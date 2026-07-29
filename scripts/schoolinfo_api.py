"""학교알리미 학교별 공시 「졸업생의 진로 현황」 클라이언트.

    https://www.schoolinfo.go.kr/ei/pp/Pneipp_b06_s0p.do

학교별 특목고·자사고 진학 실적은 이 화면에만 있다. 확인한 다른 경로는 전부
막혀 있었다:

- 학교알리미 OpenAPI: apiType 0~139 을 sidoCode/sggCode 까지 채워 전수 호출해도
  진학 유형별 항목이 없다(데이터가 오는 26개 중 하나도).
- 학교알리미 「항목별 공시정보」(지역 일괄 조회): 2023~2026 전 연도, 전 분류에
  「졸업생의 진로 현황」이 목록에 없다.
- EDSS 신청 자료: 학교 단위 행은 있지만 식별자가 익명 `개방ID` 뿐이라 학교를
  특정할 수 없다(data/edss/README.md).

여기는 학교명이 붙은 화면이라 조인이 필요 없다. 대신 학교 하나에 요청 하나라
서울·경기 1,147곳 × 연도만큼 부른다.

순수 파싱 함수(parse_progression)와 네트워크 I/O 를 분리해 두어 파싱은 단독으로
검증할 수 있다. 응답은 UTF-8 이 아니라 cp949 다.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.schoolinfo.go.kr"
SIGUNGU_URL = f"{BASE}/ei/ss/pneiss_a03_s0_sigungu_json.do"
SCHOOL_URL = f"{BASE}/ei/ss/pneiss_a03_s0_school_json.do"
PROGRESSION_URL = f"{BASE}/ei/pp/Pneipp_b06_s0p.do"
REFERER = f"{BASE}/ei/ss/Pneiss_b01_s0.do"

# 학교급 코드. 01 은 전체, 02 초, 03 중, 04 고 — 실측으로 확인했다.
MIDDLE_SCHOOL = "03"

SIDO = {"서울특별시": "1100000000", "경기도": "4100000000"}

# 「졸업생의 진로 현황」 화면을 여는 고정 파라미터. GS_HANGMOK_CD=06 이 항목
# 코드이고 나머지는 화면이 함께 넘기는 값이라 그대로 둔다.
FIXED_PARAMS = {
    "GS_HANGMOK_CD": "06",
    "GS_HANGMOK_NO": "13-다",
    "GS_HANGMOK_NM": "졸업생의 진로 현황",
    "GS_BURYU_CD": "JG040",
    "JG_BURYU_CD": "JG130",
    "JG_HANGMOK_CD": "52",
    "JG_GUBUN": "1",
    "GS_TYPE": "Y",
    "SORT": "BR",
    "LOAD_TYPE": "single",
}

HEADERS = {
    "Accept": "text/html, */*; q=0.01",
    "Accept-Language": "ko",
    "Referer": REFERER,
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

# 표의 '합계' 행에서 읽을 값의 순서. 화면 헤더와 같은 순서다.
#   구분 | 졸업자 | 일반고 | 특성화고 | 과학고 | 외고국제고 | 예고체고 |
#   마이스터고 | 특목고소계 | 자사고 | 자공고 | 자율고소계 | 기타 | 진학자계 |
#   취업자 | 대안교육기관진학 | 무직자및미상
TOTAL_COLUMNS = [
    "grad", "general", "vocational", "science", "foreign_intl", "art_pe",
    "meister", "special_sum", "auto_private", "auto_public", "auto_sum",
    "etc", "advanced", "employed", "alternative", "none",
]


class SchoolInfoError(Exception):
    """응답은 왔지만 표를 찾을 수 없을 때."""


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).replace("&nbsp;", " ").strip()


def parse_progression(html: str) -> dict[str, int] | None:
    """공시 화면 HTML 에서 '합계' 행을 뽑는다. 표가 없으면 None.

    표가 아예 없는 학교(그 해에 졸업생이 없거나 미공시)가 있으므로 예외 대신
    None 을 돌려준다 — 수집을 멈출 일이 아니라 그 학교만 건너뛸 일이다.
    """
    for table in re.findall(r"<table.*?</table>", html, re.S):
        for row in re.findall(r"<tr.*?</tr>", table, re.S):
            cells = [_text(c) for c in re.findall(r"<t[hd].*?</t[hd]>", row, re.S)]
            if not cells or cells[0].replace(" ", "") != "합계":
                continue
            values = []
            for cell in cells[1:]:
                digits = cell.replace(",", "")
                if not re.fullmatch(r"-?\d+", digits):
                    return None
                values.append(int(digits))
            if len(values) < len(TOTAL_COLUMNS):
                return None
            return dict(zip(TOTAL_COLUMNS, values))
    return None


def special_rate(row: dict[str, int]) -> float | None:
    """특목고·자사고 진학률(%). 졸업자가 0이면 None.

    분자는 특목고 소계 + 자율고 소계다 — 페이지의 시·도 차트(progression.json,
    EDSS 기반)와 같은 정의라야 두 숫자를 나란히 놓을 수 있다.
    """
    grad = row.get("grad") or 0
    if grad <= 0:
        return None
    return (row.get("special_sum", 0) + row.get("auto_sum", 0)) / grad * 100


def _get(url: str, params: dict | None = None, data: dict | None = None,
         retries: int = 3) -> bytes:
    body = urllib.parse.urlencode(data, encoding="utf-8").encode() if data else None
    full = f"{url}?{urllib.parse.urlencode(params, encoding='utf-8')}" if params else url
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, data=body, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=40) as res:
                return res.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{retries}회 재시도 후에도 실패했습니다: {last}")


def fetch_sigungu(sido_code: str) -> list[dict]:
    """시도 안의 시군구 목록."""
    import json
    raw = _get(SIGUNGU_URL, data={"SIDO_CODE": sido_code})
    payload = json.loads(raw.decode("utf-8", "replace"))
    return payload["list"] if isinstance(payload, dict) else payload


def fetch_schools(sido_code: str, gugun_code: str,
                  level: str = MIDDLE_SCHOOL) -> list[dict]:
    """시군구 안의 학교 목록. SHL_NM(학교명)과 SHL_IDF_CD(UUID)를 쓴다."""
    import json
    raw = _get(SCHOOL_URL, data={
        "HG_JONGRYU_GB": level, "SIDO_CODE": sido_code, "GUGUN_CODE": gugun_code})
    payload = json.loads(raw.decode("utf-8", "replace"))
    return payload["list"] if isinstance(payload, dict) else payload


def fetch_progression(shl_idf_cd: str, school_name: str, year: str) -> dict | None:
    """학교 하나의 그 해 졸업생 진로 현황. 표가 없으면 None."""
    params = dict(FIXED_PARAMS)
    params.update({
        "HG_NM": school_name,
        "SHL_IDF_CD": shl_idf_cd,
        "JG_YEAR": year,
        "JG_YEAR2": year,
        "CHOSEN_JG_YEAR": year,
        "PRE_JG_YEAR": year,
    })
    # 응답은 cp949 다. utf-8 로 읽으면 학교명·항목명이 전부 깨진다.
    html = _get(PROGRESSION_URL, params=params).decode("cp949", "replace")
    return parse_progression(html)
