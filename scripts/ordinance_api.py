"""법제처 국가법령정보에서 시·군 도시계획조례의 용적률 상한을 읽는다. 순수 함수만 둔다.

I/O 는 collect_ordinance.py 가 담당한다.

왜 필요한가 — 용적률 상한은 광역이 아니라 **시·군 조례**가 정한다. 서울시 값
(1종 150 · 2종 200 · 3종 250 · 준주거 400%)을 경기 단지에 붙이면 틀린다.
실측하면 가평 3종 300%, 용인 290%, 성남·안양 280%, 고양 250% 로 제각각이라
서울 값을 쓰면 3종에서 30~50%p 를 과소평가한다.

인증은 OC 파라미터 하나다 (open.law.go.kr 가입 이메일 ID). 별도 활용신청이 없다.

조례마다 표기가 달라 파싱이 늘 성공하지는 않는다. 못 읽은 곳은 빈 값으로 남기고
건수를 로그로 알린다 — 지어내는 것보다 비우는 게 낫다.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.parse
import xml.etree.ElementTree as ET

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"

# 조례명이 시는 '도시계획 조례', 군은 '군계획 조례' 다 (가평군이 그렇다).
ORDINANCE_NAME_RE = re.compile(r"(도시|군)계획\s*조례$")

# 용도지역명. 조례마다 '제3종일반주거지역' 과 '제3종 일반주거지역' 을 섞어 쓴다.
ZONE_RE = (
    r"(제\s*\d\s*종\s*(?:전용|일반)주거지역"
    r"|준주거지역|중심상업지역|일반상업지역|근린상업지역|유통상업지역"
    r"|전용공업지역|일반공업지역|준공업지역"
    r"|보전녹지지역|생산녹지지역|자연녹지지역)"
)
ZONE_HEAD_RE = re.compile(rf"{ZONE_RE}\s*[::]")

# 값 표기가 조례마다 다르다. '250퍼센트' 와 '100분의 250' 을 모두 받는다
# (군포시가 후자를 쓴다). 개정일자('2015.10.08') 같은 숫자는 두 형태 어디에도
# 걸리지 않아 오인식하지 않는다.
VALUE_RE = re.compile(r"(?:([\d,]+)\s*퍼센트|100\s*분의\s*([\d,]+))")

# 호 단위로 자른다. '1. 제1종전용주거지역 : 100퍼센트 이하 2. 제2종...' 꼴이다.
CLAUSE_SPLIT_RE = re.compile(r"(?:(?<=\s)|^)\d{1,2}\.\s")

# 정비사업에만 따로 적용되는 단서. 이 대시보드가 정확히 그 맥락이라 이 값을 우선한다.
#   용인시  '제2종 일반주거지역 : 240퍼센트 이하. 다만, 「도시 및 주거환경정비법」
#            제2조제2호의 정비사업으로 건설하는 공동주택은 250퍼센트 이하'
#   군포시  '제2종 일반주거지역 : 100분의 230 이하. 다만, 「도시 및 주거환경정비법」의
#            주택재건축사업은 100분의 250 이하'
#   수원시  '제2종일반주거지역 : 일반건축은 250퍼센트 이하(다만, 공동주택은 230퍼센트
#            이하, …「도시 및 주거환경정비법」…에 따른 공동주택은 250퍼센트 이하)'
# '정비사업' 이라는 낱말이 없는 조례가 있어 근거 법률명까지 단서로 인정한다.
PROVISO_MARK_RE = re.compile(r"(정비사업|재건축|재개발|주거환경정비법)")

COLUMNS = [
    "city",
    "law_id",
    "law_name",
    "promulgated",
    "zone",
    "far",
    "far_redev",
    "has_proviso",
]


class OrdinanceError(Exception):
    """응답이 예상한 형태가 아닐 때."""


def normalize_zone(name: str) -> str:
    """'제3종 일반주거지역' → '제3종일반주거지역'. 브이월드 용도지역명과 맞추기 위함."""
    return re.sub(r"\s+", "", name or "")


def search_url(oc: str, query: str, display: int = 20) -> str:
    params = {"OC": oc, "target": "ordin", "type": "XML", "query": query, "display": str(display)}
    return f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"


def service_url(oc: str, law_id: str) -> str:
    params = {"OC": oc, "target": "ordin", "ID": law_id, "type": "XML"}
    return f"{SERVICE_URL}?{urllib.parse.urlencode(params)}"


def pick_ordinance(xml_text: str, org_name: str) -> dict | None:
    """검색 결과에서 그 지자체의 도시(군)계획 조례를 고른다.

    org_name 은 '경기도 성남시' 처럼 법제처가 쓰는 지자체기관명이다.
    이름이 비슷한 다른 조례('성남시 도시계획변경 사전협상 운영에 관한 조례')가
    함께 오므로 끝이 '도시계획 조례' / '군계획 조례' 인 것만 남긴다.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise OrdinanceError(f"검색 응답을 파싱하지 못했습니다: {exc}") from exc

    for law in root.iter("law"):
        org = (law.findtext("지자체기관명") or "").strip()
        name = (law.findtext("자치법규명") or "").strip()
        if org == org_name and ORDINANCE_NAME_RE.search(name):
            return {
                "law_id": (law.findtext("자치법규ID") or "").strip(),
                "law_name": name,
                "promulgated": (law.findtext("공포일자") or "").strip(),
            }
    return None


def parse_far(xml_text: str) -> dict[str, dict]:
    """조례 본문에서 용도지역별 용적률 상한을 뽑는다.

    조제목에 '용적률' 이 들어간 조만 본다 — 건폐율 조에도 같은 용도지역 이름과
    퍼센트가 나와서, 본문 전체에 정규식을 걸면 건폐율(50~70%)을 용적률로 읽는다.

    돌려주는 값: {정규화한 용도지역명: {"far": 240, "far_redev": 250|None,
                                        "has_proviso": bool}}
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise OrdinanceError(f"본문 응답을 파싱하지 못했습니다: {exc}") from exc

    for jo in root.iter("조"):
        title = (jo.findtext("조제목") or "").strip()
        if "용적률" not in title:
            continue
        body = " ".join("".join(jo.itertext()).split())

        out: dict[str, dict] = {}
        for clause in CLAUSE_SPLIT_RE.split(body):
            head = ZONE_HEAD_RE.search(clause)
            if not head:
                continue
            rest = clause[head.end():]
            # 콜론 바로 뒤에 값이 오지 않는 조례가 있다 (수원시 '일반건축은 250퍼센트').
            # 호 안에서 처음 나오는 값을 기본 상한으로 본다.
            base_match = VALUE_RE.search(rest)
            if not base_match:
                continue
            zone = normalize_zone(head.group(1))
            base = int((base_match.group(1) or base_match.group(2)).replace(",", ""))

            # 단서는 '다만' 뒤부터 본다. 그 안에서 정비사업 근거가 보이면
            # 뒤따르는 첫 값을 정비사업 적용 상한으로 잡는다.
            redev = None
            for marker in ("다만", "("):
                pos = rest.find(marker, base_match.end())
                if pos < 0:
                    continue
                tail = rest[pos:]
                mark = PROVISO_MARK_RE.search(tail)
                if not mark:
                    continue
                value = VALUE_RE.search(tail[mark.end():])
                if value:
                    redev = int((value.group(1) or value.group(2)).replace(",", ""))
                    break

            # 같은 조에 같은 용도지역이 두 번 나오면 처음 것을 남긴다.
            out.setdefault(
                zone,
                {"far": base, "far_redev": redev, "has_proviso": redev is not None},
            )
        if out:
            return out
    return {}


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r.get("city", ""), r.get("zone", ""))):
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return buf.getvalue()


def effective_far(row: dict) -> int | None:
    """정비사업에 실제로 적용되는 상한.

    단서로 정비사업 전용값이 따로 있으면 그것을, 없으면 기본값을 쓴다.
    이 대시보드는 재건축·재개발을 보는 화면이라 단서 쪽이 맞는 값이다.
    """
    for key in ("far_redev", "far"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None
