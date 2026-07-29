"""NEIS 교육정보 개방 포털 학교기본정보 API 클라이언트.

    https://open.neis.go.kr/hub/schoolInfo

고등학교의 '종류'(일반고/특목고/특성화고/자율고) 와 특목고 계열을 여기서만
얻을 수 있다. 위치 표준데이터(schools_api.py) 에는 학교급(초/중/고) 과
설립구분(공립/사립) 뿐이라 특목고를 가려낼 수 없고, 이름으로 거르는 것도
불가능하다 — 서울·경기 고등학교 중 이름에 '과학고' 가 든 20곳 가운데 실제
특목고는 5곳뿐이고 나머지는 조리·의료·자동차과학고 같은 특성화고다.

순수 파싱 함수와 네트워크 I/O 를 분리해 두어 파싱은 단독으로 검증할 수 있다.
키가 없으면 55건만 내려오므로(무료 발급, open.neis.go.kr) 키는 필수다.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://open.neis.go.kr/hub/schoolInfo"

# 시도교육청 코드 → 위치 표준데이터 주소의 시도 접두사. 조인 키를 (시도, 학교명)
# 으로 잡기 위해 필요하다 — 학교명은 시도 안에서만 유일하다.
OFFICES = {"B10": "서울특별시", "J10": "경기도"}

# 지도에 올릴 특목고 계열. 예술·체육 계열과 마이스터고(산업수요 맞춤형)는
# 전국 모집이라 '근처' 와의 연관이 약해 뺀다.
TARGET_COURSES = ("과학계열", "외국어계열", "국제계열")

# 계열이 비어 있는 특목고를 이름으로 구제할 때 쓴다. 경기외국어고·동두천
# 외국어고가 여기 해당한다 — HS_SC_NM 은 '특목고' 인데 SPCLY_PURPS_HS_ORD_NM
# 이 비어 있어, 계열만 보고 거르면 외고인데도 조용히 빠진다.
NAME_TO_COURSE = {
    "과학고등학교": "과학계열",
    "외국어고등학교": "외국어계열",
    "국제고등학교": "국제계열",
}

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class NeisError(Exception):
    """RESULT.CODE 가 정상(INFO-000)이 아닐 때. code 로 원인을 구분한다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def load_key() -> str:
    """NEIS_API_KEY 를 환경변수에서, 없으면 저장소 루트 .env 에서 읽는다.

    .env 는 .gitignore 에 등록돼 있어 커밋되지 않는다. 키를 저장소 파일에
    그냥 두면 gh-pages 가 그대로 웹에 서빙해 공개된다 — 실제로 한 번 그랬다.
    """
    key = os.environ.get("NEIS_API_KEY")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("NEIS_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise SystemExit(
        "NEIS_API_KEY 를 찾지 못했습니다. open.neis.go.kr 에서 인증키를 받아 "
        "환경변수로 지정하거나 저장소 루트 .env 에 NEIS_API_KEY=... 로 넣으세요."
    )


def parse_response(payload: dict) -> tuple[list[dict], int]:
    """응답 본문에서 (항목 목록, 전체 건수) 를 꺼낸다. 오류면 NeisError.

    NEIS 는 정상일 때 {'schoolInfo': [head, rows]} 를, 오류일 때
    {'RESULT': {...}} 를 준다 — 결과 없음(INFO-200) 도 오류 모양으로 온다.
    그 경우는 빈 목록으로 돌려주는 게 호출부에서 다루기 쉽다.
    """
    if "schoolInfo" not in payload:
        result = payload.get("RESULT") or {}
        code = str(result.get("CODE", "")).strip()
        message = str(result.get("MESSAGE", "")).strip()
        if code == "INFO-200":  # 해당하는 데이터가 없습니다
            return [], 0
        raise NeisError(code or "UNKNOWN", message or json.dumps(payload)[:200])
    blocks = payload["schoolInfo"]
    head = blocks[0].get("head") or []
    total = 0
    for entry in head:
        if "list_total_count" in entry:
            total = int(entry["list_total_count"])
            break
    rows = blocks[1].get("row") or []
    return list(rows), total


def course_of(item: dict) -> str | None:
    """특목고면 계열을, 아니면 None. 대상 계열(TARGET_COURSES) 밖도 None.

    계열이 비어 있는 특목고는 이름으로 보완한다. 반대로 이름만 보고 판정하지는
    않는다 — HS_SC_NM 이 '특목고' 인 학교에 한해서만 이름을 참고한다.
    """
    if (item.get("HS_SC_NM") or "").strip() != "특목고":
        return None
    course = (item.get("SPCLY_PURPS_HS_ORD_NM") or "").strip()
    if course in TARGET_COURSES:
        return course
    if course:
        return None
    name = (item.get("SCHUL_NM") or "").strip()
    for suffix, guessed in NAME_TO_COURSE.items():
        if name.endswith(suffix):
            return guessed
    return None


def fetch_page(key: str, office: str, page: int, rows: int = 1000,
               retries: int = 3) -> tuple[list[dict], int]:
    """한 시도교육청의 고등학교 한 페이지. 네트워크 오류는 지수 백오프로 재시도."""
    query = urllib.parse.urlencode({
        "KEY": key, "Type": "json", "pIndex": page, "pSize": rows,
        "ATPT_OFCDC_SC_CODE": office, "SCHUL_KND_SC_NM": "고등학교",
    })
    url = f"{ENDPOINT}?{query}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                payload = json.loads(res.read().decode("utf-8"))
            return parse_response(payload)
        except NeisError:
            raise  # 키 문제는 재시도해도 소용없다
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{retries}회 재시도 후에도 실패했습니다: {last}")


def fetch_courses(key: str, page_size: int = 1000) -> dict[tuple[str, str], str]:
    """{(시도, 학교명): 계열} 을 만든다. 대상 계열 특목고만 담는다."""
    out: dict[tuple[str, str], str] = {}
    for office, sido in OFFICES.items():
        seen = 0
        page = 1
        while True:
            items, total = fetch_page(key, office, page, page_size)
            if not items:
                break
            seen += len(items)
            for item in items:
                course = course_of(item)
                if course is None:
                    continue
                name = (item.get("SCHUL_NM") or "").strip()
                if name:
                    out[(sido, name)] = course
            if seen >= total:
                break
            page += 1
            time.sleep(0.3)
    return out
