"""전국초중등학교위치표준데이터 API 클라이언트.

    https://api.data.go.kr/openapi/tn_pubr_public_elesch_mskul_lc_api

순수 파싱 함수와 네트워크 I/O 를 분리해 두어 파싱은 단독으로 검증할 수 있다.
전국 12,011건을 numOfRows=1000 으로 13번 부르면 다 온다.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://api.data.go.kr/openapi/tn_pubr_public_elesch_mskul_lc_api"

# 이 도구가 다루는 학교급. 일반 고등학교는 배정이라 '근처'가 성립하지 않지만,
# 특목고(과학고·외고·국제고)는 지원해서 가는 학교라 위치가 의미를 가진다.
# 여기서는 고등학교를 일단 다 통과시키고, 어느 고등학교가 특목고인지는
# collect_schools.py 가 NEIS 분류(neis_api.py)와 조인해 가른다 — 위치
# 표준데이터에는 고등학교 종류를 구분할 필드가 아예 없다.
LEVELS = ("초등학교", "중학교", "고등학교")
REGIONS = ("서울특별시", "경기도")

COLUMNS = [
    "school_id", "school_name", "level", "found_type",
    "addr", "road_addr", "sido_office", "lat", "lon",
    # 특목고 계열(과학계열/외국어계열/국제계열). 초·중은 빈 문자열이다.
    "course",
]


class ApiError(Exception):
    """resultCode 가 정상(00)이 아닐 때. code 로 원인을 구분한다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def parse_response(payload: dict) -> tuple[list[dict], int]:
    """응답 본문에서 (항목 목록, 전체 건수) 를 꺼낸다. 오류면 ApiError."""
    resp = payload.get("response") or {}
    header = resp.get("header") or {}
    code = str(header.get("resultCode", "")).strip()
    if code != "00":
        raise ApiError(code, str(header.get("resultMsg", "")).strip())
    body = resp.get("body") or {}
    return list(body.get("items") or []), int(body.get("totalCount") or 0)


def normalize(item: dict) -> dict | None:
    """원본 항목을 CSV 한 행으로. 범위 밖이면 None.

    버리는 것: 서울·경기 밖, 초·중·고 아닌 학교, 분교, 운영 중이 아닌 학교,
    좌표 결측. 고등학교를 특목고만 남기는 일은 여기서 하지 않는다 — 그 판정은
    NEIS 응답이 있어야 하고, 이 함수는 항목 하나만 보고 결정하는 순수 함수로
    둔다(collect_schools.py 가 조인 뒤에 거른다).
    """
    addr = (item.get("lnmadr") or "").strip()
    if not addr.startswith(REGIONS):
        return None
    if (item.get("schoolSe") or "").strip() not in LEVELS:
        return None
    if (item.get("bnhhSe") or "").strip() not in ("", "본교"):
        return None
    if (item.get("operSttus") or "").strip() not in ("", "운영"):
        return None
    lat = (item.get("latitude") or "").strip()
    lon = (item.get("longitude") or "").strip()
    if not lat or not lon:
        return None
    return {
        "school_id": (item.get("schoolId") or "").strip(),
        "school_name": (item.get("schoolNm") or "").strip(),
        "level": (item.get("schoolSe") or "").strip(),
        "found_type": (item.get("fondType") or "").strip(),
        "addr": addr,
        "road_addr": (item.get("rdnmadr") or "").strip(),
        "sido_office": (item.get("cddcNm") or "").strip(),
        "lat": lat,
        "lon": lon,
        "course": "",
    }


def fetch_page(key: str, page: int, rows: int = 1000, retries: int = 3) -> tuple[list[dict], int]:
    """한 페이지를 받는다. 네트워크 오류는 지수 백오프로 재시도한다."""
    query = urllib.parse.urlencode(
        {"serviceKey": key, "type": "json", "pageNo": page, "numOfRows": rows})
    url = f"{ENDPOINT}?{query}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                payload = json.loads(res.read().decode("utf-8"))
            return parse_response(payload)
        except ApiError:
            raise  # 키·서비스 문제는 재시도해도 소용없다
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{retries}회 재시도 후에도 실패했습니다: {last}")
