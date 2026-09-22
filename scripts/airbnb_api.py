"""Airbnb 지도검색 응답 파싱과 bbox 계산. 순수 함수만 — I/O 없다.

좌표가 어디 있는가
------------------
`https://www.airbnb.co.kr/s/homes?ne_lat=…&sw_lng=…&search_by_map=true` 의
**서버렌더 HTML** 안, `<script id="data-deferred-state-0">` 에 들어 있다.

    {"__typename":"Coordinate","latitude":37.4973,"longitude":127.0374}

GraphQL API(`/api/v3/StaysSearch`)에는 없다 — 거기 오는 건 `listingId` 뿐인
`SkinnyListingItem` 이다. 그래서 HTML 을 읽는다. 브라우저는 필요 없다.
평범한 Chrome User-Agent 를 단 `urllib` 으로 200 이 온다.

실측으로 알아낸 것 (2026-09-22)
-------------------------------
  · bbox 하나로 받을 수 있는 건 **약 200건**이 끝이다. 커서는 15장인데 페이지
    끼리 좌표가 겹쳐서, bbox 안에 712건이 있어도 나머지는 못 받는다.
    그래서 `split()` 으로 쪼개 가며 훑는다.
  · 총건수(`숙소 N개`)는 **1,000 에서 잘린다**. 전국이든 서울이든 제주든 전부
    `1,000` 이다. 그래서 총건수는 "더 쪼개야 하나" 를 판정하는 데만 쓰고,
    최종 숫자는 절대 여기서 가져오지 않는다 — 모은 좌표를 세서 낸다.
    `is_capped()` 가 이 경계를 알려준다.
  · 한 응답 안에 같은 숙소의 좌표가 두 번 실려 온다. `parse_coordinates()`
    가 중복을 없애 돌려준다.
  · 경로는 `/s/homes` — 세그먼트가 하나다. robots.txt 의 `Disallow: /s/*/*`
    는 세그먼트 둘 이상을 막으므로 여기에 걸리지 않는다. `/s/서울/homes`
    같은 주소를 쓰면 걸린다. **경로를 바꾸지 말 것.**

Airbnb 는 공식 API 가 아니다. 화면이 바뀌면 이 파싱은 조용히 0건을 돌려준다.
그래서 파싱 함수는 예외를 던지지 않고, 0건 판정은 호출부(`collect_airbnb.py`)
가 실패로 다룬다. 회귀는 `tests/fixtures/airbnb_search.html.gz` 가 잡는다.
"""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.parse
from typing import NamedTuple

BASE_URL = "https://www.airbnb.co.kr/s/homes"

# 총건수가 이 값이면 "1,000개 이상" 이라는 뜻이다. 집계에 쓰면 안 된다.
TOTAL_CAP = 1000

# bbox 하나에서 실제로 받아지는 좌표 수의 실측 상한(약 200). 분할 기준은 이보다
# 낮게 잡아 여유를 둔다 — 상한에 딱 붙여 두면 경계에서 놓친다.
COLLECTED_CAP = 200


class Box(NamedTuple):
    """위경도 사각형. Airbnb 의 파라미터 이름을 그대로 쓴다."""

    ne_lat: float
    ne_lng: float
    sw_lat: float
    sw_lng: float


def search_url(box: Box, cursor: str | None = None) -> str:
    """지도검색 주소 하나. 경로는 반드시 `/s/homes`(세그먼트 1개)다."""
    params = [
        ("ne_lat", f"{box.ne_lat:g}"),
        ("ne_lng", f"{box.ne_lng:g}"),
        ("sw_lat", f"{box.sw_lat:g}"),
        ("sw_lng", f"{box.sw_lng:g}"),
        ("search_by_map", "true"),
        ("search_type", "user_map_move"),
    ]
    if cursor:
        params.append(("cursor", cursor))
    return BASE_URL + "?" + urllib.parse.urlencode(params)


# `__typename` 이 앞에 오는 것까지 묶어 좁게 잡는다. 넓게 `"latitude"` 만
# 잡으면 지도 중심 좌표 같은 것까지 숙소로 세어 버린다.
_COORD = re.compile(r'"Coordinate","latitude":(-?[\d.]+),"longitude":(-?[\d.]+)')
_TOTAL = re.compile(r"숙소\s*([\d,]+)개")
_CURSORS = re.compile(r'"pageCursors":\[(.*?)\]', re.S)
_CURSOR = re.compile(r'"([A-Za-z0-9+/=]{16,})"')


def parse_coordinates(html: str) -> list[tuple[float, float]]:
    """숙소 좌표를 나온 순서대로, 중복 없이 돌려준다. 못 찾으면 빈 목록."""
    seen: dict[tuple[float, float], None] = {}
    for lat, lng in _COORD.findall(html or ""):
        try:
            seen.setdefault((float(lat), float(lng)))
        except ValueError:
            continue
    return list(seen)


def parse_total(html: str) -> int | None:
    """화면에 적힌 총건수. 없으면 None.

    0 으로 바꾸지 않는다 — 0 이면 '숙소가 없다' 가 되어 분할이 멈춘다.
    """
    match = _TOTAL.search(html or "")
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def is_capped(total: int | None) -> bool:
    """총건수가 상한에 닿았는가 — 닿았으면 실제 값은 그 이상이다."""
    return total is not None and total >= TOTAL_CAP


def parse_cursors(html: str) -> list[str]:
    """페이지 커서를 순서대로. 없으면 빈 목록."""
    match = _CURSORS.search(html or "")
    return _CURSOR.findall(match.group(1)) if match else []


def cursor_offset(cursor: str) -> int | None:
    """커서가 가리키는 items_offset. 못 읽으면 None."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return int(json.loads(base64.b64decode(padded).decode())["items_offset"])
    except Exception:  # noqa: BLE001 — 커서 모양이 바뀌어도 수집은 계속돼야 한다
        return None


def split(box: Box) -> list[Box]:
    """bbox 를 사분면으로 쪼갠다. 넷을 합치면 원래 bbox 와 정확히 같다."""
    mid_lat = (box.ne_lat + box.sw_lat) / 2
    mid_lng = (box.ne_lng + box.sw_lng) / 2
    return [
        Box(box.ne_lat, box.ne_lng, mid_lat, mid_lng),
        Box(box.ne_lat, mid_lng, mid_lat, box.sw_lng),
        Box(mid_lat, box.ne_lng, box.sw_lat, mid_lng),
        Box(mid_lat, mid_lng, box.sw_lat, box.sw_lng),
    ]


def too_small(box: Box, min_deg: float) -> bool:
    """더 쪼개도 의미가 없을 만큼 작은가. 짧은 변을 기준으로 본다."""
    return min(box.ne_lat - box.sw_lat, box.ne_lng - box.sw_lng) < min_deg


def contains(box: Box, lat: float, lng: float) -> bool:
    """좌표가 bbox 안인가. 경계는 안으로 친다."""
    return box.sw_lat <= lat <= box.ne_lat and box.sw_lng <= lng <= box.ne_lng


def snap(lat: float, lng: float, size: float) -> tuple[float, float]:
    """좌표를 격자 칸의 남서쪽 꼭짓점으로 내린다.

    `lat / size` 를 그냥 floor 하면 격자 경계에 딱 걸린 값이 부동소수점 오차로
    한 칸 아래로 떨어진다(37.495 / 0.005 = 7498.999…). 먼저 반올림해 털어낸다.
    """
    def one(value: float) -> float:
        return round(math.floor(round(value / size, 6)) * size, 6)

    return one(lat), one(lng)


def looks_empty(total: int | None, found: int) -> bool:
    """이 bbox 에 숙소가 없는가.

    실측(2026-09-22): 동해 한복판·설악산 산속처럼 진짜 빈 bbox 는 총건수 문구도
    좌표도 없이 온다. 파싱이 고장났을 때와 응답이 똑같이 생겼다는 뜻이라, 이
    둘을 가르는 일은 호출부가 한다 — 수집기는 숙소가 확실히 있는 bbox 로
    주기적으로 확인한다(`collect_airbnb.py` 의 canary).

    여기서는 '없어 보인다' 만 판정한다. 이 구분이 중요한 이유는 `decide()` 가
    모를 때 쪼개는 쪽으로 기울기 때문이다 — 빈 bbox 를 쪼개기 시작하면 전국
    bbox 하나가 4^11 개로 불어난다.
    """
    return found == 0 and not total


def decide(total: int | None, found: int, box: Box,
           split_at: int, min_deg: float) -> str:
    """bbox 하나를 보고 `"split"` 인지 `"collect"` 인지 정한다.

    쪼개는 쪽으로 기운다. 덜 쪼개면 그 지역 숙소가 조용히 빠지지만, 더 쪼개면
    호출만 몇 번 더 쓰기 때문이다. 다만 두 곳에서 멈춘다.

      · 빈 bbox(`looks_empty`) — 안 멈추면 바다에서 4^n 으로 터진다.
      · `min_deg` 보다 작은 bbox — 명동·홍대는 400m 로 쪼개도 상한을 넘는다.
        그렇게 멈춘 칸은 `under_collected()` 로 표시해 화면에서 '실제보다
        적음' 이라 밝힌다.
    """
    if looks_empty(total, found) or too_small(box, min_deg):
        return "collect"
    if total is None or total > split_at or is_capped(total):
        return "split"
    return "collect"


def under_collected(total: int | None, found: int, split_at: int) -> bool:
    """이 bbox 에서 거둔 좌표가 실제보다 적은가.

    총건수를 못 읽었는데 좌표는 있으면 `True` 다 — 모른다는 것을 '다 거뒀다'
    로 바꾸지 않는다. 빈 bbox 는 덜 걷힌 게 아니라 원래 없는 것이다.
    """
    if looks_empty(total, found):
        return False
    return total is None or total > split_at


def published_points(state: dict) -> dict:
    """화면에 내보낼 좌표.

    수집기는 전국 한 바퀴를 며칠에 걸쳐 돈다. 그동안 모으는 중인 것은
    `pending`, 지난 바퀴를 다 돌아 확정된 것은 `points` 다. 화면에는 확정본을
    쓰되, **첫 바퀴를 아직 못 끝냈으면 모으는 중인 것이라도 보여준다** —
    그러지 않으면 처음 며칠 동안 지도가 통째로 비어 있다.
    """
    return state.get("points") or state.get("pending") or {}


def snapshot_complete(state: dict) -> bool:
    """확정 스냅샷이 있는가 — 전국을 한 바퀴 다 돈 결과인가.

    화면 각주가 "전국을 다 훑었다" 고 말할 근거다. 틀리면 반쪽 자료를 완주한
    것처럼 보여주게 된다.

    `pending` 이 생기기 전에 쓰인 상태 파일은 확정본과 모으는 중인 것이 한
    통에 있었다. 그때는 `frontier` 가 남아 있으면 아직 돌던 중이었다는 뜻이다.
    """
    if not state.get("points"):
        return False
    if "pending" in state:
        return True
    return not state.get("frontier")


def needs_empty_confirmation(box: Box, min_deg: float) -> bool:
    """'비었다' 는 판정을 한 번 더 확인해야 할 만큼 큰 bbox 인가.

    빈 응답과 고장난 응답은 생김새가 같다(`looks_empty` 참고). 작은 bbox 를
    잘못 가지치면 동네 하나가 빠지지만, 큰 bbox 를 잘못 가지치면 **도 하나가
    통째로 지도에서 사라진다.** 그런데 바다에는 빈 bbox 가 수없이 많으므로
    전부 확인하면 비용이 두 배가 된다. 그래서 위쪽 두세 단계(기본 1도 초과)만
    확인한다 — 몇 번 더 묻는 값으로 최악의 실패를 막는다.

    짧은 변으로 잰다. 긴 변만 보면 가늘고 긴 bbox 를 큰 것으로 잘못 센다.
    """
    return min(box.ne_lat - box.sw_lat, box.ne_lng - box.sw_lng) > min_deg


def seed_boxes(geo: dict, pad: float = 0.05) -> list[Box]:
    """쿼드트리를 시작할 bbox 들. 시도마다 하나씩, 시군구 경계상자를 합쳐 만든다.

    전국을 사각형 하나로 덮으면 규슈가 통째로 들어온다 — 실측(2026-09-22)으로
    후쿠오카 숙소 1,694건을 받아다 버렸다. 받은 만큼 예산이 사라지고, 그 위를
    쪼개느라 더 쓴다. 시도별로 자르면 바다와 일본이 대부분 빠진다.

    `pad` 는 경계에 딱 붙은 숙소가 빠지지 않게 두는 여유다. 시도끼리 조금씩
    겹치지만 좌표는 키로 중복이 걸러지므로 숫자가 틀어지지 않는다.
    """
    bounds: dict[str, list[float]] = {}
    for feature in geo.get("features") or []:
        code = (feature.get("properties") or {}).get("sgg")
        geometry = feature.get("geometry") or {}
        if not code or geometry.get("type") != "MultiPolygon":
            continue
        sido = code[:2]
        for polygon in geometry.get("coordinates") or []:
            for ring in polygon:
                for lng, lat in ring:
                    box = bounds.get(sido)
                    if box is None:
                        bounds[sido] = [lat, lat, lng, lng]
                    else:
                        box[0] = min(box[0], lat)
                        box[1] = max(box[1], lat)
                        box[2] = min(box[2], lng)
                        box[3] = max(box[3], lng)
    return [Box(ne_lat=b[1] + pad, ne_lng=b[3] + pad,
                sw_lat=b[0] - pad, sw_lng=b[2] - pad)
            for _, b in sorted(bounds.items())]
