"""브이월드(V-World) 데이터 API 조회. 순수 함수만 둔다.

I/O 는 collect_vworld.py 가 담당한다.

왜 필요한가 — 경기도 용도지역을 얻는 유일한 길이다.
  - 서울시 도시계획포털(UPIS)의 지적도·용도지역은 서울 전용이다.
  - 건축물대장 지역지구구역(getBrJijiguInfo)은 종 구분을 주지 않는다.
    은마조차 '일반주거지역'·'도시지역' 으로만 답해 용적률 상한(1종 150 /
    2종 200 / 3종 250)을 특정할 수 없다.
  - 브이월드 LT_C_UQ111 은 '제2종일반주거지역' 처럼 종까지 준다.

국가공간정보포털(NSDI, apis.data.go.kr/1611000/nsdi/...)은 2024-01-01 종료됐고
브이월드로 통합됐다. 아직 그 경로를 안내하는 문서가 많지만 400 만 돌아온다.

인증키는 저장소에 두지 않는다. 환경변수 VWORLD_API_KEY 를 쓴다.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
from pathlib import Path

BASE = "https://api.vworld.kr/req/data"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# 연속지적도(부번). PNU 로 직접 조회된다 — data/complexes.csv.gz 와 그대로 붙는다.
PARCEL_LAYER = "LP_PA_CBND_BUBUN"
# 용도지역. 폴리곤이라 PNU 가 없다. 필지 대표점을 넣어 포함 관계로 찾는다.
LANDUSE_LAYER = "LT_C_UQ111"

# 서울특별시 도시계획조례 기준 용적률 상한(%)을 그대로 쓰지 않는다.
# 지자체마다 조례가 달라 경기도 단지에 서울 값을 붙이면 틀린다.
# 종 구분만 넘기고 상한 해석은 build 쪽에서 한다.

COLUMNS = [
    "pnu",
    "sgg_cd",
    "jibun",
    "addr",
    "area_sqm",
    "jiga_won_sqm",
    "lon",
    "lat",
    "landuse_nm",
    "sido_name",
]

EARTH_RADIUS_M = 6371000.0


class VworldError(Exception):
    """응답 status 가 OK 도 NOT_FOUND 도 아닐 때."""


class KeyMissing(Exception):
    """인증키가 없다."""


def load_key() -> str:
    """VWORLD_API_KEY 를 환경변수에서, 없으면 저장소 루트 .env 에서 읽는다.

    .env 는 .gitignore 에 등록돼 있어 커밋되지 않는다. 키를 저장소 파일에
    그냥 두면 gh-pages 가 그대로 웹에 서빙해 공개된다 (neis_api.load_key 와
    같은 이유·같은 방식).
    """
    key = os.environ.get("VWORLD_API_KEY")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("VWORLD_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise KeyMissing(
        "VWORLD_API_KEY 를 찾지 못했습니다. www.vworld.kr/dev/v4api.do 에서 "
        "인증키를 받아 환경변수로 지정하거나 저장소 루트 .env 에 "
        "VWORLD_API_KEY=... 로 넣으세요."
    )


def query_url(key: str, layer: str, *, attr_filter: str = "", geom_filter: str = "",
              size: int = 10, domain: str = "localhost") -> str:
    params = {
        "service": "data",
        "request": "GetFeature",
        "key": key,
        "data": layer,
        "format": "json",
        "domain": domain,
        "size": str(size),
    }
    if attr_filter:
        params["attrFilter"] = attr_filter
    if geom_filter:
        params["geomFilter"] = geom_filter
    return f"{BASE}?{urllib.parse.urlencode(params)}"


def parcel_url(key: str, pnu: str) -> str:
    return query_url(key, PARCEL_LAYER, attr_filter=f"pnu:=:{pnu}", size=10)


def landuse_url(key: str, lon: float, lat: float) -> str:
    return query_url(key, LANDUSE_LAYER, geom_filter=f"POINT({lon} {lat})", size=10)


def _features(payload: str | dict) -> list[dict]:
    """응답에서 feature 목록을 꺼낸다. 자료 없음은 빈 목록이지 오류가 아니다."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    response = data.get("response") or {}
    status = response.get("status")
    if status == "NOT_FOUND":
        return []
    if status != "OK":
        error = response.get("error") or {}
        raise VworldError(f"{status}: {error.get('code')} {error.get('text', '')[:120]}")
    collection = (response.get("result") or {}).get("featureCollection") or {}
    return collection.get("features") or []


def _rings(geometry: dict) -> list[list[list[float]]]:
    """Polygon / MultiPolygon 을 링 목록으로 편다."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return coords
    if kind == "MultiPolygon":
        return [ring for polygon in coords for ring in polygon]
    return []


def ring_area_sqm(ring: list[list[float]]) -> float:
    """WGS84 링의 면적(㎡). 링 평균 위도에서 등장방형으로 펴고 신발끈 공식을 쓴다.

    필지 하나는 수백 미터 규모라 이 근사의 오차가 0.1% 아래다. 정확한 측지
    면적이 필요한 게 아니라 대지지분(대지면적 ÷ 세대수)을 내는 게 목적이다.
    """
    if len(ring) < 4:
        return 0.0
    lat_mean = sum(p[1] for p in ring) / len(ring)
    k = math.cos(math.radians(lat_mean))
    deg = math.pi / 180 * EARTH_RADIUS_M
    pts = [(p[0] * k * deg, p[1] * deg) for p in ring]
    total = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def polygon_area_sqm(geometry: dict) -> float:
    """첫 링을 외곽, 나머지를 구멍으로 보고 면적을 낸다.

    MultiPolygon 은 조각마다 외곽/구멍 구분이 따로라 여기서는 조각 구분을 잃는다.
    필지는 구멍이 거의 없어 실무상 문제가 되지 않는다 — 대신 가장 큰 링을
    외곽으로 삼아 순서에 기대지 않는다.
    """
    rings = _rings(geometry)
    if not rings:
        return 0.0
    areas = [ring_area_sqm(r) for r in rings]
    outer = max(areas)
    holes = sum(a for a in areas if a is not outer and a < outer)
    return max(0.0, outer - holes) if len(areas) > 1 else outer


def centroid(geometry: dict) -> tuple[float, float] | None:
    """가장 넓은 링의 무게중심. 조각난 필지에서 대표점이 빈 땅에 찍히는 걸 막는다."""
    best: tuple[float, float, float] | None = None
    for ring in _rings(geometry):
        if len(ring) < 4:
            continue
        a = cx = cy = 0.0
        for i in range(len(ring) - 1):
            x0, y0 = ring[i][0], ring[i][1]
            x1, y1 = ring[i + 1][0], ring[i + 1][1]
            cross = x0 * y1 - x1 * y0
            a += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if a == 0:
            continue
        a *= 0.5
        candidate = (abs(a), cx / (6 * a), cy / (6 * a))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2]


def parse_parcel(payload: str | dict) -> dict | None:
    """지적도 응답에서 필지 하나를 뽑는다. 같은 PNU 조각이 여럿이면 면적을 합친다."""
    features = _features(payload)
    if not features:
        return None
    total_area = 0.0
    best_geom = None
    best_area = -1.0
    props: dict = {}
    for feature in features:
        geometry = feature.get("geometry") or {}
        area = polygon_area_sqm(geometry)
        total_area += area
        if area > best_area:
            best_area, best_geom = area, geometry
            props = feature.get("properties") or {}
    point = centroid(best_geom) if best_geom else None
    pnu = (props.get("pnu") or "").strip()
    return {
        "pnu": pnu,
        "sgg_cd": pnu[:5],
        "jibun": (props.get("jibun") or "").strip(),
        "addr": (props.get("addr") or "").strip(),
        "area_sqm": round(total_area, 1),
        "jiga_won_sqm": (props.get("jiga") or "").strip(),
        "lon": round(point[0], 6) if point else "",
        "lat": round(point[1], 6) if point else "",
    }


def parse_landuse(payload: str | dict) -> dict | None:
    """용도지역 응답에서 이름을 뽑는다.

    한 점에 여러 폴리곤이 겹칠 수 있다 ('도시지역' 같은 상위 구분과
    '제2종일반주거지역' 이 함께 온다). 종 구분이 있는 쪽을 고른다 —
    용적률 상한을 정하려면 그것만이 쓸모 있다.
    """
    features = _features(payload)
    if not features:
        return None
    names = [(f.get("properties") or {}) for f in features]
    graded = [n for n in names if "종" in (n.get("uname") or "")]
    pick = graded[0] if graded else names[0]
    return {
        "landuse_nm": (pick.get("uname") or "").strip(),
        "sido_name": (pick.get("sido_name") or "").strip(),
    }
