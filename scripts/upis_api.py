"""서울도시계획포털(UPIS) 정비구역 도형 조회. 순수 함수만 둔다.

I/O 는 collect_zones.py 가 담당한다.

포털은 내부 ArcGIS 10.81 을 /proxy/proxy.jsp 로 열어둔다. 표준 ArcGIS REST
query 문법이 그대로 통해서 별도 라이브러리 없이 읽을 수 있다. 다만 비공식
경로이므로 언제든 막힐 수 있다 — 막히면 구역 경계 없이 대표지번 좌표만으로도
화면이 성립하도록 build_redevelopment.py 쪽에서 폴리곤을 선택 항목으로 다룬다.

주의: 이 서비스는 pagination(resultOffset/resultRecordCount)을 지원하지 않는다.
레이어당 건수가 maxRecordCount(1000)보다 적어 where=1=1 로 한 번에 받는다.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_api  # noqa: E402  구역명 정규화를 한 곳에서만 정의하기 위해 빌려 쓴다

PROXY = "https://urban.seoul.go.kr/proxy/proxy.jsp"
MAPSERVER = "http://98.33.2.225:6080/arcgis/rest/services/UPIS/20200526_WMS/MapServer"
REFERER = "https://urban.seoul.go.kr/view/map/mapPopup.html"

# 레이어 번호 → (레이어 코드, 사업유형). 도시계획포털 mapLayers.js 의 분류를 옮긴 것이다.
# 정비사업·소규모정비사업·재정비촉진사업 세 묶음만 받는다.
ZONE_LAYERS: dict[int, tuple[str, str]] = {
    94: ("BZ101", "신속통합기획"),
    95: ("BZ102", "재개발(도시정비형)"),
    96: ("BZ103", "재개발(주택정비형)"),
    97: ("BZ104", "재건축(단독)"),
    98: ("BZ105", "재건축(공동)"),
    99: ("BZ107", "주거환경개선(관리형)"),
    100: ("BZ108", "주거환경개선(정비형)"),
    101: ("BZ201", "모아타운"),
    102: ("BZ202", "가로주택정비사업"),
    103: ("BZ203", "자율주택정비사업"),
    104: ("BZ204", "소규모재건축사업"),
    105: ("BZ205", "소규모재개발사업"),
    112: ("BZ401", "재정비촉진지구"),
    113: ("BZ402", "재정비촉진구역"),
}

OUT_FIELDS = ["PRESENT_SN", "DGM_NM", "DGM_AR", "SIGNGU_SE", "PROPEL_CD", "WTNNC_SN"]

# 연속지적도. WFS 서비스 쪽에 있고 PNU 로 바로 조회된다 —
# data/complexes.csv.gz 가 단지마다 PNU 를 갖고 있어 그대로 조인된다.
WFS_MAPSERVER = "http://98.33.2.225:6080/arcgis/rest/services/UPIS/20200526_WFS/MapServer"
PARCEL_LAYER = 1
PARCEL_FIELDS = ["PNU", "JIBUN", "SPACE_AREA", "JIGA", "JIMOK", "OWN"]

# 용도지역. 허용 용적률 상한을 여기서 얻는다.
LANDUSE_LAYER = 123
LANDUSE_FIELDS = ["ATRB_SE", "DGM_NM", "SIGNGU_SE"]

LANDUSE_LABELS = {
    "UQA111": "제1종전용주거지역",
    "UQA112": "제2종전용주거지역",
    "UQA119": "미분류전용주거지역",
    "UQA121": "제1종일반주거지역",
    "UQA122": "제2종일반주거지역",
    "UQA123": "제3종일반주거지역",
    "UQA124": "제2종일반주거지역(7층이하)",
    "UQA129": "미분류일반주거지역",
    "UQA130": "준주거지역",
    "UQA190": "기타주거지역",
    "UQA210": "중심상업지역",
    "UQA220": "일반상업지역",
    "UQA230": "근린상업지역",
    "UQA240": "유통상업지역",
    "UQA290": "기타상업지역",
    "UQA310": "전용공업지역",
    "UQA320": "일반공업지역",
    "UQA330": "준공업지역",
    "UQA390": "기타공업지역",
    "UQA410": "보전녹지지역",
    "UQA420": "생산녹지지역",
    "UQA430": "자연녹지지역",
    "UQA490": "기타녹지지역",
}

# 서울특별시 도시계획조례 기준 용적률 상한(%).
#
# 화면이 쓰는 상한은 이제 여기가 아니라 collect_ordinance.py 가 법제처에서 받아
# 오는 실측값이다 (서울·경기 32개 지자체, 정비사업 단서까지 반영).
# 이 표는 collect_parcels.py 가 parcels.csv.gz 에 참고용으로 남기는 값이고,
# 서울 조례 실측값과 대조해 두 값이 같은 것을 확인했다.
FAR_LIMIT = {
    "UQA111": 100,
    "UQA112": 120,
    "UQA121": 150,
    "UQA122": 200,
    "UQA123": 250,
    "UQA124": 200,
    "UQA130": 400,
    "UQA210": 1000,
    "UQA220": 800,
    "UQA230": 600,
    "UQA240": 600,
    "UQA310": 200,
    "UQA320": 200,
    "UQA330": 400,
    "UQA410": 50,
    "UQA420": 50,
    "UQA430": 50,
}

PARCEL_COLUMNS = [
    "pnu",
    "sgg_cd",
    "jibun",
    "area_sqm",
    "jiga_won_sqm",
    "jimok",
    "own",
    "lon",
    "lat",
    "landuse_cd",
    "landuse_nm",
    "far_limit",
]

ZONE_COLUMNS = [
    "layer_code",
    "bsns_se",
    "present_sn",
    "zone_name",
    "zone_name_norm",
    "area_sqm",
    "sgg_cd",
    "propel_cd",
    "wtnnc_sn",
    "lon",
    "lat",
]


class UpisError(Exception):
    """ArcGIS 가 error 객체를 돌려줬거나 응답이 예상 형태가 아닐 때."""


def query_url(layer_id: int, *, geometry: bool) -> str:
    """레이어 하나를 통째로 받는 프록시 URL.

    geometry=False 는 속성만 받아 가볍다. 폴리곤이 필요할 때만 True 로 부른다.
    outSR=4326 으로 WGS84 경위도를 요청한다 — 대시보드 투영이 경위도 기준이라
    여기서 맞춰 받지 않으면 좌표계를 직접 변환해야 한다.
    """
    params = {
        "where": "1=1",
        "outFields": ",".join(OUT_FIELDS),
        "returnGeometry": "true" if geometry else "false",
        "f": "json",
    }
    if geometry:
        params["outSR"] = "4326"
    target = f"{MAPSERVER}/{layer_id}/query?{urllib.parse.urlencode(params)}"
    # 프록시는 대상 URL 을 인코딩 없이 쿼리스트링 뒤에 그대로 붙이는 방식이다.
    return f"{PROXY}?{target}"


def parcel_query_url(pnus: list[str]) -> str:
    """필지 여러 개를 PNU IN (...) 로 한 번에 받는다.

    한 건씩 부르면 서울 아파트만 6천 콜이 넘는다. maxRecordCount 가 10000 이라
    수백 개씩 묶어도 잘린 응답이 오지 않는다.
    """
    quoted = ",".join("'" + p.replace("'", "") + "'" for p in pnus)
    params = {
        "where": f"PNU IN ({quoted})",
        "outFields": ",".join(PARCEL_FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }
    target = f"{WFS_MAPSERVER}/{PARCEL_LAYER}/query?{urllib.parse.urlencode(params)}"
    return f"{PROXY}?{target}"


def landuse_ids_url() -> str:
    """용도지역 OBJECTID 전체 목록. 8,331건이 한 응답에 다 온다.

    자치구(SIGNGU_SE)로 못 나눈다 — 이 레이어는 대부분의 폴리곤이 '11000'
    (서울시 전체)으로 태깅돼 있어 강남구로 걸러도 8건만 나온다.
    OBJECTID 도 794293~4745454 로 띄엄띄엄해 구간 조건으로 훑을 수 없다.
    그래서 ID 를 먼저 통째로 받아 놓고 명시적으로 묶어 부른다.
    """
    params = {"where": "1=1", "returnIdsOnly": "true", "f": "json"}
    target = f"{MAPSERVER}/{LANDUSE_LAYER}/query?{urllib.parse.urlencode(params)}"
    return f"{PROXY}?{target}"


def landuse_query_url(object_ids: list[int]) -> str:
    """용도지역 폴리곤을 OBJECTID 묶음으로 받는다 (maxRecordCount 1000, pagination 미지원)."""
    params = {
        "objectIds": ",".join(str(i) for i in object_ids),
        "outFields": ",".join(LANDUSE_FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }
    target = f"{MAPSERVER}/{LANDUSE_LAYER}/query?{urllib.parse.urlencode(params)}"
    return f"{PROXY}?{target}"


def parse_object_ids(payload: str | dict) -> list[int]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if "error" in data:
        raise UpisError(str(data["error"]))
    return list(data.get("objectIds") or [])


def parse_parcels(payload: str | dict) -> list[dict]:
    """필지 응답을 레코드로. 대표점은 폴리곤 무게중심을 쓴다."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if "error" in data:
        raise UpisError(str(data["error"]))
    out: list[dict] = []
    for feature in data.get("features", []):
        attrs = feature.get("attributes") or {}
        pnu = (attrs.get("PNU") or "").strip()
        if not pnu:
            continue
        rings = (feature.get("geometry") or {}).get("rings") or []
        centroid = ring_centroid(rings) if rings else None
        out.append(
            {
                "pnu": pnu,
                "sgg_cd": pnu[:5],
                "jibun": (attrs.get("JIBUN") or "").strip(),
                "area_sqm": attrs.get("SPACE_AREA") or 0,
                "jiga_won_sqm": attrs.get("JIGA") or 0,
                "jimok": (attrs.get("JIMOK") or "").strip(),
                "own": (attrs.get("OWN") or "").strip(),
                "lon": round(centroid[0], 6) if centroid else "",
                "lat": round(centroid[1], 6) if centroid else "",
            }
        )
    return out


def parse_landuse(payload: str | dict) -> list[dict]:
    """용도지역 폴리곤. 점 판정을 빨리 하려고 경계상자를 미리 붙여 둔다."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if "error" in data:
        raise UpisError(str(data["error"]))
    out: list[dict] = []
    for feature in data.get("features", []):
        attrs = feature.get("attributes") or {}
        rings = (feature.get("geometry") or {}).get("rings") or []
        if not rings:
            continue
        xs = [p[0] for ring in rings for p in ring]
        ys = [p[1] for ring in rings for p in ring]
        code = (attrs.get("ATRB_SE") or "").strip()
        out.append(
            {
                "code": code,
                "name": LANDUSE_LABELS.get(code, code),
                "sgg_cd": (attrs.get("SIGNGU_SE") or "").strip(),
                "rings": rings,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            }
        )
    return out


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    """레이 캐스팅. 경계에 정확히 걸친 점은 어느 쪽으로 가든 상관없다."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def find_landuse(lon: float, lat: float, polygons: list[dict]) -> dict | None:
    """점이 속한 용도지역을 찾는다. 첫 링은 외곽, 나머지는 구멍으로 본다."""
    for poly in polygons:
        x0, y0, x1, y1 = poly["bbox"]
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        rings = poly["rings"]
        if not _point_in_ring(lon, lat, rings[0]):
            continue
        if any(_point_in_ring(lon, lat, hole) for hole in rings[1:]):
            continue
        return poly
    return None


def ring_centroid(rings: list[list[list[float]]]) -> tuple[float, float] | None:
    """구역 대표점. 가장 큰 링의 면적 가중 무게중심을 쓴다.

    구역이 여러 조각으로 나뉘는 경우가 있어 단순 평균을 쓰면 대표점이 조각
    사이 빈 땅에 찍힌다. 가장 넓은 조각 안에 찍히도록 한다.
    """
    best: tuple[float, float, float] | None = None  # (|면적|, cx, cy)
    for ring in rings:
        if len(ring) < 3:
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
        centroid = (abs(a), cx / (6 * a), cy / (6 * a))
        if best is None or centroid[0] > best[0]:
            best = centroid
    if best is None:
        # 면적이 0 인 퇴화 폴리곤. 점 평균으로 물러선다.
        pts = [p for ring in rings for p in ring]
        if not pts:
            return None
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return best[1], best[2]


def rdp(points: list[list[float]], eps: float) -> list[list[float]]:
    """Douglas-Peucker 단순화. build_geo.rdp 와 같은 식·같은 이유(스택 방식)다."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i][0], points[i][1]
        bx, by = points[j][0], points[j][1]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        best, best_i = -1.0, -1
        for m in range(i + 1, j):
            px, py = points[m][0], points[m][1]
            if seg == 0:
                dist = math.hypot(px - ax, py - ay)
            else:
                dist = abs(dy * px - dx * py + bx * ay - by * ax) / seg
            if dist > best:
                best, best_i = dist, m
        if best > eps:
            keep[best_i] = True
            stack.append((i, best_i))
            stack.append((best_i, j))
    return [p for p, k in zip(points, keep) if k]


def simplify_rings(
    rings: list[list[list[float]]], eps: float, min_points: int = 4
) -> list[list[list[float]]]:
    """링마다 단순화하고 너무 작아진 조각은 버린다."""
    out: list[list[list[float]]] = []
    for ring in rings:
        simplified = rdp([[p[0], p[1]] for p in ring], eps)
        if len(simplified) >= min_points:
            out.append([[round(x, 6), round(y, 6)] for x, y in simplified])
    return out


def parse_zones(payload: str | dict, layer_id: int) -> list[dict]:
    """ArcGIS query 응답을 구역 레코드 목록으로 바꾼다.

    Raises:
        UpisError: 응답에 error 가 들어 있거나 features 가 없을 때.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    if "error" in data:
        raise UpisError(str(data["error"]))
    if "features" not in data:
        raise UpisError(f"features 가 없습니다: {sorted(data)[:6]}")

    code, bsns_se = ZONE_LAYERS.get(layer_id, (f"L{layer_id}", ""))
    out: list[dict] = []
    for feature in data["features"]:
        attrs = feature.get("attributes") or {}
        name = (attrs.get("DGM_NM") or "").strip()
        rings = (feature.get("geometry") or {}).get("rings") or []
        centroid = ring_centroid(rings) if rings else None
        out.append(
            {
                "layer_code": code,
                "bsns_se": bsns_se,
                "present_sn": attrs.get("PRESENT_SN") or "",
                "zone_name": name,
                "zone_name_norm": cleanup_api.normalize_zone_name(name),
                "area_sqm": attrs.get("DGM_AR") or 0,
                "sgg_cd": (attrs.get("SIGNGU_SE") or "").strip(),
                "propel_cd": (attrs.get("PROPEL_CD") or "").strip(),
                "wtnnc_sn": attrs.get("WTNNC_SN") or "",
                "lon": round(centroid[0], 6) if centroid else "",
                "lat": round(centroid[1], 6) if centroid else "",
                "rings": rings,
            }
        )
    return out
