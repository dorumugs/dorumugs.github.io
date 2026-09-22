"""수집한 Airbnb 좌표를 시군구·격자로 집계해 대시보드용 JSON 을 만든다.

    python3 scripts/build_airbnb.py
    python3 scripts/build_airbnb.py --state data/airbnb/jeju.json.gz --dry-run

만드는 것
---------
  assets/realestate/airbnb.json         시군구 집계 + 전국 격자 + 수집 메타
  assets/realestate/airbnb/<sgg>.json   시군구별 좌표. 그 지역을 볼 때만 받아간다

전국 좌표를 한 파일에 담으면 첫 화면이 무겁다. 실거래 대시보드가
`assets/realestate/sgg/<code>.json` 으로 가른 것과 같은 이유다.

공개 범위
---------
좌표만 쓴다. 숙소 ID·이름·가격·사진·링크는 `data/` 안에만 있고 여기로 나오지
않는다. 목적이 밀집도라 좌표면 충분하고, 숙소 콘텐츠를 재배포하는 모양이
되지 않는다.

색칠은 면적당 밀도로 한다
-------------------------
절대 수로 칠하면 서울 몇 개 구가 눈금을 다 먹어 나머지 전국이 한 색이 된다.
`density = count / area_km2` 로 칠하고, 절대 수는 표에서 본다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airbnb_api as api  # noqa: E402
import build_dashboard  # noqa: E402
import upis_api  # noqa: E402

STATE_FILE = ROOT / "data" / "airbnb" / "state.json.gz"
GEO_FILE = ROOT / "data" / "geo" / "sgg_kr.geojson.gz"
PROJECTION_FILE = ROOT / "data" / "geo" / "projection_kr.json"
OUT_FILE = ROOT / "assets" / "realestate" / "airbnb.json"
POINTS_DIR = ROOT / "assets" / "realestate" / "airbnb"

# 격자 한 칸. 0.005도는 위도로 약 550m, 경도로 약 460m(위도 37도)다.
GRID = 0.005

# 좌표 소수 자릿수. Airbnb 가 주는 값 자체가 소수 4~5자리 근사 좌표다.
NDIGITS = 5

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320


def load_polygons(geo: dict) -> list[dict]:
    """GeoJSON 을 점 판정에 쓰기 좋은 모양으로 바꾼다.

    링마다 bbox 를 미리 재 둔다. 좌표 하나마다 256개 시군구의 모든 변을 훑으면
    전국 수만 건에서 몇 분씩 걸린다 — bbox 로 먼저 걸러 내면 거의 다 튕겨난다.
    """
    out: list[dict] = []
    for feature in geo.get("features", []):
        props = feature.get("properties") or {}
        code = props.get("sgg")
        geometry = feature.get("geometry") or {}
        if not code or geometry.get("type") != "MultiPolygon":
            continue
        rings: list[dict] = []
        area = 0.0
        for polygon in geometry.get("coordinates") or []:
            if not polygon or not polygon[0]:
                continue
            ring = polygon[0]
            lons = [pt[0] for pt in ring]
            lats = [pt[1] for pt in ring]
            rings.append({"ring": ring,
                          "bbox": (min(lons), min(lats), max(lons), max(lats))})
            area += ring_area_km2(ring)
        out.append({"sgg": code, "name": props.get("name") or code,
                    "rings": rings, "area_km2": area})
    return out


def find_sgg(lat: float, lng: float, areas: list[dict]) -> str | None:
    """좌표가 속한 시군구 코드. 어디에도 안 들면 None.

    GeoJSON 좌표는 `[경도, 위도]` 순이다 — 뒤집어 넣으면 전국이 조용히 어긋난다.
    """
    for area in areas:
        for part in area["rings"]:
            x0, y0, x1, y1 = part["bbox"]
            if not (x0 <= lng <= x1 and y0 <= lat <= y1):
                continue
            if upis_api._point_in_ring(lng, lat, part["ring"]):  # noqa: SLF001
                return area["sgg"]
    return None


def ring_area_km2(ring: list[list[float]]) -> float:
    """링의 면적(km²). 등장방형 근사 — 시군구 크기에서는 오차가 눈에 안 띈다."""
    if len(ring) < 3:
        return 0.0
    lats = [pt[1] for pt in ring]
    mean_lat = sum(lats) / len(lats)
    scale_x = KM_PER_DEG_LON * math.cos(math.radians(mean_lat))
    total = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][0] * scale_x, ring[i][1] * KM_PER_DEG_LAT
        x2, y2 = ring[(i + 1) % len(ring)][0] * scale_x, ring[(i + 1) % len(ring)][1] * KM_PER_DEG_LAT
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def sgg_bboxes(areas: list[dict]) -> dict[str, list[float]]:
    """시군구별 경계 상자 `[위도최소, 위도최대, 경도최소, 경도최대]`.

    화면에서 시도를 고르면 그 범위의 격자 칸만 그린다 — 없으면 확대했을 때
    옆 시도 점이 끼어든다.
    """
    out: dict[str, list[float]] = {}
    for area in areas:
        if not area["rings"]:
            continue
        x0 = min(part["bbox"][0] for part in area["rings"])
        y0 = min(part["bbox"][1] for part in area["rings"])
        x1 = max(part["bbox"][2] for part in area["rings"])
        y1 = max(part["bbox"][3] for part in area["rings"])
        out[area["sgg"]] = [y0, y1, x0, x1]
    return out


def is_weak(lat: float, lng: float, weak: list[list[float]], size: float) -> bool:
    """이 격자 칸이 '상한에 걸려 덜 걷힌' bbox 와 겹치는가.

    칸의 원점이 bbox 안에 드는지만 보면 안 된다. 덜 걷힌 bbox 는 분할 하한인
    400m(`MIN_DEG`)이고 격자 칸은 500m 라, **bbox 가 칸보다 작아** 칸의 남서쪽
    꼭짓점을 품지 못하는 일이 흔하다 — 실측으로 bbox 13개 중 6개만 잡혔다.
    겹치기만 하면 그 칸은 실제보다 적게 잡힌 칸이다.
    """
    top, right = lat + size, lng + size
    for ne_lat, ne_lng, sw_lat, sw_lng in weak:
        if sw_lat <= top and ne_lat >= lat and sw_lng <= right and ne_lng >= lng:
            return True
    return False


def points_by_sgg(points, areas: list[dict]) -> dict[str, list[list[float]]]:
    """좌표를 시군구별로 가른다. 어디에도 안 드는 좌표는 버린다."""
    out: dict[str, list[list[float]]] = {}
    for lat, lng in points:
        code = find_sgg(lat, lng, areas)
        if code:
            out.setdefault(code, []).append(
                [round(lat, NDIGITS), round(lng, NDIGITS)])
    return {code: sorted(pts) for code, pts in sorted(out.items())}


def aggregate(points, areas: list[dict], grid: float = GRID,
              weak: list[list[float]] | None = None) -> dict:
    """시군구 집계와 격자 집계를 한 번에 낸다.

    격자 칸은 `[위도, 경도, 개수, 덜걷힘]` 이다. 덜걷힘이 1 인 칸은 상한에 걸려
    실제보다 적게 잡힌 곳이라 화면에서 그렇게 밝힌다.
    """
    weak = weak or []
    counts: dict[str, int] = {area["sgg"]: 0 for area in areas}
    cells: dict[tuple[float, float], int] = {}
    total = placed = 0

    for lat, lng in points:
        total += 1
        code = find_sgg(lat, lng, areas)
        if code is None:
            continue
        placed += 1
        counts[code] += 1
        key = api.snap(lat, lng, grid)
        cells[key] = cells.get(key, 0) + 1

    sgg = {}
    for area in areas:
        area_km2 = area["area_km2"]
        count = counts[area["sgg"]]
        sgg[area["sgg"]] = {
            "name": area["name"],
            "count": count,
            "area_km2": round(area_km2, 2),
            # 반올림하지 않는다. 자릿수를 줄이면 넓고 한산한 군에서 밀도가
            # 0 으로 뭉개져 지도에서 서로 구분이 안 된다. 256개뿐이라 용량은
            # 문제가 되지 않는다.
            "density": count / area_km2 if area_km2 > 0 else 0.0,
        }

    return {
        "sgg": sgg,
        "sgg_bbox": sgg_bboxes(areas),
        "grid": [[lat, lng, n, 1 if is_weak(lat, lng, weak, grid) else 0]
                 for (lat, lng), n in sorted(cells.items())],
        "meta": {"total": total, "placed": placed, "dropped": total - placed,
                 "grid_deg": grid, "weak_boxes": len(weak)},
    }


def load_state(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def load_geojson(path: Path) -> dict:
    """시군구 경계를 읽는다. build_geo.py 가 gzip 으로 쓴다(전국본이 6MB 라서)."""
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def state_points(state: dict) -> list[tuple[float, float]]:
    """상태 파일에서 화면에 쓸 좌표를 꺼낸다.

    확정 스냅샷이 있으면 그것, 없으면(첫 바퀴 진행 중) 모으는 중인 것.
    """
    out = []
    for key in api.published_points(state):
        lat, _, lng = key.partition(",")
        try:
            out.append((float(lat), float(lng)))
        except ValueError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE_FILE)
    parser.add_argument("--geo", type=Path, default=GEO_FILE)
    parser.add_argument("--projection", type=Path, default=PROJECTION_FILE)
    parser.add_argument("--grid", type=float, default=GRID)
    parser.add_argument("--generated", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    args = parser.parse_args()

    if not args.state.exists():
        print(f"수집 상태가 없습니다: {args.state}", file=sys.stderr)
        return 1
    if not args.geo.exists() or not args.projection.exists():
        print(f"전국 경계·투영이 없습니다: {args.geo}, {args.projection}. "
              "build_geo.py --sido all --suffix _kr 을 먼저 돌리세요.", file=sys.stderr)
        return 1

    state = load_state(args.state)
    points = state_points(state)
    if not points:
        print("좌표가 없습니다 — 기존 집계를 건드리지 않고 끝냅니다.", file=sys.stderr)
        return 1

    areas = load_polygons(load_geojson(args.geo))
    complete = api.snapshot_complete(state)
    weak = state.get("weak") if complete else state.get("weak_pending")
    result = aggregate(points, areas, args.grid, weak)
    # 투영 파라미터를 같이 실어 보낸다. 화면의 캔버스 점 레이어가 위경도를
    # map_kr.svg 와 같은 좌표계로 옮기는 데 쓴다 — 별도 요청을 만들지 않으려고
    # 여기 넣는다(data/ 는 수집물 디렉터리라 웹에서 받아 쓰지 않는다).
    result["projection"] = json.loads(args.projection.read_text(encoding="utf-8"))
    # check_freshness.py 는 최상위 `generated` 를 본다. meta 안에도 같은 값을
    # 두지만, 검사기가 보는 자리는 여기다.
    result["generated"] = args.generated
    result["meta"].update({
        "generated": args.generated,
        "collected": (state.get("completed_at") if complete
                      else state.get("updated")) or state.get("updated"),
        "complete": complete,
        "source": "airbnb.co.kr 지도검색",
        "pass_in_progress": len(state.get("frontier") or []) > 0,
    })

    meta = result["meta"]
    print(f"좌표 {meta['total']} → 시군구 배정 {meta['placed']}, "
          f"버림 {meta['dropped']} ({meta['dropped'] / meta['total']:.1%})")
    print(f"격자 {len(result['grid'])}칸, 덜걷힌 bbox {meta['weak_boxes']}개, "
          f"완주={meta['complete']}")
    top = sorted(result["sgg"].items(), key=lambda kv: -kv[1]["count"])[:10]
    for code, entry in top:
        print(f"  {code} {entry['name']:12s} {entry['count']:6d}건  "
              f"{entry['density']:7.2f}건/km²")

    if args.dry_run:
        return 0

    changed = build_dashboard.write_json(OUT_FILE, result)
    print(f"{OUT_FILE.relative_to(ROOT)} {'갱신' if changed else '그대로'} "
          f"({OUT_FILE.stat().st_size:,}B)")

    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = points_by_sgg(points, areas)
    written = 0
    for code, pts in groups.items():
        if build_dashboard.write_json(POINTS_DIR / f"{code}.json",
                                      {"sgg": code, "points": pts}):
            written += 1
    # 이번에 숙소가 없어진 시군구의 옛 파일은 지운다 — 남겨 두면 화면이 옛 점을 그린다.
    stale = 0
    for old in POINTS_DIR.glob("*.json"):
        if old.stem not in groups:
            old.unlink()
            stale += 1
    print(f"{POINTS_DIR.relative_to(ROOT)}/ 시군구 {len(groups)}개 "
          f"(갱신 {written}, 삭제 {stale})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
