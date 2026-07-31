#!/usr/bin/env python3
"""서울 정비구역 도형(경계·면적·추진단계)을 UPIS 에서 받아 저장한다.

레이어가 14개뿐이고 레이어당 한 번에 다 받으므로 예산 관리가 필요 없다.
매일 돌릴 필요도 없다 — 구역 지정·변경은 드물다. 주 1회면 충분하다.

출력: data/zones/zones.csv.gz     구역 속성 + 대표점(경위도)
      data/zones/zones.geojson.gz 단순화한 구역 경계

    python3 scripts/collect_zones.py            # 전체
    python3 scripts/collect_zones.py --no-geometry   # 속성만 (빠름)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_api  # noqa: E402
import upis_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZONES_DIR = ROOT / "data" / "zones"

# 경위도 기준 단순화 강도. 1e-5 도는 대략 1m 다. 구역은 시군구 경계보다 훨씬
# 작아 build_geo 의 --eps 보다 촘촘해야 모양이 남는다.
DEFAULT_EPS = 3e-5


def _request(url: str, retries: int = 3) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KayserDocs-realestate/1.0)",
            "Referer": upis_api.REFERER,
        },
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def _write(path: Path, data: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def zones_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=upis_api.ZONE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["layer_code"], r["present_sn"])):
        writer.writerow({c: row.get(c, "") for c in upis_api.ZONE_COLUMNS})
    return buf.getvalue()


def zones_to_geojson(rows: list[dict], eps: float) -> str:
    features = []
    for row in sorted(rows, key=lambda r: (r["layer_code"], r["present_sn"])):
        rings = upis_api.simplify_rings(row.get("rings") or [], eps)
        if not rings:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "present_sn": row["present_sn"],
                    "layer_code": row["layer_code"],
                    "bsns_se": row["bsns_se"],
                    "zone_name": row["zone_name"],
                    "sgg_cd": row["sgg_cd"],
                    "propel_cd": row["propel_cd"],
                    "area_sqm": row["area_sqm"],
                },
                # ArcGIS 의 rings 는 GeoJSON Polygon 의 coordinates 와 같은 구조다
                # (첫 링이 외곽, 나머지가 구멍). 감는 방향 규약은 다르지만
                # 우리 렌더러는 방향을 보지 않는다.
                "geometry": {"type": "Polygon", "coordinates": rings},
            }
        )
    return json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False, separators=(",", ":")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="폴리곤 단순화 강도(도)")
    parser.add_argument("--no-geometry", action="store_true", help="속성만 받는다")
    parser.add_argument("--sleep", type=float, default=0.5, help="레이어 사이 대기 초")
    args = parser.parse_args()

    want_geometry = not args.no_geometry
    all_rows: list[dict] = []
    failed: list[str] = []

    for layer_id, (code, name) in sorted(upis_api.ZONE_LAYERS.items()):
        url = upis_api.query_url(layer_id, geometry=want_geometry)
        try:
            rows = upis_api.parse_zones(_request(url), layer_id)
        except (upis_api.UpisError, urllib.error.URLError, OSError) as exc:
            failed.append(f"{code}({name}): {exc}")
            print(f"  {code} {name:20s} 실패: {exc}", file=sys.stderr)
            continue
        with_geom = sum(1 for r in rows if r.get("rings"))
        print(f"  {code} {name:20s} {len(rows):4d}건 (경계 {with_geom}건)")
        all_rows.extend(rows)
        if args.sleep:
            time.sleep(args.sleep)

    if not all_rows:
        print("구역을 하나도 받지 못했습니다. 프록시가 막혔을 수 있습니다.", file=sys.stderr)
        return 1

    changed = 0
    if _write(ZONES_DIR / "zones.csv.gz", cleanup_api.gzip_bytes(zones_to_csv(all_rows))):
        changed += 1
    if want_geometry:
        geojson = zones_to_geojson(all_rows, args.eps)
        if _write(ZONES_DIR / "zones.geojson.gz", cleanup_api.gzip_bytes(geojson)):
            changed += 1
        print(f"\nGeoJSON {len(geojson) / 1024:.0f}KB (단순화 전 대비 압축은 gzip 이 처리)")

    sgg_count = len({r["sgg_cd"] for r in all_rows if r["sgg_cd"]})
    print(f"구역 {len(all_rows)}건 · 자치구 {sgg_count}개 · 파일 {changed}개 갱신")
    if failed:
        print(f"실패 레이어 {len(failed)}개:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
