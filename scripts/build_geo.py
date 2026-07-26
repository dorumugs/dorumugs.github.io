"""행정동 경계 GeoJSON 을 시군구 단위로 병합해 대시보드용 SVG 를 만든다.

원본: vuski/admdongkor ver20260701 (약 34MB). 저장소에 넣지 않고 결과물만 커밋한다.

    python3 scripts/build_geo.py --input /tmp/geo/hjd.geojson

서울/경기 탭은 같은 SVG 의 viewBox 를 바꿔 구현하므로 지도는 한 장만 만든다.
따라서 단순화 강도는 가장 확대되는 뷰(서울) 기준으로 잡아야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import regions  # noqa: E402

GEO_DIR = ROOT / "data" / "geo"
GEO_FILE = GEO_DIR / "sgg_seoul_gyeonggi.geojson"
SVG_FILE = ROOT / "_includes" / "realestate" / "map.svg"

SIDO = ("11", "41")
SVG_WIDTH = 1000.0

# 원본 SVG 예산. 넘으면 실패시킨다. gzip 후 대략 1/4 로 줄어든다.
MAX_SVG_BYTES = 250 * 1024

Ring = list[list[float]]
Point = tuple[float, float]


def merge_sgg(features: list[dict]) -> dict[str, list[Ring]]:
    """행정동 feature 목록을 시군구 5자리로 묶는다. 외곽 링만 남긴다.

    구멍(내부 링)은 시군구 경계에서는 의미가 없어 버린다. 시도 11/41 밖은 제외.
    """
    out: dict[str, list[Ring]] = {}
    for f in features:
        props = f.get("properties") or {}
        if props.get("sido") not in SIDO:
            continue
        code = props.get("sgg")
        geom = f.get("geometry")
        if not code or not geom:
            continue
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["coordinates"]
        else:
            continue
        for poly in polygons:
            if poly and poly[0]:
                out.setdefault(code, []).append(poly[0])
    return out


ROUND_NDIGITS = 7  # 위경도 7자리 ~= 1cm. 부동소수점 잡음 없이 정점을 매칭하기 위한 키.


def _round_pt(pt: list[float] | Point) -> Point:
    return (round(pt[0], ROUND_NDIGITS), round(pt[1], ROUND_NDIGITS))


def _ring_area2(ring: list[Point]) -> float:
    """부호 있는 면적의 2배(신발끈 공식). 양수면 반시계, 음수면 시계 방향."""
    n = len(ring)
    total = 0.0
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def _open_ccw_ring(ring: Ring) -> list[Point]:
    """닫힌 링(첫점==끝점)을 열고, 좌표를 반올림한 뒤 반시계 방향으로 통일한다."""
    pts = [_round_pt(p) for p in ring]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) >= 3 and _ring_area2(pts) < 0:
        pts.reverse()
    return pts


def _drop_exact_collinear(ring: list[Point]) -> list[Point]:
    """스티칭 직후 정확히 일직선인 통과점을 제거한다.

    상쇄된 변의 양 끝점(원래 이웃 행정동 경계의 접점)은 진짜 꼭짓점이 아니라
    합쳐진 변 위의 통과점일 수 있다. 이건 나중에 하는 eps 기반 단순화와는
    별개로, 오차 없이 딱 일직선인 점만 정리하는 단계다.
    """
    n = len(ring)
    if n < 3:
        return ring
    keep: list[Point] = []
    for i in range(n):
        ax, ay = ring[i - 1]
        bx, by = ring[i]
        cx, cy = ring[(i + 1) % n]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross != 0:
            keep.append(ring[i])
    return keep if len(keep) >= 3 else ring


def dissolve(rings: list[Ring]) -> list[Ring]:
    """한 시군구에 속한 행정동 외곽 링들을, 겹치는 경계를 상쇄해 하나로 합친다.

    핵심 아이디어(정확한 변 상쇄): 모든 링을 반시계 방향으로 통일하면, 이웃한
    두 행정동이 공유하는 변은 한쪽에서는 A->B, 다른 쪽에서는 B->A 로 정반대
    방향으로 나타난다. 방향이 정확히 반대인 변끼리 상쇄해 지우면, 남는 변은
    시군구의 진짜 외곽(과 진짜 구멍·떨어진 섬)뿐이다. 좌표는 투영·단순화 전
    원본 위경도 상태여야 한다 — RDP 로 단순화하면 꼭짓점이 미세하게 어긋나서
    더 이상 정확히 매칭되지 않기 때문이다.

    남은 변은 끝점을 따라가며 이어붙여(스티칭) 닫힌 링들로 복원한다.
    """
    edges: list[tuple[Point, Point]] = []
    for ring in rings:
        pts = _open_ccw_ring(ring)
        n = len(pts)
        if n < 3:
            continue
        for i in range(n):
            edges.append((pts[i], pts[(i + 1) % n]))

    counts = Counter(edges)
    result_edges: list[tuple[Point, Point]] = []
    seen_pairs: set[tuple[Point, Point]] = set()
    for (a, b), c in counts.items():
        if (a, b) in seen_pairs or (b, a) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        seen_pairs.add((b, a))
        rc = counts.get((b, a), 0)
        net = c - rc
        if net > 0:
            result_edges.extend([(a, b)] * net)
        elif net < 0:
            result_edges.extend([(b, a)] * (-net))
        # net == 0 이면 완전히 상쇄된 내부 경계 -> 버린다

    adj: dict[Point, list[Point]] = {}
    for a, b in result_edges:
        adj.setdefault(a, []).append(b)

    out: list[Ring] = []
    while any(adj.values()):
        start = next(v for v, lst in adj.items() if lst)
        ring_pts = [start]
        current = start
        while True:
            nxts = adj.get(current)
            if not nxts:
                break  # in=out 차수가 어긋나는 비정상 그래프에 대한 방어
            current = nxts.pop()
            if current == start:
                break
            ring_pts.append(current)
        cleaned = _drop_exact_collinear(ring_pts)
        if len(cleaned) >= 3:
            closed = cleaned + [cleaned[0]]
            out.append([list(p) for p in closed])
    return out


def rdp(points: list[Point], eps: float) -> list[Point]:
    """Douglas-Peucker 단순화. 재귀 대신 스택으로 돌아 깊은 링에서도 안전하다."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        best, best_i = -1.0, -1
        for m in range(i + 1, j):
            px, py = points[m]
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


def project(rings: dict[str, list[Ring]], width: float) -> tuple[dict[str, list[list[Point]]], float, float]:
    """등장방형 투영 + 위도 보정. 전체가 width 에 꽉 차도록 맞춘다.

    이 정도 면적(서울·경기)에서는 왜곡이 눈에 띄지 않아 별도 라이브러리가 필요 없다.
    y 는 위가 북쪽이 되도록 뒤집는다.
    """
    pts = [pt for rs in rings.values() for r in rs for pt in r]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    k = math.cos(math.radians((min_lat + max_lat) / 2))
    span_x = (max_lon - min_lon) * k
    span_y = max_lat - min_lat
    height = width * span_y / span_x

    def to_xy(pt: list[float]) -> Point:
        return (
            (pt[0] - min_lon) * k / span_x * width,
            (max_lat - pt[1]) / span_y * height,
        )

    out = {code: [[to_xy(pt) for pt in ring] for ring in rs] for code, rs in rings.items()}
    return out, width, height


def to_svg(projected: dict[str, list[list[Point]]], names: dict[str, str],
           width: float, height: float) -> str:
    """시군구별 path 하나씩. 색은 넣지 않는다 — 런타임에 JS 가 fill 을 칠한다."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        'class="re-map" role="img" aria-label="서울·경기 시군구 지도">'
    ]
    for code in sorted(projected):
        d = "".join(
            "M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in ring) + "Z"
            for ring in projected[code]
        )
        name = names.get(code, code)
        lines.append(f'<path id="sgg-{code}" data-sgg="{code}" data-name="{name}" d="{d}"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def simplify(projected: dict[str, list[list[Point]]], eps: float,
             min_area: float) -> dict[str, list[list[Point]]]:
    """링마다 단순화하고, 너무 작아진 조각(먼 섬·자투리)은 버린다."""
    out: dict[str, list[list[Point]]] = {}
    for code, rings in projected.items():
        kept = []
        for ring in rings:
            simple = rdp(ring, eps)
            if len(simple) < 4:
                continue
            area = abs(sum(
                simple[i][0] * simple[i - 1][1] - simple[i - 1][0] * simple[i][1]
                for i in range(len(simple))
            )) / 2
            if area < min_area:
                continue
            kept.append(simple)
        out[code] = kept
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="행정동 GeoJSON 경로")
    parser.add_argument("--eps", type=float, default=1.0,
                        help="단순화 강도. 서울 뷰 기준으로 정한다")
    parser.add_argument("--min-area", type=float, default=4.0)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    merged = merge_sgg(data["features"])

    expected = {code: name for code, name in regions.sgg_codes()}
    if set(merged) != set(expected):
        missing = sorted(set(expected) - set(merged))
        extra = sorted(set(merged) - set(expected))
        print(f"시군구 불일치. 누락={missing} 잉여={extra}", file=sys.stderr)
        return 1

    # 투영·단순화 전에 원본 위경도 상태에서 행정동 경계를 시군구 외곽으로 합친다.
    dissolved = {code: dissolve(rings) for code, rings in merged.items()}

    projected, w, h = project(dissolved, SVG_WIDTH)
    projected = simplify(projected, args.eps, args.min_area)

    empty = sorted(c for c, rings in projected.items() if not rings)
    if empty:
        print(f"단순화 후 비어버린 시군구={empty}. eps 를 낮추세요.", file=sys.stderr)
        return 1

    names = {code: name.split(" ")[-1] if code.startswith("11") else
             " ".join(name.split(" ")[1:]) for code, name in expected.items()}

    svg = to_svg(projected, names, w, h)
    if len(svg.encode("utf-8")) > MAX_SVG_BYTES:
        print(f"SVG 가 예산({MAX_SVG_BYTES}B)을 넘었습니다: {len(svg.encode('utf-8'))}B",
              file=sys.stderr)
        return 1

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    SVG_FILE.parent.mkdir(parents=True, exist_ok=True)

    geo = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"sgg": code, "name": names[code]},
             "geometry": {"type": "MultiPolygon",
                          "coordinates": [[[[round(c, 5) for c in pt] for pt in ring]]
                                          for ring in dissolved[code]]}}
            for code in sorted(merged)
        ],
    }
    GEO_FILE.write_text(
        json.dumps(geo, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    SVG_FILE.write_text(svg, encoding="utf-8")

    subpaths = sum(len(rings) for rings in projected.values())
    print(f"시군구 {len(projected)}개, 서브패스 {subpaths}개, "
          f"SVG {len(svg.encode('utf-8')):,}B, viewBox {w:.0f}x{h:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
