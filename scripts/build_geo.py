"""행정동 경계 GeoJSON 을 시군구 단위로 병합해 대시보드용 SVG 를 만든다.

원본: vuski/admdongkor ver20260701 (약 34MB). 저장소에 넣지 않고 결과물만 커밋한다.

    python3 scripts/build_geo.py --input /tmp/geo/hjd.geojson --eps 0.05

서울/경기 탭은 같은 SVG 의 viewBox 를 바꿔 구현하므로 지도는 한 장만 만든다.
따라서 단순화 강도는 가장 확대되는 뷰(서울) 기준으로 잡아야 한다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import regions  # noqa: E402

GEO_DIR = ROOT / "data" / "geo"
GEO_FILE = GEO_DIR / "sgg_seoul_gyeonggi.geojson.gz"
SVG_FILE = ROOT / "_includes" / "realestate" / "map.svg"
PROJECTION_FILE = GEO_DIR / "projection.json"

SIDO = ("11", "41")
SVG_WIDTH = 1000.0

# 원본 SVG 예산. 넘으면 실패시킨다. gzip 후 대략 1/4 로 줄어든다.
MAX_SVG_BYTES = 250 * 1024

Ring = list[list[float]]
Point = tuple[float, float]


def merge_sgg(features: list[dict],
              sido: tuple[str, ...] | None = SIDO) -> dict[str, list[Ring]]:
    """행정동 feature 목록을 시군구 5자리로 묶는다. 외곽 링만 남긴다.

    구멍(내부 링)은 시군구 경계에서는 의미가 없어 버린다.

    `sido` 기본값은 서울·경기다 — 커밋된 `map.svg` 를 그대로 재현해야 하므로
    바꾸지 말 것. 전국 지도를 만들 때만 `None` 을 준다.
    """
    out: dict[str, list[Ring]] = {}
    for f in features:
        props = f.get("properties") or {}
        if sido is not None and props.get("sido") not in sido:
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


def sgg_names(features: list[dict]) -> dict[str, str]:
    """GeoJSON 에서 시군구 이름을 뽑는다.

    전국에는 `regions.py`(서울·경기 전용 코드표)가 없어서 데이터에 실려 온
    `sggnm` 을 쓴다. 서울·경기 기본 빌드는 이 함수를 쓰지 않는다 — `sggnm` 은
    '수원시장안구' 처럼 공백이 없어서 커밋된 지도의 '수원시 장안구' 와 다르다.
    """
    out: dict[str, str] = {}
    for f in features:
        props = f.get("properties") or {}
        code = props.get("sgg")
        if code and code not in out:
            out[code] = (props.get("sggnm") or "").strip() or code
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


def projection_params(rings: dict[str, list[Ring]], width: float) -> dict[str, float]:
    """투영에 필요한 상수를 계산한다.

    학교 점처럼 나중에 같은 지도 위에 올릴 좌표가 이 값을 그대로 써야 한다.
    두 곳에서 각자 계산하면 경계 데이터나 --eps 를 갱신할 때 조용히 어긋난다.
    """
    pts = [pt for rs in rings.values() for r in rs for pt in r]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    k = math.cos(math.radians((min_lat + max_lat) / 2))
    span_x = (max_lon - min_lon) * k
    span_y = max_lat - min_lat
    return {
        "min_lon": min_lon,
        "max_lat": max_lat,
        "k": k,
        "span_x": span_x,
        "span_y": span_y,
        "width": width,
        "height": width * span_y / span_x,
    }


def project(rings: dict[str, list[Ring]], width: float) -> tuple[dict[str, list[list[Point]]], float, float]:
    """등장방형 투영 + 위도 보정. 전체가 width 에 꽉 차도록 맞춘다.

    이 정도 면적(서울·경기)에서는 왜곡이 눈에 띄지 않아 별도 라이브러리가 필요 없다.
    y 는 위가 북쪽이 되도록 뒤집는다.
    """
    p = projection_params(rings, width)

    def to_xy(pt: list[float]) -> Point:
        return (
            (pt[0] - p["min_lon"]) * p["k"] / p["span_x"] * p["width"],
            (p["max_lat"] - pt[1]) / p["span_y"] * p["height"],
        )

    out = {code: [[to_xy(pt) for pt in ring] for ring in rs] for code, rs in rings.items()}
    return out, p["width"], p["height"]


DEFAULT_LABEL = "서울·경기 시군구 지도"


def to_xy(lon: float, lat: float, params: dict) -> Point:
    """projection.json 파라미터 하나로 점을 투영한다. project() 와 같은 식이다.

    지도 위에 나중에 얹는 것들(학교 점, 행정동 경계)은 반드시 이 함수를 거쳐야
    한다 — 각자 계산하면 경계 데이터나 --eps 를 갱신할 때 조용히 어긋난다.
    """
    return (
        (lon - params["min_lon"]) * params["k"] / params["span_x"] * params["width"],
        (params["max_lat"] - lat) / params["span_y"] * params["height"],
    )


def _ring_area(ring: list[Point]) -> float:
    return abs(sum(ring[i][0] * ring[i - 1][1] - ring[i - 1][0] * ring[i][1]
                   for i in range(len(ring)))) / 2


def dong_paths(features: list[dict], params: dict, eps: float,
               min_area: float) -> dict[str, list[dict]]:
    """행정동을 시군구별로 묶어 SVG path 문자열로 낸다.

    좌표계는 `params` 가 가리키는 지도(전국이면 map_kr.svg)와 같다. 그래야
    같은 viewBox 에 그대로 얹힌다.

    **동 하나가 통째로 사라지는 일은 없다.** `min_area` 로 자투리 섬은 버리되,
    그러다 남는 게 없으면 가장 큰 링 하나는 남긴다 — 동이 빠지면 그 구에
    구멍이 뚫리고, 화면에서는 '그런 동이 없다' 로 읽힌다.
    """
    out: dict[str, list[dict]] = {}
    for feature in features:
        props = feature.get("properties") or {}
        sgg = props.get("sgg")
        geometry = feature.get("geometry") or {}
        # Polygon 과 MultiPolygon 을 둘 다 받는다. 전국 3,558개 중 울릉군 서면
        # 하나만 Polygon 인데, MultiPolygon 만 받으면 그 동이 조용히 사라진다
        # (실제로 사라졌다). merge_sgg 도 같은 이유로 둘 다 받는다.
        if geometry.get("type") == "Polygon":
            polygons = [geometry.get("coordinates") or []]
        elif geometry.get("type") == "MultiPolygon":
            polygons = geometry.get("coordinates") or []
        else:
            continue
        if not sgg:
            continue
        rings: list[list[Point]] = []
        for polygon in polygons:
            if not polygon or not polygon[0]:
                continue
            raw = [to_xy(pt[0], pt[1], params) for pt in polygon[0]]
            simple = rdp(raw, eps)
            # 단순화가 링을 못 쓰게 뭉갰으면 원본을 쓴다. 아주 작은 동에서
            # 일어나는데, 버리면 그 동이 지도에서 통째로 사라진다.
            if len(simple) < 3:
                simple = raw
            if len(simple) >= 3:
                rings.append(simple)
        if not rings:
            continue
        kept = [r for r in rings if _ring_area(r) >= min_area]
        if not kept:
            kept = [max(rings, key=_ring_area)]
        name = (props.get("adm_nm") or "").split()
        out.setdefault(sgg, []).append({
            "code": props.get("adm_cd2") or "",
            "name": name[-1] if name else (props.get("adm_cd2") or ""),
            "d": "".join("M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in r) + "Z"
                         for r in kept),
        })
    return {sgg: sorted(items, key=lambda d: d["code"])
            for sgg, items in sorted(out.items())}


def to_svg(projected: dict[str, list[list[Point]]], names: dict[str, str],
           width: float, height: float, label: str = DEFAULT_LABEL) -> str:
    """시군구별 path 하나씩. 색은 넣지 않는다 — 런타임에 JS 가 fill 을 칠한다."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'class="re-map" role="img" aria-label="{label}">'
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="행정동 GeoJSON 경로")
    # 0.05 는 dissolve 도입 후 커밋된 지도(225,350B, 88 서브패스)를 그대로
    # 재현하는 값이다. 기본값을 바꾸면 이 주석과 커밋된 SVG 도 함께 갱신할 것.
    parser.add_argument("--eps", type=float, default=0.05,
                        help="단순화 강도. 서울 뷰 기준으로 정한다")
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument("--sido", default=",".join(SIDO),
                        help="쉼표로 구분한 시도 코드, 또는 'all'. 기본은 서울·경기")
    parser.add_argument("--suffix", default="",
                        help="산출물 이름 뒤에 붙일 꼬리표. 전국은 '_kr'")
    parser.add_argument("--label", default=DEFAULT_LABEL,
                        help="SVG 의 aria-label. 스크린리더가 읽는다")
    parser.add_argument("--max-bytes", type=int, default=MAX_SVG_BYTES,
                        help="SVG 바이트 예산")
    parser.add_argument("--dong", type=Path, metavar="DIR",
                        help="행정동 경계를 시군구별 JSON 으로 이 디렉터리에 낸다. "
                             "지도와 같은 좌표계라 같은 viewBox 에 그대로 얹힌다")
    # 0.05 사용자 단위는 약 21m 다. 시군구 하나로 20배쯤 확대했을 때 화면에서
    # 1px 이라 눈에 띄지 않으면서, 시군구당 중위 9.4KB 로 가볍다(실측).
    parser.add_argument("--dong-eps", type=float, default=0.05)
    parser.add_argument("--dong-min-area", type=float, default=0.05)
    return parser


def outputs(suffix: str) -> tuple[Path, Path, Path]:
    """(SVG, projection, geojson) 경로. 꼬리표가 없으면 기존 이름 그대로."""
    if not suffix:
        return SVG_FILE, PROJECTION_FILE, GEO_FILE
    return (SVG_FILE.with_name(f"map{suffix}.svg"),
            GEO_DIR / f"projection{suffix}.json",
            GEO_DIR / f"sgg{suffix}.geojson.gz")


def main() -> int:
    args = build_parser().parse_args()

    nationwide = args.sido.strip().lower() == "all"
    sido = None if nationwide else tuple(s.strip() for s in args.sido.split(",") if s.strip())

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    merged = merge_sgg(data["features"], sido=sido)
    if not merged:
        print(f"시도 {args.sido} 에 해당하는 행정동이 없습니다.", file=sys.stderr)
        return 1

    # 서울·경기 기본 빌드는 수집기의 코드표와 정확히 맞아야 한다 — 실거래
    # 대시보드가 같은 시군구 집합을 쓰기 때문이다. 전국에는 그 코드표가 없어서
    # GeoJSON 의 코드·이름을 그대로 신뢰한다.
    if sido == SIDO:
        expected = {code: name for code, name in regions.sgg_codes()}
        if set(merged) != set(expected):
            missing = sorted(set(expected) - set(merged))
            extra = sorted(set(merged) - set(expected))
            print(f"시군구 불일치. 누락={missing} 잉여={extra}", file=sys.stderr)
            return 1
        names = {code: name.split(" ")[-1] if code.startswith("11") else
                 " ".join(name.split(" ")[1:]) for code, name in expected.items()}
    else:
        names = {code: name for code, name in sgg_names(data["features"]).items()
                 if code in merged}

    # 투영·단순화 전에 원본 위경도 상태에서 행정동 경계를 시군구 외곽으로 합친다.
    dissolved = {code: dissolve(rings) for code, rings in merged.items()}

    projected, w, h = project(dissolved, SVG_WIDTH)
    params = projection_params(dissolved, SVG_WIDTH)
    projected = simplify(projected, args.eps, args.min_area)

    empty = sorted(c for c, rings in projected.items() if not rings)
    if empty:
        print(f"단순화 후 비어버린 시군구={empty}. eps 를 낮추세요.", file=sys.stderr)
        return 1

    svg_file, projection_file, geo_file = outputs(args.suffix)

    svg = to_svg(projected, names, w, h, label=args.label)
    if len(svg.encode("utf-8")) > args.max_bytes:
        print(f"SVG 가 예산({args.max_bytes}B)을 넘었습니다: {len(svg.encode('utf-8'))}B. "
              "--eps 를 올리거나 --max-bytes 를 조정하세요.", file=sys.stderr)
        return 1

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    svg_file.parent.mkdir(parents=True, exist_ok=True)

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
    # gzip 으로 쓴다. 전국본은 날것으로 6.1MB 라 저장소에서 가장 큰 파일이
    # 되는데, git 히스토리는 되돌릴 수 없다(압축하면 1.6MB). mtime 을 0 으로
    # 고정해 내용이 같으면 바이트도 같게 만든다 — 안 그러면 다시 만들 때마다
    # 새 blob 이 쌓인다.
    body = (json.dumps(geo, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")
    with gzip.GzipFile(geo_file, "wb", compresslevel=9, mtime=0) as f:
        f.write(body)
    svg_file.write_text(svg, encoding="utf-8")
    if args.dong:
        dong = dong_paths(data["features"], params, args.dong_eps, args.dong_min_area)
        dong = {code: items for code, items in dong.items() if code in merged}
        args.dong.mkdir(parents=True, exist_ok=True)
        written = 0
        for code, items in dong.items():
            body = json.dumps({"sgg": code, "dong": items},
                              ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n"
            path = args.dong / f"{code}.json"
            data_bytes = body.encode("utf-8")
            if not path.exists() or path.read_bytes() != data_bytes:
                path.write_bytes(data_bytes)
                written += 1
        missing = sorted(set(merged) - set(dong))
        if missing:
            print(f"행정동이 없는 시군구={missing}", file=sys.stderr)
            return 1
        # 입력의 동 수와 산출물의 동 수를 대조한다. 하나가 조용히 빠지면 그
        # 구에 구멍이 뚫리는데, 화면에서는 '그런 동이 없다' 로 읽힌다 —
        # 실제로 울릉군 서면이 geometry 종류 때문에 빠진 적이 있다.
        wanted = sum(1 for f in data["features"]
                     if (f.get("properties") or {}).get("sgg") in merged)
        got = sum(len(v) for v in dong.values())
        if got != wanted:
            print(f"행정동 수가 맞지 않습니다: 입력 {wanted} → 산출 {got}",
                  file=sys.stderr)
            return 1
        total = sum(len((args.dong / f"{code}.json").read_bytes()) for code in dong)
        print(f"행정동 {sum(len(v) for v in dong.values())}개, 시군구 {len(dong)}개, "
              f"{total / 1024:.0f}KB (갱신 {written})")

    projection_file.write_text(
        json.dumps(params, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")

    subpaths = sum(len(rings) for rings in projected.values())
    print(f"시군구 {len(projected)}개, 서브패스 {subpaths}개, "
          f"SVG {len(svg.encode('utf-8')):,}B, viewBox {w:.0f}x{h:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
