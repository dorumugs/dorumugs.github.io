# 부동산 실거래가 대시보드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이미 수집된 서울·경기 아파트 실거래 435만 건을 집계해 `/real-estate/` 에 지도 기반 대시보드를 띄운다.

**Architecture:** 파이썬 빌드 스크립트가 원본 CSV.gz 를 읽어 정적 JSON 집계본과 SVG 지도를 만들고, 페이지의 바닐라 JS 가 그 JSON 만 읽어 그린다. 원본 → 집계 → 표현 3단 분리로, 지표가 바뀌면 집계만 다시 돌린다. 외부 라이브러리는 파이썬·자바스크립트 양쪽 모두 쓰지 않는다.

**Tech Stack:** Python 3.12 표준 라이브러리, ES 모듈 자바스크립트, SVG, Jekyll + minimal-mistakes

**설계 문서:** `_dev/specs/2026-07-26-real-estate-dashboard-ui-design.md`

## Global Constraints

- **파이썬은 표준 라이브러리만.** pandas·numpy·requests 금지. 기존 수집기(`scripts/rtms.py`, `scripts/regions.py`)와 같은 제약이다.
- **테스트는 stdlib `unittest`.** 이 환경에 pytest 가 없다. 실행은 `python3 -m unittest discover -s tests -v`.
- **모든 새 파이썬 파일은** `from __future__ import annotations` 로 시작하고, 함수에 타입 힌트를 달고, docstring·주석을 한국어로 쓴다. 기존 `scripts/*.py` 와 동일한 스타일이다.
- **자바스크립트 외부 라이브러리 금지.** CDN·npm 모두 사용하지 않는다.
- **결정론적 출력.** 모든 생성 파일은 같은 입력에 대해 바이트가 동일해야 한다. JSON 은 `sort_keys=True`, `separators=(",", ":")`, 부동소수는 정수 또는 소수점 이하 자리수 고정.
- **390px 가로 스크롤 금지.** `document.documentElement.scrollWidth <= 390` 을 만족해야 한다.
- **테마 원본 파일을 수정하지 않는다.** `_includes/realestate/`, `assets/realestate/` 하위 디렉터리를 새로 만들어 그 안에만 쓴다. `_layouts/`, `_sass/`, `docs/`, `CHANGELOG.md` 는 손대지 않는다.
- **색은 아래 값을 그대로 쓴다.** (dataviz 기준 팔레트, 라이트 표면 `#fcfcfb`)
  - 순차 램프 8단: `#cde2fb` `#9ec5f4` `#6da7ec` `#3987e5` `#2a78d6` `#256abf` `#184f95` `#0d366b`
  - 발산 램프 7단: `#184f95` `#2a78d6` `#86b6ef` `#f0efec` `#f0a8a8` `#d03b3b` `#a02020`
  - 라인 `#2a78d6`, 그리드 `#e1e0d9`, 축 `#c3c2b7`, 주 잉크 `#0b0b0b`, 보조 잉크 `#52514e`, 흐림 `#898781`
  - 상승 텍스트 `#006300`, 하락 텍스트 `#d03b3b`
- **평당 환산 상수는 `3.3058`.** 모든 파일에서 같은 값을 쓴다.
- **커밋 신원**은 매 커밋마다 환경변수로 지정한다 (전역 `git config` 를 바꾸지 않는다):
  ```bash
  export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
  export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
  ```
- **커밋 메시지**는 영문 한 줄 요약 + 빈 줄 + 한국어 본문 1~2줄.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `scripts/build_geo.py` | 행정동 GeoJSON → 시군구 병합 → 단순화 → SVG. 순수 함수 + `main()` |
| `scripts/aggregate.py` | 집계 순수 함수 (평당가, 중위값, 변화율, 회전율). I/O 없음 |
| `scripts/build_dashboard.py` | 원본 읽기 → 조인 → `aggregate` 호출 → JSON 쓰기 |
| `tests/test_geo.py` | 경계 병합·단순화 검증 |
| `tests/test_aggregate.py` | 집계 함수 검증 |
| `tests/test_build_dashboard.py` | 조인·출력 포맷·결정론 검증 |
| `_includes/realestate/map.svg` | 생성물. 72개 `<path>` 만 |
| `_pages/real-estate.md` | 페이지 front matter + 마크업 |
| `assets/realestate/dashboard.css` | 대시보드 전용 스타일 |
| `assets/realestate/palette.js` | 색 상수와 램프 보간. 다른 JS 가 공유 |
| `assets/realestate/data.js` | JSON fetch + 캐시 |
| `assets/realestate/map.js` | 지도 채색·탭·툴팁 |
| `assets/realestate/charts.js` | KPI 타일, 라인 차트, 단지 표 |
| `assets/realestate/app.js` | 상태 관리, URL 파라미터, 배선 |
| `assets/realestate/summary.json` | 생성물. 구 × 월 집계 |
| `assets/realestate/sgg/{code}.json` | 생성물. 구별 최근 12개월 단지 집계 |

### 설계 문서에서 한 군데 조정

설계 문서는 `sgg/{code}.json` 을 "단지 × 월 전체 기간" 으로 적었다. 그런데 1차
화면이 실제로 읽는 것은 **최근 12개월 단지 집계뿐**이다 (단지 상세 팝업은 1차
제외). 전 기간 단지×월을 내려받게 하면 구당 수백 KB 를 모바일에서 쓰지도 않고
받는다.

따라서 1차에서는 `sgg/{code}.json` 을 **최근 12개월 단지 집계로 좁힌다**
(구당 20~40KB 예상). 전 기간 단지 시계열은 단지 상세 팝업을 켜는 2차에서
`sgg/{code}-history.json` 으로 따로 만든다. 집계 로직은 Task 4 에서 기간만
파라미터로 받게 짜 두므로 2차 비용은 함수 호출 한 줄이다.

---

## Task 1: 경계 병합과 SVG 생성

**Files:**
- Create: `scripts/build_geo.py`
- Create: `tests/test_geo.py`
- Generates: `data/geo/sgg_seoul_gyeonggi.geojson`, `_includes/realestate/map.svg`

**Interfaces:**
- Consumes: `regions.sgg_codes() -> list[tuple[str, str]]` (기존 모듈, 72개 (코드, 이름))
- Produces:
  - `merge_sgg(features: list[dict]) -> dict[str, list[list[list[float]]]]` — 시군구 5자리 → 외곽 링 목록
  - `rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]`
  - `project(rings, width) -> tuple[dict[str, list], float, float]` — (코드→투영 링, 폭, 높이)
  - `to_svg(projected: dict[str, list], names: dict[str, str], w: float, h: float) -> str`

원본 GeoJSON 은 34MB 라 저장소에 넣지 않는다. 스크립트가 URL 에서 받거나
`--input` 으로 로컬 경로를 받는다. 결과물만 커밋한다.

- [ ] **Step 1: 원본 GeoJSON 을 내려받아 작업 디렉터리에 둔다**

```bash
mkdir -p /tmp/geo && curl -sL --max-time 300 \
  -o /tmp/geo/hjd.geojson \
  "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20260701/HangJeongDong_ver20260701.geojson"
ls -lh /tmp/geo/hjd.geojson
```

기대: 약 34MB.

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`tests/test_geo.py` 를 만든다. 34MB 원본에 의존하지 않도록 병합·단순화·투영은
합성 입력으로 검증하고, 실제 원본과의 대조는 파일이 있을 때만 도는 테스트로
분리한다.

```python
"""경계 병합·단순화·투영 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_geo  # noqa: E402
import regions  # noqa: E402


def _feature(sgg: str, name: str, ring: list[list[float]]) -> dict:
    return {
        "properties": {"sgg": sgg, "sggnm": name, "sido": sgg[:2]},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


class TestMergeSgg(unittest.TestCase):
    def test_merges_dong_into_sgg(self) -> None:
        square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        other = [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [2.0, 0.0]]
        feats = [
            _feature("11680", "강남구", square),
            _feature("11680", "강남구", other),
            _feature("41135", "성남시분당구", square),
        ]
        merged = build_geo.merge_sgg(feats)
        self.assertEqual(sorted(merged), ["11680", "41135"])
        self.assertEqual(len(merged["11680"]), 2)

    def test_drops_other_sido(self) -> None:
        square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        feats = [_feature("26110", "부산중구", square)]
        self.assertEqual(build_geo.merge_sgg(feats), {})

    def test_multipolygon_contributes_every_part(self) -> None:
        a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        b = [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 5.0]]
        feats = [{
            "properties": {"sgg": "41570", "sggnm": "김포시", "sido": "41"},
            "geometry": {"type": "MultiPolygon", "coordinates": [[a], [b]]},
        }]
        merged = build_geo.merge_sgg(feats)
        self.assertEqual(len(merged["41570"]), 2)

    def test_skips_null_geometry(self) -> None:
        feats = [{"properties": {"sgg": "11680", "sggnm": "강남구", "sido": "11"},
                  "geometry": None}]
        self.assertEqual(build_geo.merge_sgg(feats), {})


class TestRdp(unittest.TestCase):
    def test_collinear_points_removed(self) -> None:
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        self.assertEqual(build_geo.rdp(pts, 0.1), [(0.0, 0.0), (3.0, 0.0)])

    def test_keeps_point_beyond_eps(self) -> None:
        pts = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]
        self.assertEqual(len(build_geo.rdp(pts, 1.0)), 3)

    def test_short_input_untouched(self) -> None:
        pts = [(0.0, 0.0), (1.0, 1.0)]
        self.assertEqual(build_geo.rdp(pts, 99.0), pts)


class TestProject(unittest.TestCase):
    def test_fills_requested_width_and_flips_y(self) -> None:
        ring = [[126.0, 37.0], [127.0, 37.0], [127.0, 38.0], [126.0, 37.0]]
        proj, w, h = build_geo.project({"11680": [ring]}, 1000.0)
        xs = [p[0] for p in proj["11680"][0]]
        ys = [p[1] for p in proj["11680"][0]]
        self.assertAlmostEqual(min(xs), 0.0, places=6)
        self.assertAlmostEqual(max(xs), 1000.0, places=6)
        self.assertAlmostEqual(w, 1000.0, places=6)
        # 위도가 큰 점(북쪽)이 화면 위(y 작음)로 가야 한다
        north = proj["11680"][0][2]
        self.assertAlmostEqual(north[1], 0.0, places=6)
        self.assertGreater(h, 0.0)

    def test_latitude_correction_applied(self) -> None:
        """위도 37도에서 경도 1도는 위도 1도보다 짧다. 종횡비에 반영돼야 한다."""
        ring = [[126.0, 37.0], [127.0, 37.0], [127.0, 38.0], [126.0, 37.0]]
        _, w, h = build_geo.project({"x": [ring]}, 1000.0)
        self.assertGreater(h, w)  # 가로 1도 < 세로 1도 이므로 세로가 길다


class TestToSvg(unittest.TestCase):
    def test_emits_one_path_per_sgg_with_id_and_name(self) -> None:
        proj = {"11680": [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]]}
        svg = build_geo.to_svg(proj, {"11680": "강남구"}, 100.0, 120.0)
        self.assertIn('viewBox="0 0 100 120"', svg)
        self.assertIn('id="sgg-11680"', svg)
        self.assertIn('data-name="강남구"', svg)
        self.assertNotIn("fill=", svg)  # 색은 런타임에 칠한다

    def test_output_is_deterministic(self) -> None:
        proj = {"41135": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]],
                "11680": [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 2.0)]]}
        names = {"41135": "성남시분당구", "11680": "강남구"}
        a = build_geo.to_svg(proj, names, 10.0, 10.0)
        b = build_geo.to_svg(proj, names, 10.0, 10.0)
        self.assertEqual(a, b)
        self.assertLess(a.index("sgg-11680"), a.index("sgg-41135"))  # 코드 정렬


class TestAgainstRealSource(unittest.TestCase):
    """원본 GeoJSON 이 있을 때만 도는 대조 테스트."""

    SRC = Path("/tmp/geo/hjd.geojson")

    @unittest.skipUnless(SRC.exists(), "원본 GeoJSON 없음")
    def test_merges_to_exactly_the_collector_regions(self) -> None:
        data = json.loads(self.SRC.read_text(encoding="utf-8"))
        merged = build_geo.merge_sgg(data["features"])
        expected = {code for code, _ in regions.sgg_codes()}
        self.assertEqual(set(merged), expected)
        self.assertEqual(len(merged), 72)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_geo -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_geo'`

- [ ] **Step 4: `scripts/build_geo.py` 를 구현한다**

```python
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

    projected, w, h = project(merged, SVG_WIDTH)
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
                                          for ring in merged[code]]}}
            for code in sorted(merged)
        ],
    }
    GEO_FILE.write_text(
        json.dumps(geo, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    SVG_FILE.write_text(svg, encoding="utf-8")

    print(f"시군구 {len(projected)}개, SVG {len(svg.encode('utf-8')):,}B, "
          f"viewBox {w:.0f}x{h:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_geo -v`
Expected: PASS (원본 대조 테스트 포함 — `/tmp/geo/hjd.geojson` 이 있으므로 skip 되지 않는다)

- [ ] **Step 6: 실제로 생성하고 eps 를 정한다**

```bash
python3 scripts/build_geo.py --input /tmp/geo/hjd.geojson --eps 1.0
```

기대 출력: `시군구 72개, SVG ...B, viewBox 1000x1201`

SVG 가 250KB 를 넘으면 `--eps` 를 올리고, 서울 뷰가 각져 보이면 낮춘다.
실측 기준선: eps 2.5 → 110KB, eps 1.5 → 152KB, eps 0.6 → 254KB.
**eps 1.0 에서 시작해 250KB 예산 안에 들어오는 가장 작은 값을 고른다.**

서울 뷰를 눈으로 확인한다 (통합 투영에서 서울은 약 2.6배 확대된다):

```bash
python3 - <<'PY'
import re, pathlib
svg = pathlib.Path('_includes/realestate/map.svg').read_text(encoding='utf-8')
# 서울(11로 시작) path 의 좌표 범위로 viewBox 를 잡아 미리보기 파일을 만든다
xs, ys = [], []
for m in re.finditer(r'data-sgg="(11\d{3})" data-name="[^"]*" d="([^"]+)"', svg):
    for pt in re.findall(r'(-?\d+\.?\d*),(-?\d+\.?\d*)', m.group(2)):
        xs.append(float(pt[0])); ys.append(float(pt[1]))
pad = 8
vb = f"{min(xs)-pad:.0f} {min(ys)-pad:.0f} {max(xs)-min(xs)+2*pad:.0f} {max(ys)-min(ys)+2*pad:.0f}"
out = svg.replace('viewBox="0 0 1000 1201"', f'viewBox="{vb}"', 1)
out = out.replace('<path ', '<path fill="#cde2fb" stroke="#2a78d6" stroke-width="1" ')
pathlib.Path('/tmp/geo/seoul-preview.svg').write_text(out, encoding='utf-8')
print('viewBox', vb)
PY
```

`/tmp/geo/seoul-preview.svg` 를 열어(또는 `google-chrome --headless=new
--screenshot=/tmp/geo/seoul.png --window-size=800,700 file:///tmp/geo/seoul-preview.svg`
로 캡처해) 구 경계가 각져 보이지 않는지 확인한다.

- [ ] **Step 7: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_geo.py tests/test_geo.py data/geo _includes/realestate/map.svg
git commit -m "Add Seoul/Gyeonggi boundary merge and SVG builder

행정동 경계를 시군구 72개로 병합해 대시보드용 SVG 한 장을 만든다.
수집기 지역 코드와 일치하지 않으면 빌드를 실패시킨다."
```

---

## Task 2: 집계 순수 함수

**Files:**
- Create: `scripts/aggregate.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: 없음 (표준 라이브러리만)
- Produces:
  - `PYEONG = 3.3058`
  - `pyeong_price(price_10k: int, area_sqm: float) -> float | None` — 만원/평
  - `median(values: list[float]) -> float | None`
  - `pct_change(now: float | None, before: float | None) -> float | None`
  - `rolling_median(series: list[float | None], window: int) -> list[float | None]`
  - `from_peak(series: list[float | None], index: int) -> float | None`
  - `turnover(trade_count: int, households: int) -> float | None`
  - `area_bucket(area_sqm: float) -> int` — 0:~60, 1:60~85, 2:85~135, 3:135~
  - `THIN_SAMPLE = 5` — 이 미만이면 얇은 달

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_aggregate.py`:

```python
"""집계 순수 함수 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate  # noqa: E402


class TestPyeongPrice(unittest.TestCase):
    def test_converts_using_fixed_constant(self) -> None:
        # 84.0㎡ = 25.41평, 100,000만원 -> 3,935만원/평
        got = aggregate.pyeong_price(100_000, 84.0)
        self.assertAlmostEqual(got, 100_000 / (84.0 / 3.3058), places=6)

    def test_rejects_zero_or_negative_area(self) -> None:
        self.assertIsNone(aggregate.pyeong_price(100_000, 0.0))
        self.assertIsNone(aggregate.pyeong_price(100_000, -1.0))


class TestMedian(unittest.TestCase):
    def test_odd_count(self) -> None:
        self.assertEqual(aggregate.median([3.0, 1.0, 2.0]), 2.0)

    def test_even_count_averages_middle_pair(self) -> None:
        self.assertEqual(aggregate.median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(aggregate.median([]))


class TestPctChange(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertAlmostEqual(aggregate.pct_change(110.0, 100.0), 10.0)

    def test_negative(self) -> None:
        self.assertAlmostEqual(aggregate.pct_change(90.0, 100.0), -10.0)

    def test_none_operand_is_none(self) -> None:
        self.assertIsNone(aggregate.pct_change(None, 100.0))
        self.assertIsNone(aggregate.pct_change(100.0, None))

    def test_zero_base_is_none(self) -> None:
        self.assertIsNone(aggregate.pct_change(100.0, 0.0))


class TestRollingMedian(unittest.TestCase):
    def test_smooths_over_window(self) -> None:
        got = aggregate.rolling_median([1.0, 2.0, 3.0, 4.0], 3)
        self.assertIsNone(got[0])
        self.assertIsNone(got[1])
        self.assertEqual(got[2], 2.0)
        self.assertEqual(got[3], 3.0)

    def test_ignores_none_inside_window(self) -> None:
        got = aggregate.rolling_median([1.0, None, 3.0], 3)
        self.assertEqual(got[2], 2.0)

    def test_all_none_window_is_none(self) -> None:
        got = aggregate.rolling_median([None, None, None], 3)
        self.assertIsNone(got[2])


class TestFromPeak(unittest.TestCase):
    def test_percent_below_historical_max(self) -> None:
        series = [100.0, 200.0, 150.0]
        self.assertAlmostEqual(aggregate.from_peak(series, 2), -25.0)

    def test_at_peak_is_zero(self) -> None:
        self.assertAlmostEqual(aggregate.from_peak([100.0, 200.0], 1), 0.0)

    def test_peak_only_looks_at_or_before_index(self) -> None:
        """미래의 고점을 끌어와 '전고점 대비'를 계산하면 안 된다."""
        series = [100.0, 150.0, 400.0]
        self.assertAlmostEqual(aggregate.from_peak(series, 1), 0.0)

    def test_none_current_is_none(self) -> None:
        self.assertIsNone(aggregate.from_peak([100.0, None], 1))


class TestTurnover(unittest.TestCase):
    def test_ratio_in_percent(self) -> None:
        self.assertAlmostEqual(aggregate.turnover(50, 1000), 5.0)

    def test_zero_households_is_none(self) -> None:
        self.assertIsNone(aggregate.turnover(50, 0))


class TestAreaBucket(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(aggregate.area_bucket(59.9), 0)
        self.assertEqual(aggregate.area_bucket(60.0), 1)
        self.assertEqual(aggregate.area_bucket(84.9), 1)
        self.assertEqual(aggregate.area_bucket(85.0), 2)
        self.assertEqual(aggregate.area_bucket(134.9), 2)
        self.assertEqual(aggregate.area_bucket(135.0), 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_aggregate -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aggregate'`

- [ ] **Step 3: `scripts/aggregate.py` 를 구현한다**

```python
"""대시보드 집계에 쓰는 순수 함수. I/O 를 하지 않아 단독으로 검증할 수 있다."""

from __future__ import annotations

import statistics

# 1평 = 3.3058㎡. 저장소 전체에서 이 값을 쓴다.
PYEONG = 3.3058

# 월 거래가 이 미만이면 '얇은 달'로 표시하고 변화율은 이동중위로 낸다.
THIN_SAMPLE = 5

# 전용면적 구간 경계 (㎡)
AREA_EDGES = (60.0, 85.0, 135.0)


def pyeong_price(price_10k: int, area_sqm: float) -> float | None:
    """만원/평. 면적이 0 이하면 None."""
    if area_sqm <= 0:
        return None
    return price_10k / (area_sqm / PYEONG)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def pct_change(now: float | None, before: float | None) -> float | None:
    """퍼센트 변화율. 어느 쪽이든 없거나 기준이 0 이면 None."""
    if now is None or before is None or before == 0:
        return None
    return (now / before - 1) * 100


def rolling_median(series: list[float | None], window: int) -> list[float | None]:
    """창 안의 None 은 무시하고 중위값을 낸다. 창이 다 차기 전 구간은 None."""
    out: list[float | None] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = [v for v in series[i + 1 - window:i + 1] if v is not None]
        out.append(statistics.median(chunk) if chunk else None)
    return out


def from_peak(series: list[float | None], index: int) -> float | None:
    """해당 시점까지의 역대 최고 대비 몇 % 인지.

    고점은 index 이하 구간에서만 찾는다. 미래의 고점을 끌어오면
    '지금 전고점 대비 얼마'라는 뜻이 깨진다.
    """
    now = series[index]
    if now is None:
        return None
    past = [v for v in series[:index + 1] if v is not None]
    if not past:
        return None
    peak = max(past)
    if peak == 0:
        return None
    return (now / peak - 1) * 100


def turnover(trade_count: int, households: int) -> float | None:
    """세대수 대비 거래 회전율(%)."""
    if households <= 0:
        return None
    return trade_count / households * 100


def area_bucket(area_sqm: float) -> int:
    """0: ~60, 1: 60~85, 2: 85~135, 3: 135~"""
    for i, edge in enumerate(AREA_EDGES):
        if area_sqm < edge:
            return i
    return len(AREA_EDGES)
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_aggregate -v`
Expected: PASS (전부)

- [ ] **Step 5: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/aggregate.py tests/test_aggregate.py
git commit -m "Add pure aggregation helpers for the dashboard

평당가·중위값·변화율·전고점 대비·회전율을 I/O 없이 계산하는 함수 모음.
전고점은 해당 시점 이전 구간에서만 찾는다."
```

---

## Task 3: 구 × 월 집계 (`summary.json`)

**Files:**
- Create: `scripts/build_dashboard.py`
- Create: `tests/test_build_dashboard.py`
- Generates: `assets/realestate/summary.json`

**Interfaces:**
- Consumes: `aggregate.*` (Task 2), `regions.make_pnu`, `regions.sgg_codes`, `rtms.csv_to_rows`, `rtms.gunzip_text`
- Produces:
  - `load_complexes() -> dict[str, dict]` — PNU → `{"name","dong","hh","sgg"}`
  - `join_household(row: dict, complexes: dict, by_name: dict) -> int | None`
  - `month_labels() -> list[str]` — `["2006-01", ...]` 오름차순
  - `build_summary(...) -> dict` — 아래 스키마
  - `write_json(path: Path, payload: dict) -> bool` — 내용이 같으면 안 쓰고 False

### `summary.json` 스키마

```json
{
  "generated": "2026-07-26",
  "months": ["2006-01", "...", "2026-07"],
  "partial": "2026-07",
  "sgg": {
    "11680": {"name": "강남구", "sido": "11", "hh": {"300": 91234, "all": 120345}}
  },
  "series": {
    "300": {"11680": {"med": [2599, null, ...], "n": [243, ...], "cancel": [0, ...]}},
    "all": {"11680": {"med": [...], "n": [...], "cancel": [...]}}
  }
}
```

`med` 는 만원/평 **정수 반올림**. 거래가 없는 달은 `null`. 배열 길이는 `months`
와 같다. `partial` 은 신고가 아직 덜 들어온 마지막 달이다.

`cancel`(계약해제 건수)은 1차 화면에서 쓰지 않지만 지금 담아 둔다. 2차에서
해제율 지표를 켤 때 247개월을 다시 집계하지 않아도 된다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_build_dashboard.py`:

```python
"""집계 빌드 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dashboard  # noqa: E402


def _trade(sgg="11680", umd="역삼동", jibun="755-1", apt="개나리푸르지오",
           area="84.0", price="100000", date="2026-06-27", cdeal="") -> dict:
    return {
        "sgg_cd": sgg, "umd_nm": umd, "jibun": jibun, "apt_name": apt,
        "apt_dong": "", "build_year": "2006", "area_sqm": area, "floor": "7",
        "price_10k": price, "trade_date": date, "deal_type": "중개거래",
        "seller_gbn": "개인", "buyer_gbn": "개인", "land_leasehold": "N",
        "cdeal_type": cdeal, "cdeal_day": "",
    }


class TestJoinHousehold(unittest.TestCase):
    def setUp(self) -> None:
        self.complexes = {
            "1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                    "hh": 332, "sgg": "11680"},
        }
        self.by_name = {("11680", "역삼동", "개나리푸르지오"): 332}

    def test_matches_by_pnu(self) -> None:
        got = build_dashboard.join_household(_trade(), self.complexes, self.by_name)
        self.assertEqual(got, 332)

    def test_falls_back_to_name_when_pnu_missing(self) -> None:
        row = _trade(jibun="9999-9999")  # 마스터에 없는 지번
        got = build_dashboard.join_household(row, self.complexes, self.by_name)
        self.assertEqual(got, 332)

    def test_unknown_returns_none(self) -> None:
        row = _trade(apt="없는단지", jibun="9999-9999")
        self.assertIsNone(build_dashboard.join_household(row, self.complexes, self.by_name))


class TestBuildSummary(unittest.TestCase):
    def _run(self, by_month: dict[str, list[dict]], hh_lookup=None) -> dict:
        complexes = {"1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                             "hh": 332, "sgg": "11680"}}
        by_name = {("11680", "역삼동", "개나리푸르지오"): 332}
        return build_dashboard.build_summary(
            by_month, sorted(by_month), complexes, by_name,
            {"11680": "강남구"}, generated="2026-07-26")

    def test_median_is_pyeong_price_rounded(self) -> None:
        out = self._run({"2026-06": [_trade(area="84.0", price="100000")]})
        # 100000 / (84/3.3058) = 3935.5 -> 3936
        self.assertEqual(out["series"]["all"]["11680"]["med"][0], 3936)

    def test_cancelled_trade_excluded_from_price_but_counted(self) -> None:
        rows = [_trade(price="100000"), _trade(price="900000", cdeal="O")]
        out = self._run({"2026-06": rows})
        s = out["series"]["all"]["11680"]
        self.assertEqual(s["med"][0], 3936)   # 해제 건이 중위값을 흔들지 않는다
        self.assertEqual(s["n"][0], 1)        # 유효 거래만 센다
        self.assertEqual(s["cancel"][0], 1)

    def test_household_filter_splits_series(self) -> None:
        rows = [_trade(), _trade(apt="없는단지", jibun="9999-9999", price="50000")]
        out = self._run({"2026-06": rows})
        self.assertEqual(out["series"]["300"]["11680"]["n"][0], 1)
        self.assertEqual(out["series"]["all"]["11680"]["n"][0], 2)

    def test_month_with_no_trades_is_null(self) -> None:
        out = self._run({"2026-05": [], "2026-06": [_trade()]})
        self.assertIsNone(out["series"]["all"]["11680"]["med"][0])
        self.assertIsNotNone(out["series"]["all"]["11680"]["med"][1])

    def test_zero_area_row_dropped(self) -> None:
        out = self._run({"2026-06": [_trade(area="0")]})
        self.assertIsNone(out["series"]["all"]["11680"]["med"][0])

    def test_last_month_marked_partial(self) -> None:
        out = self._run({"2026-05": [_trade()], "2026-06": [_trade()]})
        self.assertEqual(out["partial"], "2026-06")

    def test_households_summed_per_filter(self) -> None:
        out = self._run({"2026-06": [_trade()]})
        self.assertEqual(out["sgg"]["11680"]["hh"]["300"], 332)


class TestWriteJson(unittest.TestCase):
    def test_skips_write_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            payload = {"b": 2, "a": 1}
            self.assertTrue(build_dashboard.write_json(p, payload))
            self.assertFalse(build_dashboard.write_json(p, payload))

    def test_output_is_key_sorted_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            build_dashboard.write_json(p, {"b": 2, "a": 1})
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.startswith('{"a":1,"b":2}'))

    def test_same_payload_two_orders_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            build_dashboard.write_json(a, {"x": 1, "y": 2})
            build_dashboard.write_json(b, {"y": 2, "x": 1})
            self.assertEqual(a.read_bytes(), b.read_bytes())


class TestAgainstRealData(unittest.TestCase):
    """실제 원본으로 만든 summary.json 이 있을 때만 도는 대조 테스트.

    설계 문서 작성 시 원본에서 직접 계산한 강남구 값과 맞춰 본다.
    """

    OUT = ROOT / "assets" / "realestate" / "summary.json"

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_gangnam_matches_measured_values(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        months = data["months"]
        med = data["series"]["all"]["11680"]["med"]
        self.assertEqual(med[months.index("2006-01")], 2599)
        self.assertEqual(med[months.index("2026-07")], 12659)
        self.assertEqual(med[months.index("2026-06")], 12252)

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_covers_all_regions_and_months(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        self.assertEqual(len(data["sgg"]), 72)
        self.assertEqual(len(data["months"]), 247)
        self.assertEqual(data["months"][0], "2006-01")

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_within_size_budget(self) -> None:
        self.assertLess(self.OUT.stat().st_size, 200 * 1024)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_build_dashboard -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_dashboard'`

- [ ] **Step 3: `scripts/build_dashboard.py` 를 구현한다**

```python
"""수집한 원본에서 대시보드용 집계 JSON 을 만든다.

    python3 scripts/build_dashboard.py

원본은 건드리지 않는다. 지표 정의가 바뀌면 이 스크립트만 다시 돌린다.
출력은 결정론적이라 값이 바뀌지 않은 파일은 건드리지 않고 넘어간다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

TRADES_DIR = ROOT / "data" / "trades"
COMPLEX_FILE = ROOT / "data" / "complexes.csv.gz"
OUT_DIR = ROOT / "assets" / "realestate"
SUMMARY_FILE = OUT_DIR / "summary.json"

HOUSEHOLD_MIN = 300
MAX_SUMMARY_BYTES = 200 * 1024

_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_name(name: str) -> str:
    """단지명 2차 조인용 정규화. 괄호 꼬리표와 공백을 없앤다.

    '현대5차(71,72동)' 와 '현대5차' 가 같은 단지로 붙게 한다.
    """
    return _SUFFIX_RE.sub("", (name or "").strip()).replace(" ", "")


def load_complexes() -> tuple[dict[str, dict], dict[tuple[str, str, str], int]]:
    """단지 마스터를 PNU 인덱스와 (시군구, 법정동, 정규화 단지명) 인덱스로 읽는다."""
    rows = rtms.csv_to_rows(rtms.gunzip_text(COMPLEX_FILE.read_bytes()))
    by_pnu: dict[str, dict] = {}
    by_name: dict[tuple[str, str, str], int] = {}
    for row in rows:
        try:
            hh = int(row.get("household_count") or 0)
        except ValueError:
            continue
        pnu = row.get("pnu") or ""
        sgg = row.get("sgg_cd") or ""
        name = row.get("complex_name") or ""
        # address 는 '서울특별시 강남구 역삼동 755-1' 꼴. 법정동은 뒤에서 두 번째 토큰.
        parts = (row.get("address") or "").split()
        dong = parts[-2] if len(parts) >= 2 else ""
        if pnu:
            by_pnu[pnu] = {"name": name, "dong": dong, "hh": hh, "sgg": sgg}
        key = (sgg, dong, normalize_name(name))
        # 같은 키에 여러 단지가 오면 세대수가 큰 쪽을 남긴다 (대단지 대표값)
        if key not in by_name or by_name[key] < hh:
            by_name[key] = hh
    return by_pnu, by_name


def join_household(row: dict, by_pnu: dict[str, dict],
                   by_name: dict[tuple[str, str, str], int]) -> int | None:
    """거래 한 건의 세대수를 찾는다. PNU 완전일치 → 단지명 → 실패."""
    pnu = regions.make_pnu(row["sgg_cd"], row["umd_nm"], row["jibun"])
    if pnu:
        hit = by_pnu.get(pnu)
        if hit:
            return hit["hh"]
    dong = (row["umd_nm"] or "").split(" ")[-1]
    return by_name.get((row["sgg_cd"], dong, normalize_name(row["apt_name"])))


def month_labels() -> list[str]:
    """data/trades 에 실제로 있는 월 목록. 오름차순."""
    return sorted(p.stem for p in TRADES_DIR.glob("*/*.csv.gz"))


def read_month(ym: str) -> list[dict]:
    path = TRADES_DIR / ym[:4] / f"{ym}.csv.gz"
    if not path.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(path.read_bytes()))


def build_summary(by_month: dict[str, list[dict]], months: list[str],
                  by_pnu: dict[str, dict], by_name: dict[tuple[str, str, str], int],
                  sgg_names: dict[str, str], generated: str) -> dict:
    """구 × 월 집계를 만든다. 필터 두 갈래('300', 'all')를 동시에 낸다."""
    filters = ("300", "all")
    prices: dict[str, dict[str, dict[int, list[float]]]] = {
        f: defaultdict(lambda: defaultdict(list)) for f in filters}
    cancels: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    households: dict[str, dict[str, int]] = {f: defaultdict(int) for f in filters}
    seen_complex: dict[str, set[tuple[str, str]]] = {f: set() for f in filters}

    index = {ym: i for i, ym in enumerate(months)}

    for ym, rows in by_month.items():
        mi = index[ym]
        for row in rows:
            sgg = row["sgg_cd"]
            if sgg not in sgg_names:
                continue
            if row.get("cdeal_type") == "O":
                cancels[sgg][mi] += 1
                continue
            try:
                area = float(row["area_sqm"])
                price = int(row["price_10k"])
            except (ValueError, KeyError):
                continue
            pp = aggregate.pyeong_price(price, area)
            if pp is None:
                continue
            hh = join_household(row, by_pnu, by_name)
            targets = ["all"]
            if hh is not None and hh >= HOUSEHOLD_MIN:
                targets.append("300")
            key = (sgg, normalize_name(row["apt_name"]))
            for f in targets:
                prices[f][sgg][mi].append(pp)
                if hh and key not in seen_complex[f]:
                    seen_complex[f].add(key)
                    households[f][sgg] += hh

    series: dict[str, dict[str, dict]] = {}
    for f in filters:
        series[f] = {}
        for sgg in sgg_names:
            med: list[int | None] = []
            cnt: list[int] = []
            can: list[int] = []
            for mi in range(len(months)):
                vals = prices[f][sgg].get(mi, [])
                m = aggregate.median(vals)
                med.append(round(m) if m is not None else None)
                cnt.append(len(vals))
                can.append(cancels[sgg].get(mi, 0) if f == "all" else 0)
            series[f][sgg] = {"med": med, "n": cnt, "cancel": can}

    return {
        "generated": generated,
        "months": months,
        "partial": months[-1] if months else None,
        "sgg": {
            sgg: {
                "name": name,
                "sido": sgg[:2],
                "hh": {f: households[f].get(sgg, 0) for f in filters},
            }
            for sgg, name in sorted(sgg_names.items())
        },
        "series": series,
    }


def write_json(path: Path, payload: dict) -> bool:
    """결정론적으로 쓴다. 내용이 같으면 건드리지 않고 False."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")) + "\n"
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat(),
                        help="생성일. 테스트에서 고정하려면 지정한다")
    args = parser.parse_args()

    months = month_labels()
    if not months:
        print("data/trades 가 비어 있습니다.", file=sys.stderr)
        return 1

    by_pnu, by_name = load_complexes()
    sgg_names = {}
    for code, full in regions.sgg_codes():
        short = full.split(" ")[-1] if code.startswith("11") else " ".join(full.split(" ")[1:])
        sgg_names[code] = short

    by_month = {ym: read_month(ym) for ym in months}
    total = sum(len(v) for v in by_month.values())
    print(f"원본 {total:,}건 / {len(months)}개월 / 시군구 {len(sgg_names)}개")

    summary = build_summary(by_month, months, by_pnu, by_name, sgg_names, args.generated)
    changed = write_json(SUMMARY_FILE, summary)
    size = SUMMARY_FILE.stat().st_size
    print(f"summary.json {size:,}B {'갱신' if changed else '변경 없음'}")
    if size > MAX_SUMMARY_BYTES:
        print(f"summary.json 이 예산({MAX_SUMMARY_BYTES}B)을 넘었습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 합성 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_build_dashboard -v`
Expected: PASS. `TestAgainstRealData` 3건은 `summary.json` 이 아직 없어 skip.

- [ ] **Step 5: 실제 데이터로 빌드한다**

Run: `python3 scripts/build_dashboard.py`
Expected: `원본 4,352,431건 / 247개월 / 시군구 72개` 그리고 `summary.json ...B 갱신`

몇 분 걸린다 (435만 행 파싱). 예산 초과로 실패하면 `med` 를 그대로 두고
`cancel` 배열에서 전부 0 인 구를 생략하는 것으로 줄인다.

- [ ] **Step 6: 실측 대조 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_build_dashboard -v`
Expected: PASS 전부. 특히 `test_gangnam_matches_measured_values` 가 통과해야
한다 — 2006-01 = 2599, 2026-06 = 12252, 2026-07 = 12659.

값이 다르면 조인이나 해제 건 처리가 설계와 어긋난 것이다. 구현을 고친다.

- [ ] **Step 7: 결정론을 확인한다**

```bash
cp assets/realestate/summary.json /tmp/summary-1.json
python3 scripts/build_dashboard.py --generated 2026-07-26
python3 scripts/build_dashboard.py --generated 2026-07-26
cmp /tmp/summary-1.json assets/realestate/summary.json && echo "동일"
```

Expected: 두 번째 실행이 `변경 없음` 을 출력한다.

- [ ] **Step 8: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_dashboard.py tests/test_build_dashboard.py assets/realestate/summary.json
git commit -m "Build per-district monthly aggregate for the dashboard

435만 건을 구×월 중위 평당가로 집계한다. 세대수 필터 두 갈래를 함께 낸다.
계약해제 건은 가격에서 빼되 건수는 따로 보존한다."
```

---

## Task 4: 구별 단지 집계 (`sgg/{code}.json`)

**Files:**
- Modify: `scripts/build_dashboard.py` (함수 추가 + `main()` 확장)
- Modify: `tests/test_build_dashboard.py` (테스트 클래스 추가)
- Generates: `assets/realestate/sgg/{code}.json` × 72

**Interfaces:**
- Consumes: Task 3 의 `join_household`, `normalize_name`, `write_json`
- Produces: `build_sgg_detail(rows_by_month, months, window, by_pnu, by_name, sgg) -> dict`

### `sgg/{code}.json` 스키마

```json
{
  "sgg": "11680",
  "window": ["2025-08", "2026-07"],
  "complexes": [
    {"name": "개나리푸르지오", "dong": "역삼동", "hh": 332,
     "med": 8123, "n": 14, "bk": [0, 0, 12, 2]}
  ]
}
```

`med` 는 창 전체의 중위 평당가(만원/평), `n` 은 유효 거래 수, `bk` 는
면적 구간별 거래 수 (0:~60, 1:60~85, 2:85~135, 3:135~). `hh` 가 `null` 이면
세대수 미상(마스터 미매칭)이고 300세대+ 필터에서 빠진다.

`window` 를 파라미터로 받으므로, 2차에서 전 기간 단지 시계열이 필요해지면
같은 함수를 기간만 바꿔 호출하면 된다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_build_dashboard.py` 의 `TestAgainstRealData` **앞에** 아래 클래스를
추가한다. 파일 상단 import 는 그대로 쓴다.

```python
class TestBuildSggDetail(unittest.TestCase):
    def setUp(self) -> None:
        self.by_pnu = {"1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                               "hh": 332, "sgg": "11680"}}
        self.by_name = {("11680", "역삼동", "개나리푸르지오"): 332}
        self.months = ["2026-05", "2026-06", "2026-07"]

    def _run(self, by_month: dict[str, list[dict]], window: int = 2) -> dict:
        return build_dashboard.build_sgg_detail(
            by_month, self.months, window, self.by_pnu, self.by_name, "11680")

    def test_only_window_months_counted(self) -> None:
        by_month = {
            "2026-05": [_trade(price="100000")],   # 창 밖
            "2026-06": [_trade(price="200000")],
            "2026-07": [_trade(price="200000")],
        }
        out = self._run(by_month, window=2)
        self.assertEqual(out["window"], ["2026-06", "2026-07"])
        self.assertEqual(out["complexes"][0]["n"], 2)

    def test_median_over_window(self) -> None:
        by_month = {"2026-06": [_trade(price="100000")],
                    "2026-07": [_trade(price="300000")]}
        out = self._run(by_month, window=2)
        # (100000 + 300000)/2 규모의 평당가 중위 = 두 값의 평균
        lo = 100_000 / (84.0 / 3.3058)
        hi = 300_000 / (84.0 / 3.3058)
        self.assertEqual(out["complexes"][0]["med"], round((lo + hi) / 2))

    def test_area_buckets_counted(self) -> None:
        by_month = {"2026-07": [_trade(area="59.0"), _trade(area="84.0"),
                                _trade(area="84.0"), _trade(area="200.0")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"][0]["bk"], [1, 2, 0, 1])

    def test_cancelled_excluded(self) -> None:
        by_month = {"2026-07": [_trade(), _trade(cdeal="O")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"][0]["n"], 1)

    def test_unmatched_complex_has_null_households(self) -> None:
        by_month = {"2026-07": [_trade(apt="없는단지", jibun="9999-9999")]}
        out = self._run(by_month, window=2)
        self.assertIsNone(out["complexes"][0]["hh"])

    def test_other_districts_excluded(self) -> None:
        by_month = {"2026-07": [_trade(sgg="11110", umd="교북동", apt="경희궁자이")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"], [])

    def test_sorted_by_median_desc_then_name(self) -> None:
        by_month = {"2026-07": [
            _trade(apt="싼단지", jibun="1-1", price="50000"),
            _trade(apt="비싼단지", jibun="2-2", price="500000"),
        ]}
        out = self._run(by_month, window=2)
        self.assertEqual([c["name"] for c in out["complexes"]], ["비싼단지", "싼단지"])
```

그리고 `TestAgainstRealData` 안에 아래 두 테스트를 덧붙인다.

```python
    SGG_DIR = ROOT / "assets" / "realestate" / "sgg"

    @unittest.skipUnless(SGG_DIR.exists(), "구별 JSON 없음 — 먼저 빌드하세요")
    def test_every_region_has_a_detail_file(self) -> None:
        files = sorted(p.stem for p in self.SGG_DIR.glob("*.json"))
        self.assertEqual(len(files), 72)

    @unittest.skipUnless(SGG_DIR.exists(), "구별 JSON 없음 — 먼저 빌드하세요")
    def test_detail_files_within_size_budget(self) -> None:
        oversized = [(p.name, p.stat().st_size) for p in self.SGG_DIR.glob("*.json")
                     if p.stat().st_size > 300 * 1024]
        self.assertEqual(oversized, [])
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_build_dashboard.TestBuildSggDetail -v`
Expected: FAIL — `AttributeError: module 'build_dashboard' has no attribute 'build_sgg_detail'`

- [ ] **Step 3: `build_sgg_detail` 을 구현한다**

`scripts/build_dashboard.py` 의 `write_json` **앞에** 아래를 넣는다.

```python
SGG_DIR = OUT_DIR / "sgg"
DETAIL_WINDOW = 12          # 단지 랭킹은 최근 12개월
MAX_DETAIL_BYTES = 300 * 1024


def build_sgg_detail(by_month: dict[str, list[dict]], months: list[str], window: int,
                     by_pnu: dict[str, dict], by_name: dict[tuple[str, str, str], int],
                     sgg: str) -> dict:
    """구 하나의 단지별 집계. 창(window)은 months 의 마지막 n개월.

    1차 화면은 최근 12개월 랭킹만 쓴다. 창을 파라미터로 받아 두었으므로
    단지 시계열이 필요해지면 같은 함수를 다시 부르면 된다.
    """
    target = months[-window:] if window else months
    bucket_count = len(aggregate.AREA_EDGES) + 1

    prices: dict[tuple[str, str], list[float]] = defaultdict(list)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * bucket_count)
    households: dict[tuple[str, str], int | None] = {}
    labels: dict[tuple[str, str], str] = {}

    for ym in target:
        for row in by_month.get(ym, []):
            if row["sgg_cd"] != sgg or row.get("cdeal_type") == "O":
                continue
            try:
                area = float(row["area_sqm"])
                price = int(row["price_10k"])
            except (ValueError, KeyError):
                continue
            pp = aggregate.pyeong_price(price, area)
            if pp is None:
                continue
            dong = (row["umd_nm"] or "").split(" ")[-1]
            key = (dong, normalize_name(row["apt_name"]))
            prices[key].append(pp)
            buckets[key][aggregate.area_bucket(area)] += 1
            labels.setdefault(key, row["apt_name"])
            if key not in households:
                households[key] = join_household(row, by_pnu, by_name)

    complexes = []
    for key, vals in prices.items():
        dong, _ = key
        med = aggregate.median(vals)
        complexes.append({
            "name": labels[key],
            "dong": dong,
            "hh": households.get(key),
            "med": round(med) if med is not None else None,
            "n": len(vals),
            "bk": buckets[key],
        })
    # 비싼 순, 같으면 이름순 — 정렬이 고정돼야 출력이 결정론적이다
    complexes.sort(key=lambda c: (-(c["med"] or 0), c["name"]))

    return {
        "sgg": sgg,
        "window": [target[0], target[-1]] if target else [],
        "complexes": complexes,
    }
```

`main()` 의 `changed = write_json(SUMMARY_FILE, summary)` 아래, `return 0` 앞에
아래를 넣는다.

```python
    SGG_DIR.mkdir(parents=True, exist_ok=True)
    updated, oversized = 0, []
    for sgg in sorted(sgg_names):
        detail = build_sgg_detail(by_month, months, DETAIL_WINDOW,
                                  by_pnu, by_name, sgg)
        path = SGG_DIR / f"{sgg}.json"
        if write_json(path, detail):
            updated += 1
        if path.stat().st_size > MAX_DETAIL_BYTES:
            oversized.append((path.name, path.stat().st_size))
    print(f"구별 JSON {len(sgg_names)}개 중 {updated}개 갱신")
    if oversized:
        print(f"예산({MAX_DETAIL_BYTES}B) 초과: {oversized}", file=sys.stderr)
        return 1
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_build_dashboard -v`
Expected: `TestBuildSggDetail` 전부 PASS. 구별 JSON 대조 2건은 아직 skip.

- [ ] **Step 5: 실제로 빌드하고 크기를 확인한다**

```bash
python3 scripts/build_dashboard.py
ls -S assets/realestate/sgg | head -3
du -sh assets/realestate
python3 -m unittest tests.test_build_dashboard -v
```

Expected: 구별 JSON 72개 생성, 가장 큰 파일도 300KB 미만, 전체 테스트 PASS.

300KB 를 넘는 구가 있으면 거래 3건 미만 단지를 빼서 줄인다 (화면 표는 5건
이상만 보여주므로 손실이 없다) — `build_sgg_detail` 마지막에 필터를 넣는다.

- [ ] **Step 6: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_dashboard.py tests/test_build_dashboard.py assets/realestate/sgg
git commit -m "Build per-district complex rankings for the dashboard

구별 최근 12개월 단지 집계를 시군구마다 한 파일로 낸다.
구를 처음 누를 때만 받도록 요약본과 분리했다."
```

---

## Task 5: 페이지 골격과 지도

**Files:**
- Create: `_pages/real-estate.md`
- Create: `assets/realestate/dashboard.css`
- Create: `assets/realestate/palette.js`
- Create: `assets/realestate/data.js`
- Create: `assets/realestate/map.js`
- Create: `assets/realestate/app.js`

**Interfaces:**
- Consumes: `assets/realestate/summary.json` (Task 3), `_includes/realestate/map.svg` (Task 1)
- Produces (ES 모듈 export):
  - `palette.js`: `SEQUENTIAL: string[]`, `DIVERGING: string[]`, `INK`, `INK2`, `MUTED`, `GRID`, `AXIS`, `LINE`, `UP`, `DOWN`, `NO_DATA`, `rampColor(ramp, t)`, `divergingColor(value, span)`, `sequentialColor(value, min, max)`
  - `data.js`: `loadSummary(): Promise<object>`, `loadSgg(code): Promise<object>`
  - `map.js`: `initMap(root, {onSelect}): MapHandle`, `MapHandle.paint(values, kind)`, `MapHandle.setView(view)`, `MapHandle.setSelected(code)`
  - `app.js`: 진입점. DOM 로드 후 배선

- [ ] **Step 1: 페이지와 마크업을 만든다**

`_pages/real-estate.md`:

```markdown
---
layout: single
title: "서울·경기 아파트 실거래 대시보드"
permalink: /real-estate/
classes: wide
author_profile: false
toc: false
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}">

<div class="re-app" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs" role="tablist" aria-label="지역 선택">
      <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
      <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
      <button class="re-tab" data-view="all" role="tab" aria-selected="false">전체</button>
    </div>
    <div class="re-filters">
      <label class="re-field">
        <span class="re-field-label">지표</span>
        <select class="re-metric">
          <option value="level">중위 평당가</option>
          <option value="chg3">3개월 변화율</option>
          <option value="chg6">6개월 변화율</option>
          <option value="chg12" selected>12개월 변화율</option>
          <option value="peak">전고점 대비</option>
          <option value="turnover">거래 회전율</option>
        </select>
      </label>
      <label class="re-field">
        <span class="re-field-label">기준월</span>
        <select class="re-month"></select>
      </label>
      <button class="re-toggle is-on" data-filter="300" aria-pressed="true">300세대+</button>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-legend" aria-hidden="true"></div>
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <h2 class="re-panel-title">지역을 선택하세요</h2>
      <div class="re-kpis"></div>
      <h3 class="re-section-title">평당가 추이</h3>
      <div class="re-chart"></div>
    </div>
  </div>

  <h3 class="re-section-title">단지 랭킹 · 최근 12개월</h3>
  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-footnote"></p>
</div>

<script type="module" src="{{ '/assets/realestate/app.js' | relative_url }}"></script>
```

- [ ] **Step 2: 색 모듈을 만든다**

`assets/realestate/palette.js`:

```javascript
// dataviz 기준 팔레트(라이트 표면 #fcfcfb). 값을 바꾸지 말 것 — 검증을 통과한 조합이다.

export const SEQUENTIAL = [
  '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
  '#2a78d6', '#256abf', '#184f95', '#0d366b',
];

// 파랑(하락) ↔ 회색(변화 없음) ↔ 빨강(상승). 국내 관행과 같은 방향이다.
export const DIVERGING = [
  '#184f95', '#2a78d6', '#86b6ef', '#f0efec', '#f0a8a8', '#d03b3b', '#a02020',
];

export const INK = '#0b0b0b';
export const INK2 = '#52514e';
export const MUTED = '#898781';
export const GRID = '#e1e0d9';
export const AXIS = '#c3c2b7';
export const LINE = '#2a78d6';
export const UP = '#006300';
export const DOWN = '#d03b3b';
export const NO_DATA = '#e8e8e4';

export function rampColor(ramp, t) {
  if (!Number.isFinite(t)) return NO_DATA;
  const i = Math.min(ramp.length - 1, Math.max(0, Math.floor(t * ramp.length)));
  return ramp[i];
}

// span 은 한쪽 팔의 길이. -span..+span 을 발산 램프 전체에 대응시킨다.
export function divergingColor(value, span) {
  if (!Number.isFinite(value) || !(span > 0)) return NO_DATA;
  const clamped = Math.max(-span, Math.min(span, value));
  return rampColor(DIVERGING, (clamped + span) / (2 * span) * 0.999);
}

export function sequentialColor(value, min, max) {
  if (!Number.isFinite(value) || !(max > min)) return NO_DATA;
  const t = (Math.max(min, Math.min(max, value)) - min) / (max - min);
  return rampColor(SEQUENTIAL, t * 0.999);
}
```

- [ ] **Step 3: 데이터 로더를 만든다**

`assets/realestate/data.js`:

```javascript
// 집계 JSON 을 받아 캐시한다. 구별 상세는 처음 누를 때만 받는다.

let base = '/assets/realestate';
let summaryPromise = null;
const sggCache = new Map();

export function setBase(path) {
  base = path.replace(/\/$/, '');
}

async function getJson(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

export function loadSummary() {
  if (!summaryPromise) summaryPromise = getJson(`${base}/summary.json`);
  return summaryPromise;
}

export function loadSgg(code) {
  if (!sggCache.has(code)) sggCache.set(code, getJson(`${base}/sgg/${code}.json`));
  return sggCache.get(code);
}
```

- [ ] **Step 4: 지도 모듈을 만든다**

`assets/realestate/map.js`:

```javascript
import { NO_DATA } from './palette.js';

// 서울/경기/전체 뷰는 같은 SVG 의 viewBox 를 바꿔 만든다. 지도는 한 장뿐이다.
const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };
const PAD = 10;

function boundsOf(paths) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of paths) {
    const box = p.getBBox();
    minX = Math.min(minX, box.x);
    minY = Math.min(minY, box.y);
    maxX = Math.max(maxX, box.x + box.width);
    maxY = Math.max(maxY, box.y + box.height);
  }
  return { minX, minY, maxX, maxY };
}

export function initMap(root, { onSelect }) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  const paths = Array.from(svg.querySelectorAll('path[data-sgg]'));
  const byCode = new Map(paths.map((p) => [p.dataset.sgg, p]));
  let labels = new Map();
  let selected = null;

  function showTip(path, evt) {
    const code = path.dataset.sgg;
    const text = labels.get(code) || path.dataset.name;
    tip.textContent = text;
    tip.hidden = false;
    const box = root.getBoundingClientRect();
    tip.style.left = `${evt.clientX - box.left}px`;
    tip.style.top = `${evt.clientY - box.top}px`;
  }

  for (const path of paths) {
    path.setAttribute('tabindex', '0');
    path.setAttribute('role', 'button');
    path.addEventListener('click', () => onSelect(path.dataset.sgg));
    path.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onSelect(path.dataset.sgg);
      }
    });
    path.addEventListener('mousemove', (e) => showTip(path, e));
    path.addEventListener('mouseleave', () => { tip.hidden = true; });
    path.addEventListener('focus', () => {
      const box = path.getBoundingClientRect();
      showTip(path, { clientX: box.left + box.width / 2, clientY: box.top });
    });
    path.addEventListener('blur', () => { tip.hidden = true; });
  }

  return {
    // values: Map<code, {color, label}>
    paint(values) {
      labels = new Map();
      for (const [code, path] of byCode) {
        const hit = values.get(code);
        path.setAttribute('fill', hit ? hit.color : NO_DATA);
        if (hit) labels.set(code, hit.label);
      }
    },
    setView(view) {
      const prefix = VIEW_PREFIX[view] ?? '';
      const shown = paths.filter((p) => p.dataset.sgg.startsWith(prefix));
      const visible = new Set(shown);
      for (const p of paths) {
        p.style.display = visible.has(p) ? '' : 'none';
      }
      const b = boundsOf(shown);
      svg.setAttribute('viewBox',
        `${b.minX - PAD} ${b.minY - PAD} ${b.maxX - b.minX + 2 * PAD} ${b.maxY - b.minY + 2 * PAD}`);
    },
    setSelected(code) {
      if (selected) selected.classList.remove('is-selected');
      selected = code ? byCode.get(code) : null;
      if (selected) selected.classList.add('is-selected');
    },
    codesIn(view) {
      const prefix = VIEW_PREFIX[view] ?? '';
      return paths.map((p) => p.dataset.sgg).filter((c) => c.startsWith(prefix));
    },
  };
}
```

- [ ] **Step 5: 지표 계산과 배선을 만든다**

`assets/realestate/app.js`:

```javascript
import { setBase, loadSummary, loadSgg } from './data.js';
import { initMap } from './map.js';
import { divergingColor, sequentialColor, DIVERGING, SEQUENTIAL } from './palette.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);

const state = { view: 'seoul', metric: 'chg12', filter: '300', ym: null, sgg: null };
let summary = null;
let map = null;

const METRICS = {
  level: { label: '중위 평당가', unit: '만원/평', kind: 'sequential' },
  chg3: { label: '3개월 변화율', unit: '%', kind: 'diverging', lag: 3 },
  chg6: { label: '6개월 변화율', unit: '%', kind: 'diverging', lag: 6 },
  chg12: { label: '12개월 변화율', unit: '%', kind: 'diverging', lag: 12 },
  peak: { label: '전고점 대비', unit: '%', kind: 'diverging' },
  turnover: { label: '거래 회전율', unit: '%', kind: 'sequential' },
};

// 얇은 달(거래 5건 미만)이 변화율을 흔들지 않도록 3개월 이동중위를 쓴다.
const THIN = 5;

function smoothed(series, index) {
  const out = [];
  for (let i = 0; i <= index; i += 1) {
    const chunk = [];
    for (let k = Math.max(0, i - 2); k <= i; k += 1) {
      if (series.med[k] != null) chunk.push(series.med[k]);
    }
    out.push(chunk.length ? chunk.sort((a, b) => a - b)[Math.floor(chunk.length / 2)] : null);
  }
  return out;
}

function metricValue(code, metric, index) {
  const series = summary.series[state.filter][code];
  if (!series) return null;
  const spec = METRICS[metric];
  if (metric === 'level') return series.med[index];
  if (metric === 'turnover') {
    const hh = summary.sgg[code].hh[state.filter];
    if (!hh) return null;
    let n = 0;
    for (let i = Math.max(0, index - 11); i <= index; i += 1) n += series.n[i] || 0;
    return (n / hh) * 100;
  }
  const thin = (series.n[index] || 0) < THIN;
  const base = thin ? smoothed(series, index) : series.med;
  const now = base[index];
  if (now == null) return null;
  if (metric === 'peak') {
    let peak = -Infinity;
    for (let i = 0; i <= index; i += 1) if (base[i] != null) peak = Math.max(peak, base[i]);
    return peak > 0 ? (now / peak - 1) * 100 : null;
  }
  const before = base[index - spec.lag];
  if (before == null || before === 0) return null;
  return (now / before - 1) * 100;
}

function repaint() {
  const index = summary.months.indexOf(state.ym);
  const codes = map.codesIn(state.view);
  const raw = new Map();
  for (const code of codes) raw.set(code, metricValue(code, state.metric, index));

  const spec = METRICS[state.metric];
  const nums = [...raw.values()].filter((v) => Number.isFinite(v));
  const values = new Map();

  if (spec.kind === 'diverging') {
    const span = Math.max(1, ...nums.map((v) => Math.abs(v)));
    for (const [code, v] of raw) {
      values.set(code, {
        color: divergingColor(v, span),
        label: `${summary.sgg[code].name} ${Number.isFinite(v) ? `${v.toFixed(1)}%` : '자료 없음'}`,
      });
    }
    drawLegend(-span, span, 'diverging', spec.unit);
  } else {
    const min = nums.length ? Math.min(...nums) : 0;
    const max = nums.length ? Math.max(...nums) : 1;
    for (const [code, v] of raw) {
      values.set(code, {
        color: sequentialColor(v, min, max),
        label: `${summary.sgg[code].name} ${Number.isFinite(v) ? v.toLocaleString() : '자료 없음'}`,
      });
    }
    drawLegend(min, max, 'sequential', spec.unit);
  }
  map.paint(values);
}

function drawLegend(min, max, kind, unit) {
  const el = root.querySelector('.re-legend');
  // 램프는 palette.js 한 곳에서만 정의한다. 여기에 색을 다시 적지 말 것.
  const ramp = kind === 'diverging' ? DIVERGING : SEQUENTIAL;
  const swatches = ramp.map((c) => `<i style="background:${c}"></i>`).join('');
  const fmt = (v) => (kind === 'diverging' ? `${v.toFixed(0)}%` : Math.round(v).toLocaleString());
  el.innerHTML = `<span>${fmt(min)}</span>${swatches}<span>${fmt(max)}${
    kind === 'sequential' ? ` ${unit}` : ''}</span>`;
}

function fillMonths() {
  const sel = root.querySelector('.re-month');
  sel.innerHTML = summary.months
    .map((m) => `<option value="${m}">${m}${m === summary.partial ? ' (집계 중)' : ''}</option>`)
    .join('');
  sel.value = state.ym;
}

async function selectSgg(code) {
  state.sgg = code;
  map.setSelected(code);
  root.querySelector('.re-panel-title').textContent = summary.sgg[code].name;
  const detail = await loadSgg(code);
  const { renderPanel } = await import('./charts.js');
  renderPanel(root, summary, detail, state);
}

function bind() {
  root.querySelectorAll('.re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      root.querySelectorAll('.re-tab').forEach((b) => {
        const on = b === btn;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-selected', String(on));
      });
      map.setView(state.view);
      repaint();
    });
  });
  root.querySelector('.re-metric').addEventListener('change', (e) => {
    state.metric = e.target.value;
    repaint();
  });
  root.querySelector('.re-month').addEventListener('change', (e) => {
    state.ym = e.target.value;
    repaint();
    if (state.sgg) selectSgg(state.sgg);
  });
  const toggle = root.querySelector('.re-toggle');
  toggle.addEventListener('click', () => {
    state.filter = state.filter === '300' ? 'all' : '300';
    const on = state.filter === '300';
    toggle.classList.toggle('is-on', on);
    toggle.setAttribute('aria-pressed', String(on));
    toggle.textContent = on ? '300세대+' : '전체 거래';
    repaint();
    if (state.sgg) selectSgg(state.sgg);
  });
}

async function start() {
  summary = await loadSummary();
  // 마지막 달은 신고가 덜 들어와 항상 미완성이다. 직전 완료 월을 기본으로 둔다.
  const last = summary.months.length - 1;
  state.ym = summary.months[Math.max(0, last - 1)];
  map = initMap(root, { onSelect: selectSgg });
  fillMonths();
  bind();
  map.setView(state.view);
  repaint();
  root.querySelector('.re-footnote').textContent =
    `국토교통부 실거래가 · ${summary.months[0]} ~ ${summary.months[last]} · 갱신 ${summary.generated}`;
}

start();
```

- [ ] **Step 6: 스타일을 만든다**

`assets/realestate/dashboard.css`:

```css
/* 대시보드 전용. 테마 스타일을 건드리지 않도록 .re- 접두사만 쓴다. */

.re-app {
  --re-ink: #0b0b0b;
  --re-ink2: #52514e;
  --re-muted: #898781;
  --re-grid: #e1e0d9;
  --re-line: #2a78d6;
  --re-border: rgba(11, 11, 11, 0.10);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--re-ink);
  max-width: 100%;
}

.re-controls {
  display: flex; justify-content: space-between; align-items: center;
  gap: 10px; flex-wrap: wrap; margin-bottom: 14px;
}
.re-tabs { display: flex; gap: 4px; }
.re-tab, .re-toggle {
  font-size: 13px; padding: 5px 13px; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--re-border); background: transparent; color: var(--re-ink2);
}
.re-tab.is-on { background: var(--re-line); color: #fff; border-color: var(--re-line); }
.re-toggle.is-on {
  background: rgba(42, 120, 214, 0.10); color: var(--re-line);
  border-color: rgba(42, 120, 214, 0.35);
}
.re-filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.re-field { display: flex; align-items: center; gap: 5px; font-size: 12px; }
.re-field-label { color: var(--re-muted); }
.re-field select {
  font-size: 13px; padding: 4px 8px; border-radius: 6px;
  border: 1px solid var(--re-border); background: transparent; color: var(--re-ink);
}

.re-body { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 20px; align-items: start; }
.re-map-wrap { position: relative; }
svg.re-map { width: 100%; height: auto; display: block; }
svg.re-map path {
  stroke: rgba(255, 255, 255, 0.85); stroke-width: 1;
  cursor: pointer; transition: fill 120ms linear;
}
svg.re-map path:hover { stroke: var(--re-ink); stroke-width: 1.5; }
svg.re-map path:focus-visible { outline: none; stroke: var(--re-ink); stroke-width: 2; }
svg.re-map path.is-selected { stroke: var(--re-ink); stroke-width: 2; }

.re-legend {
  display: flex; align-items: center; gap: 0;
  margin-top: 8px; font-size: 10.5px; color: var(--re-muted);
}
.re-legend i { width: 26px; height: 9px; display: block; }
.re-legend span:first-child { margin-right: 6px; }
.re-legend span:last-child { margin-left: 6px; }

.re-tip {
  position: absolute; transform: translate(-50%, -140%); pointer-events: none;
  background: var(--re-ink); color: #fff; font-size: 11.5px;
  padding: 3px 8px; border-radius: 5px; white-space: nowrap; z-index: 3;
}

.re-panel-title { font-size: 17px; margin: 0 0 10px; }
.re-section-title { font-size: 13px; font-weight: 650; margin: 16px 0 8px; }
.re-kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.re-kpi { border: 1px solid var(--re-border); border-radius: 8px; padding: 9px 11px; }
.re-kpi-label { font-size: 10.5px; color: var(--re-muted); margin-bottom: 4px; }
.re-kpi-value { font-size: 20px; font-weight: 650; letter-spacing: -0.01em; }
.re-kpi-value span { font-size: 11.5px; font-weight: 400; color: var(--re-ink2); margin-left: 2px; }
.re-kpi-delta { font-size: 10.5px; margin-top: 3px; color: var(--re-ink2); }
.re-kpi-delta.is-up { color: #006300; }
.re-kpi-delta.is-down { color: #d03b3b; }

.re-chart svg { width: 100%; height: auto; display: block; }

.re-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table.re-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.re-table th {
  text-align: left; font-size: 10.5px; font-weight: 600; color: var(--re-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
  padding: 6px; border-bottom: 1px solid var(--re-grid); white-space: nowrap;
}
table.re-table td { padding: 6px; border-bottom: 1px solid var(--re-grid); }
table.re-table td.is-num, table.re-table th.is-num {
  text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;
}
table.re-table td.is-dim { color: var(--re-ink2); }

.re-footnote { font-size: 11.5px; color: var(--re-muted); margin-top: 12px; }

@media (max-width: 720px) {
  .re-body { grid-template-columns: 1fr; }
  .re-kpis { grid-template-columns: 1fr; }
  .re-controls { align-items: flex-start; }
}
```

- [ ] **Step 7: 임시 `charts.js` 를 만든다 (Task 6 에서 채운다)**

`app.js` 가 동적 import 하므로 빈 모듈이라도 있어야 지도가 뜬다.

```javascript
// assets/realestate/charts.js — Task 6 에서 구현한다.
export function renderPanel() {}
```

- [ ] **Step 8: 로컬에서 빌드해 지도가 뜨는지 확인한다**

이 리눅스 머신에는 루비가 없다. Docker 로 빌드한다.

```bash
cat > _tmp_Gemfile <<'EOF'
source "https://rubygems.org"
gemspec
gem "jekyll-sass-converter", "~> 2.0"
gem "webrick"
EOF

docker run --rm -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll jekyll/jekyll:4.2.2 \
  sh -c "bundle install && jekyll build"

ls -la _site/real-estate/index.html _site/assets/realestate/summary.json
```

Expected: 두 파일 모두 존재.

```bash
docker run -d --name kayser_serve -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll -p 4000:4000 jekyll/jekyll:4.2.2 \
  sh -c "bundle install && jekyll serve --host 0.0.0.0 --skip-initial-build --no-watch"
sleep 20 && curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:4000/real-estate/
```

Expected: `200`

- [ ] **Step 9: 헤드리스 크롬으로 지도 렌더링을 확인한다**

```bash
google-chrome --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1200,900 --virtual-time-budget=8000 \
  --screenshot=/tmp/re-desktop.png http://127.0.0.1:4000/real-estate/
```

`/tmp/re-desktop.png` 를 Read 로 열어 확인한다. 합격 기준:

- 서울 25개 구가 파랑↔빨강 발산색으로 칠해져 있다 (전부 회색이면 채색 실패)
- 지도가 왼쪽, 빈 패널이 오른쪽에 있다
- 범례에 숫자 범위가 보인다
- 콘솔 오류가 없다 (아래로 확인)

```bash
google-chrome --headless=new --disable-gpu --dump-dom \
  --virtual-time-budget=8000 http://127.0.0.1:4000/real-estate/ 2>&1 \
  | grep -iE "error|failed" | head
```

Expected: 출력 없음.

- [ ] **Step 10: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add _pages/real-estate.md assets/realestate/*.css assets/realestate/*.js
git commit -m "Add real estate dashboard page with interactive map

서울/경기/전체 탭은 같은 SVG 의 viewBox 를 바꿔 만든다.
지도 채색은 지표에 따라 순차·발산 램프를 갈아 쓴다."
```

---

## Task 6: KPI 타일과 평당가 추이 차트

**Files:**
- Modify: `assets/realestate/charts.js` (Task 5 의 빈 모듈을 채운다)

**Interfaces:**
- Consumes: `palette.js` 의 `LINE`, `GRID`, `AXIS`, `MUTED`, `INK`; `app.js` 가 넘기는 `(root, summary, detail, state)`
- Produces: `renderPanel(root, summary, detail, state)`, `lineChart(months, values, options) -> string` (SVG 문자열)

- [ ] **Step 1: 차트와 KPI 를 구현한다**

`assets/realestate/charts.js` 전체를 아래로 바꾼다.

```javascript
import { LINE, GRID, AXIS, MUTED, INK } from './palette.js';

const W = 520, H = 190, PAD_L = 46, PAD_R = 10, PAD_T = 12, PAD_B = 24;

function niceTicks(max) {
  const step = Math.pow(10, Math.floor(Math.log10(max))) / 2;
  const out = [];
  for (let v = 0; v <= max; v += step) out.push(v);
  return out.length > 6 ? out.filter((_, i) => i % 2 === 0) : out;
}

export function lineChart(months, values, { partialFrom } = {}) {
  const finite = values.filter((v) => v != null);
  if (!finite.length) {
    return `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" `
      + `text-anchor="middle" font-size="12" fill="${MUTED}">자료 없음</text></svg>`;
  }
  const max = Math.max(...finite) * 1.08;
  const n = months.length;
  const iw = W - PAD_L - PAD_R, ih = H - PAD_T - PAD_B;
  const X = (i) => PAD_L + (n > 1 ? (i / (n - 1)) * iw : iw / 2);
  const Y = (v) => PAD_T + (1 - v / max) * ih;

  const parts = [`<svg viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" `
    + `role="img" aria-label="평당가 추이">`];

  for (const t of niceTicks(max)) {
    parts.push(`<line x1="${PAD_L}" y1="${Y(t).toFixed(1)}" x2="${W - PAD_R}" `
      + `y2="${Y(t).toFixed(1)}" stroke="${GRID}" stroke-width="1"/>`);
    parts.push(`<text x="${PAD_L - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end" `
      + `font-size="9" fill="${MUTED}">${t >= 1000 ? `${Math.round(t / 1000)}천` : Math.round(t)}</text>`);
  }

  for (let i = 0; i < n; i += 1) {
    if (!months[i].endsWith('-01') || Number(months[i].slice(0, 4)) % 5 !== 0) continue;
    parts.push(`<text x="${X(i).toFixed(1)}" y="${H - 7}" text-anchor="middle" `
      + `font-size="9" fill="${MUTED}">${months[i].slice(0, 4)}</text>`);
  }

  parts.push(`<line x1="${PAD_L}" y1="${Y(0).toFixed(1)}" x2="${W - PAD_R}" `
    + `y2="${Y(0).toFixed(1)}" stroke="${AXIS}" stroke-width="1"/>`);

  // 값이 빈 달이 있어도 선이 끊기도록 구간별로 나눠 그린다
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      parts.push(`<polyline points="${run.join(' ')}" fill="none" stroke="${LINE}" `
        + `stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
    }
    run = [];
  };
  values.forEach((v, i) => {
    if (v == null) { flush(); return; }
    run.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  });
  flush();

  let lastIdx = -1;
  for (let i = n - 1; i >= 0; i -= 1) if (values[i] != null) { lastIdx = i; break; }
  if (lastIdx >= 0) {
    const lx = X(lastIdx), ly = Y(values[lastIdx]);
    parts.push(`<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="4" fill="${LINE}" `
      + `stroke="#fcfcfb" stroke-width="2"/>`);
    parts.push(`<text x="${(lx - 7).toFixed(1)}" y="${(ly - 9).toFixed(1)}" text-anchor="end" `
      + `font-size="11" font-weight="600" fill="${INK}">`
      + `${values[lastIdx].toLocaleString()}만원/평</text>`);
  }
  if (partialFrom != null && partialFrom >= 0) {
    parts.push(`<line x1="${X(partialFrom).toFixed(1)}" y1="${PAD_T}" `
      + `x2="${X(partialFrom).toFixed(1)}" y2="${(H - PAD_B).toFixed(1)}" `
      + `stroke="${MUTED}" stroke-width="1" stroke-dasharray="3 3"/>`);
  }
  parts.push('</svg>');
  return parts.join('');
}

function kpi(label, value, unit, delta, direction) {
  const cls = direction > 0 ? 'is-up' : direction < 0 ? 'is-down' : '';
  return `<div class="re-kpi"><div class="re-kpi-label">${label}</div>`
    + `<div class="re-kpi-value">${value}<span>${unit}</span></div>`
    + `<div class="re-kpi-delta ${cls}">${delta}</div></div>`;
}

function pct(v) {
  return v == null ? '자료 없음' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
}

export function renderPanel(root, summary, detail, state) {
  const series = summary.series[state.filter][detail.sgg];
  const index = summary.months.indexOf(state.ym);
  const med = series.med.slice(0, index + 1);
  const months = summary.months.slice(0, index + 1);

  const now = med[index];
  const yoy = index >= 12 && med[index - 12] ? (now / med[index - 12] - 1) * 100 : null;

  let n12 = 0, prev12 = 0;
  for (let i = Math.max(0, index - 11); i <= index; i += 1) n12 += series.n[i] || 0;
  for (let i = Math.max(0, index - 23); i <= index - 12; i += 1) prev12 += series.n[i] || 0;
  const volDelta = prev12 ? (n12 / prev12 - 1) * 100 : null;

  let peak = -Infinity, peakAt = null;
  med.forEach((v, i) => { if (v != null && v > peak) { peak = v; peakAt = months[i]; } });
  const fromPeak = now != null && peak > 0 ? (now / peak - 1) * 100 : null;

  root.querySelector('.re-kpis').innerHTML = [
    kpi('중위 평당가', now != null ? now.toLocaleString() : '—', '만원',
      `전년 대비 ${pct(yoy)}`, yoy == null ? 0 : Math.sign(yoy)),
    kpi('최근 12개월 거래', n12.toLocaleString(), '건',
      `직전 12개월 대비 ${pct(volDelta)}`, volDelta == null ? 0 : Math.sign(volDelta)),
    kpi('전고점 대비', fromPeak != null ? fromPeak.toFixed(1) : '—', '%',
      peakAt ? `${peakAt} ${peak.toLocaleString()}만원/평` : '—', 0),
  ].join('');

  const partialFrom = summary.partial ? months.indexOf(summary.partial) : -1;
  root.querySelector('.re-chart').innerHTML = lineChart(months, med, { partialFrom });

  renderTable(root, detail, state);
}

function renderTable(root, detail, state) {
  const min300 = state.filter === '300';
  const rows = detail.complexes
    .filter((c) => c.n >= 5 && (!min300 || (c.hh != null && c.hh >= 300)))
    .slice(0, 30);
  const head = '<thead><tr><th></th><th>단지</th><th>법정동</th>'
    + '<th class="is-num">평당가(만원)</th><th class="is-num">세대</th>'
    + '<th class="is-num">거래</th></tr></thead>';
  const body = rows.map((c, i) => `<tr><td class="is-dim">${i + 1}</td>`
    + `<td>${c.name}</td><td class="is-dim">${c.dong}</td>`
    + `<td class="is-num">${c.med != null ? c.med.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.hh != null ? c.hh.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.n}</td></tr>`).join('');
  root.querySelector('.re-table').innerHTML = rows.length
    ? `${head}<tbody>${body}</tbody>`
    : `${head}<tbody><tr><td colspan="6">최근 12개월 거래 5건 이상 단지가 없습니다.</td></tr></tbody>`;
}
```

- [ ] **Step 2: 브라우저에서 강남구를 눌러 확인한다**

Docker 서버가 떠 있어야 한다 (Task 5 Step 8). 크롬을 CDP 로 몰아 강남구를
클릭한 뒤 캡처한다.

```bash
cat > /tmp/re-click.py <<'PY'
"""헤드리스 크롬을 CDP 로 몰아 특정 구를 클릭하고 화면을 캡처한다."""
import base64, json, os, socket, subprocess, time, urllib.request, struct, hashlib

PORT = 9333
proc = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu",
                         f"--remote-debugging-port={PORT}", "--hide-scrollbars",
                         "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")

host, rest = ws.split("://")[1].split("/", 1)
h, p = host.split(":")
sock = socket.create_connection((h, int(p)))
key = base64.b64encode(os.urandom(16)).decode()
sock.send((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
buf = b""
while b"\r\n\r\n" not in buf:
    buf += sock.recv(4096)

def send(mid, method, params=None):
    payload = json.dumps({"id": mid, "method": method, "params": params or {}}).encode()
    mask = os.urandom(4)
    n = len(payload)
    hdr = b"\x81"
    if n < 126:
        hdr += bytes([0x80 | n])
    elif n < 65536:
        hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
    sock.send(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

def recv():
    def rd(n):
        out = b""
        while len(out) < n:
            out += sock.recv(n - len(out))
        return out
    b1, b2 = rd(2)
    ln = b2 & 0x7F
    if ln == 126:
        ln = struct.unpack(">H", rd(2))[0]
    elif ln == 127:
        ln = struct.unpack(">Q", rd(8))[0]
    return json.loads(rd(ln))

def call(mid, method, params=None):
    send(mid, method, params)
    while True:
        msg = recv()
        if msg.get("id") == mid:
            return msg

call(1, "Page.enable")
call(2, "Runtime.enable")
call(3, "Emulation.setDeviceMetricsOverride",
     {"width": 1200, "height": 900, "deviceScaleFactor": 1, "mobile": False})
call(4, "Page.navigate", {"url": "http://127.0.0.1:4000/real-estate/"})
time.sleep(6)
call(5, "Runtime.evaluate",
     {"expression": "document.querySelector('path[data-sgg=\"11680\"]').dispatchEvent("
                    "new MouseEvent('click', {bubbles:true}))"})
time.sleep(3)
res = call(6, "Runtime.evaluate", {
    "expression": "JSON.stringify({title: document.querySelector('.re-panel-title').textContent,"
                  "kpis: document.querySelectorAll('.re-kpi').length,"
                  "chart: document.querySelectorAll('.re-chart svg polyline').length,"
                  "rows: document.querySelectorAll('.re-table tbody tr').length,"
                  "overflow: document.documentElement.scrollWidth})",
    "returnByValue": True})
print(res["result"]["result"]["value"])
shot = call(7, "Page.captureScreenshot", {"format": "png"})
open("/tmp/re-gangnam.png", "wb").write(base64.b64decode(shot["result"]["data"]))
proc.terminate()
PY
python3 /tmp/re-click.py
```

Expected 출력:

```
{"title":"강남구","kpis":3,"chart":1,"rows":30,"overflow":1200}
```

- `title` 이 `강남구` — 구 선택이 패널에 반영됐다
- `kpis` 가 3 — KPI 타일 3장
- `chart` 가 1 이상 — 라인이 그려졌다 (0 이면 데이터가 안 붙은 것)
- `rows` 가 1 이상 — 단지 표가 채워졌다
- `overflow` 가 1200 — 데스크톱에서 가로로 밀리지 않았다

`/tmp/re-gangnam.png` 를 Read 로 열어 눈으로도 확인한다. KPI 중위 평당가가
`12,252만원` (기본 기준월이 직전 완료 월인 2026-06 이므로) 이어야 하고, 라인이
2006년 왼쪽 아래에서 2026년 오른쪽 위로 올라가야 한다.

- [ ] **Step 3: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add assets/realestate/charts.js
git commit -m "Render KPI tiles, price trend chart and complex table

선택한 구의 22년치 중위 평당가를 라인으로 그리고 최근 12개월 단지 랭킹을 붙인다.
자료가 빈 달에서는 선을 끊어 없는 값을 이어 그리지 않는다."
```

---

## Task 7: 모바일 대응 검증

**Files:**
- Modify: `assets/realestate/dashboard.css` (측정 결과에 따라)
- Create: `_dev/tools/check_mobile.py` (재사용할 검증 스크립트)

**Interfaces:**
- Consumes: Task 5·6 의 페이지
- Produces: `_dev/tools/check_mobile.py` — 390px 에서 `scrollWidth` 를 재고 합격/불합격을 출력

저장소 규칙상 **모든 페이지가 390px 에서 가로로 넘치면 안 된다.** 스크린샷은
뷰포트 폭으로 잘려 오버플로가 드러나지 않으므로 JS 로 `scrollWidth` 를 재야 한다.

- [ ] **Step 1: 검증 스크립트를 만든다**

`_dev/tools/check_mobile.py`:

```python
"""390px 뷰포트에서 가로 오버플로가 있는지 잰다.

    python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/

이 환경에는 puppeteer/selenium 이 없다. 원시 소켓으로 CDP WebSocket 을 직접 쓴다.
스크린샷은 뷰포트 폭으로 잘려 오버플로가 안 보이므로 scrollWidth 를 재는 게 핵심이다.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request

WIDTH = 390
HEIGHT = 844
PORT = 9334


class CDP:
    def __init__(self, ws_url: str) -> None:
        host, rest = ws_url.split("://")[1].split("/", 1)
        h, p = host.split(":")
        self.sock = socket.create_connection((h, int(p)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.send((
            f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        self.mid = 0

    def _send(self, method: str, params: dict) -> int:
        self.mid += 1
        payload = json.dumps({"id": self.mid, "method": method,
                              "params": params}).encode()
        mask = os.urandom(4)
        n = len(payload)
        hdr = b"\x81"
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.send(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        return self.mid

    def _read(self) -> dict:
        def rd(n: int) -> bytes:
            out = b""
            while len(out) < n:
                out += self.sock.recv(n - len(out))
            return out
        _, b2 = rd(2)
        ln = b2 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", rd(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", rd(8))[0]
        return json.loads(rd(ln))

    def call(self, method: str, params: dict | None = None) -> dict:
        mid = self._send(method, params or {})
        while True:
            msg = self._read()
            if msg.get("id") == mid:
                return msg


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: check_mobile.py <URL>", file=sys.stderr)
        return 2
    url = sys.argv[1]

    proc = subprocess.Popen(
        ["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={PORT}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3)
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        cdp = CDP(ws)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride",
                 {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": 2, "mobile": True})
        cdp.call("Page.navigate", {"url": url})
        time.sleep(7)
        # 구를 하나 눌러 표까지 채운 상태로 잰다. 빈 표는 넘칠 수가 없다.
        cdp.call("Runtime.evaluate", {"expression":
            "(document.querySelector('path[data-sgg=\"11680\"]')"
            "||document.querySelector('path[data-sgg]'))"
            ".dispatchEvent(new MouseEvent('click',{bubbles:true}))"})
        time.sleep(3)

        res = cdp.call("Runtime.evaluate", {"returnByValue": True, "expression": """
          (() => {
            const doc = document.documentElement.scrollWidth;
            const bad = [];
            for (const el of document.querySelectorAll('.re-app *')) {
              const r = el.getBoundingClientRect();
              if (r.width <= window.innerWidth + 1) continue;
              let p = el.parentElement, scrollable = false;
              while (p) {
                if (getComputedStyle(p).overflowX === 'auto'
                    || getComputedStyle(p).overflowX === 'scroll') { scrollable = true; break; }
                p = p.parentElement;
              }
              if (!scrollable) bad.push(el.className + ' w=' + Math.round(r.width));
            }
            return JSON.stringify({doc, inner: window.innerWidth, bad: bad.slice(0, 8)});
          })()
        """})
        out = json.loads(res["result"]["result"]["value"])
        shot = cdp.call("Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": True})
        open("/tmp/re-mobile.png", "wb").write(base64.b64decode(shot["result"]["data"]))

        ok = out["doc"] <= out["inner"] and not out["bad"]
        print(f"scrollWidth={out['doc']} innerWidth={out['inner']}")
        if out["bad"]:
            print("스크롤 컨테이너 밖에서 넘치는 요소:")
            for b in out["bad"]:
                print(f"  - {b}")
        print("합격" if ok else "불합격")
        print("스크린샷: /tmp/re-mobile.png")
        return 0 if ok else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 측정한다**

Docker 서버가 떠 있어야 한다.

```bash
python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/
```

Expected: `scrollWidth=390 innerWidth=390` 그리고 `합격`

- [ ] **Step 3: 불합격이면 원인별로 고친다**

`불합격` 이면 출력에 넘치는 요소가 나온다. 원인별 처방:

| 넘치는 요소 | 처방 |
|---|---|
| `re-table` | `.re-table-wrap` 이 `overflow-x:auto` 인지 확인. 표 자체에 `min-width` 를 주지 말 것 |
| `re-controls` / `re-filters` | 이미 `flex-wrap:wrap` 이다. `select` 에 `max-width:100%` 를 추가 |
| `re-chart svg` | SVG 에 `width:100%;height:auto` 가 걸렸는지 확인 |
| `re-map` | `svg.re-map { width:100% }` 확인. `viewBox` 는 폭을 늘리지 않는다 |
| `re-tip` | 툴팁이 화면 밖으로 나간 것. `.re-tip` 에 `max-width:60vw` 추가 |

고친 뒤 Docker 를 다시 빌드하고 Step 2 를 반복한다.

```bash
docker exec kayser_serve sh -c "cd /srv/jekyll && jekyll build" || \
  docker run --rm -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
    -v "$PWD":/srv/jekyll -w /srv/jekyll jekyll/jekyll:4.2.2 \
    sh -c "bundle install && jekyll build"
```

- [ ] **Step 4: 스크린샷을 눈으로 확인한다**

`/tmp/re-mobile.png` 를 Read 로 연다. 확인할 것:

- 지도가 화면 폭에 맞게 들어가고 서울 구가 손가락으로 누를 만한 크기다
- 컨트롤이 두 줄로 접혔고 잘리지 않았다
- KPI 3장이 세로로 쌓였다
- 표가 오른쪽으로 잘려 보이더라도, 표를 옆으로 밀면 나머지 컬럼이 나온다
  (페이지 전체가 아니라 표만 스크롤돼야 한다)

- [ ] **Step 5: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add _dev/tools/check_mobile.py assets/realestate/dashboard.css
git commit -m "Verify dashboard has no horizontal overflow at 390px

CDP 로 scrollWidth 를 재는 검증 스크립트를 넣었다. 스크린샷은 뷰포트 폭으로
잘려 오버플로가 드러나지 않기 때문이다."
```

---

## Task 8: URL 파라미터, 네비게이션, 일일 빌드 연동

**Files:**
- Modify: `assets/realestate/app.js`
- Modify: `_data/navigation.yml`
- Modify: `scripts/daily.sh`

**Interfaces:**
- Consumes: Task 5 의 `state` 객체와 `start()`
- Produces: `readParams()`, `writeParams()` (`app.js` 내부 함수)

- [ ] **Step 1: URL 파라미터를 붙인다**

`assets/realestate/app.js` 의 `async function start()` **앞에** 아래 두 함수를 넣는다.

```javascript
// 글에서 특정 화면을 바로 가리킬 수 있게 상태를 주소에 싣는다.
//   /real-estate/?sgg=11680&metric=chg12&ym=2026-06&filter=300&view=seoul
function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  if (['seoul', 'gyeonggi', 'all'].includes(view)) state.view = view;
  const metric = q.get('metric');
  if (metric && METRICS[metric]) state.metric = metric;
  const filter = q.get('filter');
  if (filter === '300' || filter === 'all') state.filter = filter;
  const ym = q.get('ym');
  if (ym && summary.months.includes(ym)) state.ym = ym;
  const sgg = q.get('sgg');
  if (sgg && summary.sgg[sgg]) state.sgg = sgg;
}

function writeParams() {
  const q = new URLSearchParams();
  q.set('view', state.view);
  q.set('metric', state.metric);
  q.set('ym', state.ym);
  q.set('filter', state.filter);
  if (state.sgg) q.set('sgg', state.sgg);
  window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
}
```

`repaint()` 의 마지막 줄 `map.paint(values);` 바로 아래에 `writeParams();` 를 넣는다.

`selectSgg()` 의 `renderPanel(root, summary, detail, state);` 아래에도
`writeParams();` 를 넣는다.

`start()` 를 아래로 바꾼다 (`readParams()` 호출과 초기 선택 반영이 추가된다).

```javascript
async function start() {
  summary = await loadSummary();
  // 마지막 달은 신고가 덜 들어와 항상 미완성이다. 직전 완료 월을 기본으로 둔다.
  const last = summary.months.length - 1;
  state.ym = summary.months[Math.max(0, last - 1)];
  readParams();

  map = initMap(root, { onSelect: selectSgg });
  fillMonths();
  bind();

  root.querySelectorAll('.re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  root.querySelector('.re-metric').value = state.metric;
  const toggle = root.querySelector('.re-toggle');
  const on300 = state.filter === '300';
  toggle.classList.toggle('is-on', on300);
  toggle.setAttribute('aria-pressed', String(on300));
  toggle.textContent = on300 ? '300세대+' : '전체 거래';

  map.setView(state.view);
  repaint();
  if (state.sgg) await selectSgg(state.sgg);

  root.querySelector('.re-footnote').textContent =
    `국토교통부 실거래가 · ${summary.months[0]} ~ ${summary.months[last]} · 갱신 ${summary.generated}`;
}
```

- [ ] **Step 2: 파라미터가 먹는지 확인한다**

Docker 를 다시 빌드한 뒤:

```bash
python3 - <<'PY'
import json, subprocess
out = subprocess.run([
  "google-chrome", "--headless=new", "--disable-gpu", "--virtual-time-budget=9000",
  "--dump-dom",
  "http://127.0.0.1:4000/real-estate/?sgg=41135&metric=level&view=gyeonggi&filter=all",
], capture_output=True, text=True).stdout
print("성남시분당구" in out, "패널 제목 반영")
PY
```

Expected: `True 패널 제목 반영`

- [ ] **Step 3: 네비게이션에 추가한다**

`_data/navigation.yml` 의 `main:` 목록 **맨 앞에** 넣는다.

```yaml
main:
  - title: "부동산"
    url: /real-estate/
  - title: "Category"
    url: /categories/
```

- [ ] **Step 4: 일일 빌드에 집계를 연결한다**

`scripts/daily.sh` 의 아래 줄을

```bash
python3 -u scripts/collect_trades.py --max-calls "$MAX_CALLS"
```

아래로 바꾼다.

```bash
python3 -u scripts/collect_trades.py --max-calls "$MAX_CALLS"

# 수집이 일일 한도로 중간에 멈춘 날에도 집계는 돌린다.
# 그날까지 받은 데이터로 만든 대시보드가 어제 것보다 낫다.
echo "----- 집계 시작 -----"
python3 -u scripts/build_dashboard.py
```

그리고 커밋 대상에 집계본을 더한다. 아래 줄을

```bash
if [ -z "$(git status --porcelain data)" ]; then
```

아래로 바꾼다.

```bash
if [ -z "$(git status --porcelain data assets/realestate)" ]; then
```

그리고

```bash
git add data
```

를 아래로 바꾼다.

```bash
git add data assets/realestate
```

커밋 메시지도 집계본을 포함하도록 바꾼다.

```bash
git commit -q -m "Accumulate Seoul/Gyeonggi apartment trade data

수집 스크립트가 자동 갱신한 월별 실거래가 파일 ${MONTHS}개와 대시보드 집계본."
```

- [ ] **Step 5: 일일 빌드를 수집 없이 예행한다**

```bash
MAX_CALLS=0 AUTO_COMMIT=0 bash scripts/daily.sh
```

Expected: 수집이 0콜로 끝나고 `집계 시작` 뒤 `summary.json ... 변경 없음`,
`구별 JSON 72개 중 0개 갱신` 이 나온다. 이미 최신이므로 아무것도 안 바뀌는 게 정상이다.

- [ ] **Step 6: 전체 테스트를 돌린다**

```bash
python3 -m unittest discover -s tests -v
```

Expected: 전부 PASS (기존 `test_collector.py` 포함).

- [ ] **Step 7: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add assets/realestate/app.js _data/navigation.yml scripts/daily.sh
git commit -m "Wire dashboard URL params, nav entry and daily aggregation

글에서 특정 화면을 바로 가리킬 수 있게 상태를 주소에 싣는다.
매일 수집이 끝나면 집계까지 돌려 변경분만 커밋한다."
```

- [ ] **Step 8: 정리하고 푸시 여부를 묻는다**

```bash
docker rm -f kayser_serve 2>/dev/null
rm -f _tmp_Gemfile
git status --short
```

`_tmp_Gemfile` 은 커밋하지 않는다. 푸시는 **사용자가 명시적으로 요청할 때만**
한다 — 저장소 규칙이다.

---

## 자체 검토

### 설계 문서 대비 커버리지

| 설계 항목 | 담당 |
|---|---|
| 경계 병합 ver20260701 → 72개 검증 | Task 1 |
| 단순화 eps·예산·빈 path 검사 | Task 1 Step 4·6 |
| 등장방형 투영 + 위도 보정 | Task 1 |
| 평당가 = price / (area/3.3058) | Task 2 |
| 중위값·변화율·전고점·회전율 | Task 2 |
| 얇은 달 3개월 이동중위 | Task 2 (`rolling_median`), Task 5 (`smoothed`) |
| 계약해제 가격 제외·건수 보존 | Task 3 |
| 세대수 2단 조인 (PNU → 단지명) | Task 3 (`join_household`) |
| 300세대+ / 전체 두 갈래 집계 | Task 3 |
| `summary.json` 스키마·200KB 예산 | Task 3 |
| 결정론적 JSON | Task 3 (`write_json`) + Step 7 |
| 강남구 실측 대조 | Task 3 Step 6 |
| 구별 단지 집계·300KB 예산 | Task 4 |
| 면적 구간 | Task 2 (`area_bucket`), Task 4 |
| 탭 전환 = viewBox 변경 | Task 5 (`map.js`) |
| 2단 레이아웃 + 하단 표 | Task 5 (CSS `.re-body`) |
| 지표 5종 셀렉트 | Task 5 (`METRICS`) |
| 순차/발산 램프, 팔레트 고정값 | Task 5 (`palette.js`) |
| 범례·툴팁·키보드 접근 | Task 5 (`map.js`, CSS) |
| 미완성 기준월 기본값 | Task 5 (`start()`), Task 6 (차트 점선) |
| KPI 3장 | Task 6 |
| 라인 차트 (범례 없음, 단일 계열) | Task 6 |
| 단지 표 5건 이상 | Task 6 (`renderTable`) |
| 390px 오버플로 0 | Task 7 |
| URL 파라미터 | Task 8 |
| 카테고리 루트 `/real-estate/` | Task 5 (`permalink`) |
| 네비게이션 추가 | Task 8 |
| `daily.sh` 연동 | Task 8 |
| 계약해제율 지표 (2차) | 범위 밖 — 데이터만 Task 3 에서 확보 |

빠진 항목 없음.

### 이름 일관성

- `join_household`, `normalize_name`, `write_json`, `build_summary`,
  `build_sgg_detail` — Task 3 에서 정의하고 Task 4 에서 같은 이름으로 쓴다
- `initMap` 이 돌려주는 핸들의 `paint` / `setView` / `setSelected` / `codesIn` —
  Task 5 `map.js` 에서 정의하고 같은 파일의 `app.js` 에서 그 이름으로 부른다
- `renderPanel(root, summary, detail, state)` — Task 5 에서 빈 껍데기로 만들고
  Task 6 에서 같은 시그니처로 채운다
- `METRICS` 는 `app.js` 안에서만 쓴다. Task 8 의 `readParams()` 도 같은 파일이라
  접근할 수 있다
- `aggregate.AREA_EDGES` — Task 2 에서 정의하고 Task 4 에서 버킷 개수 계산에 쓴다

### 남는 위험

1. **`load_complexes()` 의 법정동 파싱** — `address` 를 공백으로 쪼개 뒤에서 두
   번째를 법정동으로 본다. `서울특별시 강남구 역삼동 755-1` 에서는 맞지만 읍면
   지역(`경기도 여주시 가남읍 태평리 123`)에서는 `태평리` 가 아니라 `가남읍` 이
   잡힐 수 있다. Task 3 Step 6 의 강남구 대조는 이걸 잡아내지 못한다. 2차 조인
   실패는 세대수 미상으로만 이어져 300세대+ 집계에서 빠지는 정도라 치명적이진
   않지만, Task 4 실행 후 세대수 미상 비율이 10% 를 넘으면 파싱을 고쳐야 한다.
2. **`build_dashboard.py` 가 435만 행을 전부 메모리에 올린다** (`by_month` 딕셔너리).
   행당 16개 문자열이라 대략 3~5GB 를 쓸 수 있다. Task 3 Step 5 에서 메모리
   부족이 나면 월별로 스트리밍하도록 고쳐야 한다 — `build_summary` 와
   `build_sgg_detail` 이 둘 다 `by_month` 를 훑으므로, 월 단위 누산기로 바꾸면
   된다.
3. **Docker 빌드 환경** — 이 리눅스 머신에 루비가 없어 `run.sh` 가 동작하지
   않는다. Task 5 Step 8 의 Docker 절차가 실패하면 이후 브라우저 검증
   (Task 5·6·7·8)이 전부 막힌다. 그 경우 GitHub Pages 에 푸시한 뒤 라이브 URL 로
   검증하는 방법으로 우회한다 (빌드에 50~60초).
