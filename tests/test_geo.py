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


class TestDissolve(unittest.TestCase):
    """행정동 경계 상쇄. 겹치는 변만 놓치면 대시보드 지도가 동 단위로 보인다."""

    def test_dissolves_shared_edge_into_one_ring(self) -> None:
        # 두 1x1 정사각형이 x=1 변을 공유 -> 2x1 직사각형 하나(꼭짓점 4개)로 합쳐져야 한다
        a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        b = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        rings = build_geo.dissolve([a, b])
        self.assertEqual(len(rings), 1)
        corners = rings[0][:-1] if rings[0][0] == rings[0][-1] else rings[0]
        self.assertEqual(len(corners), 4)
        self.assertEqual(
            {tuple(p) for p in corners},
            {(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)},
        )

    def test_disjoint_squares_stay_separate(self) -> None:
        a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        b = [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.0, 5.0]]
        rings = build_geo.dissolve([a, b])
        self.assertEqual(len(rings), 2)

    def test_handles_mixed_input_winding(self) -> None:
        """원본 링의 시계/반시계 방향이 섞여 있어도 상쇄가 되어야 한다."""
        a_cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        b_ccw = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        rings = build_geo.dissolve([a_cw, b_ccw])
        self.assertEqual(len(rings), 1)
        corners = rings[0][:-1] if rings[0][0] == rings[0][-1] else rings[0]
        self.assertEqual(len(corners), 4)


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


class TestProjectionParams(unittest.TestCase):
    RING = [[126.0, 37.0], [127.0, 37.0], [127.0, 38.0], [126.0, 37.0]]

    def test_returns_every_key_the_consumer_needs(self) -> None:
        p = build_geo.projection_params({"11680": [self.RING]}, 1000.0)
        self.assertEqual(
            sorted(p),
            ["height", "k", "max_lat", "min_lon", "span_x", "span_y", "width"],
        )

    def test_matches_project_output(self) -> None:
        """같은 점을 params 로 직접 변환한 값과 project() 결과가 같아야 한다."""
        rings = {"11680": [self.RING]}
        projected, _, _ = build_geo.project(rings, 1000.0)
        p = build_geo.projection_params(rings, 1000.0)
        lon, lat = self.RING[1]
        x = (lon - p["min_lon"]) * p["k"] / p["span_x"] * p["width"]
        y = (p["max_lat"] - lat) / p["span_y"] * p["height"]
        got_x, got_y = projected["11680"][0][1]
        self.assertAlmostEqual(x, got_x, places=9)
        self.assertAlmostEqual(y, got_y, places=9)

    def test_height_follows_aspect_ratio(self) -> None:
        p = build_geo.projection_params({"x": [self.RING]}, 1000.0)
        self.assertAlmostEqual(p["height"], p["width"] * p["span_y"] / p["span_x"], places=9)


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


class TestBuildParser(unittest.TestCase):
    def test_eps_default_reproduces_committed_svg(self) -> None:
        """기본 --eps 는 dissolve 도입 후 커밋된 지도를 낸 값(0.05)과 같아야 한다.

        어긋나면 인자 없이 재실행했을 때 커밋된 SVG 와 다른(더 거친) 지도가
        조용히 만들어진다.
        """
        args = build_geo.build_parser().parse_args(["--input", "x.geojson"])
        self.assertEqual(args.eps, 0.05)


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


class TestSidoFilter(unittest.TestCase):
    """전국 지도를 만들려면 시도 필터가 인자여야 한다. 기본값은 바뀌면 안 된다."""

    SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]

    def features(self) -> list[dict]:
        return [
            _feature("11110", "종로구", self.SQUARE),
            _feature("41111", "수원시장안구", self.SQUARE),
            _feature("50110", "제주시", self.SQUARE),
        ]

    def test_default_keeps_only_seoul_and_gyeonggi(self) -> None:
        """기본 동작이 바뀌면 커밋된 map.svg 가 조용히 달라진다."""
        merged = build_geo.merge_sgg(self.features())
        self.assertEqual(set(merged), {"11110", "41111"})

    def test_none_keeps_every_sido(self) -> None:
        merged = build_geo.merge_sgg(self.features(), sido=None)
        self.assertEqual(set(merged), {"11110", "41111", "50110"})

    def test_explicit_sido_narrows_to_it(self) -> None:
        merged = build_geo.merge_sgg(self.features(), sido=("50",))
        self.assertEqual(set(merged), {"50110"})


class TestSggNames(unittest.TestCase):
    """전국에는 서울·경기 코드표(regions)가 없다. 이름을 GeoJSON 에서 뽑는다."""

    SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]

    def test_reads_the_short_name_from_the_feature(self) -> None:
        names = build_geo.sgg_names([_feature("50110", "제주시", self.SQUARE)])
        self.assertEqual(names, {"50110": "제주시"})

    def test_one_name_per_sgg_even_with_many_dong(self) -> None:
        feats = [_feature("11110", "종로구", self.SQUARE) for _ in range(5)]
        self.assertEqual(build_geo.sgg_names(feats), {"11110": "종로구"})

    def test_missing_name_falls_back_to_the_code(self) -> None:
        """이름이 비어도 시군구를 지도에서 떨어뜨리지 않는다."""
        feat = _feature("99999", "", self.SQUARE)
        self.assertEqual(build_geo.sgg_names([feat]), {"99999": "99999"})


class TestSvgLabel(unittest.TestCase):
    """전국 지도에 '서울·경기' 라고 적히면 스크린리더가 틀린 말을 읽는다."""

    SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]

    def svg(self, **kwargs) -> str:
        projected = {"11110": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]}
        return build_geo.to_svg(projected, {"11110": "종로구"}, 10.0, 10.0, **kwargs)

    def test_default_label_unchanged(self) -> None:
        self.assertIn('aria-label="서울·경기 시군구 지도"', self.svg())

    def test_label_can_be_set(self) -> None:
        self.assertIn('aria-label="전국 시군구 지도"', self.svg(label="전국 시군구 지도"))
