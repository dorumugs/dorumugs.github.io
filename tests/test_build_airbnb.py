"""Airbnb 좌표 집계 검증 — 시군구 배정·격자·면적 밀도.

    python3 -m unittest tests.test_build_airbnb -v

좌표를 엉뚱한 시군구에 넣으면 지도는 멀쩡해 보이면서 숫자만 틀린다.
합이 맞는지, 버린 좌표를 세는지가 핵심이다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_airbnb  # noqa: E402

# 한 변 1도짜리 정사각형 두 개. 겹치지 않는다.
WEST = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
EAST = [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [2.0, 0.0]]


def geo(*features) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def feature(code: str, name: str, *rings) -> dict:
    return {
        "type": "Feature",
        "properties": {"sgg": code, "name": name},
        "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]},
    }


class AssignSgg(unittest.TestCase):
    def setUp(self) -> None:
        self.areas = build_airbnb.load_polygons(
            geo(feature("11110", "종로구", WEST), feature("50110", "제주시", EAST)))

    def test_point_lands_in_the_polygon_that_holds_it(self):
        self.assertEqual(build_airbnb.find_sgg(0.5, 0.5, self.areas), "11110")
        self.assertEqual(build_airbnb.find_sgg(0.5, 2.5, self.areas), "50110")

    def test_point_outside_every_polygon_is_none(self):
        """바다 위 좌표를 아무 시군구에나 밀어 넣으면 그 구 숫자가 부풀어 오른다."""
        self.assertIsNone(build_airbnb.find_sgg(0.5, 9.0, self.areas))

    def test_latitude_and_longitude_are_not_swapped(self):
        """GeoJSON 은 [경도, 위도] 다. 뒤집으면 전국이 조용히 어긋난다."""
        self.assertIsNone(build_airbnb.find_sgg(2.5, 0.5, self.areas))

    def test_multipolygon_parts_all_count(self):
        areas = build_airbnb.load_polygons(geo(feature("50110", "제주시", WEST, EAST)))
        self.assertEqual(build_airbnb.find_sgg(0.5, 0.5, areas), "50110")
        self.assertEqual(build_airbnb.find_sgg(0.5, 2.5, areas), "50110")


class Aggregate(unittest.TestCase):
    def setUp(self) -> None:
        self.areas = build_airbnb.load_polygons(
            geo(feature("11110", "종로구", WEST), feature("50110", "제주시", EAST)))

    def test_counts_every_point_it_placed(self):
        points = [(0.1, 0.1), (0.2, 0.2), (0.5, 2.5)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(result["sgg"]["11110"]["count"], 2)
        self.assertEqual(result["sgg"]["50110"]["count"], 1)

    def test_sgg_counts_and_placed_total_agree(self):
        """합이 안 맞으면 어딘가에서 좌표가 새고 있다는 뜻이다."""
        points = [(0.1, 0.1), (0.2, 0.2), (0.5, 2.5), (9.0, 9.0)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(sum(s["count"] for s in result["sgg"].values()),
                         result["meta"]["placed"])

    def test_points_outside_are_counted_not_silently_dropped(self):
        points = [(0.1, 0.1), (9.0, 9.0), (9.1, 9.1)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(result["meta"]["dropped"], 2)
        self.assertEqual(result["meta"]["total"], 3)

    def test_empty_sgg_still_appears_with_zero(self):
        """지도에서 시군구가 사라지는 대신 0 으로 칠해져야 한다."""
        result = build_airbnb.aggregate([(0.1, 0.1)], self.areas, grid=0.005)
        self.assertEqual(result["sgg"]["50110"]["count"], 0)

    def test_grid_merges_neighbours_into_one_cell(self):
        points = [(0.1001, 0.1001), (0.1002, 0.1002)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(len(result["grid"]), 1)
        self.assertEqual(result["grid"][0][2], 2)

    def test_grid_keeps_distant_points_apart(self):
        points = [(0.1, 0.1), (0.9, 0.9)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(len(result["grid"]), 2)

    def test_grid_counts_sum_to_placed(self):
        points = [(0.1, 0.1), (0.1001, 0.1001), (0.5, 2.5), (9.0, 9.0)]
        result = build_airbnb.aggregate(points, self.areas, grid=0.005)
        self.assertEqual(sum(cell[2] for cell in result["grid"]),
                         result["meta"]["placed"])

    def test_grid_is_ordered_so_the_file_is_reproducible(self):
        forward = build_airbnb.aggregate([(0.9, 0.9), (0.1, 0.1)], self.areas, grid=0.005)
        backward = build_airbnb.aggregate([(0.1, 0.1), (0.9, 0.9)], self.areas, grid=0.005)
        self.assertEqual(forward["grid"], backward["grid"])


class Density(unittest.TestCase):
    """절대 수로 칠하면 서울이 다 먹는다. 면적당으로 칠한다."""

    def test_area_of_a_unit_square_near_the_equator(self):
        area = build_airbnb.ring_area_km2([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertAlmostEqual(area, 111.32 * 111.32, delta=500)

    def test_area_shrinks_with_latitude(self):
        low = build_airbnb.ring_area_km2([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        high = build_airbnb.ring_area_km2(
            [[0.0, 60.0], [1.0, 60.0], [1.0, 61.0], [0.0, 61.0]])
        self.assertLess(high, low)

    def test_density_is_count_over_area(self):
        areas = build_airbnb.load_polygons(geo(feature("11110", "종로구", WEST)))
        result = build_airbnb.aggregate([(0.1, 0.1)] , areas, grid=0.005)
        entry = result["sgg"]["11110"]
        self.assertAlmostEqual(entry["density"], entry["count"] / entry["area_km2"], places=6)

    def test_zero_area_does_not_divide_by_zero(self):
        degenerate = build_airbnb.load_polygons(
            geo(feature("99999", "점", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])))
        result = build_airbnb.aggregate([], degenerate, grid=0.005)
        self.assertEqual(result["sgg"]["99999"]["density"], 0.0)


class Points(unittest.TestCase):
    """시군구별 점 파일 — 화면에서 그 지역을 볼 때만 받아간다."""

    def setUp(self) -> None:
        self.areas = build_airbnb.load_polygons(
            geo(feature("11110", "종로구", WEST), feature("50110", "제주시", EAST)))

    def test_splits_points_by_sgg(self):
        groups = build_airbnb.points_by_sgg([(0.1, 0.1), (0.5, 2.5)], self.areas)
        self.assertEqual(groups["11110"], [[0.1, 0.1]])
        self.assertEqual(groups["50110"], [[0.5, 2.5]])

    def test_drops_points_outside_every_sgg(self):
        groups = build_airbnb.points_by_sgg([(9.0, 9.0)], self.areas)
        self.assertEqual(groups, {})

    def test_points_are_sorted_so_the_file_is_reproducible(self):
        a = build_airbnb.points_by_sgg([(0.9, 0.9), (0.1, 0.1)], self.areas)
        b = build_airbnb.points_by_sgg([(0.1, 0.1), (0.9, 0.9)], self.areas)
        self.assertEqual(a, b)

    def test_coordinates_are_rounded_to_keep_the_file_small(self):
        groups = build_airbnb.points_by_sgg([(0.123456789, 0.987654321)], self.areas)
        for lat, lng in groups["11110"]:
            self.assertEqual(lat, round(lat, 5))
            self.assertEqual(lng, round(lng, 5))


class WeakCells(unittest.TestCase):
    """상한에 걸려 덜 걷힌 구역은 화면에서 '실제보다 적음' 이라 밝혀야 한다.

    덜 걷힌 bbox 는 400m(분할 하한), 격자 칸은 500m 다. **bbox 가 칸보다
    작으므로** 칸의 원점만 보면 대부분 놓친다 — 실측으로 bbox 13개 중 6개만
    잡혔다. 칸과 bbox 가 겹치는지를 본다.
    """

    CELL = 0.005

    def test_marks_a_cell_the_box_covers(self):
        weak = [[0.52, 0.52, 0.50, 0.50]]
        self.assertTrue(build_airbnb.is_weak(0.505, 0.505, weak, self.CELL))

    def test_marks_a_cell_when_the_box_is_smaller_and_sits_inside_it(self):
        """400m bbox 가 500m 칸 안에 쏙 들어가면 칸의 원점을 품지 않는다."""
        weak = [[0.5040, 0.5040, 0.5020, 0.5020]]
        self.assertTrue(build_airbnb.is_weak(0.500, 0.500, weak, self.CELL))

    def test_marks_a_cell_the_box_only_clips(self):
        weak = [[0.5010, 0.5010, 0.4995, 0.4995]]
        self.assertTrue(build_airbnb.is_weak(0.500, 0.500, weak, self.CELL))

    def test_leaves_other_cells_alone(self):
        weak = [[0.52, 0.52, 0.50, 0.50]]
        self.assertFalse(build_airbnb.is_weak(0.1, 0.1, weak, self.CELL))

    def test_the_next_cell_over_is_not_marked(self):
        weak = [[0.5040, 0.5040, 0.5020, 0.5020]]
        self.assertFalse(build_airbnb.is_weak(0.495, 0.500, weak, self.CELL))

    def test_no_weak_boxes_means_nothing_is_marked(self):
        self.assertFalse(build_airbnb.is_weak(0.505, 0.505, [], self.CELL))


class SggBbox(unittest.TestCase):
    """화면이 시도를 고를 때 그 범위의 격자만 그리려면 시군구 경계 상자가 필요하다."""

    def setUp(self) -> None:
        self.areas = build_airbnb.load_polygons(
            geo(feature("11110", "종로구", WEST), feature("50110", "제주시", EAST)))

    def test_bbox_is_lat_min_max_then_lng_min_max(self):
        boxes = build_airbnb.sgg_bboxes(self.areas)
        self.assertEqual(boxes["11110"], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(boxes["50110"], [0.0, 1.0, 2.0, 3.0])

    def test_multipolygon_bbox_spans_every_part(self):
        areas = build_airbnb.load_polygons(geo(feature("50110", "제주시", WEST, EAST)))
        self.assertEqual(build_airbnb.sgg_bboxes(areas)["50110"], [0.0, 1.0, 0.0, 3.0])

    def test_aggregate_carries_the_boxes(self):
        result = build_airbnb.aggregate([(0.1, 0.1)], self.areas, grid=0.005)
        self.assertEqual(set(result["sgg_bbox"]), {"11110", "50110"})


class WeakMeta(unittest.TestCase):
    """덜걷힌 구역 수는 집계가 실제로 쓴 목록에서 나와야 한다.

    화면이 "도심 N곳은 실제보다 적게 잡힙니다" 라고 말하는 근거다. 다른
    목록을 세면 경고가 있는데도 0 이라고 말한다(실제로 그랬다).
    """

    def setUp(self) -> None:
        self.areas = build_airbnb.load_polygons(geo(feature("11110", "종로구", WEST)))

    def test_counts_the_boxes_it_was_given(self):
        weak = [[0.52, 0.52, 0.50, 0.50], [0.62, 0.62, 0.60, 0.60]]
        result = build_airbnb.aggregate([(0.1, 0.1)], self.areas, 0.005, weak)
        self.assertEqual(result["meta"]["weak_boxes"], 2)

    def test_no_boxes_means_zero(self):
        result = build_airbnb.aggregate([(0.1, 0.1)], self.areas, 0.005, [])
        self.assertEqual(result["meta"]["weak_boxes"], 0)

    def test_a_cell_inside_a_weak_box_is_flagged(self):
        weak = [[0.52, 0.52, 0.50, 0.50]]
        result = build_airbnb.aggregate([(0.505, 0.505), (0.1, 0.1)],
                                        self.areas, 0.005, weak)
        flags = {(lat, lng): w for lat, lng, _, w in result["grid"]}
        self.assertEqual(flags[build_airbnb.api.snap(0.505, 0.505, 0.005)], 1)
        self.assertEqual(flags[build_airbnb.api.snap(0.1, 0.1, 0.005)], 0)
