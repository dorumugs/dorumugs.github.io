"""Airbnb 지도검색 HTML 파싱 검증.

    python3 -m unittest tests.test_airbnb_api -v

fixtures/airbnb_search.html.gz 는 2026-09-22 실제 응답에서 좌표가 든
`data-deferred-state-0` 스크립트와 총건수 문구만 잘라낸 것이다. Airbnb 는 공식
API 가 아니라 화면이 바뀌면 조용히 깨진다 — 그때 여기서 잡는다.

고정된 사실(실측):
    bbox 37.49,127.02 ~ 37.52,127.05
    좌표 20건 · 커서 15장 · 총건수 712
"""

from __future__ import annotations

import gzip
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airbnb_api  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
BOX = airbnb_api.Box(ne_lat=37.52, ne_lng=127.05, sw_lat=37.49, sw_lng=127.02)


def html(name: str = "airbnb_search.html.gz") -> str:
    with gzip.open(FIXTURES / name, "rb") as f:
        return f.read().decode("utf-8")


class SearchUrl(unittest.TestCase):
    def test_carries_the_bbox(self):
        url = airbnb_api.search_url(BOX)
        for part in ("ne_lat=37.52", "ne_lng=127.05", "sw_lat=37.49", "sw_lng=127.02"):
            self.assertIn(part, url)

    def test_single_segment_path_so_robots_disallow_does_not_match(self):
        """robots.txt 는 `/s/*/*` 를 막는다. `/s/homes` 는 세그먼트가 하나여야 한다."""
        url = airbnb_api.search_url(BOX)
        path = url.split("?", 1)[0]
        self.assertEqual(path, "https://www.airbnb.co.kr/s/homes")

    def test_cursor_is_url_encoded(self):
        url = airbnb_api.search_url(BOX, cursor="eyJhIjoxfQ==")
        self.assertIn("cursor=eyJhIjoxfQ%3D%3D", url)

    def test_no_cursor_param_without_cursor(self):
        self.assertNotIn("cursor=", airbnb_api.search_url(BOX))


class ParseCoordinates(unittest.TestCase):
    def test_reads_every_listing_once(self):
        found = airbnb_api.parse_coordinates(html())
        self.assertEqual(len(found), 20)

    def test_coordinates_are_floats_in_the_box(self):
        for lat, lng in airbnb_api.parse_coordinates(html()):
            self.assertIsInstance(lat, float)
            self.assertIsInstance(lng, float)
            self.assertTrue(37.49 <= lat <= 37.52, lat)
            self.assertTrue(127.02 <= lng <= 127.05, lng)

    def test_deduplicates_repeated_listings(self):
        """좌표는 한 응답 안에 두 번씩 실려 온다. 같은 숙소를 두 번 세면 안 된다."""
        doubled = html() + html()
        self.assertEqual(len(airbnb_api.parse_coordinates(doubled)), 20)

    def test_empty_page_yields_nothing_without_raising(self):
        self.assertEqual(airbnb_api.parse_coordinates(html("airbnb_search_empty.html.gz")), [])

    def test_garbage_yields_nothing_without_raising(self):
        self.assertEqual(airbnb_api.parse_coordinates(""), [])
        self.assertEqual(airbnb_api.parse_coordinates("<html>차단되었습니다</html>"), [])


class ParseTotal(unittest.TestCase):
    def test_reads_the_headline_count(self):
        self.assertEqual(airbnb_api.parse_total(html()), 712)

    def test_missing_count_is_none_not_zero(self):
        """0 으로 바꾸면 '숙소가 없다' 는 없는 사실을 지어내 분할을 멈춘다."""
        self.assertIsNone(airbnb_api.parse_total(html("airbnb_search_empty.html.gz")))
        self.assertIsNone(airbnb_api.parse_total(""))

    def test_strips_thousands_separator(self):
        self.assertEqual(airbnb_api.parse_total("<span>숙소 1,000개</span>"), 1000)

    def test_flags_the_capped_count(self):
        """총건수는 1,000 에서 잘린다. 집계값으로 쓰면 안 된다는 걸 알려야 한다."""
        self.assertTrue(airbnb_api.is_capped(1000))
        self.assertFalse(airbnb_api.is_capped(999))
        self.assertFalse(airbnb_api.is_capped(None))


class ParseCursors(unittest.TestCase):
    def test_reads_every_page_cursor(self):
        self.assertEqual(len(airbnb_api.parse_cursors(html())), 15)

    def test_cursors_are_ordered_by_offset(self):
        cursors = airbnb_api.parse_cursors(html())
        self.assertEqual(airbnb_api.cursor_offset(cursors[0]), 0)
        self.assertEqual(airbnb_api.cursor_offset(cursors[1]), 18)
        self.assertEqual(airbnb_api.cursor_offset(cursors[2]), 36)

    def test_missing_cursors_is_empty_list(self):
        self.assertEqual(airbnb_api.parse_cursors(""), [])


class SplitBox(unittest.TestCase):
    def test_makes_four_quadrants(self):
        self.assertEqual(len(airbnb_api.split(BOX)), 4)

    def test_quadrants_cover_the_parent_exactly(self):
        quads = airbnb_api.split(BOX)
        self.assertEqual(min(q.sw_lat for q in quads), BOX.sw_lat)
        self.assertEqual(max(q.ne_lat for q in quads), BOX.ne_lat)
        self.assertEqual(min(q.sw_lng for q in quads), BOX.sw_lng)
        self.assertEqual(max(q.ne_lng for q in quads), BOX.ne_lng)

    def test_quadrants_have_a_quarter_of_the_area_each(self):
        for q in airbnb_api.split(BOX):
            self.assertAlmostEqual(q.ne_lat - q.sw_lat, (BOX.ne_lat - BOX.sw_lat) / 2)
            self.assertAlmostEqual(q.ne_lng - q.sw_lng, (BOX.ne_lng - BOX.sw_lng) / 2)

    def test_smallest_side_decides_whether_to_split_further(self):
        tiny = airbnb_api.Box(ne_lat=37.5010, ne_lng=127.0300, sw_lat=37.5000, sw_lng=127.0200)
        self.assertTrue(airbnb_api.too_small(tiny, 0.004))
        self.assertFalse(airbnb_api.too_small(BOX, 0.004))


class ContainsPoint(unittest.TestCase):
    def test_keeps_points_inside(self):
        self.assertTrue(airbnb_api.contains(BOX, 37.50, 127.03))

    def test_drops_points_outside(self):
        self.assertFalse(airbnb_api.contains(BOX, 37.60, 127.03))
        self.assertFalse(airbnb_api.contains(BOX, 37.50, 127.10))

    def test_boundary_counts_as_inside(self):
        self.assertTrue(airbnb_api.contains(BOX, BOX.sw_lat, BOX.sw_lng))
        self.assertTrue(airbnb_api.contains(BOX, BOX.ne_lat, BOX.ne_lng))


class Grid(unittest.TestCase):
    def test_snaps_to_the_cell_origin(self):
        self.assertEqual(airbnb_api.snap(37.4973, 127.0374, 0.005), (37.495, 127.035))

    def test_points_in_one_cell_share_a_key(self):
        a = airbnb_api.snap(37.4951, 127.0351, 0.005)
        b = airbnb_api.snap(37.4999, 127.0399, 0.005)
        self.assertEqual(a, b)

    def test_next_cell_differs(self):
        a = airbnb_api.snap(37.4999, 127.0351, 0.005)
        b = airbnb_api.snap(37.5001, 127.0351, 0.005)
        self.assertNotEqual(a, b)

    def test_cell_origin_snaps_to_itself(self):
        """경계값이 위 칸으로 튀면 격자 경계마다 한 줄이 비어 보인다."""
        self.assertEqual(airbnb_api.snap(37.495, 127.035, 0.005), (37.495, 127.035))



class Decide(unittest.TestCase):
    """bbox 하나를 보고 쪼갤지 거둘지 정하는 판단. 수집기의 심장이다."""

    BIG = BOX                                                    # 0.03도
    TINY = airbnb_api.Box(37.5010, 127.0210, 37.5000, 127.0200)  # 0.001도

    def decide(self, total, found, box):
        return airbnb_api.decide(total, found, box, split_at=180, min_deg=0.004)

    def test_collects_when_the_box_fits_under_the_cap(self):
        self.assertEqual(self.decide(150, 150, self.BIG), "collect")

    def test_splits_when_the_box_holds_more_than_the_cap(self):
        self.assertEqual(self.decide(712, 20, self.BIG), "split")

    def test_splits_when_the_count_is_capped(self):
        """1,000 은 '1,000개 이상' 이다. 그대로 거두면 조용히 빠뜨린다."""
        self.assertEqual(self.decide(1000, 20, self.BIG), "split")

    def test_collects_once_the_box_is_too_small_to_split(self):
        """명동·홍대는 400m 로 쪼개도 상한을 넘는다. 무한 분할을 막는다."""
        self.assertEqual(self.decide(1000, 20, self.TINY), "collect")

    def test_empty_box_terminates_instead_of_splitting(self):
        """실측: 바다·산은 총건수도 좌표도 없이 온다. 쪼개면 4^n 으로 터진다."""
        self.assertEqual(self.decide(None, 0, self.BIG), "collect")

    def test_missing_count_but_listings_present_splits(self):
        """좌표는 왔는데 건수만 못 읽었다 — 화면이 바뀐 것이다. 쪼개서 안전하게 훑는다."""
        self.assertEqual(self.decide(None, 20, self.BIG), "split")

    def test_zero_count_is_collected_not_split(self):
        self.assertEqual(self.decide(0, 0, self.BIG), "collect")


class IsEmpty(unittest.TestCase):
    """빈 지역과 파싱 고장은 응답이 똑같이 생겼다. 경계를 한 곳에 모아 둔다."""

    def test_no_count_and_no_coordinates_is_empty(self):
        self.assertTrue(airbnb_api.looks_empty(None, 0))

    def test_coordinates_without_a_count_is_not_empty(self):
        self.assertFalse(airbnb_api.looks_empty(None, 20))

    def test_a_count_of_zero_is_empty(self):
        self.assertTrue(airbnb_api.looks_empty(0, 0))

    def test_a_real_count_is_not_empty(self):
        self.assertFalse(airbnb_api.looks_empty(712, 20))


class UnderCollected(unittest.TestCase):
    """상한에 걸려 실제보다 적게 잡힌 칸은 화면에 그렇게 표시해야 한다."""

    def test_flags_a_tiny_box_that_still_overflowed(self):
        self.assertTrue(airbnb_api.under_collected(1000, 20, 180))

    def test_does_not_flag_a_box_within_the_cap(self):
        self.assertFalse(airbnb_api.under_collected(150, 150, 180))

    def test_empty_box_is_not_under_collected(self):
        """빈 바다를 '덜 걷혔다' 고 표시하면 지도가 온통 경고가 된다."""
        self.assertFalse(airbnb_api.under_collected(None, 0, 180))

    def test_listings_without_a_count_is_under_collected(self):
        self.assertTrue(airbnb_api.under_collected(None, 20, 180))


class SnapshotComplete(unittest.TestCase):
    """확정 스냅샷이 있는가 — 화면 각주가 '전국을 다 훑었다' 고 말할 근거."""

    def test_finished_pass_is_complete(self):
        self.assertTrue(airbnb_api.snapshot_complete(
            {"points": {"a": 1}, "pending": {}, "frontier": []}))

    def test_first_pass_in_progress_is_not_complete(self):
        self.assertFalse(airbnb_api.snapshot_complete(
            {"points": {}, "pending": {"a": 1}, "frontier": [[1, 1, 0, 0]]}))

    def test_a_finished_snapshot_stays_complete_while_the_next_pass_runs(self):
        self.assertTrue(airbnb_api.snapshot_complete(
            {"points": {"a": 1}, "pending": {"b": 1}, "frontier": [[1, 1, 0, 0]]}))

    def test_an_old_unfinished_state_is_not_complete(self):
        """`pending` 이 생기기 전 파일. frontier 가 남았으면 돌던 중이었다 —
        여기서 True 를 내면 화면이 완주했다고 거짓말한다."""
        self.assertFalse(airbnb_api.snapshot_complete(
            {"points": {"a": 1}, "frontier": [[1, 1, 0, 0]]}))

    def test_an_old_finished_state_is_complete(self):
        self.assertTrue(airbnb_api.snapshot_complete(
            {"points": {"a": 1}, "frontier": []}))

    def test_nothing_collected_is_not_complete(self):
        self.assertFalse(airbnb_api.snapshot_complete({"points": {}, "frontier": []}))


class NeedsEmptyConfirmation(unittest.TestCase):
    """큰 bbox 가 '비었다' 고 나오면 한 번 더 묻는다.

    빈 응답과 고장난 응답은 생김새가 같다. 작은 bbox 하나를 잘못 가지치면
    동네 하나가 빠지지만, 큰 bbox 를 잘못 가지치면 **도 하나가 통째로 지도에서
    사라진다.** 큰 것만 확인해 비용을 몇 번으로 묶는다.
    """

    BIG = airbnb_api.Box(38.0, 130.0, 35.0, 127.0)      # 3도
    SMALL = airbnb_api.Box(37.52, 127.05, 37.49, 127.02)  # 0.03도

    def test_large_empty_box_is_confirmed(self):
        self.assertTrue(airbnb_api.needs_empty_confirmation(self.BIG, 1.0))

    def test_small_empty_box_is_taken_at_face_value(self):
        self.assertFalse(airbnb_api.needs_empty_confirmation(self.SMALL, 1.0))

    def test_boundary_size_is_not_confirmed(self):
        exact = airbnb_api.Box(38.0, 128.0, 37.0, 127.0)  # 정확히 1도
        self.assertFalse(airbnb_api.needs_empty_confirmation(exact, 1.0))

    def test_the_shorter_side_decides(self):
        """긴 변만 보면 가늘고 긴 bbox 를 큰 것으로 잘못 센다."""
        thin = airbnb_api.Box(38.0, 130.0, 37.9, 127.0)
        self.assertFalse(airbnb_api.needs_empty_confirmation(thin, 1.0))


class SeedBoxes(unittest.TestCase):
    """쿼드트리를 어디서 시작하나.

    전국을 사각형 하나로 덮으면 규슈가 통째로 들어온다 — 실측으로 후쿠오카
    숙소 1,694건을 받아다 버렸다. 예산은 그만큼 사라진다. 시도별 경계상자로
    시작하면 바다와 일본이 대부분 빠진다.
    """

    GEO = {"features": [
        {"properties": {"sgg": "11110"},
         "geometry": {"type": "MultiPolygon",
                      "coordinates": [[[[126.9, 37.5], [127.0, 37.5], [127.0, 37.6]]]]}},
        {"properties": {"sgg": "11140"},
         "geometry": {"type": "MultiPolygon",
                      "coordinates": [[[[126.8, 37.4], [126.95, 37.4], [126.95, 37.55]]]]}},
        {"properties": {"sgg": "50110"},
         "geometry": {"type": "MultiPolygon",
                      "coordinates": [[[[126.2, 33.4], [126.6, 33.4], [126.6, 33.6]]]]}},
    ]}

    def test_one_box_per_sido(self):
        boxes = airbnb_api.seed_boxes(self.GEO, pad=0.0)
        self.assertEqual(len(boxes), 2)

    def test_box_covers_every_sgg_of_that_sido(self):
        boxes = {round(b.sw_lng, 4): b for b in airbnb_api.seed_boxes(self.GEO, pad=0.0)}
        seoul = boxes[126.8]
        self.assertAlmostEqual(seoul.sw_lat, 37.4)
        self.assertAlmostEqual(seoul.ne_lat, 37.6)
        self.assertAlmostEqual(seoul.ne_lng, 127.0)

    def test_pad_widens_the_box(self):
        """경계에 딱 붙은 숙소가 빠지지 않게 조금 넓힌다."""
        plain = airbnb_api.seed_boxes(self.GEO, pad=0.0)
        padded = airbnb_api.seed_boxes(self.GEO, pad=0.05)
        for a, b in zip(sorted(plain), sorted(padded)):
            self.assertAlmostEqual(b.ne_lat - a.ne_lat, 0.05)
            self.assertAlmostEqual(a.sw_lng - b.sw_lng, 0.05)

    def test_no_features_yields_nothing(self):
        self.assertEqual(airbnb_api.seed_boxes({"features": []}), [])

    def test_result_is_ordered_so_runs_are_reproducible(self):
        self.assertEqual(airbnb_api.seed_boxes(self.GEO),
                         airbnb_api.seed_boxes(self.GEO))
