"""수집기의 순수한 부분 — 한 바퀴(pass) 관리와 예산 소진 처리.

    python3 -m unittest tests.test_collect_airbnb -v

전국을 한 번 훑는 데 며칠이 걸린다. 그 사이 화면에 무엇을 보여줄지, 다 훑고
나면 무엇을 버릴지가 여기 있다. 틀리면 지도가 조용히 반쯤 비거나, 사라진
숙소가 영원히 남는다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airbnb_api  # noqa: E402
import collect_airbnb as C  # noqa: E402


ONE = [C.api.Box(1.0, 2.0, 0.0, 1.0)]


class NewState(unittest.TestCase):
    def test_starts_with_the_given_boxes_and_nothing_collected(self):
        state = C.new_state(ONE)
        self.assertEqual(len(state["frontier"]), 1)
        self.assertEqual(state["points"], {})
        self.assertEqual(state["pending"], {})

    def test_default_seed_is_one_box_per_sido_not_one_for_the_country(self):
        """전국 사각형 하나로 시작하면 규슈까지 훑는다 — 실측으로 후쿠오카
        숙소 1,694건을 받아다 버렸다."""
        boxes = C.seed()
        self.assertGreater(len(boxes), 10)
        self.assertNotIn(list(C.KOREA), [list(b) for b in boxes])

    def test_every_seed_box_sits_over_korea(self):
        for box in C.seed():
            self.assertTrue(32.0 < box.sw_lat < box.ne_lat < 39.5, box)
            self.assertTrue(124.0 < box.sw_lng < box.ne_lng < 132.5, box)


class BeginPass(unittest.TestCase):
    """frontier 가 비면 새 바퀴다. 지난 바퀴 결과는 화면용으로 남겨 둔다."""

    def test_empty_frontier_starts_a_new_pass(self):
        state = C.new_state(ONE)
        state["frontier"] = []
        state["points"] = {"37.5,127.0": 1}
        C.begin_pass(state, ONE)
        self.assertEqual(len(state["frontier"]), 1)
        self.assertEqual(state["pending"], {})

    def test_new_pass_keeps_the_published_snapshot(self):
        """새 바퀴를 도는 며칠 동안 지도가 비면 안 된다."""
        state = C.new_state(ONE)
        state["frontier"] = []
        state["points"] = {"37.5,127.0": 1, "33.5,126.5": 1}
        C.begin_pass(state, ONE)
        self.assertEqual(len(state["points"]), 2)

    def test_new_pass_throws_away_the_previous_pending(self):
        """중간에 끊긴 바퀴의 찌꺼기가 다음 바퀴에 섞이면 안 된다."""
        state = C.new_state(ONE)
        state["frontier"] = []
        state["pending"] = {"37.5,127.0": 1}
        C.begin_pass(state, ONE)
        self.assertEqual(state["pending"], {})

    def test_unfinished_pass_is_left_alone(self):
        state = C.new_state(ONE)
        state["frontier"] = [[1.0, 2.0, 0.0, 1.0], [3.0, 4.0, 2.0, 3.0]]
        state["pending"] = {"37.5,127.0": 1}
        C.begin_pass(state, ONE)
        self.assertEqual(len(state["frontier"]), 2)
        self.assertEqual(len(state["pending"]), 1)

    def test_an_old_unfinished_pass_becomes_pending_not_a_snapshot(self):
        """`pending` 이 생기기 전에 쓰인 상태 파일. 아직 돌던 중이었으므로
        그 좌표는 확정 스냅샷이 아니다 — 확정으로 두면 화면이 '완주' 라고
        거짓말한다."""
        old = {"frontier": [[1.0, 2.0, 0.0, 1.0]], "points": {"37.5,127.0": 1},
               "weak": [[1.0, 1.0, 0.0, 0.0]], "stats": {"calls": 0}, "complete": False}
        C.begin_pass(old, ONE)
        self.assertEqual(old["points"], {})
        self.assertEqual(old["pending"], {"37.5,127.0": 1})
        self.assertEqual(old["weak_pending"], [[1.0, 1.0, 0.0, 0.0]])

    def test_an_old_finished_pass_stays_a_snapshot(self):
        old = {"frontier": [], "points": {"37.5,127.0": 1}, "weak": [],
               "stats": {"calls": 0}, "complete": True}
        C.begin_pass(old, ONE)
        self.assertEqual(old["points"], {"37.5,127.0": 1})
        self.assertEqual(old["pending"], {})
        self.assertEqual(len(old["frontier"]), 1)


class FinishPass(unittest.TestCase):
    """다 훑으면 이번 바퀴 결과가 지난 바퀴를 대신한다."""

    def test_pending_replaces_points(self):
        state = C.new_state(ONE)
        state["frontier"] = []
        state["points"] = {"old": 1}
        state["pending"] = {"new": 1}
        state["weak_pending"] = [[1.0, 1.0, 0.0, 0.0]]
        C.finish_pass(state)
        self.assertEqual(state["points"], {"new": 1})
        self.assertEqual(state["weak"], [[1.0, 1.0, 0.0, 0.0]])
        self.assertTrue(state["complete"])

    def test_disappeared_listings_do_not_survive(self):
        """지난 바퀴에만 있던 좌표가 남으면 숙소가 영원히 안 사라진다."""
        state = C.new_state(ONE)
        state["frontier"] = []
        state["points"] = {"a": 1, "b": 1}
        state["pending"] = {"a": 1}
        C.finish_pass(state)
        self.assertEqual(set(state["points"]), {"a"})

    def test_does_nothing_while_boxes_remain(self):
        state = C.new_state(ONE)
        state["frontier"] = [[1.0, 2.0, 0.0, 1.0]]
        state["points"] = {"old": 1}
        state["pending"] = {"new": 1}
        C.finish_pass(state)
        self.assertEqual(state["points"], {"old": 1})
        self.assertFalse(state["complete"])

    def test_an_empty_pass_does_not_wipe_the_map(self):
        """수집이 통째로 깨진 바퀴가 빈 결과로 지도를 지워서는 안 된다."""
        state = C.new_state(ONE)
        state["frontier"] = []
        state["points"] = {"a": 1, "b": 1}
        state["pending"] = {}
        C.finish_pass(state)
        self.assertEqual(set(state["points"]), {"a", "b"})


class PublishedPoints(unittest.TestCase):
    """화면에 내보낼 좌표 — 첫 바퀴 중에는 모으는 중인 것이라도 보여준다."""

    def test_uses_the_finished_snapshot(self):
        state = {"points": {"a": 1}, "pending": {"b": 1}}
        self.assertEqual(airbnb_api.published_points(state), {"a": 1})

    def test_falls_back_to_pending_before_the_first_pass_finishes(self):
        state = {"points": {}, "pending": {"b": 1}}
        self.assertEqual(airbnb_api.published_points(state), {"b": 1})

    def test_empty_when_nothing_collected_yet(self):
        self.assertEqual(airbnb_api.published_points({"points": {}, "pending": {}}), {})

    def test_collector_and_builder_use_the_same_rule(self):
        self.assertIs(C.published_points, airbnb_api.published_points)


class Retry(unittest.TestCase):
    """실패한 bbox 를 어디에 되돌리나."""

    def test_failed_box_goes_to_the_back_of_the_queue(self):
        """스택 끝(다음에 꺼낼 자리)에 두면 영구적으로 깨진 bbox 하나가
        매 실행을 다섯 번 만에 중단시킨다."""
        frontier = [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0]]
        C.requeue(frontier, C.api.Box(3.0, 3.0, 2.0, 2.0))
        self.assertEqual(frontier[0], [3.0, 3.0, 2.0, 2.0])
        self.assertEqual(frontier[-1], [2.0, 2.0, 1.0, 1.0])
