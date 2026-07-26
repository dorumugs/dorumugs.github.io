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
