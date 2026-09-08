"""집계 산출물 신선도 판정.

조용한 붕괴 — 크론이 멈추거나 수집이 깨져도 어제 파일이 그대로 남아 오늘
값인 척하는 고장 — 을 잡는 판정이다. 여기가 틀리면 경보가 아예 안 울린다.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import freshness_api as f  # noqa: E402

TODAY = date(2026, 8, 12)


class DaysSinceTest(unittest.TestCase):
    def test_두_형식을_다_읽는다(self) -> None:
        self.assertEqual(f.days_since("2026-08-10", TODAY), 2)
        self.assertEqual(f.days_since("20260810", TODAY), 2)

    def test_못_읽으면_None(self) -> None:
        for bad in ("", "어제", "2026/08/10", None):
            self.assertIsNone(f.days_since(bad, TODAY))


class EnglishDateTest(unittest.TestCase):
    def test_로케일에_안_기댄다(self) -> None:
        """포켓몬 환율이 '11 Aug 2026' 으로 온다. %b 는 크론 LC_TIME 에 따라 깨진다."""
        self.assertEqual(f.days_since_english("11 Aug 2026", TODAY), 1)
        self.assertEqual(f.days_since_english("1 January 2026", TODAY), 223)

    def test_못_읽으면_None(self) -> None:
        for bad in ("", "11 Xxx 2026", "2026-08-11", "11 Aug"):
            self.assertIsNone(f.days_since_english(bad, TODAY))


class MonthsBehindTest(unittest.TestCase):
    def test_해를_넘겨도_센다(self) -> None:
        self.assertEqual(f.months_behind("2026-08", TODAY), 0)
        self.assertEqual(f.months_behind("2026-07", TODAY), 1)
        self.assertEqual(f.months_behind("2025-08", TODAY), 12)

    def test_이상한_값은_None(self) -> None:
        for bad in ("", "2026-13", "202608", "지난달", None):
            self.assertIsNone(f.months_behind(bad, TODAY))


class StaleTest(unittest.TestCase):
    def test_주말을_건너도_신선하다(self) -> None:
        """크론이 평일에만 도는 경우가 있다 — 금요일 자료가 월요일에 울리면 안 된다."""
        self.assertFalse(f.is_stale("2026-08-09", 3, TODAY))

    def test_한계를_넘기면_낡았다(self) -> None:
        self.assertTrue(f.is_stale("2026-08-08", 3, TODAY))

    def test_날짜를_못_읽으면_낡은_것으로_본다(self) -> None:
        """빠진 값을 '최신' 으로 처리하면 필드 이름이 바뀌는 순간 검사기가 무력해진다."""
        self.assertTrue(f.is_stale("", 3, TODAY))
        self.assertTrue(f.is_stale("어제", 3, TODAY))

    def test_달_단위도_같은_원칙(self) -> None:
        self.assertFalse(f.is_month_stale("2026-07", 1, TODAY))
        self.assertTrue(f.is_month_stale("2026-06", 1, TODAY))
        self.assertTrue(f.is_month_stale("", 1, TODAY))


class LimitsTest(unittest.TestCase):
    def test_대시보드마다_한계가_있다(self) -> None:
        for name in ("trades", "schools", "redev", "pokemon", "supply"):
            self.assertIn(name, f.BUILD_LIMITS)

    def test_매월_도는_학군은_더_넉넉하다(self) -> None:
        self.assertGreater(f.BUILD_LIMITS["schools"], f.BUILD_LIMITS["trades"])

    def test_월_2회_도는_착공은_더_넉넉하다(self) -> None:
        self.assertGreater(f.BUILD_LIMITS["supply"], f.BUILD_LIMITS["trades"])

    def test_착공은_발표_시차만큼_뒤처져도_된다(self) -> None:
        """통계누리 발표 시차가 약 1.5개월이라 실거래보다 한 달 더 준다."""
        self.assertGreater(f.MONTH_LAG_LIMITS["supply"], f.MONTH_LAG_LIMITS["trades"])
