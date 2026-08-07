"""감정 등급(PSA) 시세 파싱 검증.

    python3 -m unittest tests.test_ppt_api -v

fixtures/ppt_card_graded.json 은 실제 응답을 줄인 것이다 (base1-4 Charizard).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ppt_api  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "ppt_card_graded.json"


def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def card() -> dict:
    return ppt_api.cards_from_response(payload())[0]


class Extract(unittest.TestCase):
    def test_cards_from_list(self):
        self.assertEqual(len(ppt_api.cards_from_response(payload())), 1)

    def test_cards_from_single_object(self):
        """단일 조회는 data 가 객체로 온다. 배열로 통일해야 한다."""
        one = payload()["data"][0]
        self.assertEqual(ppt_api.cards_from_response({"data": one}), [one])

    def test_cards_from_junk(self):
        self.assertEqual(ppt_api.cards_from_response({}), [])
        self.assertEqual(ppt_api.cards_from_response(None), [])


class GradeBlock(unittest.TestCase):
    def test_known_grade(self):
        block = ppt_api.grade_block(card(), "psa10")
        self.assertTrue(block)
        self.assertIn("count", block)

    def test_missing_grade_is_empty(self):
        self.assertEqual(ppt_api.grade_block(card(), "psa1"), {})

    def test_card_without_ebay(self):
        self.assertEqual(ppt_api.grade_block({"name": "x"}, "psa10"), {})


class Price(unittest.TestCase):
    def test_prefers_smart_market_price(self):
        """중앙값만 쓰면 1년 전 한 건이 오늘 시세인 척한다."""
        block = {"medianPrice": 999, "smartMarketPrice": {"price": 500}}
        self.assertEqual(ppt_api.price_of(block), 500)

    def test_falls_back_to_median(self):
        self.assertEqual(ppt_api.price_of({"medianPrice": 999}), 999)

    def test_no_price(self):
        self.assertIsNone(ppt_api.price_of({}))

    def test_confidence_and_date(self):
        block = ppt_api.grade_block(card(), "psa10")
        self.assertIn(ppt_api.confidence_of(block), ("low", "medium", "high", ""))
        self.assertRegex(ppt_api.sale_date_of(block), r"^\d{4}-\d{2}-\d{2}$|^$")

    def test_count(self):
        self.assertGreaterEqual(ppt_api.count_of(ppt_api.grade_block(card(), "psa9")), 1)
        self.assertEqual(ppt_api.count_of({}), 0)


class ParseCard(unittest.TestCase):
    def setUp(self):
        self.row = ppt_api.parse_card(card(), "base1-4", "2026-08-08")

    def test_has_every_column(self):
        self.assertEqual(sorted(self.row), sorted(ppt_api.COLUMNS))

    def test_ids(self):
        self.assertEqual(self.row["card_id"], "base1-4")
        self.assertEqual(self.row["tcg_product_id"], "42382")

    def test_psa_values_present(self):
        self.assertIsNotNone(self.row["psa10"])
        self.assertIsNotNone(self.row["psa9"])
        self.assertTrue(ppt_api.has_any_grade(self.row))

    def test_sample_size_travels_with_price(self):
        """표본 크기 없이 값만 들고 다니면 화면이 정직할 수 없다."""
        self.assertIsInstance(self.row["psa10_n"], int)
        self.assertIsInstance(self.row["psa9_n"], int)

    def test_card_without_grades(self):
        row = ppt_api.parse_card({"tcgPlayerId": "1"}, "x-1", "2026-08-08")
        self.assertFalse(ppt_api.has_any_grade(row))
        self.assertIsNone(row["psa10"])


class Premium(unittest.TestCase):
    def test_multiple(self):
        self.assertEqual(ppt_api.premium({"psa10": 1000.0}, 100), 10.0)

    def test_missing_inputs(self):
        self.assertIsNone(ppt_api.premium({"psa10": None}, 100))
        self.assertIsNone(ppt_api.premium({"psa10": 100.0}, 0))
        self.assertIsNone(ppt_api.premium({"psa10": 100.0}, None))
        self.assertIsNone(ppt_api.premium({"psa10": 100.0}, "—"))


class Budget(unittest.TestCase):
    def test_two_credits_per_card(self):
        self.assertEqual(ppt_api.CREDITS_PER_CARD, 2)
        self.assertEqual(ppt_api.daily_budget(100), 50)

    def test_reserve_is_held_back(self):
        self.assertEqual(ppt_api.daily_budget(100, reserve=10), 45)

    def test_never_negative(self):
        self.assertEqual(ppt_api.daily_budget(2, reserve=10), 0)
        self.assertEqual(ppt_api.daily_budget(0), 0)


if __name__ == "__main__":
    unittest.main()
