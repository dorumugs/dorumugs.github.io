"""KREAM 시세표 응답 파싱 검증.

    python3 -m unittest tests.test_kream_api -v

fixtures/kream_chart.json 은 실제 응답에서 표본을 뽑아 줄인 것이다. KREAM 은
공식 API 가 아니라 화면이 바뀌면 조용히 깨지므로 응답 모양을 고정해 둔다.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import kream_api  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "kream_chart.json"
C = {name: i for i, name in enumerate(kream_api.PRODUCT_COLUMNS)}
M = {name: i for i, name in enumerate(kream_api.MARKET_COLUMNS)}


def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ParsePct(unittest.TestCase):
    def test_scientific_notation(self):
        self.assertEqual(kream_api.parse_pct("-1.72E1"), -17.2)
        self.assertEqual(kream_api.parse_pct("2.4E0"), 2.4)
        self.assertEqual(kream_api.parse_pct("4.0E-1"), 0.4)

    def test_missing_is_none_not_zero(self):
        """'-' 를 0 으로 바꾸면 '안 움직였다' 는 없는 사실을 지어내게 된다."""
        self.assertIsNone(kream_api.parse_pct("-"))
        self.assertIsNone(kream_api.parse_pct(""))
        self.assertIsNone(kream_api.parse_pct(None))
        self.assertIsNotNone(kream_api.parse_pct("0E0"))

    def test_garbage(self):
        self.assertIsNone(kream_api.parse_pct("어제보다"))


class ParsePrice(unittest.TestCase):
    def test_rounds_to_won(self):
        self.assertEqual(kream_api.parse_price(154812.5), 154813)
        self.assertEqual(kream_api.parse_price(12000000), 12000000)

    def test_missing(self):
        self.assertIsNone(kream_api.parse_price(None))
        self.assertIsNone(kream_api.parse_price("-"))


class SplitLanguage(unittest.TestCase):
    def test_japanese(self):
        name, lang = kream_api.split_language(
            "Pokemon TCG Rayquaza VMAX HR Blue Sky Stream (Japanese Ver.)")
        self.assertEqual(name, "Pokemon TCG Rayquaza VMAX HR Blue Sky Stream")
        self.assertEqual(lang, "일어판")

    def test_korean_and_english(self):
        self.assertEqual(kream_api.split_language("x (Korean Ver.)")[1], "한글판")
        self.assertEqual(kream_api.split_language("x (English Ver.)")[1], "영문판")

    def test_no_suffix_keeps_name(self):
        name, lang = kream_api.split_language("Pokemon TCG Ditto's Time Capsule")
        self.assertEqual(name, "Pokemon TCG Ditto's Time Capsule")
        self.assertEqual(lang, "")


class ParseMarket(unittest.TestCase):
    def setUp(self):
        self.rows = kream_api.parse_market(payload())

    def test_sorted_ascending(self):
        dates = [r[M["date"]] for r in self.rows]
        self.assertEqual(dates, sorted(dates))
        self.assertGreater(len(dates), 1)

    def test_values(self):
        first = self.rows[0]
        self.assertEqual(first[M["date"]], "2026-07-08")
        self.assertIsInstance(first[M["count"]], int)
        self.assertGreater(first[M["median"]], 0)

    def test_empty_payload(self):
        self.assertEqual(kream_api.parse_market({}), [])


class ParseProducts(unittest.TestCase):
    def setUp(self):
        self.payload = payload()
        self.dates = kream_api.history_dates(self.payload)
        self.rows = kream_api.parse_products(self.payload, self.dates)

    def test_shared_date_axis(self):
        self.assertEqual(self.dates[0], "2026-07-08")
        for row in self.rows:
            self.assertEqual(len(row[C["history"]]), len(self.dates))

    def test_name_prefixes_stripped(self):
        for row in self.rows:
            self.assertFalse(row[C["name_ko"]].startswith("포켓몬 TCG"))
            self.assertFalse(row[C["name_en"]].startswith("Pokemon TCG"))
            self.assertNotIn("Ver.)", row[C["name_en"]])

    def test_sorted_by_price_desc(self):
        prices = [r[C["price"]] or 0 for r in self.rows]
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_image_prefix_stripped(self):
        for row in self.rows:
            self.assertFalse(row[C["image"]].startswith("http"))

    def test_missing_image_is_blank_not_broken(self):
        blanks = [r for r in self.rows if not r[C["image"]]]
        self.assertTrue(blanks, "이미지 없는 표본이 fixture 에 있어야 한다")

    def test_change_missing_stays_none(self):
        row = next(r for r in self.rows if r[C["code"]] == "S8A-P002-025")
        self.assertIsNone(row[C["change_30d"]])

    def test_change_parsed(self):
        row = next(r for r in self.rows if r[C["code"]] == "SV2A-168-165_JP")
        self.assertEqual(row[C["change_30d"]], -17.2)
        self.assertEqual(row[C["lang"]], "일어판")


class Summaries(unittest.TestCase):
    def setUp(self):
        data = payload()
        self.rows = kream_api.parse_products(data, kream_api.history_dates(data))

    def test_thin_count(self):
        self.assertEqual(kream_api.thin_count(self.rows),
                         sum(1 for r in self.rows if r[C["tx"]] <= 1))

    def test_language_counts_sum(self):
        counts = kream_api.language_counts(self.rows)
        self.assertEqual(sum(counts.values()), len(self.rows))


class Validity(unittest.TestCase):
    """빈 껍데기로 좋은 데이터를 덮어쓰지 않기 위한 관문."""

    def test_real_payload_needs_enough_products(self):
        # fixture 는 5종뿐이라 문턱을 넘지 못한다 — 그게 맞는 동작이다.
        self.assertFalse(kream_api.is_valid(payload()))

    def test_rejects_junk(self):
        self.assertFalse(kream_api.is_valid({}))
        self.assertFalse(kream_api.is_valid(None))
        self.assertFalse(kream_api.is_valid({"market": {"rows": []}}))

    def test_accepts_full_shape(self):
        data = payload()
        data["product"]["rows"] = data["product"]["rows"] * 25  # 125종
        self.assertTrue(kream_api.is_valid(data))


if __name__ == "__main__":
    unittest.main()
