"""PSA 10 통합 표 집계 검증.

    python3 -m unittest tests.test_build_unified -v

이 표의 규칙 두 가지를 지킨다.
  · PSA 10 만 담는다 (raw 를 섞으면 가격순 정렬이 거짓말이 된다)
  · 병합은 하드 키(품번)로만 한다 (이름으로 묶으면 다른 카드가 엉킨다)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_unified  # noqa: E402
import kream_api  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "kream_chart.json"
U = {name: i for i, name in enumerate(build_unified.COLUMNS)}


def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class BaseCode(unittest.TestCase):
    def test_language_suffix_stripped(self):
        self.assertEqual(kream_api.base_code("S7R-083-067_JP"), "S7R083067")
        self.assertEqual(kream_api.base_code("S7R083-067_KR"), "S7R083067")
        self.assertEqual(kream_api.base_code("BS4/102_EN"), "BS4102")

    def test_same_card_different_language_collides(self):
        """일어판과 한글판이 같은 열쇠로 떨어져야 한 줄로 묶인다."""
        self.assertEqual(kream_api.base_code("M2116-080_JP"),
                         kream_api.base_code("M2116080_KR"))

    def test_different_cards_do_not_collide(self):
        self.assertNotEqual(kream_api.base_code("S7R-083-067_JP"),
                            kream_api.base_code("S7R-076-067_JP"))

    def test_blank(self):
        self.assertEqual(kream_api.base_code(""), "")
        self.assertEqual(kream_api.base_code(None), "")


class StripLanguageSuffix(unittest.TestCase):
    def test_strips(self):
        self.assertEqual(
            kream_api.strip_language_suffix("레쿠쟈 VMAX HR 창공스트림 (일어판)"),
            "레쿠쟈 VMAX HR 창공스트림")

    def test_leaves_other_parens(self):
        self.assertEqual(kream_api.strip_language_suffix("피카츄 (프로모)"),
                         "피카츄 (프로모)")


class MergeLanguages(unittest.TestCase):
    def setUp(self):
        data = payload()
        self.products = kream_api.parse_products(data, kream_api.history_dates(data))
        self.groups = kream_api.merge_languages(self.products)

    def test_never_loses_a_product(self):
        seen = sum(len(g["by_lang"]) for g in self.groups)
        self.assertEqual(seen, len(self.products))

    def test_group_has_no_language_suffix_in_name(self):
        for g in self.groups:
            self.assertNotIn("(일어판)", g["name_ko"])

    def test_ratio_needs_two_languages(self):
        for g in self.groups:
            if len(g["by_lang"]) < 2:
                self.assertIsNone(kream_api.language_ratio(g))

    def test_ratio_is_max_over_min(self):
        price_at = kream_api.PRODUCT_COLUMNS.index("price")
        fake = {"by_lang": {
            "일어판": [None] * len(kream_api.PRODUCT_COLUMNS),
            "한글판": [None] * len(kream_api.PRODUCT_COLUMNS),
        }}
        fake["by_lang"]["일어판"][price_at] = 12000000
        fake["by_lang"]["한글판"][price_at] = 1150000
        self.assertEqual(kream_api.language_ratio(fake), 10.4)


class Build(unittest.TestCase):
    def setUp(self):
        self.view = build_unified.build(payload(), [], [], {}, "2026-08-08")

    def test_columns_match(self):
        self.assertEqual(self.view["columns"], build_unified.COLUMNS)

    def test_every_row_has_every_column(self):
        for row in self.view["rows"]:
            self.assertEqual(len(row), len(build_unified.COLUMNS))

    def test_domestic_rows_are_krw(self):
        for row in self.view["rows"]:
            if row[U["market"]] == "국내":
                self.assertEqual(row[U["unit"]], "KRW")

    def test_every_row_has_at_least_one_price(self):
        """값이 없는 줄은 이 표에 있을 이유가 없다."""
        for row in self.view["rows"]:
            self.assertTrue(row[U["prices"]])

    def test_samples_travel_with_prices(self):
        for row in self.view["rows"]:
            self.assertEqual(set(row[U["samples"]]), set(row[U["prices"]]))

    def test_stats_add_up(self):
        stats = self.view["stats"]
        self.assertEqual(stats["total"], stats["global"] + stats["domestic"])
        self.assertEqual(stats["total"], len(self.view["rows"]))

    def test_json_serialisable(self):
        json.dumps(self.view, ensure_ascii=False)


class GlobalRows(unittest.TestCase):
    """글로벌 줄은 PSA 10 이 있는 카드만. raw 만 있는 카드는 안 들어온다."""

    def _card(self, cid="base1-4", dex="6"):
        return {"card_id": cid, "set_id": "base1", "local_id": "4",
                "name_en": "Charizard", "dex_id": dex}

    def test_skips_cards_without_psa10(self):
        rows = build_unified.global_rows(
            [self._card()], [{"card_id": "base1-4", "psa10": None}], {})
        self.assertEqual(rows, [])

    def test_includes_cards_with_psa10(self):
        rows = build_unified.global_rows(
            [self._card()], [{"card_id": "base1-4", "psa10": 14875.0,
                              "psa10_n": 1}], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][U["unit"]], "USD")
        self.assertEqual(rows[0][U["prices"]], {"영문판": 14875.0})

    def test_korean_name_is_appended_for_search(self):
        """국내 줄은 한글뿐이라, 한글이 없으면 '리자몽' 검색에서 글로벌이 빠진다."""
        rows = build_unified.global_rows(
            [self._card()], [{"card_id": "base1-4", "psa10": 1.0, "psa10_n": 1}],
            {"6": "리자몽"})
        self.assertIn("리자몽", rows[0][U["name"]])

    def test_unknown_card_is_skipped(self):
        rows = build_unified.global_rows(
            [], [{"card_id": "nope-1", "psa10": 1.0}], {})
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
