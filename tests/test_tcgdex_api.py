"""TCGdex 응답 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"

import tcgdex_api  # noqa: E402


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestParseSetList(unittest.TestCase):
    def test_extracts_id_name_and_count(self) -> None:
        rows = tcgdex_api.parse_set_list(_fixture("tcgdex_sets.json"))
        self.assertTrue(rows)
        base = [r for r in rows if r["set_id"] == "base1"]
        self.assertEqual(len(base), 1)
        self.assertEqual(base[0]["name"], "Base Set")
        self.assertEqual(base[0]["card_count"], 102)

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(tcgdex_api.ApiError):
            tcgdex_api.parse_set_list({"error": "nope"})


class TestParseSetDetail(unittest.TestCase):
    def test_extracts_release_date_and_card_ids(self) -> None:
        d = tcgdex_api.parse_set_detail(_fixture("tcgdex_set_base1.json"))
        self.assertEqual(d["set_id"], "base1")
        self.assertEqual(d["release_date"], "1999-01-09")
        self.assertIn("base1-4", d["card_ids"])
        self.assertEqual(len(d["card_ids"]), 102)

    def test_missing_cards_yields_empty_list(self) -> None:
        d = tcgdex_api.parse_set_detail({"id": "x", "name": "X", "releaseDate": "2020-01-01"})
        self.assertEqual(d["card_ids"], [])

    def test_missing_id_raises(self) -> None:
        with self.assertRaises(tcgdex_api.ApiError):
            tcgdex_api.parse_set_detail({"name": "X"})


class TestEraOf(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(tcgdex_api.era_of("1999-01-09"), "빈티지")
        self.assertEqual(tcgdex_api.era_of("2003-12-31"), "빈티지")
        self.assertEqual(tcgdex_api.era_of("2004-01-01"), "클래식")
        self.assertEqual(tcgdex_api.era_of("2010-12-31"), "클래식")
        self.assertEqual(tcgdex_api.era_of("2011-01-01"), "모던")
        self.assertEqual(tcgdex_api.era_of("2019-12-31"), "모던")
        self.assertEqual(tcgdex_api.era_of("2020-01-01"), "최신")

    def test_blank_is_none(self) -> None:
        self.assertIsNone(tcgdex_api.era_of(""))
        self.assertIsNone(tcgdex_api.era_of("abcd-01-01"))


class TestPickVariant(unittest.TestCase):
    def test_prefers_holofoil(self) -> None:
        tp = {"normal": {"marketPrice": 1.0}, "holofoil": {"marketPrice": 9.0}}
        name, block = tcgdex_api.pick_variant(tp)
        self.assertEqual(name, "holofoil")
        self.assertEqual(block["marketPrice"], 9.0)

    def test_falls_back_to_normal(self) -> None:
        name, _ = tcgdex_api.pick_variant({"normal": {"marketPrice": 1.0}})
        self.assertEqual(name, "normal")

    def test_skips_variant_without_market_price(self) -> None:
        tp = {"holofoil": {"lowPrice": 3.0}, "normal": {"marketPrice": 1.0}}
        name, _ = tcgdex_api.pick_variant(tp)
        self.assertEqual(name, "normal")

    def test_none_when_no_usable_variant(self) -> None:
        self.assertIsNone(tcgdex_api.pick_variant({"unit": "USD", "updated": "x"}))
        self.assertIsNone(tcgdex_api.pick_variant(None))


class TestShortImage(unittest.TestCase):
    def test_strips_cdn_prefix(self) -> None:
        self.assertEqual(
            tcgdex_api.short_image("https://assets.tcgdex.net/en/base/base1/4"),
            "base/base1/4")

    def test_blank_stays_blank(self) -> None:
        self.assertEqual(tcgdex_api.short_image(None), "")
        self.assertEqual(tcgdex_api.short_image(""), "")

    def test_unexpected_host_is_left_alone(self) -> None:
        self.assertEqual(tcgdex_api.short_image("https://other/x"), "https://other/x")


class TestFirstDexId(unittest.TestCase):
    def test_takes_first_of_list(self) -> None:
        self.assertEqual(tcgdex_api.first_dex_id([6]), "6")
        self.assertEqual(tcgdex_api.first_dex_id([25, 26]), "25")

    def test_missing_is_blank(self) -> None:
        self.assertEqual(tcgdex_api.first_dex_id(None), "")
        self.assertEqual(tcgdex_api.first_dex_id([]), "")


class TestParseCardFull(unittest.TestCase):
    def test_priced_card_has_meta_and_prices(self) -> None:
        row = tcgdex_api.parse_card_full(_fixture("tcgdex_card_priced.json"), "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(set(row), set(tcgdex_api.CARD_COLUMNS))
        self.assertEqual(row["card_id"], "base1-4")
        self.assertEqual(row["set_id"], "base1")
        self.assertEqual(row["name_en"], "Charizard")
        self.assertEqual(row["dex_id"], "6")
        self.assertEqual(row["image"], "base/base1/4")
        self.assertTrue(row["tp_product_id"], "TCGplayer productId 를 걷어야 한다")
        self.assertGreater(row["tp_market"], 0)
        self.assertGreater(row["tp_high"], 0)
        self.assertEqual(row["updated"], "2026-08-07")

    def test_observed_max_starts_at_current_price(self) -> None:
        row = tcgdex_api.parse_card_full(_fixture("tcgdex_card_priced.json"), "2026-08-07")
        self.assertEqual(row["obs_max"], row["tp_market"])
        self.assertEqual(row["obs_max_date"], "2026-08-07")

    def test_missing_product_id_is_blank(self) -> None:
        payload = {"id": "x-1", "pricing": {"cardmarket": {"avg": 5.0}}}
        self.assertEqual(tcgdex_api.parse_card_full(payload, "2026-08-07")["tp_product_id"], "")

    def test_energy_card_has_no_dex_id(self) -> None:
        row = tcgdex_api.parse_card_full(_fixture("tcgdex_card_energy.json"), "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(row["dex_id"], "")
        self.assertEqual(row["category"], "Energy")

    def test_unpriced_card_yields_none(self) -> None:
        self.assertIsNone(
            tcgdex_api.parse_card_full(_fixture("tcgdex_card_unpriced.json"), "2026-08-07"))

    def test_missing_id_yields_none(self) -> None:
        self.assertIsNone(tcgdex_api.parse_card_full({"name": "X"}, "2026-08-07"))

    def test_cardmarket_only_still_yields_row(self) -> None:
        payload = {"id": "x-1", "name": "X", "pricing": {"cardmarket": {"avg": 5.0}}}
        row = tcgdex_api.parse_card_full(payload, "2026-08-07")
        self.assertIsNotNone(row)
        self.assertIsNone(row["tp_market"])
        self.assertEqual(row["cm_avg"], 5.0)
        self.assertEqual(row["obs_max"], 5.0)

    def test_zero_price_is_not_a_price(self) -> None:
        payload = {"id": "x-1", "pricing": {"cardmarket": {"avg": 0}}}
        self.assertIsNone(tcgdex_api.parse_card_full(payload, "2026-08-07"))


class TestMergeCard(unittest.TestCase):
    def _row(self, market, obs, obs_date, updated="2026-08-08"):
        row = {c: None for c in tcgdex_api.CARD_COLUMNS}
        row.update({"card_id": "a-1", "tp_market": market, "obs_max": obs,
                    "obs_max_date": obs_date, "updated": updated})
        return row

    def test_no_previous_row_is_kept_as_is(self) -> None:
        new = self._row(10.0, 10.0, "2026-08-08")
        self.assertEqual(tcgdex_api.merge_card(None, new), new)

    def test_higher_new_price_raises_the_record(self) -> None:
        old = self._row(10.0, 10.0, "2026-08-07")
        new = self._row(15.0, 15.0, "2026-08-08")
        m = tcgdex_api.merge_card(old, new)
        self.assertEqual(m["obs_max"], 15.0)
        self.assertEqual(m["obs_max_date"], "2026-08-08")

    def test_lower_new_price_keeps_the_old_record(self) -> None:
        old = self._row(20.0, 20.0, "2026-08-07")
        new = self._row(8.0, 8.0, "2026-08-08")
        m = tcgdex_api.merge_card(old, new)
        self.assertEqual(m["obs_max"], 20.0)
        self.assertEqual(m["obs_max_date"], "2026-08-07")
        self.assertEqual(m["tp_market"], 8.0, "현재가는 새 값이어야 한다")

    def test_missing_new_price_keeps_the_old_record(self) -> None:
        old = self._row(20.0, 20.0, "2026-08-07")
        new = self._row(None, None, "")
        m = tcgdex_api.merge_card(old, new)
        self.assertEqual(m["obs_max"], 20.0)
        self.assertEqual(m["obs_max_date"], "2026-08-07")

    def test_metadata_comes_from_the_new_row(self) -> None:
        old = self._row(10.0, 10.0, "2026-08-07")
        old["name_en"] = "Old Name"
        new = self._row(10.0, 10.0, "2026-08-08")
        new["name_en"] = "New Name"
        self.assertEqual(tcgdex_api.merge_card(old, new)["name_en"], "New Name")


if __name__ == "__main__":
    unittest.main()
