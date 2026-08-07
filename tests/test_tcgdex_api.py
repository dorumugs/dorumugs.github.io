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
        self.assertEqual(tcgdex_api.era_of("2026-08-07"), "최신")

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

    def test_none_input_is_safe(self) -> None:
        self.assertIsNone(tcgdex_api.pick_variant(None))


class TestParseCardPricing(unittest.TestCase):
    def test_priced_card_yields_full_row(self) -> None:
        row = tcgdex_api.parse_card_pricing(_fixture("tcgdex_card_priced.json"), "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(row["date"], "2026-08-07")
        self.assertEqual(row["card_id"], "base1-4")
        self.assertEqual(row["variant"], "holofoil")
        self.assertGreater(row["tp_market"], 0)
        self.assertGreater(row["cm_avg30"], 0)
        self.assertEqual(set(row), set(tcgdex_api.COLUMNS))

    def test_unpriced_card_yields_none(self) -> None:
        self.assertIsNone(
            tcgdex_api.parse_card_pricing(_fixture("tcgdex_card_unpriced.json"), "2026-08-07")
        )

    def test_cardmarket_only_still_yields_row(self) -> None:
        payload = {"id": "x-1", "name": "X", "pricing": {"cardmarket": {"avg": 5.0, "trend": 4.0}}}
        row = tcgdex_api.parse_card_pricing(payload, "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(row["variant"], "")
        self.assertIsNone(row["tp_market"])
        self.assertEqual(row["cm_avg"], 5.0)

    def test_zero_price_is_not_a_price(self) -> None:
        payload = {"id": "x-1", "pricing": {"cardmarket": {"avg": 0}}}
        self.assertIsNone(tcgdex_api.parse_card_pricing(payload, "2026-08-07"))

    def test_missing_id_yields_none(self) -> None:
        payload = {"pricing": {"cardmarket": {"avg": 5.0}}}
        self.assertIsNone(tcgdex_api.parse_card_pricing(payload, "2026-08-07"))


class TestTercileBounds(unittest.TestCase):
    def test_splits_into_thirds(self) -> None:
        lo, hi = tcgdex_api.tercile_bounds([float(i) for i in range(1, 10)])
        self.assertAlmostEqual(lo, 4.0)
        self.assertAlmostEqual(hi, 7.0)

    def test_single_value_gives_equal_bounds(self) -> None:
        lo, hi = tcgdex_api.tercile_bounds([5.0])
        self.assertEqual(lo, 5.0)
        self.assertEqual(hi, 5.0)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            tcgdex_api.tercile_bounds([])


class TestBandOf(unittest.TestCase):
    def test_assigns_by_bounds(self) -> None:
        bounds = (4.0, 7.0)
        self.assertEqual(tcgdex_api.band_of(2.0, bounds), "저가")
        self.assertEqual(tcgdex_api.band_of(5.0, bounds), "중가")
        self.assertEqual(tcgdex_api.band_of(9.0, bounds), "고가")

    def test_boundary_goes_upward(self) -> None:
        bounds = (4.0, 7.0)
        self.assertEqual(tcgdex_api.band_of(4.0, bounds), "중가")
        self.assertEqual(tcgdex_api.band_of(7.0, bounds), "고가")


class TestStratify(unittest.TestCase):
    def _cards(self, era: str, n: int, offset: float = 0.0):
        return [
            {"card_id": f"{era}-{i}", "era": era, "price": float(i) + offset}
            for i in range(1, n + 1)
        ]

    def test_picks_per_cell_from_every_cell(self) -> None:
        cards = []
        for era in tcgdex_api.ERAS:
            cards += self._cards(era, 60)
        picked = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        self.assertEqual(len(picked), 5 * 3 * len(tcgdex_api.ERAS))
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cell = [c for c in picked if c["era"] == era and c["band"] == band]
                self.assertEqual(len(cell), 5, f"{era}/{band}")

    def test_is_deterministic_for_same_seed(self) -> None:
        cards = []
        for era in tcgdex_api.ERAS:
            cards += self._cards(era, 60)
        a = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        b = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        self.assertEqual([c["card_id"] for c in a], [c["card_id"] for c in b])

    def test_takes_all_when_cell_smaller_than_quota(self) -> None:
        cards = self._cards("빈티지", 6)
        picked = tcgdex_api.stratify(cards, per_cell=5, seed=1)
        self.assertEqual(len(picked), 6)

    def test_ignores_cards_without_era(self) -> None:
        cards = self._cards("빈티지", 30) + [{"card_id": "x", "era": None, "price": 1.0}]
        picked = tcgdex_api.stratify(cards, per_cell=3, seed=1)
        self.assertNotIn("x", [c["card_id"] for c in picked])

    def test_spans_the_whole_price_range_of_a_cell(self) -> None:
        """분위 추출이라 한 칸의 표본이 저가 쪽에만 몰리지 않아야 한다."""
        cards = self._cards("빈티지", 300)
        picked = [c for c in tcgdex_api.stratify(cards, per_cell=10, seed=7)
                  if c["band"] == "저가"]
        prices = [c["price"] for c in picked]
        self.assertGreater(max(prices) - min(prices), 40)


class TestFillForward(unittest.TestCase):
    def test_carries_last_value(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([1.0, None, None, 4.0], max_days=7),
            [1.0, 1.0, 1.0, 4.0],
        )

    def test_stops_after_max_days(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([1.0, None, None, None], max_days=2),
            [1.0, 1.0, 1.0, None],
        )

    def test_leading_none_stays_none(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([None, None, 3.0], max_days=7),
            [None, None, 3.0],
        )


class TestIndexPoint(unittest.TestCase):
    def _universe(self):
        out = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                out.append({"card_id": f"{era}-{band}", "era": era, "band": band})
        return out

    def test_all_flat_is_one_hundred(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        point = tcgdex_api.index_point(uni, dict(base), base)
        self.assertAlmostEqual(point["index"], 100.0)
        self.assertEqual(point["missing"], 0)

    def test_uniform_doubling_is_two_hundred(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 20.0 for k in base}
        self.assertAlmostEqual(tcgdex_api.index_point(uni, day, base)["index"], 200.0)

    def test_cells_are_equally_weighted(self) -> None:
        """한 칸만 2배가 되면 지수는 1/12 만큼만 오른다."""
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = dict(base)
        day["빈티지-고가"] = 20.0
        expected = 100.0 * (11 * 1.0 + 2.0) / 12
        self.assertAlmostEqual(tcgdex_api.index_point(uni, day, base)["index"], expected)

    def test_missing_card_is_excluded_and_counted(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 10.0 for k in base if k != "빈티지-고가"}
        point = tcgdex_api.index_point(uni, day, base)
        self.assertEqual(point["missing"], 1)
        self.assertAlmostEqual(point["index"], 100.0)

    def test_empty_cell_drops_out_of_the_average(self) -> None:
        """칸이 통째로 비면 그 칸을 빼고 남은 칸으로 평균낸다."""
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 20.0 for k in base if k != "빈티지-고가"}
        point = tcgdex_api.index_point(uni, day, base)
        self.assertAlmostEqual(point["index"], 200.0)

    def test_sub_indices_are_reported(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = dict(base)
        for band in tcgdex_api.BANDS:
            day[f"빈티지-{band}"] = 20.0
        point = tcgdex_api.index_point(uni, day, base)
        self.assertAlmostEqual(point["by_era"]["빈티지"], 200.0)
        self.assertAlmostEqual(point["by_era"]["최신"], 100.0)
        self.assertAlmostEqual(point["by_band"]["고가"], 100.0 * (1 * 2.0 + 3 * 1.0) / 4)

    def test_no_usable_prices_yields_none(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        point = tcgdex_api.index_point(uni, {}, base)
        self.assertIsNone(point["index"])
        self.assertEqual(point["missing"], len(uni))


if __name__ == "__main__":
    unittest.main()
