"""지수 집계 검증.

    python3 -m unittest tests.test_build_pokemon -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_pokemon  # noqa: E402
import tcgdex_api  # noqa: E402


def _scan_row(cid: str, market: float) -> dict:
    row = {c: None for c in tcgdex_api.COLUMNS}
    row.update({"date": "2026-08-07", "card_id": cid, "variant": "normal", "tp_market": market})
    return row


class TestBuildUniverse(unittest.TestCase):
    def _inputs(self):
        rows, meta, names = [], {}, {}
        for si, (era, rd) in enumerate(
            [("빈티지", "2001-06-01"), ("클래식", "2007-08-01"),
             ("모던", "2014-05-07"), ("최신", "2022-07-01")]
        ):
            sid = f"s{si}"
            meta[sid] = {"name": f"Set {si}", "release_date": rd, "era": era,
                         "included": True, "coverage": 1.0, "card_count": 90}
            for i in range(90):
                rows.append(_scan_row(f"{sid}-{i}", float(i + 1)))
                names[f"{sid}-{i}"] = f"카드{si}_{i}"
        return rows, meta, names

    def test_picks_per_cell_from_every_cell(self) -> None:
        rows, meta, names = self._inputs()
        uni = build_pokemon.build_universe(rows, meta, names, seed=1, per_cell=5)
        self.assertEqual(len(uni["cards"]), 5 * 3 * 4)
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cell = [c for c in uni["cards"] if c["era"] == era and c["band"] == band]
                self.assertEqual(len(cell), 5, f"{era}/{band}")

    def test_records_base_price_set_and_name(self) -> None:
        rows, meta, names = self._inputs()
        uni = build_pokemon.build_universe(rows, meta, names, seed=1, per_cell=5)
        card = uni["cards"][0]
        self.assertGreater(card["base_price"], 0)
        self.assertTrue(card["set_name"])
        self.assertTrue(card["name"].startswith("카드"))
        self.assertEqual(uni["seed"], 1)

    def test_falls_back_to_card_id_when_name_unknown(self) -> None:
        rows, meta, _ = self._inputs()
        uni = build_pokemon.build_universe(rows, meta, {}, seed=1, per_cell=5)
        self.assertTrue(uni["cards"][0]["name"])

    def test_excluded_sets_are_ignored(self) -> None:
        rows, meta, names = self._inputs()
        meta["s0"]["included"] = False
        uni = build_pokemon.build_universe(rows, meta, names, seed=1, per_cell=5)
        self.assertFalse([c for c in uni["cards"] if c["set_id"] == "s0"])


class TestSeriesFromPrices(unittest.TestCase):
    def _universe(self):
        cards = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cards.append({"card_id": f"{era}-{band}", "era": era, "band": band,
                              "base_price": 10.0, "name": "x", "set_id": "s",
                              "set_name": "S"})
        return {"base_date": "2026-08-07", "seed": 1, "cards": cards}

    def _rows(self, day: str, price: float):
        out = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                r = {c: None for c in tcgdex_api.COLUMNS}
                r.update({"date": day, "card_id": f"{era}-{band}", "tp_market": price})
                out.append(r)
        return out

    def test_base_date_is_one_hundred(self) -> None:
        uni = self._universe()
        s = build_pokemon.series_from_prices(uni, self._rows("2026-08-07", 10.0))
        self.assertEqual(s["dates"], ["2026-08-07"])
        self.assertAlmostEqual(s["index"][0], 100.0)

    def test_doubling_next_day(self) -> None:
        uni = self._universe()
        rows = self._rows("2026-08-07", 10.0) + self._rows("2026-08-08", 20.0)
        s = build_pokemon.series_from_prices(uni, rows)
        self.assertAlmostEqual(s["index"][1], 200.0)
        self.assertAlmostEqual(s["by_era"]["빈티지"][1], 200.0)

    def test_dates_are_sorted(self) -> None:
        uni = self._universe()
        rows = self._rows("2026-08-09", 10.0) + self._rows("2026-08-07", 10.0)
        s = build_pokemon.series_from_prices(uni, rows)
        self.assertEqual(s["dates"], ["2026-08-07", "2026-08-09"])

    def test_cardmarket_is_used_when_tcgplayer_missing(self) -> None:
        uni = self._universe()
        rows = self._rows("2026-08-07", 10.0)
        for r in rows:
            r["tp_market"] = None
            r["cm_avg"] = 20.0
        s = build_pokemon.series_from_prices(uni, rows)
        self.assertAlmostEqual(s["index"][0], 200.0)


class TestBackcast(unittest.TestCase):
    def _universe(self):
        return {"base_date": "2026-08-07", "cards": [
            {"card_id": "a-1", "era": "빈티지", "band": "고가", "base_price": 10.0,
             "name": "A", "set_id": "s", "set_name": "S"},
        ]}

    def test_uses_cardmarket_averages(self) -> None:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "a-1", "tp_market": 10.0,
                    "cm_avg": 10.0, "cm_avg7": 8.0, "cm_avg30": 5.0})
        out = build_pokemon.backcast_from_cardmarket(self._universe(), [row])
        self.assertEqual(len(out["dates"]), 3)
        self.assertLess(out["index"][0], out["index"][-1])
        self.assertAlmostEqual(out["index"][-1], 100.0)
        self.assertTrue(out["estimated"])

    def test_dates_run_backwards_from_base(self) -> None:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "a-1",
                    "cm_avg": 10.0, "cm_avg7": 8.0, "cm_avg30": 5.0})
        out = build_pokemon.backcast_from_cardmarket(self._universe(), [row])
        self.assertEqual(out["dates"], ["2026-07-08", "2026-07-31", "2026-08-07"])

    def test_no_cardmarket_yields_empty(self) -> None:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "a-1", "tp_market": 10.0})
        self.assertEqual(build_pokemon.backcast_from_cardmarket(self._universe(), [row])["dates"], [])


if __name__ == "__main__":
    unittest.main()
