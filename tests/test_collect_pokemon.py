"""수집기의 순수 부분 검증. 네트워크는 타지 않는다.

    python3 -m unittest tests.test_collect_pokemon -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import tcgdex_api  # noqa: E402


class TestIsCandidateSet(unittest.TestCase):
    def test_accepts_regular_expansions(self) -> None:
        for name in ("Base Set", "Neo Discovery", "Flashfire", "Phantasmal Flames"):
            self.assertTrue(collect_pokemon.is_candidate_set(name, "2014-05-07"), name)

    def test_rejects_promos_and_kits(self) -> None:
        for name in ("Nintendo Black Star Promos", "DP trainer Kit (Manaphy)",
                     "Unseen Forces Unown Collection", "Miscellaneous Promos"):
            self.assertFalse(collect_pokemon.is_candidate_set(name, "2005-08-22"), name)

    def test_rejects_missing_release_date(self) -> None:
        self.assertFalse(collect_pokemon.is_candidate_set("Base Set", ""))


class TestCoverageOf(unittest.TestCase):
    def test_ratio_of_priced_cards(self) -> None:
        rows = [{"card_id": "a-1"}, {"card_id": "a-2"}]
        self.assertAlmostEqual(collect_pokemon.coverage_of(rows, ["a-1", "a-2", "a-3", "a-4"]), 0.5)

    def test_empty_set_is_zero(self) -> None:
        self.assertEqual(collect_pokemon.coverage_of([], []), 0.0)

    def test_full_coverage(self) -> None:
        rows = [{"card_id": "a-1"}, {"card_id": "a-2"}]
        self.assertEqual(collect_pokemon.coverage_of(rows, ["a-1", "a-2"]), 1.0)


class TestRowsToCsv(unittest.TestCase):
    def test_writes_header_and_none_as_blank(self) -> None:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "base1-4",
                    "variant": "holofoil", "tp_market": 818.65})
        out = collect_pokemon.rows_to_csv([row])
        lines = out.strip().split("\n")
        self.assertEqual(lines[0], ",".join(tcgdex_api.COLUMNS))
        self.assertIn("base1-4,holofoil,818.65", lines[1])
        self.assertTrue(lines[1].endswith(",,,,"), lines[1])


class TestMergeRows(unittest.TestCase):
    def _row(self, date: str, cid: str, market: float) -> dict:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": date, "card_id": cid, "variant": "normal", "tp_market": market})
        return row

    def test_appends_to_existing(self) -> None:
        first = collect_pokemon.rows_to_csv([self._row("2026-08-07", "a-1", 1.0)])
        merged = collect_pokemon.merge_rows(first, [self._row("2026-08-08", "a-1", 2.0)])
        self.assertEqual(len(merged.strip().split("\n")), 3)

    def test_same_date_and_card_is_replaced_not_duplicated(self) -> None:
        first = collect_pokemon.rows_to_csv([self._row("2026-08-07", "a-1", 1.0)])
        merged = collect_pokemon.merge_rows(first, [self._row("2026-08-07", "a-1", 9.0)])
        lines = merged.strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("9.0", lines[1])

    def test_empty_existing_is_fine(self) -> None:
        merged = collect_pokemon.merge_rows("", [self._row("2026-08-07", "a-1", 1.0)])
        self.assertEqual(len(merged.strip().split("\n")), 2)

    def test_rows_are_sorted_by_date_then_card(self) -> None:
        merged = collect_pokemon.merge_rows("", [
            self._row("2026-08-08", "b-1", 1.0),
            self._row("2026-08-07", "z-9", 1.0),
            self._row("2026-08-07", "a-1", 1.0),
        ])
        lines = merged.strip().split("\n")[1:]
        self.assertTrue(lines[0].startswith("2026-08-07,a-1"))
        self.assertTrue(lines[1].startswith("2026-08-07,z-9"))
        self.assertTrue(lines[2].startswith("2026-08-08,b-1"))


class TestLoadUniverseIds(unittest.TestCase):
    def test_extracts_card_ids_in_order(self) -> None:
        uni = {"base_date": "2026-08-07", "cards": [
            {"card_id": "b-2"}, {"card_id": "a-1"},
        ]}
        self.assertEqual(collect_pokemon.load_universe_ids(uni), ["b-2", "a-1"])

    def test_missing_cards_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            collect_pokemon.load_universe_ids({"base_date": "2026-08-07"})


if __name__ == "__main__":
    unittest.main()
