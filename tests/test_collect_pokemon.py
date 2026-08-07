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


def _row(cid: str, market=None, obs=None, obs_date="", name="X", updated="2026-08-08") -> dict:
    row = {c: None for c in tcgdex_api.CARD_COLUMNS}
    row.update({"card_id": cid, "set_id": cid.split("-")[0], "local_id": "1",
                "name_en": name, "dex_id": "", "rarity": "Common", "category": "Pokemon",
                "image": "s/s1/1", "tp_market": market, "obs_max": obs,
                "obs_max_date": obs_date, "updated": updated})
    return row


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
        self.assertAlmostEqual(
            collect_pokemon.coverage_of(rows, ["a-1", "a-2", "a-3", "a-4"]), 0.5)

    def test_empty_set_is_zero(self) -> None:
        self.assertEqual(collect_pokemon.coverage_of([], []), 0.0)


class TestCsvRoundTrip(unittest.TestCase):
    def test_header_matches_columns(self) -> None:
        out = collect_pokemon.rows_to_csv([_row("a-1", 1.5, 1.5, "2026-08-08")])
        self.assertEqual(out.strip().split("\n")[0], ",".join(tcgdex_api.CARD_COLUMNS))

    def test_round_trip_preserves_values(self) -> None:
        rows = [_row("a-1", 12.34, 20.0, "2026-08-01")]
        back = collect_pokemon.csv_to_rows(collect_pokemon.rows_to_csv(rows))
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["card_id"], "a-1")
        self.assertEqual(back[0]["tp_market"], 12.34)
        self.assertEqual(back[0]["obs_max"], 20.0)
        self.assertEqual(back[0]["obs_max_date"], "2026-08-01")

    def test_none_becomes_none_again(self) -> None:
        back = collect_pokemon.csv_to_rows(collect_pokemon.rows_to_csv([_row("a-1")]))
        self.assertIsNone(back[0]["tp_market"])
        self.assertIsNone(back[0]["obs_max"])

    def test_empty_text_is_empty_list(self) -> None:
        self.assertEqual(collect_pokemon.csv_to_rows(""), [])


class TestMergeCards(unittest.TestCase):
    def test_new_card_is_added(self) -> None:
        out = collect_pokemon.merge_cards([], [_row("a-1", 1.0, 1.0, "2026-08-08")])
        self.assertEqual(len(out), 1)

    def test_same_card_is_replaced_not_duplicated(self) -> None:
        old = [_row("a-1", 1.0, 1.0, "2026-08-07")]
        out = collect_pokemon.merge_cards(old, [_row("a-1", 2.0, 2.0, "2026-08-08")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tp_market"], 2.0)

    def test_observed_max_survives_a_price_drop(self) -> None:
        old = [_row("a-1", 30.0, 30.0, "2026-08-07")]
        out = collect_pokemon.merge_cards(old, [_row("a-1", 5.0, 5.0, "2026-08-08")])
        self.assertEqual(out[0]["tp_market"], 5.0)
        self.assertEqual(out[0]["obs_max"], 30.0)
        self.assertEqual(out[0]["obs_max_date"], "2026-08-07")

    def test_rows_are_sorted_by_card_id(self) -> None:
        out = collect_pokemon.merge_cards([], [_row("b-1"), _row("a-1"), _row("c-1")])
        self.assertEqual([r["card_id"] for r in out], ["a-1", "b-1", "c-1"])

    def test_untouched_cards_are_kept(self) -> None:
        old = [_row("a-1", 1.0, 1.0, "2026-08-07"), _row("z-9", 9.0, 9.0, "2026-08-07")]
        out = collect_pokemon.merge_cards(old, [_row("a-1", 2.0, 2.0, "2026-08-08")])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["card_id"], "z-9")


if __name__ == "__main__":
    unittest.main()
