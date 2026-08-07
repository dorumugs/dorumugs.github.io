"""화면용 JSON 집계 검증.

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

COL = {name: i for i, name in enumerate(build_pokemon.VIEW_COLUMNS)}


def _card(cid, set_id, name, dex="", market=10.0, high=None, obs=None,
          obs_date="", cm=None, rarity="Rare") -> dict:
    row = {c: None for c in tcgdex_api.CARD_COLUMNS}
    row.update({"card_id": cid, "set_id": set_id, "local_id": cid.split("-")[-1],
                "name_en": name, "dex_id": dex, "rarity": rarity, "category": "Pokemon",
                "image": f"x/{set_id}/1", "tp_market": market, "tp_high": high,
                "obs_max": obs, "obs_max_date": obs_date, "cm_avg": cm,
                "updated": "2026-08-08"})
    return row


class TestViewRow(unittest.TestCase):
    def test_maps_columns_in_order(self) -> None:
        r = build_pokemon.view_row(
            _card("base1-4", "base1", "Charizard", dex="6", market=818.65,
                  high=4590.63, obs=900.0, obs_date="2026-08-07", cm=446.7),
            {"6": "리자몽"})
        self.assertEqual(len(r), len(build_pokemon.VIEW_COLUMNS))
        self.assertEqual(r[COL["card_id"]], "base1-4")
        self.assertEqual(r[COL["name_en"]], "Charizard")
        self.assertEqual(r[COL["name_ko"]], "리자몽")
        self.assertEqual(r[COL["price"]], 818.65)
        self.assertEqual(r[COL["high_ask"]], 4590.63)
        self.assertEqual(r[COL["obs_max"]], 900.0)
        self.assertEqual(r[COL["obs_max_date"]], "2026-08-07")

    def test_no_dex_id_means_no_korean_name(self) -> None:
        r = build_pokemon.view_row(_card("base1-102", "base1", "Water Energy"), {"6": "리자몽"})
        self.assertEqual(r[COL["name_ko"]], "")

    def test_cardmarket_fills_in_when_tcgplayer_missing(self) -> None:
        r = build_pokemon.view_row(
            _card("a-1", "s", "A", market=None, cm=12.5), {})
        self.assertEqual(r[COL["price"]], 12.5)

    def test_missing_price_is_none_not_zero(self) -> None:
        r = build_pokemon.view_row(_card("a-1", "s", "A", market=None), {})
        self.assertIsNone(r[COL["price"]])


class TestBuildCards(unittest.TestCase):
    def _meta(self):
        return {"s1": {"name": "Set One", "release_date": "2020-01-01",
                       "era": "최신", "included": True},
                "s2": {"name": "Set Two", "release_date": "1999-01-09",
                       "era": "빈티지", "included": False}}

    def test_excluded_sets_are_dropped(self) -> None:
        cards = [_card("s1-1", "s1", "A"), _card("s2-1", "s2", "B")]
        out = build_pokemon.build_cards(cards, self._meta(), {})
        self.assertEqual([r[COL["card_id"]] for r in out], ["s1-1"])

    def test_sorted_by_price_desc(self) -> None:
        cards = [_card("s1-1", "s1", "A", market=5.0),
                 _card("s1-2", "s1", "B", market=50.0),
                 _card("s1-3", "s1", "C", market=20.0)]
        out = build_pokemon.build_cards(cards, self._meta(), {})
        self.assertEqual([r[COL["price"]] for r in out], [50.0, 20.0, 5.0])

    def test_priceless_cards_sort_last(self) -> None:
        cards = [_card("s1-1", "s1", "A", market=None),
                 _card("s1-2", "s1", "B", market=1.0)]
        out = build_pokemon.build_cards(cards, self._meta(), {})
        self.assertEqual(out[0][COL["card_id"]], "s1-2")


class TestBuildSets(unittest.TestCase):
    def test_counts_only_sets_with_cards(self) -> None:
        meta = {"s1": {"name": "One", "release_date": "2020-01-01", "era": "최신",
                       "included": True},
                "s9": {"name": "Nine", "release_date": "2021-01-01", "era": "최신",
                       "included": True}}
        cards = [build_pokemon.view_row(_card("s1-1", "s1", "A"), {}),
                 build_pokemon.view_row(_card("s1-2", "s1", "B"), {})]
        out = build_pokemon.build_sets(meta, cards)
        self.assertEqual(list(out), ["s1"])
        self.assertEqual(out["s1"]["count"], 2)

    def test_newest_set_first(self) -> None:
        meta = {"old": {"name": "Old", "release_date": "1999-01-09", "era": "빈티지",
                        "included": True},
                "new": {"name": "New", "release_date": "2025-01-01", "era": "최신",
                        "included": True}}
        cards = [build_pokemon.view_row(_card("old-1", "old", "A"), {}),
                 build_pokemon.view_row(_card("new-1", "new", "B"), {})]
        self.assertEqual(list(build_pokemon.build_sets(meta, cards)), ["new", "old"])


if __name__ == "__main__":
    unittest.main()
