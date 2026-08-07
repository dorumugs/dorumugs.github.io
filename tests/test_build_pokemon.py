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

C = {name: i for i, name in enumerate(build_pokemon.VIEW_COLUMNS)}


def _card(cid, set_id, name, dex="", market=10.0, high=None, obs=None,
          obs_date="", cm=None, rarity="Rare", serie="x") -> dict:
    local = cid.split("-")[-1]
    row = {c: None for c in tcgdex_api.CARD_COLUMNS}
    row.update({"card_id": cid, "set_id": set_id, "local_id": local,
                "name_en": name, "dex_id": dex, "rarity": rarity, "category": "Pokemon",
                "image": f"{serie}/{set_id}/{local}", "tp_market": market, "tp_high": high,
                "obs_max": obs, "obs_max_date": obs_date, "cm_avg": cm,
                "updated": "2026-08-08"})
    return row


def _meta():
    return {"s1": {"name": "Set One", "release_date": "2020-01-01",
                   "era": "최신", "included": True},
            "s2": {"name": "Set Two", "release_date": "1999-01-09",
                   "era": "빈티지", "included": False}}


class TestSerieOf(unittest.TestCase):
    def test_takes_first_segment(self) -> None:
        self.assertEqual(build_pokemon.serie_of("base/base1/4"), "base")

    def test_unexpected_shape_is_blank(self) -> None:
        self.assertEqual(build_pokemon.serie_of(""), "")
        self.assertEqual(build_pokemon.serie_of("base1/4"), "")


class TestDictionary(unittest.TestCase):
    def test_same_value_gets_same_index(self) -> None:
        d = build_pokemon.Dictionary()
        self.assertEqual(d.index("a"), 0)
        self.assertEqual(d.index("b"), 1)
        self.assertEqual(d.index("a"), 0)
        self.assertEqual(d.values, ["a", "b"])

    def test_none_becomes_blank(self) -> None:
        d = build_pokemon.Dictionary()
        self.assertEqual(d.index(None), 0)
        self.assertEqual(d.values, [""])


class TestBuildPayload(unittest.TestCase):
    def test_row_shape_and_dictionary_lookup(self) -> None:
        cards = [_card("s1-4", "s1", "Charizard", dex="6", market=818.65, high=4590.63,
                       obs=900.0, obs_date="2026-08-07", cm=402.79, rarity="Rare Holo",
                       serie="base")]
        p = build_pokemon.build_payload(cards, _meta(), {"6": "리자몽"})
        row = p["rows"][0]
        self.assertEqual(len(row), len(build_pokemon.VIEW_COLUMNS))
        self.assertEqual(p["sets"][row[C["set"]]], "s1")
        self.assertEqual(p["series"][row[C["set"]]], "base")
        self.assertEqual(row[C["local_id"]], "4")
        self.assertEqual(row[C["name_en"]], "Charizard")
        self.assertEqual(row[C["name_ko"]], "리자몽")
        self.assertEqual(p["rarities"][row[C["rarity"]]], "Rare Holo")
        self.assertEqual(row[C["price"]], 818.65)
        self.assertEqual(row[C["high_ask"]], 4590.63)
        self.assertEqual(row[C["obs_max"]], 900.0)
        self.assertEqual(p["dates"][row[C["obs_date"]]], "2026-08-07")

    def test_card_id_and_image_are_derivable(self) -> None:
        """저장하지 않는 두 값이 규칙으로 되만들어지는지."""
        cards = [_card("s1-4", "s1", "A", serie="base")]
        p = build_pokemon.build_payload(cards, _meta(), {})
        row = p["rows"][0]
        sid = p["sets"][row[C["set"]]]
        serie = p["series"][row[C["set"]]]
        self.assertEqual(sid + "-" + row[C["local_id"]], "s1-4")
        self.assertEqual(serie + "/" + sid + "/" + row[C["local_id"]], "base/s1/4")

    def test_no_dex_id_means_no_korean_name(self) -> None:
        p = build_pokemon.build_payload(
            [_card("s1-102", "s1", "Water Energy")], _meta(), {"6": "리자몽"})
        self.assertEqual(p["rows"][0][C["name_ko"]], "")

    def test_cardmarket_fills_in_when_tcgplayer_missing(self) -> None:
        p = build_pokemon.build_payload(
            [_card("s1-1", "s1", "A", market=None, cm=12.5)], _meta(), {})
        self.assertEqual(p["rows"][0][C["price"]], 12.5)

    def test_excluded_sets_are_dropped(self) -> None:
        cards = [_card("s1-1", "s1", "A"), _card("s2-1", "s2", "B")]
        p = build_pokemon.build_payload(cards, _meta(), {})
        self.assertEqual(len(p["rows"]), 1)
        self.assertEqual(p["sets"], ["s1"])

    def test_sorted_by_price_desc(self) -> None:
        cards = [_card("s1-1", "s1", "A", market=5.0),
                 _card("s1-2", "s1", "B", market=50.0),
                 _card("s1-3", "s1", "C", market=20.0)]
        p = build_pokemon.build_payload(cards, _meta(), {})
        self.assertEqual([r[C["price"]] for r in p["rows"]], [50.0, 20.0, 5.0])

    def test_priceless_cards_sort_last(self) -> None:
        cards = [_card("s1-1", "s1", "A", market=None),
                 _card("s1-2", "s1", "B", market=1.0)]
        p = build_pokemon.build_payload(cards, _meta(), {})
        self.assertEqual(p["rows"][0][C["local_id"]], "2")

    def test_repeated_strings_are_stored_once(self) -> None:
        cards = [_card(f"s1-{i}", "s1", f"C{i}", rarity="Rare Holo") for i in range(20)]
        p = build_pokemon.build_payload(cards, _meta(), {})
        self.assertEqual(p["sets"], ["s1"])
        self.assertEqual(p["rarities"], ["Rare Holo"])


class TestBuildSets(unittest.TestCase):
    def test_counts_only_sets_with_cards(self) -> None:
        meta = dict(_meta())
        meta["s9"] = {"name": "Nine", "release_date": "2021-01-01", "era": "최신",
                      "included": True}
        p = build_pokemon.build_payload(
            [_card("s1-1", "s1", "A"), _card("s1-2", "s1", "B")], meta, {})
        out = build_pokemon.build_sets(meta, p)
        self.assertEqual(list(out), ["s1"])
        self.assertEqual(out["s1"]["count"], 2)

    def test_newest_set_first(self) -> None:
        meta = {"old": {"name": "Old", "release_date": "1999-01-09", "era": "빈티지",
                        "included": True},
                "new": {"name": "New", "release_date": "2025-01-01", "era": "최신",
                        "included": True}}
        p = build_pokemon.build_payload(
            [_card("old-1", "old", "A"), _card("new-1", "new", "B")], meta, {})
        self.assertEqual(list(build_pokemon.build_sets(meta, p)), ["new", "old"])


if __name__ == "__main__":
    unittest.main()
