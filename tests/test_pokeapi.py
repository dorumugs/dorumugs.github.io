"""PokéAPI 종 이름 파싱 검증.

    python3 -m unittest tests.test_pokeapi -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"

import pokeapi  # noqa: E402


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestParseSpecies(unittest.TestCase):
    def test_extracts_korean_name(self) -> None:
        self.assertEqual(pokeapi.parse_species(_fixture("pokeapi_species_6.json")),
                         ("6", "리자몽"))

    def test_no_korean_entry_yields_none(self) -> None:
        payload = {"id": 1, "names": [{"language": {"name": "en"}, "name": "Bulbasaur"}]}
        self.assertIsNone(pokeapi.parse_species(payload))

    def test_blank_korean_name_yields_none(self) -> None:
        payload = {"id": 1, "names": [{"language": {"name": "ko"}, "name": "  "}]}
        self.assertIsNone(pokeapi.parse_species(payload))

    def test_missing_id_raises(self) -> None:
        with self.assertRaises(pokeapi.ApiError):
            pokeapi.parse_species({"names": []})


class TestKoreanName(unittest.TestCase):
    def test_looks_up_by_dex_id(self) -> None:
        self.assertEqual(pokeapi.korean_name("6", {"6": "리자몽"}), "리자몽")

    def test_missing_dex_id_is_blank(self) -> None:
        """트레이너·에너지 카드는 도감번호가 없다."""
        self.assertEqual(pokeapi.korean_name("", {"6": "리자몽"}), "")

    def test_unknown_dex_id_is_blank(self) -> None:
        self.assertEqual(pokeapi.korean_name("9999", {"6": "리자몽"}), "")


if __name__ == "__main__":
    unittest.main()
