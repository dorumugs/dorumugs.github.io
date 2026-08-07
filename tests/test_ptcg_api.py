"""pokemontcg.io 대체 사진 경로 검증.

    python3 -m unittest tests.test_ptcg_api -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ptcg_api  # noqa: E402


class NumberVariants(unittest.TestCase):
    """순서대로 시도할 후보 목록이다. 우리 번호가 언제나 먼저다."""

    def test_our_number_comes_first(self):
        self.assertEqual(ptcg_api.number_variants("SV001")[0], "SV001")
        self.assertEqual(ptcg_api.number_variants("GG01")[0], "GG01")

    def test_leading_zero_variant(self):
        """Skyridge 는 우리가 'H09' 로 적는 걸 저쪽은 'H9' 로 적는다."""
        self.assertEqual(ptcg_api.number_variants("H09"), ["H09", "H9"])
        self.assertEqual(ptcg_api.number_variants("SV001"), ["SV001", "SV1"])

    def test_no_duplicate_when_nothing_to_strip(self):
        self.assertEqual(ptcg_api.number_variants("GG11"), ["GG11"])

    def test_pure_number_unchanged(self):
        self.assertEqual(ptcg_api.number_variants("12"), ["12"])

    def test_blank(self):
        self.assertEqual(ptcg_api.number_variants(""), [])
        self.assertEqual(ptcg_api.number_variants(None), [])


class CandidatePaths(unittest.TestCase):
    def test_mapped_set(self):
        self.assertEqual(ptcg_api.candidate_paths("swsh4.5sv", "SV001")[0],
                         "swsh45sv/SV001")

    def test_variant_included(self):
        self.assertEqual(ptcg_api.candidate_paths("ecard3", "H09"),
                         ["ecard3/H09", "ecard3/H9"])

    def test_unmapped_set_gives_nothing(self):
        """My First Battle 은 저쪽에도 없다. 없는 주소를 지어내지 않는다."""
        self.assertEqual(ptcg_api.candidate_paths("mfb", "1"), [])


class Urls(unittest.TestCase):
    def test_thumb_and_full(self):
        self.assertEqual(ptcg_api.thumb_url("swsh45sv/SV001"),
                         "https://images.pokemontcg.io/swsh45sv/SV001.png")
        self.assertEqual(ptcg_api.full_url("swsh45sv/SV001"),
                         "https://images.pokemontcg.io/swsh45sv/SV001_hires.png")


class NeedsArt(unittest.TestCase):
    def test_tcgdex_image_is_enough(self):
        self.assertFalse(ptcg_api.needs_art("base/base1/4", ""))

    def test_tcgplayer_id_is_enough(self):
        self.assertFalse(ptcg_api.needs_art("", "123456"))

    def test_nothing_means_needs_art(self):
        self.assertTrue(ptcg_api.needs_art("", ""))

    def test_partial_path_does_not_count(self):
        self.assertTrue(ptcg_api.needs_art("base1/4", ""))
        self.assertTrue(ptcg_api.needs_art("base//4", ""))


if __name__ == "__main__":
    unittest.main()
