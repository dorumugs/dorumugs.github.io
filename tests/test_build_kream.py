"""국내 원화 화면용 JSON 집계 검증.

    python3 -m unittest tests.test_build_kream -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_kream  # noqa: E402
import kream_api  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "kream_chart.json"
C = {name: i for i, name in enumerate(kream_api.PRODUCT_COLUMNS)}


def view() -> dict:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return build_kream.build(payload, "2026-08-08")


class Shape(unittest.TestCase):
    def setUp(self):
        self.view = view()

    def test_columns_match_parser(self):
        """화면이 배열을 이름 없이 읽으므로 순서가 어긋나면 조용히 틀린다."""
        self.assertEqual(self.view["columns"], kream_api.PRODUCT_COLUMNS)
        self.assertEqual(self.view["market_columns"], kream_api.MARKET_COLUMNS)

    def test_every_row_has_every_column(self):
        for row in self.view["rows"]:
            self.assertEqual(len(row), len(kream_api.PRODUCT_COLUMNS))

    def test_history_aligned_to_shared_dates(self):
        days = len(self.view["dates"])
        self.assertGreater(days, 0)
        for row in self.view["rows"]:
            self.assertEqual(len(row[C["history"]]), days)

    def test_market_aligned_to_columns(self):
        for row in self.view["market"]:
            self.assertEqual(len(row), len(kream_api.MARKET_COLUMNS))

    def test_json_serialisable(self):
        json.dumps(self.view, ensure_ascii=False)


class Stats(unittest.TestCase):
    def setUp(self):
        self.stats = view()["stats"]

    def test_counts(self):
        self.assertEqual(self.stats["product_count"], 5)
        self.assertEqual(self.stats["days"], 4)

    def test_date_range_ascending(self):
        self.assertLessEqual(self.stats["first_date"], self.stats["last_date"])

    def test_median_and_max(self):
        rows = view()["rows"]
        prices = sorted(r[C["price"]] for r in rows)
        self.assertEqual(self.stats["max_price"], prices[-1])
        self.assertEqual(self.stats["median_price"], prices[len(prices) // 2])

    def test_thin_is_reported(self):
        """표본이 얇다는 사실을 화면이 숨기지 않도록 집계에 남긴다."""
        self.assertIn("thin", self.stats)
        self.assertGreater(self.stats["thin"], 0)

    def test_languages_cover_all_rows(self):
        self.assertEqual(sum(self.stats["languages"].values()),
                         self.stats["product_count"])


class Generated(unittest.TestCase):
    def test_generated_passed_through(self):
        self.assertEqual(view()["generated"], "2026-08-08")

    def test_version_kept(self):
        self.assertTrue(view()["version"])

    def test_image_prefix_present(self):
        """행에서 호스트를 떼어냈으니 화면이 다시 붙일 접두가 있어야 한다."""
        self.assertTrue(view()["image_prefix"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
