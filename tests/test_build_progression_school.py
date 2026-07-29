"""학교별 진학률 집계 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_progression_school as bps  # noqa: E402


def _row(**kw) -> dict:
    base = {"year": "2025", "sido": "서울특별시", "sgg": "강남구",
            "sgg_code": "1168000000", "school_name": "개원중학교",
            "shl_idf_cd": "uuid-1", "grad": "291", "general": "241",
            "vocational": "19", "science": "0", "foreign_intl": "9",
            "art_pe": "3", "meister": "0", "special_sum": "12",
            "auto_private": "15", "auto_public": "0", "auto_sum": "15",
            "etc": "4", "advanced": "291", "employed": "0",
            "alternative": "0", "none": "0"}
    base.update({k: str(v) for k, v in kw.items()})
    return base


class TestBuild(unittest.TestCase):
    def test_rate_and_shape(self) -> None:
        out = bps.build([_row()], "2026-07-29")
        self.assertEqual(out["years"], ["2025"])
        self.assertEqual(len(out["schools"]), 1)
        s = out["schools"][0]
        self.assertEqual(s["name"], "개원중학교")
        self.assertEqual(s["sgg"], "11680")  # 1168000000 앞 5자리
        self.assertEqual(s["r"], [round(27 / 291 * 100, 1)])
        self.assertEqual(s["g"], 291)

    def test_same_school_in_two_sigungu_deduped(self) -> None:
        """학교알리미 시군구 목록은 '수원시'와 '수원시 장안구'를 모두 준다 —
        같은 학교가 두 번 잡히므로 UUID 로 합치고, 구까지 있는 이름을 남긴다."""
        rows = [
            _row(sgg="수원시", sgg_code="4111000000", school_name="A중학교", shl_idf_cd="u2"),
            _row(sgg="수원시 장안구", sgg_code="4111100000", school_name="A중학교", shl_idf_cd="u2"),
        ]
        out = bps.build(rows, "2026-07-29")
        self.assertEqual(len(out["schools"]), 1)
        self.assertEqual(out["schools"][0]["sgg"], "41111")

    def test_years_aligned_with_nulls(self) -> None:
        """연도 배열과 값 배열의 길이가 같아야 화면이 축을 맞출 수 있다."""
        rows = [_row(year="2023"), _row(year="2025", grad=100, special_sum=5, auto_sum=5)]
        out = bps.build(rows, "2026-07-29")
        self.assertEqual(out["years"], ["2023", "2025"])
        s = out["schools"][0]
        self.assertEqual(len(s["r"]), 2)
        self.assertEqual(s["r"][1], 10.0)

    def test_zero_graduates_dropped(self) -> None:
        """졸업자가 0인 해는 비율을 만들 수 없다. 그 학교에 값이 하나도 없으면 뺀다."""
        out = bps.build([_row(grad=0, special_sum=0, auto_sum=0)], "2026-07-29")
        self.assertEqual(out["schools"], [])

    def test_missing_year_is_null(self) -> None:
        rows = [_row(year="2023", shl_idf_cd="u3", school_name="B중학교"),
                _row(year="2025", shl_idf_cd="u4", school_name="C중학교")]
        out = bps.build(rows, "2026-07-29")
        by = {s["name"]: s["r"] for s in out["schools"]}
        self.assertIsNone(by["B중학교"][1])
        self.assertIsNone(by["C중학교"][0])


class TestAgainstRealOutput(unittest.TestCase):
    OUT = ROOT / "assets" / "realestate" / "progression_school.json"

    @unittest.skipUnless(OUT.exists(), "progression_school.json 없음 — 먼저 빌드하세요")
    def test_within_size_budget(self) -> None:
        self.assertLess(self.OUT.stat().st_size, bps.MAX_BYTES)

    @unittest.skipUnless(OUT.exists(), "progression_school.json 없음 — 먼저 빌드하세요")
    def test_every_school_has_aligned_rates(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        n = len(data["years"])
        self.assertTrue(data["schools"])
        for s in data["schools"]:
            self.assertEqual(len(s["r"]), n, s["name"])
            self.assertEqual(len(s["sgg"]), 5, s["name"])
            self.assertTrue(any(v is not None for v in s["r"]), s["name"])

    @unittest.skipUnless(OUT.exists(), "progression_school.json 없음 — 먼저 빌드하세요")
    def test_rates_within_range(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        for s in data["schools"]:
            for v in s["r"]:
                if v is not None:
                    self.assertGreaterEqual(v, 0.0, s["name"])
                    self.assertLessEqual(v, 100.0, s["name"])


if __name__ == "__main__":
    unittest.main()
