"""EDSS 진학률 집계 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_progression  # noqa: E402


def _row(region="서울특별시교육청", year="2025", sci_m="0", sci_f="0",
        lang_m="0", lang_f="0", auto_m="0", auto_f="0",
        grad_m="0", grad_f="0") -> dict:
    return {
        "시도교육청명": region,
        "공시년도": year,
        "특수과학고진학남학생수": sci_m,
        "특수과학고진학여학생수": sci_f,
        "특수외국어고진학남학생수": lang_m,
        "특수외국어고진학여학생수": lang_f,
        "자율사립고진학남학생수": auto_m,
        "자율사립고진학여학생수": auto_f,
        "졸업생진로_중_남자졸업생수": grad_m,
        "졸업생진로_중_여자졸업생수": grad_f,
    }


class TestAggregateRows(unittest.TestCase):
    def test_six_columns_summed_correctly(self) -> None:
        row = _row(sci_m="10", sci_f="20", lang_m="30", lang_f="40",
                  auto_m="50", auto_f="60", grad_m="500", grad_f="500")
        agg = build_progression.aggregate_rows([row])
        num, den = agg["서울"]["2025"]
        self.assertEqual(num, 10 + 20 + 30 + 40 + 50 + 60)
        self.assertEqual(den, 1000)

    def test_rows_for_same_region_year_accumulate(self) -> None:
        rows = [_row(sci_m="10", grad_m="100"), _row(sci_m="5", grad_m="50")]
        agg = build_progression.aggregate_rows(rows)
        num, den = agg["서울"]["2025"]
        self.assertEqual(num, 15)
        self.assertEqual(den, 150)

    def test_regions_other_than_seoul_gyeonggi_dropped(self) -> None:
        rows = [_row(region="부산광역시교육청", sci_m="10", grad_m="100"),
                _row(region="교육부", sci_m="10", grad_m="100")]
        agg = build_progression.aggregate_rows(rows)
        self.assertEqual(agg, {})

    def test_gyeonggi_kept_alongside_seoul(self) -> None:
        rows = [_row(region="서울특별시교육청", sci_m="1", grad_m="10"),
                _row(region="경기도교육청", sci_m="2", grad_m="20")]
        agg = build_progression.aggregate_rows(rows)
        self.assertIn("서울", agg)
        self.assertIn("경기", agg)

    def test_blank_or_missing_values_treated_as_zero(self) -> None:
        row = _row(sci_m="", grad_m="100")
        del row["특수과학고진학여학생수"]
        agg = build_progression.aggregate_rows([row])
        num, den = agg["서울"]["2025"]
        self.assertEqual(num, 0)
        self.assertEqual(den, 100)


class TestBuildPayload(unittest.TestCase):
    def test_rate_is_numerator_over_denominator(self) -> None:
        agg = {"서울": {"2025": (250, 1000)}, "경기": {"2025": (10, 1000)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        self.assertEqual(payload["regions"]["서울"]["rate"], [25.0])
        self.assertEqual(payload["regions"]["경기"]["rate"], [1.0])

    def test_zero_denominator_does_not_crash(self) -> None:
        agg = {"서울": {"2025": (0, 0)}, "경기": {"2025": (0, 100)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        self.assertIsNone(payload["regions"]["서울"]["rate"][0])
        self.assertEqual(payload["regions"]["경기"]["rate"][0], 0.0)

    def test_excluded_years_are_dropped_from_output(self) -> None:
        # 2009 는 (원본에서 실제로 그렇듯) 여섯 컬럼 합이 0인 해라도 하드코딩
        # 제외 대상이다 — '집계 안 됨'을 '0%'로 보여주면 안 되기 때문.
        agg = {"서울": {"2009": (0, 100000), "2011": (100, 1000)},
              "경기": {"2009": (0, 100000), "2011": (50, 1000)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        self.assertNotIn("2009", payload["years"])
        self.assertEqual(payload["years"], ["2011"])

    def test_excluded_years_recorded_in_payload(self) -> None:
        payload = build_progression.build_payload({}, generated="2026-07-28")
        self.assertEqual(payload["excluded_years"], ["2009", "2010"])
        self.assertIn("공시되지 않아", payload["excluded_note"])

    def test_denominator_series_exposed_for_cohort_size(self) -> None:
        agg = {"서울": {"2011": (100, 2000), "2025": (50, 1000)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        self.assertEqual(payload["regions"]["서울"]["den"], [2000, 1000])

    def test_missing_region_year_defaults_to_zero(self) -> None:
        # 경기 자료가 그 해에 아예 없어도(다른 해만 있어도) years 목록에는
        # 서울 기준으로 그 해가 남을 수 있다 — 그 때 경기 쪽은 0/0(=rate None)
        # 으로 채워져야지 KeyError 로 죽으면 안 된다.
        agg = {"서울": {"2011": (10, 100)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        self.assertEqual(payload["regions"]["경기"]["num"], [0])
        self.assertEqual(payload["regions"]["경기"]["den"], [0])
        self.assertIsNone(payload["regions"]["경기"]["rate"][0])


class TestWriteJsonIntegration(unittest.TestCase):
    def test_output_is_deterministic_across_two_runs(self) -> None:
        agg = {"서울": {"2011": (10, 100)}, "경기": {"2011": (5, 100)}}
        payload = build_progression.build_payload(agg, generated="2026-07-28")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "progression.json"
            self.assertTrue(build_progression.write_json(p, payload))
            first = p.read_bytes()
            self.assertFalse(build_progression.write_json(p, payload))
            self.assertEqual(first, p.read_bytes())


if __name__ == "__main__":
    unittest.main()
