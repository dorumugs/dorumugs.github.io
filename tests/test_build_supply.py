"""착공 × 금리 집계 검증.

여기서 잡는 것: 광주·전남 통합 합산, 12개월 이동합계의 결측 전파,
평년 기준선, 유효 개월 문턱, 그리고 **`총계` 대조 실패 시 빌드 실패**.

마지막 하나가 이 파일의 핵심이다 — 전남광주 같은 통합이 또 일어나면 새 라벨이
화이트리스트에 없어 조용히 누락되는데, 지역 개수를 세는 검사로는 못 잡는다.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_supply  # noqa: E402
from molit_stat_api import Row  # noqa: E402


def _months(start_year: int, count: int) -> list[str]:
    out = []
    year, month = start_year, 1
    for _ in range(count):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


class TestRollingSum(unittest.TestCase):
    def test_first_eleven_points_are_undefined(self) -> None:
        got = build_supply.rolling_sum([1] * 13, 12)
        self.assertEqual(got[:11], [None] * 11)

    def test_twelfth_point_is_the_sum_of_the_window(self) -> None:
        got = build_supply.rolling_sum(list(range(1, 14)), 12)
        self.assertEqual(got[11], sum(range(1, 13)))
        self.assertEqual(got[12], sum(range(2, 14)))

    def test_missing_inside_the_window_poisons_the_point(self) -> None:
        """창 안에 결측이 하나라도 있으면 그 지점은 결측이다. 0 으로 때우지 않는다."""
        values = [1] * 24
        values[5] = None
        got = build_supply.rolling_sum(values, 12)
        self.assertIsNone(got[11])
        self.assertIsNone(got[16])   # 창이 5번을 아직 물고 있다
        self.assertEqual(got[17], 12)  # 창이 5번을 지나갔다

    def test_zero_is_not_missing(self) -> None:
        values = [0] + [1] * 11
        self.assertEqual(build_supply.rolling_sum(values, 12)[11], 11)


class TestUnification(unittest.TestCase):
    def test_both_present_are_summed(self) -> None:
        rows = [Row("2026-06", "광주", 569, True), Row("2026-06", "전남", 677, True)]
        got = build_supply.merge_unification(rows)
        self.assertEqual(got[("2026-06", "전남광주")], 1246)
        self.assertNotIn(("2026-06", "광주"), got)
        self.assertNotIn(("2026-06", "전남"), got)

    def test_unified_label_passes_through(self) -> None:
        rows = [Row("2026-07", "전남광주", 638, True)]
        got = build_supply.merge_unification(rows)
        self.assertEqual(got[("2026-07", "전남광주")], 638)

    def test_half_present_is_missing_not_half_the_sum(self) -> None:
        """반쪽만 더하면 통합 이전 구간을 조용히 과소계상한다."""
        rows = [Row("2026-06", "광주", 569, True), Row("2026-06", "전남", None, True)]
        got = build_supply.merge_unification(rows)
        self.assertIsNone(got[("2026-06", "전남광주")])

    def test_zero_and_value_still_sum(self) -> None:
        rows = [Row("2026-05", "광주", 0, True), Row("2026-05", "전남", 1031, True)]
        got = build_supply.merge_unification(rows)
        self.assertEqual(got[("2026-05", "전남광주")], 1031)

    def test_other_regions_untouched(self) -> None:
        rows = [Row("2011-01", "서울", 103, False)]
        got = build_supply.merge_unification(rows)
        self.assertEqual(got[("2011-01", "서울")], 103)


class TestFillObservedGaps(unittest.TestCase):
    """관측이 시작된 뒤의 `'-'` 는 0 이다.

    근거는 추측이 아니라 합계 대조다 — `'-'` 를 빼고 더한 시도 합이 187개월
    **전부** `총계` 와 맞는다. 즉 그 달의 그 시도는 실제로 착공이 0 이었다.
    관측 시작 전(세종의 2011~2012)은 다르다. 그때는 시도가 없었으므로 0 이
    아니라 결측이고, 0 으로 깔면 평년 기준선이 바닥으로 내려가 지수가 폭주한다.
    """

    def test_gap_after_the_first_observation_becomes_zero(self) -> None:
        self.assertEqual(build_supply.fill_observed_gaps([None, 5, None, 7]),
                         [None, 5, 0, 7])

    def test_leading_gap_stays_missing(self) -> None:
        got = build_supply.fill_observed_gaps([None, None, 5])
        self.assertEqual(got[:2], [None, None])

    def test_all_missing_stays_all_missing(self) -> None:
        self.assertEqual(build_supply.fill_observed_gaps([None, None]), [None, None])

    def test_a_leading_zero_counts_as_an_observation(self) -> None:
        """0 은 관측값이다. 그 뒤의 결측은 채운다."""
        self.assertEqual(build_supply.fill_observed_gaps([0, None]), [0, 0])

    def test_negative_revision_counts_as_an_observation(self) -> None:
        self.assertEqual(build_supply.fill_observed_gaps([-12, None]), [-12, 0])


class TestBaselineIndex(unittest.TestCase):
    def test_baseline_average_becomes_one_hundred(self) -> None:
        months = _months(2011, 120)
        mavg = [None] * 11 + [100.0] * 109
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertEqual(index[11], 100.0)

    def test_double_the_baseline_is_two_hundred(self) -> None:
        months = _months(2011, 120)
        mavg = [None] * 11 + [100.0] * 97 + [200.0] * 12
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertEqual(index[-1], 200.0)

    def test_months_outside_the_window_do_not_move_the_baseline(self) -> None:
        months = _months(2011, 120)
        mavg = [None] * 11 + [100.0] * 97 + [9999.0] * 12
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertEqual(index[11], 100.0)

    def test_too_few_valid_months_yields_no_index_at_all(self) -> None:
        """짧은 기준선으로 만든 지수는 숫자만 그럴듯하고 뜻이 없다."""
        months = _months(2011, 120)
        mavg = [None] * 70 + [100.0] * 50
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertEqual(index, [None] * 120)

    def test_missing_points_stay_missing(self) -> None:
        months = _months(2011, 120)
        mavg = [None] * 11 + [100.0] * 109
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertIsNone(index[0])

    def test_zero_baseline_yields_no_index(self) -> None:
        """기준이 0 이면 나눌 수 없다. 무한대를 그리지 않는다."""
        months = _months(2011, 120)
        mavg = [None] * 11 + [0.0] * 109
        index = build_supply.baseline_index(mavg, months, "2011-12", "2019-12", 60)
        self.assertEqual(index, [None] * 120)


class TestCrossCheckTotal(unittest.TestCase):
    def _rows(self, *, drop: str | None = None) -> list[Row]:
        parts = {"서울": 100, "경기": 200, "부산": 300}
        rows = [Row("2011-01", r, v, False) for r, v in parts.items() if r != drop]
        rows.append(Row("2011-01", "총계", sum(parts.values()), False))
        return rows

    def test_matching_total_passes(self) -> None:
        self.assertIsNone(build_supply.cross_check_total(self._rows()))

    def test_missing_region_fails_the_build(self) -> None:
        """시도 하나가 화이트리스트에서 빠지면 합계가 1% 넘게 어긋난다."""
        problem = build_supply.cross_check_total(self._rows(drop="부산"))
        self.assertIsNotNone(problem)
        self.assertIn("2011-01", problem)

    def test_tiny_difference_is_tolerated(self) -> None:
        rows = [Row("2011-01", "서울", 1000, False), Row("2011-01", "총계", 1005, False)]
        self.assertIsNone(build_supply.cross_check_total(rows))

    def test_month_without_a_total_row_is_skipped(self) -> None:
        rows = [Row("2011-01", "서울", 100, False)]
        self.assertIsNone(build_supply.cross_check_total(rows))

    def test_missing_value_is_treated_as_absent_and_still_compared(self) -> None:
        """세종의 `'-'` 는 실제로 0 이라 합계가 맞는다. 건너뛰면 검사가 약해진다."""
        rows = [Row("2011-01", "세종", None, False), Row("2011-01", "서울", 300, False),
                Row("2011-01", "총계", 300, False)]
        self.assertIsNone(build_supply.cross_check_total(rows))

    def test_value_hidden_behind_a_missing_marker_fails(self) -> None:
        rows = [Row("2011-01", "세종", None, False), Row("2011-01", "서울", 300, False),
                Row("2011-01", "총계", 800, False)]
        self.assertIsNotNone(build_supply.cross_check_total(rows))


class TestBuild(unittest.TestCase):
    def _starts(self) -> list[Row]:
        months = _months(2011, 120)
        rows: list[Row] = []
        for month in months:
            per = {"서울": 100, "경기": 200, "광주": 50, "전남": 70}
            for region, value in per.items():
                rows.append(Row(month, region, value, False))
            rows.append(Row(month, "총계", sum(per.values()), False))
        return rows

    def _rates(self) -> dict:
        months = _months(2011, 120)
        return {"base": [(m, 2.0) for m in months],
                "mortgage": [(m, 4.0) for m in months]}

    def test_nationwide_series_comes_first_and_is_the_sum_of_regions(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        self.assertEqual(out["regions"][0]["code"], "00")
        self.assertEqual(out["regions"][0]["name"], "전국")
        self.assertEqual(out["regions"][0]["units"][0], 420)

    def test_gwangju_and_jeonnam_appear_only_as_one_merged_region(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        names = [r["name"] for r in out["regions"]]
        self.assertIn("전남광주", names)
        self.assertNotIn("광주", names)
        self.assertNotIn("전남", names)

    def test_total_row_is_not_published_as_a_region(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        self.assertNotIn("총계", [r["name"] for r in out["regions"]])

    def test_every_series_matches_the_month_axis_length(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        n = len(out["months"])
        for region in out["regions"]:
            for key in ("units", "mavg", "index"):
                self.assertEqual(len(region[key]), n, f"{region['name']}.{key}")
        self.assertEqual(len(out["rates"]["base"]), n)

    def test_rates_align_to_the_month_axis(self) -> None:
        rates = {"base": [("2011-05", 3.0)], "mortgage": []}
        out = build_supply.build(self._starts(), rates, "2026-09-07")
        idx = out["months"].index("2011-05")
        self.assertEqual(out["rates"]["base"][idx], 3.0)
        self.assertIsNone(out["rates"]["base"][0])
        self.assertEqual(out["rates"]["mortgage"], [None] * len(out["months"]))

    def test_provisional_start_is_reported(self) -> None:
        rows = self._starts()
        rows = [r._replace(provisional=True) if r.month >= "2020-01" else r for r in rows]
        out = build_supply.build(rows, self._rates(), "2026-09-07")
        self.assertEqual(out["provisional_from"], "2020-01")

    def test_no_provisional_months_reports_none(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        self.assertIsNone(out["provisional_from"])

    def test_mismatched_total_raises(self) -> None:
        rows = [r for r in self._starts() if r.region != "경기"]
        with self.assertRaises(build_supply.BuildError):
            build_supply.build(rows, self._rates(), "2026-09-07")

    def test_month_axis_has_no_gaps(self) -> None:
        rows = [r for r in self._starts() if r.month != "2015-06"]
        out = build_supply.build(rows, self._rates(), "2026-09-07")
        self.assertIn("2015-06", out["months"])
        idx = out["months"].index("2015-06")
        self.assertIsNone(out["regions"][0]["units"][idx])

    def test_scattered_gaps_no_longer_erase_the_index(self) -> None:
        """대전처럼 `'-'` 가 흩뿌려진 시도도 지수를 얻는다. 결측 하나가 12개
        지점을 죽이므로, 채우지 않으면 절반의 시도가 지수를 잃는다."""
        rows = [r for r in self._starts()
                if not (r.region == "서울" and r.month.endswith("-02"))]
        rows += [Row(m, "서울", None, False) for m in _months(2011, 120)
                 if m.endswith("-02")]
        rows = [r for r in rows if not (r.region == "총계")]
        rows += [Row(m, "총계", 320 if m.endswith("-02") else 420, False)
                 for m in _months(2011, 120)]
        out = build_supply.build(rows, self._rates(), "2026-09-07")
        seoul = next(r for r in out["regions"] if r["name"] == "서울")
        self.assertIsNotNone(seoul["index"][-1])

    def test_region_that_starts_late_keeps_its_leading_gap(self) -> None:
        """세종은 2012-10 부터다. 그 앞을 0 으로 깔면 기준선이 바닥이 된다."""
        rows = self._starts()
        late = [Row(m, "세종", None if m < "2013-01" else 10, False)
                for m in _months(2011, 120)]
        rows = [r for r in rows if r.region != "총계"] + late
        rows += [Row(m, "총계", 420 + (0 if m < "2013-01" else 10), False)
                 for m in _months(2011, 120)]
        out = build_supply.build(rows, self._rates(), "2026-09-07")
        sejong = next(r for r in out["regions"] if r["name"] == "세종")
        self.assertIsNone(sejong["units"][0])
        self.assertIsNone(sejong["mavg"][11])

    def test_latest_month_is_the_last_month(self) -> None:
        out = build_supply.build(self._starts(), self._rates(), "2026-09-07")
        self.assertEqual(out["latest_month"], out["months"][-1])


if __name__ == "__main__":
    unittest.main()
