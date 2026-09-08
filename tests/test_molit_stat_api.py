"""통계누리 주택유형별 착공실적(formId 5387) 파싱 검증.

픽스처는 실제 응답이다. 함정 다섯 개를 여기서 잡는다 —
60개월 초과 오류, 잠정치 `p)`, 집계행 혼입, 전남광주 통합, `'-'` 결측.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import molit_stat_api  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _rows(name: str) -> list:
    rows, err = molit_stat_api.parse_starts(_fixture(name))
    assert err is None, err
    return rows


def _at(rows: list, month: str, region: str):
    got = [r for r in rows if r.month == month and r.region == region]
    return got[0] if got else None


class TestErrorResponses(unittest.TestCase):
    def test_60_month_limit_is_an_error_not_empty_data(self) -> None:
        """`result: false` 가 HTTP 200 으로 온다. 빈 데이터로 읽으면 조용히 망한다."""
        rows, err = molit_stat_api.parse_starts(_fixture("molit_starts_toolong.json"))
        self.assertEqual(rows, [])
        self.assertIsNotNone(err)
        self.assertIn("60개월", err)

    def test_unexpected_shape_is_an_error(self) -> None:
        rows, err = molit_stat_api.parse_starts({"result": True, "data": "nope"})
        self.assertEqual(rows, [])
        self.assertIsNotNone(err)

    def test_non_dict_is_an_error(self) -> None:
        rows, err = molit_stat_api.parse_starts([])
        self.assertEqual(rows, [])
        self.assertIsNotNone(err)


class TestProvisional(unittest.TestCase):
    def test_p_suffix_is_stripped_into_a_flag(self) -> None:
        row = _at(_rows("molit_starts_2026.json"), "2026-07", "전남광주")
        self.assertIsNotNone(row)
        self.assertTrue(row.provisional)

    def test_confirmed_month_has_no_flag(self) -> None:
        row = _at(_rows("molit_starts_2011.json"), "2011-01", "서울")
        self.assertIsNotNone(row)
        self.assertFalse(row.provisional)
        self.assertEqual(row.units, 103)


class TestAggregateRows(unittest.TestCase):
    def test_subtotal_rows_are_dropped(self) -> None:
        """수도권소계·지방소계·기타광역시·기타지방은 이중계상이다."""
        regions = {r.region for r in _rows("molit_starts_2026.json")}
        for label in ("수도권소계", "지방소계", "기타광역시", "기타지방"):
            self.assertNotIn(label, regions)

    def test_unknown_label_is_dropped(self) -> None:
        """화이트리스트에 없으면 버린다 — 새 합계행이 생겨도 조용히 안 섞인다."""
        payload = {"result": True, "data": [
            {"0": "2026-07 p)", "1": "남부권소계", "2": "아파트",
             "3": "아파트", "4": "아파트", "5": "999"}]}
        rows, err = molit_stat_api.parse_starts(payload)
        self.assertIsNone(err)
        self.assertEqual(rows, [])

    def test_total_row_is_kept_for_cross_check(self) -> None:
        """`총계` 는 시도가 아니지만 build 의 합계 대조에 쓰므로 남긴다."""
        row = _at(_rows("molit_starts_2026.json"), "2026-07", "총계")
        self.assertIsNotNone(row)
        self.assertEqual(row.units, 18047)


class TestCategoryFilter(unittest.TestCase):
    def test_only_apartment_rows_survive(self) -> None:
        payload = {"result": True, "data": [
            {"0": "2011-01", "1": "서울", "2": "단독", "3": "단독", "4": "단독", "5": "111"},
            {"0": "2011-01", "1": "서울", "2": "연립", "3": "연립", "4": "연립", "5": "222"},
            {"0": "2011-01", "1": "서울", "2": "아파트", "3": "아파트", "4": "아파트", "5": "103"}]}
        rows, err = molit_stat_api.parse_starts(payload)
        self.assertIsNone(err)
        self.assertEqual([(r.region, r.units) for r in rows], [("서울", 103)])


class TestMissingValues(unittest.TestCase):
    def test_dash_is_missing_not_zero(self) -> None:
        """세종은 2011-01 에 행이 있지만 값이 `'-'` 다. 0 으로 읽으면 지수가 폭주한다."""
        row = _at(_rows("molit_starts_2011.json"), "2011-01", "세종")
        self.assertIsNotNone(row)
        self.assertIsNone(row.units)

    def test_zero_is_a_real_observation(self) -> None:
        row = _at(_rows("molit_starts_2026.json"), "2026-05", "세종")
        self.assertIsNotNone(row)
        self.assertEqual(row.units, 0)

    def test_blank_and_garbage_are_missing(self) -> None:
        payload = {"result": True, "data": [
            {"0": "2011-01", "1": "서울", "2": "아파트", "3": "", "4": "", "5": ""},
            {"0": "2011-02", "1": "서울", "2": "아파트", "3": "", "4": "", "5": "n/a"}]}
        rows, err = molit_stat_api.parse_starts(payload)
        self.assertIsNone(err)
        self.assertEqual([r.units for r in rows], [None, None])

    def test_negative_is_a_real_downward_revision(self) -> None:
        """2011-12 충남이 실제로 `-1494` 다. 결측으로 버리면 총계와 어긋난다 —
        합계 대조가 이걸 잡아냈다. 12개월 이동합계에서 앞달의 과대계상을 상쇄한다."""
        row = _at(_rows("molit_starts_revision.json"), "2011-12", "충남")
        self.assertEqual(row.units, -1494)

    def test_negative_month_still_reconciles_with_the_total(self) -> None:
        rows = _rows("molit_starts_revision.json")
        sido = sum(r.units for r in rows
                   if r.region != "총계" and r.units is not None)
        total = _at(rows, "2011-12", "총계").units
        self.assertEqual(sido, total)


class TestUnificationLabel(unittest.TestCase):
    def test_gwangju_and_jeonnam_before_unification(self) -> None:
        rows = _rows("molit_starts_2026.json")
        self.assertEqual(_at(rows, "2026-06", "광주").units, 569)
        self.assertEqual(_at(rows, "2026-06", "전남").units, 677)
        self.assertIsNone(_at(rows, "2026-06", "전남광주"))

    def test_unified_label_after(self) -> None:
        rows = _rows("molit_starts_2026.json")
        self.assertEqual(_at(rows, "2026-07", "전남광주").units, 638)
        self.assertIsNone(_at(rows, "2026-07", "광주"))
        self.assertIsNone(_at(rows, "2026-07", "전남"))

    def test_parser_does_not_merge_them(self) -> None:
        """합산은 build 의 일이다. 파서는 원본에 충실하다."""
        rows = _rows("molit_starts_2026.json")
        self.assertIsNotNone(_at(rows, "2026-05", "광주"))
        self.assertIsNone(_at(rows, "2026-05", "전남광주"))


class TestSanityGate(unittest.TestCase):
    def _month(self, total: int) -> dict:
        return {"result": True, "data": [
            {"0": "2011-01", "1": "서울", "2": "아파트", "3": "", "4": "", "5": str(total)},
            {"0": "2011-02", "1": "서울", "2": "아파트", "3": "", "4": "", "5": "103"}]}

    def test_absurdly_large_month_is_dropped_whole(self) -> None:
        rows, err = molit_stat_api.parse_starts(self._month(200001))
        self.assertIsNone(err)
        self.assertEqual([r.month for r in rows], ["2011-02"])

    def test_all_zero_month_is_dropped_whole(self) -> None:
        rows, err = molit_stat_api.parse_starts(self._month(0))
        self.assertIsNone(err)
        self.assertEqual([r.month for r in rows], ["2011-02"])

    def test_real_months_pass_the_gate(self) -> None:
        months = {r.month for r in _rows("molit_starts_2026.json")}
        self.assertEqual(months, {"2026-05", "2026-06", "2026-07"})


if __name__ == "__main__":
    unittest.main()
