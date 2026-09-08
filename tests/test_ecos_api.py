"""한국은행 ECOS 통계 응답 파싱 검증.

ECOS 는 오류를 **HTTP 200 과 함께** `{"RESULT": {"CODE": …}}` 로 돌려준다.
상태코드만 보면 실패를 놓친다. 그걸 여기서 잡는다.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ecos_api  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _payload(points: list[tuple[str, str]]) -> dict:
    return {"StatisticSearch": {
        "list_total_count": len(points),
        "row": [{"TIME": t, "DATA_VALUE": v, "UNIT_NAME": "연%"} for t, v in points]}}


class TestParseSeries(unittest.TestCase):
    def test_base_rate_fixture(self) -> None:
        series, err = ecos_api.parse_series(_fixture("ecos_base_rate.json"))
        self.assertIsNone(err)
        self.assertEqual(series[0], ("2011-01", 2.75))
        self.assertEqual(len(series), 10)

    def test_mortgage_rate_fixture(self) -> None:
        series, err = ecos_api.parse_series(_fixture("ecos_mortgage_rate.json"))
        self.assertIsNone(err)
        self.assertEqual(series[0], ("2011-01", 4.8))

    def test_time_is_reformatted_to_dashed_month(self) -> None:
        series, _ = ecos_api.parse_series(_payload([("202601", "3.5")]))
        self.assertEqual(series, [("2026-01", 3.5)])

    def test_series_is_sorted_by_month(self) -> None:
        series, _ = ecos_api.parse_series(_payload([("201103", "3.0"), ("201101", "2.75")]))
        self.assertEqual([m for m, _ in series], ["2011-01", "2011-03"])


class TestErrorResponses(unittest.TestCase):
    def test_error_payload_is_not_read_as_data(self) -> None:
        """인증키 오류가 HTTP 200 으로 온다. 데이터로 오독하면 빈 시계열이 된다."""
        series, err = ecos_api.parse_series(_fixture("ecos_error.json"))
        self.assertEqual(series, [])
        self.assertIsNotNone(err)
        self.assertIn("INFO-100", err)

    def test_sample_key_limit_error(self) -> None:
        payload = {"RESULT": {"CODE": "ERROR-301", "MESSAGE": "sample 은 최대 10건"}}
        series, err = ecos_api.parse_series(payload)
        self.assertEqual(series, [])
        self.assertIn("ERROR-301", err)

    def test_missing_container_is_an_error(self) -> None:
        series, err = ecos_api.parse_series({})
        self.assertEqual(series, [])
        self.assertIsNotNone(err)

    def test_non_dict_is_an_error(self) -> None:
        series, err = ecos_api.parse_series("nope")
        self.assertEqual(series, [])
        self.assertIsNotNone(err)


class TestBadPoints(unittest.TestCase):
    def test_unparseable_value_drops_only_that_point(self) -> None:
        series, err = ecos_api.parse_series(
            _payload([("201101", "2.75"), ("201102", ""), ("201103", "3.0")]))
        self.assertIsNone(err)
        self.assertEqual([m for m, _ in series], ["2011-01", "2011-03"])

    def test_out_of_range_rate_is_dropped(self) -> None:
        """금리가 30% 를 넘거나 음수면 응답이 이상한 것이다."""
        series, _ = ecos_api.parse_series(
            _payload([("201101", "2.75"), ("201102", "31.0"), ("201103", "-1.0")]))
        self.assertEqual([m for m, _ in series], ["2011-01"])

    def test_zero_is_kept(self) -> None:
        """0% 는 있을 수 있는 값이다. 범위 밖과 섞지 않는다."""
        series, _ = ecos_api.parse_series(_payload([("202001", "0.0")]))
        self.assertEqual(series, [("2020-01", 0.0)])

    def test_bad_time_drops_only_that_point(self) -> None:
        series, _ = ecos_api.parse_series(
            _payload([("2011", "2.75"), ("201102", "3.0")]))
        self.assertEqual(series, [("2011-02", 3.0)])


if __name__ == "__main__":
    unittest.main()
