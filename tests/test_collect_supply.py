"""착공·금리 수집기의 순수 부분 검증 — 60개월 청크와 잠정치 재수집 판단.

통계누리는 61개월 이상을 요청하면 `{"result": false}` 를 **HTTP 200 으로** 준다.
청크를 한 달이라도 잘못 자르면 조용히 빈 구간이 생긴다.

잠정치는 확정되면서 값이 바뀐다. 캐시가 잠정월을 잡아먹으면 영원히 옛 값을 쓴다.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_supply  # noqa: E402


class TestMonthChunks(unittest.TestCase):
    def test_chunks_never_exceed_the_limit(self) -> None:
        for start, end in collect_supply.month_chunks("201101", "202609", 60):
            span = collect_supply.months_between(start, end)
            self.assertLessEqual(len(span), 60, f"{start}~{end}")

    def test_chunks_cover_the_whole_range_without_gaps_or_overlap(self) -> None:
        chunks = collect_supply.month_chunks("201101", "202609", 60)
        seen: list[str] = []
        for start, end in chunks:
            seen.extend(collect_supply.months_between(start, end))
        self.assertEqual(seen, collect_supply.months_between("201101", "202609"))

    def test_exactly_one_chunk_when_range_fits(self) -> None:
        self.assertEqual(collect_supply.month_chunks("202101", "202512", 60),
                         [("202101", "202512")])

    def test_one_extra_month_forces_a_second_chunk(self) -> None:
        self.assertEqual(collect_supply.month_chunks("202101", "202601", 60),
                         [("202101", "202512"), ("202601", "202601")])

    def test_single_month_range(self) -> None:
        self.assertEqual(collect_supply.month_chunks("202609", "202609", 60),
                         [("202609", "202609")])

    def test_end_before_start_yields_nothing(self) -> None:
        self.assertEqual(collect_supply.month_chunks("202609", "202601", 60), [])


class TestMonthsBetween(unittest.TestCase):
    def test_crosses_the_year_boundary(self) -> None:
        self.assertEqual(collect_supply.months_between("201111", "201202"),
                         ["2011-11", "2011-12", "2012-01", "2012-02"])

    def test_single_month(self) -> None:
        self.assertEqual(collect_supply.months_between("202609", "202609"), ["2026-09"])


class TestChunkNeedsRefetch(unittest.TestCase):
    def test_chunk_of_only_confirmed_months_is_skipped(self) -> None:
        confirmed = {"2011-01", "2011-02", "2011-03"}
        self.assertFalse(
            collect_supply.chunk_needs_refetch("201101", "201103", confirmed))

    def test_one_provisional_month_forces_the_whole_chunk(self) -> None:
        """잠정월은 값이 바뀐다. 캐시가 잡아먹으면 영원히 옛 값을 쓴다."""
        confirmed = {"2011-01", "2011-02"}
        self.assertTrue(
            collect_supply.chunk_needs_refetch("201101", "201103", confirmed))

    def test_unseen_month_forces_a_fetch(self) -> None:
        self.assertTrue(collect_supply.chunk_needs_refetch("202609", "202609", set()))


if __name__ == "__main__":
    unittest.main()
