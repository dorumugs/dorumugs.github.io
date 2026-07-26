"""실거래가 수집기 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_trades  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestParseResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.xml = (FIXTURES / "apt_trade_sample.xml").read_text(encoding="utf-8")

    def test_parses_real_sample(self) -> None:
        rows, total = rtms.parse_response(self.xml)
        self.assertEqual(total, 211)
        self.assertTrue(rows)
        first = rows[0]
        self.assertEqual(first["apt_name"], "개나리푸르지오")
        self.assertEqual(first["sgg_cd"], "11680")
        self.assertEqual(first["umd_nm"], "역삼동")
        self.assertEqual(first["jibun"], "755-1")
        self.assertEqual(first["price_10k"], "290000")  # '290,000' 의 쉼표 제거
        self.assertEqual(first["trade_date"], "2026-06-27")

    def test_every_column_present(self) -> None:
        rows, _ = rtms.parse_response(self.xml)
        for row in rows:
            self.assertEqual(set(row), set(rtms.COLUMNS))

    def test_zero_pads_single_digit_month_and_day(self) -> None:
        xml = _wrap('<dealAmount>1,000</dealAmount><dealYear>2026</dealYear>'
                    '<dealMonth>6</dealMonth><dealDay>7</dealDay>')
        rows, _ = rtms.parse_response(xml)
        self.assertEqual(rows[0]["trade_date"], "2026-06-07")

    def test_keeps_cancelled_deals_as_flag(self) -> None:
        """MCP 파서는 계약해제 건을 버리지만 여기서는 지표로 쓰려고 남긴다."""
        xml = _wrap('<dealAmount>1,000</dealAmount><dealYear>2026</dealYear>'
                    '<dealMonth>6</dealMonth><dealDay>7</dealDay>'
                    '<cdealType>O</cdealType><cdealDay>26.07.01</cdealDay>')
        rows, _ = rtms.parse_response(xml)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cdeal_type"], "O")

    def test_skips_row_with_unparseable_amount(self) -> None:
        xml = _wrap('<dealAmount>-</dealAmount><dealYear>2026</dealYear>'
                    '<dealMonth>6</dealMonth><dealDay>7</dealDay>')
        rows, _ = rtms.parse_response(xml)
        self.assertEqual(rows, [])

    def test_raises_on_error_result_code(self) -> None:
        xml = ('<response><header><resultCode>04</resultCode>'
               '<resultMsg>HTTP ROUTING ERROR</resultMsg></header></response>')
        with self.assertRaises(rtms.ApiError) as ctx:
            rtms.parse_response(xml)
        self.assertEqual(ctx.exception.code, "04")
        self.assertFalse(ctx.exception.is_limit)

    def test_flags_quota_exceeded(self) -> None:
        for code in ("22", "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"):
            xml = (f'<response><header><resultCode>{code}</resultCode>'
                   '<resultMsg>quota</resultMsg></header></response>')
            with self.assertRaises(rtms.ApiError) as ctx:
                rtms.parse_response(xml)
            self.assertTrue(ctx.exception.is_limit, code)


def _wrap(item_body: str) -> str:
    return (
        "<response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header>"
        f"<body><totalCount>1</totalCount><items><item>{item_body}</item></items></body></response>"
    )


class TestPnu(unittest.TestCase):
    def test_matches_value_from_complex_api(self) -> None:
        """단지정보 API 가 대치동 1014-3 에 대해 실제로 돌려준 PNU."""
        self.assertEqual(
            regions.make_pnu("11680", "대치동", "1014-3"), "1168010600110140003"
        )

    def test_pads_bonbeon_and_bubeon(self) -> None:
        self.assertEqual(regions.make_pnu("11680", "대치동", "988"), "1168010600109880000")

    def test_mountain_jibun_uses_ledger_two(self) -> None:
        pnu = regions.make_pnu("11680", "대치동", "산 12")
        self.assertEqual(pnu[10], "2")
        self.assertEqual(pnu, "1168010600200120000")

    def test_mountain_without_space(self) -> None:
        self.assertEqual(regions.parse_jibun("산12-3"), ("2", "0012", "0003"))

    def test_two_token_umd_name_for_eup_myeon_area(self) -> None:
        """읍면 지역은 umdNm 이 '고덕면 궁리' 처럼 두 토큰으로 온다.

        말단 토큰만 키로 잡으면 2006-01 기준 5.6% 가 조인에 실패했다.
        """
        pnu = regions.make_pnu("41220", "고덕면 궁리", "75")
        self.assertIsNotNone(pnu)
        self.assertEqual(len(pnu), 19)
        self.assertTrue(pnu.startswith("41220"))
        self.assertEqual(pnu[10:], "100750000")

    def test_single_token_dong_still_resolves(self) -> None:
        self.assertEqual(regions.dong_code("11680", "역삼동"), "1168010100")

    def test_unknown_dong_returns_none(self) -> None:
        self.assertIsNone(regions.make_pnu("11680", "없는동", "1"))

    def test_malformed_jibun_returns_none(self) -> None:
        for bad in ("", "   ", "1-2-3", "abc", "12345", "1-12345"):
            self.assertIsNone(regions.parse_jibun(bad), bad)


class TestRegionTable(unittest.TestCase):
    def test_seoul_has_25_gu(self) -> None:
        seoul = [c for c, _ in regions.sgg_codes() if c.startswith("11")]
        self.assertEqual(len(seoul), 25)

    def test_excludes_parent_city_codes_of_split_cities(self) -> None:
        """실측 결과 상위 시 코드는 과거치까지 전부 0건이라 조회 대상이 아니다."""
        codes = {c for c, _ in regions.sgg_codes()}
        for parent in ("41110", "41130", "41170", "41190", "41270", "41280", "41460", "41590"):
            self.assertNotIn(parent, codes, parent)
        for child in ("41111", "41135", "41192", "41597"):
            self.assertIn(child, codes, child)

    def test_keeps_undivided_cities_and_counties(self) -> None:
        codes = {c for c, _ in regions.sgg_codes()}
        for solo in ("41210", "41830", "41500"):  # 광명시, 양평군, 이천시
            self.assertIn(solo, codes, solo)

    def test_normalizes_trailing_whitespace_in_names(self) -> None:
        names = dict((c, n) for c, n in regions.sgg_codes())
        self.assertEqual(names["41192"], "경기도 부천시 원미구")


class TestMergeAndSerialize(unittest.TestCase):
    def test_dedupes_and_sorts_deterministically(self) -> None:
        a = _row(umd_nm="역삼동", price_10k="100")
        b = _row(umd_nm="개포동", price_10k="200")
        merged = rtms.merge_rows([a, b], [dict(a)])
        self.assertEqual(len(merged), 2)
        self.assertEqual([r["umd_nm"] for r in merged], ["개포동", "역삼동"])

    def test_later_batch_wins_on_same_key(self) -> None:
        """재수집 시 계약해제 플래그 같은 갱신분이 반영되어야 한다."""
        old = _row(cdeal_type="")
        new = _row(cdeal_type="O")
        merged = rtms.merge_rows([old], [new])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["cdeal_type"], "O")

    def test_csv_roundtrip(self) -> None:
        rows = [_row(umd_nm="역삼동"), _row(umd_nm="개포동")]
        back = rtms.csv_to_rows(rtms.rows_to_csv(rows))
        self.assertEqual(rtms.merge_rows(back), rtms.merge_rows(rows))

    def test_gzip_is_byte_identical_across_runs(self) -> None:
        """mtime 을 고정하지 않으면 매일 재작성마다 git 에 새 blob 이 쌓인다."""
        text = rtms.rows_to_csv([_row()])
        self.assertEqual(rtms.gzip_bytes(text), rtms.gzip_bytes(text))

    def test_gzip_roundtrip(self) -> None:
        text = rtms.rows_to_csv([_row(apt_name="개나리푸르지오")])
        self.assertEqual(rtms.gunzip_text(rtms.gzip_bytes(text)), text)


def _row(**overrides: str) -> dict:
    base = {c: "" for c in rtms.COLUMNS}
    base.update(
        {
            "sgg_cd": "11680",
            "umd_nm": "역삼동",
            "jibun": "755-1",
            "apt_name": "개나리푸르지오",
            "area_sqm": "59.6851",
            "floor": "11",
            "price_10k": "290000",
            "trade_date": "2026-06-27",
        }
    )
    base.update(overrides)
    return base


class TestWorklist(unittest.TestCase):
    MONTHS = ["200601", "200602", "202605", "202606", "202607"]
    SGGS = ["11680", "11110"]

    def test_backfill_goes_oldest_first(self) -> None:
        state = {"done": {}, "failed": {}, "daily": {}}
        work = collect_trades.build_worklist(state, self.MONTHS, self.SGGS, refresh=3)
        self.assertEqual([ym for ym, _ in work], self.MONTHS)

    def test_no_refresh_while_backfill_pending(self) -> None:
        """백필 중에는 최근 3개월 재수집(216콜)에 예산을 쓰지 않는다."""
        state = {"done": {"11680|200601": 1}, "failed": {}, "daily": {}}
        work = collect_trades.build_worklist(state, self.MONTHS, self.SGGS, refresh=3)
        self.assertEqual(len(work), len(self.MONTHS))
        self.assertEqual(work[0], ("200601", ["11110"]))  # 이미 받은 칸은 빠진다

    def test_switches_to_refresh_once_backfill_completes(self) -> None:
        done = {collect_trades.cell(s, m): 1 for s in self.SGGS for m in self.MONTHS}
        state = {"done": done, "failed": {}, "daily": {}}
        work = collect_trades.build_worklist(state, self.MONTHS, self.SGGS, refresh=3)
        self.assertEqual([ym for ym, _ in work], ["202607", "202606", "202605"])
        for _, pending in work:
            self.assertEqual(pending, self.SGGS)  # 갱신은 전 시군구 대상

    def test_refresh_zero_yields_nothing_when_complete(self) -> None:
        done = {collect_trades.cell(s, m): 1 for s in self.SGGS for m in self.MONTHS}
        state = {"done": done, "failed": {}, "daily": {}}
        self.assertEqual(collect_trades.build_worklist(state, self.MONTHS, self.SGGS, 0), [])


class TestBudget(unittest.TestCase):
    def test_take_decrements_until_exhausted(self) -> None:
        budget = collect_trades.Budget(2)
        budget.take()
        budget.take()
        with self.assertRaises(collect_trades.LimitReached):
            budget.take()
        self.assertEqual(budget.used, 2)

    def test_drain_stops_other_workers_immediately(self) -> None:
        """한 워커가 한도 초과를 만나면 나머지도 즉시 멈춰야 한다."""
        budget = collect_trades.Budget(100)
        budget.take()
        budget.drain()
        self.assertEqual(budget.left, 0)
        with self.assertRaises(collect_trades.LimitReached):
            budget.take()

    def test_concurrent_takes_never_oversubscribe(self) -> None:
        import threading

        budget = collect_trades.Budget(50)
        granted = []
        lock = threading.Lock()

        def worker() -> None:
            while True:
                try:
                    budget.take()
                except collect_trades.LimitReached:
                    return
                with lock:
                    granted.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(granted), 50)


class TestQuotaSignal(unittest.TestCase):
    """공공데이터포털은 일일 한도 초과를 resultCode 가 아니라 HTTP 429 로 알린다.

    이걸 일반 오류로 처리하면 한도에 닿고도 멈추지 않는다. 실제로 그 버그 때문에
    8,319칸이 헛돌며 남은 예산을 전부 태웠다.
    """

    def _raise_429(self, *_args, **_kwargs):
        import urllib.error

        raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", {}, None)

    def test_http_429_stops_collection(self) -> None:
        budget = collect_trades.Budget(500)
        with unittest.mock.patch.object(collect_trades, "fetch_page", self._raise_429):
            with self.assertRaises(collect_trades.LimitReached):
                collect_trades.fetch_cell("k", "11680", "202606", budget, 0)

    def test_http_429_drains_budget_so_other_workers_stop(self) -> None:
        budget = collect_trades.Budget(500)
        with unittest.mock.patch.object(collect_trades, "fetch_page", self._raise_429):
            with self.assertRaises(collect_trades.LimitReached):
                collect_trades.fetch_cell("k", "11680", "202606", budget, 0)
        self.assertEqual(budget.left, 0)

    def test_other_http_errors_still_propagate(self) -> None:
        import urllib.error

        def raise_500(*_args, **_kwargs):
            raise urllib.error.HTTPError("http://x", 503, "Server Error", {}, None)

        budget = collect_trades.Budget(500)
        with unittest.mock.patch.object(collect_trades, "fetch_page", raise_500):
            with self.assertRaises(urllib.error.HTTPError):
                collect_trades.fetch_cell("k", "11680", "202606", budget, 0)
        self.assertGreater(budget.left, 0)  # 한도가 아니므로 예산을 비우지 않는다


class TestCallAccounting(unittest.TestCase):
    def test_does_not_double_count_across_month_flushes(self) -> None:
        """월마다 기록하고 마지막에 또 기록하므로 차액만 더해야 한다."""
        state = {"done": {}, "failed": {}, "daily": {}}
        collect_trades._record_calls(state, "2026-07-25", 72, False, "")
        collect_trades._record_calls(state, "2026-07-25", 72, False, "")
        self.assertEqual(state["daily"]["2026-07-25"]["calls"], 144)

    def test_accumulates_across_runs_on_same_day(self) -> None:
        state = {"done": {}, "failed": {}, "daily": {"2026-07-25": {"calls": 900, "limited": False}}}
        collect_trades._record_calls(state, "2026-07-25", 500, True, "quota")
        entry = state["daily"]["2026-07-25"]
        self.assertEqual(entry["calls"], 1400)
        self.assertTrue(entry["limited"])
        self.assertEqual(entry["reason"], "quota")

    def test_limited_flag_is_sticky(self) -> None:
        state = {"done": {}, "failed": {}, "daily": {}}
        collect_trades._record_calls(state, "2026-07-25", 10, True, "quota")
        collect_trades._record_calls(state, "2026-07-25", 10, False, "")
        self.assertTrue(state["daily"]["2026-07-25"]["limited"])


class TestMonthRange(unittest.TestCase):
    def test_starts_at_public_data_boundary(self) -> None:
        from datetime import date

        months = collect_trades.all_months(date(2026, 7, 25))
        self.assertEqual(months[0], "200601")
        self.assertEqual(months[-1], "202607")

    def test_crosses_year_boundary(self) -> None:
        from datetime import date

        months = collect_trades.all_months(date(2006, 12, 1))
        self.assertEqual(months, ["200601", "200602", "200603", "200604", "200605", "200606",
                                  "200607", "200608", "200609", "200610", "200611", "200612"])

    def test_month_path_layout(self) -> None:
        path = collect_trades.month_path("202606")
        self.assertEqual(path.parent.name, "2026")
        self.assertEqual(path.name, "2026-06.csv.gz")


if __name__ == "__main__":
    unittest.main()
