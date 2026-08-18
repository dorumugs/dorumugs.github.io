"""미국 ETF 집계 테스트.

국내와 다른 두 가지만 여기서 본다 — 영어 이름에서 배수를 읽는 것과,
원화 환산에 환율을 맞춰 붙이는 것.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import build_us_etf as u  # noqa: E402
import naver_us_api as us_api  # noqa: E402


class LeverageTest(unittest.TestCase):
    def test_영어_이름에서_배수를_읽는다(self):
        cases = {
            "Direxion Daily Semiconductor Bull 3X ETF": 3.0,
            "Direxion Daily Semiconductor Bear 3X ETF": -3.0,
            "ProShares UltraPro QQQ": 3.0,
            "ProShares UltraPro Short QQQ": -3.0,
            "ProShares Ultra QQQ": 2.0,
            "ProShares UltraShort S&P500": -2.0,
            "Direxion Daily AAPL Bear 1X ETF": -1.0,
            "Invesco QQQ Trust  Series 1": 1.0,
            "SPDR S&P 500 ETF Trust": 1.0,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(u.leverage_of(name), expected)

    def test_미국은_3배까지_있다(self):
        # 국내는 최대 2배다. 3배에 1배 잣대를 대면 늘 '과열' 로 찍힌다.
        self.assertEqual(u.leverage_of("ProShares UltraPro S&P500"), 3.0)


class FxReturnTest(unittest.TestCase):
    def test_보유_기간_환율_변동을_잰다(self):
        fx = {"20260101": 1000.0, "20260201": 1100.0}
        dates = ["20260101", "20260115", "20260201"]
        self.assertAlmostEqual(u.fx_return(fx, dates, 2), 0.10)

    def test_고시가_없는_날은_직전_값을_쓴다(self):
        # 한국과 미국의 공휴일이 어긋나서 날짜가 딱 안 맞는 일이 흔하다.
        fx = {"20260101": 1000.0, "20260130": 1100.0}
        dates = ["20260101", "20260115", "20260201"]
        self.assertAlmostEqual(u.fx_return(fx, dates, 2), 0.10)

    def test_환율이_없으면_None이다(self):
        self.assertIsNone(u.fx_return({}, ["20260101", "20260201"], 1))

    def test_거래일이_모자라면_None이다(self):
        fx = {"20260101": 1000.0}
        self.assertIsNone(u.fx_return(fx, ["20260101"], 5))


if __name__ == "__main__":
    unittest.main()


class UsTickerTest(unittest.TestCase):
    """구성종목 티커를 네이버 표기로 맞추기."""

    def test_클래스_구분자를_점으로_바꾼다(self):
        # 발행사는 'BRK/B' 로 주는데 네이버는 'BRK.B' 로 찾는다. 안 바꾸면
        # 자동완성이 못 찾아 그 종목이 통째로 폭 계산에서 빠진다.
        self.assertEqual(us_api.normalize_ticker("BRK/B"), "BRK.B")
        self.assertEqual(us_api.normalize_ticker("HEI/A"), "HEI.A")

    def test_평범한_티커는_그대로다(self):
        for t in ("AAPL", "NVDA", "LLY", "FXI"):
            self.assertEqual(us_api.normalize_ticker(t), t)

    def test_공백과_소문자를_정리한다(self):
        self.assertEqual(us_api.normalize_ticker("  aapl "), "AAPL")

    def test_빈_값은_빈_문자열이다(self):
        self.assertEqual(us_api.normalize_ticker(""), "")
        self.assertEqual(us_api.normalize_ticker(None), "")


class Lev3BreadthTest(unittest.TestCase):
    """3배 ETF 의 폭을 어떤 종목으로 재는가."""

    def test_불_3배만_고른다(self):
        rows = [
            {"ticker": "SOXL", "lev": 3.0}, {"ticker": "SOXS", "lev": -3.0},
            {"ticker": "TQQQ", "lev": 3.0}, {"ticker": "QQQ", "lev": 1.0},
            {"ticker": "QLD", "lev": 2.0},
        ]
        self.assertEqual(sorted(u.lev3_bull_tickers(rows)), ["SOXL", "TQQQ"])

    def test_인버스는_뺀다(self):
        # 스왑 구조라 구성종목이 0개고, '구성종목 70% 상승' 은 인버스에서
        # ETF 가 내린다는 뜻이라 의미가 뒤집힌다.
        self.assertEqual(u.lev3_bull_tickers([{"ticker": "SQQQ", "lev": -3.0}]), [])

    def test_보유_티커를_정규화해_모은다(self):
        holdings = {"SOXL": {"rows": [{"ticker": "NVDA"}, {"ticker": "BRK/B"}, {"name": "CASH"}]}}
        self.assertEqual(u.holding_tickers(holdings, ["SOXL"]), ["BRK.B", "NVDA"])

    def test_구성종목이_없으면_빈_목록이다(self):
        self.assertEqual(u.holding_tickers({"SQQQ": {"rows": []}}, ["SQQQ"]), [])

    def test_티커가_아닌_내부_식별자를_버린다(self):
        # 발행사 파일의 티커 칸에 '2200963' 같은 숫자가 섞여 온다. 안 버리면
        # 매일 '못 찾음' 에 쌓여 진짜 파손 신호를 덮는다.
        holdings = {"SPXL": {"rows": [{"ticker": "AAPL"}, {"ticker": "2200963"}]}}
        self.assertEqual(u.holding_tickers(holdings, ["SPXL"]), ["AAPL"])
