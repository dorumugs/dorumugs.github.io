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
