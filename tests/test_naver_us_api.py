"""네이버 해외증시 파싱 회귀 테스트.

국내와 서버가 달라 함정도 다르다. 거래소 접미사, ETF/ETN 구분, 쉼표 박힌
환율 문자열이 여기서 깨지면 미국 화면 전체가 조용히 틀린다.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import naver_us_api as us  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class SeedTest(unittest.TestCase):
    def test_주석과_빈줄을_버린다(self):
        text = "# 설명\nSPY QQQ\n\n  TQQQ  # 꼬리 주석\n"
        self.assertEqual(us.parse_seed(text), ["SPY", "QQQ", "TQQQ"])

    def test_중복을_없애고_순서를_지킨다(self):
        self.assertEqual(us.parse_seed("SPY QQQ SPY"), ["SPY", "QQQ"])

    def test_소문자를_대문자로_올린다(self):
        self.assertEqual(us.parse_seed("spy\ntqqq"), ["SPY", "TQQQ"])


class AutocompleteTest(unittest.TestCase):
    def test_거래소_접미사가_붙은_코드를_준다(self):
        # 같은 티커라도 거래소마다 코드가 다르다. 이걸 틀리면 일봉을 못 받는다.
        row = us.parse_autocomplete(load("naver_us_autocomplete.json"), "SOXL")
        self.assertEqual(row["code"], "SOXL.K")
        self.assertEqual(row["ticker"], "SOXL")
        self.assertTrue(row["etf"])
        self.assertIn("Semiconductor", row["name"])

    def test_티커가_정확히_맞는_것만_고른다(self):
        # 'SPY' 로 물으면 'Spyre Therapeutics' 도 같이 온다.
        self.assertIsNone(us.parse_autocomplete(load("naver_us_autocomplete.json"), "SOX"))

    def test_응답이_망가지면_None이다(self):
        self.assertIsNone(us.parse_autocomplete(b"not json", "SPY"))
        self.assertIsNone(us.parse_autocomplete(b'{"items":[]}', "SPY"))

    def test_ETN은_ETF가_아니라고_표시한다(self):
        # url 이 /worldstock/stock/ 이면 ETN 이다. 발행사 신용위험이 붙는
        # 다른 물건이라 같이 취급하면 안 된다.
        raw = (b'{"items":[{"code":"VXX","reutersCode":"VXX","name":"iPath VIX",'
               b'"nationCode":"USA","url":"/worldstock/stock/VXX/total"}]}')
        row = us.parse_autocomplete(raw, "VXX")
        self.assertFalse(row["etf"])

    def test_미국이_아니면_거른다(self):
        raw = (b'{"items":[{"code":"SPY","reutersCode":"SPY","name":"x",'
               b'"nationCode":"KOR","url":"/worldstock/etf/SPY"}]}')
        self.assertIsNone(us.parse_autocomplete(raw, "SPY"))


class ChartTest(unittest.TestCase):
    def setUp(self):
        self.rows = us.parse_chart(load("naver_us_chart.json"))

    def test_국내_파서와_같은_모양으로_돌려준다(self):
        # 지표 계산 코드를 둘로 나누지 않으려면 키 이름이 같아야 한다.
        self.assertEqual(set(self.rows[0]), {"date", "open", "high", "low", "close", "volume"})

    def test_날짜_오름차순이다(self):
        dates = [r["date"] for r in self.rows]
        self.assertEqual(dates, sorted(dates))

    def test_고가는_저가보다_크거나_같다(self):
        self.assertTrue(all(r["high"] >= r["low"] for r in self.rows))

    def test_거래량이_없으면_0으로_둔다(self):
        rows = us.parse_chart(b'[{"localDate":"20260810","closePrice":100}]')
        self.assertEqual(rows[0]["volume"], 0)
        self.assertEqual(rows[0]["high"], 100)

    def test_종가가_0이하인_줄은_버린다(self):
        self.assertEqual(us.parse_chart(b'[{"localDate":"20260810","closePrice":0}]'), [])

    def test_망가진_응답은_빈_목록이다(self):
        self.assertEqual(us.parse_chart(b"<html>"), [])


class FxTest(unittest.TestCase):
    def setUp(self):
        self.rows = us.parse_fx(load("naver_us_fx.json"))

    def test_쉼표_박힌_문자열을_숫자로_바꾼다(self):
        # '1,417.20' 을 그대로 float 하면 터진다.
        self.assertGreater(self.rows[-1]["close"], 500)
        self.assertIsInstance(self.rows[-1]["close"], float)

    def test_날짜에서_하이픈을_뗀다(self):
        # 일봉은 '20260810' 인데 환율은 '2026-08-10' 이라 맞춰야 붙는다.
        self.assertTrue(all(len(r["date"]) == 8 and r["date"].isdigit() for r in self.rows))

    def test_오름차순이다(self):
        dates = [r["date"] for r in self.rows]
        self.assertEqual(dates, sorted(dates))


if __name__ == "__main__":
    unittest.main()
