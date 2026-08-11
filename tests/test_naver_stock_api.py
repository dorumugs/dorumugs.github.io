"""네이버 금융 파싱 회귀 테스트.

공식 API 가 아니라 화면을 읽는 곳이 대부분이라 네이버가 마크업을 바꾸면 조용히
깨진다. tests/fixtures/ 에 실제 응답을 고정해 두고 그걸로 잡는다.
"""

from __future__ import annotations

import datetime
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import naver_stock_api as api  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class ThemeListTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_theme_list(load("naver_theme_list.html"))

    def test_한_페이지에_40개가_나온다(self):
        self.assertEqual(len(self.rows), 40)

    def test_한글이_깨지지_않는다(self):
        # EUC-KR 을 UTF-8 로 읽으면 여기서 깨진다.
        names = [r["name"] for r in self.rows]
        self.assertIn("정유", names)

    def test_필드가_숫자로_파싱된다(self):
        row = next(r for r in self.rows if r["no"] == "185")
        self.assertEqual(row["type"], "theme")
        self.assertEqual(row["name"], "정유")
        self.assertIsInstance(row["change_rate"], float)
        self.assertIsInstance(row["change_3d"], float)
        self.assertEqual(row["up"] + row["flat"] + row["down"], 3)

    def test_주도주는_코드와_이름으로_온다(self):
        row = next(r for r in self.rows if r["no"] == "185")
        codes = [x["code"] for x in row["leaders"]]
        self.assertIn("010950", codes)
        self.assertTrue(all(len(c) == 6 for c in codes))


class UpjongListTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_upjong_list(load("naver_upjong_list.html"))

    def test_업종이_모두_나온다(self):
        self.assertEqual(len(self.rows), 79)

    def test_상승보합하락_합이_전체와_같다(self):
        row = next(r for r in self.rows if r["name"] == "손해보험")
        self.assertEqual(row["up"] + row["flat"] + row["down"], row["total"])

    def test_업종에는_최근3일이_없다(self):
        self.assertNotIn("change_3d", self.rows[0])


class GroupDetailTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_group_detail(load("naver_group_detail.html"))

    def test_구성종목이_중복없이_나온다(self):
        codes = [r["code"] for r in self.rows]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(codes), 3)

    def test_종목명이_붙어_온다(self):
        by_code = {r["code"]: r["name"] for r in self.rows}
        self.assertEqual(by_code["010950"], "S-Oil")


class MarketCodesTest(unittest.TestCase):
    def test_시총목록에서_종목코드만_뽑는다(self):
        codes = api.parse_market_codes(load("naver_market_sum.html"))
        self.assertIn("005930", codes)
        self.assertEqual(len(codes), len(set(codes)))
        # 한 페이지는 50종목이다. 이보다 훨씬 많으면 광고·추천 링크를 주운 것이다.
        self.assertLessEqual(len(codes), 50)


class EtfListTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_etf_list(load("naver_etf_list.json"))

    def test_전종목이_나온다(self):
        self.assertGreater(len(self.rows), 1000)

    def test_한글_종목명이_깨지지_않는다(self):
        # 이 JSON 도 EUC-KR 이다. UTF-8 로 읽으면 예외가 난다.
        names = [r["name"] for r in self.rows]
        self.assertIn("KODEX 200", names)
        self.assertTrue(any("반도체" in n for n in names))

    def test_분류코드가_1에서_7이다(self):
        tabs = {r["tab"] for r in self.rows}
        self.assertTrue(tabs <= {1, 2, 3, 4, 5, 6, 7}, tabs)

    def test_거래대금과_시총이_온다(self):
        row = next(r for r in self.rows if r["code"] == "069500")
        self.assertIsNotNone(row["amount_mn"])
        self.assertIsNotNone(row["market_cap_100m"])


class EtfHoldingsTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_etf_holdings(load("naver_etf_holdings.html"))

    def test_구성종목과_비중이_나온다(self):
        self.assertEqual(len(self.rows), 26)
        self.assertEqual(self.rows[0]["name"], "LG에너지솔루션")
        self.assertAlmostEqual(self.rows[0]["weight"], 21.45, places=2)

    def test_이_서버만_UTF8이다(self):
        # EUC-KR 로 읽으면 여기서 깨진다.
        self.assertTrue(any("삼성" in r["name"] for r in self.rows))

    def test_CU_data가_없으면_빈_리스트다(self):
        self.assertEqual(api.parse_etf_holdings(b"<html>no data</html>"), [])


class SiseJsonTest(unittest.TestCase):
    def setUp(self):
        self.rows = api.parse_sise_json(load("naver_sise_json.txt"))

    def test_일봉이_파싱된다(self):
        self.assertGreater(len(self.rows), 40)
        first = self.rows[0]
        self.assertEqual(set(first), {"date", "open", "high", "low", "close", "volume"})

    def test_날짜_오름차순이다(self):
        dates = [r["date"] for r in self.rows]
        self.assertEqual(dates, sorted(dates))

    def test_헤더행을_데이터로_줍지_않는다(self):
        # 응답 첫 줄은 ['날짜','시가',...] 라는 헤더다.
        self.assertTrue(all(r["date"].isdigit() for r in self.rows))

    def test_고가는_저가보다_크거나_같다(self):
        self.assertTrue(all(r["high"] >= r["low"] for r in self.rows))


class NormalizeNameTest(unittest.TestCase):
    def test_공백과_기호를_턴다(self):
        self.assertEqual(api.normalize_name("POSCO 홀딩스"), "POSCO홀딩스")
        self.assertEqual(api.normalize_name("SK 하이닉스"), "SK하이닉스")

    def test_우선주는_보통주와_다르게_남는다(self):
        self.assertNotEqual(
            api.normalize_name("삼성전자(우)"), api.normalize_name("삼성전자")
        )


class LeverageTest(unittest.TestCase):
    def test_배수를_읽는다(self):
        cases = {
            "KODEX 레버리지": 2.0,
            "KODEX 코스닥150레버리지": 2.0,
            "TIGER 반도체TOP10레버리지": 2.0,
            "KODEX 인버스": -1.0,
            "KODEX 200선물인버스2X": -2.0,
            "TIGER 반도체TOP10": 1.0,
            "KODEX 200": 1.0,
            "TIGER 미국S&P500": 1.0,
            "KODEX 2차전지산업": 1.0,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(api.leverage_of(name), expected)

    def test_실제_목록에_말도_안되는_배수가_없다(self):
        # 국내 상장은 최대 2배다. 3배가 나오면 이름 파싱이 틀린 것이다.
        rows = api.parse_etf_list(load("naver_etf_list.json"))
        for row in rows:
            with self.subTest(name=row["name"]):
                self.assertLessEqual(abs(api.leverage_of(row["name"])), 2.0)


class IntradayTest(unittest.TestCase):
    """장중 수집을 알아채는가.

    실제로 당한 적이 있다 — 11:09 에 받은 KODEX 200 의 '오늘 종가' 가 99,100
    이었는데 9분 뒤엔 98,880 이었다. siseJson 이 장중에도 오늘 행을 주는데 그
    종가 칸이 확정값이 아니라 현재가이기 때문이다.
    """

    def test_장중이면_참이다(self):
        # 2026-08-11 은 화요일.
        self.assertTrue(api.is_intraday(datetime.datetime(2026, 8, 11, 11, 9)))
        self.assertTrue(api.is_intraday(datetime.datetime(2026, 8, 11, 9, 0)))

    def test_마감_뒤에는_거짓이다(self):
        self.assertFalse(api.is_intraday(datetime.datetime(2026, 8, 11, 18, 30)))
        self.assertFalse(api.is_intraday(datetime.datetime(2026, 8, 11, 15, 40)))

    def test_마감_직전과_직후가_갈린다(self):
        self.assertTrue(api.is_intraday(datetime.datetime(2026, 8, 11, 15, 39)))
        self.assertFalse(api.is_intraday(datetime.datetime(2026, 8, 11, 15, 40)))

    def test_주말은_새_거래일이_없어_거짓이다(self):
        # 2026-08-15 토요일, 08-16 일요일.
        self.assertFalse(api.is_intraday(datetime.datetime(2026, 8, 15, 11, 0)))
        self.assertFalse(api.is_intraday(datetime.datetime(2026, 8, 16, 11, 0)))


class HedgedTest(unittest.TestCase):
    def test_환헤지_표기를_읽는다(self):
        self.assertTrue(api.is_hedged("KODEX 골드선물(H)"))
        self.assertFalse(api.is_hedged("TIGER 미국S&P500"))


if __name__ == "__main__":
    unittest.main()
