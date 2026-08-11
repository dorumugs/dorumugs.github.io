"""미국 ETF 구성종목 파싱 회귀 테스트.

세 발행사 실제 응답을 tests/fixtures/ 에 고정해 두고 그걸로 잡는다(2026-08-11
캡처). 발행사 화면이 바뀌면 여기서 조용히 걸린다.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import us_holdings_api as h  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class DirexionSoxlTest(unittest.TestCase):
    def setUp(self):
        self.result = h.parse_direxion(load("us_holdings_direxion_soxl.csv"))

    def test_기준일이_나온다(self):
        self.assertEqual(self.result["asOf"], "2026-08-11")

    def test_비중_상위_실제값(self):
        top = self.result["rows"][0]
        self.assertEqual(top["ticker"], "NVDA")
        self.assertAlmostEqual(top["weight"], 6.4832, places=3)
        second = self.result["rows"][1]
        self.assertEqual(second["ticker"], "AVGO")

    def test_현금에_드레이퍼스가_반영된다(self):
        # DREYFUS GOVT CASH MAN INS 가 StockTicker 없이 11.5% 로 온다. 같은
        # 성격의 다른 현금성 MMF(DREYFUS TRSRY·GOLDMAN·JPMORGAN)도 섞여 있어
        # cash 총합은 11.5% 보다 크지만, 적어도 그 11.5%p 이상은 반영돼야 한다.
        self.assertGreaterEqual(self.result["cash"], 11.4972 - 0.01)

    def test_현금이_최상위_종목으로_나오지_않는다(self):
        # 이게 원래 버그였다 — DREYFUS 가 종목코드 없이 1위 비중으로 온다.
        names = [r["name"] for r in self.result["rows"]]
        self.assertFalse(any("DREYFUS" in n for n in names))
        tickers = [r["ticker"] for r in self.result["rows"]]
        self.assertNotIn("", tickers)

    def test_토탈리턴스왑은_rows에도_cash에도_없다(self):
        # SOXL(3배)은 스왑으로 레버리지를 만든다. 스왑 명목가치를 현금으로 세면
        # cash 가 271% 로 찍힌다 — 현금이 아니라 버려야 한다.
        names = [r["name"] for r in self.result["rows"]]
        self.assertFalse(any("SWAP" in n.upper() for n in names))
        total = h.total_weight(self.result["rows"], self.result["cash"])
        self.assertLess(total, 200.0)  # 스왑(약 228%p)을 걸러내지 않았다면 340%대가 나온다

    def test_비중_내림차순_정렬(self):
        weights = [r["weight"] for r in self.result["rows"]]
        self.assertEqual(weights, sorted(weights, reverse=True))


class ArkArkgTest(unittest.TestCase):
    def setUp(self):
        self.result = h.parse_ark(load("us_holdings_ark_arkg.csv"))

    def test_기준일이_나온다(self):
        self.assertEqual(self.result["asOf"], "2026-08-11")

    def test_10x_지노믹스가_10_11퍼센트(self):
        top = self.result["rows"][0]
        self.assertEqual(top["ticker"], "TXG")
        self.assertIn("10X GENOMICS", top["name"])
        self.assertAlmostEqual(top["weight"], 10.11, places=2)

    def test_현금성_MMF는_rows에_없다(self):
        names = [r["name"] for r in self.result["rows"]]
        self.assertFalse(any("GOLDMAN" in n for n in names))
        self.assertGreater(self.result["cash"], 0)

    def test_꼬리_법적고지문이_행으로_들어오지_않는다(self):
        # 마지막 줄은 weight 칸에 '%' 가 없는 통째 법적고지문이다.
        for row in self.result["rows"]:
            self.assertTrue(row["ticker"])
            self.assertIsInstance(row["weight"], float)

    def test_유효성_검사를_통과한다(self):
        self.assertTrue(h.total_weight_ok(self.result["rows"], self.result["cash"]))


class SpdrXlvTest(unittest.TestCase):
    def setUp(self):
        self.result = h.parse_spdr(load("us_holdings_spdr_xlv.xlsx"))

    def test_기준일이_나온다(self):
        self.assertEqual(self.result["asOf"], "2026-08-10")

    def test_최상위_종목은_일라이릴리(self):
        top = self.result["rows"][0]
        self.assertEqual(top["ticker"], "LLY")
        self.assertAlmostEqual(top["weight"], 16.0183, places=3)

    def test_MMF와_달러잔여는_rows에_없고_cash로_간다(self):
        tickers = [r["ticker"] for r in self.result["rows"]]
        self.assertNotIn("-", tickers)
        self.assertGreater(self.result["cash"], 0)

    def test_유효성_검사를_통과한다(self):
        self.assertTrue(h.total_weight_ok(self.result["rows"], self.result["cash"]))


class HoldingsUrlTest(unittest.TestCase):
    def test_direxion_url(self):
        self.assertEqual(h.holdings_url("SOXL"), "https://www.direxion.com/holdings/SOXL.csv")

    def test_spdr_url_은_소문자_티커(self):
        url = h.holdings_url("XLV")
        self.assertTrue(url.endswith("holdings-daily-us-en-xlv.xlsx"))

    def test_ark_url_은_티커가_아니라_펀드파일명(self):
        url = h.holdings_url("ARKG")
        self.assertIn("ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS", url)
        self.assertNotIn("/ARKG.csv", url)

    def test_모르는_발행사는_None(self):
        # iShares·ProShares·Invesco·GlobalX 등은 URL 을 확인 못 했다 — 추측하지 않는다.
        self.assertIsNone(h.holdings_url("TQQQ"))
        self.assertIsNone(h.holdings_url("존재하지않는티커"))

    def test_금_실물신탁은_같은_URL_패턴이_없다(self):
        self.assertIsNone(h.holdings_url("GLD"))
        self.assertIsNone(h.holdings_url("GLDM"))


class TotalWeightOkTest(unittest.TestCase):
    def test_100퍼센트_근방이면_통과(self):
        rows = [{"ticker": "A", "name": "A", "weight": 60.0}, {"ticker": "B", "name": "B", "weight": 39.0}]
        self.assertTrue(h.total_weight_ok(rows, 1.0))

    def test_너무_적으면_실패(self):
        rows = [{"ticker": "A", "name": "A", "weight": 10.0}]
        self.assertFalse(h.total_weight_ok(rows, 0.0))

    def test_레버리지면_임계값이_배수로_늘어난다(self):
        rows = [{"ticker": "A", "name": "A", "weight": 290.0}]
        self.assertFalse(h.total_weight_ok(rows, 0.0, leverage=1.0))
        self.assertTrue(h.total_weight_ok(rows, 0.0, leverage=3.0))


if __name__ == "__main__":
    unittest.main()
