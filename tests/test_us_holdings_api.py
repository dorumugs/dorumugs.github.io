"""미국 ETF 구성종목 파싱 회귀 테스트.

세 발행사 실제 응답을 tests/fixtures/ 에 고정해 두고 그걸로 잡는다(2026-08-11
캡처). 발행사 화면이 바뀌면 여기서 조용히 걸린다.
"""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
import zipfile

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

    def test_토탈리턴스왑은_rows에도_cash에도_없고_swap_버킷으로_간다(self):
        # SOXL(3배)은 스왑으로 레버리지를 만든다. 스왑 명목가치를 현금으로 세면
        # cash 가 271% 로 찍힌다 — 그렇다고 버리면(예전 버그) "현금 44%짜리
        # 펀드"로 보이는 착시가 생긴다. 버리지 않고 swap 버킷으로 살린다.
        names = [r["name"] for r in self.result["rows"]]
        self.assertFalse(any("SWAP" in n.upper() for n in names))
        self.assertIn("ICE SEMICONDUCTOR INDEX SWAP", self.result["swapNote"])

    def test_네_버킷_실측값(self):
        # 리포트 실측: 주식 30행 72.2% · 스왑 8행 219.6% · 현금 4행 43.7% ·
        # 기타("Semiconductor Bull 3x") 1행 8.1% = 합계 343.6%.
        equity = sum(r["weight"] for r in self.result["rows"])
        self.assertAlmostEqual(equity, 72.2, places=1)
        self.assertAlmostEqual(self.result["swap"], 219.6, places=1)
        self.assertAlmostEqual(self.result["cash"], 43.7, places=1)
        self.assertAlmostEqual(self.result["other"], 8.1, places=1)
        total = h.total_weight(
            self.result["rows"], self.result["cash"], self.result["swap"], self.result["other"]
        )
        self.assertAlmostEqual(total, 343.6, delta=0.2)

    def test_펀드_자체_배수_이름은_other로_간다(self):
        # "Semiconductor Bull 3x" 는 스왑과 짝을 이루는 명목가치지만 이름에
        # SWAP 이 없다 — rows·cash·swap 어디에도 안 들어가고 other 로 간다.
        names = [r["name"] for r in self.result["rows"]]
        self.assertFalse(any("bull" in n.lower() for n in names))
        self.assertGreater(self.result["other"], 0)

    def test_유효성_검사는_스왑을_포함해야_통과한다(self):
        # 3배 레버리지 기준(scale=3)으로 봤을 때, 스왑을 뺀 채로 검사하면 실패하고
        # (72.2+43.7+8.1=124 는 300 근처가 아니다) 스왑을 포함해야 통과한다.
        self.assertFalse(
            h.total_weight_ok(self.result["rows"], self.result["cash"], 0.0, self.result["other"], leverage=3.0)
        )
        self.assertTrue(
            h.total_weight_ok(
                self.result["rows"], self.result["cash"], self.result["swap"], self.result["other"], leverage=3.0
            )
        )

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

    def test_주식형이라_noTicker가_거짓이다(self):
        self.assertFalse(self.result["noTicker"])
        self.assertEqual(self.result["swap"], 0.0)


class SpdrBilTest(unittest.TestCase):
    """BIL(SPDR 1-3개월 T-Bill ETF) — Ticker 열 자체가 없는 채권형.

    이전 코드는 Name·Ticker·Weight 셋 다 있어야 헤더로 인정해서, Ticker 가
    없는 이 파일은 rows·cash 모두 빈 채로 돌아갔다 — 그 결과 화면에는
    "구성종목을 받지 못했습니다 (SPDR)" 가 떴다. 파일은 200 으로 받았고
    파싱도 가능한데 실패로 보고하는 거짓말이었다.
    """

    def setUp(self):
        self.result = h.parse_spdr(load("us_holdings_spdr_bil.xlsx"))

    def test_기준일이_나온다(self):
        self.assertEqual(self.result["asOf"], "2026-08-10")

    def test_noTicker가_참이고_구성종목이_실린다(self):
        self.assertTrue(self.result["noTicker"])
        self.assertGreater(len(self.result["rows"]), 0)

    def test_티커_없는_채권도_rows에_실리고_cash로_쓸려가지_않는다(self):
        # 채권형에서 티커 없음은 "현금"이 아니라 "이 파일엔 티커 칸이 없다"는
        # 뜻이다 — cash 로 합치면 국채 사다리를 통째로 현금으로 오분류하게 된다.
        self.assertEqual(self.result["cash"], 0.0)
        top = self.result["rows"][0]
        self.assertEqual(top["ticker"], "")
        self.assertIn("TREASURY BILL", top["name"])

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


class ZipBombGuardTest(unittest.TestCase):
    """압축 해제 크기 상한. cron 무인 실행이라 오염된 응답에 서버가 죽으면 안 된다."""

    def _bomb(self) -> bytes:
        """헤더의 file_size 가 진실인 zip. 상한을 넘는 sheet1.xml 하나만 담는다."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("xl/worksheets/sheet1.xml", b"\0" * (h.MAX_UNZIPPED_BYTES + 1))
        return buf.getvalue()

    def test_상한_넘는_시트는_빈_결과로_떨어진다(self) -> None:
        parsed = h.parse_spdr(self._bomb())
        self.assertEqual(parsed["rows"], [])
        self.assertEqual(parsed["cash"], 0.0)

    def test_압축률이_높아도_메모리로_풀지_않는다(self) -> None:
        raw = self._bomb()
        self.assertLess(len(raw), 1024 * 1024, "압축본은 작다 — 그래서 상한이 필요하다")
        archive = zipfile.ZipFile(io.BytesIO(raw))
        with self.assertRaises(ValueError):
            h._read_capped(archive, "xl/worksheets/sheet1.xml")

    def test_정상_파일은_그대로_읽힌다(self) -> None:
        """상한 때문에 멀쩡한 파일이 막히면 안 된다 — XLV 는 그대로 파싱된다."""
        parsed = h.parse_spdr((FIXTURES / "us_holdings_spdr_xlv.xlsx").read_bytes())
        self.assertGreater(len(parsed["rows"]), 0)
