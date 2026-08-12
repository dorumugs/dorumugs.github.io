"""미국 ETF 구성종목 파싱 회귀 테스트.

세 발행사 실제 응답을 tests/fixtures/ 에 고정해 두고 그걸로 잡는다(2026-08-11
캡처). 발행사 화면이 바뀌면 여기서 조용히 걸린다.
"""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from datetime import date, timedelta
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


class IsharesTest(unittest.TestCase):
    """iShares — 주식형(SOXX)과 채권형(TLT)의 열 구성이 다르다."""

    def test_주식형은_티커와_비중을_읽는다(self) -> None:
        p = h.parse_ishares(load("us_holdings_ishares_soxx.csv"))
        self.assertEqual(p["asOf"], "2026-08-10")
        self.assertFalse(p["noTicker"])
        self.assertEqual(len(p["rows"]), 34)
        self.assertEqual(p["rows"][0]["ticker"], "NVDA")
        self.assertAlmostEqual(sum(r["weight"] for r in p["rows"]), 99.99, places=1)

    def test_채권형은_티커_열이_없다(self) -> None:
        """TLT 는 Ticker 열 자체가 없다. 없다고 현금으로 쓸어담으면 안 된다."""
        p = h.parse_ishares(load("us_holdings_ishares_tlt.csv"))
        self.assertTrue(p["noTicker"])
        self.assertEqual(p["cash"], 0.0)
        self.assertEqual(len(p["rows"]), 48)
        self.assertEqual(p["rows"][0]["name"], "TREASURY BOND")
        self.assertTrue(all(r["ticker"] == "" for r in p["rows"]))

    def test_헤더를_못_찾으면_빈_결과(self) -> None:
        self.assertEqual(h.parse_ishares(b"<!DOCTYPE html><html></html>")["rows"], [])


class GlobalxTest(unittest.TestCase):
    def test_비중_티커_이름을_읽는다(self) -> None:
        p = h.parse_globalx(load("us_holdings_globalx_lit.csv"))
        self.assertEqual(p["asOf"], "2026-08-10")
        self.assertEqual(len(p["rows"]), 41)
        self.assertAlmostEqual(sum(r["weight"] for r in p["rows"]), 99.98, places=1)

    def test_해외상장_티커의_접미를_안_뗀다(self) -> None:
        """'ERA FP' 에서 ' FP' 를 떼면 미국 티커와 충돌해 엉뚱한 회사가 된다."""
        p = h.parse_globalx(load("us_holdings_globalx_lit.csv"))
        self.assertTrue(any(" " in r["ticker"] for r in p["rows"]))

    def test_URL_은_최신_날짜부터_나열한다(self) -> None:
        urls = h.globalx_urls("LIT", date(2026, 8, 12), back=2)
        self.assertEqual(len(urls), 3)
        self.assertIn("lit_full-holdings_20260812.csv", urls[0])
        self.assertIn("lit_full-holdings_20260810.csv", urls[2])
        self.assertEqual(h.globalx_urls("SPY", date(2026, 8, 12)), [])


class VanguardTest(unittest.TestCase):
    def test_주식형은_티커와_비중을_읽는다(self) -> None:
        p = h.parse_vanguard(load("us_holdings_vanguard_voo.json"))
        self.assertEqual(p["asOf"], "2026-06-30")
        self.assertFalse(p["noTicker"])
        self.assertEqual(p["rows"][0]["ticker"], "NVDA")

    def test_한_장만_받아도_진짜_종목수를_남긴다(self) -> None:
        """500행 상한이라 받은 줄 수가 종목 수가 아니다 — 화면이 거짓말하면 안 된다."""
        p = h.parse_vanguard(load("us_holdings_vanguard_voo.json"))
        self.assertEqual(p["totalCount"], 504)
        self.assertLess(len(p["rows"]), p["totalCount"])

    def test_채권형은_티커를_통째로_버린다(self) -> None:
        """채권 원장의 ticker 는 발행사의 '주식' 티커라 그대로 쓰면 오해를 만든다."""
        p = h.parse_vanguard(load("us_holdings_vanguard_bnd.json"), bond=True)
        self.assertTrue(p["noTicker"])
        self.assertTrue(all(r["ticker"] == "" for r in p["rows"]))
        self.assertEqual(p["totalCount"], 10065)

    def test_JSON_이_아니면_빈_결과(self) -> None:
        self.assertEqual(h.parse_vanguard(b"<html>nope</html>")["rows"], [])


class ProsharesTest(unittest.TestCase):
    """전 종목 단일 파일. 비중 열이 없어 금액에서 만든다 — 분모가 핵심이다."""

    def setUp(self) -> None:
        self.all = h.parse_proshares_all(load("us_holdings_proshares_daily.csv"))

    def test_펀드별로_갈라진다(self) -> None:
        self.assertEqual(sorted(self.all), ["NOBL", "SQQQ", "TQQQ", "UVXY"])
        self.assertEqual(self.all["TQQQ"]["asOf"], "2026-08-10")

    def test_UVXY_선물이_정확히_1점5배로_떨어진다(self) -> None:
        """분모를 Exposure 합으로 잡으면 이 값이 안 나온다 — NAV 는 Market Value 합이다."""
        p = self.all["UVXY"]
        self.assertAlmostEqual(p["swap"], 150.0, places=1)

    def test_선물을_현금으로_세지_않는다(self) -> None:
        """VIX 선물을 현금에 넣으면 '현금 100% 펀드' 라는 거짓 화면이 된다."""
        p = self.all["UVXY"]
        self.assertLess(p["cash"], 25.0)
        self.assertEqual(p["rows"], [])

    def test_TQQQ_는_주식과_스왑을_함께_든다(self) -> None:
        p = self.all["TQQQ"]
        self.assertGreater(p["swap"], 200.0)
        self.assertGreater(len(p["rows"]), 50)
        self.assertIn("NASDAQ 100", p["swapNote"])

    def test_인버스는_스왑이_음수다(self) -> None:
        self.assertLess(self.all["SQQQ"]["swap"], -200.0)

    def test_레버리지_아닌_상품은_스왑이_없다(self) -> None:
        p = self.all["NOBL"]
        self.assertEqual(p["swap"], 0.0)
        self.assertAlmostEqual(sum(r["weight"] for r in p["rows"]), 99.9, places=0)

    def test_NAV_잔여항목은_개별_칸으로_안_센다(self) -> None:
        """'Net Other Assets' 는 분모에만 들어간다 — 현금에 또 더하면 이중계상이다."""
        for p in self.all.values():
            self.assertNotIn("Net Other Assets", p["swapNote"])
            names = [r["name"] for r in p["rows"]]
            self.assertFalse(any("Net Other Assets" in n for n in names))


class InverseValidityTest(unittest.TestCase):
    def test_인버스는_비중합을_검사하지_않는다(self) -> None:
        """부호 섞인 합은 배수와 무관하다 — 여기에 임계값을 맞추면 진짜 오류를 놓친다."""
        rows = [{"ticker": "X", "name": "x", "weight": 87.7}]
        self.assertTrue(h.total_weight_ok(rows, 34.0, -100.0, 0.0, leverage=-1.0))

    def test_롱_레버리지는_여전히_검사한다(self) -> None:
        rows = [{"ticker": "X", "name": "x", "weight": 10.0}]
        self.assertFalse(h.total_weight_ok(rows, 0.0, 0.0, 0.0, leverage=3.0))


class CashLikeTest(unittest.TestCase):
    """티커가 있어도 실질이 현금인 줄은 종목 표에 올리지 않는다."""

    def test_자사_MMF_는_1위_보유가_아니다(self) -> None:
        """TQQQ 표 1위가 'PROSHARES GENIUS MNY MKT ETF' 로 찍히던 문제."""
        p = h.parse_proshares_all(load("us_holdings_proshares_daily.csv"))["TQQQ"]
        self.assertFalse(any(r["ticker"] == "IQMM" for r in p["rows"]))
        self.assertEqual(p["rows"][0]["ticker"], "NVDA")
        self.assertGreater(p["cash"], 30.0)

    def test_현금성_판정(self) -> None:
        self.assertTrue(h._is_cash_like("PROSHARES GENIUS MNY MKT ETF"))
        self.assertTrue(h._is_cash_like("Goldman Sachs Money Market Fund"))
        self.assertFalse(h._is_cash_like("NVIDIA CORP"))
        self.assertFalse(h._is_cash_like("MARKET AXESS HOLDINGS"))


class InvescoTest(unittest.TestCase):
    """securityTypeCode 로 가른다 — 이름 추측이 아니라."""

    def test_주식형은_100퍼센트로_떨어진다(self) -> None:
        p = h.parse_invesco(load("us_holdings_invesco_qqq.json"))
        self.assertEqual(p["asOf"], "2026-08-10")
        self.assertEqual(p["rows"][0]["ticker"], "NVDA")
        total = sum(r["weight"] for r in p["rows"]) + p["cash"] + p["swap"] + p["other"]
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_원자재_담보를_종목으로_안_센다(self) -> None:
        """DBC 는 담보 MMF 가 80.2% 라, 이름만 보고 담으면 1위가 MMF 가 되고 합이 198% 다."""
        p = h.parse_invesco(load("us_holdings_invesco_dbc.json"))
        self.assertFalse(any(r["ticker"] == "AGPXX" for r in p["rows"]))
        self.assertGreater(p["swap"], 100.0)  # 원자재 선물
        total = sum(r["weight"] for r in p["rows"]) + p["cash"] + p["swap"] + p["other"]
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_이름의_HTML_엔티티를_푼다(self) -> None:
        """issuerName 에 &amp; 가 그대로 온다 — 화면에 'Government &amp; Agency' 로 찍히면 안 된다."""
        p = h.parse_invesco(load("us_holdings_invesco_dbc.json"))
        self.assertFalse(any("&amp;" in r["name"] for r in p["rows"]))

    def test_JSON_이_아니면_빈_결과(self) -> None:
        self.assertEqual(h.parse_invesco(b"<html>406</html>")["rows"], [])

    def test_QQQ_만_티커로_조회한다(self) -> None:
        """나머지는 티커로 부르면 500 이라 CUSIP 을 박아 뒀다."""
        self.assertIn("idType=ticker", h.invesco_url("QQQ"))
        self.assertIn("idType=cusip", h.invesco_url("RSP"))
        self.assertIsNone(h.invesco_url("SPY"))


class FirstTrustTest(unittest.TestCase):
    def test_HTML_표에서_티커와_비중을_읽는다(self) -> None:
        p = h.parse_firsttrust(load("us_holdings_firsttrust_cibr.html"))
        self.assertEqual(p["asOf"], "2026-08-10")
        self.assertEqual(p["rows"][0]["ticker"], "PANW")
        self.assertGreater(len(p["rows"]), 30)

    def test_헤더가_없으면_빈_결과(self) -> None:
        """열 이름이 바뀌면 엉뚱한 값을 비중으로 읽느니 빈 결과가 낫다."""
        self.assertEqual(h.parse_firsttrust(b"<table><tr><td>x</td></tr></table>")["rows"], [])


class KraneSharesTest(unittest.TestCase):
    def test_현지_티커를_그대로_둔다(self) -> None:
        """텐센트는 700, 알리바바는 9988 이다 — 미국 티커로 바꾸면 다른 회사가 된다."""
        p = h.parse_kraneshares(load("us_holdings_kraneshares_kweb.csv"))
        self.assertEqual(p["asOf"], "2026-08-10")
        self.assertEqual(p["rows"][0]["ticker"], "700")
        self.assertAlmostEqual(sum(r["weight"] for r in p["rows"]) + p["cash"], 100.0, places=0)

    def test_URL_은_MMDDYYYY_형식이다(self) -> None:
        urls = h.kraneshares_urls("KWEB", date(2026, 8, 12), back=2)
        self.assertIn("08_12_2026_kweb_holdings.csv", urls[0])
        self.assertIn("08_10_2026_kweb_holdings.csv", urls[2])
        self.assertEqual(h.kraneshares_urls("SPY", date(2026, 8, 12)), [])


class PhysicalTrustTest(unittest.TestCase):
    def test_실물_신탁은_발행사_목록에_들어_있다(self) -> None:
        """받으려다 실패한 게 아니라 구성종목이 없는 상품이라, 화면이 구별해야 한다."""
        for ticker in ("GLD", "GLDM", "IAU", "SLV"):
            self.assertEqual(h.ISSUER_BY_TICKER[ticker], h.ISSUER_PHYSICAL)
            self.assertIn(h.PHYSICAL_TRUSTS[ticker], ("금괴", "은괴"))

    def test_실물_신탁은_URL_을_안_만든다(self) -> None:
        self.assertIsNone(h.holdings_url("GLD"))


class FreshnessTest(unittest.TestCase):
    """조용한 붕괴를 막는 관문.

    예전 게이트는 "캐시에 있느냐" 로 절반을 봤다. 그건 이 고장을 절대 못 잡는다 —
    발행사가 화면을 바꾸면 파서가 빈 결과를 내고 수집기는 어제 캐시를 그대로
    들고 가서, 개수는 162개 그대로다. 오늘 전부 깨져도 통과한다.
    """

    TODAY = date(2026, 8, 12)

    def _cache(self, spec: dict[str, tuple[str, int]]) -> dict[str, dict]:
        """{티커: (발행사, 며칠 전)} 을 캐시 모양으로."""
        out = {}
        for ticker, (issuer, age) in spec.items():
            out[ticker] = {
                "issuer": issuer,
                "fetchedDate": (self.TODAY - timedelta(days=age)).isoformat(),
            }
        return out

    def test_발행사_하나가_통째로_낡으면_파손이다(self) -> None:
        cache = self._cache({f"A{i}": ("iShares", 30) for i in range(41)}
                            | {f"B{i}": ("SPDR", 0) for i in range(22)})
        self.assertEqual(h.freshness(cache, self.TODAY)["broken"], ["iShares"])

    def test_개별_종목_몇_개는_파손이_아니다(self) -> None:
        """발행사 서버가 하루 못 버티는 건 흔하다 — 그걸로 울리면 아무도 안 본다."""
        cache = self._cache({f"A{i}": ("iShares", 0) for i in range(38)}
                            | {f"C{i}": ("iShares", 30) for i in range(3)})
        report = h.freshness(cache, self.TODAY)
        self.assertEqual(report["broken"], [])
        self.assertEqual(len(report["staleTickers"]), 3)

    def test_종목이_적은_발행사는_파손으로_안_센다(self) -> None:
        """1~2개짜리는 우연히 0이 될 수 있다."""
        cache = self._cache({"KWEB": ("KraneShares", 30)})
        self.assertEqual(h.freshness(cache, self.TODAY)["broken"], [])

    def test_주말을_건너도_신선하다(self) -> None:
        """크론은 평일에만 돈다 — 금요일 자료가 월요일에 낡음으로 찍히면 안 된다."""
        cache = self._cache({f"A{i}": ("SPDR", 3) for i in range(22)})
        report = h.freshness(cache, self.TODAY)
        self.assertEqual(report["broken"], [])
        self.assertEqual(report["staleTickers"], [])

    def test_날짜가_없으면_낡은_것으로_본다(self) -> None:
        """빠진 값을 '최신' 으로 보면 그게 바로 조용한 붕괴다."""
        cache = {f"A{i}": {"issuer": "SPDR", "fetchedDate": ""} for i in range(5)}
        self.assertEqual(h.freshness(cache, self.TODAY)["broken"], ["SPDR"])

    def test_days_since(self) -> None:
        self.assertEqual(h.days_since("2026-08-10", self.TODAY), 2)
        self.assertIsNone(h.days_since("", self.TODAY))
        self.assertIsNone(h.days_since("어제", self.TODAY))
