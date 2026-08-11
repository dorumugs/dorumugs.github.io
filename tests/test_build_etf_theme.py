"""지표 계산과 등급 규칙 테스트.

실데이터로는 규칙이 다 안 밟힌다 — 오늘 장에 '눌림 매수' 가 한 종목도 없으면
그 가지는 한 번도 실행되지 않는다. 그래서 여기서는 합성 시계열로 각 규칙을
일부러 밟게 만든다. 경계값에서 정확히 갈리는지도 같이 본다.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import build_etf_theme as b  # noqa: E402


def line(start: float, step: float, n: int) -> list[float]:
    """매일 step 씩 곧게 오르는 종가."""
    return [start * (1 + step) ** i for i in range(n)]


def healthy_kwargs(**over):
    """'추세 진행' 이 나오는 정상 입력. 테스트마다 필요한 것만 덮어쓴다."""
    base = dict(
        bars_count=130,
        r20=0.10,
        r5=0.02,
        straight=0.8,
        straight_prior=0.8,
        vol=0.40,
        gap=0.05,
        dd=-0.01,
        turnover=100e8,
        beats_market=True,
        breadth_ok=True,
        leverage=1.0,
    )
    base.update(over)
    return base


class ReturnTest(unittest.TestCase):
    def test_20일_수익률은_21번째_전_종가와_비교한다(self):
        closes = [100.0] * 20 + [110.0]
        self.assertAlmostEqual(b.pct_return(closes, 20), 0.10)

    def test_데이터가_모자라면_None이다(self):
        self.assertIsNone(b.pct_return([100.0] * 10, 20))


class StraightnessTest(unittest.TestCase):
    def test_완벽한_직선은_1이다(self):
        self.assertAlmostEqual(b.straightness(line(100, 0.01, 21)), 1.0, places=3)

    def test_내려가면_0이다(self):
        # 기울기가 음수면 '곧게 내려간' 것이라도 0 을 준다. 이 지표는 상승
        # 추세의 품질을 재는 것이지 방향을 재는 게 아니다.
        self.assertEqual(b.straightness(line(100, -0.01, 21)), 0.0)

    def test_하루만_급등하고_제자리면_낮다(self):
        # 20일 수익률은 같아도 이쪽은 추세가 아니다.
        spike = [100.0] * 10 + [130.0] * 11
        steady = line(100, 0.0132, 21)
        self.assertAlmostEqual(spike[-1] / spike[0], steady[-1] / steady[0], places=1)
        self.assertLess(b.straightness(spike), b.straightness(steady))

    def test_평평하면_0이다(self):
        self.assertEqual(b.straightness([100.0] * 21), 0.0)


class StraightnessPriorTest(unittest.TestCase):
    def test_눌리기_전_추세를_본다(self):
        # 20일 곧게 오른 뒤 5일 쉬는 모양. 최근 창은 망가지지만 이전 창은 곧다.
        closes = line(100, 0.01, 21) + [121.0, 120.0, 119.0, 118.5, 118.0]
        self.assertGreater(b.straightness_prior(closes), 0.9)
        self.assertLess(b.straightness(closes[-21:]), b.straightness_prior(closes))

    def test_데이터가_26일_미만이면_0이다(self):
        self.assertEqual(b.straightness_prior(line(100, 0.01, 25)), 0.0)


class VolatilityTest(unittest.TestCase):
    def test_한결같이_오르면_변동성이_0에_가깝다(self):
        # CD금리 ETF 가 이 모양이다. 모멘텀 개념이 성립하지 않는다.
        self.assertLess(b.annualized_vol(line(100, 0.0001, 30)), 0.01)

    def test_출렁이면_커진다(self):
        zigzag = [100.0 * (1.05 if i % 2 else 0.95) for i in range(30)]
        self.assertGreater(b.annualized_vol(zigzag), 0.5)


class GapAndDrawdownTest(unittest.TestCase):
    def test_20일선_이격(self):
        self.assertAlmostEqual(b.ma_gap([100.0] * 19 + [110.0], 20), 110 / 100.5 - 1, places=4)

    def test_고점_대비는_항상_0_이하다(self):
        closes = list(range(80, 100)) + [90]
        self.assertLessEqual(b.drawdown([float(c) for c in closes], 20), 0)

    def test_신고가면_0이다(self):
        self.assertAlmostEqual(b.drawdown(line(100, 0.01, 20), 20), 0.0)


class TurnoverTest(unittest.TestCase):
    def test_하루짜리_대량거래에_속지_않는다(self):
        # 19일은 1억, 하루만 1000억. 평균은 50억이 넘지만 중앙값은 1억이다.
        closes = [100.0] * 20
        volumes = [1_000_000] * 19 + [1_000_000_000]
        self.assertAlmostEqual(b.median_turnover(closes, volumes, 20), 1e8)


class BreadthTest(unittest.TestCase):
    def test_양수_비율을_센다(self):
        self.assertAlmostEqual(b.breadth_of([0.1, 0.2, -0.1, -0.2]), 0.5)

    def test_None은_분모에서_빠진다(self):
        self.assertAlmostEqual(b.breadth_of([0.1, None, -0.1, None]), 0.5)

    def test_전부_None이면_None이다(self):
        self.assertIsNone(b.breadth_of([None, None]))


class GroupIndexTest(unittest.TestCase):
    def test_한_종목의_상한가에_끌려가지_않는다(self):
        # 아홉 종목은 제자리, 한 종목만 매일 +30%. 합산이면 지수가 치솟지만
        # 중위값이라 제자리여야 한다. 테마 대세를 개별 이슈와 가르는 핵심이다.
        calendar = ["1", "2", "3"]
        flat = {"1": 100.0, "2": 100.0, "3": 100.0}
        moon = {"1": 100.0, "2": 130.0, "3": 169.0}
        index = b.group_index(calendar, [dict(flat) for _ in range(9)] + [moon])
        self.assertAlmostEqual(index[-1], 100.0, places=6)

    def test_모두_오르면_지수도_오른다(self):
        calendar = ["1", "2"]
        rising = {"1": 100.0, "2": 110.0}
        index = b.group_index(calendar, [dict(rising) for _ in range(5)])
        self.assertAlmostEqual(index[-1], 110.0, places=6)


class GradeTest(unittest.TestCase):
    def test_거래일이_모자라면_판정불가다(self):
        grade, momentum, _ = b.grade_of(**healthy_kwargs(bars_count=15))
        self.assertEqual(grade, b.GRADE_UNKNOWN)
        self.assertEqual(momentum, b.GRADE_UNKNOWN)

    def test_금리형은_과열이나_추세보다_먼저_걸린다(self):
        # CD금리 ETF 는 R² 가 1.0 이라 순서를 잘못 두면 최고 등급을 받아버린다.
        grade, _, reasons = b.grade_of(**healthy_kwargs(vol=0.001, straight=1.0, r20=0.30, r5=-0.01))
        self.assertEqual(grade, b.GRADE_RATE)
        self.assertIn("금리형", reasons[0])

    def test_변동성_경계에서_갈린다(self):
        self.assertEqual(b.grade_of(**healthy_kwargs(vol=0.079))[0], b.GRADE_RATE)
        self.assertNotEqual(b.grade_of(**healthy_kwargs(vol=0.081))[0], b.GRADE_RATE)

    def test_거래대금_경계에서_갈린다(self):
        self.assertEqual(b.grade_of(**healthy_kwargs(turnover=4.9e8))[0], b.GRADE_ILLIQUID)
        self.assertNotEqual(b.grade_of(**healthy_kwargs(turnover=5.1e8))[0], b.GRADE_ILLIQUID)

    def test_유동성이_막아도_모멘텀_판단은_남는다(self):
        # '추세는 좋은데 거래가 안 되는 것' 과 '추세도 나쁜 것' 은 다르다.
        grade, momentum, _ = b.grade_of(**healthy_kwargs(turnover=1e8))
        self.assertEqual(grade, b.GRADE_ILLIQUID)
        self.assertEqual(momentum, b.GRADE_TREND)

    def test_급등_뒤_꺾이면_과열주의다(self):
        grade, _, _ = b.grade_of(**healthy_kwargs(r20=0.30, r5=-0.02))
        self.assertEqual(grade, b.GRADE_OVERHEAT)

    def test_2배_ETF는_과열선이_두배다(self):
        # 같은 +30% 라도 2배 ETF 는 기초자산이 15% 오른 것뿐이라 과열이 아니다.
        self.assertEqual(b.grade_of(**healthy_kwargs(r20=0.30, r5=-0.02))[0], b.GRADE_OVERHEAT)
        self.assertNotEqual(
            b.grade_of(**healthy_kwargs(r20=0.30, r5=-0.02, leverage=2.0))[0],
            b.GRADE_OVERHEAT,
        )
        self.assertEqual(
            b.grade_of(**healthy_kwargs(r20=0.55, r5=-0.02, leverage=2.0))[0],
            b.GRADE_OVERHEAT,
        )

    def test_곧게_오르다_쉬면_눌림매수다(self):
        grade, _, reasons = b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.05))
        self.assertEqual(grade, b.GRADE_PULLBACK)
        self.assertTrue(any("쉬는 중" in r for r in reasons))

    def test_눌림_이전_추세가_없으면_눌림매수가_아니다(self):
        # 그냥 흘러내리는 것과 쉬는 것을 가르는 조건이다.
        grade, _, _ = b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.05, straight_prior=0.1))
        self.assertNotEqual(grade, b.GRADE_PULLBACK)

    def test_너무_깊게_빠지면_눌림이_아니라_이탈이다(self):
        self.assertEqual(b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.05))[0], b.GRADE_PULLBACK)
        self.assertNotEqual(
            b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.30))[0], b.GRADE_PULLBACK
        )

    def test_아직_안_눌렸으면_눌림이_아니다(self):
        self.assertNotEqual(
            b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.01))[0], b.GRADE_PULLBACK
        )

    def test_시장을_못_넘으면_눌림도_추세도_아니다(self):
        for over in ({"beats_market": False}, {"breadth_ok": False}):
            with self.subTest(over=over):
                grade, _, _ = b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.05, **over))
                self.assertNotIn(grade, {b.GRADE_PULLBACK, b.GRADE_TREND})

    def test_흐름을_이어가면_추세진행이다(self):
        self.assertEqual(b.grade_of(**healthy_kwargs())[0], b.GRADE_TREND)

    def test_저점_반등은_이제_약세로_묶인다(self):
        # '바닥다지기' 를 뺐다. 가설 없이 만든 규칙이었고, 오르는 걸 산다는
        # 화면 전체의 전제와 모순됐다.
        grade, _, _ = b.grade_of(
            **healthy_kwargs(r20=-0.08, r5=0.03, gap=0.02, beats_market=False)
        )
        self.assertEqual(grade, b.GRADE_WEAK)
        self.assertFalse(hasattr(b, "GRADE_BASING"))

    def test_근거가_없으면_약세다(self):
        grade, _, reasons = b.grade_of(
            **healthy_kwargs(r20=-0.05, r5=-0.03, gap=-0.04, straight=0.1, beats_market=False)
        )
        self.assertEqual(grade, b.GRADE_WEAK)
        self.assertTrue(reasons)

    def test_모든_등급이_한번씩은_나온다(self):
        # 규칙 표에 있는데 영영 안 나오는 등급이 있으면 죽은 가지다.
        produced = {
            b.grade_of(**healthy_kwargs(bars_count=10))[0],
            b.grade_of(**healthy_kwargs(vol=0.01))[0],
            b.grade_of(**healthy_kwargs(turnover=1e8))[0],
            b.grade_of(**healthy_kwargs(r20=0.30, r5=-0.02))[0],
            b.grade_of(**healthy_kwargs(r5=-0.02, dd=-0.05))[0],
            b.grade_of(**healthy_kwargs())[0],
            b.grade_of(**healthy_kwargs(beats_market=False, r20=-0.05, r5=-0.03, gap=-0.02))[0],
        }
        self.assertEqual(produced, set(b.GRADE_ORDER))


class AtrTest(unittest.TestCase):
    def test_갭을_반영한다(self):
        # 고가-저가는 1인데 전날 종가에서 10 만큼 갭이 떴다. 갭을 무시하면
        # 하루 변동폭을 1 로 잘못 재고, 손절선이 터무니없이 좁아진다.
        closes = [100.0] * 15 + [110.0]
        highs = [100.5] * 15 + [110.5]
        lows = [99.5] * 15 + [109.5]
        value = b.atr(highs, lows, closes, span=1)
        self.assertAlmostEqual(value, 10.5, places=6)

    def test_데이터가_모자라면_None이다(self):
        self.assertIsNone(b.atr([1.0] * 5, [1.0] * 5, [1.0] * 5, span=14))


class StopLevelTest(unittest.TestCase):
    def test_ATR과_노이즈_중_더_먼_쪽을_쓴다(self):
        # 손절은 보유기간 노이즈 **바깥**에 있어야 한다. 안쪽에 두면 논지가
        # 깨져서가 아니라 평범한 출렁임에 걸린다.
        closes = [100.0] * 30
        # 2×ATR = 10 이 1σ(=8) 보다 멀다 → ATR 쪽
        self.assertAlmostEqual(b.stop_level(closes, 5.0, 0.08), 90.0)
        # 1σ(=12) 가 2×ATR(=4) 보다 멀다 → 노이즈 쪽
        self.assertAlmostEqual(b.stop_level(closes, 2.0, 0.12), 88.0)

    def test_기대변동폭을_모르면_ATR만_쓴다(self):
        self.assertAlmostEqual(b.stop_level([100.0] * 30, 5.0, None), 90.0)

    def test_손절이_현재가_위면_None이다(self):
        self.assertIsNone(b.stop_level([100.0] * 30, 0.0, 0.0))

    def test_예전_규칙보다_반드시_넓다(self):
        # 예전에는 max(10일 저점, 현재가-2ATR) 로 더 가까운 쪽을 골랐다. 새 규칙은
        # 어떤 입력에서도 그것보다 좁아지지 않아야 한다.
        closes = [100.0] * 30
        for atr_v, exp in ((5.0, 0.08), (2.0, 0.12), (1.0, 0.05)):
            with self.subTest(atr=atr_v, exp=exp):
                new = b.stop_level(closes, atr_v, exp)
                old_atr_only = 100.0 - 2 * atr_v
                self.assertLessEqual(new, old_atr_only + 1e-9)


class StopTouchProbabilityTest(unittest.TestCase):
    def test_1시그마면_약_32퍼센트다(self):
        # 반사원리: 경로 최솟값이 -1σ 아래로 갈 확률 = 2Φ(-1) ≈ 0.317
        self.assertAlmostEqual(b.stop_touch_probability(-0.06, 0.06), 0.3173, places=3)

    def test_예전_손절폭이면_절반이_걸린다(self):
        # 손절 4%, 2주 폭 6% → 0.67σ → 약 50%. 백테스트 관측(37~60%)과 맞는다.
        p = b.stop_touch_probability(-0.04, 0.06)
        self.assertGreater(p, 0.45)
        self.assertLess(p, 0.55)

    def test_넓을수록_낮아진다(self):
        wide = b.stop_touch_probability(-0.12, 0.06)
        narrow = b.stop_touch_probability(-0.03, 0.06)
        self.assertLess(wide, narrow)

    def test_기대변동폭이_0이면_None이다(self):
        self.assertIsNone(b.stop_touch_probability(-0.05, 0.0))
        self.assertIsNone(b.stop_touch_probability(None, 0.06))


class ExpectedMoveTest(unittest.TestCase):
    def test_날짜의_제곱근만큼_커진다(self):
        # 하루 변동이 일정할 때 10일 기대폭은 하루의 √10 배다. 10배가 아니다 —
        # 오르내림이 서로 상쇄되기 때문이다.
        closes = [100.0 * (1.02 if i % 2 else 0.98) ** 1 for i in range(30)]
        one = b.expected_move(closes, days=1)
        ten = b.expected_move(closes, days=10)
        self.assertAlmostEqual(ten / one, 10 ** 0.5, places=6)

    def test_움직이지_않으면_0이다(self):
        self.assertAlmostEqual(b.expected_move([100.0] * 30), 0.0)


class PremiumTest(unittest.TestCase):
    def test_NAV보다_비싸면_양수다(self):
        self.assertAlmostEqual(b.premium_of(101, 100), 0.01)

    def test_NAV가_없거나_0이면_None이다(self):
        self.assertIsNone(b.premium_of(101, 0))
        self.assertIsNone(b.premium_of(101, None))


class VolumeProfileTest(unittest.TestCase):
    def test_거래된_가격대에만_쌓인다(self):
        # 100~110 에서만 거래됐으면 그 아래위 칸은 비어 있어야 한다.
        highs = [110.0] * 5
        lows = [100.0] * 5
        vols = [1000] * 5
        centers, buckets = b.volume_profile(highs, lows, vols, span=5, bins=10)
        self.assertAlmostEqual(sum(buckets), 5000)
        self.assertTrue(all(100 <= c <= 110 for c in centers))

    def test_하루_거래량이_고가저가에_균등하게_퍼진다(self):
        # 장중 체결 분포가 없으니 균등 배분이 표준 근사다.
        centers, buckets = b.volume_profile([110.0], [100.0], [1000], span=1, bins=10)
        self.assertEqual(len(set(round(v, 6) for v in buckets)), 1)

    def test_좁게_거래된_날은_한_칸에_몰린다(self):
        highs = [100.0] * 4 + [200.0]
        lows = [100.0] * 4 + [100.0]
        vols = [1000] * 4 + [10]
        centers, buckets = b.volume_profile(highs, lows, vols, span=5, bins=10)
        # 100 근처 칸에 4000 이 몰려 최대여야 한다.
        peak = centers[buckets.index(max(buckets))]
        self.assertLess(peak, 120)

    def test_데이터가_없으면_None이다(self):
        self.assertIsNone(b.volume_profile([], [], []))
        self.assertIsNone(b.volume_profile([100.0], [100.0], [10]))


class ProfileStatsTest(unittest.TestCase):
    def make(self, highs, lows, vols):
        return b.volume_profile(highs, lows, vols, span=len(highs), bins=10)

    def test_위에_다_물려_있으면_overhead가_1에_가깝다(self):
        # 200~210 에서만 거래됐는데 지금 100 이면 전부 물려 있다.
        prof = self.make([210.0] * 5, [200.0] * 5, [1000] * 5)
        stats = b.profile_stats(prof, 100.0)
        self.assertAlmostEqual(stats["overhead"], 1.0, places=6)

    def test_아래에만_있으면_0이다(self):
        prof = self.make([110.0] * 5, [100.0] * 5, [1000] * 5)
        self.assertAlmostEqual(b.profile_stats(prof, 500.0)["overhead"], 0.0)

    def test_첫_저항은_현재가_위에서_가장_두꺼운_칸이다(self):
        # 100~105 에 얇게, 150~155 에 두껍게 쌓였다. 지금 110 이면 저항은 150대다.
        highs = [105.0] * 2 + [155.0] * 8
        lows = [100.0] * 2 + [150.0] * 8
        vols = [100] * 2 + [5000] * 8
        stats = b.profile_stats(self.make(highs, lows, vols), 110.0)
        self.assertGreater(stats["wall"], 145)
        self.assertGreater(stats["wallGap"], 0)

    def test_현재가_위에_아무것도_없으면_저항이_None이다(self):
        prof = self.make([110.0] * 5, [100.0] * 5, [1000] * 5)
        self.assertIsNone(b.profile_stats(prof, 500.0)["wall"])

    def test_매물대가_없으면_전부_None이다(self):
        empty = b.profile_stats(None, 100.0)
        self.assertTrue(all(v is None for v in empty.values()))


class HighestTest(unittest.TestCase):
    def test_구간_최고가를_고른다(self):
        self.assertEqual(b.highest([1.0, 9.0, 3.0, 5.0], 3), 9.0)

    def test_데이터가_짧으면_있는_만큼으로_잰다(self):
        self.assertEqual(b.highest([1.0, 4.0], 250), 4.0)

    def test_비어_있으면_None이다(self):
        self.assertIsNone(b.highest([], 20))


class MarketRegimeTest(unittest.TestCase):
    def test_둘_다_20일선_위면_순풍이다(self):
        up = line(100, 0.01, 25)
        self.assertEqual(b.market_regime(up, up)["label"], "순풍")

    def test_둘_다_아래면_역풍이다(self):
        down = line(100, -0.01, 25)
        self.assertEqual(b.market_regime(down, down)["label"], "역풍")

    def test_하나만_위면_엇갈림이고_어느_쪽인지_알려준다(self):
        up, down = line(100, 0.01, 25), line(100, -0.01, 25)
        r = b.market_regime(up, down)
        self.assertEqual(r["label"], "엇갈림")
        self.assertIn("코스피", r["note"])
        self.assertTrue(r["kospiAbove"])
        self.assertFalse(r["kosdaqAbove"])


class PercentileTest(unittest.TestCase):
    def test_최고값은_상위_100이다(self):
        self.assertEqual(b.percentile_rank(10, [1, 2, 3, 10]), 75.0)

    def test_빈_모집단은_0이다(self):
        self.assertEqual(b.percentile_rank(10, []), 0.0)


class StockHoldingTest(unittest.TestCase):
    def test_종목이_아닌_줄을_걸러낸다(self):
        # 이걸 분모에 넣으면 파서가 멀쩡해도 매칭률이 낮게 나온다.
        for name in ("원화현금", "설정현금액", "선물2026년09월물", "코스피위클리M C"):
            with self.subTest(name=name):
                self.assertFalse(b.is_stock_holding(name))
        for name in ("삼성전자", "LG에너지솔루션", "POSCO홀딩스"):
            with self.subTest(name=name):
                self.assertTrue(b.is_stock_holding(name))


if __name__ == "__main__":
    unittest.main()
