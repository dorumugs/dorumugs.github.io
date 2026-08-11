"""백테스트 채점 로직 테스트.

여기서 틀리면 성적표 전체가 조용히 거짓말을 한다. 특히 표본을 겹치게 세거나
손절을 관대하게 매기면 결과가 실제보다 좋게 나온다.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import backtest_etf as bt  # noqa: E402


class EvaluationIndexTest(unittest.TestCase):
    def test_관측이_겹치지_않는다(self):
        # step 을 horizon 과 같게 두면 앞 관측의 결과 구간이 다음 관측 시점과
        # 맞닿기만 하고 겹치지 않는다. 겹치면 표본이 실제보다 열 배 많아 보인다.
        idx = bt.evaluation_indexes(200, min_history=30, horizon=10, step=10)
        self.assertTrue(all(b - a == 10 for a, b in zip(idx, idx[1:])))

    def test_앞으로_볼_구간이_없으면_안_넣는다(self):
        idx = bt.evaluation_indexes(200, min_history=30, horizon=10, step=10)
        self.assertLess(max(idx) + 10, 200)

    def test_과거가_모자라면_안_넣는다(self):
        idx = bt.evaluation_indexes(200, min_history=30, horizon=10, step=10)
        self.assertGreaterEqual(min(idx), 30)

    def test_데이터가_짧으면_빈_목록이다(self):
        self.assertEqual(bt.evaluation_indexes(35, min_history=30, horizon=10), [])


class ForwardReturnTest(unittest.TestCase):
    def test_10일_뒤와_비교한다(self):
        closes = [100.0] * 10 + [110.0]
        self.assertAlmostEqual(bt.forward_return(closes, 0, horizon=10), 0.10)

    def test_앞이_모자라면_None이다(self):
        self.assertIsNone(bt.forward_return([100.0] * 5, 0, horizon=10))


class StopOutcomeTest(unittest.TestCase):
    def test_저가가_손절선에_닿으면_거기서_나온다(self):
        # 종가는 끝에 올라가지만 도중에 손절선을 찍었다. 종가만 보면 이겼다고
        # 세는데, 실제로는 손절에 털린 뒤였다.
        closes = [100.0] + [100.0] * 9 + [120.0]
        lows = [100.0] + [89.0] + [100.0] * 8 + [120.0]
        hit, realized = bt.stop_outcome(lows, closes, 0, stop=90.0, horizon=10)
        self.assertTrue(hit)
        self.assertAlmostEqual(realized, -0.10)

    def test_안_닿으면_그냥_보유_수익이다(self):
        closes = [100.0] * 10 + [110.0]
        lows = [95.0] * 11
        hit, realized = bt.stop_outcome(lows, closes, 0, stop=90.0, horizon=10)
        self.assertFalse(hit)
        self.assertAlmostEqual(realized, 0.10)

    def test_손절선이_없으면_걸리지_않은_것으로_본다(self):
        closes = [100.0] * 10 + [110.0]
        hit, realized = bt.stop_outcome([1.0] * 11, closes, 0, stop=None, horizon=10)
        self.assertFalse(hit)
        self.assertAlmostEqual(realized, 0.10)

    def test_당일_저가는_보지_않는다(self):
        # 진입일 저가는 이미 지나간 값이라 손절 대상이 아니다.
        closes = [100.0] * 10 + [110.0]
        lows = [50.0] + [99.0] * 10
        hit, _ = bt.stop_outcome(lows, closes, 0, stop=90.0, horizon=10)
        self.assertFalse(hit)


class SummarizeTest(unittest.TestCase):
    def make(self, fwds, excess=None, stops=None):
        return [
            bt.Trial(
                fwd=f,
                excess=(excess[i] if excess else None),
                stopHit=(stops[i] if stops else False),
                realized=f,
            )
            for i, f in enumerate(fwds)
        ]

    def test_표본이_적으면_숫자를_내지_않는다(self):
        # 없는 근거를 만들지 않는다. 관측 다섯 개로 승률 60% 를 쓰면 안 된다.
        s = bt.summarize(self.make([0.01] * 5))
        self.assertTrue(s["thin"])
        self.assertNotIn("winRate", s)

    def test_아예_없으면_None이다(self):
        self.assertIsNone(bt.summarize([]))

    def test_승률과_중위값을_센다(self):
        fwds = [0.01] * 60 + [-0.01] * 40
        s = bt.summarize(self.make(fwds))
        self.assertEqual(s["n"], 100)
        self.assertAlmostEqual(s["winRate"], 0.60)
        self.assertFalse(s["thin"])

    def test_손절_걸린_비율을_센다(self):
        stops = [True] * 30 + [False] * 70
        s = bt.summarize(self.make([0.01] * 100, stops=stops))
        self.assertAlmostEqual(s["stopHitRate"], 0.30)

    def test_시장초과가_없으면_None으로_둔다(self):
        s = bt.summarize(self.make([0.01] * 50))
        self.assertIsNone(s["medianExcess"])
        self.assertIsNone(s["beatRate"])


if __name__ == "__main__":
    unittest.main()
