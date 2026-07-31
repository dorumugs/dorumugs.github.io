"""재개발·재건축 집계 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_redevelopment as br  # noqa: E402


class TestMonthAdd(unittest.TestCase):
    def test_같은_해(self):
        self.assertEqual(br.month_add("2020-08", 1), "2020-09")
        self.assertEqual(br.month_add("2020-08", -1), "2020-07")

    def test_해를_넘는다(self):
        self.assertEqual(br.month_add("2020-08", 12), "2021-08")
        self.assertEqual(br.month_add("2020-01", -1), "2019-12")
        self.assertEqual(br.month_add("2020-12", 1), "2021-01")

    def test_12개월_창(self):
        self.assertEqual(br.month_add("2007-11", -12), "2006-11")
        self.assertEqual(br.month_add("2007-11", 12), "2008-11")


class TestImpliedFar(unittest.TestCase):
    def test_은마_수준(self):
        # 4,424세대 · 전용 중위 84㎡ · 대지 239,226㎡ → 실제 알려진 값(약 200%) 언저리
        far = br.implied_far(4424, 84.0, 239225.8)
        self.assertIsNotNone(far)
        self.assertTrue(180 <= far <= 220, far)

    def test_값이_없으면_None(self):
        self.assertIsNone(br.implied_far(0, 84.0, 1000))
        self.assertIsNone(br.implied_far(100, None, 1000))
        self.assertIsNone(br.implied_far(100, 84.0, 0))


class TestLandSharePyeong(unittest.TestCase):
    def test_은마_대지지분(self):
        # 239,225.8㎡ ÷ 4,424세대 = 16.4평. 알려진 값과 맞는다.
        share = br.land_share_pyeong(239225.8, 4424, far=189.0)
        self.assertAlmostEqual(share, 16.4, places=1)

    def test_역산_용적률이_말이_안_되면_감춘다(self):
        # 56세대에 63,321㎡ — 등록 필지가 단지 조각과 어긋난 실제 사례.
        # 이 조합은 역산 용적률이 10% 언저리라 대지지분을 내면 안 된다.
        self.assertIsNone(br.land_share_pyeong(63321.0, 56, far=11.0))

    def test_역산_용적률이_너무_높아도_감춘다(self):
        self.assertIsNone(br.land_share_pyeong(300.0, 234, far=2000.0))

    def test_검증을_못_하면_감춘다(self):
        # 거래가 없어 역산이 불가능한 단지. 검증 못 한 값을 섞지 않는다.
        self.assertIsNone(br.land_share_pyeong(50000.0, 1000, far=None))

    def test_경계값(self):
        self.assertIsNotNone(br.land_share_pyeong(50000.0, 1000, far=br.IMPLIED_FAR_MIN))
        self.assertIsNotNone(br.land_share_pyeong(50000.0, 1000, far=br.IMPLIED_FAR_MAX))
        self.assertIsNone(br.land_share_pyeong(50000.0, 1000, far=br.IMPLIED_FAR_MIN - 0.1))
        self.assertIsNone(br.land_share_pyeong(50000.0, 1000, far=br.IMPLIED_FAR_MAX + 0.1))

    def test_입력이_비면_None(self):
        self.assertIsNone(br.land_share_pyeong(0, 100, far=200.0))
        self.assertIsNone(br.land_share_pyeong(1000, 0, far=200.0))


class TestExcessReturn(unittest.TestCase):
    def test_대조군을_뺀다(self):
        # 단지 +20%, 자치구 +12% → 초과 +8%p
        self.assertAlmostEqual(br.excess_return(100, 120, 100, 112), 8.0, places=6)

    def test_시장이_더_오르면_음수(self):
        self.assertAlmostEqual(br.excess_return(100, 105, 100, 115), -10.0, places=6)

    def test_한쪽이라도_없으면_None(self):
        self.assertIsNone(br.excess_return(None, 120, 100, 112))
        self.assertIsNone(br.excess_return(100, 120, None, 112))
        self.assertIsNone(br.excess_return(100, 120, 0, 112))


class TestWindowMedian(unittest.TestCase):
    def setUp(self):
        self.series = {
            "2020-01": [100.0, 200.0],
            "2020-02": [300.0],
            "2020-06": [999.0],
        }

    def test_구간_안만_센다(self):
        value, n = br.window_median(self.series, "2020-01", "2020-02")
        self.assertEqual(n, 3)
        self.assertEqual(value, 200.0)

    def test_구간_밖은_제외(self):
        value, n = br.window_median(self.series, "2020-06", "2020-06")
        self.assertEqual((value, n), (999.0, 1))

    def test_비면_None(self):
        self.assertEqual(br.window_median(self.series, "2021-01", "2021-12"), (None, 0))


class TestDerivePropelLabels(unittest.TestCase):
    def test_구역명으로_단계를_되짚는다(self):
        zones = [
            {"propel_cd": "PP0206", "sgg_cd": "11620", "zone_name_norm": "봉천14"},
            {"propel_cd": "PP0206", "sgg_cd": "11620", "zone_name_norm": "봉천15"},
        ]
        projects = [
            {"sgg_nm": "관악구", "name": "봉천14구역 주택재개발정비사업조합", "stage": "조합설립인가"},
            {"sgg_nm": "관악구", "name": "봉천15구역 주택재개발정비사업조합", "stage": "조합설립인가"},
        ]
        labels = br.derive_propel_labels(zones, projects)
        self.assertEqual(labels["PP0206"]["label"], "조합설립인가")
        self.assertEqual(labels["PP0206"]["hits"], 2)

    def test_못_붙이면_코드가_빠진다(self):
        zones = [{"propel_cd": "PP9999", "sgg_cd": "11620", "zone_name_norm": "없는구역"}]
        self.assertEqual(br.derive_propel_labels(zones, []), {})


if __name__ == "__main__":
    unittest.main()
