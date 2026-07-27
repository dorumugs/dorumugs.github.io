"""집계 빌드 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dashboard  # noqa: E402


def _trade(sgg="11680", umd="역삼동", jibun="755-1", apt="개나리푸르지오",
           area="84.0", price="100000", date="2026-06-27", cdeal="") -> dict:
    return {
        "sgg_cd": sgg, "umd_nm": umd, "jibun": jibun, "apt_name": apt,
        "apt_dong": "", "build_year": "2006", "area_sqm": area, "floor": "7",
        "price_10k": price, "trade_date": date, "deal_type": "중개거래",
        "seller_gbn": "개인", "buyer_gbn": "개인", "land_leasehold": "N",
        "cdeal_type": cdeal, "cdeal_day": "",
    }


class TestJoinHousehold(unittest.TestCase):
    def setUp(self) -> None:
        self.complexes = {
            "1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                    "hh": 332, "sgg": "11680"},
        }
        self.by_name = {("11680", "역삼동", "개나리푸르지오"): 332}

    def test_matches_by_pnu(self) -> None:
        got = build_dashboard.join_household(_trade(), self.complexes, self.by_name)
        self.assertEqual(got, 332)

    def test_falls_back_to_name_when_pnu_missing(self) -> None:
        row = _trade(jibun="9999-9999")  # 마스터에 없는 지번
        got = build_dashboard.join_household(row, self.complexes, self.by_name)
        self.assertEqual(got, 332)

    def test_unknown_returns_none(self) -> None:
        row = _trade(apt="없는단지", jibun="9999-9999")
        self.assertIsNone(build_dashboard.join_household(row, self.complexes, self.by_name))


class TestBuildSummary(unittest.TestCase):
    def _run(self, by_month: dict[str, list[dict]], hh_lookup=None) -> dict:
        complexes = {"1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                             "hh": 332, "sgg": "11680"}}
        by_name = {("11680", "역삼동", "개나리푸르지오"): 332}
        return build_dashboard.build_summary(
            by_month, sorted(by_month), complexes, by_name,
            {"11680": "강남구"}, generated="2026-07-26")

    def test_median_is_pyeong_price_rounded(self) -> None:
        out = self._run({"2026-06": [_trade(area="84.0", price="100000")]})
        # 100000 / (84/3.3058) = 3935.476... -> 3935
        self.assertEqual(out["series"]["all"]["11680"]["med"][0], 3935)

    def test_cancelled_trade_excluded_from_price_but_counted(self) -> None:
        rows = [_trade(price="100000"), _trade(price="900000", cdeal="O")]
        out = self._run({"2026-06": rows})
        s = out["series"]["all"]["11680"]
        self.assertEqual(s["med"][0], 3935)   # 해제 건이 중위값을 흔들지 않는다
        self.assertEqual(s["n"][0], 1)        # 유효 거래만 센다
        self.assertEqual(s["cancel"][0], 1)

    def test_household_filter_splits_series(self) -> None:
        rows = [_trade(), _trade(apt="없는단지", jibun="9999-9999", price="50000")]
        out = self._run({"2026-06": rows})
        self.assertEqual(out["series"]["300"]["11680"]["n"][0], 1)
        self.assertEqual(out["series"]["all"]["11680"]["n"][0], 2)

    def test_month_with_no_trades_is_null(self) -> None:
        out = self._run({"2026-05": [], "2026-06": [_trade()]})
        self.assertIsNone(out["series"]["all"]["11680"]["med"][0])
        self.assertIsNotNone(out["series"]["all"]["11680"]["med"][1])

    def test_zero_area_row_dropped(self) -> None:
        out = self._run({"2026-06": [_trade(area="0")]})
        self.assertIsNone(out["series"]["all"]["11680"]["med"][0])

    def test_last_month_marked_partial(self) -> None:
        out = self._run({"2026-05": [_trade()], "2026-06": [_trade()]})
        self.assertEqual(out["partial"], "2026-06")

    def test_households_summed_per_filter(self) -> None:
        out = self._run({"2026-06": [_trade()]})
        self.assertEqual(out["sgg"]["11680"]["hh"]["300"], 332)

    def test_cancel_present_and_full_length_for_every_filter(self) -> None:
        # '300' 필터는 해제 건을 별도로 세지 않아 배열 값이 전부 0이지만,
        # 키 자체는 두 필터 모두 항상 있어야 한다 — 프런트가 조건 없이
        # series[filter][sgg].cancel 을 읽을 수 있어야 하기 때문.
        rows = [_trade(price="100000"), _trade(price="900000", cdeal="O")]
        out = self._run({"2026-05": [], "2026-06": rows})
        for f in ("300", "all"):
            s = out["series"][f]["11680"]
            self.assertIn("cancel", s)
            self.assertEqual(len(s["cancel"]), len(out["months"]))
        self.assertEqual(out["series"]["all"]["11680"]["cancel"], [0, 1])
        self.assertEqual(out["series"]["300"]["11680"]["cancel"], [0, 0])


class TestWriteJson(unittest.TestCase):
    def test_skips_write_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            payload = {"b": 2, "a": 1}
            self.assertTrue(build_dashboard.write_json(p, payload))
            self.assertFalse(build_dashboard.write_json(p, payload))

    def test_output_is_key_sorted_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            build_dashboard.write_json(p, {"b": 2, "a": 1})
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.startswith('{"a":1,"b":2}'))

    def test_same_payload_two_orders_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            build_dashboard.write_json(a, {"x": 1, "y": 2})
            build_dashboard.write_json(b, {"y": 2, "x": 1})
            self.assertEqual(a.read_bytes(), b.read_bytes())


class TestBuildSggDetail(unittest.TestCase):
    def setUp(self) -> None:
        self.by_pnu = {"1168010100107550001": {"name": "개나리푸르지오", "dong": "역삼동",
                                               "hh": 332, "sgg": "11680"}}
        self.by_name = {("11680", "역삼동", "개나리푸르지오"): 332}
        self.months = ["2026-05", "2026-06", "2026-07"]

    def _run(self, by_month: dict[str, list[dict]], window: int = 2) -> dict:
        return build_dashboard.build_sgg_detail(
            by_month, self.months, window, self.by_pnu, self.by_name, "11680")

    def test_only_window_months_counted(self) -> None:
        by_month = {
            "2026-05": [_trade(price="100000")],   # 창 밖
            "2026-06": [_trade(price="200000")],
            "2026-07": [_trade(price="200000")],
        }
        out = self._run(by_month, window=2)
        self.assertEqual(out["window"], ["2026-06", "2026-07"])
        self.assertEqual(out["complexes"][0]["n"], 2)

    def test_median_over_window(self) -> None:
        by_month = {"2026-06": [_trade(price="100000")],
                    "2026-07": [_trade(price="300000")]}
        out = self._run(by_month, window=2)
        # (100000 + 300000)/2 규모의 평당가 중위 = 두 값의 평균
        lo = 100_000 / (84.0 / 3.3058)
        hi = 300_000 / (84.0 / 3.3058)
        self.assertEqual(out["complexes"][0]["med"], round((lo + hi) / 2))

    def test_area_buckets_counted(self) -> None:
        by_month = {"2026-07": [_trade(area="59.0"), _trade(area="84.0"),
                                _trade(area="84.0"), _trade(area="200.0")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"][0]["bk"], [1, 2, 0, 1])

    def test_cancelled_excluded(self) -> None:
        by_month = {"2026-07": [_trade(), _trade(cdeal="O")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"][0]["n"], 1)

    def test_unmatched_complex_has_null_households(self) -> None:
        by_month = {"2026-07": [_trade(apt="없는단지", jibun="9999-9999")]}
        out = self._run(by_month, window=2)
        self.assertIsNone(out["complexes"][0]["hh"])

    def test_other_districts_excluded(self) -> None:
        by_month = {"2026-07": [_trade(sgg="11110", umd="교북동", apt="경희궁자이")]}
        out = self._run(by_month, window=2)
        self.assertEqual(out["complexes"], [])

    def test_sorted_by_median_desc_then_name(self) -> None:
        by_month = {"2026-07": [
            _trade(apt="싼단지", jibun="1-1", price="50000"),
            _trade(apt="비싼단지", jibun="2-2", price="500000"),
        ]}
        out = self._run(by_month, window=2)
        self.assertEqual([c["name"] for c in out["complexes"]], ["비싼단지", "싼단지"])


class TestAgainstRealData(unittest.TestCase):
    """실제 원본으로 만든 summary.json 이 있을 때만 도는 대조 테스트.

    설계 문서 작성 시 원본에서 직접 계산한 강남구 값과 맞춰 본다.
    """

    OUT = ROOT / "assets" / "realestate" / "summary.json"

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_gangnam_matches_measured_values(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        months = data["months"]
        med = data["series"]["all"]["11680"]["med"]
        # 마감된 달만 정확값으로 못박는다. 손으로 원본에서 계산한 대조값이다.
        self.assertEqual(med[months.index("2006-01")], 2599)

        # 최근 3개월은 신고 지연으로 계속 움직인다 (프로덕션의 settling 창과 같다).
        # 값을 못박으면 수집이 돌 때마다 테스트가 깨지고, 깨진 값을 결과물에서 읽어
        # 갱신하는 순간 '원본에서 손으로 계산한 값과 맞춰 본다'는 대조 의미 자체가
        # 사라진다 — 파이프라인이 제 지난 출력을 재현하는지만 확인하게 된다.
        # 그래서 열려 있는 달은 존재와 타당성만 본다.
        for ym in months[-3:]:
            value = med[months.index(ym)]
            self.assertIsNotNone(value, f"{ym} 중위 평당가가 비어 있다")
            self.assertGreater(value, 0)

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_covers_all_regions_and_months(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        self.assertEqual(len(data["sgg"]), 72)
        self.assertEqual(data["months"][0], "2006-01")
        # 개수를 못박으면 달이 하나 넘어가는 순간 깨진다. 2006-01 부터 마지막 달까지
        # 빠짐없이 이어지는지를 대신 본다 — 중간에 구멍이 나는 것이 진짜 문제다.
        first, last = data["months"][0], data["months"][-1]
        y0, m0 = (int(x) for x in first.split("-"))
        y1, m1 = (int(x) for x in last.split("-"))
        self.assertEqual(len(data["months"]), (y1 - y0) * 12 + (m1 - m0) + 1)

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_cancel_present_for_every_sgg_and_filter(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        n_months = len(data["months"])
        for f in ("300", "all"):
            for sgg, s in data["series"][f].items():
                self.assertIn("cancel", s, f"{f}/{sgg} 에 cancel 키가 없음")
                self.assertEqual(len(s["cancel"]), n_months, f"{f}/{sgg}")

    @unittest.skipUnless(OUT.exists(), "summary.json 없음 — 먼저 빌드하세요")
    def test_within_size_budget(self) -> None:
        self.assertLess(self.OUT.stat().st_size, 400 * 1024)

    SGG_DIR = ROOT / "assets" / "realestate" / "sgg"

    @unittest.skipUnless(SGG_DIR.exists(), "구별 JSON 없음 — 먼저 빌드하세요")
    def test_every_region_has_a_detail_file(self) -> None:
        files = sorted(p.stem for p in self.SGG_DIR.glob("*.json"))
        self.assertEqual(len(files), 72)

    @unittest.skipUnless(SGG_DIR.exists(), "구별 JSON 없음 — 먼저 빌드하세요")
    def test_detail_files_within_size_budget(self) -> None:
        oversized = [(p.name, p.stat().st_size) for p in self.SGG_DIR.glob("*.json")
                     if p.stat().st_size > 300 * 1024]
        self.assertEqual(oversized, [])


if __name__ == "__main__":
    unittest.main()
