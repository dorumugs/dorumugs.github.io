"""학교 집계 빌드 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_schools  # noqa: E402

PARAMS = {"min_lon": 126.0, "max_lat": 38.0, "k": 0.8,
          "span_x": 1.6, "span_y": 1.2, "width": 1000.0, "height": 750.0}


def _school(**kw) -> dict:
    base = {
        "school_id": "S1", "school_name": "계성초등학교", "level": "초등학교",
        "found_type": "사립", "addr": "서울특별시 서초구 내곡동 1",
        "road_addr": "", "sido_office": "서울특별시교육청",
        "lat": "37.5", "lon": "127.0",
    }
    base.update(kw)
    return base


class TestToSvgXy(unittest.TestCase):
    def test_matches_projection_formula(self) -> None:
        x, y = build_schools.to_svg_xy(37.5, 127.0, PARAMS)
        self.assertAlmostEqual(x, (127.0 - 126.0) * 0.8 / 1.6 * 1000.0, places=6)
        self.assertAlmostEqual(y, (38.0 - 37.5) / 1.2 * 750.0, places=6)

    def test_north_is_up(self) -> None:
        _, y_north = build_schools.to_svg_xy(37.9, 127.0, PARAMS)
        _, y_south = build_schools.to_svg_xy(37.1, 127.0, PARAMS)
        self.assertLess(y_north, y_south)


class TestParseAddr(unittest.TestCase):
    def test_seoul_two_tokens(self) -> None:
        self.assertEqual(build_schools.parse_addr("서울특별시 강남구 대치동 123"),
                         ("서울특별시 강남구", "대치동"))

    def test_gyeonggi_with_gu_takes_longest_match(self) -> None:
        """'경기도 성남시' 로 먼저 맞으면 분당구를 놓친다. 긴 이름이 이겨야 한다."""
        self.assertEqual(build_schools.parse_addr("경기도 성남시 분당구 서현동 100"),
                         ("경기도 성남시 분당구", "서현동"))

    def test_gyeonggi_without_gu(self) -> None:
        self.assertEqual(build_schools.parse_addr("경기도 광명시 하안동 50"),
                         ("경기도 광명시", "하안동"))

    def test_eup_myeon_ri_keeps_full_suffix(self) -> None:
        """읍면 지역은 '가남읍 태평리' 처럼 두 토큰이다. 접미사 전체를 넘겨야 한다."""
        self.assertEqual(build_schools.parse_addr("경기도 여주시 가남읍 태평리 12-3"),
                         ("경기도 여주시", "가남읍 태평리"))

    def test_unknown_region_is_none(self) -> None:
        self.assertIsNone(build_schools.parse_addr("부산광역시 해운대구 우동 1"))


class TestSelectPrivateElementary(unittest.TestCase):
    def test_keeps_only_private_elementary(self) -> None:
        rows = [
            _school(school_id="A"),
            _school(school_id="B", found_type="공립"),
            _school(school_id="C", level="중학교", found_type="사립"),
            _school(school_id="D", found_type="국립"),
        ]
        got = build_schools.select_private_elementary(rows)
        self.assertEqual([r["school_id"] for r in got], ["A"])


class TestBuild(unittest.TestCase):
    def _run(self, rows: list[dict]) -> dict:
        return build_schools.build(rows, PARAMS, generated="2026-07-27")

    def test_emits_expected_shape(self) -> None:
        out = self._run([_school()])
        self.assertEqual(out["generated"], "2026-07-27")
        s = out["schools"][0]
        self.assertEqual(s["name"], "계성초등학교")
        self.assertEqual(s["lvl"], "초")
        self.assertEqual(s["found"], "사립")
        self.assertEqual(s["dong"], "내곡동")
        self.assertEqual(s["sgg"], "11650")
        self.assertEqual(s["dong_cd"], "1165010900")

    def test_coordinates_rounded_to_one_decimal(self) -> None:
        s = self._run([_school()])["schools"][0]
        self.assertEqual(s["x"], round(s["x"], 1))
        self.assertEqual(s["y"], round(s["y"], 1))

    def test_unresolvable_address_dropped(self) -> None:
        out = self._run([_school(addr="부산광역시 해운대구 우동 1")])
        self.assertEqual(out["schools"], [])

    def test_sorted_by_sgg_then_name(self) -> None:
        """sgg 코드 오름차순이 1차 키다. 종로구(11110)가 강남구(11680)보다
        코드값이 작아 이름 자모순과 무관하게 먼저 온다."""
        rows = [
            _school(school_id="A", school_name="나초등학교", addr="서울특별시 강남구 대치동 1"),
            _school(school_id="B", school_name="가초등학교", addr="서울특별시 강남구 대치동 2"),
            _school(school_id="C", school_name="다초등학교", addr="서울특별시 종로구 청운동 3"),
        ]
        got = [s["name"] for s in self._run(rows)["schools"]]
        self.assertEqual(got, ["다초등학교", "가초등학교", "나초등학교"])


class TestAgainstRealOutput(unittest.TestCase):
    OUT = ROOT / "assets" / "realestate" / "schools.json"

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_all_points_inside_viewbox(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        for s in data["schools"]:
            self.assertGreaterEqual(s["x"], 0.0, s["name"])
            self.assertLessEqual(s["x"], 1000.0, s["name"])
            self.assertGreaterEqual(s["y"], 0.0, s["name"])
            self.assertLessEqual(s["y"], 1201.0, s["name"])

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_only_private_elementary(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        self.assertTrue(data["schools"])
        for s in data["schools"]:
            self.assertEqual(s["lvl"], "초")
            self.assertEqual(s["found"], "사립")

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_within_size_budget(self) -> None:
        self.assertLess(self.OUT.stat().st_size, 100 * 1024)


if __name__ == "__main__":
    unittest.main()
