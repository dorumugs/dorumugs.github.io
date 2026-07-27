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

import build_geo  # noqa: E402
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


class TestCrossModuleProjection(unittest.TestCase):
    """build_schools.to_svg_xy 와 build_geo.project 가 같은 투영식을 쓰는지 묶어 검증한다.

    두 모듈은 각자 파일에 같은 산식을 손으로 다시 적어 뒀다(build_schools.py 의
    to_svg_xy, build_geo.py 의 project). test_geo.py 는 build_geo.py 만, 이 파일의
    TestToSvgXy 는 build_schools.py 만 각자 자기 공식을 되풀이해 검증할 뿐 서로
    대조하지 않는다 — build_geo.py 의 식을 바꾸고 build_schools.py 를 그대로 두면
    어느 쪽도 실패하지 않는다. 이 테스트가 그 틈을 잇는다: build_geo.projection_params
    가 낸 파라미터로 build_schools.to_svg_xy 가 계산한 좌표가, 같은 점을
    build_geo.project 로 투영한 좌표와 정확히 같아야 한다.
    """

    # (위도, 경도). 서울시청은 설계 문서(2026-07-27-school-map-design.md)의
    # 투영 일치 검증에 쓰인 좌표다.
    POINTS = [
        (37.5663, 126.9779),  # 서울시청
        (37.0, 126.0),
        (38.0, 127.0),
        (37.25, 126.75),
        (37.9, 126.1),
    ]
    # project() 는 링이 기하학적으로 유효한 폴리곤인지 따지지 않고 점마다 독립적으로
    # 투영한다 — 검증하려는 점들을 그대로 링 하나에 담아 project() 를 통과시킨다.
    RING = [[lon, lat] for lat, lon in POINTS]
    RINGS = {"11680": [RING]}
    WIDTH = 1000.0

    def test_to_svg_xy_matches_build_geo_project(self) -> None:
        params = build_geo.projection_params(self.RINGS, self.WIDTH)
        projected, _, _ = build_geo.project(self.RINGS, self.WIDTH)
        got_points = projected["11680"][0]
        self.assertEqual(len(got_points), len(self.POINTS))
        for (lat, lon), (px, py) in zip(self.POINTS, got_points):
            x, y = build_schools.to_svg_xy(lat, lon, params)
            self.assertAlmostEqual(x, px, places=9, msg=f"x mismatch at ({lat}, {lon})")
            self.assertAlmostEqual(y, py, places=9, msg=f"y mismatch at ({lat}, {lon})")


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

    def test_unspaced_mountain_parcel_stripped(self) -> None:
        """'산26-127' 처럼 '산' 과 번지가 붙어 있으면 마지막 토큰이 숫자로
        시작하지도, '산' 그 자체이지도 않아 놓치기 쉽다. 실제 조인 실패 사례
        (명지초등학교)와 같은 모양이다."""
        self.assertEqual(build_schools.parse_addr("서울특별시 서대문구 홍은동 산26-127"),
                         ("서울특별시 서대문구", "홍은동"))

    def test_spaced_mountain_parcel_stripped(self) -> None:
        """'산 26-127' 처럼 '산' 과 번지가 띄어 쓰인 경우도 같이 떨어져야 한다."""
        self.assertEqual(build_schools.parse_addr("서울특별시 마포구 성산동 산 11-31"),
                         ("서울특별시 마포구", "성산동"))

    def test_plain_hyphenated_jibun_stripped(self) -> None:
        """'산' 없는 일반 하이픈 지번(100-4)도 그대로 떨어져야 한다."""
        self.assertEqual(build_schools.parse_addr("서울특별시 금천구 시흥동 100-4"),
                         ("서울특별시 금천구", "시흥동"))

    def test_eup_myeon_ri_with_mountain_parcel(self) -> None:
        """읍면리 두 토큰 접미사와 산 지번 제거가 함께 작동해야 한다."""
        self.assertEqual(build_schools.parse_addr("경기도 여주시 가남읍 태평리 산12-3"),
                         ("경기도 여주시", "가남읍 태평리"))


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


class TestBuildDeterminism(unittest.TestCase):
    """build() 는 순수 함수라 같은 입력을 몇 번 빌드해도 바이트가 같아야 한다.

    이전엔 좌표를 밀어내는 fan-out 단계가 있었고 이 테스트가 그 결정론을
    같이 검증했다. fan-out 은 되돌렸다(1 SVG 단위가 약 130m 라, 겹친 점을
    벌리는 게 최대 3.5km 까지 실제 좌표를 왜곡했다 — "이 학교 근처 집은
    얼마인가" 를 답하는 도구에서 학교 위치 자체가 틀리면 안 된다). 좌표는 이제
    to_svg_xy() 의 순수 투영 결과 그대로이므로, 결정론은 여전히 지켜야 할
    성질로 남아 이 테스트를 그대로 유지한다.
    """

    def test_two_runs_produce_identical_bytes(self) -> None:
        rows = [_school(school_id=str(i), school_name=f"{chr(65 + i)}초등학교")
                for i in range(6)]
        out1 = build_schools.build(rows, PARAMS, generated="2026-07-27")
        out2 = build_schools.build(rows, PARAMS, generated="2026-07-27")
        dump = lambda o: json.dumps(  # noqa: E731
            o, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(dump(out1), dump(out2))


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
