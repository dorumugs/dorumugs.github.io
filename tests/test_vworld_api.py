"""브이월드 데이터 API 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import vworld_api  # noqa: E402


def _ok(features: list) -> dict:
    return {
        "response": {
            "status": "OK",
            "result": {"featureCollection": {"features": features}},
        }
    }


def _square(lon: float, lat: float, d: float) -> dict:
    """한 변이 d 도인 정사각형 폴리곤."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat],
        ]],
    }


class TestStatusHandling(unittest.TestCase):
    def test_자료없음은_오류가_아니다(self):
        payload = {"response": {"status": "NOT_FOUND", "record": {"total": "0"}}}
        self.assertIsNone(vworld_api.parse_parcel(payload))
        self.assertIsNone(vworld_api.parse_landuse(payload))

    def test_오류는_VworldError(self):
        payload = {
            "response": {
                "status": "ERROR",
                "error": {"code": "INVALID_KEY", "text": "등록되지 않은 인증키입니다."},
            }
        }
        with self.assertRaises(vworld_api.VworldError):
            vworld_api.parse_parcel(payload)


class TestArea(unittest.TestCase):
    def test_위도_보정을_한다(self):
        # 같은 각도 크기라도 위도가 높을수록 동서 길이가 짧아 면적이 작아진다.
        low = vworld_api.polygon_area_sqm(_square(127.0, 10.0, 0.001))
        high = vworld_api.polygon_area_sqm(_square(127.0, 60.0, 0.001))
        self.assertGreater(low, high)

    def test_알려진_크기와_맞는다(self):
        # 위도 37.5 에서 0.001도 정사각형: 남북 약 111.2m, 동서 약 88.2m → 약 9,800㎡
        area = vworld_api.polygon_area_sqm(_square(127.0, 37.5, 0.001))
        self.assertAlmostEqual(area, 9810, delta=200)

    def test_MultiPolygon_도_읽는다(self):
        square = _square(127.0, 37.5, 0.001)["coordinates"]
        multi = {"type": "MultiPolygon", "coordinates": [square]}
        self.assertAlmostEqual(
            vworld_api.polygon_area_sqm(multi),
            vworld_api.polygon_area_sqm(_square(127.0, 37.5, 0.001)),
            places=3,
        )

    def test_점이_모자라면_0(self):
        self.assertEqual(vworld_api.ring_area_sqm([[127.0, 37.5], [127.1, 37.5]]), 0.0)
        self.assertEqual(vworld_api.polygon_area_sqm({"type": "Point", "coordinates": []}), 0.0)


class TestParseParcel(unittest.TestCase):
    def setUp(self):
        self.payload = _ok([
            {
                "geometry": _square(127.0, 37.5, 0.001),
                "properties": {
                    "pnu": "4111113000103950000",
                    "jibun": "395대",
                    "addr": "경기도 수원시 장안구 정자동 395",
                    "jiga": "1734000",
                },
            }
        ])

    def test_필드(self):
        row = vworld_api.parse_parcel(self.payload)
        self.assertEqual(row["pnu"], "4111113000103950000")
        self.assertEqual(row["sgg_cd"], "41111")
        self.assertEqual(row["jiga_won_sqm"], "1734000")
        self.assertGreater(row["area_sqm"], 9000)
        # 대표점은 정사각형 한가운데
        self.assertAlmostEqual(row["lon"], 127.0005, places=3)
        self.assertAlmostEqual(row["lat"], 37.5005, places=3)

    def test_같은_PNU_조각은_면적을_합친다(self):
        two = _ok([
            {"geometry": _square(127.0, 37.5, 0.001), "properties": {"pnu": "X"}},
            {"geometry": _square(127.01, 37.5, 0.001), "properties": {"pnu": "X"}},
        ])
        one = vworld_api.parse_parcel(_ok([
            {"geometry": _square(127.0, 37.5, 0.001), "properties": {"pnu": "X"}}
        ]))
        self.assertAlmostEqual(
            vworld_api.parse_parcel(two)["area_sqm"], one["area_sqm"] * 2, delta=1.0
        )


class TestParseLanduse(unittest.TestCase):
    def test_종_구분이_있는_쪽을_고른다(self):
        # 한 점에 '도시지역'(상위 구분)과 '제2종일반주거지역'이 함께 온다.
        # 용적률 상한을 정하려면 종이 있는 쪽만 쓸모 있다.
        payload = _ok([
            {"properties": {"uname": "도시지역", "sido_name": "경기도"}},
            {"properties": {"uname": "제2종일반주거지역", "sido_name": "경기도"}},
        ])
        self.assertEqual(vworld_api.parse_landuse(payload)["landuse_nm"], "제2종일반주거지역")

    def test_종이_없으면_첫_값(self):
        payload = _ok([{"properties": {"uname": "자연녹지지역", "sido_name": "경기도"}}])
        self.assertEqual(vworld_api.parse_landuse(payload)["landuse_nm"], "자연녹지지역")


class TestUrls(unittest.TestCase):
    def test_필지는_PNU_속성필터(self):
        url = vworld_api.parcel_url("KEY", "1168010600103160000")
        self.assertIn("LP_PA_CBND_BUBUN", url)
        self.assertIn("pnu%3A%3D%3A1168010600103160000", url)

    def test_용도지역은_좌표필터(self):
        url = vworld_api.landuse_url("KEY", 127.0, 37.5)
        self.assertIn("LT_C_UQ111", url)
        self.assertIn("POINT", url)

    def test_키가_URL에_들어간다(self):
        # 키는 저장소에 두지 않지만 URL 조립은 되어야 한다.
        self.assertIn("key=KEY", vworld_api.parcel_url("KEY", "1"))


class TestLoadKey(unittest.TestCase):
    def test_환경변수_우선(self):
        import os

        old = os.environ.get("VWORLD_API_KEY")
        os.environ["VWORLD_API_KEY"] = "FROM-ENV"
        try:
            self.assertEqual(vworld_api.load_key(), "FROM-ENV")
        finally:
            if old is None:
                del os.environ["VWORLD_API_KEY"]
            else:
                os.environ["VWORLD_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
