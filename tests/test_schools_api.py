"""학교 위치 API 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import schools_api  # noqa: E402


def _item(**kw) -> dict:
    base = {
        "schoolId": "B000008352", "schoolNm": "세화여자중학교", "schoolSe": "중학교",
        "fondType": "사립", "bnhhSe": "본교", "operSttus": "운영",
        "lnmadr": "서울특별시 서초구 반포동 753",
        "rdnmadr": "서울특별시 서초구 신반포로 56-7",
        "cddcNm": "서울특별시교육청", "latitude": "37.5019828", "longitude": "126.994230",
    }
    base.update(kw)
    return base


def _payload(items: list[dict], total: int) -> dict:
    return {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                         "body": {"totalCount": total, "items": items}}}


class TestParseResponse(unittest.TestCase):
    def test_returns_items_and_total(self) -> None:
        rows, total = schools_api.parse_response(_payload([_item()], 12011))
        self.assertEqual(total, 12011)
        self.assertEqual(len(rows), 1)

    def test_empty_items_is_not_an_error(self) -> None:
        rows, total = schools_api.parse_response(_payload([], 0))
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_missing_items_key_treated_as_empty(self) -> None:
        payload = {"response": {"header": {"resultCode": "00"}, "body": {"totalCount": 0}}}
        rows, _ = schools_api.parse_response(payload)
        self.assertEqual(rows, [])

    def test_unregistered_key_raises_with_code(self) -> None:
        payload = {"response": {"header": {"resultCode": "30",
                                           "resultMsg": "SERVICE KEY IS NOT REGISTERED ERROR."}}}
        with self.assertRaises(schools_api.ApiError) as ctx:
            schools_api.parse_response(payload)
        self.assertEqual(ctx.exception.code, "30")

    def test_missing_service_raises_with_code(self) -> None:
        payload = {"response": {"header": {"resultCode": "12",
                                           "resultMsg": "NO OPENAPI SERVICE ERROR."}}}
        with self.assertRaises(schools_api.ApiError) as ctx:
            schools_api.parse_response(payload)
        self.assertEqual(ctx.exception.code, "12")


class TestNormalize(unittest.TestCase):
    def test_keeps_seoul(self) -> None:
        got = schools_api.normalize(_item())
        self.assertEqual(got["school_name"], "세화여자중학교")
        self.assertEqual(got["level"], "중학교")
        self.assertEqual(got["found_type"], "사립")
        self.assertEqual(got["lat"], "37.5019828")

    def test_keeps_gyeonggi(self) -> None:
        got = schools_api.normalize(_item(lnmadr="경기도 성남시 분당구 서현동 100"))
        self.assertIsNotNone(got)

    def test_drops_other_region(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(lnmadr="부산광역시 해운대구 우동 1")))

    def test_drops_missing_coordinates(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(latitude="")))
        self.assertIsNone(schools_api.normalize(_item(longitude=None)))

    def test_drops_high_school(self) -> None:
        """고등학교는 이 도구의 범위 밖이라 수집 단계에서 버린다."""
        self.assertIsNone(schools_api.normalize(_item(schoolSe="고등학교")))

    def test_drops_closed_school(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(operSttus="폐교")))

    def test_drops_branch_school(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(bnhhSe="분교")))

    def test_every_column_present(self) -> None:
        got = schools_api.normalize(_item())
        self.assertEqual(sorted(got), sorted(schools_api.COLUMNS))


if __name__ == "__main__":
    unittest.main()
