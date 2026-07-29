"""NEIS 학교기본정보 파싱·특목고 판정 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_schools  # noqa: E402
import neis_api  # noqa: E402


def _ok(rows: list[dict], total: int | None = None) -> dict:
    return {"schoolInfo": [
        {"head": [{"list_total_count": total if total is not None else len(rows)},
                  {"RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다."}}]},
        {"row": rows},
    ]}


def _hs(**kw) -> dict:
    base = {"SCHUL_NM": "한성과학고등학교", "HS_SC_NM": "특목고",
            "SPCLY_PURPS_HS_ORD_NM": "과학계열"}
    base.update(kw)
    return base


class TestParseResponse(unittest.TestCase):
    def test_returns_rows_and_total(self) -> None:
        rows, total = neis_api.parse_response(_ok([_hs()], total=319))
        self.assertEqual(len(rows), 1)
        self.assertEqual(total, 319)

    def test_no_data_is_empty_not_error(self) -> None:
        """INFO-200(해당 데이터 없음)은 오류 모양으로 오지만 정상 종료 조건이다."""
        payload = {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
        rows, total = neis_api.parse_response(payload)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_invalid_key_raises_with_code(self) -> None:
        payload = {"RESULT": {"CODE": "ERROR-290", "MESSAGE": "인증키가 유효하지 않습니다."}}
        with self.assertRaises(neis_api.NeisError) as ctx:
            neis_api.parse_response(payload)
        self.assertEqual(ctx.exception.code, "ERROR-290")


class TestCourseOf(unittest.TestCase):
    def test_target_courses_pass(self) -> None:
        for course in ("과학계열", "외국어계열", "국제계열"):
            self.assertEqual(neis_api.course_of(_hs(SPCLY_PURPS_HS_ORD_NM=course)), course)

    def test_general_high_school_is_none(self) -> None:
        self.assertIsNone(neis_api.course_of(
            _hs(HS_SC_NM="일반고", SPCLY_PURPS_HS_ORD_NM=None)))

    def test_specialized_vocational_is_none(self) -> None:
        """이름에 '과학고'가 들어가도 특성화고면 대상이 아니다 — 조리·의료과학고 등."""
        self.assertIsNone(neis_api.course_of(
            _hs(SCHUL_NM="한국조리과학고등학교", HS_SC_NM="특성화고",
                SPCLY_PURPS_HS_ORD_NM=None)))

    def test_art_and_meister_courses_excluded(self) -> None:
        for course in ("예술계열", "체육계열", "산업수요 맞춤형 고등학교"):
            self.assertIsNone(neis_api.course_of(_hs(SPCLY_PURPS_HS_ORD_NM=course)))

    def test_missing_course_recovered_by_name(self) -> None:
        """경기외고·동두천외고는 특목고인데 계열이 비어 있다. 이름으로 구제한다."""
        self.assertEqual(
            neis_api.course_of(_hs(SCHUL_NM="경기외국어고등학교",
                                   SPCLY_PURPS_HS_ORD_NM=None)), "외국어계열")
        self.assertEqual(
            neis_api.course_of(_hs(SCHUL_NM="동두천외국어고등학교",
                                   SPCLY_PURPS_HS_ORD_NM="")), "외국어계열")

    def test_name_recovery_only_applies_to_special_purpose(self) -> None:
        """이름 구제는 HS_SC_NM 이 '특목고'일 때만. 이름만 보고 판정하지 않는다."""
        self.assertIsNone(neis_api.course_of(
            _hs(SCHUL_NM="경기외국어고등학교", HS_SC_NM="자율고",
                SPCLY_PURPS_HS_ORD_NM=None)))


def _row(**kw) -> dict:
    base = {"school_id": "S1", "school_name": "한성과학고등학교", "level": "고등학교",
            "found_type": "공립", "addr": "서울특별시 강서구 등촌동 1",
            "road_addr": "", "sido_office": "서울특별시교육청",
            "lat": "37.5", "lon": "127.0", "course": ""}
    base.update(kw)
    return base


class TestApplyCourses(unittest.TestCase):
    COURSES = {("서울특별시", "한성과학고등학교"): "과학계열"}

    def test_matched_high_school_gets_course(self) -> None:
        got = collect_schools.apply_courses([_row()], self.COURSES)
        self.assertEqual([r["course"] for r in got], ["과학계열"])

    def test_unmatched_high_school_dropped(self) -> None:
        rows = [_row(school_name="양천고등학교")]
        self.assertEqual(collect_schools.apply_courses(rows, self.COURSES), [])

    def test_elementary_and_middle_pass_through_untouched(self) -> None:
        rows = [_row(level="초등학교", school_name="계성초등학교"),
                _row(level="중학교", school_name="경신중학교")]
        got = collect_schools.apply_courses(rows, self.COURSES)
        self.assertEqual(len(got), 2)
        self.assertEqual([r["course"] for r in got], ["", ""])

    def test_same_name_in_other_sido_does_not_match(self) -> None:
        """조인 키는 (시도, 학교명)이다 — 시도를 빼면 동명 학교가 서로 섞인다."""
        rows = [_row(addr="경기도 성남시 분당구 정자동 1")]
        self.assertEqual(collect_schools.apply_courses(rows, self.COURSES), [])


if __name__ == "__main__":
    unittest.main()
