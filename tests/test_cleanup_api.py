"""정비사업 정보몽땅 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v

픽스처는 2026-07-31 에 실제로 받은 응답이다. 정보몽땅은 공식 API 가 아니라
화면이 바뀌면 파서가 조용히 깨진다. 여기서 잡는 게 목적이다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cleanup_api  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestSplitJibun(unittest.TestCase):
    def test_동_지번_분리(self):
        self.assertEqual(cleanup_api.split_jibun("아현동 613-10"), ("아현동", "613-10"))
        self.assertEqual(cleanup_api.split_jibun("화곡동 956-37"), ("화곡동", "956-37"))
        self.assertEqual(cleanup_api.split_jibun("염리동 105"), ("염리동", "105"))

    def test_꼬리표가_붙어도_앞_두_토큰만(self):
        self.assertEqual(cleanup_api.split_jibun("북아현동 1-1 일대"), ("북아현동", "1-1"))

    def test_산지번은_붙여_쓴다(self):
        # 실거래 CSV 의 jibun 은 '산101' 이라 떨어진 '산 101' 을 합쳐야 조인된다.
        self.assertEqual(cleanup_api.split_jibun("봉천동 산 101"), ("봉천동", "산101"))

    def test_지번이_숫자가_아니면_비운다(self):
        # 실거래 CSV 의 jibun 과 못 맞추는 값을 넣느니 비워서 조인에서 빠지게 한다.
        self.assertEqual(cleanup_api.split_jibun("상계동 일대"), ("상계동", ""))
        self.assertEqual(cleanup_api.split_jibun("공덕동"), ("공덕동", ""))
        self.assertEqual(cleanup_api.split_jibun(""), ("", ""))


class TestParseProjectList(unittest.TestCase):
    def setUp(self):
        self.rows = cleanup_api.parse_project_list(_fixture("cleanup_project_list.html"))

    def test_행수(self):
        self.assertEqual(len(self.rows), 10)

    def test_첫_행_전체_필드(self):
        row = self.rows[0]
        self.assertEqual(row["seq"], "1102")
        self.assertEqual(row["sgg_nm"], "마포구")
        self.assertEqual(row["bsns_se"], "재개발(도시정비형)")
        self.assertEqual(row["name"], "마포로3구역제3지구 도시환경정비사업조합")
        self.assertEqual(row["jibun_addr"], "아현동 613-10")
        self.assertEqual(row["umd_nm"], "아현동")
        self.assertEqual(row["jibun"], "613-10")
        self.assertEqual(row["stage"], "착공")

    def test_이동_링크에서_키를_캔다(self):
        row = self.rows[0]
        self.assertEqual(row["cafe_url"], "mapo33")
        self.assertEqual(row["wtnnc_sn"], "11000AGZ201308079074")

    def test_지도링크가_없는_사업장도_수집된다(self):
        # 목록 1,102건 중 지도 링크는 365건뿐이다. 없다고 행을 버리면 안 된다.
        no_map = [r for r in self.rows if not r["wtnnc_sn"]]
        self.assertTrue(no_map)
        self.assertTrue(all(r["cafe_url"] for r in no_map))

    def test_모든_행이_사업장명과_카페주소를_갖는다(self):
        for row in self.rows:
            self.assertTrue(row["name"], row)
            self.assertTrue(row["cafe_url"], row)

    def test_표가_없으면_ParseError(self):
        with self.assertRaises(cleanup_api.ParseError):
            cleanup_api.parse_project_list("<html><body>점검 중입니다</body></html>")


class TestParseCafeKeys(unittest.TestCase):
    def test_키_추출(self):
        keys = cleanup_api.parse_cafe_keys(_fixture("cleanup_cafe_main.html"))
        self.assertEqual(keys["cafe_id"], "440100003003n67")
        self.assertEqual(keys["bsns_pk"], "11440-100003003")

    def test_없으면_ParseError(self):
        with self.assertRaises(cleanup_api.ParseError):
            cleanup_api.parse_cafe_keys("<form><input name='cafeId' value='x'></form>")


class TestParseProgress(unittest.TestCase):
    def setUp(self):
        self.events = cleanup_api.parse_progress(_fixture("cleanup_progress.html"))

    def test_이벤트가_충분히_나온다(self):
        self.assertGreater(len(self.events), 40)

    def test_날짜는_모두_ISO(self):
        for e in self.events:
            self.assertRegex(e["event_date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_구역지정_고시(self):
        found = [
            e
            for e in self.events
            if e["stage"] == "정비구역지정" and e["event_date"] == "2007-08-23"
        ]
        self.assertEqual(len(found), 1)
        e = found[0]
        self.assertEqual(e["event"], "구역지정(변경)고시")
        self.assertEqual(e["issuer"], "서울시")
        self.assertIn("서울특별시고시 제2007279호", e["notice_no"])

    def test_동의율을_숫자로_뽑는다(self):
        approvals = [
            e for e in self.events if e["stage"] == "조합설립인가" and e["consent_rate"]
        ]
        self.assertTrue(approvals)
        self.assertIn("80.88", {e["consent_rate"] for e in approvals})

    def test_선정업체명(self):
        vendors = {e["vendor"] for e in self.events if e["stage"] == "시공자선정"}
        self.assertIn("주식회사 대우건설", vendors)

    def test_도달하지_않은_단계는_빠진다(self):
        # 이 사업장은 준공인가·이전고시·조합해산 전이다. 빈 아코디언이 이벤트로 새면 안 된다.
        stages = {e["stage"] for e in self.events}
        self.assertNotIn("준공인가", stages)
        self.assertNotIn("이전고시", stages)
        self.assertIn("관리처분인가", stages)

    def test_단계_이름이_알려진_목록_안에_있다(self):
        known = set(cleanup_api.PROGRESS_STAGES)
        for e in self.events:
            self.assertIn(e["stage"], known, f"모르는 단계: {e['stage']}")


class TestMilestoneDates(unittest.TestCase):
    def setUp(self):
        self.events = cleanup_api.parse_progress(_fixture("cleanup_progress.html"))
        self.dates = cleanup_api.milestone_dates(self.events)

    def test_세_관문_모두_잡힌다(self):
        self.assertEqual(set(self.dates), set(cleanup_api.MILESTONE_STAGES))

    def test_최초_인가일을_쓴다(self):
        # 조합설립인가는 2007-11-27 최초 인가 뒤 2007-12-21 · 2018 · 2020 · 2021
        # 변경인가가 이어진다. 가격이 반응하는 건 최초 인가다.
        self.assertEqual(self.dates["조합설립인가"], "2007-11-27")
        self.assertEqual(self.dates["관리처분인가"], "2020-08-27")

    def test_신청일은_사건일이_아니다(self):
        # 사업시행인가 최초 '신청'은 2008-03-12, 최초 '인가'는 2012-10-26 이다.
        self.assertEqual(self.dates["사업시행인가"], "2012-10-26")


class TestNormalizeZoneName(unittest.TestCase):
    def test_사업장명과_구역명이_같아진다(self):
        self.assertEqual(
            cleanup_api.normalize_zone_name("봉천14구역 주택재개발정비사업조합"),
            cleanup_api.normalize_zone_name("봉천14"),
        )

    def test_뉴타운_꼬리표를_뗀다(self):
        self.assertEqual(
            cleanup_api.normalize_zone_name("아현3구역 주택재개발정비사업 조합(뉴타운)"),
            cleanup_api.normalize_zone_name("아현3"),
        )


class TestSerialization(unittest.TestCase):
    def test_왕복(self):
        rows = cleanup_api.parse_project_list(_fixture("cleanup_project_list.html"))
        text = cleanup_api.projects_to_csv(rows)
        back = cleanup_api.csv_to_rows(text)
        self.assertEqual(len(back), len(rows))
        self.assertEqual(
            {r["cafe_url"] for r in back}, {r["cafe_url"] for r in rows}
        )

    def test_gzip_은_재현_가능하다(self):
        a = cleanup_api.gzip_bytes("같은 내용")
        b = cleanup_api.gzip_bytes("같은 내용")
        self.assertEqual(a, b)
        self.assertEqual(cleanup_api.gunzip_text(a), "같은 내용")


if __name__ == "__main__":
    unittest.main()
