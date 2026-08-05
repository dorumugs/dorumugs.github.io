"""도시계획조례 용적률 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v

본문 발췌는 2026-08-05 에 법제처에서 실제로 받은 조례에서 옮겼다. 조례마다
표기가 제각각이라 여기가 회귀를 잡는 자리다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ordinance_api  # noqa: E402


def _doc(body: str, title: str = "용도지역 안에서의 용적률") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<LawService><조><조문번호>006700</조문번호><조제목>{title}</조제목>
<조내용><![CDATA[{body}]]></조내용></조></LawService>"""


class TestPickOrdinance(unittest.TestCase):
    XML = """<?xml version="1.0" encoding="UTF-8"?><OrdinSearch>
    <law id="1"><자치법규명><![CDATA[성남시 도시계획변경 사전협상 운영에 관한 조례]]></자치법규명>
      <자치법규ID>2251917</자치법규ID><지자체기관명>경기도 성남시</지자체기관명><공포일자>20250101</공포일자></law>
    <law id="2"><자치법규명><![CDATA[성남시 도시계획 조례]]></자치법규명>
      <자치법규ID>2146953</자치법규ID><지자체기관명>경기도 성남시</지자체기관명><공포일자>20260101</공포일자></law>
    <law id="3"><자치법규명><![CDATA[수원시 도시계획 조례]]></자치법규명>
      <자치법규ID>9999</자치법규ID><지자체기관명>경기도 수원시</지자체기관명><공포일자>20260101</공포일자></law>
    </OrdinSearch>"""

    def test_이름이_비슷한_다른_조례를_거른다(self):
        got = ordinance_api.pick_ordinance(self.XML, "경기도 성남시")
        self.assertEqual(got["law_id"], "2146953")
        self.assertEqual(got["law_name"], "성남시 도시계획 조례")

    def test_다른_지자체는_안_고른다(self):
        self.assertEqual(
            ordinance_api.pick_ordinance(self.XML, "경기도 수원시")["law_id"], "9999"
        )
        self.assertIsNone(ordinance_api.pick_ordinance(self.XML, "경기도 용인시"))

    def test_군계획_조례도_인정한다(self):
        xml = """<OrdinSearch><law><자치법규명><![CDATA[가평군 군계획 조례]]></자치법규명>
        <자치법규ID>2019668</자치법규ID><지자체기관명>경기도 가평군</지자체기관명>
        <공포일자>20260420</공포일자></law></OrdinSearch>"""
        self.assertEqual(
            ordinance_api.pick_ordinance(xml, "경기도 가평군")["law_name"], "가평군 군계획 조례"
        )


class TestParseFar(unittest.TestCase):
    def test_성남시_정비사업_단서(self):
        # 실제 원문: '280퍼센트(…정비사업으로 건설하고자 하는 …아파트는 300퍼센트)'
        body = (
            "① 각 용도지역의 용적률은 다음 각 호와 같다. "
            "1. 제1종일반주거지역: 160퍼센트 "
            "2. 제3종일반주거지역: 280퍼센트(「도시 및 주거환경정비법」제2조제2호의 "
            "정비사업으로 건설하고자 하는 「건축법 시행령」별표1 제2호의 공동주택 중 "
            "아파트는 300퍼센트)"
        )
        far = ordinance_api.parse_far(_doc(body))
        self.assertEqual(far["제1종일반주거지역"]["far"], 160)
        third = far["제3종일반주거지역"]
        self.assertEqual(third["far"], 280)
        self.assertEqual(third["far_redev"], 300)
        self.assertTrue(third["has_proviso"])
        # 이 화면은 정비사업 맥락이라 300 이 적용값이다.
        self.assertEqual(ordinance_api.effective_far(third), 300)

    def test_군포시_100분의_표기(self):
        body = (
            "1. 제1종 전용주거지역 : 100분의 80 이하 "
            "4. 제2종 일반주거지역 : 100분의 230 이하. 다만, 「도시 및 주거환경정비법」의 "
            "주택재건축사업은 100분의 250 이하 "
            "5. 제3종 일반주거지역 : 100분의 280 이하"
        )
        far = ordinance_api.parse_far(_doc(body))
        self.assertEqual(far["제1종전용주거지역"]["far"], 80)
        self.assertEqual(far["제2종일반주거지역"]["far"], 230)
        self.assertEqual(far["제2종일반주거지역"]["far_redev"], 250)
        self.assertEqual(far["제3종일반주거지역"]["far"], 280)

    def test_수원시_콜론_뒤_수식어(self):
        # '제2종일반주거지역 : 일반건축은 250퍼센트 이하(다만, 공동주택은 230…, 정비법…250…)'
        body = (
            "4. 제2종일반주거지역 : 일반건축은 250퍼센트 이하"
            "(다만, 공동주택 또는 공동주택 용도가 복합된 건축물은 230퍼센트 이하, "
            "「도시 및 주거환경정비법」에 따른 공동주택은 250퍼센트 이하) (개정 2020.10.05)"
        )
        far = ordinance_api.parse_far(_doc(body))
        self.assertEqual(far["제2종일반주거지역"]["far"], 250)
        self.assertEqual(far["제2종일반주거지역"]["far_redev"], 250)

    def test_공백_섞인_용도지역명(self):
        body = "5. 제3종 일반주거지역 : 290퍼센트 이하"
        far = ordinance_api.parse_far(_doc(body))
        self.assertIn("제3종일반주거지역", far)

    def test_개정일자를_값으로_읽지_않는다(self):
        # '개정 2015.10.08' 의 2015 를 용적률로 오인하면 안 된다.
        body = "3. 제1종일반주거지역 : 200퍼센트 이하 <개정 2015.10.08, 2019.03.29>"
        self.assertEqual(ordinance_api.parse_far(_doc(body))["제1종일반주거지역"]["far"], 200)

    def test_건폐율_조는_보지_않는다(self):
        # 건폐율 조에도 같은 용도지역 이름과 퍼센트가 나온다. 조제목으로 걸러야 한다.
        body = "1. 제3종일반주거지역: 50퍼센트"
        self.assertEqual(ordinance_api.parse_far(_doc(body, title="용도지역 안에서의 건폐율")), {})

    def test_용적률_조가_없으면_빈_결과(self):
        self.assertEqual(ordinance_api.parse_far("<LawService></LawService>"), {})

    def test_깨진_XML은_OrdinanceError(self):
        with self.assertRaises(ordinance_api.OrdinanceError):
            ordinance_api.parse_far("<LawService")


class TestEffectiveFar(unittest.TestCase):
    def test_단서가_없으면_기본값(self):
        self.assertEqual(ordinance_api.effective_far({"far": 250, "far_redev": None}), 250)

    def test_빈_문자열도_넘긴다(self):
        # CSV 왕복 뒤에는 None 이 '' 로 온다.
        self.assertEqual(ordinance_api.effective_far({"far": "250", "far_redev": ""}), 250)

    def test_값이_없으면_None(self):
        self.assertIsNone(ordinance_api.effective_far({"far": "", "far_redev": ""}))


class TestNormalizeZone(unittest.TestCase):
    def test_공백_제거(self):
        self.assertEqual(ordinance_api.normalize_zone("제 3 종 일반주거지역"), "제3종일반주거지역")


if __name__ == "__main__":
    unittest.main()
