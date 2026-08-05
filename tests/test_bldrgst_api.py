"""건축물대장 총괄표제부 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bldrgst_api  # noqa: E402


def _xml(items: str, code: str = "00", total: int = 1) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<response><header><resultCode>{code}</resultCode><resultMsg>NORMAL SERVICE</resultMsg></header>
<body><items>{items}</items><totalCount>{total}</totalCount></body></response>"""


# 은마아파트 실제 응답에서 옮긴 값. platArea 와 vlRat 이 0 인 것까지 그대로다.
EUMA = """<item>
  <platPlc>서울특별시 강남구 대치동 316번지</platPlc>
  <sigunguCd>11680</sigunguCd><bjdongCd>10600</bjdongCd><platGbCd>0</platGbCd>
  <bun>0316</bun><ji>0000</ji>
  <bldNm>은마아파트</bldNm><mainBldCnt>31</mainBldCnt><hhldCnt>4424</hhldCnt>
  <platArea>0</platArea><archArea>48104.55</archArea><totArea>528772</totArea>
  <vlRatEstmTotArea>488472.85</vlRatEstmTotArea><vlRat>0</vlRat><bcRat>0</bcRat>
  <useAprDay></useAprDay><mainPurpsCdNm>공동주택</mainPurpsCdNm>
</item>"""

SHOP = """<item>
  <sigunguCd>11680</sigunguCd><bjdongCd>10600</bjdongCd><platGbCd>0</platGbCd>
  <bun>0100</bun><ji>0001</ji>
  <bldNm>상가</bldNm><totArea>500</totArea><mainPurpsCdNm>제1종근린생활시설</mainPurpsCdNm>
</item>"""

SAN = """<item>
  <sigunguCd>41135</sigunguCd><bjdongCd>11000</bjdongCd><platGbCd>1</platGbCd>
  <bun>0012</bun><ji>0000</ji>
  <bldNm>산지아파트</bldNm><hhldCnt>300</hhldCnt><platArea>10000</platArea>
  <totArea>20000</totArea><vlRatEstmTotArea>18000</vlRatEstmTotArea><vlRat>180</vlRat>
  <mainPurpsCdNm>공동주택</mainPurpsCdNm>
</item>"""


class TestMakePnu(unittest.TestCase):
    def test_일반_지번(self):
        # platGbCd 0(일반) → PNU 대장구분 1
        self.assertEqual(
            bldrgst_api.make_pnu("11680", "10600", "0", "0316", "0000"),
            "1168010600103160000",
        )

    def test_산_지번(self):
        # platGbCd 1(산) → PNU 대장구분 2
        self.assertEqual(
            bldrgst_api.make_pnu("41135", "11000", "1", "0012", "0000"),
            "4113511000200120000",
        )

    def test_자릿수를_채운다(self):
        self.assertEqual(
            bldrgst_api.make_pnu("11110", "10100", "0", "1", ""),
            "1111010100100010000",
        )

    def test_주소가_모자라면_None(self):
        self.assertIsNone(bldrgst_api.make_pnu("", "10600", "0", "0316", "0000"))
        self.assertIsNone(bldrgst_api.make_pnu("11680", "10600", "0", "", "0000"))


class TestParseResponse(unittest.TestCase):
    def test_은마_필드(self):
        rows, total = bldrgst_api.parse_response(_xml(EUMA))
        self.assertEqual(total, 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["pnu"], "1168010600103160000")
        self.assertEqual(row["hhld_cnt"], "4424")
        self.assertEqual(row["main_bld_cnt"], "31")
        self.assertEqual(row["vl_rat_estm_tot_area"], "488472.85")
        # 대장에 비어 있는 값은 만들어내지 않고 그대로 옮긴다.
        self.assertEqual(row["plat_area"], "0")
        self.assertEqual(row["vl_rat"], "0")

    def test_공동주택이_아니면_버린다(self):
        rows, _ = bldrgst_api.parse_response(_xml(EUMA + SHOP, total=2))
        self.assertEqual([r["bld_nm"] for r in rows], ["은마아파트"])

    def test_산지번도_읽는다(self):
        rows, _ = bldrgst_api.parse_response(_xml(SAN))
        self.assertEqual(rows[0]["pnu"], "4113511000200120000")

    def test_에러코드는_ApiError(self):
        with self.assertRaises(bldrgst_api.ApiError):
            bldrgst_api.parse_response(_xml(EUMA, code="30"))

    def test_한도초과를_구분한다(self):
        try:
            bldrgst_api.parse_response(_xml(EUMA, code="22"))
        except bldrgst_api.ApiError as exc:
            self.assertTrue(exc.is_limit)
        else:
            self.fail("ApiError 가 나지 않았습니다")


class TestFloorAreaAndFar(unittest.TestCase):
    def setUp(self):
        self.euma = bldrgst_api.parse_response(_xml(EUMA))[0][0]

    def test_용적률_산정_연면적을_먼저_쓴다(self):
        self.assertEqual(bldrgst_api.floor_area(self.euma), 488472.85)

    def test_연면적으로_물러선다(self):
        row = dict(self.euma, vl_rat_estm_tot_area="0")
        self.assertEqual(bldrgst_api.floor_area(row), 528772.0)

    def test_대장_용적률이_있으면_그대로(self):
        rows, _ = bldrgst_api.parse_response(_xml(SAN))
        self.assertEqual(bldrgst_api.far(rows[0]), 180.0)

    def test_대장_용적률이_0이면_연면적으로_낸다(self):
        # 은마는 대장 대지면적이 0 이라 지적도 면적(239,225.8㎡)을 넣어야 나온다.
        far = bldrgst_api.far(self.euma, land_sqm=239225.8)
        self.assertIsNotNone(far)
        self.assertAlmostEqual(far, 204.2, places=1)

    def test_대지면적이_없으면_None(self):
        self.assertIsNone(bldrgst_api.far(self.euma))


class TestMergeRows(unittest.TestCase):
    def test_연면적이_큰_쪽을_남긴다(self):
        old = {"pnu": "X", "tot_area": "1000"}
        new = {"pnu": "X", "tot_area": "5000"}
        self.assertEqual(bldrgst_api.merge_rows([old], [new]), [new])
        self.assertEqual(bldrgst_api.merge_rows([new], [old]), [new])

    def test_PNU_로_정렬한다(self):
        rows = bldrgst_api.merge_rows([{"pnu": "B"}, {"pnu": "A"}])
        self.assertEqual([r["pnu"] for r in rows], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
