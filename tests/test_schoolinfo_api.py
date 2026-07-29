"""학교알리미 졸업생 진로 현황 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import schoolinfo_api as api  # noqa: E402

# 개원중학교 2025년 실제 응답의 표 구조를 그대로 줄여 옮긴 것.
# 순서: 졸업자 | 일반고 | 특성화고 | 과학고 | 외고국제고 | 예고체고 | 마이스터고 |
#       특목소계 | 자사고 | 자공고 | 자율소계 | 기타 | 진학자계 | 취업 | 대안 | 무직
REAL_TABLE = """
<table>
 <tr><th>구  분</th><th>졸업자</th><th>진학자</th></tr>
 <tr><td>남</td><td>98</td><td>82</td><td>4</td><td>0</td><td>0</td><td>0</td>
     <td>0</td><td>0</td><td>10</td><td>0</td><td>10</td><td>2</td><td>98</td>
     <td>0</td><td>0</td><td>0</td></tr>
 <tr><td>합계</td><td>291</td><td>241</td><td>19</td><td>0</td><td>9</td><td>3</td>
     <td>0</td><td>12</td><td>15</td><td>0</td><td>15</td><td>4</td><td>291</td>
     <td>0</td><td>0</td><td>0</td></tr>
 <tr><td>비  율</td><td>82.8</td><td>6.5</td></tr>
</table>
"""


class TestParseProgression(unittest.TestCase):
    def test_reads_total_row(self) -> None:
        row = api.parse_progression(REAL_TABLE)
        self.assertIsNotNone(row)
        self.assertEqual(row["grad"], 291)
        self.assertEqual(row["general"], 241)
        self.assertEqual(row["science"], 0)
        self.assertEqual(row["foreign_intl"], 9)
        self.assertEqual(row["special_sum"], 12)
        self.assertEqual(row["auto_sum"], 15)
        self.assertEqual(row["advanced"], 291)

    def test_ignores_gender_and_ratio_rows(self) -> None:
        """'남'/'여' 행과 소수점이 든 '비율' 행을 합계로 오인하면 안 된다."""
        row = api.parse_progression(REAL_TABLE)
        self.assertNotEqual(row["grad"], 98)  # 남 행
        self.assertNotEqual(row["grad"], 82)

    def test_missing_table_is_none(self) -> None:
        self.assertIsNone(api.parse_progression("<html><body>준비중</body></html>"))

    def test_short_total_row_is_none(self) -> None:
        """열이 모자라면 조용히 잘린 값을 쓰지 않고 None 을 준다."""
        html = "<table><tr><td>합계</td><td>10</td><td>5</td></tr></table>"
        self.assertIsNone(api.parse_progression(html))

    def test_non_numeric_cell_is_none(self) -> None:
        cells = "".join(f"<td>{i}</td>" for i in range(15))
        html = f"<table><tr><td>합계</td><td>-</td>{cells}</tr></table>"
        self.assertIsNone(api.parse_progression(html))

    def test_commas_stripped(self) -> None:
        cells = "".join("<td>0</td>" for _ in range(15))
        html = f"<table><tr><td>합계</td><td>1,234</td>{cells}</tr></table>"
        self.assertEqual(api.parse_progression(html)["grad"], 1234)


class TestSpecialRate(unittest.TestCase):
    def test_special_plus_autonomous_over_graduates(self) -> None:
        """분자는 특목고 소계 + 자율고 소계다 — 시·도 차트와 같은 정의라야 비교된다."""
        rate = api.special_rate({"grad": 291, "special_sum": 12, "auto_sum": 15})
        self.assertAlmostEqual(rate, 27 / 291 * 100, places=6)

    def test_zero_graduates_is_none(self) -> None:
        self.assertIsNone(api.special_rate({"grad": 0, "special_sum": 0, "auto_sum": 0}))

    def test_missing_keys_treated_as_zero(self) -> None:
        self.assertEqual(api.special_rate({"grad": 100}), 0.0)


class TestFixedParams(unittest.TestCase):
    def test_item_code_is_progression(self) -> None:
        """GS_HANGMOK_CD=06 이 「졸업생의 진로 현황」이다. 바뀌면 다른 항목을 긁는다."""
        self.assertEqual(api.FIXED_PARAMS["GS_HANGMOK_CD"], "06")
        self.assertEqual(api.FIXED_PARAMS["GS_HANGMOK_NM"], "졸업생의 진로 현황")

    def test_middle_school_level_code(self) -> None:
        self.assertEqual(api.MIDDLE_SCHOOL, "03")


if __name__ == "__main__":
    unittest.main()
