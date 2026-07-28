"""EDSS 중학교 졸업생진로현황에서 서울·경기 특목고·자사고 진학률을 뽑는다.

    python3 scripts/build_progression.py

시도교육청 단위 집계만 낼 수 있다 — 학교별로는 못 낸다. 이 파일에는 학교명도
주소도 없고 식별자(개방ID)를 우리 위치 데이터(data/schools.csv.gz)의
school_id 로 이을 방법이 없다. 자세한 내용과 시도한 조인 다섯 가지는
data/edss/README.md 참고.

출력은 assets/realestate/progression.json. 원본은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_dashboard import write_json  # noqa: E402

SRC = ROOT / "data" / "edss" / "중학교_졸업생진로현황_2009-2025.csv.gz"
OUT = ROOT / "assets" / "realestate" / "progression.json"

# 원본 시도교육청명 -> 화면에 쓸 짧은 이름. 이 두 지역만 다룬다(그 외 시도는
# 우리 학교 지도가 다루는 범위 밖이라 버린다).
REGIONS = {
    "서울특별시교육청": "서울",
    "경기도교육청": "경기",
}

# 특목·자사고 진학자 = 과학고 + 외고·국제고 + 자사고 (남 + 여). data/edss/README.md
# "쓸 수 있는 컬럼" 절과 같다.
NUMERATOR_COLS = [
    "특수과학고진학남학생수", "특수과학고진학여학생수",
    "특수외국어고진학남학생수", "특수외국어고진학여학생수",
    "자율사립고진학남학생수", "자율사립고진학여학생수",
]
DENOMINATOR_COLS = ["졸업생진로_중_남자졸업생수", "졸업생진로_중_여자졸업생수"]

# 2009·2010년은 특목고·자사고 진학 항목이 아예 공시되지 않아 위 여섯 컬럼이
# 원본에서 전부 0이다("그 해엔 아무도 안 갔다"가 아니라 "그 해엔 이 항목을
# 공시하지 않았다"). 0%로 그리면 "아무도 안 갔다"는 거짓 신호가 되므로
# 하드코딩으로 제외하고, 이유를 payload 에도 그대로 남긴다.
EXCLUDED_YEARS = {"2009", "2010"}
EXCLUDED_NOTE = (
    "2009~2010년은 특목고·자사고 진학 항목이 공시되지 않아 원본 값이 전부 "
    "0입니다. '진학자 0명'이 아니라 '집계 안 됨'이라 제외했습니다."
)


def _int(value: str | None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def aggregate_rows(rows) -> dict[str, dict[str, tuple[int, int]]]:
    """행을 (지역, 연도)로 묶어 (특목·자사 진학자, 졸업자) 합을 낸다.

    REGIONS 에 없는 시도(서울·경기 외)는 조용히 버린다. rows 는 csv.DictReader
    가 주는 dict 이터러블이면 되므로, 실제 gzip 파일이든 테스트용 합성
    데이터든 그대로 넘길 수 있다.
    """
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        short = REGIONS.get(row.get("시도교육청명"))
        if short is None:
            continue
        year = row.get("공시년도")
        num = sum(_int(row.get(c)) for c in NUMERATOR_COLS)
        den = sum(_int(row.get(c)) for c in DENOMINATOR_COLS)
        bucket = agg[short][year]
        bucket[0] += num
        bucket[1] += den
    return {region: {year: tuple(nd) for year, nd in years.items()}
            for region, years in agg.items()}


def build_payload(agg: dict[str, dict[str, tuple[int, int]]], generated: str) -> dict:
    """지역별 연도 시계열(rate/num/den)을 만든다. EXCLUDED_YEARS 는 통째로 뺀다."""
    all_years = {y for years in agg.values() for y in years}
    years = sorted(y for y in all_years if y not in EXCLUDED_YEARS)

    regions: dict[str, dict[str, list]] = {}
    for short in sorted(REGIONS.values()):
        by_year = agg.get(short, {})
        num_list, den_list, rate_list = [], [], []
        for y in years:
            num, den = by_year.get(y, (0, 0))
            num_list.append(num)
            den_list.append(den)
            rate_list.append(round(num / den * 100, 2) if den else None)
        regions[short] = {"num": num_list, "den": den_list, "rate": rate_list}

    return {
        "generated": generated,
        "source": "EDSS 중학교 졸업생진로현황(시도교육청 단위 공시자료)",
        "years": years,
        "excluded_years": sorted(EXCLUDED_YEARS),
        "excluded_note": EXCLUDED_NOTE,
        "regions": regions,
    }


def read_rows() -> list[dict]:
    with gzip.open(SRC, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat(),
                        help="생성일. 테스트에서 고정하려면 지정한다")
    args = parser.parse_args()

    if not SRC.exists():
        print(f"{SRC} 가 없습니다.", file=sys.stderr)
        return 1

    rows = read_rows()
    agg = aggregate_rows(rows)
    payload = build_payload(agg, args.generated)

    changed = write_json(OUT, payload)
    size = OUT.stat().st_size
    print(f"progression.json {size:,}B {'갱신' if changed else '변경 없음'}")
    for short in sorted(REGIONS.values()):
        r = payload["regions"][short]
        last = len(payload["years"]) - 1
        print(f"  {short} {payload['years'][last]}: "
              f"{r['rate'][last]}% ({r['num'][last]:,}/{r['den'][last]:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
