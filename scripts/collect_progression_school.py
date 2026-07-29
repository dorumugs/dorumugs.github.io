"""학교알리미에서 서울·경기 중학교별 졸업생 진로 현황을 받아 저장한다.

    python3 scripts/collect_progression_school.py
    python3 scripts/collect_progression_school.py --years 2025 --limit 20

학교 하나에 요청 하나라 1,147곳 × 연도만큼 부른다. 실거래처럼 매일 돌릴 필요는
없다 — 이 항목은 연 1회(11월) 공시라 1년에 한 번이면 충분하다.

시·도 단위 집계(build_progression.py, EDSS 기반)와 달리 여기는 학교별이다.
두 값의 정의를 맞춰 두었으므로(특목고 소계 + 자율고 소계 ÷ 졸업자) 화면에서
"이 학교 vs 시·도 평균"을 나란히 놓아도 말이 된다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rtms  # noqa: E402
import schoolinfo_api as api  # noqa: E402

OUT_FILE = ROOT / "data" / "progression_school.csv.gz"

# 학교알리미는 최근 3년치만 공시한다. 그 이전은 EDSS 신청 자료뿐인데 그건
# 학교를 특정할 수 없다(data/edss/README.md).
DEFAULT_YEARS = ("2023", "2024", "2025")

COLUMNS = ["year", "sido", "sgg", "sgg_code", "school_name", "shl_idf_cd"] + api.TOTAL_COLUMNS

# 표를 못 읽은 학교가 이 비율을 넘으면 화면 구조가 바뀐 것으로 본다. 졸업생이
# 없는 신설 학교 등으로 몇 곳은 정상적으로 비니 0 으로 두지 않는다.
MISS_LIMIT = 0.15


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return buf.getvalue()


def collect(years: tuple[str, ...], delay: float, limit: int | None) -> tuple[list[dict], int, int]:
    """(행 목록, 시도한 학교 수, 표가 없던 학교 수)."""
    rows: list[dict] = []
    tried = 0
    missed = 0
    for sido_nm, sido_cd in api.SIDO.items():
        for sgg in api.fetch_sigungu(sido_cd):
            sgg_nm = sgg["ADRCD_ID_LAST_NM"]
            sgg_cd = sgg["ADDR_CD_ID"]
            schools = api.fetch_schools(sido_cd, sgg_cd)
            for school in schools:
                if limit is not None and tried >= limit:
                    return rows, tried, missed
                tried += 1
                for year in years:
                    data = api.fetch_progression(
                        school["SHL_IDF_CD"], school["SHL_NM"], year)
                    time.sleep(delay)
                    if data is None:
                        missed += 1
                        continue
                    rows.append({
                        "year": year, "sido": sido_nm, "sgg": sgg_nm,
                        "sgg_code": sgg_cd, "school_name": school["SHL_NM"],
                        "shl_idf_cd": school["SHL_IDF_CD"], **data,
                    })
            print(f"  {sido_nm} {sgg_nm}: 학교 {len(schools):>3}곳 "
                  f"(누적 {tried:,}곳 / 행 {len(rows):,})", flush=True)
    return rows, tried, missed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--delay", type=float, default=0.25,
                        help="요청 사이 간격(초). 공개 사이트라 예의상 둔다")
    parser.add_argument("--limit", type=int, default=None, help="학교 수 상한(시험용)")
    args = parser.parse_args()

    years = tuple(args.years)
    print(f"연도 {', '.join(years)} 수집 시작")
    rows, tried, missed = collect(years, args.delay, args.limit)

    if not rows:
        print("수집된 행이 0건입니다. 화면 구조가 바뀌었는지 확인하세요.", file=sys.stderr)
        return 1
    attempts = tried * len(years)
    if attempts and missed / attempts > MISS_LIMIT:
        print(f"표를 못 읽은 비율이 {missed}/{attempts} 로 한도 {MISS_LIMIT:.0%} 를 "
              "넘었습니다. 파일을 쓰지 않습니다.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["year"], r["sgg_code"], r["school_name"]))
    print(f"학교 {tried:,}곳 / 행 {len(rows):,}건 (표 없음 {missed}건)")

    data = rtms.gzip_bytes(rows_to_csv(rows))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        print("변경 없음")
        return 0
    OUT_FILE.write_bytes(data)
    print(f"{OUT_FILE.name} 갱신 ({len(data):,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
