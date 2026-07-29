"""학교별 졸업생 진로 현황을 지도용 JSON 으로 굽는다.

    python3 scripts/build_progression_school.py

출력은 assets/realestate/progression_school.json.

두 가지로 쓴다:
1. 학군 지도 목록에서 사립중·국제중의 특목고·자사고 진학률 열
2. 고른 중학교를 "진학률이 비슷한 학교들"과 비교하는 차트 — 그러려면 지도에
   없는 공립중까지 포함한 전체 분포가 있어야 하므로 1,396곳을 다 담는다.

진학률 정의는 schoolinfo_api.special_rate() 하나만 쓴다(특목고 소계 + 자율고
소계 ÷ 졸업자). 시·도 단위 차트(progression.json, EDSS 기반)와 같은 정의라
두 값을 나란히 놓을 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import schoolinfo_api as api  # noqa: E402
from build_dashboard import write_json  # noqa: E402

SRC = ROOT / "data" / "progression_school.csv.gz"
OUT = ROOT / "assets" / "realestate" / "progression_school.json"

MAX_BYTES = 120 * 1024

# 졸업생이 너무 적은 학교는 비율이 한 명에 크게 흔들린다(1명 = 10%p 이상).
# 값을 지우지는 않고 thin 표시만 달아 화면에서 조심해 쓰게 한다.
THIN_GRADUATES = 30


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(rows: list[dict], generated: str) -> dict:
    """{학교: 연도별 진학률} 로 접는다. 같은 학교가 두 번 나오면 한 번만 담는다.

    학교알리미 시군구 목록은 '수원시' 와 '수원시 장안구' 를 모두 포함해, 같은
    학교가 두 시군구에서 각각 잡힌다. 학교 UUID 로 중복을 없앤다.
    """
    years = sorted({r["year"] for r in rows})
    by_school: dict[str, dict] = {}
    for row in rows:
        key = row["shl_idf_cd"]
        entry = by_school.setdefault(key, {
            # 시군구는 '수원시 장안구' 처럼 구까지 있는 쪽을 남긴다 — 지도의
            # 시군구 코드(41113 등)와 맞춰야 조인된다.
            "name": row["school_name"],
            "sgg": row["sgg"],
            # 지도 데이터(schools.json)의 sgg 는 5자리다. 학교알리미 행정코드
            # 1168000000 의 앞 5자리가 곧 11680 이다.
            "sgg_cd": row["sgg_code"][:5],
            "rates": {},
            "grad": {},
        })
        if len(row["sgg"]) > len(entry["sgg"]):
            entry["sgg"] = row["sgg"]
            entry["sgg_cd"] = row["sgg_code"][:5]
        data = {k: int(row[k] or 0) for k in api.TOTAL_COLUMNS}
        rate = api.special_rate(data)
        if rate is None:
            continue
        entry["rates"][row["year"]] = round(rate, 1)
        entry["grad"][row["year"]] = data["grad"]

    out = []
    last = years[-1]
    for entry in by_school.values():
        if not entry["rates"]:
            continue
        out.append({
            "name": entry["name"],
            "sgg": entry["sgg_cd"],
            # 연도 순서는 years 와 같다. 없는 해는 null.
            "r": [entry["rates"].get(y) for y in years],
            "g": entry["grad"].get(last),
        })
    out.sort(key=lambda s: (s["sgg"], s["name"]))
    thin = sum(1 for s in out if (s["g"] or 0) < THIN_GRADUATES)
    return {"generated": generated, "years": years, "thin": THIN_GRADUATES,
            "schools": out, "thin_count": thin}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not SRC.exists():
        print(f"{SRC} 가 없습니다. 먼저 collect_progression_school.py 를 돌리세요.",
              file=sys.stderr)
        return 1

    rows = load(SRC)
    payload = build(rows, args.generated)
    thin = payload.pop("thin_count")
    print(f"원본 {len(rows):,}행 → 학교 {len(payload['schools']):,}곳 "
          f"({', '.join(payload['years'])}) / 졸업생 {THIN_GRADUATES}명 미만 {thin}곳")

    changed = write_json(OUT, payload)
    size = OUT.stat().st_size
    print(f"{OUT.name} {size:,}B {'갱신' if changed else '변경 없음'}")
    if size > MAX_BYTES:
        print(f"예산({MAX_BYTES}B) 초과", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
