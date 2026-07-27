"""학교 위치 데이터를 받아 서울·경기 초·중만 저장한다.

    python3 scripts/collect_schools.py

전국 12,011건을 13번에 나눠 받아 필터한 뒤 data/schools.csv.gz 로 쓴다.
실거래처럼 매일 돌릴 필요는 없다 — 학교는 자주 바뀌지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_trades  # noqa: E402
import rtms  # noqa: E402
import schools_api  # noqa: E402

OUT_FILE = ROOT / "data" / "schools.csv.gz"


def rows_to_csv(rows: list[dict]) -> str:
    """schools_api.COLUMNS 순서로 쓴다. rtms.rows_to_csv 는 실거래 컬럼 고정이라 못 쓴다."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=schools_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in schools_api.COLUMNS})
    return buf.getvalue()


def collect(key: str, page_size: int = 1000) -> list[dict]:
    """전수를 받아 서울·경기 초·중만 남긴다. 정렬해 결정론을 지킨다."""
    kept: list[dict] = []
    seen = 0
    page = 1
    while True:
        items, total = schools_api.fetch_page(key, page, page_size)
        if not items:
            break
        seen += len(items)
        for item in items:
            row = schools_api.normalize(item)
            if row is not None:
                kept.append(row)
        print(f"  {seen:,}/{total:,} 수신, 대상 {len(kept):,}건")
        if seen >= total:
            break
        page += 1
    kept.sort(key=lambda r: (r["school_id"], r["school_name"]))
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()

    key = collect_trades.load_api_key()
    try:
        rows = collect(key, args.page_size)
    except schools_api.ApiError as exc:
        if exc.code == "30":
            print("이 키로는 학교 위치 데이터를 쓸 수 없습니다. data.go.kr 에서 "
                  "'전국초중등학교위치표준데이터' 활용신청을 하세요.", file=sys.stderr)
        elif exc.code == "12":
            print(f"엔드포인트가 존재하지 않습니다: {schools_api.ENDPOINT}", file=sys.stderr)
        else:
            print(f"API 오류: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("대상 학교가 0건입니다. 필터가 과한지 확인하세요.", file=sys.stderr)
        return 1

    levels: dict[str, int] = {}
    for row in rows:
        levels[row["level"]] = levels.get(row["level"], 0) + 1
    print(f"서울·경기 대상 {len(rows):,}건 " + " / ".join(f"{k} {v:,}" for k, v in sorted(levels.items())))

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
