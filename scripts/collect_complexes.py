#!/usr/bin/env python3
"""서울·경기 공동주택 단지 마스터(세대수)를 수집한다.

실거래가에는 세대수가 없다. 300세대 이상 필터를 걸려면 단지 식별정보를 따로
받아 PNU 로 조인해야 한다. odcloud AptIdInfoSvc 는 전국 30만 단지 테이블이라
cond[PNU::LIKE] 로 시군구별로 잘라 받는다.

LIKE 는 부분일치라 '11680' 이 지번 자리에 들어간 다른 시군구 단지도 딸려온다.
받은 뒤 PNU 접두사로 다시 걸러 정확한 집합만 남긴다.

출력: data/complexes.csv.gz
상태: data/state/complex_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import regions  # noqa: E402
import rtms  # noqa: E402
from collect_trades import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "complexes.csv.gz"
STATE_FILE = ROOT / "data" / "state" / "complex_state.json"

API_URL = "https://api.odcloud.kr/api/AptIdInfoSvc/v1/getAptInfo"
PAGE_SIZE = 1000

COLUMNS = [
    "pnu",
    "sgg_cd",
    "complex_name",
    "address",
    "household_count",
    "building_count",
    "use_approval_date",
    "complex_type_code",
]


def fetch_page(key: str, sgg: str, page: int, retries: int = 3) -> dict:
    params = {
        "serviceKey": key,
        "page": page,
        "perPage": PAGE_SIZE,
        "cond[PNU::LIKE]": sgg,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def collect_sgg(key: str, sgg: str, sleep: float) -> tuple[list[dict], int]:
    """한 시군구의 단지 목록. (정제된 행, 소비한 호출 수)"""
    rows: list[dict] = []
    page = 1
    calls = 0
    while True:
        payload = fetch_page(key, sgg, page)
        calls += 1
        data = payload.get("data") or []
        for rec in data:
            pnu = (rec.get("PNU") or "").strip()
            # LIKE 부분일치로 딸려온 타 시군구 단지를 여기서 걷어낸다.
            if not pnu.startswith(sgg):
                continue
            rows.append(
                {
                    "pnu": pnu,
                    "sgg_cd": sgg,
                    "complex_name": (rec.get("COMPLEX_NM2") or rec.get("COMPLEX_NM1") or "").strip(),
                    "address": (rec.get("ADRES") or "").strip(),
                    "household_count": str(rec.get("UNIT_CNT") or ""),
                    "building_count": str(rec.get("DONG_CNT") or ""),
                    "use_approval_date": (rec.get("USEAPR_DT") or "").strip(),
                    "complex_type_code": (rec.get("COMPLEX_GB_CD") or "").strip(),
                }
            )
        if len(data) < PAGE_SIZE:
            return rows, calls
        page += 1
        if sleep:
            time.sleep(sleep)


def read_existing() -> list[dict]:
    if not OUT_FILE.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(OUT_FILE.read_bytes()))


def write_master(rows: list[dict]) -> bool:
    """PNU 기준 중복 제거 후 결정론적으로 쓴다. 내용이 같으면 건드리지 않는다."""
    merged: dict[str, dict] = {}
    for row in rows:
        merged[row["pnu"]] = row
    ordered = [merged[k] for k in sorted(merged)]

    import csv
    import io

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({c: row.get(c, "") for c in COLUMNS})

    data = rtms.gzip_bytes(buf.getvalue())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        return False
    OUT_FILE.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=600, help="이번 실행 최대 호출 수")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="이미 받은 시군구도 다시 받는다")
    args = parser.parse_args()

    key = load_api_key()
    state = (
        json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if STATE_FILE.exists()
        else {"version": 1, "done": {}}
    )
    done = state["done"]

    sggs = [c for c, _ in regions.sgg_codes()]
    pending = sggs if args.force else [s for s in sggs if s not in done]
    print(f"단지 마스터 · 대상 시군구 {len(pending)}/{len(sggs)} · 예산 {args.max_calls}콜")

    rows = read_existing()
    budget = args.max_calls
    for sgg in pending:
        if budget <= 0:
            print("예산 소진. 다음 실행에서 이어서 진행합니다.")
            break
        fetched, calls = collect_sgg(key, sgg, args.sleep)
        budget -= calls
        rows.extend(fetched)
        done[sgg] = len(fetched)
        big = sum(1 for r in fetched if (r["household_count"] or "0").isdigit()
                  and int(r["household_count"]) >= 300)
        print(f"  {sgg} 단지 {len(fetched):>5}개 (300세대+ {big:>4}) · {calls}콜 · 잔여 {budget}")
        if args.sleep:
            time.sleep(args.sleep)

    changed = write_master(rows)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = date.today().isoformat()
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    total = len(read_existing())
    print(f"\n단지 마스터 {total}건 · 파일 {'갱신' if changed else '변화 없음'} · 완료 {len(done)}/{len(sggs)} 시군구")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
