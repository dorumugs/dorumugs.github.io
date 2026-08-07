"""포켓몬 카드 가격을 TCGdex 에서 받아 저장한다.

    python3 scripts/collect_pokemon.py --mode scan  --max-calls 6000
    python3 scripts/collect_pokemon.py --mode daily

두 단계로 나뉜다.

  scan   정규 확장팩 후보 2만여 장을 훑어 가격 분포를 만든다. 1회성이고
         --max-calls 로 하루 예산을 끊어 여러 날에 나눠 받는다.
  daily  확정된 유니버스 300장만 매일 받는다. 약 70초.

동시 요청은 4로 고정한다. 실측으로 8이면 두 배 빠르지만 무료 API 를 그렇게
두들길 이유가 없다 (4로도 2만 장에 78분).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rtms  # noqa: E402
import tcgdex_api  # noqa: E402

DATA_DIR = ROOT / "data" / "pokemon"
SCAN_STATE = DATA_DIR / "scan_state.json"
SCAN_FILE = DATA_DIR / "scan.csv.gz"
SETS_FILE = DATA_DIR / "sets.json"
PRICES_FILE = DATA_DIR / "prices.csv.gz"
UNIVERSE_FILE = DATA_DIR / "universe.json"
NAMES_FILE = DATA_DIR / "names.json"

WORKERS = 4
TIMEOUT = 30
RETRIES = 3

# 세트가 가격 유니버스에 들어가려면 이 비율 이상이 가격을 가져야 한다.
MIN_COVERAGE = 0.8

# 1차 프리필터. 최종 판정은 가격 커버리지 실측치다 (MIN_COVERAGE).
EXCLUDE_PATTERN = re.compile(
    r"promo|trainer kit|collection|deck|tin|box|kit|misc|jumbo|energy", re.I
)


def is_candidate_set(name: str, release_date: str) -> bool:
    """정규 확장팩 후보인지. 이름 프리필터 + 발매일 유무."""
    if not release_date:
        return False
    return not EXCLUDE_PATTERN.search(name or "")


def coverage_of(rows: list[dict], card_ids: list[str]) -> float:
    """세트 카드 중 가격이 잡힌 비율."""
    if not card_ids:
        return 0.0
    priced = {r["card_id"] for r in rows}
    return len([c for c in card_ids if c in priced]) / len(card_ids)


def rows_to_csv(rows: list[dict]) -> str:
    """tcgdex_api.COLUMNS 순서로 쓴다. None 은 빈 칸."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=tcgdex_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in tcgdex_api.COLUMNS})
    return buf.getvalue()


def merge_rows(existing: str, new_rows: list[dict]) -> str:
    """기존 CSV 에 새 행을 합친다. (date, card_id) 가 같으면 새 값으로 덮는다.

    같은 날 두 번 돌려도 행이 불어나지 않아야 한다 — 수동 실행과 크론이
    겹치는 일이 실제로 생긴다.
    """
    merged: dict[tuple[str, str], dict] = {}
    if existing.strip():
        for row in csv.DictReader(io.StringIO(existing)):
            merged[(row["date"], row["card_id"])] = row
    for row in new_rows:
        merged[(row["date"], row["card_id"])] = row
    ordered = sorted(merged.values(), key=lambda r: (r["date"], r["card_id"]))
    return rows_to_csv(ordered)


def load_universe_ids(universe: dict) -> list[str]:
    """유니버스 JSON 에서 카드 id 를 순서대로 꺼낸다."""
    return [c["card_id"] for c in universe["cards"]]


def _get_json(url: str):
    """GET 후 JSON. 일시 오류는 재시도한다."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kayserdocs-pokemon/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise tcgdex_api.ApiError(f"{url} 요청 실패: {last}")


def _fetch_cards(card_ids: list[str], on_date: str, names: dict | None = None) -> list[dict]:
    """카드 가격을 동시 WORKERS 개로 받는다. 실패한 카드는 조용히 빠진다.

    names 를 주면 card_id → 카드 이름을 함께 채운다. 가격 CSV 에는 이름 칸이
    없지만(컬럼이 고정) 화면 표에는 이름이 필요하다. 어차피 응답을 받는 김에
    같이 걷어 둔다.
    """
    def one(cid: str):
        try:
            payload = _get_json(f"{tcgdex_api.BASE}/cards/{cid}")
        except tcgdex_api.ApiError:
            return None
        if names is not None and payload.get("name"):
            names[cid] = payload["name"]
        return tcgdex_api.parse_card_pricing(payload, on_date)

    with cf.ThreadPoolExecutor(WORKERS) as pool:
        return [r for r in pool.map(one, card_ids) if r]


def _read_gz(path: Path) -> str:
    return rtms.gunzip_text(path.read_bytes()) if path.exists() else ""


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rtms.gzip_bytes(text))


def _load_state() -> dict:
    if SCAN_STATE.exists():
        return json.loads(SCAN_STATE.read_text(encoding="utf-8"))
    return {"done_sets": [], "complete": False}


def _save_state(state: dict) -> None:
    SCAN_STATE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def run_scan(max_calls: int, on_date: str) -> int:
    """정규 확장팩 후보를 세트 단위로 훑는다. 예산이 떨어지면 중단하고 다음에 잇는다."""
    state = _load_state()
    if state.get("complete"):
        print("전수 스캔이 이미 끝났습니다. --mode daily 를 쓰세요.")
        return 0

    sets = tcgdex_api.parse_set_list(_get_json(f"{tcgdex_api.BASE}/sets"))
    done = set(state.get("done_sets") or [])
    scanned = _read_gz(SCAN_FILE)
    set_meta: dict[str, dict] = {}
    if SETS_FILE.exists():
        set_meta = json.loads(SETS_FILE.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    if NAMES_FILE.exists():
        names = json.loads(NAMES_FILE.read_text(encoding="utf-8"))

    spent = 0
    for entry in sets:
        if entry["set_id"] in done:
            continue
        if spent >= max_calls:
            print(f"예산 {max_calls} 소진. 다음 실행에서 이어받습니다.")
            break

        detail = tcgdex_api.parse_set_detail(_get_json(f"{tcgdex_api.BASE}/sets/{entry['set_id']}"))
        spent += 1
        if not is_candidate_set(detail["name"], detail["release_date"]):
            set_meta[detail["set_id"]] = {
                "name": detail["name"], "release_date": detail["release_date"],
                "included": False, "reason": "이름 프리필터 제외", "coverage": None,
            }
            done.add(entry["set_id"])
            continue

        card_ids = detail["card_ids"]
        rows = _fetch_cards(card_ids, on_date, names)
        spent += len(card_ids)

        cov = coverage_of(rows, card_ids)
        included = cov >= MIN_COVERAGE
        set_meta[detail["set_id"]] = {
            "name": detail["name"], "release_date": detail["release_date"],
            "era": tcgdex_api.era_of(detail["release_date"]),
            "included": included,
            "reason": "" if included else f"가격 커버리지 {cov:.0%} < {MIN_COVERAGE:.0%}",
            "coverage": round(cov, 4), "card_count": len(card_ids),
        }
        if included:
            scanned = merge_rows(scanned, rows)
        done.add(entry["set_id"])
        print(f"  {detail['set_id']:<12} {detail['name'][:28]:<28} "
              f"{len(rows):>4}/{len(card_ids):<4} 커버리지 {cov:>4.0%} "
              f"{'포함' if included else '제외'}")

    _write_gz(SCAN_FILE, scanned)
    SETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETS_FILE.write_text(json.dumps(set_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    NAMES_FILE.write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    state["done_sets"] = sorted(done)
    state["complete"] = len(done) >= len(sets)
    _save_state(state)

    print(f"진행 {len(done)}/{len(sets)} 세트")
    if state["complete"]:
        print("전수 스캔 완료. build_pokemon.py 로 유니버스를 확정하세요.")
    return 0


def run_daily(on_date: str) -> int:
    """유니버스 카드만 받아 prices.csv.gz 에 덧쓴다."""
    if not UNIVERSE_FILE.exists():
        print(
            f"{UNIVERSE_FILE} 가 없습니다. 전수 스캔을 끝내고 build_pokemon.py 로 "
            "유니버스를 확정하세요.",
            file=sys.stderr,
        )
        return 1

    universe = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    card_ids = load_universe_ids(universe)
    rows = _fetch_cards(card_ids, on_date)

    # 절반도 못 받았으면 API 가 이상한 것이다. 반쪽짜리 하루를 시계열에 넣으면
    # 지수가 그날만 튀고, 그 이유를 나중에 알아내기 어렵다.
    if len(rows) < len(card_ids) // 2:
        print(f"수신 {len(rows)}/{len(card_ids)} 건 — 절반 미만이라 저장하지 않습니다.",
              file=sys.stderr)
        return 1

    _write_gz(PRICES_FILE, merge_rows(_read_gz(PRICES_FILE), rows))
    print(f"{on_date}: {len(rows)}/{len(card_ids)} 건 저장")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scan", "daily"), default="daily")
    parser.add_argument("--max-calls", type=int, default=6000)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    if args.mode == "scan":
        return run_scan(args.max_calls, args.date)
    return run_daily(args.date)


if __name__ == "__main__":
    raise SystemExit(main())
