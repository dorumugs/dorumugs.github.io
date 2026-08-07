"""포켓몬 카드 현재가를 TCGdex 에서 받아 저장한다.

    python3 scripts/collect_pokemon.py                    # 전 카드 갱신
    python3 scripts/collect_pokemon.py --max-calls 6000   # 예산 끊어 나눠 받기
    python3 scripts/collect_pokemon.py --species          # 한글 이름 1회 수집

가격이 잡히는 카드 전부(약 17,700장)를 매일 다시 받는다. 동시 4로 약 70분
걸린다. 실측으로 8이면 두 배 빠르지만 무료 API 를 그렇게 두들길 이유가 없다.

시계열을 쌓지 않는다. 현재가만 보는 화면이라 하루치 이력을 1만 7천 줄씩
누적할 이유가 없다 — 대신 관측 최고가(obs_max)를 갱신하며 이어간다.

세트 하나가 끝날 때마다 저장한다. 70분짜리 실행이 중간에 죽어도 다음 실행이
이어받는다.
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

import pokeapi  # noqa: E402
import rtms  # noqa: E402
import tcgdex_api  # noqa: E402

DATA_DIR = ROOT / "data" / "pokemon"
CARDS_FILE = DATA_DIR / "cards.csv.gz"
SETS_FILE = DATA_DIR / "sets.json"
SPECIES_FILE = DATA_DIR / "species_ko.json"
STATE_FILE = DATA_DIR / "sweep_state.json"

WORKERS = 4
TIMEOUT = 30
RETRIES = 3
USER_AGENT = "kayserdocs-pokemon/1.0"

# 세트가 목록에 들어가려면 이 비율 이상이 가격을 가져야 한다. 프로모·트레이너킷은
# 가격이 아예 안 잡혀서 (2026-08-07 실측) 여기서 걸러진다.
MIN_COVERAGE = 0.8

# 1차 프리필터. 최종 판정은 가격 커버리지 실측치다.
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
    """tcgdex_api.CARD_COLUMNS 순서로 쓴다. None 은 빈 칸."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=tcgdex_api.CARD_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: ("" if row.get(c) is None else row.get(c))
                         for c in tcgdex_api.CARD_COLUMNS})
    return buf.getvalue()


def csv_to_rows(text: str) -> list[dict]:
    """CSV 를 행 목록으로. 숫자 칸은 float 로 되돌린다."""
    if not text.strip():
        return []
    out = []
    numeric = set(tcgdex_api.PRICE_COLUMNS) | {"obs_max"}
    for row in csv.DictReader(io.StringIO(text)):
        for col in numeric:
            row[col] = float(row[col]) if row.get(col) else None
        out.append(row)
    return out


def merge_cards(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """카드 단위로 겹친다. 관측 최고가는 tcgdex_api.merge_card 가 지킨다."""
    by_id = {r["card_id"]: r for r in existing}
    for row in new_rows:
        by_id[row["card_id"]] = tcgdex_api.merge_card(by_id.get(row["card_id"]), row)
    return sorted(by_id.values(), key=lambda r: r["card_id"])


def _get_json(url: str):
    """GET 후 JSON. 일시 오류는 재시도한다."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise tcgdex_api.ApiError(f"{url} 요청 실패: {last}")


def _fetch_cards(card_ids: list[str], on_date: str) -> list[dict]:
    """카드를 동시 WORKERS 개로 받는다. 실패한 카드는 조용히 빠진다."""
    def one(cid: str):
        try:
            return tcgdex_api.parse_card_full(
                _get_json(f"{tcgdex_api.BASE}/cards/{cid}"), on_date)
        except tcgdex_api.ApiError:
            return None

    with cf.ThreadPoolExecutor(WORKERS) as pool:
        return [r for r in pool.map(one, card_ids) if r]


def _read_gz(path: Path) -> str:
    return rtms.gunzip_text(path.read_bytes()) if path.exists() else ""


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rtms.gzip_bytes(text))


def _read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def collect_species() -> int:
    """도감번호 → 한글 이름. 1회만 받으면 되고 거의 바뀌지 않는다."""
    names = _read_json(SPECIES_FILE, {})
    todo = [i for i in range(1, pokeapi.SPECIES_MAX + 1) if str(i) not in names]
    if not todo:
        print(f"한글 이름 {len(names)}종 — 이미 다 받았습니다.")
        return 0

    def one(dex: int):
        try:
            return pokeapi.parse_species(_get_json(f"{pokeapi.BASE}/pokemon-species/{dex}/"))
        except (pokeapi.ApiError, tcgdex_api.ApiError):
            return None

    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for got in pool.map(one, todo):
            if got:
                names[got[0]] = got[1]

    _write_json(SPECIES_FILE, dict(sorted(names.items(), key=lambda kv: int(kv[0]))))
    print(f"한글 이름 {len(names)}종 저장")
    return 0


def run_sweep(max_calls: int, on_date: str) -> int:
    """가격 있는 카드를 전부 다시 받는다."""
    sets = tcgdex_api.parse_set_list(_get_json(f"{tcgdex_api.BASE}/sets"))
    cards = csv_to_rows(_read_gz(CARDS_FILE))
    set_meta = _read_json(SETS_FILE, {})
    state = _read_json(STATE_FILE, {})

    # 이번 회차에 이미 훑은 세트는 건너뛴다. 날짜가 바뀌면 처음부터 다시 돈다.
    done = set(state.get("done_sets") or []) if state.get("date") == on_date else set()

    def flush() -> None:
        _write_gz(CARDS_FILE, rows_to_csv(cards))
        _write_json(SETS_FILE, set_meta)
        _write_json(STATE_FILE, {"date": on_date, "done_sets": sorted(done),
                                 "complete": len(done) >= len(sets)})

    spent = 0
    for entry in sets:
        if entry["set_id"] in done:
            continue
        if spent >= max_calls:
            print(f"예산 {max_calls} 소진. 다음 실행에서 이어받습니다.")
            break

        detail = tcgdex_api.parse_set_detail(
            _get_json(f"{tcgdex_api.BASE}/sets/{entry['set_id']}"))
        spent += 1

        if not is_candidate_set(detail["name"], detail["release_date"]):
            set_meta[detail["set_id"]] = {
                "name": detail["name"], "release_date": detail["release_date"],
                "era": tcgdex_api.era_of(detail["release_date"]),
                "included": False, "reason": "이름 프리필터 제외", "coverage": None,
            }
            done.add(entry["set_id"])
            flush()
            continue

        card_ids = detail["card_ids"]
        rows = _fetch_cards(card_ids, on_date)
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
            cards = merge_cards(cards, rows)
        done.add(entry["set_id"])
        flush()
        print(f"  {detail['set_id']:<12} {detail['name'][:26]:<26} "
              f"{len(rows):>4}/{len(card_ids):<4} 커버리지 {cov:>4.0%} "
              f"{'포함' if included else '제외'}")

    flush()
    print(f"진행 {len(done)}/{len(sets)} 세트 · 카드 {len(cards):,}장")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=30000)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--species", action="store_true",
                        help="도감번호→한글 이름만 받고 끝낸다")
    args = parser.parse_args()

    if args.species:
        return collect_species()
    return run_sweep(args.max_calls, args.date)


if __name__ == "__main__":
    raise SystemExit(main())
