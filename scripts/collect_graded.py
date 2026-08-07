"""등급(PSA) 시세를 PokemonPriceTracker 에서 받아 저장한다.

    python3 scripts/collect_graded.py                 # 남은 크레딧만큼
    python3 scripts/collect_graded.py --min-price 200 # 대상 좁히기
    python3 scripts/collect_graded.py --max-cards 20  # 예산 직접 지정

왜 조금씩 나눠 받는가
---------------------
무료 등급이 하루 100 크레딧이고 카드 한 장에 2 크레딧이라 **하루 50장**이다.
우리 카드는 17,683장이지만 등급이 의미 있는 건 비싼 카드뿐이다.

  raw $500 이상    79장   ->  2일 주기
  raw $200 이상   270장   ->  5일 주기
  raw $100 이상   541장   -> 11일 주기   (기본값)

등급 시장은 거래가 워낙 얇아서 — 베이스셋 리자몽 PSA 10 이 최근 1년에 1건
팔렸다 — 이 주기로 충분하다. 더 넓게·더 자주 원하면 유료 등급($9.99/월,
하루 1만장)으로 올리면 되고, 코드는 그대로 쓴다.

비싼 카드부터 돌고, 가장 오래된 것을 먼저 갱신한다. 중간에 죽어도 상태
파일이 남아 다음 실행이 이어받는다.

인증키는 `.env` 의 `POKEMONPRICETRACKER_API_KEY`. 저장소에 두지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import ppt_api  # noqa: E402
import rtms  # noqa: E402

DATA_DIR = ROOT / "data" / "pokemon"
GRADED_FILE = DATA_DIR / "graded.csv.gz"
STATE_FILE = DATA_DIR / "graded_state.json"
ENV_FILE = ROOT / ".env"

USER_AGENT = "kayserdocs-pokemon/1.0"
TIMEOUT = 60
PAUSE = 1.1          # 분당 60 제한. 여유를 두고 천천히 간다.
MIN_PRICE = 100.0    # 이보다 싼 카드는 감정 자체가 수지가 안 맞는다


def api_key() -> str:
    key = os.environ.get("POKEMONPRICETRACKER_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("POKEMONPRICETRACKER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def fetch(path: str, key: str) -> tuple[int, dict, dict]:
    """(status, 헤더, 본문). 헤더에 남은 크레딧이 들어 있다."""
    request = urllib.request.Request(
        ppt_api.BASE_URL + path,
        headers={"Authorization": "Bearer " + key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), {}
    except Exception:
        return 0, {}, {}


def remaining_credits(headers: dict) -> int:
    for name in ("X-Ratelimit-Daily-Remaining", "X-RateLimit-Daily-Remaining"):
        if name in headers:
            try:
                return int(headers[name])
            except ValueError:
                pass
    return 0


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=ppt_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: ("" if row.get(c) is None else row.get(c))
                         for c in ppt_api.COLUMNS})
    return buf.getvalue()


def csv_to_rows(text: str) -> list[dict]:
    if not text.strip():
        return []
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        for key in ("psa10", "psa9", "ebay_raw"):
            row[key] = float(row[key]) if row.get(key) else None
        for key in ("psa10_n", "psa9_n", "ebay_raw_n"):
            row[key] = int(row[key]) if row.get(key) else 0
        out.append(row)
    return out


def targets(cards: list[dict], known: dict, min_price: float) -> list[dict]:
    """비쌀수록 먼저, 오래 안 본 것부터. 아직 한 번도 안 본 카드가 최우선."""
    def raw_price(card):
        try:
            return float(card.get("tp_market") or 0)
        except (TypeError, ValueError):
            return 0.0

    pool = [c for c in cards
            if raw_price(c) >= min_price and (c.get("tp_product_id") or "").strip()]
    pool.sort(key=lambda c: (known.get(c["card_id"], {}).get("updated", ""),
                             -raw_price(c)))
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-price", type=float, default=MIN_PRICE,
                        help=f"이 raw 가격 이상만 본다 (기본 ${MIN_PRICE:.0f})")
    parser.add_argument("--max-cards", type=int, default=0,
                        help="0 이면 남은 크레딧이 허락하는 만큼")
    parser.add_argument("--reserve", type=int, default=4,
                        help="다른 용도로 남겨둘 크레딧")
    args = parser.parse_args()

    key = api_key()
    if not key:
        print("POKEMONPRICETRACKER_API_KEY 가 없습니다. .env 를 확인하세요.",
              file=sys.stderr)
        return 1

    if not collect_pokemon.CARDS_FILE.exists():
        print("카드 원본이 없습니다. collect_pokemon.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1

    cards = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    existing = csv_to_rows(collect_pokemon._read_gz(GRADED_FILE))
    known = {r["card_id"]: r for r in existing}

    pool = targets(cards, known, args.min_price)
    if not pool:
        print(f"raw ${args.min_price:.0f} 이상인 카드가 없습니다.")
        return 0

    # 첫 호출로 남은 크레딧을 확인한다. 그 자체도 크레딧을 쓴다.
    on_date = date.today().isoformat()
    first = pool[0]
    status, headers, payload = fetch(
        f"/cards?tcgPlayerId={urllib.parse.quote(first['tp_product_id'])}"
        "&includeEbay=true&days=90", key)
    if status != 200:
        print(f"첫 호출이 실패했습니다 (HTTP {status}). 키와 잔여 크레딧을 확인하세요.",
              file=sys.stderr)
        return 1

    left = remaining_credits(headers)
    budget = args.max_cards or ppt_api.daily_budget(left, args.reserve)
    print(f"대상 {len(pool):,}장 (raw ${args.min_price:.0f} 이상) · "
          f"남은 크레딧 {left} · 이번에 볼 카드 {budget:,}장")

    done = 0
    for card in pool[:max(1, budget)]:
        if done:  # 첫 장은 위에서 이미 받았다
            status, headers, payload = fetch(
                f"/cards?tcgPlayerId={urllib.parse.quote(card['tp_product_id'])}"
                "&includeEbay=true&days=90", key)
            if status == 429:
                print("크레딧이 떨어졌습니다. 여기까지 저장하고 멈춥니다.")
                break
            if status != 200:
                print(f"  {card['card_id']}: HTTP {status} — 건너뜁니다", file=sys.stderr)
                time.sleep(PAUSE)
                continue

        found = ppt_api.cards_from_response(payload)
        if found:
            row = ppt_api.parse_card(found[0], card["card_id"], on_date)
            if ppt_api.has_any_grade(row):
                known[row["card_id"]] = row
            else:
                # 등급 거래가 없는 카드도 기록해 둔다. 안 그러면 매일 다시 묻는다.
                known[row["card_id"]] = row
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{budget}")
        time.sleep(PAUSE)

    rows = sorted(known.values(), key=lambda r: r["card_id"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRADED_FILE.write_bytes(rtms.gzip_bytes(rows_to_csv(rows)))
    STATE_FILE.write_text(json.dumps({
        "date": on_date, "min_price": args.min_price,
        "pool": len(pool), "collected": len(rows),
        "with_psa": sum(1 for r in rows if ppt_api.has_any_grade(r)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    with_psa = sum(1 for r in rows if ppt_api.has_any_grade(r))
    print(f"이번에 {done}장 · 누적 {len(rows):,}장 (PSA 값 있음 {with_psa:,}장) "
          f"-> {GRADED_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
