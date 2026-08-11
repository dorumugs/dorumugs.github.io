#!/usr/bin/env python3
"""미국 상장 ETF 일봉을 받아 data/stocks/ 에 쌓는다.

파싱은 naver_us_api.py 가 한다. 여기는 I/O 와 예산만 다룬다.

받는 것

  1. 티커 해석   data/us_etf_seed.txt 의 티커를 자동완성에 물어 네이버 코드와
                 정식 이름을 얻는다. 거래소 접미사(.O/.K/없음)가 여기서 정해진다.
                 주 1회. 씨앗이 바뀌면 그날 바로 다시 푼다.
  2. 일봉        종목당 요청 1번에 3.5년(약 900행)이 온다.
  3. S&P500      벤치마크. 미국 ETF 를 코스피와 비교하는 건 말이 안 된다.
  4. 원달러      원화 환산과 '환율이 수익률에 얼마나 섞였나' 를 보여주는 데 쓴다.

왜 별도 파일인가

  통화가 다르다. 국내 캐시(bars.csv.gz)에 섞으면 종가가 원인지 달러인지 코드가
  기억해야 하는데, 그런 건 언젠가 틀린다. 거래일 달력도 다르다 — 미국은 추수감사절에
  쉬고 한국은 추석에 쉰다.

기준일이 하루 어긋나는 건 정상이다. 미국 종가는 한국 시간 다음 날 새벽에
확정되므로, 저녁 크론이 받는 마지막 줄은 '어제 미국 장' 이다.

사용법

    python3 scripts/collect_us_etf.py
    python3 scripts/collect_us_etf.py --full-bars     # 일봉 전체 재수집
    python3 scripts/collect_us_etf.py --refresh-seed  # 티커 강제 재해석
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import pathlib
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import naver_us_api as us  # noqa: E402
from collect_stocks import Budget, fetch, load_state, save_state, stale, write_json_gz, read_json_gz  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"
SEED_FILE = REPO / "data" / "us_etf_seed.txt"

UNIVERSE_FILE = DATA / "us_universe.json.gz"
BARS_FILE = DATA / "us_bars.csv.gz"
FX_FILE = DATA / "us_fx.json.gz"
STATE_FILE = DATA / "state.json"

URL_AC = "https://ac.stock.naver.com/ac?q={q}&target=stock"
URL_CHART = (
    "https://api.stock.naver.com/chart/foreign/item/{code}/day"
    "?startDateTime={start}0000&endDateTime={end}0000"
)
URL_INDEX = (
    "https://api.stock.naver.com/chart/foreign/index/{code}/day"
    "?startDateTime={start}0000&endDateTime={end}0000"
)
# 환율은 한 번에 많이 못 준다. pageSize 90 이면 400 이 떨어지고 60 까지만 받는다.
# 그래서 페이지를 넘겨 가며 모은다.
URL_FX = "https://api.stock.naver.com/marketindex/exchange/FX_USDKRW/prices?page={page}&pageSize={size}"
FX_PAGE_SIZE = 60
FX_PAGES = 16          # 60 × 16 ≈ 960 영업일 (약 3.7년)

# 벤치마크. 미국 ETF 는 S&P500 과 견준다.
INDEX_SYMBOL = ".INX"
INDEX_KEY = "SPX"

SEED_MAX_AGE_DAYS = 7
FULL_BARS_MAX_AGE_DAYS = 30
FULL_HISTORY_DAYS = 1300
INCREMENTAL_DAYS = 30
KEEP_HISTORY_DAYS = 900

BARS_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def resolve_universe(tickers: list[str], budget: Budget, workers: int) -> list[dict]:
    """씨앗 티커를 네이버 코드로 푼다.

    같은 티커라도 거래소마다 코드가 달라(.O/.K/없음) 규칙으로는 못 맞춘다.
    자동완성에 물어보는 게 유일하게 확실한 방법이다.
    """
    found: list[dict] = []
    missing: list[str] = []

    def one(ticker: str) -> None:
        raw = fetch(URL_AC.format(q=urllib.parse.quote(ticker)), budget)
        if raw is None:
            missing.append(ticker)
            return
        row = us.parse_autocomplete(raw, ticker)
        if row:
            found.append(row)
        else:
            missing.append(ticker)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, tickers))

    found.sort(key=lambda r: r["ticker"])
    etf = sum(1 for r in found if r["etf"])
    print(f"  티커 해석 {len(found)}/{len(tickers)} (ETF {etf} · ETN 등 {len(found) - etf})")
    if missing:
        print(f"  못 찾음 {len(missing)}: {' '.join(sorted(missing)[:20])}", file=sys.stderr)
    return found


def load_bars() -> dict[str, dict[str, list]]:
    bars: dict[str, dict[str, list]] = {}
    if not BARS_FILE.exists():
        return bars
    try:
        with gzip.open(BARS_FILE, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                bars.setdefault(row["symbol"], {})[row["date"]] = [
                    float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row["close"]), int(row["volume"]),
                ]
    except (OSError, csv.Error, ValueError, KeyError):
        return {}
    return bars


def save_bars(bars: dict[str, dict[str, list]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(BARS_COLUMNS)
    for symbol in sorted(bars):
        for day in sorted(bars[symbol]):
            writer.writerow([symbol, day] + list(bars[symbol][day]))
    body = buffer.getvalue().encode("utf-8")
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(BARS_FILE, "wb"), mtime=0) as fh:
        fh.write(body)


def collect_bars(codes: list[str], bars: dict, budget: Budget, workers: int, span: int) -> int:
    start = (date.today() - timedelta(days=span)).strftime("%Y%m%d")
    end = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    ok = 0

    def one(code: str) -> None:
        nonlocal ok
        url = (URL_INDEX if code.startswith(".") else URL_CHART).format(
            code=urllib.parse.quote(code), start=start, end=end
        )
        raw = fetch(url, budget)
        if raw is None:
            return
        rows = us.parse_chart(raw)
        if not rows:
            return
        key = INDEX_KEY if code == INDEX_SYMBOL else code
        slot = bars.setdefault(key, {})
        for row in rows:
            slot[row["date"]] = [row["open"], row["high"], row["low"], row["close"], row["volume"]]
        ok += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, codes))
    return ok


def trim(bars: dict[str, dict[str, list]]) -> None:
    for symbol, series in bars.items():
        if len(series) > KEEP_HISTORY_DAYS:
            keep = sorted(series)[-KEEP_HISTORY_DAYS:]
            bars[symbol] = {d: series[d] for d in keep}


def main() -> int:
    parser = argparse.ArgumentParser(description="미국 ETF 일봉 수집")
    parser.add_argument("--max-calls", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-bars", action="store_true")
    parser.add_argument("--refresh-seed", action="store_true")
    args = parser.parse_args()

    if not SEED_FILE.exists():
        print(f"{SEED_FILE} 가 없습니다.", file=sys.stderr)
        return 1
    tickers = us.parse_seed(SEED_FILE.read_text(encoding="utf-8"))
    DATA.mkdir(parents=True, exist_ok=True)
    state = load_state()
    budget = Budget(args.max_calls)
    failed = False

    universe = read_json_gz(UNIVERSE_FILE)
    # 씨앗을 고쳤으면 낡음과 무관하게 바로 다시 푼다.
    seed_changed = bool(universe) and {r["ticker"] for r in universe} != set(tickers)
    if args.refresh_seed or universe is None or seed_changed or stale(state, "us_seed_date", SEED_MAX_AGE_DAYS):
        print(f"씨앗 {len(tickers)}개를 네이버 코드로 풉니다." + (" (씨앗 변경 감지)" if seed_changed else ""))
        fresh = resolve_universe(tickers, budget, args.workers)
        if len(fresh) >= len(tickers) * 0.8:
            universe = fresh
            write_json_gz(UNIVERSE_FILE, universe)
            state["us_seed_date"] = date.today().isoformat()
        else:
            failed = True
            print("  해석이 80%도 안 됐습니다 — 지난 목록을 그대로 씁니다.", file=sys.stderr)
    if not universe:
        print("미국 ETF 목록이 없어 진행할 수 없습니다.", file=sys.stderr)
        return 1

    full = args.full_bars or not BARS_FILE.exists() or stale(
        state, "us_full_bars_date", FULL_BARS_MAX_AGE_DAYS
    )
    bars = load_bars()
    codes = [r["code"] for r in universe] + [INDEX_SYMBOL]
    span = FULL_HISTORY_DAYS if full else INCREMENTAL_DAYS
    print(f"일봉을 받습니다 — {len(codes)}종목, {'전체 3.5년' if full else f'증분 {span}일'}")
    ok = collect_bars(codes, bars, budget, args.workers, span)
    print(f"  일봉 {ok}/{len(codes)}종목")
    if ok < len(codes) * 0.8:
        failed = True
        print("  일봉이 80%도 안 왔습니다.", file=sys.stderr)

    # 환율. 원화 환산과 '환율이 얼마나 섞였나' 에 쓴다.
    merged: dict[str, float] = {}
    for page in range(1, FX_PAGES + 1):
        raw = fetch(URL_FX.format(page=page, size=FX_PAGE_SIZE), budget)
        rows = us.parse_fx(raw) if raw else []
        if not rows:
            break
        before = len(merged)
        for row in rows:
            merged[row["date"]] = row["close"]
        if len(merged) == before:      # 같은 페이지가 반복되면 끝이다
            break
    fx = [{"date": d, "close": merged[d]} for d in sorted(merged)]
    if fx:
        write_json_gz(FX_FILE, fx)
        print(f"  원달러 {len(fx)}일 · 최근 {fx[-1]['date']} {fx[-1]['close']:,.2f}원")
    else:
        failed = True
        print("  환율을 못 받았습니다 — 지난 값을 그대로 씁니다.", file=sys.stderr)

    trim(bars)
    save_bars(bars)
    if full and ok >= len(codes) * 0.8:
        state["us_full_bars_date"] = date.today().isoformat()
    state["us_bars_date"] = date.today().isoformat()
    state["us_last_run"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    print(f"요청 {budget.used}건 사용, 캐시 {len(bars)}종목")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
