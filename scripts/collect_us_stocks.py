#!/usr/bin/env python3
"""미국 3배 레버리지 ETF **구성종목**의 일봉을 받아 data/stocks/ 에 쌓는다.

왜 따로 받나

  us_bars.csv.gz 에는 ETF 192개만 있다. 그것만으로는 '이 ETF 가 담은 종목 중
  몇 %가 올랐나' 를 못 잰다 — ETF 가 오른 건 결과지 원인이 아니다. 국내 화면이
  테마 구성종목으로 폭을 재듯, 미국도 구성종목 일봉이 있어야 같은 것을 잰다.

  ETF 와 섞지 않고 us_stock_bars.csv.gz 로 나눈 이유는 us_bars.csv.gz 를 읽는
  코드가 전부 '여기 있는 건 ETF' 를 전제하기 때문이다 (universe 와 1:1 로 돈다).
  섞으면 그 전제가 조용히 깨진다.

무엇을 받나

  1. 티커 해석   3배 **불** ETF 가 들고 있는 종목 티커를 자동완성에 물어
                 네이버 코드를 얻는다. 주 1회. 보유 종목이 바뀌면 그날 다시 푼다.
  2. 일봉        종목당 요청 1번. 폭 계산에 30 거래일이면 되지만 한 번에
                 오는 김에 그대로 받아 둔다.

  베어 3배(SQQQ·SOXS 등)는 스왑만 들고 있어 구성종목이 0개다. 짝이 되는 불
  ETF 의 종목을 빌려 오지 않는다 — 인버스에서 '구성종목 상승' 은 그 ETF 가
  내린다는 뜻이라 색이 거꾸로 읽힌다.

사용법

    python3 scripts/collect_us_stocks.py
    python3 scripts/collect_us_stocks.py --full-bars      # 일봉 전체 재수집
    python3 scripts/collect_us_stocks.py --refresh-seed   # 티커 강제 재해석
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import pathlib
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import naver_us_api as us  # noqa: E402
from collect_stocks import (  # noqa: E402
    Budget, fetch, load_state, save_state, stale, write_json_gz, read_json_gz,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"

US_UNIVERSE_FILE = DATA / "us_universe.json.gz"
HOLDINGS_FILE = DATA / "us_holdings.json.gz"
UNIVERSE_FILE = DATA / "us_stock_universe.json.gz"
BARS_FILE = DATA / "us_stock_bars.csv.gz"

URL_AC = "https://ac.stock.naver.com/ac?q={q}&target=stock"
URL_CHART = (
    "https://api.stock.naver.com/chart/foreign/item/{code}/day"
    "?startDateTime={start}0000&endDateTime={end}0000"
)

SEED_MAX_AGE_DAYS = 7
FULL_BARS_MAX_AGE_DAYS = 30
FULL_HISTORY_DAYS = 400     # 폭은 30 거래일이면 되지만 넉넉히
INCREMENTAL_DAYS = 30
KEEP_HISTORY_DAYS = 260

BARS_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def wanted_tickers() -> list[str]:
    """3배 불 ETF 가 들고 있는 개별 종목 티커.

    build_us_etf 의 판정과 어긋나면 안 되므로 같은 함수를 쓴다.
    """
    import build_us_etf as bu  # noqa: PLC0415 — 무거운 임포트를 여기서만

    universe = read_json_gz(US_UNIVERSE_FILE) or []
    holdings = read_json_gz(HOLDINGS_FILE) or {}
    rows = [{"ticker": r["ticker"], "lev": bu.leverage_of(r["name"])} for r in universe]
    return bu.holding_tickers(holdings, bu.lev3_bull_tickers(rows))


def resolve(tickers: list[str], budget: Budget, workers: int) -> list[dict]:
    found: list[dict] = []
    missing: list[str] = []

    def one(ticker: str) -> None:
        raw = fetch(URL_AC.format(q=urllib.parse.quote(ticker)), budget)
        row = us.parse_autocomplete(raw, ticker) if raw is not None else None
        (found if row else missing).append(row or ticker)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, tickers))
    found.sort(key=lambda r: r["ticker"])
    print(f"  티커 해석 {len(found)}/{len(tickers)}")
    if missing:
        # 못 찾은 종목은 폭 계산의 분모에서 빠진다. 조용히 빠지면 폭이 낮게
        # 나오는 이유를 못 찾으므로 반드시 이름을 남긴다.
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
    """gzip mtime 을 0 으로 고정한다. 안 그러면 내용이 같아도 매일 새 blob 이
    쌓여 저장소가 계속 커진다."""
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
        raw = fetch(URL_CHART.format(code=urllib.parse.quote(code), start=start, end=end), budget)
        if raw is None:
            return
        rows = us.parse_chart(raw)
        if not rows:
            return
        slot = bars.setdefault(code, {})
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
    parser = argparse.ArgumentParser(description="미국 3배 ETF 구성종목 일봉 수집")
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-bars", action="store_true")
    parser.add_argument("--refresh-seed", action="store_true")
    args = parser.parse_args()

    tickers = wanted_tickers()
    if not tickers:
        print("3배 불 ETF 의 구성종목이 없습니다. collect_us_holdings.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1
    print(f"3배 불 ETF 구성종목 {len(tickers)}개")

    DATA.mkdir(parents=True, exist_ok=True)
    state = load_state()
    budget = Budget(args.max_calls)

    universe = read_json_gz(UNIVERSE_FILE)
    known = {r["ticker"] for r in universe} if universe else set()
    # 보유 종목이 바뀌었으면 낡음과 무관하게 새 티커만 바로 푼다.
    fresh = [t for t in tickers if t not in known]
    if args.refresh_seed or universe is None or stale(state, "us_stock_seed_date", SEED_MAX_AGE_DAYS):
        universe = resolve(tickers, budget, args.workers)
        state["us_stock_seed_date"] = date.today().isoformat()
        write_json_gz(UNIVERSE_FILE, universe)
    elif fresh:
        print(f"  새 구성종목 {len(fresh)}개만 해석")
        universe = universe + resolve(fresh, budget, args.workers)
        universe.sort(key=lambda r: r["ticker"])
        write_json_gz(UNIVERSE_FILE, universe)

    # 지금 3배 불이 안 담는 종목은 굳이 매일 받지 않는다.
    wanted = set(tickers)
    codes = [r["code"] for r in universe if r["ticker"] in wanted]
    if not codes:
        print("해석된 코드가 없습니다.", file=sys.stderr)
        return 1

    bars = load_bars()
    full = args.full_bars or not bars or stale(state, "us_stock_bars_full", FULL_BARS_MAX_AGE_DAYS)
    span = FULL_HISTORY_DAYS if full else INCREMENTAL_DAYS
    ok = collect_bars(codes, bars, budget, args.workers, span)
    print(f"  일봉 {ok}/{len(codes)} ({'전체' if full else '증분'} {span}일)")
    if full and ok:
        state["us_stock_bars_full"] = date.today().isoformat()

    if not ok:
        print("일봉을 하나도 못 받았습니다.", file=sys.stderr)
        return 1

    trim(bars)
    save_bars(bars)
    state["us_stock_date"] = date.today().isoformat()
    save_state(state)

    days = sorted({d for s in bars.values() for d in s})
    print(f"  종목 {len(bars)}개 · 거래일 {len(days)}일 "
          f"({days[0]}~{days[-1]}) · {BARS_FILE.stat().st_size / 1024:.0f}KB")
    print(f"  API 호출 {budget.used}회")
    return 0


if __name__ == "__main__":
    sys.exit(main())
