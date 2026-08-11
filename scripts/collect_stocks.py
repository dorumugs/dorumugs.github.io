#!/usr/bin/env python3
"""네이버 금융에서 테마·업종·ETF·일봉을 받아 data/stocks/ 에 쌓는다.

파싱은 naver_stock_api.py 가 한다. 여기는 I/O 와 예산과 재개만 다룬다.

받는 것과 주기

  그룹 목록·구성종목   주 1회   344 요청  테마 265 · 업종 79
  시장 구분표          주 1회    87 요청  코스피/코스닥 소속. 벤치마크 고르기용
  ETF 목록             매일       1 요청  1,160개
  ETF 구성종목         주 1회  1,160 요청  종목명만 온다(코드 없음)
  일봉                 매일   약 5,600     증분 30일 / 월 1회 전체 250일

주기는 날짜로 가르지 않고 **파일이 얼마나 낡았는지**로 가른다. 크론이 하루
걸러도 다음 실행이 알아서 메운다. 요일로 가르면 그날 실패한 주는 한 주를 통째로
날린다.

일봉은 왜 다시 받나

  수정주가 때문이다. 액면분할·병합이 나면 네이버는 **과거 종가를 통째로 고쳐서**
  준다. 증분만 이어붙이면 분할 이전 구간이 옛 가격으로 남아 20일 수익률이
  엉뚱하게 나온다. 그래서 매일은 최근 30일만 덮어쓰고, 30일에 한 번은 250일을
  통째로 다시 받아 갈아끼운다.

캐시는 커밋하지 않는다

  data/stocks/ 는 .gitignore 로 막혀 있다. 5,600 심볼 × 120 거래일이 gzip 6MB 라
  매일 커밋하면 1년에 2GB 다. 언제든 2분이면 다시 받을 수 있는 재생성 가능
  캐시라 저장소에 넣을 값어치가 없다. 커밋하는 건 집계 결과뿐이다.

사용법

    python3 scripts/collect_stocks.py                 # 알아서 판단
    python3 scripts/collect_stocks.py --full-bars     # 일봉 전체 재수집
    python3 scripts/collect_stocks.py --refresh-groups
    python3 scripts/collect_stocks.py --max-calls 500 # 예산만 쓰고 멈춤
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import pathlib
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import naver_stock_api as api  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"

GROUPS_FILE = DATA / "groups.json.gz"
MARKET_FILE = DATA / "market.json.gz"
ETFS_FILE = DATA / "etfs.json.gz"
HOLDINGS_FILE = DATA / "holdings.json.gz"
BARS_FILE = DATA / "bars.csv.gz"
STATE_FILE = DATA / "state.json"

THEME_PAGES = 8
MARKET_PAGES = {0: 50, 1: 40}

GROUPS_MAX_AGE_DAYS = 7
HOLDINGS_MAX_AGE_DAYS = 7
FULL_BARS_MAX_AGE_DAYS = 30

# 보관 기간이 둘이다.
#
#   종목(4,405)      120 거래일. 테마 20일 지표를 만드는 데 그 이상은 필요 없다.
#   ETF·지수(1,162)  880 거래일(3.5년). 백테스트가 여기서 나온다.
#
# 왜 종목은 길게 안 받나. 백테스트를 테마 단위로 하려면 **과거 시점의 구성종목
# 명단**이 있어야 하는데 네이버가 그걸 안 준다. 오늘 명단으로 2년 전을 계산하면
# 올라서 편입된 종목이 과거에도 있던 걸로 잡혀 폭이 부풀려진다. 편향된 백테스트는
# 없느니만 못하다 — 숫자는 인용되는데 편향은 안 따라다니기 때문이다.
# 그래서 검증은 편향이 없는 ETF 가격 층에서만 한다.
#
# 전 종목을 3.5년 받으면 원본 300MB 다. ETF·지수만 받으면 63MB 로 끝난다.
FULL_BACKFILL_DAYS = 250
FULL_HISTORY_DAYS = 1300
INCREMENTAL_DAYS = 30
KEEP_TRADING_DAYS = 130
KEEP_HISTORY_DAYS = 900

INDEX_SYMBOLS = ["KOSPI", "KOSDAQ"]

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 KHTML, like Gecko Chrome/126 Safari/537.36"

URL_THEME_LIST = "https://finance.naver.com/sise/theme.naver?page={page}"
URL_UPJONG_LIST = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
URL_GROUP_DETAIL = "https://finance.naver.com/sise/sise_group_detail.naver?type={type}&no={no}"
URL_MARKET_SUM = "https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
URL_ETF_LIST = "https://finance.naver.com/api/sise/etfItemList.nhn"
URL_ETF_HOLDINGS = "https://navercomp.wisereport.co.kr/v2/ETF/index.aspx?cmp_cd={code}&target=cu_more"
URL_SISE = (
    "https://api.finance.naver.com/siseJson.naver"
    "?symbol={symbol}&requestType=1&startTime={start}&endTime={end}&timeframe=day"
)

BARS_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


class Budget:
    """요청 예산. 다 쓰면 남은 일은 다음 실행이 이어받는다."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True

    @property
    def left(self) -> int:
        return max(0, self.limit - self.used)


def fetch(url: str, budget: Budget, referer: str | None = None, tries: int = 3) -> bytes | None:
    """한 번 받는다. 예산이 없으면 None.

    네이버는 간헐적으로 끊는다. 실패를 전체 실행의 실패로 치지 않는다 —
    한 종목의 일봉이 빠지면 그 종목만 판정에서 빠지면 될 일이다.
    """
    if not budget.take():
        return None
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ko,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == tries - 1:
                return None
            time.sleep(0.5 * (attempt + 1) + random.random() * 0.3)
    return None


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_state(state: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def write_json_gz(path: pathlib.Path, payload) -> None:
    """gzip 은 mtime=0 으로 고정한다. 내용이 같으면 바이트도 같아야 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"), mtime=0) as fh:
        fh.write(body)


def read_json_gz(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stale(state: dict, key: str, max_age_days: int) -> bool:
    stamp = state.get(key)
    if not stamp:
        return True
    try:
        then = datetime.strptime(stamp, "%Y-%m-%d").date()
    except ValueError:
        return True
    return (date.today() - then).days >= max_age_days


def collect_groups(budget: Budget, workers: int) -> list[dict] | None:
    """테마 265 + 업종 79 의 목록과 구성종목."""
    groups: list[dict] = []

    seen_no = set()
    for page in range(1, THEME_PAGES + 1):
        raw = fetch(URL_THEME_LIST.format(page=page), budget)
        if raw is None:
            break
        rows = api.parse_theme_list(raw)
        fresh = [r for r in rows if r["no"] not in seen_no]
        if not fresh:
            # 마지막 페이지를 넘기면 네이버는 같은 페이지를 다시 준다.
            break
        seen_no.update(r["no"] for r in fresh)
        groups.extend(fresh)

    raw = fetch(URL_UPJONG_LIST, budget)
    if raw is not None:
        groups.extend(api.parse_upjong_list(raw))

    if not groups:
        return None

    def detail(group: dict) -> None:
        raw = fetch(URL_GROUP_DETAIL.format(type=group["type"], no=group["no"]), budget)
        group["members"] = api.parse_group_detail(raw) if raw else []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(detail, groups))

    kept = [g for g in groups if g.get("members")]
    print(f"  그룹 {len(kept)}개 (구성종목 {sum(len(g['members']) for g in kept)}건)")

    # 예산이 중간에 떨어지면 절반만 받고도 '오늘 갱신함' 으로 찍혀 다음 실행이
    # 건너뛴다. 그러면 반쪽짜리 그룹 표로 일주일을 간다. 온전히 못 받았으면
    # 실패로 돌려 지난 파일을 그대로 쓰게 한다.
    if len(kept) < len(groups) * 0.9 or len(kept) < 300:
        print(
            f"  그룹을 온전히 못 받았습니다 ({len(kept)}/{len(groups)}). 저장하지 않습니다.",
            file=sys.stderr,
        )
        return None
    return kept


def collect_market_map(budget: Budget, workers: int) -> dict[str, str] | None:
    """종목코드 -> 'KOSPI' | 'KOSDAQ'.

    그룹 상세도 ETF 구성종목도 시장을 안 알려준다. 벤치마크를 고르려면 필요하다.
    """
    result: dict[str, str] = {}
    lock = threading.Lock()

    def one(job: tuple[int, int]) -> None:
        sosok, page = job
        raw = fetch(URL_MARKET_SUM.format(sosok=sosok, page=page), budget)
        if raw is None:
            return
        label = "KOSPI" if sosok == 0 else "KOSDAQ"
        codes = api.parse_market_codes(raw)
        with lock:
            for code in codes:
                result.setdefault(code, label)

    jobs = [(s, p) for s, pages in MARKET_PAGES.items() for p in range(1, pages + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))

    # 페이지 대부분을 못 받았으면 시장 구분이 뻥 뚫린 채로 굳는다.
    if len(result) < 2000:
        print(f"  시장 구분이 {len(result)}종목뿐입니다. 저장하지 않습니다.", file=sys.stderr)
        return None
    kospi = sum(1 for v in result.values() if v == "KOSPI")
    print(f"  시장 구분 {len(result)}종목 (코스피 {kospi} · 코스닥 {len(result) - kospi})")
    return result


def collect_etfs(budget: Budget) -> list[dict] | None:
    raw = fetch(URL_ETF_LIST, budget)
    if raw is None:
        return None
    etfs = api.parse_etf_list(raw)
    print(f"  ETF {len(etfs)}개")
    return etfs


def collect_holdings(etfs: list[dict], budget: Budget, workers: int) -> dict[str, list[dict]]:
    """ETF 구성종목. 종목명만 오므로 나중에 이름으로 테마와 맞댄다."""
    result: dict[str, list[dict]] = {}
    lock = threading.Lock()

    def one(etf: dict) -> None:
        raw = fetch(
            URL_ETF_HOLDINGS.format(code=etf["code"]),
            budget,
            referer="https://finance.naver.com/",
        )
        if raw is None:
            return
        rows = api.parse_etf_holdings(raw)
        if rows:
            with lock:
                result[etf["code"]] = rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, etfs))

    print(f"  ETF 구성종목 {len(result)}/{len(etfs)}개 ETF")
    return result


def load_bars() -> dict[str, dict[str, list]]:
    """{symbol: {date: [o,h,l,c,v]}}"""
    bars: dict[str, dict[str, list]] = {}
    if not BARS_FILE.exists():
        return bars
    try:
        with gzip.open(BARS_FILE, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                bars.setdefault(row["symbol"], {})[row["date"]] = [
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["volume"]),
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
            o, h, low, c, v = bars[symbol][day]
            writer.writerow([symbol, day, o, h, low, c, v])
    body = buffer.getvalue().encode("utf-8")
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(BARS_FILE, "wb"), mtime=0) as fh:
        fh.write(body)


def collect_bars(
    symbols: list[str], bars: dict, budget: Budget, workers: int, span: int
) -> int:
    """일봉을 받아 기존 캐시에 덮어쓴다.

    덮어쓰기(update)라는 점이 중요하다. 수정주가로 과거 종가가 바뀌면 받은
    구간은 새 값으로 갈린다. 그래서 30일에 한 번 전체를 다시 받는다.
    """
    start = (date.today() - timedelta(days=span)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    lock = threading.Lock()
    ok = 0

    def one(symbol: str) -> None:
        nonlocal ok
        raw = fetch(URL_SISE.format(symbol=symbol, start=start, end=end), budget)
        if raw is None:
            return
        rows = api.parse_sise_json(raw)
        if not rows:
            return
        with lock:
            slot = bars.setdefault(symbol, {})
            for row in rows:
                slot[row["date"]] = [
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                ]
            ok += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, symbols))
    return ok


def trim_bars(bars: dict[str, dict[str, list]], long_symbols: set[str]) -> None:
    """보관 기간을 넘긴 과거를 버린다.

    ETF·지수는 백테스트가 쓰므로 880 거래일, 나머지 종목은 130 거래일이다.
    안 자르면 캐시가 끝없이 자란다.
    """
    for symbol, series in bars.items():
        limit = KEEP_HISTORY_DAYS if symbol in long_symbols else KEEP_TRADING_DAYS
        if len(series) <= limit:
            continue
        keep = sorted(series)[-limit:]
        bars[symbol] = {d: series[d] for d in keep}


def main() -> int:
    parser = argparse.ArgumentParser(description="네이버 테마·업종·ETF·일봉 수집")
    parser.add_argument("--max-calls", type=int, default=20000, help="이번 실행 요청 예산")
    parser.add_argument("--workers", type=int, default=8, help="동시 요청 수")
    parser.add_argument("--full-bars", action="store_true", help="일봉 전체 재수집")
    parser.add_argument("--refresh-groups", action="store_true", help="그룹·구성종목 강제 갱신")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    state = load_state()
    budget = Budget(args.max_calls)
    failed = False

    groups = read_json_gz(GROUPS_FILE)
    if args.refresh_groups or groups is None or stale(state, "groups_date", GROUPS_MAX_AGE_DAYS):
        print("테마·업종 목록과 구성종목을 받습니다.")
        fresh = collect_groups(budget, args.workers)
        if fresh:
            groups = fresh
            write_json_gz(GROUPS_FILE, groups)
            state["groups_date"] = date.today().isoformat()
        else:
            failed = True
            print("  그룹 수집 실패 — 지난 파일을 그대로 씁니다.", file=sys.stderr)
    if not groups:
        print("그룹 데이터가 아예 없어 더 진행할 수 없습니다.", file=sys.stderr)
        return 1

    market = read_json_gz(MARKET_FILE)
    if args.refresh_groups or market is None or stale(state, "market_date", GROUPS_MAX_AGE_DAYS):
        print("시장 구분표를 받습니다.")
        fresh_market = collect_market_map(budget, args.workers)
        if fresh_market:
            market = fresh_market
            write_json_gz(MARKET_FILE, market)
            state["market_date"] = date.today().isoformat()
        else:
            failed = True
            print("  시장 구분 수집 실패 — 지난 파일을 그대로 씁니다.", file=sys.stderr)

    print("ETF 목록을 받습니다.")
    etfs = collect_etfs(budget)
    if etfs:
        write_json_gz(ETFS_FILE, etfs)
        state["etfs_date"] = date.today().isoformat()
    else:
        etfs = read_json_gz(ETFS_FILE) or []
        failed = True
        print("  ETF 목록 수집 실패 — 지난 파일을 그대로 씁니다.", file=sys.stderr)

    holdings = read_json_gz(HOLDINGS_FILE)
    if etfs and (holdings is None or stale(state, "holdings_date", HOLDINGS_MAX_AGE_DAYS)):
        print("ETF 구성종목을 받습니다.")
        fresh_holdings = collect_holdings(etfs, budget, args.workers)
        # 절반도 못 받았으면 지난 파일이 낫다.
        if len(fresh_holdings) >= len(etfs) * 0.5:
            holdings = fresh_holdings
            write_json_gz(HOLDINGS_FILE, holdings)
            state["holdings_date"] = date.today().isoformat()
        else:
            failed = True
            print("  ETF 구성종목이 절반도 안 왔습니다 — 지난 파일을 그대로 씁니다.", file=sys.stderr)

    # ETF·지수는 백테스트가 쓰므로 3.5년, 나머지 종목은 20일 지표에 필요한 만큼만.
    long_symbols = set(INDEX_SYMBOLS) | {e["code"] for e in etfs}
    short_symbols: list[str] = []
    seen = set(long_symbols)
    for group in groups:
        for member in group.get("members", []):
            if member["code"] not in seen:
                seen.add(member["code"])
                short_symbols.append(member["code"])
    total = len(long_symbols) + len(short_symbols)

    full = args.full_bars or not BARS_FILE.exists() or stale(
        state, "full_bars_date", FULL_BARS_MAX_AGE_DAYS
    )
    bars = load_bars()
    print(
        f"일봉을 받습니다 — ETF·지수 {len(long_symbols)} + 종목 {len(short_symbols)}, "
        f"{'전체 재수집' if full else '증분 ' + str(INCREMENTAL_DAYS) + '일'}, 예산 {budget.left}"
    )
    ok = collect_bars(
        sorted(long_symbols), bars, budget, args.workers,
        FULL_HISTORY_DAYS if full else INCREMENTAL_DAYS,
    )
    print(f"  ETF·지수 {ok}/{len(long_symbols)}심볼 ({'3.5년' if full else '증분'})")
    ok += collect_bars(
        short_symbols, bars, budget, args.workers,
        FULL_BACKFILL_DAYS if full else INCREMENTAL_DAYS,
    )
    print(f"  합계 {ok}/{total}심볼")

    if ok < total * 0.8:
        failed = True
        print("  일봉이 80%도 안 왔습니다 — 예산이나 네이버 응답을 확인하세요.", file=sys.stderr)

    # 장중에 받으면 마지막 행의 '종가' 는 확정 종가가 아니라 그 순간 현재가다.
    # 크론은 18:30 이라 걸릴 일이 없지만 손으로 돌리면 조용히 섞인다. 상태에
    # 남겨 두면 집계가 화면에 '장중 시세' 라고 알릴 수 있다.
    intraday = api.is_intraday(datetime.now())
    if intraday:
        print(
            "  ※ 장 마감(15:30) 전이라 마지막 일봉이 확정 종가가 아닌 장중 시세입니다.",
            file=sys.stderr,
        )
    state["bars_intraday"] = intraday

    trim_bars(bars, long_symbols)
    save_bars(bars)
    if full and ok >= total * 0.8:
        state["full_bars_date"] = date.today().isoformat()
    state["bars_date"] = date.today().isoformat()
    state["last_run"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    print(f"요청 {budget.used}건 사용, 캐시 {len(bars)}심볼")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
