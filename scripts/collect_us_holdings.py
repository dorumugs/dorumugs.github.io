#!/usr/bin/env python3
"""미국 ETF 구성종목을 받아 data/stocks/us_holdings.json.gz 에 쌓는다.

파싱은 us_holdings_api.py 가 한다. 여기는 I/O·예산·재개만 다룬다.

192개 중 발행사를 확인한 건 162개다(iShares 41·Direxion 37·SPDR 22·ProShares 19·
Vanguard 12·Global X 9·Invesco 8·ARK 6·실물 신탁 4·First Trust 3·KraneShares 1).
나머지 30개는 ISSUER_BY_TICKER 에 없어 애초에 요청을 안 보낸다 — "못 받음" 이
아니라 "발행사를 모름" 이라 따로 센다.

발행사마다 받는 방식이 다르다.

  대부분        티커별 URL 하나(holdings_url)
  ProShares     전 종목이 파일 하나 — 하루 1요청이면 19개가 다 온다
  Global X·KraneShares  파일명에 날짜가 박혀 최근 영업일부터 거슬러 올라간다
  Invesco       TLS 지문으로 urllib 을 막아 브라우저로만 받는다(browser_fetch.py).
                --no-browser 로 끌 수 있고, 실패해도 나머지 수집은 그대로 간다
  실물 신탁     받을 게 없다 — 요청 없이 "구성종목 없음" 을 기록한다

예의상 워커를 적게 쓴다(기본 3) — 발행사 서버는 우리 트래픽을 기대하지 않는다.

재개: 캐시에 있는 종목은 fetchedDate 가 오늘이면 다시 안 받는다. 예산이 부족해
이번 실행에서 못 받은 종목은 캐시에 있던 값을 그대로 들고 다음 실행으로 넘어간다.

사용법

    python3 scripts/collect_us_holdings.py
    python3 scripts/collect_us_holdings.py --max-calls 80 --workers 3
    python3 scripts/collect_us_holdings.py --force   # 캐시 무시하고 전부 다시 받기
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import us_holdings_api as h  # noqa: E402
from build_us_etf import leverage_of  # noqa: E402
from collect_stocks import Budget, fetch, read_json_gz, write_json_gz  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"
UNIVERSE_FILE = DATA / "us_universe.json.gz"
HOLDINGS_FILE = DATA / "us_holdings.json.gz"

PARSERS = {
    h.ISSUER_DIREXION: h.parse_direxion,
    h.ISSUER_SPDR: h.parse_spdr,
    h.ISSUER_ARK: h.parse_ark,
    h.ISSUER_ISHARES: h.parse_ishares,
    h.ISSUER_VANGUARD: h.parse_vanguard,
    h.ISSUER_FIRSTTRUST: h.parse_firsttrust,
}

# 파일명에 날짜가 박혀 최근 영업일부터 거슬러 올라가며 찾아야 하는 발행사.
DATED_URLS = {
    h.ISSUER_GLOBALX: (h.globalx_urls, h.parse_globalx),
    h.ISSUER_KRANESHARES: (h.kraneshares_urls, h.parse_kraneshares),
}

# Invesco 워밍업 페이지. 이 출처의 세션을 얻어야 dng-api 가 응답한다.
INVESCO_WARMUP = "https://www.invesco.com/us/en/home.html"


def _fresh(entry: dict | None) -> bool:
    """오늘 이미 받았으면 다시 안 받는다."""
    if not entry:
        return False
    return entry.get("fetchedDate") == date.today().isoformat()


def _finish(parsed: dict, ticker: str, name: str, issuer: str) -> dict | None:
    """파싱 결과를 검사하고 발행사·수집일을 얹는다. 버릴 것만 None."""
    # 스왑·기타도 "받은 데이터" 다 — 인버스·채권 레버리지 상품(TZA·TMF 등)은
    # 개별 종목·현금이 거의 없고 스왑뿐이라, rows·cash 만 보면 아무것도 못 받은
    # 걸로 착각해 통째로 버리게 된다.
    if not parsed["rows"] and not parsed["cash"] and not parsed.get("swap") and not parsed.get("other"):
        return None
    lev = leverage_of(name)
    swap = parsed.get("swap", 0.0)
    other = parsed.get("other", 0.0)
    # 뱅가드는 첫 500행만 받으므로(VT 는 10,032종목) 비중 합이 100 에 한참
    # 못 미치는 게 정상이다 — 검사에서 뺀다. 대신 totalCount 로 진짜 종목 수를
    # 남겨 화면에 "상위 25/10032" 로 적는다.
    if issuer != h.ISSUER_VANGUARD and not h.total_weight_ok(
        parsed["rows"], parsed["cash"], swap, other, leverage=lev
    ):
        total = h.total_weight(parsed["rows"], parsed["cash"], swap, other)
        print(f"    {ticker}: 비중 합 {total:.1f}% — 80~130%×{max(1, abs(lev)):.0f} 범위 밖", file=sys.stderr)
    parsed["issuer"] = issuer
    parsed["fetchedDate"] = date.today().isoformat()
    return parsed


def collect_dated(ticker: str, name: str, issuer: str, budget: Budget) -> dict | None:
    """파일명에 날짜가 박힌 발행사. 최근 날짜부터 200 이 나올 때까지 내려간다."""
    make_urls, parser = DATED_URLS[issuer]
    for url in make_urls(ticker, date.today()):
        raw = fetch(url, budget, max_bytes=h.MAX_RESPONSE_BYTES)
        if raw is None:
            continue
        try:
            parsed = parser(raw)
        except Exception:  # noqa: BLE001 — 발행사 파일 포맷이 어떻게 깨질지 모른다
            continue
        if parsed["rows"]:
            return _finish(parsed, ticker, name, issuer)
    return None


def collect_invesco(names: dict[str, str]) -> dict[str, dict]:
    """Invesco 는 TLS 지문으로 막혀 브라우저로만 받는다 — browser_fetch.py 참고.

    예산(Budget)에는 안 넣는다. 그 예산은 urllib 요청 수를 재는 것이고 이건
    브라우저 세션 하나이기 때문이다. 실패하면 빈 딕셔너리라 나머지 수집은
    그대로 굴러간다.
    """
    from browser_fetch import fetch_json_via_browser  # noqa: PLC0415 — 여기서만 쓴다

    urls = {t: h.invesco_url(t) for t in h.INVESCO_ID}
    urls = {t: u for t, u in urls.items() if u}
    bodies = fetch_json_via_browser(urls, INVESCO_WARMUP)
    out: dict[str, dict] = {}
    for ticker, raw in bodies.items():
        try:
            parsed = h.parse_invesco(raw)
        except Exception:  # noqa: BLE001
            continue
        done = _finish(parsed, ticker, names.get(ticker, ticker), h.ISSUER_INVESCO)
        if done:
            out[ticker] = done
    return out


def physical_entry(ticker: str) -> dict:
    """실물 신탁(GLD·SLV 등). 받으려다 실패한 게 아니라 애초에 구성종목이 없다.

    화면이 "받지 못했습니다" 라고 하면 거짓말이 된다 — 없는 걸 없다고 적어야
    한다. rows 는 비우되 asset 에 무엇을 담는지 남겨 화면이 그대로 쓴다.
    """
    return {
        "asOf": "", "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "",
        "other": 0.0, "noTicker": False,
        "physical": h.PHYSICAL_TRUSTS[ticker],
        "issuer": h.ISSUER_PHYSICAL,
        "fetchedDate": date.today().isoformat(),
    }


def collect_proshares(budget: Budget, universe_names: dict[str, str]) -> dict[str, dict]:
    """ProShares 전 종목을 요청 한 번으로. 실패하면 빈 딕셔너리."""
    raw = fetch(URL := h.URL_PROSHARES_ALL, budget, max_bytes=h.MAX_RESPONSE_BYTES)
    if raw is None:
        print(f"    ProShares 일별보유 파일을 못 받았습니다: {URL}", file=sys.stderr)
        return {}
    try:
        by_ticker = h.parse_proshares_all(raw)
    except Exception:  # noqa: BLE001
        print("    ProShares 일별보유 파일 파싱 실패", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    for ticker in h.PROSHARES_TICKERS:
        parsed = by_ticker.get(ticker)
        if not parsed:
            continue
        done = _finish(parsed, ticker, universe_names.get(ticker, ticker), h.ISSUER_PROSHARES)
        if done:
            out[ticker] = done
    return out


def collect_one(ticker: str, name: str, issuer: str, budget: Budget) -> dict | None:
    """구성종목 하나. 실패해도 예외를 밖으로 던지지 않는다 — 실행 전체를 죽이면 안 된다."""
    if issuer in DATED_URLS:
        return collect_dated(ticker, name, issuer, budget)
    if issuer == h.ISSUER_PHYSICAL:
        return physical_entry(ticker)  # 받을 게 없다 — 요청도 안 한다
    url = h.holdings_url(ticker)
    if url is None:
        return None
    raw = fetch(url, budget, max_bytes=h.MAX_RESPONSE_BYTES)
    if raw is None:
        return None
    parser = PARSERS[issuer]
    try:
        if issuer == h.ISSUER_VANGUARD:
            # 채권 원장의 ticker 는 그 채권을 발행한 회사의 '주식' 티커라 그대로
            # 쓰면 안 된다 — parse_vanguard 의 bond 인자 설명 참고.
            parsed = parser(raw, bond=h.VANGUARD_PATH.get(ticker) == "bond")
        else:
            parsed = parser(raw)
    except Exception:  # noqa: BLE001 — 발행사 파일 포맷이 어떻게 깨질지 모른다
        return None
    return _finish(parsed, ticker, name, issuer)


def main() -> int:
    parser = argparse.ArgumentParser(description="미국 ETF 구성종목 수집")
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 전부 다시 받기")
    parser.add_argument("--no-browser", action="store_true",
                        help="브라우저가 필요한 발행사(Invesco) 를 건너뛴다")
    args = parser.parse_args()

    universe = read_json_gz(UNIVERSE_FILE)
    if not universe:
        print(f"{UNIVERSE_FILE} 가 없습니다. collect_us_etf.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1

    cache: dict[str, dict] = read_json_gz(HOLDINGS_FILE) or {}
    budget = Budget(args.max_calls)

    names = {row["ticker"]: row["name"] for row in universe}

    no_issuer = 0
    todo: list[tuple[str, str, str]] = []  # (ticker, name, issuer)
    proshares_stale = False
    invesco_stale = False
    for row in universe:
        ticker = row["ticker"]
        issuer = h.ISSUER_BY_TICKER.get(ticker)
        if issuer is None:
            no_issuer += 1
            continue
        if not args.force and _fresh(cache.get(ticker)):
            continue
        if issuer == h.ISSUER_PROSHARES:
            # 티커별로 안 받는다 — 하나라도 낡았으면 파일 하나를 받아 19개를 다 채운다.
            proshares_stale = True
            continue
        if issuer == h.ISSUER_INVESCO:
            # 브라우저 세션 하나로 8개를 다 받는다.
            invesco_stale = True
            continue
        todo.append((ticker, row["name"], issuer))

    lock = threading.Lock()
    fetched = 0
    failed: list[str] = []
    results: dict[str, dict] = {}

    def one(item: tuple[str, str, str]) -> None:
        nonlocal fetched
        ticker, name, issuer = item
        result = collect_one(ticker, name, issuer, budget)
        with lock:
            if result is None:
                failed.append(ticker)
            else:
                results[ticker] = result
                fetched += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))

    if proshares_stale:
        bulk = collect_proshares(budget, names)
        results.update(bulk)
        fetched += len(bulk)
        failed.extend(t for t in h.PROSHARES_TICKERS if t not in bulk)
        print(f"  ProShares 일별보유 파일 1건으로 {len(bulk)}개 채움")

    if invesco_stale and not args.no_browser:
        inv = collect_invesco(names)
        results.update(inv)
        fetched += len(inv)
        failed.extend(t for t in h.INVESCO_ID if t not in inv)
        print(f"  Invesco 브라우저 세션 1건으로 {len(inv)}개 채움")

    cache.update(results)
    # 이번에 실패했고 캐시에도 없던 티커는 실패로 남긴다(다음 실행이 재시도).
    still_missing = [t for t in failed if t not in cache]

    known = len(universe) - no_issuer
    have_holdings = sum(1 for row in universe if row["ticker"] in cache)
    print(f"미국 ETF {len(universe)}개 · 발행사 확인 {known}개(발행사 모름 {no_issuer}개)")
    print(f"  이번 실행 수집 {fetched}개 · 실패 {len(failed)}개(캐시로 보충 {len(failed) - len(still_missing)}개)")
    print(f"  구성종목 보유 {have_holdings}/{len(universe)}개 · 요청 {budget.used}건 사용")
    if still_missing:
        print(f"  캐시도 없이 완전히 빈 티커 {len(still_missing)}개: {' '.join(sorted(still_missing)[:20])}", file=sys.stderr)

    write_json_gz(HOLDINGS_FILE, {t: cache[t] for t in sorted(cache)})

    # 절반도 못 채웠으면 신호를 준다 — 발행사 화면이 통째로 바뀐 것일 수 있다.
    return 1 if known and have_holdings < known * 0.5 else 0


if __name__ == "__main__":
    sys.exit(main())
