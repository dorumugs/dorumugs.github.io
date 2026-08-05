#!/usr/bin/env python3
"""서울 연립·다세대 실거래가를 월별로 누적 수집한다.

재개발 정비구역 안은 아파트가 아니라 다세대·연립이다. 이 데이터가 있어야
재개발 사업장의 인가 전후 가격 변화를 볼 수 있다 (아파트만으로는 재건축
374곳만 보인다).

서울만 받는다 — 정비사업 진행단계 데이터가 서울시 정보몽땅에만 있어
경기 다세대를 받아도 붙일 사건이 없다.

출력: data/villa_trades/YYYY/YYYY-MM.csv.gz
상태: data/state/villa_state.json

일일 한도는 서비스마다 따로 잡힌다. 아파트 실거래(RTMSDataSvcAptTrade)를
그날 다 써도 이 서비스(RTMSDataSvcRHTrade)는 별도 예산이 남아 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_trades  # noqa: E402  인증키 로딩·월 목록을 한 곳에서만 정의한다
import villa_rtms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "villa_trades"
STATE_FILE = ROOT / "data" / "state" / "villa_state.json"

API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade"
PAGE_SIZE = 1000
SEOUL = "11"


class LimitReached(Exception):
    """일일 호출 한도에 도달했거나 이번 실행 예산을 다 썼다."""


class Budget:
    """남은 호출 수. 실제로 쓴 횟수를 따로 센다.

    total-left 로 사용량을 내면 drain() 이 잔여를 0 으로 만드는 순간 사용량이
    예산 전체로 부풀어 보인다. 실제로 810콜을 쓰고 8,500콜로 보고되는 일이
    있었다 — 한도 진단을 통째로 틀리게 만든다.
    """

    def __init__(self, total: int) -> None:
        self._left = total
        self._used = 0
        self._lock = threading.Lock()
        self.total = total

    def take(self) -> None:
        with self._lock:
            if self._left <= 0:
                raise LimitReached("이번 실행 호출 예산 소진")
            self._left -= 1
            self._used += 1

    def drain(self) -> None:
        with self._lock:
            self._left = 0

    @property
    def left(self) -> int:
        with self._lock:
            return self._left

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


def fetch_page(key: str, sgg: str, ym: str, page: int, retries: int = 5) -> str:
    """한 페이지를 받는다.

    429 를 곧바로 일일 한도로 보지 않는다. 이 서비스는 동시 요청이 몰리면
    잠깐 429 를 내고 곧 회복한다 — 실제로 810콜에서 429 를 맞고 멈췄는데
    1분 뒤 같은 키로 정상 응답이 왔다. 물러섰다 다시 걸어 보고, 그래도
    계속 429 면 그때 한도로 판단한다.
    """
    url = (
        f"{API_URL}?serviceKey={urllib.parse.quote(key, safe='')}"
        f"&LAWD_CD={sgg}&DEAL_YMD={ym}&numOfRows={PAGE_SIZE}&pageNo={page}"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                if attempt == retries - 1:
                    raise LimitReached("HTTP 429 가 계속됩니다 (일일 한도로 판단)") from exc
                time.sleep(3 * (attempt + 1))
                last = exc
                continue
            if 400 <= exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def fetch_cell(key: str, sgg: str, ym: str, budget: Budget, sleep: float) -> list[dict]:
    rows: list[dict] = []
    page = 1
    while True:
        budget.take()
        try:
            text = fetch_page(key, sgg, ym, page)
        except LimitReached:
            budget.drain()
            raise
        try:
            batch, total = villa_rtms.parse_response(text)
        except villa_rtms.ApiError as exc:
            if exc.is_limit:
                budget.drain()
                raise LimitReached(str(exc)) from exc
            raise
        rows.extend(batch)
        if page * PAGE_SIZE >= total or not batch:
            return rows
        page += 1
        if sleep:
            time.sleep(sleep)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "done": {}, "failed": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def month_path(ym: str) -> Path:
    return OUT_DIR / ym[:4] / f"{ym[:4]}-{ym[4:]}.csv.gz"


def read_month(ym: str) -> list[dict]:
    path = month_path(ym)
    if not path.exists():
        return []
    return villa_rtms.csv_to_rows(villa_rtms.gunzip_text(path.read_bytes()))


def write_month(ym: str, rows: list[dict]) -> bool:
    path = month_path(ym)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = villa_rtms.gzip_bytes(villa_rtms.rows_to_csv(rows))
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=8000, help="이번 실행 최대 호출 수")
    parser.add_argument("--sleep", type=float, default=0.05, help="페이지 간 대기 초")
    parser.add_argument("--workers", type=int, default=6, help="동시 요청 수")
    args = parser.parse_args()

    key = collect_trades.load_api_key()
    state = load_state()
    today = date.today()

    sggs = [c for c, _ in __import__("regions").sgg_codes() if c.startswith(SEOUL)]
    months = collect_trades.all_months(today)
    budget = Budget(args.max_calls)
    limited = False
    changed_files = 0

    def cell(s: str, m: str) -> str:
        return f"{s}|{m}"

    total_cells = len(sggs) * len(months)
    done_cells = sum(1 for s in sggs for m in months if cell(s, m) in state["done"])
    print(f"서울 {len(sggs)}개 구 × {len(months)}개월 = {total_cells}칸 · 완료 {done_cells} · 예산 {args.max_calls}콜")

    for ym in months:
        if budget.left <= 0 or limited:
            break
        pending = [s for s in sggs if cell(s, ym) not in state["done"]]
        if not pending:
            continue

        collected: list[list[dict]] = []
        completed: list[tuple[str, int]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_cell, key, s, ym, budget, args.sleep): s for s in pending}
            for future in as_completed(futures):
                sgg = futures[future]
                try:
                    rows = future.result()
                except LimitReached as exc:
                    limited = True
                    print(f"  중단: {exc}", file=sys.stderr)
                    continue
                except villa_rtms.ApiError as exc:
                    state["failed"][cell(sgg, ym)] = f"{exc.code}: {exc.message}"
                    continue
                except Exception as exc:
                    state["failed"][cell(sgg, ym)] = f"error: {exc}"
                    continue
                collected.append(rows)
                completed.append((sgg, len(rows)))

        if collected:
            merged = villa_rtms.merge_rows(read_month(ym), *collected)
            if write_month(ym, merged):
                changed_files += 1
            for sgg, count in completed:
                state["done"][cell(sgg, ym)] = count
                state["failed"].pop(cell(sgg, ym), None)
            print(f"  {ym[:4]}-{ym[4:]}  구 {len(completed)}/{len(pending)}  누적 {len(merged):>5}건  잔여 {budget.left}")
        save_state(state)

    save_state(state)
    done_cells = sum(1 for s in sggs for m in months if cell(s, m) in state["done"])
    print(
        f"\n호출 {budget.used}콜 · 파일 {changed_files}개 갱신 · "
        f"진행률 {done_cells}/{total_cells} ({done_cells / total_cells:.1%})"
    )
    if limited:
        print("한도에 걸려 중단했습니다. 다음 실행에서 이어받습니다.")
    if state["failed"]:
        print(f"실패로 남은 칸 {len(state['failed'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
