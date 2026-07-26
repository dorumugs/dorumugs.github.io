#!/usr/bin/env python3
"""서울·경기 아파트 실거래가를 월별로 누적 수집한다.

매일 실행하면
  1. 최근 몇 개월(기본 3)을 다시 받아 신고 지연·계약해제분을 갱신하고
  2. 아직 못 채운 과거 (시군구 × 월) 칸을 오래된 것부터 채운다.
일일 호출 한도에 걸리면 그 지점을 저장하고 정상 종료한다. 다음 날 이어서 진행한다.

출력: data/trades/YYYY/YYYY-MM.csv.gz   (한 파일 = 한 달치 서울+경기 전량)
상태: data/state/collect_state.json

인증키는 저장소에 두지 않는다. 환경변수 DATA_GO_KR_API_KEY 를 우선 쓰고,
없으면 ~/.claude.json 의 real-estate MCP 서버 env 에서 읽는다.
"""

from __future__ import annotations

import argparse
import json
import os
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

import regions  # noqa: E402
import rtms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRADES_DIR = ROOT / "data" / "trades"
STATE_FILE = ROOT / "data" / "state" / "collect_state.json"

API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
FIRST_MONTH = "200601"  # 국토부 실거래가 공개 시작
PAGE_SIZE = 1000


# --------------------------------------------------------------------------
# 인증키
# --------------------------------------------------------------------------


def load_api_key() -> str:
    key = os.environ.get("DATA_GO_KR_API_KEY")
    if key:
        return key

    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        found = _find_mcp_key(json.loads(claude_json.read_text(encoding="utf-8")))
        if found:
            return found

    raise SystemExit(
        "DATA_GO_KR_API_KEY 를 찾지 못했습니다. 환경변수로 지정하거나 "
        "~/.claude.json 의 real-estate MCP 설정을 확인하세요."
    )


def _find_mcp_key(node: object) -> str | None:
    if not isinstance(node, dict):
        return None
    servers = node.get("mcpServers")
    if isinstance(servers, dict) and "real-estate" in servers:
        env = servers["real-estate"].get("env") or {}
        if env.get("DATA_GO_KR_API_KEY"):
            return env["DATA_GO_KR_API_KEY"]
    for value in node.values():
        found = _find_mcp_key(value)
        if found:
            return found
    return None


# --------------------------------------------------------------------------
# 상태
# --------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "done": {}, "failed": {}, "daily": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cell(sgg: str, ym: str) -> str:
    return f"{sgg}|{ym}"


# --------------------------------------------------------------------------
# 월 목록
# --------------------------------------------------------------------------


def all_months(today: date) -> list[str]:
    months: list[str] = []
    year, month = int(FIRST_MONTH[:4]), int(FIRST_MONTH[4:])
    while (year, month) <= (today.year, today.month):
        months.append(f"{year}{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def fetch_page(key: str, sgg: str, ym: str, page: int, retries: int = 3) -> str:
    url = (
        f"{API_URL}?serviceKey={urllib.parse.quote(key, safe='')}"
        f"&LAWD_CD={sgg}&DEAL_YMD={ym}&numOfRows={PAGE_SIZE}&pageNo={page}"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            # HTTP 4xx 는 재시도해도 같은 결과다. 5xx 와 네트워크 오류만 물러섰다 재시도한다.
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


class LimitReached(Exception):
    """일일 호출 한도에 도달했거나 이번 실행 예산을 다 썼다."""


class Budget:
    """남은 호출 수. 워커 여러 개가 동시에 깎으므로 잠금이 필요하다."""

    def __init__(self, total: int) -> None:
        self._left = total
        self._lock = threading.Lock()
        self.total = total

    def take(self) -> None:
        with self._lock:
            if self._left <= 0:
                raise LimitReached("이번 실행 호출 예산 소진")
            self._left -= 1

    def drain(self) -> None:
        """한도 초과를 만났을 때 남은 예산을 즉시 0 으로 만들어 다른 워커도 멈춘다."""
        with self._lock:
            self._left = 0

    @property
    def left(self) -> int:
        with self._lock:
            return self._left

    @property
    def used(self) -> int:
        return self.total - self.left


def fetch_cell(key: str, sgg: str, ym: str, budget: Budget, sleep: float) -> list[dict]:
    """한 (시군구 × 월) 칸을 페이지 끝까지 받아온다."""
    rows: list[dict] = []
    page = 1
    while True:
        budget.take()
        try:
            text = fetch_page(key, sgg, ym, page)
        except urllib.error.HTTPError as exc:
            # 공공데이터포털은 일일 한도 초과를 resultCode 가 아니라 HTTP 429 로 알린다.
            # 이걸 일반 오류로 처리하면 한도에 닿고도 멈추지 않고 남은 예산을 전부 태운다.
            if exc.code == 429:
                budget.drain()
                raise LimitReached("HTTP 429 일일 호출 한도 초과") from exc
            raise
        try:
            batch, total = rtms.parse_response(text)
        except rtms.ApiError as exc:
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


# --------------------------------------------------------------------------
# 월 파일 입출력
# --------------------------------------------------------------------------


def month_path(ym: str) -> Path:
    return TRADES_DIR / ym[:4] / f"{ym[:4]}-{ym[4:]}.csv.gz"


def read_month(ym: str) -> list[dict]:
    path = month_path(ym)
    if not path.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(path.read_bytes()))


def write_month(ym: str, rows: list[dict]) -> bool:
    """월 파일을 쓴다. 내용이 이전과 같으면 건드리지 않고 False 를 돌려준다."""
    path = month_path(ym)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rtms.gzip_bytes(rtms.rows_to_csv(rows))
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------


def build_worklist(state: dict, months: list[str], sggs: list[str], refresh: int) -> list[tuple[str, list[str]]]:
    """처리할 (월, 시군구목록) 목록. 갱신 대상 최근 월이 먼저, 그다음 과거 백필.

    백필이 끝나기 전에는 갱신을 돌리지 않는다. 최근 3개월 갱신은 매번 216콜을
    먹는데, 아직 대시보드가 없는 백필 기간에는 그만큼 백필이 늦어질 뿐이다.
    백필이 끝나면 자동으로 갱신 모드로 넘어간다.
    """
    done = state["done"]
    backlog = [
        (ym, [s for s in sggs if cell(s, ym) not in done])
        for ym in months
    ]
    backlog = [(ym, pending) for ym, pending in backlog if pending]

    refresh_months = months[-refresh:] if (refresh and not backlog) else []

    work: list[tuple[str, list[str]]] = []
    for ym in reversed(refresh_months):  # 최신 달부터
        work.append((ym, list(sggs)))
    for ym, pending in backlog:  # 과거부터
        if ym not in refresh_months:
            work.append((ym, pending))
    return work


def _record_calls(state: dict, day: str, calls: int, limited: bool, reason: str) -> None:
    """그날 성공한 호출 수를 누적한다. 일일 한도를 실측하는 근거가 된다."""
    entry = state["daily"].setdefault(day, {"calls": 0, "limited": False})
    entry["calls"] += calls
    entry["limited"] = entry["limited"] or limited
    if limited and reason:
        entry["reason"] = reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=900,
        help="이번 실행에서 허용할 최대 API 호출 수 (기본 900). 일일 한도 실측 전 보수적 기본값.",
    )
    parser.add_argument("--refresh", type=int, default=3, help="매번 다시 받을 최근 개월 수 (기본 3)")
    parser.add_argument("--sleep", type=float, default=0.1, help="페이지 간 대기 초 (기본 0.1)")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="동시 요청 수 (기본 6). 호출당 2초 넘게 걸려 순차 실행이면 백필에 12시간이 든다.",
    )
    parser.add_argument("--only-month", help="특정 월만 처리 (YYYYMM). 디버깅용")
    args = parser.parse_args()

    key = load_api_key()
    state = load_state()
    today = date.today()
    today_str = today.isoformat()

    sggs = [c for c, _ in regions.sgg_codes()]
    months = all_months(today)
    if args.only_month:
        months = [m for m in months if m == args.only_month]
        if not months:
            print(f"대상 월 없음: {args.only_month}", file=sys.stderr)
            return 1

    work = build_worklist(state, months, sggs, 0 if args.only_month else args.refresh)
    budget = Budget(args.max_calls)
    recorded_calls = 0
    limited = False
    limit_reason = ""
    changed_files = 0
    fetched_cells = 0

    total_cells = len(sggs) * len(months)
    done_cells = sum(1 for s in sggs for m in months if cell(s, m) in state["done"])
    print(f"진행률 {done_cells}/{total_cells} 칸 · 예산 {args.max_calls}콜 · 대상 {len(work)}개월")

    for ym, pending in work:
        if budget.left <= 0 or limited:
            break

        collected: list[list[dict]] = []
        completed: list[tuple[str, int]] = []

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_cell, key, sgg, ym, budget, args.sleep): sgg
                for sgg in pending
            }
            for future in as_completed(futures):
                sgg = futures[future]
                try:
                    rows = future.result()
                except LimitReached as exc:
                    limited = True
                    limit_reason = str(exc)
                    continue
                except rtms.ApiError as exc:
                    # 이 칸만 실패로 남기고 다음 실행에서 재시도한다.
                    state["failed"][cell(sgg, ym)] = f"{exc.code}: {exc.message}"
                    continue
                except Exception as exc:  # 네트워크 재시도까지 실패
                    state["failed"][cell(sgg, ym)] = f"error: {exc}"
                    continue
                collected.append(rows)
                completed.append((sgg, len(rows)))
                fetched_cells += 1

        if collected:
            # 기존 파일과 병합해 다시 쓴다. 중간에 끊겨도 다음 실행에서 이어붙는다.
            merged = rtms.merge_rows(read_month(ym), *collected)
            if write_month(ym, merged):
                changed_files += 1
            for sgg, count in completed:
                state["done"][cell(sgg, ym)] = count
                state["failed"].pop(cell(sgg, ym), None)
            print(
                f"  {ym[:4]}-{ym[4:]}  시군구 {len(completed)}/{len(pending)}  "
                f"누적 {len(merged):>6}건  잔여예산 {budget.left}"
            )

        # 월 단위로 상태를 저장한다. 중간에 죽어도 여기까지의 진행은 남는다.
        _record_calls(state, today_str, budget.used - recorded_calls, limited, limit_reason)
        recorded_calls = budget.used
        save_state(state)

    used = budget.used
    # 월 루프 안에서 이미 기록한 몫을 빼고 남은 것만 더한다 (이중 계상 방지).
    _record_calls(state, today_str, used - recorded_calls, limited, limit_reason)
    save_state(state)
    day = state["daily"][today_str]

    done_cells = sum(1 for s in sggs for m in months if cell(s, m) in state["done"])
    print(
        f"\n호출 {used}콜 · 칸 {fetched_cells}개 · 파일 {changed_files}개 갱신 · "
        f"진행률 {done_cells}/{total_cells} ({done_cells / total_cells:.1%})"
    )
    if limited:
        print(f"중단 사유: {limit_reason}")
        print(f"오늘 누적 성공 호출: {day['calls']}콜")
    if state["failed"]:
        print(f"실패로 남은 칸 {len(state['failed'])}개 (다음 실행에서 재시도)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
