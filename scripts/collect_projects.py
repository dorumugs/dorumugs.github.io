#!/usr/bin/env python3
"""서울 정비사업장 목록과 사업장별 추진경과를 누적 수집한다.

매일 실행하면
  1. 사업장 목록(약 1,100건)을 한 번에 다시 받아 진행단계 변화를 반영하고
  2. 추진경과를 아직 못 받았거나 단계가 바뀐 사업장부터 채운다.
이번 실행 예산을 다 쓰면 그 지점을 저장하고 정상 종료한다. 다음 날 이어서 진행한다.

출력: data/projects/projects.csv.gz   사업장 목록
      data/projects/events.csv.gz     사업장별 단계 이벤트 (일자·동의율·고시번호)
상태: data/state/project_state.json

정보몽땅은 공식 OpenAPI 가 아니라 공공 웹사이트다. 인증키가 없는 대신
예의를 지켜야 한다 — 동시 요청을 낮게 잡고 요청 사이에 간격을 둔다.
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

import cleanup_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "data" / "projects"
STATE_FILE = ROOT / "data" / "state" / "project_state.json"

BASE = "https://cleanup.seoul.go.kr"
LIST_URL = f"{BASE}/cleanup/bsnssttus/lsubBsnsSttus.do"
CAFE_URL = f"{BASE}/cafe/mainIndx.do"
PROGRESS_URL = f"{BASE}/cafe/mainIndx/cleanup-prtnelapse/vscr.do"
REFERER = f"{BASE}/cleanup/bsnssttus/lscrMainIndx.do"

# 목록을 한 번에 다 받는다. 실측 1,102건이라 넉넉히 잡아도 응답이 720KB 남짓이다.
LIST_PAGE_SIZE = 3000
SEOUL_ALL = "11000"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _request(url: str, data: bytes | None = None, retries: int = 3) -> str:
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KayserDocs-realestate/1.0)",
            "Referer": REFERER,
        },
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            # 4xx 는 다시 걸어도 같은 결과다. 5xx 와 네트워크 오류만 물러섰다 재시도한다.
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def fetch_project_list() -> list[dict[str, str]]:
    """사업장 목록 전체를 한 번에 받는다."""
    params = [("bsnsSeCodeList", c) for c in cleanup_api.BSNS_SE_CODES]
    params += [("bsnsEfctMthdList", c) for c in cleanup_api.BSNS_EFCT_CODES]
    params += [
        ("sigunguCd", SEOUL_ALL),
        ("legaldongCode", ""),
        ("bsnsProgrsSttusCode", ""),
        ("asscNm", ""),
        ("orderValue", ""),
        ("cpage", "1"),
        ("pageSize", str(LIST_PAGE_SIZE)),
    ]
    query = urllib.parse.urlencode(params)
    # 이 화면은 쿼리스트링과 폼 본문을 둘 다 봐야 필터가 걸린다.
    return cleanup_api.parse_project_list(_request(f"{LIST_URL}?{query}", query.encode("utf-8")))


def fetch_progress(cafe_url: str) -> list[dict[str, str]]:
    """사업장 하나의 추진경과를 받는다. 카페 메인에서 키를 캔 뒤 본 조회 — 2콜."""
    main = _request(f"{CAFE_URL}?{urllib.parse.urlencode({'cafeUrl': cafe_url})}")
    keys = cleanup_api.parse_cafe_keys(main)
    query = urllib.parse.urlencode({"cafeId": keys["cafe_id"], "bsnsPk": keys["bsns_pk"]})
    events = cleanup_api.parse_progress(_request(f"{PROGRESS_URL}?{query}"))
    for event in events:
        event["cafe_url"] = cafe_url
    return events


# --------------------------------------------------------------------------
# 예산
# --------------------------------------------------------------------------


class LimitReached(Exception):
    """이번 실행 호출 예산을 다 썼다."""


class Budget:
    """남은 호출 수. 워커 여러 개가 동시에 깎으므로 잠금이 필요하다."""

    def __init__(self, total: int) -> None:
        self._left = total
        self._lock = threading.Lock()
        self.total = total

    def take(self, n: int = 1) -> None:
        with self._lock:
            if self._left < n:
                raise LimitReached("이번 실행 호출 예산 소진")
            self._left -= n

    @property
    def left(self) -> int:
        with self._lock:
            return self._left

    @property
    def used(self) -> int:
        return self.total - self.left


# --------------------------------------------------------------------------
# 상태·파일
# --------------------------------------------------------------------------


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


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return cleanup_api.csv_to_rows(cleanup_api.gunzip_text(path.read_bytes()))


def _write(path: Path, text: str) -> bool:
    """내용이 이전과 같으면 건드리지 않고 False 를 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cleanup_api.gzip_bytes(text)
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def projects_path() -> Path:
    return PROJECTS_DIR / "projects.csv.gz"


def events_path() -> Path:
    return PROJECTS_DIR / "events.csv.gz"


# --------------------------------------------------------------------------
# 작업 목록
# --------------------------------------------------------------------------


def build_worklist(projects: list[dict[str, str]], state: dict, refresh: int) -> list[str]:
    """추진경과를 받을 사업장(cafe_url) 목록.

    순서는
      1. 한 번도 못 받은 사업장
      2. 목록의 진행단계가 마지막 수집 때와 달라진 사업장 (단계가 넘어갔다)
      3. 가장 오래 안 본 사업장 refresh 개
    이다. 정비사업은 몇 달 단위로 움직여 매일 전량을 다시 받을 이유가 없다.
    """
    done = state["done"]
    fresh: list[str] = []
    changed: list[str] = []
    seen: list[tuple[str, str]] = []

    for project in projects:
        url = project.get("cafe_url", "")
        if not url or project.get("suspended"):
            continue  # 카페가 닫힌 사업장은 받아봐야 404 다
        entry = done.get(url)
        if not entry:
            fresh.append(url)
        elif entry.get("stage") != project.get("stage", ""):
            changed.append(url)
        else:
            seen.append((entry.get("fetched", ""), url))

    seen.sort()
    stale = [url for _, url in seen[:refresh]]
    return fresh + changed + stale


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=400,
        help="이번 실행에서 허용할 최대 요청 수 (기본 400). 사업장 하나에 2콜이 든다.",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="단계 변화가 없어도 다시 받아볼 사업장 수 (기본 30). 오래된 것부터.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3, help="요청 사이 대기 초 (기본 0.3)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="동시 요청 수 (기본 3). 공공 웹사이트라 낮게 잡는다.",
    )
    parser.add_argument("--list-only", action="store_true", help="목록만 갱신하고 끝낸다")
    args = parser.parse_args()

    state = load_state()
    today = date.today().isoformat()

    # 1) 목록
    try:
        projects = fetch_project_list()
    except Exception as exc:
        print(f"목록 수집 실패: {exc}", file=sys.stderr)
        return 1
    if not projects:
        print("목록이 비어 있습니다. 화면 구조가 바뀌었을 수 있습니다.", file=sys.stderr)
        return 1

    changed_files = 0
    if _write(projects_path(), cleanup_api.projects_to_csv(projects)):
        changed_files += 1
    print(f"사업장 목록 {len(projects)}건 · 자치구 {len({p['sgg_nm'] for p in projects})}개")

    if args.list_only:
        save_state(state)
        return 0

    # 2) 추진경과
    work = build_worklist(projects, state, args.refresh)
    stage_by_url = {p["cafe_url"]: p.get("stage", "") for p in projects}
    budget = Budget(args.max_calls)
    limited = False

    existing: dict[str, list[dict[str, str]]] = {}
    for row in _read(events_path()):
        existing.setdefault(row["cafe_url"], []).append(row)

    print(f"추진경과 대상 {len(work)}곳 · 예산 {args.max_calls}콜 (곳당 2콜)")

    def worker(cafe_url: str) -> tuple[str, list[dict[str, str]]]:
        budget.take(2)  # 카페 메인 + 추진경과
        if args.sleep:
            time.sleep(args.sleep)
        return cafe_url, fetch_progress(cafe_url)

    # 1,000곳 가까이 받는 실행은 20분 넘게 걸린다. 끝에서 한 번만 저장하면
    # 중간에 죽었을 때 그때까지 받은 걸 통째로 잃고 다음 실행이 처음부터 다시 긁는다.
    # 주기적으로 흘려 둔다.
    SAVE_EVERY = 25

    collected = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, url): url for url in work}
        for done_count, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            if done_count % SAVE_EVERY == 0:
                flat = [row for rows in existing.values() for row in rows]
                _write(events_path(), cleanup_api.events_to_csv(flat))
                save_state(state)
                print(f"  ... {collected}곳 수집 · 잔여예산 {budget.left}콜", flush=True)
            try:
                url, events = future.result()
            except LimitReached:
                limited = True
                continue
            except cleanup_api.CafeMissing:
                # 목록이 '일시중단' 으로 표시하지 않았는데도 카페가 닫힌 경우.
                # 실패가 아니라 확정된 상태이므로 done 에 넣어 재시도를 멈춘다.
                # 나중에 진행단계가 바뀌면 build_worklist 가 다시 데려온다.
                existing.pop(url, None)
                state["done"][url] = {
                    "events": 0,
                    "stage": stage_by_url.get(url, ""),
                    "fetched": today,
                    "no_cafe": True,
                }
                state["failed"].pop(url, None)
                continue
            except cleanup_api.ParseError as exc:
                state["failed"][url] = f"parse: {exc}"
                continue
            except Exception as exc:
                state["failed"][url] = f"error: {exc}"
                continue

            existing[url] = events
            state["done"][url] = {
                "events": len(events),
                "stage": stage_by_url.get(url, ""),
                "fetched": today,
            }
            state["failed"].pop(url, None)
            collected += 1

    # 3) 저장
    flat = [row for rows in existing.values() for row in rows]
    if _write(events_path(), cleanup_api.events_to_csv(flat)):
        changed_files += 1
    save_state(state)

    print(
        f"\n요청 {budget.used}콜 · 추진경과 {collected}곳 수집 · "
        f"이벤트 누적 {len(flat)}건 · 파일 {changed_files}개 갱신"
    )
    print(f"진행률 {len(state['done'])}/{len(projects)}곳")
    if limited:
        print("예산을 다 써서 중단했습니다. 다음 실행에서 이어받습니다.")
    if state["failed"]:
        print(f"실패로 남은 사업장 {len(state['failed'])}곳 (다음 실행에서 재시도)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
