#!/usr/bin/env python3
"""서울·경기 아파트 단지의 건축물대장 총괄표제부를 법정동 단위로 수집한다.

필지마다 부르면 아파트만 17,160콜이다. 총괄표제부는 bun/ji 를 생략하면
그 법정동 전체를 한 번에 주고 (표본 30~50건), 아파트가 실제로 있는 법정동은
서울·경기를 합쳐 930개뿐이라 하루치 한도 안에서 끝난다.

출력: data/bldrgst/bldrgst.csv.gz
상태: data/state/bldrgst_state.json

인증키는 collect_trades.py 와 같은 것을 쓴다 (DATA_GO_KR_API_KEY).
다만 활용신청은 실거래가와 별도다 (2026-08-05 승인). 403 이 오면 신청이
풀렸는지부터 확인한다 — 키가 틀린 게 아니라 그 서비스만 막힌 것이다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bldrgst_api  # noqa: E402
import cleanup_api  # noqa: E402
import collect_trades  # noqa: E402  인증키 로딩을 한 곳에서만 정의한다

ROOT = Path(__file__).resolve().parent.parent
COMPLEX_FILE = ROOT / "data" / "complexes.csv.gz"
OUT_FILE = ROOT / "data" / "bldrgst" / "bldrgst.csv.gz"
STATE_FILE = ROOT / "data" / "state" / "bldrgst_state.json"

API = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrRecapTitleInfo"
PAGE_SIZE = 100


class LimitReached(Exception):
    """일일 호출 한도에 도달했거나 이번 실행 예산을 다 썼다."""


def fetch_page(key: str, sgg: str, bjd: str, page: int, retries: int = 3) -> str:
    params = {
        "serviceKey": key,
        "sigunguCd": sgg,
        "bjdongCd": bjd,
        "platGbCd": "0",
        "numOfRows": str(PAGE_SIZE),
        "pageNo": str(page),
        "_type": "xml",
    }
    url = f"{API}?{urllib.parse.urlencode(params, safe='')}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise LimitReached("HTTP 429 일일 호출 한도 초과") from exc
            if exc.code == 403:
                raise SystemExit(
                    "건축물대장 API 가 403 입니다. data.go.kr 에서 "
                    "'국토교통부_건축HUB_건축물대장정보 서비스' 활용신청을 확인하세요."
                ) from exc
            if 400 <= exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def target_dongs() -> list[tuple[str, str]]:
    """아파트가 실제로 등록된 법정동 (시군구5, 법정동5) 목록.

    준공연도로 거르지 않는다 — 화면에서 연차 기준을 바꿀 때 재수집을 피한다.
    """
    text = gzip.decompress(COMPLEX_FILE.read_bytes()).decode("utf-8")
    dongs = {
        (r["pnu"][:5], r["pnu"][5:10])
        for r in csv.DictReader(io.StringIO(text, newline=""))
        if r.get("complex_type_code") == "1" and len(r.get("pnu") or "") >= 10
    }
    return sorted(dongs)


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


def read_existing() -> list[dict]:
    if not OUT_FILE.exists():
        return []
    return list(
        csv.DictReader(
            io.StringIO(gzip.decompress(OUT_FILE.read_bytes()).decode("utf-8"), newline="")
        )
    )


def write_out(rows: list[dict]) -> bool:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = cleanup_api.gzip_bytes(bldrgst_api.rows_to_csv(rows))
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        return False
    OUT_FILE.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=900, help="이번 실행 최대 호출 수")
    parser.add_argument("--sleep", type=float, default=0.1, help="요청 사이 대기 초")
    parser.add_argument("--refresh", action="store_true", help="이미 받은 법정동도 다시 받는다")
    args = parser.parse_args()

    key = collect_trades.load_api_key()
    state = load_state()
    today = date.today().isoformat()

    dongs = target_dongs()
    pending = [d for d in dongs if args.refresh or f"{d[0]}{d[1]}" not in state["done"]]
    print(f"아파트 있는 법정동 {len(dongs)}개 · 이번에 받을 곳 {len(pending)}개 · 예산 {args.max_calls}콜")

    collected = list(read_existing())
    calls = 0
    limited = False

    for sgg, bjd in pending:
        if calls >= args.max_calls:
            limited = True
            break
        key_str = f"{sgg}{bjd}"
        page = 1
        rows: list[dict] = []
        try:
            while True:
                if calls >= args.max_calls:
                    limited = True
                    break
                calls += 1
                batch, total = bldrgst_api.parse_response(fetch_page(key, sgg, bjd, page))
                rows.extend(batch)
                if page * PAGE_SIZE >= total or not total:
                    break
                page += 1
                if args.sleep:
                    time.sleep(args.sleep)
        except LimitReached as exc:
            print(f"  중단: {exc}", file=sys.stderr)
            limited = True
            break
        except bldrgst_api.ApiError as exc:
            if exc.is_limit:
                print(f"  중단: {exc}", file=sys.stderr)
                limited = True
                break
            state["failed"][key_str] = f"{exc.code}: {exc.message}"
            continue
        except Exception as exc:
            state["failed"][key_str] = f"error: {exc}"
            continue

        if limited and not rows:
            break
        collected = bldrgst_api.merge_rows(collected, rows)
        state["done"][key_str] = {"rows": len(rows), "fetched": today}
        state["failed"].pop(key_str, None)

        if len(state["done"]) % 50 == 0:
            write_out(collected)
            save_state(state)
            print(f"  ... 법정동 {len(state['done'])}/{len(dongs)} · 단지 {len(collected)}곳 · {calls}콜")
        if args.sleep:
            time.sleep(args.sleep)

    changed = write_out(collected)
    save_state(state)

    with_land = sum(1 for r in collected if bldrgst_api._num(r.get("plat_area")) > 0)
    with_hh = sum(1 for r in collected if bldrgst_api._num(r.get("hhld_cnt")) > 0)
    with_gfa = sum(1 for r in collected if bldrgst_api.floor_area(r))
    print(
        f"\n호출 {calls}콜 · 법정동 {len(state['done'])}/{len(dongs)} · 단지 {len(collected)}곳 "
        f"· 파일 {'갱신' if changed else '변화 없음'}"
    )
    print(
        f"채움률: 대지면적 {with_land}/{len(collected)} · 세대수 {with_hh}/{len(collected)} "
        f"· 연면적 {with_gfa}/{len(collected)}"
    )
    if limited:
        print("예산을 다 써서 중단했습니다. 다음 실행에서 이어받습니다.")
    if state["failed"]:
        print(f"실패로 남은 법정동 {len(state['failed'])}개 (다음 실행에서 재시도)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
