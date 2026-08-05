#!/usr/bin/env python3
"""브이월드에서 경기 아파트 단지의 필지·용도지역을 받는다.

경기도는 지금까지 용도지역이 비어 있었다. 서울시 도시계획포털(UPIS)의
지적도·용도지역은 서울 전용이고, 건축물대장 지역지구구역은 종 구분을 주지
않아('일반주거지역'), 용적률 상한을 특정할 수 없었다. 브이월드는
'제2종일반주거지역'처럼 종까지 준다.

덤으로 필지 면적도 받는다. 건축물대장 platArea 가 21% 비어 있어 경기 단지의
대지지분이 그만큼 빠져 있었는데, 이걸로 메운다.

  단지 하나에 2콜 — 지적도(PNU 조회) + 용도지역(대표점 조회)

출력: data/vworld/parcels.csv.gz
상태: data/state/vworld_state.json

인증키: 환경변수 VWORLD_API_KEY 또는 저장소 루트 .env (커밋되지 않음)
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
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_api  # noqa: E402
import vworld_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMPLEX_FILE = ROOT / "data" / "complexes.csv.gz"
OUT_FILE = ROOT / "data" / "vworld" / "parcels.csv.gz"
STATE_FILE = ROOT / "data" / "state" / "vworld_state.json"

# 기본 대상은 경기다. 서울은 UPIS 지적도로 이미 채워져 있다.
# --sido 로 바꿔 서울을 다시 받아 교차검증할 수 있다 (은마 면적 오차 0.06% 였다).
DEFAULT_SIDO = "41"


class LimitReached(Exception):
    """이번 실행 예산을 다 썼거나 브이월드가 한도를 알렸다."""


def _get(url: str, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                if attempt == retries - 1:
                    raise LimitReached("HTTP 429 가 계속됩니다") from exc
                time.sleep(3 * (attempt + 1))
                last = exc
                continue
            if 400 <= exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def target_pnus(sido: str, min_age: int, this_year: int) -> list[str]:
    """대상 아파트 PNU. 준공연도로 한 번 거른다.

    브이월드는 단지당 2콜이라 전량(경기 아파트 7,771곳)을 받으면 15,000콜이다.
    노후 단지만 받아 예산을 아낀다 — 화면에서 25년 미만은 어차피 감춘다.
    """
    text = gzip.decompress(COMPLEX_FILE.read_bytes()).decode("utf-8")
    out: list[str] = []
    for row in csv.DictReader(io.StringIO(text, newline="")):
        if row.get("complex_type_code") != "1":
            continue
        pnu = row.get("pnu") or ""
        if not pnu.startswith(sido):
            continue
        approval = (row.get("use_approval_date") or "").strip()
        year = int(approval[:4]) if len(approval) >= 4 and approval[:4].isdigit() else 0
        if not year or this_year - year < min_age:
            continue
        out.append(pnu)
    return sorted(set(out))


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


def read_existing() -> dict[str, dict]:
    if not OUT_FILE.exists():
        return {}
    text = gzip.decompress(OUT_FILE.read_bytes()).decode("utf-8")
    return {r["pnu"]: r for r in csv.DictReader(io.StringIO(text, newline="")) if r.get("pnu")}


def write_out(rows: dict[str, dict]) -> bool:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=vworld_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for pnu in sorted(rows):
        writer.writerow({c: rows[pnu].get(c, "") for c in vworld_api.COLUMNS})
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = cleanup_api.gzip_bytes(buf.getvalue())
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        return False
    OUT_FILE.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sido", default=DEFAULT_SIDO, help="시도 2자리 (기본 41=경기)")
    parser.add_argument("--min-age", type=int, default=25, help="이 연차 이상만 받는다")
    parser.add_argument("--max-calls", type=int, default=7000, help="이번 실행 최대 호출 수")
    parser.add_argument("--sleep", type=float, default=0.05, help="요청 사이 대기 초")
    parser.add_argument("--skip-landuse", action="store_true", help="필지만 받고 용도지역은 건너뛴다")
    args = parser.parse_args()

    try:
        key = vworld_api.load_key()
    except vworld_api.KeyMissing as exc:
        print(exc, file=sys.stderr)
        return 1

    state = load_state()
    today = date.today()
    pnus = target_pnus(args.sido, args.min_age, today.year)
    pending = [p for p in pnus if p not in state["done"]]
    rows = read_existing()

    print(
        f"시도 {args.sido} · 준공 {args.min_age}년 이상 아파트 {len(pnus)}곳 · "
        f"이번에 받을 곳 {len(pending)}곳 · 예산 {args.max_calls}콜 (곳당 2콜)"
    )

    calls = 0
    limited = False
    got_parcel = got_landuse = 0

    for i, pnu in enumerate(pending, 1):
        if calls + 2 > args.max_calls:
            limited = True
            break
        try:
            calls += 1
            parcel = vworld_api.parse_parcel(_get(vworld_api.parcel_url(key, pnu)))
            if parcel is None or not parcel.get("lon"):
                # 지적도에 없는 필지. 재시도해도 같으니 done 으로 닫는다.
                state["done"][pnu] = {"found": False, "fetched": today.isoformat()}
                state["failed"].pop(pnu, None)
                continue
            got_parcel += 1

            if not args.skip_landuse:
                calls += 1
                landuse = vworld_api.parse_landuse(
                    _get(vworld_api.landuse_url(key, float(parcel["lon"]), float(parcel["lat"])))
                )
                if landuse:
                    parcel.update(landuse)
                    got_landuse += 1

            rows[pnu] = parcel
            state["done"][pnu] = {"found": True, "fetched": today.isoformat()}
            state["failed"].pop(pnu, None)
        except LimitReached as exc:
            print(f"  중단: {exc}", file=sys.stderr)
            limited = True
            break
        except vworld_api.VworldError as exc:
            state["failed"][pnu] = str(exc)[:120]
            continue
        except Exception as exc:
            state["failed"][pnu] = f"error: {exc}"[:120]
            continue

        if i % 200 == 0:
            write_out(rows)
            save_state(state)
            print(f"  ... {i}/{len(pending)} · 필지 {got_parcel} · 용도지역 {got_landuse} · {calls}콜", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    changed = write_out(rows)
    save_state(state)

    with_zone = sum(1 for r in rows.values() if r.get("landuse_nm"))
    print(
        f"\n호출 {calls}콜 · 필지 {got_parcel}곳 새로 받음 · 누적 {len(rows)}곳 "
        f"(용도지역 있음 {with_zone}곳) · 파일 {'갱신' if changed else '변화 없음'}"
    )
    print(f"진행률 {len(state['done'])}/{len(pnus)}")
    if limited:
        print("예산/한도로 중단했습니다. 다음 실행에서 이어받습니다.")
    if state["failed"]:
        print(f"실패로 남은 단지 {len(state['failed'])}곳 (다음 실행에서 재시도)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
