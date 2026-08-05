#!/usr/bin/env python3
"""서울·경기 지자체 도시계획조례에서 용도지역별 용적률 상한을 받는다.

용적률 상한은 광역이 아니라 시·군 조례가 정한다. 서울시 값(3종 250%)을 경기에
붙이면 틀린다 — 실측하면 가평 300%, 용인 290%, 성남·안양 280%, 고양 250% 다.

서울 1 + 경기 31 = 32곳이고 곳당 2콜(조례 검색 + 본문)이라 64콜이면 끝난다.
서울도 함께 받아 화면이 쓰는 상한을 한 출처로 통일했다 (예전에는 서울 값만
upis_api.FAR_LIMIT 에 상수로 박혀 있었고, 실측값이 그 표와 일치해 교차검증됐다).
개정이 잦지 않아 월 1회면 충분하다 — redev_daily.sh 가 매월 5일에 부른다.

출력: data/ordinance/far_limits.csv.gz
상태: data/state/ordinance_state.json

인증: 환경변수 LAW_OC (open.law.go.kr 가입 이메일 ID). 없으면 'test' 로 넘어간다 —
법제처가 공개 시험용으로 열어둔 값이고 실제로 응답한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_api  # noqa: E402
import ordinance_api  # noqa: E402
import regions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "ordinance" / "far_limits.csv.gz"
STATE_FILE = ROOT / "data" / "state" / "ordinance_state.json"

# 서울은 자치구가 아니라 특별시 조례 하나가 25개 구를 다 덮는다.
# 경기는 시·군마다 따로다. 둘을 같은 파이프라인에서 받아 하드코딩 표를 없앤다.
SIDO_NAMES = ("서울특별시", "경기도")


def load_oc() -> str:
    return os.environ.get("LAW_OC") or "test"


def cities() -> list[tuple[str, str]]:
    """조례를 받을 (지자체기관명, 조례 검색어) 목록.

    '경기도 수원시 장안구' 4개는 모두 수원시 조례 하나를 따르므로 자치구를 접는다.
    서울은 자치구 조례가 없고 '서울특별시 도시계획 조례' 하나뿐이다.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, full in regions.sgg_codes():
        parts = full.split()
        if not parts or parts[0] not in SIDO_NAMES:
            continue
        if parts[0] == "서울특별시":
            org, name = "서울특별시", "서울특별시"
        else:
            if len(parts) < 2:
                continue
            org, name = f"{parts[0]} {parts[1]}", parts[1]
        if org in seen:
            continue
        seen.add(org)
        out.append((org, name))
    return out


def _get(url: str, retries: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep", type=float, default=0.4, help="요청 사이 대기 초")
    parser.add_argument("--only", help="특정 시·군만 (디버깅용)")
    args = parser.parse_args()

    oc = load_oc()
    state = load_state()
    today = date.today().isoformat()
    targets = [c for c in cities() if not args.only or c[1] == args.only]
    if not targets:
        print(f"대상 시·군이 없습니다: {args.only}", file=sys.stderr)
        return 1

    print(f"서울·경기 지자체 {len(targets)}곳 · OC={oc} · 곳당 2콜")

    rows: list[dict] = []
    ok = partial = failed = 0

    for org, city in targets:
        # 시는 '도시계획 조례', 군은 '군계획 조례' 라 두 이름을 다 시도한다.
        found = None
        for suffix in ("도시계획 조례", "군계획 조례"):
            try:
                found = ordinance_api.pick_ordinance(
                    _get(ordinance_api.search_url(oc, f"{city} {suffix}")), org
                )
            except (ordinance_api.OrdinanceError, urllib.error.URLError, OSError) as exc:
                state["failed"][city] = f"search: {exc}"[:120]
                found = None
            if found:
                break
            if args.sleep:
                time.sleep(args.sleep)

        if not found:
            failed += 1
            state["failed"].setdefault(city, "조례를 찾지 못했습니다")
            print(f"  {city:8s} ❌ 조례 없음")
            continue

        try:
            far = ordinance_api.parse_far(_get(ordinance_api.service_url(oc, found["law_id"])))
        except (ordinance_api.OrdinanceError, urllib.error.URLError, OSError) as exc:
            failed += 1
            state["failed"][city] = f"body: {exc}"[:120]
            print(f"  {city:8s} ❌ 본문 실패")
            continue

        if not far:
            failed += 1
            state["failed"][city] = "용적률 조를 찾지 못했습니다"
            print(f"  {city:8s} ⚠️  용적률 조 없음 ({found['law_name']})")
            continue

        for zone, value in far.items():
            rows.append(
                {
                    "city": city,
                    "law_id": found["law_id"],
                    "law_name": found["law_name"],
                    "promulgated": found["promulgated"],
                    "zone": zone,
                    "far": value["far"],
                    "far_redev": value["far_redev"] if value["far_redev"] else "",
                    "has_proviso": "1" if value["has_proviso"] else "",
                }
            )

        # 주거지역 4종이 다 나왔는지로 완전/부분을 가른다. 화면이 쓰는 건 이쪽이다.
        need = {"제1종일반주거지역", "제2종일반주거지역", "제3종일반주거지역", "준주거지역"}
        got = need & set(far)
        if got == need:
            ok += 1
            mark = "✅"
        else:
            partial += 1
            mark = "△"
        provisos = sum(1 for v in far.values() if v["has_proviso"])
        print(
            f"  {city:8s} {mark} 용도지역 {len(far)}종 · 주거 {len(got)}/4"
            + (f" · 정비사업 단서 {provisos}건" if provisos else "")
        )
        state["done"][city] = {
            "law_id": found["law_id"],
            "promulgated": found["promulgated"],
            "zones": len(far),
            "fetched": today,
        }
        state["failed"].pop(city, None)
        if args.sleep:
            time.sleep(args.sleep)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = cleanup_api.gzip_bytes(ordinance_api.rows_to_csv(rows))
    changed = not (OUT_FILE.exists() and OUT_FILE.read_bytes() == data)
    if changed:
        OUT_FILE.write_bytes(data)
    save_state(state)

    print(
        f"\n완전 {ok}곳 · 부분 {partial}곳 · 실패 {failed}곳 · "
        f"행 {len(rows)}개 · 파일 {'갱신' if changed else '변화 없음'}"
    )
    if state["failed"]:
        print(f"못 읽은 시·군 {len(state['failed'])}곳 — 상한을 비워 둡니다 (지어내지 않음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
