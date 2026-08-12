#!/usr/bin/env python3
"""대시보드 산출물이 낡았는지 검사한다. 크론 스크립트 끝에서 부른다.

판정은 freshness_api.py 가 한다. 여기는 파일을 읽고 결과를 찍는 일만 한다.

두 가지를 따로 본다.

  빌드 날짜(generated)   집계가 돌긴 했나 — 크론이 멈추면 여기서 걸린다
  데이터 날짜            담긴 자료가 최신인가 — **수집만 깨진 경우는 여기서만 걸린다**

둘을 나눈 이유가 중요하다. 수집이 조용히 깨져도 집계는 옛 원본으로 성공한다.
그러면 generated 는 오늘로 갱신되고 데이터는 지난달에 멈춘다. 빌드 날짜만
보는 검사는 이 고장을 절대 못 잡는다.

데이터 날짜가 없는 산출물(학군)은 "빌드 날짜만 확인" 이라고 적는다 — 검사한
척하면 안 본 것보다 나쁘다.

사용법

    python3 scripts/check_freshness.py                # 전부
    python3 scripts/check_freshness.py trades redev   # 일부만

종료코드 1 이면 하나 이상이 허용 나이를 넘겼다는 뜻이다.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from freshness_api import (  # noqa: E402
    BUILD_LIMITS, MONTH_LAG_LIMITS, days_since, days_since_english,
    is_month_stale, is_stale, months_behind,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

# 이름 -> (파일, 사람이 읽을 이름, 데이터 날짜를 꺼내는 방법)
#
# 데이터 날짜 방법은 (종류, 경로) 다.
#   ("month", ["months", -1])    월 단위. MONTH_LAG_LIMITS 로 본다
#   ("english", ["fx", "date"])  "11 Aug 2026" 꼴. BUILD_LIMITS 로 본다
#   None                          데이터 날짜가 없다
TARGETS: dict[str, tuple[str, str, tuple | None]] = {
    "trades": ("assets/realestate/summary.json", "실거래 대시보드",
               ("month", ["months", -1])),
    "schools": ("assets/realestate/schools.json", "학군 지도", None),
    "redev": ("assets/realestate/redev.json", "재개발·재건축",
              ("month", ["latest_month"])),
    "pokemon": ("assets/pokemon/meta.json", "포켓몬 카드",
                ("english", ["fx", "date"])),
}


def _dig(data, path: list):
    """중첩 경로를 따라간다. 없으면 None."""
    cur = data
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def check(name: str, today: date) -> bool:
    """하나를 본다. 낡았으면 False."""
    rel, label, data_spec = TARGETS[name]
    path = REPO / rel
    limit = BUILD_LIMITS[name]

    if not path.exists():
        print(f"  ✖ {label}: {rel} 이 없습니다.", file=sys.stderr)
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ✖ {label}: {rel} 을 읽지 못했습니다({exc}).", file=sys.stderr)
        return False

    ok = True

    built = data.get("generated", "")
    age = days_since(built, today)
    if is_stale(built, limit, today):
        shown = f"{age}일 전" if age is not None else f"날짜를 못 읽음({built!r})"
        print(f"  ✖ {label}: 집계가 {shown} — 허용 {limit}일을 넘겼습니다. "
              f"크론이 멈췄는지 확인하세요.", file=sys.stderr)
        ok = False

    if data_spec is None:
        note = "데이터 날짜 없음(빌드 날짜만 확인)"
    else:
        kind, dig_path = data_spec
        raw = _dig(data, dig_path)
        note, ok = _check_data_date(name, label, kind, raw, today, ok)

    if ok:
        print(f"    {label:16s} 집계 {built}({age}일 전) · {note}")
    return ok


def _check_data_date(name: str, label: str, kind: str, raw, today: date,
                     ok: bool) -> tuple[str, bool]:
    """데이터 날짜를 본다. (설명, 통과여부) 를 돌려준다."""
    if kind == "month":
        lag_limit = MONTH_LAG_LIMITS[name]
        if is_month_stale(raw or "", lag_limit, today):
            lag = months_behind(raw or "", today)
            shown = f"{lag}달 뒤처짐" if lag is not None else f"못 읽음({raw!r})"
            print(f"  ✖ {label}: 데이터가 {raw!r} 까지뿐입니다({shown}, 허용 "
                  f"{lag_limit}달). **집계는 돌았는데 수집이 멈춘 모양입니다.**",
                  file=sys.stderr)
            return f"데이터 {raw}", False
        return f"데이터 {raw}({months_behind(raw, today)}달 뒤)", ok

    # english: 상위 자료의 날짜 문자열
    limit = BUILD_LIMITS[name]
    age = days_since_english(raw or "", today)
    if age is None or age > limit:
        shown = f"{age}일 전" if age is not None else f"못 읽음({raw!r})"
        print(f"  ✖ {label}: 원본 날짜가 {shown} — 허용 {limit}일을 넘겼습니다. "
              f"**집계는 돌았는데 수집이 멈춘 모양입니다.**", file=sys.stderr)
        return f"원본 {raw}", False
    return f"원본 {raw}({age}일 전)", ok


def main(argv: list[str]) -> int:
    names = argv or sorted(TARGETS)
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        print(f"모르는 이름: {' '.join(unknown)} — 가능한 값: {' '.join(sorted(TARGETS))}",
              file=sys.stderr)
        return 2

    today = date.today()
    print("  산출물 신선도:")
    results = [check(name, today) for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
