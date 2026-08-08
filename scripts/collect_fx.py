"""달러/원 환율을 받아 저장한다. 두 시장을 같은 축에 놓으려면 이게 필요하다.

    python3 scripts/collect_fx.py

open.er-api.com 은 키가 필요 없고 기준일을 같이 준다. 환율은 매일 바뀌므로
**기준일을 반드시 들고 다닌다** — 화면에 "1,423원 기준" 이라고만 적으면
언제 값인지 몰라 오해한다.

실패하면 기존 파일을 그대로 둔다. 어제 환율이 빈 값보다 낫다.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

FX_FILE = ROOT / "data" / "pokemon" / "fx.json"
URL = "https://open.er-api.com/v6/latest/USD"
USER_AGENT = "kayserdocs-pokemon/1.0"
TIMEOUT = 30

# 이 범위를 벗어나면 응답이 이상한 것이다. 조용히 틀린 환율로 덮어쓰지 않는다.
SANE_RANGE = (500.0, 3000.0)


def parse(payload: dict) -> dict | None:
    """{'rate': 1423.48, 'date': '2026-08-07'} 또는 None."""
    if not isinstance(payload, dict):
        return None
    rate = (payload.get("rates") or {}).get("KRW")
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        return None
    if not (SANE_RANGE[0] <= rate <= SANE_RANGE[1]):
        return None
    stamp = str(payload.get("time_last_update_utc") or "")
    return {
        "rate": round(rate, 2),
        "date": stamp[5:16].strip() or date.today().isoformat(),
        "source": "open.er-api.com",
    }


def main() -> int:
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except Exception as err:
        print(f"환율을 받지 못했습니다 ({err}).", file=sys.stderr)
        return 1 if not FX_FILE.exists() else 0

    parsed = parse(payload)
    if parsed is None:
        print("환율 응답이 이상합니다. 기존 값을 유지합니다.", file=sys.stderr)
        return 1 if not FX_FILE.exists() else 0

    FX_FILE.parent.mkdir(parents=True, exist_ok=True)
    FX_FILE.write_text(json.dumps(parsed, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"USD 1 = {parsed['rate']:,.2f}원 ({parsed['date']} 기준) -> {FX_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
