"""환율을 받아 저장한다. 두 시장을 같은 축에 놓으려면 이게 필요하다.

    python3 scripts/collect_fx.py

달러·원·유로·엔 넷을 들고 다닌다. 기준은 USD 다.

open.er-api.com 은 키가 필요 없고 기준일을 같이 준다. 환율은 매일 바뀌므로
**기준일을 반드시 들고 다닌다** — 화면에 "1,423원 기준" 이라고만 적으면
언제 값인지 몰라 오해한다.

주의: 화면의 유로에는 두 종류가 있다. 여기서 만든 건 **달러를 환산한 유로**고,
카드 데이터의 `cm_avg` 는 **Cardmarket 유럽 시장의 실제 체결가**다. 둘은 다른
값이므로 화면에서 섞지 않는다.

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

# USD 1 당 이 범위를 벗어나면 응답이 이상한 것이다. 조용히 틀린 환율로
# 덮어쓰면 화면의 모든 환산이 함께 틀어지므로 통화마다 문을 달아 둔다.
SANE = {
    "KRW": (500.0, 3000.0),
    "EUR": (0.5, 2.0),
    "JPY": (60.0, 400.0),
}


def parse(payload: dict) -> dict | None:
    """{'rates': {'USD':1,'KRW':…,'EUR':…,'JPY':…}, 'date': …} 또는 None.

    넷 중 하나라도 이상하면 통째로 버린다 — 일부만 맞는 환율표는 화면에서
    어느 칸이 틀렸는지 알 수 없어 더 위험하다.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("rates")
    if not isinstance(raw, dict):
        return None

    rates = {"USD": 1.0}
    for code, (low, high) in SANE.items():
        try:
            value = float(raw.get(code))
        except (TypeError, ValueError):
            return None
        if not (low <= value <= high):
            return None
        rates[code] = round(value, 6 if code == "EUR" else 4)

    stamp = str(payload.get("time_last_update_utc") or "")
    return {
        "base": "USD",
        "rates": rates,
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
    rates = parsed["rates"]
    print(f"USD 1 = {rates['KRW']:,.2f}원 · {rates['EUR']:.4f}유로 · "
          f"{rates['JPY']:,.2f}엔 ({parsed['date']} 기준) -> {FX_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
