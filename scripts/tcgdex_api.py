"""TCGdex 카드 가격 API 파싱과 지수 계산.

    https://api.tcgdex.net/v2/en

API 키가 필요 없다. 순수 함수만 두어 네트워크 없이 검증한다 — 공식 정부 API 가
아니라 응답 형태가 예고 없이 바뀔 수 있으므로 fixtures 로 회귀를 잡는다.

세트 목록에는 가격이 없다. 가격은 카드 1장당 1요청이다 (2026-08-07 확인).
"""

from __future__ import annotations

import random

BASE = "https://api.tcgdex.net/v2/en"

# 가격 CSV 컬럼 순서. collect_pokemon.py 가 이 순서로 쓴다.
COLUMNS = [
    "date", "card_id", "variant",
    "tp_market", "tp_low", "tp_mid",
    "cm_avg", "cm_trend", "cm_avg7", "cm_avg30",
]

ERAS = ("빈티지", "클래식", "모던", "최신")
BANDS = ("고가", "중가", "저가")

# TCGplayer variant 우선순위. 홀로가 그 카드의 '대표 시세'로 통용된다.
VARIANT_PRIORITY = ("holofoil", "normal", "reverseHolofoil")

# 칸당 표본 수. 12칸 × 25 = 300장. 동시 4로 약 70초면 다 받는다.
PER_CELL = 25

# 가격이 빠졌을 때 마지막 값을 이월하는 최대 일수.
CARRY_FORWARD_DAYS = 7


class ApiError(Exception):
    """응답이 기대한 형태가 아닐 때."""


def parse_set_list(payload) -> list[dict]:
    """세트 목록 응답을 [{'set_id','name','card_count'}] 로."""
    if not isinstance(payload, list):
        raise ApiError(f"세트 목록이 배열이 아닙니다: {type(payload).__name__}")
    rows = []
    for item in payload:
        sid = (item.get("id") or "").strip()
        if not sid:
            continue
        rows.append({
            "set_id": sid,
            "name": (item.get("name") or "").strip(),
            "card_count": int((item.get("cardCount") or {}).get("total") or 0),
        })
    return rows


def parse_set_detail(payload: dict) -> dict:
    """세트 상세를 {'set_id','name','release_date','card_ids'} 로.

    이 응답의 카드 배열에는 가격이 없다. id 만 걷어 카드별로 다시 부른다.
    """
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ApiError("세트 상세에 id 가 없습니다")
    return {
        "set_id": payload["id"],
        "name": (payload.get("name") or "").strip(),
        "release_date": (payload.get("releaseDate") or "").strip(),
        "card_ids": [c["id"] for c in (payload.get("cards") or []) if c.get("id")],
    }


def era_of(release_date: str) -> str | None:
    """발매일을 시대 구간으로. 빈 값이면 None."""
    if not release_date or len(release_date) < 4 or not release_date[:4].isdigit():
        return None
    year = int(release_date[:4])
    if year <= 2003:
        return "빈티지"
    if year <= 2010:
        return "클래식"
    if year <= 2019:
        return "모던"
    return "최신"


def _num(value) -> float | None:
    """숫자로 바꾼다. None·빈 문자열·0 이하는 None (0원은 시세가 아니다)."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def pick_variant(tcgplayer) -> tuple[str, dict] | None:
    """대표 variant 를 고른다. marketPrice 가 있는 것만 후보다."""
    if not isinstance(tcgplayer, dict):
        return None
    for name in VARIANT_PRIORITY:
        block = tcgplayer.get(name)
        if isinstance(block, dict) and _num(block.get("marketPrice")) is not None:
            return name, block
    return None


def parse_card_pricing(payload: dict, on_date: str) -> dict | None:
    """카드 응답에서 가격 한 행을 만든다. 양쪽 다 값이 없으면 None.

    TCGplayer(USD) 가 지수 기준이고 Cardmarket(EUR) 은 avg7/avg30 을 주므로
    0일차 변화율의 유일한 근거다. 한쪽만 있어도 행을 남긴다.
    """
    pricing = (payload or {}).get("pricing") or {}
    tp_pick = pick_variant(pricing.get("tcgplayer") or {})
    cm = pricing.get("cardmarket") or {}

    variant, tp = ("", {}) if tp_pick is None else tp_pick
    row = {
        "date": on_date,
        "card_id": (payload or {}).get("id") or "",
        "variant": variant,
        "tp_market": _num(tp.get("marketPrice")),
        "tp_low": _num(tp.get("lowPrice")),
        "tp_mid": _num(tp.get("midPrice")),
        "cm_avg": _num(cm.get("avg")),
        "cm_trend": _num(cm.get("trend")),
        "cm_avg7": _num(cm.get("avg7")),
        "cm_avg30": _num(cm.get("avg30")),
    }
    if not row["card_id"]:
        return None
    if all(row[k] is None for k in COLUMNS[3:]):
        return None
    return row
