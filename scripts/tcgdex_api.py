"""TCGdex 카드 API 파싱.

    https://api.tcgdex.net/v2/en

API 키가 필요 없다. 순수 함수만 두어 네트워크 없이 검증한다 — 공식 정부 API 가
아니라 응답 형태가 예고 없이 바뀔 수 있으므로 fixtures 로 회귀를 잡는다.

세트 목록에는 가격이 없다. 가격은 카드 1장당 1요청이다 (2026-08-07 확인).
"""

from __future__ import annotations

BASE = "https://api.tcgdex.net/v2/en"

# 이미지 URL 앞머리. 저장할 때는 떼고 'base/base1/4' 만 남긴다 — 1만 7천 줄에
# 같은 45자를 반복해 넣을 이유가 없다. 화면에서 다시 붙인다.
IMAGE_PREFIX = "https://assets.tcgdex.net/en/"

# 카드 CSV 컬럼 순서. collect_pokemon.py 가 이 순서로 쓴다.
CARD_COLUMNS = [
    "card_id", "set_id", "local_id",
    "name_en", "dex_id", "rarity", "category", "image",
    "tp_market", "tp_low", "tp_mid", "tp_high",
    "cm_avg", "cm_low", "cm_trend",
    "obs_max", "obs_max_date", "updated",
]

# 가격 컬럼만 따로 — 갱신할 때 메타는 두고 이것만 덮는다.
PRICE_COLUMNS = [
    "tp_market", "tp_low", "tp_mid", "tp_high",
    "cm_avg", "cm_low", "cm_trend",
]

ERAS = ("빈티지", "클래식", "모던", "최신")

# TCGplayer variant 우선순위. 홀로가 그 카드의 '대표 시세'로 통용된다.
VARIANT_PRIORITY = ("holofoil", "normal", "reverseHolofoil")


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


def short_image(image: str | None) -> str:
    """이미지 URL 에서 CDN 앞머리를 뗀다. 화면에서 다시 붙인다."""
    if not image:
        return ""
    return image[len(IMAGE_PREFIX):] if image.startswith(IMAGE_PREFIX) else image


def first_dex_id(dex) -> str:
    """dexId 는 배열로 온다([6]). 첫 번째만 쓴다. 트레이너·에너지는 없다."""
    if isinstance(dex, list) and dex:
        return str(dex[0])
    if isinstance(dex, int):
        return str(dex)
    return ""


def parse_card_full(payload: dict, on_date: str) -> dict | None:
    """카드 응답에서 메타와 현재가를 한 행으로. 가격이 하나도 없으면 None.

    관측 최고가는 이 시점의 대표 시세로 시작한다. 다음 수집에서
    merge_card 가 더 높은 값이 나오면 갱신한다.
    """
    if not isinstance(payload, dict) or not payload.get("id"):
        return None

    pricing = payload.get("pricing") or {}
    tp_pick = pick_variant(pricing.get("tcgplayer") or {})
    cm = pricing.get("cardmarket") or {}
    _, tp = ("", {}) if tp_pick is None else tp_pick

    row = {
        "card_id": payload["id"],
        "set_id": (payload.get("set") or {}).get("id") or "",
        "local_id": str(payload.get("localId") or ""),
        "name_en": (payload.get("name") or "").strip(),
        "dex_id": first_dex_id(payload.get("dexId")),
        "rarity": (payload.get("rarity") or "").strip(),
        "category": (payload.get("category") or "").strip(),
        "image": short_image(payload.get("image")),
        "tp_market": _num(tp.get("marketPrice")),
        "tp_low": _num(tp.get("lowPrice")),
        "tp_mid": _num(tp.get("midPrice")),
        "tp_high": _num(tp.get("highPrice")),
        "cm_avg": _num(cm.get("avg")),
        "cm_low": _num(cm.get("low")),
        "cm_trend": _num(cm.get("trend")),
        "obs_max": None,
        "obs_max_date": "",
        "updated": on_date,
    }
    if all(row[c] is None for c in PRICE_COLUMNS):
        return None

    current = row["tp_market"] or row["cm_avg"]
    if current is not None:
        row["obs_max"] = current
        row["obs_max_date"] = on_date
    return row


def merge_card(old: dict | None, new: dict) -> dict:
    """새로 받은 행을 기존 행에 겹친다. 관측 최고가는 더 높은 쪽을 남긴다.

    최고가를 새 값으로 덮어쓰면 '관측 최고가'가 그냥 현재가가 된다. 값이
    내려간 날에도 최고 기록은 남아야 한다.
    """
    if not old:
        return new
    merged = dict(new)
    old_max = old.get("obs_max")
    new_max = new.get("obs_max")
    if old_max is not None and (new_max is None or old_max >= new_max):
        merged["obs_max"] = old_max
        merged["obs_max_date"] = old.get("obs_max_date", "")
    return merged
