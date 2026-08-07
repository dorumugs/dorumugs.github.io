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


def tercile_bounds(prices: list[float]) -> tuple[float, float]:
    """가격 분포의 3분위 경계 (하한, 상한) 를 낸다."""
    if not prices:
        raise ValueError("가격이 비었습니다")
    ordered = sorted(prices)
    n = len(ordered)
    return ordered[n // 3], ordered[(2 * n) // 3]


def band_of(price: float, bounds: tuple[float, float]) -> str:
    """가격을 가격대 이름으로. 경계값은 위 칸에 넣는다."""
    low, high = bounds
    if price >= high:
        return "고가"
    if price >= low:
        return "중가"
    return "저가"


def stratify(cards: list[dict], per_cell: int = PER_CELL, seed: int = 20260807) -> list[dict]:
    """시대 × 가격대 12칸에서 고정 표본을 뽑는다.

    가격대 경계는 시대 안에서 잡는다. 시대를 가로질러 절대 금액으로 자르면
    빈티지가 전부 고가, 최신이 전부 저가가 되어 칸이 무너진다.

    칸 안에서 다시 5분위로 나눠 균등하게 뽑는다. 그냥 무작위로 뽑으면 저가
    쪽에 몰려 그 칸의 상단이 지수에 안 들어간다.
    """
    by_era: dict[str, list[dict]] = {}
    for card in cards:
        if card.get("era") in ERAS and card.get("price"):
            by_era.setdefault(card["era"], []).append(card)

    picked: list[dict] = []
    for era in ERAS:
        pool = by_era.get(era) or []
        if not pool:
            continue
        bounds = tercile_bounds([c["price"] for c in pool])
        cells: dict[str, list[dict]] = {b: [] for b in BANDS}
        for card in pool:
            band = band_of(card["price"], bounds)
            cells[band].append({**card, "band": band})

        for band in BANDS:
            cell = sorted(cells[band], key=lambda c: (c["price"], c["card_id"]))
            if len(cell) <= per_cell:
                picked.extend(cell)
                continue
            # 칸을 5분위로 갈라 각 분위에서 균등하게 뽑는다.
            rng = random.Random(f"{seed}-{era}-{band}")
            chunks = 5
            quota, extra = divmod(per_cell, chunks)
            chosen: list[dict] = []
            taken: set[str] = set()
            for k in range(chunks):
                start = (len(cell) * k) // chunks
                end = (len(cell) * (k + 1)) // chunks
                slice_ = cell[start:end]
                want = quota + (1 if k < extra else 0)
                for card in rng.sample(slice_, min(want, len(slice_))):
                    chosen.append(card)
                    taken.add(card["card_id"])
            # 분위가 짧아 못 채웠으면 남은 데서 채운다.
            if len(chosen) < per_cell:
                rest = [c for c in cell if c["card_id"] not in taken]
                chosen.extend(rng.sample(rest, min(per_cell - len(chosen), len(rest))))
            picked.extend(sorted(chosen, key=lambda c: c["card_id"]))
    return picked


def fill_forward(values: list[float | None], max_days: int = CARRY_FORWARD_DAYS) -> list[float | None]:
    """결측을 마지막 값으로 이월한다. max_days 를 넘으면 결측으로 둔다.

    영원히 이월하면 상장폐지된 종목을 계속 들고 있는 지수가 된다. 그게 우리가
    깐 생존편향의 반대편 오류다.
    """
    out: list[float | None] = []
    last: float | None = None
    run = 0
    for value in values:
        if value is not None:
            out.append(value)
            last = value
            run = 0
            continue
        run += 1
        out.append(last if (last is not None and run <= max_days) else None)
    return out


def _cell_relative(card_ids: list[str], day_prices: dict, base_prices: dict) -> float | None:
    """칸의 평균 상대가격. 쓸 수 있는 카드가 없으면 None."""
    rels = []
    for cid in card_ids:
        now = day_prices.get(cid)
        base = base_prices.get(cid)
        if now is not None and base:
            rels.append(now / base)
    return sum(rels) / len(rels) if rels else None


def index_point(universe: list[dict], day_prices: dict, base_prices: dict) -> dict:
    """하루치 지수와 하위지수를 낸다. 기준일 = 100.

    12칸을 균등가중한다. 시장 규모 비례가 이론적으로 낫지만 포켓몬 카드는
    유통 물량이 공개되지 않아 아무도 시장 규모를 모른다. 모르는 걸 아는 척
    가중치에 넣으면 우리가 깐 지수와 같은 짓이 된다.
    """
    cells: dict[tuple[str, str], list[str]] = {}
    for card in universe:
        cells.setdefault((card["era"], card["band"]), []).append(card["card_id"])

    rel: dict[tuple[str, str], float] = {}
    for key, ids in cells.items():
        value = _cell_relative(ids, day_prices, base_prices)
        if value is not None:
            rel[key] = value

    def mean_of(keys) -> float | None:
        vals = [rel[k] for k in keys if k in rel]
        return 100.0 * sum(vals) / len(vals) if vals else None

    return {
        "index": mean_of(list(rel)),
        "by_era": {e: mean_of([k for k in rel if k[0] == e]) for e in ERAS},
        "by_band": {b: mean_of([k for k in rel if k[1] == b]) for b in BANDS},
        "missing": sum(
            1 for c in universe
            if day_prices.get(c["card_id"]) is None or not base_prices.get(c["card_id"])
        ),
    }
