"""PokemonPriceTracker 응답 파싱 — 순수 함수만.

등급별 시세를 주는 유일한 정식 통로다. 다른 곳은 이렇다 (2026-08-08 실측).

  TCGdex·TCGplayer·tcgcsv   등급 축이 아예 없다. 인쇄 판본(홀로/논홀로)만 나눈다
  PSA 공식 낙찰가(APR)      카드 상세가 로그인 벽 뒤다
  PriceCharting             데이터는 제일 좋지만 API 가 유료 구독 전용이다

여기 값은 **eBay 실제 낙찰가**다. 호가가 아니라 팔린 값이라는 점이 중요하다.
대신 표본이 얇다 — 베이스셋 리자몽 PSA 10 이 최근 1년에 1건이다. 그래서
건수·최근 거래일·신뢰도를 값과 함께 들고 다닌다. 숫자만 크게 띄우면 안 된다.

I/O 는 없다. 네트워크는 collect_graded.py 가 한다.
"""

from __future__ import annotations

BASE_URL = "https://www.pokemonpricetracker.com/api/v2"

# 화면에 싣는 등급. 응답은 PSA·BGS·CGC·SGC·TAG·ACE 를 다 주지만, 국내(KREAM)
# 쪽이 PSA 10 기준이라 비교가 되려면 PSA 계열이어야 한다. 9 는 10 을 못 받았을
# 때의 현실적인 낙착지라 같이 싣는다.
GRADES = ("psa10", "psa9", "ungraded")

COLUMNS = [
    "card_id", "tcg_product_id",
    "psa10", "psa10_n", "psa10_date", "psa10_conf",
    "psa9", "psa9_n", "psa9_date", "psa9_conf",
    "ebay_raw", "ebay_raw_n",
    "updated",
]

# 카드 1장에 크레딧 2개(카드 1 + eBay 1). 무료 등급이 하루 100 이다.
CREDITS_PER_CARD = 2


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2)


def grade_block(card: dict, grade: str) -> dict:
    """`ebay.salesByGrade` 에서 등급 하나를 꺼낸다. 없으면 빈 dict."""
    sales = ((card or {}).get("ebay") or {}).get("salesByGrade") or {}
    block = sales.get(grade)
    return block if isinstance(block, dict) else {}


def price_of(block: dict):
    """대표가. 저쪽이 계산해 주는 smartMarketPrice 를 먼저 쓴다.

    중앙값만 쓰면 1년 전 한 건이 오늘 시세인 척한다. smartMarketPrice 는
    기간을 잘라 가중한 값이라 그보다 낫다. 없으면 중앙값으로 물러선다.
    """
    smart = block.get("smartMarketPrice") or {}
    return _num(smart.get("price")) if smart.get("price") is not None \
        else _num(block.get("medianPrice"))


def confidence_of(block: dict) -> str:
    return str((block.get("smartMarketPrice") or {}).get("confidence") or "")


def sale_date_of(block: dict) -> str:
    """마지막으로 팔린 날. 값이 얼마나 묵었는지 화면이 보여줘야 한다."""
    return str(block.get("lastSaleDate") or "")[:10]


def count_of(block: dict) -> int:
    try:
        return int(block.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def parse_card(card: dict, card_id: str, on_date: str) -> dict:
    """카드 하나를 COLUMNS 모양의 행으로."""
    psa10 = grade_block(card, "psa10")
    psa9 = grade_block(card, "psa9")
    raw = grade_block(card, "ungraded")
    return {
        "card_id": card_id,
        "tcg_product_id": str(card.get("tcgPlayerId") or ""),
        "psa10": price_of(psa10),
        "psa10_n": count_of(psa10),
        "psa10_date": sale_date_of(psa10),
        "psa10_conf": confidence_of(psa10),
        "psa9": price_of(psa9),
        "psa9_n": count_of(psa9),
        "psa9_date": sale_date_of(psa9),
        "psa9_conf": confidence_of(psa9),
        "ebay_raw": price_of(raw),
        "ebay_raw_n": count_of(raw),
        "updated": on_date,
    }


def has_any_grade(row: dict) -> bool:
    """PSA 값이 하나도 없으면 화면에 실을 이유가 없다."""
    return row.get("psa10") is not None or row.get("psa9") is not None


def premium(row: dict, raw_price) -> float | None:
    """raw 대비 PSA 10 이 몇 배인가. 등급이 무엇을 사는지 보여주는 숫자다."""
    psa10 = row.get("psa10")
    try:
        raw_price = float(raw_price)
    except (TypeError, ValueError):
        return None
    if psa10 is None or raw_price <= 0:
        return None
    return round(psa10 / raw_price, 1)


def cards_from_response(payload: dict) -> list[dict]:
    """`data` 가 배열일 때도 객체일 때도 있다. 배열로 통일한다."""
    data = (payload or {}).get("data")
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    return [data] if isinstance(data, dict) else []


def daily_budget(remaining_credits: int, reserve: int = 0) -> int:
    """남은 크레딧으로 몇 장을 볼 수 있나."""
    usable = max(0, int(remaining_credits) - int(reserve))
    return usable // CREDITS_PER_CARD
