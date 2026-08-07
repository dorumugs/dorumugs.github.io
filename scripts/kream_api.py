"""KREAM 포켓몬 카드 시세표 응답 파싱 — 순수 함수만.

응답은 두 덩어리다.

  market   하루 한 줄. 그날 거래된 카드 전체의 중앙값·건수·저가·고가·거래대금.
  product  상품 한 줄. 최근 중앙값·변동률·30일 고저·31일 가격 이력.

주의할 점이 몇 가지 있다.

  변동률   숫자가 아니라 문자열이고 지수표기다. '-1.2E0' 은 -1.2% 다.
           값이 없으면 '-' 로 온다 — 0% 가 아니라 '모른다' 는 뜻이다.
  이름     한글·영문 모두 '포켓몬 TCG ' / 'Pokemon TCG ' 로 시작한다. 899종
           전부 그래서 떼어낸다. 끝의 '(일어판)' 은 언어로 따로 뽑는다.
  기준     이 값들은 PSA 10 등급 · 대부분 일본판이다. TCGdex 쪽 raw 영문판
           시세와 같은 축에 놓으면 안 된다. 섞지 않는 건 화면의 몫이지만,
           여기서 언어만은 뽑아 둔다.

I/O 는 없다. 네트워크는 collect_kream.py 가 한다.
"""

from __future__ import annotations

import math
import re

IMAGE_PREFIX = "https://kream-phinf.pstatic.net/"

# 응답이 주는 주소는 원본이라 한 장에 845KB(1120px PNG)다. 그리드에 48장이면
# 40MB 라 그대로 쓰면 안 된다. 네이버 CDN 의 리사이즈 파라미터로 줄인다
# (2026-08-08 실측: s_webp 180px 7KB · m_webp 525px 32KB · l_webp 1120px 76KB).
IMAGE_THUMB_PARAM = "?type=m_webp"
IMAGE_FULL_PARAM = "?type=l_webp"

NAME_PREFIX_KO = "포켓몬 TCG "
NAME_PREFIX_EN = "Pokemon TCG "

# 이름 끝의 '(Japanese Ver.)' 를 언어로 바꾼다.
LANGUAGES = {
    "Japanese": "일어판",
    "Korean": "한글판",
    "English": "영문판",
}

MARKET_COLUMNS = [
    "date", "median", "count", "low", "high", "avg", "total",
]

PRODUCT_COLUMNS = [
    "name_ko", "name_en", "code", "lang", "price",
    "change_1d", "change_7d", "change_30d",
    "tx", "high_30d", "low_30d", "image", "history",
]


def parse_pct(value) -> float | None:
    """'-1.2E0' -> -1.2 · '-' -> None.

    '-' 를 0 으로 바꾸면 안 된다. '안 움직였다' 가 아니라 '거래가 없어 모른다'
    는 뜻이라, 0% 로 적으면 없는 사실을 지어내는 게 된다.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def parse_price(value) -> int | None:
    """원 단위 정수. 소수점은 의미가 없다.

    파이썬 기본 round() 는 은행가 반올림이라 154812.5 를 154812 로 내린다.
    1원 차이지만 읽는 사람이 손으로 계산한 값과 어긋나므로 0.5 는 올린다.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return math.floor(number + 0.5) if number >= 0 else math.ceil(number - 0.5)


def strip_prefix(text: str, prefix: str) -> str:
    text = (text or "").strip()
    return text[len(prefix):].strip() if text.startswith(prefix) else text


def split_language(name_en: str) -> tuple[str, str]:
    """('… Blue Sky Stream (Japanese Ver.)') -> ('… Blue Sky Stream', '일어판')."""
    match = re.search(r"\(([A-Za-z]+) Ver\.\)\s*$", name_en or "")
    if not match:
        return (name_en or "").strip(), ""
    trimmed = (name_en[:match.start()]).strip()
    return trimmed, LANGUAGES.get(match.group(1), match.group(1))


def short_image(url: str) -> str:
    """호스트를 떼어 짧게. 899장이면 호스트만 29KB 다."""
    return strip_prefix(url or "", IMAGE_PREFIX)


def parse_market(payload: dict) -> list[list]:
    """날짜 오름차순으로 정렬한 시장 일별 행. MARKET_COLUMNS 순서."""
    rows = ((payload or {}).get("market") or {}).get("rows") or []
    out = []
    for row in rows:
        date = (row.get("trade_date") or "").strip()
        if not date:
            continue
        out.append([
            date,
            parse_price(row.get("median_price")),
            int(row.get("transaction_count") or 0),
            parse_price(row.get("low_price")),
            parse_price(row.get("high_price")),
            parse_price(row.get("avg_price")),
            parse_price(row.get("transaction_total")),
        ])
    out.sort(key=lambda r: r[0])
    return out


def history_dates(payload: dict) -> list[str]:
    """모든 상품이 같은 날짜축을 쓴다(실측). 축을 한 번만 저장한다."""
    for row in ((payload or {}).get("product") or {}).get("rows") or []:
        history = row.get("price_history") or []
        if history:
            return [point.get("date", "") for point in history]
    return []


def parse_product(row: dict, dates: list[str]) -> list:
    """상품 한 줄을 PRODUCT_COLUMNS 순서의 배열로."""
    name_en, lang = split_language(row.get("name") or "")
    history_by_date = {
        point.get("date"): parse_price(point.get("price"))
        for point in (row.get("price_history") or [])
    }
    return [
        strip_prefix(row.get("translated_name") or "", NAME_PREFIX_KO),
        strip_prefix(name_en, NAME_PREFIX_EN),
        (row.get("style_code") or "").strip(),
        lang,
        parse_price(row.get("latest_median_price")),
        parse_pct(row.get("change_1d_pct")),
        parse_pct(row.get("change_7d_pct")),
        parse_pct(row.get("change_30d_pct")),
        int(row.get("transaction_count_total") or 0),
        parse_price(row.get("high_price_30d")),
        parse_price(row.get("low_price_30d")),
        short_image(row.get("image_url") or ""),
        [history_by_date.get(date) for date in dates],
    ]


def parse_products(payload: dict, dates: list[str]) -> list[list]:
    """현재가 높은 순. 화면 기본 정렬과 같게 미리 맞춰 둔다."""
    rows = ((payload or {}).get("product") or {}).get("rows") or []
    out = [parse_product(row, dates) for row in rows]
    price_at = PRODUCT_COLUMNS.index("price")
    out.sort(key=lambda r: -(r[price_at] or 0))
    return out


def thin_count(products: list[list], threshold: int = 1) -> int:
    """30일 거래가 threshold 건 이하인 상품 수. 화면에 그대로 적는다."""
    tx_at = PRODUCT_COLUMNS.index("tx")
    return sum(1 for row in products if (row[tx_at] or 0) <= threshold)


def language_counts(products: list[list]) -> dict:
    lang_at = PRODUCT_COLUMNS.index("lang")
    counts: dict = {}
    for row in products:
        key = row[lang_at] or "기타"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def is_valid(payload: dict) -> bool:
    """수집기가 받은 게 진짜 시세 응답인지. 빈 껍데기로 덮어쓰지 않으려는 것."""
    if not isinstance(payload, dict):
        return False
    market = parse_market(payload)
    products = ((payload or {}).get("product") or {}).get("rows") or []
    return bool(market) and len(products) >= 100
