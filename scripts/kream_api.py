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


def base_code(style_code: str) -> str:
    """언어 접미사와 구분기호를 떼어 같은 카드끼리 묶는 열쇠.

    `S7R-083-067_JP` 와 `S7R083-067_KR` 은 같은 카드의 일어판·한글판이다.
    KREAM 이 하이픈을 일정하게 쓰지 않아 기호를 다 지우고 대문자로 맞춘다.

    이름으로 묶으면 안 된다 — "리자몽" 하나에 국내 45종·글로벌 76장이라
    엉킨다. 품번은 하드 키다.
    """
    text = re.sub(r"_(JP|KR|EN)$", "", (style_code or "").strip(), flags=re.I)
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def strip_language_suffix(name: str) -> str:
    """'레쿠쟈 VMAX HR 창공스트림 (일어판)' -> '레쿠쟈 VMAX HR 창공스트림'.

    언어가 열로 빠진 표에서는 이름 뒤의 '(일어판)' 이 거짓이 된다 — 그 줄은
    일어판과 한글판을 같이 들고 있기 때문이다.
    """
    return re.sub(r"\s*\((?:일어판|한글판|영문판|기타)\)\s*$", "", name or "").strip()


def merge_languages(products: list[list]) -> list[dict]:
    """같은 품번의 언어판을 한 줄로 묶는다.

    돌려주는 각 줄:
      {"key", "name_ko", "name_en", "by_lang": {"일어판": row, …}, "langs": [...]}

    899종이 859줄이 된다(40쌍이 접힌다). 접힌 쌍은 일어판이 한글판의 3~10배인
    경우가 많아, 따로 흩어 놓으면 안 보이던 사실이 드러난다.
    """
    code_at = PRODUCT_COLUMNS.index("code")
    lang_at = PRODUCT_COLUMNS.index("lang")
    name_at = PRODUCT_COLUMNS.index("name_ko")
    en_at = PRODUCT_COLUMNS.index("name_en")
    price_at = PRODUCT_COLUMNS.index("price")

    groups: dict[str, dict] = {}
    for row in products:
        key = base_code(row[code_at]) or f"~{row[code_at]}"
        group = groups.get(key)
        if group is None:
            group = groups[key] = {
                "key": key,
                "name_ko": strip_language_suffix(row[name_at]),
                "name_en": row[en_at],
                "by_lang": {}, "langs": [],
            }
        lang = row[lang_at] or "기타"
        # 같은 언어판이 두 번 나오면 비싼 쪽을 남긴다. 값이 붙은 쪽이 본품이다.
        kept = group["by_lang"].get(lang)
        if kept is None or (row[price_at] or 0) > (kept[price_at] or 0):
            group["by_lang"][lang] = row
        if lang not in group["langs"]:
            group["langs"].append(lang)

    out = list(groups.values())
    out.sort(key=lambda g: -max(
        (r[price_at] or 0) for r in g["by_lang"].values()))
    return out


def language_ratio(group: dict) -> float | None:
    """한 줄 안에서 가장 비싼 언어판이 가장 싼 것의 몇 배인가.

    같은 카드의 일어판이 한글판의 10배인 일이 흔하다. 그 숫자가 이 표의
    존재 이유다. 언어판이 하나뿐이면 비교할 것이 없으므로 None.
    """
    price_at = PRODUCT_COLUMNS.index("price")
    prices = [r[price_at] for r in group.get("by_lang", {}).values() if r[price_at]]
    if len(prices) < 2:
        return None
    low = min(prices)
    return round(max(prices) / low, 1) if low else None


def is_valid(payload: dict) -> bool:
    """수집기가 받은 게 진짜 시세 응답인지. 빈 껍데기로 덮어쓰지 않으려는 것."""
    if not isinstance(payload, dict):
        return False
    market = parse_market(payload)
    products = ((payload or {}).get("product") or {}).get("rows") or []
    return bool(market) and len(products) >= 100
