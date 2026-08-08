"""KREAM 원화 시세를 화면용 JSON 으로 굽는다.

    python3 scripts/build_kream.py

산출물은 assets/pokemon/krw.json 하나다. 상품이 899종뿐이라 카드 브라우저처럼
사전 인코딩까지 갈 필요는 없지만, 배열로 담아 키 반복은 없앤다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_kream  # noqa: E402
import kream_api  # noqa: E402
import rtms  # noqa: E402

OUT_DIR = ROOT / "assets" / "pokemon"
OUT_FILE = OUT_DIR / "krw.json"
XREF_FILE = OUT_DIR / "xref.json"


def build(payload: dict, generated: str) -> dict:
    dates = kream_api.history_dates(payload)
    market = kream_api.parse_market(payload)
    products = kream_api.parse_products(payload, dates)

    price_at = kream_api.PRODUCT_COLUMNS.index("price")
    prices = sorted(row[price_at] for row in products if row[price_at])

    return {
        "generated": generated,
        "version": payload.get("version", ""),
        "image_prefix": kream_api.IMAGE_PREFIX,
        "image_thumb": kream_api.IMAGE_THUMB_PARAM,
        "image_full": kream_api.IMAGE_FULL_PARAM,
        "dates": dates,
        "market_columns": kream_api.MARKET_COLUMNS,
        "market": market,
        "columns": kream_api.PRODUCT_COLUMNS,
        "rows": products,
        "stats": {
            "product_count": len(products),
            "days": len(market),
            "first_date": market[0][0] if market else "",
            "last_date": market[-1][0] if market else "",
            "median_price": prices[len(prices) // 2] if prices else None,
            "max_price": prices[-1] if prices else None,
            "thin": kream_api.thin_count(products),
            "languages": kream_api.language_counts(products),
        },
    }


def build_xref(products: list[list], cards_payload: dict) -> dict:
    """두 시장을 **종(species) 단위**로 잇는다. 카드 단위로는 못 잇는다.

    왜 종 단위인가
    --------------
    국내 영문판 14종의 품번으로 카드 단위 결합을 시험했을 때 확실히 이어진 게
    1건이었다(2026-08-08 실측). 나머지는 일어판·한글판이라 영문판과 다른
    물건이다. 그래서 "이 카드가 저쪽에도 있다" 는 말은 할 수 없다.

    할 수 있는 말은 "이 **포켓몬**이 저쪽에 몇 개 있다" 다. 글로벌 카드에는
    도감번호로 붙인 한글 종 이름이 있고, 국내 상품명에서 그 종 이름이 899종 중
    801종(89%) 잡힌다.

    **참/거짓이 아니라 개수를 돌려준다.** "국내 있음" 이라고 적으면 사람은
    '같은 카드가 저 값에 거래된다' 로 읽는다. "국내 리자몽 46종" 이라고 적으면
    특정 카드가 아니라 종 얘기라는 게 숫자 자체로 드러난다.

    {종 이름: {"g": 글로벌 카드 수, "d": 국내 상품 수}} — 양쪽에 다 있는 종만.
    """
    ko_at = kream_api.PRODUCT_COLUMNS.index("name_ko")
    name_at = cards_payload["columns"].index("name_ko")

    global_count: dict = {}
    for row in cards_payload["rows"]:
        species = row[name_at]
        if species:
            global_count[species] = global_count.get(species, 0) + 1

    domestic_count: dict = {}
    for row in products:
        title = row[ko_at]
        # 가장 긴 이름부터 맞춘다. '리자몽' 과 '메가리자몽' 이 같이 걸리면
        # 긴 쪽이 실제 종이다.
        hits = [s for s in global_count if s and s in title]
        if hits:
            best = max(hits, key=len)
            domestic_count[best] = domestic_count.get(best, 0) + 1

    return {species: {"g": global_count[species], "d": count}
            for species, count in sorted(domestic_count.items())
            if global_count.get(species)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not collect_kream.KREAM_FILE.exists():
        print(f"{collect_kream.KREAM_FILE} 가 없습니다. 먼저 collect_kream.py 를 돌리세요.",
              file=sys.stderr)
        return 1

    payload = json.loads(rtms.gunzip_text(collect_kream.KREAM_FILE.read_bytes()))
    if not kream_api.is_valid(payload):
        print("저장된 응답이 시세 데이터가 아닙니다.", file=sys.stderr)
        return 1

    view = build(payload, args.generated)
    cards_payload = json.loads((OUT_DIR / "cards.json").read_text(encoding="utf-8")) \
        if (OUT_DIR / "cards.json").exists() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(view, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    if cards_payload:
        dates = kream_api.history_dates(payload)
        xref = build_xref(kream_api.parse_products(payload, dates), cards_payload)
        XREF_FILE.write_text(
            json.dumps(xref, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        print(f"두 시장에 다 있는 종 {len(xref):,}종 -> {XREF_FILE}")

    stats = view["stats"]
    packed = len(gzip.compress(OUT_FILE.read_bytes(), 9))
    print(f"상품 {stats['product_count']:,}종 · 시장 {stats['days']}일 "
          f"({stats['first_date']} ~ {stats['last_date']})")
    print(f"언어 {stats['languages']} · 30일 거래 1건 이하 {stats['thin']:,}종")
    print(f"krw.json 원본 {OUT_FILE.stat().st_size / 1024:.0f}KB · gzip {packed / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
