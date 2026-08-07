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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(view, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    stats = view["stats"]
    packed = len(gzip.compress(OUT_FILE.read_bytes(), 9))
    print(f"상품 {stats['product_count']:,}종 · 시장 {stats['days']}일 "
          f"({stats['first_date']} ~ {stats['last_date']})")
    print(f"언어 {stats['languages']} · 30일 거래 1건 이하 {stats['thin']:,}종")
    print(f"krw.json 원본 {OUT_FILE.stat().st_size / 1024:.0f}KB · gzip {packed / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
