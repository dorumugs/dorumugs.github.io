"""PSA 10 통합 표를 굽는다.

    python3 scripts/build_unified.py

왜 PSA 10 만 담는가
-------------------
글로벌 표는 **raw**, 국내 표는 **PSA 10** 을 재고 있었다. 값의 종류가 달라서
한 표에 섞으면 가격순 정렬이 곧바로 거짓말이 된다 — raw $819 카드가 PSA 10
1,000만원짜리보다 위로 올라간다.

**PSA 10 으로 한정하면 그 문제가 사라진다.** 같은 등급, 같은 종류의 값이고
통화만 환산하면 된다. 그래서 이 표에는 PSA 10 만 넣는다. raw 브라우저는
그대로 따로 둔다.

무엇을 합치고 무엇을 안 합치는가
--------------------------------
합친다   국내 언어판. `S7R-083-067_JP` 와 `S7R083-067_KR` 은 같은 카드다.
         품번에서 언어 접미사를 떼는 **하드 키**라 틀릴 여지가 없다.
         899종이 859줄로 접힌다.

안 합친다 글로벌 ↔ 국내. 영문판 14종의 품번으로 시도해 봤더니 유일하게
         이어진 것이 1건이었다(2026-08-08 실측). 앞자리가 PTCGO 약칭이라
         우리 세트 id 와 체계가 다르다. 근거 없이 붙이면 "리자몽" 76장과
         45종이 엉킨다. 같은 표에 나란히 두되 줄은 따로 간다.
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

import build_pokemon  # noqa: E402
import collect_graded  # noqa: E402
import collect_kream  # noqa: E402
import collect_pokemon  # noqa: E402
import kream_api  # noqa: E402
import pokeapi  # noqa: E402
import ppt_api  # noqa: E402
import rtms  # noqa: E402

OUT_FILE = ROOT / "assets" / "pokemon" / "unified.json"

# 한 줄의 배열 순서. app 이 같은 순서로 읽는다.
COLUMNS = [
    "name",        # 화면에 크게 나올 이름
    "sub",         # 세트·품번
    "market",      # '글로벌' | '국내'
    "prices",      # {언어판: PSA10 금액}
    "unit",        # prices 의 통화 — 'USD' | 'KRW'
    "samples",     # {언어판: 표본 건수}
    "ratio",       # 한 줄 안 최고/최저 배수. 언어판이 하나면 null
]


def domestic_rows(payload: dict) -> list[list]:
    """국내 — 언어판을 묶어 한 줄로."""
    dates = kream_api.history_dates(payload)
    products = kream_api.parse_products(payload, dates)
    price_at = kream_api.PRODUCT_COLUMNS.index("price")
    code_at = kream_api.PRODUCT_COLUMNS.index("code")
    tx_at = kream_api.PRODUCT_COLUMNS.index("tx")

    out = []
    for group in kream_api.merge_languages(products):
        prices, samples = {}, {}
        for lang, row in group["by_lang"].items():
            if row[price_at]:
                prices[lang] = row[price_at]
                samples[lang] = row[tx_at]
        if not prices:
            continue
        first = next(iter(group["by_lang"].values()))
        out.append([
            group["name_ko"],
            "품번 " + kream_api.base_code(first[code_at]),
            "국내",
            prices,
            "KRW",
            samples,
            kream_api.language_ratio(group),
        ])
    return out


def global_rows(cards: list[dict], graded: list[dict], species: dict) -> list[list]:
    """글로벌 — PSA 10 값이 잡힌 카드만. raw 만 있는 카드는 이 표에 안 넣는다.

    이름에 한글 종 이름을 같이 담는다. 국내 줄은 한글뿐이라, 안 담으면
    "리자몽" 으로 찾았을 때 국내 줄만 나오고 글로벌 줄이 통째로 빠진다."""
    by_id = {c["card_id"]: c for c in cards}
    out = []
    for row in graded:
        if row.get("psa10") is None:
            continue
        card = by_id.get(row["card_id"])
        if not card:
            continue
        ko = pokeapi.korean_name(card.get("dex_id", ""), species)
        out.append([
            (card.get("name_en") or row["card_id"]) + (f" · {ko}" if ko else ""),
            f"{card.get('set_id', '')} · #{card.get('local_id', '')}",
            "글로벌",
            {"영문판": row["psa10"]},
            "USD",
            {"영문판": row.get("psa10_n") or 0},
            None,
        ])
    return out


def build(payload: dict, cards: list[dict], graded: list[dict],
          species: dict, generated: str) -> dict:
    rows = global_rows(cards, graded, species) + domestic_rows(payload)
    merged = sum(1 for r in rows if len(r[COLUMNS.index("prices")]) > 1)
    return {
        "generated": generated,
        "columns": COLUMNS,
        "rows": rows,
        "stats": {
            "total": len(rows),
            "global": sum(1 for r in rows if r[COLUMNS.index("market")] == "글로벌"),
            "domestic": sum(1 for r in rows if r[COLUMNS.index("market")] == "국내"),
            "merged": merged,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not collect_kream.KREAM_FILE.exists():
        print("국내 시세가 없습니다. collect_kream.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1

    payload = json.loads(rtms.gunzip_text(collect_kream.KREAM_FILE.read_bytes()))
    cards = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    graded = collect_graded.csv_to_rows(
        collect_pokemon._read_gz(collect_graded.GRADED_FILE))

    species = collect_pokemon._read_json(collect_pokemon.SPECIES_FILE, {})
    view = build(payload, cards, graded, species, args.generated)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(view, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    stats = view["stats"]
    packed = len(gzip.compress(OUT_FILE.read_bytes(), 9))
    print(f"통합 {stats['total']:,}줄 (글로벌 {stats['global']:,} · "
          f"국내 {stats['domestic']:,}) · 언어판이 둘 이상 붙은 줄 {stats['merged']}개")
    print(f"unified.json {OUT_FILE.stat().st_size / 1024:.0f}KB · gzip {packed / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
