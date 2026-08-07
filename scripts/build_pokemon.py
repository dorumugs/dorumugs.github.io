"""포켓몬 카드 현재가를 화면용 JSON 으로 굽는다.

    python3 scripts/build_pokemon.py

카드가 1만 7천 장이라 객체로 쓰면 같은 키가 그만큼 반복된다. 배열 한 줄에
값만 담고 컬럼 순서를 따로 알려준다 — 용량이 절반 아래로 떨어진다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import pokeapi  # noqa: E402
import tcgdex_api  # noqa: E402

OUT_DIR = ROOT / "assets" / "pokemon"

# cards.json 의 배열 컬럼 순서. app.js 가 같은 순서로 읽는다.
VIEW_COLUMNS = [
    "card_id", "set_id", "local_id", "name_en", "name_ko",
    "rarity", "image", "price", "high_ask", "obs_max", "obs_max_date", "cm_avg",
]


def _round(value, digits=2):
    return round(value, digits) if isinstance(value, (int, float)) else None


def view_row(card: dict, species: dict) -> list:
    """카드 한 장을 화면용 배열로. VIEW_COLUMNS 순서."""
    return [
        card["card_id"],
        card["set_id"],
        card["local_id"],
        card["name_en"],
        pokeapi.korean_name(card.get("dex_id", ""), species),
        card.get("rarity", ""),
        card.get("image", ""),
        _round(card.get("tp_market") or card.get("cm_avg")),
        _round(card.get("tp_high")),
        _round(card.get("obs_max")),
        card.get("obs_max_date", "") or "",
        _round(card.get("cm_avg")),
    ]


def build_cards(cards: list[dict], set_meta: dict, species: dict) -> list[list]:
    """포함된 세트의 카드만, 현재가 높은 순으로."""
    included = {sid for sid, m in set_meta.items() if m.get("included")}
    rows = [view_row(c, species) for c in cards if c["set_id"] in included]
    price_at = VIEW_COLUMNS.index("price")
    rows.sort(key=lambda r: (-(r[price_at] or 0), r[0]))
    return rows


def build_sets(set_meta: dict, cards: list[list]) -> dict:
    """화면 필터용 세트 정보. 실제로 카드가 남은 세트만."""
    set_at = VIEW_COLUMNS.index("set_id")
    counts: dict[str, int] = {}
    for row in cards:
        counts[row[set_at]] = counts.get(row[set_at], 0) + 1
    out = {}
    for sid, meta in set_meta.items():
        if sid in counts:
            out[sid] = {
                "name": meta.get("name", ""),
                "release_date": meta.get("release_date", ""),
                "era": meta.get("era") or "",
                "count": counts[sid],
            }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["release_date"], reverse=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not collect_pokemon.CARDS_FILE.exists():
        print(f"{collect_pokemon.CARDS_FILE} 가 없습니다. 먼저 collect_pokemon.py 를 돌리세요.",
              file=sys.stderr)
        return 1

    cards_raw = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    set_meta = collect_pokemon._read_json(collect_pokemon.SETS_FILE, {})
    species = collect_pokemon._read_json(collect_pokemon.SPECIES_FILE, {})

    cards = build_cards(cards_raw, set_meta, species)
    sets = build_sets(set_meta, cards)

    ko_at = VIEW_COLUMNS.index("name_ko")
    with_ko = sum(1 for r in cards if r[ko_at])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cards.json").write_text(json.dumps(
        {"columns": VIEW_COLUMNS, "image_prefix": tcgdex_api.IMAGE_PREFIX, "rows": cards},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "generated": args.generated,
        "card_count": len(cards),
        "set_count": len(sets),
        "with_korean_name": with_ko,
        "species_count": len(species),
        "sets_excluded": sum(1 for m in set_meta.values() if not m.get("included")),
    }, ensure_ascii=False), encoding="utf-8")

    size = (OUT_DIR / "cards.json").stat().st_size
    print(f"카드 {len(cards):,}장 · 세트 {len(sets)}개 · 한글명 {with_ko:,}장 "
          f"· cards.json {size/1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
