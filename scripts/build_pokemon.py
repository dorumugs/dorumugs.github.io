"""포켓몬 카드 현재가를 화면용 JSON 으로 굽는다.

    python3 scripts/build_pokemon.py

카드가 1만 7천 장이라 그대로 쓰면 payload 가 2MB 다. 두 가지로 줄인다.

  파생   card_id 는 '세트-번호', 이미지 경로는 '시리즈/세트/번호' 로 100%
         규칙적이다 (17,683장 전수 확인). 저장하지 않고 화면에서 만든다.
  사전   세트 141·등급 28·날짜 2종뿐이다. 문자열을 반복하지 않고 번호로 넣는다.

기존 대시보드 중 가장 큰 payload 가 gzip 133KB 라 그 언저리를 목표로 한다.
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

import collect_pokemon  # noqa: E402
import pokeapi  # noqa: E402
import tcgdex_api  # noqa: E402

OUT_DIR = ROOT / "assets" / "pokemon"

# cards.json 의 배열 컬럼 순서. app.js 가 같은 순서로 읽는다.
# card_id 와 image 는 여기 없다 — 화면에서 만든다.
VIEW_COLUMNS = [
    "set", "local_id", "name_en", "name_ko", "rarity",
    "price", "high_ask", "obs_max", "obs_date", "cm_avg", "tcg_pid",
]

# TCGdex 가 이미지를 안 주는 카드용 대체 CDN. productId 로 제품 사진을 준다.
TCGPLAYER_IMAGE_PREFIX = "https://tcgplayer-cdn.tcgplayer.com/product/"


def _round(value, digits=2):
    return round(value, digits) if isinstance(value, (int, float)) else None


def serie_of(image: str) -> str:
    """이미지 경로 'base/base1/4' 의 첫 칸이 시리즈다. 화면이 경로를 되만들 때 쓴다."""
    parts = (image or "").split("/")
    return parts[0] if len(parts) == 3 else ""


class Dictionary:
    """반복되는 문자열을 번호로 바꾼다. 같은 값은 같은 번호를 받는다."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self._index: dict[str, int] = {}

    def index(self, value: str) -> int:
        value = value or ""
        if value not in self._index:
            self._index[value] = len(self.values)
            self.values.append(value)
        return self._index[value]


def build_payload(cards: list[dict], set_meta: dict, species: dict) -> dict:
    """포함된 세트의 카드만, 현재가 높은 순으로 사전 인코딩해 담는다."""
    included = {sid for sid, m in set_meta.items() if m.get("included")}

    sets = Dictionary()
    rarities = Dictionary()
    dates = Dictionary()
    series_by_set: dict[str, str] = {}

    rows = []
    for card in cards:
        sid = card["set_id"]
        if sid not in included:
            continue
        serie = serie_of(card.get("image", ""))
        if serie and sid not in series_by_set:
            series_by_set[sid] = serie
        price = _round(card.get("tp_market") or card.get("cm_avg"))
        rows.append([
            sets.index(sid),
            card["local_id"],
            card["name_en"],
            pokeapi.korean_name(card.get("dex_id", ""), species),
            rarities.index(card.get("rarity", "")),
            price,
            _round(card.get("tp_high")),
            _round(card.get("obs_max")),
            dates.index(card.get("obs_max_date", "")),
            _round(card.get("cm_avg")),
            # TCGdex 이미지가 있으면 대체 사진은 필요 없다. 없을 때만 담는다.
            (card.get("tp_product_id") or "") if not serie else "",
        ])

    price_at = VIEW_COLUMNS.index("price")
    rows.sort(key=lambda r: (-(r[price_at] or 0), r[0], r[1]))

    return {
        "columns": VIEW_COLUMNS,
        "image_prefix": tcgdex_api.IMAGE_PREFIX,
        "tcgplayer_image_prefix": TCGPLAYER_IMAGE_PREFIX,
        "sets": sets.values,
        "series": [series_by_set.get(sid, "") for sid in sets.values],
        "rarities": rarities.values,
        "dates": dates.values,
        "rows": rows,
    }


def build_sets(set_meta: dict, payload: dict) -> dict:
    """화면 필터용 세트 정보. 실제로 카드가 남은 세트만."""
    set_at = VIEW_COLUMNS.index("set")
    counts: dict[str, int] = {}
    for row in payload["rows"]:
        sid = payload["sets"][row[set_at]]
        counts[sid] = counts.get(sid, 0) + 1
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

    cards = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    set_meta = collect_pokemon._read_json(collect_pokemon.SETS_FILE, {})
    species = collect_pokemon._read_json(collect_pokemon.SPECIES_FILE, {})

    payload = build_payload(cards, set_meta, species)
    sets = build_sets(set_meta, payload)

    ko_at = VIEW_COLUMNS.index("name_ko")
    with_ko = sum(1 for r in payload["rows"] if r[ko_at])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cards.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "generated": args.generated,
        "card_count": len(payload["rows"]),
        "set_count": len(sets),
        "with_korean_name": with_ko,
        "species_count": len(species),
        "sets_excluded": sum(1 for m in set_meta.values() if not m.get("included")),
    }, ensure_ascii=False), encoding="utf-8")

    raw = (OUT_DIR / "cards.json").stat().st_size
    packed = len(gzip.compress((OUT_DIR / "cards.json").read_bytes(), 9))
    print(f"카드 {len(payload['rows']):,}장 · 세트 {len(sets)}개 · 한글명 {with_ko:,}장")
    print(f"cards.json 원본 {raw/1024:.0f}KB · gzip {packed/1024:.0f}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
