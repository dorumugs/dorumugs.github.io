"""TCGdex·TCGplayer 둘 다 사진을 못 주는 카드를 pokemontcg.io 로 메운다.

    python3 scripts/collect_card_art.py

경로 규칙을 화면에 심어 두면 없는 이미지까지 요청해 404 를 흘린다. 그래서
여기서 **실제로 존재하는 것만 확인해** data/pokemon/art.json 에 적는다.
화면은 이 표에 있는 카드만 그린다.

이미 확인된 카드는 다시 두들기지 않는다. 새 세트가 들어와야 요청이 생긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import ptcg_api  # noqa: E402

ART_FILE = ROOT / "data" / "pokemon" / "art.json"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/149.0.0.0 Safari/537.36")


def exists(url: str, timeout: int = 20) -> bool:
    """HEAD 로 존재만 확인한다. 이미지를 내려받지 않는다."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def resolve(card: dict) -> tuple[str, str] | None:
    """이 카드에 쓸 수 있는 첫 경로. 없으면 None."""
    for path in ptcg_api.candidate_paths(card["set_id"], card["local_id"]):
        if exists(ptcg_api.thumb_url(path)):
            return card["card_id"], path
    return None


def gaps(cards: list[dict], known: dict) -> list[dict]:
    """아직 사진이 없고 아직 확인해 보지 않은 카드."""
    return [c for c in cards
            if c["card_id"] not in known
            and ptcg_api.needs_art(c.get("image", ""), c.get("tp_product_id", ""))
            and ptcg_api.SET_MAP.get(c["set_id"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8,
                        help="동시 요청 수. CDN 이지만 예의상 낮게 둔다")
    parser.add_argument("--recheck", action="store_true",
                        help="이미 확인된 카드도 다시 두들긴다")
    args = parser.parse_args()

    if not collect_pokemon.CARDS_FILE.exists():
        print(f"{collect_pokemon.CARDS_FILE} 가 없습니다. collect_pokemon.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1

    cards = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    known: dict = {} if args.recheck else collect_pokemon._read_json(ART_FILE, {})

    todo = gaps(cards, known)
    print(f"사진 없는 카드 중 확인할 대상 {len(todo):,}장 (이미 확인 {len(known):,}장)")
    if not todo:
        print("새로 확인할 것이 없습니다.")
        return 0

    found = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(resolve, todo):
            if result:
                known[result[0]] = result[1]
                found += 1

    ART_FILE.parent.mkdir(parents=True, exist_ok=True)
    ART_FILE.write_text(
        json.dumps(dict(sorted(known.items())), ensure_ascii=False,
                   separators=(",", ":")),
        encoding="utf-8")

    missed = len(todo) - found
    print(f"찾음 {found:,}장 · 저쪽에도 없음 {missed:,}장 -> {ART_FILE}")

    still = [c for c in cards
             if c["card_id"] not in known
             and ptcg_api.needs_art(c.get("image", ""), c.get("tp_product_id", ""))]
    print(f"세 출처를 다 뒤져도 사진이 없는 카드: {len(still):,}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
