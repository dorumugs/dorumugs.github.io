"""TCGdex·TCGplayer 둘 다 사진을 못 주는 카드를 pokemontcg.io 로 메운다.

    python3 scripts/collect_card_art.py

경로 규칙을 화면에 심어 두면 없는 이미지까지 요청해 404 를 흘린다. 그래서
여기서 **실제로 존재하는 것만 확인해** data/pokemon/art.json 에 적는다.
화면은 이 표에 있는 카드만 그린다.

이미 확인된 카드는 다시 두들기지 않는다. 새 세트가 들어와야 요청이 생긴다.

    python3 scripts/collect_card_art.py --verify-tcgdex

**TCGdex 는 경로를 주면서 파일이 없는 카드가 있다.** e-Card 시절 Aquapolis·
Skyridge 의 H 번호 홀로들이 그렇고, 하필 블래키·후딘·마기라스 같은 인기 카드가
거기 몰려 있다. "경로가 없는 카드"만 세면 이걸 영영 못 잡는다 — 화면에서만
조용히 '이미지 없음' 이 된다.

`--verify-tcgdex` 는 TCGdex 경로 전부(약 17,000장)를 두들겨 깨진 것을 찾아
대체 출처로 메운다. 느리므로 매일 돌리지 않고 가끔 돌린다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bulba_api  # noqa: E402
import collect_pokemon  # noqa: E402
import ptcg_api  # noqa: E402

ART_FILE = ROOT / "data" / "pokemon" / "art.json"
# 경로는 있는데 파일이 없는 TCGdex 카드. 수집기는 매일 원본을 다시 받아 경로를
# 되살리므로, 여기에 따로 적어 두고 집계 때 그 경로를 무시한다.
BROKEN_FILE = ROOT / "data" / "pokemon" / "art_broken.json"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/149.0.0.0 Safari/537.36")
TCGPLAYER_PREFIX = "https://tcgplayer-cdn.tcgplayer.com/product/"


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


def resolve(card: dict):
    """이 카드에 쓸 수 있는 첫 경로. 없으면 None.

    돌려주는 값은 화면이 그대로 쓸 수 있는 모양이다 —
      pokemontcg.io  ("card_id", {"src": "ptcg", "path": "swsh45sv/SV001"})
    """
    for path in ptcg_api.candidate_paths(card["set_id"], card["local_id"]):
        if exists(ptcg_api.thumb_url(path)):
            return card["card_id"], {"src": "ptcg", "path": path}
    return None


def resolve_wiki(cards: list[dict]) -> dict:
    """Bulbagarden Archives 에서 한 번에 찾는다.

    앞선 세 출처가 못 덮는 아주 오래되거나 비매품인 세트용이다. 상업 시세
    사이트는 이런 카드를 안 다루고 위키가 다룬다.

    썸네일 URL 은 손으로 만들면 404 다 — MediaWiki 가 요청받을 때 생성하므로
    API 로 물어봐야 한다."""
    wanted = {}
    for card in cards:
        titles = bulba_api.candidate_titles(
            card["set_id"], card.get("name_en", ""), card["local_id"])
        if titles:
            wanted[card["card_id"]] = titles
    if not wanted:
        return {}

    every = [t for titles in wanted.values() for t in titles]
    found = {}
    for start in range(0, len(every), 40):   # 위키 한 번에 최대 50개
        url = bulba_api.query_url(every[start:start + 40])
        request = urllib.request.Request(
            url, headers={"User-Agent": bulba_api.USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                found.update(bulba_api.parse_images(json.load(response)))
        except Exception as err:
            print(f"  위키 조회 실패: {err}", file=sys.stderr)
        time.sleep(0.6)

    out = {}
    for card_id, titles in wanted.items():
        hit = bulba_api.pick(titles, found)
        if hit:
            out[card_id] = {"src": "bulba", "thumb": hit[0], "full": hit[1]}
    return out


def broken_tcgdex(cards: list[dict], workers: int) -> list[dict]:
    """TCGdex 경로가 있는데 실제로는 404 인 카드를 찾는다.

    경로가 있다고 이미지가 있는 게 아니다. 전수로 두들겨 봐야 안다."""
    targets = [c for c in cards if ptcg_api.tcgdex_thumb_url(c.get("image", ""))]
    print(f"TCGdex 경로 {len(targets):,}장을 확인합니다 (몇 분 걸립니다)")

    def alive(card):
        return card, exists(ptcg_api.tcgdex_thumb_url(card["image"]))

    bad = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (card, ok) in enumerate(pool.map(alive, targets), 1):
            if not ok:
                bad.append(card)
            if index % 2000 == 0:
                print(f"  {index:,}/{len(targets):,} · 깨진 것 {len(bad)}")
    print(f"깨진 TCGdex 경로 {len(bad):,}장")
    return bad


def gaps(cards: list[dict], known: dict, broken: set) -> list[dict]:
    """아직 사진이 없고 아직 확인해 보지 않은 카드 (pokemontcg.io 대상).

    경로가 깨진 카드는 TCGplayer productId 가 있어도 대상이다 — 화면이 경로를
    **세트 단위**로 만들기 때문에, 카드 하나만 깨졌어도 확인된 주소를 따로
    쥐여 주지 않으면 그 카드는 계속 깨진 경로를 쓴다."""
    return [c for c in cards
            if c["card_id"] not in known
            and (ptcg_api.needs_art(c.get("image", ""), c.get("tp_product_id", ""))
                 or c["card_id"] in broken)
            and ptcg_api.SET_MAP.get(c["set_id"])]


def wiki_gaps(cards: list[dict], known: dict) -> list[dict]:
    """위 셋으로도 안 되고 아직 위키를 안 뒤져 본 카드."""
    return [c for c in cards
            if c["card_id"] not in known
            and ptcg_api.needs_art(c.get("image", ""), c.get("tp_product_id", ""))
            and bulba_api.SET_PATTERNS.get(c["set_id"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8,
                        help="동시 요청 수. CDN 이지만 예의상 낮게 둔다")
    parser.add_argument("--recheck", action="store_true",
                        help="이미 확인된 카드도 다시 두들긴다")
    parser.add_argument("--verify-tcgdex", action="store_true",
                        help="TCGdex 경로 전부를 두들겨 깨진 것을 찾아 메운다 (느림)")
    args = parser.parse_args()

    if not collect_pokemon.CARDS_FILE.exists():
        print(f"{collect_pokemon.CARDS_FILE} 가 없습니다. collect_pokemon.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1

    cards = collect_pokemon.csv_to_rows(
        collect_pokemon._read_gz(collect_pokemon.CARDS_FILE))
    known: dict = {} if args.recheck else collect_pokemon._read_json(ART_FILE, {})

    # 경로가 있어도 파일이 없는 카드를 먼저 골라낸다. 이걸 안 하면 화면에서만
    # 조용히 '이미지 없음' 이 되고 집계에는 안 잡힌다.
    broken_ids = set(collect_pokemon._read_json(BROKEN_FILE, []))
    if args.verify_tcgdex:
        broken_ids = {c["card_id"] for c in broken_tcgdex(cards, args.workers)}
        BROKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        collect_pokemon._write_json(BROKEN_FILE, sorted(broken_ids))
    for card in cards:
        if card["card_id"] in broken_ids:
            card["image"] = ""   # 아래 단계들이 '사진 없는 카드' 로 보게 한다
            known.pop(card["card_id"], None)

    todo = gaps(cards, known, broken_ids)
    print(f"pokemontcg.io 에서 확인할 대상 {len(todo):,}장 (이미 확인 {len(known):,}장)")

    found = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(resolve, todo):
                if result:
                    known[result[0]] = result[1]
                    found += 1

    # pokemontcg.io 에도 없지만 TCGplayer 사진은 있는 경우. 깨진 경로를 그대로
    # 두느니 제품 사진이라도 쥐여 준다.
    for card in cards:
        if (card["card_id"] in broken_ids and card["card_id"] not in known
                and (card.get("tp_product_id") or "").strip()):
            pid = card["tp_product_id"].strip()
            known[card["card_id"]] = {
                "src": "tcgplayer",
                "thumb": f"{TCGPLAYER_PREFIX}{pid}_200w.jpg",
                "full": f"{TCGPLAYER_PREFIX}{pid}_in_1000x1000.jpg",
            }
            found += 1

    # 마지막 그물. 상업 사이트가 안 다루는 옛 세트를 위키에서 찾는다.
    wiki_todo = wiki_gaps(cards, known)
    print(f"위키에서 확인할 대상 {len(wiki_todo):,}장")
    if wiki_todo:
        wiki = resolve_wiki(wiki_todo)
        known.update(wiki)
        found += len(wiki)

    if not todo and not wiki_todo:
        print("새로 확인할 것이 없습니다.")
        return 0

    ART_FILE.parent.mkdir(parents=True, exist_ok=True)
    ART_FILE.write_text(
        json.dumps(dict(sorted(known.items())), ensure_ascii=False,
                   separators=(",", ":")),
        encoding="utf-8")

    missed = (len(todo) + len(wiki_todo)) - found
    print(f"찾음 {found:,}장 · 저쪽에도 없음 {missed:,}장 -> {ART_FILE}")

    still = [c for c in cards
             if c["card_id"] not in known
             and ptcg_api.needs_art(c.get("image", ""), c.get("tp_product_id", ""))]
    print(f"세 출처를 다 뒤져도 사진이 없는 카드: {len(still):,}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
