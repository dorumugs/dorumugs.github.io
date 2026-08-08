"""Bulbagarden Archives(위키) 이미지 조회 — 순수 함수만.

TCGdex·TCGplayer·pokemontcg.io 셋 다 없는 카드가 39장 남았다. 전부 아주
오래되거나 비매품인 세트다 (My First Battle 34장, Poké Card Creator Pack 5장).
이런 건 상업 시세 사이트가 안 다루고 위키가 다룬다.

파일 이름 규칙 (실측으로 확인)

  My First Battle          `{이름}MyFirstBattle.jpg`     예) ArcanineMyFirstBattle.jpg
  Poké Card Creator Pack   `{이름}CreatorContest{번호}.png`  예) MudkipCreatorContest4.png

썸네일은 URL 을 손으로 만들면 404 다. MediaWiki 가 요청받을 때 생성하므로
API 에 `iiurlwidth` 를 줘서 `thumburl` 을 받아야 한다.

I/O 는 없다. 네트워크는 collect_card_art.py 가 한다.
"""

from __future__ import annotations

import urllib.parse

API = "https://archives.bulbagarden.net/w/api.php"

# 위키 규칙에 예의를 지킨다. 누가 왜 긁는지 밝히는 UA 를 쓴다.
USER_AGENT = "kayserdocs-pokemon/1.0 (https://dorumugs.github.io)"

# 카드 그리드가 245px 폭이라 그 크기로 받는다.
THUMB_WIDTH = 245

# 이 세트만 여기서 찾는다. 나머지는 앞선 세 출처가 이미 덮는다.
SET_PATTERNS = {
    "mfb": "{name}MyFirstBattle",
    "ex5.5": "{name}CreatorContest{local_id}",
}


def candidate_titles(set_id: str, name_en: str, local_id: str) -> list[str]:
    """이 카드에 시도해 볼 파일 제목들. 대응 세트가 아니면 빈 목록."""
    pattern = SET_PATTERNS.get(set_id)
    if not pattern or not name_en:
        return []
    stem = pattern.format(name=(name_en or "").replace(" ", ""), local_id=local_id)
    return [f"File:{stem}.jpg", f"File:{stem}.png"]


def query_url(titles: list[str], width: int = THUMB_WIDTH) -> str:
    """여러 제목을 한 번에 묻는다. 위키를 여러 번 두들길 이유가 없다."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(width),
        "titles": "|".join(titles),
    }
    return API + "?" + urllib.parse.urlencode(params)


def parse_images(payload: dict) -> dict:
    """응답에서 {파일 제목: (썸네일, 원본)} 만 추린다. 없는 파일은 빠진다."""
    pages = ((payload or {}).get("query") or {}).get("pages") or {}
    out = {}
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info or not info.get("url"):
            continue
        out[page.get("title", "")] = (info.get("thumburl") or info["url"], info["url"])
    return out


def pick(titles: list[str], found: dict) -> tuple[str, str] | None:
    """후보 제목 중 먼저 걸리는 것. 순서가 곧 우선순위다."""
    for title in titles:
        if title in found:
            return found[title]
    return None
