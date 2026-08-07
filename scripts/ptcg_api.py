"""pokemontcg.io 이미지 CDN 파싱·경로 순수 함수.

TCGdex 가 이미지를 안 주고 TCGplayer productId 도 없는 카드가 남는다. 세트
통째로 빠지는 경우가 대부분이라(Shiny Vault, Galarian Gallery, Trainer
Gallery) 눈에 잘 띈다. pokemontcg.io 가 그 세트들을 가지고 있어 마지막
대체 출처로 쓴다.

I/O 는 없다. 네트워크는 collect_card_art.py 가 한다.
"""

from __future__ import annotations

import re

IMAGE_PREFIX = "https://images.pokemontcg.io/"

# 우리(TCGdex) 세트 id -> pokemontcg.io 세트 id.
# 세트 이름을 맞춰 뽑고 실제 이미지를 두들겨 확인한 것만 남긴다.
# My First Battle(mfb)·Poké Card Creator Pack(ex5.5) 은 저쪽에도 없어서 뺐다.
SET_MAP = {
    "swsh4.5sv": "swsh45sv",       # Shining Fates Shiny Vault
    "swsh12.5gg": "swsh12pt5gg",   # Crown Zenith Galarian Gallery
    "swsh9.5tg": "swsh9tg",        # Brilliant Stars Trainer Gallery
    "swsh10.5tg": "swsh10tg",      # Astral Radiance Trainer Gallery
    "swsh11.5tg": "swsh11tg",      # Lost Origin Trainer Gallery
    "swsh12.5tg": "swsh12tg",      # Silver Tempest Trainer Gallery
    "ecard3": "ecard3",            # Skyridge
    "sm3.5": "sm35",               # Shining Legends
    "sm7.5": "sm75",               # Dragon Majesty
    "xy8": "xy8",                  # BREAKthrough
}


def number_variants(local_id: str) -> list[str]:
    """저쪽이 쓸 법한 번호 표기를 우선순위대로.

    대부분 우리 번호를 그대로 쓴다. 다만 Skyridge 는 우리가 'H09' 로 적는 걸
    저쪽은 'H9' 로 적는다. 접두 글자를 남기고 숫자의 앞 0 만 떼어 본다.
    """
    local_id = (local_id or "").strip()
    if not local_id:
        return []
    out = [local_id]
    m = re.fullmatch(r"([A-Za-z]*)0*(\d+)([A-Za-z]*)", local_id)
    if m:
        stripped = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        if stripped != local_id:
            out.append(stripped)
    return out


def image_path(ptcg_set: str, number: str) -> str:
    """화면에 저장하는 짧은 경로. 'swsh45sv/SV001' 꼴."""
    return f"{ptcg_set}/{number}"


def thumb_url(path: str) -> str:
    return f"{IMAGE_PREFIX}{path}.png"


def full_url(path: str) -> str:
    return f"{IMAGE_PREFIX}{path}_hires.png"


def candidate_paths(set_id: str, local_id: str) -> list[str]:
    """이 카드에 시도해 볼 경로들. 대응 세트가 없으면 빈 목록."""
    ptcg_set = SET_MAP.get(set_id)
    if not ptcg_set:
        return []
    return [image_path(ptcg_set, n) for n in number_variants(local_id)]


def needs_art(image: str, tp_product_id: str) -> bool:
    """TCGdex 경로도 없고 TCGplayer 사진도 없는 카드인가."""
    parts = (image or "").split("/")
    has_tcgdex = len(parts) == 3 and all(parts)
    return not has_tcgdex and not (tp_product_id or "").strip()
