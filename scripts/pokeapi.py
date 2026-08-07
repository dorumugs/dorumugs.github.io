"""PokéAPI 종 이름 파싱 — 도감번호를 한글 이름으로.

    https://pokeapi.co/api/v2/pokemon-species/{id}/

TCGdex 의 한국어 데이터가 사실상 비어 있어(95세트 중 3세트, 7,646장 중 235장,
2026-08-07 실측) 카드 한글명을 거기서 가져올 수 없다. 대신 카드마다 붙은
dexId 로 포켓몬 *종* 이름을 한국어로 받아 붙인다.

한계가 있다. 이건 종 이름이지 카드 정식 한글명이 아니다. 트레이너·에너지
카드는 도감번호가 없어 한글명이 없다. 화면에 그렇게 적는다.

User-Agent 를 안 보내면 403 이다.
"""

from __future__ import annotations

BASE = "https://pokeapi.co/api/v2"

# 2026-08-07 기준 전국도감 등재 종 수. 넘겨 부르면 404 라 상한으로 쓴다.
SPECIES_MAX = 1025


class ApiError(Exception):
    """응답이 기대한 형태가 아닐 때."""


def parse_species(payload: dict) -> tuple[str, str] | None:
    """종 응답에서 (도감번호, 한글 이름). 한글 이름이 없으면 None."""
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ApiError("종 응답에 id 가 없습니다")
    for entry in payload.get("names") or []:
        if (entry.get("language") or {}).get("name") == "ko":
            name = (entry.get("name") or "").strip()
            if name:
                return str(payload["id"]), name
    return None


def korean_name(dex_id: str, names: dict[str, str]) -> str:
    """도감번호로 한글 이름을 찾는다. 없으면 빈 문자열."""
    return names.get(str(dex_id), "") if dex_id else ""
