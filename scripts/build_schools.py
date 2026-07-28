"""학교 원본을 지도에 올릴 수 있는 집계 JSON 으로 굽는다.

    python3 scripts/build_schools.py

사립 초등학교·중학교를 함께 다룬다. 진학 실적(특목고·자사고 비율) 기반으로
중학교를 상위권만 거르는 안은 그 데이터가 학교알리미 OpenAPI 로도, 공개용데이터
목록에도, 학교별 공시 화면에도 없어 접었다 — 대신 사립초와 같은 논리를 그대로
써서 사립 중학교 전체를 낸다: 배정이 아니라 지원으로 가는 학교라 '근처'가
실제로 의미를 가진다.
좌표 변환은 build_geo.py 가 내보낸 projection.json 을 그대로 쓴다 —
파라미터를 각자 계산하면 지도와 점이 조용히 어긋난다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dashboard  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

SCHOOL_FILE = ROOT / "data" / "schools.csv.gz"
PROJECTION_FILE = ROOT / "data" / "geo" / "projection.json"
OUT_FILE = ROOT / "assets" / "realestate" / "schools.json"

MAX_BYTES = 100 * 1024
JOIN_FAIL_LIMIT = 0.10

# 학교급 표기를 화면용 한 글자로 줄인다.
LEVEL_SHORT = {"초등학교": "초", "중학교": "중"}


@lru_cache(maxsize=1)
def _sgg_by_name() -> list[tuple[str, str]]:
    """(정식이름, 코드) 를 이름 긴 순으로. 긴 이름이 먼저 맞아야 한다.

    '경기도 성남시' 가 '경기도 성남시 분당구' 보다 먼저 맞으면 구를 놓친다.
    """
    pairs = [(name, code) for code, name in regions.sgg_codes()]
    return sorted(pairs, key=lambda p: -len(p[0]))


# 지번 토큰: '1', '123-45', '산26-127', '산 26-127' 처럼 붙거나 띄어 쓴다.
_JIBUN_TOKEN = re.compile(r"^산?\d+(-\d+)?$")


def _is_jibun_token(token: str) -> bool:
    return token == "산" or bool(_JIBUN_TOKEN.fullmatch(token))


def parse_addr(addr: str) -> tuple[str, str] | None:
    """지번주소를 (시군구 정식이름, 읍면동 접미사) 로 나눈다. 못 찾으면 None.

    읍면 지역은 '가남읍 태평리' 처럼 두 토큰이라 접미사 전체를 넘긴다 —
    regions.dong_code 가 접미사 전체를 1차 키로 쓰기 때문이다.
    마지막 지번 토큰들('산26-127', '산 26-127' 등 붙거나 띄어 쓴 경우 모두)은 떼어낸다.
    """
    text = " ".join((addr or "").split())
    for name, _code in _sgg_by_name():
        if not text.startswith(name + " "):
            continue
        rest = text[len(name):].strip()
        tokens = rest.split(" ")
        while tokens and _is_jibun_token(tokens[-1]):
            tokens = tokens[:-1]
        if not tokens:
            return None
        return name, " ".join(tokens)
    return None


def to_svg_xy(lat: float, lon: float, params: dict) -> tuple[float, float]:
    """build_geo.py 의 project() 와 같은 식. 파라미터는 projection.json 에서 온다."""
    x = (lon - params["min_lon"]) * params["k"] / params["span_x"] * params["width"]
    y = (params["max_lat"] - lat) / params["span_y"] * params["height"]
    return x, y


def select_private_schools(rows: list[dict]) -> list[dict]:
    """대상: 사립 초등학교·중학교. 국립은 요구사항이 '사립' 이라 넣지 않는다.

    고등학교는 애초에 수집 대상이 아니다(schools_api.LEVELS). 공립·국립
    중학교도 여기서 걸러진다 — 진학 실적 기반 순위를 매길 데이터가 없어서
    '지원으로 가는 학교'라는, 사립초에 이미 쓰던 논리를 그대로 재사용한다.
    """
    return [r for r in rows
            if r.get("level") in LEVEL_SHORT and r.get("found_type") == "사립"]


def build(rows: list[dict], params: dict, generated: str) -> dict:
    """선택된 학교를 지도 좌표와 법정동 코드가 붙은 형태로 만든다."""
    name_to_code = dict(_sgg_by_name())
    out: list[dict] = []
    failed: list[str] = []

    for row in rows:
        parsed = parse_addr(row["addr"])
        if parsed is None:
            failed.append(row["school_name"])
            continue
        sgg_name, umd = parsed
        sgg = name_to_code[sgg_name]
        dong_cd = regions.dong_code(sgg, umd)
        if dong_cd is None:
            failed.append(row["school_name"])
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (TypeError, ValueError):
            failed.append(row["school_name"])
            continue
        x, y = to_svg_xy(lat, lon, params)
        out.append({
            "name": row["school_name"],
            "lvl": LEVEL_SHORT[row["level"]],
            "found": row["found_type"],
            "sgg": sgg,
            "dong": umd.split(" ")[-1],
            "dong_cd": dong_cd,
            "addr": row["addr"],
            "x": round(x, 1),
            "y": round(y, 1),
        })

    # 완전 전순서. 동점이 흔들리면 재빌드마다 바이트가 달라진다.
    out.sort(key=lambda s: (s["sgg"], s["name"], s["dong_cd"]))
    return {"generated": generated, "schools": out, "join_failed": len(failed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not PROJECTION_FILE.exists():
        print(f"{PROJECTION_FILE} 가 없습니다. 먼저 build_geo.py 를 돌리세요.", file=sys.stderr)
        return 1
    if not SCHOOL_FILE.exists():
        print(f"{SCHOOL_FILE} 가 없습니다. 먼저 collect_schools.py 를 돌리세요.", file=sys.stderr)
        return 1

    params = json.loads(PROJECTION_FILE.read_text(encoding="utf-8"))
    rows = rtms.csv_to_rows(rtms.gunzip_text(SCHOOL_FILE.read_bytes()))
    picked = select_private_schools(rows)
    print(f"원본 {len(rows):,}건 중 사립 초·중 {len(picked):,}건")

    payload = build(picked, params, args.generated)
    failed = payload.pop("join_failed")
    if picked and failed / len(picked) > JOIN_FAIL_LIMIT:
        print(f"법정동 조인 실패 {failed}/{len(picked)} — 한도 "
              f"{JOIN_FAIL_LIMIT:.0%} 초과", file=sys.stderr)
        return 1
    if failed:
        print(f"법정동 조인 실패 {failed}건 (한도 안)")

    changed = build_dashboard.write_json(OUT_FILE, payload)
    size = OUT_FILE.stat().st_size
    print(f"schools.json {size:,}B {'갱신' if changed else '변경 없음'} "
          f"/ 학교 {len(payload['schools']):,}개")
    if size > MAX_BYTES:
        print(f"예산({MAX_BYTES}B) 초과", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
