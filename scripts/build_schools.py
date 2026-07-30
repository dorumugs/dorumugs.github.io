"""학교 원본을 지도에 올릴 수 있는 집계 JSON 으로 굽는다.

    python3 scripts/build_schools.py

사립 초등학교·중학교와 특목고(과학·외국어·국제 계열)를 함께 다루고, 중학교
안에서 국제중을 따로 뗀다. 사립초와 같은 논리로 사립 중학교 전체를 낸다 —
배정이 아니라 지원으로 가는 학교라 '근처'가 실제로 의미를 가진다.

학교별 특목고·자사고 진학률은 여기서 다루지 않는다. 지도에 없는 공립중까지
모집단에 넣어야 해서 별도 파이프라인(collect_progression_school.py →
build_progression_school.py)이 progression_school.json 으로 따로 굽고, 화면에서
학교명으로 조인한다.
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

# 학교급 표기를 화면용으로 줄인다. 고등학교는 collect_schools.py 가 이미
# 특목고(과학·외국어·국제 계열)만 남겨 두었으므로 그대로 '특목고' 가 된다.
LEVEL_SHORT = {"초등학교": "초", "중학교": "중", "고등학교": "특목고"}

# 국제중은 중학교 안에서 따로 뗀다. 전국 국제중은 공식 명칭이 '○○국제중학교'
# 라 이름 판정이 확실하다(서울·경기는 대원·영훈·청심 3곳). NEIS 에는 중학교
# 종류를 구분하는 필드가 없어 이름 말고는 근거가 없다.
INTL_MIDDLE_SUFFIX = "국제중학교"
INTL_MIDDLE_LABEL = "국제중"


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
    """대상: 사립 초·중과 특목고.

    사립 초·중만 넣는 이유는 그대로다 — 진학 실적 기반 순위를 매길 데이터가
    없어서 '지원으로 가는 학교'라는 논리를 쓴다. 공립·국립 중학교는 배정이라
    여기서 걸러진다.

    고등학교는 예외로 설립구분을 보지 않는다. collect_schools.py 가 NEIS
    분류로 특목고(과학·외국어·국제 계열)만 남겨 두었고, 그 학교들은 공립이든
    사립이든 모두 지원해서 가기 때문이다 — 서울과학고·경기과학고처럼 공립인
    곳을 설립구분으로 거르면 정작 대상이 빠진다.
    """
    return [r for r in rows
            if (r.get("level") == "고등학교"
                or (r.get("level") in LEVEL_SHORT and r.get("found_type") == "사립"))]


def level_label(row: dict) -> str:
    """화면에 쓸 학교급. 국제중은 중학교에서 따로 뗀다."""
    level = LEVEL_SHORT[row["level"]]
    if level == "중" and row["school_name"].endswith(INTL_MIDDLE_SUFFIX):
        return INTL_MIDDLE_LABEL
    return level


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
        school = {
            "name": row["school_name"],
            "lvl": level_label(row),
            "found": row["found_type"],
            "sgg": sgg,
            "dong": umd.split(" ")[-1],
            "dong_cd": dong_cd,
            "addr": row["addr"],
            "x": round(x, 1),
            "y": round(y, 1),
        }
        # 계열은 특목고에만 있다. 초·중까지 빈 문자열로 채우면 JSON 이 그만큼
        # 커지는데 예산(100KB)이 빠듯하다.
        if row.get("course"):
            school["course"] = row["course"]
        out.append(school)

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
    print(f"원본 {len(rows):,}건 중 대상 {len(picked):,}건 "
          f"(특목고 {sum(1 for r in picked if r['level'] == '고등학교')}곳 포함)")

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
