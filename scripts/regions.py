"""서울·경기 지역 코드 테이블과 PNU 조립.

실거래가 API 는 aptSeq 를 주지 않지만 sggCd / umdNm / jibun 을 준다.
단지정보 API 의 pnu 와 구조가 같으므로 결정론적으로 조립해 조인할 수 있다.

    PNU = 법정동코드(10) + 대장구분(1) + 본번(4) + 부번(4)
    예) 대치동 1014-3 -> 1168010600 + 1 + 1014 + 0003
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGION_FILE = DATA_DIR / "region_codes_11_41.tsv"

SEOUL_PREFIX = "11"
GYEONGGI_PREFIX = "41"

# 시도 루트. 시군구 목록에서 제외한다.
_SIDO_ROOTS = {"11000", "41000"}


def _rows() -> list[tuple[str, str]]:
    """(10자리 코드, 법정동명) 중 '존재' 상태인 행만 돌려준다."""
    out: list[tuple[str, str]] = []
    with REGION_FILE.open(encoding="utf-8") as f:
        next(f)  # 헤더
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            code, name, status = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if status == "존재" and len(code) == 10:
                # 부천시 구 이름 등에 후행 공백이 섞여 있어 내부 공백까지 정규화한다.
                out.append((code, re.sub(r"\s+", " ", name)))
    return out


@lru_cache(maxsize=1)
def sgg_codes() -> list[tuple[str, str]]:
    """조회 대상 시군구 코드 목록. (5자리 코드, 이름)

    구가 설치된 시는 **하위 구 코드만** 포함하고 상위 시 코드는 제외한다.
    실거래가 API 는 과거 데이터를 현재 행정구역으로 소급 재배치하기 때문이다.
    실측 결과 41590(화성시)/41190(부천시)/41110(수원시) 등 구가 있는 시의
    상위 코드는 2006년치까지 전부 0건이고, 41597(동탄구)은 구 신설 10년 전인
    2015-06 에도 269건을 돌려준다. 상위 코드를 병행 조회할 필요가 없다.
    """
    seen: set[str] = set()
    for code, name in _rows():
        if code[5:] != "00000":
            continue  # 시군구 레벨이 아님
        sgg = code[:5]
        if sgg in _SIDO_ROOTS or not sgg.startswith((SEOUL_PREFIX, GYEONGGI_PREFIX)):
            continue
        seen.add(sgg)

    out: list[tuple[str, str]] = []
    for code, name in _rows():
        if code[5:] != "00000":
            continue
        sgg = code[:5]
        if sgg not in seen:
            continue
        # 구가 달린 시(XXXX0)는 제외. 같은 4자리 접두사의 구 코드(XXXXn)가 있으면 상위다.
        if sgg.endswith("0") and any(
            other[:4] == sgg[:4] and not other.endswith("0") for other in seen
        ):
            continue
        out.append((sgg, name))
    return sorted(set(out))


@lru_cache(maxsize=1)
def _sgg_names() -> dict[str, str]:
    """시군구 5자리 -> 시도까지 포함한 정식 이름."""
    return {
        code[:5]: name
        for code, name in _rows()
        if code[5:] == "00000" and code[:5] not in _SIDO_ROOTS
    }


@lru_cache(maxsize=2)
def dong_index() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """(시군구 5자리, 지명) -> 법정동 10자리 코드. (정확 접미사, 말단 토큰) 두 벌.

    실거래가 API 의 umdNm 은 시군구 아래 전체 접미사를 준다. 동 지역은
    '역삼동' 처럼 한 토큰이지만, 읍면 지역은 '고덕면 궁리' 처럼 두 토큰이다.
    따라서 접미사 전체를 1차 키로 쓰고, 말단 토큰을 2차 폴백으로 둔다.

    말단 토큰 키는 같은 시군구 안에서 충돌할 수 있어(서로 다른 읍의 같은 리 이름)
    충돌하면 모호한 것으로 보고 아예 버린다. 잘못된 PNU 를 만드는 것보다
    조인 실패로 남기는 편이 낫다.
    """
    sgg_names = _sgg_names()
    exact: dict[tuple[str, str], str] = {}
    leaf: dict[tuple[str, str], str] = {}
    leaf_conflicts: set[tuple[str, str]] = set()

    for code, name in _rows():
        if code[5:] == "00000":
            continue  # 시군구 레벨은 제외
        sgg = code[:5]
        prefix = sgg_names.get(sgg)
        if not prefix or not name.startswith(prefix):
            continue
        suffix = name[len(prefix):].strip()
        if not suffix:
            continue
        exact[(sgg, suffix)] = code

        token = suffix.split(" ")[-1]
        key = (sgg, token)
        if key in leaf and leaf[key] != code:
            leaf_conflicts.add(key)
        else:
            leaf[key] = code

    for key in leaf_conflicts:
        leaf.pop(key, None)
    return exact, leaf


_JIBUN_RE = re.compile(r"^(산)?\s*(\d+)(?:-(\d+))?$")


def parse_jibun(jibun: str) -> tuple[str, str, str] | None:
    """지번 문자열을 (대장구분, 본번4, 부번4) 로 분해한다.

    '755-1' -> ('1', '0755', '0001')
    '산 12'  -> ('2', '0012', '0000')
    """
    m = _JIBUN_RE.match((jibun or "").strip())
    if not m:
        return None
    is_san, bon, bu = m.group(1), m.group(2), m.group(3)
    if len(bon) > 4 or (bu and len(bu) > 4):
        return None
    ledger = "2" if is_san else "1"
    return ledger, bon.zfill(4), (bu or "0").zfill(4)


def dong_code(sgg_cd: str, umd_nm: str) -> str | None:
    """(시군구, umdNm) 에 해당하는 법정동 10자리 코드. 접미사 일치 우선."""
    exact, leaf = dong_index()
    name = re.sub(r"\s+", " ", (umd_nm or "").strip())
    if not name:
        return None
    found = exact.get((sgg_cd, name))
    if found:
        return found
    return leaf.get((sgg_cd, name.split(" ")[-1]))


def make_pnu(sgg_cd: str, umd_nm: str, jibun: str) -> str | None:
    """실거래가 레코드에서 19자리 PNU 를 조립한다. 실패하면 None."""
    code = dong_code(sgg_cd, umd_nm)
    if code is None:
        return None
    parsed = parse_jibun(jibun)
    if parsed is None:
        return None
    ledger, bon, bu = parsed
    return f"{code}{ledger}{bon}{bu}"
