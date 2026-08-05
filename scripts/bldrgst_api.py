"""건축HUB 건축물대장 총괄표제부 파싱. 순수 함수만 둔다.

I/O 는 collect_bldrgst.py 가 담당한다.

총괄표제부(getBrRecapTitleInfo)는 단지 하나를 한 줄로 준다 — 동별로 쪼개지지
않아 세대수·연면적이 단지 전체 기준이다. 은마아파트를 조회하면 hhldCnt 4424,
mainBldCnt 31 이 그대로 나온다. data/complexes.csv.gz 를 (시군구, 법정동,
단지명)으로 묶어 세대수를 합산하던 어림짐작을 이 값으로 대체할 수 있다.

주의 — 대장에 비어 있는 칸이 많다:
  - platArea(대지면적)와 vlRat(용적률)이 0 인 단지가 20% 남짓 된다. 은마가 그렇다.
  - hhldCnt 가 비는 단지도 있다 (고양 마두동 표본은 전부 비어 있었다).
그래서 이 파일은 값을 만들어내지 않고 있는 그대로 옮기기만 한다. 무엇으로
메울지는 build_redevelopment.py 가 정한다.

용적률은 vlRat 를 그대로 믿지 않는다. vlRatEstmTotArea(용적률 산정 연면적)를
대지면적으로 나누는 편이 채움률이 높다. 은마는 488,472.85 ÷ 239,225.8 = 204.2%
로, 알려진 값과 맞는다.
"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

COLUMNS = [
    "pnu",
    "sgg_cd",
    "bld_nm",
    "main_bld_cnt",
    "hhld_cnt",
    "plat_area",
    "arch_area",
    "tot_area",
    "vl_rat_estm_tot_area",
    "vl_rat",
    "bc_rat",
    "use_apr_day",
    "main_purps",
]

# 공동주택만 남긴다. 같은 법정동을 통째로 받으면 근린생활시설·업무시설이 섞여 온다.
HOUSING_PURPS = ("공동주택", "아파트", "연립주택", "다세대주택", "주상복합")

# 한도 초과를 뜻하는 응답 코드. 만나면 그날 수집을 통째로 중단한다.
LIMIT_CODES = {
    "22",
    "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
}


class ApiError(Exception):
    """resultCode 가 정상(00/000)이 아닐 때."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"resultCode={code} {message}")
        self.code = code
        self.message = message

    @property
    def is_limit(self) -> bool:
        return self.code in LIMIT_CODES


def _txt(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def make_pnu(sgg_cd: str, bjdong_cd: str, plat_gb_cd: str, bun: str, ji: str) -> str | None:
    """대장 응답의 주소 코드로 19자리 PNU 를 만든다.

    platGbCd 는 0=일반 / 1=산 인데 PNU 의 대장구분은 1=일반 / 2=산 이라 하나씩 밀린다.
    (regions.parse_jibun 이 쓰는 규약과 맞춘다.)
    """
    if not (sgg_cd and bjdong_cd and bun):
        return None
    ledger = "2" if str(plat_gb_cd).strip() == "1" else "1"
    return f"{sgg_cd}{bjdong_cd}{ledger}{bun.zfill(4)}{(ji or '0').zfill(4)}"


def parse_response(xml_text: str) -> tuple[list[dict[str, str]], int]:
    """총괄표제부 XML 을 (레코드 목록, totalCount) 로 파싱한다.

    공동주택이 아닌 건물은 버린다.

    Raises:
        ApiError: resultCode 가 정상이 아닐 때.
    """
    root = ET.fromstring(xml_text)
    code = (root.findtext(".//resultCode") or "").strip()
    if code not in ("00", "000"):
        raise ApiError(code, (root.findtext(".//resultMsg") or "").strip())

    total = int((root.findtext(".//totalCount") or "0").strip() or 0)

    rows: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        purps = _txt(item, "mainPurpsCdNm")
        if purps and not any(p in purps for p in HOUSING_PURPS):
            continue
        pnu = make_pnu(
            _txt(item, "sigunguCd"),
            _txt(item, "bjdongCd"),
            _txt(item, "platGbCd"),
            _txt(item, "bun"),
            _txt(item, "ji"),
        )
        if not pnu:
            continue
        rows.append(
            {
                "pnu": pnu,
                "sgg_cd": _txt(item, "sigunguCd"),
                "bld_nm": _txt(item, "bldNm"),
                "main_bld_cnt": _txt(item, "mainBldCnt"),
                "hhld_cnt": _txt(item, "hhldCnt"),
                "plat_area": _txt(item, "platArea"),
                "arch_area": _txt(item, "archArea"),
                "tot_area": _txt(item, "totArea"),
                "vl_rat_estm_tot_area": _txt(item, "vlRatEstmTotArea"),
                "vl_rat": _txt(item, "vlRat"),
                "bc_rat": _txt(item, "bcRat"),
                "use_apr_day": _txt(item, "useAprDay"),
                "main_purps": purps,
            }
        )
    return rows, total


def _num(value: str | float | None) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def floor_area(row: dict) -> float | None:
    """용적률 산정 연면적. 비면 연면적으로 물러선다."""
    for key in ("vl_rat_estm_tot_area", "tot_area"):
        value = _num(row.get(key))
        if value > 0:
            return value
    return None


def far(row: dict, land_sqm: float | None = None) -> float | None:
    """용적률(%). 대장의 vlRat 를 먼저 쓰고, 비면 연면적 ÷ 대지면적으로 낸다.

    land_sqm 을 주면 대장의 plat_area 대신 그 값을 쓴다 — 대장 대지면적이
    0 인 단지에 지적도에서 받은 면적을 넣어 되살리기 위한 통로다.
    """
    recorded = _num(row.get("vl_rat"))
    if recorded > 0:
        return recorded
    land = land_sqm if land_sqm else _num(row.get("plat_area"))
    gross = floor_area(row)
    if not land or not gross:
        return None
    return gross / land * 100


def merge_rows(*batches: list[dict[str, str]]) -> list[dict[str, str]]:
    """PNU 기준으로 합치고 정렬한다. 나중 배치가 앞의 것을 덮어쓴다.

    같은 PNU 에 총괄표제부가 여럿 오면 연면적이 가장 큰 것을 남긴다 —
    재건축·증축으로 옛 대장이 함께 조회되는 경우가 있고, 그중 실제 단지는
    가장 큰 쪽이다.
    """
    merged: dict[str, dict[str, str]] = {}
    for batch in batches:
        for row in batch:
            old = merged.get(row["pnu"])
            if old is None or _num(row.get("tot_area")) > _num(old.get("tot_area")):
                merged[row["pnu"]] = row
    return [merged[k] for k in sorted(merged)]


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return buf.getvalue()
