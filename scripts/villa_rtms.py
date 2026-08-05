"""연립·다세대 실거래가 XML 파싱과 월별 CSV 직렬화. 순수 함수만 둔다.

아파트용 rtms.py 와 같은 역할이고 구조도 같지만, 응답 태그가 달라 따로 둔다
(aptNm→mhouseNm, 추가로 houseType·landAr). 아파트 수집은 매일 도는 크론이
쓰고 있어 그쪽 파일을 건드리지 않는 편이 안전하다.

왜 필요한가 — 재개발 정비구역 안은 다세대·연립·단독이지 아파트가 아니다.
아파트 실거래만 가지고는 재개발 사업장 690곳의 인가 전후 가격 변화를 볼 수 없다.

단독·다가구(RTMSDataSvcSHTrade)는 쓰지 않는다. 그 API 는 지번을 '3**' 처럼
마스킹해서 내보내 정비구역 대표지번과 맞출 수 없다.
"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

from rtms import ApiError, LIMIT_CODES, gunzip_text, gzip_bytes  # noqa: F401  재사용

COLUMNS = [
    "sgg_cd",
    "umd_nm",
    "jibun",
    "house_name",
    "house_type",
    "build_year",
    "area_sqm",
    "land_ar",
    "floor",
    "price_10k",
    "trade_date",
    "deal_type",
    "seller_gbn",
    "buyer_gbn",
    "cdeal_type",
    "cdeal_day",
]

# 정렬 겸 중복제거 키. 파일 바이트를 재현 가능하게 만드는 근거이기도 하다.
KEY_COLUMNS = [
    "sgg_cd",
    "umd_nm",
    "jibun",
    "house_name",
    "area_sqm",
    "floor",
    "trade_date",
    "price_10k",
]


def _txt(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def _amount(raw: str) -> int | None:
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def parse_response(xml_text: str) -> tuple[list[dict[str, str]], int]:
    """연립·다세대 실거래 XML 을 (레코드 목록, totalCount) 로 파싱한다.

    계약해제 건도 버리지 않고 플래그로 남긴다 (rtms.parse_response 와 같은 이유).

    Raises:
        ApiError: resultCode 가 000 이 아닐 때.
    """
    root = ET.fromstring(xml_text)
    code = (root.findtext(".//resultCode") or "").strip()
    if code not in ("000", "00"):
        raise ApiError(code, (root.findtext(".//resultMsg") or "").strip())

    total = int((root.findtext(".//totalCount") or "0").strip() or 0)

    rows: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        price = _amount(_txt(item, "dealAmount"))
        if price is None:
            continue
        year = _txt(item, "dealYear")
        if not year:
            continue
        date = f"{year}-{_txt(item, 'dealMonth').zfill(2)}-{_txt(item, 'dealDay').zfill(2)}"
        rows.append(
            {
                "sgg_cd": _txt(item, "sggCd"),
                "umd_nm": _txt(item, "umdNm"),
                "jibun": _txt(item, "jibun"),
                "house_name": _txt(item, "mhouseNm"),
                "house_type": _txt(item, "houseType"),
                "build_year": _txt(item, "buildYear"),
                "area_sqm": _txt(item, "excluUseAr"),
                "land_ar": _txt(item, "landAr"),
                "floor": _txt(item, "floor"),
                "price_10k": str(price),
                "trade_date": date,
                "deal_type": _txt(item, "dealingGbn"),
                "seller_gbn": _txt(item, "slerGbn"),
                "buyer_gbn": _txt(item, "buyerGbn"),
                "cdeal_type": _txt(item, "cdealType"),
                "cdeal_day": _txt(item, "cdealDay"),
            }
        )
    return rows, total


def sort_key(row: dict[str, str]) -> tuple:
    return tuple(row.get(c, "") for c in KEY_COLUMNS)


def merge_rows(*batches: list[dict[str, str]]) -> list[dict[str, str]]:
    """여러 배치를 합치고 중복을 제거한 뒤 정렬한다. 나중 배치가 앞의 것을 덮는다."""
    merged: dict[tuple, dict[str, str]] = {}
    for batch in batches:
        for row in batch:
            merged[sort_key(row)] = row
    return [merged[k] for k in sorted(merged)]


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
    return buf.getvalue()


def csv_to_rows(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text, newline="")))
