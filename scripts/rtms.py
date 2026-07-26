"""실거래가 XML 파싱과 월별 CSV 직렬화. 순수 함수만 둔다.

I/O 는 collect_trades.py 가 담당한다. 여기 있는 함수는 전부 부수효과가 없어
저장해 둔 샘플 응답만으로 검증할 수 있다.
"""

from __future__ import annotations

import csv
import gzip
import io
import xml.etree.ElementTree as ET

# 한도 초과를 뜻하는 응답 코드. 만나면 그날 수집을 통째로 중단한다.
LIMIT_CODES = {
    "22",
    "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
}

COLUMNS = [
    "sgg_cd",
    "umd_nm",
    "jibun",
    "apt_name",
    "apt_dong",
    "build_year",
    "area_sqm",
    "floor",
    "price_10k",
    "trade_date",
    "deal_type",
    "seller_gbn",
    "buyer_gbn",
    "land_leasehold",
    "cdeal_type",
    "cdeal_day",
]

# 정렬 겸 중복제거 키. 파일 바이트를 재현 가능하게 만드는 근거이기도 하다.
KEY_COLUMNS = [
    "sgg_cd",
    "umd_nm",
    "jibun",
    "apt_name",
    "apt_dong",
    "area_sqm",
    "floor",
    "trade_date",
    "price_10k",
]


class ApiError(Exception):
    """resultCode 가 정상(000)이 아닐 때."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"resultCode={code} {message}")
        self.code = code
        self.message = message

    @property
    def is_limit(self) -> bool:
        return self.code in LIMIT_CODES


def _txt(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def _amount(raw: str) -> int | None:
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def parse_response(xml_text: str) -> tuple[list[dict[str, str]], int]:
    """실거래가 XML 을 (레코드 목록, totalCount) 로 파싱한다.

    MCP 파서와 달리 계약해제 건(cdealType == 'O')도 버리지 않고 플래그로 남긴다.
    해제 거래 자체가 대시보드에서 의미 있는 지표이기 때문이다.

    Raises:
        ApiError: resultCode 가 000 이 아닐 때.
    """
    root = ET.fromstring(xml_text)
    code = (root.findtext(".//resultCode") or "").strip()
    if code != "000":
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
                "apt_name": _txt(item, "aptNm"),
                "apt_dong": _txt(item, "aptDong"),
                "build_year": _txt(item, "buildYear"),
                "area_sqm": _txt(item, "excluUseAr"),
                "floor": _txt(item, "floor"),
                "price_10k": str(price),
                "trade_date": date,
                "deal_type": _txt(item, "dealingGbn"),
                "seller_gbn": _txt(item, "slerGbn"),
                "buyer_gbn": _txt(item, "buyerGbn"),
                "land_leasehold": _txt(item, "landLeaseholdGbn"),
                "cdeal_type": _txt(item, "cdealType"),
                "cdeal_day": _txt(item, "cdealDay"),
            }
        )
    return rows, total


def sort_key(row: dict[str, str]) -> tuple:
    return tuple(row.get(c, "") for c in KEY_COLUMNS)


def merge_rows(*batches: list[dict[str, str]]) -> list[dict[str, str]]:
    """여러 배치를 합치고 중복을 제거한 뒤 정렬한다.

    같은 (시군구, 동, 지번, 단지, 동호, 면적, 층, 날짜, 금액) 조합은 한 건으로 본다.
    같은 날 같은 층에서 같은 금액의 별개 거래가 두 건 있으면 하나로 합쳐지지만,
    실거래가 API 가 이를 구분할 식별자를 주지 않아 불가피하다.
    나중에 받은 배치가 앞의 것을 덮어써 갱신분(계약해제 등)이 반영된다.
    """
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


def gzip_bytes(text: str) -> bytes:
    """재현 가능한 gzip 바이트를 만든다.

    gzip 은 기본적으로 헤더에 현재 시각을 기록하므로, 내용이 같아도 다시 쓰면
    바이트가 달라지고 git 에 새 blob 이 쌓인다. 매일 돌리면 히스토리가 계속
    불어나기 때문에 mtime 을 0 으로 고정한다. 파일명도 헤더에 안 들어가도록
    GzipFile 에 filename='' 을 준다.
    """
    raw = text.encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0, filename="") as gz:
        gz.write(raw)
    return buf.getvalue()


def gunzip_text(data: bytes) -> str:
    return gzip.decompress(data).decode("utf-8")
