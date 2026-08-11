"""미국 ETF 구성종목 파싱 — 순수 함수만. I/O 없음.

세 발행사만 다룬다. iShares·ProShares·Invesco·GlobalX 등은 URL을 찾지 못했다 —
확인 안 된 URL은 추측하지 않는다. 나머지는 collect_us_holdings.py 가 "발행사
모름" 으로 건너뛴다.

  Direxion  www.direxion.com/holdings/{TICKER}.csv                 CSV
  ARK       assets.ark-funds.com/fund-documents/funds-etf-csv/{FUND_FILE}.csv  CSV
  SPDR      www.ssga.com/.../holdings-daily-us-en-{ticker}.xlsx     XLSX(zip+xml)

ISSUER_BY_TICKER 는 data/stocks/us_universe.json.gz(2026-08-11 스냅샷)의 종목명을
훑어 **한 번 만들고 하드코딩**했다. 이름으로 발행사를 맞추는 걸 실행 시점 정규식에
맡기면, 피드가 이름 표기를 바꾸는 순간(예: "State Street SPDR" ↔ "SPDR") 조용히
빠지는 종목이 생긴다. 티커는 상장폐지되지 않는 한 안 바뀐다.

밟은 지뢰를 적어 둔다.

  레버리지 상품의 스왑   Direxion 3배 상품(SOXL 등)은 토탈리턴스왑으로 레버리지를
                        만든다. StockTicker 가 빈 줄 중 일부(ICE SEMICONDUCTOR
                        INDEX SWAP·"Semiconductor Bull 3x" 같은)는 현금이 아니라
                        그 스왑의 명목가치다 — 그대로 cash 에 더하면 SOXL 하나가
                        '현금 271%' 로 찍힌다. SecurityDescription 이 SWAP 이거나
                        "Bull/Bear Nx" 패턴(펀드 자신의 목표 배수를 되풀이하는
                        이름)이면 진짜 현금이 아니라 레버리지 장치로 보고
                        **rows·cash 어디에도 넣지 않고 버린다.** 인버스·채권
                        레버리지 상품(TZA·TMF 등)은 개별 종목을 아예 안 담고 이런
                        스왑뿐이라 rows 가 거의 빈다 — 이것도 맞는 결과다.
                        그래도 직접보유(주식+진짜 현금)의 합이 100 을 10~15%p
                        웃도는 경우가 있다(과담보). 유효성 검사는 그 여유를 둔다.

  ARK 파일명            URL 에 티커가 아니라 펀드 전체 이름이 들어간다. 정식 이름과
                        철자가 미묘하게 다르다 — ARKQ 는 "Technology" 가 아니라
                        "Tech." 로 줄어 있고 "&" 앞뒤에 마침표가 붙는다. 6개 전부
                        실제로 요청해 200 이 오는 철자를 확인하고 아래에 박았다.

  SPDR 금 신탁          GLD·GLDM 은 같은 URL 패턴이 404 다 — 실물 금만 담아서
                        '구성종목' 개념 자체가 없는 상품이다. ISSUER_BY_TICKER 에서
                        아예 뺐다.

  ARK 꼬리 문구          CSV 마지막 줄에 법적 고지문이 필드 수가 안 맞는 채로 한
                        줄 통째로 붙어 있다. csv.DictReader 가 그 줄을 None 값
                        섞인 딕셔너리로 뱉는다. weight 칸이 '%' 를 안 담고 있으면
                        건너뛴다.

  SPDR 표 끝            xlsx 시트는 표가 끝난 뒤에도 안내문·법적고지 행이 계속
                        이어진다. Name 칸이 비는 첫 행에서 멈춘다.

  종목코드 없음(Direxion) StockTicker 칸이 아예 비어 있는 줄(현금성 MMF·스왑
                        모두)이 있다. 빈 칸을 '이름 없는 종목' 취급하면 안 되고
                        cash 로 보내야 한다 — DREYFUS GOVT CASH MAN INS 가 대표
                        사례다.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

# --- 발행사별 종목 ----------------------------------------------------------
# data/stocks/us_universe.json.gz 192개를 이름으로 한 번 훑어 만든 결과.
# 아래 세 리스트만 손으로 고친다 — 유니버스가 바뀌면 이 파일도 다시 만든다.

DIREXION_TICKERS = [
    "AAPD", "AAPU", "AMDD", "AMZU", "CURE", "DFEN", "DPST", "DRIP", "DRN",
    "DUST", "ERX", "ERY", "GGLL", "GUSH", "JDST", "JNUG", "LABD", "LABU",
    "METU", "MSFD", "MSFU", "NAIL", "NUGT", "NVDU", "SOXL", "SOXS", "SPXL",
    "SPXS", "TMF", "TMV", "TNA", "TSLL", "TSLS", "TZA", "WEBL", "YANG", "YINN",
]

# GLD·GLDM(금 실물 신탁)은 뺐다 — 같은 URL 패턴이 404 다.
SPDR_TICKERS = [
    "BIL", "DIA", "JNK", "KRE", "MDY", "SPY", "XBI", "XHB", "XLB", "XLC",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XME",
    "XOP",
]

# 티커 -> assets.ark-funds.com 파일명. 실제로 HEAD 요청해 200 을 확인했다.
ARK_FUND_FILE = {
    "ARKB": "ARK_21SHARES_BITCOIN_ETF_ARKB_HOLDINGS",
    "ARKF": "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS",
    "ARKG": "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS",
    "ARKK": "ARK_INNOVATION_ETF_ARKK_HOLDINGS",
    "ARKQ": "ARK_AUTONOMOUS_TECH._&_ROBOTICS_ETF_ARKQ_HOLDINGS",
    "ARKW": "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS",
}

ISSUER_DIREXION = "Direxion"
ISSUER_SPDR = "SPDR"
ISSUER_ARK = "ARK"

# 티커 -> 발행사. holdings_url() 과 collect_us_holdings.py 가 여기만 본다.
ISSUER_BY_TICKER: dict[str, str] = {
    **{t: ISSUER_DIREXION for t in DIREXION_TICKERS},
    **{t: ISSUER_SPDR for t in SPDR_TICKERS},
    **{t: ISSUER_ARK for t in ARK_FUND_FILE},
}

URL_DIREXION = "https://www.direxion.com/holdings/{ticker}.csv"
URL_SPDR = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
)
URL_ARK = "https://assets.ark-funds.com/fund-documents/funds-etf-csv/{file}.csv"

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RE_COL = re.compile(r"[A-Z]+")
_RE_SPDR_ASOF = re.compile(r"As of\s+(\d{1,2})-([A-Za-z]{3})-(\d{4})")

# Direxion 레버리지 상품의 토탈리턴스왑. "ICE SEMICONDUCTOR INDEX SWAP" 처럼
# SWAP 이 박혀 있거나, "Semiconductor Bull 3x"·"20+ Year Treasury Bull 3x" 처럼
# 펀드 자신의 목표 배수 문구를 되풀이한다. 둘 다 진짜 현금이 아니다.
_RE_DIREXION_SWAP = re.compile(r"SWAP|\bBULL\s*\d|\bBEAR\s*\d", re.I)


def holdings_url(ticker: str) -> str | None:
    """발행사별 구성종목 URL. 발행사를 모르면 None — 추측하지 않는다."""
    issuer = ISSUER_BY_TICKER.get(ticker)
    if issuer == ISSUER_DIREXION:
        return URL_DIREXION.format(ticker=ticker)
    if issuer == ISSUER_SPDR:
        return URL_SPDR.format(ticker=ticker.lower())
    if issuer == ISSUER_ARK:
        fname = ARK_FUND_FILE.get(ticker)
        return URL_ARK.format(file=urllib.parse.quote(fname)) if fname else None
    return None


def _sort_rows(rows: list[dict]) -> list[dict]:
    """비중 내림차순. 동률은 티커 오름차순으로 묶어 실행마다 순서가 안 흔들리게 한다."""
    return sorted(rows, key=lambda r: (-r["weight"], r["ticker"]))


def total_weight(rows: list[dict], cash: float) -> float:
    """구성종목 비중 합 + 현금. 유효성 판단의 재료."""
    return sum(r["weight"] for r in rows) + cash


def total_weight_ok(rows: list[dict], cash: float, leverage: float = 1.0) -> bool:
    """비중 합이 그럴듯한 범위(95~105)에 드는가.

    레버리지 상품은 스왑 명목가치 때문에 합이 배수만큼 커진다 — 3배 상품이면
    300% 언저리가 정상이다. 그래서 임계값도 배수로 늘린다. 1배 상품인데 범위
    밖이면 진짜로 파싱이 잘못됐을 확률이 높다.
    """
    scale = max(1.0, abs(leverage))
    total = total_weight(rows, cash)
    return 95.0 * scale <= total <= 105.0 * scale


def _to_iso_date(text: str, fmt: str) -> str:
    try:
        return datetime.strptime(text.strip(), fmt).date().isoformat()
    except (ValueError, AttributeError):
        return ""


def parse_direxion(raw: bytes) -> dict:
    """Direxion CSV.

    파일 앞 3~4줄은 펀드명·티커·발행주식수다. 진짜 헤더는 "TradeDate" 로 시작하는
    줄부터다. StockTicker 가 빈 줄은 현금성 MMF 이거나 레버리지용 토탈리턴스왑
    이다 — 스왑은 진짜 현금이 아니므로 버리고, 나머지만 cash 로 보낸다.
    """
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"TradeDate"')), None)
    if start is None:
        return {"asOf": "", "rows": [], "cash": 0.0}

    rows: list[dict] = []
    cash = 0.0
    as_of = ""
    for row in csv.DictReader(lines[start:]):
        pct = row.get("HoldingsPercent")
        if not pct:
            continue
        try:
            weight = round(float(pct), 4)
        except ValueError:
            continue
        if not as_of:
            as_of = _to_iso_date(row.get("TradeDate") or "", "%m/%d/%Y %I:%M:%S %p")
        ticker = (row.get("StockTicker") or "").strip()
        name = (row.get("SecurityDescription") or "").strip()
        if not ticker:
            if not _RE_DIREXION_SWAP.search(name):
                cash += weight
            continue  # 스왑이면 rows·cash 어디에도 안 넣고 버린다
        rows.append({"ticker": ticker, "name": name, "weight": weight})

    return {"asOf": as_of, "rows": _sort_rows(rows), "cash": round(cash, 4)}


def parse_ark(raw: bytes) -> dict:
    """ARK CSV. 열은 date,fund,company,ticker,cusip,shares,market value ($),weight (%).

    마지막 줄에 법적 고지문이 필드 수가 안 맞는 채로 붙어 있다 — weight 칸에
    '%' 가 없으면 그 줄이다. 조용히 건너뛴다. ticker 가 빈 줄(현금성 MMF)은
    cash 로 보낸다.
    """
    text = raw.decode("utf-8", "replace")
    rows: list[dict] = []
    cash = 0.0
    as_of = ""
    for row in csv.DictReader(io.StringIO(text)):
        pct = row.get("weight (%)")
        if not pct or "%" not in pct:
            continue
        try:
            weight = round(float(pct.strip().rstrip("%")), 4)
        except ValueError:
            continue
        if not as_of:
            as_of = _to_iso_date(row.get("date") or "", "%m/%d/%Y")
        ticker = (row.get("ticker") or "").strip()
        name = (row.get("company") or "").strip()
        if not ticker:
            cash += weight
            continue
        rows.append({"ticker": ticker, "name": name, "weight": weight})

    return {"asOf": as_of, "rows": _sort_rows(rows), "cash": round(cash, 4)}


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"m": _XLSX_NS}
    out = []
    for si in root.findall("m:si", ns):
        out.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
    return out


def _col_letters(cell_ref: str) -> str:
    m = _RE_COL.match(cell_ref)
    return m.group(0) if m else cell_ref


def parse_spdr(raw: bytes) -> dict:
    """SPDR/SSGA xlsx. zipfile + xml.etree 로만 읽는다 — openpyxl 등 외부 라이브러리 없이.

    열 순서가 상품마다 조금씩 다르다(Sector 가 없는 채권형도 있다). 그래서 위치가
    아니라 헤더 행의 라벨(Name·Ticker·Weight)로 열을 찾는다. 표 뒤로 안내문·법적
    고지가 계속 이어지므로 Name 칸이 비는 첫 행에서 멈춘다. Ticker 가 '-' 인 줄
    (현금·잔여 통화·단기 국채 MMF)은 cash 로 보낸다.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return {"asOf": "", "rows": [], "cash": 0.0}

    strings = _shared_strings(archive)
    ns = {"m": _XLSX_NS}

    grid: list[dict[str, str | None]] = []
    for row_el in sheet.findall(".//m:row", ns):
        cells: dict[str, str | None] = {}
        for c in row_el.findall("m:c", ns):
            ref = c.get("r")
            if not ref:
                continue
            t = c.get("t")
            v = c.find("m:v", ns)
            val = v.text if v is not None else None
            if t == "s" and val is not None:
                idx = int(val)
                val = strings[idx] if 0 <= idx < len(strings) else None
            cells[_col_letters(ref)] = val
        grid.append(cells)

    as_of = ""
    header_idx = None
    for i, cells in enumerate(grid):
        values = [v for v in cells.values() if v]
        if not as_of:
            for v in values:
                m = _RE_SPDR_ASOF.search(v)
                if m:
                    as_of = _to_iso_date(m.group(0)[len("As of "):], "%d-%b-%Y")
        if "Name" in values and "Ticker" in values and "Weight" in values:
            header_idx = i
            header_cells = cells
            break
    if header_idx is None:
        return {"asOf": as_of, "rows": [], "cash": 0.0}

    label_of_col = {col: label for col, label in header_cells.items() if label}
    col_name = next((c for c, l in label_of_col.items() if l == "Name"), None)
    col_ticker = next((c for c, l in label_of_col.items() if l == "Ticker"), None)
    col_weight = next((c for c, l in label_of_col.items() if l == "Weight"), None)
    if col_name is None or col_weight is None:
        return {"asOf": as_of, "rows": [], "cash": 0.0}

    rows: list[dict] = []
    cash = 0.0
    for cells in grid[header_idx + 1:]:
        name = cells.get(col_name)
        if not name:
            break  # 표가 끝나고 안내문·법적고지가 시작되는 지점
        try:
            weight = round(float(cells.get(col_weight)), 4)
        except (TypeError, ValueError):
            continue
        ticker = (cells.get(col_ticker) or "").strip() if col_ticker else ""
        if not ticker or ticker == "-":
            cash += weight
            continue
        rows.append({"ticker": ticker, "name": name.strip(), "weight": weight})

    return {"asOf": as_of, "rows": _sort_rows(rows), "cash": round(cash, 4)}
