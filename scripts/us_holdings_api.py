"""미국 ETF 구성종목 파싱 — 순수 함수만. I/O 없음.

일곱 발행사를 다룬다(유니버스 192개 중 145개). Invesco 는 모든 경로가 406 이라
못 넣었다 — 헤드리스 크롬으로도 국가선택 스플래시에 막힌다. 확인 안 된 URL 은
추측하지 않는다. 나머지 47개는 collect_us_holdings.py 가 "발행사 모름" 으로
건너뛰고, 화면이 그 사실을 그대로 적는다.

  Direxion   www.direxion.com/holdings/{TICKER}.csv                  CSV
  ARK        assets.ark-funds.com/fund-documents/funds-etf-csv/{FILE}.csv   CSV
  SPDR       www.ssga.com/.../holdings-daily-us-en-{ticker}.xlsx      XLSX(zip+xml)
  iShares    www.blackrock.com/varnish-api/.../get-fund-document      CSV
  ProShares  accounts.profunds.com/etfdata/psdlyhld.csv               CSV(전 종목 1파일)
  Vanguard   investor.vanguard.com/.../portfolio-holding/{stock|bond} JSON
  Global X   assets.globalxetfs.com/funds/holdings/{t}_full-holdings_{YMD}.csv  CSV

ISSUER_BY_TICKER 는 data/stocks/us_universe.json.gz(2026-08-11 스냅샷)의 종목명을
훑어 **한 번 만들고 하드코딩**했다. 이름으로 발행사를 맞추는 걸 실행 시점 정규식에
맡기면, 피드가 이름 표기를 바꾸는 순간(예: "State Street SPDR" ↔ "SPDR") 조용히
빠지는 종목이 생긴다. 티커는 상장폐지되지 않는 한 안 바뀐다.

밟은 지뢰를 적어 둔다.

  레버리지 상품의 스왑   Direxion 3배 상품(SOXL 등)은 토탈리턴스왑으로 레버리지를
                        만든다. 3배 상품은 주식 72%·스왑 220%·현금성 44% 를 합쳐
                        300%대 노출을 만드는 게 **정상**이다 — 스왑을 버리고
                        "주식 72% + 현금 44%" 만 보여주면 레버리지 펀드를 현금
                        많이 든 펀드처럼 보이게 만드는, 원래 화면(빈 표)보다 더
                        나쁜 착시가 생긴다(실측: SOXL 스왑 219.6%). 그래서 버리지
                        않고 **네 번째 칸으로 살린다.**
                        StockTicker 가 빈 줄은 SecurityDescription 으로 다시
                        나눈다 — "SWAP" 이 들어간 줄은 진짜 토탈리턴스왑(swap,
                        swapNote 에 어느 지수인지 적는다). "Semiconductor Bull
                        3x" 처럼 펀드 자신의 목표 배수 이름을 되풀이하는 줄은
                        스왑과 짝을 이루는 명목가치이지만 이름에 SWAP 이 없어
                        따로(other) 묶는다. 나머지 빈 줄만 진짜 현금성 MMF다
                        (cash). 인버스·채권 레버리지 상품(TZA·TMF 등)은 개별
                        종목을 아예 안 담고 이런 스왑뿐이라 rows 가 거의 빈다 —
                        이것도 맞는 결과다.
                        직접보유(주식+진짜 현금)+스왑+기타의 합이 100×배수를
                        10~30%p 웃도는 경우가 있다(과담보). 유효성 검사는 그
                        여유를 둔다 — total_weight_ok() 참고.

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

  SPDR 채권형(Ticker 열 없음)  BIL·JNK 같은 채권형은 열 구성 자체가 다르다 —
                        Name·Weight 는 있지만 Ticker 칸이 없다(대신 Coupon·
                        Maturity 로 채권을 나열한다). 예전 코드는 "Name·Ticker·
                        Weight 셋 다 있어야 헤더"로 판정해서 이 셋을 못 찾으면
                        rows·cash 모두 빈 채로 돌려줬다 — 실제로는 파일을 받고
                        파싱도 됐는데 "발행사가 그날 파일을 안 올렸다"는 실패
                        문구가 뜨는 거짓말이었다. 지금은 헤더 판정을 Name·Weight
                        둘로 낮추고, Ticker 칸이 없으면 noTicker=True 를 얹어
                        모든 행을 rows 에 (ticker="") 그대로 담는다 — 티커 없는
                        줄을 현금으로 쓸어담지 않는다(채권 원장에서 티커 없음은
                        "현금"이 아니라 "이 파일엔 티커가 없다"는 뜻이다).

  종목코드 없음(Direxion) StockTicker 칸이 아예 비어 있는 줄(현금성 MMF·스왑
                        모두)이 있다. 빈 칸을 '이름 없는 종목' 취급하면 안 되고
                        cash 로 보내야 한다 — DREYFUS GOVT CASH MAN INS 가 대표
                        사례다.

  iShares .ajax 폐기     널리 알려진 /{slug}/1467271812596.ajax 는 이제 제품 페이지
                        HTML 을 content-type 만 text/csv 로 붙여 돌려준다. 쿠키
                        동의·Referer·헤드리스 크롬 다운로드까지 다 같은 결과라,
                        "받았는데 파싱이 안 된다"로 오래 헤맬 수 있다. 페이지를
                        렌더해 링크를 캐면 varnish-api 주소가 나온다 — portfolioId
                        하나만 쓰고 슬러그가 안 끼어서 펀드명이 바뀌어도 안 깨진다.

  ProShares 비중 열 없음  psdlyhld.csv 에는 비중이 없고 금액만 있다. 분모를 Exposure
                        합으로 잡으면 틀린다 — 실물은 Market Value, 파생은 Exposure
                        로 열이 갈리기 때문이다. NAV 는 Market Value 열의 합이다.
                        UVXY 로 검산했다: VIX 선물 454.2M ÷ NAV 302.9M = 150.0% 로
                        상품의 1.5배와 정확히 맞는다.

  선물은 현금이 아니다   UVXY 처럼 VIX 선물만 든 상품은 티커 없는 줄이 전부 선물이다.
                        SWAP 만 파생으로 보면 이 150% 가 통째로 '현금' 으로 넘어가
                        "현금 100% 펀드" 라는 거짓 화면이 된다. _RE_SWAP 이 FUTURE·
                        E-MINI 까지 잡는 이유다.

  뱅가드 500행 상한      한 번에 500행까지만 준다(VT 는 10,032종목). next 를 다 따라가면
                        펀드 하나에 21요청이라, 첫 장만 받고 진짜 종목 수는 응답의
                        size(totalCount)로 남긴다. 그래서 비중 합이 100 에 한참 못
                        미친다 — 유효성 검사에서 뱅가드를 빼는 이유다.

  뱅가드 채권의 ticker   채권 원장의 ticker 는 그 채권을 발행한 회사의 **주식** 티커다.
                        그대로 쓰면 "Amazon.com Inc. 회사채" 가 AMZN 으로 찍혀 아마존
                        주식을 담은 것처럼 보인다. 게다가 국채엔 티커가 없어 한 표
                        안에서 채워진 줄과 빈 줄이 섞인다(BND 상위 25줄 중 1줄,
                        VCIT 는 9줄). 채권형은 bond=True 로 티커를 통째로 버린다.

  인버스는 합이 안 맞는다  스왑 명목가치가 음수로 오는데 담보 주식·현금은 양수라,
                        부호 섞인 합이 배수와 무관해진다(SH: 주식 +87.7·현금 +34.0·
                        스왑 −100.0 → 합 +21.7, 목표 100). 여기에 임계값을 맞춰
                        깎으면 그 임계값이 다음 발행사의 진짜 파싱 오류를 놓친다.
                        그래서 인버스는 아예 검사하지 않는다.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta

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

# 티커 -> BlackRock portfolioId. 제품 스크리너(product-screener-v3.1.jsn)로 한 번
# 뽑아 하드코딩했다 — 매일 1.9MB 를 받아 티커를 되찾는 건 낭비고, portfolioId 는
# 펀드가 청산되지 않는 한 안 바뀐다. 43개 전수 호출로 200 을 확인했다.
#
# IAU·SLV 는 뺐다 — 실물 금·은 신탁이라 400("No value for this component: holdings")
# 이 돌아온다. 구성종목 개념 자체가 없는 상품이라 GLD 와 같은 이유로 제외다.
ISHARES_PORTFOLIO = {
    "AGG": "239458", "DVY": "239500", "EEM": "239637", "EFA": "239623",
    "EMB": "239572", "ETHA": "337614", "EWA": "239607", "EWC": "239615",
    "EWG": "239650", "EWH": "239657", "EWJ": "239665", "EWT": "239686",
    "EWU": "239690", "EWY": "239681", "EWZ": "239612", "EZU": "239644",
    "FXI": "239536", "HYG": "239565", "IBIT": "333011", "ICLN": "239738",
    "IEF": "239456", "IEFA": "244049", "IEI": "239455", "IEMG": "244050",
    "IGV": "239771", "INDA": "239659", "ITB": "239512", "IVV": "239726",
    "IWM": "239710", "IYR": "239520", "LQD": "239566", "MCHI": "239619",
    "MUB": "239766", "PFF": "239826", "REM": "239543", "SGOV": "314116",
    "SHV": "239466", "SHY": "239452", "SOXX": "239705", "TIP": "239467",
    "TLT": "239454",
}

# ProShares 는 전 종목이 파일 하나(psdlyhld.csv)에 들어 있다 — 티커별로 받지
# 않는다. 그래서 여기는 "그 파일에서 꺼내 올 티커" 목록일 뿐이다.
PROSHARES_TICKERS = [
    "DOG", "NOBL", "PSQ", "QLD", "SDOW", "SDS", "SH", "SPXU", "SQQQ", "SSO",
    "SVXY", "TBT", "TQQQ", "TWM", "UBT", "UDOW", "UPRO", "UVXY", "VIXY",
]

# 티커 -> 뱅가드 API 의 자산군 경로. 주식형은 stock, 채권형은 bond 다. 아무거나
# 넣으면 200 에 size=0 인 빈 응답이 와서 "받았는데 비었다"로 오인한다. 12개를
# 두 경로 다 찔러 확정했다.
VANGUARD_PATH = {
    "BND": "bond", "VCIT": "bond", "VCSH": "bond",
    "VEA": "stock", "VIG": "stock", "VNQ": "stock", "VOO": "stock",
    "VT": "stock", "VTI": "stock", "VWO": "stock", "VXUS": "stock",
    "VYM": "stock",
}

GLOBALX_TICKERS = [
    "AIQ", "BOTZ", "COPX", "LIT", "QYLD", "RYLD", "SIL", "URA", "XYLD",
]

ISSUER_DIREXION = "Direxion"
ISSUER_SPDR = "SPDR"
ISSUER_ARK = "ARK"
ISSUER_ISHARES = "iShares"
ISSUER_PROSHARES = "ProShares"
ISSUER_VANGUARD = "Vanguard"
ISSUER_GLOBALX = "Global X"

# 티커 -> 발행사. holdings_url() 과 collect_us_holdings.py 가 여기만 본다.
ISSUER_BY_TICKER: dict[str, str] = {
    **{t: ISSUER_DIREXION for t in DIREXION_TICKERS},
    **{t: ISSUER_SPDR for t in SPDR_TICKERS},
    **{t: ISSUER_ARK for t in ARK_FUND_FILE},
    **{t: ISSUER_ISHARES for t in ISHARES_PORTFOLIO},
    **{t: ISSUER_PROSHARES for t in PROSHARES_TICKERS},
    **{t: ISSUER_VANGUARD for t in VANGUARD_PATH},
    **{t: ISSUER_GLOBALX for t in GLOBALX_TICKERS},
}

URL_DIREXION = "https://www.direxion.com/holdings/{ticker}.csv"
URL_SPDR = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
)
URL_ARK = "https://assets.ark-funds.com/fund-documents/funds-etf-csv/{file}.csv"

# iShares 는 예전에 널리 쓰이던 /{slug}/1467271812596.ajax 엔드포인트가 폐기됐다.
# 지금 그 주소로 요청하면 content-type 만 text/csv 로 붙여 제품 페이지 HTML 을
# 돌려준다 — 쿠키 동의·Referer·헤드리스 크롬 다운로드까지 다 같은 결과다.
# 제품 페이지를 렌더해 링크를 캐 보면 아래 주소가 나온다. portfolioId 하나만
# 있으면 되고 슬러그가 안 끼어서, 펀드명이 바뀌어도 안 깨진다.
URL_ISHARES = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/"
    "api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES"
    "&targetSite=us-ishares&locale=en_US&portfolioId={pid}&userType=individual"
    "&component=holdings"
)
# ProShares 전 종목 일별 보유. 종목당 한 번이 아니라 하루 한 번이면 끝난다.
URL_PROSHARES_ALL = "https://accounts.profunds.com/etfdata/psdlyhld.csv"
URL_VANGUARD = (
    "https://investor.vanguard.com/investment-products/etfs/profile/api/"
    "{ticker}/portfolio-holding/{path}"
)
# 날짜가 파일명에 박힌다. 펀드 페이지(291KB)를 긁어 링크를 찾는 대신 최근
# 영업일부터 거슬러 올라가며 찔러 본다 — 보통 1~2번에 맞는다.
URL_GLOBALX = "https://assets.globalxetfs.com/funds/holdings/{lower}_full-holdings_{ymd}.csv"

# 응답·압축해제 크기 상한. 실측 최대는 JNK(채권 1,211종) 의 xlsx 로 원본 약 250 KB,
# sheet1.xml 해제 후 약 6 MB 다. 여유를 20배 두되 무제한은 두지 않는다 — cron 무인
# 실행이라 오염된 응답 하나로 서버가 OOM 으로 죽는 걸 막는 게 목적이다.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_UNZIPPED_BYTES = 128 * 1024 * 1024

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RE_COL = re.compile(r"[A-Z]+")
_RE_SPDR_ASOF = re.compile(r"As of\s+(\d{1,2})-([A-Za-z]{3})-(\d{4})")
_RE_GLOBALX_ASOF = re.compile(r"as of\s+(\d{1,2}/\d{1,2}/\d{4})", re.I)
_RE_PROSHARES_ASOF = re.compile(r"AS OF\s+(\d{1,2}/\d{1,2}/\d{4})", re.I)
_RE_PROSHARES_RESIDUAL = re.compile(r"Net Other Assets", re.I)

# 티커 없는 줄을 스왑/기타/현금 셋으로 나눈다. "ICE SEMICONDUCTOR INDEX SWAP" 처럼
# SWAP 이 박힌 줄은 진짜 토탈리턴스왑. "Semiconductor Bull 3x"·"20+ Year Treasury
# Bull 3x" 처럼 펀드 자신의 목표 배수 문구를 되풀이하는 줄은 스왑과 짝을 이루는
# 명목가치지만 이름에 SWAP 이 없어 other 로 따로 묶는다. 셋 다 세 발행사에 공통으로
# 적용한다 — ARK·SPDR 은 지금 견본에 스왑이 없지만, 나중에 생겨도 조용히 현금에
# 합쳐지는 대신 여기서 걸리게 하기 위해서다.
_RE_SWAP = re.compile(r"\bSWAP\b|\bFUTURE\b|\bFUT\b|E-MINI", re.I)
_RE_SELF_LEV_NAME = re.compile(r"\bBULL\s*\d|\bBEAR\s*\d", re.I)


# 티커가 붙어 있어도 실질이 현금인 줄이 있다. ProShares 는 자사 MMF(IQMM)를
# 담보로 들고 있는데, 티커가 있다는 이유로 종목 표에 올리면 TQQQ 의 1위 보유가
# "PROSHARES GENIUS MNY MKT ETF 16.3%" 로 찍힌다 — 나스닥 3배를 사는 사람이
# 보려는 건 그게 아니다. 이름으로 걸러 현금으로 보낸다.
_RE_CASH_LIKE = re.compile(r"MNY\s*MKT|MONEY\s*MARKET|CASH\s+MGMT|TRSRY\s+CASH", re.I)


def _is_cash_like(name: str) -> bool:
    """티커가 있어도 실질이 현금인 줄인가."""
    return bool(_RE_CASH_LIKE.search(name))


def _classify_no_ticker(name: str) -> str:
    """티커 없는 줄의 성격 판정. "swap" · "other" · "cash" 중 하나를 돌려준다."""
    if _RE_SWAP.search(name):
        return "swap"
    if _RE_SELF_LEV_NAME.search(name):
        return "other"
    return "cash"


def holdings_url(ticker: str) -> str | None:
    """발행사별 구성종목 URL. 발행사를 모르면 None — 추측하지 않는다.

    ProShares 와 Global X 는 여기서 못 만든다 — 앞은 전 종목 단일 파일이라
    티커가 URL 에 안 들어가고(URL_PROSHARES_ALL), 뒤는 파일명에 날짜가 박혀
    globalx_urls() 로 후보를 여러 개 만들어야 한다. 둘 다 None 을 돌려주고
    collect_us_holdings.py 가 따로 처리한다.
    """
    issuer = ISSUER_BY_TICKER.get(ticker)
    if issuer == ISSUER_DIREXION:
        return URL_DIREXION.format(ticker=ticker)
    if issuer == ISSUER_SPDR:
        return URL_SPDR.format(ticker=ticker.lower())
    if issuer == ISSUER_ARK:
        fname = ARK_FUND_FILE.get(ticker)
        return URL_ARK.format(file=urllib.parse.quote(fname)) if fname else None
    if issuer == ISSUER_ISHARES:
        pid = ISHARES_PORTFOLIO.get(ticker)
        return URL_ISHARES.format(pid=pid) if pid else None
    if issuer == ISSUER_VANGUARD:
        path = VANGUARD_PATH.get(ticker)
        return URL_VANGUARD.format(ticker=ticker, path=path) if path else None
    return None


def globalx_urls(ticker: str, today: date, back: int = 7) -> list[str]:
    """Global X 후보 URL 을 최신 날짜부터 나열한다.

    파일명이 lit_full-holdings_20260810.csv 꼴이라 날짜를 알아야 한다. 주말·
    공휴일에는 그날 파일이 없으니 하루씩 거슬러 올라가며 200 이 나오는 첫
    주소를 쓴다. 호출부가 순서대로 시도하고 성공하면 멈춘다.
    """
    if ticker not in GLOBALX_TICKERS:
        return []
    lower = ticker.lower()
    return [
        URL_GLOBALX.format(lower=lower, ymd=(today - timedelta(days=d)).strftime("%Y%m%d"))
        for d in range(back + 1)
    ]


def _sort_rows(rows: list[dict]) -> list[dict]:
    """비중 내림차순. 동률은 티커 오름차순으로 묶어 실행마다 순서가 안 흔들리게 한다."""
    return sorted(rows, key=lambda r: (-r["weight"], r["ticker"]))


def total_weight(rows: list[dict], cash: float, swap: float = 0.0, other: float = 0.0) -> float:
    """구성종목 비중 합 + 현금 + 스왑 + 기타. 유효성 판단의 재료."""
    return sum(r["weight"] for r in rows) + cash + swap + other


def total_weight_ok(
    rows: list[dict], cash: float, swap: float = 0.0, other: float = 0.0, leverage: float = 1.0
) -> bool:
    """비중 합이 그럴듯한 범위에 드는가.

    1배 상품은 100% 언저리가 정상이다. 레버리지 상품은 스왑 명목가치 때문에 합이
    배수만큼 커진다 — 3배 상품이면 300% 언저리다. 그래서 기준선을 100×배수로 잡고,
    거기서 20%p 는 적게(80%), 30%p 는 많게(130%) 허용한다 — SOXL 을 실측하면
    72.2(주식)+219.6(스왑)+43.7(현금)+8.1(기타)=343.6% 로 300 의 114.5% 인데, 이건
    과담보(collateral over-posting)라 정상이다. 대칭이 아니라 위쪽을 더 넉넉히 둔
    이유가 이거다 — 부족(파싱 누락)은 더 엄격히, 초과(과담보)는 더 느슨히 잡는다.

    인버스 상품은 **검사하지 않는다**(항상 True). 스왑 명목가치가 음수로 오는데
    담보로 쌓은 주식·현금은 양수라, 부호가 섞인 합이 상품의 배수와 아무 관계가
    없어진다 — ProShares 19개를 실측하니 SH 가 주식 +87.7·현금 +34.0·스왑
    −100.0 으로 합 +21.7(목표 100), SDS 가 합 −66.0(목표 200)이었다. 어느 쪽도
    파싱 오류가 아니다. 여기에 맞는 임계값을 억지로 깎아 만들면 그 임계값이
    다음 발행사에서 진짜 파싱 오류를 놓친다. 그래서 아예 안 본다.

    범위 밖이어도 데이터는 버리지 않는다 — 로그만 남긴다. 잘못된 임계값 때문에
    멀쩡한 펀드가 조용히 사라지면 안 된다. collect_us_holdings.py 의 collect_one
    참고.
    """
    if leverage < 0:
        return True
    scale = max(1.0, abs(leverage))
    total = abs(total_weight(rows, cash, swap, other))
    return 80.0 * scale <= total <= 130.0 * scale


def _to_iso_date(text: str, fmt: str) -> str:
    try:
        return datetime.strptime(text.strip(), fmt).date().isoformat()
    except (ValueError, AttributeError):
        return ""


def parse_direxion(raw: bytes) -> dict:
    """Direxion CSV.

    파일 앞 3~4줄은 펀드명·티커·발행주식수다. 진짜 헤더는 "TradeDate" 로 시작하는
    줄부터다. StockTicker 가 빈 줄은 현금성 MMF 이거나 레버리지용 토탈리턴스왑
    이다 — _classify_no_ticker() 로 swap·other·cash 셋 중 하나로 보낸다. 스왑은
    버리지 않는다: SOXL 처럼 3배 상품은 스왑이 명목가치의 대부분(220%p 안팎)을
    차지해서, 버리면 "현금 많은 펀드"로 보이는 착시가 생긴다.
    """
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"TradeDate"')), None)
    if start is None:
        return {"asOf": "", "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "", "other": 0.0, "noTicker": False}

    rows: list[dict] = []
    cash = 0.0
    swap = 0.0
    other = 0.0
    swap_names: list[str] = []
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
            bucket = _classify_no_ticker(name)
            if bucket == "swap":
                swap += weight
                if name not in swap_names:
                    swap_names.append(name)
            elif bucket == "other":
                other += weight
            else:
                cash += weight
            continue
        rows.append({"ticker": ticker, "name": name, "weight": weight})

    return {
        "asOf": as_of,
        "rows": _sort_rows(rows),
        "cash": round(cash, 4),
        "swap": round(swap, 4),
        "swapNote": " · ".join(swap_names),
        "other": round(other, 4),
        "noTicker": False,
    }


def parse_ark(raw: bytes) -> dict:
    """ARK CSV. 열은 date,fund,company,ticker,cusip,shares,market value ($),weight (%).

    마지막 줄에 법적 고지문이 필드 수가 안 맞는 채로 붙어 있다 — weight 칸에
    '%' 가 없으면 그 줄이다. 조용히 건너뛴다. ticker 가 빈 줄은 대부분 현금성
    MMF 지만, _classify_no_ticker() 로 한 번 더 걸러서(지금 견본엔 없지만) 스왑
    성격 줄이 조용히 cash 에 섞이는 걸 막는다.
    """
    text = raw.decode("utf-8", "replace")
    rows: list[dict] = []
    cash = 0.0
    swap = 0.0
    other = 0.0
    swap_names: list[str] = []
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
            bucket = _classify_no_ticker(name)
            if bucket == "swap":
                swap += weight
                if name not in swap_names:
                    swap_names.append(name)
            elif bucket == "other":
                other += weight
            else:
                cash += weight
            continue
        rows.append({"ticker": ticker, "name": name, "weight": weight})

    return {
        "asOf": as_of,
        "rows": _sort_rows(rows),
        "cash": round(cash, 4),
        "swap": round(swap, 4),
        "swapNote": " · ".join(swap_names),
        "other": round(other, 4),
        "noTicker": False,
    }


def _read_capped(archive: zipfile.ZipFile, name: str) -> bytes:
    """압축 해제 크기를 확인하고 읽는다.

    xlsx 는 zip 이라 작은 응답이 해제되면 수 GB 로 부풀 수 있다(zip bomb).
    cron 으로 무인 실행되는 수집기라 한 번 터지면 서버가 OOM 으로 죽는다.
    ZipInfo.file_size 는 헤더에 적힌 값이라 위조될 수 있으니, 읽을 때도
    상한+1 바이트만 받아 실제 크기로 한 번 더 막는다.
    """
    if archive.getinfo(name).file_size > MAX_UNZIPPED_BYTES:
        raise ValueError(f"{name}: 압축 해제 크기 상한 초과")
    with archive.open(name) as fp:
        data = fp.read(MAX_UNZIPPED_BYTES + 1)
    if len(data) > MAX_UNZIPPED_BYTES:
        raise ValueError(f"{name}: 압축 해제 크기 상한 초과")
    return data


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(_read_capped(archive, "xl/sharedStrings.xml"))
    except (KeyError, ValueError):
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

    열 순서가 상품마다 조금씩 다르다. 그래서 위치가 아니라 헤더 행의 라벨(Name·
    Ticker·Weight)로 열을 찾는다. 표 뒤로 안내문·법적고지가 계속 이어지므로 Name
    칸이 비는 첫 행에서 멈춘다.

    BIL·JNK 같은 채권형은 Ticker 칸 자체가 없다(Coupon·Maturity 로 채권을
    나열한다) — 헤더 판정을 Name·Weight 둘로 낮추고, Ticker 열을 못 찾으면
    noTicker=True 를 얹어 모든 행을 rows 에 담는다(ticker=""). 이 파일에서 티커
    없음은 "현금"이 아니라 "채권형이라 애초에 티커 칸이 없다"는 뜻이라 cash 로
    보내면 안 된다.

    Ticker 열이 있는(주식형) 파일에서 Ticker 가 '-' 인 줄(현금·잔여 통화·단기
    국채 MMF)은 _classify_no_ticker() 로 swap·other·cash 를 가른다.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        sheet = ET.fromstring(_read_capped(archive, "xl/worksheets/sheet1.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, ValueError):
        return {"asOf": "", "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "", "other": 0.0, "noTicker": False}

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
        if "Name" in values and "Weight" in values:
            header_idx = i
            header_cells = cells
            break
    if header_idx is None:
        return {"asOf": as_of, "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "", "other": 0.0, "noTicker": False}

    label_of_col = {col: label for col, label in header_cells.items() if label}
    col_name = next((c for c, l in label_of_col.items() if l == "Name"), None)
    col_ticker = next((c for c, l in label_of_col.items() if l == "Ticker"), None)
    col_weight = next((c for c, l in label_of_col.items() if l == "Weight"), None)
    if col_name is None or col_weight is None:
        return {"asOf": as_of, "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "", "other": 0.0, "noTicker": False}

    no_ticker = col_ticker is None  # BIL·JNK 같은 채권형 — Ticker 칸 자체가 없다

    rows: list[dict] = []
    cash = 0.0
    swap = 0.0
    other = 0.0
    swap_names: list[str] = []
    for cells in grid[header_idx + 1:]:
        name = cells.get(col_name)
        if not name:
            break  # 표가 끝나고 안내문·법적고지가 시작되는 지점
        try:
            weight = round(float(cells.get(col_weight)), 4)
        except (TypeError, ValueError):
            continue
        if no_ticker:
            # 채권형: 티커 칸이 없으니 전부 이름만 있는 구성종목이다. 현금으로
            # 쓸어담지 않는다 — 못 받은 게 아니라 애초에 이런 모양의 파일이다.
            rows.append({"ticker": "", "name": name.strip(), "weight": weight})
            continue
        ticker = (cells.get(col_ticker) or "").strip()
        if not ticker or ticker == "-":
            bucket = _classify_no_ticker(name)
            if bucket == "swap":
                swap += weight
                if name not in swap_names:
                    swap_names.append(name.strip())
            elif bucket == "other":
                other += weight
            else:
                cash += weight
            continue
        rows.append({"ticker": ticker, "name": name.strip(), "weight": weight})

    return {
        "asOf": as_of,
        "rows": _sort_rows(rows),
        "cash": round(cash, 4),
        "swap": round(swap, 4),
        "swapNote": " · ".join(swap_names),
        "other": round(other, 4),
        "noTicker": no_ticker,
    }


def _to_float(text: str | None) -> float | None:
    """'1,234.56' · '2.80' · '' 을 float 로. 못 읽으면 None."""
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    if not cleaned or cleaned == "-":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _empty() -> dict:
    return {"asOf": "", "rows": [], "cash": 0.0, "swap": 0.0, "swapNote": "",
            "other": 0.0, "noTicker": False}


def _pack(rows: list[dict], cash: float, swap: float, swap_names: list[str],
          other: float, as_of: str, no_ticker: bool) -> dict:
    """네 칸(rows·cash·swap·other)을 공통 모양으로 묶는다."""
    return {
        "asOf": as_of,
        "rows": _sort_rows(rows),
        "cash": round(cash, 4),
        "swap": round(swap, 4),
        "swapNote": " · ".join(swap_names),
        "other": round(other, 4),
        "noTicker": no_ticker,
    }


def parse_ishares(raw: bytes) -> dict:
    """iShares CSV.

    앞 9줄이 펀드명·기준일·설정일·발행주식수다. 진짜 헤더는 첫 칸이 'Ticker'
    (주식형) 또는 'Name'(채권형) 인 줄이다. 채권형(AGG·TLT·HYG·SGOV·LQD·MUB
    등 12개)은 Ticker 열 자체가 없어 SPDR 채권형과 같은 취급을 한다 —
    noTicker=True 로 이름만 담는다. 티커 없음을 현금으로 쓸어담으면 안 된다.

    표가 끝난 뒤 빈 줄과 주석이 이어지므로 첫 칸이 비는 행에서 멈춘다.
    'Weight (%)' 가 비중이다. 티커가 '-' 인 줄(주식형의 현금·파생)은
    _classify_no_ticker() 로 swap·other·cash 를 가른다.
    """
    rows_all = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", "replace"))))
    head = next((i for i, r in enumerate(rows_all)
                 if r and r[0].strip() in ("Ticker", "Name")), None)
    if head is None:
        return _empty()

    cols = [c.strip() for c in rows_all[head]]
    if "Weight (%)" not in cols:
        return _empty()
    i_weight = cols.index("Weight (%)")
    i_name = cols.index("Name") if "Name" in cols else None
    i_ticker = cols.index("Ticker") if "Ticker" in cols else None
    no_ticker = i_ticker is None

    as_of = ""
    for r in rows_all[:head]:
        if len(r) >= 2 and r[0].strip() == "Fund Holdings as of":
            as_of = _to_iso_date(r[1], "%b %d, %Y")
            break

    rows: list[dict] = []
    cash = swap = other = 0.0
    swap_names: list[str] = []
    for r in rows_all[head + 1:]:
        if not r or not r[0].strip():
            break
        weight = _to_float(r[i_weight] if i_weight < len(r) else None)
        if weight is None:
            continue
        name = (r[i_name].strip() if i_name is not None and i_name < len(r) else "")
        ticker = (r[i_ticker].strip() if i_ticker is not None and i_ticker < len(r) else "")
        if no_ticker:
            rows.append({"ticker": "", "name": name, "weight": round(weight, 4)})
            continue
        if not ticker or ticker == "-":
            bucket = _classify_no_ticker(name)
            if bucket == "swap":
                swap += weight
                if name not in swap_names:
                    swap_names.append(name)
            elif bucket == "other":
                other += weight
            else:
                cash += weight
            continue
        if _is_cash_like(name):
            cash += weight
            continue
        rows.append({"ticker": ticker, "name": name, "weight": round(weight, 4)})

    return _pack(rows, cash, swap, swap_names, other, as_of, no_ticker)


def parse_vanguard(raw: bytes, bond: bool = False) -> dict:
    """뱅가드 JSON. fund.entity[] 에 ticker·longName·percentWeight 가 있다.

    한 번에 500행까지만 준다(next 링크로 이어지지만 VT 는 10,032종목이라 21번을
    더 받아야 한다). 화면에는 상위 25종목만 쓰므로 첫 장만 받는다 — 대신
    실제 종목 수(size)를 count 로 남겨 "상위 25/10032" 로 정직하게 적는다.
    비중 합이 100 에 한참 못 미치는 건 그래서다. 유효성 검사에서 뱅가드를
    빼는 이유이기도 하다(collect_us_holdings.py 참고).

    채권형(BND·VCIT·VCSH)은 bond=True 로 부른다 — 아래 noTicker 처리 참고.
    """
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _empty()

    entities = (data.get("fund") or {}).get("entity") or []
    if not entities:
        return _empty()

    as_of = _to_iso_date((data.get("asOfDate") or "")[:10], "%Y-%m-%d")
    rows: list[dict] = []
    for e in entities:
        weight = _to_float(str(e.get("percentWeight") or ""))
        if weight is None:
            continue
        rows.append({
            "ticker": (e.get("ticker") or "").strip(),
            "name": (e.get("longName") or e.get("shortName") or "").strip(),
            "weight": round(weight, 4),
        })
    if not rows:
        return _empty()

    # bond=True 면 티커를 통째로 버린다. 뱅가드 채권 원장의 ticker 는 그 채권을
    # 발행한 회사의 **주식** 티커라, 그대로 두면 "Amazon.com Inc. 회사채"가
    # AMZN 으로 찍혀 아마존 주식을 담은 것처럼 보인다. 게다가 국채에는 티커가
    # 없어 한 표 안에서 채워진 줄과 빈 줄이 뒤섞인다(BND 는 상위 25줄 중 1줄,
    # VCIT 는 9줄). 채권형은 이름만 보여주는 게 맞다.
    no_ticker = bond or not any(r["ticker"] for r in rows)
    if no_ticker:
        for r in rows:
            r["ticker"] = ""
    packed = _pack(rows, 0.0, 0.0, [], 0.0, as_of, no_ticker)
    # 받은 건 첫 장뿐이라 진짜 종목 수를 따로 실어 보낸다.
    packed["totalCount"] = int(data.get("size") or len(rows))
    return packed


def parse_globalx(raw: bytes) -> dict:
    """Global X CSV. 1행 펀드명, 2행 'Fund Holdings Data as of MM/DD/YYYY',
    3행이 헤더('% of Net Assets,Ticker,Name,SEDOL,...') 다.

    해외 상장 종목은 Ticker 가 'ERA FP'(블룸버그 표기)처럼 접미가 붙는다 —
    그대로 둔다. 접미를 떼면 미국 티커와 충돌해 엉뚱한 회사가 된다.
    """
    rows_all = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", "replace"))))
    head = next((i for i, r in enumerate(rows_all)
                 if r and r[0].strip() == "% of Net Assets"), None)
    if head is None:
        return _empty()

    cols = [c.strip() for c in rows_all[head]]
    i_pct, i_ticker, i_name = 0, cols.index("Ticker"), cols.index("Name")

    as_of = ""
    for r in rows_all[:head]:
        m = _RE_GLOBALX_ASOF.search(",".join(r))
        if m:
            as_of = _to_iso_date(m.group(1), "%m/%d/%Y")
            break

    rows: list[dict] = []
    cash = swap = other = 0.0
    swap_names: list[str] = []
    for r in rows_all[head + 1:]:
        if not r or len(r) <= max(i_ticker, i_name):
            continue
        weight = _to_float(r[i_pct])
        if weight is None:
            continue
        ticker = r[i_ticker].strip()
        name = r[i_name].strip()
        if not ticker or ticker == "-":
            bucket = _classify_no_ticker(name)
            if bucket == "swap":
                swap += weight
                if name not in swap_names:
                    swap_names.append(name)
            elif bucket == "other":
                other += weight
            else:
                cash += weight
            continue
        if _is_cash_like(name):
            cash += weight
            continue
        rows.append({"ticker": ticker, "name": name, "weight": round(weight, 4)})

    return _pack(rows, cash, swap, swap_names, other, as_of, False)


def parse_proshares_all(raw: bytes) -> dict[str, dict]:
    """ProShares 전 종목 일별 보유(psdlyhld.csv) 를 티커별로 가른다.

    다른 발행사와 달리 **파일 하나에 전 펀드**가 들어 있어 하루 한 번이면 끝난다.
    2행이 'AS OF M/D/YYYY', 4행이 헤더다.

    **비중 열이 없어 금액에서 만든다.** 두 금액 열의 쓰임이 다르다 —
    실물 보유(주식)와 잔여자산은 Market Value 에, 스왑·선물·MMF 는 명목가치가
    Exposure Value 에 들어간다. 그래서 분모(NAV)는 Market Value 열의 합으로
    잡고, 각 줄은 Market Value 가 있으면 그걸, 없으면 Exposure Value 를 쓴다.

    UVXY 로 검산했다 — VIX 선물 명목 454.2M ÷ NAV 302.9M = 150.0% 로 상품의
    1.5배 목표와 정확히 맞는다. Exposure 합을 분모로 쓰면 이 값이 안 나온다.

    'Net Other Assets (Liabilities)' 는 NAV 를 맞추는 잔여 항목이라 분모에는
    넣고 개별 칸(현금)으로는 안 센다 — MMF 를 따로 세면서 이것까지 현금에
    더하면 같은 돈을 두 번 세게 된다.

    TQQQ 는 주식 73%·스왑 190%대로 3배 노출을 만든다. 스왑 칸이 비면 그건
    파싱이 틀린 것이다.
    """
    rows_all = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", "replace"))))
    head = next((i for i, r in enumerate(rows_all)
                 if r and r[0].strip() == "Fund Ticker"), None)
    if head is None:
        return {}

    cols = [c.strip() for c in rows_all[head]]
    try:
        i_fund = cols.index("Fund Ticker")
        i_sec = cols.index("Security Ticker")
        i_desc = cols.index("Security Description")
        i_exp = cols.index("Exposure Value (Notional + G/L)")
        i_mkt = cols.index("Market Value")
    except ValueError:
        return {}

    as_of = ""
    for r in rows_all[:head]:
        m = _RE_PROSHARES_ASOF.search(",".join(r))
        if m:
            as_of = _to_iso_date(m.group(1), "%m/%d/%Y")
            break

    # 1차: 펀드별로 (티커, 이름, 금액, NAV 기여분) 을 모은다.
    by_fund: dict[str, list[tuple[str, str, float, float]]] = {}
    for r in rows_all[head + 1:]:
        if not r or len(r) <= i_mkt:
            continue
        fund = r[i_fund].strip()
        if not fund:
            continue
        mkt = _to_float(r[i_mkt])
        exp = _to_float(r[i_exp])
        amount = mkt if mkt is not None else exp
        if amount is None:
            continue
        by_fund.setdefault(fund, []).append(
            (r[i_sec].strip(), r[i_desc].strip(), amount, mkt or 0.0))

    out: dict[str, dict] = {}
    for fund, items in by_fund.items():
        nav = sum(m for _, _, _, m in items)
        if not nav:
            continue
        rows: list[dict] = []
        cash = swap = other = 0.0
        swap_names: list[str] = []
        for ticker, name, amount, _mkt in items:
            weight = round(amount / nav * 100.0, 4)
            if _RE_PROSHARES_RESIDUAL.search(name):
                continue  # NAV 잔여 항목 — 분모에만 들어간다
            if not ticker or ticker == "-":
                bucket = _classify_no_ticker(name)
                if bucket == "swap":
                    swap += weight
                    if name not in swap_names:
                        swap_names.append(name)
                elif bucket == "other":
                    other += weight
                else:
                    cash += weight
                continue
            if _is_cash_like(name):
                cash += weight
                continue
            rows.append({"ticker": ticker, "name": name, "weight": weight})
        out[fund] = _pack(rows, cash, swap, swap_names, other, as_of, False)
    return out
