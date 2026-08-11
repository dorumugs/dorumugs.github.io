"""네이버 금융 응답 파싱 — 순수 함수만.

여섯 군데를 읽는다. 전부 인증키가 없다.

  테마 목록      finance.naver.com/sise/theme.naver?page=N          265개(7페이지)
  업종 목록      finance.naver.com/sise/sise_group.naver?type=upjong 79개
  그룹 구성종목  finance.naver.com/sise/sise_group_detail.naver      테마 평균 24 · 업종 평균 56
  시장 구분      finance.naver.com/sise/sise_market_sum.naver        코스피 50 · 코스닥 37페이지
  ETF 목록       finance.naver.com/api/sise/etfItemList.nhn          1,160개
  ETF 구성종목   navercomp.wisereport.co.kr/v2/ETF/index.aspx        var CU_data
  일봉           api.finance.naver.com/siseJson.naver                3.5년치가 요청 1번

밟은 지뢰를 적어 둔다.

  인코딩   같은 네이버인데 서버마다 다르다. finance.naver.com 은 HTML 도 JSON 도
           EUC-KR 이고, navercomp.wisereport.co.kr 만 UTF-8 이다. 그래서 이 모듈의
           함수는 전부 str 이 아니라 **bytes 를 받아** 각자 아는 인코딩으로 푼다.
           호출부가 인코딩을 기억하게 두면 언젠가 틀린다.

  종목코드 ETF 구성종목(CU_data)에는 **종목코드가 없다.** 종목명만 온다. 그래서
           테마 구성종목과 이어붙이려면 이름을 맞대야 한다. normalize_name() 이
           그 일을 한다.

  일봉 응답 JSON 처럼 보이지만 JSON 이 아니다. 작은따옴표를 쓰고 헤더 행이 섞여
           있어 json.loads 가 실패한다. 정규식으로 행만 뽑는다.

  등락률   '+3.11%' · '-1.20%' · '0.00%' 로 오고, 값이 없으면 아예 빈 칸이다.

I/O 는 없다. 네트워크는 collect_stocks.py 가 한다.
"""

from __future__ import annotations

import json
import re

ENCODING_FINANCE = "euc-kr"
ENCODING_WISEREPORT = "utf-8"

# 네이버가 붙이는 종목 링크. 목록·구성종목·주도주 어디서나 같은 모양이다.
RE_ITEM_LINK = re.compile(r'/item/main\.naver\?code=([0-9A-Z]{6})"[^>]*>([^<]+)</a>')

RE_THEME_ROW = re.compile(
    r'<td class="col_type1"><a href="/sise/sise_group_detail\.naver\?type=theme&no=(\d+)">'
    r"(.*?)</a></td>(.*?)</tr>",
    re.S,
)

RE_UPJONG_ROW = re.compile(
    r'<td style="padding-left:10px;"><a href="/sise/sise_group_detail\.naver\?type=upjong&no=(\d+)">'
    r"(.*?)</a></td>(.*?)</tr>",
    re.S,
)

RE_NUMBER_CELL = re.compile(r'<td class="number[^"]*">(.*?)</td>', re.S)
RE_TAG = re.compile(r"<[^>]+>")

# 시가총액 목록은 종목명 링크에만 class="tltle" 이 붙는다. 이걸로 골라야
# 페이지 아래 '다음' 링크나 광고에 섞인 종목 링크를 안 줍는다.
RE_MARKET_SUM_ITEM = re.compile(r'/item/main\.naver\?code=([0-9A-Z]{6})"\s+class="tltle"')

RE_CU_DATA = re.compile(r"var\s+CU_data\s*=\s*(\{.*?\});", re.S)

# 일봉. ["20260601", 319500, 354500, 319500, 349000, 45052488, 48.3]
RE_SISE_ROW = re.compile(
    r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+)'
)

# ETF 이름에서 배수를 읽는다. 국내 상장은 최대 2배까지다.
# 곱하기 기호가 X·x·배 로 섞여 오고, '2X레버리지' 처럼 붙어 오기도 한다.
RE_MULTIPLIER = re.compile(r"(\d)\s*[Xx배]")

# 이름 대조용. 우선주·괄호주석·공백·가운뎃점을 털어낸다.
RE_NAME_NOISE = re.compile(r"[\s()\[\]·.,'\"-]")


def _text(html: str) -> str:
    """태그를 걷어내고 공백을 하나로 줄인다."""
    return " ".join(RE_TAG.sub(" ", html).split())


def parse_pct(value: str | None) -> float | None:
    """'+3.11%' -> 3.11 · '' -> None.

    빈 칸을 0 으로 바꾸지 않는다. '안 움직였다' 와 '모른다' 는 다른 사실이다.
    """
    if value is None:
        return None
    text = value.strip().replace("%", "").replace(",", "").replace("+", "")
    if not text or text in {"-", "--"}:
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    """'1,234' -> 1234 · '' -> None."""
    if value is None:
        return None
    text = value.strip().replace(",", "").replace("+", "")
    if not text or text in {"-", "--"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_name(name: str) -> str:
    """종목명 대조용 정규화.

    ETF 구성종목(CU_data)이 종목코드를 안 주기 때문에 이름으로 맞대야 한다.
    공백·괄호·가운뎃점·마침표를 털고 대문자로 올린다.

      'LG에너지솔루션'  -> 'LG에너지솔루션'
      'POSCO 홀딩스'    -> 'POSCO홀딩스'
      '삼성전자(우)'    -> '삼성전자우'

    우선주의 '(우)' 는 붙여서 남긴다. 보통주와 다른 종목이라 합치면 안 된다.
    """
    return RE_NAME_NOISE.sub("", name).upper()


def parse_theme_list(raw: bytes) -> list[dict]:
    """테마 목록 한 페이지.

    열 순서는 테마명 · 전일대비 · 최근3일 · 상승 · 보합 · 하락 · 주도주1 · 주도주2 다.
    주도주는 등락 아이콘 <img> 가 앞에 붙어 오므로 링크만 뽑는다.
    """
    html = raw.decode(ENCODING_FINANCE, "replace")
    rows = []
    for no, name, rest in RE_THEME_ROW.findall(html):
        cells = [_text(c) for c in RE_NUMBER_CELL.findall(rest)]
        leaders = [
            {"code": code, "name": _text(nm)} for code, nm in RE_ITEM_LINK.findall(rest)
        ]
        rows.append(
            {
                "type": "theme",
                "no": no,
                "name": _text(name),
                "change_rate": parse_pct(cells[0] if len(cells) > 0 else None),
                "change_3d": parse_pct(cells[1] if len(cells) > 1 else None),
                "up": parse_int(cells[2] if len(cells) > 2 else None),
                "flat": parse_int(cells[3] if len(cells) > 3 else None),
                "down": parse_int(cells[4] if len(cells) > 4 else None),
                "leaders": leaders,
            }
        )
    return rows


def parse_upjong_list(raw: bytes) -> list[dict]:
    """업종 목록.

    열 순서는 업종명 · 전일대비 · 전체 · 상승 · 보합 · 하락 이다.
    테마와 달리 최근3일이 없고 전체 종목수가 있다.
    """
    html = raw.decode(ENCODING_FINANCE, "replace")
    rows = []
    for no, name, rest in RE_UPJONG_ROW.findall(html):
        cells = [_text(c) for c in RE_NUMBER_CELL.findall(rest)]
        rows.append(
            {
                "type": "upjong",
                "no": no,
                "name": _text(name),
                "change_rate": parse_pct(cells[0] if len(cells) > 0 else None),
                "total": parse_int(cells[1] if len(cells) > 1 else None),
                "up": parse_int(cells[2] if len(cells) > 2 else None),
                "flat": parse_int(cells[3] if len(cells) > 3 else None),
                "down": parse_int(cells[4] if len(cells) > 4 else None),
            }
        )
    return rows


def parse_group_detail(raw: bytes) -> list[dict]:
    """테마·업종 구성종목.

    한 종목이 표 안에서 두 번 링크된다 — 종목명 칸과 '테마 편입 사유' 툴팁이다.
    코드로 중복을 걷어내되 등장 순서(시가총액 순)는 지킨다.
    """
    html = raw.decode(ENCODING_FINANCE, "replace")
    seen: dict[str, str] = {}
    for code, name in RE_ITEM_LINK.findall(html):
        text = _text(name)
        if code not in seen and text:
            seen[code] = text
    return [{"code": code, "name": name} for code, name in seen.items()]


def parse_market_codes(raw: bytes) -> list[str]:
    """시가총액 목록 한 페이지의 종목코드.

    코스피(sosok=0)·코스닥(sosok=1) 소속을 가려내는 데 쓴다. 그룹 상세와 ETF
    구성종목이 시장을 안 알려줘서 벤치마크를 못 고르기 때문에 따로 받는다.
    """
    html = raw.decode(ENCODING_FINANCE, "replace")
    out: list[str] = []
    for code in RE_MARKET_SUM_ITEM.findall(html):
        if code not in out:
            out.append(code)
    return out


def parse_etf_list(raw: bytes) -> list[dict]:
    """ETF 전종목.

    etfTabCode 는 1 국내시장지수 · 2 국내업종테마 · 3 국내파생 · 4 해외주식
    · 5 원자재 · 6 채권 · 7 기타혼합 이다.

    amonut(거래대금·백만원)은 오타가 아니라 응답 필드 이름 그대로다.
    marketSum 은 억원이다.
    """
    data = json.loads(raw.decode(ENCODING_FINANCE, "replace"))
    items = data.get("result", {}).get("etfItemList", [])
    out = []
    for it in items:
        code = str(it.get("itemcode", "")).strip()
        name = str(it.get("itemname", "")).strip()
        if not code or not name:
            continue
        out.append(
            {
                "code": code,
                "name": name,
                "tab": int(it.get("etfTabCode") or 0),
                "price": it.get("nowVal"),
                "change_rate": it.get("changeRate"),
                "nav": it.get("nav"),
                "volume": it.get("quant"),
                "amount_mn": it.get("amonut"),
                "market_cap_100m": it.get("marketSum"),
                "return_3m": it.get("threeMonthEarnRate"),
            }
        )
    return out


def parse_etf_holdings(raw: bytes) -> list[dict]:
    """ETF 구성종목 — var CU_data 안의 JSON.

    이 서버만 UTF-8 이다. 종목코드가 없고 종목명(STK_NM_KOR)만 온다.
    비중(ETF_WEIGHT)은 %, 합이 100 이 안 될 수 있다 — 현금·선물이 빠져 있다.
    """
    text = raw.decode(ENCODING_WISEREPORT, "replace")
    match = RE_CU_DATA.search(text)
    if not match:
        return []
    try:
        grid = json.loads(match.group(1)).get("grid_data", [])
    except json.JSONDecodeError:
        return []
    out = []
    for row in grid:
        name = str(row.get("STK_NM_KOR", "")).strip()
        if not name:
            continue
        try:
            weight = float(row.get("ETF_WEIGHT") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        out.append({"name": name, "weight": round(weight, 4), "date": str(row.get("TRD_DT", ""))})
    return out


def parse_sise_json(raw: bytes) -> list[dict]:
    """일봉.

    JSON 처럼 보이지만 아니다. 작은따옴표에 헤더 행이 섞여 있어 json.loads 가
    실패한다. 정규식으로 데이터 행만 뽑는다.

    지수(KOSPI·KOSDAQ)도 같은 모양으로 오고, 종가가 소수점이라 float 로 받는다.
    오름차순(과거→현재)으로 오지만 믿지 않고 날짜로 다시 정렬한다.
    """
    text = raw.decode("utf-8", "replace")
    rows = []
    for date, op, hi, lo, cl, vol in RE_SISE_ROW.findall(text):
        close = float(cl)
        if close <= 0:
            continue
        rows.append(
            {
                "date": date,
                "open": float(op),
                "high": float(hi),
                "low": float(lo),
                "close": close,
                "volume": int(vol),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def leverage_of(name: str) -> float:
    """ETF 이름에서 배수를 읽는다.

      'KODEX 레버리지'          ->  2.0
      'KODEX 200선물인버스2X'   -> -2.0
      'KODEX 인버스'            -> -1.0
      'TIGER 반도체TOP10'       ->  1.0

    왜 필요한가. 2배 ETF 는 기초자산이 12.5% 만 올라도 25% 가 된다. 과열선이나
    눌림 구간 같은 임계값을 1배와 같은 값으로 대면 레버리지는 늘 '과열' 로
    찍힌다. 판정할 때 이 배수로 임계값을 늘려 준다.

    국내 상장은 최대 2배다. 3배는 KRX 에 없다.
    """
    upper = name.upper()
    inverse = "인버스" in name
    multiplier = 1.0
    match = RE_MULTIPLIER.search(upper.replace("TOP", " "))
    if match:
        multiplier = float(match.group(1))
    elif "레버리지" in name:
        multiplier = 2.0
    return -multiplier if inverse else multiplier


def is_intraday(now) -> bool:
    """지금 받으면 마지막 일봉이 '종가' 가 아니라 '장중 시세' 인가.

    siseJson 은 장중에도 오늘 날짜 행을 준다. 그런데 그 종가 칸에 들어 있는 건
    확정 종가가 아니라 **그 순간의 현재가**다. 모르고 받으면 장중 값이 종가인
    척 캐시에 박히고, 20일 수익률이 몇 분마다 달라진다.

    실제로 겪었다. 11:09 에 받은 KODEX 200 의 '20260811 종가' 가 99,100 이었는데
    9분 뒤에 다시 받으니 98,880 이었다.

    한국거래소 정규장은 09:00~15:30 이다. 종가 단일가와 반영까지 감안해 15:40
    이후를 확정으로 본다. 주말은 애초에 새 거래일이 없다.
    """
    if now.weekday() >= 5:
        return False
    return (now.hour, now.minute) < (15, 40)


def is_hedged(name: str) -> bool:
    """환헤지 여부. 이름 끝의 '(H)' 로 판별한다.

    해외 ETF 는 원화로 표시되므로 헤지가 없으면 환율이 수익률에 섞인다.
    기초자산이 그대로여도 달러가 오르면 ETF 가 오른다.
    """
    return "(H)" in name.upper()
