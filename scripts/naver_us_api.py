"""네이버 해외증시 응답 파싱 — 순수 함수만.

국내 쪽(naver_stock_api.py)과 서버가 다르다. 이쪽은 전부 JSON 이고 **UTF-8** 이라
EUC-KR 지옥이 없다. 대신 다른 함정이 있다.

  거래소 접미사   같은 티커라도 상장 거래소에 따라 코드가 다르다.
                  TQQQ.O(나스닥) · SOXL.K(아멕스) · SPY(접미사 없음, NYSE Arca).
                  규칙이 없어서 **자동완성에 물어봐야** 한다.

  ETF 판별        자동완성 응답의 url 이 '/worldstock/etf/' 로 시작하면 ETF 다.
                  FNGO·VXX 처럼 이름에 ETF 가 있어도 실제로는 ETN 인 것이 있는데,
                  ETN 은 발행사 신용위험이 붙는 다른 물건이라 태그해서 구분한다.

  숫자가 문자열   환율 응답은 '1,417.20' 처럼 쉼표가 박힌 문자열로 온다.

  기준일          미국 종가는 한국 시간 다음 날 새벽에 확정된다. 저녁에 받으면
                  '어제 미국 장' 이 마지막이다. 국내 기준일과 하루 어긋나는 게
                  정상이다.

I/O 는 없다. 네트워크는 collect_us_etf.py 가 한다.
"""

from __future__ import annotations

import json

ETF_URL_MARK = "/worldstock/etf/"


def parse_seed(text: str) -> list[str]:
    """씨앗 목록에서 티커만 뽑는다. '#' 주석과 빈 줄은 버리고, 한 줄에 여러 개 OK."""
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for token in line.split():
            ticker = token.strip().upper()
            if ticker and ticker not in seen:
                seen.add(ticker)
                out.append(ticker)
    return out


def parse_autocomplete(raw: bytes, ticker: str) -> dict | None:
    """자동완성 응답에서 그 티커에 정확히 맞는 미국 종목 하나를 고른다.

    'SPY' 로 물으면 'Spyre Therapeutics' 같은 것도 같이 온다. 코드가 정확히
    일치하고 미국인 것만 취한다.

    돌려주는 것
      code    네이버가 쓰는 코드(reutersCode). 'TQQQ.O' · 'SPY' 처럼 접미사가
              있을 수도 없을 수도 있다. 일봉을 부를 때 이 값을 쓴다.
      name    정식 이름
      etf     ETF 인가 (아니면 ETN 등)
    """
    try:
        items = json.loads(raw.decode("utf-8", "replace")).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    for item in items:
        if str(item.get("code", "")).upper() != ticker.upper():
            continue
        if item.get("nationCode") != "USA":
            continue
        code = item.get("reutersCode") or item.get("code")
        if not code:
            continue
        return {
            "ticker": ticker.upper(),
            "code": code,
            "name": str(item.get("name", "")).strip(),
            "exchange": str(item.get("typeCode", "")),
            "etf": ETF_URL_MARK in str(item.get("url", "")),
        }
    return None


def parse_chart(raw: bytes) -> list[dict]:
    """해외 일봉.

    국내 siseJson 과 달리 진짜 JSON 이고 키 이름이 길다. 국내 파서와 같은
    모양으로 맞춰서 돌려준다 — 지표 계산 코드를 둘로 나누지 않기 위해서다.

    거래량이 없는 날(지수 등)은 0 으로 둔다. 종가가 0 이하인 줄은 버린다.
    """
    try:
        rows = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        try:
            close = float(row["closePrice"])
            date = str(row["localDate"])
        except (KeyError, TypeError, ValueError):
            continue
        if close <= 0 or len(date) != 8:
            continue
        out.append(
            {
                "date": date,
                "open": float(row.get("openPrice") or close),
                "high": float(row.get("highPrice") or close),
                "low": float(row.get("lowPrice") or close),
                "close": close,
                "volume": int(row.get("accumulatedTradingVolume") or 0),
            }
        )
    out.sort(key=lambda r: r["date"])
    return out


def parse_fx(raw: bytes) -> list[dict]:
    """원달러 일별 종가.

    '1,417.20' 처럼 쉼표가 박힌 문자열로 온다. 날짜도 '2026-08-11' 형식이라
    일봉과 맞추려면 하이픈을 떼야 한다.
    """
    try:
        rows = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        try:
            close = float(str(row["closePrice"]).replace(",", ""))
            date = str(row["localTradedAt"])[:10].replace("-", "")
        except (KeyError, TypeError, ValueError):
            continue
        if close <= 0 or len(date) != 8:
            continue
        out.append({"date": date, "close": close})
    out.sort(key=lambda r: r["date"])
    return out
