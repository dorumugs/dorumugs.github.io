#!/usr/bin/env python3
"""미국 ETF 집계 — assets/etf/us.json.

지표 계산은 build_etf_theme.py 의 함수를 그대로 쓴다. 국내와 미국에 다른 잣대를
대면 두 화면을 나란히 놓을 수 없다.

국내와 다른 점만 여기서 다룬다.

  벤치마크    S&P500(SPX). 미국 ETF 를 코스피와 견주는 건 말이 안 된다.
  통화        가격·거래대금이 달러다. **거래대금 하한은 환율로 환산해 국내와
              같은 원화 기준(5억 원)으로 맞춘다** — 규칙을 하나만 둔다.
  환율        원화로 사는 사람의 수익률에는 환율이 섞인다. 달러 수익률과 원화
              수익률을 **둘 다** 낸다. 기초자산이 그대로여도 달러가 오르면
              원화 수익은 플러스다.
  폭          국내 테마와 연결이 없으므로 확인 불가. 국내 해외ETF 와 같은 처리다.
  괴리율      네이버가 해외 ETF 의 NAV 를 주지 않는다. 비워 둔다.
  기준일      미국 종가는 한국 시간 다음 날 새벽에 확정된다. 국내 기준일보다
              하루 이른 것이 정상이다.
"""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_etf_theme as b  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"
OUT = REPO / "assets" / "etf"

BARS_FILE = DATA / "us_bars.csv.gz"
HOLDINGS_FILE = DATA / "us_holdings.json.gz"
INDEX_KEY = "SPX"
HOLDINGS_TOP_N = 25

# 레버리지·인버스 판별. 국내는 이름에 '레버리지' 가 박혀 있지만 미국은 영어다.
LEV_WORDS = {
    "3X": 3.0, "2X": 2.0, "1.5X": 1.5,
    "ULTRAPRO": 3.0, "ULTRA": 2.0,
    "BULL 3X": 3.0, "BULL 2X": 2.0,
}
INVERSE_WORDS = ("BEAR", "SHORT", "INVERSE", "-1X", "ULTRASHORT")


def leverage_of(name: str) -> float:
    """이름에서 배수를 읽는다.

    'Direxion Daily Semiconductor Bull 3X ETF' -> 3.0
    'ProShares UltraPro Short QQQ'             -> -3.0
    'Direxion Daily AAPL Bear 1X ETF'          -> -1.0

    미국은 3배까지 있다. 국내(최대 2배)와 달라서 과열선·눌림 구간이 세 배로
    벌어진다 — 3배 ETF 에 1배 잣대를 대면 늘 '과열' 로 찍힌다.
    """
    upper = name.upper()
    multiplier = 1.0
    for word, value in LEV_WORDS.items():
        if word in upper:
            multiplier = max(multiplier, value)
    if "ULTRAPRO" in upper:
        multiplier = 3.0
    inverse = any(word in upper for word in INVERSE_WORDS)
    return -multiplier if inverse else multiplier


def load_bars() -> dict[str, dict[str, list]]:
    bars: dict[str, dict[str, list]] = {}
    with gzip.open(BARS_FILE, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bars.setdefault(row["symbol"], {})[row["date"]] = [
                float(row["close"]), int(row["volume"]),
                float(row["high"]), float(row["low"]),
            ]
    return bars


def fx_return(fx: dict[str, float], dates: list[str], span: int) -> float | None:
    """보유 기간 동안 원달러가 몇 % 움직였나.

    미국 장이 열린 날짜에 맞춰 본다. 환율 고시가 없는 날은 가장 가까운 이전 값을
    쓴다 — 공휴일이 서로 어긋나기 때문이다.
    """
    if len(dates) < span + 1:
        return None
    keys = sorted(fx)
    if not keys:
        return None

    def at(target: str) -> float | None:
        prior = [k for k in keys if k <= target]
        return fx[prior[-1]] if prior else None

    now, past = at(dates[-1]), at(dates[-span - 1])
    if not now or not past:
        return None
    return now / past - 1


def build_holdings(universe: list[dict]) -> dict:
    """assets/etf/us_holdings.json 에 쓸 모양으로 다듬는다.

    data/stocks/us_holdings.json.gz(collect_us_holdings.py 가 채운다)를 그대로
    내보내지 않는다 — 종목마다 30~60줄이라 192개 다 합치면 화면이 안 쓰는 무게가
    된다. 카드 하나가 보여줄 상위 25개(rows)만 자르고, cash·swap·swapNote·
    other·noTicker 는 통째로 넘긴다 — 이 넷은 배열이 아니라 스칼라라 무겁지
    않고, 레버리지 상품의 스왑 비중을 화면에서 보여주려면 다 있어야 한다.

    발행사를 아는데(ISSUER_BY_TICKER) 캐시에 없는 티커는 빈 항목으로 넣어 둔다 —
    그래야 화면이 "아직 못 받음" 과 "애초에 발행사를 모름" 을 구별할 수 있다.
    """
    import us_holdings_api as hapi  # noqa: PLC0415 — main() 에서만 쓰는 지연 임포트

    cache = b.read_json_gz(HOLDINGS_FILE) or {}
    out: dict[str, dict] = {}
    for item in universe:
        ticker = item["ticker"]
        if ticker not in hapi.ISSUER_BY_TICKER:
            continue  # 발행사를 모른다 — 화면에 항목 자체를 안 만든다
        entry = cache.get(ticker)
        if not entry:
            out[ticker] = {
                "issuer": hapi.ISSUER_BY_TICKER[ticker],
                "asOf": "", "rows": [], "cash": 0.0,
                "swap": 0.0, "swapNote": "", "other": 0.0, "noTicker": False,
                "count": 0,
            }
            continue
        rows = entry.get("rows") or []
        out[ticker] = {
            "issuer": entry.get("issuer", hapi.ISSUER_BY_TICKER[ticker]),
            "asOf": entry.get("asOf", ""),
            "rows": rows[:HOLDINGS_TOP_N],
            "cash": entry.get("cash", 0.0),
            "swap": entry.get("swap", 0.0),
            "swapNote": entry.get("swapNote", ""),
            "other": entry.get("other", 0.0),
            "noTicker": entry.get("noTicker", False),
            "count": len(rows),
        }
    return out


def main() -> int:
    universe = b.read_json_gz(DATA / "us_universe.json.gz")
    if not universe or not BARS_FILE.exists():
        print("미국 캐시가 없습니다. collect_us_etf.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1
    fx_rows = b.read_json_gz(DATA / "us_fx.json.gz") or []
    fx = {r["date"]: r["close"] for r in fx_rows}
    rate = fx[max(fx)] if fx else None
    if not rate:
        print("환율이 없어 원화 환산을 할 수 없습니다.", file=sys.stderr)
        return 1

    bars = load_bars()
    index = b.series_of(bars, INDEX_KEY)
    if len(index.closes) < b.LOOKBACK + 1:
        print("S&P500 일봉이 모자랍니다.", file=sys.stderr)
        return 1
    bench_r20 = b.pct_return(index.closes, b.LOOKBACK) or 0.0
    bench_swing = b.pct_return(index.closes, b.SWING_LOOKBACK) or 0.0
    base_date = index.dates[-1]

    # 시장 국면도 S&P500 로 본다. 두 지수가 없으니 하나로 판정한다.
    index_gap = b.ma_gap(index.closes)
    regime = {
        "label": "순풍" if (index_gap or 0) > 0 else "역풍",
        "note": ("S&P500 이 20일선 위입니다." if (index_gap or 0) > 0
                 else "S&P500 이 20일선 아래입니다. 규모를 줄이거나 쉬는 것을 먼저 고려하세요."),
    }

    rows = []
    for item in universe:
        code = item["code"]
        s = b.series_of(bars, code)
        if not s.closes:
            continue
        m = b.metrics_of(s)
        lev = leverage_of(item["name"])
        r20, r_swing = m["r20"], m["rSwing"]
        excess = None if r20 is None else r20 - bench_r20
        # 거래대금은 달러다. 국내와 같은 잣대를 대려면 원화로 바꿔야 한다.
        turnover_krw = None if m["turnover"] is None else m["turnover"] * rate
        grade, momentum, reasons = b.grade_of(
            bars_count=m["bars"], r20=r20, r5=m["r5"], straight=m["straight"],
            straight_prior=m["straightPrior"], vol=m["vol"], gap=m["gap"], dd=m["dd"],
            turnover=turnover_krw,
            beats_market=bool(excess is not None and excess > 0),
            breadth_ok=True, leverage=lev,
        )
        if grade in (b.GRADE_PULLBACK, b.GRADE_TREND):
            reasons.append("대세 확인 불가 — 국내 테마와 연결되지 않음")

        fx_move = fx_return(fx, s.dates, b.SWING_LOOKBACK)
        row = {
            "code": code,
            "ticker": item["ticker"],
            "name": item["name"],
            "exchange": item.get("exchange", ""),
            "isEtf": bool(item.get("etf", True)),
            "lev": lev,
            "grade": grade,
            "momentum": momentum,
            "reasons": reasons,
            "bench": "S&P500",
            "excess": None if excess is None else round(excess, 5),
            "turnoverKrw": None if turnover_krw is None else round(turnover_krw),
            "priceKrw": None if m["price"] is None else round(m["price"] * rate),
            "fxMove": None if fx_move is None else round(fx_move, 5),
            # 원화로 산 사람의 수익률. (1+달러수익)×(1+환율변동)−1
            "rSwingKrw": (None if (r_swing is None or fx_move is None)
                          else round((1 + r_swing) * (1 + fx_move) - 1, 5)),
            "premium": None,     # 네이버가 해외 ETF NAV 를 주지 않는다
            "breadth": None,
            "spark": b.spark(s.closes),
        }
        row.update(b.round_metrics(m))
        rows.append(row)

    order = {g: i for i, g in enumerate(b.GRADE_ORDER)}
    rows.sort(key=lambda r: (order.get(r["grade"], 99),
                             -(r["riskAdj"] if r["riskAdj"] is not None else -9)))
    grades = Counter(r["grade"] for r in rows)

    meta = {
        "baseDate": base_date,
        "fxRate": round(rate, 2),
        "fxDate": max(fx),
        "spx20": round(bench_r20, 5),
        "spxSwing": round(bench_swing, 5),
        "regime": regime,
        "count": len(rows),
        "swingLookback": b.SWING_LOOKBACK,
        "swingWeeks": round(b.SWING_LOOKBACK / 5),
        "grades": dict(grades),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "us.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    holdings = build_holdings(universe)
    (OUT / "us_holdings.json").write_text(
        json.dumps(holdings, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    have = sum(1 for v in holdings.values() if v["rows"] or v["cash"])

    lev3 = sum(1 for r in rows if abs(r["lev"]) >= 3)
    print(f"미국 ETF {len(rows)}개 · 기준 {base_date} · 환율 {rate:,.2f}원 ({max(fx)})")
    print(f"  S&P500 20일 {bench_r20:+.2%} · 국면 {regime['label']} · 3배 레버리지 {lev3}개")
    print("  등급: " + ", ".join(f"{g} {grades[g]}" for g in b.GRADE_ORDER if grades.get(g)))
    print(f"  assets/etf/us.json  {(OUT / 'us.json').stat().st_size / 1024:.0f}KB")
    print(f"  구성종목 {have}/{len(holdings)}개 발행사 확인 종목 중 보유 · "
          f"assets/etf/us_holdings.json {(OUT / 'us_holdings.json').stat().st_size / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
