#!/usr/bin/env python3
"""등급이 실제로 다음 2주를 맞혔는지 채점한다 — assets/etf/backtest.json.

## 무엇을 하나

3.5년치 ETF 일봉을 놓고, 과거 각 시점에서 **그날까지의 데이터만으로** 지금과
똑같은 등급을 매긴 다음, 그 뒤 10 거래일에 실제로 어떻게 됐는지 센다.

규칙은 `build_etf_theme.py` 의 함수를 그대로 가져다 쓴다. 백테스트용으로 따로
구현하면 화면과 다른 걸 채점하게 되고, 그러면 검증이 아니라 딴 얘기가 된다.

## 이 백테스트가 주장할 수 있는 것과 없는 것

**주장할 수 있는 것** — ETF 가격에서만 나오는 판단(수익률·추세 곧기·변동성·
거래대금·손절선)이 과거 3.5년 동안 다음 10 거래일과 어떤 관계였는가.

**주장할 수 없는 것** — 아래 넷은 데이터가 없어서 못 고친다. 숫자를 읽을 때
반드시 같이 읽어야 한다.

  1. 구성종목 폭(breadth)      네이버가 과거 편입 명단을 안 준다. 오늘 명단으로
                               과거를 재면 '올라서 편입된' 종목이 과거에도 있던
                               걸로 잡혀 폭이 부풀려진다. 그래서 백테스트에서는
                               폭 조건을 **아예 빼고**(항상 통과) 계산한다.
                               화면의 등급은 여기에 폭 필터가 하나 더 붙으므로,
                               실제 성적은 이 결과와 다를 수 있다.
  2. 상장폐지 ETF              지금 살아있는 것만 있다. 사라진 건 대개 성적이
                               나빴다. 결과가 낙관 쪽으로 기운다.
  3. 신규상장 ETF              3.5년 내내 있던 게 아니라, 기간마다 표본이 다르다.
  4. 거래 비용                 스프레드·세금을 빼지 않은 값이다.

## 과최적화를 어떻게 막나

**이 스크립트는 채점만 한다.** 성적이 나쁘다고 임계값을 되맞추면 그 순간
백테스트는 검증이 아니라 곡선 맞추기가 된다.

규칙을 고칠 때는 순서를 지킨다 — **논리로 가설을 먼저 세우고 그다음에 채점받는다.**
실제로 두 번 그렇게 고쳤다. 손절이 보유기간 노이즈(σ×√H) 안쪽에 있다는 산수
오류를 찾아 1σ 바깥으로 옮겼고, 애초에 가설이 없던 '바닥다지기' 등급을 뺐다.
둘 다 성적표를 보고 숫자를 맞춘 게 아니라 근거를 먼저 세운 뒤 결과를 확인했다.

고치지 않고 남긴 것도 있다. 추세진행의 얇은 우위, 눌림매수가 시장을 못 이기는
것, 과열주의의 표본 부족은 그대로 화면에 쓴다.

표본도 부풀리지 않는다. 매일 관측하면서 10일 선행수익을 재면 이웃한 관측끼리
90% 가 겹쳐 표본이 실제보다 열 배 많아 보인다. 그래서 **10 거래일마다 한 번씩만**
평가한다(비중첩).
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_etf_theme as b  # noqa: E402
import naver_stock_api as api  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"
OUT = REPO / "assets" / "etf"

HORIZON = b.SWING_LOOKBACK          # 10 거래일 = 2주
STEP = HORIZON                       # 비중첩
MIN_HISTORY = b.LOOKBACK + b.SHORT_LOOKBACK + 5
RECENT_DAYS = 250                    # 최근 1년
MIN_SAMPLE = 30                      # 이보다 적으면 숫자를 내지 않는다

# 왕복 거래 비용 가정. ETF 는 증권거래세가 면제라 남는 건 위탁수수료와 스프레드다.
# 유동성이 받쳐주는 ETF 기준으로 0.10%p 로 잡았다. 정확한 값이 아니라 **얇은
# 우위가 비용을 넘는지** 를 가늠하려고 두는 잣대다. 유동성이 얕으면 이보다 크다.
ROUND_TRIP_COST = 0.001


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """승률의 95% 신뢰구간 (윌슨 구간).

    표본 86건으로 승률 60% 가 나왔다고 '60%' 라고 쓰면 안 된다. 그 정도 표본이면
    실제 값이 50~70% 어디든 될 수 있고, 그러면 기준선(58.8%)과 구별이 안 된다.
    구간을 같이 보여줘야 어떤 차이가 진짜인지 판단할 수 있다.

    단순 정규근사 대신 윌슨을 쓰는 이유는 비율이 0 이나 1 에 가까울 때 구간이
    범위를 벗어나지 않기 때문이다.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


class Trial(dict):
    """평가 한 건. dict 를 그대로 쓰되 이름만 붙인다."""


def evaluation_indexes(n: int, min_history: int = MIN_HISTORY,
                       horizon: int = HORIZON, step: int = STEP) -> list[int]:
    """평가 시점의 인덱스.

    앞으로 horizon 일이 남아 있어야 결과를 볼 수 있고, 뒤로 min_history 일이
    있어야 지표를 만들 수 있다. step 을 horizon 과 같게 두면 관측이 겹치지 않는다.
    """
    if n <= min_history + horizon:
        return []
    return list(range(min_history, n - horizon, step))


def forward_return(closes: list[float], i: int, horizon: int = HORIZON) -> float | None:
    if i + horizon >= len(closes) or closes[i] <= 0:
        return None
    return closes[i + horizon] / closes[i] - 1


def stop_outcome(lows: list[float], closes: list[float], i: int, stop: float | None,
                 horizon: int = HORIZON) -> tuple[bool, float | None]:
    """손절이 걸렸나, 그리고 손절을 지켰을 때 실제 수익은 얼마인가.

    앞으로 horizon 일 안에 저가가 손절선을 건드리면 거기서 나온 것으로 본다.
    갭으로 손절선을 훌쩍 뛰어넘어 시작하는 경우는 더 나쁘게 체결되므로,
    여기 숫자는 **실제보다 좋게 나온 값**이다.
    """
    fwd = forward_return(closes, i, horizon)
    if stop is None or fwd is None:
        return False, fwd
    window = lows[i + 1: i + 1 + horizon]
    if window and min(window) <= stop:
        return True, stop / closes[i] - 1
    return False, fwd


def summarize(trials: list[Trial]) -> dict | None:
    """한 무리의 평가 결과를 요약한다. 표본이 적으면 None — 없는 근거를 만들지 않는다."""
    if len(trials) < MIN_SAMPLE:
        return {"n": len(trials), "thin": True} if trials else None
    fwd = [t["fwd"] for t in trials]
    excess = [t["excess"] for t in trials if t["excess"] is not None]
    realized = [t["realized"] for t in trials if t["realized"] is not None]
    wins = sum(1 for v in fwd if v > 0)
    lo, hi = wilson_interval(wins, len(fwd))
    median_excess = statistics.median(excess) if excess else None
    return {
        "n": len(trials),
        "thin": False,
        "winRate": round(wins / len(fwd), 4),
        "winLo": round(lo, 4),
        "winHi": round(hi, 4),
        # 얇은 우위가 거래 비용을 넘는지. 넘지 못하면 통계적으로 유의해도 못 먹는다.
        "netExcess": None if median_excess is None else round(median_excess - ROUND_TRIP_COST, 5),
        "medianFwd": round(statistics.median(fwd), 5),
        "meanFwd": round(sum(fwd) / len(fwd), 5),
        "beatRate": (round(sum(1 for t in trials if t["excess"] is not None and t["excess"] > 0)
                           / len(excess), 4) if excess else None),
        "medianExcess": round(median_excess, 5) if median_excess is not None else None,
        "stopHitRate": round(sum(1 for t in trials if t["stopHit"]) / len(trials), 4),
        "medianRealized": round(statistics.median(realized), 5) if realized else None,
        "p25": round(statistics.quantiles(fwd, n=4)[0], 5),
        "p75": round(statistics.quantiles(fwd, n=4)[2], 5),
    }


def run(bars: dict, etfs: list[dict], bench_of: dict[str, str]) -> dict:
    kospi = b.series_of(bars, "KOSPI")
    kosdaq = b.series_of(bars, "KOSDAQ")
    calendar = kospi.dates
    if len(calendar) < MIN_HISTORY + HORIZON:
        raise SystemExit("지수 일봉이 모자라 백테스트를 못 합니다.")

    kospi_at = {d: i for i, d in enumerate(kospi.dates)}
    kosdaq_at = {d: i for i, d in enumerate(kosdaq.dates)}

    series: dict[str, b.Series] = {}
    index_at: dict[str, dict[str, int]] = {}
    for etf in etfs:
        s = b.series_of(bars, etf["code"])
        if len(s.closes) > MIN_HISTORY + HORIZON:
            series[etf["code"]] = s
            index_at[etf["code"]] = {d: i for i, d in enumerate(s.dates)}

    meta_of = {e["code"]: e for e in etfs}
    trials: list[Trial] = []
    eval_dates: list[str] = []

    for ci in evaluation_indexes(len(calendar)):
        date = calendar[ci]
        eval_dates.append(date)

        bench_fwd = {}
        bench_r20 = {}
        for label, s, at in (("KOSPI", kospi, kospi_at), ("KOSDAQ", kosdaq, kosdaq_at)):
            i = at.get(date)
            if i is None or i < b.LOOKBACK:
                bench_r20[label] = None
                bench_fwd[label] = None
                continue
            bench_r20[label] = b.pct_return(s.closes[: i + 1], b.LOOKBACK)
            bench_fwd[label] = forward_return(s.closes, i)
        if bench_r20["KOSPI"] is None:
            continue

        # 1차: 이 날짜의 지표를 전부 구해 둔다. 해외 ETF 의 분류내 백분위를
        # 매기려면 그날의 단면 전체가 필요하다.
        snapshot = []
        by_tab: dict[int, list[float]] = defaultdict(list)
        for code, s in series.items():
            i = index_at[code].get(date)
            if i is None or i < MIN_HISTORY or i + HORIZON >= len(s.closes):
                continue
            past = b.Series(
                s.dates[: i + 1], s.closes[: i + 1], s.volumes[: i + 1],
                s.highs[: i + 1], s.lows[: i + 1],
            )
            m = b.metrics_of(past)
            if m["r20"] is None or m["r10"] is None:
                continue
            snapshot.append((code, i, s, m))
            by_tab[meta_of[code]["tab"]].append(m["r20"])

        # 2차: 등급을 매기고 앞으로 10일을 본다.
        for code, i, s, m in snapshot:
            etf = meta_of[code]
            domestic = etf["tab"] in b.DOMESTIC_TABS
            bench = bench_of.get(code, "KOSPI")
            if domestic:
                base = bench_r20.get(bench)
                beats = base is not None and m["r20"] > base
                fwd_bench = bench_fwd.get(bench)
            else:
                pct = b.percentile_rank(m["r20"], by_tab[etf["tab"]])
                beats = pct >= b.FOREIGN_PCT_OK
                fwd_bench = None

            grade, momentum, _ = b.grade_of(
                bars_count=m["bars"],
                r20=m["r20"],
                r5=m["r5"],
                straight=m["straight"],
                straight_prior=m["straightPrior"],
                vol=m["vol"],
                gap=m["gap"],
                dd=m["dd"],
                turnover=m["turnover"],
                beats_market=beats,
                # 폭은 과거 명단이 없어 검증할 수 없다. 조건에서 뺀다.
                breadth_ok=True,
                leverage=api.leverage_of(etf["name"]),
            )
            fwd = forward_return(s.closes, i)
            if fwd is None:
                continue
            hit, realized = stop_outcome(s.lows, s.closes, i, m["stop"])
            trials.append(Trial(
                date=date, code=code, tab=etf["tab"], grade=grade, momentum=momentum,
                fwd=fwd, excess=None if fwd_bench is None else fwd - fwd_bench,
                stopHit=hit, realized=realized,
                # 매물대 지표도 같이 채점한다. 새로 넣은 숫자가 실제로 다음 2주와
                # 관계가 있는지 묻지 않으면, 그럴듯해 보인다는 이유만으로 화면에
                # 남게 된다.
                overhead=m["overhead"],
            ))

    recent_from = calendar[-RECENT_DAYS] if len(calendar) > RECENT_DAYS else calendar[0]
    recent = [t for t in trials if t["date"] >= recent_from]

    def by_overhead(rows: list[Trial]) -> dict:
        """'위에 물린 물량' 구간별 성적.

        구간을 20% 단위로 나눈 건 읽기 쉬우라고 그런 것이지 성적이 잘 갈리는
        지점을 찾은 게 아니다. **이 결과로 규칙을 바꾸지 않는다** — 지표가
        실제로 뜻이 있는지 재기만 한다.
        """
        edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        out = {}
        for lo, hi in edges:
            bucket = [t for t in rows
                      if t["overhead"] is not None and lo <= t["overhead"] < hi]
            summary = summarize(bucket)
            if summary:
                out[f"{int(lo * 100)}-{min(int(hi * 100), 100)}%"] = summary
        return out

    def by_grade(rows: list[Trial]) -> dict:
        buckets: dict[str, list[Trial]] = defaultdict(list)
        for t in rows:
            buckets[t["grade"]].append(t)
        out = {}
        for grade in b.GRADE_ORDER:
            summary = summarize(buckets.get(grade, []))
            if summary:
                out[grade] = summary
        return out

    return {
        "asOf": calendar[-1],
        "from": eval_dates[0] if eval_dates else None,
        "to": eval_dates[-1] if eval_dates else None,
        "horizon": HORIZON,
        "step": STEP,
        "evalDates": len(eval_dates),
        "trials": len(trials),
        "etfs": len(series),
        "recentFrom": recent_from,
        "all": {"baseline": summarize(trials), "byGrade": by_grade(trials),
                "byOverhead": by_overhead(trials)},
        "recent": {"baseline": summarize(recent), "byGrade": by_grade(recent),
                   "byOverhead": by_overhead(recent)},
        "cost": ROUND_TRIP_COST,
        "limits": [
            "구성종목 폭(breadth)은 과거 편입 명단이 없어 조건에서 뺐습니다. "
            "화면의 등급에는 폭 필터가 하나 더 붙으므로 실제 성적은 다를 수 있습니다.",
            "지금 상장돼 있는 ETF 만 들어 있습니다. 상장폐지된 것은 대개 성적이 "
            "나빴으므로 결과가 낙관 쪽으로 기웁니다.",
            "신규상장 ETF 때문에 기간마다 표본 구성이 다릅니다.",
            "스프레드·세금 같은 거래 비용을 빼지 않았습니다.",
            "손절은 장중 저가가 손절선에 닿으면 거기서 체결된 것으로 봅니다. "
            "갭으로 뛰어넘으면 더 나쁘게 체결되므로 실제보다 좋게 나온 값입니다.",
            "성적이 나쁘다고 임계값을 되맞추지 않았습니다. 규칙을 고칠 때는 논리로 "
            "가설을 먼저 세우고 그다음에 채점받았습니다 — 손절을 보유기간 1σ 바깥으로 "
            "옮긴 것과 '바닥다지기' 등급을 뺀 것이 그렇게 나온 결과입니다.",
        ],
    }


def main() -> int:
    etfs = b.read_json_gz(DATA / "etfs.json.gz")
    if not etfs:
        print("data/stocks 캐시가 없습니다. collect_stocks.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1
    bars = b.load_bars()

    # 벤치마크(코스피/코스닥)는 현재 구성종목 기준이다. 반도체 ETF 가 3년 사이
    # 코스닥으로 옮겨가지는 않으므로 영향이 작다고 보고 그대로 쓴다.
    live = OUT / "etfs.json"
    bench_of = {}
    if live.exists():
        for row in json.loads(live.read_text(encoding="utf-8")):
            if row.get("bench"):
                bench_of[row["code"]] = row["bench"]

    result = run(bars, etfs, bench_of)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backtest.json").write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    print(f"{result['from']} ~ {result['to']} · 평가일 {result['evalDates']}회 "
          f"· ETF {result['etfs']}개 · 관측 {result['trials']:,}건 (비중첩)")
    base = result["all"]["baseline"]
    print(f"\n기준선(전체 평균적인 ETF): 승률 {base['winRate']:.1%} "
          f"· 중위 {base['medianFwd']:+.2%} · 손절 걸림 {base['stopHitRate']:.1%}")
    print(f"\n{'등급':10s}{'표본':>7s}{'승률 (95% 구간)':>20s}{'중위 2주':>10s}"
          f"{'초과':>8s}{'비용후':>8s}{'손절걸림':>9s}{'손절적용':>10s}")
    for label, block in (("전체 3.5년", result["all"]), ("최근 1년", result["recent"])):
        print(f"-- {label}")
        base = block["baseline"]
        if base and not base.get("thin"):
            print(f"{'기준선':10s}{base['n']:>7,}"
                  f"{base['winRate']:>9.1%} ({base['winLo']:.1%}~{base['winHi']:.1%})"
                  f"{base['medianFwd']:>+10.2%}")
        for grade, s in block["byGrade"].items():
            if s.get("thin"):
                print(f"{grade:10s}{s['n']:>7,}   표본 부족")
                continue
            ex = "—" if s["medianExcess"] is None else f"{s['medianExcess']:+.2%}"
            net = "—" if s["netExcess"] is None else f"{s['netExcess']:+.2%}"
            print(f"{grade:10s}{s['n']:>7,}"
                  f"{s['winRate']:>9.1%} ({s['winLo']:.1%}~{s['winHi']:.1%})"
                  f"{s['medianFwd']:>+10.2%}{ex:>8s}{net:>8s}"
                  f"{s['stopHitRate']:>9.1%}{s['medianRealized']:>+10.2%}")
    print(f"\n{'위에 물린 물량':14s}{'표본':>7s}{'승률 (95% 구간)':>20s}{'중위 2주':>10s}{'초과':>8s}")
    for band, s in result["all"]["byOverhead"].items():
        if s.get("thin"):
            print(f"{band:14s}{s['n']:>7,}   표본 부족")
            continue
        ex = "—" if s["medianExcess"] is None else f"{s['medianExcess']:+.2%}"
        print(f"{band:14s}{s['n']:>7,}"
              f"{s['winRate']:>9.1%} ({s['winLo']:.1%}~{s['winHi']:.1%})"
              f"{s['medianFwd']:>+10.2%}{ex:>8s}")

    print(f"\n  assets/etf/backtest.json  {(OUT / 'backtest.json').stat().st_size / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
