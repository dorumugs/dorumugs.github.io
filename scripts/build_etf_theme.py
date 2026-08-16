#!/usr/bin/env python3
"""테마·업종·ETF 20일 모멘텀 집계 — assets/etf/*.json 을 굽는다.

읽는 것은 data/stocks/ 캐시뿐이다. 네트워크를 쓰지 않는다.

무엇을 계산하나

  레이어 1  ETF 자체     20일·5일 수익률, 추세 곧기, 변동성, 20일선 이격,
                         20일 고점 대비, 20일 중앙 거래대금
  레이어 2  테마·업종    구성종목 **중위** 20일 수익률, 폭, 21일 지수
  레이어 3  시장         KOSPI·KOSDAQ 20일 대비 초과 (해외는 같은 분류 백분위)

왜 평균이 아니라 중위값인가

  테마 구성종목 하나가 +300% 가면 평균은 테마 전체가 오른 것처럼 보인다. 하지만
  그건 대세가 아니라 개별 이슈다. 중위값은 거기 안 흔들린다. 여기에 폭(20일
  수익률이 양인 종목 비율)을 붙이면 '이 판이 통째로 오르는가' 가 한 숫자로 나온다.

왜 가중합 점수가 아니라 규칙인가

  근거 없이 정한 가중치가 숫자의 권위를 빌리기 때문이다. 대신 조건을 그대로
  드러내는 규칙을 쓰고, 어떤 조건에서 갈렸는지 화면에 같이 내보낸다.
  채점은 backtest_etf.py 가 따로 하되, **그 결과로 임계값을 되맞추지 않는다.**

레버리지 임계값

  2배 ETF 는 기초자산이 12.5% 만 올라도 25% 가 된다. 1배와 같은 잣대를 대면
  레버리지는 늘 '과열' 로 찍힌다. 그래서 배수만큼 임계값을 늘린다.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import naver_stock_api as api  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "stocks"
OUT = REPO / "assets" / "etf"

LOOKBACK = 20
SHORT_LOOKBACK = 5
# 보유 목표 기간. 처음엔 2주(10 거래일)로 뒀다가 **8주(40 거래일)** 로 늘렸다.
#
# 왜 늘렸나. 가설을 먼저 세웠다 — 왕복 비용 0.10%p 를 10일마다 내면 얇은 우위가
# 다 깎인다. 신호가 같다면 오래 들고 갈수록 비용을 나눠 내게 된다. 재봤더니
# 그것보다 결과가 좋았다. '추세진행 + 물린물량 20% 미만' 기준으로
#
#   보유 10일  건당 초과 +0.22%  비용후 +0.12%  연 환산 +3.0%
#   보유 20일           +0.52%         +0.42%         +5.2%
#   보유 40일           +1.43%         +1.33%         +8.3%
#   보유 60일           +2.23%         +2.13%         +8.9%
#
# 건당 초과가 기간보다 빠르게 늘었다(4배 기간에 6.5배). 비용 분산만이 아니라
# **신호 자체가 긴 구간에서 더 잘 듣는다**는 뜻이고, 모멘텀 문헌과도 맞는다.
#
# 40 과 60 중 어느 쪽이 나은지는 **못 가린다** — 겹치지 않는 평가 시점이 각각
# 21회·14회뿐이라 그 차이를 주장할 표본이 없다. 방향만 믿고 40 을 골랐다.
#
# 이건 임계값 튜닝이 아니다. 보유 기간은 규칙 안의 파라미터가 아니라 **투자자가
# 정하는 조건**이라, 여러 값을 재서 고르는 게 맞다. 판정 규칙(LOOKBACK 20,
# SHORT_LOOKBACK 5)은 그대로 뒀다 — 측정할 때도 그대로였기 때문이다.
SWING_LOOKBACK = 40
ATR_SPAN = 14
# 손절폭. 2×ATR 은 하루 평균 등락의 두 배라, 평범한 출렁임에는 안 털리고
# 추세가 깨지면 잡히는 자리다.
ATR_STOP_MULT = 2.0
SPARK_POINTS = 21
# 지수를 만들 거래일 수. 눌림 이전 추세(5일 전에서 끝나는 20일 창)를 재려면
# 스파크라인용 21일로는 모자라 26일이 필요하다.
CALENDAR_POINTS = LOOKBACK + SHORT_LOOKBACK + 1

# 폭 스트립. 날짜별 '오른 구성종목 비율' 을 30 거래일치 늘어놓는다.
#
# 20일 상승 비율은 한 숫자라 **언제부터** 대세였는지를 못 말한다. 20일 내내
# 꾸준히 올라 64% 인 테마와 18일 죽어 있다가 이틀 급등해 64% 가 된 테마가
# 화면에서 똑같이 보인다. 스윙 진입 시점을 정할 때 이 둘은 다른 물건이다.
#
# 이 배열은 등급 계산에 **쓰지 않는다.** 달력도 따로 만든다 — CALENDAR_POINTS
# 를 늘리면 group_index 가 길어지고 그게 straightness 를 거쳐 등급을 조용히
# 바꾼다. 판정 규칙은 건드리지 않는다.
STRIP_SPAN = 30
# 그날 양쪽 종가가 다 있는 종목이 이보다 적으면 비율을 내지 않는다.
# 2종목 중 2종목이 올랐다고 '100% 대세' 라고 칠할 수는 없다.
STRIP_MIN_VALID = 3

# --- 판정 임계값 ------------------------------------------------------------
# 검증된 최적값이 아니라 합리적 출발점이다. 근거는 _dev/specs 문서에 적어 뒀다.
BREADTH_OK = 0.60          # 구성종목 60% 이상이 올라야 '대세'
STRAIGHT_OK = 0.50         # 추세 곧기(R²) 하한
VOL_FLOOR = 0.08           # 연율 8% 미만이면 금리형. CD금리 ETF 는 0.12% 다
TURNOVER_FLOOR = 5e8       # 20일 중앙 거래대금 5억 원
OVERHEAT_R20 = 0.25        # 20일 +25% 이상이면 과열 후보
PULLBACK_R5_LOW = -0.05    # 눌림 구간의 최근 5일 하한
PULLBACK_R5_HIGH = 0.01    # 상한
PULLBACK_DD_DEEP = -0.10   # 20일 고점 대비 이보다 깊으면 추세 이탈
PULLBACK_DD_SHALLOW = -0.03  # 이보다 얕으면 아직 안 눌린 것
FOREIGN_PCT_OK = 60        # 해외·채권 등은 같은 분류 상위 40% 안

# 테마 상세에 'ETF 로 살 수 있는가' 를 붙일 때, 편입비중 합이 이만큼은 돼야
# 그 테마에 실제로 노출됐다고 본다. 2% 담은 걸 '반도체 ETF' 라고 부를 수는 없다.
GROUP_ETF_MIN_WEIGHT = 10.0
GROUP_ETF_TOP = 5          # 시총 상위 N · 거래대금 상위 N
# 노출이 이보다 낮으면 '부분 노출' 로 표시한다. 목록에서 빼지는 않는다 —
# 없는 것보다 낫고, 얼마나 담고 있는지는 숫자로 보여 주기 때문이다.
GROUP_ETF_PARTIAL = 30.0

# 국내 시장지수 ETF(tab 1)는 그룹별 ETF 목록에서 뺀다.
#
# 코스피200 은 지금 반도체 비중이 60% 라 '반도체' 테마에 노출 59.9% 로 잡히고,
# 시가총액이 크니 거의 모든 큰 테마의 1위로 올라온다. 계산은 맞지만 **지수를
# 사는 건 범주상 테마를 사는 게 아니다.** 테마 트레이더에게는 소음이다.
# 파생(tab 3)은 남긴다 — 'TIGER 반도체TOP10레버리지' 처럼 진짜 테마 레버리지가
# 거기 있고, 지수 레버리지는 노출 40% 로 구별된다.
GROUP_ETF_EXCLUDE_TABS = {1}

# 매물대. 120 거래일(약 6개월)이면 물린 사람이 생길 만큼 길고, 지금 판단에
# 쓸 만큼 최근이다. 종목 캐시가 130일이라 그 이상은 어차피 못 본다.
PROFILE_SPAN = 120
PROFILE_BINS = 40
YEAR_SPAN = 250            # 52주 고점

# 후보 간 상관. 60 거래일이면 최근 국면을 반영하면서도 표본이 충분하다.
# 상관행렬은 후보군에만 계산한다 — 1,160개 전부면 67만 쌍이라 실을 수 없다.
CORR_SPAN = 60
CORR_MAX_CANDIDATES = 160

DOMESTIC_TABS = {1, 2, 3}

TAB_NAMES = {
    1: "국내 시장지수",
    2: "국내 업종·테마",
    3: "국내 파생",
    4: "해외 주식",
    5: "원자재",
    6: "채권",
    7: "기타·혼합",
}

GRADE_UNKNOWN = "판정불가"
GRADE_RATE = "금리형"
GRADE_ILLIQUID = "유동성부족"
GRADE_OVERHEAT = "과열주의"
GRADE_PULLBACK = "눌림매수"
GRADE_TREND = "추세진행"
GRADE_WEAK = "약세"

# '바닥다지기'(20일 마이너스인데 5일 플러스이고 20일선 위)를 뺐다.
#
# 성적이 나빠서가 아니다 — 그랬으면 다른 등급도 성적을 보고 고쳐야 한다.
# **애초에 가설이 없던 규칙**이라 뺐다. 추세진행·눌림매수는 모멘텀 지속이라는
# 근거가 있지만, 저점에서 반등하는 걸 산다는 건 이 화면 전체의 전제(오르는 걸
# 산다)와 정면으로 모순된다. 근거 없이 만들어 놓고 '관찰 대상' 이라는 이름으로
# 기회처럼 보이게 한 것이 문제였다.
#
# 참고로 채점 결과도 나빴다 — 승률 43.6%, 중위 −0.81% 로 기준선(58.8%, +0.27%)
# 아래였다. 해당하던 것들은 이제 약세로 들어간다.
GRADE_ORDER = [
    GRADE_PULLBACK,
    GRADE_TREND,
    GRADE_OVERHEAT,
    GRADE_WEAK,
    GRADE_ILLIQUID,
    GRADE_RATE,
    GRADE_UNKNOWN,
]


# --- 순수 계산 --------------------------------------------------------------


def pct_return(closes: list[float], span: int) -> float | None:
    """span 거래일 전 대비 수익률. 데이터가 모자라면 None."""
    if len(closes) < span + 1:
        return None
    past = closes[-span - 1]
    if past <= 0:
        return None
    return closes[-1] / past - 1


def straightness(closes: list[float]) -> float:
    """추세가 얼마나 곧은지 — 로그 종가 직선 맞춤의 결정계수 R².

    가격 점들을 그래프에 찍고 자로 직선을 하나 긋는다고 생각하면 된다. 점들이
    자에 딱 붙어 있으면 1 에 가깝고, 사방으로 흩어져 있으면 0 에 가깝다.
    '오르냐' 가 아니라 '깔끔하게 오르냐' 를 재는 숫자다.

    하루 급등하고 19일 제자리인 것과 매일 조금씩 오른 것을 가른다.
    기울기가 0 이하면(= 내려가는 추세면) 0 을 준다.
    """
    if len(closes) < 3 or any(c <= 0 for c in closes):
        return 0.0
    ys = [math.log(c) for c in closes]
    n = len(ys)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return 0.0
    slope = sxy / sxx
    if slope <= 0:
        return 0.0
    return round(max(0.0, min(1.0, (sxy * sxy) / (sxx * syy))), 4)


def straightness_prior(closes: list[float]) -> float:
    """눌림 **이전** 구간의 추세 곧기 — 5거래일 전에서 끝나는 20일 창.

    왜 따로 재나. '눌림 매수' 는 곧게 오르던 것이 최근 며칠 쉬고 있는 상태다.
    그런데 최근 21일로 R² 를 재면 그 쉬는 구간이 직선을 망가뜨려 값이 떨어진다.
    즉 '추세가 곧다' 와 '지금 눌렸다' 가 서로를 배제해 버려서, 두 조건을 같은
    창으로 걸면 해당하는 종목이 영영 안 나온다 (실제로 1,160개 중 0개였다).

    그래서 눌림 판정에서는 쉬기 직전까지의 추세를 본다.
    """
    if len(closes) < LOOKBACK + SHORT_LOOKBACK + 1:
        return 0.0
    return straightness(closes[-(LOOKBACK + SHORT_LOOKBACK + 1):-SHORT_LOOKBACK])


def annualized_vol(closes: list[float], span: int = LOOKBACK) -> float | None:
    """연율 환산 변동성. 하루 출렁임을 1년치로 부풀린 값(×√252)."""
    if len(closes) < span + 1:
        return None
    window = closes[-span - 1:]
    daily = [
        window[i] / window[i - 1] - 1
        for i in range(1, len(window))
        if window[i - 1] > 0
    ]
    if len(daily) < 2:
        return None
    return statistics.pstdev(daily) * math.sqrt(252)


def ma_gap(closes: list[float], span: int = LOOKBACK) -> float | None:
    """20일 평균선에서 얼마나 떨어져 있나."""
    if len(closes) < span:
        return None
    window = closes[-span:]
    avg = sum(window) / len(window)
    if avg <= 0:
        return None
    return closes[-1] / avg - 1


def drawdown(closes: list[float], span: int = LOOKBACK) -> float | None:
    """최근 span 거래일 최고가 대비 지금 위치. 항상 0 이하."""
    if len(closes) < span:
        return None
    peak = max(closes[-span:])
    if peak <= 0:
        return None
    return closes[-1] / peak - 1


def median_turnover(closes: list[float], volumes: list[int], span: int = LOOKBACK) -> float | None:
    """20일 거래대금의 **중앙값**.

    평균이 아니라 중앙값을 쓰는 이유는 하루짜리 대량 거래에 속지 않기 위해서다.
    스윙에서 가장 자주 다치는 건 지표가 틀려서가 아니라 못 빠져나와서다.
    """
    if len(closes) < span or len(volumes) < span:
        return None
    values = [c * v for c, v in zip(closes[-span:], volumes[-span:])]
    if not values:
        return None
    return statistics.median(values)


def atr(highs: list[float], lows: list[float], closes: list[float], span: int = ATR_SPAN) -> float | None:
    """평균 실제 변동폭(ATR). 하루에 보통 얼마나 움직이나를 '원' 으로 잰다.

    하루 변동폭을 고가-저가로만 재면 갭(전날 종가에서 훌쩍 뛰어 시작하는 것)을
    놓친다. 그래서 셋 중 가장 큰 값을 그날의 실제 변동폭으로 본다.

      고가 - 저가            그날 안에서 움직인 폭
      |고가 - 전날 종가|     위로 갭이 뜬 경우
      |저가 - 전날 종가|     아래로 갭이 뜬 경우

    손절선을 그으려면 이 값이 필요하다. 변동성 퍼센트로는 '몇 원 밑에 걸까' 를
    못 정한다.
    """
    if min(len(highs), len(lows), len(closes)) < span + 1:
        return None
    trs = []
    for i in range(len(closes) - span, len(closes)):
        prev = closes[i - 1]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - prev), abs(lows[i] - prev)))
    return sum(trs) / len(trs) if trs else None


def stop_level(closes: list[float], atr_value: float | None, expected: float | None) -> float | None:
    """손절 가격 — 보유기간 노이즈의 **바깥**에 둔다.

    처음에는 `max(최근 저점, 현재가 − 2×ATR)` 로 둘 중 더 가까운 쪽을 썼다.
    손실을 먼저 제한한다는 생각이었는데, 산수가 틀렸다.

      보유 H일 동안의 노이즈 크기는 하루 변동성 × √H 다. 2주면 √10 ≈ 3.16
      일간단위인데, 2×ATR 은 2 일간단위다. **2 < 3.16 이라 손절선이 노이즈
      밴드 안에 있다.** 논지가 깨져서가 아니라 평범한 출렁임에 걸린다.
      거기에 '최근 저점' 을 더 가까운 쪽으로 골랐으니 더 좁아졌다.

    보유를 8주로 늘린 뒤에는 √40 ≈ 6.3 이라 손절이 훨씬 넓어진다. 같은 규칙이
    기간을 따라 자동으로 조정되는 것이지 값을 새로 고른 게 아니다.

    백테스트가 이걸 그대로 보여줬다 — 2주 안에 손절이 37~60% 걸렸고, 그 바람에
    원수익이 플러스인 등급도 손절을 지키면 마이너스가 됐다.

    그래서 **보유기간 1σ 바깥**으로 옮긴다. 1σ 는 제약을 만족하는 최솟값이지
    성적이 제일 좋게 나오는 값을 찾은 게 아니다. 최근 10일 저점 항은 뺐다 —
    노이즈 바깥에 둔다는 요구와 충돌하고, 좁아지는 쪽으로만 작동했기 때문이다.
    """
    if not closes or atr_value is None:
        return None
    close = closes[-1]
    floor = ATR_STOP_MULT * atr_value
    if expected is not None:
        floor = max(floor, expected * close)
    stop = close - floor
    if stop >= close or stop <= 0:
        return None
    return stop


def stop_touch_probability(stop_pct: float | None, expected: float | None) -> float | None:
    """2주 안에 손절선을 건드릴 확률 (이론치).

    방향성 없는 무작위 걸음에서 경로의 최솟값이 −d 아래로 내려갈 확률은
    반사원리로 `2Φ(−d/σ)` 다. 종가가 −d 아래로 끝날 확률의 **두 배**인데,
    도중에 찍고 올라오는 경로까지 세기 때문이다.

    손절을 '얼마나 자주 걸릴 자리에 뒀는가' 로 바꿔 보여주려고 쓴다. 손절폭이
    1σ 면 약 32%, 0.67σ 면 약 50% 다 — 예전 규칙이 딱 후자였고 실제로도
    그만큼 걸렸다.

    이 값은 **검증 가능하다.** 백테스트의 실제 손절 적중률과 견줘 볼 수 있다.
    """
    if stop_pct is None or not expected or expected <= 0:
        return None
    z = abs(stop_pct) / expected
    # 표준정규 누적분포 Φ(−z) = 0.5 × erfc(z/√2)
    return round(min(1.0, math.erfc(z / math.sqrt(2))), 4)


def expected_move(closes: list[float], days: int = SWING_LOOKBACK) -> float | None:
    """앞으로 `days` 거래일 동안 보통 이만큼 움직인다 — 관측된 변동성의 1σ.

    √시간 법칙을 쓴다. 하루 변동폭이 2% 면 40일은 2%×√40 ≈ 12.6% 다. 40일이니까
    80% 가 아니다 — 오르내림이 서로 상쇄되기 때문에 날짜의 제곱근만큼만 커진다.

    **방향을 맞히는 값이 아니다.** '보유 기간에 이 정도 폭으로 흔들리는 물건'
    이라는 크기 감각일 뿐이다. 손절폭과 견줘 볼 잣대로 쓴다.
    """
    if len(closes) < LOOKBACK + 1:
        return None
    window = closes[-LOOKBACK - 1:]
    daily = [
        window[i] / window[i - 1] - 1 for i in range(1, len(window)) if window[i - 1] > 0
    ]
    if len(daily) < 2:
        return None
    return statistics.pstdev(daily) * math.sqrt(days)


# 손익비(2주 기대폭 ÷ 손절폭)는 뺐다. 손절을 1σ 바깥으로 옮긴 뒤로는 이 값이
# 정의상 1.0 이하로 고정돼 아무것도 가르지 못한다. 대신 같은 두 숫자로 만드는
# stop_touch_probability() 를 쓴다 — 그건 백테스트와 대조까지 된다.


def premium_of(price, nav) -> float | None:
    """괴리율 — 시장가가 순자산가치(NAV)보다 얼마나 비싼가.

    ETF 는 담고 있는 자산의 가치(NAV)가 정해져 있는데 시장가는 수급으로 따로
    논다. 괴리가 +1% 인 걸 사면 **사는 순간 1% 를 얹어 주는 것**이고, 나중에
    괴리가 붙어 있으리라는 보장이 없다. 2주 스윙에서 1% 는 작지 않다.

    거래가 얕은 ETF 와 해외 자산 ETF(시차 때문에)에서 특히 벌어진다.
    """
    try:
        price = float(price)
        nav = float(nav)
    except (TypeError, ValueError):
        return None
    if nav <= 0:
        return None
    return price / nav - 1


def volume_profile(
    highs: list[float], lows: list[float], volumes: list[int],
    span: int = PROFILE_SPAN, bins: int = PROFILE_BINS,
) -> tuple[list[float], list[float]] | None:
    """매물대 — 어느 가격대에서 거래가 많이 됐나.

    가격 구간을 bins 칸으로 나누고, 하루치 거래량을 그날 고가~저가에 **균등하게**
    나눠 담는다. 장중 체결 분포가 없으니(일봉만 있다) 이게 표준 근사다. 실제로는
    종가 근처에 더 몰리지만, 하루 범위 안에서의 치우침은 여러 날을 쌓으면 대체로
    씻긴다.

    왜 필요한가. 위에 매물이 쌓여 있으면 오를 때마다 **본전 찾는 매도**가 나온다.
    2주 스윙에서 이건 지표보다 직접적인 장애물이다. 아래에 쌓여 있으면 반대로
    받쳐 준다.

    돌려주는 것은 (가격 중심값 배열, 거래량 배열) 이다.
    """
    # 하루치라도 고가와 저가가 다르면 매물대는 만들어진다. 값이 하나도 없거나
    # 가격 폭이 0 인 경우만 걸러낸다 (아래 top <= bottom).
    if min(len(highs), len(lows), len(volumes)) < 1:
        return None
    hs, ls, vs = highs[-span:], lows[-span:], volumes[-span:]
    top, bottom = max(hs), min(ls)
    if top <= bottom:
        return None
    width = (top - bottom) / bins
    buckets = [0.0] * bins

    def index_of(price: float) -> int:
        return max(0, min(bins - 1, int((price - bottom) / width)))

    for high, low, volume in zip(hs, ls, vs):
        if volume <= 0:
            continue
        lo_i, hi_i = index_of(low), index_of(high)
        share = volume / (hi_i - lo_i + 1)
        for i in range(lo_i, hi_i + 1):
            buckets[i] += share

    centers = [bottom + width * (i + 0.5) for i in range(bins)]
    return centers, buckets


def profile_stats(profile, price: float | None) -> dict:
    """매물대에서 뽑아내는 세 가지.

      poc       가장 많이 거래된 가격대. '평균 매입가' 에 가장 가까운 자리다
      overhead  현재가 **위**에 쌓인 거래량 비율. 이 사람들이 물려 있다
      wall      현재가 위에서 매물이 가장 두꺼운 가격대. 첫 저항이다

    overhead 가 70% 면 지난 반년 거래의 70% 가 지금보다 비싼 값에 이뤄졌다는
    뜻이다. 오를 때마다 팔 사람이 그만큼 기다리고 있다.
    """
    out = {"poc": None, "overhead": None, "wall": None, "wallGap": None}
    if not profile or price is None or price <= 0:
        return out
    centers, buckets = profile
    total = sum(buckets)
    if total <= 0:
        return out
    out["poc"] = round(centers[buckets.index(max(buckets))], 2)
    above = [(c, v) for c, v in zip(centers, buckets) if c > price]
    out["overhead"] = round(sum(v for _, v in above) / total, 4)
    if above:
        wall_price = max(above, key=lambda cv: cv[1])[0]
        out["wall"] = round(wall_price, 2)
        out["wallGap"] = round(wall_price / price - 1, 5)
    return out


def highest(values: list[float], span: int) -> float | None:
    """최근 span 거래일 최고가. 데이터가 짧으면 있는 만큼으로 잰다."""
    if not values:
        return None
    return max(values[-span:])


def median_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def breadth_of(returns: list[float | None]) -> float | None:
    """20일 수익률이 양(+)인 종목 비율. 이 판이 통째로 오르는가."""
    clean = [r for r in returns if r is not None]
    if not clean:
        return None
    return sum(1 for r in clean if r > 0) / len(clean)


def daily_breadth(
    calendar: list[str],
    closes_by_date: list[dict],
    min_valid: int = STRIP_MIN_VALID,
) -> list[int | None]:
    """날짜별 '전일보다 오른 구성종목 비율' — 정수 퍼센트 0~100.

    돌려주는 길이는 `len(calendar) - 1` 이다. 첫날은 전일이 없어 상승·하락을
    가를 수 없다.

    세 가지를 지킨다.

      양쪽 종가가 다 있는 종목만 분모  거래정지·상장 전은 빠진다. 안 빼면
                                       가만히 있는 종목이 '안 오른 쪽' 으로
                                       세어져 폭이 낮게 나온다
      보합은 오른 것이 아니다           제자리인 날은 상승이 아니다
      유효 종목이 모자라면 None         비율을 낼 표본이 없다는 뜻이고,
                                       화면에서는 빈칸으로 그린다

    **임계값은 여기서 적용하지 않는다.** 70/30 은 화면 쪽 상수다. 원값을
    실어야 칸마다 실제 비율을 보여줄 수 있고, 기준을 바꿔도 다시 굽지 않는다.
    """
    out: list[int | None] = []
    for i in range(1, len(calendar)):
        today, yesterday = calendar[i], calendar[i - 1]
        up = 0
        valid = 0
        for prices in closes_by_date:
            a, b_ = prices.get(yesterday), prices.get(today)
            if not a or not b_ or a <= 0 or b_ <= 0:
                continue
            valid += 1
            if b_ > a:
                up += 1
        out.append(round(100 * up / valid) if valid >= min_valid else None)
    return out


def group_index(calendar: list[str], closes_by_date: list[dict]) -> list[float]:
    """구성종목 일간수익률의 **중위값**을 누적곱한 지수. 시작을 100 으로 둔다.

    합산이 아니라 중위값이라 한 종목의 상한가에 지수가 끌려가지 않는다.
    """
    index = [100.0]
    for i in range(1, len(calendar)):
        today, yesterday = calendar[i], calendar[i - 1]
        daily = []
        for prices in closes_by_date:
            a, b = prices.get(yesterday), prices.get(today)
            if a and b and a > 0:
                daily.append(b / a - 1)
        step = statistics.median(daily) if daily else 0.0
        index.append(index[-1] * (1 + step))
    return [round(v, 3) for v in index]


def grade_of(
    *,
    bars_count: int,
    r20: float | None,
    r5: float | None,
    straight: float,
    straight_prior: float = 0.0,
    vol: float | None,
    gap: float | None,
    dd: float | None,
    turnover: float | None,
    beats_market: bool,
    breadth_ok: bool,
    leverage: float = 1.0,
) -> tuple[str, str, list[str]]:
    """(최종 등급, 모멘텀 등급, 이유) 를 돌려준다.

    둘로 나누는 이유가 있다. 금리형·유동성 부족·판정 불가는 '살 수 있는
    물건인가' 의 문제라 모멘텀보다 앞선다. 하지만 그걸로 덮어 버리면 '추세는
    좋은데 거래가 안 되는 것' 과 '추세도 나쁜 것' 이 화면에서 똑같이 보인다.
    그래서 막는 사유는 grade 에, 흐름 판단은 momentum 에 따로 남긴다.

    임계값은 배수만큼 늘린다. 2배 ETF 에 1배 잣대를 대면 늘 과열로 찍힌다.
    """
    scale = max(1.0, abs(leverage))

    if bars_count < LOOKBACK + 1 or r20 is None or r5 is None:
        return GRADE_UNKNOWN, GRADE_UNKNOWN, [f"거래일 {bars_count}일 — 20일 지표를 만들 수 없음"]

    blocker = None
    blocker_reason = None
    if vol is not None and vol < VOL_FLOOR:
        blocker = GRADE_RATE
        blocker_reason = f"연율 변동성 {vol:.1%} — 금리형이라 모멘텀이 성립하지 않음"
    elif turnover is not None and turnover < TURNOVER_FLOOR:
        blocker = GRADE_ILLIQUID
        blocker_reason = (
            f"20일 중앙 거래대금 {turnover / 1e8:.1f}억 — 스윙으로 빠져나오기 어려움"
        )

    reasons: list[str] = []
    if r20 >= OVERHEAT_R20 * scale and r5 < 0:
        momentum = GRADE_OVERHEAT
        reasons = [f"20일 {r20:+.1%} 급등 뒤 최근 5일 {r5:+.1%} 로 꺾임 — 추격 금지"]
    else:
        if beats_market:
            reasons.append("시장보다 낫다")
        if breadth_ok:
            reasons.append("구성종목 다수가 상승")

        pullback = (
            beats_market
            and breadth_ok
            and straight_prior >= STRAIGHT_OK
            and dd is not None
            and PULLBACK_R5_LOW * scale <= r5 <= PULLBACK_R5_HIGH * scale
            and PULLBACK_DD_DEEP * scale <= dd <= PULLBACK_DD_SHALLOW * scale
        )
        if pullback:
            reasons.append(f"쉬기 전까지 추세가 곧았다 (R² {straight_prior:.2f})")
            reasons.append(f"고점 대비 {dd:+.1%} 로 쉬는 중")
            momentum = GRADE_PULLBACK
        elif beats_market and breadth_ok and r5 >= 0:
            if straight >= STRAIGHT_OK:
                reasons.append(f"추세가 곧다 (R² {straight:.2f})")
            reasons.append(f"최근 5일 {r5:+.1%} 로 흐름 유지")
            momentum = GRADE_TREND
        else:
            momentum = GRADE_WEAK
            missing = []
            if not beats_market:
                missing.append("시장을 못 넘음")
            if not breadth_ok:
                missing.append("구성종목 상승 비율이 낮음")
            if straight < STRAIGHT_OK:
                missing.append(f"추세가 고르지 않음 (R² {straight:.2f})")
            reasons = missing or ["뚜렷한 상승 근거 없음"]

    if blocker:
        return blocker, momentum, [blocker_reason] + reasons
    return momentum, momentum, reasons


def correlation(a: list[float], b_: list[float]) -> float | None:
    """두 수익률 배열의 상관계수.

    ETF 두 개를 나눠 담아도 둘이 같이 움직이면 분산이 아니다. 반도체 ETF 세 개는
    이름만 셋이지 사실상 한 베팅이다. 그걸 숫자로 잡는다.

    가격이 아니라 **일간 수익률**로 재야 한다. 가격끼리 재면 둘 다 우상향이라는
    이유만으로 상관이 높게 나온다.
    """
    n = min(len(a), len(b_))
    if n < 20:
        return None
    x, y = a[-n:], b_[-n:]
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((u - mx) * (v - my) for u, v in zip(x, y))
    return round(max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy))), 3)


def daily_returns(closes: list[float], span: int = CORR_SPAN) -> list[float]:
    window = closes[-(span + 1):]
    return [
        window[i] / window[i - 1] - 1
        for i in range(1, len(window))
        if window[i - 1] > 0
    ]


def market_regime(kospi: list[float], kosdaq: list[float]) -> dict:
    """지금 롱을 잡아도 되는 국면인가.

    두 지수가 각각 20일 이동평균 위에 있는지만 본다. 예측이 아니라 위치 확인이다.

      둘 다 위       순풍 — 평소대로
      하나만 위      엇갈림 — 오른 쪽 시장에 속한 테마만
      둘 다 아래     역풍 — 후보가 좋아 보여도 시장에 쓸려 나간다

    2주 스윙에서 이게 왜 앞에 오냐면, 보유 기간이 짧을수록 개별 테마의 힘보다
    시장 전체의 방향이 결과를 더 많이 정하기 때문이다. 테마를 고르기 전에
    들어갈 때인지부터 봐야 한다.
    """
    def above(closes: list[float]) -> bool | None:
        gap = ma_gap(closes)
        return None if gap is None else gap > 0

    kospi_up = above(kospi)
    kosdaq_up = above(kosdaq)
    count = sum(1 for v in (kospi_up, kosdaq_up) if v)
    if kospi_up is None or kosdaq_up is None:
        label, note = "알 수 없음", "지수 데이터가 모자랍니다."
    elif count == 2:
        label = "순풍"
        note = "코스피·코스닥 둘 다 20일선 위입니다. 평소대로 봐도 되는 국면입니다."
    elif count == 1:
        up = "코스피" if kospi_up else "코스닥"
        label = "엇갈림"
        note = f"{up}만 20일선 위입니다. {up}에 속한 테마로 좁히는 편이 낫습니다."
    else:
        label = "역풍"
        note = ("코스피·코스닥 둘 다 20일선 아래입니다. 2주 스윙은 시장 방향에 크게 끌려가므로 "
                "후보가 좋아 보여도 규모를 줄이거나 쉬는 것을 우선 고려하세요.")
    return {"label": label, "note": note, "kospiAbove": kospi_up, "kosdaqAbove": kosdaq_up}


def percentile_rank(value: float, population: list[float]) -> float:
    """population 안에서 value 가 상위 몇 %인지 (100 이 최고)."""
    if not population:
        return 0.0
    below = sum(1 for v in population if v < value)
    return round(100.0 * below / len(population), 1)


# --- 입출력 ----------------------------------------------------------------


def read_state() -> dict:
    path = DATA / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_json_gz(path: pathlib.Path):
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def load_bars() -> dict[str, dict[str, list]]:
    """{symbol: {date: [종가, 거래량, 고가, 저가]}}

    고가·저가까지 읽는 이유는 ATR 때문이다. 종가만으로는 하루에 얼마나 흔들리는지
    알 수 없고, 그걸 모르면 손절선을 그을 수 없다.
    """
    bars: dict[str, dict[str, list]] = {}
    with gzip.open(DATA / "bars.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bars.setdefault(row["symbol"], {})[row["date"]] = [
                float(row["close"]),
                int(row["volume"]),
                float(row["high"]),
                float(row["low"]),
            ]
    return bars


class Series(NamedTuple):
    dates: list[str]
    closes: list[float]
    volumes: list[int]
    highs: list[float]
    lows: list[float]


def series_of(bars: dict, symbol: str) -> Series:
    data = bars.get(symbol) or {}
    dates = sorted(data)
    return Series(
        dates,
        [data[d][0] for d in dates],
        [data[d][1] for d in dates],
        [data[d][2] for d in dates],
        [data[d][3] for d in dates],
    )


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def spark(closes: list[float], points: int = SPARK_POINTS) -> list[float]:
    """스파크라인용. 시작을 100 으로 맞춘 최근 points 개 종가."""
    window = closes[-points:]
    if not window or window[0] <= 0:
        return []
    return [round(c / window[0] * 100, 2) for c in window]


# ETF 구성종목에는 종목이 아닌 줄이 섞여 온다. 원화현금·설정현금액·선물·옵션은
# 애초에 종목코드가 없으므로 이름 매칭률을 잴 때 분모에서 빼야 한다. 안 빼면
# 파서가 멀쩡한데도 매칭률이 낮게 나와 없는 문제를 쫓게 된다.
NON_STOCK_HOLDING = ("현금", "예금", "선물", "옵션", "설정", "위클리", "스왑", "CD금리")


def is_stock_holding(name: str) -> bool:
    return not any(token in name for token in NON_STOCK_HOLDING)


def metrics_of(s: Series) -> dict:
    closes = s.closes
    atr_value = atr(s.highs, s.lows, closes)
    expected = expected_move(closes)
    stop = stop_level(closes, atr_value, expected)
    stop_pct = None if stop is None else stop / closes[-1] - 1
    vol = annualized_vol(closes)
    r_swing = pct_return(closes, SWING_LOOKBACK)
    price = closes[-1] if closes else None
    high_year = highest(s.highs, YEAR_SPAN)
    profile = volume_profile(s.highs, s.lows, s.volumes)
    stats = profile_stats(profile, price)
    return {
        "price": price,
        "high52": None if high_year is None else round(high_year, 2),
        # 실제로 몇 거래일을 본 고점인지. 상장 1년이 안 됐으면 '52주 고점' 이
        # 아니라 '상장 후 최고' 라고 불러야 한다. 250 미만이면 화면이 말을 바꾼다.
        "high52Days": min(len(s.highs), YEAR_SPAN),
        "fromHigh52": (None if not high_year or not price
                       else round(price / high_year - 1, 5)),
        "high20": (lambda h: None if h is None else round(h, 2))(highest(s.highs, LOOKBACK)),
        "poc": stats["poc"],
        "overhead": stats["overhead"],
        "wall": stats["wall"],
        "wallGap": stats["wallGap"],
        # 40칸짜리 매물대 원본. 목록에 실으면 ETF 1,160개에 4만 줄이라
        # 지연 로딩하는 details.json 쪽에만 넣고 여기서는 떼어낸다.
        "profile": profile,
        "r20": pct_return(closes, LOOKBACK),
        "rSwing": r_swing,
        "r5": pct_return(closes, SHORT_LOOKBACK),
        "straight": straightness(closes[-(LOOKBACK + 1):]),
        "straightPrior": straightness_prior(closes),
        "vol": vol,
        "gap": ma_gap(closes),
        "dd": drawdown(closes),
        "turnover": median_turnover(closes, s.volumes),
        "atr": atr_value,
        "stop": None if stop is None else round(stop, 2),
        "stopPct": stop_pct,
        "expectedSwing": expected,
        "stopProb": stop_touch_probability(stop_pct, expected),
        # 위험조정 2주 모멘텀. 변동성 100% 짜리의 +20% 와 30% 짜리의 +8% 를
        # 같은 줄에 세우려면 위험 한 단위당 얼마를 벌었는지로 봐야 한다.
        "riskAdj": None if (r_swing is None or not vol) else round(r_swing / vol, 4),
        "bars": len(closes),
    }


def round_metrics(m: dict) -> dict:
    out = dict(m)
    out.pop("profile", None)   # 원본 매물대는 목록에 싣지 않는다
    for key in ("r20", "rSwing", "r5", "vol", "gap", "dd", "stopPct", "expectedSwing",
                "fromHigh52", "wallGap"):
        if out.get(key) is not None:
            out[key] = round(out[key], 5)
    if out.get("atr") is not None:
        out["atr"] = round(out["atr"], 2)
    if out.get("turnover") is not None:
        out["turnover"] = round(out["turnover"])
    return out


def main() -> int:
    groups = read_json_gz(DATA / "groups.json.gz")
    etfs = read_json_gz(DATA / "etfs.json.gz")
    holdings = read_json_gz(DATA / "holdings.json.gz") or {}
    market = read_json_gz(DATA / "market.json.gz") or {}
    if not groups or not etfs:
        print("data/stocks 캐시가 없습니다. collect_stocks.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1

    bars = load_bars()
    kospi = series_of(bars, "KOSPI")
    calendar = kospi.dates[-CALENDAR_POINTS:]
    if not calendar:
        print("KOSPI 일봉이 없습니다. 거래일 달력을 만들 수 없습니다.", file=sys.stderr)
        return 1

    # 스트립 달력은 **따로** 만든다. calendar 를 늘리면 group_index 가 길어지고
    # 그게 straightness 를 거쳐 등급을 바꾼다. 표시용 지표가 판정 규칙을
    # 건드리게 두면 안 된다.
    strip_calendar = kospi.dates[-(STRIP_SPAN + 1):]

    kosdaq = series_of(bars, "KOSDAQ")
    benchmark = {
        "KOSPI": pct_return(kospi.closes, LOOKBACK) or 0.0,
        "KOSDAQ": pct_return(kosdaq.closes, LOOKBACK) or 0.0,
    }
    base_date = calendar[-1]

    # 시장 국면. 2주 스윙에서 시장이 20일선 아래로 흘러내리면 테마가 아무리
    # 좋아도 같이 쓸려 나간다. 롱을 얼마나 세게 잡을지는 여기서 먼저 정해진다.
    regime = market_regime(kospi.closes, kosdaq.closes)

    # 종목별 수익률을 한 번만 계산해 두고 돌려 쓴다.
    stock_return: dict[str, float | None] = {}
    stock_short: dict[str, float | None] = {}
    stock_swing: dict[str, float | None] = {}
    stock_overhead: dict[str, float | None] = {}
    for code in bars:
        s = series_of(bars, code)
        closes = s.closes
        stock_return[code] = pct_return(closes, LOOKBACK)
        stock_short[code] = pct_return(closes, SHORT_LOOKBACK)
        stock_swing[code] = pct_return(closes, SWING_LOOKBACK)
        stock_overhead[code] = profile_stats(
            volume_profile(s.highs, s.lows, s.volumes),
            closes[-1] if closes else None,
        )["overhead"]

    # 이름 -> 종목코드. ETF 구성종목이 코드를 안 줘서 이름으로 맞대야 한다.
    name_to_code: dict[str, str] = {}
    code_to_name: dict[str, str] = {}
    for group in groups:
        for member in group.get("members", []):
            name_to_code.setdefault(api.normalize_name(member["name"]), member["code"])
            code_to_name.setdefault(member["code"], member["name"])

    # 네이버는 ETF 1,160개를 전부 업종 '기타' 에 넣어 둔다. 그룹 지표에서 ETF 를
    # 빼야 하는 이유가 둘이다. 하나, 테마 대세는 개별 종목이 움직여야 대세지
    # ETF 가 오른 건 결과지 원인이 아니다. 둘, ETF 를 ETF 섞인 그룹으로 확인하면
    # 자기가 자기를 보증하는 순환 참조가 된다.
    etf_codes = {e["code"] for e in etfs}

    # --- 테마·업종 -----------------------------------------------------------
    group_rows = []
    group_details = {}
    for group in groups:
        members = [m for m in group.get("members", []) if m["code"] not in etf_codes]
        # 업종 '기타' 는 분류가 안 되는 것들을 담아 둔 자루라 섹터가 아니다.
        if not members or (group["type"] == "upjong" and group["name"] == "기타"):
            continue
        codes = [m["code"] for m in members]
        r20s = [stock_return.get(c) for c in codes]
        r5s = [stock_short.get(c) for c in codes]
        closes_by_date = [
            {d: bars[c][d][0] for d in bars[c]} for c in codes if c in bars
        ]
        index = group_index(calendar, closes_by_date)
        markets = Counter(market.get(c) for c in codes if market.get(c))
        bench = "KOSDAQ" if markets.get("KOSDAQ", 0) > markets.get("KOSPI", 0) else "KOSPI"

        r_swings = [stock_swing.get(c) for c in codes]
        overheads = [stock_overhead.get(c) for c in codes]
        median20 = median_or_none(r20s)
        median_swing = median_or_none(r_swings)
        median5 = median_or_none(r5s)
        breadth = breadth_of(r20s)
        straight = straightness(index)
        excess = None if median20 is None else median20 - benchmark[bench]
        valid = sum(1 for r in r20s if r is not None)

        gap = ma_gap(index) if len(index) >= LOOKBACK else None
        dd = drawdown(index) if len(index) >= LOOKBACK else None
        grade, momentum, reasons = grade_of(
            bars_count=len(index),
            r20=median20,
            r5=median5,
            straight=straight,
            straight_prior=straightness_prior(index),
            vol=None,
            gap=gap,
            dd=dd,
            turnover=None,
            beats_market=bool(excess is not None and excess > 0),
            breadth_ok=bool(breadth is not None and breadth >= BREADTH_OK),
        )

        key = f"{group['type']}-{group['no']}"
        group_rows.append(
            {
                "key": key,
                "type": group["type"],
                "no": group["no"],
                "name": group["name"],
                "members": len(codes),
                "valid": valid,
                "r20": None if median20 is None else round(median20, 5),
                "rSwing": None if median_swing is None else round(median_swing, 5),
                "r5": None if median5 is None else round(median5, 5),
                "breadthSwing": (lambda x: None if x is None else round(x, 4))(breadth_of(r_swings)),
                # 구성종목의 매물대 부담 중위값. 테마가 통째로 물려 있는지.
                "overhead": (lambda x: None if x is None else round(x, 4))(median_or_none(overheads)),
                "breadth": None if breadth is None else round(breadth, 4),
                "straight": straight,
                # 날짜별 상승 종목 비율 30칸(과거→최근). 화면에서 색띠로 그린다.
                "strip": daily_breadth(strip_calendar, closes_by_date),
                "bench": bench,
                "excess": None if excess is None else round(excess, 5),
                "grade": grade,
                "momentum": momentum,
                "reasons": reasons,
                "spark": spark(index),
            }
        )
        ranked = sorted(
            (
                {
                    "code": m["code"],
                    "name": m["name"],
                    "r20": None
                    if stock_return.get(m["code"]) is None
                    else round(stock_return[m["code"]], 5),
                }
                for m in members
            ),
            key=lambda x: (x["r20"] is None, -(x["r20"] or 0)),
        )
        group_details[key] = ranked

    # --- ETF -----------------------------------------------------------------
    # 대표 그룹: ETF 편입비중이 어느 테마·업종에 가장 많이 겹치는지.
    group_codes = {
        f"{g['type']}-{g['no']}": {m["code"] for m in g.get("members", [])} for g in groups
    }
    group_name = {f"{g['type']}-{g['no']}": g["name"] for g in groups}
    group_by_key = {r["key"]: r for r in group_rows}

    matched_names = 0
    total_names = 0
    group_etf_overlap: dict[str, list[tuple[str, float]]] = defaultdict(list)

    etf_rows = []
    etf_details = {}
    etf_profile = {}
    by_tab: dict[int, list[float]] = defaultdict(list)

    prepared = []
    for etf in etfs:
        code = etf["code"]
        s = series_of(bars, code)
        m = metrics_of(s)
        held = holdings.get(code, [])
        resolved = []
        for row in held:
            stock_code = name_to_code.get(api.normalize_name(row["name"]))
            # 매칭률은 국내 ETF 의 '종목' 줄만 센다. 해외 ETF 는 미국 주식을
            # 담으니 국내 종목코드가 없는 게 정상이고, 원화현금·선물 줄은 애초에
            # 종목이 아니다. 둘을 분모에 넣으면 파서가 멀쩡해도 낮게 나온다.
            if etf["tab"] in DOMESTIC_TABS and is_stock_holding(row["name"]):
                total_names += 1
                if stock_code:
                    matched_names += 1
            resolved.append({**row, "code": stock_code})

        overlap: dict[str, float] = defaultdict(float)
        for row in resolved:
            if not row["code"]:
                continue
            for key, codes in group_codes.items():
                if row["code"] in codes:
                    overlap[key] += row["weight"]
        best_key = max(overlap, key=overlap.get) if overlap else None
        # 테마를 눌렀을 때 '이걸 살 수 있는 ETF' 를 보여주려면 대표 그룹 하나로는
        # 모자란다. 한 ETF 가 여러 테마에 걸치고, 어떤 테마의 최대 노출 ETF 가
        # 그 ETF 입장에서는 두 번째 테마일 수 있기 때문이다. 노출이 의미 있는
        # 만큼(10% 이상) 되는 조합을 전부 모아 둔다.
        if etf["tab"] not in GROUP_ETF_EXCLUDE_TABS:
            for key, weight in overlap.items():
                if weight >= GROUP_ETF_MIN_WEIGHT:
                    group_etf_overlap[key].append((code, round(weight, 1)))

        markets = Counter(
            market.get(r["code"]) for r in resolved if r["code"] and market.get(r["code"])
        )
        bench = "KOSDAQ" if markets.get("KOSDAQ", 0) > markets.get("KOSPI", 0) else "KOSPI"

        if m["r20"] is not None:
            by_tab[etf["tab"]].append(m["r20"])
        prepared.append((etf, m, resolved, best_key, overlap, bench, s.closes))

    for etf, m, resolved, best_key, overlap, bench, closes in prepared:
        code = etf["code"]
        leverage = api.leverage_of(etf["name"])
        domestic = etf["tab"] in DOMESTIC_TABS
        r20 = m["r20"]

        if domestic:
            excess = None if r20 is None else r20 - benchmark[bench]
            beats = bool(excess is not None and excess > 0)
            pct = None
        else:
            excess = None
            pct = None if r20 is None else percentile_rank(r20, by_tab[etf["tab"]])
            beats = bool(pct is not None and pct >= FOREIGN_PCT_OK)

        group_row = group_by_key.get(best_key) if best_key else None
        breadth = group_row["breadth"] if group_row else None
        # 대표 그룹이 없으면 확인할 폭이 없다. 통과시키되 화면에 '대세 확인 불가' 로
        # 드러낸다. 근거가 한 겹 얇다는 사실을 숨기지 않는다.
        breadth_ok = True if breadth is None else breadth >= BREADTH_OK

        grade, momentum, reasons = grade_of(
            bars_count=m["bars"],
            r20=r20,
            r5=m["r5"],
            straight=m["straight"],
            straight_prior=m["straightPrior"],
            vol=m["vol"],
            gap=m["gap"],
            dd=m["dd"],
            turnover=m["turnover"],
            beats_market=beats,
            breadth_ok=breadth_ok,
            leverage=leverage,
        )
        if breadth is None and grade in {GRADE_PULLBACK, GRADE_TREND}:
            reasons.append("대세 확인 불가 — 국내 테마와 연결되지 않음")

        row = {
            "code": code,
            "name": etf["name"],
            "tab": etf["tab"],
            "tabName": TAB_NAMES.get(etf["tab"], "기타"),
            "lev": leverage,
            "hedged": api.is_hedged(etf["name"]),
            "cap": etf.get("market_cap_100m"),
            "grade": grade,
            "momentum": momentum,
            "reasons": reasons,
            "bench": bench if domestic else None,
            "excess": None if excess is None else round(excess, 5),
            "pct": pct,
            "group": best_key,
            "groupName": group_name.get(best_key) if best_key else None,
            "groupWeight": round(overlap[best_key], 1) if best_key else None,
            "breadth": breadth,
            "premium": (lambda p: None if p is None else round(p, 5))(
                premium_of(etf.get("price"), etf.get("nav"))
            ),
            "spark": spark(closes),
        }
        row.update(round_metrics(m))
        etf_rows.append(row)

        # 매물대 막대. 40칸을 그대로 두면 details.json 이 커지므로 거래량을
        # 최대값 대비 0~100 정수로 눌러 담는다. 화면은 비율만 쓴다.
        prof = m.get("profile")
        if prof:
            centers, buckets = prof
            peak = max(buckets) or 1
            # 가격 배열은 안 싣는다. 칸이 균등하므로 최저·최고만 있으면 화면에서
            # 되만들 수 있고, 그것만으로 파일이 절반이 된다.
            step = (centers[-1] - centers[0]) / (len(centers) - 1) if len(centers) > 1 else 0
            etf_profile[code] = {
                "lo": round(centers[0] - step / 2, 2),
                "hi": round(centers[-1] + step / 2, 2),
                "vol": [round(v / peak * 100) for v in buckets],
            }

        etf_details[code] = [
            {
                "name": r["name"],
                "weight": r["weight"],
                "code": r["code"],
                "r20": None
                if not r["code"] or stock_return.get(r["code"]) is None
                else round(stock_return[r["code"]], 5),
            }
            for r in resolved[:10]
        ]

    # 같은 테마에 붙은 ETF 가 열 개씩 뜨면 후보 목록이 아니라 소음이다. 반도체
    # ETF 열 개를 다 사는 사람은 없다 — 사실상 같은 베팅이다. 대표 그룹마다
    # 20일 중앙 거래대금이 가장 큰 하나만 '대표' 로 세운다. 유동성으로 고르는
    # 이유는 2주 안에 나와야 하기 때문이다.
    best_by_group: dict[str, tuple[float, str]] = {}
    for row in etf_rows:
        key = row["group"]
        if not key or row["turnover"] is None:
            continue
        if key not in best_by_group or row["turnover"] > best_by_group[key][0]:
            best_by_group[key] = (row["turnover"], row["code"])
    primary_codes = {code for _, code in best_by_group.values()}
    for row in etf_rows:
        # 대표 그룹이 없는 ETF(해외·채권 등)는 겹칠 상대가 없으니 그대로 둔다.
        row["primary"] = row["group"] is None or row["code"] in primary_codes

    # --- 후보 간 상관 --------------------------------------------------------
    # 분산은 개수가 아니라 상관이 정한다. 후보를 3~5개 담을 때 '사실상 같은 베팅'
    # 인지 알려주려면 상관이 필요하다. 전 종목은 못 싣고(67만 쌍), 실제로 담을
    # 만한 후보군에만 계산한다.
    candidates = [
        r for r in etf_rows
        if r["primary"] and r["grade"] in (GRADE_TREND, GRADE_PULLBACK, GRADE_OVERHEAT)
        and r["turnover"] and r["turnover"] >= TURNOVER_FLOOR
    ]
    candidates.sort(key=lambda r: -(r["turnover"] or 0))
    candidates = candidates[:CORR_MAX_CANDIDATES]
    corr_codes = [r["code"] for r in candidates]
    returns_of = {c: daily_returns(series_of(bars, c).closes) for c in corr_codes}
    corr_pairs: dict[str, float] = {}
    for i, a in enumerate(corr_codes):
        for b_code in corr_codes[i + 1:]:
            value = correlation(returns_of[a], returns_of[b_code])
            if value is not None:
                corr_pairs[f"{a}:{b_code}"] = value
    print(f"후보 {len(corr_codes)}개 상관 {len(corr_pairs)}쌍 (최근 {CORR_SPAN}거래일)")
    write_json(OUT / "corr.json", {"span": CORR_SPAN, "codes": corr_codes, "pairs": corr_pairs})

    match_rate = matched_names / total_names if total_names else 0.0
    print(f"국내 ETF 구성종목 이름 매칭률 {match_rate:.1%} ({matched_names}/{total_names})")
    if total_names and match_rate < 0.80:
        print("  매칭률이 80% 미만입니다 — normalize_name 을 확인하세요.", file=sys.stderr)

    grades = Counter(r["grade"] for r in etf_rows)
    order = {g: i for i, g in enumerate(GRADE_ORDER)}
    etf_rows.sort(key=lambda r: (order.get(r["grade"], 99), -(r["r20"] or -9)))
    group_rows.sort(key=lambda r: (order.get(r["grade"], 99), -(r["r20"] or -9)))

    state = read_state()
    meta = {
        "baseDate": base_date,
        "generated": state.get("last_run", base_date),
        # 장중에 수집한 캐시로 구웠으면 마지막 종가가 확정값이 아니다. 숨기지
        # 않고 화면에 알린다.
        "intraday": bool(state.get("bars_intraday")),
        "lookback": LOOKBACK,
        "stripSpan": STRIP_SPAN,
        # 스트립 30칸이 각각 어느 거래일인지. 화면 툴팁에 쓴다.
        "stripDates": strip_calendar[1:],
        "swingLookback": SWING_LOOKBACK,
        "swingWeeks": round(SWING_LOOKBACK / 5),
        "regime": regime,
        "primaryCount": sum(1 for r in etf_rows if r["primary"]),
        "kospi": round(benchmark["KOSPI"], 5),
        "kosdaq": round(benchmark["KOSDAQ"], 5),
        "etfCount": len(etf_rows),
        "groupCount": len(group_rows),
        "themeCount": sum(1 for g in group_rows if g["type"] == "theme"),
        "upjongCount": sum(1 for g in group_rows if g["type"] == "upjong"),
        "trendGroups": sum(
            1 for g in group_rows if (g["breadth"] or 0) >= BREADTH_OK and (g["r20"] or 0) > 0
        ),
        "matchRate": round(match_rate, 4),
        "grades": dict(grades),
        "thresholds": {
            "breadth": BREADTH_OK,
            "straight": STRAIGHT_OK,
            "vol": VOL_FLOOR,
            "turnover": TURNOVER_FLOOR,
            "overheat": OVERHEAT_R20,
            # 스트립 색 경계. 계산이 아니라 화면 표시용이라 여기에만 둔다.
            "stripUp": 0.70,
            "stripDown": 0.30,
        },
    }

    write_json(OUT / "meta.json", meta)
    write_json(OUT / "etfs.json", etf_rows)
    write_json(OUT / "groups.json", group_rows)
    # 그룹 -> 그 그룹을 살 수 있는 ETF. 시가총액 상위 5 와 거래대금 상위 5 를
    # 미리 잘라서 넣는다. 전부 실으면 details.json 이 2MB 를 넘고, 화면에서
    # 쓰는 건 열 줄뿐이다.
    row_by_code = {r["code"]: r for r in etf_rows}
    group_etf: dict[str, dict] = {}
    for key, pairs in group_etf_overlap.items():
        entries = []
        for code, weight in pairs:
            row = row_by_code.get(code)
            if not row:
                continue
            entries.append({
                "code": code,
                "name": row["name"],
                "weight": weight,          # 이 ETF 안에서 해당 테마가 차지하는 비중(%)
                "cap": row["cap"],          # 시가총액(억원)
                "turnover": row["turnover"],
                "rSwing": row["rSwing"],
                "grade": row["grade"],
                "lev": row["lev"],
                "partial": weight < GROUP_ETF_PARTIAL,
            })
        if not entries:
            continue
        by_cap = sorted(entries, key=lambda e: -(e["cap"] or 0))[:GROUP_ETF_TOP]
        by_turnover = sorted(entries, key=lambda e: -(e["turnover"] or 0))[:GROUP_ETF_TOP]
        group_etf[key] = {"total": len(entries), "byCap": by_cap, "byTurnover": by_turnover}

    for row in group_rows:
        row["etfCount"] = group_etf.get(row["key"], {}).get("total", 0)
    write_json(OUT / "groups.json", group_rows)

    covered = sum(1 for g in group_rows if g["key"] in group_etf)
    print(f"ETF 로 살 수 있는 테마·업종 {covered}/{len(group_rows)}개 "
          f"(편입비중 {GROUP_ETF_MIN_WEIGHT:.0f}% 이상 기준)")

    write_json(OUT / "details.json", {
        "etf": etf_details, "group": group_details, "groupEtf": group_etf,
        "profile": etf_profile,
    })

    print(f"기준 거래일 {base_date} · ETF {len(etf_rows)} · 그룹 {len(group_rows)}")
    print(f"시장 국면: {regime['label']} — {regime['note']}")
    print(f"대표 ETF {meta['primaryCount']}개 (중복 접기 전 {len(etf_rows)}개)")
    print("등급 분포: " + ", ".join(f"{g} {grades[g]}" for g in GRADE_ORDER if grades.get(g)))
    for path in ("meta.json", "etfs.json", "groups.json", "details.json"):
        print(f"  assets/etf/{path}  {(OUT / path).stat().st_size / 1024:.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
