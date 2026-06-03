---
layout: single
title:  "퀀트 팩터 직접 계산하기 — PER · RSI · MACD · Sharpe 를 Python + LaTeX 로 풀어보기"
date: 2026-06-03 09:00:00 +0900
categories: finance
tag: [quant, python, pandas, factor, PER, PBR, ROE, RSI, MACD, Bollinger, Sharpe, Beta]
author_profile: false
toc: true
use_math: true
description: "PER·ROE 같은 펀더멘털 팩터부터 RSI·MACD·Bollinger·Sharpe 까지, 수식 정의와 함께 pandas 로 직접 계산하는 코드를 한 글에 모았어요."
---

## Summary

퀀트 팩터는 라이브러리 한 줄(`ta.rsi(...)`)로 뽑을 수 있어요. 다만 그렇게만 쓰다 보면 **"왜 분모가 저거고, 왜 14일이고, 왜 표준편차로 나누는지"** 가 통째로 비어 있더라고요. 팩터 의미가 비어 있으면 백테스트 결과를 해석할 때도 흔들립니다.

그래서 자주 쓰는 팩터들을 **정의(LaTeX) → 왜 그렇게 계산하나 → pandas 구현 → 해석/함정** 의 같은 4단 포맷으로 묶었어요.

> 💡 이 글에서 다루는 것
> - 가치/수익성 팩터 — **EPS · PER · PBR · ROE**
> - 가격 팩터 — **단순/로그 수익률 · RSI · MACD · Bollinger Bands**
> - 위험조정 팩터 — **Sharpe Ratio · Beta**
> - 각 팩터의 수식과 pandas / numpy 구현, "왜 이렇게 정의되는가" 한 줄 해설

본문에 들어가기 전에 환경부터 잡고 갈게요. 각 팩터 섹션의 정의식 바로 아래에 **"💡 기호 풀기"** 박스를 둬서, 처음 보는 기호는 그 자리에서 읽는 법까지 같이 짚고 넘어가요.

<br>

<br>



## 0. 공통 환경

코드 예시는 전부 다음 데이터프레임이 있다고 가정합니다.

```python
import numpy as np
import pandas as pd

# df: 일별 OHLCV 데이터, index 는 DatetimeIndex
# columns: ['open', 'high', 'low', 'close', 'volume']
# market: 동일 인덱스의 시장지수 종가 시리즈 (예: KOSPI)
```

라이브러리는 `pandas`, `numpy` 만 씁니다. `ta`, `talib` 같은 wrapper 는 한 번 직접 짜 보고 나면 그 다음에 써도 늦지 않아요.

> ⚠️ 펀더멘털 팩터(EPS, PER, …)는 손익계산서/재무상태표에서 오는 값이라 데이터 소스에 따라 정의가 살짝 다릅니다. 아래 정의는 **국제 회계 기준에서 가장 일반적인 형태** 기준이에요.

<br>

<br>



## 1. EPS — 주당순이익

### 정의

$$
\text{EPS} = \frac{\text{Net Income} - \text{Preferred Dividends}}{\text{Weighted Average Shares Outstanding}}
$$

### 왜 이렇게 계산하나

- 분자에서 **우선주 배당을 빼는 이유** — EPS 는 "보통주 한 주에 떨어지는 이익"이라서, 우선주 몫은 미리 떼야 보통주 주주의 몫만 남습니다.
- 분모가 **가중평균 발행주식 수** 인 이유 — 회기 중간에 유상증자/자사주 매입이 있으면 단순 기말 주식 수로 나누면 왜곡돼요. 기간별로 가중평균을 내야 "1주당" 이라는 단위가 의미를 가집니다.

### Python

> **`eps`**  
> 입력: `net_income` (`float`) — 회기 순이익(원). `preferred_div` (`float`) — 우선주 배당총액(원). `wa_shares` (`float`) — 가중평균 발행주식수(주).  
> 반환: `float` — 보통주 1주당 순이익(원).

```python
def eps(net_income: float, preferred_div: float, wa_shares: float) -> float:
    return (net_income - preferred_div) / wa_shares
```

벡터화하려면 가중평균 주식수만 미리 만들어두면 돼요.

> **`weighted_avg_shares`**  
> 입력: `shares_timeline` (`pd.Series`, `index=DatetimeIndex`) — 발행주식수 변경 시점별 누적 주식수.  
> 반환: `float` — 기간 가중평균 발행주식수.

```python
def weighted_avg_shares(shares_timeline: pd.Series) -> float:
    # shares_timeline: 변경 시점별 발행주식수 (index=date)
    days = (shares_timeline.index.to_series().diff().dt.days
            .shift(-1).fillna(0))
    return (shares_timeline * days).sum() / days.sum()
```

### 해석

EPS 절대값은 회사 사이즈에 휘둘려요. **YoY 성장률(EPS growth)** 이나 다음에 나올 PER 처럼 가격과 결합해야 비교 가능한 숫자가 됩니다.

<br>

<br>



## 2. PER — 주가수익비율

### 정의

$$
\text{PER} = \frac{P}{\text{EPS}}
$$

여기서 $P$ 는 주가, EPS 는 위 정의대로의 연간 주당순이익이에요.

### 왜 이렇게 계산하나

- **분자 / 분모의 단위가 같은 "원" 이라 무차원** — 그래서 종목 간 비교가 가능해집니다. "이 회사 시총 1조원" 같은 절대값으로는 비교가 안 돼요.
- 의미는 **"이 회사의 이익을 1년치 그대로 받는 데 몇 년이 걸리나"** 입니다. PER 10 이면 10년 치 이익으로 주가를 회수하는 셈.
- EPS 가 음수면 PER 가 음수로 떨어져서 의미가 무너집니다. 이때는 `NaN` 처리하는 게 안전해요.

### Python

> **`per`**  
> 입력: `price` (`pd.Series`) — 같은 인덱스 위의 주가. `eps_annual` (`pd.Series`) — 같은 인덱스 위의 연간 EPS.  
> 반환: `pd.Series` — 같은 인덱스의 PER 값. EPS ≤ 0 인 시점은 `NaN`.

```python
def per(price: pd.Series, eps_annual: pd.Series) -> pd.Series:
    # 두 시리즈는 같은 인덱스(분기/연도)로 정렬돼 있다고 가정
    out = price / eps_annual
    return out.where(eps_annual > 0, np.nan)   # 적자기업은 의미 없음
```

### 해석

같은 산업 안에서 PER 가 낮은 종목을 "싸다" 고 보는 게 전통적 가치투자 접근이에요. 다만 **산업이 다르면 직접 비교 금지** — 성장주는 구조적으로 PER 가 높습니다. 그래서 보통 산업 중앙값 대비 z-score 로 변환해서 씁니다.

> **`per_zscore`**  
> 입력: `per_series` (`pd.Series`) — 종목별 PER. `industry` (`pd.Series`) — 같은 길이/인덱스의 산업 라벨.  
> 반환: `pd.Series` — 산업 그룹 안에서 중앙값 대비 z-score (median/std 사용).

```python
def per_zscore(per_series: pd.Series, industry: pd.Series) -> pd.Series:
    g = per_series.groupby(industry)
    return (per_series - g.transform("median")) / g.transform("std")
```

<br>

<br>



## 3. PBR — 주가순자산비율

### 정의

$$
\text{PBR} = \frac{P}{\text{BPS}}, \quad \text{BPS} = \frac{\text{Total Equity} - \text{Preferred Equity}}{\text{Common Shares Outstanding}}
$$

### 왜 이렇게 계산하나

- PER 가 "**이익**" 기준이라면 PBR 은 "**장부가**" 기준이에요. 이익은 분기별로 크게 흔들리지만 자본은 천천히 변해서, **이익 변동성이 큰 산업**(은행/철강/조선)에서는 PBR 이 더 안정적인 가치 지표가 됩니다.
- PBR 1 미만은 "장부상 청산가치보다 시장이 더 싸게 평가" 라는 뜻. 다만 그게 진짜 저평가인지, 자산 자체가 부실해서 시장이 디스카운트하는 건지는 **재무제표를 따로 봐야** 알 수 있어요.

### Python

> **`pbr`**  
> 입력: `price` (`pd.Series`) — 주가. `total_equity`, `preferred_equity`, `common_shares` (`pd.Series`) — 같은 인덱스의 총자본/우선주 자본/보통주 발행주식수.  
> 반환: `pd.Series` — 같은 인덱스의 PBR. BPS ≤ 0 인 시점은 `NaN`.

```python
def pbr(price: pd.Series, total_equity: pd.Series,
        preferred_equity: pd.Series, common_shares: pd.Series) -> pd.Series:
    bps = (total_equity - preferred_equity) / common_shares
    return (price / bps).where(bps > 0, np.nan)
```

### 해석

PBR 단독으로 쓰지 않고 다음에 나올 ROE 와 결합해서 보는 게 정석이에요. **PBR 이 낮은데 ROE 가 높으면** 진짜 저평가일 가능성이 높고, **PBR 이 낮고 ROE 도 낮으면** 자본을 굴리지 못하는 "값싼 이유" 가 있는 회사인 거죠.

<br>

<br>



## 4. ROE — 자기자본이익률

### 정의

$$
\text{ROE} = \frac{\text{Net Income}}{\text{Average Equity}}
$$

평균자본은 보통 $(E_{\text{begin}} + E_{\text{end}}) / 2$ 로 잡습니다.

### 왜 이렇게 계산하나

- 회사가 "**주주가 맡긴 돈으로 이익을 얼마나 만들어냈나**" 를 직접 보는 비율이에요. 자본수익률.
- **평균을 쓰는 이유** — 분자(이익)는 한 해 동안 누적된 양인데 분모를 기말 자본만 쓰면 기간이 안 맞아요. 자본조달이 기중에 일어나면 더 심하게 어긋납니다.
- ROE 는 **Du Pont 분해** 로 풀어보면 의미가 더 또렷해져요.

$$
\text{ROE} = \underbrace{\frac{\text{Net Income}}{\text{Sales}}}_{\text{순이익률}} \cdot \underbrace{\frac{\text{Sales}}{\text{Assets}}}_{\text{자산회전율}} \cdot \underbrace{\frac{\text{Assets}}{\text{Equity}}}_{\text{재무레버리지}}
$$

같은 ROE 라도 "**마진이 높은 ROE**" 와 "**레버리지가 큰 ROE**" 는 위험 프로필이 전혀 달라요.

### Python

> **`roe`**  
> 입력: `net_income`, `equity_begin`, `equity_end` (`pd.Series`) — 같은 인덱스의 회기 순이익/기초자본/기말자본.  
> 반환: `pd.Series` — 같은 인덱스의 ROE. 평균자본 ≤ 0 인 시점은 `NaN`.

> **`dupont`**  
> 입력: `net_income`, `sales`, `assets`, `equity` (`pd.Series`) — 같은 인덱스로 정렬된 4개 시리즈.  
> 반환: `pd.DataFrame` — 같은 인덱스, 컬럼 `margin`/`turnover`/`leverage`/`ROE`. ROE = 셋의 곱.

```python
def roe(net_income: pd.Series, equity_begin: pd.Series, equity_end: pd.Series) -> pd.Series:
    avg_eq = (equity_begin + equity_end) / 2
    return (net_income / avg_eq).where(avg_eq > 0, np.nan)

def dupont(net_income, sales, assets, equity):
    margin = net_income / sales
    turnover = sales / assets
    leverage = assets / equity
    return pd.DataFrame({"margin": margin, "turnover": turnover, "leverage": leverage,
                         "ROE": margin * turnover * leverage})
```

### 해석

ROE 가 **자본비용($k_e$)** 보다 높아야 진짜 가치를 만드는 회사예요. ROE 15% 가 좋아 보여도 자본비용이 18% 면 가치 파괴 중인 셈.

<br>

<br>



## 5. 단순 수익률 vs 로그 수익률

### 정의

단순 수익률(simple return):

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

로그 수익률(log return):

$$
r_t^{\log} = \ln\!\left(\frac{P_t}{P_{t-1}}\right) = \ln P_t - \ln P_{t-1}
$$

> 💡 기호 풀기
> - $P_t$ — "피 티". 첨자(아래에 작게 붙는 글자)는 **시점 인덱스**. $P_{t-1}$ 은 한 시점 전 가격.
> - $\ln$ — "엘 엔". 자연로그(밑이 $e \approx 2.718$ 인 로그). **곱셈을 덧셈으로 바꿔주는 함수** 라서, 수익률처럼 "곱해서 누적되는 값" 을 다룰 때 편해요.
> - $\sum_{t=1}^{T}$ — "시그마". $t = 1$ 부터 $T$ 까지 차례로 다 더하라는 **합 기호** 예요.
> - 분수는 한국어로 **분모부터** 읽어요. $\frac{P_t}{P_{t-1}}$ → "피 티 마이너스 일 분의, 피 티".

### 왜 로그 수익률을 자주 쓰나

핵심 이유는 **시간 가산성** 입니다. 일별 로그 수익률을 그냥 더하면 기간 수익률이 돼요.

$$
\sum_{t=1}^{T} r_t^{\log} = \sum_{t=1}^{T} \bigl(\ln P_t - \ln P_{t-1}\bigr) = \ln\!\left(\frac{P_T}{P_0}\right)
$$

단순 수익률은 이게 안 돼요. $(1+r_1)(1+r_2)\dots(1+r_T) - 1$ 로 곱해야 합니다. 이게 모델링 단계에서 차이를 만들어요.

- **정규성 가정** 이 필요한 모델(GARCH, 옵션 가격) → 로그 수익률이 더 정규에 가까움.
- **수익률 합산/평균** 하는 모든 통계량 → 로그 수익률이 자연스러움.
- 다만 **포트폴리오 단위 합산**(자산별 비중 가중) 에서는 단순 수익률이 정확해요. $r_p = \sum_i w_i r_i$ 가 그대로 성립.

### Python

> **`simple_return` / `log_return`**  
> 입력: `close` (`pd.Series`, `index=DatetimeIndex`) — 일별 종가.  
> 반환: `pd.Series` — 같은 인덱스의 일간 수익률. 첫 행은 `NaN` (직전 값이 없음).

```python
def simple_return(close: pd.Series) -> pd.Series:
    return close.pct_change()

def log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))
```

### 해석

작은 수익률에서는 $\ln(1+r) \approx r$ 이라 둘이 거의 같지만, 일변동이 큰 날(코로나 쇼크 같은)은 차이가 커요. **변동성/위험 모델링은 로그**, **포트폴리오 비중 가중은 단순** — 이 한 줄만 외우면 헷갈리지 않습니다.

<br>

<br>



## 6. RSI — 상대강도지수

### 정의

기간 $n$(보통 14)에 대해,

$$
U_t = \max(P_t - P_{t-1},\, 0), \qquad D_t = \max(P_{t-1} - P_t,\, 0)
$$

Wilder smoothing 을 적용한 평균:

$$
\bar{U}_t = \frac{(n-1)\,\bar{U}_{t-1} + U_t}{n}, \qquad \bar{D}_t = \frac{(n-1)\,\bar{D}_{t-1} + D_t}{n}
$$

상대강도와 RSI:

$$
\text{RS}_t = \frac{\bar{U}_t}{\bar{D}_t}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}
$$

> 💡 기호 풀기
> - $\max(a, 0)$ — "맥스". 두 값 중 **더 큰 쪽**. 여기선 음수를 0 으로 잘라내는 트릭으로 써요. 가격이 내려간 날은 $U_t = 0$ 이 됩니다.
> - $\bar{U}_t$ — "유 바". 문자 위 막대(`bar`)는 **"평균값"** 이라는 표기 관습이에요. $\bar{U}$ 는 $U$ 의 평균.
> - $\text{RS}_t$ — t 시점의 RS 값. RS 는 "Relative Strength" 약자 그대로.

### 왜 이렇게 계산하나

- **상승폭/하락폭을 따로 평균** 내는 이유 — 단순 수익률 평균은 양수/음수가 상쇄돼서 "추세의 강도" 가 안 잡혀요. 절대값으로 떼서 평균하면 "올라가는 힘 vs 내려가는 힘" 의 비율이 됩니다.
- **Wilder smoothing** 은 사실상 $\alpha = 1/n$ 인 지수이동평균이에요. 일반 EMA(2/(N+1)) 보다 더 매끄럽고, 한 번 값이 큰 충격에 흔들리는 양이 적습니다.
- **0~100 으로 정규화** 하는 이유 — $\text{RS} \in [0, \infty)$ 라 그대로 쓰면 시각화/비교가 어려워서, $100 - 100/(1+\text{RS})$ 변환으로 범위를 잘라요. RS = 1 이면 RSI = 50.

### Python

> **`rsi`**  
> 입력: `close` (`pd.Series`, `index=DatetimeIndex`) — 일별 종가. `n` (`int`) — Wilder smoothing 기간(보통 14).  
> 반환: `pd.Series` — 같은 인덱스의 RSI 값(0~100). 초반 `n` 칸은 사실상 워밍업 구간.

```python
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder smoothing == EWM with alpha = 1/n, adjust=False
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = down.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_up / avg_dn
    return 100 - 100 / (1 + rs)
```

### 해석

- **RSI > 70 과매수, < 30 과매도** 는 관용 기준이지만 추세장에서는 위쪽에 한참 붙어있을 수 있어요. 절대 임계값보다 **다이버전스**(가격은 신고가인데 RSI 는 직전 고점을 못 넘는 등)가 더 정보량이 큽니다.
- 14일은 Wilder 의 원논문 기본값. 단기 트레이딩은 9~10, 장기는 21~25 도 자주 써요.

<br>

<br>



## 7. MACD — 이동평균 수렴/발산

### 정의

EMA(지수이동평균):

$$
\alpha_N = \frac{2}{N+1}, \qquad \text{EMA}_N(P)_t = \alpha_N P_t + (1 - \alpha_N)\,\text{EMA}_N(P)_{t-1}
$$

MACD 선과 시그널 선, 히스토그램:

$$
\text{MACD}_t = \text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t
$$

$$
\text{Signal}_t = \text{EMA}_{9}(\text{MACD})_t, \qquad \text{Hist}_t = \text{MACD}_t - \text{Signal}_t
$$

> 💡 기호 풀기
> - $\alpha$ — "알파". 그리스 문자. 여기선 EMA 의 **가중치** (0~1 사이 비율) 로 써요. $\alpha_N$ 처럼 첨자로 어떤 기간 $N$ 의 가중치인지 구분.
> - $\text{EMA}_N(P)_t$ — "이엠에이 N, t 시점". 가격 $P$ 에 기간 $N$ 으로 EMA 를 씌운 값을 $t$ 시점에서 본 것. 식이 어렵다 싶으면 **"최근 값에 더 큰 비중을 두는 평균"** 한 줄로 우선 읽고 넘어가도 OK.
> - $\text{MACD}_t, \text{Signal}_t, \text{Hist}_t$ — 같은 시점 $t$ 의 세 값. **MACD ↔ Signal 의 차이가 Hist** 라는 구조만 잡으면 끝.

### 왜 이렇게 계산하나

- **EMA - EMA** 의 차이는 단순이동평균 차이와 달리, **가까운 과거에 더 큰 가중치** 를 줘서 추세 전환을 더 빨리 잡습니다.
- 단기 EMA(12)가 장기 EMA(26)를 위로 뚫으면 MACD > 0 → 단기 추세가 장기 추세 위로 올라옴 → 매수 신호로 해석.
- **시그널(EMA9)을 한 번 더 씌우는 이유** — MACD 자체가 노이즈에 흔들리니까, MACD 의 추세를 다시 한 번 추적하기 위함이에요. 둘이 교차하는 시점이 실질적인 진입 시그널.
- **히스토그램** 은 MACD 와 시그널의 차이라서, 0 을 위/아래로 가르는 시점이 "추세 가속/감속" 의 변곡점.

### Python

> **`macd`**  
> 입력: `close` (`pd.Series`, `index=DatetimeIndex`) — 일별 종가. `fast`/`slow`/`signal` (`int`) — 단기/장기/시그널 EMA 기간(기본 12/26/9).  
> 반환: `pd.DataFrame` — `close` 와 같은 인덱스, 컬럼 `macd`(단기-장기), `signal`(MACD 의 EMA9), `hist`(둘의 차).

```python
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})
```

`span=N` 이 위 정의의 $\alpha = 2/(N+1)$ 과 정확히 일치해요. `adjust=False` 가 핵심 — 기본값 `True` 면 초반 구간에서 가중치를 정규화해서 수치가 살짝 다르게 나옵니다.

### 해석

MACD 는 **추세추종**이라 횡보장에서는 거짓 신호가 많아요. RSI(역추세) 와 같이 보는 게 정석이에요. "MACD 골든크로스인데 RSI 도 50 이상 추세전환" 같은 **2중 확인** 으로 진입.

<br>

<br>



## 8. Bollinger Bands — 변동성 밴드

### 정의

기간 $N$(보통 20), 표준편차 배수 $k$(보통 2):

$$
\text{MB}_t = \frac{1}{N}\sum_{i=t-N+1}^{t} P_i, \qquad
\sigma_t = \sqrt{\frac{1}{N}\sum_{i=t-N+1}^{t}(P_i - \text{MB}_t)^2}
$$

$$
\text{UB}_t = \text{MB}_t + k\,\sigma_t, \qquad \text{LB}_t = \text{MB}_t - k\,\sigma_t
$$

> 💡 기호 풀기
> - $\sigma$ — "시그마(소문자)". **표준편차** 예요. 데이터가 평균에서 얼마나 떨어져 있는지를 잰 값. $\sigma$ 가 크면 변동이 심한 것.
> - $(P_i - \text{MB}_t)^2$ — 각 가격이 평균으로부터 떨어진 거리를 **제곱** 한 것. 제곱하는 이유는 +/- 부호가 상쇄되지 않게 양수로 통일하려고.
> - $\sum_{i=t-N+1}^{t}$ — "최근 $N$ 일" 의 의미예요. $t$ 가 오늘이면 $t-N+1$ 이 $N$ 일 전 → 끝 인덱스에서 거꾸로 $N$ 칸.
> - $\text{MB, UB, LB}$ — Middle Band / Upper Band / Lower Band 약자 그대로.

### 왜 이렇게 계산하나

- 정규분포 가정 아래에서 가격이 $\text{MB} \pm 2\sigma$ 안에 있을 확률은 약 **95.4%** 입니다. 그래서 $k=2$ 가 "통계적으로 의미있는 이탈" 의 기준선처럼 쓰여요.
- 분모가 $N$ 이냐 $N-1$ 이냐 차이가 있는데, Bollinger 원안은 **모표준편차(N으로 나눔)** 입니다. `pandas.rolling.std()` 는 기본이 표본표준편차($N-1$)라서 `ddof=0` 으로 맞춰야 해요.
- **밴드 폭(UB - LB)** 자체가 변동성 지표예요. 좁아지면(squeeze) 변동성 압축 → 곧 큰 움직임이 나올 가능성이 높다는 식의 해석이 가능합니다.

### Python

> **`bollinger`**  
> 입력: `close` (`pd.Series`, `index=DatetimeIndex`) — 일별 종가. `n` (`int`) — 윈도우(기본 20). `k` (`float`) — 표준편차 배수(기본 2).  
> 반환: `pd.DataFrame` — `close` 와 같은 인덱스, 컬럼 `mb`(중심), `ub`(상단), `lb`(하단), `width`((UB-LB)/MB, 변동성 압축 모니터링용).

```python
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mb = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)   # 모표준편차
    ub = mb + k * sd
    lb = mb - k * sd
    width = (ub - lb) / mb              # 변동성 압축 모니터링용
    return pd.DataFrame({"mb": mb, "ub": ub, "lb": lb, "width": width})
```

### 해석

- **밴드 터치 = 매매 신호** 는 단순한 해석이고, 실제로는 **추세장에서는 상단 밴드를 따라 올라가는** "워킹 더 밴드" 가 잦아요.
- 더 안전한 해석은 **밴드 폭** + **MACD** 조합. 밴드가 압축돼 있다가 확장 시작하는 시점 + MACD 가 0선 돌파면 추세 전환의 1차 후보로 봅니다.

<br>

<br>



## 9. Sharpe Ratio — 위험조정수익

### 정의

$$
\text{Sharpe} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p}
$$

일별 데이터로 추정하고 연환산할 때(주식 거래일 252일 기준):

$$
\text{Sharpe}_{\text{ann}} = \frac{\bar{r} - r_f^{\text{daily}}}{s} \cdot \sqrt{252}
$$

여기서 $\bar{r}, s$ 는 일별 (로그) 수익률의 평균과 표준편차예요.

> 💡 기호 풀기
> - $\mathbb{E}[\cdot]$ — "기댓값". 확률적인 평균이에요. 실제 데이터로 계산할 땐 표본평균 $\bar{r}$ 로 대체.
> - $R_p, R_f$ — 포트폴리오 수익률(p = portfolio) 과 무위험수익률(f = risk-free). 첨자는 **"종류를 가리키는 라벨"** 역할이에요. 시점 인덱스가 아니라는 점만 주의.
> - $\sqrt{252}$ — "루트 252". 1 년 거래일이 약 252 일. 분산은 시간에 비례하고, 표준편차는 그 제곱근에 비례해서 $\sqrt{252}$ 가 등장합니다.

### 왜 이렇게 계산하나

- **분모를 표준편차로 나누는 의미** — 같은 평균수익률이라도 변동성이 크면 실제 손에 들어오는 결과는 더 불확실해요. 그 불확실성을 페널티로 깔자는 게 핵심 아이디어.
- $\sqrt{252}$ **로 곱하는 이유** — 분산은 시간에 비례(독립가정), 표준편차는 $\sqrt{\text{시간}}$ 에 비례. 평균은 시간에 비례하니까 분자는 $\times 252$, 분모는 $\times \sqrt{252}$ → 전체 $\sqrt{252}$ 배가 됩니다.
- 무위험수익률($R_f$)을 빼는 이유 — "내가 위험을 감수해서 얻은 **추가** 수익" 만 평가하려는 거예요. 예금 이자보다 못한 펀드는 Sharpe 가 음수가 돼야 정상.

### Python

> **`sharpe`**  
> 입력: `returns` (`pd.Series`, `index=DatetimeIndex`) — 일별 (로그) 수익률. `rf_annual` (`float`) — 연 무위험금리(0.03 = 3%). `periods_per_year` (`int`) — 연환산 시점 수(주식 252).  
> 반환: `float` — 연환산 Sharpe.

```python
def sharpe(returns: pd.Series, rf_annual: float = 0.03,
           periods_per_year: int = 252) -> float:
    rf_daily = rf_annual / periods_per_year
    excess = returns - rf_daily
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(periods_per_year)
```

### 해석

- **Sharpe > 1** 정도면 "괜찮은" 전략, **> 2** 면 "꽤 좋은" 전략. 다만 **단점이 큼** — 표준편차는 위/아래 변동성을 똑같이 페널티로 잡기 때문에, **수익률 분포가 비대칭**(꼬리위험 큰 경우)이면 Sharpe 가 과대평가될 수 있어요.
- 그래서 실무에서는 **Sortino**(하방 편차만 쓰는 Sharpe 변형)와 같이 봅니다.

$$
\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{down}}}, \quad
\sigma_{\text{down}} = \sqrt{\mathbb{E}\bigl[\min(r - r_f, 0)^2\bigr]}
$$

<br>

<br>



## 10. Beta — 시장 민감도

### 정의

종목 $i$ 의 시장 베타:

$$
\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}
$$

OLS 회귀 형태로 보면 $R_i = \alpha_i + \beta_i R_m + \varepsilon_i$ 의 회귀계수 그 자체예요.

> 💡 기호 풀기
> - $\text{Cov}(X, Y)$ — "공분산". $X$ 가 평균보다 클 때 $Y$ 도 평균보다 큰 경향이 있는지(같이 움직이는지)를 잰 값. 양수면 같은 방향, 음수면 반대 방향.
> - $\text{Var}(X)$ — "분산". $X$ 가 자기 평균에서 얼마나 흩어져 있는지. **표준편차 $\sigma$ 의 제곱($\sigma^2$)** 이에요.
> - $\alpha_i, \beta_i, \varepsilon_i$ — 회귀식의 절편($\alpha$, 알파), 기울기($\beta$, 베타), 오차($\varepsilon$, 엡실론). 첨자 $i$ 는 **종목 인덱스** 예요.

### 왜 이렇게 계산하나

- **공분산 / 시장분산** 의 의미 — 시장이 1% 움직일 때 이 종목이 평균적으로 몇 % 움직이는지의 비율이에요. 그래서 $\beta=1$ 은 "시장과 같이", $\beta=1.5$ 는 "시장보다 1.5배 출렁", $\beta < 0$ 은 "시장과 반대로".
- CAPM 의 기대수익률 모형 $\mathbb{E}[R_i] = R_f + \beta_i(\mathbb{E}[R_m] - R_f)$ 에서 베타가 그대로 위험 프리미엄의 계수로 들어가요. 그래서 **베타 = 시장위험의 노출도** 라고 해석.
- 분산 $\text{Var}(R_m)$ 으로 나누는 이유는 회귀의 정의 그 자체. 두 변수 사이의 선형 의존성을 시장 분산에 대해 정규화한 양입니다.

### Python

> **`beta`**  
> 입력: `stock_ret`, `market_ret` (`pd.Series`) — 같은 인덱스의 일별 수익률.  
> 반환: `float` — 전 기간 평균 시장 베타.

> **`rolling_beta`**  
> 입력: 위와 동일 + `window` (`int`) — 롤링 윈도우 길이(거래일).  
> 반환: `pd.Series` (`index=date`) — 시점별 롤링 베타. 초반 `window-1` 칸은 `NaN`.

```python
def beta(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    cov = df.cov().iloc[0, 1]
    var_m = df.iloc[:, 1].var(ddof=1)
    return cov / var_m

def rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = stock_ret.rolling(window).cov(market_ret)
    var_m = market_ret.rolling(window).var(ddof=1)
    return cov / var_m
```

회귀로 직접 풀어도 동일해요.

> **`beta_ols`**  
> 입력: `stock_ret`, `market_ret` (`pd.Series`) — 같은 인덱스의 일별 수익률.  
> 반환: `statsmodels` OLS 결과 객체. `.params.iloc[1]` 이 베타, `.params.iloc[0]` 이 알파 절편.

```python
import statsmodels.api as sm

def beta_ols(stock_ret: pd.Series, market_ret: pd.Series):
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    X = sm.add_constant(df.iloc[:, 1])
    return sm.OLS(df.iloc[:, 0], X).fit()   # params.iloc[1] 이 beta
```

### 해석

- **베타는 시간에 따라 변합니다**. 한 시점에서 잰 베타를 5년 뒤에 그대로 쓰면 위험합니다. 보통 **60~252일 롤링 베타** 를 같이 봐요.
- **저베타 ≠ 안전** — 베타는 시장 방향성에만 대한 민감도라서, 베타가 낮아도 종목 고유 위험(idiosyncratic risk)이 클 수 있어요. 분산투자로 줄어드는 건 후자입니다.

<br>

<br>



## 11. 한 번에 묶어 쓰는 패턴

위 함수들을 같은 `df` 위에서 한 번에 만들면 보통 이렇게 됩니다.

> **`build_factors`**  
> 입력: `df` (`pd.DataFrame`, `index=DatetimeIndex`) — `open`/`high`/`low`/`close`/`volume` 컬럼 보유의 OHLCV. `market` (`pd.Series`, 같은 인덱스) — 시장지수 종가.  
> 반환: `pd.DataFrame` — `df` 와 같은 인덱스, 컬럼 `ret`/`rsi14`/`macd`/`macd_sig`/`macd_hist`/`bb_mb`/`bb_ub`/`bb_lb`/`bb_w`/`beta60`. 펀더멘털 팩터(EPS/PER 등)는 분기 데이터라 별도 매핑 후 합치는 게 안전.

```python
def build_factors(df: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret"] = log_return(df["close"])
    out["rsi14"] = rsi(df["close"], n=14)

    macd_df = macd(df["close"])
    out[["macd", "macd_sig", "macd_hist"]] = macd_df[["macd", "signal", "hist"]]

    bb = bollinger(df["close"], n=20, k=2)
    out[["bb_mb", "bb_ub", "bb_lb", "bb_w"]] = bb

    out["beta60"] = rolling_beta(out["ret"], log_return(market), window=60)
    return out
```

> ✅ 같은 인덱스로 정렬된 OHLCV + 시장지수만 있으면, 위 한 함수로 **가격 기반 팩터 + 베타** 를 한 번에 뽑을 수 있어요. 펀더멘털 팩터(EPS/PER/PBR/ROE)는 분기 데이터라서 별도의 일자 매핑(예: `reindex().ffill()`)을 거쳐서 합쳐주면 OK.

<br>

<br>



## 12. 자주 빠지는 함정

- **Look-ahead bias** — 분기 펀더멘털을 사용할 때 "공시일" 이 아니라 "결산일" 기준으로 붙이면 미래 정보를 미리 본 백테스트가 돼요. 늘 **공시 가능 시점 이후** 의 인덱스로 ffill.
- **Survivorship bias** — 상장폐지된 종목을 데이터셋에서 빼고 백테스트하면 과거 성과가 부풀려져요. 폐지 종목까지 포함된 데이터 소스를 쓰세요.
- **결측치 처리** — RSI/MACD/Bollinger 는 모두 워밍업 구간에 `NaN` 이 생깁니다. 백테스트 진입 신호를 만들 때는 워밍업 끝난 이후 인덱스로만 트레이딩.
- **표본/모표준편차 차이** — Bollinger 는 모표준편차(`ddof=0`), Sharpe 는 표본표준편차(`ddof=1`) 가 관례. 작아 보여도 값이 살짝 달라져서 시각화/리서치 비교가 안 맞을 수 있어요.

<br>

<br>



일단 오늘은 여기까지.....   
다음 글에서는 위 팩터들을 같은 데이터프레임에 묶어서 **간단한 멀티팩터 알파를 만들고 백테스트** 하는 흐름을 정리해볼게요.
