---
layout: single
title:  "(1/2) 미국 금융인들이 보는 거시 지표 — 어디서 보고 Python 으로 어떻게 가져오나"
date: 2026-06-06 21:00:00 +0900
categories: finance
tag: [macro, FRED, CPI, PCE, 고용지표, 기준금리, 장단기금리차, VIX, yfinance, pandas-datareader, 경제지표, python]
author_profile: false
toc: true
description: "월가가 매달 챙겨 보는 거시 지표(CPI · PCE · 고용 · 기준금리 · 장단기 금리차 · VIX)를 어디서 확인하고, FRED · yfinance · pandas-datareader 로 Python 에서 직접 받아오는 패턴까지 한 글에 정리했어요."
---

{% include series-macro-indicators.html current="1" %}

## Summary

미국 주식이나 채권을 조금이라도 보다 보면 "이번 주 CPI 가…", "PCE 가 예상보다…", "장단기 금리차가 역전돼서…" 같은 말을 계속 만나게 돼요. 이게 다 미국 금융인들이 매달, 매주 챙겨 보는 **거시 지표(macro indicator)** 들입니다.

문제는 이게 종류도 많고, 출처도 제각각이고(BLS, BEA, 연준, ISM…), "그래서 이걸 어디서 봐야 하지?" 가 항상 막막하다는 거예요. 그리고 한 번 어디서 보는지 알고 나면, 그 다음 궁금증은 똑같이 흘러갑니다. "이거 Python 으로 받아서 그래프로 그릴 수는 없나?"

이 글은 그 두 가지를 한 번에 정리해요. **대표 지표가 뭔지 → 1차 출처가 어디인지 → Python 으로 어떻게 받아오는지** 순서로, 복붙하면 바로 도는 코드까지 같이 둡니다.

> 💡 이 글에서 다루는 것
> - 거시 지표를 4묶음(인플레이션 · 고용 · 성장 · 금리/시장)으로 큰 그림 잡기
> - 각 지표의 1차 출처와 발표 주기 — 어디서 진짜 원본을 보나
> - `FRED` API 키 발급 + `fredapi` · `pandas-datareader` · `yfinance` 설치
> - CPI · PCE · 실업률 · 기준금리 · 장단기 금리차 · VIX 를 Python 으로 받는 패턴
> - 전년 대비(YoY) 인플레이션 계산, 침체 신호로 쓰는 장단기 금리차
> - 여러 지표를 한 번에 받아서 미니 대시보드 만들기

<br>

<br>



## 0. 큰 그림 — 지표는 결국 4묶음

지표 이름이 수십 개라 처음엔 압도되는데, 연준과 시장이 실제로 보는 흐름은 크게 네 덩어리예요. 이 틀만 잡으면 새 지표를 만나도 "아 이건 고용 쪽이구나" 하고 자리를 찾을 수 있어요.

| 묶음 | 질문 | 대표 지표 |
|------|------|-----------|
| 인플레이션 | 물가가 오르나? | CPI, Core CPI, PCE, Core PCE, PPI |
| 고용 | 사람들이 일하고 있나? | 비농업 고용(NFP), 실업률, 신규 실업수당 청구, JOLTS |
| 성장·활동 | 경제가 돌아가나? | GDP, 소매판매, 산업생산, ISM PMI |
| 금리·시장 | 돈값과 분위기는? | 기준금리, 국채 2년·10년물, 장단기 금리차, VIX |

연준의 목표가 **물가 안정 + 완전 고용** 두 가지라(이중 책무, dual mandate), 위 표에서 인플레이션과 고용이 가장 무겁게 다뤄져요. 나머지 둘은 그 두 개를 둘러싼 배경이라고 보면 편해요.

<br>

<br>



## 1. 어디서 보나 — 1차 출처 지도

지표마다 그걸 만들어 발표하는 **기관(1차 출처)** 이 정해져 있어요. 뉴스는 결국 이 원본을 받아서 옮기는 거라, 원본 위치를 알면 가장 빠르고 정확해요.

| 출처 | 무엇을 발표하나 | 주소 |
|------|----------------|------|
| BLS (노동통계국) | CPI, 비농업 고용,<br>실업률, PPI | <https://www.bls.gov> |
| BEA (경제분석국) | GDP,<br>PCE(개인소비지출) | <https://www.bea.gov> |
| Federal Reserve<br>(연준) | 기준금리(FOMC),<br>산업생산 | <https://www.federalreserve.gov> |
| ISM | 제조업·서비스 PMI | <https://www.ismworld.org> |
| U.S. Treasury | 국채 금리(yield curve) | <https://home.treasury.gov> |
| CME FedWatch | 기준금리 인상/인하 확률 | <https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html> |

그런데 이걸 기관별로 일일이 돌아다니면 너무 번거로워요. 그래서 실무에선 한 군데로 모아주는 허브를 씁니다.

> 💡 FRED 가 사실상 표준 허브  
> 세인트루이스 연준이 운영하는 [FRED(Federal Reserve Economic Data)](https://fred.stlouisfed.org) 는 위 기관들의 시계열을 거의 다 끌어모아 둔 곳이에요. 80만 개가 넘는 시계열에 **시리즈 ID** 가 붙어 있고(예: CPI 는 `CPIAUCSL`), 무료 API 까지 열어줘요. 이 글의 Python 예시도 대부분 FRED 를 거칩니다.

발표 일정 자체를 한눈에 보고 싶으면 **경제 캘린더** 가 편해요. [Investing.com 경제 캘린더](https://www.investing.com/economic-calendar/) 나 [Trading Economics](https://tradingeconomics.com/calendar) 에서 "다음 CPI 발표가 언제, 예상치는 얼마"까지 한 화면에서 볼 수 있어요.

<br>

<br>



## 2. Python 준비 — 라이브러리 3개 + FRED 키

거시 지표를 Python 으로 받을 때 쓰는 도구는 사실상 세 개면 충분해요.

| 라이브러리 | 쓰는 곳 | API 키 |
|------------|---------|--------|
| `fredapi` | FRED 시계열 (정석) | 필요 |
| `pandas-datareader` | FRED · Stooq 등 (키 없이도 FRED 됨) | FRED 소스는 불필요 |
| `yfinance` | 주가지수·VIX·시장 데이터 | 불필요 |

설치는 한 줄이에요.

```shell
pip install fredapi pandas-datareader yfinance
```

### 2-1. FRED API 키 발급

`fredapi` 를 쓰려면 무료 키가 하나 필요해요. [FRED API Keys 페이지](https://fredaccount.stlouisfed.org/apikeys) 에서 계정 만들고 "Request API Key" 누르면 바로 발급돼요. 발급된 키는 코드에 그대로 박지 말고 환경변수로 빼두는 걸 추천드려요.

```shell
export FRED_API_KEY="<FRED_API_KEY>"
```

```python
import os
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])
```

> ✅ 키 없이 시작하고 싶다면  
> `pandas-datareader` 의 FRED 소스는 키 없이도 동작해요. "일단 그래프부터 보자" 단계라면 이쪽이 더 빠릅니다. 본격적으로 여러 시리즈를 자주 받을 거면 `fredapi` + 키 조합이 안정적이에요.

```python
import pandas_datareader.data as web

# 키 없이 바로 — CPI 받아오기
cpi = web.DataReader("CPIAUCSL", "fred", "2020-01-01")
print(cpi.tail(3))
```

```text
            CPIAUCSL
DATE                
2026-03-01   320.421
2026-04-01   321.087
2026-05-01   321.755
```

> ⚠️ 위 출력의 날짜·숫자는 예시예요. 실제로 돌리면 그 시점의 최신 값이 찍혀요. CPI 는 "지수(index)" 라 숫자 자체(예: 321.7)보다 **변화율** 이 중요해요. 변화율 계산은 바로 다음 절에서 다룹니다.

<br>

<br>



## 3. 인플레이션 — CPI · Core CPI · PCE

물가 지표는 시장이 가장 예민하게 반응하는 묶음이에요. 발표 당일 주가·채권이 크게 출렁이는 게 보통 이쪽입니다.

| 지표 | FRED 코드 | 의미 |
|------|-----------|------|
| CPI | `CPIAUCSL` | 소비자물가지수 (헤드라인) |
| Core CPI | `CPILFESL` | 식품·에너지 뺀 근원 CPI |
| PCE | `PCEPI` | 개인소비지출 물가지수 |
| Core PCE | `PCEPILFE` | 근원 PCE — **연준이 가장 중시** |
| PPI | `PPIACO` | 생산자물가지수 (선행 성격) |

뉴스에 "근원(Core)" 이 자주 나오는 건, 식품·에너지는 출렁임이 커서 빼고 봐야 추세가 보이기 때문이에요. 특히 **Core PCE 는 연준이 금리 결정의 기준으로 삼는 지표** 라 가장 무겁게 봐야 해요.

### 3-1. 전년 대비(YoY) 인플레이션 계산

CPI 원본은 지수라서, 우리가 뉴스에서 듣는 "물가 3.2% 상승" 으로 바꾸려면 12개월 전 대비 변화율로 환산해야 해요. 월간 데이터니까 `pct_change(12)` 면 끝이에요.

```python
import pandas_datareader.data as web

cpi = web.DataReader("CPIAUCSL", "fred", "2018-01-01")["CPIAUCSL"]

# 12개월 전 대비 변화율(%) = 우리가 아는 "전년 대비 물가상승률"
cpi_yoy = cpi.pct_change(12) * 100
print(cpi_yoy.dropna().tail(3).round(2))
```

```text
DATE
2026-03-01    3.05
2026-04-01    2.94
2026-05-01    2.88
```

`pct_change(12)` 가 "지금 값 ÷ 12개월 전 값 − 1" 을 해주는 거라, 결과는 곧바로 전년 대비 상승률이 돼요. 연준의 목표가 2% 라, 위 예시처럼 2%대 후반이면 "아직 목표보다 살짝 높네" 정도로 읽으면 돼요.

### 3-2. 헤드라인 vs 근원 같이 보기

헤드라인과 근원을 나란히 두면 "에너지 때문에 튄 건지, 추세적으로 오르는 건지" 가 한눈에 보여요.

```python
import pandas_datareader.data as web

codes = {"CPIAUCSL": "CPI", "CPILFESL": "Core CPI"}
df = web.DataReader(list(codes), "fred", "2018-01-01").rename(columns=codes)

yoy = df.pct_change(12).mul(100).dropna()
print(yoy.tail(3).round(2))
```

```text
                CPI  Core CPI
DATE                         
2026-03-01     3.05      3.21
2026-04-01     2.94      3.15
2026-05-01     2.88      3.08
```

근원(Core)이 헤드라인보다 천천히 내려오는 모습이 자주 보이는데, 이게 "물가의 끈적함(stickiness)" 이라고 부르는 부분이에요. 연준이 금리 인하를 망설이는 이유가 보통 여기 있어요.

<br>

<br>



## 4. 고용 — 비농업 고용 · 실업률 · 신규 실업수당 청구

고용은 인플레이션과 함께 연준 이중 책무의 한 축이에요. 매달 첫째 주 금요일의 **고용보고서(Jobs Report)** 가 그 달 가장 큰 이벤트 중 하나입니다.

| 지표 | FRED 코드 | 발표 주기 | 포인트 |
|------|-----------|-----------|--------|
| 비농업 고용 (NFP) | `PAYEMS` | 매월 | 전월 대비 "증감" 으로 봄 |
| 실업률 | `UNRATE` | 매월 | 낮을수록 뜨거운 노동시장 |
| 신규 실업수당 청구 | `ICSA` | **매주** | 가장 빠른 고용 체온계 |
| 구인건수 (JOLTS) | `JTSJOL` | 매월 | 일자리 수요 |

비농업 고용은 절대 수치보다 **전월 대비 증감(몇만 명 늘었나)** 으로 봐요. FRED 의 `PAYEMS` 는 누적 고용자 수라, `diff()` 로 증감을 뽑아야 뉴스에서 듣는 "이번 달 +18만 명" 같은 숫자가 나와요.

```python
import pandas_datareader.data as web

payems = web.DataReader("PAYEMS", "fred", "2024-01-01")["PAYEMS"]

# 전월 대비 증감(천 명 단위) → 1000 곱해 "명" 으로
nfp_change = payems.diff().dropna() * 1000
print(nfp_change.tail(3).astype(int))
```

```text
DATE
2026-03-01    192000
2026-04-01    165000
2026-05-01    158000
```

신규 실업수당 청구(`ICSA`)는 **주간** 이라 가장 빠르게 노동시장 온도를 알려줘요. 매주 목요일에 나오고, 갑자기 숫자가 튀면 "고용이 식고 있다" 는 조기 신호로 읽혀요.

```python
import pandas_datareader.data as web

claims = web.DataReader("ICSA", "fred", "2025-01-01")["ICSA"]
print(claims.tail(3).astype(int))
```

```text
DATE
2026-05-16    223000
2026-05-23    219000
2026-05-30    228000
```

<br>

<br>



## 5. 성장·활동 — GDP · 소매판매 · ISM PMI

경제가 실제로 얼마나 돌아가는지 보는 묶음이에요.

| 지표 | FRED 코드 | 주기 | 의미 |
|------|-----------|------|------|
| 실질 GDP | `GDPC1` | 분기 | 경제 규모 (물가 보정) |
| GDP 성장률 | `A191RL1Q225SBEA` | 분기 | 연율화 성장률(%) |
| 소매판매 | `RSAFS` | 매월 | 소비 체력 |
| 산업생산 | `INDPRO` | 매월 | 공장 가동 |

GDP 는 분기마다 나오고, 시장은 "전분기 대비 연율 몇 %" 형태를 봐요. FRED 에는 그 연율 성장률이 `A191RL1Q225SBEA` 로 따로 들어가 있어서 변환 없이 바로 읽을 수 있어요.

```python
import pandas_datareader.data as web

gdp = web.DataReader("A191RL1Q225SBEA", "fred", "2023-01-01")
print(gdp.tail(4).round(1))
```

```text
            A191RL1Q225SBEA
DATE                       
2025-06-01              2.4
2025-09-01              3.1
2025-12-01              2.6
2026-03-01              2.2
```

### 5-1. ISM PMI 는 FRED 에 없어요 (라이선스 함정)

여기서 한 번 막히는 분이 많아요. ISM 제조업·서비스 **PMI 는 ISM 의 저작권 정책 때문에 FRED 에서 빠졌어요.** `NAPM` 같은 옛 코드를 넣으면 안 나옵니다.

> ⚠️ ISM PMI 받는 법  
> PMI 원본 수치는 [ISM 공식 사이트](https://www.ismworld.org) 의 리포트나 경제 캘린더에서 확인하는 게 정석이에요. Python 으로 자동화하려면 별도 유료 데이터 공급사(예: Nasdaq Data Link)를 거쳐야 해요. 무료로 흐름만 보고 싶으면, FRED 에 있는 **산업생산(`INDPRO`)** 이나 **제조업 신규주문** 으로 대체 신호를 잡는 방법이 있어요.

50 이라는 기준선만 기억하면 PMI 읽기는 쉬워요. **50 위면 확장, 아래면 위축** 이에요.

<br>

<br>



## 6. 금리·통화정책 — 기준금리 · 국채 · 장단기 금리차

연준이 물가·고용을 보고 내리는 결론이 결국 **기준금리** 예요. 그리고 그 기준금리에 대한 시장의 기대가 **국채 금리** 로 나타나요.

| 지표 | FRED 코드 | 의미 |
|------|-----------|------|
| 실효 기준금리 | `FEDFUNDS` | 연준 정책금리(월평균) |
| 국채 3개월물 | `DGS3MO` | 초단기 금리 |
| 국채 2년물 | `DGS2` | 정책금리 기대 반영 |
| 국채 10년물 | `DGS10` | 장기 성장·물가 기대 |
| 10년−2년 스프레드 | `T10Y2Y` | **장단기 금리차** |
| 10년−3개월 스프레드 | `T10Y3M` | 연준이 더 선호하는 침체 신호 |

### 6-1. 장단기 금리차 — 침체의 단골 신호

보통은 돈을 오래 빌려줄수록 금리가 높아서, 10년물이 2년물보다 높아요(스프레드 양수). 그런데 이게 **역전(음수)** 되면, 즉 단기 금리가 장기보다 높아지면, 역사적으로 경기 침체를 앞두고 자주 나타난 신호로 유명해요.

FRED 에 `T10Y2Y` 라는 스프레드 시리즈가 아예 만들어져 있어서, 빼기 계산도 필요 없어요.

```python
import pandas_datareader.data as web

spread = web.DataReader("T10Y2Y", "fred", "2022-01-01")["T10Y2Y"]
latest = spread.dropna().iloc[-1]

print(f"최근 10Y-2Y 스프레드: {latest:.2f}%p")
print("상태:", "역전(침체 경고)" if latest < 0 else "정상(우상향)")
```

```text
최근 10Y-2Y 스프레드: 0.34%p
상태: 정상(우상향)
```

직접 빼서 만들고 싶다면 두 시리즈를 받아 빼면 돼요. 결과는 `T10Y2Y` 와 거의 같아요.

```python
import pandas_datareader.data as web

df = web.DataReader(["DGS10", "DGS2"], "fred", "2022-01-01")
df["spread"] = df["DGS10"] - df["DGS2"]
print(df[["DGS10", "DGS2", "spread"]].dropna().tail(3).round(2))
```

```text
            DGS10  DGS2  spread
DATE                           
2026-06-02   4.28  3.95    0.33
2026-06-03   4.30  3.97    0.33
2026-06-04   4.31  3.97    0.34
```

### 6-2. 다음 FOMC 에서 금리를 올릴까 내릴까

이건 지표라기보다 **시장의 기대 확률** 인데, 금융인들이 매일 봐요. CME 의 **FedWatch Tool** 이 기준금리 선물 가격에서 "다음 회의에서 인하할 확률 몇 %" 를 뽑아줘요. [CME FedWatch](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html) 에서 무료로 볼 수 있어요. 발표를 앞두고 이 확률이 어떻게 움직이는지가 단기 시장의 방향을 많이 좌우해요.

<br>

<br>



## 7. 시장 심리·리스크 — VIX · 신용 스프레드 · 주가지수

지표가 "데이터" 라면, 이 묶음은 "분위기" 예요. 그중 대표가 VIX 인데, 이건 FRED 에도 있고 `yfinance` 로도 받을 수 있어요.

| 지표 | 받는 법 | 의미 |
|------|---------|------|
| VIX | FRED `VIXCLS` / yfinance `^VIX` | 변동성·공포 지수 |
| 하이일드 스프레드 | FRED `BAMLH0A0HYM2` | 신용 위험 (CDS 대체) |
| 투자등급 스프레드 | FRED `BAMLC0A0CM` | 우량 회사채 가산금리 |
| S&P 500 | yfinance `^GSPC` | 대표 주가지수 |
| 나스닥 100 | yfinance `^NDX` | 기술주 |
| 소비자심리 | FRED `UMCSENT` | 미시간대 소비자심리지수 |

VIX 는 "공포 지수" 라는 별명대로, 시장이 불안할수록 올라가요. 보통 20 아래면 평온, 30 을 넘으면 시장이 꽤 긴장한 상태로 읽어요.

```python
import yfinance as yf

vix = yf.download("^VIX", start="2025-01-01", progress=False)["Close"]
print(vix.tail(3).round(2))
```

```text
Ticker        ^VIX
Date              
2026-06-02   16.84
2026-06-03   17.21
2026-06-04   16.55
```

S&P 500 지수도 같은 방식이에요. 티커만 `^GSPC` 로 바꾸면 돼요.

```python
import yfinance as yf

sp500 = yf.download("^GSPC", start="2025-01-01", progress=False)["Close"]
print(sp500.tail(3).round(2))
```

```text
Ticker          ^GSPC
Date                 
2026-06-02    6312.45
2026-06-03    6340.18
2026-06-04    6328.77
```

> 💡 Fear & Greed Index  
> CNN 의 [Fear & Greed Index](https://www.cnn.com/markets/fear-and-greed) 도 분위기 지표로 인기가 많아요. 0(극단적 공포)~100(극단적 탐욕) 한 숫자로 시장 심리를 요약해줘요. 공식 Python API 는 없지만, 비공식 엔드포인트를 `requests` 로 받아 쓰는 패키지들이 있어요.

### 7-1. CDS / 신용 스프레드 — 채권 시장의 공포

VIX 가 "주식 공포" 라면, **신용 위험** 은 채권 쪽 공포예요. 여기서 가장 많이 듣는 게 **CDS(Credit Default Swap)** 인데, "이 채권 발행자가 부도날 위험" 을 사고파는 보험이에요. 그 가격(CDS 스프레드, bp 단위)이 오르면 시장이 부도 위험을 크게 본다는 뜻이라, 신용 리스크의 온도계로 쓰여요. 국가 CDS(한국물 CDS 프리미엄 같은)나 `CDX`·`iTraxx` 같은 인덱스가 대표적이에요.

> ⚠️ CDS 원본은 FRED 에 없어요  
> CDS 데이터는 Markit·S&P Global, Bloomberg·Refinitiv 같은 유료 단말의 독점 데이터예요. 1편에서 쓴 무료 FRED·`yfinance` 루트로는 받을 수 없습니다. 그래서 무료로 같은 신호를 잡을 땐 **신용 스프레드(credit spread)** 를 대체재로 써요. CDS 와 스프레드는 둘 다 신용 위험의 가격이라 거의 같이 움직이거든요.

FRED 에 있는 **하이일드 스프레드(`BAMLH0A0HYM2`)** 가 가장 실용적인 대체재예요. 정크본드(투기등급 회사채)에 시장이 요구하는 가산금리라, 위험회피 심리가 커지면 같이 치솟아요.

```python
import pandas_datareader.data as web

hy = web.DataReader("BAMLH0A0HYM2", "fred", "2007-01-01")["BAMLH0A0HYM2"]
print(hy.tail(3))
```

```text
DATE
2026-06-02    3.12
2026-06-03    3.18
2026-06-04    3.15
```

단위는 %p 예요. 3%대면 평온한 상태고, 2008년 금융위기·2020년 코로나 충격 때는 10%를 훌쩍 넘었어요. 그래서 이 숫자가 갑자기 벌어지면 "채권 시장이 긴장하고 있다" 는 신호로 읽어요. 더 우량한 회사채 쪽을 보고 싶으면 투자등급 스프레드(`BAMLC0A0CM`)로 코드만 바꾸면 돼요.

<br>

<br>



## 8. 한 번에 받아서 미니 대시보드 만들기

지표를 하나씩 받는 법을 알았으니, 마지막으로 **여러 개를 한 번에** 받아 현재 상태를 한 표로 모아볼게요. `pandas-datareader` 는 시리즈 코드를 리스트로 넘기면 한 번에 받아줘요.

```python
import pandas_datareader.data as web

codes = {
    "CPIAUCSL":     "CPI지수",
    "UNRATE":       "실업률",
    "FEDFUNDS":     "기준금리",
    "DGS10":        "국채10년",
    "T10Y2Y":       "장단기차",
    "VIXCLS":       "VIX",
    "BAMLH0A0HYM2": "신용스프레드",
}

df = web.DataReader(list(codes), "fred", "2024-01-01").rename(columns=codes)

# 지표마다 발표 주기가 달라서 빈칸이 생겨요 → 마지막 유효값으로 채우기
snapshot = df.ffill().iloc[-1]
print(snapshot.round(2))
```

```text
CPI지수      321.76
실업률         4.10
기준금리        4.08
국채10년       4.31
장단기차        0.34
VIX         16.55
신용스프레드      3.15
Name: 2026-06-04 00:00:00, dtype: float64
```

여기서 한 가지 함정이 있어요. **지표마다 발표 주기가 달라서**(CPI 는 월간, VIX 는 일간) 같은 표에 넣으면 빈칸이 잔뜩 생겨요. 그래서 `ffill()`(직전 유효값으로 채우기)로 메운 뒤 마지막 행을 꺼내는 거예요. 이렇게 하면 "오늘 기준 각 지표의 최신값" 한 줄이 나와요.

추세를 그래프로 겹쳐 보고 싶으면, 단위가 제각각이라(실업률 4% vs CPI 321) 그대로 그리면 안 돼요. 시작점을 100으로 맞추는 정규화를 한 번 거치면 같은 화면에서 비교가 됩니다.

```python
# 각 시리즈를 시작값=100 으로 정규화 → 한 그래프에서 추세 비교
normalized = df.ffill().dropna()
normalized = normalized / normalized.iloc[0] * 100

normalized.plot(figsize=(11, 5), title="거시 지표 추세 (시작=100)")
```

이러면 "금리는 횡보하는데 VIX 만 튀었네" 같은 상대 비교가 한눈에 들어와요.

<br>

<br>



## 9. 발표 시점만 알아도 절반은 따라간다

마지막으로 실전 팁 하나. 지표는 **발표 직전·직후에 시장이 가장 크게 움직여요.** 그래서 "값" 만큼이나 "언제 나오나" 가 중요해요. 대략 이런 리듬이에요.

| 시점 | 지표 |
|------|------|
| 매주 목요일 | 신규 실업수당 청구 |
| 매월 첫째 주 금요일 | 고용보고서(NFP·실업률) |
| 매월 중순 | CPI → 그 며칠 뒤 PPI·소매판매 |
| 매월 말 | PCE(Core PCE) |
| 6주마다 | FOMC 기준금리 결정 |
| 분기 말 다음 달 | GDP |

발표 캘린더는 [Investing.com 경제 캘린더](https://www.investing.com/economic-calendar/) 에서 국가를 미국으로 걸고, 중요도 별표 3개짜리만 보면 위 핵심 지표가 추려져요. "예상치(forecast) 대비 실제(actual)" 가 얼마나 벗어나느냐가 그날 시장 반응의 크기를 만들어요.

<br>

<br>



## 마무리

거시 지표는 처음엔 용어부터 낯설지만, **4묶음(인플레이션·고용·성장·금리) → 1차 출처 → FRED 시리즈 코드** 순서로 한 번 자리를 잡아두면 그 다음부턴 코드 몇 줄로 직접 받아볼 수 있어요. 특히 FRED 하나만 익숙해져도 거의 모든 지표를 같은 패턴으로 끌어올 수 있다는 게 가장 큰 무기예요.

정리하면 이렇게 시작하시는 걸 추천드려요.

- [x] FRED 무료 키 발급하고 `pandas-datareader` 로 CPI 한 번 받아보기
- [x] `pct_change(12)` 로 전년 대비 물가상승률 뽑아보기
- [x] `T10Y2Y` 로 장단기 금리차가 지금 양수인지 음수인지 확인
- [x] `yfinance` 로 VIX·S&P500 받아서 시장 분위기 같이 보기

일단 오늘은 여기까지.....   
다음 글에서는 이렇게 받은 지표들을 실제로 그래프 대시보드로 엮고, 발표 서프라이즈(예상 대비 실제)를 자동으로 계산하는 부분을 정리해볼게요.

---

**다음 글 →:** [(2/2) 거시 지표 대시보드 + 발표 서프라이즈](/finance/거시지표_대시보드_서프라이즈_Python/)
