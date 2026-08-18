---
layout: single
title: "ETF 테마 모멘텀"
permalink: /dashboard/etf-theme/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/etf-theme/header.svg
  teaser: /assets/images/etf-theme/header.svg
description: "국내 테마 265개와 업종 79개가 최근 20 거래일 동안 실제로 오르고 있는지를 구성종목 중위 수익률과 상승 종목 비율로 판정하고, 최근 30 거래일 중 이 판이 통째로 움직인 날을 색띠로 보여줍니다. 그 흐름을 살 수 있는 ETF 1,160개로 연결하고, 판정 조건을 전부 공개하며 매일 자동 갱신합니다."
---

<link rel="stylesheet" href="{{ '/assets/etf/etf.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="ef-app" data-base="{{ '/assets/etf' | relative_url }}">

  <div class="ef-market" id="ef-market" aria-live="polite">불러오는 중…</div>

  <div class="ef-tabs" role="tablist" aria-label="보기 고르기">
    <button type="button" id="ef-tab-etf" class="is-on" role="tab" aria-selected="true" aria-controls="ef-view-etf">ETF</button>
    <button type="button" id="ef-tab-theme" role="tab" aria-selected="false" aria-controls="ef-view-theme">테마</button>
    <button type="button" id="ef-tab-upjong" role="tab" aria-selected="false" aria-controls="ef-view-upjong">업종</button>
    <button type="button" id="ef-tab-us" role="tab" aria-selected="false" aria-controls="ef-view-us">미국 ETF</button>
    <button type="button" id="ef-tab-us3" role="tab" aria-selected="false" aria-controls="ef-view-us3">미국 3배</button>
  </div>

<section id="ef-view-etf" role="tabpanel" aria-labelledby="ef-tab-etf" markdown="0">

  <div class="ef-pick" id="ef-pick"></div>

  <div class="ef-filters">
    <input type="search" id="ef-q" placeholder="ETF 이름으로 검색 — 반도체, KODEX" autocomplete="off" aria-label="ETF 검색">
    <select id="ef-grade" aria-label="등급 거르기"><option value="">등급 전체</option></select>
    <select id="ef-tabcode" aria-label="분류 거르기"><option value="">분류 전체</option></select>
    <select id="ef-lev" aria-label="배수 거르기">
      <option value="">배수 전체</option>
      <option value="1">1배만</option>
      <option value="lev">레버리지만</option>
      <option value="inv">인버스만</option>
    </select>
    <select id="ef-liq" aria-label="거래대금 하한">
      <option value="0">거래대금 전체</option>
      <option value="500000000" selected>5억 이상</option>
      <option value="5000000000">50억 이상</option>
      <option value="50000000000">500억 이상</option>
    </select>
    <select id="ef-sort" aria-label="정렬">
      <option value="overhead">위에 물린 물량 적은 순</option>
      <option value="grade">등급순</option>
      <option value="riskAdj">위험 대비 2주 수익순</option>
      <option value="stopProb">손절 걸릴 확률 낮은 순</option>
      <option value="rSwing" data-swing-label>수익률순</option>
      <option value="r20">20일 수익률순</option>
      <option value="turnover">거래대금순</option>
    </select>
    <label class="ef-check"><input type="checkbox" id="ef-primary" checked> 테마별 대표 하나만</label>
    <label class="ef-check ef-risk-box">한 번에 잃어도 되는 금액
      <input type="number" id="ef-risk" min="10000" step="10000" value="300000" aria-label="한 번에 잃어도 되는 금액(원)"> 원
    </label>
  </div>

  <h3 class="ef-tbl-title">담은 것 — 정말 분산이 되나</h3>
  <div class="ef-basket" id="ef-basket"></div>

  <p class="ef-count" id="ef-count"></p>
  <div class="ef-grid" id="ef-list"></div>
  <p class="ef-more"><button type="button" id="ef-more" hidden>더 보기</button></p>

</section>

<section id="ef-view-theme" role="tabpanel" aria-labelledby="ef-tab-theme" hidden markdown="0">

  <div class="ef-filters">
    <input type="search" id="ef-th-q" placeholder="테마 이름으로 검색 — 2차전지, 원자력" autocomplete="off" aria-label="테마 검색">
    <select id="ef-th-grade" aria-label="등급 거르기"><option value="">등급 전체</option></select>
    <select id="ef-th-sort" aria-label="정렬">
      <option value="strip">최근 30일 빨강 많은 순</option>
      <option value="overhead">위에 물린 물량 적은 순</option>
      <option value="rSwing" data-swing-label>수익률순</option>
      <option value="r20">20일 수익률순</option>
      <option value="breadth">상승 종목 비율순</option>
      <option value="grade">등급순</option>
    </select>
    <label class="ef-check"><input type="checkbox" id="ef-th-buyable"> ETF 로 살 수 있는 것만</label>
    <label class="ef-check"><input type="checkbox" id="ef-th-hot"> <span data-hot-label>최근 3일 연속 상승만</span></label>
  </div>

  <div class="ef-striplegend" id="ef-th-legend"></div>

  <p class="ef-note" style="margin-top:0">줄을 누르면 <strong>그 테마를 살 수 있는 ETF</strong>를 시가총액 상위 5개·거래대금 상위 5개로 보여줍니다.</p>

  <p class="ef-count" id="ef-th-count"></p>
  <div id="ef-th-table"></div>
  <p class="ef-more"><button type="button" id="ef-th-more" hidden>테마 더 보기</button></p>

</section>

<section id="ef-view-upjong" role="tabpanel" aria-labelledby="ef-tab-upjong" hidden markdown="0">

  <div class="ef-filters">
    <input type="search" id="ef-up-q" placeholder="업종 이름으로 검색 — 반도체, 은행" autocomplete="off" aria-label="업종 검색">
    <select id="ef-up-grade" aria-label="등급 거르기"><option value="">등급 전체</option></select>
    <select id="ef-up-sort" aria-label="정렬">
      <option value="strip">최근 30일 빨강 많은 순</option>
      <option value="overhead">위에 물린 물량 적은 순</option>
      <option value="rSwing" data-swing-label>수익률순</option>
      <option value="r20">20일 수익률순</option>
      <option value="breadth">상승 종목 비율순</option>
      <option value="grade">등급순</option>
    </select>
    <label class="ef-check"><input type="checkbox" id="ef-up-buyable"> ETF 로 살 수 있는 것만</label>
    <label class="ef-check"><input type="checkbox" id="ef-up-hot"> <span data-hot-label>최근 3일 연속 상승만</span></label>
  </div>

  <div class="ef-striplegend" id="ef-up-legend"></div>

  <p class="ef-note" style="margin-top:0">업종은 거래소 분류라 테마보다 넓고 겹치지 않습니다. 줄을 누르면 <strong>그 업종을 살 수 있는 ETF</strong>가 나옵니다.</p>

  <p class="ef-count" id="ef-up-count"></p>
  <div id="ef-up-table"></div>
  <p class="ef-more"><button type="button" id="ef-up-more" hidden>업종 더 보기</button></p>

</section>

<section id="ef-view-us" role="tabpanel" aria-labelledby="ef-tab-us" hidden markdown="0">

  <div class="ef-market" id="ef-us-market" aria-live="polite"></div>

  <div class="ef-filters">
    <input type="search" id="ef-us-q" placeholder="티커나 이름으로 검색 — TQQQ, SOXL, semiconductor" autocomplete="off" aria-label="미국 ETF 검색">
    <select id="ef-us-grade" aria-label="등급 거르기"><option value="">등급 전체</option></select>
    <select id="ef-us-lev" aria-label="배수 거르기">
      <option value="">배수 전체</option>
      <option value="1">1배만</option>
      <option value="lev">레버리지만</option>
      <option value="lev3">3배만</option>
      <option value="inv">인버스만</option>
    </select>
    <select id="ef-us-liq" aria-label="거래대금 하한">
      <option value="0">거래대금 전체</option>
      <option value="500000000" selected>5억 이상</option>
      <option value="10000000000">100억 이상</option>
      <option value="100000000000">1000억 이상</option>
    </select>
    <select id="ef-us-sort" aria-label="정렬">
      <option value="grade">등급순</option>
      <option value="riskAdj">위험 대비 수익순</option>
      <option value="overhead">위에 물린 물량 적은 순</option>
      <option value="rSwing">달러 수익률순</option>
      <option value="rSwingKrw">원화 수익률순</option>
      <option value="turnoverKrw">거래대금순</option>
    </select>
  </div>

  <p class="ef-count" id="ef-us-count"></p>
  <div class="ef-grid" id="ef-us-list"></div>
  <p class="ef-more"><button type="button" id="ef-us-more" hidden>더 보기</button></p>

</section>

<section id="ef-view-us3" role="tabpanel" aria-labelledby="ef-tab-us3" hidden markdown="0">

  <div class="ef-filters">
    <input type="search" id="ef-u3-q" placeholder="티커나 이름으로 검색 — SOXL, 반도체" autocomplete="off" aria-label="3배 ETF 검색">
    <select id="ef-u3-grade" aria-label="등급 거르기"><option value="">등급 전체</option></select>
    <select id="ef-u3-sort" aria-label="정렬">
      <option value="strip">최근 30일 빨강 많은 순</option>
      <option value="rSwing">달러 수익률순</option>
      <option value="rSwingKrw">원화 수익률순</option>
      <option value="r20">20일 수익률순</option>
      <option value="overhead">위에 물린 물량 적은 순</option>
      <option value="turnoverKrw">거래대금순</option>
      <option value="grade">등급순</option>
    </select>
    <label class="ef-check"><input type="checkbox" id="ef-u3-hot"> <span data-hot-label>최근 3일 연속 상승만</span></label>
  </div>

  <div class="ef-striplegend" id="ef-u3-legend"></div>

  <p class="ef-count" id="ef-u3-count"></p>
  <div id="ef-u3-table"></div>

</section>

  <div class="ef-panel" id="ef-panel" hidden role="dialog" aria-modal="true" aria-labelledby="ef-panel-title">
    <div class="ef-panel-inner">
      <button type="button" class="ef-panel-close" id="ef-panel-close" aria-label="닫기">✕</button>
      <h2 id="ef-panel-title"></h2>
      <div id="ef-panel-body"></div>
    </div>
  </div>

</div>

## 미국 3배 탭 — 여기만 폭을 잽니다

미국 ETF 는 원래 **상승 비율(폭)을 확인할 수 없었습니다.** 국내 테마와 연결이 없어서
"구성종목 중 몇 %가 올랐나"를 셀 수가 없었거든요. 3배 탭에서는 그걸 직접 잽니다 —
발행사가 공시한 **보유 종목의 일봉을 따로 받아서** 국내 테마와 똑같이 계산합니다.
같은 잣대를 써야 두 탭의 빨강이 같은 뜻이 됩니다.

**불(Bull) 3배 15개만 있습니다.** 인버스 3배(SQQQ·SOXS·SPXU 등 9개)는 스왑만 들고
있어 구성종목이 0개입니다. 설령 짝이 되는 불 ETF 의 종목을 빌려 온다 해도, 인버스에서
"구성종목 70% 상승"은 그 ETF 가 **내린다**는 뜻이라 색이 거꾸로 읽힙니다. 이 화면
전체가 "오르는 걸 산다"는 전제 위에 있으니 아예 뺐습니다. 기존 **미국 ETF** 탭에서는
그대로 보입니다.

**셋은 폭을 못 냅니다.** `TMF`·`TNA`·`YINN` 은 보유 종목이 1개뿐이라(사실상 스왑)
**확인 불가**로 표시합니다. 빈칸으로 두면 "오른 날이 없다"로 읽히기 때문에 글자로
적습니다.

**표본 크기가 제각각입니다.** `SPXL`·`UPRO` 는 500종목, `SOXL` 은 30종목, `UDOW` 는
30종목입니다. 같은 70%라도 500개 중 350개와 30개 중 21개는 신뢰도가 다릅니다. 표에
**보유** 열을 같이 둔 이유입니다.

**3배는 임계값이 세 배입니다.** 기초자산이 8.3%만 올라도 25%가 되므로, 1배와 같은
잣대를 대면 늘 "과열"로 찍힙니다. 등급 판정에서 배수만큼 늘려 봅니다. 대신 손절폭도
그만큼 벌어지니 **수량을 그만큼 줄여야** 같은 위험이 됩니다.

## 미국 ETF는 다르게 읽어야 합니다

**환율이 섞여 있습니다.** 원화로 사는 사람의 수익률은 `(1+달러수익) × (1+환율변동) − 1`
입니다. 지금 기준 최근 8주 원달러가 **−6.55%** 라, 달러로 +30.6% 오른 ETF도 원화로는
**+22.0%** 입니다. 카드에 **달러 수익률과 원화 수익률을 나란히** 놓은 이유입니다.

**세금 체계가 완전히 다릅니다.** 국내 상장 ETF는 매매차익이 배당소득으로 잡히지만, 미국
상장 ETF는 **양도소득세 22%**(연 250만원 기본공제, 분리과세)입니다. 8주마다 회전하는
매매라면 이 차이가 수익률을 크게 갉습니다. 이 화면의 어떤 숫자에도 세금은 반영돼 있지
않습니다.

**괴리율을 알 수 없습니다.** 네이버가 해외 ETF의 NAV를 주지 않습니다. 국내 화면에 있던
괴리 경고가 여기엔 없으니, 유동성 낮은 것은 스프레드를 직접 확인하세요.

**대세 확인이 안 됩니다.** 국내 테마와 연결이 없어 상승 비율(폭)을 계산할 수 없습니다.
근거가 국내보다 한 겹 얇습니다.

**기준일이 하루 이릅니다.** 미국 종가는 한국 시간 다음 날 새벽에 확정됩니다. 저녁에
갱신하면 마지막 줄은 '어제 미국 장'입니다.

**3배 레버리지가 있습니다.** 국내는 최대 2배지만 미국은 3배(TQQQ·SOXL·UPRO·CURE 등
24개)까지 있습니다. 임계값은 배수만큼 늘려서 판정하므로 3배 ETF는 과열선이 세 배입니다.
대신 손절폭도 −20%대로 벌어지니 **수량을 그만큼 줄여야** 같은 위험이 됩니다.

**벤치마크는 S&P500입니다.** 미국 ETF를 코스피와 견주는 건 말이 안 되니까요.
거래대금 하한만은 환율로 환산해 국내와 같은 원화 기준(5억 원)을 씁니다 — 잣대를 하나만
두려고요.

## 주 1회, 같은 시각에만 여세요

8주짜리 신호를 매일 들여다보면 과매매가 됩니다. **주 1회 같은 요일 같은 시각**에만 열고,
그때 진입할 것만 정하세요.

카드를 누르면 아래에 **"진입 기록 복사"** 버튼이 있습니다. 날짜·등급·진입가·손절·수량·
물린 물량·근거·시장 국면이 한 덩어리로 클립보드에 담깁니다. 청산일과 손익을 적을 빈칸도
같이 들어갑니다. 개인 투자자 성과를 가장 크게 바꾸는 건 지표를 하나 더 만드는 것이 아니라
**진입 전에 근거와 손절을 적어 두는 것**입니다.

## 이 등급을 믿어도 되나 — 검증 결과

규칙을 만들었으면 채점을 받아야 합니다. 3.5년치 ETF 일봉을 놓고, 과거 각 시점에서
**그날까지의 데이터만으로** 지금과 똑같은 등급을 매긴 다음 그 뒤 10 거래일에 실제로
어떻게 됐는지 셌습니다. 화면과 같은 코드를 씁니다.

**성적이 나쁘다고 임계값을 되맞추지 않습니다.** 고치는 순간 검증이 아니라 곡선 맞추기가
됩니다. 규칙을 손볼 때는 **먼저 논리로 가설을 세우고 그다음에 채점**받는 순서를 지킵니다.
아래에는 규칙이 잘한 칸과 못한 칸이 그대로 있습니다.

<div class="ef-backtest" id="ef-backtest">불러오는 중…</div>

### 첫 채점에서 드러난 결함과, 고친 방법

첫 채점에서 손절이 2주 안에 **37~60%** 걸렸고, 그 바람에 원수익이 플러스인 등급도 손절을
지키면 마이너스가 됐습니다.

원인은 통계가 아니라 **산수였습니다.** 보유 H일 동안의 노이즈 크기는 하루 변동성 × √H입니다.
2주면 √10 ≈ 3.16 일간단위인데 손절은 2×ATR ≈ 2 일간단위였습니다. **2 < 3.16이니 손절선이
노이즈 밴드 안에** 있었던 겁니다. 논지가 깨져서가 아니라 평범한 출렁임에 걸립니다.

이론값도 맞아떨어졌습니다. 손절선을 건드릴 확률은 `2Φ(−손절폭/σ)`인데 당시 비율이
4%/6% = 0.67σ → **예측 50%**, 관측 37~60%였습니다.

그래서 **손절을 보유기간 1σ 바깥으로** 옮겼습니다. 제약을 만족하는 최솟값을 택한 것이지
성적이 제일 좋게 나오는 값을 찾은 게 아닙니다. 결과는 위 표에 있습니다 — 손절 적중률이
절반 아래로 떨어졌고 손절 적용 수익이 플러스로 돌아섰습니다.

**바닥다지기 등급은 뺐습니다.** 성적이 나빠서가 아니라 **애초에 가설이 없던 규칙**이라
뺐습니다. 추세진행·눌림매수는 모멘텀 지속이라는 근거가 있지만, "저점에서 반등하는 걸 산다"는
이 화면 전체의 전제(오르는 걸 산다)와 정면으로 모순됩니다. 근거 없이 만들어 놓고 '관찰 대상'
이라는 이름으로 기회처럼 보이게 한 것이 문제였습니다. 해당하던 것들은 약세로 들어갑니다.

### 아직 남아 있는 것 — 고치지 않았습니다

**추세진행의 우위는 여전히 얇습니다.** 시장 대비 2주 +0.24%p, 거래 비용을 빼면 +0.14%p입니다.
있긴 하지만 크지 않습니다.

**눌림매수는 시장을 못 이깁니다.** 절대 수익은 +1.14%로 가장 높지만 시장 대비로는 −0.07%입니다.
지수를 그냥 사는 것과 견주면 나을 게 없다는 뜻입니다.

**과열주의는 숫자를 믿으면 안 됩니다.** 표본 86건이라 승률 신뢰구간이 49.9~70.1%로,
기준선(58.8%)과 겹칩니다. 표에 "구별 안 됨"을 붙여 뒀습니다.

## 먼저 시장 국면부터 봅니다

맨 위 띠가 **코스피·코스닥이 각각 20일선 위인지**를 알려줍니다. 예측이 아니라 위치 확인입니다.
보유 기간이 짧을수록 개별 테마의 힘보다 시장 전체의 방향이 결과를 더 많이 정하기 때문에,
2주 스윙에서는 테마를 고르기 전에 **들어갈 때인지**부터 봐야 합니다. 둘 다 20일선 아래인
"역풍"이면 후보가 좋아 보여도 규모를 줄이거나 쉬는 쪽이 먼저입니다.

## 두 개의 시간축

카드에 **2주(10일)** 와 **20일** 두 숫자가 나란히 있습니다. 20일은 "이 판이 대세인가"를 보는
배경이고, 10 거래일은 실제로 들고 갈 구간입니다. 신호의 기간과 보유 기간이 어긋나면 20일짜리
흐름을 보고 들어가 10일 만에 나오는, 아귀가 안 맞는 매매가 됩니다.

## 주문을 내려면 이 세 개가 필요합니다

| | |
|---|---|
| **손절** | `현재가 − max(2×ATR, 2주 기대 변동폭)`. **보유기간 노이즈의 바깥**에 둡니다 |
| **2주 폭** | 관측된 변동성의 1σ를 10 거래일로 환산했습니다. **방향을 맞히는 값이 아니라** 크기 감각입니다 |
| **손절 걸릴 확률** | `2Φ(−손절폭/σ)`. 방향성 없는 움직임을 가정했을 때 2주 안에 손절선을 건드릴 확률입니다 |

손절이 노이즈 안쪽에 있으면 논지가 깨져서가 아니라 평범한 출렁임에 걸립니다.

손절을 1σ에 맞춰 두었기 때문에 **대부분의 ETF에서 걸릴 확률이 약 32%로 같습니다.** 변동성이
큰 ETF는 손절폭이 −22%까지 벌어지고 작은 ETF는 −3%에 붙지만, *걸릴 확률*은 같게 정규화돼
있다는 뜻입니다. 그래서 카드에는 손절폭만 적고, 확률은 기본값에서 벗어난 경우(갭이 커서
2×ATR이 1σ보다 넓은 ETF, 전체의 22%)에만 "손절 여유"로 표시합니다.

## 보유는 8주입니다 (2주에서 늘렸습니다)

처음엔 2주로 만들었다가 **8주(40 거래일)** 로 늘렸습니다. 가설을 먼저 세우고 쟀습니다 —
*"왕복 비용 0.10%p를 10일마다 내면 얇은 우위가 다 깎인다. 오래 들고 갈수록 비용을 나눠 낸다."*

'추세진행 + 물린물량 20% 미만' 기준입니다.

| 보유 | 건당 초과 | 비용 후 | 연 환산 |
|---|---|---|---|
| 2주 | +0.22% | +0.12% | +3.0% |
| 4주 | +0.52% | +0.42% | +5.2% |
| **8주** | **+1.43%** | **+1.33%** | **+8.3%** |
| 12주 | +2.23% | +2.13% | +8.9% |

**건당 초과가 기간보다 빠르게 늘었습니다** — 4배 기간에 6.5배. 비용 분산만이 아니라 신호
자체가 긴 구간에서 더 잘 듣는다는 뜻이고, 모멘텀 문헌과도 맞습니다.

8주와 12주 중 어느 쪽이 나은지는 **못 가립니다.** 겹치지 않는 평가 시점이 각각 21회·14회
뿐이라 그 차이를 주장할 표본이 없습니다. 방향만 믿고 8주를 골랐습니다.

이건 임계값 튜닝이 아닙니다. **보유 기간은 규칙 안의 파라미터가 아니라 투자자가 정하는
조건**이라 여러 값을 재서 고르는 게 맞습니다. 판정 규칙(20일·5일)은 그대로 뒀습니다.

## 담은 것 — 정말 분산이 되나

카드의 **담기**를 눌러 3~5개를 모으면 계산해 드립니다.

**분산은 개수가 아니라 상관이 정합니다.** 반도체 ETF 세 개는 이름만 셋이지 사실상 한
베팅입니다. 실제로 재보면 `KODEX 미국나스닥100(H)`와 `TIGER 미국나스닥100(H)`의 상관은
**+1.00**이고, `TIGER 미국S&P500(H)`와 `TIGER 원유선물(H)`은 **−0.54**입니다.

포트폴리오 변동폭은 `√(ΣΣ aᵢaⱼρᵢⱼ)`로 구합니다. 상관이 1이면 그냥 합이라 분산 효과가 0,
낮을수록 줄어듭니다. **상관을 모르는 쌍은 1로 봅니다** — 모르면 최악을 가정해야 분산 효과를
실제보다 크게 보여주는 일이 없습니다.

**"다 손절되면"** 이 실질적인 한도입니다. 5개를 담으면 위험금액의 5배가 한꺼번에 날아갈 수
있고, 상관이 높으면 실제로 같이 날아갑니다. 그 금액이 감당 가능한지가 먼저입니다.

## 몇 주를 살 것인가

**손절폭이 클수록 같은 금액을 넣으면 안 됩니다.** 손절 −22%짜리와 −3%짜리에 같은 돈을 넣으면
전자에서 일곱 배를 잃습니다.

그래서 필터 줄에 **"한 번에 잃어도 되는 금액"** 칸을 뒀습니다. 이 값 하나를 넣으면 카드마다
살 수량이 나옵니다.

```
수량 = 잃어도 되는 금액 ÷ (현재가 × 손절폭)
```

이렇게 잡으면 손절폭이 제각각인 종목에 **같은 금액이 아니라 같은 위험**을 걸게 됩니다.
30만원을 걸기로 했으면 어느 종목에서 손절당해도 잃는 돈은 30만원 근처입니다.

수량에는 **유동성 상한**이 걸려 있습니다. 하루 거래대금의 1%를 넘겨 담으면 넣고 빼는 데
값이 밀리므로 거기서 끊고, "유동성 상한"이라고 표시합니다. 손절폭이 아주 좁은 물건에
위험 기반 수량만 쓰면 "10억어치 사라" 같은 답이 나오는데 산수는 맞아도 현실이 아닙니다. 유동성 하한 5억은 모두에게 같은
값이지만, 실제로 못 빠져나오는지는 **내 주문 크기**에 달렸기 때문입니다. 한 주만 사도 한도를
넘으면 "담을 수 없음"으로 표시합니다. 값은 브라우저에 저장돼 다음에도 남습니다.

위 검증 결과의 **실제 손절 적중률**(21~28%)과 이 이론값(32%)을 견줘 보실 수 있습니다 —
추세가 있는 자산이라 실제가 조금 낮게 나옵니다. 둘이 크게 어긋나면 계산이 틀린 겁니다.

ATR은 갭까지 반영한 하루 평균 등락폭입니다. 고가−저가만 보면 전날 종가에서 훌쩍 뛰어 시작하는
갭을 놓쳐 손절선이 터무니없이 좁아집니다.

**정렬 기본값이 "위험 대비 2주 수익"인 이유**가 여기 있습니다. 변동성 100%짜리의 +20%와
30%짜리의 +8% 중 뒤가 더 나은 스윙일 수 있습니다. 수익률만으로 줄세우면 그냥 변동성이 큰 것만
위로 올라옵니다.

## 고점 · 현재가 · 매물대

카드에 **현재가**, **52주 고점 대비**, **위에 물린 물량**을 함께 놓습니다. 카드를 누르면
매물대 그래프가 나옵니다 — 세로축이 가격이고 위가 비싼 쪽, **붉은 칸이 현재가보다 비싼
값에 거래된 물량**, 점선이 현재가입니다.

**"위에 물린 물량"이 왜 중요한가.** 이 값이 80%라면 지난 120 거래일 거래의 80%가 지금보다
비싼 값에 이뤄졌다는 뜻입니다. 그 사람들은 본전 근처에 오면 팝니다. 2주 스윙에서 이건
지표보다 직접적인 장애물입니다 — 아무리 추세가 좋아도 머리 위에 매도 대기 물량이 층층이
쌓여 있으면 그 구간을 통과하는 데 시간이 걸립니다. 반대로 20%면 길이 비어 있습니다.

**매물이 가장 두꺼운 값(POC)** 은 가장 많이 거래된 가격대라 **평균 매입가에 가장 가까운
자리**입니다. 여기를 위로 넘기면 물린 사람이 크게 줄고, 아래로 깨지면 늘어납니다.

**첫 저항 매물대**는 현재가 위에서 매물이 가장 두꺼운 가격입니다. 2주 목표가 여기를 넘어야
한다면 쉽지 않다고 봐야 합니다.

한계가 있습니다. **일봉만 있어 장중 체결 분포를 모릅니다.** 그래서 하루 거래량을 그날
고가~저가에 균등하게 나눠 담은 근사치입니다. 실제로는 종가 근처에 더 몰리지만, 하루 안의
치우침은 120일을 쌓으면 대체로 씻깁니다. 정밀한 매물대가 아니라 **어느 쪽이 무거운지**를
보는 도구로 쓰세요.

## 괴리율

ETF는 담고 있는 자산의 가치(NAV)가 정해져 있는데 시장가는 수급으로 따로 놉니다. 괴리가 **+1%**
인 걸 사면 **사는 순간 1%를 얹어 주는 것**이고, 나중에도 괴리가 붙어 있으리라는 보장이 없습니다.
2주 스윙에서 1%는 작지 않습니다. 0.5%를 넘으면 카드에 경고를 답니다. 거래가 얕은 ETF와 해외
자산 ETF(시차 때문에)에서 특히 벌어집니다.

## 테마별 대표 하나만

반도체 ETF 열 개를 다 사는 사람은 없습니다 — 사실상 같은 베팅이니까요. 기본값은 대표 테마마다
**20일 중앙 거래대금이 가장 큰 하나**만 보여줍니다. 유동성으로 고르는 이유는 2주 안에 나와야
하기 때문입니다. 체크를 풀면 전부 나옵니다.

## 최근 30일 — 언제부터 몰렸나

테마·업종 표 맨 오른쪽에 색띠가 있습니다. **최근 30 거래일 동안 날마다 구성종목 중 몇
%가 전일보다 올랐는지**를 왼쪽(과거)에서 오른쪽(최근)으로 늘어놓은 것입니다.

| 칸 색 | 그날 오른 구성종목 |
|---|---|
| 빨강 | 70% 이상 — 이 판이 통째로 움직인 날 |
| 파랑 | 30% 이하 — 통째로 밀린 날 |
| 회색 | 그 사이 — 방향이 갈린 날 |
| 빈칸 | 그날 값을 만들 종목이 3개가 안 됨 |

**왜 이 띠가 필요한가.** 상승 비율은 한 숫자라 *언제부터* 그랬는지를 못 말합니다. 20일
내내 꾸준히 올라 64%인 테마와, 18일 죽어 있다가 이틀 급등해 64%가 된 테마가 표에서
똑같이 보입니다. 스윙 진입 시점을 정할 때 이 둘은 전혀 다른 물건입니다. 오른쪽 끝에
빨강이 몰려 있으면 지금 붙은 것이고, 왼쪽에만 몰려 있으면 이미 지나간 판입니다.

가운데를 회색으로 비워 둔 이유는, 55% 같은 애매한 날까지 색을 주면 **진짜 몰린 날이
안 보이기** 때문입니다. 칸에 손을 올리면 그날 실제 비율이 나옵니다.

**띠 끝 3칸이 모두 빨강인 줄은 배경을 연하게 칠했습니다.** 최근 3 거래일 내내 구성종목
70% 이상이 오른, *지금 막 붙은* 판입니다. 왼쪽(과거)에 몰린 빨강은 이미 지나간 판이라
끝에서부터 셉니다. 3일로 잡은 이유는 하루는 소음이고, 5일이면 20일 지표가 이미 잡아내기
때문입니다. **연속이라는 것 말고는 아무 뜻이 없습니다** — 3일 올랐으니 4일째도 오른다는
근거는 이 화면 어디에도 없습니다.

**한계가 셋 있습니다.**

**종가만 봅니다.** 장중에 아무리 밀렸어도 종가가 전일보다 높으면 빨강입니다.

**동일가중입니다.** 시가총액 1위가 −5%여도 나머지 아홉이 +0.1%면 90%로 찍힙니다.
*얼마나* 올랐는지가 아니라 **몇 개가** 올랐는지를 재는 지표입니다. 폭과 수익률을
같이 봐야 하는 이유입니다.

**구성종목 수가 제각각입니다.** 5종목 테마의 80%와 60종목 테마의 80%는 신뢰도가
다릅니다. 표에 종목 수를 나란히 둔 이유입니다.

70%·30%도 검증된 값이 아니라 논리로 정한 출발점입니다 — 이 화면의 다른 임계값과
같습니다.

## 나머지 지표

**20일 수익률**은 최근 20 거래일 동안 얼마나 올랐는지입니다. 테마·업종은 구성종목 수익률의
**중위값**을 씁니다. 평균이 아닌 이유는, 구성종목 하나가 세 배가 되면 평균은 테마 전체가
오른 것처럼 보이지만 그건 대세가 아니라 개별 이슈이기 때문입니다.

**상승 비율(폭)** 은 구성종목 중 20일 수익률이 양(+)인 종목의 비율입니다. 이 판이 통째로
움직이는지를 한 숫자로 보여줍니다. 60% 이상을 대세로 봅니다.

**추세 곧기**는 20일 종가를 그래프에 찍고 자로 직선을 그었을 때 점들이 자에 얼마나 붙어
있는지입니다. 1에 가까울수록 매일 조금씩 꾸준히 오른 것이고, 0에 가까울수록 하루 급등하고
나머지는 제자리인 것입니다. "오르냐"가 아니라 "**깔끔하게** 오르냐"를 재는 숫자입니다.

**거래대금**은 20일 중앙값입니다. 평균이 아니라 중앙값을 쓰는 이유는 하루짜리 대량 거래에
속지 않기 위해서입니다. 스윙에서 가장 자주 다치는 건 지표가 틀려서가 아니라 사고 나서
못 빠져나와서입니다.

## 등급이 무슨 뜻인가

| 등급 | 뜻 |
|---|---|
| **눌림매수** | 시장보다 낫고, 구성종목 다수가 오르고, 쉬기 전까지 추세가 곧았는데 지금 고점 대비 3~10% 쉬는 중 |
| **추세진행** | 시장보다 낫고, 구성종목 다수가 오르고, 최근 5일도 꺾이지 않음 |
| **과열주의** | 20일에 25% 이상 올랐는데 최근 5일이 마이너스. 급등 뒤 꺾임 |
| **약세** | 위 어디에도 안 듦 |
| **유동성부족** | 20일 중앙 거래대금 5억 원 미만. 흐름과 무관하게 스윙으로 빠져나오기 어려움 |
| **금리형** | 연율 변동성 8% 미만. CD금리·머니마켓처럼 모멘텀 개념이 성립하지 않음 |
| **판정불가** | 상장·거래정지로 20 거래일이 안 됨 |

레버리지 ETF는 임계값을 **배수만큼 늘려서** 봅니다. 2배 ETF는 기초자산이 12.5%만 올라도
25%가 되므로, 1배와 같은 잣대를 대면 늘 "과열"로 찍히기 때문입니다.

해외·채권·원자재 ETF는 국내 종목을 담지 않아 **상승 비율을 확인할 수 없습니다.** 그런
경우 "대세 확인 불가"로 표시합니다. 근거가 한 겹 얇다는 뜻입니다. 또 해외 ETF는 원화로
표시되므로 수익률에 **환율이 섞여 있습니다** — 기초자산이 그대로여도 달러가 오르면
오릅니다. 이름 끝에 `(H)`가 붙은 것이 환헤지 상품입니다.

<p class="ef-disclaimer">
이 화면은 과거 가격을 계산해 보여줄 뿐이고, 앞으로의 수익을 예측하지 않습니다.
등급은 관찰 구간 분류이지 매수·매도 신호가 아닙니다. <strong>손절선과 2주 기대 변동폭도
예측이 아닙니다</strong> — 관측된 변동성을 환산한 값이라, 실제로는 그보다 크게 움직일 수
있고 갭으로 손절선을 건너뛸 수도 있습니다. 임계값(상승 비율 60%, 변동성 8%, 거래대금 5억,
손절폭 등)은 검증된 최적값이 아니라 논리로 정한 출발점입니다. 백테스트는 이 규칙을
<strong>채점만</strong> 하고 임계값을 되맞추지 않습니다 — 그렇게 하면 검증이 아니라 곡선
맞추기가 되기 때문입니다. 검증 결과에는 상장폐지 ETF 누락 등 걷어내지 못한 편향이 있습니다.
투자 판단과 그 결과는 투자자 본인에게 있습니다.
</p>

<script src="{{ '/assets/etf/app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
