---
layout: single
title: "아파트 착공 × 금리"
permalink: /dashboard/supply/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/dashboard-supply/header.svg
  teaser: /assets/images/dashboard-supply/header.svg
description: "시도 16곳의 아파트 착공을 2011년부터 월별로 쌓고, 각 지역의 평년(2011~2019) 대비 지수로 바꿔 한눈에 비교합니다. 착공은 입주보다 2~3년 앞서므로 지금의 착공이 미래 입주물량의 예고편입니다. 한국은행 기준금리와 예금은행 주택담보대출 금리를 같은 x축 아래 패널에 나란히 놓았습니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}?v={{ site.time | date: '%s' }}">
<link rel="stylesheet" href="{{ '/assets/realestate/supply.css' | relative_url }}?v={{ site.time | date: '%s' }}">

착공은 준공보다 **2~3년 앞섭니다.** 지금 착공이 줄고 있다면 그 지역은 2~3년 뒤
입주가 마릅니다. 절대량으로는 서울과 대구를 같은 축에 놓을 수 없으니,
**각 시도의 평년 대비 지수**로 봅니다.

<div class="re-app is-supply" data-base="{{ '/assets/realestate' | relative_url }}">

  <div class="re-freshness"></div>

  <div class="sp-controls">
    <div class="sp-metric" role="group" aria-label="지표 고르기">
      <button type="button" class="is-on" data-metric="index" aria-pressed="true">평년=100 지수</button>
      <button type="button" data-metric="mavg" aria-pressed="false">12개월 이동합계</button>
      <button type="button" data-metric="units" aria-pressed="false">월별 착공 호수</button>
    </div>
    <dl class="sp-defs"></dl>

    <div class="sp-picks" role="group" aria-label="시도 고르기">불러오는 중…</div>
  </div>

  <div class="sp-chart"></div>
  <div class="sp-legend"></div>
  <p class="sp-note"></p>

  <div class="sp-cards"></div>
  <p class="sp-cards-foot"></p>

  <h2>이 화면이 하지 않는 것</h2>

  <p markdown="1">
  **예측하지 않습니다.** 착공에서 입주 시점을 역산해 주지 않습니다 — 시차는
  사업마다 다릅니다. **상관계수나 회귀도 내지 않습니다.** 두 선을 나란히 놓기만
  합니다. 이중 Y축을 쓰지 않은 것도 같은 이유입니다. 이중축은 두 축의 눈금을
  어떻게 잡느냐로 상관관계를 있어 보이게도 없어 보이게도 만들 수 있어서,
  눈금을 고르는 순간 만든 사람이 결론을 정하는 셈이 됩니다.
  </p>

  <h2>읽을 때 알고 계셔야 할 것</h2>

  <ul class="sp-limits"></ul>

  <p class="sp-note" markdown="1">
  자료는 국토교통 통계누리 **주택유형별 착공실적(월계)** 과 한국은행 **ECOS** 입니다.
  시군구 단위로 내려오는 주택 통계는 미분양 하나뿐이라 착공은 시도까지만 봅니다.
  건축HUB 주택인허가 대장으로 시군구 착공을 직접 집계하는 길도 확인해 봤지만,
  표본에서 착공일이 채워진 건이 1/5, 총세대수는 전부 0이라 **만들어도 믿을 수 없는
  숫자**가 나와 하지 않았습니다.
  </p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/supply-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
