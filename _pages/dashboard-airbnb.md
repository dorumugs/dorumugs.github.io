---
layout: single
title: "전국 Airbnb 밀집 지도"
permalink: /dashboard/airbnb/
classes: wide
author_profile: false
toc: false
description: "에어비앤비 지도검색을 전국 격자로 훑어 모은 숙소 위치입니다. 256개 시군구를 면적당 밀도로 칠하고, 구를 누르면 그 지역 숙소 점을 지도에 찍습니다. 어디에 숙소가 몰려 있는지 한 장으로 봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}?v={{ site.time | date: '%s' }}">
<link rel="stylesheet" href="{{ '/assets/realestate/airbnb.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="re-app is-airbnb" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs re-metric-tabs" role="tablist" aria-label="색칠 기준">
      <button class="re-tab is-on" data-metric="density" role="tab" aria-selected="true">면적당 밀도</button>
      <button class="re-tab" data-metric="count" role="tab" aria-selected="false">숙소 수</button>
    </div>
    <div class="re-legend"></div>
    <div class="re-filters">
      <label class="re-field">
        <span class="re-field-label">지역</span>
        <select class="re-region"></select>
      </label>
      <button type="button" class="re-toggle is-on" data-layer="on" aria-pressed="true">점 표시</button>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map_kr.svg %}
      <canvas class="re-points" aria-hidden="true"></canvas>
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <div class="re-panel-head">
        <h2 class="re-panel-title">지역을 선택하세요</h2>
        <button type="button" class="re-back-btn" hidden>← 전체로</button>
      </div>
      <div class="re-kpis"></div>
      <p class="re-panel-note"></p>
      <h3 class="re-section-title">시군구 랭킹</h3>
      <div class="re-table-wrap"><table class="re-table"></table></div>
    </div>
  </div>

  <p class="re-footnote"></p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/airbnb-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
