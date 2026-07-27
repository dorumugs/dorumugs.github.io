---
layout: single
title: "서울·경기 아파트 실거래 대시보드"
permalink: /real-estate/trades/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/real-estate-dashboard/header.svg
  teaser: /assets/images/real-estate-dashboard/header.svg
description: "국토교통부 실거래가 435만 건으로 만든 서울·경기 아파트 대시보드입니다. 72개 시군구를 지도에서 눌러 평당가 수준·변화율·전고점 대비·거래 회전율을 비교하고, 구별 단지 랭킹까지 봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}">

<div class="re-app" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs" role="tablist" aria-label="지역 선택">
      <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
      <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
      <button class="re-tab" data-view="all" role="tab" aria-selected="false">전체</button>
    </div>
    <div class="re-filters">
      <label class="re-field">
        <span class="re-field-label">지표</span>
        <select class="re-metric">
          <option value="level">중위 평당가</option>
          <option value="chg3">3개월 변화율</option>
          <option value="chg6">6개월 변화율</option>
          <option value="chg12" selected>12개월 변화율</option>
          <option value="peak">전고점 대비</option>
          <option value="turnover">거래 회전율</option>
        </select>
      </label>
      <label class="re-field">
        <span class="re-field-label">기준월</span>
        <select class="re-month"></select>
      </label>
      <button class="re-toggle is-on" data-filter="300" aria-pressed="true">300세대+</button>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-legend"></div>
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <div class="re-panel-head">
        <h2 class="re-panel-title">지역을 선택하세요</h2>
        <button type="button" class="re-back-btn" hidden>← 목록으로</button>
      </div>
      <div class="re-kpis"></div>
      <h3 class="re-section-title re-chart-heading" hidden>평당가 추이</h3>
      <div class="re-chart"></div>
    </div>
  </div>

  <h3 class="re-section-title re-rank-heading" hidden>단지 랭킹 · 최근 12개월</h3>
  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-footnote"></p>
</div>

<script type="module" src="{{ '/assets/realestate/app.js' | relative_url }}"></script>
