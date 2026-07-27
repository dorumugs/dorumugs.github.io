---
layout: single
title: "서울·경기 학군 지도"
permalink: /real-estate/schools/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/real-estate-schools/header.svg
  teaser: /assets/images/real-estate-schools/header.svg
description: "서울·경기 사립초 위치를 지도에 올리고, 학교를 누르면 그 학교가 속한 법정동의 아파트 실거래 시세를 보여줍니다. 국토교통부 실거래가와 학교 위치 공공데이터를 겹쳐 봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/realestate/schools.css' | relative_url }}">

<div class="re-app is-schools" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs" role="tablist" aria-label="지역 선택">
      <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
      <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
      <button class="re-tab" data-view="all" role="tab" aria-selected="false">전체</button>
    </div>
    <div class="re-legend-dots">
      <span><i class="is-초"></i>사립초</span>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <h2 class="re-panel-title">학교를 선택하세요</h2>
      <p class="re-school-meta"></p>
      <h3 class="re-section-title re-rank-heading" hidden>같은 법정동 아파트</h3>
    </div>
  </div>

  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-caveat">
    같은 법정동 기준입니다. 실제 배정 학교는 통학구역에 따라 다릅니다.
    사립초는 배정이 아니라 지원으로 가는 학교라 '근처'의 의미가 또 다릅니다.
  </p>
  <p class="re-footnote"></p>
</div>

<script type="module" src="{{ '/assets/realestate/schools-app.js' | relative_url }}"></script>
