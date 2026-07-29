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
description: "서울·경기 사립초·사립중·국제중·특목고(과학고·외고·국제고) 위치를 지도에 올리고, 학교를 누르면 그 학교가 속한 법정동의 아파트 실거래 시세를 보여줍니다. 국토교통부 실거래가와 학교 위치 공공데이터를 겹쳐 봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}?v={{ site.time | date: '%s' }}">
<link rel="stylesheet" href="{{ '/assets/realestate/schools.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="re-app is-schools" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tab-groups">
      <div class="re-tabs re-view-tabs" role="tablist" aria-label="지역 선택">
        <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
        <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
        <button class="re-tab" data-view="all" role="tab" aria-selected="false">전체</button>
      </div>
      <div class="re-tabs re-lvl-tabs" role="tablist" aria-label="학교급 선택">
        <button class="re-tab is-on" data-lvl="all" role="tab" aria-selected="true">전체</button>
        <button class="re-tab" data-lvl="초" role="tab" aria-selected="false">사립초</button>
        <button class="re-tab" data-lvl="중" role="tab" aria-selected="false">사립중</button>
        <button class="re-tab" data-lvl="국제중" role="tab" aria-selected="false">국제중</button>
        <button class="re-tab" data-lvl="특목고" role="tab" aria-selected="false">특목고</button>
      </div>
    </div>
    <div class="re-legend-dots">
      <span><i class="is-초"></i>사립초</span>
      <span><i class="is-중"></i>사립중</span>
      <span><i class="is-국제중"></i>국제중</span>
      <span><i class="is-특목고"></i>특목고</span>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <div class="re-panel-head">
        <h2 class="re-panel-title">학교를 선택하세요</h2>
        <button type="button" class="re-back-btn" hidden>← 목록으로</button>
      </div>
      <p class="re-school-meta"></p>
      <h3 class="re-section-title re-peer-prog-heading" hidden>진학률이 비슷한 학교 · 특목고·자사고</h3>
      <div class="re-peer-prog" hidden>
        <div class="re-peer-chart" role="img" aria-label="진학률이 비슷한 학교 비교"></div>
        <div class="re-peer-legend" aria-hidden="true"></div>
        <p class="re-peer-note"></p>
      </div>
      <h3 class="re-section-title re-rank-heading" hidden>학교별 법정동 아파트 · 비교군 포함</h3>
    </div>
  </div>

  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-rank-note"></p>
  <p class="re-caveat">
    같은 법정동 기준입니다. 실제 배정 학교는 통학구역에 따라 다릅니다.
    사립초·사립중·국제중은 배정이 아니라 지원으로 가는 학교라 '근처'의 의미가 또 다릅니다.
    특목고(과학고·외고·국제고)는 시·도 단위로 모집해 통학 거리와의 관계가 이들보다 훨씬
    약합니다 — 순위나 배정과는 무관한, 위치 참고용 레이어로 보세요.
    중학교는 사립중·국제중만 다룹니다 — 특목고·자사고 진학 비율 기반 필터를 넣고 싶었지만
    해당 데이터가 공개되어 있지 않아 넣지 못했습니다. 대신 시·도 단위 참고 자료를
    아래에 붙였습니다.
  </p>

  <div class="re-progression">
    <h2 class="re-section-title">특목고·자사고 진학률(시·도 단위)</h2>
    <p class="re-prog-note">
      학교별 특목고·자사고 진학 실적은 어떤 공개 경로로도 구할 수 없어(위 지도 목록의
      학교별 순위나 필터로는 쓰이지 않습니다), 서울·경기 전체 중학교 졸업생을 놓고 낸
      시·도 단위 집계만 보여드립니다. 특목고(과학고·외고/국제고)·자사고 진학자를
      졸업자 수로 나눈 비율입니다.
    </p>
    <div class="re-prog-chart" role="img" aria-label="진학률 추이 불러오는 중"></div>
    <div class="re-prog-legend" aria-hidden="true"></div>
    <p class="re-prog-cohort"></p>
  </div>

  <p class="re-footnote"></p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/schools-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
