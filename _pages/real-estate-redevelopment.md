---
layout: single
title: "재개발·재건축 데이터"
permalink: /real-estate/redevelopment/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/real-estate-redevelopment/header.svg
  teaser: /assets/images/real-estate-redevelopment/header.svg
description: "서울·경기 노후 아파트의 대지지분·용도지역을 표로 세우고, 서울 정비사업장 1,102곳의 진행단계를 지도에 올렸습니다. 조합설립·사업시행·관리처분 인가가 실거래가를 실제로 얼마나 움직였는지 435만 건으로 따져봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}?v={{ site.time | date: '%s' }}">
<link rel="stylesheet" href="{{ '/assets/realestate/redev.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="re-app is-redev" data-base="{{ '/assets/realestate' | relative_url }}">

  <div class="re-controls">
    <div class="re-tabs re-mode-tabs" role="tablist" aria-label="보기 선택">
      <button class="re-tab is-on" data-mode="complexes" role="tab" aria-selected="true">노후 단지</button>
      <button class="re-tab" data-mode="projects" role="tab" aria-selected="false">진행 단계</button>
      <button class="re-tab" data-mode="premium" role="tab" aria-selected="false">단계별 프리미엄</button>
    </div>
    <div class="re-tabs re-view-tabs" role="tablist" aria-label="지역 선택">
      <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
      <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <div class="re-panel-head">
        <h2 class="re-panel-title">구를 선택하세요</h2>
      </div>
      <p class="re-panel-meta"></p>
      <div class="re-legend"></div>
    </div>
  </div>

  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-note"></p>

  <div class="re-premium" hidden>
    <h2 class="re-section-title">인가 통과 전후 12개월, 실거래 평당가는 얼마나 움직였나</h2>
    <p class="re-premium-lead">
      각 사업장이 인가를 받은 달을 0으로 놓고, 직전 12개월과 직후 12개월의 중위 평당가를
      비교합니다. 그대로 두면 시장 전체의 상승분이 섞이므로,
      <b>같은 자치구 아파트 전체가 같은 기간에 움직인 만큼을 빼서</b> 초과분만 남겼습니다.
      아래 숫자는 그 초과분입니다.
    </p>
    <div class="re-premium-cards"></div>
    <h3 class="re-section-title">초과 상승이 컸던 사업장</h3>
    <div class="re-table-wrap"><table class="re-table re-premium-table"></table></div>
  </div>

  <p class="re-caveat">
    <b>대지지분</b>은 그 단지가 깔고 앉은 땅을 세대수로 나눈 값입니다. 재건축에서 가장 먼저 보는
    숫자이지만, 이것만으로 사업이 되는지는 정해지지 않습니다. 종합 점수를 만들지 않은 이유이기도
    합니다 — 가중치를 지어내면 근거 없는 확신을 주기 때문에, 지표를 그대로 두고 원하는 열로
    정렬해서 보도록 했습니다.
    <br><br>
    <b>용적률 상한</b>은 서울시 도시계획조례의 기본값입니다. 실제 정비계획에서는 공공기여·
    임대주택 등으로 완화되는 경우가 많아, 이 값을 근거로 분담금이나 사업성을 단정할 수 없습니다.
    <br><br>
    <b>추정 용적률</b>은 실제 연면적이 아닙니다. 국토교통부 건축물대장 API 활용신청이 안 돼 있어
    연면적을 받을 수 없고, 대신 그 단지에서 실제 거래된 전용면적의 중위값에 세대수를 곱하고
    전용률 0.75로 나눠 어림한 값입니다. 이 값을 쓰는 진짜 이유는 검산입니다 — 단지 등록 지번의
    필지가 실제 단지 땅보다 넓으면 대지지분이 부풀어 오르는데, 그때 추정 용적률이 비정상적으로
    낮게 나옵니다. <b class="re-doubt-sample">100% 아래인 칸에 빨간 물음표</b>를 달아둔 이유입니다.
    그런 줄에서는 대지지분을 믿지 마세요. 역산 값이 80~400% 밖으로 벗어난 단지는 대지지분을
    아예 감췄습니다.
    <br><br>
    <b>대지지분·용도지역은 서울만</b> 있습니다. 서울시 도시계획포털의 연속지적도에서 받는데
    경기도는 이 경로로 열리지 않습니다. 경기 단지는 준공연차·세대수·실거래가만 나옵니다.
    <b>정비사업 진행단계도 서울만</b>입니다. 서울시 정비사업 정보몽땅이 상시 갱신되는 유일한
    출처이고, 경기도 자료는 2025년 7월 이후 멈춰 있어 넣지 않았습니다.
    <br><br>
    <b>단계별 프리미엄은 재건축만</b> 집계했습니다. 재개발 구역 안은 빌라·단독주택이라
    아파트 실거래에 잡히지 않기 때문입니다. 표본이 10건 미만인 단계는 값을 감췄고,
    사분위 범위를 함께 적었습니다 — 중위값만 보면 편차가 얼마나 큰지 놓칩니다.
    과거에 그랬다는 기록이지 앞으로도 그렇다는 뜻이 아닙니다.
  </p>

  <p class="re-footnote"></p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/redev-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
