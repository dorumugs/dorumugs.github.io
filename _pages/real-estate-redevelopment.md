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
      <div class="re-picks" hidden></div>
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
      <b>같은 자치구가 같은 기간에 움직인 만큼을 빼서</b> 초과분만 남겼습니다.
      아래 숫자는 그 초과분입니다. 재건축은 아파트 실거래로, 재개발은 연립·다세대
      실거래로 봅니다 — 붙이는 단위가 달라 블록을 나눴습니다.
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
    <b>용적률 상한</b>은 그 단지가 속한 지자체의 도시계획조례에서 직접 읽었습니다
    (법제처 국가법령정보, 서울·경기 32곳). 상한은 광역이 아니라 시·군이 정해서
    제각각입니다 — 같은 제3종일반주거지역이라도 서울 250 · 고양 250 · 군포 280 ·
    성남 300 · 남양주 300%입니다.
    <br><br>
    조례에 <b>정비사업 단서</b>가 붙은 경우 그 값을 썼습니다. 예를 들어 성남시는
    제3종을 280%로 정하면서 "정비사업으로 건설하는 아파트는 300퍼센트"라는 단서를
    두는데, 이 화면이 재건축·재개발을 보는 곳이라 300이 맞는 값입니다. 다만 실제
    정비계획에서는 공공기여·임대주택 등으로 또 달라지므로, 이 값을 근거로 분담금이나
    사업성을 단정할 수는 없습니다.
    <br><br>
    <b>용적률</b>은 대부분 국토교통부 건축물대장의 용적률 산정 연면적을 대지면적으로 나눈
    실측값입니다. 대장에 연면적이 없는 일부 단지만 실거래 전용면적의 중위값에 세대수를 곱하고
    전용률 0.75로 나눠 역산했고, 그런 칸은 <b>값 뒤에 ~ 표시</b>를 달았습니다.
    용적률을 굳이 함께 보여주는 건 검산 때문입니다 — 대지면적이 실제 단지 땅보다 넓게 잡히면
    대지지분이 부풀어 오르는데, 그때 용적률이 비정상적으로 낮게 나옵니다.
    <b>100% 아래인 칸에 빨간 물음표</b>를 달아둔 이유이고, 그런 줄에서는 대지지분을 믿지 마세요.
    80~400% 밖으로 벗어난 단지는 대지지분을 아예 감췄습니다.
    <br><br>
    <b>세대수와 대지면적</b>은 건축물대장을 먼저 봅니다. 단지 식별정보는 한 단지를 지번마다
    쪼개 등록해 세대수가 조각나 있는데, 대장의 총괄표제부는 단지 전체를 한 줄로 주기
    때문입니다. 대장에 대지면적이 없는 단지(약 20%)는 연속지적도로 메웠습니다 — 서울은
    서울시 도시계획포털, 경기는 브이월드입니다. 두 지적도를 은마아파트로 교차검증했을 때
    면적 차이가 0.06%였습니다.
    <br><br>
    <b>용도지역</b>은 브이월드에서 받습니다. 건축물대장에도 지역지구 항목이 있지만
    '일반주거지역'까지만 답하고 <b>몇 종인지를 주지 않아</b> 쓸 수 없었습니다 — 용적률
    상한이 1종 150 · 2종 200 · 3종 250%로 갈리는데 그걸 못 정하면 의미가 없습니다.
    <br><br>
    <b>정비사업 진행단계는 서울만</b>입니다. 서울시 정비사업 정보몽땅이 상시 갱신되는 유일한
    출처이고, 경기도 자료는 2025년 7월 이후 멈춰 있어 넣지 않았습니다.
    <br><br>
    <b>단계별 프리미엄은 재건축과 재개발을 따로</b> 집계했습니다. 붙이는 방식이 다르기
    때문입니다. 재건축은 대표지번으로 그 아파트 단지를 정확히 특정할 수 있지만, 재개발
    구역 안은 다세대·연립이고 구역 경계를 알 수 없어 <b>사업장이 속한 법정동 전체</b>를
    구역의 대리 지표로 썼습니다. 법정동은 구역보다 넓어 구역 밖 거래가 섞이고, 한 동에
    사업장이 여럿이면 서로 영향을 줍니다. 모두 효과를 <b>희석하는</b> 방향이라, 재개발
    수치가 0에 가깝다고 해서 "효과가 없다"고 읽으시면 안 됩니다.
    <br><br>
    표본이 10건 미만인 단계는 값을 감췄고, 사분위 범위를 함께 적었습니다 — 중위값만 보면
    편차가 얼마나 큰지 놓칩니다. 과거에 그랬다는 기록이지 앞으로도 그렇다는 뜻이 아닙니다.
    단독·다가구 실거래는 쓰지 않았습니다. 그 API 는 지번을 <code>3**</code> 처럼 가려서
    내보내 정비구역과 맞출 수 없습니다.
  </p>

  <p class="re-footnote"></p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/redev-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
