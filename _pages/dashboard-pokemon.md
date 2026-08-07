---
layout: single
title: "포켓몬 카드 가격지수"
permalink: /dashboard/pokemon/
classes: wide
author_profile: false
toc: false
description: "포켓몬 카드 시장을 시대 4 × 가격대 3 = 12칸으로 나눠 층화추출한 300장의 가격지수입니다. 구성 종목·가중치·산식을 전부 공개합니다. TCGplayer 시세를 매일 자동 갱신합니다."
---

<link rel="stylesheet" href="{{ '/assets/pokemon/pokemon.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="pk-app" data-base="{{ '/assets/pokemon' | relative_url }}">

  <p class="pk-lede">
    포켓몬 카드 시장을 <strong>시대 4 × 가격대 3 = 12칸</strong>으로 나누고, 각 칸에서
    25장씩 뽑은 <strong>300장</strong>의 가격을 매일 추적합니다. 칸은 균등가중입니다.
    구성 종목과 산식은 아래에 전부 적어 두었습니다.
  </p>

  <div class="pk-stats" id="pk-stats"></div>

  <div class="pk-tabs" role="tablist" aria-label="지수 선택">
    <button class="pk-tab is-on" data-series="index" role="tab" aria-selected="true">전체</button>
    <button class="pk-tab" data-series="era" role="tab" aria-selected="false">시대별</button>
    <button class="pk-tab" data-series="band" role="tab" aria-selected="false">가격대별</button>
  </div>

  <div class="pk-chart-wrap">
    <svg class="pk-chart" id="pk-chart" viewBox="0 0 720 340" preserveAspectRatio="xMidYMid meet" role="img" aria-label="가격지수 추이"></svg>
    <div class="pk-legend" id="pk-legend"></div>
  </div>

  <p class="pk-note" id="pk-note"></p>

  <h2>구성 종목</h2>
  <p class="pk-sub">지수에 들어가는 300장 전부입니다. 숨기는 종목이 없습니다.</p>

  <div class="pk-filters">
    <select id="pk-era" aria-label="시대 거르기"><option value="">시대 전체</option></select>
    <select id="pk-band" aria-label="가격대 거르기"><option value="">가격대 전체</option></select>
  </div>

  <div class="pk-table-wrap">
    <table class="pk-table" id="pk-table">
      <thead><tr>
        <th>카드</th>
        <th class="is-num">현재가</th><th class="is-num">변화</th><th class="is-num">기준가</th>
        <th>시대</th><th>가격대</th><th>세트</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p class="pk-more"><button id="pk-more" type="button">더 보기</button></p>

  <h2>산식</h2>

  <p class="pk-formula" id="pk-formula"></p>

  <p>
    수식이 어려우면 이렇게 읽으면 됩니다. 카드마다 <strong>"기준일에 견줘 지금 몇 배냐"</strong>를
    구합니다. 기준일에 100달러였던 게 지금 120달러면 1.2배입니다. 이 배수를 칸 안에서
    평균 내면 그 칸의 성적이 나옵니다. 칸 12개 성적을 다시 평균 내고 100을 곱하면 지수입니다.
    처음이 100이고 120이 되면 시장이 20% 올랐다는 뜻입니다.
  </p>

  <p>
    가격대는 <strong>시대 안에서</strong> 3등분합니다. 시대를 가로질러 절대 금액으로 자르면
    빈티지가 전부 고가, 최신이 전부 저가로 몰려 칸이 무너지기 때문입니다.
  </p>

  <h2>이 지수가 하지 않는 것</h2>

  <ul>
    <li><strong>시장 전체를 대표한다고 주장하지 않습니다.</strong> 이건 공개된 300장의 성적입니다.</li>
    <li><strong>오른 카드로 갈아타지 않습니다.</strong> 표본은 6개월 고정입니다.</li>
    <li><strong>사라진 카드를 영원히 들고 있지 않습니다.</strong> 가격이 7일 넘게 결측되면 그 카드를 빼고, 뺀 사실을 위에 표시합니다.</li>
    <li><strong>왕복 비용은 반영돼 있지 않습니다.</strong> 감정·수수료·세금은 별도입니다. <a href="{{ '/finance/포켓몬카드_투자_총비용_감정_수수료_세금/' | relative_url }}">2편</a>에서 다뤘습니다.</li>
    <li><strong>프로모·트레이너킷은 들어 있지 않습니다.</strong> 유통 경로가 불규칙하고 시세 자체가 잡히지 않습니다.</li>
  </ul>

  <p class="pk-source">
    출처: <a href="https://tcgdex.dev/" rel="noopener">TCGdex</a> (TCGplayer USD · Cardmarket EUR).
    이 지수는 시세 참고용이며 투자 권유가 아닙니다.
  </p>

</div>

<script src="{{ '/assets/pokemon/app.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
