---
layout: single
title: "포켓몬 카드 시세"
permalink: /dashboard/pokemon/
classes: wide
author_profile: false
toc: false
description: "포켓몬 카드 시세를 카드 사진과 함께 봅니다. 글로벌은 TCGplayer·Cardmarket 기준 달러·유로 시세를, 국내는 KREAM 기준 원화 시세와 최근 한 달 거래 추이를 보여줍니다. 영어 이름과 한글 이름 둘 다로 검색할 수 있고 매일 자동 갱신합니다."
---

<link rel="stylesheet" href="{{ '/assets/pokemon/pokemon.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="pk-app" data-base="{{ '/assets/pokemon' | relative_url }}">

  <div class="pk-tabs" role="tablist" aria-label="시장 고르기">
    <button type="button" id="pk-tab-usd" class="is-on" role="tab" aria-selected="true" aria-controls="pk-view-usd">글로벌 · 달러</button>
    <button type="button" id="pk-tab-krw" role="tab" aria-selected="false" aria-controls="pk-view-krw">국내 · 원화</button>
    <button type="button" id="pk-tab-cmp" role="tab" aria-selected="false" aria-controls="pk-view-cmp">PSA 10</button>
  </div>

  <p class="pk-warn">
    두 시장은 <strong>파는 물건이 다릅니다.</strong> 글로벌은 <strong>영문판</strong>이고
    등급 없는 raw 가 기본이며 비싼 카드에만 PSA 10 이 붙어 있습니다. 국내는
    <strong>대부분 일본판</strong>이고 전부 <strong>PSA 10</strong>입니다.
    같은 포켓몬이라도 세트·판본이 다르면 다른 카드이니, 값을 빼기 전에
    <strong>[PSA 10]</strong> 탭에서 같은 등급끼리만 견주세요.
  </p>

<section id="pk-view-usd" role="tabpanel" aria-labelledby="pk-tab-usd" markdown="0">

  <div class="pk-search">
    <input type="search" id="pk-q" placeholder="카드 이름으로 검색 — 리자몽, Charizard" autocomplete="off" aria-label="카드 검색">
  </div>

  <div class="pk-filters">
    <select id="pk-set" aria-label="세트 거르기"><option value="">세트 전체</option></select>
    <select id="pk-era" aria-label="시대 거르기"><option value="">시대 전체</option></select>
    <select id="pk-sort" aria-label="정렬">
      <option value="price-desc">비싼 순</option>
      <option value="price-asc">싼 순</option>
      <option value="name">이름순</option>
      <option value="obs-desc">관측 최고가 순</option>
      <option value="psa-desc">PSA 10 비싼 순</option>
    </select>
  </div>

  <p class="pk-count" id="pk-count">불러오는 중…</p>

  <div class="pk-grid" id="pk-grid"></div>

  <p class="pk-more"><button id="pk-more" type="button">더 보기</button></p>

  <h2>가격이 무슨 뜻인가</h2>

  <div class="pk-table-wrap">
    <table class="pk-table">
      <thead><tr><th>표시</th><th>정체</th><th>주의할 점</th></tr></thead>
      <tbody>
        <tr>
          <td><strong>현재가</strong></td>
          <td>TCGplayer market price (USD)</td>
          <td>최근 실제 거래를 반영한 값입니다. 기준으로 삼기에 가장 낫습니다.</td>
        </tr>
        <tr>
          <td>최고 호가</td>
          <td>지금 시장에 올라온 가장 비싼 <strong>매물</strong></td>
          <td><strong>팔린 값이 아닙니다.</strong> 터무니없는 호가가 섞입니다.</td>
        </tr>
        <tr>
          <td>관측 최고가</td>
          <td>이 사이트가 매일 재면서 본 최고 현재가</td>
          <td>2026-08-07 이후만 봅니다. 그 전 기록은 없습니다.</td>
        </tr>
        <tr>
          <td>EUR</td>
          <td>Cardmarket 평균 낙찰가</td>
          <td>유럽 시장이라 미국과 다르게 움직입니다.</td>
        </tr>
        <tr>
          <td><strong>PSA 10 · PSA 9</strong></td>
          <td>eBay 에서 <strong>실제로 팔린</strong> 감정 카드 가격</td>
          <td>옆의 <strong>건수와 마지막 거래일</strong>을 꼭 같이 보세요. 1건이면 그건 시세가 아니라 사례 하나입니다.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h2>알아두실 것</h2>

  <ul>
    <li><strong>역대 최고 낙찰가는 없습니다.</strong> 무료로 열린 어떤 API 도 주지 않습니다. 위 세 가지가 구할 수 있는 전부입니다.</li>
    <li><strong>한글 이름은 포켓몬 이름입니다.</strong> 카드 정식 한글명이 공개된 데이터가 없어, 도감번호로 포켓몬 종 이름을 붙였습니다. <code>Charizard ex</code> 는 <code>리자몽 ex</code> 로 나옵니다.</li>
    <li><strong>트레이너·에너지 카드는 한글 이름이 없습니다.</strong> 도감번호가 없기 때문입니다.</li>
    <li><strong>가격이 안 잡히는 카드는 빠져 있습니다.</strong> 프로모·트레이너킷은 유통 경로가 불규칙해 시세 자체가 잡히지 않습니다.</li>
    <li><strong>사진이 없는 카드가 39장 있습니다.</strong> 세 출처를 다 뒤져도 없는 것들입니다 (My First Battle, Poké Card Creator Pack).</li>
    <li><strong>감정 등급 시세는 비싼 카드에만 붙습니다.</strong> raw 시세가 $100 이 넘는 카드부터 순서대로 모읍니다. 그 아래는 감정료가 카드값을 넘어서 감정 자체를 안 합니다.</li>
    <li><strong>등급 시장은 거래가 아주 얇습니다.</strong> 베이스셋 리자몽 PSA 10 조차 최근 1년에 <strong>1건</strong> 팔렸습니다. 건수가 한 자리면 "시세"라고 부르기 어렵습니다.</li>
    <li><strong>raw 와 PSA 10 은 몇 배씩 벌어집니다.</strong> 지금 모인 카드 기준 중앙값이 <strong>8.6배</strong>, 최대 40배가 넘습니다. 그 차이가 곧 감정료·대기시간·등급이 안 나올 위험의 값입니다.</li>
  </ul>

  <p class="pk-source">
    출처: <a href="https://tcgdex.dev/" rel="noopener">TCGdex</a> (카드·시세·이미지) ·
    <a href="https://pokeapi.co/" rel="noopener">PokéAPI</a> (한글 이름) ·
    TCGplayer 와 <a href="https://pokemontcg.io/" rel="noopener">pokemontcg.io</a>
    (TCGdex 에 이미지가 없는 카드의 사진) ·
    <a href="https://www.pokemonpricetracker.com/" rel="noopener">PokemonPriceTracker</a>
    (eBay 감정 등급 낙찰가).
    시세 참고용이며 투자 권유가 아닙니다.
  </p>

  <p class="pk-note" id="pk-meta"></p>

</section>

<section id="pk-view-krw" role="tabpanel" aria-labelledby="pk-tab-krw" hidden markdown="0">

  <h2>최근 한 달 국내 거래</h2>

  <div class="krw-stats" id="krw-stats"></div>
  <div class="krw-chart" id="krw-chart"></div>
  <p class="krw-readout" id="krw-readout"></p>
  <p class="pk-note">
    선은 그날 거래된 카드 전체의 <strong>중앙값</strong>, 옅은 막대는 <strong>거래건수</strong>입니다.
    차트를 누르거나 문지르면 그날 값이 나옵니다.
  </p>

  <h2>상품별 시세</h2>

  <div class="pk-search">
    <input type="search" id="krw-q" placeholder="이름·품번으로 검색 — 리자몽, Charizard, SV5A" autocomplete="off" aria-label="국내 상품 검색">
  </div>

  <div class="pk-filters">
    <select id="krw-lang" aria-label="언어판 거르기"><option value="">언어판 전체</option></select>
    <select id="krw-sort" aria-label="정렬">
      <option value="price-desc">비싼 순</option>
      <option value="price-asc">싼 순</option>
      <option value="chg-desc">30일 상승 순</option>
      <option value="chg-asc">30일 하락 순</option>
      <option value="tx-desc">거래 많은 순</option>
      <option value="name">이름순</option>
    </select>
    <label class="krw-check"><input type="checkbox" id="krw-liquid"> 거래 2건 이상만</label>
  </div>

  <p class="pk-count" id="krw-count">불러오는 중…</p>

  <div class="pk-grid" id="krw-grid"></div>

  <p class="pk-more"><button id="krw-more" type="button">더 보기</button></p>

  <h2>이 숫자를 믿기 전에</h2>

  <div class="pk-table-wrap">
    <table class="pk-table">
      <thead><tr><th>표시</th><th>정체</th><th>주의할 점</th></tr></thead>
      <tbody>
        <tr>
          <td><strong>현재가</strong></td>
          <td>KREAM 최근 체결 중앙값 (원)</td>
          <td>호가가 아니라 <strong>실제 체결가</strong>입니다. 다만 아래처럼 표본이 얇습니다.</td>
        </tr>
        <tr>
          <td>30일 고가·저가</td>
          <td>최근 30일 체결가의 최대·최소</td>
          <td>거래가 한 건이면 고가·저가·현재가가 모두 같은 값입니다.</td>
        </tr>
        <tr>
          <td>변동률</td>
          <td>1·7·30일 전 대비</td>
          <td>거래가 없으면 <strong>—</strong> 입니다. 0% 가 아니라 <strong>모른다</strong>는 뜻입니다.</td>
        </tr>
        <tr>
          <td>거래건수</td>
          <td>최근 30일 체결 건수</td>
          <td>이 숫자가 작을수록 위 값들을 믿을 근거가 약합니다.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h2>알아두실 것</h2>

  <ul>
    <li><strong>PSA 10 등급 기준입니다.</strong> 등급이 없는 카드나 낮은 등급은 값이 크게 다릅니다. 글로벌 탭의 <strong>raw 가격과 빼면 안 되고</strong>, 글로벌 쪽 PSA 10 과 견주세요 — <a href="#psa10" id="krw-to-cmp">PSA 10 탭</a>이 그것만 골라 계산해 줍니다.</li>
    <li><strong>대부분 일본판입니다.</strong> 899종 중 830종이 일어판이고 한글판은 54종뿐입니다.</li>
    <li><strong>표본이 얇습니다.</strong> 899종 가운데 404종은 30일 거래가 1건 이하입니다. 한 사람이 한 번 판 값이 그대로 "시세"가 됩니다.</li>
    <li><strong>KREAM 에 올라온 상품만 있습니다.</strong> 번개장터·중고나라 등 다른 경로의 거래는 안 잡힙니다.</li>
    <li><strong>수수료·감정료가 빠진 값입니다.</strong> 실제 손익은 <a href="/finance/포켓몬카드_투자_총비용_감정_수수료_세금/">총비용 편</a>을 참고하세요.</li>
  </ul>

  <p class="pk-source">
    출처: <a href="https://content.kream.co.kr/pokemon-tcg-chart" rel="noopener" target="_blank">KREAM 포켓몬 카드 시세표</a>.
    시세 참고용이며 투자 권유가 아닙니다.
  </p>

  <p class="pk-note" id="krw-meta"></p>

</section>

<section id="pk-view-cmp" role="tabpanel" aria-labelledby="pk-tab-cmp" hidden markdown="0">

  <h2>PSA 10 으로 두 시장 나란히 보기</h2>

  <p class="pk-warn">
    <strong>합치는 건 품번이 같을 때만입니다.</strong> 국내 일어판·한글판은 품번이 같아
    한 줄로 묶었습니다(40줄). 하지만 <strong>글로벌과 국내는 안 붙입니다</strong> —
    영문판 14종으로 품번을 이어봤더니 확실히 이어진 게 1건뿐이었습니다.
    이름으로 이으면 "피카츄" 하나에 국내 112종·글로벌 98장이 엉킵니다.
    <strong>표에서 글로벌 줄과 국내 줄을 하나씩 고르면</strong> 그 둘만 견줍니다.
  </p>

  <div class="cmp-modes" role="radiogroup" aria-label="찾는 방법">
    <label><input type="radio" name="cmp-mode" value="name" checked> 이름으로</label>
    <label><input type="radio" name="cmp-mode" value="band"> 가격대로</label>
    <span class="cmp-curpick">
      <label for="cmp-display">표시 통화</label>
      <select id="cmp-display" aria-label="목록에 함께 보일 통화">
        <option value="KRW">원 (₩)</option>
        <option value="USD">달러 ($)</option>
        <option value="EUR">유로 (€)</option>
        <option value="JPY">엔 (¥)</option>
      </select>
    </span>
  </div>

  <div class="pk-search" id="cmp-by-name">
    <input type="search" id="cmp-q" placeholder="포켓몬 이름으로 양쪽 동시 검색 — 리자몽, 피카츄" autocomplete="off" aria-label="양쪽 시장 검색">
  </div>

  <div class="cmp-band" id="cmp-by-band" hidden>
    <input type="text" id="cmp-amount" inputmode="decimal" placeholder="금액 — 예: 10000000" autocomplete="off" aria-label="기준 금액">
    <select id="cmp-cur" aria-label="입력한 금액의 통화">
      <option value="KRW">원</option>
      <option value="USD">달러</option>
      <option value="EUR">유로</option>
      <option value="JPY">엔</option>
    </select>
    <select id="cmp-tol" aria-label="허용 범위">
      <option value="10">±10%</option>
      <option value="20" selected>±20%</option>
      <option value="35">±35%</option>
      <option value="50">±50%</option>
    </select>
    <p class="pk-note">이 금액 근처의 <strong>PSA 10</strong> 카드를 양쪽에서 찾습니다. 이름이 달라도 값이 비슷하면 견줄 거리가 됩니다.</p>
  </div>

  <div class="cmp-picked" id="cmp-panel" hidden>
    <div class="pk-table-wrap">
      <table class="pk-table cmp-table">
        <thead>
          <tr><th>항목</th><th>글로벌 · 영문판</th><th>국내 · KREAM</th><th>차이</th></tr>
        </thead>
        <tbody id="cmp-tbody"></tbody>
      </table>
    </div>
    <p class="cmp-verdict" id="cmp-gap"></p>
  </div>

  <h2>PSA 10 통합 표</h2>

  <p class="pk-note">
    한 줄이 한 카드입니다. <strong>PSA 10 만</strong> 담았습니다 — 글로벌의 raw 가격은
    종류가 다른 값이라 섞으면 가격순 정렬이 곧바로 거짓말이 됩니다. raw 는
    <strong>글로벌 탭</strong>에 그대로 있습니다.
  </p>

  <div class="cmp-uni-bar">
    <span class="pk-count" id="uni-count">불러오는 중…</span>
    <label class="krw-check"><input type="checkbox" id="uni-merged"> 언어판이 둘 이상인 것만</label>
    <select id="uni-sort" aria-label="정렬">
      <option value="price-desc">비싼 순</option>
      <option value="price-asc">싼 순</option>
      <option value="ratio-desc">언어판 격차 큰 순</option>
      <option value="name">이름순</option>
    </select>
  </div>

  <div class="pk-table-wrap cmp-scroll">
    <table class="pk-table cmp-pick">
      <thead>
        <tr>
          <th>카드</th><th>시장 · 품번</th>
          <th>PSA 10</th><th>표본</th><th>언어판 격차</th>
        </tr>
      </thead>
      <tbody id="uni-list"></tbody>
    </table>
  </div>

  <h2>이 표를 읽는 법</h2>

  <ul>
    <li><strong>김치 프리미엄이 뭔가요.</strong> 국내 가격이 해외보다 비싼 정도입니다. <strong>+</strong> 면 국내가 비싸고(김프), <strong>−</strong> 면 국내가 싼 <strong>역프리미엄</strong>입니다. 코인 시장에서 굳은 말을 그대로 씁니다 — 개념은 <a href="/finance/김치프리미엄_차익_세후계산_심화/">김프 편</a>에 정리해 두었습니다.</li>
    <li><strong>이 폭이 그대로 차익은 아닙니다.</strong> 수수료·감정료·관세·환전비용이 다 빠진 값입니다. 코인 김프가 그렇듯, <strong>마찰비용이 프리미엄보다 크면 남는 게 없습니다.</strong> <a href="/finance/포켓몬카드_투자_총비용_감정_수수료_세금/">총비용 편</a>에 왕복 계산이 있습니다.</li>
    <li><strong>같은 등급끼리만 뺍니다.</strong> 국내는 전부 PSA 10 이라, 글로벌 카드에 PSA 10 값이 없으면 <strong>계산하지 않습니다</strong>. raw 와 PSA 10 을 빼면 나오는 건 나라 차이가 아니라 등급 프리미엄이니까요.</li>
    <li><strong>영문판과 일본판은 다른 물건입니다.</strong> 같은 포켓몬·같은 기술이라도 인쇄·유통량이 달라 값이 따로 움직입니다. 국내 목록의 <strong>영문판</strong> 상품끼리 견주면 가장 깔끔합니다.</li>
    <li><strong>언어판 격차를 보세요.</strong> 같은 품번인데 일어판이 한글판의 <strong>3~10배</strong>인 카드가 있습니다. 나라 사이 차이보다 이쪽이 더 클 때가 많습니다.</li>
    <li><strong>여기 없는 값은 글로벌 탭에 있습니다.</strong> raw 시세, Cardmarket 유럽 실거래, PSA 9 는 종류가 다른 값이라 이 표에 넣지 않았습니다.</li>
    <li><strong>환율은 매일 바뀝니다.</strong> 환산에 쓴 환율과 기준일을 맨 아래에 적어 두었습니다.</li>
  </ul>

  <p class="pk-note" id="cmp-meta"></p>

</section>

</div>

<script src="{{ '/assets/pokemon/app.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
<script src="{{ '/assets/pokemon/krw.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
