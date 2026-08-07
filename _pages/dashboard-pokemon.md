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
  </div>

  <p class="pk-warn">
    두 탭은 <strong>같은 카드의 다른 값이 아닙니다.</strong> 글로벌은 등급 없는 raw 영문판,
    국내는 <strong>PSA 10 등급</strong>에 대부분 일본판입니다. 나란히 놓고 빼면 안 됩니다.
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
    <li><strong>PSA 10 등급 기준입니다.</strong> 등급이 없는 카드나 낮은 등급은 값이 크게 다릅니다. 글로벌 탭(raw)과 나란히 비교하면 안 됩니다.</li>
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

</div>

<script src="{{ '/assets/pokemon/app.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
<script src="{{ '/assets/pokemon/krw.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
