---
layout: single
title: "포켓몬 카드 시세"
permalink: /dashboard/pokemon/
classes: wide
author_profile: false
toc: false
description: "포켓몬 카드 현재 시세를 카드 사진과 함께 봅니다. 영어 이름과 한글 이름 둘 다로 검색할 수 있고, 현재가·현재 최고 호가·관측 최고가를 나란히 보여줍니다. TCGplayer·Cardmarket 시세를 매일 자동 갱신합니다."
---

<link rel="stylesheet" href="{{ '/assets/pokemon/pokemon.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="pk-app" data-base="{{ '/assets/pokemon' | relative_url }}">

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
      </tbody>
    </table>
  </div>

  <h2>알아두실 것</h2>

  <ul>
    <li><strong>역대 최고 낙찰가는 없습니다.</strong> 무료로 열린 어떤 API 도 주지 않습니다. 위 세 가지가 구할 수 있는 전부입니다.</li>
    <li><strong>한글 이름은 포켓몬 이름입니다.</strong> 카드 정식 한글명이 공개된 데이터가 없어, 도감번호로 포켓몬 종 이름을 붙였습니다. <code>Charizard ex</code> 는 <code>리자몽 ex</code> 로 나옵니다.</li>
    <li><strong>트레이너·에너지 카드는 한글 이름이 없습니다.</strong> 도감번호가 없기 때문입니다.</li>
    <li><strong>가격이 안 잡히는 카드는 빠져 있습니다.</strong> 프로모·트레이너킷은 유통 경로가 불규칙해 시세 자체가 잡히지 않습니다.</li>
    <li><strong>영문판·달러 기준입니다.</strong> 국내 원화 시세는 공개 API 가 없어 여기서 다루지 않습니다. 원화로 보시려면 아래 링크를 쓰세요.</li>
  </ul>

  <h2>원화 시세는 여기서</h2>

  <p>
    국내 시세를 여는 공개 API 가 없어 이 대시보드는 달러·유로만 다룹니다.
    원화로 보시려면 <a href="https://content.kream.co.kr/pokemon-tcg-chart" rel="noopener" target="_blank">KREAM 포켓몬 카드 시세표</a>가
    <strong>PSA 10 등급</strong> 기준 원화 시세를 보여줍니다. 등급 카드만 다루므로
    이 대시보드(등급 없는 raw 카드 시세)와는 기준이 다릅니다 — 같은 카드라도
    금액이 크게 벌어집니다.
  </p>

  <p class="pk-source">
    출처: <a href="https://tcgdex.dev/" rel="noopener">TCGdex</a> (카드·시세·이미지) ·
    <a href="https://pokeapi.co/" rel="noopener">PokéAPI</a> (한글 이름) ·
    TCGplayer (TCGdex 에 이미지가 없는 카드의 제품 사진).
    시세 참고용이며 투자 권유가 아닙니다.
  </p>

  <p class="pk-note" id="pk-meta"></p>

</div>

<script src="{{ '/assets/pokemon/app.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
