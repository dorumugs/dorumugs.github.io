---
layout: single
title: "대시보드"
permalink: /dashboard/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/dashboard-hub/header.svg
  teaser: /assets/images/dashboard-hub/header.svg
description: "데이터로 보는 대시보드를 모았습니다. 국토교통부 실거래가 435만 건으로 만든 부동산 도구 셋과, 구성·가중치·산식을 전부 공개한 포켓몬 카드 가격지수."
---

<link rel="stylesheet" href="{{ '/assets/realestate/hub.css' | relative_url }}?v={{ site.time | date: '%s' }}">

글로 한 번 정리한 주제를 눌러볼 수 있는 화면으로 옮기고 있습니다.
지금은 수도권 부동산과 포켓몬 카드 시장을 다룹니다.

## 부동산

<div class="rh-grid">

  <a class="rh-card is-live" href="{{ '/dashboard/real-estate/trades/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">실거래 대시보드</h2>
    <p class="rh-desc">
      국토교통부 실거래가 <strong>435만 건</strong>(2006년~현재)을 서울·경기 72개 시군구 지도에 올렸습니다.
      구를 누르면 22년치 평당가 추이와 단지 랭킹이 나옵니다.
    </p>
    <ul class="rh-points">
      <li>중위 평당가 · 3/6/12개월 변화율 · 전고점 대비 · 거래 회전율</li>
      <li>300세대 이상 단지만 보기 토글</li>
      <li>매일 새 실거래가 자동 반영</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>

  <a class="rh-card is-live" href="{{ '/dashboard/real-estate/schools/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">학군 지도</h2>
    <p class="rh-desc">
      서울·경기 <strong>사립초 41곳 · 사립중 197곳</strong>을 지도에 올렸습니다.
      학교를 누르면 그 학교가 속한 법정동의 아파트 실거래 시세가 나옵니다.
    </p>
    <ul class="rh-points">
      <li>학교 위치와 그 동네 평당가를 한 화면에서</li>
      <li>학교급(사립초/사립중)으로 걸러보기</li>
      <li>진학 실적 기반 필터는 데이터가 공개되지 않아 없습니다</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>

  <a class="rh-card is-live" href="{{ '/dashboard/real-estate/redevelopment/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">재개발·재건축</h2>
    <p class="rh-desc">
      서울 정비사업장 <strong>1,102곳</strong>의 진행단계와 인가 일자를 모았습니다.
      노후 아파트의 <strong>대지지분</strong>을 세워 두고, 인가 통과가 실거래가를
      실제로 얼마나 움직였는지 435만 건으로 따집니다.
    </p>
    <ul class="rh-points">
      <li>대지지분 · 용도지역 · 용적률 상한 · 준공연차로 정렬</li>
      <li>조합설립 → 사업시행 → 관리처분 인가일과 동의율</li>
      <li>인가 전후 12개월 초과수익 — 재건축은 아파트, 재개발은 연립·다세대 실거래로</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>

  <div class="rh-card is-planned">
    <span class="rh-badge">구상 중</span>
    <h2 class="rh-title">입지 분석</h2>
    <p class="rh-desc">
      교통·생활권·정비사업처럼 가격에 앞서 움직이는 것들을 모읍니다.
      금액대별로 어디까지 갈 수 있는지 보여주는 쪽으로 생각하고 있습니다.
    </p>
    <span class="rh-go is-muted">글 먼저 읽기 ↓</span>
  </div>

</div>

## 수집품

<div class="rh-grid">

  <a class="rh-card is-live" href="{{ '/dashboard/pokemon/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">포켓몬 카드 가격지수</h2>
    <p class="rh-desc">
      시대 4 × 가격대 3 = <strong>12칸</strong>으로 나눠 층화추출한 <strong>300장</strong>의
      가격지수입니다. 남의 지수를 인용하지 않고 원가격에서 직접 만듭니다.
    </p>
    <ul class="rh-points">
      <li>구성 종목 300장·가중치·산식 전부 공개</li>
      <li>시대별·가격대별 하위지수</li>
      <li>매일 TCGplayer 시세 자동 갱신</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>

</div>

## 바탕이 된 글

대시보드는 이 글들에서 출발했습니다.

### 부동산 — 실거래·가격

- [수도권 아파트 전용 84 구별 중위가격 지도](/finance/수도권_아파트_전용84_구별_중위가격_지도/)
- [세대수로 보정한 중위가격](/finance/수도권_아파트_전용84_세대수_보정_중위가격/)
- [토지거래허가구역, 세 낀 집 입주와 전세 퇴거](/finance/토허구역_세낀집_입주_전세퇴거/)

### 부동산 — 학군

- [학군지 (1) 서울 3대 — 대치·목동·중계](/finance/학군지_1_서울3대_대치목동중계/)
- [학군지 (2) 1기 신도시 — 분당·평촌·일산](/finance/학군지_2_1기신도시_분당평촌일산/)
- [학군지 (3) 신흥 도심 — 광장동·송도·대흥](/finance/학군지_3_신흥도심_광장동송도대흥/)
- [사립초 이후 중학교 코스 — 일반중·국제중·사립중](/finance/사립초_이후_중학교_코스_일반중_국제중_사립중/)
- [서울·경기 국제중·사립중 학교지도](/finance/서울_경기_국제중_사립중_학교지도_위치_특징_장단점/)

### 부동산 — 입지

- [서대문구 입지분석 — 금액대별 투자지도](/finance/서대문구_부동산_입지분석_금액대별_투자지도/)

### 수집품 — 포켓몬 카드

가격지수 대시보드는 이 시리즈 1편에서 "남의 지수는 근거로 못 쓴다"고 쓴 데 대한 답입니다.

- [(1/5) 포켓몬 카드는 자산인가 — 시장 구조와 '3,821% 지수'의 진실](/finance/포켓몬카드_투자_자산군_지수해부/)
- [(2/5) 왕복 비용부터 계산하자 — 감정료·수수료·세금](/finance/포켓몬카드_투자_총비용_감정_수수료_세금/)
- [(3/5) 공급이 최대 리스크 — 인쇄량과 감정 인구](/finance/포켓몬카드_투자_공급구조_인쇄량_감정인구/)
- [(4/5) 수요는 어디서 오고 언제 멈추나 — 사이클과 규제 리스크](/finance/포켓몬카드_투자_수요사이클_규제리스크/)
- [(5/5) 그래도 한다면 — 배분·선별·보관 체크리스트](/finance/포켓몬카드_투자_실행규칙_포트폴리오_체크리스트/)
