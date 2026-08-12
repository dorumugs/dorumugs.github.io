/* ETF·테마 20일 모멘텀 대시보드.

   외부 라이브러리를 쓰지 않는다. 스파크라인도 인라인 SVG 로 직접 그린다.

   파일을 셋으로 나눈 이유가 있다. etfs.json 과 groups.json 은 목록을 그리는 데
   바로 필요하지만, details.json 은 1MB 가 넘는데 카드를 눌러야 쓴다. 처음부터
   받으면 모바일에서 첫 화면이 그만큼 늦어진다. 그래서 지연 로딩한다. */
(function () {
  'use strict';

  var app = document.querySelector('.ef-app');
  if (!app) { return; }
  var BASE = app.dataset.base;
  var PAGE = 24;
  var GROUP_PAGE = 30;

  var state = {
    meta: null,
    backtest: null,
    etfs: [],
    groups: [],
    details: null,
    detailsPromise: null,
    etfShown: PAGE,
    themeShown: GROUP_PAGE,
    upjongShown: GROUP_PAGE,
    risk: 300000,
    us: null,
    usHoldings: null,
    usShown: PAGE,
    basket: [],
    corr: null,
    corrPromise: null
  };

  var $ = function (id) { return document.getElementById(id); };

  /* 보유 기간 라벨. meta 에서 읽어 만든다 — '2주(10일)' 을 문자열에 박아 두면
     기간을 바꿀 때마다 화면 곳곳이 거짓말을 한다. */
  function swingLabel() {
    var m = state.meta || {};
    return (m.swingWeeks || 8) + '주(' + (m.swingLookback || 40) + '일)';
  }
  function swingWeeks() { return (state.meta && state.meta.swingWeeks) || 8; }

  function fetchJson(name) {
    return fetch(BASE + '/' + name + '.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) { throw new Error(name + ' ' + r.status); }
      return r.json();
    });
  }

  /* --- 표시 도우미 ------------------------------------------------------- */

  function pct(v, digits) {
    if (v === null || v === undefined) { return '—'; }
    var d = digits === undefined ? 1 : digits;
    return (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
  }

  function moneyShort(v) {
    if (v === null || v === undefined) { return '—'; }
    if (v >= 1e12) { return (v / 1e12).toFixed(1) + '조'; }
    if (v >= 1e8) { return Math.round(v / 1e8) + '억'; }
    return Math.round(v / 1e4) + '만';
  }

  function dirClass(v) {
    if (v === null || v === undefined) { return 'ef-flat'; }
    if (v > 0) { return 'ef-up'; }
    if (v < 0) { return 'ef-down'; }
    return 'ef-flat';
  }

  function prettyDate(yyyymmdd) {
    if (!yyyymmdd || yyyymmdd.length !== 8) { return yyyymmdd || ''; }
    return yyyymmdd.slice(0, 4) + '-' + yyyymmdd.slice(4, 6) + '-' + yyyymmdd.slice(6);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* 스파크라인. 값은 시작을 100 으로 맞춘 배열이라 그대로 최소·최대만 잡으면 된다. */
  function sparkSvg(values, up) {
    if (!values || values.length < 2) { return ''; }
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = max - min || 1;
    var step = 100 / (values.length - 1);
    var points = values.map(function (v, i) {
      return (i * step).toFixed(2) + ',' + (28 - ((v - min) / span) * 26).toFixed(2);
    }).join(' ');
    var color = up > 0 ? '#c0392b' : (up < 0 ? '#2471c4' : '#999');
    return '<svg class="ef-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true">' +
      '<polyline fill="none" stroke="' + color + '" stroke-width="1.4" ' +
      'vector-effect="non-scaling-stroke" points="' + points + '"></polyline></svg>';
  }

  function badge(grade) {
    return '<span class="ef-badge ef-g-' + escapeHtml(grade) + '">' + escapeHtml(grade) + '</span>';
  }

  function breadthBar(breadth) {
    if (breadth === null || breadth === undefined) {
      return '<div class="ef-breadth"><span>상승 비율 확인 불가</span></div>';
    }
    var w = Math.round(breadth * 100);
    return '<div class="ef-breadth"><span class="ef-breadth-bar"><i style="width:' + w + '%"></i></span>' +
      '<span>구성종목 ' + w + '% 상승</span></div>';
  }

  /* --- 카드 -------------------------------------------------------------- */

  function levLabel(lev) {
    if (lev === 2) { return '레버리지 2배'; }
    if (lev === -1) { return '인버스'; }
    if (lev === -2) { return '인버스 2배'; }
    return null;
  }

  /* 보유 기간 매매에 필요한 세 숫자. 어디서 자를지, 보통 얼마나 흔들리는지, 그 손절이
     얼마나 자주 걸릴 자리인지. 이게 없으면 '오른다' 는 정보만으로 주문을 못 낸다. */
  function swingRow(row) {
    if (row.stopPct === null || row.stopPct === undefined) { return ''; }
    // 손절을 보유기간 1σ 로 맞춰 두었으므로 걸릴 확률이 대부분 32% 로 같다.
    // 같은 숫자를 천 번 찍으면 정보가 아니라 벽지가 된다. 기본값에서 벗어난
    // 경우(2×ATR 이 1σ 보다 넓어 손절이 더 여유로운 경우)만 짚는다.
    var loose = row.stopProb !== null && row.stopProb !== undefined && row.stopProb < 0.28;
    return '<span class="ef-swing">' +
      '<span>손절 <em class="ef-down">' + pct(row.stopPct) + '</em></span>' +
      '<span>' + swingWeeks() + '주 폭 <em>±' + (row.expectedSwing * 100).toFixed(1) + '%</em></span>' +
      (loose ? '<span>손절 여유 <em>걸릴 확률 ' + (row.stopProb * 100).toFixed(0) + '%</em></span>' : '') +
      '</span>';
  }

  /* 몇 주를 살 것인가.

     "손절폭으로 나눠서 수량을 잡으세요" 라고 써 놓고 계산을 안 해 주면 아무도
     안 한다. 산수는 자명하다 — 잃어도 되는 금액을 손절폭으로 나누면 수량이다.

       수량 = 위험금액 ÷ (현재가 × 손절폭)

     이렇게 잡으면 손절 −22% 짜리와 −3% 짜리에 **같은 금액이 아니라 같은 위험**을
     걸게 된다. 손절폭이 넓은 물건에 같은 돈을 넣으면 일곱 배를 잃는다.

     투입금액이 하루 거래대금의 1% 를 넘으면 경고한다. 유동성 하한(5억)은 모두에게
     같은 값이지만, 실제로 못 빠져나오는지는 **내 주문 크기**에 달렸다. */
  function sizing(row) {
    if (!row.stopPct || row.price === null || row.price === undefined) { return null; }
    // 금리형·판정불가에는 수량을 내지 않는다. 손절폭이 0.03% 라 위험 기반
    // 수량이 천문학적으로 나오는데, 애초에 스윙으로 담을 물건이 아니다.
    if (row.grade === '금리형' || row.grade === '판정불가') { return null; }
    var perShare = row.price * Math.abs(row.stopPct);
    if (perShare <= 0) { return null; }
    var byRisk = Math.floor(state.risk / perShare);
    if (byRisk < 1) {
      return { shares: 0, amount: 0, capped: false,
               note: '한 주만 사도 손실 한도를 넘습니다. 이 종목은 지금 규모로 못 담습니다.' };
    }
    /* 유동성 상한. 손절폭이 아주 좁은 물건(머니마켓·CD금리류는 0.03% 다)에
       위험 기반 수량만 쓰면 '10억어치 사라' 같은 답이 나온다. 산수는 맞지만
       현실이 아니다 — 하루 거래대금의 1% 를 넘겨 담으면 넣고 빼는 데 값이
       밀리므로, 거기서 끊는다. 어느 쪽에 걸렸는지도 같이 알려준다. */
    var byLiquidity = row.turnover
      ? Math.floor(row.turnover * 0.01 / row.price)
      : byRisk;
    var shares = Math.max(0, Math.min(byRisk, byLiquidity));
    if (shares < 1) {
      return { shares: 0, amount: 0, capped: true,
               note: '하루 거래대금이 너무 적어 한 주도 감당이 안 됩니다.' };
    }
    return {
      shares: shares,
      amount: shares * row.price,
      capped: shares < byRisk,
      byRisk: byRisk,
      note: ''
    };
  }

  /* 고점 · 현재가 · 물린 물량. 위에 매물이 쌓여 있으면 오를 때마다 본전 찾는
     매도가 나온다. 짧은 보유에서는 지표보다 직접적인 장애물이다. */
  function supplyRow(row) {
    if (row.price === null || row.price === undefined) { return ''; }
    var heavy = row.overhead !== null && row.overhead >= 0.6;
    return '<span class="ef-swing ef-supply">' +
      '<span>현재 <em>' + Math.round(row.price).toLocaleString() + '</em></span>' +
      (row.high52 === null ? '' :
        '<span>' + ((row.high52Days || 0) >= 250 ? '52주 고점' : '상장 후 최고') +
        ' <em>' + Math.round(row.high52).toLocaleString() + '</em> ' +
        '<i class="' + dirClass(row.fromHigh52) + '">' + pct(row.fromHigh52) + '</i></span>') +
      (row.overhead === null ? '' :
        '<span>위에 물린 물량 <em' + (heavy ? ' class="ef-down"' : '') + '>' +
        Math.round(row.overhead * 100) + '%</em></span>') +
      (function () {
        var s = sizing(row);
        if (!s) { return ''; }
        if (!s.shares) { return '<span class="ef-down">한도 초과 — 못 담음</span>'; }
        return '<span>수량 <em>' + s.shares.toLocaleString() + '주</em> · 투입 <em>' +
          moneyShort(s.amount) + '</em>' + (s.capped ? ' <i class="ef-down">유동성 상한</i>' : '') +
          '</span>';
      })() +
      '</span>';
  }

  function etfCard(row) {
    var tags = [];
    var lev = levLabel(row.lev);
    if (lev) { tags.push('<span class="ef-tag is-warn">' + lev + '</span>'); }
    if (row.hedged) { tags.push('<span class="ef-tag">환헤지</span>'); }
    if (row.premium !== null && Math.abs(row.premium) > 0.005) {
      tags.push('<span class="ef-tag is-warn">괴리 ' + pct(row.premium, 2) + '</span>');
    }
    if (!row.primary) { tags.push('<span class="ef-tag">같은 테마 중복</span>'); }
    tags.push('<span class="ef-tag">' + escapeHtml(row.tabName) + '</span>');

    var relative = row.bench
      ? escapeHtml(row.bench) + ' 대비 <em>' + pct(row.excess) + '</em>'
      : (row.pct === null || row.pct === undefined ? '' : '같은 분류 상위 <em>' + (100 - row.pct).toFixed(0) + '%</em>');

    var inBasket = state.basket.indexOf(row.code) >= 0;
    return '<div class="ef-cardwrap">' +
      '<button type="button" class="ef-basket-toggle' + (inBasket ? ' is-on' : '') +
      '" data-basket="' + escapeHtml(row.code) + '" aria-pressed="' + inBasket + '">' +
      (inBasket ? '담음 ✓' : '담기') + '</button>' +
      '<button type="button" class="ef-card" data-kind="etf" data-code="' + escapeHtml(row.code) + '">' +
      '<span class="ef-card-top"><span class="ef-name">' + escapeHtml(row.name) + '</span></span>' +
      '<span class="ef-heads">' +
        '<span class="ef-head"><i>' + swingLabel() + '</i><b class="' + dirClass(row.rSwing) + '">' + pct(row.rSwing) + '</b></span>' +
        '<span class="ef-head"><i>20일</i><b class="' + dirClass(row.r20) + '">' + pct(row.r20) + '</b></span>' +
      '</span>' +
      badge(row.grade) +
      sparkSvg(row.spark, row.r20) +
      swingRow(row) +
      supplyRow(row) +
      '<span class="ef-sub">' +
        '<span>거래대금 <em>' + moneyShort(row.turnover) + '</em></span>' +
        (relative ? '<span>' + relative + '</span>' : '') +
        (row.groupName ? '<span>대표 <em>' + escapeHtml(row.groupName) + '</em></span>' : '') +
      '</span>' +
      breadthBar(row.breadth) +
      '<span class="ef-tagrow">' + tags.join('') + '</span>' +
      '</button></div>';
  }

  /* 테마 265개를 카드로 늘어놓으면 훑을 수가 없다. 표로 놓고 테마·업종을
     따로 세운다. 390px 에서는 자체 스크롤 안에서만 옆으로 넘친다. */
  function groupTable(rows, shown) {
    if (!rows.length) {
      return '<p class="ef-note">조건에 맞는 항목이 없습니다.</p>';
    }
    var body = rows.slice(0, shown).map(function (r) {
      var mark = r.etfCount ? r.etfCount + '개' : '—';
      return '<tr class="ef-row" data-kind="group" data-code="' + escapeHtml(r.key) + '" tabindex="0">' +
        '<td class="ef-rowname">' + escapeHtml(r.name) + '</td>' +
        '<td class="' + dirClass(r.rSwing) + '">' + pct(r.rSwing) + '</td>' +
        '<td class="' + dirClass(r.r20) + '">' + pct(r.r20) + '</td>' +
        '<td>' + (r.breadth === null ? '—' : Math.round(r.breadth * 100) + '%') + '</td>' +
        '<td class="' + dirClass(r.excess) + '">' + pct(r.excess) + '</td>' +
        '<td' + (r.overhead !== null && r.overhead >= 0.6 ? ' class="ef-down"' : '') + '>' +
          (r.overhead === null ? '—' : Math.round(r.overhead * 100) + '%') + '</td>' +
        '<td>' + badge(r.grade) + '</td>' +
        '<td>' + r.members + '</td>' +
        '<td>' + mark + '</td></tr>';
    }).join('');
    return '<div class="ef-tablewrap"><table class="ef-table ef-grouptable"><thead><tr>' +
      '<th>이름</th><th>' + swingWeeks() + '주</th><th>20일</th><th>상승비율</th><th>시장대비</th>' +
      '<th>물린 물량</th><th>등급</th><th>종목</th><th>ETF</th></tr></thead><tbody>' + body +
      '</tbody></table></div>';
  }

  /* --- 미국 ETF ----------------------------------------------------------- */

  function usLevLabel(lev) {
    if (lev === 0 || lev === 1) { return null; }
    var abs = Math.abs(lev);
    return (lev < 0 ? '인버스 ' : '레버리지 ') + abs + '배';
  }

  /* 미국 ETF 는 국내와 다른 게 많아 카드를 따로 그린다.

     가장 중요한 차이는 **환율**이다. 원화로 사는 사람의 수익률은 달러 수익률이
     아니다. 둘을 나란히 놓지 않으면 달러로 +30% 인데 원화로 +22% 인 것을 모른다. */
  function usCard(row) {
    var tags = [];
    var lev = usLevLabel(row.lev);
    if (lev) { tags.push('<span class="ef-tag is-warn">' + lev + '</span>'); }
    if (!row.isEtf) { tags.push('<span class="ef-tag is-warn">ETN — 발행사 신용위험</span>'); }
    tags.push('<span class="ef-tag">' + escapeHtml(row.exchange || '미국') + '</span>');

    var s = usSizing(row);
    return '<div class="ef-cardwrap">' +
      '<button type="button" class="ef-card" data-kind="us" data-code="' + escapeHtml(row.code) + '">' +
      '<span class="ef-card-top"><span class="ef-name">' +
        '<b class="ef-ticker">' + escapeHtml(row.ticker) + '</b> ' + escapeHtml(row.name) + '</span></span>' +
      '<span class="ef-heads">' +
        '<span class="ef-head"><i>' + swingWeeks() + '주 · 달러</i><b class="' +
          dirClass(row.rSwing) + '">' + pct(row.rSwing) + '</b></span>' +
        '<span class="ef-head"><i>' + swingWeeks() + '주 · 원화</i><b class="' +
          dirClass(row.rSwingKrw) + '">' + pct(row.rSwingKrw) + '</b></span>' +
        '<span class="ef-head"><i>20일 · 달러</i><b class="' +
          dirClass(row.r20) + '">' + pct(row.r20) + '</b></span>' +
      '</span>' +
      badge(row.grade) +
      sparkSvg(row.spark, row.r20) +
      (row.stopPct === null ? '' :
        '<span class="ef-swing">' +
        '<span>손절 <em class="ef-down">' + pct(row.stopPct) + '</em></span>' +
        '<span>' + swingWeeks() + '주 폭 <em>±' + (row.expectedSwing * 100).toFixed(1) + '%</em></span>' +
        (s && s.shares ? '<span>수량 <em>' + s.shares.toLocaleString() + '주</em> · 투입 <em>' +
          moneyShort(s.amount) + '</em></span>' : '') +
        '</span>') +
      '<span class="ef-sub">' +
        '<span>현재 <em>$' + (row.price === null ? '—' : row.price.toLocaleString()) + '</em>' +
        (row.priceKrw ? ' <i>' + row.priceKrw.toLocaleString() + '원</i>' : '') + '</span>' +
        '<span>S&P500 대비 <em>' + pct(row.excess) + '</em></span>' +
        '<span>거래대금 <em>' + moneyShort(row.turnoverKrw) + '</em></span>' +
        (row.overhead === null ? '' : '<span>위에 물린 물량 <em' +
          (row.overhead >= 0.6 ? ' class="ef-down"' : '') + '>' +
          Math.round(row.overhead * 100) + '%</em></span>') +
      '</span>' +
      '<span class="ef-tagrow">' + tags.join('') + '</span>' +
      '</button></div>';
  }

  /* 미국 ETF 는 달러로 거래하지만 잃어도 되는 금액은 원화로 정한다.
     수량 = 위험금액(원) ÷ (원화 환산 가격 × 손절폭). */
  function usSizing(row) {
    if (!row.stopPct || !row.priceKrw) { return null; }
    if (row.grade === '금리형' || row.grade === '판정불가') { return null; }
    var perShare = row.priceKrw * Math.abs(row.stopPct);
    if (perShare <= 0) { return null; }
    var byRisk = Math.floor(state.risk / perShare);
    var byLiquidity = row.turnoverKrw
      ? Math.floor(row.turnoverKrw * 0.01 / row.priceKrw) : byRisk;
    var shares = Math.max(0, Math.min(byRisk, byLiquidity));
    if (shares < 1) { return { shares: 0, amount: 0, capped: byLiquidity < byRisk }; }
    return { shares: shares, amount: shares * row.priceKrw,
             capped: shares < byRisk, byRisk: byRisk };
  }

  function filteredUs() {
    if (!state.us) { return []; }
    var q = $('ef-uq').value.trim().toLowerCase();
    var grade = $('ef-ugrade').value;
    var lev = $('ef-ulev').value;
    var liq = parseFloat($('ef-uliq').value) || 0;
    return sortRows(state.us.rows.filter(function (r) {
      if (q && r.name.toLowerCase().indexOf(q) < 0 && r.ticker.toLowerCase().indexOf(q) < 0) { return false; }
      if (grade && r.grade !== grade) { return false; }
      if (lev === '1' && r.lev !== 1) { return false; }
      if (lev === 'lev' && r.lev <= 1) { return false; }
      if (lev === 'lev3' && r.lev < 3) { return false; }
      if (lev === 'inv' && r.lev >= 0) { return false; }
      if (liq && (r.turnoverKrw === null || r.turnoverKrw < liq)) { return false; }
      return true;
    }), $('ef-usort').value);
  }

  function renderUs() {
    if (!state.us) { return; }
    var rows = filteredUs();
    var shown = rows.slice(0, state.usShown);
    $('ef-ulist').innerHTML = shown.map(usCard).join('');
    $('ef-ucount').textContent = rows.length
      ? '미국 ETF ' + rows.length + '개 중 ' + shown.length + '개 표시'
      : '조건에 맞는 ETF 가 없습니다.';
    $('ef-umore2').hidden = shown.length >= rows.length;

    var m = state.us.meta;
    var regCls = m.regime.label === '역풍'
      ? 'background:#fdf0f0;border-color:#f0d0d0;color:#8a2b2b'
      : 'background:#e9f5ee;border-color:#c3e2d0;color:#16704a';
    $('ef-us-market').innerHTML =
      '<span class="ef-stat" style="flex:1 1 100%;' + regCls + '">' +
        '<b style="font-size:1em">시장 국면 · ' + escapeHtml(m.regime.label) + '</b>' +
        '<span>' + escapeHtml(m.regime.note) + '</span></span>' +
      '<span class="ef-stat" style="flex:1 1 100%;background:#fff8e1;border-color:#f0e0a8;color:#7a5b00">' +
        '<b style="font-size:1em">환율이 섞여 있습니다</b>' +
        '<span>원달러 ' + m.fxRate.toLocaleString() + '원. 원화 수익률은 달러 수익률에 ' +
        '환율 변동을 곱한 값입니다. <strong>양도소득세 22%</strong>(연 250만원 공제)도 ' +
        '어떤 숫자에도 반영돼 있지 않습니다.</span></span>' +
      '<span class="ef-stat"><span>기준 거래일</span><b>' + prettyDate(m.baseDate) + '</b></span>' +
      '<span class="ef-stat"><span>S&P500 20일</span><b class="' + dirClass(m.spx20) + '">' +
        pct(m.spx20) + '</b></span>' +
      '<span class="ef-stat"><span>대상 ETF</span><b>' + m.count + '</b></span>';
  }

  /* --- 바구니: 분산이 되는가 --------------------------------------------- */

  function loadCorr() {
    if (state.corr) { return Promise.resolve(state.corr); }
    if (!state.corrPromise) {
      state.corrPromise = fetchJson('corr').then(function (d) { state.corr = d; return d; })
        .catch(function () { state.corr = { pairs: {} }; return state.corr; });
    }
    return state.corrPromise;
  }

  function corrOf(a, c) {
    if (!state.corr || !state.corr.pairs) { return null; }
    var p = state.corr.pairs;
    var v = p[a + ':' + c];
    if (v === undefined) { v = p[c + ':' + a]; }
    return v === undefined ? null : v;
  }

  /* 분산은 개수가 아니라 상관이 정한다.

     각 종목의 보유기간 변동 금액을 a_i = 투입금액 × 기대변동폭 이라 하면,
     포트폴리오 변동 금액은 sqrt(ΣΣ a_i a_j ρ_ij) 다. 상관이 1 이면 그냥 합이고
     (분산 효과 0), 0 이면 제곱합의 제곱근으로 줄어든다.

     상관을 모르는 쌍은 **1 로 본다** — 모르면 최악을 가정하는 게 맞다.
     분산 효과를 실제보다 크게 보여주면 안 되기 때문이다. */
  function basketStats() {
    var rows = state.basket
      .map(function (c) { return state.etfs.filter(function (r) { return r.code === c; })[0]; })
      .filter(Boolean);
    if (!rows.length) { return null; }
    var items = rows.map(function (r) {
      var s = sizing(r);
      var amount = s && s.shares ? s.amount : 0;
      return { row: r, amount: amount, swing: amount * (r.expectedSwing || 0) };
    });
    var naive = items.reduce(function (a, i) { return a + i.swing; }, 0);
    var varSum = 0;
    for (var i = 0; i < items.length; i++) {
      for (var j = 0; j < items.length; j++) {
        var rho = i === j ? 1 : corrOf(items[i].row.code, items[j].row.code);
        if (rho === null) { rho = 1; }
        varSum += items[i].swing * items[j].swing * rho;
      }
    }
    var combined = Math.sqrt(Math.max(0, varSum));
    var worst = null;
    for (var a = 0; a < rows.length; a++) {
      for (var c = a + 1; c < rows.length; c++) {
        var v = corrOf(rows[a].code, rows[c].code);
        if (v !== null && (!worst || v > worst.v)) {
          worst = { v: v, a: rows[a], b: rows[c] };
        }
      }
    }
    return {
      rows: rows, items: items,
      amount: items.reduce(function (a, i) { return a + i.amount; }, 0),
      naive: naive, combined: combined,
      benefit: naive > 0 ? 1 - combined / naive : 0,
      allStopped: state.risk * rows.length,
      worst: worst,
      unknown: rows.length > 1 && worst === null
    };
  }

  function renderBasket() {
    var host = $('ef-basket');
    if (!host) { return; }
    var s = basketStats();
    if (!s) {
      host.innerHTML = '<p class="ef-note">카드의 <strong>담기</strong>를 눌러 3~5개를 모으면 ' +
        '<strong>정말 분산이 되는지</strong> 계산해 드립니다. 반도체 ETF 세 개는 이름만 셋이지 ' +
        '사실상 한 베팅입니다.</p>';
      return;
    }
    var names = s.rows.map(function (r) {
      return '<span class="ef-tag">' + escapeHtml(r.name) +
        ' <button type="button" class="ef-basket-x" data-basket="' + escapeHtml(r.code) +
        '" aria-label="빼기">✕</button></span>';
    }).join('');
    var warn = '';
    if (s.worst && s.worst.v >= 0.85) {
      warn = '<p class="ef-record-note"><strong>' + escapeHtml(s.worst.a.name) + '</strong> 과 ' +
        '<strong>' + escapeHtml(s.worst.b.name) + '</strong> 의 상관이 ' + s.worst.v.toFixed(2) +
        ' 입니다 — 사실상 같은 베팅이라 나눠 담은 뜻이 없습니다.</p>';
    } else if (s.unknown) {
      warn = '<p class="ef-record-note">상관을 모르는 조합이 있어 <strong>분산 효과를 0으로</strong> ' +
        '잡았습니다. 후보군(거래대금 상위·추세 등급) 밖의 ETF 는 상관을 계산해 두지 않습니다.</p>';
    }
    host.innerHTML =
      '<div class="ef-basket-names">' + names + '</div>' +
      '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
      '<th>담은 것</th><th>값</th></tr></thead><tbody>' +
      '<tr><td>합산 투입</td><td><b>' + Math.round(s.amount).toLocaleString() + '원</b></td></tr>' +
      '<tr><td>다 손절되면</td><td class="ef-down"><b>−' +
        Math.round(s.allStopped).toLocaleString() + '원</b> (' + s.rows.length + '× 위험금액)</td></tr>' +
      '<tr><td>' + swingWeeks() + '주 기대 변동폭</td><td><b>±' +
        Math.round(s.combined).toLocaleString() + '원</b></td></tr>' +
      '<tr><td>따로 담았다면</td><td>±' + Math.round(s.naive).toLocaleString() + '원</td></tr>' +
      '<tr><td>분산 효과</td><td><b>' + (s.benefit * 100).toFixed(0) + '% 감소</b>' +
        (s.worst ? ' · 가장 닮은 쌍 상관 ' + s.worst.v.toFixed(2) : '') + '</td></tr>' +
      '</tbody></table></div>' + warn +
      '<p class="ef-note">상관이 1 이면 합산과 같아 분산 효과가 0 이고, 낮을수록 줄어듭니다. ' +
      '<strong>모르는 쌍은 상관 1 로 봅니다</strong> — 모르면 최악을 가정해야 분산 효과를 ' +
      '실제보다 크게 보여주는 일이 없습니다. ' +
      '"다 손절되면" 은 담은 것이 동시에 무너지는 경우라, 이 금액이 감당 가능한지가 먼저입니다.</p>';
  }

  function toggleBasket(code) {
    var i = state.basket.indexOf(code);
    if (i >= 0) { state.basket.splice(i, 1); } else {
      if (state.basket.length >= 8) { return; }
      state.basket.push(code);
    }
    try { window.localStorage.setItem('ef-basket', JSON.stringify(state.basket)); } catch (e) { /* 무시 */ }
    loadCorr().then(function () { renderBasket(); renderEtfs(); renderPick(); });
  }

  /* --- 거르기와 정렬 ------------------------------------------------------ */

  var GRADE_ORDER = ['눌림매수', '추세진행', '과열주의', '약세', '유동성부족', '금리형', '판정불가'];

  function gradeRank(g) {
    var i = GRADE_ORDER.indexOf(g);
    return i < 0 ? 99 : i;
  }

  function sortRows(rows, key) {
    var copy = rows.slice();
    if (key === 'grade') {
      copy.sort(function (a, b) {
        return gradeRank(a.grade) - gradeRank(b.grade) || (b.riskAdj || b.rSwing || -9) - (a.riskAdj || a.rSwing || -9);
      });
    } else if (key === 'riskAdj') {
      // 위험 한 단위당 얼마를 벌었나. 변동성이 다른 물건을 같은 줄에 세우려면
      // 수익률만으로는 안 된다.
      copy.sort(function (a, b) { return (b.riskAdj === null ? -9 : b.riskAdj) - (a.riskAdj === null ? -9 : a.riskAdj); });
    } else if (key === 'stopProb') {
      // 손절이 덜 걸릴 자리부터. 낮을수록 좋으므로 오름차순이다.
      copy.sort(function (a, b) { return (a.stopProb === null ? 9 : a.stopProb) - (b.stopProb === null ? 9 : b.stopProb); });
    } else if (key === 'breadth') {
      copy.sort(function (a, b) { return (b.breadth || -1) - (a.breadth || -1); });
    } else if (key === 'overhead') {
      // 위에 물린 물량이 적은 쪽부터. 검증에서 이 구간의 승률이 가장 높았다.
      copy.sort(function (a, b) { return (a.overhead === null ? 9 : a.overhead) - (b.overhead === null ? 9 : b.overhead); });
    } else {
      copy.sort(function (a, b) { return (b[key] === null ? -Infinity : b[key]) - (a[key] === null ? -Infinity : a[key]); });
    }
    return copy;
  }

  function filteredEtfs() {
    var q = $('ef-q').value.trim().toLowerCase();
    var grade = $('ef-grade').value;
    var tab = $('ef-tabcode').value;
    var lev = $('ef-lev').value;
    var liq = parseFloat($('ef-liq').value) || 0;
    var primaryOnly = $('ef-primary').checked;

    return sortRows(state.etfs.filter(function (r) {
      if (primaryOnly && !r.primary) { return false; }
      if (q && r.name.toLowerCase().indexOf(q) < 0 && r.code.indexOf(q) < 0) { return false; }
      if (grade && r.grade !== grade) { return false; }
      if (tab && String(r.tab) !== tab) { return false; }
      if (lev === '1' && r.lev !== 1) { return false; }
      if (lev === 'lev' && r.lev !== 2) { return false; }
      if (lev === 'inv' && r.lev >= 0) { return false; }
      if (liq && (r.turnover === null || r.turnover < liq)) { return false; }
      return true;
    }), $('ef-sort').value);
  }

  function filteredGroups(type) {
    var q = $('ef-gq').value.trim().toLowerCase();
    var grade = $('ef-ggrade').value;
    var buyableOnly = $('ef-gbuyable').checked;
    return sortRows(state.groups.filter(function (r) {
      if (r.type !== type) { return false; }
      if (q && r.name.toLowerCase().indexOf(q) < 0) { return false; }
      if (grade && r.grade !== grade) { return false; }
      if (buyableOnly && !r.etfCount) { return false; }
      return true;
    }), $('ef-gsort').value);
  }

  /* --- 그리기 ------------------------------------------------------------ */

  function renderEtfs() {
    var rows = filteredEtfs();
    var shown = rows.slice(0, state.etfShown);
    $('ef-list').innerHTML = shown.map(etfCard).join('');
    $('ef-count').textContent = rows.length
      ? 'ETF ' + rows.length + '개 중 ' + shown.length + '개 표시'
      : '조건에 맞는 ETF 가 없습니다.';
    $('ef-more').hidden = shown.length >= rows.length;
  }

  function renderGroups() {
    [['theme', 'ef-theme-table', 'ef-tcount', 'ef-tmore', 'themeShown'],
     ['upjong', 'ef-upjong-table', 'ef-ucount', 'ef-umore', 'upjongShown']
    ].forEach(function (cfg) {
      var rows = filteredGroups(cfg[0]);
      var shown = Math.min(state[cfg[4]], rows.length);
      $(cfg[1]).innerHTML = groupTable(rows, shown);
      $(cfg[2]).textContent = rows.length ? shown + ' / ' + rows.length + '개' : '0개';
      $(cfg[3]).hidden = shown >= rows.length;
    });
  }

  /* 성적표. 규칙을 자랑하는 자리가 아니라 규칙이 얼마나 못 미더운지 보여주는
     자리다. 나쁜 칸은 색으로 드러낸다. */
  function renderBacktest() {
    var host = $('ef-backtest');
    if (!host) { return; }
    var bt = state.backtest;
    if (!bt) {
      host.innerHTML = '<p class="ef-note">검증 결과를 아직 만들지 못했습니다.</p>';
      return;
    }
    function table(block, label) {
      var base = block.baseline;
      var head = '<tr><th>등급</th><th>표본</th><th>승률 (95% 구간)</th><th>중위 ' + swingWeeks() + '주</th>' +
        '<th>시장대비</th><th>비용 차감 후</th><th>손절 걸림</th><th>손절 적용</th></tr>';
      var rows = Object.keys(block.byGrade).map(function (g) {
        var s = block.byGrade[g];
        if (s.thin) {
          return '<tr><td>' + escapeHtml(g) + '</td><td>' + s.n +
            '</td><td colspan="6">표본이 적어 숫자를 내지 않습니다</td></tr>';
        }
        var realCls = s.medianRealized < 0 ? ' class="ef-down"' : '';
        var exCls = s.medianExcess !== null && s.medianExcess < 0 ? ' class="ef-down"' : '';
        var netCls = s.netExcess !== null && s.netExcess < 0 ? ' class="ef-down"' : '';
        // 기준선과 승률 구간이 겹치면 '차이가 있다' 고 말할 수 없다.
        var overlaps = base && !base.thin && s.winLo <= base.winHi && s.winHi >= base.winLo;
        return '<tr><td>' + badge(g) + (overlaps ? '<span class="ef-tag">구별 안 됨</span>' : '') + '</td>' +
          '<td>' + s.n.toLocaleString() + '</td>' +
          '<td>' + (s.winRate * 100).toFixed(1) + '% <i>(' +
            (s.winLo * 100).toFixed(1) + '~' + (s.winHi * 100).toFixed(1) + ')</i></td>' +
          '<td>' + pct(s.medianFwd, 2) + '</td>' +
          '<td' + exCls + '>' + (s.medianExcess === null ? '—' : pct(s.medianExcess, 2)) + '</td>' +
          '<td' + netCls + '>' + (s.netExcess === null ? '—' : pct(s.netExcess, 2)) + '</td>' +
          '<td>' + (s.stopHitRate * 100).toFixed(1) + '%</td>' +
          '<td' + realCls + '><b>' + pct(s.medianRealized, 2) + '</b></td></tr>';
      }).join('');
      var baseRow = base && !base.thin
        ? '<tr class="ef-baseline"><td>기준선(전체)</td><td>' + base.n.toLocaleString() + '</td><td>' +
          (base.winRate * 100).toFixed(1) + '% <i>(' + (base.winLo * 100).toFixed(1) + '~' +
          (base.winHi * 100).toFixed(1) + ')</i></td><td>' + pct(base.medianFwd, 2) +
          '</td><td>—</td><td>—</td><td>' + (base.stopHitRate * 100).toFixed(1) + '%</td><td>—</td></tr>'
        : '';
      return '<h3 class="ef-plan-title">' + label + '</h3>' +
        '<div class="ef-tablewrap"><table class="ef-table"><thead>' + head +
        '</thead><tbody>' + baseRow + rows + '</tbody></table></div>';
    }

      /* 누적 곡선. 이 화면에서 가장 답하기 어려운 질문에 답하는 자리다 —
       '이걸 계속 하는 게 그냥 지수를 사놓는 것보다 나은가'. 답이 '아니오' 여도
       그대로 그린다. */
    function curveBlock(c) {
      if (!c || !c.strategy) { return ''; }
      var lines = [
        { key: 'strategyTimed', color: '#0b7a4b', label: '전략 + 손절 + 국면필터' },
        { key: 'strategyStop', color: '#c0392b', label: '전략 + 손절' },
        { key: 'benchmark', color: '#888', label: '그냥 지수 보유' }
      ];
      var all = lines.reduce(function (acc, l) { return acc.concat(c[l.key] || []); }, []);
      var min = Math.min.apply(null, all), max = Math.max.apply(null, all);
      var span = (max - min) || 1;
      var n = c.strategy.length;
      var paths = lines.map(function (l) {
        var pts = (c[l.key] || []).map(function (v, i) {
          return (i / (n - 1) * 100).toFixed(2) + ',' + (100 - (v - min) / span * 100).toFixed(2);
        }).join(' ');
        return '<polyline fill="none" stroke="' + l.color + '" stroke-width="1.6" ' +
          'vector-effect="non-scaling-stroke" points="' + pts + '"></polyline>';
      }).join('');
      var legend = lines.map(function (l) {
        return '<span class="ef-legend"><i style="background:' + l.color + '"></i>' + l.label + '</span>';
      }).join('');

      function row(label, key, annualKey, mddKey) {
        var last = c[key][c[key].length - 1];
        var cum = last / 100 - 1;
        return '<tr><td>' + label + '</td>' +
          '<td class="' + dirClass(cum) + '">' + pct(cum) + '</td>' +
          '<td class="' + dirClass(c[annualKey]) + '">' + pct(c[annualKey]) + '</td>' +
          '<td>' + (mddKey && c[mddKey] !== undefined ? pct(c[mddKey]) : '—') + '</td></tr>';
      }
      return '<h3 class="ef-plan-title">그냥 지수를 사놓는 것보다 나은가</h3>' +
        '<p class="ef-note">매 회전마다 <strong>추세진행 중 위험조정 상위 ' + c.topN +
        '개를 같은 금액으로</strong> 사서 보유하는 전략입니다. 회전할 때마다 비용을 뺐습니다. ' +
        c.rounds + '회 회전 · ' + c.years + '년.</p>' +
        '<div class="ef-curve"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' +
        paths + '</svg><div class="ef-legendrow">' + legend + '</div></div>' +
        '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
        '<th>방식</th><th>누적</th><th>연 환산</th><th>최대낙폭</th></tr></thead><tbody>' +
        row('전략 + 손절 + 국면필터', 'strategyTimed', 'annualStrategyTimed', 'mddStrategyTimed') +
        row('전략 + 손절', 'strategyStop', 'annualStrategyStop', 'mddStrategy') +
        row('전략 (손절 없이)', 'strategy', 'annualStrategy', null) +
        '<tr class="ef-baseline">' +
        '<td>그냥 지수 보유</td>' +
        '<td>' + pct(c.benchmark[c.benchmark.length - 1] / 100 - 1) + '</td>' +
        '<td>' + pct(c.annualBenchmark) + '</td>' +
        '<td>' + pct(c.mddBenchmark) + '</td></tr>' +
        '</tbody></table></div>' +
        '<p class="ef-note"><strong>수익만 보면 지수 보유가 낫습니다.</strong> ' +
        '국면 필터를 켜면 최대낙폭이 ' + pct(c.mddStrategy) + ' → ' + pct(c.mddStrategyTimed) +
        ' 로 줄지만 수익도 같이 줄어듭니다(역풍이라 쉰 회차 ' + c.restRounds + '/' + c.rounds + '). ' +
        '이 화면의 값어치는 <strong>수익을 올려주는 것보다 나쁜 걸 안 사게 하고 손실 크기를 ' +
        '묶어주는 쪽</strong>에 있다고 읽는 게 맞습니다. ' +
        '회전이 ' + c.rounds + '회뿐이라 이 차이도 확정적이지 않습니다.</p>';
    }

    function overheadTable(block) {
      if (!block || !Object.keys(block).length) { return ''; }
      var rows = Object.keys(block).map(function (band) {
        var s = block[band];
        if (s.thin) {
          return '<tr><td>' + band + '</td><td>' + s.n + '</td><td colspan="3">표본 부족</td></tr>';
        }
        var exCls = s.medianExcess !== null && s.medianExcess < 0 ? ' class="ef-down"' : '';
        return '<tr><td>' + band + '</td><td>' + s.n.toLocaleString() + '</td>' +
          '<td>' + (s.winRate * 100).toFixed(1) + '% <i>(' +
            (s.winLo * 100).toFixed(1) + '~' + (s.winHi * 100).toFixed(1) + ')</i></td>' +
          '<td>' + pct(s.medianFwd, 2) + '</td>' +
          '<td' + exCls + '>' + (s.medianExcess === null ? '—' : pct(s.medianExcess, 2)) + '</td></tr>';
      }).join('');
      return '<h3 class="ef-plan-title">위에 물린 물량으로 나눠 보면 (전체 기간)</h3>' +
        '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
        '<th>위에 물린 물량</th><th>표본</th><th>승률 (95% 구간)</th><th>중위 ' + swingWeeks() + '주</th><th>시장대비</th>' +
        '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
        '<p class="ef-note">등급보다 이쪽이 더 잘 갈립니다. 물린 물량이 적을수록 승률이 높고, ' +
        '구간끼리 신뢰구간이 겹치지 않습니다. ' +
        '<strong>다만 이 결과로 등급 규칙을 바꾸지 않았습니다</strong> — 성적을 보고 기준을 ' +
        '되맞추면 곡선 맞추기가 됩니다. 숫자를 보여주고 정렬·필터를 드리는 데까지만 합니다.</p>';
    }

    host.innerHTML =
      '<p class="ef-note"><strong>' + prettyDate(bt.from) + ' ~ ' + prettyDate(bt.to) + '</strong> · ' +
      'ETF ' + bt.etfs.toLocaleString() + '개 · 평가 ' + bt.evalDates + '회 · ' +
      '관측 ' + bt.trials.toLocaleString() + '건. ' +
      '10 거래일마다 한 번씩만 평가해 관측이 겹치지 않게 했습니다. 매일 갱신합니다. ' +
      '승률 괄호는 95% 신뢰구간이고, 기준선과 구간이 겹치면 <b>구별 안 됨</b>을 붙였습니다 — ' +
      '표본이 적어 차이를 주장할 수 없다는 뜻입니다. ' +
      '비용은 왕복 ' + (bt.cost * 100).toFixed(2) + '%p 로 가정했습니다.</p>' +
      curveBlock(bt.curve) +
      table(bt.all, '전체 기간') +
      table(bt.recent, '최근 1년') +
      overheadTable(bt.all.byOverhead) +
      '<h3 class="ef-plan-title">이 숫자가 말하지 못하는 것</h3>' +
      '<ul class="ef-why">' + bt.limits.map(function (l) {
        return '<li>' + escapeHtml(l) + '</li>';
      }).join('') + '</ul>';
  }

  function renderMarket() {
    var m = state.meta;
    var warn = m.intraday
      ? '<span class="ef-stat" style="flex:1 1 100%;background:#fff8e1;border-color:#f0e0a8;color:#7a5b00">' +
        '<b style="font-size:1em">장중 시세로 계산됨</b>' +
        '<span>장 마감 전에 받은 값이라 마지막 종가가 확정값이 아닙니다. 18:30 갱신 뒤 값이 바뀝니다.</span></span>'
      : '';
    /* 시장 국면을 맨 앞에 둔다. 보유가 짧을수록 개별 테마보다 시장
       방향이 결과를 더 많이 정한다. 테마를 고르기 전에 볼 것. */
    var reg = m.regime || {};
    var regCls = reg.label === '역풍' ? 'background:#fdf0f0;border-color:#f0d0d0;color:#8a2b2b'
      : (reg.label === '순풍' ? 'background:#e9f5ee;border-color:#c3e2d0;color:#16704a'
        : 'background:#fff8e1;border-color:#f0e0a8;color:#7a5b00');
    var regime = '<span class="ef-stat" style="flex:1 1 100%;' + regCls + '">' +
      '<b style="font-size:1em">시장 국면 · ' + escapeHtml(reg.label || '—') + '</b>' +
      '<span>' + escapeHtml(reg.note || '') + '</span></span>';

    $('ef-market').innerHTML = warn + regime +
      '<span class="ef-stat"><span>기준 거래일</span><b>' + prettyDate(m.baseDate) + '</b></span>' +
      '<span class="ef-stat"><span>코스피 20일</span><b class="' + dirClass(m.kospi) + '">' + pct(m.kospi) + '</b></span>' +
      '<span class="ef-stat"><span>코스닥 20일</span><b class="' + dirClass(m.kosdaq) + '">' + pct(m.kosdaq) + '</b></span>' +
      '<span class="ef-stat"><span>대세 테마·업종</span><b>' + m.trendGroups + ' / ' + m.groupCount + '</b></span>';
  }

  /* 오늘의 후보. 없으면 없다고 말한다 — 빈 자리를 남기면 로딩 실패로 읽힌다. */
  function renderPick() {
    // 목록은 기본이 '테마별 대표 하나만' 인데 후보 카드가 그걸 안 따르면,
    // 맨 앞에 '같은 테마 중복' 딱지가 붙은 ETF 가 추천처럼 놓인다. 같은 규칙을 쓴다.
    var pool = state.etfs.filter(function (r) { return r.primary; });
    var pullback = pool.filter(function (r) { return r.grade === '눌림매수'; });
    var trend = pool.filter(function (r) { return r.grade === '추세진행'; });
    var html = '';

    if (pullback.length) {
      html += '<h3>눌림 매수 구간 ' + pullback.length + '개</h3>' +
        '<div class="ef-grid">' + sortRows(pullback, 'r20').slice(0, 3).map(etfCard).join('') + '</div>';
    } else {
      html += '<div class="ef-pick-empty"><strong>오늘은 눌림 매수 구간에 든 ETF 가 없습니다.</strong> ' +
        '곧게 오르던 것이 잠깐 쉬는 모양을 찾는 조건인데, 지금 장에는 그런 종목이 없습니다. ' +
        '조건을 느슨하게 해서 억지로 후보를 만들지 않습니다.</div>';
    }

    if (trend.length) {
      html += '<h3 style="margin-top:1em">추세 진행 상위</h3>' +
        '<div class="ef-grid">' + sortRows(trend, 'r20').slice(0, 3).map(etfCard).join('') + '</div>';
    }
    $('ef-pick').innerHTML = html;
  }

  /* --- 상세 패널 --------------------------------------------------------- */

  function loadDetails() {
    if (state.details) { return Promise.resolve(state.details); }
    if (!state.detailsPromise) {
      state.detailsPromise = fetchJson('details').then(function (d) {
        state.details = d;
        return d;
      });
    }
    return state.detailsPromise;
  }

  /* '이 테마를 살 수 있는 ETF'. 시가총액 상위 5 와 거래대금 상위 5 를 따로
     놓는다. 큰 게 곧 잘 팔리는 게 아니라서다 — 시총 1조짜리가 하루 3억밖에
     안 거래되면 보유 기간 안에 못 빠져나온다. */
  function groupEtfTable(list, unitLabel, unit) {
    var body = list.map(function (e) {
      var lev = levLabel(e.lev);
      return '<tr><td>' + escapeHtml(e.name) +
        (lev ? ' <span class="ef-tag is-warn">' + lev + '</span>' : '') +
        (e.partial ? ' <span class="ef-tag">부분 노출</span>' : '') + '</td>' +
        '<td>' + unit(e) + '</td>' +
        '<td>' + e.weight.toFixed(0) + '%</td>' +
        '<td class="' + dirClass(e.rSwing) + '">' + pct(e.rSwing) + '</td>' +
        '<td>' + badge(e.grade) + '</td></tr>';
    }).join('');
    return '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
      '<th>ETF</th><th>' + unitLabel + '</th><th>노출</th><th>' + swingWeeks() + '주</th><th>등급</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table></div>';
  }

  function groupEtfSection(key) {
    var d = state.details && state.details.groupEtf ? state.details.groupEtf[key] : null;
    if (!d) {
      return '<h3 class="ef-plan-title">이 테마를 살 수 있는 ETF</h3>' +
        '<p class="ef-note">편입비중 10% 이상으로 이 테마에 노출된 ETF 가 없습니다. ' +
        '개별 종목으로만 접근할 수 있는 테마입니다.</p>';
    }
    return '<h3 class="ef-plan-title">이 테마를 살 수 있는 ETF · 시가총액 상위</h3>' +
      groupEtfTable(d.byCap, '시가총액', function (e) {
        return e.cap ? e.cap.toLocaleString() + '억' : '—';
      }) +
      '<h3 class="ef-plan-title">거래대금 상위</h3>' +
      groupEtfTable(d.byTurnover, '20일 중앙', function (e) {
        return moneyShort(e.turnover);
      }) +
      '<p class="ef-note">노출 ' + d.total + '개 중 상위 5개씩입니다. ' +
      '<strong>노출</strong>은 그 ETF 자산에서 이 테마가 차지하는 비중이라, ' +
      '낮으면 테마가 올라도 ETF 는 덜 움직입니다. ' +
      '국내 시장지수 ETF 는 뺐습니다 — 코스피200 은 지금 반도체가 60% 라 큰 테마마다 ' +
      '1위로 올라오지만, 지수를 사는 건 테마를 사는 게 아니기 때문입니다.</p>';
  }

  function holdingsTable(rows, weightHeader) {
    if (!rows || !rows.length) { return '<p class="ef-note">구성종목 정보가 없습니다.</p>'; }
    var head = '<tr><th>종목</th>' + (weightHeader ? '<th>비중</th>' : '') + '<th>20일</th></tr>';
    var body = rows.map(function (r) {
      return '<tr><td>' + escapeHtml(r.name) + '</td>' +
        (weightHeader ? '<td>' + (r.weight === undefined ? '—' : r.weight.toFixed(1) + '%') + '</td>' : '') +
        '<td class="' + dirClass(r.r20) + '">' + pct(r.r20) + '</td></tr>';
    }).join('');
    return '<div class="ef-tablewrap"><table class="ef-table"><thead>' + head +
      '</thead><tbody>' + body + '</tbody></table></div>';
  }

  /* 매매 계획. 등급만 보고는 주문을 못 낸다 — 어디서 자를지가 있어야 한다.
     목표가를 쓰지 않는 이유는 그건 예측이기 때문이다. 대신 '보통 이만큼
     흔들린다' 는 관측치를 놓고 손절폭과 견주게 한다. */
  function tradePlan(row, kind) {
    if (kind !== 'etf' || row.stop === null || row.stop === undefined) { return ''; }
    var rows = [
      ['손절 가격', row.stop.toLocaleString() + '원 <span class="ef-down">(' + pct(row.stopPct) + ')</span>',
        '<code>현재가 − max(2×ATR, 보유기간 기대 변동폭)</code>. <strong>보유기간 노이즈의 바깥</strong>에 둡니다 — ' +
        '안쪽에 두면 논지가 깨져서가 아니라 평범한 출렁임에 걸립니다'],
      [swingWeeks() + '주 기대 변동폭', '±' + (row.expectedSwing * 100).toFixed(1) + '%',
        '관측된 변동성의 1σ를 보유 기간으로 환산한 값입니다. <strong>방향을 맞히는 값이 아니라</strong> 크기 감각입니다'],
      ['손절 걸릴 확률', (row.stopProb === null ? '—' : (row.stopProb * 100).toFixed(0) + '%'),
        '방향성 없는 움직임을 가정했을 때 보유 기간 안에 손절선을 건드릴 확률입니다(<code>2Φ(−손절폭/σ)</code>). ' +
        '아래 검증 결과의 실제 손절 적중률과 견줘 볼 수 있습니다'],
      ['하루 변동폭(ATR)', (row.atr === null ? '—' : row.atr.toLocaleString() + '원'),
        '갭까지 반영한 하루 평균 등락폭입니다']
    ];
    var size = sizing(row);
    if (size) {
      rows.push(['살 수량', size.shares
          ? size.shares.toLocaleString() + '주 · ' + Math.round(size.amount).toLocaleString() + '원'
          : '담을 수 없음',
        size.shares
          ? '잃어도 되는 금액 ' + state.risk.toLocaleString() + '원을 손절폭 ' +
            pct(row.stopPct) + ' 로 나눈 값입니다. 이렇게 잡아야 손절폭이 다른 종목에 ' +
            '<strong>같은 금액이 아니라 같은 위험</strong>을 걸게 됩니다.' +
            (size.capped
              ? ' <strong class="ef-down">위험 기준으로는 ' + size.byRisk.toLocaleString() +
                '주지만 하루 거래대금의 1%에서 끊었습니다</strong> — 그 이상은 넣고 빼는 데 ' +
                '값이 밀립니다'
              : '')
          : size.note]);
    }
    if (row.premium !== null && row.premium !== undefined) {
      rows.push(['괴리율', pct(row.premium, 2),
        'NAV 대비 시장가입니다. 양(+)이면 <strong>사는 순간 그만큼 얹어 주는 것</strong>이라 작지 않습니다']);
    }
    return '<h3 class="ef-plan-title">' + swingWeeks() + '주 스윙 계획</h3><dl class="ef-plan">' +
      rows.map(function (r) {
        return '<dt>' + r[0] + '</dt><dd><b>' + r[1] + '</b><span>' + r[2] + '</span></dd>';
      }).join('') + '</dl>';
  }

  /* 이 등급이 과거에 실제로 어땠는지. 등급 옆에 성적표를 붙여 두면 배지를
     근거로 착각할 일이 줄어든다. 좋은 숫자든 나쁜 숫자든 그대로 쓴다. */
  function gradeRecord(grade) {
    if (!state.backtest) { return ''; }
    var s = (state.backtest.all.byGrade || {})[grade];
    if (!s || s.thin) { return ''; }
    var base = state.backtest.all.baseline;
    var overlaps = base && !base.thin && s.winLo <= base.winHi && s.winHi >= base.winLo;
    var netBad = s.netExcess !== null && s.netExcess <= 0;
    var warn = netBad || overlaps;
    var note = '';
    if (overlaps) {
      note = '표본이 적어 <strong>기준선과 구별되지 않습니다</strong>. 이 등급이 낫다고 말할 근거가 없습니다.';
    } else if (netBad) {
      note = '거래 비용을 빼면 시장 대비 우위가 <strong>남지 않습니다</strong>. 절대 수익은 플러스지만 ' +
        '같은 돈으로 지수를 샀을 때와 견주면 이깁니다 라고 말하기 어렵습니다.';
    }
    return '<div class="ef-record' + (warn ? ' is-warn' : '') + '">' +
      '<b>이 등급의 과거 ' + swingWeeks() + '주 성적</b>' +
      '<span>표본 ' + s.n.toLocaleString() + '건 · 승률 ' + (s.winRate * 100).toFixed(1) +
      '% (95% 구간 ' + (s.winLo * 100).toFixed(1) + '~' + (s.winHi * 100).toFixed(1) + ')' +
      ' · 중위 ' + pct(s.medianFwd, 2) + '</span>' +
      (s.medianExcess === null ? '' :
        '<span>시장 대비 ' + pct(s.medianExcess, 2) + ' → 거래 비용 차감 후 <b>' +
        pct(s.netExcess, 2) + '</b></span>') +
      '<span>손절이 2주 안에 걸린 비율 <b>' + (s.stopHitRate * 100).toFixed(1) + '%</b>' +
      ' → 손절을 지켰을 때 중위 <b>' + pct(s.medianRealized, 2) + '</b></span>' +
      (note ? '<span class="ef-record-note">' + note + '</span>' : '') +
      '</div>';
  }

  /* 매물대 막대그래프. 가격이 세로축이고 위가 비싼 쪽이다. 현재가에 선을 긋고
     그 위쪽 막대를 다른 색으로 칠하면 '얼마나 물려 있나' 가 한눈에 보인다. */
  function profileChart(row) {
    var prof = state.details && state.details.profile ? state.details.profile[row.code] : null;
    if (!prof || !prof.vol.length || row.price === null) { return ''; }
    var n = prof.vol.length;
    var span = prof.hi - prof.lo;
    if (span <= 0) { return ''; }
    var rowH = 100 / n;
    var bars = prof.vol.map(function (v, i) {
      // 배열 0 번이 가장 싼 칸이다. 위가 비싸 보이게 뒤집어 그린다.
      var y = 100 - (i + 1) * rowH;
      var center = prof.lo + span * (i + 0.5) / n;
      var above = center > row.price;
      return '<rect x="0" y="' + y.toFixed(2) + '" width="' + Math.max(v, 0.6) +
        '" height="' + (rowH * 0.82).toFixed(2) + '" fill="' +
        (above ? '#e3a8a8' : '#a8c4e3') + '"></rect>';
    }).join('');
    var priceY = 100 - ((row.price - prof.lo) / span) * 100;
    priceY = Math.max(0, Math.min(100, priceY));
    return '<div class="ef-profile">' +
      '<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' + bars +
      '<line x1="0" y1="' + priceY.toFixed(2) + '" x2="100" y2="' + priceY.toFixed(2) +
      '" stroke="#222" stroke-width="0.6" stroke-dasharray="2 1.5"></line></svg>' +
      '<div class="ef-profile-axis">' +
        '<span>' + Math.round(prof.hi).toLocaleString() + '</span>' +
        '<span>' + Math.round(prof.lo).toLocaleString() + '</span>' +
      '</div>' +
      '<p class="ef-note">붉은 칸이 <strong>현재가보다 비싼 값에 거래된 물량</strong>입니다. ' +
      '점선이 현재가입니다. 일봉만 있어 하루 거래량을 그날 고가~저가에 균등하게 ' +
      '나눠 담은 근사치이고, 최근 120 거래일을 봅니다.</p>' +
      '</div>';
  }

  function supplySection(row) {
    if (row.price === null || row.price === undefined) { return ''; }
    var rows = [
      ['현재가', Math.round(row.price).toLocaleString() + '원', ''],
    ];
    if (row.high52 !== null) {
      var full = (row.high52Days || 0) >= 250;
      rows.push([full ? '52주 고점' : '상장 후 최고',
        Math.round(row.high52).toLocaleString() + '원 ' +
        '<span class="' + dirClass(row.fromHigh52) + '">(' + pct(row.fromHigh52) + ')</span>',
        full ? '고점에서 얼마나 내려와 있는지입니다'
             : '거래일이 ' + row.high52Days + '일뿐이라 아직 1년치가 아닙니다. ' +
               '<strong>52주 고점이라고 부를 수 없습니다</strong>']);
    }
    if (row.high20 !== null) {
      rows.push(['20일 고점', Math.round(row.high20).toLocaleString() + '원 ' +
        '<span class="' + dirClass(row.dd) + '">(' + pct(row.dd) + ')</span>',
        '최근 흐름 안에서의 위치입니다']);
    }
    if (row.poc !== null) {
      rows.push(['매물이 가장 두꺼운 값', Math.round(row.poc).toLocaleString() + '원',
        '지난 120 거래일 중 가장 많이 거래된 가격대입니다. <strong>평균 매입가에 가장 가까운 자리</strong>라, ' +
        '여기를 위로 넘기면 물린 사람이 크게 줄고 아래로 깨지면 늘어납니다']);
    }
    if (row.overhead !== null) {
      rows.push(['위에 물린 물량', Math.round(row.overhead * 100) + '%',
        '지난 120 거래일 거래의 이만큼이 <strong>지금보다 비싼 값</strong>에 이뤄졌습니다. ' +
        '오를 때마다 본전 찾는 매도가 그만큼 기다립니다']);
    }
    if (row.wall !== null) {
      rows.push(['첫 저항 매물대', Math.round(row.wall).toLocaleString() + '원 ' +
        '<span>(' + pct(row.wallGap) + ')</span>',
        '현재가 위에서 매물이 가장 두꺼운 가격대입니다. 2주 목표가 여기를 넘어야 한다면 쉽지 않습니다']);
    }
    return '<h3 class="ef-plan-title">고점 · 현재가 · 매물대</h3>' +
      profileChart(row) +
      '<dl class="ef-plan">' + rows.map(function (r) {
        return '<dt>' + r[0] + '</dt><dd><b>' + r[1] + '</b>' +
          (r[2] ? '<span>' + r[2] + '</span>' : '') + '</dd>';
      }).join('') + '</dl>';
  }

  /* 진입 기록. 화면이 숫자를 다 주니 옮겨 적기만 하면 되는데, 옮겨 적는 순간이
     귀찮으면 아무도 안 한다. 한 번 눌러 클립보드로 보낸다.

     진입 전에 근거와 손절을 적어 두는 것이 개인 투자자 성과를 가장 크게 바꾼다.
     지표를 하나 더 만드는 것보다 이쪽이 낫다. */
  function recordText(row) {
    var s = sizing(row);
    var m = state.meta || {};
    var lines = [
      '[' + prettyDate(m.baseDate) + '] ' + row.name + ' (' + row.code + ')',
      '등급: ' + row.grade + ' · ' + swingWeeks() + '주 ' + pct(row.rSwing) + ' · 20일 ' + pct(row.r20),
      '진입: ' + (row.price === null ? '—' : Math.round(row.price).toLocaleString() + '원'),
      '손절: ' + (row.stop === null ? '—' : Math.round(row.stop).toLocaleString() + '원 (' + pct(row.stopPct) + ')'),
      '수량: ' + (s && s.shares ? s.shares.toLocaleString() + '주 · ' + Math.round(s.amount).toLocaleString() + '원' : '—'),
      '위에 물린 물량: ' + (row.overhead === null ? '—' : Math.round(row.overhead * 100) + '%'),
      '대표 테마: ' + (row.groupName || '없음') +
        (row.breadth === null ? '' : ' (상승비율 ' + Math.round(row.breadth * 100) + '%)'),
      '근거: ' + row.reasons.join(' / '),
      '시장 국면: ' + ((m.regime && m.regime.label) || '—'),
      '청산 예정: ' + swingWeeks() + '주 뒤 또는 손절',
      '', '실제 청산일:            청산가:            손익:',
      '되돌아보기:'
    ];
    return lines.join('\n');
  }

  /* 미국 ETF 에만 붙는 주의. 국내와 다른 것을 매번 상기시킨다 —
     환율·세금·괴리율·대세 확인 불가. */
  function usNotes(row) {
    var m = state.us.meta;
    var items = [
      '<strong>원화 수익률 ' + pct(row.rSwingKrw) + '</strong> = 달러 ' + pct(row.rSwing) +
        ' × 환율 ' + pct(row.fxMove) + '. 기초자산이 그대로여도 환율만으로 움직입니다',
      '<strong>양도소득세 22%</strong>(연 250만원 공제)가 어떤 숫자에도 반영돼 있지 않습니다',
      'NAV 를 못 받아 <strong>괴리율을 알 수 없습니다</strong>. 유동성 낮으면 스프레드를 직접 확인하세요',
      '국내 테마와 연결이 없어 <strong>상승 비율(폭)을 확인할 수 없습니다</strong>',
      '기준일 ' + prettyDate(m.baseDate) + ' — 미국 종가는 한국 시간 다음 날 새벽에 확정됩니다'
    ];
    return '<div class="ef-record is-warn"><b>미국 ETF 라 다른 점</b>' +
      items.map(function (i) { return '<span>· ' + i + '</span>'; }).join('') + '</div>';
  }

  /* 발행사 이름 짐작. us_holdings.json 에 없는(=수집 대상이 아닌) ETF 라도
     '누가 운용하는지' 는 종목명 앞부분으로 알려줄 수 있다.

     단, 첫 낱말을 무조건 집으면 안 된다 — "United States Oil Fund LP" 가
     "United" 로 잘려 발행사인 척하게 된다. 아는 이름에만 맞히고, 모르면
     빈 문자열을 돌려 안내문에서 괄호를 통째로 뺀다. 틀린 이름을 대는 것보다
     아무 이름도 안 대는 게 낫다. */
  var KNOWN_ISSUERS = [
    'Global X', 'State Street', 'First Trust', 'JPMorgan', 'iShares', 'Shares',
    'ProShares', 'Direxion', 'Vanguard', 'Invesco', 'SPDR', 'ARK', 'VanEck',
    'WisdomTree', 'Schwab', 'Fidelity', 'Amplify', 'GraniteShares', 'Teucrium',
    'Sprott', 'KraneShares', 'Bitwise', 'Grayscale', 'Defiance', 'Roundhill',
    'YieldMax', 'Simplify', 'Tradr'
  ];
  function issuerGuess(name) {
    for (var i = 0; i < KNOWN_ISSUERS.length; i++) {
      if (name.indexOf(KNOWN_ISSUERS[i]) === 0) {
        return KNOWN_ISSUERS[i] === 'Shares' ? 'iShares' : KNOWN_ISSUERS[i];
      }
    }
    return '';
  }

  /* 미국 ETF 구성종목. 국내 표(holdingsTable)를 그대로 못 쓴다 — 국내는 20일
     수익률 칸이 있지만 미국 구성종목은 발행사 CSV/xlsx 에 그 값이 없다.
     종목명·티커·비중만 확실하다.

     레버리지 상품(SOXL 등)은 스왑으로 배수를 만든다 — swap 이 있으면 그
     비중을 표 아래 그대로 적는다. 표에 실린 건 "주식 몇 %" 뿐인데 그게
     펀드 전체라고 오해하면 안 되기 때문이다(72% 만 보고 "현금 많은
     펀드"로 착각하는 게 실제로 있었던 버그다).

     BIL·JNK 같은 채권형은 noTicker=true 로 온다 — 개별 채권에 주식 티커가
     없는 게 정상이라 티커 칸 자체를 빼고 보여준다.

     인버스 상품(SOXS·TZA·SQQQ 등)은 개별 종목을 아예 안 담는다 — rows 가
     비고 스왑이 음수로만 온다. 표를 빈 채로 그리면 "못 받았다"로 보이므로
     표를 아예 만들지 않고 설명 줄만 남긴다. */
  /* 파생 이름 목록을 짧게 줄인다. DBC 같은 원자재 펀드는 선물이 10종을 넘어
     한 줄이 화면을 다 먹는다 — 앞 셋만 두고 나머지는 개수로 적는다. */
  function shortNote(note) {
    var parts = note.split(' · ').filter(Boolean);
    if (parts.length <= 3) { return note; }
    return parts.slice(0, 3).join(' · ') + ' 외 ' + (parts.length - 3) + '건';
  }

  function usHoldingsTable(row) {
    var entry = state.usHoldings ? state.usHoldings[row.ticker] : null;
    /* 실물 신탁(GLD·SLV 등)은 금괴·은괴만 담는다. 받으려다 실패한 게 아니라
       구성종목이라는 개념 자체가 없는 상품이라, "받지 못했습니다" 로 적으면
       거짓말이 된다. 없는 것은 없다고 적는다. */
    if (entry && entry.physical) {
      return '<h3 class="ef-plan-title">구성종목</h3>' +
        '<p class="ef-note">' + escapeHtml(entry.physical) + '를 그대로 보관하는 실물 신탁이라 ' +
        '구성종목이 없습니다. 값은 ' + escapeHtml(entry.physical) + ' 시세를 그대로 따라갑니다.</p>';
    }
    var hasData = entry && (entry.rows.length || entry.cash || entry.swap || entry.other);
    if (!hasData) {
      var issuer = (entry && entry.issuer) || issuerGuess(row.name);
      return '<h3 class="ef-plan-title">구성종목</h3>' +
        '<p class="ef-note">구성종목을 받지 못했습니다' +
        (issuer ? ' (' + escapeHtml(issuer) + ')' : '') + '. 확인된 발행사' +
        '(Direxion·ARK·SPDR·iShares·ProShares·Vanguard·Global X·Invesco·First Trust·' +
        'KraneShares)가 아니거나, ' +
        '발행사가 그날 파일을 아직 안 올린 경우입니다.</p>';
    }
    var title = '<h3 class="ef-plan-title">구성종목 · ' + escapeHtml(entry.issuer || '') +
      (entry.asOf ? ' · ' + prettyDate(entry.asOf) + ' 기준' : '') + '</h3>';
    var countTail = entry.count > entry.rows.length
      ? ' · 상위 ' + entry.rows.length + '/' + entry.count + '종목만 놓았습니다' : '';

    if (entry.noTicker) {
      var bondBody = entry.rows.map(function (r) {
        return '<tr><td>' + escapeHtml(r.name) + '</td><td>' + r.weight.toFixed(2) + '%</td></tr>';
      }).join('');
      return title +
        '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
        '<th>종목명</th><th>비중</th></tr></thead><tbody>' + bondBody + '</tbody></table></div>' +
        '<p class="ef-note">채권형이라 개별 종목 티커가 없습니다' + countTail + '.</p>';
    }

    var body = entry.rows.map(function (r) {
      return '<tr><td>' + escapeHtml(r.name) + '</td><td>' + escapeHtml(r.ticker) + '</td>' +
        '<td>' + r.weight.toFixed(2) + '%</td></tr>';
    }).join('');
    var swap = entry.swap || 0;
    var cash = entry.cash || 0;
    var other = entry.other || 0;
    var note;
    if (swap !== 0) {
      /* 파생은 스왑만이 아니다 — UVXY 는 VIX 선물, TQQQ 는 지수스왑이다.
         부호는 방향이다: 인버스는 음수로 온다(SOXS −300%).

         배수는 row.lev 로 말하지 않는다. 그 값은 종목명에서 짐작한 것이라
         틀릴 때가 있다 — UVXY 는 이름에 'Ultra' 가 붙어 2배로 읽히지만
         2018년부터 1.5배이고, 실제 파생 노출도 150.0% 로 측정된다. 짐작한
         배수와 측정한 노출이 어긋나면 화면이 거짓말을 하게 되므로, 여기서는
         **측정된 수치만** 말한다. */
      var dir = swap < 0 ? '하락에 베팅하는 ' : '';
      note = '<p class="ef-note">파생(스왑·선물) ' + Math.abs(swap).toFixed(1) + '%' +
        (entry.swapNote ? ' · ' + escapeHtml(shortNote(entry.swapNote)) : '') +
        ' · 현금성 ' + cash.toFixed(1) + '%' +
        (other > 0 ? ' · 기타(펀드 자체 명목가치) ' + other.toFixed(1) + '%' : '') +
        ' — ' + dir + '배수를 주식이 아니라 파생으로 만듭니다' + countTail + '.</p>';
    } else if (cash > 0 || other > 0) {
      note = '<p class="ef-note">현금·기타 ' + (cash + other).toFixed(1) + '%' + countTail + '</p>';
    } else if (countTail) {
      note = '<p class="ef-note">' + countTail.replace(/^ · /, '') + '.</p>';
    } else {
      note = '';
    }
    /* 개별 종목이 하나도 없으면 표를 만들지 않는다 — 머리글만 있고 몸통이 빈
       표는 "데이터를 못 받았다"로 읽힌다. 인버스 상품이 여기로 온다. */
    var table = entry.rows.length
      ? '<div class="ef-tablewrap"><table class="ef-table"><thead><tr>' +
        '<th>종목명</th><th>티커</th><th>비중</th></tr></thead><tbody>' + body + '</tbody></table></div>'
      : '<p class="ef-note">개별 종목을 담지 않는 상품입니다.</p>';
    return title + table + note;
  }

  function openPanel(kind, code) {
    var panel = $('ef-panel');
    var title = $('ef-panel-title');
    var body = $('ef-panel-body');
    var row = kind === 'etf'
      ? state.etfs.filter(function (r) { return r.code === code; })[0]
      : (kind === 'us'
        ? (state.us ? state.us.rows.filter(function (r) { return r.code === code; })[0] : null)
        : state.groups.filter(function (r) { return r.key === code; })[0]);
    if (!row) { return; }

    title.innerHTML = (kind === 'us' ? escapeHtml(row.ticker) + ' · ' : '') +
      escapeHtml(row.name) + ' ' + badge(row.grade);
    // 미국 ETF 는 구성종목을 애초에 못 받은 채로 등급을 매긴다(build_us_etf.py 가
    // breadth_ok 를 항상 true 로 둔다) — 그래서 나온 '구성종목 다수가 상승' 은
    // 확인 안 된 근거다. 국내와 달리 실제로 아는 게 없으니 문구를 빼고 보여준다.
    var reasons = kind === 'us'
      ? row.reasons.filter(function (r) { return r.indexOf('구성종목') < 0; })
      : row.reasons;
    var head = '<ul class="ef-why">' +
      reasons.map(function (r) { return '<li>' + escapeHtml(r) + '</li>'; }).join('') +
      '</ul>' + (kind === 'us' ? usNotes(row) : gradeRecord(row.grade)) +
      tradePlan(row, kind === 'us' ? 'etf' : kind) +
      (kind === 'etf'
        ? '<button type="button" class="ef-copy" id="ef-copy" data-code="' +
          escapeHtml(row.code) + '">진입 기록 복사</button>'
        : '');
    body.innerHTML = head + '<p class="ef-note">구성종목을 불러오는 중…</p>';
    panel.hidden = false;
    document.body.style.overflow = 'hidden';

    if (kind === 'us') {
      body.innerHTML = head + supplySection(row) + usHoldingsTable(row);
      return;
    }
    loadDetails().then(function (d) {
      var rows = kind === 'etf' ? (d.etf[code] || []) : (d.group[code] || []);
      var extra = '';
      if (kind === 'etf') {
        extra = '<p class="ef-note">비중 상위 10종목입니다. 20일 수익률이 비어 있는 줄은 ' +
          '해외 종목이거나 현금·선물이라 국내 종목코드로 이어지지 않는 것입니다.</p>';
      } else {
        extra = '<p class="ef-note">구성종목 ' + rows.length + '개를 20일 수익률 순으로 놓았습니다. ' +
          '테마 수익률은 이 값들의 <strong>중위값</strong>이라 맨 위 한두 종목에 끌려가지 않습니다.</p>';
      }
      // 테마·업종은 'ETF 로 어떻게 사나' 가 먼저다. 구성종목은 근거일 뿐이다.
      var etfPart = kind === 'group' ? groupEtfSection(code) : supplySection(row);
      if (kind === 'us') { body.innerHTML = head + etfPart; return; }
      var holdTitle = kind === 'group'
        ? '<h3 class="ef-plan-title">구성종목</h3>' : '';
      body.innerHTML = head + etfPart + holdTitle + holdingsTable(rows, kind === 'etf') + extra;
    }).catch(function () {
      body.innerHTML += '<p class="ef-error">구성종목을 불러오지 못했습니다.</p>';
    });
  }

  function closePanel() {
    $('ef-panel').hidden = true;
    document.body.style.overflow = '';
  }

  /* --- 배선 -------------------------------------------------------------- */

  function fillSelect(select, values) {
    values.forEach(function (v) {
      var opt = document.createElement('option');
      opt.value = v.value;
      opt.textContent = v.label;
      select.appendChild(opt);
    });
  }

  function wire() {
    var riskInput = $('ef-risk');
    if (riskInput) {
      var saved = null;
      try { saved = window.localStorage.getItem('ef-risk'); } catch (e) { saved = null; }
      if (saved && parseInt(saved, 10) > 0) { state.risk = parseInt(saved, 10); }
      riskInput.value = state.risk;
      riskInput.addEventListener('input', function () {
        var v = parseInt(riskInput.value, 10);
        if (!v || v <= 0) { return; }
        state.risk = v;
        try { window.localStorage.setItem('ef-risk', String(v)); } catch (e) { /* 무시 */ }
        renderEtfs();
        renderPick();
        renderBasket();
        renderUs();
      });
    }

    var etfInputs = ['ef-q', 'ef-grade', 'ef-tabcode', 'ef-lev', 'ef-liq', 'ef-sort', 'ef-primary'];
    etfInputs.forEach(function (id) {
      $(id).addEventListener('input', function () { state.etfShown = PAGE; renderEtfs(); });
      $(id).addEventListener('change', function () { state.etfShown = PAGE; renderEtfs(); });
    });
    var groupInputs = ['ef-gq', 'ef-ggrade', 'ef-gsort', 'ef-gbuyable'];
    groupInputs.forEach(function (id) {
      function reset() {
        state.themeShown = GROUP_PAGE;
        state.upjongShown = GROUP_PAGE;
        renderGroups();
      }
      $(id).addEventListener('input', reset);
      $(id).addEventListener('change', reset);
    });

    $('ef-more').addEventListener('click', function () { state.etfShown += PAGE; renderEtfs(); });
    $('ef-tmore').addEventListener('click', function () { state.themeShown += GROUP_PAGE; renderGroups(); });
    $('ef-umore').addEventListener('click', function () { state.upjongShown += GROUP_PAGE; renderGroups(); });

    document.addEventListener('click', function (e) {
      var copy = e.target.closest ? e.target.closest('.ef-copy') : null;
      if (copy) {
        e.preventDefault();
        var target = state.etfs.filter(function (r) { return r.code === copy.dataset.code; })[0];
        if (target && navigator.clipboard) {
          navigator.clipboard.writeText(recordText(target)).then(function () {
            copy.textContent = '복사했습니다 ✓';
            copy.classList.add('is-done');
          });
        }
        return;
      }
      var basket = e.target.closest ? e.target.closest('[data-basket]') : null;
      if (basket) { e.preventDefault(); e.stopPropagation(); toggleBasket(basket.dataset.basket); return; }
      var hit = e.target.closest ? e.target.closest('.ef-card, .ef-row') : null;
      if (hit) { openPanel(hit.dataset.kind, hit.dataset.code); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') { return; }
      var row = e.target.closest ? e.target.closest('.ef-row') : null;
      if (row) { e.preventDefault(); openPanel(row.dataset.kind, row.dataset.code); }
    });
    $('ef-panel-close').addEventListener('click', closePanel);
    $('ef-panel').addEventListener('click', function (e) {
      if (e.target === $('ef-panel')) { closePanel(); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !$('ef-panel').hidden) { closePanel(); }
    });

    var TABS = [
      ['ef-tab-etf', 'ef-view-etf'],
      ['ef-tab-group', 'ef-view-group'],
      ['ef-tab-us', 'ef-view-us']
    ];
    function showTab(active) {
      TABS.forEach(function (pair) {
        var on = pair[0] === active;
        $(pair[0]).classList.toggle('is-on', on);
        $(pair[0]).setAttribute('aria-selected', String(on));
        $(pair[1]).hidden = !on;
      });
    }
    TABS.forEach(function (pair) {
      $(pair[0]).addEventListener('click', function () { showTab(pair[0]); });
    });

    var usInputs = ['ef-uq', 'ef-ugrade', 'ef-ulev', 'ef-uliq', 'ef-usort'];
    usInputs.forEach(function (id) {
      function reset() { state.usShown = PAGE; renderUs(); }
      $(id).addEventListener('input', reset);
      $(id).addEventListener('change', reset);
    });
    $('ef-umore2').addEventListener('click', function () { state.usShown += PAGE; renderUs(); });
  }

  Promise.all([
    fetchJson('meta'), fetchJson('etfs'), fetchJson('groups'),
    fetchJson('backtest').catch(function () { return null; }),
    fetchJson('us').catch(function () { return null; }),
    fetchJson('us_holdings').catch(function () { return null; })
  ])
    .then(function (res) {
      state.meta = res[0];
      state.etfs = res[1];
      state.groups = res[2];
      state.backtest = res[3];
      state.us = res[4];
      state.usHoldings = res[5] || {};

      var etfGrades = GRADE_ORDER.filter(function (g) {
        return state.etfs.some(function (r) { return r.grade === g; });
      });
      fillSelect($('ef-grade'), etfGrades.map(function (g) { return { value: g, label: g }; }));

      var groupGrades = GRADE_ORDER.filter(function (g) {
        return state.groups.some(function (r) { return r.grade === g; });
      });
      fillSelect($('ef-ggrade'), groupGrades.map(function (g) { return { value: g, label: g }; }));

      var tabs = {};
      state.etfs.forEach(function (r) { tabs[r.tab] = r.tabName; });
      fillSelect($('ef-tabcode'), Object.keys(tabs).sort().map(function (t) {
        return { value: t, label: tabs[t] };
      }));

      try {
        var savedBasket = JSON.parse(window.localStorage.getItem('ef-basket') || '[]');
        if (Array.isArray(savedBasket)) { state.basket = savedBasket.slice(0, 8); }
      } catch (e) { state.basket = []; }

      wire();
      renderMarket();
      renderBacktest();
      renderPick();
      renderEtfs();
      renderGroups();
      renderBasket();
      if (state.basket.length) { loadCorr().then(renderBasket); }

      if (state.us) {
        fillSelect($('ef-ugrade'), GRADE_ORDER.filter(function (g) {
          return state.us.rows.some(function (r) { return r.grade === g; });
        }).map(function (g) { return { value: g, label: g }; }));
        renderUs();
      } else {
        $('ef-us-market').innerHTML =
          '<span class="ef-error">미국 ETF 데이터를 불러오지 못했습니다.</span>';
      }
    })
    .catch(function (err) {
      $('ef-market').innerHTML = '<span class="ef-error">데이터를 불러오지 못했습니다 — ' +
        escapeHtml(err.message) + '</span>';
    });
})();
