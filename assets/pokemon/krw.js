/* 국내(KREAM) 원화 시세 탭.

   글로벌 탭(app.js)과는 데이터도 기준도 완전히 다르다. 여기 값은 PSA 10
   등급 · 대부분 일본판이고, 저쪽은 등급 없는 raw 영문판이다. 그래서 한 화면에
   섞지 않고 탭으로 갈라 둔다.

   krw.json 이 gzip 147KB 라 첫 화면부터 받지 않는다. 탭을 처음 열 때 받는다.
   외부 라이브러리는 쓰지 않는다 — 차트도 SVG 를 직접 그린다. */
(function () {
  'use strict';

  var app = document.querySelector('.pk-app');
  if (!app) { return; }
  var BASE = app.dataset.base;
  var PAGE = 48;

  var state = {
    data: null, loading: false, filtered: [], shown: PAGE,
    currency: 'KRW'   // 목록에 원래 통화와 함께 보일 통화
  };
  var C = {};
  var M = {};

  function el(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function norm(s) {
    return String(s || '').toLowerCase().replace(/\s+/g, '');
  }

  /* 1,234,000원. 원 단위라 소수점은 의미가 없다. */
  function won(v) {
    if (v === null || v === undefined) { return '—'; }
    return Number(v).toLocaleString('ko-KR') + '원';
  }

  /* 축 눈금처럼 좁은 자리용. 1250000 -> '125만'. */
  function shortWon(v) {
    if (v === null || v === undefined) { return '—'; }
    var n = Number(v);
    if (n >= 100000000) { return (n / 100000000).toFixed(1).replace(/\.0$/, '') + '억'; }
    if (n >= 10000) { return Math.round(n / 10000).toLocaleString('ko-KR') + '만'; }
    return n.toLocaleString('ko-KR');
  }

  /* 변동률. null 은 0% 가 아니라 '거래가 없어 모른다' 는 뜻이라 — 로 적는다. */
  function pct(v) {
    if (v === null || v === undefined) { return { text: '—', cls: 'is-flat' }; }
    var n = Number(v);
    return {
      text: (n > 0 ? '+' : '') + n.toFixed(1) + '%',
      cls: n > 0 ? 'is-up' : (n < 0 ? 'is-down' : 'is-flat')
    };
  }

  /* 응답이 주는 주소는 1120px 원본이라 한 장에 845KB 다. 그리드엔 줄인 것을
     쓰고, 눌러서 여는 것만 큰 것을 준다. */
  function imageOf(row) {
    var path = row[C.image];
    if (!path) { return null; }
    var base = state.data.image_prefix + path;
    return {
      thumb: base + (state.data.image_thumb || ''),
      full: base + (state.data.image_full || '')
    };
  }

  /* ---- 시장 추이 차트 ---------------------------------------------------
     거래건수를 옅은 막대로 깔고 중앙값을 선으로 얹는다. 값이 두 자리수
     차이라 축을 따로 쓴다. viewBox 로 그려 폭은 100% 로 늘어난다. */

  var W = 720, H = 240, PAD_L = 52, PAD_R = 44, PAD_T = 16, PAD_B = 30;

  function drawChart() {
    var rows = state.data.market;
    if (!rows.length) { return; }
    var medians = rows.map(function (r) { return r[M.median] || 0; });
    var counts = rows.map(function (r) { return r[M.count] || 0; });
    var maxMedian = Math.max.apply(null, medians);
    var minMedian = Math.min.apply(null, medians);
    var maxCount = Math.max.apply(null, counts);
    var span = maxMedian - minMedian || 1;

    var plotW = W - PAD_L - PAD_R;
    var plotH = H - PAD_T - PAD_B;
    var step = rows.length > 1 ? plotW / (rows.length - 1) : 0;

    function x(i) { return PAD_L + i * step; }
    function yMedian(v) { return PAD_T + plotH - ((v - minMedian) / span) * plotH; }
    function yCount(v) { return PAD_T + plotH - (v / (maxCount || 1)) * plotH * 0.55; }

    var barW = Math.max(2, step * 0.55);
    var bars = rows.map(function (r, i) {
      var top = yCount(r[M.count] || 0);
      return '<rect class="krw-bar" x="' + (x(i) - barW / 2).toFixed(1) +
        '" y="' + top.toFixed(1) + '" width="' + barW.toFixed(1) +
        '" height="' + Math.max(0, PAD_T + plotH - top).toFixed(1) + '"></rect>';
    }).join('');

    var line = rows.map(function (r, i) {
      return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + yMedian(r[M.median] || 0).toFixed(1);
    }).join(' ');

    var ticks = [minMedian, (minMedian + maxMedian) / 2, maxMedian].map(function (v) {
      var y = yMedian(v);
      return '<line class="krw-grid" x1="' + PAD_L + '" y1="' + y.toFixed(1) +
        '" x2="' + (W - PAD_R) + '" y2="' + y.toFixed(1) + '"></line>' +
        '<text class="krw-axis" x="' + (PAD_L - 8) + '" y="' + (y + 3.5).toFixed(1) +
        '" text-anchor="end">' + esc(shortWon(Math.round(v))) + '</text>';
    }).join('');

    var countTop = yCount(maxCount);
    var countAxis = '<text class="krw-axis" x="' + (W - PAD_R + 8) + '" y="' +
      (countTop + 3.5).toFixed(1) + '">' + maxCount + '건</text>';

    var firstDate = rows[0][M.date];
    var lastDate = rows[rows.length - 1][M.date];
    var dateAxis =
      '<text class="krw-axis" x="' + PAD_L + '" y="' + (H - 8) + '">' +
        esc(firstDate.slice(5)) + '</text>' +
      '<text class="krw-axis" x="' + (W - PAD_R) + '" y="' + (H - 8) +
        '" text-anchor="end">' + esc(lastDate.slice(5)) + '</text>';

    el('krw-chart').innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
      'aria-label="최근 ' + rows.length + '일 국내 포켓몬 카드 거래 중앙값과 거래건수">' +
      ticks + bars + countAxis +
      '<path class="krw-line" d="' + line + '"></path>' +
      dateAxis +
      '<rect id="krw-hit" x="' + PAD_L + '" y="' + PAD_T + '" width="' + plotW +
      '" height="' + plotH + '" fill="transparent"></rect>' +
      '<line id="krw-cursor" class="krw-cursor" x1="0" y1="' + PAD_T + '" x2="0" y2="' +
      (PAD_T + plotH) + '" style="display:none"></line>' +
      '</svg>';

    bindChartReadout(rows, x, step);
    showReadout(rows[rows.length - 1]);
  }

  function bindChartReadout(rows, x, step) {
    var svg = el('krw-chart').querySelector('svg');
    var cursor = el('krw-cursor');

    function at(event) {
      var box = svg.getBoundingClientRect();
      var point = event.touches ? event.touches[0] : event;
      var vx = ((point.clientX - box.left) / box.width) * W;
      var i = step ? Math.round((vx - PAD_L) / step) : 0;
      i = Math.max(0, Math.min(rows.length - 1, i));
      cursor.setAttribute('x1', x(i));
      cursor.setAttribute('x2', x(i));
      cursor.style.display = '';
      showReadout(rows[i]);
    }

    svg.addEventListener('mousemove', at);
    svg.addEventListener('touchstart', at, { passive: true });
    svg.addEventListener('touchmove', at, { passive: true });
    svg.addEventListener('mouseleave', function () {
      cursor.style.display = 'none';
      showReadout(rows[rows.length - 1]);
    });
  }

  function showReadout(row) {
    el('krw-readout').innerHTML =
      '<b>' + esc(row[M.date]) + '</b>' +
      '<span>중앙값 ' + esc(won(row[M.median])) + '</span>' +
      '<span>거래 ' + Number(row[M.count]).toLocaleString('ko-KR') + '건</span>' +
      '<span>거래대금 ' + esc(shortWon(row[M.total])) + '원</span>';
  }

  /* ---- 상품 목록 -------------------------------------------------------- */

  function applyFilters() {
    var q = norm(el('krw-q').value);
    var lang = el('krw-lang').value;
    var liquid = el('krw-liquid').checked;
    var sort = el('krw-sort').value;

    var out = state.data.rows.filter(function (r) {
      if (lang && r[C.lang] !== lang) { return false; }
      if (liquid && (r[C.tx] || 0) < 2) { return false; }
      if (q && norm(r[C.name_ko]).indexOf(q) < 0 &&
              norm(r[C.name_en]).indexOf(q) < 0 &&
              norm(r[C.code]).indexOf(q) < 0) { return false; }
      return true;
    });

    var by = {
      'price-desc': function (a, b) { return (b[C.price] || 0) - (a[C.price] || 0); },
      'price-asc': function (a, b) { return (a[C.price] || 0) - (b[C.price] || 0); },
      'tx-desc': function (a, b) { return (b[C.tx] || 0) - (a[C.tx] || 0); },
      'name': function (a, b) { return a[C.name_ko].localeCompare(b[C.name_ko], 'ko'); }
    };
    /* 변동률 정렬은 값이 없는 상품을 뒤로 민다. 모르는 걸 0% 로 쳐서
       한가운데 끼워 넣으면 순위가 거짓말이 된다. */
    if (sort === 'chg-desc' || sort === 'chg-asc') {
      var dir = sort === 'chg-desc' ? -1 : 1;
      out.sort(function (a, b) {
        var av = a[C.change_30d], bv = b[C.change_30d];
        if (av === null && bv === null) { return 0; }
        if (av === null) { return 1; }
        if (bv === null) { return -1; }
        return dir * (bv - av);
      });
    } else {
      out.sort(by[sort] || by['price-desc']);
    }

    state.filtered = out;
    state.shown = PAGE;
    render();
  }

  function cardHtml(r) {
    var pic = imageOf(r);
    var change = pct(r[C.change_30d]);
    /* referrerpolicy="no-referrer" 가 반드시 있어야 한다. 네이버 CDN 은
       Referer 가 KREAM 도메인이 아니면 403 + text/html 을 주고, 그러면
       Chrome 이 ORB 로 막아 이미지가 통째로 안 뜬다 (실측). 링크도 같은
       이유로 noreferrer 를 준다. */
    var thumb = '<span class="pk-noimg">이미지 없음</span>' + (pic
      ? '<img class="pk-img" src="' + esc(pic.thumb) + '" alt="' + esc(r[C.name_ko]) +
        '" loading="lazy" width="525" height="525" referrerpolicy="no-referrer"' +
        ' onerror="this.style.display=\'none\'">'
      : '');

    return '<article class="pk-card">' +
      (pic
        ? '<a class="pk-imgwrap krw-imgwrap" href="' + esc(pic.full) +
          '" target="_blank" rel="noopener noreferrer">' + thumb + '</a>'
        : '<div class="pk-imgwrap krw-imgwrap">' + thumb + '</div>') +
      '<div class="pk-body">' +
        '<h3 class="pk-name">' + esc(r[C.name_ko]) + '</h3>' +
        '<p class="pk-ko krw-en">' + esc(r[C.name_en]) + '</p>' +
        '<p class="pk-set">' + esc(r[C.lang] || '기타') + ' · ' + esc(r[C.code]) + '</p>' +
        '<p class="pk-price">' + esc(won(r[C.price])) +
          ' <span class="krw-chg ' + change.cls + '">' + esc(change.text) + '</span></p>' +
        '<div class="pk-sub">' +
          row('30일 고가', won(r[C.high_30d])) +
          row('30일 저가', won(r[C.low_30d])) +
          row('30일 거래', Number(r[C.tx] || 0).toLocaleString('ko-KR') + '건') +
        '</div>' +
      '</div>' +
    '</article>';
  }

  function row(label, value) {
    return '<div class="pk-row"><span class="pk-lbl">' + esc(label) +
      '</span><span class="pk-val">' + esc(value) + '</span></div>';
  }

  function render() {
    var slice = state.filtered.slice(0, state.shown);
    el('krw-grid').innerHTML = slice.length
      ? slice.map(cardHtml).join('')
      : '<p class="pk-empty">찾는 카드가 없습니다. 철자나 필터를 확인해 보세요.</p>';
    el('krw-count').textContent =
      state.filtered.length.toLocaleString('ko-KR') + '종 중 ' +
      slice.length.toLocaleString('ko-KR') + '종 표시';
    el('krw-more').style.display =
      state.filtered.length > state.shown ? '' : 'none';
  }

  function fillStats() {
    var s = state.data.stats;
    var langs = Object.keys(s.languages).map(function (k) {
      return k + ' ' + s.languages[k].toLocaleString('ko-KR');
    }).join(' · ');
    el('krw-stats').innerHTML =
      stat('상품', s.product_count.toLocaleString('ko-KR') + '종') +
      stat('중앙값', shortWon(s.median_price) + '원') +
      stat('최고가', shortWon(s.max_price) + '원') +
      stat('30일 거래 1건 이하', s.thin.toLocaleString('ko-KR') + '종');
    el('krw-meta').textContent =
      '집계 ' + s.first_date + ' ~ ' + s.last_date + ' · ' + langs +
      ' · 마지막 갱신 ' + state.data.generated + '.';

    var sel = el('krw-lang');
    Object.keys(s.languages).forEach(function (k) {
      sel.insertAdjacentHTML('beforeend',
        '<option value="' + esc(k) + '">' + esc(k) + ' (' + s.languages[k] + ')</option>');
    });
  }

  function stat(label, value) {
    return '<div class="krw-stat"><span>' + esc(label) + '</span><b>' +
      esc(value) + '</b></div>';
  }

  function debounce(fn, ms) {
    var t;
    return function () { clearTimeout(t); t = setTimeout(fn, ms); };
  }

  function bind() {
    el('krw-q').addEventListener('input', debounce(applyFilters, 180));
    ['krw-lang', 'krw-sort'].forEach(function (id) {
      el(id).addEventListener('change', applyFilters);
    });
    el('krw-liquid').addEventListener('change', applyFilters);
    el('krw-more').addEventListener('click', function () {
      state.shown += PAGE;
      render();
    });
  }

  function load() {
    if (state.data || state.loading) { return; }
    state.loading = true;
    fetch(BASE + '/krw.json', { cache: 'no-cache' })
      .then(function (r) {
        if (!r.ok) { throw new Error('krw ' + r.status); }
        return r.json();
      })
      .then(function (data) {
        state.data = data;
        data.columns.forEach(function (name, i) { C[name] = i; });
        data.market_columns.forEach(function (name, i) { M[name] = i; });
        fillStats();
        drawChart();
        bind();
        applyFilters();
      })
      .catch(function (err) {
        el('krw-count').textContent =
          '국내 시세를 불러오지 못했습니다. (' + err.message + ')';
        state.loading = false;
      });
  }

  /* ---- 비교 -------------------------------------------------------------
     자동 매칭은 하지 않는다. "피카츄"만 해도 국내 112종 · 글로벌 98장이라
     기계가 이으면 베이스셋 카드와 최신 일본판 SAR 이 묶인다. 양쪽을 나란히
     보여주고 고르는 건 사람이 한다. 고른 둘만 같은 통화로 환산해 견준다. */

  var cmp = { usd: null, krw: null, global: null, ready: false };

  /* 글로벌 데이터는 app.js 가 들고 있다. 같은 파일을 두 번 받지 않도록
     전역으로 넘겨받는다. 아직 안 왔으면 직접 받는다. */
  function globalData() {
    if (cmp.global) { return Promise.resolve(cmp.global); }
    if (window.__pkGlobal) { cmp.global = window.__pkGlobal; return Promise.resolve(cmp.global); }
    return fetch(BASE + '/cards.json', { cache: 'no-cache' })
      .then(function (r) { return r.json(); })
      .then(function (d) { cmp.global = d; return d; });
  }

  /* ---- 통화 -------------------------------------------------------------
     모든 값을 일단 달러로 모은 뒤 원·유로·엔으로 편다. 환율은 USD 기준
     하나뿐이라 교차 환산이 어긋날 일이 없다.

     주의: 여기서 만드는 유로는 **환산 유로**다. 카드의 `cm_avg` 는 Cardmarket
     유럽 시장의 **실제 체결가**라 다른 값이다. 화면에서 섞지 않는다. */

  var CURRENCIES = [
    { code: 'KRW', label: '원', suffix: '원', digits: 0 },
    { code: 'USD', label: '달러', prefix: '$', digits: 0 },
    { code: 'EUR', label: '유로', prefix: '€', digits: 0 },
    { code: 'JPY', label: '엔', suffix: '엔', digits: 0 }
  ];

  function rate(code) {
    var fx = (cmp.meta && cmp.meta.fx) || {};
    return (fx.rates && fx.rates[code]) || null;
  }

  function fromUsd(usd, code) {
    var r = rate(code);
    return (usd === null || usd === undefined || !r) ? null : usd * r;
  }

  function toUsd(amount, code) {
    var r = rate(code);
    return (amount === null || amount === undefined || !r) ? null : amount / r;
  }

  function fmt(amount, code) {
    if (amount === null || amount === undefined) { return '—'; }
    var c = null;
    CURRENCIES.forEach(function (x) { if (x.code === code) { c = x; } });
    if (!c) { return String(amount); }
    return (c.prefix || '') +
      Number(amount).toLocaleString('ko-KR', { maximumFractionDigits: c.digits }) +
      (c.suffix || '');
  }

  /* 목록용 — 원래 통화와 사용자가 고른 통화 둘만. 넷을 다 넣으면 표가 못 읽힌다. */
  function pairFromUsd(usd) {
    if (usd === null || usd === undefined) { return '—'; }
    var pick = state.currency;
    if (pick === 'USD') { return fmt(usd, 'USD'); }
    return fmt(usd, 'USD') + '<small>' + fmt(fromUsd(usd, pick), pick) + '</small>';
  }

  function pairFromWon(won) {
    if (won === null || won === undefined) { return '—'; }
    var pick = state.currency;
    if (pick === 'KRW') { return fmt(won, 'KRW'); }
    return fmt(won, 'KRW') +
      '<small>' + fmt(fromUsd(toUsd(won, 'KRW'), pick), pick) + '</small>';
  }

  /* 비교표용 — 네 통화를 다 편다. 여기서는 자리가 있다. */
  function allFromUsd(usd) {
    if (usd === null || usd === undefined) { return '—'; }
    return CURRENCIES.map(function (c, i) {
      var v = fmt(fromUsd(usd, c.code), c.code);
      return i === 0 ? '<b>' + v + '</b>' : '<i>' + v + '</i>';
    }).join('');
  }

  function allFromWon(won) {
    return allFromUsd(toUsd(won, 'KRW'));
  }

  /* ---- 검색: 이름으로 / 가격대로 -----------------------------------------
     이름이 달라도 값이 비슷하면 견줄 거리가 된다. 가격대 모드는 고른 금액의
     ±범위에 드는 카드를 양쪽에서 찾아 준다. 기준은 **PSA 10** 이다 — 국내가
     전부 PSA 10 이라 그래야 같은 것끼리 걸린다. */

  function searchMode() {
    var m = document.querySelector('input[name="cmp-mode"]:checked');
    return m ? m.value : 'name';
  }

  function bandBounds() {
    var amount = parseFloat(String(el('cmp-amount').value).replace(/[^0-9.]/g, ''));
    if (!amount || amount <= 0) { return null; }
    var usd = toUsd(amount, el('cmp-cur').value);
    if (usd === null) { return null; }
    var tol = Number(el('cmp-tol').value) / 100;
    return { usd: usd, lo: usd * (1 - tol), hi: usd * (1 + tol), tol: tol };
  }

  function cmpSearch() {
    if (searchMode() === 'band') {
      var band = bandBounds();
      if (!band) {
        emptyLists('금액을 입력하세요.');
        return;
      }
      renderGlobalList(null, band);
      renderKrwList(null, band);
      return;
    }
    var q = norm(el('cmp-q').value);
    if (!q) {
      emptyLists('검색어를 입력하세요.');
      return;
    }
    renderGlobalList(q, null);
    renderKrwList(q, null);
  }

  function emptyLists(message) {
    el('cmp-list-usd').innerHTML = '';
    el('cmp-list-krw').innerHTML = '';
    el('cmp-count-usd').textContent = message;
    el('cmp-count-krw').textContent = message;
  }

  var LIMIT = 30;

  function renderGlobalList(q, band) {
    var d = cmp.global;
    if (!d) { return; }
    var GC = {};
    d.columns.forEach(function (n, i) { GC[n] = i; });
    function psaOf(r) {
      var g = cmp.graded && cmp.graded[d.sets[r[GC.set]] + '-' + r[GC.local_id]];
      return g ? g[cmp.gcol.psa10] : null;
    }

    var hits;
    if (band) {
      /* 가격대 모드는 PSA 10 이 있는 카드만 본다. raw 를 섞으면 등급이 다른
         것끼리 걸려서 "비슷한 가격" 이라는 말이 무의미해진다. */
      hits = d.rows.filter(function (r) {
        var p = psaOf(r);
        return p !== null && p !== undefined && p >= band.lo && p <= band.hi;
      });
      hits.sort(function (a, b) {
        return Math.abs(psaOf(a) - band.usd) - Math.abs(psaOf(b) - band.usd);
      });
    } else {
      hits = d.rows.filter(function (r) {
        return norm(r[GC.name_en]).indexOf(q) >= 0 || norm(r[GC.name_ko]).indexOf(q) >= 0;
      });
      hits.sort(function (a, b) { return (b[GC.price] || 0) - (a[GC.price] || 0); });
    }

    el('cmp-count-usd').textContent =
      hits.length.toLocaleString('ko-KR') + '장 중 ' +
      Math.min(LIMIT, hits.length) + '장' +
      (band ? ' · PSA 10 이 있는 카드만' : '');

    el('cmp-list-usd').innerHTML = hits.slice(0, LIMIT).map(function (r) {
      var sid = d.sets[r[GC.set]];
      var cardId = sid + '-' + r[GC.local_id];
      var g = (cmp.graded && cmp.graded[cardId]) || null;
      var psa = g ? g[cmp.gcol.psa10] : null;
      return '<tr class="cmp-item" tabindex="0" data-side="usd" data-id="' +
        esc(cardId) + '" data-psa="' + (psa ? '1' : '0') + '">' +
        '<td><b>' + esc(r[GC.name_en]) + '</b>' +
          (r[GC.name_ko] ? '<i>' + esc(r[GC.name_ko]) + '</i>' : '') + '</td>' +
        '<td class="cmp-dim">' + esc(sid) + ' · #' + esc(r[GC.local_id]) + '</td>' +
        '<td class="cmp-num">' + pairFromUsd(r[GC.price]) + '</td>' +
        '<td class="cmp-num' + (psa ? ' is-psa' : ' is-none') + '">' +
          (psa ? pairFromUsd(psa) : '—') + '</td>' +
        '</tr>';
    }).join('') || '<tr><td colspan="4" class="pk-empty">글로벌에 없습니다.</td></tr>';
  }

  function renderKrwList(q, band) {
    var hits;
    if (band) {
      hits = state.data.rows.filter(function (r) {
        var usd = toUsd(r[C.price], 'KRW');
        return usd !== null && usd >= band.lo && usd <= band.hi;
      });
      hits.sort(function (a, b) {
        return Math.abs(toUsd(a[C.price], 'KRW') - band.usd) -
               Math.abs(toUsd(b[C.price], 'KRW') - band.usd);
      });
    } else {
      hits = state.data.rows.filter(function (r) {
        return norm(r[C.name_ko]).indexOf(q) >= 0 || norm(r[C.name_en]).indexOf(q) >= 0;
      });
      hits.sort(function (a, b) { return (b[C.price] || 0) - (a[C.price] || 0); });
    }

    el('cmp-count-krw').textContent =
      hits.length.toLocaleString('ko-KR') + '종 중 ' +
      Math.min(LIMIT, hits.length) + '종';

    el('cmp-list-krw').innerHTML = hits.slice(0, LIMIT).map(function (r) {
      var idx = state.data.rows.indexOf(r);
      var tx = Number(r[C.tx] || 0);
      return '<tr class="cmp-item" tabindex="0" data-side="krw" data-id="' + idx +
        '" data-psa="1">' +
        '<td><b>' + esc(r[C.name_ko]) + '</b></td>' +
        '<td class="cmp-dim">' + esc(r[C.lang] || '기타') + ' · ' + esc(r[C.code]) + '</td>' +
        /* 거래가 한 건이면 그 값은 시세가 아니라 사례 하나다. 눈에 띄게 둔다. */
        '<td class="cmp-num' + (tx <= 1 ? ' is-thin' : '') + '">' +
          tx.toLocaleString('ko-KR') + '건</td>' +
        '<td class="cmp-num is-psa">' + pairFromWon(r[C.price]) + '</td>' +
        '</tr>';
    }).join('') || '<tr><td colspan="4" class="pk-empty">국내에 없습니다.</td></tr>';
  }

  function pickCard(side, id) {
    if (side === 'usd') {
      var d = cmp.global, GC = {};
      d.columns.forEach(function (n, i) { GC[n] = i; });
      var row = null;
      d.rows.some(function (r) {
        if (d.sets[r[GC.set]] + '-' + r[GC.local_id] === id) { row = r; return true; }
        return false;
      });
      if (!row) { return; }
      var g = (cmp.graded && cmp.graded[id]) || null;
      cmp.usd = {
        title: row[GC.name_en], sub: d.sets[row[GC.set]] + ' · #' + row[GC.local_id],
        ko: row[GC.name_ko], raw: row[GC.price], cm: row[GC.cm_avg],
        psa10: g ? g[cmp.gcol.psa10] : null,
        psa10n: g ? g[cmp.gcol.psa10_n] : null,
        psa10date: g ? g[cmp.gcol.psa10_date] : '',
        psa9: g ? g[cmp.gcol.psa9] : null
      };
    } else {
      var r2 = state.data.rows[Number(id)];
      if (!r2) { return; }
      cmp.krw = {
        title: r2[C.name_ko], sub: (r2[C.lang] || '기타') + ' · ' + r2[C.code],
        en: r2[C.name_en], lang: r2[C.lang] || '기타',
        psa10won: r2[C.price], tx: r2[C.tx],
        high: r2[C.high_30d], low: r2[C.low_30d]
      };
    }
    renderPanel();
  }

  /* 고른 두 장을 항목별로 나란히 놓는다. 차이 칸은 **같은 등급끼리만** 채운다 —
     빈칸을 남기는 게 잘못된 뺄셈을 보여주는 것보다 낫다. */
  /* label 은 이 파일 안의 고정 문자열이라 그대로 넣는다 (일부는 <small> 을
     쓴다). 사용자 입력이 들어오는 left/right 는 부르는 쪽에서 esc 한다. */
  function row(label, left, right, diff, cls) {
    return '<tr' + (cls ? ' class="' + cls + '"' : '') + '>' +
      '<th scope="row">' + label + '</th>' +
      '<td class="cmp-num">' + (left || '—') + '</td>' +
      '<td class="cmp-num">' + (right || '—') + '</td>' +
      '<td class="cmp-num">' + (diff || '') + '</td>' +
      '</tr>';
  }

  function renderPanel() {
    var panel = el('cmp-panel');
    if (!cmp.usd && !cmp.krw) { panel.hidden = true; return; }
    panel.hidden = false;

    var u = cmp.usd, k = cmp.krw;
    var out = '';

    out += row('카드',
      u ? '<b>' + esc(u.title) + '</b>' + (u.ko ? '<i>' + esc(u.ko) + '</i>' : '')
        : '<span class="cmp-empty">왼쪽 표에서 한 줄 고르세요</span>',
      k ? '<b>' + esc(k.title) + '</b>' : '<span class="cmp-empty">오른쪽 표에서 한 줄 고르세요</span>',
      '');

    out += row('세트 · 품번', u ? esc(u.sub) : '', k ? esc(k.sub) : '', '');
    out += row('언어판', u ? '영문판' : '', k ? esc(k.lang) : '', '');
    out += row('raw 현재가', u ? allFromUsd(u.raw) : '', '—',
      k ? '<span class="cmp-na">국내는 raw 를 안 팝니다</span>' : '');

    var diff = psa10Diff();
    out += row('PSA 10',
      u ? (u.psa10 ? allFromUsd(u.psa10) : '<span class="cmp-none">아직 없음</span>') : '',
      k ? allFromWon(k.psa10won) : '',
      diff.cell, 'cmp-key');

    out += row('PSA 9', u ? (u.psa9 ? allFromUsd(u.psa9) : '—') : '', '—', '');

    /* 이 유로만은 환산이 아니다. Cardmarket 유럽 시장의 실제 체결가라
       달러를 환산한 값과 다르고, 그 차이 자체가 정보다. */
    if (u && u.cm) {
      var conv = fromUsd(u.raw, 'EUR');
      out += row('유럽 실거래 <small>Cardmarket</small>',
        fmt(u.cm, 'EUR') + (conv
          ? '<i>달러 환산은 ' + fmt(conv, 'EUR') + '</i>' : ''),
        '—',
        '<span class="cmp-na">환산이 아니라 실제 유럽 시세</span>');
    }

    out += row('표본',
      u && u.psa10 ? esc(u.psa10n + '건') +
        (u.psa10date ? '<i>' + esc(u.psa10date) + '</i>' : '') : '',
      k ? esc(Number(k.tx || 0).toLocaleString('ko-KR') + '건') + '<i>최근 30일</i>' : '',
      '');
    /* 고가와 저가가 같으면 거래가 한 건뿐이라는 뜻이다. 그 사실을 적어 준다. */
    out += row('30일 고·저', '',
      k ? (k.high === k.low
            ? '<b>' + esc(Number(k.high || 0).toLocaleString('ko-KR')) + '원</b>' +
              '<i>고·저가 같음 — 거래 한 건</i>'
            : '<b>고 ' + esc(Number(k.high || 0).toLocaleString('ko-KR')) + '원</b>' +
              '<i>저 ' + esc(Number(k.low || 0).toLocaleString('ko-KR')) + '원</i>')
        : '', '');

    el('cmp-tbody').innerHTML = out;
    el('cmp-gap').innerHTML = diff.note;
  }

  /* 같은 등급끼리만 뺀다. 글로벌에 PSA 10 이 없으면 뺄 것이 없다고 적는다 —
     raw 와 PSA 10 을 빼면 그 차이는 나라 차이가 아니라 등급 프리미엄이다.
     {cell: 표의 '차이' 칸, note: 표 아래 한 줄} 을 돌려준다. */
  function psa10Diff() {
    if (!cmp.usd || !cmp.krw) {
      return { cell: '', note: '<span class="cmp-note">양쪽에서 하나씩 고르면 차이를 계산합니다.</span>' };
    }
    if (cmp.usd.psa10 === null || cmp.usd.psa10 === undefined) {
      return {
        cell: '<span class="cmp-na">계산 안 함</span>',
        note: '<span class="cmp-note">이 글로벌 카드는 <b>PSA 10 값이 없어</b> 뺄 수 없습니다. ' +
          'raw 와 PSA 10 을 빼면 나라 차이가 아니라 등급 프리미엄이 나옵니다.</span>'
      };
    }
    var a = fromUsd(cmp.usd.psa10, 'KRW');
    var b = cmp.krw.psa10won;
    if (!a) {
      return { cell: '', note: '<span class="cmp-note">환율을 불러오지 못했습니다.</span>' };
    }
    var gap = b - a;
    var pct = (gap / a) * 100;
    var up = gap > 0;
    /* 김치 프리미엄 — 국내가 해외보다 비싼 정도. 음수면 역프리미엄이다.
       암호화폐 쪽에서 굳은 말이라 뜻이 바로 통한다. */
    var label = up ? '김치 프리미엄' : '역프리미엄';
    return {
      cell: '<b class="cmp-diff ' + (up ? 'is-up' : 'is-down') + '">' +
        (up ? '+' : '−') + Math.abs(pct).toFixed(1) + '%</b>' +
        '<i>' + esc(label) + '</i>' +
        '<i>' + (up ? '+' : '−') +
        Math.abs(Math.round(gap)).toLocaleString('ko-KR') + '원</i>',
      note: '<span class="cmp-note">둘 다 PSA 10 이라 비교됩니다 — <b>' +
        esc(label) + ' ' + (up ? '+' : '−') + Math.abs(pct).toFixed(1) + '%.</b> ' +
        '국내가 해외보다 ' + Math.abs(pct).toFixed(1) + '% ' +
        (up ? '비쌉니다' : '쌉니다') + '. ' +
        '<span class="cmp-caveat">영문판과 일본판은 다른 카드입니다. ' +
        '수수료·감정료·관세·환전비용은 빠져 있어, 이 폭이 그대로 차익이 되지는 ' +
        '않습니다.</span></span>'
    };
  }

  function bindCompare() {
    el('cmp-q').addEventListener('input', debounce(cmpSearch, 200));
    el('cmp-amount').addEventListener('input', debounce(cmpSearch, 250));
    ['cmp-cur', 'cmp-tol'].forEach(function (id) {
      el(id).addEventListener('change', cmpSearch);
    });
    el('cmp-display').addEventListener('change', function () {
      state.currency = el('cmp-display').value;
      cmpSearch();
    });
    Array.prototype.forEach.call(
      document.querySelectorAll('input[name="cmp-mode"]'), function (radio) {
        radio.addEventListener('change', function () {
          var band = searchMode() === 'band';
          el('cmp-by-name').hidden = band;
          el('cmp-by-band').hidden = !band;
          cmpSearch();
        });
      });

    ['cmp-list-usd', 'cmp-list-krw'].forEach(function (id) {
      var body = el(id);

      function choose(target) {
        var tr = target && target.closest ? target.closest('.cmp-item') : null;
        if (!tr) { return; }
        Array.prototype.forEach.call(body.querySelectorAll('.cmp-item'), function (x) {
          x.classList.remove('is-on');
        });
        tr.classList.add('is-on');
        pickCard(tr.dataset.side, tr.dataset.id);
      }

      body.addEventListener('click', function (ev) { choose(ev.target); });
      /* 표의 줄을 키보드로도 고를 수 있어야 한다. tr 에 tabindex 를 줬다. */
      body.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          choose(ev.target);
        }
      });
    });
  }

  function loadCompare() {
    if (cmp.ready) { return; }
    load();  // 국내 데이터가 필요하다
    Promise.all([
      globalData(),
      fetch(BASE + '/graded.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }),
      fetch(BASE + '/meta.json', { cache: 'no-cache' }).then(function (r) { return r.json(); })
    ]).then(function (res) {
      cmp.global = res[0];
      cmp.gcol = {};
      res[1].columns.forEach(function (n, i) { cmp.gcol[n] = i; });
      cmp.graded = res[1].cards;
      cmp.meta = res[2];
      cmp.ready = true;
      bindCompare();
      renderPanel();
      var fx = cmp.meta.fx || {};
      el('cmp-meta').textContent = (fx.rates && fx.rates.KRW)
        ? '환산 환율 USD 1 = ' + fx.rates.KRW.toLocaleString('ko-KR') + '원 · ' +
          fx.rates.EUR + '유로 · ' + fx.rates.JPY.toLocaleString('ko-KR') + '엔 (' +
          fx.date + ' 기준 · ' + fx.source + '). ' +
          '유럽 실거래(Cardmarket) 유로만은 환산이 아니라 실제 시세입니다. ' +
          '글로벌 PSA 10 이 붙은 카드는 ' +
          (cmp.meta.graded_count || 0).toLocaleString('ko-KR') + '장입니다.'
        : '환율을 불러오지 못해 환산이 표시되지 않습니다.';
      if (el('cmp-q').value) { cmpSearch(); }
    }).catch(function (err) {
      el('cmp-count-usd').textContent = '비교 데이터를 불러오지 못했습니다. (' + err.message + ')';
    });
  }

  /* ---- 탭 --------------------------------------------------------------- */

  function activate(which) {
    ['usd', 'krw', 'cmp'].forEach(function (name) {
      var on = name === which;
      var tab = el('pk-tab-' + name);
      var view = el('pk-view-' + name);
      if (!tab || !view) { return; }
      tab.classList.toggle('is-on', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      view.hidden = !on;
    });
    if (which === 'krw') { load(); }
    if (which === 'cmp') { loadCompare(); }
    if (history.replaceState) {
      history.replaceState(null, '',
        which === 'usd' ? location.pathname : '#' + which);
    }
  }

  ['usd', 'krw', 'cmp'].forEach(function (name) {
    var tab = el('pk-tab-' + name);
    if (tab) {
      tab.addEventListener('click', function () { activate(name); });
    }
  });

  if (location.hash === '#krw') { activate('krw'); }
  if (location.hash === '#cmp') { activate('cmp'); }
})();
