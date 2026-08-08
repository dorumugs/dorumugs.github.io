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

  /* ---- PSA 10 통합 표 ----------------------------------------------------
     한 줄이 한 카드다. **PSA 10 만** 담는다 — 글로벌의 raw 는 종류가 다른 값이라
     같은 표에 넣으면 가격순 정렬이 곧바로 거짓말이 된다.

     국내 언어판은 품번에서 언어 접미사를 뗀 하드 키로 이미 서버에서 묶여
     한 줄에 들어 있다. 글로벌↔국내는 붙이지 않는다 — 영문판 14종으로
     시험했을 때 유일하게 이어진 게 1건이었다. */

  var UNI_LIMIT = 200;

  function uniPrices(row) {
    return row[cmp.ucol.prices] || {};
  }

  function uniTopUsd(row) {
    var unit = row[cmp.ucol.unit];
    var values = Object.keys(uniPrices(row)).map(function (k) {
      return uniPrices(row)[k];
    });
    if (!values.length) { return null; }
    var top = Math.max.apply(null, values);
    return unit === 'USD' ? top : toUsd(top, 'KRW');
  }

  function uniRender() {
    if (!cmp.unified) { return; }
    var q = norm(el('cmp-q').value);
    var onlyMerged = el('uni-merged').checked;
    var sort = el('uni-sort').value;
    var U = cmp.ucol;

    /* 원본 순서를 기억해 둔다. 정렬해도 줄을 되짚을 수 있어야 한다. */
    cmp.unified.rows.forEach(function (r, i) { r.__i = i; });

    var band = searchMode() === 'band' ? bandBounds() : null;
    var rows = cmp.unified.rows.filter(function (r) {
      if (onlyMerged && Object.keys(uniPrices(r)).length < 2) { return false; }
      if (band) {
        var top = uniTopUsd(r);
        return top !== null && top >= band.lo && top <= band.hi;
      }
      if (q && norm(r[U.name]).indexOf(q) < 0 && norm(r[U.sub]).indexOf(q) < 0) {
        return false;
      }
      return true;
    });
    if (band) {
      rows.sort(function (a, b) {
        return Math.abs(uniTopUsd(a) - band.usd) - Math.abs(uniTopUsd(b) - band.usd);
      });
    }

    if (band) {
      /* 가격대 모드는 기준 금액에 가까운 순이 곧 정렬이다. */
    } else if (sort === 'name') {
      rows.sort(function (a, b) { return a[U.name].localeCompare(b[U.name], 'ko'); });
    } else if (sort === 'ratio-desc') {
      /* 격차가 없는 줄(언어판 하나)은 뒤로. 0 으로 쳐서 섞으면 순위가 거짓이 된다. */
      rows.sort(function (a, b) {
        var av = a[U.ratio], bv = b[U.ratio];
        if (!av && !bv) { return 0; }
        if (!av) { return 1; }
        if (!bv) { return -1; }
        return bv - av;
      });
    } else {
      var dir = sort === 'price-asc' ? 1 : -1;
      rows.sort(function (a, b) {
        return dir * ((uniTopUsd(b) || 0) - (uniTopUsd(a) || 0));
      });
    }

    el('uni-count').textContent =
      rows.length.toLocaleString('ko-KR') + '줄 중 ' +
      Math.min(UNI_LIMIT, rows.length).toLocaleString('ko-KR') + '줄 표시';

    el('uni-list').innerHTML = rows.slice(0, UNI_LIMIT).map(function (r) {
      var unit = r[U.unit];
      var prices = uniPrices(r);
      var samples = r[U.samples] || {};
      var priceCells = Object.keys(prices).map(function (lang) {
        var native = unit === 'USD' ? fmt(prices[lang], 'USD') : fmt(prices[lang], 'KRW');
        var other = unit === 'USD'
          ? fmt(fromUsd(prices[lang], state.currency), state.currency)
          : fmt(fromUsd(toUsd(prices[lang], 'KRW'), state.currency), state.currency);
        var same = (unit === 'USD' && state.currency === 'USD') ||
                   (unit === 'KRW' && state.currency === 'KRW');
        return '<span class="uni-lang">' + esc(lang) + '</span>' +
          '<b>' + native + '</b>' + (same ? '' : '<small>' + other + '</small>');
      }).join('<hr class="uni-sep">');

      var sampleCells = Object.keys(prices).map(function (lang) {
        var n = samples[lang] || 0;
        return '<span class="' + (n <= 1 ? 'is-thin' : '') + '">' +
          n.toLocaleString('ko-KR') + '건</span>';
      }).join('<hr class="uni-sep">');

      var ratio = r[U.ratio]
        ? '<b class="cmp-diff is-up">' + r[U.ratio] + '배</b>' +
          '<i>언어판 사이</i>'
        : '<span class="cmp-none">—</span>';

      return '<tr class="cmp-item" tabindex="0" data-index="' + r.__i +
        '" data-market="' + esc(r[U.market]) + '">' +
        '<td><b>' + esc(r[U.name]) + '</b></td>' +
        '<td class="cmp-dim"><span class="uni-market is-' +
          (r[U.market] === '글로벌' ? 'g' : 'k') + '">' + esc(r[U.market]) + '</span>' +
          '<i>' + esc(r[U.sub]) + '</i></td>' +
        '<td class="cmp-num is-psa">' + priceCells + '</td>' +
        '<td class="cmp-num">' + sampleCells + '</td>' +
        '<td class="cmp-num">' + ratio + '</td>' +
        '</tr>';
    }).join('') ||
      '<tr><td colspan="5" class="pk-empty">해당하는 카드가 없습니다.</td></tr>';
  }

  function cmpSearch() {
    uniRender();
  }

  /* 표에서 고른 두 줄. 글로벌 줄과 국내 줄을 하나씩 고르면 견준다. */
  function pickRow(index) {
    var r = cmp.unified.rows[index];
    if (!r) { return; }
    if (r[cmp.ucol.market] === '글로벌') { cmp.usd = r; } else { cmp.krw = r; }
    renderPanel();
  }

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

  function priceUsdOf(r, lang) {
    var p = (r[cmp.ucol.prices] || {})[lang];
    if (p === undefined) { return null; }
    return r[cmp.ucol.unit] === 'USD' ? p : toUsd(p, 'KRW');
  }

  function langsOf(r) { return Object.keys(r[cmp.ucol.prices] || {}); }

  function renderPanel() {
    var panel = el('cmp-panel');
    if (!cmp.usd && !cmp.krw) { panel.hidden = true; return; }
    panel.hidden = false;

    var u = cmp.usd, k = cmp.krw;
    var U = cmp.ucol;
    var out = '';

    out += row('카드',
      u ? '<b>' + esc(u[U.name]) + '</b>'
        : '<span class="cmp-empty">표에서 글로벌 줄을 하나 고르세요</span>',
      k ? '<b>' + esc(k[U.name]) + '</b>'
        : '<span class="cmp-empty">표에서 국내 줄을 하나 고르세요</span>', '');

    out += row('시장 · 품번', u ? esc(u[U.sub]) : '', k ? esc(k[U.sub]) : '', '');

    /* 국내 줄에 언어판이 둘이면 각각 따로 견준다. 하나를 임의로 고르면
       그 선택이 곧 결론을 바꾸므로 고르지 않는다. */
    var uPrice = u ? priceUsdOf(u, '영문판') : null;
    var kLangs = k ? langsOf(k) : [];

    out += row('PSA 10',
      u ? allFromUsd(uPrice) : '',
      k ? kLangs.map(function (lang) {
        return '<span class="uni-lang">' + esc(lang) + '</span>' +
          allFromUsd(priceUsdOf(k, lang));
      }).join('<hr class="uni-sep">') : '',
      premiumCells(uPrice, k, kLangs), 'cmp-key');

    out += row('표본',
      u ? esc(((u[U.samples] || {})['영문판'] || 0) + '건') : '',
      k ? kLangs.map(function (lang) {
        var n = (k[U.samples] || {})[lang] || 0;
        return '<span class="' + (n <= 1 ? 'is-thin' : '') + '">' +
          n.toLocaleString('ko-KR') + '건</span>';
      }).join('<hr class="uni-sep">') : '', '');

    if (k && k[U.ratio]) {
      out += row('언어판 격차', '',
        '<b class="cmp-diff is-up">' + k[U.ratio] + '배</b>',
        '<span class="cmp-na">같은 카드인데 언어판끼리 이만큼 벌어집니다</span>');
    }

    el('cmp-tbody').innerHTML = out;
    el('cmp-gap').innerHTML = verdict(uPrice, k, kLangs);
  }

  /* 김치 프리미엄 — 국내가 해외보다 비싼 정도. 음수면 역프리미엄이다.
     양쪽 다 PSA 10 이라 뺄 수 있다. 글로벌 값이 없으면 계산하지 않는다. */
  function premium(usdGlobal, usdDomestic) {
    if (!usdGlobal || !usdDomestic) { return null; }
    return ((usdDomestic - usdGlobal) / usdGlobal) * 100;
  }

  function premiumCells(uPrice, k, kLangs) {
    if (!uPrice || !k) { return ''; }
    return kLangs.map(function (lang) {
      var p = premium(uPrice, priceUsdOf(k, lang));
      if (p === null) { return '<span class="cmp-none">—</span>'; }
      var up = p > 0;
      /* 퍼센트만으로는 크기가 안 잡힌다. 원화 절대금액을 같이 적는다. */
      var gapWon = fromUsd(priceUsdOf(k, lang) - uPrice, 'KRW');
      return '<b class="cmp-diff ' + (up ? 'is-up' : 'is-down') + '">' +
        (up ? '+' : '−') + Math.abs(p).toFixed(1) + '%</b>' +
        '<i>' + esc(lang) + ' ' + (up ? '김치 프리미엄' : '역프리미엄') + '</i>' +
        (gapWon === null ? '' : '<i>' + (up ? '+' : '−') +
          Math.abs(Math.round(gapWon)).toLocaleString('ko-KR') + '원</i>');
    }).join('<hr class="uni-sep">');
  }

  function verdict(uPrice, k, kLangs) {
    if (!cmp.usd || !cmp.krw) {
      return '<span class="cmp-note">글로벌 줄과 국내 줄을 하나씩 고르면 ' +
        '김치 프리미엄을 계산합니다.</span>';
    }
    if (!uPrice) {
      return '<span class="cmp-note">이 글로벌 줄에 <b>PSA 10 값이 없어</b> ' +
        '뺄 수 없습니다.</span>';
    }
    var parts = kLangs.map(function (lang) {
      var p = premium(uPrice, priceUsdOf(k, lang));
      if (p === null) { return ''; }
      return lang + ' <b>' + (p > 0 ? '+' : '−') + Math.abs(p).toFixed(1) + '%</b>';
    }).filter(Boolean).join(' · ');
    return '<span class="cmp-note">둘 다 PSA 10 이라 비교됩니다 — ' + parts + '. ' +
      '<span class="cmp-caveat">영문판·일본판·한글판은 서로 다른 카드입니다. ' +
      '수수료·감정료·관세·환전비용이 빠져 있어 이 폭이 그대로 차익은 아닙니다.' +
      '</span></span>';
  }

  function bindCompare() {
    el('cmp-q').addEventListener('input', debounce(cmpSearch, 200));
    el('cmp-amount').addEventListener('input', debounce(cmpSearch, 250));
    ['cmp-cur', 'cmp-tol', 'uni-sort'].forEach(function (id) {
      el(id).addEventListener('change', cmpSearch);
    });
    el('uni-merged').addEventListener('change', cmpSearch);
    el('cmp-display').addEventListener('change', function () {
      state.currency = el('cmp-display').value;
      cmpSearch();
      renderPanel();
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

    var body = el('uni-list');

    function choose(target) {
      var tr = target && target.closest ? target.closest('.cmp-item') : null;
      if (!tr) { return; }
      var market = tr.dataset.market;
      Array.prototype.forEach.call(body.querySelectorAll('.cmp-item'), function (x) {
        if (x.dataset.market === market) { x.classList.remove('is-on'); }
      });
      tr.classList.add('is-on');
      pickRow(Number(tr.dataset.index));
    }

    body.addEventListener('click', function (ev) { choose(ev.target); });
    /* 표의 줄을 키보드로도 고를 수 있어야 한다. tr 에 tabindex 를 줬다. */
    body.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        choose(ev.target);
      }
    });
  }

  function loadCompare() {
    if (cmp.ready) { return; }
    load();  // 국내 데이터가 필요하다
    Promise.all([
      fetch(BASE + '/unified.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }),
      fetch(BASE + '/meta.json', { cache: 'no-cache' }).then(function (r) { return r.json(); })
    ]).then(function (res) {
      cmp.unified = res[0];
      cmp.ucol = {};
      res[0].columns.forEach(function (n, i) { cmp.ucol[n] = i; });
      cmp.meta = res[1];
      cmp.ready = true;
      bindCompare();
      renderPanel();
      uniRender();
      var fx = cmp.meta.fx || {};
      var s = cmp.unified.stats;
      el('cmp-meta').textContent = (fx.rates && fx.rates.KRW)
        ? '통합 ' + s.total.toLocaleString('ko-KR') + '줄 (글로벌 ' + s.global +
          ' · 국내 ' + s.domestic.toLocaleString('ko-KR') + ') · 언어판이 둘 이상 붙은 줄 ' +
          s.merged + '개. 환산 환율 USD 1 = ' +
          fx.rates.KRW.toLocaleString('ko-KR') + '원 · ' + fx.rates.EUR + '유로 · ' +
          fx.rates.JPY.toLocaleString('ko-KR') + '엔 (' + fx.date + ' 기준 · ' +
          fx.source + ').'
        : '환율을 불러오지 못해 환산이 표시되지 않습니다.';
    }).catch(function (err) {
      el('uni-count').textContent = '통합 표를 불러오지 못했습니다. (' + err.message + ')';
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
