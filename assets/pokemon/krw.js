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

  var state = { data: null, loading: false, filtered: [], shown: PAGE };
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

  function usdToWon(usd) {
    var fx = (cmp.meta && cmp.meta.fx) || {};
    return fx.rate ? usd * fx.rate : null;
  }

  function wonToUsd(won) {
    var fx = (cmp.meta && cmp.meta.fx) || {};
    return fx.rate ? won / fx.rate : null;
  }

  /* 두 통화를 항상 같이 적는다. 한쪽만 적으면 독자가 머릿속으로 환산하다 틀린다. */
  function bothFromUsd(usd) {
    if (usd === null || usd === undefined) { return '—'; }
    var w = usdToWon(usd);
    return '$' + Number(usd).toLocaleString('ko-KR', { maximumFractionDigits: 0 }) +
      (w ? ' <small>· ' + Math.round(w).toLocaleString('ko-KR') + '원</small>' : '');
  }

  function bothFromWon(won) {
    if (won === null || won === undefined) { return '—'; }
    var u = wonToUsd(won);
    return Number(won).toLocaleString('ko-KR') + '원' +
      (u ? ' <small>· $' + Math.round(u).toLocaleString('ko-KR') + '</small>' : '');
  }

  function cmpSearch() {
    var q = norm(el('cmp-q').value);
    if (!q) {
      el('cmp-list-usd').innerHTML = '';
      el('cmp-list-krw').innerHTML = '';
      el('cmp-count-usd').textContent = '검색어를 입력하세요.';
      el('cmp-count-krw').textContent = '검색어를 입력하세요.';
      return;
    }
    renderGlobalList(q);
    renderKrwList(q);
  }

  var LIMIT = 30;

  function renderGlobalList(q) {
    var d = cmp.global;
    if (!d) { return; }
    var GC = {};
    d.columns.forEach(function (n, i) { GC[n] = i; });
    var hits = d.rows.filter(function (r) {
      return norm(r[GC.name_en]).indexOf(q) >= 0 || norm(r[GC.name_ko]).indexOf(q) >= 0;
    });
    hits.sort(function (a, b) { return (b[GC.price] || 0) - (a[GC.price] || 0); });

    el('cmp-count-usd').textContent =
      hits.length.toLocaleString('ko-KR') + '장 중 ' +
      Math.min(LIMIT, hits.length) + '장';

    el('cmp-list-usd').innerHTML = hits.slice(0, LIMIT).map(function (r) {
      var sid = d.sets[r[GC.set]];
      var cardId = sid + '-' + r[GC.local_id];
      var g = (cmp.graded && cmp.graded[cardId]) || null;
      var psa = g ? g[cmp.gcol.psa10] : null;
      return '<button type="button" class="cmp-item" data-side="usd" data-id="' +
        esc(cardId) + '">' +
        '<b>' + esc(r[GC.name_en]) + '</b>' +
        (r[GC.name_ko] ? '<i>' + esc(r[GC.name_ko]) + '</i>' : '') +
        '<span class="cmp-set">' + esc(sid) + ' · #' + esc(r[GC.local_id]) + '</span>' +
        '<span class="cmp-p">raw ' + bothFromUsd(r[GC.price]) + '</span>' +
        (psa ? '<span class="cmp-p is-psa">PSA 10 ' + bothFromUsd(psa) + '</span>'
             : '<span class="cmp-p is-none">PSA 10 없음</span>') +
        '</button>';
    }).join('') || '<p class="pk-empty">글로벌에 없습니다.</p>';
  }

  function renderKrwList(q) {
    var hits = state.data.rows.filter(function (r) {
      return norm(r[C.name_ko]).indexOf(q) >= 0 || norm(r[C.name_en]).indexOf(q) >= 0;
    });
    hits.sort(function (a, b) { return (b[C.price] || 0) - (a[C.price] || 0); });

    el('cmp-count-krw').textContent =
      hits.length.toLocaleString('ko-KR') + '종 중 ' +
      Math.min(LIMIT, hits.length) + '종';

    el('cmp-list-krw').innerHTML = hits.slice(0, LIMIT).map(function (r, i) {
      var idx = state.data.rows.indexOf(r);
      return '<button type="button" class="cmp-item" data-side="krw" data-id="' + idx + '">' +
        '<b>' + esc(r[C.name_ko]) + '</b>' +
        '<span class="cmp-set">' + esc(r[C.lang] || '기타') + ' · ' + esc(r[C.code]) +
          ' · 30일 ' + Number(r[C.tx] || 0).toLocaleString('ko-KR') + '건</span>' +
        '<span class="cmp-p is-psa">PSA 10 ' + bothFromWon(r[C.price]) + '</span>' +
        '</button>';
    }).join('') || '<p class="pk-empty">국내에 없습니다.</p>';
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
        raw: row[GC.price], psa10: g ? g[cmp.gcol.psa10] : null
      };
    } else {
      var r2 = state.data.rows[Number(id)];
      if (!r2) { return; }
      cmp.krw = {
        title: r2[C.name_ko], sub: (r2[C.lang] || '기타') + ' · ' + r2[C.code],
        psa10won: r2[C.price], tx: r2[C.tx]
      };
    }
    renderPanel();
  }

  function renderPanel() {
    var panel = el('cmp-panel');
    if (!cmp.usd && !cmp.krw) { panel.hidden = true; return; }
    panel.hidden = false;

    el('cmp-pick-usd').innerHTML = cmp.usd
      ? '<span class="cmp-tag">글로벌 · 영문판</span>' +
        '<b>' + esc(cmp.usd.title) + '</b>' +
        '<span class="cmp-set">' + esc(cmp.usd.sub) + '</span>' +
        '<span class="cmp-p">raw ' + bothFromUsd(cmp.usd.raw) + '</span>' +
        (cmp.usd.psa10
          ? '<span class="cmp-p is-psa">PSA 10 ' + bothFromUsd(cmp.usd.psa10) + '</span>'
          : '<span class="cmp-p is-none">PSA 10 값이 아직 없습니다</span>')
      : '<span class="cmp-empty">왼쪽에서 한 장 고르세요</span>';

    el('cmp-pick-krw').innerHTML = cmp.krw
      ? '<span class="cmp-tag">국내 · PSA 10</span>' +
        '<b>' + esc(cmp.krw.title) + '</b>' +
        '<span class="cmp-set">' + esc(cmp.krw.sub) + '</span>' +
        '<span class="cmp-p is-psa">' + bothFromWon(cmp.krw.psa10won) + '</span>' +
        '<span class="cmp-set">30일 거래 ' +
          Number(cmp.krw.tx || 0).toLocaleString('ko-KR') + '건</span>'
      : '<span class="cmp-empty">오른쪽에서 한 종 고르세요</span>';

    el('cmp-gap').innerHTML = gapHtml();
  }

  /* 같은 등급끼리만 뺀다. 글로벌에 PSA 10 이 없으면 뺄 것이 없다고 적는다 —
     raw 와 PSA 10 을 빼면 그 차이는 나라 차이가 아니라 등급 프리미엄이다. */
  function gapHtml() {
    if (!cmp.usd || !cmp.krw) {
      return '<span class="cmp-note">양쪽에서 하나씩 고르면 차이를 계산합니다.</span>';
    }
    if (cmp.usd.psa10 === null || cmp.usd.psa10 === undefined) {
      return '<span class="cmp-note">이 글로벌 카드는 <b>PSA 10 값이 없어</b> 뺄 수 없습니다.<br>' +
        'raw 와 PSA 10 을 빼면 나라 차이가 아니라 등급 프리미엄이 나옵니다.</span>';
    }
    var a = usdToWon(cmp.usd.psa10);
    var b = cmp.krw.psa10won;
    if (!a) { return '<span class="cmp-note">환율을 불러오지 못했습니다.</span>'; }
    var diff = b - a;
    var pct = (diff / a) * 100;
    var up = diff > 0;
    return '<span class="cmp-note">둘 다 PSA 10 이라 비교됩니다</span>' +
      '<b class="cmp-diff ' + (up ? 'is-up' : 'is-down') + '">' +
        (up ? '국내가 ' : '국내가 ') +
        Math.abs(pct).toFixed(1) + '% ' + (up ? '비쌉니다' : '쌉니다') + '</b>' +
      '<span class="cmp-note">차이 ' +
        (up ? '+' : '−') + Math.abs(Math.round(diff)).toLocaleString('ko-KR') + '원</span>' +
      '<span class="cmp-note cmp-caveat">영문판과 일본판은 다른 카드입니다. ' +
        '수수료·감정료·관세는 빠져 있습니다.</span>';
  }

  function bindCompare() {
    el('cmp-q').addEventListener('input', debounce(cmpSearch, 200));
    ['cmp-list-usd', 'cmp-list-krw'].forEach(function (id) {
      el(id).addEventListener('click', function (ev) {
        var btn = ev.target.closest ? ev.target.closest('.cmp-item') : null;
        if (!btn) { return; }
        var list = el(id);
        Array.prototype.forEach.call(list.querySelectorAll('.cmp-item'), function (b) {
          b.classList.remove('is-on');
        });
        btn.classList.add('is-on');
        pickCard(btn.dataset.side, btn.dataset.id);
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
      el('cmp-meta').textContent = fx.rate
        ? '환산 환율 USD 1 = ' + fx.rate.toLocaleString('ko-KR') + '원 (' +
          fx.date + ' 기준 · ' + fx.source + '). 글로벌 PSA 10 이 붙은 카드는 ' +
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
