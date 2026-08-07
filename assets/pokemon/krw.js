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

  /* ---- 탭 --------------------------------------------------------------- */

  function activate(which) {
    ['usd', 'krw'].forEach(function (name) {
      var on = name === which;
      var tab = el('pk-tab-' + name);
      var view = el('pk-view-' + name);
      if (!tab || !view) { return; }
      tab.classList.toggle('is-on', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
      view.hidden = !on;
    });
    if (which === 'krw') { load(); }
    if (history.replaceState) {
      history.replaceState(null, '', which === 'krw' ? '#krw' : location.pathname);
    }
  }

  ['usd', 'krw'].forEach(function (name) {
    var tab = el('pk-tab-' + name);
    if (tab) {
      tab.addEventListener('click', function () { activate(name); });
    }
  });

  if (location.hash === '#krw') { activate('krw'); }
})();
