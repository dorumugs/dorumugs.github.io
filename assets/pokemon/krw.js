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
    data: null, meta: null, loading: false, filtered: [], shown: PAGE,
    currency: 'KRW',  // 목록에 원화와 함께 보일 통화
    view: 'grid',     // 'grid' 카드형 | 'table' 표형
    merge: false      // 같은 품번의 언어판을 한 줄로 묶을지
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

  /* ---- 상품 목록 --------------------------------------------------------

     언어판 묶기: `S7R-083-067_JP` 와 `S7R083-067_KR` 은 같은 카드다. 품번에서
     언어 접미사와 기호만 떼면 같아지는 **하드 키**라 틀릴 여지가 없다.
     묶어 놓으면 일어판이 한글판의 3~10배인 것이 바로 보인다 — 흩어 놓으면
     안 보이던 사실이다. 이름으로 묶으면 안 된다 ("리자몽" 하나에 45종). */

  function baseCode(code) {
    return String(code || '').replace(/_(JP|KR|EN)$/i, '').replace(/[^A-Za-z0-9]/g, '')
      .toUpperCase();
  }

  function stripLang(name) {
    return String(name || '').replace(/\s*\((?:일어판|한글판|영문판|기타)\)\s*$/, '').trim();
  }

  /* [{rows: [상품…], name, code, top, ratio}] — 묶지 않으면 한 줄에 하나씩. */
  function grouped(rows) {
    if (!state.merge) {
      return rows.map(function (r) {
        return { rows: [r], name: r[C.name_ko], code: r[C.code],
                 top: r[C.price] || 0, ratio: null };
      });
    }
    var byKey = {}, order = [];
    rows.forEach(function (r) {
      var key = baseCode(r[C.code]) || r[C.code];
      if (!byKey[key]) { byKey[key] = { rows: [], key: key }; order.push(key); }
      byKey[key].rows.push(r);
    });
    return order.map(function (key) {
      var g = byKey[key];
      var prices = g.rows.map(function (r) { return r[C.price] || 0; })
        .filter(function (p) { return p > 0; });
      var top = prices.length ? Math.max.apply(null, prices) : 0;
      var low = prices.length ? Math.min.apply(null, prices) : 0;
      return {
        rows: g.rows,
        name: stripLang(g.rows[0][C.name_ko]),
        code: key,
        top: top,
        ratio: (prices.length > 1 && low) ? Math.round((top / low) * 10) / 10 : null
      };
    });
  }

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

    var groups = grouped(out);
    if (sort === 'gap-desc') {
      /* 격차가 없는 줄(언어판 하나)은 뒤로. 0 으로 쳐서 섞으면 순위가 거짓이 된다. */
      groups.sort(function (a, b) {
        if (!a.ratio && !b.ratio) { return 0; }
        if (!a.ratio) { return 1; }
        if (!b.ratio) { return -1; }
        return b.ratio - a.ratio;
      });
    } else if (state.merge && (sort === 'price-desc' || sort === 'price-asc')) {
      var dir = sort === 'price-asc' ? 1 : -1;
      groups.sort(function (a, b) { return dir * (b.top - a.top); });
    }

    state.filtered = groups;
    state.shown = PAGE;
    render();
  }

  /* 값 하나를 원화 + (고른 통화) 로. 넷을 다 넣으면 카드가 못 읽힌다. */
  function money2(w) {
    if (w === null || w === undefined) { return '—'; }
    var pick = state.currency;
    if (pick === 'KRW') { return fmt(w, 'KRW'); }
    return fmt(w, 'KRW') + '<small>' + fmt(fromUsd(toUsd(w, 'KRW'), pick), pick) + '</small>';
  }

  function cardHtml(g) {
    var r = g.rows[0];
    var pic = imageOf(r);
    var change = pct(r[C.change_30d]);
    var thumb = '<span class="pk-noimg">이미지 없음</span>' + (pic
      ? '<img class="pk-img" src="' + esc(pic.thumb) + '" alt="' + esc(g.name) +
        '" loading="lazy" width="525" height="525" referrerpolicy="no-referrer"' +
        ' onerror="this.style.display=\'none\'">'
      : '');

    /* 묶인 줄은 언어판마다 값을 따로 적는다. 하나를 골라 대표로 쓰면 그 선택이
       곧 결론을 바꾼다. */
    var prices = g.rows.length > 1
      ? g.rows.map(function (x) {
          return row(x[C.lang] || '기타', won(x[C.price]));
        }).join('')
      : '';

    return '<article class="pk-card">' +
      (pic
        ? '<a class="pk-imgwrap krw-imgwrap" href="' + esc(pic.full) +
          '" target="_blank" rel="noopener noreferrer">' + thumb + '</a>'
        : '<div class="pk-imgwrap krw-imgwrap">' + thumb + '</div>') +
      '<div class="pk-body">' +
        '<h3 class="pk-name">' + esc(g.name) + '</h3>' +
        (g.rows.length > 1 ? '' : '<p class="pk-ko krw-en">' + esc(r[C.name_en]) + '</p>') +
        '<p class="pk-set">' +
          (g.rows.length > 1
            ? g.rows.map(function (x) { return esc(x[C.lang] || '기타'); }).join(' · ')
            : esc(r[C.lang] || '기타')) +
          ' · ' + esc(g.code) + '</p>' +
        (g.rows.length > 1
          ? '<div class="pk-sub">' + prices +
            (g.ratio ? row('언어판 격차', g.ratio + '배') : '') + '</div>'
          : '<p class="pk-price">' + money2(r[C.price]) +
            ' <span class="krw-chg ' + change.cls + '">' + esc(change.text) + '</span></p>' +
            '<div class="pk-sub">' +
              row('30일 고가', won(r[C.high_30d])) +
              row('30일 저가', won(r[C.low_30d])) +
              row('30일 거래', Number(r[C.tx] || 0).toLocaleString('ko-KR') + '건') +
            '</div>') +
      '</div>' +
    '</article>';
  }

  function row(label, value) {
    return '<div class="pk-row"><span class="pk-lbl">' + esc(label) +
      '</span><span class="pk-val">' + value + '</span></div>';
  }

  /* 표형 한 줄. 묶인 줄은 언어판을 세로로 편다. */
  function rowHtml(g) {
    var r = g.rows[0];
    var pic = imageOf(r);
    var thumb = pic
      ? '<a href="' + esc(pic.full) + '" target="_blank" rel="noopener noreferrer">' +
        '<img class="pk-thumb" src="' + esc(pic.thumb) + '" alt="' + esc(g.name) +
        '" loading="lazy" width="44" height="44" referrerpolicy="no-referrer"' +
        ' onerror="this.style.display=\'none\'"></a>'
      : '<span class="pk-thumb is-none">—</span>';

    function stack(fn) {
      return g.rows.map(fn).join('<hr class="uni-sep">');
    }

    return '<tr>' +
      '<td class="pk-thumbcell">' + thumb + '</td>' +
      '<td><b>' + esc(g.name) + '</b>' +
        (g.rows.length > 1 ? '' : '<i>' + esc(r[C.name_en]) + '</i>') + '</td>' +
      '<td class="cmp-dim">' + stack(function (x) {
          return esc(x[C.lang] || '기타');
        }) + '<i>' + esc(g.code) + '</i></td>' +
      '<td class="cmp-num is-psa">' + stack(function (x) {
          return money2(x[C.price]);
        }) + '</td>' +
      '<td class="cmp-num">' + stack(function (x) {
          var c = pct(x[C.change_30d]);
          return '<span class="krw-chg ' + c.cls + '">' + esc(c.text) + '</span>';
        }) + '</td>' +
      '<td class="cmp-num">' + stack(function (x) {
          var n = Number(x[C.tx] || 0);
          return '<span class="' + (n <= 1 ? 'is-thin' : '') + '">' +
            n.toLocaleString('ko-KR') + '건</span>';
        }) + '</td>' +
      '<td class="cmp-num">' + (g.ratio
        ? '<b class="cmp-diff is-up">' + g.ratio + '배</b>'
        : '<span class="cmp-none">—</span>') + '</td>' +
      '</tr>';
  }

  function render() {
    var grid = el('krw-grid');
    var wrap = el('krw-tablewrap');
    var slice = state.filtered.slice(0, state.shown);
    var table = state.view === 'table';

    grid.hidden = table;
    wrap.hidden = !table;

    if (table) {
      el('krw-tbody').innerHTML = slice.length
        ? slice.map(rowHtml).join('')
        : '<tr><td colspan="7" class="pk-empty">찾는 카드가 없습니다.</td></tr>';
    } else {
      grid.innerHTML = slice.length
        ? slice.map(cardHtml).join('')
        : '<p class="pk-empty">찾는 카드가 없습니다. 철자나 필터를 확인해 보세요.</p>';
    }

    el('krw-count').textContent =
      state.filtered.length.toLocaleString('ko-KR') + (state.merge ? '줄 중 ' : '종 중 ') +
      slice.length.toLocaleString('ko-KR') + (state.merge ? '줄 표시' : '종 표시');

    el('krw-more').style.display =
      state.filtered.length > state.shown ? '' : 'none';
  }

  function setLive(key, text) {
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-live="' + key + '"]'), function (e) {
        e.textContent = text;
      });
  }

  /* 본문 숫자를 데이터에서 그린다. 상품 수는 매일 바뀐다 (899 -> 903). */
  function fillLive() {
    var s = state.data.stats;
    var total = s.product_count;
    var langs = Object.keys(s.languages).map(function (k) {
      return k + ' ' + s.languages[k].toLocaleString('ko-KR') + '종';
    }).join(' · ');
    setLive('langs', total.toLocaleString('ko-KR') + '종 가운데 ' + langs + '입니다.');
    setLive('thin', total.toLocaleString('ko-KR') + '종 가운데 ' +
      s.thin.toLocaleString('ko-KR') + '종은 30일 거래가 1건 이하입니다.');

    var merged = 0;
    var seen = {};
    state.data.rows.forEach(function (r) {
      var key = baseCode(r[C.code]) || r[C.code];
      if (seen[key] === 1) { merged++; }
      seen[key] = (seen[key] || 0) + 1;
    });
    setLive('merged', merged.toLocaleString('ko-KR') + '줄');
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
    ['krw-lang', 'krw-sort', 'krw-cur'].forEach(function (id) {
      el(id).addEventListener('change', function () {
        if (id === 'krw-cur') { state.currency = el('krw-cur').value; }
        applyFilters();
      });
    });
    ['krw-liquid', 'krw-merge'].forEach(function (id) {
      el(id).addEventListener('change', function () {
        if (id === 'krw-merge') { state.merge = el('krw-merge').checked; }
        applyFilters();
      });
    });
    Array.prototype.forEach.call(
      document.querySelectorAll('input[name="krw-view"]'), function (radio) {
        radio.addEventListener('change', function () {
          state.view = radio.value;
          render();
        });
      });
    el('krw-more').addEventListener('click', function () {
      state.shown += PAGE;
      render();
    });
  }

  function load() {
    if (state.data || state.loading) { return; }
    state.loading = true;
    Promise.all([
      fetch(BASE + '/krw.json', { cache: 'no-cache' }).then(function (r) {
        if (!r.ok) { throw new Error('krw ' + r.status); }
        return r.json();
      }),
      fetch(BASE + '/meta.json', { cache: 'no-cache' }).then(function (r) { return r.json(); })
    ])
      .then(function (res) {
        var data = res[0];
        state.meta = res[1];
        state.data = data;
        data.columns.forEach(function (name, i) { C[name] = i; });
        data.market_columns.forEach(function (name, i) { M[name] = i; });
        fillStats();
        fillLive();
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
    var fx = (state.meta && state.meta.fx) || {};
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

  /* 목록용 — 원화와 사용자가 고른 통화 둘만. 넷을 다 넣으면 표가 못 읽힌다. */
  function pairFromWon(won) {
    if (won === null || won === undefined) { return '—'; }
    var pick = state.currency;
    if (pick === 'KRW') { return fmt(won, 'KRW'); }
    return fmt(won, 'KRW') +
      '<small>' + fmt(fromUsd(toUsd(won, 'KRW'), pick), pick) + '</small>';
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
      history.replaceState(null, '',
        which === 'usd' ? location.pathname : '#' + which);
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
