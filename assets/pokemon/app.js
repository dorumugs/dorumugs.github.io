/* 포켓몬 카드 지수 대시보드.
   외부 라이브러리를 쓰지 않는다. 차트는 SVG 를 직접 그린다. */
(function () {
  'use strict';

  var app = document.querySelector('.pk-app');
  if (!app) { return; }
  var BASE = app.dataset.base;

  var COLORS = ['#2c3e50', '#c0392b', '#27ae60', '#8e44ad', '#e67e22', '#16a085'];
  var PAGE = 50;

  var state = { index: null, universe: null, meta: null, series: 'index', shown: PAGE };

  function fetchJson(name) {
    return fetch(BASE + '/' + name + '.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) { throw new Error(name + ' ' + r.status); }
      return r.json();
    });
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fmt(n, digits) {
    if (n === null || n === undefined) { return '—'; }
    return Number(n).toLocaleString('ko-KR', {
      minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0
    });
  }

  function pct(now, before) {
    if (!before || now === null || now === undefined) { return null; }
    return (now / before - 1) * 100;
  }

  function signed(p) {
    if (p === null || p === undefined) { return '—'; }
    return (p >= 0 ? '+' : '') + p.toFixed(1) + '%';
  }

  function lastValid(arr) {
    for (var i = arr.length - 1; i >= 0; i--) {
      if (arr[i] !== null && arr[i] !== undefined) { return arr[i]; }
    }
    return null;
  }

  /* n 일 전 값. 관측이 그만큼 쌓이지 않았으면 가장 오래된 값을 쓴다. */
  function valueDaysAgo(series, days) {
    if (!series.length) { return null; }
    var i = series.length - 1 - days;
    return series[i < 0 ? 0 : i];
  }

  function renderStats() {
    var idx = state.index, meta = state.meta;
    var series = idx.index || [];
    var now = lastValid(series);

    var cards = [
      { v: fmt(now, 1), label: '현재 지수 (기준일 ' + idx.base_date + ' = 100)', delta: null },
      { v: signed(pct(now, valueDaysAgo(series, 7))), label: '7일 변화',
        delta: pct(now, valueDaysAgo(series, 7)) },
      { v: signed(pct(now, valueDaysAgo(series, 30))), label: '30일 변화',
        delta: pct(now, valueDaysAgo(series, 30)) },
      { v: fmt(meta.card_count) + '장', label: '구성 종목 (12칸 × ' + meta.per_cell + ')', delta: null },
      { v: fmt(meta.days) + '일', label: '누적 관측', delta: null },
      { v: fmt(meta.missing) + '장', label: '오늘 결측', delta: null }
    ];

    document.getElementById('pk-stats').innerHTML = cards.map(function (c) {
      var cls = c.delta === null || c.delta === undefined ? '' : (c.delta >= 0 ? ' is-up' : ' is-down');
      return '<div class="pk-stat' + cls + '"><b>' + esc(c.v) + '</b><span>' + esc(c.label) + '</span></div>';
    }).join('');
  }

  function seriesToDraw() {
    var idx = state.index;
    if (state.series === 'era') {
      return Object.keys(idx.by_era).map(function (k, i) {
        return { name: k, values: idx.by_era[k], color: COLORS[i % COLORS.length] };
      });
    }
    if (state.series === 'band') {
      return Object.keys(idx.by_band).map(function (k, i) {
        return { name: k, values: idx.by_band[k], color: COLORS[i % COLORS.length] };
      });
    }
    return [{ name: '전체 지수', values: idx.index, color: COLORS[0] }];
  }

  function drawChart() {
    var svg = document.getElementById('pk-chart');
    var idx = state.index;
    var lines = seriesToDraw();
    var W = 720, H = 340, P = { t: 14, r: 14, b: 26, l: 44 };

    var showBack = state.series === 'index' &&
      idx.backcast && idx.backcast.dates && idx.backcast.dates.length > 1;
    var back = showBack ? idx.backcast : null;

    /* 소급 구간의 마지막 점은 기준일과 같은 날이므로 겹치지 않게 뺀다. */
    var allDates = (back ? back.dates.slice(0, -1) : []).concat(idx.dates);
    var offset = back ? back.dates.length - 1 : 0;

    var vals = [];
    lines.forEach(function (l) {
      l.values.forEach(function (v) { if (v !== null && v !== undefined) { vals.push(v); } });
    });
    if (back) { back.index.forEach(function (v) { vals.push(v); }); }
    if (!vals.length) { svg.innerHTML = ''; return; }

    var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
    var pad = (max - min) * 0.12 || 5;
    min -= pad; max += pad;

    function x(i) {
      return P.l + (W - P.l - P.r) * (allDates.length < 2 ? 0.5 : i / (allDates.length - 1));
    }
    function y(v) { return P.t + (H - P.t - P.b) * (1 - (v - min) / (max - min)); }

    var parts = [];

    for (var g = 0; g <= 4; g++) {
      var gv = min + (max - min) * g / 4;
      parts.push('<line class="grid" x1="' + P.l + '" y1="' + y(gv).toFixed(1) +
        '" x2="' + (W - P.r) + '" y2="' + y(gv).toFixed(1) + '"/>');
      parts.push('<text class="axis" x="4" y="' + (y(gv) + 4).toFixed(1) + '">' + gv.toFixed(0) + '</text>');
    }

    if (back) {
      var bp = back.dates.map(function (_, i) {
        return (i === 0 ? 'M' : 'L') + x(i).toFixed(1) + ' ' + y(back.index[i]).toFixed(1);
      }).join(' ');
      parts.push('<path class="line is-est" d="' + bp + '" stroke="' + COLORS[0] + '"/>');
    }

    lines.forEach(function (l) {
      var d = '', started = false;
      l.values.forEach(function (v, i) {
        if (v === null || v === undefined) { return; }
        d += (started ? ' L' : 'M') + x(i + offset).toFixed(1) + ' ' + y(v).toFixed(1);
        started = true;
      });
      if (d) { parts.push('<path class="line" d="' + d + '" stroke="' + l.color + '"/>'); }
    });

    [0, Math.floor(allDates.length / 2), allDates.length - 1].forEach(function (i, k) {
      if (i < 0 || i >= allDates.length) { return; }
      var anchor = k === 0 ? 'start' : (k === 2 ? 'end' : 'middle');
      parts.push('<text class="axis" text-anchor="' + anchor + '" x="' + x(i).toFixed(1) +
        '" y="' + (H - 8) + '">' + esc(allDates[i].slice(5)) + '</text>');
    });

    svg.innerHTML = parts.join('');

    document.getElementById('pk-legend').innerHTML =
      lines.map(function (l) {
        return '<span><i style="background:' + l.color + '"></i>' + esc(l.name) + '</span>';
      }).join('') +
      (back
        ? '<span class="is-est-note">점선 = Cardmarket 7·30일 평균으로 만든 <strong>추정</strong> 소급 구간</span>'
        : '');

    document.getElementById('pk-note').textContent =
      '기준일 ' + state.meta.base_date + ' = 100. 실선은 실측분입니다. ' +
      '가격이 ' + state.meta.carry_forward_days + '일 넘게 결측된 카드는 지수에서 빠집니다. ' +
      '마지막 갱신 ' + state.meta.generated + '.';
  }

  function renderTable() {
    var era = document.getElementById('pk-era').value;
    var band = document.getElementById('pk-band').value;
    var rows = state.universe.cards.filter(function (c) {
      return (!era || c.era === era) && (!band || c.band === band);
    });

    var body = rows.slice(0, state.shown).map(function (c) {
      var p = pct(c.price, c.base_price);
      var cls = p === null ? '' : (p >= 0 ? 'up' : 'down');
      /* 모바일에서 표가 자체 스크롤되므로 중요한 열(현재가·변화)을 앞에 둔다. */
      return '<tr>' +
        '<td>' + esc(c.name) + '</td>' +
        '<td class="is-num">' + (c.price ? '$' + fmt(c.price, 2) : '—') + '</td>' +
        '<td class="is-num ' + cls + '">' + signed(p) + '</td>' +
        '<td class="is-num">$' + fmt(c.base_price, 2) + '</td>' +
        '<td>' + esc(c.era) + '</td>' +
        '<td>' + esc(c.band) + '</td>' +
        '<td>' + esc(c.set_name) + '</td>' +
        '</tr>';
    }).join('');

    document.querySelector('#pk-table tbody').innerHTML = body ||
      '<tr><td colspan="7">해당하는 카드가 없습니다.</td></tr>';

    var more = document.getElementById('pk-more');
    more.style.display = rows.length > state.shown ? '' : 'none';
    more.textContent = '더 보기 (' + Math.min(state.shown, rows.length) + ' / ' + rows.length + ')';
  }

  function fillFilters() {
    var eras = [], bands = [];
    state.universe.cards.forEach(function (c) {
      if (eras.indexOf(c.era) < 0) { eras.push(c.era); }
      if (bands.indexOf(c.band) < 0) { bands.push(c.band); }
    });
    var es = document.getElementById('pk-era'), bs = document.getElementById('pk-band');
    eras.forEach(function (e) { es.insertAdjacentHTML('beforeend', '<option>' + esc(e) + '</option>'); });
    bands.forEach(function (b) { bs.insertAdjacentHTML('beforeend', '<option>' + esc(b) + '</option>'); });
    es.addEventListener('change', function () { state.shown = PAGE; renderTable(); });
    bs.addEventListener('change', function () { state.shown = PAGE; renderTable(); });
  }

  function bindTabs() {
    Array.prototype.forEach.call(document.querySelectorAll('.pk-tab'), function (tab) {
      tab.addEventListener('click', function () {
        Array.prototype.forEach.call(document.querySelectorAll('.pk-tab'), function (t) {
          t.classList.remove('is-on');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('is-on');
        tab.setAttribute('aria-selected', 'true');
        state.series = tab.dataset.series;
        drawChart();
      });
    });
    document.getElementById('pk-more').addEventListener('click', function () {
      state.shown += PAGE;
      renderTable();
    });
  }

  Promise.all([fetchJson('index'), fetchJson('universe'), fetchJson('meta')])
    .then(function (res) {
      state.index = res[0];
      state.universe = res[1];
      state.meta = res[2];
      document.getElementById('pk-formula').textContent = state.meta.formula;
      renderStats();
      fillFilters();
      bindTabs();
      drawChart();
      renderTable();
    })
    .catch(function (err) {
      document.getElementById('pk-stats').innerHTML =
        '<p class="pk-note">데이터를 아직 불러올 수 없습니다. (' + esc(err.message) + ')</p>';
    });
})();
