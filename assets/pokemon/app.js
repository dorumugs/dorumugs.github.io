/* 포켓몬 카드 시세 브라우저.
   외부 라이브러리를 쓰지 않는다. 카드가 1만 7천 장이라 렌더는 페이지 단위로
   끊고, 이미지는 지연 로딩한다. */
(function () {
  'use strict';

  var app = document.querySelector('.pk-app');
  if (!app) { return; }
  var BASE = app.dataset.base;
  var PAGE = 60;

  var state = {
    cols: null, prefix: '', rows: [], sets: {}, meta: null,
    filtered: [], shown: PAGE
  };
  var C = {};

  function fetchJson(name) {
    return fetch(BASE + '/' + name + '.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) { throw new Error(name + ' ' + r.status); }
      return r.json();
    });
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function money(v, unit) {
    if (v === null || v === undefined) { return '—'; }
    return (unit || '$') + Number(v).toLocaleString('ko-KR',
      { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  /* 검색어 정규화. 공백을 지우고 소문자로 — '리자몽 ex' 와 'charizardex' 둘 다 잡는다. */
  function norm(s) {
    return String(s || '').toLowerCase().replace(/\s+/g, '');
  }

  function applyFilters() {
    var q = norm(document.getElementById('pk-q').value);
    var setId = document.getElementById('pk-set').value;
    var era = document.getElementById('pk-era').value;
    var sort = document.getElementById('pk-sort').value;

    var out = state.rows.filter(function (r) {
      if (setId && r[C.set_id] !== setId) { return false; }
      if (era) {
        var s = state.sets[r[C.set_id]];
        if (!s || s.era !== era) { return false; }
      }
      if (q) {
        if (norm(r[C.name_en]).indexOf(q) < 0 && norm(r[C.name_ko]).indexOf(q) < 0) {
          return false;
        }
      }
      return true;
    });

    if (sort === 'price-asc') {
      out.sort(function (a, b) {
        return (a[C.price] === null) - (b[C.price] === null) ||
          (a[C.price] || 0) - (b[C.price] || 0);
      });
    } else if (sort === 'name') {
      out.sort(function (a, b) { return a[C.name_en].localeCompare(b[C.name_en]); });
    } else if (sort === 'obs-desc') {
      out.sort(function (a, b) { return (b[C.obs_max] || 0) - (a[C.obs_max] || 0); });
    } else {
      out.sort(function (a, b) { return (b[C.price] || 0) - (a[C.price] || 0); });
    }

    state.filtered = out;
    state.shown = PAGE;
    render();
  }

  function cardHtml(r) {
    var set = state.sets[r[C.set_id]] || {};
    var img = r[C.image]
      ? state.prefix + r[C.image] + '/low.webp'
      : '';
    var full = r[C.image] ? state.prefix + r[C.image] + '/high.webp' : '';

    var thumb = img
      ? '<img class="pk-img" src="' + esc(img) + '" alt="' + esc(r[C.name_en]) +
        '" loading="lazy" width="245" height="342" onerror="this.classList.add(\'is-broken\')">'
      : '<div class="pk-img is-none">이미지 없음</div>';

    return '<article class="pk-card">' +
      (full ? '<a class="pk-imgwrap" href="' + esc(full) + '" target="_blank" rel="noopener">' + thumb + '</a>'
            : '<div class="pk-imgwrap">' + thumb + '</div>') +
      '<div class="pk-body">' +
        '<h3 class="pk-name">' + esc(r[C.name_en]) + '</h3>' +
        (r[C.name_ko] ? '<p class="pk-ko">' + esc(r[C.name_ko]) + '</p>' : '') +
        '<p class="pk-set">' + esc(set.name || r[C.set_id]) + ' · #' + esc(r[C.local_id]) +
          (r[C.rarity] ? ' · ' + esc(r[C.rarity]) : '') + '</p>' +
        '<p class="pk-price">' + money(r[C.price]) + '</p>' +
        '<dl class="pk-sub">' +
          '<dt>최고 호가</dt><dd>' + money(r[C.high_ask]) + '</dd>' +
          '<dt>관측 최고가</dt><dd>' + money(r[C.obs_max]) +
            (r[C.obs_max_date] ? ' <span class="pk-when">' + esc(r[C.obs_max_date]) + '</span>' : '') + '</dd>' +
          '<dt>Cardmarket</dt><dd>' + money(r[C.cm_avg], '€') + '</dd>' +
        '</dl>' +
      '</div>' +
    '</article>';
  }

  function render() {
    var grid = document.getElementById('pk-grid');
    var slice = state.filtered.slice(0, state.shown);
    grid.innerHTML = slice.length
      ? slice.map(cardHtml).join('')
      : '<p class="pk-empty">찾는 카드가 없습니다. 철자나 필터를 확인해 보세요.</p>';

    document.getElementById('pk-count').textContent =
      state.filtered.length.toLocaleString('ko-KR') + '장 중 ' +
      slice.length.toLocaleString('ko-KR') + '장 표시';

    var more = document.getElementById('pk-more');
    more.style.display = state.filtered.length > state.shown ? '' : 'none';
  }

  function fillFilters() {
    var setSel = document.getElementById('pk-set');
    var eraSel = document.getElementById('pk-era');
    var eras = [];
    Object.keys(state.sets).forEach(function (sid) {
      var s = state.sets[sid];
      setSel.insertAdjacentHTML('beforeend',
        '<option value="' + esc(sid) + '">' + esc(s.name) + ' (' + s.count + ')</option>');
      if (s.era && eras.indexOf(s.era) < 0) { eras.push(s.era); }
    });
    ['빈티지', '클래식', '모던', '최신'].forEach(function (e) {
      if (eras.indexOf(e) >= 0) {
        eraSel.insertAdjacentHTML('beforeend', '<option>' + esc(e) + '</option>');
      }
    });
  }

  function debounce(fn, ms) {
    var t;
    return function () { clearTimeout(t); t = setTimeout(fn, ms); };
  }

  function bind() {
    document.getElementById('pk-q').addEventListener('input', debounce(applyFilters, 180));
    ['pk-set', 'pk-era', 'pk-sort'].forEach(function (id) {
      document.getElementById(id).addEventListener('change', applyFilters);
    });
    document.getElementById('pk-more').addEventListener('click', function () {
      state.shown += PAGE;
      render();
    });
  }

  Promise.all([fetchJson('cards'), fetchJson('sets'), fetchJson('meta')])
    .then(function (res) {
      var payload = res[0];
      state.cols = payload.columns;
      state.prefix = payload.image_prefix;
      state.rows = payload.rows;
      state.sets = res[1];
      state.meta = res[2];
      payload.columns.forEach(function (name, i) { C[name] = i; });

      fillFilters();
      bind();
      applyFilters();

      document.getElementById('pk-meta').textContent =
        '카드 ' + state.meta.card_count.toLocaleString('ko-KR') + '장 · 세트 ' +
        state.meta.set_count + '개 · 한글 이름이 붙은 카드 ' +
        state.meta.with_korean_name.toLocaleString('ko-KR') + '장 · 마지막 갱신 ' +
        state.meta.generated + '.';
    })
    .catch(function (err) {
      document.getElementById('pk-count').textContent =
        '데이터를 불러오지 못했습니다. (' + err.message + ')';
    });
})();
