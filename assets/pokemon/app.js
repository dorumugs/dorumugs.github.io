/* 포켓몬 카드 시세 브라우저.
   외부 라이브러리를 쓰지 않는다. 카드가 1만 7천 장이라 렌더는 페이지 단위로
   끊고, 이미지는 지연 로딩한다.

   cards.json 은 사전 인코딩돼 있다 — 세트·등급·날짜는 번호로 오고, card_id 와
   이미지 경로는 아예 없다. 규칙이 있어서 여기서 만든다. */
(function () {
  'use strict';

  var app = document.querySelector('.pk-app');
  if (!app) { return; }
  var BASE = app.dataset.base;
  var PAGE = 60;

  var state = {
    prefix: '', tcgPrefix: '', ptcgPrefix: '', sets: [], series: [],
    rarities: [], dates: [], rows: [], setInfo: {}, art: {}, meta: null,
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

  function setIdOf(r) { return state.sets[r[C.set]] || ''; }
  function cardIdOf(r) { return setIdOf(r) + '-' + r[C.local_id]; }

  /* 이미지 주소를 만든다. {thumb, full} 이거나, 구할 수 없으면 null.

     출처가 셋이고 순서대로 떨어진다.

       1. TCGdex — 경로가 '시리즈/세트/번호' 로 규칙적이다 (전수 확인).
       2. TCGplayer — TCGdex 가 이미지를 안 주는 588장을 productId 로 메운다.
       3. pokemontcg.io — 위 둘 다 없는 세트(Shiny Vault·Galarian Gallery·
          Trainer Gallery)용. 여기만은 규칙으로 만들지 않고 art.json 에
          '실제로 있는 것만' 적어 두었다. 없는 이미지를 요청하지 않기 위해서다.

     셋 다 없는 카드가 39장 남는다 (My First Battle, Poké Card Creator Pack). */
  function imageOf(r) {
    var serie = state.series[r[C.set]];
    if (serie) {
      var path = state.prefix + serie + '/' + setIdOf(r) + '/' + r[C.local_id];
      return { thumb: path + '/low.webp', full: path + '/high.webp' };
    }
    var pid = r[C.tcg_pid];
    if (pid) {
      return {
        thumb: state.tcgPrefix + pid + '_200w.jpg',
        full: state.tcgPrefix + pid + '_in_1000x1000.jpg'
      };
    }
    var alt = state.art[cardIdOf(r)];
    if (alt) {
      return {
        thumb: state.ptcgPrefix + alt + '.png',
        full: state.ptcgPrefix + alt + '_hires.png'
      };
    }
    return null;
  }

  function applyFilters() {
    var q = norm(document.getElementById('pk-q').value);
    var setId = document.getElementById('pk-set').value;
    var era = document.getElementById('pk-era').value;
    var sort = document.getElementById('pk-sort').value;

    var out = state.rows.filter(function (r) {
      if (setId && setIdOf(r) !== setId) { return false; }
      if (era) {
        var s = state.setInfo[setIdOf(r)];
        if (!s || s.era !== era) { return false; }
      }
      if (q && norm(r[C.name_en]).indexOf(q) < 0 && norm(r[C.name_ko]).indexOf(q) < 0) {
        return false;
      }
      return true;
    });

    if (sort === 'price-asc') {
      out.sort(function (a, b) {
        var av = a[C.price], bv = b[C.price];
        if (av === null) { return 1; }
        if (bv === null) { return -1; }
        return av - bv;
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

  /* 라벨과 값을 한 줄에. 카드 폭이 좁아 둘이 안 들어가면 값이 아랫줄로
     내려가되 금액 자체는 절대 쪼개지지 않는다 ('$4,590.6 / 3' 방지). */
  function subRow(label, value, when) {
    return '<div class="pk-row">' +
      '<span class="pk-lbl">' + esc(label) + '</span>' +
      '<span class="pk-val">' + esc(value) +
        (when ? '<i class="pk-when">' + esc(when) + '</i>' : '') +
      '</span></div>';
  }

  function cardHtml(r) {
    var setId = setIdOf(r);
    var info = state.setInfo[setId] || {};
    var pic = imageOf(r);
    var rarity = state.rarities[r[C.rarity]] || '';
    var when = state.dates[r[C.obs_date]] || '';

    /* 자리표시를 항상 뒤에 깔고 이미지를 그 위에 올린다. 경로는 있는데 CDN 에
       파일이 없는 카드도 있어서, 실패하면 이미지만 사라지고 자리표시가 드러난다. */
    var thumb = '<span class="pk-noimg">이미지 없음</span>' + (pic
      ? '<img class="pk-img" src="' + esc(pic.thumb) +
        '" alt="' + esc(r[C.name_en]) + '" loading="lazy" width="245" height="342"' +
        ' onerror="this.style.display=\'none\'">'
      : '');

    return '<article class="pk-card">' +
      (pic
        ? '<a class="pk-imgwrap" href="' + esc(pic.full) +
          '" target="_blank" rel="noopener">' + thumb + '</a>'
        : '<div class="pk-imgwrap">' + thumb + '</div>') +
      '<div class="pk-body">' +
        '<h3 class="pk-name">' + esc(r[C.name_en]) + '</h3>' +
        (r[C.name_ko] ? '<p class="pk-ko">' + esc(r[C.name_ko]) + '</p>' : '') +
        '<p class="pk-set">' + esc(info.name || setId) + ' · #' + esc(r[C.local_id]) +
          (rarity ? ' · ' + esc(rarity) : '') + '</p>' +
        '<p class="pk-price">' + money(r[C.price]) + '</p>' +
        '<div class="pk-sub">' +
          subRow('최고 호가', money(r[C.high_ask])) +
          subRow('관측 최고가', money(r[C.obs_max]),
                 r[C.obs_max] !== null && r[C.obs_max] !== undefined ? when : '') +
          subRow('EUR 평균', money(r[C.cm_avg], '€')) +
        '</div>' +
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

    document.getElementById('pk-more').style.display =
      state.filtered.length > state.shown ? '' : 'none';
  }

  function fillFilters() {
    var setSel = document.getElementById('pk-set');
    var eraSel = document.getElementById('pk-era');
    var eras = [];
    Object.keys(state.setInfo).forEach(function (sid) {
      var s = state.setInfo[sid];
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

  Promise.all([fetchJson('cards'), fetchJson('sets'), fetchJson('meta'),
               fetchJson('art')])
    .then(function (res) {
      var payload = res[0];
      payload.columns.forEach(function (name, i) { C[name] = i; });
      state.prefix = payload.image_prefix;
      state.tcgPrefix = payload.tcgplayer_image_prefix || '';
      state.ptcgPrefix = payload.ptcg_image_prefix || '';
      state.sets = payload.sets;
      state.series = payload.series;
      state.rarities = payload.rarities;
      state.dates = payload.dates;
      state.rows = payload.rows;
      state.setInfo = res[1];
      state.meta = res[2];
      state.art = res[3];

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
