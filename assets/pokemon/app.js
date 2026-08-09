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
    rarities: [], dates: [], rows: [], setInfo: {}, art: {}, graded: {},
    meta: null, filtered: [], shown: PAGE,
    view: 'grid'   // 'grid' 카드형 | 'table' 표형
  };
  var C = {};
  var G = {};

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

       0. art.json — 이 카드에 대해 **실제로 열리는 것을 확인한** 주소. 규칙보다
          우선한다. TCGdex 가 경로만 주고 파일이 없는 카드가 90장 있어서다
          (Aquapolis·Skyridge 의 H 번호 홀로 등).
       1. TCGdex — 경로가 '시리즈/세트/번호' 로 규칙적이다.
       2. TCGplayer — TCGdex 가 이미지를 안 주는 588장을 productId 로 메운다.
       3. pokemontcg.io — 위 둘 다 없는 세트(Shiny Vault·Galarian Gallery·
          Trainer Gallery)용.
       4. Bulbagarden Archives — 상업 시세 사이트가 안 다루는 옛 비매품 세트
          (My First Battle, Poké Card Creator Pack).

     3·4순위는 규칙으로 만들지 않고 art.json 에 **확인된 주소만** 적어 두었다.
     없는 이미지를 요청해 404 를 흘리지 않기 위해서다.

     모든 <img> 에 referrerpolicy="no-referrer" 가 붙어야 한다. 위키는 Referer
     가 남의 도메인이면 막고, 그러면 Chrome 이 ERR_BLOCKED_BY_ORB 로 이미지를
     통째로 버린다 (실측). Referer 를 안 보내면 200 이다.

     넷 다 없는 카드는 6장뿐이다 (기본 에너지·포션·스위치). */
  function imageOf(r) {
    /* 확인된 주소가 있으면 그게 1순위다. 화면은 경로를 **세트 단위**로 만드는데
       TCGdex 는 같은 세트 안에서도 카드마다 파일이 있기도 없기도 하다 — 그래서
       "이 카드에 대해 실제로 열리는 것을 확인한 주소"가 규칙보다 우선한다. */
    var known = state.art[cardIdOf(r)];
    if (known) { return { thumb: known[0], full: known[1] }; }

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
    } else if (sort === 'psa-desc') {
      /* 등급 시세가 없는 카드는 뒤로 민다. 0 으로 쳐서 섞으면 순위가 거짓말이 된다. */
      out.sort(function (a, b) {
        var ag = state.graded[cardIdOf(a)], bg = state.graded[cardIdOf(b)];
        var av = ag ? ag[G.psa10] : null, bv = bg ? bg[G.psa10] : null;
        if (av === null || av === undefined) { return 1; }
        if (bv === null || bv === undefined) { return -1; }
        return bv - av;
      });
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

  /* 등급 시세 줄. 값만 크게 띄우면 안 된다 — 이 시장은 표본이 얇아서
     베이스셋 리자몽 PSA 10 조차 최근 1년에 1건 팔렸다. 그래서 건수와 마지막
     거래일을 값 옆에 같이 적는다. 없으면 아무것도 그리지 않는다. */
  function gradedHtml(cardId) {
    var g = state.graded[cardId];
    if (!g) { return ''; }

    function line(label, price, count, day) {
      if (price === null || price === undefined) { return ''; }
      var note = (count ? count + '건' : '거래 없음') + (day ? ' · ' + day : '');
      return '<div class="pk-row"><span class="pk-lbl">' + esc(label) +
        '</span><span class="pk-val">' + esc(money(price)) +
        '<i class="pk-when">' + esc(note) + '</i></span></div>';
    }

    var premium = g[G.premium]
      ? '<span class="pk-prem">raw 대비 ' + esc(g[G.premium]) + '배</span>'
      : '';

    return '<div class="pk-graded">' +
      '<p class="pk-gtitle">감정 등급 <small>eBay 낙찰가</small>' + premium + '</p>' +
      line('PSA 10', g[G.psa10], g[G.psa10_n], g[G.psa10_date]) +
      line('PSA 9', g[G.psa9], g[G.psa9_n], g[G.psa9_date]) +
      '</div>';
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
        ' referrerpolicy="no-referrer" onerror="this.style.display=\'none\'">'
      : '');

    return '<article class="pk-card">' +
      (pic
        ? '<a class="pk-imgwrap" href="' + esc(pic.full) +
          '" target="_blank" rel="noopener noreferrer">' + thumb + '</a>'
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
        gradedHtml(cardIdOf(r)) +
      '</div>' +
    '</article>';
  }

  /* 표형 한 줄. 카드형과 같은 데이터를 좁게 편다. 사진은 작게라도 있어야
     어떤 카드인지 알아본다 — 이름만으로는 같은 포켓몬이 수십 장이다. */
  function rowHtml(r) {
    var setId = setIdOf(r);
    var info = state.setInfo[setId] || {};
    var pic = imageOf(r);
    var rarity = state.rarities[r[C.rarity]] || '';
    var g = state.graded[cardIdOf(r)];

    var thumb = pic
      ? '<a href="' + esc(pic.full) + '" target="_blank" rel="noopener noreferrer">' +
        '<img class="pk-thumb" src="' + esc(pic.thumb) + '" alt="' +
        esc(r[C.name_en]) + '" loading="lazy" width="44" height="61"' +
        ' referrerpolicy="no-referrer" onerror="this.style.display=\'none\'"></a>'
      : '<span class="pk-thumb is-none">—</span>';

    return '<tr>' +
      '<td class="pk-thumbcell">' + thumb + '</td>' +
      '<td><b>' + esc(r[C.name_en]) + '</b>' +
        (r[C.name_ko] ? '<i>' + esc(r[C.name_ko]) + '</i>' : '') + '</td>' +
      '<td class="cmp-dim">' + esc(info.name || setId) + '<i>#' +
        esc(r[C.local_id]) + (rarity ? ' · ' + esc(rarity) : '') + '</i></td>' +
      '<td class="cmp-num"><b>' + money(r[C.price]) + '</b></td>' +
      '<td class="cmp-num">' + money(r[C.high_ask]) + '</td>' +
      '<td class="cmp-num' + (g && g[G.psa10] ? ' is-psa' : '') + '">' +
        (g && g[G.psa10] ? money(g[G.psa10]) +
          '<i>' + esc((g[G.psa10_n] || 0) + '건') + '</i>' : '—') + '</td>' +
      '</tr>';
  }

  function render() {
    var grid = document.getElementById('pk-grid');
    var wrap = document.getElementById('pk-tablewrap');
    var slice = state.filtered.slice(0, state.shown);
    var table = state.view === 'table';

    grid.hidden = table;
    wrap.hidden = !table;

    if (table) {
      document.getElementById('pk-tbody').innerHTML = slice.length
        ? slice.map(rowHtml).join('')
        : '<tr><td colspan="6" class="pk-empty">찾는 카드가 없습니다.</td></tr>';
    } else {
      grid.innerHTML = slice.length
        ? slice.map(cardHtml).join('')
        : '<p class="pk-empty">찾는 카드가 없습니다. 철자나 필터를 확인해 보세요.</p>';
    }

    document.getElementById('pk-count').textContent =
      state.filtered.length.toLocaleString('ko-KR') + '장 중 ' +
      slice.length.toLocaleString('ko-KR') + '장 표시';

    document.getElementById('pk-more').style.display =
      state.filtered.length > state.shown ? '' : 'none';
  }

  /* 본문에 박힌 숫자를 데이터에서 그려 넣는다. 하드코딩하면 다음 갱신에 곧
     거짓이 된다 — 실제로 899종/404종/830종이 하루 만에 틀어져 있었다. */
  function setLive(key, text) {
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-live="' + key + '"]'), function (e) {
        e.textContent = text;
      });
  }

  function fillLive() {
    setLive('no-art', (state.meta.no_art || 0).toLocaleString('ko-KR') + '장');

    var mult = [];
    Object.keys(state.graded).forEach(function (id) {
      var p = state.graded[id][G.premium];
      if (p) { mult.push(p); }
    });
    if (mult.length) {
      mult.sort(function (a, b) { return a - b; });
      setLive('premium', '중앙값 ' + mult[Math.floor(mult.length / 2)] +
        '배, 최대 ' + mult[mult.length - 1] + '배입니다');
    }
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
    Array.prototype.forEach.call(
      document.querySelectorAll('input[name="pk-view"]'), function (radio) {
        radio.addEventListener('change', function () {
          state.view = radio.value;
          render();
        });
      });
  }

  Promise.all([fetchJson('cards'), fetchJson('sets'), fetchJson('meta'),
               fetchJson('art'), fetchJson('graded')])
    .then(function (res) {
      var payload = res[0];
      /* 비교 탭(krw.js)이 같은 1MB 를 또 받지 않도록 넘겨둔다. */
      window.__pkGlobal = payload;
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
      res[4].columns.forEach(function (name, i) { G[name] = i; });
      state.graded = res[4].cards;

      fillFilters();
      bind();
      applyFilters();

      fillLive();
      document.getElementById('pk-meta').textContent =
        '카드 ' + state.meta.card_count.toLocaleString('ko-KR') + '장 · 세트 ' +
        state.meta.set_count + '개 · 한글 이름이 붙은 카드 ' +
        state.meta.with_korean_name.toLocaleString('ko-KR') + '장 · 감정 등급 시세가 붙은 카드 ' +
        (state.meta.graded_count || 0).toLocaleString('ko-KR') + '장 · 마지막 갱신 ' +
        state.meta.generated + '.';
    })
    .catch(function (err) {
      document.getElementById('pk-count').textContent =
        '데이터를 불러오지 못했습니다. (' + err.message + ')';
    });
})();
