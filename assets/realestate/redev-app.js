// 재개발·재건축 대시보드. 지도(시군구) + 표 세 가지 보기.
//
// 세 보기가 데이터 척추를 공유한다:
//   노후 단지   대지지분·용도지역·연차 (redev/<sgg>.json 의 complexes)
//   진행 단계   정비사업장과 구역     (redev/<sgg>.json 의 projects/zones)
//   프리미엄    이벤트 스터디 결과    (redev.json 의 premium, 전역 1회)

import { initMap } from './map.js';
import { makeSortable } from './sorttable.js';
import { SEQUENTIAL, NO_DATA, rampColor, UP, DOWN, MUTED } from './palette.js';

const root = document.querySelector('.re-app.is-redev');
const base = (root.dataset.base || '/assets/realestate').replace(/\/$/, '');

const els = {
  title: root.querySelector('.re-panel-title'),
  meta: root.querySelector('.re-panel-meta'),
  legend: root.querySelector('.re-legend'),
  tableWrap: root.querySelector('.re-body ~ .re-table-wrap'),
  table: root.querySelector('.re-body ~ .re-table-wrap .re-table'),
  note: root.querySelector('.re-note'),
  premium: root.querySelector('.re-premium'),
  premiumCards: root.querySelector('.re-premium-cards'),
  premiumTable: root.querySelector('.re-premium-table'),
  footnote: root.querySelector('.re-footnote'),
};

const state = { mode: 'complexes', view: 'seoul', sgg: null, summary: null };
const sggCache = new Map();

async function getJson(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

function loadSgg(code) {
  if (!sggCache.has(code)) {
    sggCache.set(code, getJson(`${base}/redev/${code}.json`).catch((err) => {
      sggCache.delete(code);
      throw err;
    }));
  }
  return sggCache.get(code);
}

// --------------------------------------------------------------------------
// 표시 헬퍼
// --------------------------------------------------------------------------

const DASH = '—';
const fmt = (v, digits = 0) =>
  v === null || v === undefined ? DASH : v.toLocaleString('ko-KR', { maximumFractionDigits: digits });

function pyeongPrice(v) {
  return v === null || v === undefined ? DASH : `${Math.round(v).toLocaleString('ko-KR')}만`;
}

// '38.5평'·'3,200만'·'+44.2%p' 처럼 단위가 붙은 칸은 sorttable 이 숫자로 못 읽어
// 글자 순으로 정렬한다 ('10.0평' 이 '9.0평' 앞으로 온다). 원 숫자를 data-sort 로
// 같이 실어 보낸다. 값이 없으면 속성을 아예 달지 않아 보이는 '—' 가 뒤로 간다.
const sortKey = (v) => (v === null || v === undefined ? '' : ` data-sort="${v}"`);

// 진행단계를 사업 순서대로 묶는다. 색은 진행도를 나타내는 순차 램프를 쓴다.
const STAGE_ORDER = [
  '정비계획 수립', '도시계획심의', '정비구역지정', '안전진단', '추진위구성', '추진위원회승인',
  '조합창립총회', '조합규약작성', '조합원 모집신고', '조합설립인가', '사업시행자지정',
  '지구단위계획수립/건축심의/교통심의', '사업시행인가', '사업계획승인',
  '관리처분인가', '철거', '철거 및 착공', '착공', '분양',
  '준공인가', '이전고시', '조합해산', '청산 및 조합해산', '조합청산',
];

function stageRank(stage) {
  let i = STAGE_ORDER.indexOf(stage);
  // 정보몽땅 단계명은 원문 그대로 온다. '안전진단(1차)'처럼 회차가 붙은 이름은
  // 괄호를 떼고 한 번 더 찾는다 — 못 찾으면 회색 칩이 되어 진행도가 사라진다.
  if (i < 0) i = STAGE_ORDER.indexOf(stage.replace(/\s*\([^)]*\)\s*$/, ''));
  return i < 0 ? null : i / (STAGE_ORDER.length - 1);
}

function stageColor(stage) {
  const t = stageRank(stage);
  return t === null ? NO_DATA : rampColor(SEQUENTIAL, t);
}

// --------------------------------------------------------------------------
// 지도
// --------------------------------------------------------------------------

const map = initMap(root, {
  onSelect: (code) => {
    state.sgg = code;
    map.setSelected(code);
    renderPanel();
  },
});

function paintMap() {
  const values = new Map();
  const per = state.summary?.sgg || {};
  const key = state.mode === 'projects' ? 'projects' : 'complexes';
  const codes = map.codesIn(state.view);
  const counts = codes.map((c) => per[c]?.[key] || 0);
  const max = Math.max(1, ...counts);
  for (const code of codes) {
    const n = per[code]?.[key] || 0;
    if (!n) continue;
    values.set(code, {
      color: rampColor(SEQUENTIAL, n / max),
      // 툴팁은 라벨이 있으면 구 이름을 대신 쓰지 않는다(map.js). 이름을 직접 넣는다.
      label: `${map.nameOf(code)} ${n}${key === 'projects' ? '개 사업장' : '개 단지'}`,
    });
  }
  map.paint(values);
}

// --------------------------------------------------------------------------
// 표
// --------------------------------------------------------------------------

function renderTable(head, rows, { rankColumn = null } = {}) {
  els.table.innerHTML =
    `<thead><tr>${head.map((h) => `<th${h.cls ? ` class="${h.cls}"` : ''}>${h.label}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.join('')}</tbody>`;
  makeSortable(els.table, { rankColumn });
}

function renderComplexes(detail) {
  const rows = detail.complexes;
  const head = [
    { label: '단지' }, { label: '동' }, { label: '준공' }, { label: '연차' },
    { label: '세대' }, { label: '대지지분' }, { label: '용적률' },
    { label: '용도지역' }, { label: '상한' }, { label: '최근 평당가' }, { label: '정비사업' },
  ];
  // 용적률은 대부분 건축물대장 실측이다. 연면적이 없어 실거래로 역산한 칸만
  // ~ 를 붙여 구분하고, 100% 아래인 칸은 대지면적이 단지보다 넓을 수 있다는
  // 신호라 물음표를 달아 독자가 스스로 걸러낼 수 있게 한다.
  const html = rows.map((r) => {
    const est = r.far_src === '추정';
    const doubt = r.far_est && r.far_est < 100;
    const cls = doubt ? ' class="is-doubt"' : est ? ' class="is-est"' : '';
    const title = doubt
      ? ' title="대지면적이 단지 땅보다 넓게 잡혔을 수 있습니다"'
      : est
        ? ' title="건축물대장에 연면적이 없어 실거래 전용면적으로 역산한 값입니다"'
        : ' title="건축물대장 용적률 산정 연면적 기준"';
    return `<tr>
    <td class="re-name">${r.name}</td>
    <td>${r.dong || DASH}</td>
    <td>${r.year || DASH}</td>
    <td${sortKey(r.age)}>${r.age}년</td>
    <td>${fmt(r.hh)}</td>
    <td${sortKey(r.share)}>${r.share === null ? DASH : `${r.share.toFixed(1)}평`}</td>
    <td${cls}${title}${sortKey(r.far_est)}>${r.far_est ? `${r.far_est}%${est ? '~' : ''}` : DASH}</td>
    <td>${r.zone || DASH}</td>
    <td>${r.far ? `${r.far}%` : DASH}</td>
    <td${sortKey(r.pp)}>${pyeongPrice(r.pp)}</td>
    <td>${r.stage ? `<span class="re-chip" style="--c:${stageColor(r.stage)}">${r.stage}</span>` : DASH}</td>
  </tr>`;
  });
  renderTable(head, html);

  const withLand = rows.filter((r) => r.share !== null).length;
  const measured = rows.filter((r) => r.far_src === '대장').length;
  els.note.textContent = withLand
    ? `${rows.length}곳 중 ${withLand}곳에 대지지분이 있고, 용적률 ${measured}곳은 건축물대장 실측입니다` +
      ' (나머지 ~ 표시는 실거래 역산). 오래된 순으로 정렬했습니다 —' +
      ' 머리글을 눌러 대지지분 순으로 바꿀 수 있습니다.'
    : `${rows.length}곳. 이 지역은 대지지분 자료가 없어 연차·세대수·실거래가만 나옵니다.`;
}

function renderProjects(detail) {
  const rows = detail.projects;
  const head = [
    { label: '사업장' }, { label: '구분' }, { label: '위치' }, { label: '단계' },
    { label: '조합설립' }, { label: '사업시행' }, { label: '관리처분' },
  ];
  const html = rows.map((p) => `<tr${p.suspended ? ' class="is-suspended"' : ''}>
    <td class="re-name">${p.name}</td>
    <td>${p.se}</td>
    <td>${p.addr || DASH}</td>
    <td>${p.stage ? `<span class="re-chip" style="--c:${stageColor(p.stage)}">${p.stage}</span>` : DASH}</td>
    <td>${p.milestones?.['조합설립인가'] || DASH}</td>
    <td>${p.milestones?.['사업시행인가'] || DASH}</td>
    <td>${p.milestones?.['관리처분인가'] || DASH}</td>
  </tr>`);
  renderTable(head, html);

  if (!rows.length) {
    els.note.textContent =
      '이 지역은 정비사업 진행 단계 자료가 없습니다 — 서울시 정보몽땅이 유일한 상시 출처라 서울만 있습니다.';
    return;
  }
  const suspended = rows.filter((p) => p.suspended).length;
  els.note.textContent =
    `사업장 ${rows.length}곳` +
    (suspended ? ` · 이 중 ${suspended}곳은 조합 카페가 닫혀 추진경과를 받지 못했습니다.` : '.') +
    ' 인가일은 최초 인가 기준입니다 (변경인가는 제외).';
}

function renderLegend() {
  if (state.mode === 'premium') {
    els.legend.innerHTML = '';
    return;
  }
  // 지도는 개수로, 표의 칩은 진행 단계로 칠한다. 둘이 같은 순차 램프를 쓰기 때문에
  // 무엇의 색인지 적지 않으면 지도 색을 단계 색으로 읽게 된다.
  const unit = state.mode === 'projects' ? '사업장 수' : '단지 수';
  const parts = [
    `<span class="re-legend-item"><b>지도 ${unit}</b>` +
      `<i style="background:${rampColor(SEQUENTIAL, 0.12)}"></i>적음` +
      `<i style="background:${rampColor(SEQUENTIAL, 1)}"></i>많음</span>`,
  ];
  if (state.mode === 'projects') {
    const picks = ['추진위원회승인', '조합설립인가', '사업시행인가', '관리처분인가', '착공', '준공인가'];
    parts.push(
      `<span class="re-legend-item"><b>표 진행 단계</b></span>`,
      ...picks.map((s) => `<span class="re-legend-item"><i style="background:${stageColor(s)}"></i>${s}</span>`),
    );
  }
  els.legend.innerHTML = parts.join('');
}

// --------------------------------------------------------------------------
// 프리미엄
// --------------------------------------------------------------------------

function stageCards(premium) {
  return premium.stages
    .map((s) => {
      if (s.median_excess === undefined) {
        return `<div class="re-card is-empty">
          <h3>${s.stage}</h3>
          <p class="re-card-value">${DASH}</p>
          <p class="re-card-sub">표본 ${s.n}건 · ${premium.min_sample}건 미만이라 감췄습니다</p>
        </div>`;
      }
      const color = s.median_excess > 0 ? UP : s.median_excess < 0 ? DOWN : MUTED;
      const sign = s.median_excess > 0 ? '+' : '';
      return `<div class="re-card">
        <h3>${s.stage}</h3>
        <p class="re-card-value" style="color:${color}">${sign}${s.median_excess.toFixed(1)}<small>%p</small></p>
        <p class="re-card-sub">
          사업장 ${s.n}곳 중 ${s.positive}곳이 초과 상승<br>
          사분위 ${s.q1 > 0 ? '+' : ''}${s.q1.toFixed(1)} ~ ${s.q3 > 0 ? '+' : ''}${s.q3.toFixed(1)}%p
        </p>
      </div>`;
    })
    .join('');
}

function renderPremium() {
  const blocks = [state.summary?.premium, state.summary?.premium_redev].filter(Boolean);
  if (!blocks.length) return;

  // 재건축과 재개발은 매칭 단위가 달라(단지 vs 법정동) 한 표에 섞으면 안 된다.
  // 블록을 나누고 각각의 매칭 방식을 그 자리에 적는다.
  els.premiumCards.innerHTML = blocks
    .map(
      (b) => `<section class="re-premium-block">
        <h3 class="re-block-title">${b.label} <small>사업장 ${b.matched}곳</small></h3>
        <p class="re-block-scope">${b.scope}</p>
        <div class="re-cards">${stageCards(b)}</div>
      </section>`,
    )
    .join('');

  const cases = blocks
    .flatMap((b) =>
      b.stages.flatMap((s) => (s.cases || []).map((c) => ({ ...c, stage: s.stage, kind: b.label }))),
    )
    .sort((a, b) => b.excess - a.excess)
    .slice(0, 30);

  els.premiumTable.innerHTML =
    `<thead><tr><th>사업장</th><th>구분</th><th>자치구</th><th>단계</th><th>인가일</th>
      <th>대상 변화</th><th>자치구 변화</th><th>초과분</th><th>거래</th></tr></thead>` +
    `<tbody>${cases
      .map((c) => `<tr>
        <td class="re-name">${c.name}</td>
        <td>${c.kind}</td>
        <td>${c.sgg}</td>
        <td><span class="re-chip" style="--c:${stageColor(c.stage)}">${c.stage}</span></td>
        <td>${c.date}</td>
        <td${sortKey(c.own)}>${c.own > 0 ? '+' : ''}${c.own.toFixed(1)}%</td>
        <td${sortKey(c.control)}>${c.control > 0 ? '+' : ''}${c.control.toFixed(1)}%</td>
        <td${sortKey(c.excess)} style="color:${c.excess > 0 ? UP : DOWN}"><b>${c.excess > 0 ? '+' : ''}${c.excess.toFixed(1)}%p</b></td>
        <td>${c.trades}</td>
      </tr>`)
      .join('')}</tbody>`;
  makeSortable(els.premiumTable);
}

// --------------------------------------------------------------------------
// 화면 전환
// --------------------------------------------------------------------------

async function renderPanel() {
  if (state.mode === 'premium') return;
  if (!state.sgg) {
    els.title.textContent = state.mode === 'projects' && state.view === 'gyeonggi'
      ? '경기는 진행 단계 자료가 없습니다'
      : '구를 선택하세요';
    // 정보몽땅이 서울만 상시 갱신한다. 경기 지도가 통째로 회색인 이유를 여기서 밝힌다.
    els.meta.textContent = state.mode === 'projects' && state.view === 'gyeonggi'
      ? '정비사업 진행 단계는 서울시 정보몽땅이 유일한 상시 출처라 서울만 있습니다. 경기는 노후 단지 보기를 쓰세요.'
      : '지도에서 자치구를 누르면 그 구의 목록이 나옵니다.';
    els.table.innerHTML = '';
    els.note.textContent = '';
    return;
  }
  const per = state.summary.sgg[state.sgg];
  const name = map.nameOf(state.sgg);
  els.title.textContent = name;
  els.meta.textContent = per
    ? `노후 단지 ${per.complexes}곳 · 정비사업장 ${per.projects}곳`
    : '자료 없음';

  try {
    const detail = await loadSgg(state.sgg);
    if (state.mode === 'complexes') renderComplexes(detail);
    else renderProjects(detail);
  } catch (err) {
    els.table.innerHTML = '';
    els.note.textContent = `자료를 불러오지 못했습니다: ${err.message}`;
  }
}

function applyMode() {
  const premiumMode = state.mode === 'premium';
  els.premium.hidden = !premiumMode;
  root.querySelector('.re-body').hidden = premiumMode;
  els.tableWrap.hidden = premiumMode;
  els.note.hidden = premiumMode;
  root.querySelector('.re-view-tabs').hidden = premiumMode;
  renderLegend();
  if (premiumMode) {
    renderPremium();
    return;
  }
  // 노후 단지 ↔ 진행 단계는 지도가 세는 대상이 달라진다(paintMap 이 state.mode 를 본다).
  // 다시 칠하지 않으면 처음 칠한 단지 수 색이 그대로 남는다.
  paintMap();
  renderPanel();
}

function bindTabs(selector, key, after) {
  root.querySelectorAll(`${selector} .re-tab`).forEach((tab) => {
    tab.addEventListener('click', () => {
      root.querySelectorAll(`${selector} .re-tab`).forEach((t) => {
        t.classList.remove('is-on');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('is-on');
      tab.setAttribute('aria-selected', 'true');
      state[key] = tab.dataset[key];
      after();
    });
  });
}

bindTabs('.re-mode-tabs', 'mode', applyMode);
bindTabs('.re-view-tabs', 'view', () => {
  map.setView(state.view);
  // 다른 시도로 넘어가면 이전 선택은 지도에 보이지 않는다. 선택을 비운다.
  state.sgg = null;
  map.setSelected(null);
  paintMap();
  renderPanel();
});

(async function start() {
  try {
    state.summary = await getJson(`${base}/redev.json`);
  } catch (err) {
    els.meta.textContent = `집계를 불러오지 못했습니다: ${err.message}`;
    return;
  }
  map.setView(state.view);
  applyMode(); // paintMap 을 겸한다

  const c = state.summary.counts;
  els.footnote.textContent =
    `${state.summary.latest_month} 기준 · 정비사업장 ${c.projects.toLocaleString('ko-KR')}곳` +
    `(추진경과 확보 ${c.projects_with_events.toLocaleString('ko-KR')}곳) · 정비구역 ${c.zones.toLocaleString('ko-KR')}곳 · ` +
    `준공 ${state.summary.min_age}년 이상 아파트 ${c.complexes.toLocaleString('ko-KR')}곳` +
    `(대지지분 ${c.with_land.toLocaleString('ko-KR')}곳). ` +
    `출처: 서울시 정비사업 정보몽땅 · 서울시 도시계획포털(UPIS) · 국토교통부 실거래가.`;
})();
