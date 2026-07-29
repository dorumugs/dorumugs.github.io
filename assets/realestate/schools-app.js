import { setBase, loadSgg } from './data.js';
import { initMap } from './map.js';
import { initSchoolLayer } from './schoolmap.js';
import { NO_DATA, CATEGORICAL } from './palette.js';
import { multiLineChart, legendHtml } from './charts.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);
const BASE = root.dataset.base.replace(/\/$/, '');

const state = { view: 'seoul', lvl: 'all', school: null };
let schools = [];
let map = null;
let layer = null;
let selectSeq = 0; // 학교를 고를 때마다 올라간다 — 뒤늦게 끝난 요청의 결과를 버리는 데 쓴다
let currentList = []; // 지금 목록 표에 그려진 학교들. 행 클릭/호버가 인덱스로 찾는다

const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };
const VIEW_LABEL = { seoul: '서울', gyeonggi: '경기', all: '전체' };

// 학교급(lvl) 을 화면 문구로 늘려 쓴다. school.found 는 사립·공립이 섞여
// 있으므로('서울과학고' 는 공립) '${school.found} 초등학교' 처럼 하드코딩하면
// 뜻이 맞지 않는다.
const LEVEL_LABEL = { 초: '초등학교', 중: '중학교', 국제중: '국제중학교', 특목고: '특수목적고' };

// 학교급 필터 탭·목록 제목에 쓰는 표기. 'all' 은 네 급을 같이 본다는 뜻이다.
const LVL_TAB_VALUES = ['all', '초', '중', '국제중', '특목고'];
const LVL_LIST_LABEL = {
  all: '사립초·사립중·국제중·특목고',
  초: '사립초',
  중: '사립중',
  국제중: '국제중',
  특목고: '특목고',
};

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// 학교별 특목고·자사고 진학률(progression_school.json). 서울·경기 중학교
// 1,050곳이 들어 있다 — 지도에 없는 공립중까지 담은 이유는 "비슷한 진학률
// 학교"를 고를 모집단이 필요해서다.
let progSchools = null;      // { years, thin, schools: [...] }
let progIndex = new Map();   // `${sgg}|${정규화한 이름}` → 학교

// 교명 표기가 두 자료에서 갈릴 때가 있다(가운뎃점·괄호 등). 한글·숫자만 남겨
// 맞춘다 — 예: '이화여자대학교사범대학부속이화·금란중학교' 의 가운뎃점.
function normName(name) {
  return String(name).replace(/[^가-힣0-9]/g, '');
}

function progOf(school) {
  if (!progIndex.size) return null;
  return progIndex.get(`${school.sgg}|${normName(school.name)}`) || null;
}

// 진학률을 쓸 수 있는 학교급. 초등학교는 애초에 진학 개념이 다르고, 특목고는
// 중학교 졸업생 진로의 '결과' 쪽이라 이 지표를 붙이면 뜻이 뒤집힌다.
const PROG_LEVELS = new Set(['중', '국제중']);

function visibleSchools() {
  const prefix = VIEW_PREFIX[state.view] ?? '';
  return schools.filter((s) => s.sgg.startsWith(prefix)
    && (state.lvl === 'all' || s.lvl === state.lvl));
}

// school.addr 는 "서울특별시 강남구 대치동 942" 같은 지번주소 전체다. school.dong
// 이 그 안 어딘가에 그대로 들어있으므로(build_schools.py 가 addr 에서 잘라 만든
// 값이라) 그 앞부분을 시군구로 쓰고, 시도 접두사(서울특별시/경기도)는 탭이 이미
// 지역을 말해주니 표에서는 뗀다 — "성남시 분당구" 처럼 구가 있는 경기 시군구도
// 그대로 살아남는다.
function sggLabel(school) {
  const idx = school.addr.indexOf(school.dong);
  const head = idx > -1 ? school.addr.slice(0, idx).trim() : school.addr;
  return head.replace(/^(서울특별시|경기도)\s*/, '');
}

function paintBase() {
  // 구는 배경이다. 전부 같은 옅은 색으로 깔아 점이 묻히지 않게 한다.
  const values = new Map();
  for (const code of map.codesIn(state.view)) {
    values.set(code, { color: NO_DATA, label: '' });
  }
  map.paint(values);
}

// 목록 한 칸에 넣을 최신연도 진학률. 졸업생이 적은 학교는 한 명이 몇 %p 씩
// 움직이므로 값 옆에 표시를 달아 그대로 비교하지 않게 한다.
function progCell(school) {
  if (!PROG_LEVELS.has(school.lvl)) return '<span class="re-dim">—</span>';
  const p = progOf(school);
  const last = p && p.r[p.r.length - 1];
  if (last == null) return '<span class="re-dim">자료 없음</span>';
  const thin = (p.g || 0) < progSchools.thin;
  return `${last.toFixed(1)}%${thin ? '<span class="re-thin" title="졸업생이 적어 값이 크게 흔들립니다">*</span>' : ''}`;
}

// 선택된 학교가 없을 때(첫 진입, 탭 전환, "목록으로") 지금 뷰의 학교를 전부
// 표로 뿌린다. 시군구 → 학교명 순으로 정렬해 순서가 매번 안정적이게 한다.
function renderList() {
  const list = visibleSchools()
    .slice()
    .sort((a, b) => sggLabel(a).localeCompare(sggLabel(b), 'ko') || a.name.localeCompare(b.name, 'ko'));
  currentList = list;

  root.querySelector('.re-panel-title').textContent =
    `${VIEW_LABEL[state.view]} ${LVL_LIST_LABEL[state.lvl]} ${list.length}곳`;
  root.querySelector('.re-school-meta').textContent =
    '학교를 누르면(또는 지도에서 점을 누르면) 같은 법정동 아파트 시세를 볼 수 있습니다.';
  root.querySelector('.re-rank-heading').hidden = true;
  root.querySelector('.re-back-btn').hidden = true;

  // 학교급 칸은 lvl 값을 그대로 쓴다(초/중/국제중/특목고). '사립초'처럼 더
  // 풀어 쓰면 좁은 화면에서 칸이 두 줄로 접혀 표가 들쭉날쭉해진다. 열
  // 이름(학교급)이 이미 맥락을 준다.
  const table = root.querySelector('.re-table');
  // 진학률 열은 중학교가 하나라도 보일 때만 붙인다. 사립초·특목고만 보고 있을
  // 때 값이 전부 '—' 인 빈 열이 자리를 차지하면 좁은 화면에서 손해다.
  const showProg = progSchools && list.some((s) => PROG_LEVELS.has(s.lvl));
  const progYear = showProg ? progSchools.years[progSchools.years.length - 1] : '';
  const head = '<thead><tr><th>학교명</th><th>학교급</th><th>시군구</th><th>법정동</th>'
    + (showProg ? `<th class="is-num">특목·자사고<br>진학률 ${progYear}</th>` : '')
    + '</tr></thead>';
  const body = list.map((s, i) => `<tr class="re-list-row" data-idx="${i}" tabindex="0" `
    + `role="button" aria-label="${esc(s.name)} 시세 보기">`
    + `<td>${esc(s.name)}</td><td>${esc(s.lvl)}</td>`
    + `<td>${esc(sggLabel(s))}</td><td>${esc(s.dong)}</td>`
    + (showProg ? `<td class="is-num">${progCell(s)}</td>` : '')
    + '</tr>').join('');
  const cols = showProg ? 5 : 4;
  table.innerHTML = list.length
    ? `${head}<tbody>${body}</tbody>`
    : `${head}<tbody><tr><td colspan="${cols}">이 조건에는 표시할 학교가 없습니다.</td></tr></tbody>`;
}

// 고른 중학교를 "최신연도 진학률이 가장 가까운" 학교 3곳과 함께 그린다.
// 순위를 매기는 화면이 아니라, 비슷한 수준의 학교들이 3년 동안 어떻게 움직였는지
// 보는 화면이다 — 그래서 상위권이 아니라 '가까운 값'을 고른다.
const PEER_COUNT = 3;

function renderPeerChart(school) {
  const wrap = root.querySelector('.re-peer-prog');
  if (!wrap) return;
  const heading = root.querySelector('.re-peer-prog-heading');
  const chartEl = wrap.querySelector('.re-peer-chart');
  const legendEl = wrap.querySelector('.re-peer-legend');
  const noteEl = wrap.querySelector('.re-peer-note');
  const me = PROG_LEVELS.has(school.lvl) ? progOf(school) : null;
  const last = me && me.r[me.r.length - 1];
  if (last == null) {
    wrap.hidden = true;
    if (heading) heading.hidden = true;
    return;
  }
  wrap.hidden = false;
  if (heading) heading.hidden = false;

  // 비교 대상은 지도에 없는 공립중까지 포함한 전체다. 사립중끼리만 비교하면
  // 표본이 194곳뿐이라 "비슷한 값"이 실제로는 꽤 멀어진다.
  const peers = progSchools.schools
    .filter((p) => p !== me && p.r[p.r.length - 1] != null)
    .map((p) => ({ p, d: Math.abs(p.r[p.r.length - 1] - last) }))
    .sort((a, b) => a.d - b.d || a.p.name.localeCompare(b.p.name, 'ko'))
    .slice(0, PEER_COUNT)
    .map((x) => x.p);

  const series = [me, ...peers].map((p, i) => ({
    label: p === me ? `${p.name} (선택)` : p.name,
    color: CATEGORICAL[i % CATEGORICAL.length],
    values: p.r,
  }));
  // 값이 비슷한 학교만 모아 그리므로 끝점 라벨이 서로 포개진다. 이름은
  // 아래 HTML 범례가 맡고 차트에서는 끈다.
  chartEl.innerHTML = multiLineChart(progSchools.years, series,
    { unit: '%', decimals: 1, endLabels: false });
  chartEl.setAttribute('aria-label',
    `${school.name}과 진학률이 비슷한 학교 ${peers.length}곳의 특목고·자사고 진학률 추이`);
  legendEl.innerHTML = legendHtml(series);
  const thin = [me, ...peers].filter((p) => (p.g || 0) < progSchools.thin).length;
  noteEl.textContent = '특목고·자사고 진학률이 가장 가까운 학교를 고른 것이라 순위가 아닙니다. '
    + `비교 대상은 서울·경기 중학교 ${progSchools.schools.length.toLocaleString()}곳입니다.`
    + (thin ? ` 이 중 ${thin}곳은 졸업생이 ${progSchools.thin}명 미만이라 값이 크게 흔들립니다.` : '');
}

async function selectSchool(school) {
  state.school = school;
  layer.setSelected(school.name);
  root.querySelector('.re-panel-title').textContent = school.name;
  const levelLabel = LEVEL_LABEL[school.lvl] || school.lvl;
  // 특목고만 계열(과학/외국어/국제)이 있다. 같은 '특수목적고' 라도 어느 계열인지가
  // 학교 성격을 가르므로 주소 앞에 끼워 넣는다.
  const parts = [levelLabel, school.course, school.addr].filter(Boolean);
  root.querySelector('.re-school-meta').textContent = parts.join(' · ');
  root.querySelector('.re-rank-heading').hidden = false;
  root.querySelector('.re-back-btn').hidden = false;
  renderPeerChart(school);
  writeParams();

  // selectSchool 은 제목을 동기로 세팅한 뒤 loadSgg 를 기다린다. 구를 빠르게
  // 두 번 눌러 두 번째 클릭이 진행 중일 때 첫 번째 요청이 나중에 끝나면,
  // 새 제목 아래 옛 표가 남는 경합이 생긴다. 요청마다 세대 번호를 찍어 두고
  // 자신이 최신 요청이 아니면 결과를 버린다.
  const mySeq = ++selectSeq;
  const table = root.querySelector('.re-table');
  try {
    const detail = await loadSgg(school.sgg);
    if (mySeq !== selectSeq) return;
    const rows = detail.complexes
      .filter((c) => c.dong === school.dong && c.n >= 5)
      .slice(0, 30);
    const head = '<thead><tr><th>단지</th><th class="is-num">평당가(만원)</th>'
      + '<th class="is-num">세대</th><th class="is-num">거래</th></tr></thead>';
    const body = rows.map((c) => `<tr><td>${esc(c.name)}</td>`
      + `<td class="is-num">${c.med != null ? c.med.toLocaleString() : '—'}</td>`
      + `<td class="is-num is-dim">${c.hh != null ? c.hh.toLocaleString() : '—'}</td>`
      + `<td class="is-num is-dim">${c.n.toLocaleString()}</td></tr>`).join('');
    table.innerHTML = rows.length
      ? `${head}<tbody>${body}</tbody>`
      : `${head}<tbody><tr><td colspan="4">${esc(school.dong)}에 최근 12개월 거래 `
        + '5건 이상 단지가 없습니다.</td></tr></tbody>';
  } catch (err) {
    if (mySeq !== selectSeq) return;
    table.innerHTML = '<tbody><tr><td>시세를 불러오지 못했습니다.</td></tr></tbody>';
  }
}

// 서울·경기 진학률 추이(re-progression). 지도/목록/state 와 무관하게 한 번만
// 그린다 — 학교급·지역 탭을 바꿔도 이 시·도 단위 자료는 바뀌지 않는다.
// 색은 palette.js CATEGORICAL 슬롯 1(파랑)·2(주황)를 그대로 쓴다.
const PROG_REGIONS = [
  { key: '서울', color: CATEGORICAL[0] },
  { key: '경기', color: CATEGORICAL[1] },
];

function renderProgression(data) {
  const chartEl = document.querySelector('.re-progression .re-prog-chart');
  const legendEl = document.querySelector('.re-progression .re-prog-legend');
  const cohortEl = document.querySelector('.re-progression .re-prog-cohort');
  if (!chartEl) return;
  if (!data) {
    chartEl.innerHTML = '<p class="re-error">진학률 자료를 불러오지 못했습니다.</p>';
    return;
  }

  const rateSeries = PROG_REGIONS.map(({ key, color }) => ({
    label: key, color, values: data.regions[key].rate,
  }));
  chartEl.innerHTML = multiLineChart(data.years, rateSeries, { unit: '%', decimals: 1 });
  chartEl.setAttribute('aria-label', '서울·경기 특목고·자사고 진학률 추이');
  legendEl.innerHTML = legendHtml(rateSeries);

  // 진학률만 보면 분모(졸업자 수)가 줄어드는 건 안 보인다 — 같은 15%도 10만
  // 명 중 15%와 5만 명 중 15%는 다른 이야기다. 시작 연도·끝 연도의 졸업자
  // 수를 나란히 적어 코호트가 줄어드는 걸 상호작용 없이도 보이게 한다.
  const firstYear = data.years[0];
  const lastYear = data.years[data.years.length - 1];
  cohortEl.innerHTML = PROG_REGIONS.map(({ key }) => {
    const den = data.regions[key].den;
    const first = den[0].toLocaleString();
    const last = den[den.length - 1].toLocaleString();
    return `<span>${esc(key)} 졸업자 ${esc(firstYear)}년 ${first}명 → `
      + `${esc(lastYear)}년 ${last}명</span>`;
  }).join(' · ');
}

function writeParams() {
  const q = new URLSearchParams();
  q.set('view', state.view);
  if (state.lvl !== 'all') q.set('lvl', state.lvl);
  if (state.school) q.set('school', state.school.name);
  window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  const valid = ['seoul', 'gyeonggi', 'all'].includes(view);
  if (valid) state.view = view;
  const lvl = q.get('lvl');
  if (LVL_TAB_VALUES.includes(lvl)) state.lvl = lvl;
  const name = q.get('school');
  if (name) {
    const hit = schools.find((s) => s.name === name);
    if (hit) {
      state.school = hit;
      // view= 가 쓸 만하지 않으면 고른 학교의 시도로 뷰를 맞춘다
      if (!valid) state.view = hit.sgg.startsWith('11') ? 'seoul' : 'gyeonggi';
    }
  }
}

function applyView() {
  root.querySelectorAll('.re-view-tabs .re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  root.querySelectorAll('.re-lvl-tabs .re-tab').forEach((b) => {
    const on = b.dataset.lvl === state.lvl;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  map.setView(state.view);
  paintBase();
  layer.render(visibleSchools()); // 점을 다시 그리므로 선택·호버 표시는 내부에서 초기화된다
  if (state.school) {
    layer.setSelected(state.school.name);
  } else {
    renderList();
  }
}

// 지도 점을 가리켰을 때(schoolmap.js 의 onHover) 그 학교의 표 행을 표시한다 —
// 반대 방향(행 → 점)은 layer.setHovered() 가 맡는다. currentList 는 renderList()
// 가 만든 visibleSchools() 결과라 school 객체 참조가 같으므로 indexOf 로 바로
// 행을 찾을 수 있다. 가격표가 떠 있어(currentList 가 비었거나 다른 학교라)
// 못 찾으면 조용히 아무 일도 하지 않는다.
function highlightRow(school) {
  const table = root.querySelector('.re-table');
  const prev = table.querySelector('tr.re-list-row.is-hover');
  if (prev) prev.classList.remove('is-hover');
  if (!school) return;
  const idx = currentList.indexOf(school);
  if (idx === -1) return;
  const tr = table.querySelector(`tr.re-list-row[data-idx="${idx}"]`);
  if (tr) tr.classList.add('is-hover');
}

// 표(re-table) 안의 행 인터랙션은 요소 하나하나에 리스너를 다는 대신 위임한다
// — 목록이 바뀔 때마다(탭 전환·목록으로) innerHTML 을 통째로 새로 써서 개별
// 리스너는 매번 사라지기 때문이다. 위임 리스너는 시작할 때 한 번만 건다.
function rowAt(target) {
  return target.closest ? target.closest('tr.re-list-row') : null;
}

function schoolOfRow(tr) {
  return tr ? currentList[Number(tr.dataset.idx)] : null;
}

function bindTableInteractions() {
  const table = root.querySelector('.re-table');
  table.addEventListener('click', (e) => {
    const school = schoolOfRow(rowAt(e.target));
    if (school) selectSchool(school);
  });
  table.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const school = schoolOfRow(rowAt(e.target));
    if (!school) return;
    e.preventDefault();
    selectSchool(school);
  });
  table.addEventListener('mouseover', (e) => {
    const school = schoolOfRow(rowAt(e.target));
    if (school) layer.setHovered(school.name);
  });
  table.addEventListener('mouseout', (e) => {
    const tr = rowAt(e.target);
    if (!tr) return;
    // 같은 행 안의 셀 사이를 옮겨다니는 것뿐이면(relatedTarget 이 여전히 이
    // tr 안) 무시한다 — 아니면 행을 벗어날 때마다 반짝이며 지도 점이
    // 껐다 켜졌다 한다.
    if (e.relatedTarget && tr.contains(e.relatedTarget)) return;
    layer.setHovered(null);
  });
  table.addEventListener('focusin', (e) => {
    const school = schoolOfRow(rowAt(e.target));
    if (school) layer.setHovered(school.name);
  });
  table.addEventListener('focusout', (e) => {
    const tr = rowAt(e.target);
    if (!tr) return;
    if (e.relatedTarget && tr.contains(e.relatedTarget)) return;
    layer.setHovered(null);
  });
}

function bind() {
  root.querySelectorAll('.re-view-tabs .re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      // 탭은 지역 선택이다 — 이전에 고른 학교가 새 지역에 없을 수도 있으니
      // 탭을 누르면 항상 그 지역의 전체 목록으로 돌아간다.
      state.school = null;
      applyView();
      writeParams();
    });
  });
  root.querySelectorAll('.re-lvl-tabs .re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.lvl = btn.dataset.lvl;
      // 학교급 탭도 지역 탭과 같은 이유로 선택을 초기화한다 — 고른 학교가
      // 새 필터에서 걸러질 수 있다.
      state.school = null;
      applyView();
      writeParams();
    });
  });
  root.querySelector('.re-back-btn').addEventListener('click', () => {
    state.school = null;
    layer.setSelected(null);
    layer.setHovered(null);
    renderList();
    writeParams();
  });
  bindTableInteractions();
}

async function start() {
  try {
    const res = await fetch(`${BASE}/schools.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    schools = data.schools;
    root.querySelector('.re-footnote').textContent =
      `학교 위치: 전국초중등학교위치표준데이터 · 시세: 국토교통부 실거래가 · 갱신 ${data.generated}`;
  } catch (err) {
    root.querySelector('.re-panel-title').textContent = '학교 자료를 불러오지 못했습니다';
    return;
  }

  // 학교별 진학률은 목록 열과 비교 차트에만 쓰인다. 못 받아도 지도·시세는
  // 그대로 동작해야 하므로 실패를 삼키고 진행한다(열이 안 붙을 뿐이다).
  try {
    const res = await fetch(`${BASE}/progression_school.json`, { cache: 'no-cache' });
    if (res.ok) {
      progSchools = await res.json();
      progIndex = new Map(progSchools.schools.map((p) => [`${p.sgg}|${normName(p.name)}`, p]));
    }
  } catch (err) {
    progSchools = null;
  }

  readParams();
  map = initMap(root, { onSelect: () => {}, interactive: false });
  layer = initSchoolLayer(root, { onSelect: selectSchool, onHover: highlightRow });
  bind();
  applyView();
  if (state.school) await selectSchool(state.school);

  // 진학률 추이는 지도·목록과 독립된 정적 자료라, 실패해도(fetch 만 막히고)
  // 화면의 나머지 기능(지도·시세 조회)은 그대로 동작해야 한다 — 별도로 묶어
  // 잡는다.
  try {
    const res = await fetch(`${BASE}/progression.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderProgression(await res.json());
  } catch (err) {
    renderProgression(null);
  }
}

start();
