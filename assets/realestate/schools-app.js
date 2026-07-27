import { setBase, loadSgg } from './data.js';
import { initMap } from './map.js';
import { initSchoolLayer } from './schoolmap.js';
import { NO_DATA } from './palette.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);
const BASE = root.dataset.base.replace(/\/$/, '');

const state = { view: 'seoul', school: null };
let schools = [];
let map = null;
let layer = null;
let selectSeq = 0; // 학교를 고를 때마다 올라간다 — 뒤늦게 끝난 요청의 결과를 버리는 데 쓴다

const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };

// 학교급(lvl) 을 화면 문구로 늘려 쓴다. school.found 는 초등학교에만 있어
// '${school.found} 초등학교' 처럼 하드코딩하면 중학교(2단계)가 항상
// '(빈값) 초등학교' 로 잘못 표시된다.
const LEVEL_LABEL = { 초: '초등학교', 중: '중학교' };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function visibleSchools() {
  const prefix = VIEW_PREFIX[state.view] ?? '';
  return schools.filter((s) => s.sgg.startsWith(prefix));
}

function paintBase() {
  // 구는 배경이다. 전부 같은 옅은 색으로 깔아 점이 묻히지 않게 한다.
  const values = new Map();
  for (const code of map.codesIn(state.view)) {
    values.set(code, { color: NO_DATA, label: '' });
  }
  map.paint(values);
}

async function selectSchool(school) {
  state.school = school;
  layer.setSelected(school.name);
  root.querySelector('.re-panel-title').textContent = school.name;
  const levelLabel = LEVEL_LABEL[school.lvl] || school.lvl;
  root.querySelector('.re-school-meta').textContent =
    `${levelLabel} · ${school.addr}`;
  root.querySelector('.re-rank-heading').hidden = false;
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

function writeParams() {
  const q = new URLSearchParams();
  q.set('view', state.view);
  if (state.school) q.set('school', state.school.name);
  window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  const valid = ['seoul', 'gyeonggi', 'all'].includes(view);
  if (valid) state.view = view;
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
  root.querySelectorAll('.re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  map.setView(state.view);
  paintBase();
  layer.render(visibleSchools());
  if (state.school) layer.setSelected(state.school.name);
}

function bind() {
  root.querySelectorAll('.re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      applyView();
      writeParams();
    });
  });
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

  readParams();
  map = initMap(root, { onSelect: () => {}, interactive: false });
  layer = initSchoolLayer(root, { onSelect: selectSchool });
  bind();
  applyView();
  if (state.school) await selectSchool(state.school);
}

start();
