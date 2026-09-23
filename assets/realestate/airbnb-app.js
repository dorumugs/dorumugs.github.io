// 전국 Airbnb 밀집 지도.
//
// 화면의 두 층
//   아래 — 시군구 색칠(choropleth). airbnb.json 의 sgg 집계로 칠한다.
//   위   — 캔버스 점(pointlayer.js). 전국에서는 500m 격자, 시군구 하나를
//          고르면 그 구의 개별 숙소 좌표다.
//
// 왜 기본이 면적당 밀도인가
//   절대 수로 칠하면 서울 몇 개 구가 눈금을 다 먹어 나머지 전국이 한 색이
//   된다. 밀도로 칠하고 절대 수는 표에서 본다.

import { initMap } from './map.js';
import { initPointLayer } from './pointlayer.js';
import { SEQUENTIAL, NO_DATA } from './palette.js';
import { makeSortable } from './sorttable.js';
import { showStale, LIMITS } from './freshness.js';

const root = document.querySelector('.re-app');
const BASE = root.dataset.base.replace(/\/$/, '');

const state = { metric: 'density', region: '', sgg: null, layer: true };
let data = null;      // airbnb.json
let map = null;
let layer = null;
let loadSeq = 0;      // 구를 고를 때마다 올라간다 — 뒤늦게 온 응답을 버리는 데 쓴다
const pointCache = new Map();

const SIDO_NAMES = {
  11: '서울', 12: '전남·광주', 26: '부산', 27: '대구', 28: '인천', 30: '대전',
  31: '울산', 36: '세종', 41: '경기', 43: '충북', 44: '충남', 47: '경북',
  48: '경남', 50: '제주', 51: '강원', 52: '전북',
};

const METRIC = {
  density: { label: '면적당 밀도', unit: '건/km²', of: (e) => e.density, digits: 1 },
  count: { label: '숙소 수', unit: '건', of: (e) => e.count, digits: 0 },
};

// 전국에는 같은 이름의 시군구가 여럿이다 — 중구만 서울·부산·대구·인천·대전·
// 울산에 있고 동구·남구·북구·서구도 마찬가지다. 표와 툴팁에 '중구' 라고만
// 적으면 어느 중구인지 알 길이 없으므로, 겹치는 이름에는 시도를 붙인다.
let dupeNames = new Set();

function buildDupeNames() {
  const seen = new Map();
  for (const entry of Object.values(data.sgg)) {
    seen.set(entry.name, (seen.get(entry.name) || 0) + 1);
  }
  dupeNames = new Set([...seen].filter(([, n]) => n > 1).map(([name]) => name));
}

function sggLabel(code) {
  const entry = data.sgg[code];
  if (!entry) return code;
  if (!dupeNames.has(entry.name)) return entry.name;
  return `${SIDO_NAMES[code.slice(0, 2)] || code.slice(0, 2)} ${entry.name}`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// 받침이 있으면 앞 조사, 없으면 뒤 조사. '밀도이'/'숙소 수이' 같은 말이
// 화면과 스크린리더에 그대로 나가는 걸 막는다.
function particle(word, withJong, withoutJong) {
  const last = String(word).trim().slice(-1);
  const code = last.charCodeAt(0);
  if (!(code >= 0xac00 && code <= 0xd7a3)) return withoutJong;
  return (code - 0xac00) % 28 ? withJong : withoutJong;
}

function fmt(n, digits) {
  if (!Number.isFinite(n)) return '—';
  return n.toLocaleString('ko-KR', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}

function fail(message) {
  const box = document.createElement('p');
  box.className = 're-error';
  box.textContent = message;
  root.prepend(box);
}

/* ---------- 색칠 ---------- */

// 밀도는 몇 곳이 유별나게 높다 — 실측으로 중구 138건/km², 마포구 117건/km²
// 인데 절반 넘는 시군구가 1건/km² 아래다. 최솟값~최댓값을 고르게 나누면
// 전국이 한 색이 되고, 상위 5%를 잘라도 여전히 거의 한 색이었다.
//
// 그래서 **분위수**로 자른다. 색 한 칸마다 시군구 수가 비슷하게 들어가므로
// 지역 차이가 눈에 보인다. 대신 색 간격이 값 간격과 다르다는 걸 범례가
// 숫자로 밝힌다 — 칸마다 그 칸이 어디서 시작하는지 적는다.
//
// 숙소가 0인 시군구는 이 계산에서 빼고 NO_DATA 로 칠한다. 0 을 섞으면 맨
// 아래 칸이 통째로 0 이 되어 한 칸이 놀게 된다.
function quantileBreaks(codes) {
  const values = codes
    .map((c) => METRIC[state.metric].of(data.sgg[c]))
    .filter((v) => Number.isFinite(v) && v > 0)
    .sort((a, b) => a - b);
  if (!values.length) return [];
  const breaks = [];
  for (let i = 0; i < SEQUENTIAL.length; i += 1) {
    breaks.push(values[Math.min(values.length - 1,
      Math.floor(i * values.length / SEQUENTIAL.length))]);
  }
  return breaks;
}

// 값이 몇 번째 칸인가. 칸 시작값 이상이면 그 칸이다.
function binOf(value, breaks) {
  let i = breaks.length - 1;
  while (i > 0 && value < breaks[i]) i -= 1;
  return i;
}

function paint() {
  const codes = map.codesIn(state.region || 'all').filter((c) => data.sgg[c]);
  const breaks = quantileBreaks(codes);
  const metric = METRIC[state.metric];
  const values = new Map();
  for (const code of codes) {
    const entry = data.sgg[code];
    const value = metric.of(entry);
    values.set(code, {
      color: entry.count > 0 && breaks.length
        ? SEQUENTIAL[binOf(value, breaks)] : NO_DATA,
      label: `${sggLabel(code)} · ${fmt(entry.count, 0)}건 · ${fmt(entry.density, 1)}건/km²`,
    });
  }
  map.paint(values);
  drawLegend(breaks, metric);
}

function drawLegend(breaks, metric) {
  const el = root.querySelector('.re-legend');
  // 램프는 palette.js 한 곳에서만 정의한다. 여기에 색을 다시 적지 말 것.
  // 마크업은 실거래 대시보드(app.js drawLegend)와 같게 둔다 — dashboard.css 의
  // .re-legend 규칙이 `span, i…, span` 순서를 전제로 여백을 준다.
  const swatches = SEQUENTIAL.map((c, i) => {
    const edge = breaks.length ? fmt(breaks[i], metric.digits) : '—';
    return `<i aria-hidden="true" title="${esc(edge)} 이상" style="background:${c}"></i>`;
  }).join('');
  const lo = breaks.length ? fmt(breaks[0], metric.digits) : '—';
  const hi = breaks.length ? `${fmt(breaks[breaks.length - 1], metric.digits)}+` : '—';
  el.innerHTML = `<span aria-hidden="true">${esc(lo)}</span>${swatches}`
    + `<span aria-hidden="true">${esc(hi)} ${esc(metric.unit)}</span>`;
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label',
    `지도 색상 범례. ${metric.label}${particle(metric.label, '이', '가')} `
    + `옅은 색 ${lo}에서 짙은 색 ${hi}까지. 색 한 칸마다 시군구 수가 비슷하게 `
    + '들어가도록 나눈 분위수 구간이라, 색 간격과 값 간격은 다르다.');
}

/* ---------- 점 레이어 ---------- */

// 격자 칸은 500m 다. 전국을 640px 에 담으면 한 칸이 0.6px 이라, 11,000칸을
// 그대로 찍으면 지도가 파란 덩어리가 되고 시군구 색칠이 안 보인다(실측).
// 화면에서 한 칸이 최소 이만큼은 되도록 칸을 묶어 그린다.
const MIN_CELL_PX = 5;

// 시군구 확대의 바닥(SVG 사용자 단위). 1 단위가 약 430m 이므로 40 은 약 17km다.
const MIN_VIEW = 40;

/** 격자를 `factor` 배 굵은 칸으로 다시 묶는다. 개수는 더해진다. */
function binGrid(rows, factor) {
  if (factor <= 1) return rows;
  const step = data.meta.grid_deg * factor;
  const merged = new Map();
  for (const [lat, lng, n, weak] of rows) {
    // 부동소수점 잡음은 작은 여유로 털어낸다 — 안 그러면 격자 경계에 딱
    // 걸린 칸이 한 줄 아래로 샌다(build_airbnb 의 snap 과 같은 이유).
    const la = Math.round(Math.floor(lat / step + 1e-9) * step * 1e6) / 1e6;
    const ln = Math.round(Math.floor(lng / step + 1e-9) * step * 1e6) / 1e6;
    const k = `${la},${ln}`;
    const hit = merged.get(k);
    if (hit) { hit[2] += n; hit[3] = hit[3] || weak; } else { merged.set(k, [la, ln, n, weak]); }
  }
  return [...merged.values()];
}

/** 지금 축척에서 한 칸이 화면 몇 px 인지 보고 묶을 배수를 정한다. */
function binFactor() {
  const svg = root.querySelector('svg.re-map');
  const box = svg.getBoundingClientRect();
  const view = (svg.getAttribute('viewBox') || '0 0 1000 1000').split(/\s+/).map(Number);
  const p = data.projection;
  // 격자 한 칸(경도 grid_deg)이 SVG 사용자 단위로 몇인지 → 화면 px 로.
  const unitsPerDeg = p.k / p.span_x * p.width;
  const pxPerUnit = view[2] > 0 ? box.width / view[2] : 1;
  const cellPx = data.meta.grid_deg * unitsPerDeg * pxPerUnit;
  return cellPx > 0 ? Math.max(1, Math.round(MIN_CELL_PX / cellPx)) : 1;
}

function gridFor(region) {
  if (!region) return data.grid;
  // 시도를 고르면 그 시도의 시군구 경계 상자 안쪽 격자만 그린다. 전국 격자를
  // 다 그려도 화면 밖은 안 보이지만, 확대했을 때 옆 시도 점이 끼어든다.
  const codes = new Set(map.codesIn(region));
  const boxes = data.sgg_bbox;
  if (!boxes) return data.grid;
  const mine = [...codes].map((c) => boxes[c]).filter(Boolean);
  if (!mine.length) return data.grid;
  const minLat = Math.min(...mine.map((b) => b[0]));
  const maxLat = Math.max(...mine.map((b) => b[1]));
  const minLng = Math.min(...mine.map((b) => b[2]));
  const maxLng = Math.max(...mine.map((b) => b[3]));
  return data.grid.filter(([lat, lng]) => lat >= minLat && lat <= maxLat
    && lng >= minLng && lng <= maxLng);
}

function refreshLayer() {
  // 점이 켜져 있으면 시군구 색칠을 바탕으로 낮춘다(airbnb.css 참고).
  root.classList.toggle('has-points', state.layer);
  if (!state.layer) { layer.hide(); return; }
  if (state.sgg) {
    const rows = pointCache.get(state.sgg);
    if (rows) { layer.showPoints(rows); return; }
  }
  layer.showGrid(binGrid(gridFor(state.region), binFactor()));
}

async function loadPoints(code) {
  if (pointCache.has(code)) return pointCache.get(code);
  const res = await fetch(`${BASE}/airbnb/${code}.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body = await res.json();
  const rows = body.points || [];
  pointCache.set(code, rows);
  return rows;
}

/* ---------- 패널 ---------- */

function regionLabel() {
  return state.region ? SIDO_NAMES[state.region] || state.region : '전국';
}

function showOverview() {
  const codes = map.codesIn(state.region || 'all').filter((c) => data.sgg[c]);
  const count = codes.reduce((s, c) => s + data.sgg[c].count, 0);
  const area = codes.reduce((s, c) => s + data.sgg[c].area_km2, 0);
  root.querySelector('.re-panel-title').textContent = `${regionLabel()} 전체`;
  root.querySelector('.re-back-btn').hidden = true;
  root.querySelector('.re-kpis').innerHTML = kpis([
    ['숙소', fmt(count, 0), '건'],
    ['시군구', fmt(codes.length, 0), '개'],
    ['면적당', fmt(area > 0 ? count / area : 0, 2), '건/km²'],
  ]);
  root.querySelector('.re-panel-note').textContent =
    '지도에서 시군구를 누르면 그 지역 숙소 위치를 점으로 찍습니다.';
}

function kpis(rows) {
  return rows.map(([label, value, unit]) => (
    `<div class="re-kpi"><div class="re-kpi-label">${esc(label)}</div>`
    + `<div class="re-kpi-value">${esc(value)}<small>${esc(unit)}</small></div></div>`
  )).join('');
}

async function selectSgg(code) {
  const entry = data.sgg[code];
  if (!entry) return;
  const seq = ++loadSeq;
  state.sgg = code;

  // 그 시군구로 확대한다. 전국 축척에서는 구 하나의 숙소가 몇 픽셀로 뭉개져
  // 아무것도 안 보인다(실측: 제주시 671곳이 캔버스 잉크 127px).
  // 이웃 구는 숨기지 않는다 — 한 구만 떠 있으면 어디인지 알 수 없다.
  // MIN_VIEW 는 확대의 바닥이다. map_kr.svg 는 eps 0.5 로 단순화돼 있어
  // (약 215m) 더 들어가면 경계가 각진 다각형으로 보인다 — 서울 중구는 바닥이
  // 없으면 52배까지 들어가 오차가 화면에서 23px 이 된다. 40 이면 10px 안쪽이다.
  map.focus(code, { minWidth: MIN_VIEW });
  map.setSelected(code);
  root.querySelector('.re-panel-title').textContent = sggLabel(code);
  const back = root.querySelector('.re-back-btn');
  // 시군구를 고르면 그 시도로 확대되므로, 돌아가는 곳은 전국이 아니라 그
  // 시도다. 문구가 '전체로' 면 전국으로 가는 줄 안다.
  back.textContent = `← ${regionLabel()} 전체`;
  back.hidden = false;

  const rank = Object.entries(data.sgg)
    .sort((a, b) => b[1].count - a[1].count)
    .findIndex(([c]) => c === code) + 1;
  root.querySelector('.re-kpis').innerHTML = kpis([
    ['숙소', fmt(entry.count, 0), '건'],
    ['면적당', fmt(entry.density, 2), '건/km²'],
    ['전국 순위', fmt(rank, 0), '위'],
  ]);
  const note = root.querySelector('.re-panel-note');
  // 숙소가 없는 시군구는 좌표 파일 자체가 만들어지지 않는다(build_airbnb.py 가
  // 빈 파일을 쓰지 않는다). 받으러 가면 404 를 맞고 오류 문구가 뜬다.
  if (!entry.count) {
    note.textContent = `${sggLabel(code)}에서는 숙소를 찾지 못했습니다.`;
    refreshLayer();
    return;
  }
  note.textContent = '숙소 위치를 받는 중…';
  try {
    const rows = await loadPoints(code);
    if (seq !== loadSeq) return;
    note.textContent = `${sggLabel(code)} 숙소 ${fmt(rows.length, 0)}곳을 점으로 찍었습니다.`
      + ' 위치는 에어비앤비가 공개하는 근사 좌표입니다.';
    refreshLayer();
  } catch (err) {
    if (seq !== loadSeq) return;
    note.textContent = `숙소 위치를 받지 못했습니다 (${err.message}).`;
  }
}

function clearSgg() {
  ++loadSeq;
  state.sgg = null;
  map.setSelected(null);
  map.setView(state.region || 'all', { animate: true });
  showOverview();
  refreshLayer();
}

/* ---------- 표 ---------- */

// 전국 256개를 다 그리면 표만으로 화면이 9,000px 을 넘는다(실측). 전국에서는
// 상위 몇 개만 보이고, 지역을 고르면 그 시도는 전부 보인다 — 시도 하나는
// 많아야 47개(경기)다.
const TOP_N = 50;

function drawTable() {
  const codes = map.codesIn(state.region || 'all').filter((c) => data.sgg[c]);
  const all = codes.map((c) => ({ code: c, ...data.sgg[c] }))
    .sort((a, b) => b.count - a.count);
  const capped = !state.region && all.length > TOP_N;
  const rows = capped ? all.slice(0, TOP_N) : all;
  const heading = root.querySelector('.re-section-title');
  heading.textContent = capped
    ? `시군구 랭킹 · 전국 ${fmt(all.length, 0)}개 중 상위 ${TOP_N}개`
    : `시군구 랭킹 · ${fmt(all.length, 0)}개`;
  const table = root.querySelector('.re-table');
  // 면적 열은 뺐다. 오른쪽 열이 좁고, 면적은 Airbnb 가 아니라 행정구역의
  // 성질이라 여기서 자리를 살 만한 정보가 아니다 — 밀도에 이미 들어 있다.
  table.innerHTML =
    '<thead><tr><th>시군구</th><th>숙소</th><th>건/km²</th></tr></thead><tbody>'
    + rows.map((r) => (
      `<tr data-code="${esc(r.code)}"><td>${esc(sggLabel(r.code))}</td>`
      + `<td>${fmt(r.count, 0)}</td>`
      + `<td data-sort="${r.density}">${fmt(r.density, 1)}</td></tr>`
    )).join('')
    + '</tbody>';
  makeSortable(table);
}

/* ---------- 조립 ---------- */

function setRegion(value, { animate = true } = {}) {
  state.region = value;
  state.sgg = null;
  map.setSelected(null);
  map.setView(value || 'all', { animate });
  paint();
  drawTable();
  showOverview();
  refreshLayer();
}

function wire() {
  for (const tab of root.querySelectorAll('.re-metric-tabs .re-tab')) {
    tab.addEventListener('click', () => {
      for (const t of root.querySelectorAll('.re-metric-tabs .re-tab')) {
        t.classList.toggle('is-on', t === tab);
        t.setAttribute('aria-selected', String(t === tab));
      }
      state.metric = tab.dataset.metric;
      paint();
    });
  }

  const select = root.querySelector('.re-region');
  const sidos = [...new Set(Object.keys(data.sgg).map((c) => c.slice(0, 2)))].sort();
  select.innerHTML = '<option value="">전국</option>'
    + sidos.map((s) => `<option value="${esc(s)}">${esc(SIDO_NAMES[s] || s)}</option>`).join('');
  select.addEventListener('change', () => setRegion(select.value));

  const toggle = root.querySelector('[data-layer]');
  toggle.addEventListener('click', () => {
    state.layer = !state.layer;
    toggle.classList.toggle('is-on', state.layer);
    toggle.setAttribute('aria-pressed', String(state.layer));
    refreshLayer();
  });

  root.querySelector('.re-back-btn').addEventListener('click', clearSgg);

  root.querySelector('.re-table').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-code]');
    if (tr) selectSgg(tr.dataset.code);
  });
}

function footnote() {
  const meta = data.meta;
  const el = root.querySelector('.re-footnote');
  const parts = [
    `출처 ${meta.source}. 수집 ${meta.collected ? meta.collected.slice(0, 10) : '—'}.`,
    `좌표 ${fmt(meta.total, 0)}건 중 ${fmt(meta.placed, 0)}건을 시군구에 배정했습니다`
    + `(바다·경계 밖 ${fmt(meta.dropped, 0)}건 제외).`,
  ];
  if (!meta.complete) {
    parts.push('아직 전국을 한 바퀴 다 돌지 못했습니다 — 덜 훑은 지역은 실제보다 적게 나옵니다.');
  }
  if (meta.weak_boxes) {
    parts.push(`도심 ${fmt(meta.weak_boxes, 0)}곳은 검색 결과 상한에 걸려 실제보다 적게 잡힙니다`
      + '(지도에서 붉은 테두리).');
  }
  parts.push('위치는 에어비앤비가 공개하는 근사 좌표이며, 개별 숙소 정보는 담지 않았습니다.');
  el.textContent = parts.join(' ');
}

async function main() {
  try {
    const res = await fetch(`${BASE}/airbnb.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    fail(`집계 자료를 받지 못했습니다 (${err.message}).`);
    return;
  }

  buildDupeNames();
  // onView 는 viewBox 가 바뀔 때마다(확대 애니메이션 매 프레임 포함) 불린다.
  // 캔버스는 SVG 의 화면 행렬로 좌표를 옮기므로 여기서 다시 그려야 점이
  // 지도와 함께 움직인다.
  map = initMap(root, {
    onSelect: (code) => selectSgg(code),
    // 움직이는 중에는 그리기만(싸다), 멈추면 축척에 맞춰 격자를 다시 묶는다.
    onView: (settled) => {
      if (!layer) return;
      if (settled) refreshLayer(); else layer.redraw();
    },
  });
  layer = initPointLayer(root, data.projection);
  wire();
  // 첫 그림은 움직이지 않는다 — 열자마자 지도가 스스로 움직이면 놀란다.
  setRegion('', { animate: false });
  drawTable();
  footnote();
  showStale(root.querySelector('.re-footnote'), data.meta.collected, LIMITS.airbnb,
    'Airbnb 숙소 위치');
}

main();
