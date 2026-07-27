import { setBase, loadSummary, loadSgg } from './data.js';
import { initMap } from './map.js';
import { divergingColor, sequentialColor, DIVERGING, SEQUENTIAL } from './palette.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);

const state = { view: 'seoul', metric: 'chg12', filter: '300', ym: null, sgg: null };
let summary = null;
let map = null;

const METRICS = {
  level: { label: '중위 평당가', unit: '만원/평', kind: 'sequential' },
  chg3: { label: '3개월 변화율', unit: '%', kind: 'diverging', lag: 3 },
  chg6: { label: '6개월 변화율', unit: '%', kind: 'diverging', lag: 6 },
  chg12: { label: '12개월 변화율', unit: '%', kind: 'diverging', lag: 12 },
  peak: { label: '전고점 대비', unit: '%', kind: 'diverging' },
  turnover: { label: '거래 회전율', unit: '%', kind: 'sequential' },
};

// 얇은 달(거래 5건 미만)이 변화율을 흔들지 않도록 3개월 이동중위를 쓴다.
const THIN = 5;

function smoothed(series, index) {
  const out = [];
  for (let i = 0; i <= index; i += 1) {
    const chunk = [];
    for (let k = Math.max(0, i - 2); k <= i; k += 1) {
      if (series.med[k] != null) chunk.push(series.med[k]);
    }
    out.push(chunk.length ? chunk.sort((a, b) => a - b)[Math.floor(chunk.length / 2)] : null);
  }
  return out;
}

function metricValue(code, metric, index) {
  const series = summary.series[state.filter][code];
  if (!series) return null;
  const spec = METRICS[metric];
  if (metric === 'level') return series.med[index];
  if (metric === 'turnover') {
    const hh = summary.sgg[code].hh[state.filter];
    if (!hh) return null;
    let n = 0;
    for (let i = Math.max(0, index - 11); i <= index; i += 1) n += series.n[i] || 0;
    return (n / hh) * 100;
  }
  const thin = (series.n[index] || 0) < THIN;
  const base = thin ? smoothed(series, index) : series.med;
  const now = base[index];
  if (now == null) return null;
  if (metric === 'peak') {
    let peak = -Infinity;
    for (let i = 0; i <= index; i += 1) if (base[i] != null) peak = Math.max(peak, base[i]);
    return peak > 0 ? (now / peak - 1) * 100 : null;
  }
  const before = base[index - spec.lag];
  if (before == null || before === 0) return null;
  return (now / before - 1) * 100;
}

function repaint() {
  const index = summary.months.indexOf(state.ym);
  const codes = map.codesIn(state.view);
  const raw = new Map();
  for (const code of codes) raw.set(code, metricValue(code, state.metric, index));

  const spec = METRICS[state.metric];
  const nums = [...raw.values()].filter((v) => Number.isFinite(v));
  const values = new Map();

  if (spec.kind === 'diverging') {
    const span = Math.max(1, ...nums.map((v) => Math.abs(v)));
    for (const [code, v] of raw) {
      values.set(code, {
        color: divergingColor(v, span),
        label: `${summary.sgg[code].name} ${Number.isFinite(v) ? `${v.toFixed(1)}%` : '자료 없음'}`,
      });
    }
    drawLegend(-span, span, 'diverging', spec.unit);
  } else {
    const min = nums.length ? Math.min(...nums) : 0;
    const max = nums.length ? Math.max(...nums) : 1;
    for (const [code, v] of raw) {
      values.set(code, {
        color: sequentialColor(v, min, max),
        label: `${summary.sgg[code].name} ${Number.isFinite(v) ? v.toLocaleString() : '자료 없음'}`,
      });
    }
    drawLegend(min, max, 'sequential', spec.unit);
  }
  map.paint(values);
  writeParams();
}

function drawLegend(min, max, kind, unit) {
  const el = root.querySelector('.re-legend');
  // 램프는 palette.js 한 곳에서만 정의한다. 여기에 색을 다시 적지 말 것.
  const ramp = kind === 'diverging' ? DIVERGING : SEQUENTIAL;
  const swatches = ramp.map((c) => `<i style="background:${c}"></i>`).join('');
  const fmt = (v) => (kind === 'diverging' ? `${v.toFixed(0)}%` : Math.round(v).toLocaleString());
  el.innerHTML = `<span>${fmt(min)}</span>${swatches}<span>${fmt(max)}${
    kind === 'sequential' ? ` ${unit}` : ''}</span>`;
}

function fillMonths() {
  const sel = root.querySelector('.re-month');
  sel.innerHTML = summary.months
    .map((m) => `<option value="${m}">${m}${m === summary.partial ? ' (집계 중)' : ''}</option>`)
    .join('');
  sel.value = state.ym;
}

async function selectSgg(code) {
  state.sgg = code;
  map.setSelected(code);
  root.querySelector('.re-panel-title').textContent = summary.sgg[code].name;
  const chartEl = root.querySelector('.re-chart');
  chartEl.innerHTML = '';
  try {
    const detail = await loadSgg(code);
    const { renderPanel } = await import('./charts.js');
    renderPanel(root, summary, detail, state);
    writeParams();
  } catch (err) {
    chartEl.innerHTML = '<p class="re-error">데이터를 불러오지 못했습니다. 다시 시도해 주세요.</p>';
  }
}

function bind() {
  root.querySelectorAll('.re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      root.querySelectorAll('.re-tab').forEach((b) => {
        const on = b === btn;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-selected', String(on));
      });
      map.setView(state.view);
      repaint();
    });
  });
  root.querySelector('.re-metric').addEventListener('change', (e) => {
    state.metric = e.target.value;
    repaint();
  });
  root.querySelector('.re-month').addEventListener('change', (e) => {
    state.ym = e.target.value;
    repaint();
    if (state.sgg) selectSgg(state.sgg);
  });
  const toggle = root.querySelector('.re-toggle');
  toggle.addEventListener('click', () => {
    state.filter = state.filter === '300' ? 'all' : '300';
    const on = state.filter === '300';
    toggle.classList.toggle('is-on', on);
    toggle.setAttribute('aria-pressed', String(on));
    toggle.textContent = on ? '300세대+' : '전체 거래';
    repaint();
    if (state.sgg) selectSgg(state.sgg);
  });
}

// 글에서 특정 화면을 바로 가리킬 수 있게 상태를 주소에 싣는다.
//   /real-estate/?sgg=11680&metric=chg12&ym=2026-06&filter=300&view=seoul
function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  if (['seoul', 'gyeonggi', 'all'].includes(view)) state.view = view;
  const metric = q.get('metric');
  // 대괄호 접근은 '__proto__' 같은 값에서도 진짜 값을 돌려주므로
  // hasOwnProperty 로 실제 소유 키인지 반드시 확인한다.
  if (metric && Object.prototype.hasOwnProperty.call(METRICS, metric)) state.metric = metric;
  const filter = q.get('filter');
  if (filter === '300' || filter === 'all') state.filter = filter;
  const ym = q.get('ym');
  if (ym && summary.months.includes(ym)) state.ym = ym;
  const sgg = q.get('sgg');
  if (sgg && Object.prototype.hasOwnProperty.call(summary.sgg, sgg)) state.sgg = sgg;
}

function writeParams() {
  const q = new URLSearchParams();
  q.set('view', state.view);
  q.set('metric', state.metric);
  q.set('ym', state.ym);
  q.set('filter', state.filter);
  if (state.sgg) q.set('sgg', state.sgg);
  window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

async function start() {
  try {
    summary = await loadSummary();
  } catch (err) {
    root.querySelector('.re-panel-title').textContent = '데이터를 불러오지 못했습니다';
    root.querySelector('.re-chart').innerHTML =
      '<p class="re-error">잠시 후 새로고침해 주세요.</p>';
    return;
  }
  // 마지막 달은 신고가 덜 들어와 항상 미완성이다. 직전 완료 월을 기본으로 둔다.
  const last = summary.months.length - 1;
  state.ym = summary.months[Math.max(0, last - 1)];
  readParams();

  map = initMap(root, { onSelect: selectSgg });
  fillMonths();
  bind();

  root.querySelectorAll('.re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  root.querySelector('.re-metric').value = state.metric;
  const toggle = root.querySelector('.re-toggle');
  const on300 = state.filter === '300';
  toggle.classList.toggle('is-on', on300);
  toggle.setAttribute('aria-pressed', String(on300));
  toggle.textContent = on300 ? '300세대+' : '전체 거래';

  map.setView(state.view);
  repaint();
  if (state.sgg) await selectSgg(state.sgg);

  root.querySelector('.re-footnote').textContent =
    `국토교통부 실거래가 · ${summary.months[0]} ~ ${summary.months[last]} · 갱신 ${summary.generated}`;
}

start();
