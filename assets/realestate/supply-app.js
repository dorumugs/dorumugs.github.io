// 시도별 아파트 착공 × 금리 대시보드.
//
// 답하려는 질문 둘 — (1) 지금 착공이 줄면 2~3년 뒤 입주가 마른다, (2) 어디가
// 먼저 얼어붙었나. 절대량으로는 서울과 대구가 같은 축에서 비교되지 않으므로
// 각 시도의 평년(2011~2019) 대비 지수로 본다.
//
// 금리는 배경이다. **상관을 보여줄 뿐 인과를 주장하지 않는다** — 그래서 이중축을
// 쓰지 않고 x축만 공유하는 2단 패널로 그린다(charts.js multiLinePanel 주석 참조).

import { multiLinePanel, legendHtml } from './charts.js';
import { CATEGORICAL, MUTED } from './palette.js';
import { showStale, LIMITS } from './freshness.js';

const root = document.querySelector('.re-app.is-supply');
const base = (root.dataset.base || '/assets/realestate').replace(/\/$/, '');

// 선택을 4개로 묶는 이유는 색이다. CATEGORICAL 은 인접쌍 대비를 검증한 4색이라
// 5개째부터는 그 검증이 깨진다. 전국은 회색 배경선이라 이 넷에 들지 않는다.
const MAX_PICKS = 4;

const METRICS = {
  index: { label: '평년=100 지수', unit: '평년=100', decimals: 0, reference: 100,
           note: '각 시도의 2011~2019년 평균 12개월 착공을 100으로 놓았습니다.' },
  mavg: { label: '12개월 이동합계', unit: '호', decimals: 0, reference: null,
          note: '최근 1년간 착공 호수입니다. 계절성이 정의상 제거됩니다.' },
  units: { label: '월별 착공 호수', unit: '호', decimals: 0, reference: null,
           note: '원계열입니다. 단발 대규모 사업 하나에 월값이 통째로 흔들립니다. '
               + '하향 정정이 있던 달은 음수로 옵니다.' },
};

const els = {
  banner: root.querySelector('.re-freshness'),
  metric: root.querySelector('.sp-metric'),
  picks: root.querySelector('.sp-picks'),
  chart: root.querySelector('.sp-chart'),
  legend: root.querySelector('.sp-legend'),
  note: root.querySelector('.sp-note'),
  cards: root.querySelector('.sp-cards'),
  limits: root.querySelector('.sp-limits'),
};

const state = { metric: 'index', picks: ['11'], data: null };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function byCode(code) {
  return state.data.regions.find((r) => r.code === code);
}

function fmt(value, decimals) {
  if (value == null) return '—';
  return decimals > 0 ? value.toFixed(decimals) : Math.round(value).toLocaleString();
}

function latestValue(series) {
  for (let i = series.length - 1; i >= 0; i -= 1) if (series[i] != null) return series[i];
  return null;
}

function renderPicks() {
  const nation = state.data.regions[0];
  const chips = state.data.regions.slice(1).map((r) => {
    const on = state.picks.includes(r.code);
    const has = r[state.metric].some((v) => v !== null);
    const full = !on && state.picks.length >= MAX_PICKS;
    return `<button type="button" class="sp-chip${on ? ' is-on' : ''}" data-code="${r.code}"`
      + `${has ? '' : ' disabled'} aria-pressed="${on}"`
      + `${full ? ' data-full="1"' : ''}>${esc(r.name)}</button>`;
  });
  els.picks.innerHTML = `<span class="sp-picks-label">시도 <b>${state.picks.length}</b>/`
    + `${MAX_PICKS}</span>${chips.join('')}`
    + `<span class="sp-always">${esc(nation.name)}은 회색 선으로 늘 깔립니다</span>`;
}

function renderChart() {
  const { months, regions, rates, provisional_from: provFrom } = state.data;
  const metric = METRICS[state.metric];
  const nation = regions[0];

  const series = [{ label: nation.name, color: MUTED, muted: true,
                    values: nation[state.metric] }];
  state.picks.forEach((code, i) => {
    const region = byCode(code);
    if (region) series.push({ label: region.name, color: CATEGORICAL[i % CATEGORICAL.length],
                              values: region[state.metric] });
  });

  const partialFrom = provFrom ? months.indexOf(provFrom) : -1;
  els.chart.innerHTML = multiLinePanel(months, [
    { label: `아파트 착공 — ${metric.label}`, unit: metric.unit,
      decimals: metric.decimals, reference: metric.reference, series },
    { label: '금리', unit: '%', decimals: 1, series: [
      { label: '기준금리', color: CATEGORICAL[0], values: rates.base },
      { label: '주택담보대출', color: CATEGORICAL[1], values: rates.mortgage }] },
  ], { partialFrom: partialFrom >= 0 ? partialFrom : null });

  els.legend.innerHTML = legendHtml(series.map((s) => ({ label: s.label, color: s.color })))
    + `<span class="sp-legend-sep"></span>`
    + legendHtml([{ label: '기준금리', color: CATEGORICAL[0] },
                  { label: '주택담보대출', color: CATEGORICAL[1] }]);

  els.note.innerHTML = `${esc(metric.note)}`
    + (provFrom ? ` <b>${esc(provFrom)}</b>부터는 잠정치라 점선입니다 — 확정되면서 값이 바뀝니다.` : '');
}

function renderCards() {
  const { regions, latest_month: latest } = state.data;
  const metric = METRICS[state.metric];
  const codes = ['00', ...state.picks];
  els.cards.innerHTML = codes.map((code, i) => {
    const region = byCode(code);
    if (!region) return '';
    const value = latestValue(region[state.metric]);
    const color = code === '00' ? MUTED : CATEGORICAL[(i - 1) % CATEGORICAL.length];
    return `<div class="sp-card"><span class="sp-card-dot" style="background:${color}"`
      + ` aria-hidden="true"></span><div class="sp-card-name">${esc(region.name)}</div>`
      + `<div class="sp-card-value">${fmt(value, metric.decimals)}`
      + `<span>${esc(metric.unit)}</span></div></div>`;
  }).join('') + `<div class="sp-card is-meta"><div class="sp-card-name">최신 통계</div>`
    + `<div class="sp-card-value">${esc(latest)}<span>기준</span></div></div>`;
}

function render() {
  renderPicks();
  renderChart();
  renderCards();
}

els.metric.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-metric]');
  if (!button) return;
  state.metric = button.dataset.metric;
  els.metric.querySelectorAll('button').forEach((b) => {
    const on = b.dataset.metric === state.metric;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', String(on));
  });
  render();
});

els.picks.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-code]');
  if (!button || button.disabled) return;
  const code = button.dataset.code;
  if (state.picks.includes(code)) {
    state.picks = state.picks.filter((c) => c !== code);
  } else if (state.picks.length < MAX_PICKS) {
    state.picks = [...state.picks, code];
  } else {
    return;  // 넘치면 조용히 무시한다. 먼저 고른 걸 몰래 밀어내면 더 헷갈린다
  }
  render();
});

(async function start() {
  try {
    const res = await fetch(`${base}/supply.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (err) {
    els.chart.innerHTML = `<p class="sp-error">자료를 불러오지 못했습니다 (${esc(err.message)}).</p>`;
    return;
  }
  showStale(els.banner, state.data.generated, LIMITS.supply, '착공 × 금리');
  els.limits.innerHTML = state.data.notes.map((n) => `<li>${esc(n)}</li>`).join('');
  render();
}());
