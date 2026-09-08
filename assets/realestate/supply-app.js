// 시도별 아파트 착공 × 금리 대시보드.
//
// 답하려는 질문 둘 — (1) 지금 착공이 줄면 2~3년 뒤 입주가 마른다, (2) 어디가
// 먼저 얼어붙었나. 절대량으로는 서울과 대구가 같은 축에서 비교되지 않으므로
// 각 시도의 평년(2011~2019) 대비 지수로 본다.
//
// 금리는 배경이다. **상관을 보여줄 뿐 인과를 주장하지 않는다** — 그래서 이중축을
// 쓰지 않고 x축만 공유하는 2단 패널로 그린다(charts.js multiLinePanel 주석 참조).

import { multiLinePanel, legendHtml } from './charts.js';
import { CATEGORICAL, MUTED, AXIS } from './palette.js';
import { showStale, LIMITS } from './freshness.js';

const root = document.querySelector('.re-app.is-supply');
const base = (root.dataset.base || '/assets/realestate').replace(/\/$/, '');

// 선택을 4개로 묶는 이유는 색이다. CATEGORICAL 은 인접쌍 대비를 검증한 4색이라
// 5개째부터는 그 검증이 깨진다. 전국은 회색 배경선이라 이 넷에 들지 않는다.
const MAX_PICKS = 4;

// short 는 지표 셋을 늘 펼쳐 두는 설명이다. 탭 이름만으로는 무엇을 세는 값인지
// 알 수 없어서(특히 '지수'), 고르기 전에 읽을 수 있어야 한다. note 는 고른 지표
// 하나에만 해당하는 주의사항이라 차트 밑에 남긴다.
const METRICS = {
  index: { label: '평년=100 지수', unit: '평년=100', decimals: 0, reference: 100,
           short: '그 시도의 <b>2011~2019년 평균</b>을 100으로 놓은 값. '
                + '68이면 평년의 68% 수준입니다.',
           note: '지역마다 규모가 달라 호수로는 서울과 대구를 같은 축에 놓을 수 '
               + '없어, 각자의 평년을 기준선 100으로 잡았습니다.' },
  mavg: { label: '12개월 이동합계', unit: '호', decimals: 0, reference: null,
          short: '최근 <b>12개월 착공을 더한 값</b>. 계절 효과가 정의상 사라집니다.',
          note: '창 안에 결측이 하나라도 있으면 그 달은 비워 둡니다 — 덜 더한 합을 '
              + '1년치인 척 그리지 않기 위해서입니다.' },
  units: { label: '월별 착공 호수', unit: '호', decimals: 0, reference: null,
           short: '그 달에 착공한 <b>호수 그대로</b>. 큰 사업 하나에 크게 흔들립니다.',
           note: '원계열이라 톱니처럼 보입니다. 통계누리가 하향 정정한 달은 음수로 '
               + '옵니다 — 오류가 아니라 그 달의 정정분입니다.' },
};

// 스탯 타일의 스파크라인. 12개월치를 회색으로 깔고 마지막 점만 계열 색으로 찍는다.
const SPARK = { points: 12, w: 64, h: 18 };

const els = {
  banner: root.querySelector('.re-freshness'),
  metric: root.querySelector('.sp-metric'),
  defs: root.querySelector('.sp-defs'),
  picks: root.querySelector('.sp-picks'),
  chart: root.querySelector('.sp-chart'),
  legend: root.querySelector('.sp-legend'),
  note: root.querySelector('.sp-note'),
  cards: root.querySelector('.sp-cards'),
  cardsFoot: root.querySelector('.sp-cards-foot'),
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

function latestIndex(series) {
  for (let i = series.length - 1; i >= 0; i -= 1) if (series[i] != null) return i;
  return -1;
}

function latestValue(series) {
  const i = latestIndex(series);
  return i < 0 ? null : series[i];
}

/* 최근 12개월 스파크라인. 값이 하나뿐이면 선이 그려지지 않으므로 아예 내지 않는다.
   선은 회색이고 마지막 점만 계열 색이다 — 타일에서 눈이 먼저 가야 할 곳은 큰
   숫자이지 이 미니 차트가 아니다. 점에 표면색 링을 둘러 선과 겹쳐도 읽히게 한다. */
function sparkline(series, color) {
  const end = latestIndex(series);
  if (end < 0) return '';
  const window = series.slice(Math.max(0, end - SPARK.points + 1), end + 1);
  const values = window.filter((v) => v != null);
  if (values.length < 2) return '';

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  // 끝점 지름만큼 오른쪽을 비워 둔다. 안 그러면 마지막 점이 뷰박스에 반쯤 잘린다.
  const plotW = SPARK.w - 4;
  const step = plotW / (window.length - 1);
  const y = (v) => SPARK.h - 2 - ((v - min) / span) * (SPARK.h - 4);

  const points = window
    .map((v, i) => (v == null ? null : `${(i * step).toFixed(1)},${y(v).toFixed(1)}`))
    .filter(Boolean).join(' ');
  const lastY = y(window[window.length - 1]).toFixed(1);

  return `<svg class="sp-spark" viewBox="0 0 ${SPARK.w} ${SPARK.h}" `
    + `width="${SPARK.w}" height="${SPARK.h}" aria-hidden="true" focusable="false">`
    + `<polyline points="${points}" fill="none" stroke="${AXIS}" stroke-width="1.4" `
    + `stroke-linejoin="round" stroke-linecap="round"/>`
    + `<circle cx="${plotW}" cy="${lastY}" r="2.6" fill="${color}" `
    + `stroke="#fff" stroke-width="1.6"/></svg>`;
}

/* 1년 전 대비. 방향에 색을 입히지 않는다 — 착공이 느는 게 좋은 일인지 나쁜 일인지는
   보는 사람이 집을 사려는 쪽인지 가진 쪽인지에 달렸고, 이 화면은 그걸 정해 주지
   않기로 했다(이중축을 안 쓴 것과 같은 이유). 방향은 화살표만으로 알린다. */
function yearDelta(series, decimals) {
  const end = latestIndex(series);
  if (end < 12) return null;
  const now = series[end];
  const before = series[end - 12];
  if (now == null || before == null) return null;
  const diff = now - before;
  const arrow = diff > 0 ? '▲' : (diff < 0 ? '▼' : '＝');
  return `${arrow} ${fmt(Math.abs(diff), decimals)}`;
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

function renderDefs() {
  els.defs.innerHTML = Object.entries(METRICS).map(([key, m]) => {
    const on = key === state.metric;
    return `<div class="sp-def${on ? ' is-on' : ''}">`
      + `<dt>${esc(m.label)}</dt><dd>${m.short}</dd></div>`;
  }).join('');
}

/* 스탯 타일. 지역 이름(라벨) · 큰 숫자(값) · 1년 전 대비(델타) · 12개월 스파크라인
   (추세) 넷으로 짠다. 숫자에 tabular-nums 를 쓰지 않는다 — 세로로 줄맞춤할 열이
   아니라 타일마다 홀로 서는 큰 값이라, 고정폭이면 '121' 같은 수가 헐거워 보인다. */
function renderCards() {
  const { latest_month: latest, provisional_from: provFrom } = state.data;
  const metric = METRICS[state.metric];
  const codes = ['00', ...state.picks];

  els.cards.innerHTML = codes.map((code, i) => {
    const region = byCode(code);
    if (!region) return '';
    const series = region[state.metric];
    const color = code === '00' ? MUTED : CATEGORICAL[(i - 1) % CATEGORICAL.length];
    const delta = yearDelta(series, metric.decimals);
    return `<div class="sp-card">`
      + `<div class="sp-card-name"><i style="background:${color}" aria-hidden="true"></i>`
      + `${esc(region.name)}</div>`
      + `<div class="sp-card-value">${fmt(latestValue(series), metric.decimals)}`
      + `<span>${esc(metric.unit)}</span></div>`
      + `<div class="sp-card-foot">`
      + `<span class="sp-card-delta">${delta ? `1년 전 ${delta}` : '1년 전 자료 없음'}</span>`
      + sparkline(series, color)
      + `</div></div>`;
  }).join('');

  els.cardsFoot.innerHTML = `<b>${esc(latest)}</b> 기준 · 최근 12개월 추세`
    + (provFrom ? ` · ${esc(provFrom)}부터는 잠정치` : '');
}

function render() {
  renderDefs();
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
