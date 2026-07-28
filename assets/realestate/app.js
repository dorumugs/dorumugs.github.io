import { setBase, loadSummary, loadSgg } from './data.js';
import { initMap } from './map.js';
import {
  divergingColor, sequentialColor, DIVERGING, SEQUENTIAL, INK2, LINE, DOWN,
} from './palette.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);

const state = { view: 'seoul', metric: 'chg12', filter: '300', ym: null, sgg: null };
let summary = null;
let map = null;

// 국토부 코드 목록에서 온 시군구 이름도 신뢰할 수 없는 외부 입력이다(charts.js
// 의 단지명과 같은 이유). innerHTML 에 그대로 넣기 전에 이스케이프한다.
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

const METRICS = {
  level: { label: '중위 평당가', unit: '만원/평', kind: 'sequential' },
  chg3: { label: '3개월 변화율', unit: '%', kind: 'diverging', lag: 3 },
  chg6: { label: '6개월 변화율', unit: '%', kind: 'diverging', lag: 6 },
  chg12: { label: '12개월 변화율', unit: '%', kind: 'diverging', lag: 12 },
  peak: { label: '전고점 대비', unit: '%', kind: 'diverging' },
  turnover: { label: '거래 회전율', unit: '%', kind: 'sequential' },
};

// 지표 옆 "?" 에 뜨는 설명. metricValue() 가 실제로 계산하는 방식 그대로
// 적는다 — 무엇과 비교한 값인지, 어떤 보정이 들어가는지가 핵심이다.
const METRIC_HELP = {
  level: '그 달 그 구에서 거래된 아파트의 평당가 중위값입니다(현재 세대수 필터 기준). '
    + '평형·연식·입지가 섞여 있어 구 사이 절대 비교보다는 흐름을 보는 데 쓰고, 최근 3개월치는 '
    + '신고 지연으로 아직 다 채워지지 않았습니다.',
  chg3: '현재 세대수 필터 기준으로, 기준월의 중위 평당가를 3개월 전 같은 값과 비교한 변화율입니다. '
    + '거래 5건 미만인 얇은 달은 3개월 이동중위로 대체해 비교하며, 최근 3개월치는 신고 지연으로 값이 계속 바뀔 수 있습니다.',
  chg6: '현재 세대수 필터 기준으로, 기준월의 중위 평당가를 6개월 전 같은 값과 비교한 변화율입니다. '
    + '거래 5건 미만인 얇은 달은 3개월 이동중위로 대체해 비교하며, 최근 3개월치는 신고 지연으로 값이 계속 바뀔 수 있습니다.',
  chg12: '현재 세대수 필터 기준으로, 기준월의 중위 평당가를 12개월 전 같은 값과 비교한 변화율입니다. '
    + '거래 5건 미만인 얇은 달은 3개월 이동중위로 대체해 비교하며, 최근 3개월치는 신고 지연으로 값이 계속 바뀔 수 있습니다.',
  peak: '현재 세대수 필터 기준으로, 기준월까지 있었던 역대 최고 월 중위 평당가 대비 지금이 몇 % 인지입니다. '
    + '미래의 최고가와는 비교하지 않으므로, 과거 월을 골라 보면 "그 시점까지의" 전고점 대비라는 뜻입니다.',
  turnover: '최근 12개월 거래 건수를, 이 구에서 필터 조건을 만족하는 단지들의 총 세대수로 나눈 값입니다. '
    + '거래가 얼마나 활발했는지 보는 지표일 뿐 가격 수준과는 관계없습니다.',
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
  // 얇은 달(거래 5건 미만)은 그 달만 3개월 이동중위로 대체한다. 기준월이
  // 두꺼워도 비교 대상(n개월 전) 달이 얇으면 그 달만 스무딩해야 한다 —
  // 두 끝점 중 하나를 통째로 원값/스무딩값으로 고정하면 안 된다.
  const smooth = smoothed(series, index);
  const valueAt = (i) => {
    if (i < 0 || i >= series.med.length) return null;
    return (series.n[i] || 0) < THIN ? smooth[i] : series.med[i];
  };
  const now = valueAt(index);
  if (now == null) return null;
  if (metric === 'peak') {
    let peak = -Infinity;
    for (let i = 0; i <= index; i += 1) {
      const v = valueAt(i);
      if (v != null) peak = Math.max(peak, v);
    }
    return peak > 0 ? (now / peak - 1) * 100 : null;
  }
  const before = valueAt(index - spec.lag);
  if (before == null || before === 0) return null;
  return (now / before - 1) * 100;
}

// 지도 툴팁과 랜딩 랭킹표가 같은 문구를 쓰게 한 곳에 모은다.
function formatMetric(v, spec) {
  if (!Number.isFinite(v)) return '자료 없음';
  return spec.kind === 'diverging' ? `${v.toFixed(1)}%` : v.toLocaleString();
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
        label: `${summary.sgg[code].name} ${formatMetric(v, spec)}`,
      });
    }
    drawLegend(-span, span, 'diverging', spec.unit);
  } else {
    const min = nums.length ? Math.min(...nums) : 0;
    const max = nums.length ? Math.max(...nums) : 1;
    for (const [code, v] of raw) {
      values.set(code, {
        color: sequentialColor(v, min, max),
        label: `${summary.sgg[code].name} ${formatMetric(v, spec)}`,
      });
    }
    drawLegend(min, max, 'sequential', spec.unit);
  }
  map.paint(values);
  if (!state.sgg) renderLanding(raw, spec, index);
  writeParams();
}

// 랭킹 표 한 칸에 들어가는 20년치 추이선. 축·눈금 없이 모양만 보여 주는
// 스파크라인이라, 세로 스케일은 "행마다 각자" 최소~최대로 정규화한다 — 구
// 사이 평당가가 58배까지 벌어져 공통 스케일을 쓰면 강남 말고는 전부 납작한
// 직선이 된다. 절대 수준 비교는 옆의 중위 평당가 열이 맡는다.
const SPARK_W = 100;
const SPARK_H = 24;
const SPARK_PAD = 2.5;

// 월 중위 평당가가 가장 높았던 달. 기준월과 무관한 "전 기간(2006~) 고점"이라
// 기준월을 과거로 돌려도 움직이지 않는다 — 기준월까지만 보는 '전고점 대비'
// 지표와는 기준이 다르니 두 값을 나란히 읽을 때 주의해야 한다. 같은 값이
// 오도록 최고점을 처음 찍은 달을 고르는 방식은 charts.js 의 KPI 와 맞춘다.
function peakOf(code) {
  const series = summary.series[state.filter][code];
  const med = series && series.med;
  if (!med) return null;
  let value = -Infinity;
  let index = -1;
  for (let i = 0; i < med.length; i += 1) {
    const v = med[i];
    if (v != null && v > value) { value = v; index = i; }
  }
  return index < 0 ? null : { value, index };
}

function sparkline(code, index) {
  const series = summary.series[state.filter][code];
  const med = series && series.med;
  const pts = [];
  let min = Infinity;
  let max = -Infinity;
  if (med) {
    for (let i = 0; i < med.length; i += 1) {
      const v = med[i];
      if (v == null) continue;
      pts.push([i, v]);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  // 화성시 4개 구처럼 필터 조합에 따라 전 기간이 비는 곳이 있다. 선을 못 그리면
  // 빈칸 대신 '—' 를 놓아 "그릴 게 없다"는 것이 읽히게 한다.
  if (pts.length < 2) return '<span class="re-spark-none" aria-label="자료 없음">—</span>';
  const span = max - min || 1;
  const lastX = med.length - 1 || 1;
  // 가로도 세로와 같이 SPARK_PAD 만큼 안쪽으로 들여 그린다. 기준월이 마지막
  // 달일 때 점(r=2)이 viewBox 밖으로 나가 반쪽만 보이기 때문이다.
  const x = (i) => SPARK_PAD + (i / lastX) * (SPARK_W - SPARK_PAD * 2);
  const y = (v) => SPARK_PAD + (1 - (v - min) / span) * (SPARK_H - SPARK_PAD * 2);
  // 중간에 빈 달(300세대+ 필터에서 6%)은 선을 끊지 않고 잇는다. 톱니처럼 끊어
  // 놓으면 실제로 값이 떨어진 것처럼 읽힌다.
  const d = pts.map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const at = med[index];
  const dot = at == null ? ''
    : `<circle cx="${x(index).toFixed(1)}" cy="${y(at).toFixed(1)}" r="2" fill="${LINE}"/>`;
  // 고점 표시는 기준월 점보다 뒤에 그려 겹칠 때 위로 오게 한다 — 고점이 마지막
  // 달인 구가 많아 두 점이 거의 같은 자리에 놓인다.
  const pk = peakOf(code);
  const peakDot = pk
    ? `<circle cx="${x(pk.index).toFixed(1)}" cy="${y(pk.value).toFixed(1)}" `
      + `r="2" fill="${DOWN}"/>`
    : '';
  const first = pts[0];
  const last = pts[pts.length - 1];
  const ymOf = (i) => summary.months[i];
  const label = `${summary.sgg[code].name} 평당가 추이. `
    + `${ymOf(first[0])} ${first[1].toLocaleString()}만원에서 `
    + `${ymOf(last[0])} ${last[1].toLocaleString()}만원까지, `
    + `고점 ${pk ? `${ymOf(pk.index)} ${pk.value.toLocaleString()}` : '없음'}만원.`;
  return `<svg class="re-spark" viewBox="0 0 ${SPARK_W} ${SPARK_H}" `
    + `preserveAspectRatio="none" role="img" aria-label="${esc(label)}">`
    + `<polyline points="${d}" fill="none" stroke="${INK2}" stroke-width="1" `
    + `vector-effect="non-scaling-stroke"/>${dot}${peakDot}</svg>`;
}

// 구 선택 전 첫 화면. 새로 지역 전체 중위값을 만들지 않고, 이미 지도에 칠한
// 값을 그대로 표로 늘어놓는다 — 지도 툴팁과 값·서식이 완전히 같다. 행마다
// data-code 를 심어 두 방향 호버(행 → 지도, 지도 → 행)와 클릭 선택이 코드로
// 바로 이어지게 한다(학군 페이지의 re-list-row 와 같은 얼개).
function renderLanding(raw, spec, index) {
  // 지표를 '중위 평당가'로 고르면 지표 열과 중위값 열이 같은 값이 된다. 그때만
  // 한 열로 합친다 — 똑같은 숫자를 두 번 늘어놓는 게 더 혼란스럽다.
  const dupLevel = state.metric === 'level';
  const levelSpec = METRICS.level;
  const rows = [...raw.entries()]
    .sort((a, b) => {
      const av = Number.isFinite(a[1]) ? a[1] : -Infinity;
      const bv = Number.isFinite(b[1]) ? b[1] : -Infinity;
      return bv - av;
    })
    .map(([code, v], i) => {
      const name = summary.sgg[code].name;
      // 중위값은 지도·툴팁과 같은 metricValue() 로 뽑는다. 여기서 series.med 를
      // 직접 읽으면 나중에 계산이 바뀔 때 두 곳이 조용히 어긋난다.
      const level = dupLevel ? null : metricValue(code, 'level', index);
      const pk = peakOf(code);
      const levelCell = `<td class="is-num">${formatMetric(level, levelSpec)}</td>`;
      const peakCell = `<td class="is-num is-peak">${formatMetric(pk && pk.value, levelSpec)}</td>`;
      const metricCell = `<td class="is-num">${formatMetric(v, spec)}</td>`;
      // 열 순서는 언제나 "지금 값 → 고점 → 지표". 지표가 중위 평당가일 때는
      // 지표 열이 곧 지금 값이므로 그 열을 앞에 두고 고점을 뒤에 붙인다 —
      // 그러지 않으면 고점이 지금 값보다 왼쪽에 와서 거꾸로 읽힌다.
      return `<tr class="re-list-row" data-code="${esc(code)}" tabindex="0" `
        + `role="button" aria-label="${esc(name)} 상세 보기">`
        + `<td class="is-num is-dim">${i + 1}</td>`
        + `<td>${esc(name)}</td>`
        + (dupLevel ? metricCell + peakCell : levelCell + peakCell + metricCell)
        + `<td class="is-spark">${sparkline(code, index)}</td></tr>`;
    })
    .join('');
  const since = (summary.months[0] || '').slice(0, 4);
  // 헤더도 본문과 같은 순서 규칙을 따른다(지금 값 → 고점 → 지표).
  const levelTh = `<th class="is-num">${levelSpec.label}</th>`;
  const peakTh = '<th class="is-num is-peak">중위 고점</th>';
  const metricTh = `<th class="is-num">${spec.label}</th>`;
  root.querySelector('.re-chart').innerHTML = rows
    ? `<table class="re-table is-landing"><thead><tr><th></th><th>시군구</th>`
      + (dupLevel ? metricTh + peakTh : levelTh + peakTh + metricTh)
      + `<th class="is-spark">추이 ${since}~</th></tr></thead><tbody>${rows}</tbody></table>`
    : '<p class="re-error">표시할 데이터가 없습니다.</p>';
}

// 지도 구를 가리켰을 때(map.js 의 onHover) 랭킹 표의 해당 행을 표시한다 —
// 반대 방향(행 → 구)은 map.setHovered() 가 맡는다. 구가 선택되면 .re-chart
// 는 표 대신 추이 차트가 되므로(.re-table 이 없으므로) 조용히 아무 일도
// 하지 않는다.
function highlightRankRow(code) {
  const table = root.querySelector('.re-chart .re-table');
  const prev = table && table.querySelector('tr.re-list-row.is-hover');
  if (prev) prev.classList.remove('is-hover');
  if (!code || !table) return;
  const tr = Array.from(table.querySelectorAll('tr.re-list-row'))
    .find((r) => r.dataset.code === code);
  if (tr) tr.classList.add('is-hover');
}

// 랭킹 표(.re-chart 안, 선택 전에만 존재) 행 인터랙션은 위임한다 — 탭 전환·
// "목록으로" 마다 innerHTML 을 통째로 새로 써서 개별 리스너는 매번 사라지기
// 때문이다(schools-app.js 의 bindTableInteractions 와 같은 이유). 위임 리스너는
// 항상 존재하는 .re-chart 컨테이너에 한 번만 건다.
function rowAt(target) {
  return target.closest ? target.closest('tr.re-list-row') : null;
}

function bindLandingInteractions() {
  const chart = root.querySelector('.re-chart');
  chart.addEventListener('click', (e) => {
    const tr = rowAt(e.target);
    if (tr) selectSgg(tr.dataset.code);
  });
  chart.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const tr = rowAt(e.target);
    if (!tr) return;
    e.preventDefault();
    selectSgg(tr.dataset.code);
  });
  chart.addEventListener('mouseover', (e) => {
    const tr = rowAt(e.target);
    if (tr) map.setHovered(tr.dataset.code);
  });
  chart.addEventListener('mouseout', (e) => {
    const tr = rowAt(e.target);
    if (!tr) return;
    // 같은 행 안의 셀 사이를 옮겨다니는 것뿐이면(relatedTarget 이 여전히 이
    // tr 안) 무시한다 — 아니면 행을 벗어날 때마다 반짝이며 지도 색이 껐다
    // 켜졌다 한다.
    if (e.relatedTarget && tr.contains(e.relatedTarget)) return;
    map.setHovered(null);
  });
  chart.addEventListener('focusin', (e) => {
    const tr = rowAt(e.target);
    if (tr) map.setHovered(tr.dataset.code);
  });
  chart.addEventListener('focusout', (e) => {
    const tr = rowAt(e.target);
    if (!tr) return;
    if (e.relatedTarget && tr.contains(e.relatedTarget)) return;
    map.setHovered(null);
  });
}

// "목록으로" — 구를 고르면 KPI/차트/단지 랭킹표가 랭킹 표(.re-chart)를
// 대체한다. 되돌아갈 방법이 없으면 갇히므로 학군 페이지와 같은 이름·위치의
// 버튼으로 초기 화면을 되돌린다.
function showLanding() {
  state.sgg = null;
  map.setSelected(null);
  map.setHovered(null);
  root.querySelector('.re-panel-title').textContent = '지역을 선택하세요';
  root.querySelector('.re-back-btn').hidden = true;
  root.querySelector('.re-kpis').innerHTML = '';
  root.querySelector('.re-peak').innerHTML = '';
  root.querySelector('.re-chart-heading').hidden = true;
  root.querySelector('.re-peers-heading').hidden = true;
  root.querySelector('.re-peers').innerHTML = '';
  root.querySelector('.re-rank-heading').hidden = true;
  root.querySelector('.re-table').innerHTML = '';
  repaint();
}

function drawLegend(min, max, kind, unit) {
  const el = root.querySelector('.re-legend');
  // 램프는 palette.js 한 곳에서만 정의한다. 여기에 색을 다시 적지 말 것.
  const ramp = kind === 'diverging' ? DIVERGING : SEQUENTIAL;
  // 스와치(<i>)는 색만 있고 글자가 없어 스크린리더에 읽을 게 없다. 대신
  // 컨테이너를 role="img" + aria-label 하나로 묶어 값 전체를 설명한다.
  const swatches = ramp.map((c) => `<i aria-hidden="true" style="background:${c}"></i>`).join('');
  const fmt = (v) => (kind === 'diverging' ? `${v.toFixed(0)}%` : Math.round(v).toLocaleString());
  const lo = fmt(min);
  const hi = `${fmt(max)}${kind === 'sequential' ? ` ${unit}` : ''}`;
  el.innerHTML = `<span aria-hidden="true">${lo}</span>${swatches}<span aria-hidden="true">${hi}</span>`;
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label', kind === 'diverging'
    ? `지도 색상 범례. 파랑 ${lo}에서 회색을 지나 빨강 ${hi}까지, 하락에서 상승 순서.`
    : `지도 색상 범례. 옅은 색 ${lo}에서 짙은 색 ${hi}까지, 값이 클수록 진하다.`);
}

// "?" 팝오버 내용을 현재 선택된 지표로 채운다. 지표를 바꿀 때마다 다시
// 불러야 팝오버가 항상 지금 보고 있는 지표를 설명한다.
function updateMetricHelp() {
  const pop = root.querySelector('.re-help-pop');
  if (pop) pop.textContent = METRIC_HELP[state.metric] || '';
}

// re-help-pop 은 position:fixed 라 뷰포트 기준 좌표를 직접 계산해야 한다. 버튼
// 왼쪽에 맞춰 펼치되, 지표 필드가 화면 오른쪽에 붙어 있을 때(390px 폭에서
// 흔하다) 뷰포트를 넘기지 않도록 안쪽으로 당긴다 — .re-app 의 overflow-x:hidden
// 은 넘친 걸 "숨길" 뿐 읽히게 해 주진 않으므로, 애초에 안 넘치게 좌표를 잡는다.
function positionMetricHelp() {
  const btn = root.querySelector('.re-help-btn');
  const pop = root.querySelector('.re-help-pop');
  if (!btn || !pop) return;
  const r = btn.getBoundingClientRect();
  const margin = 8;
  const pw = pop.offsetWidth || 240;
  let left = Math.min(r.left, window.innerWidth - pw - margin);
  left = Math.max(left, margin);
  pop.style.left = `${left}px`;
  pop.style.top = `${r.bottom + 6}px`;
}

// 마우스오버·키보드 포커스·탭 세 입력 모두 열 수 있어야 한다(hover 만으론
// 터치 기기에서 닿을 수 없다). hover/focus 는 손을 떼면 닫히는 임시 상태,
// 탭(click)은 다시 탭하거나 바깥을 탭하거나 Esc 를 눌러야 닫히는 고정 상태로
// 나눠 두 입력 방식이 서로 방해하지 않게 한다.
function bindMetricHelp() {
  const help = root.querySelector('.re-help');
  const btn = root.querySelector('.re-help-btn');
  const pop = root.querySelector('.re-help-pop');
  if (!help || !btn || !pop) return;
  let sticky = false;
  let hovering = false;
  const sync = () => {
    const show = sticky || hovering;
    if (show) {
      pop.style.display = 'block';
      positionMetricHelp();
    } else {
      pop.style.display = 'none';
    }
    help.classList.toggle('is-open', show);
    btn.setAttribute('aria-expanded', String(show));
  };
  btn.addEventListener('mouseenter', () => { hovering = true; sync(); });
  btn.addEventListener('mouseleave', () => { hovering = false; sync(); });
  btn.addEventListener('focus', () => { hovering = true; sync(); });
  btn.addEventListener('blur', () => { hovering = false; sync(); });
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    sticky = !sticky;
    sync();
  });
  document.addEventListener('click', (e) => {
    if (sticky && !help.contains(e.target)) { sticky = false; sync(); }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && (sticky || hovering)) {
      sticky = false;
      hovering = false;
      sync();
      btn.focus();
    }
  });
  // 뜬 채로 스크롤되면 fixed 좌표가 버튼에서 어긋나 보인다 — 탭으로 고정한
  // 상태만 닫는다(가벼운 hover는 어차피 곧 사라질 상태라 그냥 둬도 된다).
  window.addEventListener('scroll', () => {
    if (sticky) { sticky = false; sync(); }
  }, { passive: true });
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
  map.setHovered(null);
  root.querySelector('.re-panel-title').textContent = summary.sgg[code].name;
  // 랜딩 상태에서는 이 두 헤딩과 목록으로 버튼을 숨겨 뒀다. 구를 고르면 그
  // 아래 실제 내용과 함께 되살린다.
  root.querySelector('.re-chart-heading').hidden = false;
  root.querySelector('.re-peers-heading').hidden = false;
  root.querySelector('.re-rank-heading').hidden = false;
  root.querySelector('.re-back-btn').hidden = false;
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
      // 탭은 지역 선택이다 — 고른 구가 새 지역에 없을 수도 있고, 상세가 열린
      // 채로는 repaint() 가 renderLanding() 을 건너뛰어 목록이 영영 안 돌아온다.
      // 그래서 탭을 누르면 항상 그 지역의 전체 목록으로 되돌린다
      // (schools-app.js 의 지역 탭과 같은 규칙). showLanding() 이 repaint() 까지 한다.
      if (state.sgg) showLanding();
      else repaint();
    });
  });
  root.querySelector('.re-metric').addEventListener('change', (e) => {
    state.metric = e.target.value;
    repaint();
    updateMetricHelp();
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
  root.querySelector('.re-back-btn').addEventListener('click', showLanding);
  bindLandingInteractions();
  bindMetricHelp();
}

// 글에서 특정 화면을 바로 가리킬 수 있게 상태를 주소에 싣는다.
//   /real-estate/?sgg=11680&metric=chg12&ym=2026-06&filter=300&view=seoul
function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  const viewValid = ['seoul', 'gyeonggi', 'all'].includes(view);
  if (viewValid) state.view = view;
  const metric = q.get('metric');
  // 대괄호 접근은 '__proto__' 같은 값에서도 진짜 값을 돌려주므로
  // hasOwnProperty 로 실제 소유 키인지 반드시 확인한다.
  if (metric && Object.prototype.hasOwnProperty.call(METRICS, metric)) state.metric = metric;
  const filter = q.get('filter');
  if (filter === '300' || filter === 'all') state.filter = filter;
  const ym = q.get('ym');
  if (ym && summary.months.includes(ym)) state.ym = ym;
  const sgg = q.get('sgg');
  if (sgg && Object.prototype.hasOwnProperty.call(summary.sgg, sgg)) {
    state.sgg = sgg;
    // 쓸 만한 view= 가 없으면 고른 구의 시도 코드로 뷰를 추정한다. 그러지 않으면
    // '?sgg=41135' 처럼 경기 구를 가리키는 링크가 서울 지도로 열린다.
    // 값이 '없을 때'가 아니라 '유효하지 않을 때'로 판단해야 한다. 오타나 잘린 URL로
    // ?view=bogus 가 들어오면 존재 여부로만 보는 순간 추정이 막혀 같은 버그가 되살아나고,
    // writeParams 가 그 상태를 주소에 다시 써서 잘못된 링크가 그대로 굳는다.
    if (!viewValid) state.view = sgg.startsWith('11') ? 'seoul' : 'gyeonggi';
  }
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

  map = initMap(root, { onSelect: selectSgg, onHover: highlightRankRow });
  fillMonths();
  bind();

  root.querySelectorAll('.re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  root.querySelector('.re-metric').value = state.metric;
  updateMetricHelp();
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
