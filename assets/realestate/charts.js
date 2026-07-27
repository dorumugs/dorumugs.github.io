import { LINE, GRID, AXIS, MUTED, INK, SURFACE } from './palette.js';

const W = 520, H = 190, PAD_L = 46, PAD_R = 10, PAD_T = 12, PAD_B = 24;

// 목표 눈금 개수(대략 5개)에 맞춰 1/2/2.5/5/10 배수 중 가장 가까운 step 을 고른다.
// 10의 거듭제곱 경계 근처에서 눈금이 3개~10개 사이로 들쭉날쭉해지는 문제를 막는다.
function niceTicks(max) {
  if (!(max > 0)) return [0];
  const target = 5;
  const magnitude = Math.pow(10, Math.floor(Math.log10(max / target)));
  let best = null;
  for (const m of [1, 2, 2.5, 5, 10]) {
    const step = m * magnitude;
    const count = Math.floor(max / step) + 1;
    if (!best || Math.abs(count - target) < Math.abs(best.count - target)) best = { step, count };
  }
  const out = [];
  for (let v = 0; v <= max; v += best.step) out.push(v);
  return out;
}

export function lineChart(months, values, { partialFrom } = {}) {
  const finite = values.filter((v) => v != null);
  if (!finite.length) {
    return `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" `
      + `text-anchor="middle" font-size="12" fill="${MUTED}">자료 없음</text></svg>`;
  }
  const max = Math.max(...finite) * 1.08;
  const n = months.length;
  const iw = W - PAD_L - PAD_R, ih = H - PAD_T - PAD_B;
  const X = (i) => PAD_L + (n > 1 ? (i / (n - 1)) * iw : iw / 2);
  const Y = (v) => PAD_T + (1 - v / max) * ih;

  const parts = [`<svg viewBox="0 0 ${W} ${H}" font-family="system-ui,-apple-system,sans-serif" `
    + `role="img" aria-label="평당가 추이">`];

  for (const t of niceTicks(max)) {
    parts.push(`<line x1="${PAD_L}" y1="${Y(t).toFixed(1)}" x2="${W - PAD_R}" `
      + `y2="${Y(t).toFixed(1)}" stroke="${GRID}" stroke-width="1"/>`);
    parts.push(`<text x="${PAD_L - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end" `
      + `font-size="9" fill="${MUTED}">${Math.round(t).toLocaleString()}</text>`);
  }
  // 축 눈금은 만원/평 단위다. 숫자만 봐서는 단위를 알 수 없으므로 한 번만 표기한다.
  parts.push(`<text x="${PAD_L - 6}" y="${(PAD_T - 3).toFixed(1)}" text-anchor="end" `
    + `font-size="8.5" fill="${MUTED}">만원/평</text>`);

  for (let i = 0; i < n; i += 1) {
    if (!months[i].endsWith('-01') || Number(months[i].slice(0, 4)) % 5 !== 0) continue;
    parts.push(`<text x="${X(i).toFixed(1)}" y="${H - 7}" text-anchor="middle" `
      + `font-size="9" fill="${MUTED}">${months[i].slice(0, 4)}</text>`);
  }

  parts.push(`<line x1="${PAD_L}" y1="${Y(0).toFixed(1)}" x2="${W - PAD_R}" `
    + `y2="${Y(0).toFixed(1)}" stroke="${AXIS}" stroke-width="1"/>`);

  // 값이 빈 달이 있어도 선이 끊기도록 구간별로 나눠 그린다
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      parts.push(`<polyline points="${run.join(' ')}" fill="none" stroke="${LINE}" `
        + `stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`);
    }
    run = [];
  };
  values.forEach((v, i) => {
    if (v == null) { flush(); return; }
    run.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
  });
  flush();

  let lastIdx = -1;
  for (let i = n - 1; i >= 0; i -= 1) if (values[i] != null) { lastIdx = i; break; }
  if (lastIdx >= 0) {
    const lx = X(lastIdx), ly = Y(values[lastIdx]);
    parts.push(`<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="4" fill="${LINE}" `
      + `stroke="${SURFACE}" stroke-width="2"/>`);
    parts.push(`<text x="${(lx - 7).toFixed(1)}" y="${(ly - 9).toFixed(1)}" text-anchor="end" `
      + `font-size="11" font-weight="600" fill="${INK}">`
      + `${values[lastIdx].toLocaleString()}만원/평</text>`);
  }
  if (partialFrom != null && partialFrom >= 0) {
    parts.push(`<line x1="${X(partialFrom).toFixed(1)}" y1="${PAD_T}" `
      + `x2="${X(partialFrom).toFixed(1)}" y2="${(H - PAD_B).toFixed(1)}" `
      + `stroke="${MUTED}" stroke-width="1" stroke-dasharray="3 3"/>`);
  }
  parts.push('</svg>');
  return parts.join('');
}

function kpi(label, value, unit, delta, direction) {
  const cls = direction > 0 ? 'is-up' : direction < 0 ? 'is-down' : '';
  return `<div class="re-kpi"><div class="re-kpi-label">${label}</div>`
    + `<div class="re-kpi-value">${value}<span>${unit}</span></div>`
    + `<div class="re-kpi-delta ${cls}">${delta}</div></div>`;
}

function pct(v) {
  return v == null ? '자료 없음' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
}

export function renderPanel(root, summary, detail, state) {
  const series = summary.series[state.filter][detail.sgg];
  const index = summary.months.indexOf(state.ym);
  const med = series.med.slice(0, index + 1);
  const months = summary.months.slice(0, index + 1);

  const now = med[index];
  const yoy = index >= 12 && med[index - 12] ? (now / med[index - 12] - 1) * 100 : null;

  let n12 = 0, prev12 = 0;
  for (let i = Math.max(0, index - 11); i <= index; i += 1) n12 += series.n[i] || 0;
  for (let i = Math.max(0, index - 23); i <= index - 12; i += 1) prev12 += series.n[i] || 0;
  const volDelta = prev12 ? (n12 / prev12 - 1) * 100 : null;
  // 신고 지연으로 최근 3개월치는 아직 덜 들어온 상태다. 이 구간이 12개월 창에
  // 걸쳐 있으면 등락을 사실처럼 색으로 보여주지 않고 문구로만 알린다.
  const settling = index >= summary.months.length - 3;

  let peak = -Infinity, peakAt = null;
  med.forEach((v, i) => { if (v != null && v > peak) { peak = v; peakAt = months[i]; } });
  const fromPeak = now != null && peak > 0 ? (now / peak - 1) * 100 : null;

  root.querySelector('.re-kpis').innerHTML = [
    kpi('중위 평당가', now != null ? now.toLocaleString() : '—', '만원',
      `전년 대비 ${pct(yoy)}`, yoy == null ? 0 : Math.sign(yoy)),
    kpi('최근 12개월 거래', n12.toLocaleString(), '건',
      settling
        ? `직전 12개월 대비 ${pct(volDelta)} · 최근 3개월 신고 지연으로 과소 집계`
        : `직전 12개월 대비 ${pct(volDelta)}`,
      settling || volDelta == null ? 0 : Math.sign(volDelta)),
    kpi('전고점 대비', fromPeak != null ? fromPeak.toFixed(1) : '—', '%',
      peakAt ? `${peakAt} ${peak.toLocaleString()}만원/평` : '—', 0),
  ].join('');

  const partialFrom = summary.partial ? months.indexOf(summary.partial) : -1;
  root.querySelector('.re-chart').innerHTML = lineChart(months, med, { partialFrom });

  renderTable(root, detail, state);
}

function renderTable(root, detail, state) {
  const min300 = state.filter === '300';
  const rows = detail.complexes
    .filter((c) => c.n >= 5 && (!min300 || (c.hh != null && c.hh >= 300)))
    .slice(0, 30);
  const head = '<thead><tr><th></th><th>단지</th><th>법정동</th>'
    + '<th class="is-num">평당가(만원)</th><th class="is-num">세대</th>'
    + '<th class="is-num">거래</th></tr></thead>';
  const body = rows.map((c, i) => `<tr><td class="is-num is-dim">${i + 1}</td>`
    + `<td>${c.name}</td><td class="is-dim">${c.dong}</td>`
    + `<td class="is-num">${c.med != null ? c.med.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.hh != null ? c.hh.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.n}</td></tr>`).join('');
  root.querySelector('.re-table').innerHTML = rows.length
    ? `${head}<tbody>${body}</tbody>`
    : `${head}<tbody><tr><td colspan="6">최근 12개월 거래 5건 이상 단지가 없습니다.</td></tr></tbody>`;
}
