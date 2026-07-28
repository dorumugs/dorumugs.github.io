import { LINE, GRID, AXIS, MUTED, INK, SURFACE, CATEGORICAL } from './palette.js';

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

// 국토부 원본 단지명은 신뢰할 수 없는 외부 입력이다. innerHTML 에 그대로
// 넣으면 '<1동,2동>' 같은 이름이 태그로 파싱된다. HTML 특수문자를 이스케이프한다.
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

export function renderPanel(root, summary, detail, state) {
  const series = summary.series[state.filter][detail.sgg];
  const index = summary.months.indexOf(state.ym);
  const med = series.med.slice(0, index + 1);
  const months = summary.months.slice(0, index + 1);

  const now = med[index];
  const before = index >= 12 ? med[index - 12] : null;
  const yoy = now != null && before != null && before !== 0
    ? (now / before - 1) * 100
    : null;

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

  // 역대 최고 평당가 — KPI 타일과 나란히 두되, 오늘의 값이 아니라 기록임을
  // 배지("역대 최고")와 점선 테두리로 분명히 한다(dashboard.css re-peak-note).
  // 해당 필터로 유효 거래가 아예 없던 구는 summary.json 에 peak 자체가 없을
  // 수 있으므로(집계 스크립트 쪽 규칙) 조용히 생략한다.
  const recordPeak = series.peak;
  root.querySelector('.re-peak').innerHTML = recordPeak
    ? `<div class="re-peak-note"><span class="re-peak-badge">역대 최고</span>`
      + `<b>${recordPeak.pp.toLocaleString()}만원/평</b> · ${esc(recordPeak.date)} · `
      + `${esc(recordPeak.apt)}${recordPeak.dong ? ` (${esc(recordPeak.dong)})` : ''}</div>`
    : '';

  const partialFrom = summary.partial ? months.indexOf(summary.partial) : -1;
  root.querySelector('.re-chart').innerHTML = lineChart(months, med, { partialFrom });

  renderPeers(root, summary, state);
  renderTable(root, detail, state);
}

// 기준월부터 5·10·15·20년 전 같은 달을 훑는다(개월수가 아니라 연·월로 빼서
// 항상 "그 해 같은 달"이 나오게 한다). summary.months 범위 밖으로 나가는
// 시점은 아예 빼고, 있는 것만 돌려준다.
function peerPeriods(baseYm, months) {
  const [y, m] = baseYm.split('-').map(Number);
  return [0, 5, 10, 15, 20]
    .map((k) => `${y - k}-${String(m).padStart(2, '0')}`)
    .filter((ym) => months.includes(ym));
}

const PEER_PCT = 3;

// 한 시점에서 code 구의 중위 평당가와 ±3% 이내인 다른 구를 모두 찾는다. 하나도
// 없으면(강남구처럼 꼭대기에 있는 구) "짝이 없다"는 것 자체가 결과이므로 가장
// 가까운 구와 그 격차를 nearest 로 같이 돌려준다.
function peersAt(summary, filter, code, ym) {
  const idx = summary.months.indexOf(ym);
  const series = summary.series[filter];
  const sel = series[code] ? series[code].med[idx] : null;
  if (sel == null) return { ym, sel: null, peers: [], nearest: null };
  const peers = [];
  let nearest = null;
  for (const [c, s] of Object.entries(series)) {
    if (c === code) continue;
    const v = s.med[idx];
    if (v == null) continue;
    const gap = (v / sel - 1) * 100;
    if (Math.abs(gap) <= PEER_PCT) peers.push({ code: c, name: summary.sgg[c].name, gap });
    if (!nearest || Math.abs(gap) < Math.abs(nearest.gap)) nearest = { code: c, name: summary.sgg[c].name, gap };
  }
  peers.sort((a, b) => a.name.localeCompare(b.name, 'ko'));
  return { ym, sel, peers, nearest };
}

// 짝꿍 구 "집합"이 같은 동안은 같은 색, 바뀌면 팔레트의 다음 슬롯으로 넘어간다.
// 짝이 없는 시점(row.peers 가 비어 있음)은 애초에 "짝 집합"이 아니라 점선
// 테두리로만 표시하므로 색을 배정하지 않는다(dashboard.css is-nopeer).
function peerColors(rows) {
  let idx = -1;
  let prevSig = null;
  return rows.map((row) => {
    if (!row.peers.length) return null;
    const sig = row.peers.map((p) => p.code).sort().join('|');
    if (sig !== prevSig) {
      idx = (idx + 1) % CATEGORICAL.length;
      prevSig = sig;
    }
    return CATEGORICAL[idx];
  });
}

function renderPeers(root, summary, state) {
  const wrap = root.querySelector('.re-peers');
  const periods = peerPeriods(state.ym, summary.months);
  const rows = periods.map((ym) => peersAt(summary, state.filter, state.sgg, ym));
  const colors = peerColors(rows);

  wrap.innerHTML = rows.map((row, i) => {
    let body;
    if (row.sel == null) {
      body = '자료 없음';
    } else if (row.peers.length) {
      body = row.peers.map((p) => esc(p.name)).join(' · ');
    } else if (row.nearest) {
      const sign = row.nearest.gap >= 0 ? '+' : '';
      body = `가장 가까운 곳도 ${esc(row.nearest.name)} ${sign}${row.nearest.gap.toFixed(1)}%`;
    } else {
      body = '비교할 구가 없습니다';
    }
    const cls = row.peers.length ? '' : ' is-nopeer';
    const style = colors[i] ? ` style="border-left-color:${colors[i]}"` : '';
    return `<div class="re-peer-row${cls}"${style}>`
      + `<span class="re-peer-ym">${esc(row.ym)}</span>`
      + `<span class="re-peer-list">${body}</span></div>`;
  }).join('');
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
    + `<td>${esc(c.name)}</td><td class="is-dim">${esc(c.dong)}</td>`
    + `<td class="is-num">${c.med != null ? c.med.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.hh != null ? c.hh.toLocaleString() : '—'}</td>`
    + `<td class="is-num is-dim">${c.n.toLocaleString()}</td></tr>`).join('');
  root.querySelector('.re-table').innerHTML = rows.length
    ? `${head}<tbody>${body}</tbody>`
    : `${head}<tbody><tr><td colspan="6">최근 12개월 거래 5건 이상 단지가 없습니다.</td></tr></tbody>`;
}
