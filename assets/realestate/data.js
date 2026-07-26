// 집계 JSON 을 받아 캐시한다. 구별 상세는 처음 누를 때만 받는다.

let base = '/assets/realestate';
let summaryPromise = null;
const sggCache = new Map();

export function setBase(path) {
  base = path.replace(/\/$/, '');
}

async function getJson(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

export function loadSummary() {
  if (!summaryPromise) summaryPromise = getJson(`${base}/summary.json`);
  return summaryPromise;
}

export function loadSgg(code) {
  if (!sggCache.has(code)) sggCache.set(code, getJson(`${base}/sgg/${code}.json`));
  return sggCache.get(code);
}
