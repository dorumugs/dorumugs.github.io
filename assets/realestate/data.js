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
  if (!summaryPromise) {
    // 실패한 요청을 캐시에 남겨두면 재시도가 영영 막힌다 — 실패 시 비워서 다음 호출이 다시 받게 한다.
    summaryPromise = getJson(`${base}/summary.json`).catch((err) => {
      summaryPromise = null;
      throw err;
    });
  }
  return summaryPromise;
}

export function loadSgg(code) {
  if (!sggCache.has(code)) {
    sggCache.set(code, getJson(`${base}/sgg/${code}.json`).catch((err) => {
      sggCache.delete(code);
      throw err;
    }));
  }
  return sggCache.get(code);
}
