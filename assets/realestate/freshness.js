/* 자료가 낡았으면 화면이 스스로 말한다.

   이 대시보드들의 가장 위험한 고장은 "에러가 난다" 가 아니라 **아무 일도 안
   일어나는 것**이다. 크론이 멈추거나 수집원 화면이 바뀌어 파서가 빈 결과를
   내면, 어제 파일이 그대로 남아 오늘 값인 척한다. 실제로 ETF 대시보드가 크론
   등록이 빠진 채 "매일 갱신" 을 자처하고 있었다.

   크론 쪽에도 검사(scripts/check_freshness.py)를 뒀지만 그것만으로는 부족하다 —
   **크론이 아예 안 돌면 로그조차 안 생긴다.** 그때 낡았다는 사실을 알릴 수
   있는 건 페이지뿐이다.

   그래서 나이를 **보는 사람의 시계로** 잰다. 빌드 때 계산해 박아 두면 크론이
   멈추는 순간 그 숫자도 같이 멈춰서, 낡음 자체가 안 보이게 된다.

   임계값은 크론(scripts/freshness_api.py)보다 이틀 넉넉하다. 크론은 이미
   따로 울리므로, 화면은 주말·공휴일에 헛경보를 내지 않는 쪽이 낫다 —
   늑대 소년이 되면 진짜 고장 때 아무도 안 본다. */

export const DAY = 86400000;

/* 대시보드별 허용 나이(일). scripts/freshness_api.py 의 BUILD_LIMITS 와 짝이다. */
export const LIMITS = {
  trades: 5,
  schools: 45,
  redev: 5,
};

export function daysSince(iso) {
  if (!iso) { return null; }
  const text = String(iso).trim();
  const normal = text.length === 8
    ? `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6)}`
    : text;
  const t = Date.parse(normal);
  if (Number.isNaN(t)) { return null; }
  return Math.floor((Date.now() - t) / DAY);
}

/* 낡았으면 경고 문단을 만들어 돌려준다. 멀쩡하면 null.
   날짜를 못 읽는 경우도 낡은 것으로 본다 — 빠진 값을 '최신' 으로 처리하면
   필드 이름이 바뀌는 순간 이 경고가 통째로 무력화된다. */
export function staleNotice(iso, limitDays, what) {
  const age = daysSince(iso);
  if (age !== null && age <= limitDays) { return null; }

  const el = document.createElement('p');
  el.className = 're-stale';
  el.setAttribute('role', 'status');
  el.textContent = age === null
    ? `${what} 갱신 날짜를 읽지 못했습니다. 아래 숫자는 지금 자료가 아닐 수 있습니다.`
    : `${what}가 ${age}일째 그대로입니다. 자동 갱신이 멈췄거나 수집이 깨진 상태라, `
      + '아래 숫자는 최신이 아닙니다.';
  return el;
}

/* 경고를 대상 요소 앞에 끼운다. 이미 있으면 갈아 끼운다(다시 그릴 때 쌓이지 않게). */
export function showStale(anchor, iso, limitDays, what) {
  if (!anchor || !anchor.parentNode) { return; }
  const old = anchor.parentNode.querySelector('.re-stale');
  if (old) { old.remove(); }
  const notice = staleNotice(iso, limitDays, what);
  if (notice) { anchor.parentNode.insertBefore(notice, anchor); }
}
