// 데이터 색은 dataviz 기준 팔레트 그대로다 — 색맹 대비·명도 검증을 통과한 조합이라
// 바꾸지 말 것. 잉크·회색·표면만 이 사이트에 맞춰 바꿨다. 기준 팔레트의 회색은
// 따뜻한 계열인데 이 블로그는 차가운 회색($dark-gray #3d4144, $primary-color #6f777d)을
// 쓰므로, 그대로 두면 대시보드만 색이 튀어 보인다.
// dashboard.css 의 --re-ink/--re-ink2/--re-muted/--re-grid 와 같은 값을 유지할 것.

export const SEQUENTIAL = [
  '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
  '#2a78d6', '#256abf', '#184f95', '#0d366b',
];

// 파랑(하락) ↔ 회색(변화 없음) ↔ 빨강(상승). 국내 관행과 같은 방향이다.
export const DIVERGING = [
  '#184f95', '#2a78d6', '#86b6ef', '#f0efec', '#f0a8a8', '#d03b3b', '#a02020',
];

export const INK = '#3d4144';
export const INK2 = '#646769';
export const MUTED = '#7a8288';
export const GRID = '#e3e5e7';
export const AXIS = '#bcc0c4';
export const LINE = '#2a78d6';
export const UP = '#006300';
export const DOWN = '#d03b3b';
// 발산 램프의 중립(#f0efec)과 확실히 구분돼야 한다. 기존 #e8e8e4 는 중립과 거의
// 같은 밝기라, 세대수 자료가 없는 화성시 4개 구가 '변화 없음' 으로 읽혔다.
export const NO_DATA = '#c9ced1';
export const SURFACE = '#ffffff';

// 기준 팔레트의 범주형 슬롯 1~4(파랑·주황·초록·황토) — dataviz 체커로 인접쌍(라이트
// 서페이스) 전부 통과를 확인한 4색 조합이다. 슬롯 1·2(파랑·주황)는 학군 지도에서
// 학교급(초/중)을 구분하는 색으로 쓴다. 슬롯 1~4 전체는 실거래 상세의 "5년 단위
// 짝꿍 동네" 표에서 시기별 짝꿍 구 조합이 바뀔 때 색을 바꾸는 데 쓴다(charts.js
// peerColors). CSS 는 이 모듈을 import 할 수 없어 assets/realestate/schools.css 에
// 슬롯 1·2 값을 그대로 옮겨 적었다 — 그 두 슬롯 값을 바꾸면 그쪽도 바꿀 것.
export const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];

export function rampColor(ramp, t) {
  if (!Number.isFinite(t)) return NO_DATA;
  const i = Math.min(ramp.length - 1, Math.max(0, Math.floor(t * ramp.length)));
  return ramp[i];
}

// span 은 한쪽 팔의 길이. -span..+span 을 발산 램프 전체에 대응시킨다.
export function divergingColor(value, span) {
  if (!Number.isFinite(value) || !(span > 0)) return NO_DATA;
  const clamped = Math.max(-span, Math.min(span, value));
  return rampColor(DIVERGING, (clamped + span) / (2 * span) * 0.999);
}

export function sequentialColor(value, min, max) {
  if (!Number.isFinite(value) || !(max > min)) return NO_DATA;
  const t = (Math.max(min, Math.min(max, value)) - min) / (max - min);
  return rampColor(SEQUENTIAL, t * 0.999);
}
