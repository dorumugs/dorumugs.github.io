// dataviz 기준 팔레트(라이트 표면 #fcfcfb). 값을 바꾸지 말 것 — 검증을 통과한 조합이다.

export const SEQUENTIAL = [
  '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
  '#2a78d6', '#256abf', '#184f95', '#0d366b',
];

// 파랑(하락) ↔ 회색(변화 없음) ↔ 빨강(상승). 국내 관행과 같은 방향이다.
export const DIVERGING = [
  '#184f95', '#2a78d6', '#86b6ef', '#f0efec', '#f0a8a8', '#d03b3b', '#a02020',
];

export const INK = '#0b0b0b';
export const INK2 = '#52514e';
export const MUTED = '#898781';
export const GRID = '#e1e0d9';
export const AXIS = '#c3c2b7';
export const LINE = '#2a78d6';
export const UP = '#006300';
export const DOWN = '#d03b3b';
export const NO_DATA = '#e8e8e4';
export const SURFACE = '#fcfcfb';

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
