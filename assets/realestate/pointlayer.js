// 지도 위에 숙소 점을 캔버스로 얹는다.
//
// 왜 SVG 가 아니라 캔버스인가
//   학군 지도(schoolmap.js)는 점이 38개라 SVG <circle> 로 그린다. 여기는 전국
//   격자가 만 단위, 시군구 하나만 봐도 수천 개다. 그만큼의 DOM 노드를 만들면
//   모바일에서 스크롤조차 버벅인다.
//
// 좌표 변환
//   위경도 → SVG 사용자 좌표는 build_geo.py 의 project() 와 같은 식을 쓴다
//   (파라미터는 airbnb.json 의 projection). SVG 사용자 좌표 → 화면 픽셀은
//   getScreenCTM() 으로 얻는다 — viewBox 를 바꿔 확대해도, 창 크기가 바뀌어도
//   그 행렬만 다시 읽으면 맞는다.

import { CATEGORICAL } from './palette.js';

// 격자 칸의 화면 반지름 범위(px). 숫자가 커도 한없이 키우지 않는다 — 서울
// 도심에서 칸들이 서로 뭉쳐 덩어리가 된다.
const CELL_MIN_R = 1.6;
const CELL_MAX_R = 7;

// 개별 숙소 점. 한 시군구만 볼 때 쓰므로 작고 반투명하게 겹쳐 밀도를 보인다.
const DOT_R = 2.2;

// 점 색. 아래 시군구 색칠이 파랑 계열이라 같은 파랑으로 찍으면 서로 묻힌다.
// 색은 시군구 밀도, 크기는 격자 숙소 수 — 채널을 갈라야 둘 다 읽힌다.
// palette.js 의 범주형 슬롯 2(주황)로, 그 파일에서 대비가 검증된 값이다.
const CELL_COLOR = CATEGORICAL[1];
const DOT_COLOR = CATEGORICAL[1];

export function toSvgXy(lat, lon, p) {
  return [
    (lon - p.min_lon) * p.k / p.span_x * p.width,
    (p.max_lat - lat) / p.span_y * p.height,
  ];
}

export function initPointLayer(root, projection) {
  const wrap = root.querySelector('.re-map-wrap');
  const svg = wrap.querySelector('svg.re-map');
  const canvas = wrap.querySelector('canvas.re-points');
  const ctx = canvas.getContext('2d');

  // 지금 그리고 있는 것. resize·확대 때 같은 것을 다시 그린다.
  let current = { kind: 'none', rows: [], max: 1 };

  function resize() {
    const box = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(box.width * dpr));
    const h = Math.max(1, Math.round(box.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return { box, dpr };
  }

  // 위경도를 캔버스 픽셀로. SVG 의 화면 행렬을 거치므로 viewBox 가 바뀌어도 맞는다.
  function projector() {
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const canvasBox = canvas.getBoundingClientRect();
    return (lat, lng) => {
      const [x, y] = toSvgXy(lat, lng, projection);
      return [
        ctm.a * x + ctm.c * y + ctm.e - canvasBox.left,
        ctm.b * x + ctm.d * y + ctm.f - canvasBox.top,
      ];
    };
  }

  function clear() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function redraw() {
    const { box, dpr } = resize();
    clear();
    const project = projector();
    if (!project || current.kind === 'none' || !current.rows.length) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    if (current.kind === 'grid') {
      // 칸 개수의 제곱근으로 크기를 잡는다. 넓이가 개수에 비례해 눈이 읽는
      // 양과 실제 숫자가 맞는다 — 반지름을 그대로 비례시키면 큰 칸이 실제보다
      // 훨씬 많아 보인다.
      const maxRoot = Math.sqrt(current.max);
      for (const [lat, lng, n, weak] of current.rows) {
        const [x, y] = project(lat, lng);
        if (x < -10 || y < -10 || x > box.width + 10 || y > box.height + 10) continue;
        const t = maxRoot > 0 ? Math.sqrt(n) / maxRoot : 0;
        ctx.beginPath();
        ctx.arc(x, y, CELL_MIN_R + t * (CELL_MAX_R - CELL_MIN_R), 0, Math.PI * 2);
        ctx.fillStyle = CELL_COLOR;
        // 작은 칸은 옅게, 큰 칸은 진하게. 크기만으로는 작은 칸이 잘 안 보인다.
        ctx.globalAlpha = 0.45 + t * 0.45;
        ctx.fill();
        // 상한에 걸려 실제보다 적게 잡힌 칸은 테두리로 밝힌다.
        if (weak) {
          ctx.globalAlpha = 1;
          ctx.lineWidth = 1;
          ctx.strokeStyle = '#a02020';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }
    } else {
      ctx.fillStyle = DOT_COLOR;
      ctx.globalAlpha = 0.55;
      for (const [lat, lng] of current.rows) {
        const [x, y] = project(lat, lng);
        if (x < -10 || y < -10 || x > box.width + 10 || y > box.height + 10) continue;
        ctx.beginPath();
        ctx.arc(x, y, DOT_R, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  const onResize = () => redraw();
  window.addEventListener('resize', onResize);

  return {
    /** 격자 칸을 그린다. rows: [[위도, 경도, 개수, 덜걷힘], ...] */
    showGrid(rows) {
      const max = rows.reduce((m, r) => Math.max(m, r[2]), 1);
      current = { kind: 'grid', rows, max };
      redraw();
    },
    /** 개별 숙소 점을 그린다. rows: [[위도, 경도], ...] */
    showPoints(rows) {
      current = { kind: 'points', rows, max: 1 };
      redraw();
    },
    hide() {
      current = { kind: 'none', rows: [], max: 1 };
      clear();
    },
    redraw,
    destroy() {
      window.removeEventListener('resize', onResize);
    },
  };
}
