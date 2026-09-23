import { NO_DATA } from './palette.js';

// 서울/경기/전체 뷰는 같은 SVG 의 viewBox 를 바꿔 만든다. 지도는 한 장뿐이다.
const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };
const PAD = 10;

// 뷰 이름을 시군구 코드 접두사로 바꾼다. 위 세 이름 말고도 시도 코드를 그대로
// 받는다 — 전국 지도(map_kr.svg)는 시도가 16개라 이름을 일일이 적을 수 없다.
// 'all' 과 빈 값은 전부를 뜻한다.
function prefixOf(view) {
  if (view in VIEW_PREFIX) return VIEW_PREFIX[view];
  return /^\d+$/.test(view) ? view : '';
}

function boundsOf(paths) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of paths) {
    const box = p.getBBox();
    minX = Math.min(minX, box.x);
    minY = Math.min(minY, box.y);
    maxX = Math.max(maxX, box.x + box.width);
    maxY = Math.max(maxY, box.y + box.height);
  }
  return { minX, minY, maxX, maxY };
}

export function initMap(root, {
  onSelect, onHover = () => {}, interactive = true, onView = () => {},
} = {}) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  // .re-tip 의 실제 위치 기준(포함 블록)은 position:relative 인 .re-map-wrap 이다.
  // (root=.re-app 은 position 이 없어 기준이 아니다.) 클램프도 같은 상자를 써야 맞는다.
  const wrap = root.querySelector('.re-map-wrap');
  const paths = Array.from(svg.querySelectorAll('path[data-sgg]'));
  const byCode = new Map(paths.map((p) => [p.dataset.sgg, p]));
  let labels = new Map();
  let hovered = null;
  // setView 가 정한 화면 비율. focus() 가 한 구로 확대할 때 이 비율을 지킨다 —
  // 구마다 제 모양대로 viewBox 를 잡으면 svg 의 height:auto 가 따라 바뀌어
  // 지도 상자가 세로로 출렁이고 옆 칸까지 밀린다.
  let baseAspect = null;
  let anim = 0;

  // settled=false 는 "아직 움직이는 중" 이다. 호출부가 프레임마다 비싼 일을
  // 하지 않게 하려고 가른다 — 이 대시보드는 격자 1만여 칸을 축척에 맞춰 다시
  // 묶는데, 그걸 매 프레임 하면 확대가 끊긴다.
  function setBox(x, y, w, h, settled = true) {
    svg.setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
    onView(settled);
  }

  // viewBox 를 부드럽게 옮긴다. 전국에서 구 하나로 순간이동하면 어디로 갔는지
  // 알 수 없다 — 짧게 이어 줘야 눈이 따라간다.
  function animateTo(target, ms = 320) {
    const from = (svg.getAttribute('viewBox') || '0 0 1000 1000').split(/\s+/).map(Number);
    if (anim) cancelAnimationFrame(anim);
    if (ms <= 0 || from.some((n) => !Number.isFinite(n))) {
      setBox(...target);
      return;
    }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / ms);
      // ease-out cubic — 시작은 빠르고 끝은 부드럽게.
      const k = 1 - (1 - t) ** 3;
      setBox(...from.map((v, i) => v + (target[i] - v) * k), t >= 1);
      anim = t < 1 ? requestAnimationFrame(step) : 0;
    };
    anim = requestAnimationFrame(step);
  }

  function showTip(path, evt) {
    const code = path.dataset.sgg;
    const text = labels.get(code) || path.dataset.name;
    tip.textContent = text;
    tip.hidden = false;
    const box = wrap.getBoundingClientRect();
    const tipBox = tip.getBoundingClientRect();
    // transform: translate(-50%, -140%) 로 중앙정렬 + 위로 뜨므로, 그 절반/1.4배만큼
    // 여유를 두고 컨테이너 상자 안쪽으로 클램프한다. 화면 끝 근처 구를 눌러도
    // 툴팁이 잘리거나 컨테이너 밖으로 나가지 않게 하기 위함.
    const halfW = tipBox.width / 2;
    const minX = Math.min(halfW, box.width / 2);
    const maxX = Math.max(box.width - halfW, box.width / 2);
    let x = evt.clientX - box.left;
    x = Math.min(Math.max(x, minX), maxX);
    let y = evt.clientY - box.top;
    const minY = tipBox.height * 1.4;
    if (y < minY) y = minY;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }

  // 점(가운데 위)이 없어 evt 없이 프로그램적으로 가리킬 때(표 행 호버 →
  // 지도, setHovered) 쓸 좌표를 구의 화면 상자 중앙에서 만든다. 실제
  // 마우스이동은 evt.clientX/Y 를 그대로 쓴다(showTip 참고).
  function centerEvt(path) {
    const box = path.getBoundingClientRect();
    return { clientX: box.left + box.width / 2, clientY: box.top };
  }

  // 구를 "가리킨(hover/focus)" 상태로 표시한다. 실제 마우스이동·이탈, 구의
  // 키보드 포커스·블러, 그리고 표 행 호버가 부르는 setHovered() 가 전부 이
  // 함수를 거친다 — is-hover 클래스·툴팁·표 쪽으로 보내는 onHover 콜백을 한
  // 곳에서만 관리하기 위해서다(schoolmap.js 의 applyHover 와 같은 얼개).
  function applyHover(path, evt) {
    if (hovered === path) {
      if (path) showTip(path, evt || centerEvt(path));
      return;
    }
    if (hovered) hovered.classList.remove('is-hover');
    hovered = path;
    if (hovered) {
      hovered.classList.add('is-hover');
      showTip(hovered, evt || centerEvt(hovered));
    } else {
      tip.hidden = true;
    }
    onHover(hovered ? hovered.dataset.sgg : null);
  }

  // 학군 페이지처럼 구가 누를 대상이 아닌 화면에서는 포커스·클릭을 걸지 않는다.
  // 걸면 아무 동작도 하지 않는 포커스 가능한 버튼이 72개 생긴다.
  if (interactive) {
    for (const path of paths) {
      path.setAttribute('tabindex', '0');
      path.setAttribute('role', 'button');
      path.setAttribute('aria-label', path.dataset.name);
      path.addEventListener('click', () => onSelect(path.dataset.sgg));
      path.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(path.dataset.sgg);
        }
      });
      path.addEventListener('mousemove', (e) => applyHover(path, e));
      path.addEventListener('mouseleave', () => { if (hovered === path) applyHover(null); });
      path.addEventListener('focus', () => applyHover(path));
      path.addEventListener('blur', () => { if (hovered === path) applyHover(null); });
    }
  }

  return {
    // values: Map<code, {color, label}>
    paint(values) {
      labels = new Map();
      for (const [code, path] of byCode) {
        const hit = values.get(code);
        path.setAttribute('fill', hit ? hit.color : NO_DATA);
        if (hit) labels.set(code, hit.label);
      }
    },
    setView(view, { animate = false } = {}) {
      const prefix = prefixOf(view);
      const shown = paths.filter((p) => p.dataset.sgg.startsWith(prefix));
      const visible = new Set(shown);
      for (const p of paths) {
        p.style.display = visible.has(p) ? '' : 'none';
      }
      const b = boundsOf(shown);
      const box = [b.minX - PAD, b.minY - PAD,
        b.maxX - b.minX + 2 * PAD, b.maxY - b.minY + 2 * PAD];
      baseAspect = box[3] / box[2];
      if (animate) animateTo(box); else setBox(...box);
    },

    // 시군구 하나로 확대한다. **이웃을 숨기지 않는다** — 구 하나만 남기면
    // 어디를 보고 있는지 알 수 없다. 다른 시도의 이웃도 함께 드러낸다.
    // minWidth 는 확대의 바닥이다(SVG 사용자 단위). 경계선은 build_geo.py 가
    // 단순화해 둔 것이라, 무한정 확대하면 그 오차가 화면에서 각진 다각형으로
    // 드러난다 — 작은 구일수록 심하다. 호출부가 자기 지도의 단순화 강도에
    // 맞춰 정한다. 0 이면 바닥 없음.
    focus(code, { pad = 0.45, animate = true, minWidth = 0 } = {}) {
      const path = byCode.get(code);
      if (!path) return;
      for (const p of paths) p.style.display = '';
      const b = path.getBBox();
      const margin = Math.max(b.width, b.height) * pad;
      let w = Math.max(b.width + margin * 2, minWidth);
      let h = b.height + margin * 2;
      const aspect = baseAspect || h / w;
      // 비율을 맞추되 구가 잘리지 않도록 **넓히는 쪽으로만** 맞춘다.
      if (h / w < aspect) h = w * aspect; else w = h / aspect;
      const cx = b.x + b.width / 2;
      const cy = b.y + b.height / 2;
      const box = [cx - w / 2, cy - h / 2, w, h];
      if (animate) animateTo(box); else setBox(...box);
    },
    // null·단일 코드·코드 배열을 모두 받는다. 재개발 화면은 구를 여러 개 고르고,
    // 실거래 대시보드는 하나만 고른다 — 호출부를 갈라놓지 않으려고 여기서 받아준다.
    setSelected(codes) {
      const next = new Set(
        codes === null || codes === undefined ? [] : Array.isArray(codes) ? codes : [codes],
      );
      for (const [code, path] of byCode) path.classList.toggle('is-selected', next.has(code));
    },
    nameOf(code) {
      return byCode.get(code)?.dataset.name || code;
    },
    codesIn(view) {
      const prefix = prefixOf(view);
      return paths.map((p) => p.dataset.sgg).filter((c) => c.startsWith(prefix));
    },
    // 시군구 코드로 호버 상태를 프로그램적으로 건다 — 랭킹 표의 행을 마우스로
    // 가리키거나 키보드로 포커스했을 때 app.js 가 호출한다. 지금 뷰에 없는
    // (다른 지역 탭의, display:none 인) 코드나 존재하지 않는 코드면 안전하게
    // 아무 일도 하지 않는다(= 켜져 있던 호버를 지우는 것으로 끝).
    setHovered(code) {
      const path = code ? byCode.get(code) : null;
      applyHover(path && path.style.display !== 'none' ? path : null);
    },
  };
}
