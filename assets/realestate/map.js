import { NO_DATA } from './palette.js';

// 서울/경기/전체 뷰는 같은 SVG 의 viewBox 를 바꿔 만든다. 지도는 한 장뿐이다.
const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };
const PAD = 10;

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

export function initMap(root, { onSelect, onHover = () => {}, interactive = true }) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  // .re-tip 의 실제 위치 기준(포함 블록)은 position:relative 인 .re-map-wrap 이다.
  // (root=.re-app 은 position 이 없어 기준이 아니다.) 클램프도 같은 상자를 써야 맞는다.
  const wrap = root.querySelector('.re-map-wrap');
  const paths = Array.from(svg.querySelectorAll('path[data-sgg]'));
  const byCode = new Map(paths.map((p) => [p.dataset.sgg, p]));
  let labels = new Map();
  let selected = null;
  let hovered = null;

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
    setView(view) {
      const prefix = VIEW_PREFIX[view] ?? '';
      const shown = paths.filter((p) => p.dataset.sgg.startsWith(prefix));
      const visible = new Set(shown);
      for (const p of paths) {
        p.style.display = visible.has(p) ? '' : 'none';
      }
      const b = boundsOf(shown);
      svg.setAttribute('viewBox',
        `${b.minX - PAD} ${b.minY - PAD} ${b.maxX - b.minX + 2 * PAD} ${b.maxY - b.minY + 2 * PAD}`);
    },
    setSelected(code) {
      if (selected) selected.classList.remove('is-selected');
      selected = code ? byCode.get(code) : null;
      if (selected) selected.classList.add('is-selected');
    },
    nameOf(code) {
      return byCode.get(code)?.dataset.name || code;
    },
    codesIn(view) {
      const prefix = VIEW_PREFIX[view] ?? '';
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
