// 지도 SVG 위에 학교 점을 얹는다. 좌표는 빌드 때 map.svg 와 같은 투영으로
// 계산돼 있으므로 여기서는 그리기만 한다.

const NS = 'http://www.w3.org/2000/svg';

// 화면 픽셀 기준 목표 탭/클릭 반경. 데스크톱에서 기존에 보이던 점 크기(지름
// 약 21px)에 맞춰 잡았다.
const HIT_RADIUS_PX = 12;

export function initSchoolLayer(root, { onSelect, onHover = () => {} }) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  const layer = document.createElementNS(NS, 'g');
  layer.setAttribute('class', 're-school-layer');
  svg.appendChild(layer);

  let selected = null;
  let hovered = null;
  // 지금 화면에 그려진 점들. {school, dot} — 클릭/마우스이동을 svg 전체에서
  // 한 번만 받아 가장 가까운 점을 찾는 데 쓴다(아래 nearestDot 참고).
  let current = [];

  // 점끼리 SVG 사용자 단위로 몇 안 되게(때로는 fan-out 최소 간격만큼만) 떨어져
  // 있어도, 화면 픽셀 기준 히트 반경(아래 hitRadiusUserUnits)은 뷰가 축소될수록
  // 그보다 훨씬 커진다 — 전체 뷰에서는 사용자 단위 34 까지 벌어진다. 점마다
  // 각자 크게 키운 히트 원을 그리면(처음 구현) 이웃한 점들의 히트 원이 서로
  // 겹쳐 DOM 순서상 나중에 그려진 점이 항상 클릭을 가로채 버린다 — 390px 서울
  // 뷰에서 38개 중 9개가 자신이 아닌 다른(심지어 안 겹쳐 보이는) 학교를 선택하는
  // 회귀로 실측했다. 그래서 점 하나하나에 히트 영역을 그리는 대신, svg 전체에서
  // 클릭/마우스이동을 한 번만 받아 "지금 커서에서 가장 가까운 점"을 계산해
  // 고른다 — 겹치는 히트 영역이 없으니 그리는 순서와 무관하게 항상 가장 가까운
  // 점이 이긴다.
  function hitRadiusUserUnits() {
    const ctm = svg.getScreenCTM();
    const scale = ctm ? Math.abs(ctm.a) : 0;
    return scale > 0 ? HIT_RADIUS_PX / scale : HIT_RADIUS_PX;
  }

  function nearestDot(evt) {
    if (!current.length) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const svgPt = pt.matrixTransform(ctm.inverse());
    const maxDist = hitRadiusUserUnits();
    let best = null;
    let bestDist = Infinity;
    for (const item of current) {
      const dx = svgPt.x - item.school.x;
      const dy = svgPt.y - item.school.y;
      const dist = Math.hypot(dx, dy);
      if (dist <= maxDist && dist < bestDist) {
        bestDist = dist;
        best = item;
      }
    }
    return best;
  }

  // 점을 "가리킨(hover/focus)" 상태로 표시한다. 마우스 이동·이탈과 점의
  // 키보드 포커스·블러가 전부 이 함수를 거친다 — 툴팁 표시와 표 쪽으로 보내는
  // onHover 콜백을 한 곳에서만 관리하기 위해서다. 이름은 setHovered 지만
  // 외부에 노출하는 이름이 같은 API(아래 반환 객체의 setHovered(name))와는
  // 다르다 — 저건 학교 이름으로 이 함수를 대신 불러주는 얇은 래퍼다.
  function applyHover(item) {
    if (hovered === item) return;
    if (hovered) hovered.dot.classList.remove('is-hover');
    hovered = item;
    if (hovered) {
      hovered.dot.classList.add('is-hover');
      showTip(hovered.dot, hovered.school);
    } else {
      tip.hidden = true;
    }
    onHover(hovered ? hovered.school : null);
  }

  svg.addEventListener('click', (e) => {
    const hit = nearestDot(e);
    if (hit) onSelect(hit.school);
  });
  svg.addEventListener('mousemove', (e) => applyHover(nearestDot(e)));
  svg.addEventListener('mouseleave', () => applyHover(null));

  function showTip(dot, school) {
    tip.textContent = `${school.name} · ${school.dong}`;
    tip.hidden = false;
    const box = root.querySelector('.re-map-wrap').getBoundingClientRect();
    const dotBox = dot.getBoundingClientRect();
    const tipBox = tip.getBoundingClientRect();
    const halfW = tipBox.width / 2;
    const minX = Math.min(halfW, box.width / 2);
    const maxX = Math.max(box.width - halfW, box.width / 2);
    let x = dotBox.left + dotBox.width / 2 - box.left;
    x = Math.min(Math.max(x, minX), maxX);
    let y = dotBox.top - box.top;
    const minY = tipBox.height * 1.4;
    if (y < minY) y = minY;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }

  return {
    render(schools) {
      layer.textContent = '';
      selected = null;
      hovered = null;
      current = [];
      for (const school of schools) {
        const dot = document.createElementNS(NS, 'circle');
        dot.setAttribute('cx', school.x);
        dot.setAttribute('cy', school.y);
        dot.setAttribute('r', '5');
        dot.setAttribute('class', `re-dot is-${school.lvl}`);
        // 마우스/터치 클릭은 svg 의 nearestDot() 딜레이트가 처리한다(이 점
        // 자체의 지오메트리는 뷰에 따라 여전히 작다). 키보드 탭 순서·포커스만
        // 이 점이 직접 받는다.
        dot.setAttribute('tabindex', '0');
        dot.setAttribute('role', 'button');
        dot.setAttribute('aria-label', `${school.name}, ${school.dong}`);
        const item = { school, dot };
        dot.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect(school);
          }
        });
        // 키보드 포커스도 마우스 호버와 같은 경로(applyHover)를 타게 해
        // 툴팁·is-hover 표시·표 쪽 onHover 콜백이 마우스/키보드 양쪽에서
        // 똑같이 움직이게 한다.
        dot.addEventListener('focus', () => applyHover(item));
        dot.addEventListener('blur', () => { if (hovered === item) applyHover(null); });
        layer.appendChild(dot);
        current.push(item);
      }
    },
    setSelected(name) {
      if (selected) selected.classList.remove('is-selected');
      selected = null;
      if (!name) return; // 목록으로 돌아갈 때 등 — 선택만 지우면 된다
      for (const dot of layer.children) {
        if (dot.getAttribute('aria-label').startsWith(`${name},`)) {
          selected = dot;
          break;
        }
      }
      if (selected) {
        selected.classList.add('is-selected');
        // 선택된 점을 맨 위로 올려 겹친 점에 가리지 않게 한다. appendChild 로
        // 노드를 재배치하면 Chrome 이 포커스를 지운다(activeElement -> BODY) —
        // 키보드로 선택했을 때 포커스를 잃지 않도록 옮기기 전 포커스 여부를
        // 기억해 뒀다가 옮긴 뒤 되돌린다.
        const hadFocus = document.activeElement === selected;
        layer.appendChild(selected);
        if (hadFocus) selected.focus({ preventScroll: true });
      }
    },
    // 학교 이름으로 호버 상태를 프로그램적으로 건다 — 표의 행을 마우스로
    // 가리키거나 키보드로 포커스했을 때 schools-app.js 가 호출한다. 지금
    // 뷰에 없는(다른 지역 탭의) 학교 이름이면 아무 점도 못 찾으니 안전하게
    // 아무 일도 하지 않는다(= 켜져 있던 호버를 지우는 것으로 끝).
    setHovered(name) {
      if (!name) { applyHover(null); return; }
      const item = current.find((it) => it.school.name === name);
      applyHover(item || null);
    },
  };
}
