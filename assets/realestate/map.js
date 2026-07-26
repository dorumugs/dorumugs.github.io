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

export function initMap(root, { onSelect }) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  const paths = Array.from(svg.querySelectorAll('path[data-sgg]'));
  const byCode = new Map(paths.map((p) => [p.dataset.sgg, p]));
  let labels = new Map();
  let selected = null;

  function showTip(path, evt) {
    const code = path.dataset.sgg;
    const text = labels.get(code) || path.dataset.name;
    tip.textContent = text;
    tip.hidden = false;
    const box = root.getBoundingClientRect();
    tip.style.left = `${evt.clientX - box.left}px`;
    tip.style.top = `${evt.clientY - box.top}px`;
  }

  for (const path of paths) {
    path.setAttribute('tabindex', '0');
    path.setAttribute('role', 'button');
    path.addEventListener('click', () => onSelect(path.dataset.sgg));
    path.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onSelect(path.dataset.sgg);
      }
    });
    path.addEventListener('mousemove', (e) => showTip(path, e));
    path.addEventListener('mouseleave', () => { tip.hidden = true; });
    path.addEventListener('focus', () => {
      const box = path.getBoundingClientRect();
      showTip(path, { clientX: box.left + box.width / 2, clientY: box.top });
    });
    path.addEventListener('blur', () => { tip.hidden = true; });
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
    codesIn(view) {
      const prefix = VIEW_PREFIX[view] ?? '';
      return paths.map((p) => p.dataset.sgg).filter((c) => c.startsWith(prefix));
    },
  };
}
