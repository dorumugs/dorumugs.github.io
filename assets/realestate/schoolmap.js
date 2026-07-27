// 지도 SVG 위에 학교 점을 얹는다. 좌표는 빌드 때 map.svg 와 같은 투영으로
// 계산돼 있으므로 여기서는 그리기만 한다.

const NS = 'http://www.w3.org/2000/svg';

export function initSchoolLayer(root, { onSelect }) {
  const svg = root.querySelector('svg.re-map');
  const tip = root.querySelector('.re-tip');
  const layer = document.createElementNS(NS, 'g');
  layer.setAttribute('class', 're-school-layer');
  svg.appendChild(layer);

  let selected = null;

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
      for (const school of schools) {
        const dot = document.createElementNS(NS, 'circle');
        dot.setAttribute('cx', school.x);
        dot.setAttribute('cy', school.y);
        dot.setAttribute('r', '5');
        dot.setAttribute('class', `re-dot is-${school.lvl}`);
        dot.setAttribute('tabindex', '0');
        dot.setAttribute('role', 'button');
        dot.setAttribute('aria-label', `${school.name}, ${school.dong}`);
        dot.addEventListener('click', () => onSelect(school));
        dot.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect(school);
          }
        });
        dot.addEventListener('mouseenter', () => showTip(dot, school));
        dot.addEventListener('focus', () => showTip(dot, school));
        dot.addEventListener('mouseleave', () => { tip.hidden = true; });
        dot.addEventListener('blur', () => { tip.hidden = true; });
        layer.appendChild(dot);
      }
    },
    setSelected(name) {
      if (selected) selected.classList.remove('is-selected');
      selected = null;
      for (const dot of layer.children) {
        if (dot.getAttribute('aria-label').startsWith(`${name},`)) {
          selected = dot;
          break;
        }
      }
      if (selected) {
        selected.classList.add('is-selected');
        // 선택된 점을 맨 위로 올려 겹친 점에 가리지 않게 한다
        layer.appendChild(selected);
      }
    },
    clear() {
      layer.textContent = '';
      selected = null;
    },
  };
}
