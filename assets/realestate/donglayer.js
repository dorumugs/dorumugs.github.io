// 시군구 하나로 확대했을 때 그 안의 행정동 경계와 이름을 지도에 얹는다.
//
// 왜 시군구별 파일인가
//   전국 행정동은 3,558개다. 한 장의 SVG 에 다 넣으면 시군구 지도(229KB)의
//   몇 배가 되어 첫 화면이 무거워진다. 확대해야 보이는 것이라, 그 시군구를
//   고른 뒤에 그 파일 하나만 받는다(중위 9.5KB · 최대 49.5KB).
//
// 좌표계
//   경계는 build_geo.py --dong 이 **map_kr.svg 와 같은 좌표계**로 투영해 둔
//   SVG path 문자열이다. 그래서 같은 <svg> 안에 그대로 넣으면 맞는다.
//
// 글자·선 굵기
//   SVG 안의 길이는 사용자 단위라 확대하면 같이 커진다. 20배로 들어가는
//   화면이므로 그대로 두면 이름이 화면을 덮는다. rescale() 이 지금 축척을
//   보고 화면 기준 크기로 되돌린다.

const NS = 'http://www.w3.org/2000/svg';

// 이름을 띄울 최소 크기(화면 px). 글자가 들어갈 자리조차 없는 조각을 거른다.
// 44 로 뒀더니 섬이 흩어진 옹진군은 이름이 0개, 여수시는 27개 중 1개만 떠서
// 지도에서 방향을 잡을 수 없었다(실측). 겹침은 아래에서 따로 거르므로 여기는
// 느슨해도 된다.
const LABEL_MIN_PX = 26;
const LABEL_PX = 11;

export function initDongLayer(root, { onSelect = () => {} } = {}) {
  const wrap = root.querySelector('.re-map-wrap');
  const svg = wrap.querySelector('svg.re-map');

  // 경계선은 지도 SVG 안에 넣는다 — 시군구 색칠 위, 캔버스 아래.
  const layer = document.createElementNS(NS, 'g');
  layer.setAttribute('class', 're-dong-layer');
  svg.appendChild(layer);

  // 이름은 **캔버스 위**에 따로 얹는다. 숙소 점을 그리는 캔버스가 지도 SVG
  // 위에 있어서, 이름을 같은 SVG 에 두면 점이 글자를 덮는다 — 실측으로
  // 마포구에서 가장 중요한 서교동(1,028곳) 이름이 점에 묻혀 안 보였다.
  // 같은 viewBox 를 복사해 쓰므로 좌표는 지도와 정확히 같다.
  const labels = document.createElementNS(NS, 'svg');
  labels.setAttribute('class', 're-dong-labels');
  labels.setAttribute('aria-hidden', 'true');
  wrap.appendChild(labels);

  let items = [];   // { code, name, path, label, box }

  function clear() {
    while (layer.firstChild) layer.removeChild(layer.firstChild);
    while (labels.firstChild) labels.removeChild(labels.firstChild);
    items = [];
    root.classList.remove('has-dong');
  }

  /** 이름을 놓을 자리와 크기를 정한다 — **가장 큰 조각** 기준.

      동 전체의 경계 상자를 쓰면 섬을 가진 동에서 이름이 바다 한가운데로
      간다. 울릉읍은 울릉도와 독도를 함께 가져서, 상자 중심이 두 섬 사이
      바다가 되고 이름이 화면 밖(실측 x=1954, 지도 오른쪽 끝 964)으로 잘려
      아예 안 보였다. 신안군·옹진군·여수시처럼 섬이 많은 곳도 같다.

      크기 판정도 이 조각으로 한다 — 본섬이 작은데 멀리 떨어진 섬 때문에
      상자만 커져서 이름이 뜨는 일을 막는다. */
  function mainPart(d, path) {
    const subs = (d || '').split('M').filter((s) => s.trim()).map((s) => `M${s}`);
    if (subs.length <= 1) return path.getBBox();
    const probe = document.createElementNS(NS, 'path');
    layer.appendChild(probe);
    let best = null;
    for (const sub of subs) {
      probe.setAttribute('d', sub);
      const box = probe.getBBox();
      if (!best || box.width * box.height > best.width * best.height) best = box;
    }
    layer.removeChild(probe);
    return best || path.getBBox();
  }

  /** 이름 SVG 를 지도 SVG 와 같은 자리·같은 viewBox 로 맞춘다. */
  function syncLabels() {
    const s = svg.getBoundingClientRect();
    const w = wrap.getBoundingClientRect();
    labels.style.left = `${s.left - w.left}px`;
    labels.style.top = `${s.top - w.top}px`;
    labels.style.width = `${s.width}px`;
    labels.style.height = `${s.height}px`;
    const box = svg.getAttribute('viewBox');
    if (box) labels.setAttribute('viewBox', box);
  }

  function overlaps(a, b) {
    return !(a.x + a.width <= b.x || b.x + b.width <= a.x
      || a.y + a.height <= b.y || b.y + b.height <= a.y);
  }

  /** 지금 축척에서 선 굵기·글자 크기를 화면 기준으로 되돌리고, 겹치는 이름을 접는다. */
  function rescale() {
    if (!items.length) return;
    const ctm = svg.getScreenCTM();
    const scale = ctm ? Math.abs(ctm.a) : 0;
    if (!(scale > 0)) return;
    const unit = 1 / scale;                 // 화면 1px = 사용자 단위 몇인가
    syncLabels();
    labels.setAttribute('font-size', `${LABEL_PX * unit}`);

    // **숙소가 많은 동부터** 자리를 잡고, 이미 놓인 이름과 겹치는 것은 접는다.
    // 넓이 순으로 했더니 이 화면에서 정작 궁금한 동(작지만 숙소가 몰린 곳)의
    // 이름이 넓은 이웃에게 밀렸다. 겹침을 안 거르면 좁은 동이 몰린 곳에서
    // 이름이 포개져 '리동공덕동' 처럼 읽힌다.
    // 지금 보이는 범위. 화면 밖 동은 이름을 접는다 — 안 그러면 멀리 떨어진
    // 섬의 이름이 자리(겹침 판정)를 차지해 정작 화면 안 이름을 밀어낸다.
    const view = (svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
    const inView = view.length === 4 && view.every((n) => Number.isFinite(n))
      ? (b) => !(b.x + b.width < view[0] || b.x > view[0] + view[2]
        || b.y + b.height < view[1] || b.y > view[1] + view[3])
      : () => true;

    const placed = [];
    const order = [...items].sort((a, b) => (b.count - a.count)
      || (b.box.width * b.box.height - a.box.width * a.box.height));
    for (const item of order) {
      const wide = Math.max(item.box.width, item.box.height) * scale;
      if (wide < LABEL_MIN_PX || !inView(item.box)) {
        item.label.style.display = 'none';
        continue;
      }
      // getBBox 는 숨긴 요소에서 0 을 돌려주므로 먼저 보이게 한 뒤 잰다.
      item.label.style.display = '';
      const rect = item.label.getBBox();
      if (placed.some((r) => overlaps(rect, r))) {
        item.label.style.display = 'none';
        continue;
      }
      placed.push(rect);
    }
  }

  return {
    /** 동 목록을 그린다. rows: [{code, name, count, d}] */
    show(rows) {
      clear();
      for (const row of rows) {
        const path = document.createElementNS(NS, 'path');
        path.setAttribute('class', 're-dong');
        path.setAttribute('d', row.d);
        path.setAttribute('data-dong', row.code);
        path.setAttribute('data-name', row.name);
        path.setAttribute('tabindex', '0');
        path.setAttribute('role', 'button');
        path.setAttribute('aria-label', `${row.name} 숙소 ${row.count ?? 0}곳`);
        path.addEventListener('click', () => onSelect(row.code));
        path.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(row.code); }
        });
        layer.appendChild(path);

        const box = mainPart(row.d, path);
        const label = document.createElementNS(NS, 'text');
        // 이름은 지도 SVG 가 아니라 캔버스 위 오버레이에 들어간다.
        label.setAttribute('class', 're-dong-name');
        label.setAttribute('x', `${box.x + box.width / 2}`);
        label.setAttribute('y', `${box.y + box.height / 2}`);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('dominant-baseline', 'middle');
        label.textContent = row.name;
        labels.appendChild(label);

        items.push({ code: row.code, name: row.name, count: row.count || 0,
                     path, label, box });
      }
      root.classList.add('has-dong');
      rescale();
    },

    clear,
    rescale,

    /** 고른 동을 도드라지게 한다. null 이면 모두 해제. */
    setSelected(code) {
      for (const item of items) {
        item.path.classList.toggle('is-selected', item.code === code);
      }
    },

    /** 그 동의 경계 상자. 지도 확대(map.focusBox)에 쓴다. */
    boxOf(code) {
      const hit = items.find((i) => i.code === code);
      return hit ? hit.box : null;
    },

    nameOf(code) {
      const hit = items.find((i) => i.code === code);
      return hit ? hit.name : code;
    },

    get length() {
      return items.length;
    },
  };
}
