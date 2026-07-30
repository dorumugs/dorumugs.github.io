// 표 머리글을 눌러 정렬한다. 실거래 대시보드와 학군 지도의 표 네 개가 같은
// 규칙을 쓰도록 여기 한 곳에만 둔다.
//
// 데이터를 다시 만들지 않고 <tr> 노드를 그대로 옮겨 담는 방식이다. 행에 붙어
// 있는 data-code/data-idx 와 이벤트 위임(지도↔표 호버, 행 클릭)이 그대로
// 살아 있어야 하기 때문이다 — 행을 새로 그리면 그 연결이 끊긴다.

// 정렬에서 뺄 열은 <th class="no-sort">. 스파크라인처럼 값이 없는 열이 그렇다.
const SKIP = 'no-sort';

// '13,375' → 13375, '32.9%' → 32.9, '자료 없음'·'—' → null.
// 숫자로 못 읽는 값은 null 로 두고 방향과 무관하게 항상 뒤로 보낸다 — 자료가
// 없는 행이 오름차순에서 맨 위를 차지하면 표를 잘못 읽게 된다.
function numeric(text) {
  const cleaned = text.replace(/[,\s%]/g, '');
  if (!cleaned || !/^-?\d+(\.\d+)?$/.test(cleaned)) return null;
  return Number(cleaned);
}

function cellValue(row, index) {
  const cell = row.children[index];
  if (!cell) return { n: null, s: '' };
  // 화면에 안 보이는 값(묶음 표에서 두 번째 행부터 비워 둔 칸)은 data-sort 로
  // 넘겨받는다. 그게 없으면 보이는 글자를 쓴다.
  const raw = cell.dataset.sort != null ? cell.dataset.sort : cell.textContent;
  const text = String(raw).trim();
  return { n: numeric(text), s: text };
}

function compare(a, b, index, dir) {
  const va = cellValue(a, index);
  const vb = cellValue(b, index);
  const na = va.n;
  const nb = vb.n;
  if (na != null && nb != null) return (na - nb) * dir;
  // 숫자와 글자가 섞인 열은 없다고 보고, 한쪽만 숫자면 자료 없음 쪽을 뒤로.
  if (na != null) return -1;
  if (nb != null) return 1;
  if (!va.s) return 1;
  if (!vb.s) return -1;
  return va.s.localeCompare(vb.s, 'ko') * dir;
}

/**
 * @param {HTMLTableElement} table
 * @param {{rankColumn?: number}} [options] rankColumn 을 주면 정렬 뒤 그 열을
 *   1..n 으로 다시 매긴다. 실거래 랭킹 표의 순위 칸처럼 "지금 순서"를 뜻하는
 *   열이 옛 순위를 그대로 달고 있으면 표가 거짓말을 한다.
 */
export function makeSortable(table, { rankColumn = null } = {}) {
  if (!table) return;
  const head = table.querySelector('thead tr');
  const body = table.querySelector('tbody');
  if (!head || !body) return;
  // 안내 문구 한 줄짜리 '표시할 데이터가 없습니다' 같은 몸통은 정렬할 게 없다.
  if (body.querySelectorAll('tr').length < 2) return;

  const headers = [...head.children];
  headers.forEach((th, index) => {
    if (th.classList.contains(SKIP) || !th.textContent.trim()) return;
    th.classList.add('is-sortable');
    th.setAttribute('role', 'button');
    th.setAttribute('tabindex', '0');
    if (!th.hasAttribute('aria-sort')) th.setAttribute('aria-sort', 'none');

    const run = () => {
      const current = th.getAttribute('aria-sort');
      // 처음 누르면 큰 값부터 본다 — 이 표들은 대부분 "높은 쪽"이 관심사다.
      const dir = current === 'descending' ? 1 : -1;
      const rows = [...body.querySelectorAll('tr')];
      rows.sort((a, b) => compare(a, b, index, dir));
      rows.forEach((row) => body.appendChild(row));
      headers.forEach((other) => {
        if (other.hasAttribute('aria-sort')) other.setAttribute('aria-sort', 'none');
        other.classList.remove('is-asc', 'is-desc');
      });
      th.setAttribute('aria-sort', dir === 1 ? 'ascending' : 'descending');
      th.classList.add(dir === 1 ? 'is-asc' : 'is-desc');
      if (rankColumn != null) {
        rows.forEach((row, i) => {
          const cell = row.children[rankColumn];
          if (cell) cell.textContent = String(i + 1);
        });
      }
      // 묶음 표(학교별 아파트)는 같은 학교의 둘째 행부터 칸을 비워 두는데,
      // 정렬하면 묶음이 흩어져 그 행이 어느 학교인지 알 수 없게 된다. 숨겨
      // 뒀던 값(data-sort)을 채우고 묶음 구분선도 지운다.
      rows.forEach((row) => {
        row.classList.remove('is-group');
        [...row.children].forEach((cell) => {
          if (cell.dataset.sort != null && !cell.textContent.trim()) {
            cell.textContent = cell.dataset.sort;
          }
        });
      });
    };

    th.addEventListener('click', run);
    th.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); run(); }
    });
  });
}
