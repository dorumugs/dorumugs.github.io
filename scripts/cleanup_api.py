"""정비사업 정보몽땅(cleanup.seoul.go.kr) HTML 파싱. 순수 함수만 둔다.

I/O 는 collect_projects.py 가 담당한다. 여기 있는 함수는 전부 부수효과가 없어
tests/fixtures 에 저장해 둔 실제 응답만으로 검증할 수 있다.

정보몽땅은 공식 OpenAPI 가 아니다. 서울 열린데이터광장의 OA-2253
"서울시 재개발 재건축 정비사업 현황" OpenAPI 가 종료돼 원본 시스템을 직접 읽는다.
화면 구조가 바뀌면 여기가 먼저 깨진다. 그래서 파싱을 이 파일에 몰아두고
실제 응답을 픽스처로 고정해 회귀를 잡는다.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from html.parser import HTMLParser

# 사업장 목록 조회의 사업구분 코드. 전부 켜서 받는다.
BSNS_SE_CODES = ["100", "101", "102", "103", "104", "105", "106", "107"]
BSNS_EFCT_CODES = ["1", "2", "3"]

# 추진경과 아코디언에 나오는 단계 이름. 사업 진행 순서대로 둔다.
# 목록 화면의 '진행단계' 와는 표기가 조금 다르다 (목록은 '조합설립인가',
# 추진경과는 같은 이름이지만 '이주'/'철거신고' 처럼 목록에 없는 항목도 있다).
PROGRESS_STAGES = [
    "기본계획수립",
    "안전진단",
    "정비구역지정",
    "조합설립추진위원회승인",
    "정비사업전문관리업자선정",
    "설계자선정",
    "조합설립인가",
    "사업시행인가",
    "시공자선정",
    "철거업자선정",
    "관리처분인가",
    "이주",
    "철거신고",
    "착공신고",
    "일반분양승인",
    "준공인가",
    "이전고시",
    "조합해산",
    "조합청산",
]

# 이벤트 스터디에서 t=0 으로 쓸 단계. 가격이 실제로 반응한다고 보는 세 관문이다.
# 각 단계에서 '인가' 또는 '인가고시' 이벤트의 가장 이른 날짜를 사건일로 쓴다.
MILESTONE_STAGES = ["조합설립인가", "사업시행인가", "관리처분인가"]

PROJECT_COLUMNS = [
    "seq",
    "sgg_nm",
    "bsns_se",
    "name",
    "jibun_addr",
    "umd_nm",
    "jibun",
    "stage",
    "suspended",
    "cafe_url",
    "wtnnc_sn",
]

EVENT_COLUMNS = [
    "cafe_url",
    "stage",
    "event_date",
    "event",
    "issuer",
    "consent_rate",
    "notice_no",
    "vendor",
    "detail",
]

_VOID_TAGS = {"br", "img", "input", "meta", "link", "hr", "area", "base", "col"}
_WS = re.compile(r"\s+")


class ParseError(Exception):
    """응답이 예상한 화면이 아닐 때. 화면 개편을 조용히 넘기지 않기 위해 던진다."""


class CafeMissing(ParseError):
    """조합 카페가 닫혀 있어 추진경과를 볼 수 없다.

    화면 개편이 아니라 영구적인 상태다. 재시도 대상에서 빼야 매 실행마다
    같은 곳에 호출을 낭비하지 않는다.
    """


# --------------------------------------------------------------------------
# 최소 DOM
# --------------------------------------------------------------------------


class _Node:
    """파싱에 필요한 만큼만 가진 노드. 태그 이름·class·자식·텍스트.

    자식 노드와 텍스트 조각을 한 리스트(nodes)에 **문서 순서대로** 섞어 담는다.
    텍스트를 따로 모아두면 <li><h3>날짜</h3>[말머리] 설명</li> 에서 날짜가
    설명 뒤로 밀려, '날짜 다음이 본문' 이라는 파싱 전제가 무너진다.
    """

    __slots__ = ("tag", "cls", "nodes")

    def __init__(self, tag: str = "", cls: str = "") -> None:
        self.tag = tag
        self.cls = cls
        self.nodes: list[_Node | str] = []

    @property
    def children(self) -> list[_Node]:
        return [n for n in self.nodes if isinstance(n, _Node)]

    def text(self) -> str:
        """이 노드 아래 모든 텍스트를 문서 순서대로, 공백 하나로 정규화해 잇는다."""
        out: list[str] = []
        for node in self.nodes:
            out.append(node if isinstance(node, str) else node.text())
        return _WS.sub(" ", " ".join(out)).strip()

    def find_all(self, tag: str, cls: str | None = None) -> list[_Node]:
        found: list[_Node] = []
        for child in self.children:
            if child.tag == tag and (cls is None or cls in child.cls.split()):
                found.append(child)
            found.extend(child.find_all(tag, cls))
        return found


class _TreeBuilder(HTMLParser):
    """HTML 을 _Node 트리로 만든다. 닫히지 않은 태그는 부모가 닫힐 때 같이 닫는다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root")
        self._stack = [self.root]
        # 목록 표의 이동 버튼은 href 안의 자바스크립트 호출로만 식별된다.
        # 트리에는 속성을 안 담으므로 여기서 따로 모아 tr 단위로 나눠 갖는다.
        self.popups: list[tuple[int, str, str]] = []  # (열린 tr 수, 종류, 값)
        self._tr_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        adict = {k: (v or "") for k, v in attrs}
        if tag == "tr":
            self._tr_count += 1
        if tag == "a":
            href = adict.get("href", "") + " " + adict.get("onclick", "")
            for kind, pattern in (
                ("cafe_url", r"cafeOpenPopup\('([^']+)'\)"),
                ("wtnnc_sn", r"mapOpenPopup\('([^']+)'\)"),
            ):
                m = re.search(pattern, href)
                if m:
                    self.popups.append((self._tr_count, kind, m.group(1)))
        if tag in _VOID_TAGS:
            return
        node = _Node(tag, adict.get("class", ""))
        self._stack[-1].nodes.append(node)
        self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return
        # 짝 없는 닫는 태그는 무시한다. 정부 사이트 HTML 에는 흔하다.

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._stack[-1].nodes.append(data)


def _tree(html: str) -> _TreeBuilder:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder


# --------------------------------------------------------------------------
# 사업장 목록
# --------------------------------------------------------------------------


def split_jibun(addr: str) -> tuple[str, str]:
    """대표지번 '아현동 613-10' 을 (법정동, 지번) 으로 나눈다.

    실거래 CSV 의 umd_nm / jibun 컬럼과 그대로 맞물리게 하는 게 목적이다.
    '화곡동 956-37' 처럼 동 이름에 공백이 없는 게 보통이지만
    '북아현동 1-1 일대' 같이 꼬리가 붙는 경우가 있어 앞 두 토큰만 쓴다.
    """
    parts = (addr or "").split()
    if len(parts) < 2:
        return (parts[0] if parts else ""), ""
    umd = parts[0]
    jibun = parts[1]
    # '봉천동 산 101' 처럼 산지번은 '산' 이 떨어져 나온다. 실거래 CSV 는 '산101'
    # 로 붙여 쓰므로 여기서 합쳐야 조인된다.
    if jibun == "산" and len(parts) >= 3:
        jibun = "산" + parts[2]
    # 지번이 (산)숫자(-숫자) 꼴이 아니면 지번으로 보지 않는다.
    # '상계동 자력6구역 8블럭 9롯트' 같은 표기는 실거래와 맞출 수 없다.
    if not re.fullmatch(r"산?\d+(-\d+)?", jibun):
        return umd, ""
    return umd, jibun


def parse_project_list(html: str) -> list[dict[str, str]]:
    """사업장 목록 HTML 을 레코드 목록으로 바꾼다.

    표의 열 순서는
    번호 · 자치구 · 사업구분 · 사업장명 · 대표지번 · 진행단계 ·
    공개자료수 · 공개적시성 · 자료충실도 · 이동
    이고, 마지막 열의 링크에서 cafe_url(조합 카페)과 wtnnc_sn(도시계획 도형 ID)을 캔다.

    Raises:
        ParseError: 표를 하나도 못 찾았을 때. 화면 개편을 알아채기 위한 것이다.
    """
    builder = _tree(html)
    popups_by_tr: dict[int, dict[str, str]] = {}
    for tr_index, kind, value in builder.popups:
        popups_by_tr.setdefault(tr_index, {})[kind] = value

    rows: list[dict[str, str]] = []
    tr_index = 0
    for tr in builder.root.find_all("tr"):
        tr_index += 1
        cells = [td.text() for td in tr.find_all("td")]
        if len(cells) < 6:
            continue  # 머리글 행이나 '자료 없음' 행
        popup = popups_by_tr.get(tr_index, {})
        umd, jibun = split_jibun(cells[4])
        # 마지막 '이동' 칸에 '일시중단' 이 뜨면 조합 카페가 닫혀 있다 (1,102곳 중 133곳).
        # 추진경과를 받으러 가도 404 페이지만 온다. 목록에서 미리 걸러 호출을 아낀다.
        rows.append(
            {
                "seq": cells[0],
                "sgg_nm": cells[1],
                "bsns_se": cells[2],
                "name": cells[3],
                "jibun_addr": cells[4],
                "umd_nm": umd,
                "jibun": jibun,
                "stage": cells[5],
                "suspended": "1" if "일시중단" in cells[-1] else "",
                "cafe_url": popup.get("cafe_url", ""),
                "wtnnc_sn": popup.get("wtnnc_sn", ""),
            }
        )

    if not rows and "<table" not in html:
        raise ParseError("사업장 목록 표를 찾지 못했습니다. 화면 구조가 바뀌었을 수 있습니다.")
    return rows


def parse_cafe_keys(html: str) -> dict[str, str]:
    """조합 카페 메인에서 추진경과 조회에 필요한 cafeId / bsnsPk 를 캔다.

    두 값은 목록에서 주는 cafeUrl 로부터 규칙적으로 유도되지 않는다
    (cafeUrl 'mapo33' → cafeId '440100003003n67', bsnsPk '11440-100003003').
    그래서 사업장마다 메인을 한 번 더 받아야 한다.

    Raises:
        CafeMissing: 카페가 닫혀 '페이지를 찾을 수 없습니다' 안내가 온 경우.
        ParseError: 그 밖에 둘 중 하나라도 없을 때.
    """
    if "요청하신 페이지를 찾을 수 없습니다" in html:
        raise CafeMissing("조합 카페가 닫혀 있습니다")

    keys: dict[str, str] = {}
    for name in ("cafeId", "bsnsPk"):
        m = re.search(
            r'name="%s"[^>]*value="([^"]*)"' % name, html
        ) or re.search(r'value="([^"]*)"[^>]*name="%s"' % name, html)
        if m and m.group(1):
            keys["cafe_id" if name == "cafeId" else "bsns_pk"] = m.group(1)
    if len(keys) != 2:
        raise ParseError(f"cafeId/bsnsPk 를 찾지 못했습니다: {sorted(keys)}")
    return keys


# --------------------------------------------------------------------------
# 추진경과
# --------------------------------------------------------------------------


_DATE_RE = re.compile(r"(\d{4})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})")
_EVENT_RE = re.compile(r"\[([^\]:]+?)\s*(?::\s*([^\]]+?))?\]")
_CONSENT_RE = re.compile(r"동의율\s*:\s*([\d.]+)")
_NOTICE_RE = re.compile(r"고시번호\s*:\s*([^\n]+?)(?:\s*$|\s{2,})")
_VENDOR_RE = re.compile(r"선정업체명\s*:\s*([^\n]+?)(?:\s*$|\s{2,})")


def _clean_detail(text: str) -> str:
    text = _EVENT_RE.sub(" ", text, count=1)
    text = _CONSENT_RE.sub(" ", text)
    text = _NOTICE_RE.sub(" ", text)
    text = _VENDOR_RE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def parse_progress(html: str) -> list[dict[str, str]]:
    """추진경과 화면을 단계별 이벤트 목록으로 바꾼다.

    화면은 단계 하나가 아코디언 li.foldings-li 하나이고, 그 안의
    ul.check-list01 > li 가 이벤트다. 이벤트마다 h3.tit 에 날짜가,
    그 뒤 본문에 '[구역지정(변경)고시 : 서울시]' 같은 말머리와 설명이 온다.

    날짜가 없는 단계(아직 도달하지 않은 단계)는 결과에 넣지 않는다.
    """
    builder = _tree(html)
    events: list[dict[str, str]] = []

    for block in builder.root.find_all("li", "foldings-li"):
        # 단계 이름은 아코디언 머리의 첫 span 이다.
        heads = block.find_all("div", "foldings-in-wrap")
        if not heads:
            continue
        spans = heads[0].find_all("span")
        stage = spans[0].text() if spans else heads[0].text()
        if not stage:
            continue

        for lists in block.find_all("ul", "check-list01"):
            for item in lists.children:
                if item.tag != "li":
                    continue

                raw = item.text()
                m = _DATE_RE.search(raw)
                if not m:
                    continue
                year, month, day = m.groups()
                body = raw[m.end():]

                ev = _EVENT_RE.search(body)
                consent = _CONSENT_RE.search(body)
                notice = _NOTICE_RE.search(body)
                vendor = _VENDOR_RE.search(body)

                events.append(
                    {
                        "stage": stage,
                        "event_date": f"{year}-{int(month):02d}-{int(day):02d}",
                        "event": ev.group(1).strip() if ev else "",
                        "issuer": (ev.group(2) or "").strip() if ev else "",
                        "consent_rate": consent.group(1) if consent else "",
                        "notice_no": notice.group(1).strip() if notice else "",
                        "vendor": vendor.group(1).strip() if vendor else "",
                        "detail": _clean_detail(body),
                    }
                )

    return events


def milestone_dates(events: list[dict[str, str]]) -> dict[str, str]:
    """이벤트 목록에서 관문 단계별 사건일을 뽑는다.

    한 단계 안에 (변경)인가가 여러 번 있으면 **가장 이른 날짜**를 쓴다.
    최초 인가가 가격이 반응하는 시점이고, 뒤따르는 변경인가는 이미 반영된 뒤의
    행정 절차이기 때문이다. 신청일은 쓰지 않는다 — 인가 여부가 확정된 날이 사건이다.
    """
    out: dict[str, str] = {}
    for stage in MILESTONE_STAGES:
        dates = [
            e["event_date"]
            for e in events
            if e["stage"] == stage and "인가" in e["event"] and "신청" not in e["event"]
        ]
        if dates:
            out[stage] = min(dates)
    return out


# --------------------------------------------------------------------------
# 구역명 정규화 (UPIS 조인용)
# --------------------------------------------------------------------------

_NAME_NOISE = re.compile(
    r"(주택정비형|도시정비형|정비사업전문관리|도시환경정비사업|주택재개발정비사업|"
    r"주택재건축정비사업|가로주택정비사업|소규모재건축사업|소규모재개발사업|"
    r"정비사업|재개발사업|재건축사업|조합설립추진위원회|추진위원회|조합|구역|지구|"
    r"일대|사업|\(뉴타운\)|\s)"
)


def normalize_zone_name(name: str) -> str:
    """정보몽땅 사업장명과 UPIS 구역명(DGM_NM)을 맞추기 위한 정규화.

    '봉천14구역 주택재개발정비사업조합' 과 '봉천14' 가 같은 것으로 보이게 만든다.
    완벽한 매칭은 불가능하므로 조인 실패는 정상으로 취급하고 건수를 로그로 남긴다.
    """
    return _NAME_NOISE.sub("", name or "")


# --------------------------------------------------------------------------
# 직렬화
# --------------------------------------------------------------------------


def _rows_to_csv(rows: list[dict[str, str]], columns: list[str]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def projects_to_csv(rows: list[dict[str, str]]) -> str:
    return _rows_to_csv(sorted(rows, key=lambda r: r.get("cafe_url", "")), PROJECT_COLUMNS)


def events_to_csv(rows: list[dict[str, str]]) -> str:
    key = lambda r: (r.get("cafe_url", ""), r.get("event_date", ""), r.get("stage", ""))  # noqa: E731
    return _rows_to_csv(sorted(rows, key=key), EVENT_COLUMNS)


def csv_to_rows(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text, newline="")))


def gzip_bytes(text: str) -> bytes:
    """재현 가능한 gzip 바이트. mtime 을 0 으로 고정해 내용이 같으면 바이트도 같게 한다.

    매일 돌리는 수집이라 이걸 안 하면 같은 내용에도 git blob 이 계속 쌓인다.
    (rtms.gzip_bytes 와 같은 이유·같은 구현)
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0, filename="") as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def gunzip_text(data: bytes) -> str:
    return gzip.decompress(data).decode("utf-8")
