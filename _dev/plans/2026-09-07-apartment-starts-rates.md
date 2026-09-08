# 시도별 아파트 착공 × 금리 대시보드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2011-01 이후 시도별 아파트 착공 세대수와 기준금리·주택담보대출 금리를 x축을 공유하는 상하 2단 라인차트로 보여주는 대시보드를 `/dashboard/supply/` 에 만든다.

**Architecture:** 저장소의 기존 파이프라인 관례를 그대로 따른다 — `*_api.py`(순수 함수, I/O 없음) → `collect_*.py`(I/O·상태 재개) → `build_*.py`(집계) → `assets/realestate/supply.json` → 바닐라 ESM 페이지. 차트는 라이브러리 없이 `assets/realestate/charts.js` 의 기존 `multiLineChart()` 를 후방호환 옵션 3개로 확장해 재사용한다.

**Tech Stack:** Python 3 표준 라이브러리만 (`urllib`, `json`, `unittest`) · 바닐라 ES 모듈 · 손으로 만든 인라인 SVG · Jekyll(minimal-mistakes)

**Spec:** `_dev/specs/2026-09-07-apartment-starts-rates-design.md`

## 진행 상황 (2026-09-08)

| 태스크 | 상태 |
|---|---|
| 1 통계누리 파서 · 2 ECOS 파서 · 3 수집기 · 4 집계 · 5 차트 · 6 화면 · 7 크론·신선도·허브 | 구현 완료 |
| 8 모바일 검증·문서 | 완료 — 758개 테스트 통과, `/dashboard/supply/` · `/dashboard/` · 학군 세 페이지 390px `scrollWidth=390` 넘침 0 |

**전부 끝났다.** `ECOS_API_KEY` 를 받아 금리 187개월을 채웠고
(`data/supply/rates.json`), 크론도 `20 5 5,25 * *` 로 등록했다 — 기존 넷과
같은 `flock` 공유. 금리 패널의 "자료 없음" 은 사라졌다.

설계 문서의 "구현 결과" 절에 검증 수치가 정리돼 있다.

## Global Constraints

스펙과 `CLAUDE.md` 에서 그대로 가져온 값이다. **모든 태스크의 요구사항에 암묵적으로 포함된다.**

- **표준 라이브러리만 쓴다.** `pandas`·`requests` 계열 금지. `urllib`, `json`, `csv`, `gzip`, `xml.etree` 만.
- 테스트는 `pytest` 가 아니라 표준 `unittest`. 실행은 `python3 -m unittest discover -s tests`.
- 테스트 메서드 이름은 한국어를 쓴다 (`def test_집계행을_거른다`). 저장소 전체가 그렇다.
- 순수 함수 모듈(`*_api.py`)에는 네트워크 I/O 를 넣지 않는다. 인증키 로더(`load_key`)는 예외 — `scripts/vworld_api.py:60` 이 선례다.
- **키를 저장소에 두지 않는다.** `gh-pages` 는 저장소 파일을 그대로 웹에 서빙한다. `ECOS_API_KEY` 는 환경변수 또는 `.env`(gitignore 됨).
- 크론 스크립트는 `flock -w 7200 ~/.cache/realestate.lock` 을 공유한다. **반드시 유지한다** — 전부 `git commit`/`push` 를 하므로 겹쳐 돌면 커밋이 유실된다.
- git 신원은 `Jaehyun So <dorumugs@gmail.com>` 을 `GIT_AUTHOR_*`/`GIT_COMMITTER_*` 환경변수로 단발 지정한다. 전역 `git config` 변경 금지.
- **모든 페이지는 390px 에서 가로로 넘치면 안 된다.** 표는 자체 스크롤 컨테이너 안에서만 넘칠 것.
- 코드블록 안에 한글 산문을 넣지 않는다.
- 커밋 메시지는 영문 한 줄 요약 + 빈 줄 + 한국어 본문 1~2줄 + `Co-Authored-By:` 트레일러.
- 사용자가 명시 요청하기 전에는 **push 하지 않는다.** 태스크의 커밋은 로컬 커밋까지만이다.

### 데이터 상수 (스펙에서 확정)

- 착공: `https://stat.molit.go.kr/portal/stat/data.do?formId=5387&styleNum=1&apprYn=Y&startDate=YYYYMM&endDate=YYYYMM` — 인증 불필요, **1회 최대 60개월**
- 기준금리: ECOS `722Y001` / 주기 `M` / 항목 `0101000`
- 주택담보대출 금리: ECOS `121Y006` / 주기 `M` / 항목 `BECBLA0302`
- 시계열 시작: `2011-01`
- 평년 기준선: `2011-12` ~ `2019-12`, 유효 개월 60 미만이면 지수 `null`

## File Structure

| 파일 | 책임 | 태스크 |
|---|---|---|
| `scripts/molit_stat_api.py` | 통계누리 응답 파싱. 순수 함수 | 1 |
| `tests/test_molit_stat_api.py` | 위 테스트 | 1 |
| `tests/fixtures/molit_starts_2026.json` | 전남광주 통합 · 잠정치 `p)` | 1 |
| `tests/fixtures/molit_starts_2011.json` | 세종 `'-'` 결측 | 1 |
| `tests/fixtures/molit_range_error.json` | 60개월 초과 거절 응답 | 1 |
| `scripts/ecos_api.py` | ECOS 응답 파싱 + 키 로더 | 2 |
| `tests/test_ecos_api.py` | 위 테스트 | 2 |
| `tests/fixtures/ecos_base_rate.json` | 정상 응답 | 2 |
| `tests/fixtures/ecos_error.json` | 오류 응답 | 2 |
| `scripts/collect_supply.py` | 두 소스 호출, 원본 보존, 잠정월 재수집 | 3 |
| `scripts/build_supply.py` | 합산·이동합계·지수·총계 대조 | 4 |
| `tests/test_build_supply.py` | 위 테스트 | 4 |
| `assets/realestate/charts.js` | `multiLineChart` 에 옵션 3개 추가 | 5 |
| `assets/realestate/supply-app.js` | 컨트롤·렌더 | 6 |
| `assets/realestate/supply.css` | 스타일 | 6 |
| `_pages/dashboard-supply.md` | 페이지 | 6 |
| `scripts/supply_monthly.sh` | 크론 진입점 | 7 |
| `scripts/freshness_api.py` | `supply` 임계값 등록 | 7 |
| `scripts/check_freshness.py` | `supply` 대상 등록 | 7 |
| `assets/realestate/freshness.js` | `supply` 화면 임계값 | 7 |
| `_pages/dashboard.md` | 허브 카드 | 7 |

---

### Task 1: 통계누리 응답 파서 (`molit_stat_api.py`)

착공 데이터의 함정 다섯 중 넷(60개월 거절 · 잠정치 `p)` · 집계행 · `'-'` 결측)을 여기서 잡는다. 전남광주 합산은 Task 4 에서 한다 — 이 모듈은 원본에 충실해야 한다.

**Files:**
- Create: `scripts/molit_stat_api.py`
- Create: `tests/test_molit_stat_api.py`
- Create: `tests/fixtures/molit_starts_2026.json`
- Create: `tests/fixtures/molit_starts_2011.json`
- Create: `tests/fixtures/molit_range_error.json`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces:
  - `SIDO: tuple[str, ...]` — 18개 라벨 화이트리스트 (17개 시도 + 통합 라벨 `전남광주`)
  - `TOTAL_LABEL: str = "총계"`
  - `MAX_MONTHLY_NATIONWIDE: int = 200_000`
  - `parse_month(text) -> tuple[str | None, bool]` — `("2026-07", True)`
  - `parse_value(text) -> int | None` — `'-'`·빈값·비숫자·음수는 `None`
  - `parse_starts(payload: dict) -> tuple[list[dict], dict[str, int], str | None]`
    - 1번: 행 목록. 각 행은 `{"month": str, "region": str, "units": int | None, "provisional": bool}`
    - 2번: 월 → 전국 총계 아파트 착공 세대수 (Task 4 의 대조용)
    - 3번: 오류 사유. 정상이면 `None`

- [ ] **Step 1: 픽스처를 실제 응답에서 만든다**

네트워크에서 한 번 받아 필요한 부분만 잘라 고정한다. 테스트 자체는 네트워크를 쓰지 않는다.

```bash
python3 - <<'PY'
import json, urllib.request, pathlib

URL = ("https://stat.molit.go.kr/portal/stat/data.do"
       "?formId=5387&styleNum=1&apprYn=Y&startDate={s}&endDate={e}")
OUT = pathlib.Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(start, end):
    req = urllib.request.Request(URL.format(s=start, e=end),
                                 headers={"X-Requested-With": "XMLHttpRequest"})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)


def trim(payload, months, regions):
    rows = [r for r in payload["data"]
            if r.get("0", "").split()[0] in months and r.get("1") in regions]
    return {"result": True, "data": rows}


big = fetch("202501", "202607")
keep = {"총계", "수도권소계", "서울", "경기", "광주", "전남", "전남광주", "세종"}
(OUT / "molit_starts_2026.json").write_text(
    json.dumps(trim(big, {"2026-06", "2026-07"}, keep), ensure_ascii=False, indent=1),
    encoding="utf-8")

old = fetch("201101", "201512")
(OUT / "molit_starts_2011.json").write_text(
    json.dumps(trim(old, {"2011-01", "2012-09", "2012-10"},
                    {"총계", "서울", "세종"}), ensure_ascii=False, indent=1),
    encoding="utf-8")

err = fetch("201101", "202607")
(OUT / "molit_range_error.json").write_text(
    json.dumps(err, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote 3 fixtures")
PY
```

- [ ] **Step 2: 픽스처가 잡아야 할 것을 실제로 담았는지 눈으로 확인한다**

```bash
python3 - <<'PY'
import json, pathlib
f = pathlib.Path("tests/fixtures")
a = json.loads((f / "molit_starts_2026.json").read_text(encoding="utf-8"))
regions_0607 = {r["1"] for r in a["data"] if r["0"].startswith("2026-07")}
regions_0606 = {r["1"] for r in a["data"] if r["0"].startswith("2026-06")}
print("2026-07 지역:", sorted(regions_0607))
print("2026-06 지역:", sorted(regions_0606))
print("잠정 표기:", sorted({r["0"] for r in a["data"]}))
b = json.loads((f / "molit_starts_2011.json").read_text(encoding="utf-8"))
print("세종:", [(r["0"], r["5"]) for r in b["data"]
               if r["1"] == "세종" and r["2"] == "아파트"])
c = json.loads((f / "molit_range_error.json").read_text(encoding="utf-8"))
print("거절 응답:", c.get("result"), repr(c.get("msg"))[:60])
PY
```

기대 출력:
- `2026-07 지역` 에 `전남광주` 가 있고 `광주`·`전남` 이 **없다**
- `2026-06 지역` 에 `광주`·`전남` 이 있고 `전남광주` 가 **없다**
- 잠정 표기가 `2026-06 p)` 처럼 ` p)` 로 끝난다
- 세종의 2011-01 값이 `'-'` 이고 2012-10 값이 숫자다
- 거절 응답의 `result` 가 `False`

하나라도 어긋나면 Step 1 의 `keep`/`months` 를 고쳐 다시 만든다.

- [ ] **Step 3: 실패하는 테스트를 쓴다**

```python
"""국토교통 통계누리 주택유형별 착공실적(월계) 응답 파싱.

이 파서가 잡아야 할 함정이 넷이다. 전부 실제 응답에서 확인한 것들이라
픽스처에 그대로 박아 뒀다 — 통계누리는 공식 API 가 아니라 화면 뒤의
ajax 엔드포인트여서, 화면이 바뀌면 조용히 깨진다.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import molit_stat_api as m  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ParseMonthTest(unittest.TestCase):
    def test_잠정치_표기를_떼어낸다(self) -> None:
        self.assertEqual(m.parse_month("2026-07 p)"), ("2026-07", True))

    def test_확정월은_잠정이_아니다(self) -> None:
        self.assertEqual(m.parse_month("2025-09"), ("2025-09", False))

    def test_못_읽으면_None(self) -> None:
        for bad in ("", None, "2026", "2026-13", "작년"):
            self.assertIsNone(m.parse_month(bad)[0])


class ParseValueTest(unittest.TestCase):
    def test_결측_표기는_0이_아니라_None(self) -> None:
        """세종은 2012-10 이전이 '-' 다. 0 으로 읽으면 평년 기준선이 무너진다."""
        self.assertIsNone(m.parse_value("-"))
        self.assertIsNone(m.parse_value(""))
        self.assertIsNone(m.parse_value(None))

    def test_0은_실제_관측값이다(self) -> None:
        self.assertEqual(m.parse_value("0"), 0)

    def test_쉼표를_지운다(self) -> None:
        self.assertEqual(m.parse_value("21,223"), 21223)

    def test_음수는_None(self) -> None:
        self.assertIsNone(m.parse_value("-5"))


class ParseStartsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows, self.totals, self.err = m.parse_starts(load("molit_starts_2026.json"))

    def test_오류가_없다(self) -> None:
        self.assertIsNone(self.err)

    def test_집계행을_거른다(self) -> None:
        names = {r["region"] for r in self.rows}
        self.assertNotIn("총계", names)
        self.assertNotIn("수도권소계", names)

    def test_총계는_따로_돌려준다(self) -> None:
        self.assertGreater(self.totals["2026-06"], 0)

    def test_아파트만_남긴다(self) -> None:
        seoul = [r for r in self.rows
                 if r["region"] == "서울" and r["month"] == "2026-06"]
        self.assertEqual(len(seoul), 1)

    def test_전남광주_통합월에는_광주와_전남이_없다(self) -> None:
        """2026-07-01 전남광주통합특별시 출범. 라벨 자체가 바뀐다."""
        july = {r["region"] for r in self.rows if r["month"] == "2026-07"}
        self.assertIn("전남광주", july)
        self.assertNotIn("광주", july)
        self.assertNotIn("전남", july)

    def test_통합_이전_달에는_광주와_전남이_따로_있다(self) -> None:
        june = {r["region"] for r in self.rows if r["month"] == "2026-06"}
        self.assertIn("광주", june)
        self.assertIn("전남", june)
        self.assertNotIn("전남광주", june)

    def test_잠정_플래그가_붙는다(self) -> None:
        self.assertTrue(all(r["provisional"] for r in self.rows
                            if r["month"] == "2026-07"))


class SejongMissingTest(unittest.TestCase):
    def test_세종의_이른_달은_결측이다(self) -> None:
        rows, _totals, err = m.parse_starts(load("molit_starts_2011.json"))
        self.assertIsNone(err)
        early = [r for r in rows if r["region"] == "세종" and r["month"] == "2011-01"]
        self.assertEqual(len(early), 1)
        self.assertIsNone(early[0]["units"])

    def test_세종은_2012_10부터_숫자가_잡힌다(self) -> None:
        rows, _totals, _err = m.parse_starts(load("molit_starts_2011.json"))
        late = [r for r in rows if r["region"] == "세종" and r["month"] == "2012-10"]
        self.assertIsNotNone(late[0]["units"])


class RejectionTest(unittest.TestCase):
    def test_60개월_초과_거절을_성공으로_읽지_않는다(self) -> None:
        """거절이 HTTP 200 으로 온다. result 를 안 보면 조용히 빈 데이터가 된다."""
        rows, totals, err = m.parse_starts(load("molit_range_error.json"))
        self.assertEqual(rows, [])
        self.assertEqual(totals, {})
        self.assertIsNotNone(err)

    def test_이상한_구조는_오류다(self) -> None:
        for bad in (None, [], {}, {"result": True}, {"result": True, "data": []}):
            self.assertIsNotNone(m.parse_starts(bad)[2])


class SanityGateTest(unittest.TestCase):
    """한 달 전국 합계가 말이 안 되면 그 달을 통째로 버린다.

    부분적으로 틀린 시계열이 조용히 섞이는 게 빈 달보다 위험하다.
    """

    def _payload(self, total: str) -> dict:
        return {"result": True, "data": [
            {"0": "2026-06", "1": "총계", "2": "아파트", "5": total},
            {"0": "2026-06", "1": "서울", "2": "아파트", "5": "1605"},
        ]}

    def test_전국이_0이면_그_달을_버린다(self) -> None:
        rows, _t, _e = m.parse_starts(self._payload("0"))
        self.assertEqual(rows, [])

    def test_전국이_비현실적으로_크면_버린다(self) -> None:
        rows, _t, _e = m.parse_starts(self._payload("999999"))
        self.assertEqual(rows, [])

    def test_정상_범위는_통과한다(self) -> None:
        rows, _t, _e = m.parse_starts(self._payload("21223"))
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: 실패를 확인한다**

Run: `python3 -m unittest tests.test_molit_stat_api -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'molit_stat_api'`

- [ ] **Step 5: 파서를 구현한다**

```python
"""국토교통 통계누리 주택유형별 착공실적(월계)을 읽는다 — 순수 함수만. I/O 없음.

출처는 stat.molit.go.kr 의 화면 뒤 ajax 엔드포인트(`/portal/stat/data.do`)다.
공식 오픈API 가 아니라서 **화면이 바뀌면 조용히 깨진다.** 그래서 실제 응답을
tests/fixtures 에 고정해 두고 파싱 회귀를 잡는다.

응답은 컬럼 인덱스가 문자열 키인 평평한 배열이다. 컬럼 의미는
`/portal/stat/columns.do?formId=5387&styleNum=1` 이 준다.

    "0" 월    "1" 지역    "2" 대분류    "3" 중분류    "4" 소분류    "5" 착공실적

이 모듈이 잡는 함정 넷. 전부 실제 응답에서 확인했다.

1. **60개월 초과 거절이 HTTP 200 으로 온다.** `{"result": false, "msg": ...}`
   상태코드만 보면 실패를 놓치고 빈 시계열이 조용히 만들어진다.

2. **잠정치 표기.** 최근 약 10개월이 "2026-07 p)" 로 온다. 확정되면서 값이
   바뀌므로 월 문자열에서 떼어 플래그로 분리한다. 수집기가 이걸 보고 재수집한다.

3. **집계행 혼입.** 지역 라벨 23종 중 총계·수도권소계·지방소계·기타광역시·
   기타지방 5종이 합계행이다. 안 거르면 이중계상된다. **블랙리스트가 아니라
   화이트리스트로 거른다** — 블랙리스트는 합계행이 하나 추가되는 순간 뚫린다.

4. **'-' 는 0 이 아니다.** 세종은 2011-01부터 행이 있지만 값이 '-' 이고 실제
   착공은 2012-10부터다. 0 으로 읽으면 세종의 평년 기준선이 바닥으로 깔려
   지수가 폭주한다. 0 은 "그 달에 착공이 없었다" 는 관측값이고 '-' 는
   "모른다" 다. 둘을 섞지 않는다.

전남광주 통합(2026-07-01) 합산은 여기서 하지 않는다 — 이 모듈은 원본에
충실하고, 합치는 일은 build_supply.py 가 한다.
"""

from __future__ import annotations

# 실제 시도 라벨만 통과시킨다. `전남광주` 는 2026-07-01 전남광주통합특별시
# 출범으로 생긴 라벨이고, 같은 달에 `광주`·`전남` 과 동시에 오지는 않는다.
SIDO: tuple[str, ...] = (
    "서울", "인천", "경기", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "전남광주",
)

TOTAL_LABEL = "총계"
APARTMENT = "아파트"

# 한 달 전국 아파트 착공이 이 범위를 벗어나면 응답이 이상한 것이다.
# 최근 10년 최대가 월 5만 호대이므로 20만은 넉넉한 상한이다.
MAX_MONTHLY_NATIONWIDE = 200_000

_MISSING = {"", "-", "–", "—", "…", "..", "."}


def parse_month(text) -> tuple[str | None, bool]:
    """'2026-07 p)' -> ('2026-07', True). 못 읽으면 (None, 잠정여부)."""
    raw = str(text if text is not None else "").strip()
    provisional = raw.endswith("p)")
    if provisional:
        raw = raw[:-2].strip()
    if (len(raw) == 7 and raw[4] == "-"
            and raw[:4].isdigit() and raw[5:].isdigit()
            and 1 <= int(raw[5:]) <= 12):
        return raw, provisional
    return None, provisional


def parse_value(text) -> int | None:
    """착공 세대수. 결측('-')·빈값·비숫자·음수는 None. 0 은 관측값이라 살린다."""
    raw = str(text if text is not None else "").strip()
    if raw in _MISSING:
        return None
    try:
        value = int(float(raw.replace(",", "")))
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_starts(payload) -> tuple[list[dict], dict[str, int], str | None]:
    """(시도 행, 월별 전국 총계, 오류 사유).

    오류면 ([], {}, 사유) 를 돌려준다. 행은 month·region 순으로 정렬돼 있다.
    총계를 따로 돌려주는 이유는 build 단계에서 시도 합과 대조하기 위해서다 —
    지역 라벨이 또 바뀌어 화이트리스트에서 빠지면 그 대조만이 잡을 수 있다.
    """
    if not isinstance(payload, dict):
        return [], {}, "응답이 딕셔너리가 아닙니다."
    if payload.get("result") is False:
        msg = " ".join(str(payload.get("msg") or "").split())
        return [], {}, f"통계누리가 조회를 거절했습니다: {msg}"
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return [], {}, "응답에 data 배열이 없습니다."

    by_month: dict[str, dict[str, int | None]] = {}
    totals: dict[str, int] = {}
    provisional: dict[str, bool] = {}

    for row in data:
        if not isinstance(row, dict) or row.get("2") != APARTMENT:
            continue
        month, is_prov = parse_month(row.get("0"))
        if month is None:
            continue
        provisional[month] = provisional.get(month, False) or is_prov
        region = str(row.get("1") or "").strip()
        units = parse_value(row.get("5"))
        if region == TOTAL_LABEL:
            if units is not None:
                totals[month] = units
            continue
        if region not in SIDO:
            continue
        by_month.setdefault(month, {})[region] = units

    rows: list[dict] = []
    kept_totals: dict[str, int] = {}
    for month in sorted(by_month):
        regions = by_month[month]
        # 총계 행이 있으면 그걸 보고, 없으면 시도 합으로 갈음한다.
        check = totals.get(month)
        if check is None:
            check = sum(v for v in regions.values() if v is not None)
        if not (0 < check <= MAX_MONTHLY_NATIONWIDE):
            continue
        if month in totals:
            kept_totals[month] = totals[month]
        for region in sorted(regions):
            rows.append({
                "month": month,
                "region": region,
                "units": regions[region],
                "provisional": provisional[month],
            })

    if not rows:
        return [], {}, "아파트 착공 행을 한 건도 찾지 못했습니다."
    return rows, kept_totals, None
```

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_molit_stat_api -v`
Expected: PASS (전체 그린)

- [ ] **Step 7: 저장소 전체 테스트가 안 깨졌는지 확인한다**

Run: `python3 -m unittest discover -s tests`
Expected: 기존 테스트 전부 통과

- [ ] **Step 8: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/molit_stat_api.py tests/test_molit_stat_api.py tests/fixtures/molit_starts_2026.json tests/fixtures/molit_starts_2011.json tests/fixtures/molit_range_error.json
git commit -m "Add MOLIT housing-start statistics parser

통계누리 착공실적 응답 파서. 60개월 거절·잠정치 표기·집계행·'-' 결측 넷을 잡는다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: ECOS 응답 파서 (`ecos_api.py`)

**Files:**
- Create: `scripts/ecos_api.py`
- Create: `tests/test_ecos_api.py`
- Create: `tests/fixtures/ecos_base_rate.json`
- Create: `tests/fixtures/ecos_error.json`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `BASE_RATE = ("722Y001", "0101000")` — 한국은행 기준금리
  - `MORTGAGE_RATE = ("121Y006", "BECBLA0302")` — 예금은행 주택담보대출 금리
  - `class KeyMissing(RuntimeError)`
  - `load_key() -> str` — `ECOS_API_KEY` 를 환경변수 → `.env` 순으로 읽는다
  - `series_url(key, stat, item, start, end) -> str` — `start`/`end` 는 `"YYYYMM"`
  - `parse_series(payload) -> tuple[list[tuple[str, float]], str | None]` — `[("2026-01", 2.75), ...]`

- [ ] **Step 1: 픽스처를 만든다**

ECOS 의 `sample` 키는 10건 제한이 있지만 픽스처를 만들기에는 충분하다.

```bash
python3 - <<'PY'
import json, urllib.request, pathlib
OUT = pathlib.Path("tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)


def get(url):
    with urllib.request.urlopen(url, timeout=60) as res:
        return json.load(res)


ok = get("https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr"
         "/1/10/722Y001/M/202501/202510/0101000")
(OUT / "ecos_base_rate.json").write_text(
    json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")

bad = get("https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr"
          "/1/100/722Y001/M/202501/202510/0101000")
(OUT / "ecos_error.json").write_text(
    json.dumps(bad, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok rows:", len(ok["StatisticSearch"]["row"]))
print("error code:", bad["RESULT"]["CODE"])
PY
```

기대 출력: `ok rows: 10`, `error code: ERROR-301`.
오류 픽스처가 `RESULT` 키를 갖는 게 핵심이다 — 이게 **HTTP 200** 으로 온다.

- [ ] **Step 2: 실패하는 테스트를 쓴다**

```python
"""한국은행 ECOS 금리 응답 파싱.

ECOS 는 오류를 **HTTP 200 과 함께** {"RESULT": {"CODE": ...}} 로 돌려준다.
상태코드만 보면 실패를 데이터로 착각한다.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import ecos_api as e  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ParseSeriesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points, self.err = e.parse_series(load("ecos_base_rate.json"))

    def test_오류가_없다(self) -> None:
        self.assertIsNone(self.err)

    def test_월을_대시_형식으로_바꾼다(self) -> None:
        self.assertEqual(self.points[0][0][:4].isdigit(), True)
        self.assertEqual(self.points[0][0][4], "-")
        self.assertEqual(len(self.points[0][0]), 7)

    def test_값이_실수다(self) -> None:
        self.assertIsInstance(self.points[0][1], float)

    def test_월_오름차순으로_정렬한다(self) -> None:
        months = [m for m, _ in self.points]
        self.assertEqual(months, sorted(months))


class ErrorTest(unittest.TestCase):
    def test_RESULT_오류를_데이터로_읽지_않는다(self) -> None:
        points, err = e.parse_series(load("ecos_error.json"))
        self.assertEqual(points, [])
        self.assertIsNotNone(err)
        self.assertIn("ERROR-301", err)

    def test_이상한_구조는_오류다(self) -> None:
        for bad in (None, [], {}, {"StatisticSearch": {}}):
            self.assertIsNotNone(e.parse_series(bad)[1])


class RangeTest(unittest.TestCase):
    def _payload(self, value: str) -> dict:
        return {"StatisticSearch": {"row": [
            {"TIME": "202601", "DATA_VALUE": value}]}}

    def test_범위_밖_금리는_버린다(self) -> None:
        for bad in ("-1", "45", "abc", ""):
            points, _err = e.parse_series(self._payload(bad))
            self.assertEqual(points, [])

    def test_0퍼센트는_살린다(self) -> None:
        points, _err = e.parse_series(self._payload("0"))
        self.assertEqual(points, [("2026-01", 0.0)])


class SeriesUrlTest(unittest.TestCase):
    def test_경로_순서가_맞다(self) -> None:
        url = e.series_url("KEY", "722Y001", "0101000", "201101", "202612")
        self.assertTrue(url.endswith("/722Y001/M/201101/202612/0101000"))
        self.assertIn("/StatisticSearch/KEY/json/kr/1/1000/", url)


class LoadKeyTest(unittest.TestCase):
    def test_환경변수를_먼저_본다(self) -> None:
        with mock.patch.dict(os.environ, {"ECOS_API_KEY": "envkey"}):
            self.assertEqual(e.load_key(), "envkey")

    def test_없으면_env_파일을_본다(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = pathlib.Path(tmp) / ".env"
            env.write_text("# 주석\nECOS_API_KEY=filekey\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(e, "ENV_FILE", env):
                self.assertEqual(e.load_key(), "filekey")

    def test_둘_다_없으면_예외다(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = pathlib.Path(tmp) / "nope.env"
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(e, "ENV_FILE", env):
                with self.assertRaises(e.KeyMissing):
                    e.load_key()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 실패를 확인한다**

Run: `python3 -m unittest tests.test_ecos_api -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ecos_api'`

- [ ] **Step 4: 구현한다**

```python
"""한국은행 ECOS 금리 통계를 읽는다 — 파싱은 순수 함수. 네트워크 I/O 없음.

    기준금리          722Y001 / M / 0101000   1999-05 ~
    주택담보대출 금리  121Y006 / M / BECBLA0302  2001-09 ~

요청 경로의 순서가 문서와 다르면 조용히 다른 통계가 온다.

    /api/StatisticSearch/{키}/json/kr/{시작행}/{끝행}/{통계표}/{주기}/{시작}/{끝}/{항목}

**ECOS 는 오류를 HTTP 200 과 함께 돌려준다.** {"RESULT": {"CODE": "ERROR-301", ...}}
상태코드만 보면 실패를 데이터로 착각한다. 그래서 RESULT 키를 먼저 본다.

load_key 만 파일을 읽는다 — vworld_api.load_key 와 같은 이유·같은 방식이다.
키를 저장소 파일에 두면 gh-pages 가 그대로 웹에 서빙한다.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

BASE_RATE = ("722Y001", "0101000")
MORTGAGE_RATE = ("121Y006", "BECBLA0302")

# 금리가 이 범위를 벗어나면 응답이 이상한 것이다. 국내 정책·대출금리가
# 30% 를 넘은 적은 없고, 음수 금리도 한국은 겪은 적이 없다.
MIN_RATE, MAX_RATE = 0.0, 30.0


class KeyMissing(RuntimeError):
    """ECOS_API_KEY 를 못 찾았다."""


def load_key() -> str:
    """ECOS_API_KEY 를 환경변수에서, 없으면 저장소 루트 .env 에서 읽는다.

    sample 키로 조용히 대체하지 않는다 — sample 은 10건 제한이라 반쪽짜리
    시계열이 만들어지는데, 그건 빈 시계열보다 훨씬 위험하다.
    """
    key = os.environ.get("ECOS_API_KEY")
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or not line.startswith("ECOS_API_KEY="):
                continue
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise KeyMissing(
        "ECOS_API_KEY 를 찾지 못했습니다. ecos.bok.or.kr/api/ 에서 인증키를 "
        "무료로 받아 환경변수로 지정하거나 저장소 루트 .env 에 "
        "ECOS_API_KEY=... 로 넣으세요."
    )


def series_url(key: str, stat: str, item: str, start: str, end: str) -> str:
    """월별 시계열 조회 URL. start·end 는 'YYYYMM'."""
    return f"{BASE_URL}/{key}/json/kr/1/1000/{stat}/M/{start}/{end}/{item}"


def parse_series(payload) -> tuple[list[tuple[str, float]], str | None]:
    """[('2026-01', 2.75), ...] 를 월 오름차순으로. 오류면 ([], 사유)."""
    if not isinstance(payload, dict):
        return [], "응답이 딕셔너리가 아닙니다."
    result = payload.get("RESULT")
    if isinstance(result, dict):
        code = result.get("CODE", "?")
        message = " ".join(str(result.get("MESSAGE") or "").split())
        return [], f"ECOS 가 오류를 돌려줬습니다({code}): {message}"

    block = payload.get("StatisticSearch")
    rows = block.get("row") if isinstance(block, dict) else None
    if not isinstance(rows, list) or not rows:
        return [], "응답에 StatisticSearch.row 가 없습니다."

    points: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        time = str(row.get("TIME") or "").strip()
        if len(time) != 6 or not time.isdigit() or not 1 <= int(time[4:]) <= 12:
            continue
        try:
            value = float(str(row.get("DATA_VALUE") or "").strip())
        except ValueError:
            continue
        if not MIN_RATE <= value <= MAX_RATE:
            continue
        points[f"{time[:4]}-{time[4:]}"] = value

    if not points:
        return [], "쓸 수 있는 금리 값이 한 건도 없습니다."
    return sorted(points.items()), None
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_ecos_api -v`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/ecos_api.py tests/test_ecos_api.py tests/fixtures/ecos_base_rate.json tests/fixtures/ecos_error.json
git commit -m "Add BOK ECOS interest-rate parser

기준금리·주담대 금리 응답 파서. HTTP 200 으로 오는 RESULT 오류를 데이터로 읽지 않는다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 수집기 (`collect_supply.py`)

**Files:**
- Create: `scripts/collect_supply.py`

**Interfaces:**
- Consumes: `molit_stat_api.parse_starts`, `ecos_api.{load_key, series_url, parse_series, BASE_RATE, MORTGAGE_RATE}`
- Produces:
  - `data/supply/starts.json` — `{"fetched": "YYYY-MM-DD", "source": str, "months": {"2026-07": {"provisional": bool, "total": int | None, "regions": {"서울": int | None}}}}`
  - `data/supply/rates.json` — `{"fetched": "YYYY-MM-DD", "base": {"2026-01": 2.75}, "mortgage": {...}}`

- [ ] **Step 1: 구현한다**

```python
#!/usr/bin/env python3
"""아파트 착공 실적과 금리를 받아 원본을 보존한다.

    python3 scripts/collect_supply.py                # 평소
    python3 scripts/collect_supply.py --force        # 확정월까지 전부 다시

두 소스를 받는다.

    착공  국토교통 통계누리 (인증 불필요, 1회 최대 60개월)
    금리  한국은행 ECOS     (ECOS_API_KEY 필요)

**핵심은 잠정치 재수집이다.** 통계누리는 최근 약 10개월을 "2026-07 p)" 처럼
잠정치로 주고 나중에 확정하면서 값을 바꾼다. 확정월만 캐시하고 잠정월은 매
실행 다시 받는다. 이걸 안 하면 캐시가 옛 잠정치를 영원히 붙들고 있게 되는데,
개수를 세는 검사로는 절대 못 잡는 종류의 고장이다.

60개월 제한 때문에 2011-01부터 현재까지를 5년 창으로 잘라 최대 4회 호출한다.
창 안의 모든 달이 이미 있고 확정이면 그 창은 건너뛴다.

실패하면 기존 파일을 그대로 둔다. 지난달 데이터가 빈 파일보다 낫다.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ecos_api  # noqa: E402
import molit_stat_api  # noqa: E402

OUT_DIR = ROOT / "data" / "supply"
STARTS_FILE = OUT_DIR / "starts.json"
RATES_FILE = OUT_DIR / "rates.json"

MOLIT_URL = ("https://stat.molit.go.kr/portal/stat/data.do"
             "?formId=5387&styleNum=1&apprYn=Y&startDate={start}&endDate={end}")
USER_AGENT = "kayserdocs-supply/1.0"
TIMEOUT = 180
FIRST_MONTH = "2011-01"
WINDOW = 60


def month_range(first: str, last: str) -> list[str]:
    """'2011-01'..'2026-07' 를 월 목록으로."""
    year, mon = (int(x) for x in first.split("-"))
    end_year, end_mon = (int(x) for x in last.split("-"))
    out = []
    while (year, mon) <= (end_year, end_mon):
        out.append(f"{year:04d}-{mon:02d}")
        mon += 1
        if mon == 13:
            year, mon = year + 1, 1
    return out


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT,
                      "X-Requested-With": "XMLHttpRequest"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def collect_starts(today: date, force: bool) -> tuple[dict, int]:
    """(months 사전, 실패한 창 수)."""
    existing = load_existing(STARTS_FILE).get("months", {})
    months = dict(existing)
    wanted = month_range(FIRST_MONTH, f"{today.year:04d}-{today.month:02d}")

    def stale(month: str) -> bool:
        if force or month not in months:
            return True
        return bool(months[month].get("provisional"))

    failures = 0
    for i in range(0, len(wanted), WINDOW):
        window = wanted[i:i + WINDOW]
        if not any(stale(m) for m in window):
            print(f"  {window[0]}~{window[-1]} 은 전부 확정본이라 건너뜁니다.")
            continue
        start, end = window[0].replace("-", ""), window[-1].replace("-", "")
        try:
            payload = fetch_json(MOLIT_URL.format(start=start, end=end))
        except Exception as err:
            print(f"  {window[0]}~{window[-1]} 요청 실패({err}).", file=sys.stderr)
            failures += 1
            continue

        rows, totals, error = molit_stat_api.parse_starts(payload)
        if error:
            print(f"  {window[0]}~{window[-1]} 파싱 실패: {error}", file=sys.stderr)
            failures += 1
            continue

        fetched: dict[str, dict] = {}
        for row in rows:
            slot = fetched.setdefault(row["month"], {
                "provisional": row["provisional"],
                "total": totals.get(row["month"]),
                "regions": {},
            })
            slot["regions"][row["region"]] = row["units"]
        months.update(fetched)
        print(f"  {window[0]}~{window[-1]} {len(fetched)}개월 받았습니다.")

    return months, failures


def collect_rates(today: date) -> tuple[dict, int]:
    key = ecos_api.load_key()
    start = FIRST_MONTH.replace("-", "")
    end = f"{today.year:04d}{today.month:02d}"
    out, failures = {}, 0
    for name, (stat, item) in (("base", ecos_api.BASE_RATE),
                               ("mortgage", ecos_api.MORTGAGE_RATE)):
        url = ecos_api.series_url(key, stat, item, start, end)
        try:
            payload = fetch_json(url)
        except Exception as err:
            print(f"  {name} 요청 실패({err}).", file=sys.stderr)
            failures += 1
            continue
        points, error = ecos_api.parse_series(payload)
        if error:
            print(f"  {name} 파싱 실패: {error}", file=sys.stderr)
            failures += 1
            continue
        out[name] = dict(points)
        print(f"  {name} {len(points)}개월 받았습니다.")
    return out, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="아파트 착공·금리 수집")
    parser.add_argument("--force", action="store_true",
                        help="확정월까지 전부 다시 받는다")
    args = parser.parse_args()

    today = date.today()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0

    print("착공 실적을 받습니다 (국토교통 통계누리).")
    months, start_failures = collect_starts(today, args.force)
    failed += start_failures
    if months:
        STARTS_FILE.write_text(json.dumps(
            {"fetched": today.isoformat(),
             "source": "국토교통 통계누리 주택유형별 착공실적(월계) formId=5387",
             "months": dict(sorted(months.items()))},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"착공 {len(months)}개월 -> {STARTS_FILE}")
    else:
        print("착공을 한 달도 받지 못했습니다. 기존 파일을 그대로 둡니다.",
              file=sys.stderr)
        failed += 1

    print("금리를 받습니다 (한국은행 ECOS).")
    try:
        rates, rate_failures = collect_rates(today)
    except ecos_api.KeyMissing as err:
        print(f"{err}", file=sys.stderr)
        return 1
    failed += rate_failures
    if rates:
        merged = load_existing(RATES_FILE)
        merged.update(rates)
        merged["fetched"] = today.isoformat()
        RATES_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"금리 -> {RATES_FILE}")
    else:
        print("금리를 받지 못했습니다. 기존 파일을 그대로 둡니다.", file=sys.stderr)
        failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: ECOS 키가 없을 때 조용히 넘어가지 않는지 확인한다**

Run:
```bash
env -u ECOS_API_KEY python3 - <<'PY'
import pathlib, subprocess, sys
env = pathlib.Path(".env")
backup = env.read_text(encoding="utf-8") if env.exists() else None
try:
    if backup is not None:
        env.write_text("\n".join(l for l in backup.splitlines()
                                 if not l.startswith("ECOS_API_KEY=")) + "\n",
                       encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/collect_supply.py"],
                       capture_output=True, text=True)
    print("exit:", r.returncode)
    print(r.stderr.strip()[-300:])
finally:
    if backup is not None:
        env.write_text(backup, encoding="utf-8")
PY
```
Expected: `exit: 1` 이고 stderr 에 `ECOS_API_KEY 를 찾지 못했습니다` 가 보인다. 착공은 그 전에 받아 저장돼 있어도 된다.

- [ ] **Step 3: 실제로 한 번 받는다**

`.env` 에 `ECOS_API_KEY` 가 있어야 한다. 없으면 여기서 멈추고 사용자에게 발급을 요청한다 (`ecos.bok.or.kr/api/`, 무료·즉시).

Run: `python3 scripts/collect_supply.py`
Expected: 착공 창 4개를 받고 `착공 18x개월 -> …/starts.json`, `금리 -> …/rates.json`. 종료코드 0.

- [ ] **Step 4: 잠정월 재수집이 실제로 도는지 확인한다**

Run:
```bash
python3 -c "
import json,pathlib
d=json.loads(pathlib.Path('data/supply/starts.json').read_text(encoding='utf-8'))
m=d['months']
prov=[k for k,v in m.items() if v['provisional']]
print('총',len(m),'개월 · 잠정',len(prov),'개월:',prov[:3],'...',prov[-1:] if prov else '')
print('전남광주 있는 달:',[k for k,v in m.items() if '전남광주' in v['regions']])
print('세종 결측 달 수:',sum(1 for v in m.values() if v['regions'].get('세종') is None))
"
python3 scripts/collect_supply.py
```
Expected: 두 번째 실행에서 잠정월이 든 창만 다시 받고 나머지는 `전부 확정본이라 건너뜁니다` 가 찍힌다.

- [ ] **Step 5: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/collect_supply.py data/supply/
git commit -m "Add apartment-start and interest-rate collector

통계누리 착공과 ECOS 금리를 받아 원본 보존. 잠정월은 매 실행 재수집한다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 집계 (`build_supply.py`)

**Files:**
- Create: `scripts/build_supply.py`
- Create: `tests/test_build_supply.py`

**Interfaces:**
- Consumes: `data/supply/starts.json`, `data/supply/rates.json` (Task 3 산출)
- Produces:
  - `merge_honam(regions: dict[str, int | None]) -> dict[str, int | None]` — 광주+전남 → `전남광주`
  - `moving_sum(values: list[int | None], window: int = 12) -> list[float | None]`
  - `baseline_index(mavg, months, first, last, min_months) -> list[float | None]`
  - `SIDO_ORDER: tuple[str, ...]` — 화면 표시 순서 (전국 제외 16개)
  - `NATIONWIDE = "전국"`, `TOLERANCE = 0.01`
  - `class TotalMismatch(RuntimeError)`
  - `assets/realestate/supply.json` — 스펙의 스키마 그대로

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
"""아파트 착공 집계.

여기서 틀리면 화면의 모든 숫자가 함께 틀어진다. 특히 셋을 본다.

    광주·전남 합산    2026-07 통합으로 라벨이 바뀐다
    이동합계 결측 전파  창 안에 결측이 있으면 그 지점은 결측이어야 한다
    총계 대조         지역 라벨이 또 바뀌어 누락되는 걸 잡는 유일한 장치
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import build_supply as b  # noqa: E402


class MergeHonamTest(unittest.TestCase):
    def test_통합_이전_달은_더한다(self) -> None:
        merged = b.merge_honam({"광주": 100, "전남": 30, "서울": 900})
        self.assertEqual(merged["전남광주"], 130)
        self.assertNotIn("광주", merged)
        self.assertNotIn("전남", merged)
        self.assertEqual(merged["서울"], 900)

    def test_통합_이후_달은_그대로_쓴다(self) -> None:
        merged = b.merge_honam({"전남광주": 638, "서울": 900})
        self.assertEqual(merged["전남광주"], 638)

    def test_반쪽만_있으면_결측이다(self) -> None:
        """반쪽만 더하면 통합 이전 구간을 조용히 과소계상한다."""
        self.assertIsNone(b.merge_honam({"광주": 100, "전남": None})["전남광주"])
        self.assertIsNone(b.merge_honam({"광주": 100})["전남광주"])

    def test_둘_다_0이면_0이다(self) -> None:
        self.assertEqual(b.merge_honam({"광주": 0, "전남": 0})["전남광주"], 0)


class MovingSumTest(unittest.TestCase):
    def test_앞_11개월은_정의되지_않는다(self) -> None:
        got = b.moving_sum([1] * 24)
        self.assertEqual(got[:11], [None] * 11)
        self.assertEqual(got[11], 12)

    def test_창_안에_결측이_있으면_그_지점도_결측이다(self) -> None:
        values = [1] * 24
        values[5] = None
        got = b.moving_sum(values)
        self.assertIsNone(got[11])
        self.assertIsNone(got[16])
        self.assertEqual(got[17], 12)

    def test_0은_결측이_아니다(self) -> None:
        values = [1] * 24
        values[5] = 0
        self.assertEqual(b.moving_sum(values)[11], 11)


class BaselineIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.months = [f"{y:04d}-{m:02d}"
                       for y in range(2011, 2021) for m in range(1, 13)]

    def test_기준_구간_평균이_100이_된다(self) -> None:
        mavg = [None if mo < "2011-12" else 200.0 for mo in self.months]
        idx = b.baseline_index(mavg, self.months, "2011-12", "2019-12", 60)
        pos = self.months.index("2015-06")
        self.assertAlmostEqual(idx[pos], 100.0, places=6)

    def test_기준의_절반이면_50이다(self) -> None:
        mavg = [None if mo < "2011-12" else (200.0 if mo <= "2019-12" else 100.0)
                for mo in self.months]
        idx = b.baseline_index(mavg, self.months, "2011-12", "2019-12", 60)
        self.assertAlmostEqual(idx[self.months.index("2020-06")], 50.0, places=6)

    def test_유효_개월이_모자라면_전부_None(self) -> None:
        """짧은 기준선으로 만든 지수는 숫자만 그럴듯하고 뜻이 없다."""
        mavg = [200.0 if "2019-01" <= mo <= "2019-12" else None
                for mo in self.months]
        idx = b.baseline_index(mavg, self.months, "2011-12", "2019-12", 60)
        self.assertTrue(all(v is None for v in idx))

    def test_기준_평균이_0이면_전부_None(self) -> None:
        mavg = [0.0 if mo >= "2011-12" else None for mo in self.months]
        idx = b.baseline_index(mavg, self.months, "2011-12", "2019-12", 60)
        self.assertTrue(all(v is None for v in idx))


class TotalCrossCheckTest(unittest.TestCase):
    """지역 라벨이 또 바뀌어 화이트리스트에서 빠지면 이것만이 잡는다.

    개수를 세는 검사는 못 잡는다 — 지역 수가 줄어도 나머지는 멀쩡하기 때문이다.
    """

    def test_시도_합이_총계와_맞으면_통과한다(self) -> None:
        b.check_total("2026-06", {"서울": 100, "경기": 200}, 300, b.TOLERANCE)

    def test_1퍼센트_이내_차이는_봐준다(self) -> None:
        b.check_total("2026-06", {"서울": 100, "경기": 200}, 302, b.TOLERANCE)

    def test_시도가_빠지면_실패한다(self) -> None:
        with self.assertRaises(b.TotalMismatch):
            b.check_total("2026-06", {"서울": 100}, 300, b.TOLERANCE)

    def test_총계가_없으면_넘어간다(self) -> None:
        b.check_total("2026-06", {"서울": 100}, None, b.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python3 -m unittest tests.test_build_supply -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_supply'`

- [ ] **Step 3: 구현한다**

```python
#!/usr/bin/env python3
"""착공·금리 원본을 화면이 쓸 집계본으로 굽는다.

    python3 scripts/build_supply.py   ->  assets/realestate/supply.json

세 가지를 계산한다.

    units  월별 아파트 착공 세대수 (원계열)
    mavg   12개월 이동합계 — 착공은 단발 대규모 사업 하나에 월값이 통째로
           흔들려서(서울 2025-01 1,605호 → 2026-06 3,491호) 원계열은 톱니다.
           이동'평균'이 아니라 '합계' 를 쓰는 이유는 단위가 "최근 1년간 착공
           호수" 라 직관적이기 때문이다.
    index  평년 지수 — 각 시도의 2011-12~2019-12 이동합계 평균을 100 으로.
           2020 이후는 코로나·금리급등이 겹친 예외 구간이라 기준선에서 뺐다.
           절대량으로는 서울 3,491호와 대구 21호를 같은 축에서 못 비교한다.

전남광주 통합(2026-07-01)을 여기서 처리한다. **광주와 전남을 전 기간 합산**해
하나로 다룬다 — 그래야 평년 기준선의 분모가 시계열 내내 일관된다.

전국은 응답의 `총계` 행을 쓰지 않고 16개 시도를 직접 더해 만든다. `총계` 는
대조에만 쓴다. 차이가 1% 를 넘으면 화이트리스트에서 시도가 빠졌거나 라벨이
바뀐 것이므로 빌드를 실패시킨다. 개수를 세는 검사로는 이 고장을 못 잡는다.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTS_FILE = ROOT / "data" / "supply" / "starts.json"
RATES_FILE = ROOT / "data" / "supply" / "rates.json"
OUT_FILE = ROOT / "assets" / "realestate" / "supply.json"

NATIONWIDE = "전국"
MERGED = "전남광주"
HONAM = ("광주", "전남")

# 화면 표시 순서. 수도권 → 광역시 → 도. 통합 라벨은 옛 전남 자리에 둔다.
SIDO_ORDER: tuple[str, ...] = (
    "서울", "경기", "인천", "부산", "대구", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남광주", "경북", "경남", "제주",
)

CODES: dict[str, str] = {
    NATIONWIDE: "00", "서울": "11", "경기": "41", "인천": "28", "부산": "26",
    "대구": "27", "대전": "30", "울산": "31", "세종": "36", "강원": "51",
    "충북": "43", "충남": "44", "전북": "52", "전남광주": "46", "경북": "47",
    "경남": "48", "제주": "50",
}

WINDOW = 12
BASELINE_FROM, BASELINE_TO = "2011-12", "2019-12"
MIN_BASELINE_MONTHS = 60
TOLERANCE = 0.01


class TotalMismatch(RuntimeError):
    """시도 합이 응답의 총계와 어긋난다 — 지역이 누락됐다는 뜻이다."""


def merge_honam(regions: dict) -> dict:
    """광주 + 전남 -> 전남광주. 이미 통합 라벨이면 그대로 둔다.

    둘 중 하나만 값이 있으면 결측으로 둔다. 반쪽만 더한 값은 통합 이전
    구간을 조용히 과소계상한다.
    """
    out = {k: v for k, v in regions.items() if k not in HONAM}
    if MERGED in regions:
        return out
    if not any(k in regions for k in HONAM):
        return out
    parts = [regions.get(k) for k in HONAM]
    out[MERGED] = None if any(p is None for p in parts) else sum(parts)
    return out


def moving_sum(values: list, window: int = WINDOW) -> list:
    """창 안에 결측이 하나라도 있으면 그 지점은 결측이다.

    결측을 0 으로 메우면 "그 달에 안 지었다" 와 "모른다" 가 뒤섞여, 세종처럼
    시작이 늦은 시도의 이동합계가 실제보다 낮게 나온다.
    """
    out: list = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = values[i - window + 1:i + 1]
        out.append(None if any(v is None for v in chunk) else float(sum(chunk)))
    return out


def baseline_index(mavg: list, months: list, first: str, last: str,
                   min_months: int) -> list:
    """평년 = 100 지수. 기준 구간 유효 개월이 모자라면 전부 None."""
    picked = [v for mo, v in zip(months, mavg)
              if v is not None and first <= mo <= last]
    if len(picked) < min_months:
        return [None] * len(mavg)
    mean = sum(picked) / len(picked)
    if mean <= 0:
        return [None] * len(mavg)
    return [None if v is None else round(v / mean * 100, 1) for v in mavg]


def check_total(month: str, regions: dict, total, tolerance: float) -> None:
    """시도 합과 응답의 총계를 대조한다. 총계가 없으면 넘어간다."""
    if total is None:
        return
    got = sum(v for v in regions.values() if v is not None)
    if total <= 0:
        return
    if abs(got - total) / total > tolerance:
        raise TotalMismatch(
            f"{month}: 시도 합 {got:,} 이 총계 {total:,} 와 "
            f"{abs(got - total) / total:.1%} 어긋납니다. "
            f"지역 라벨이 바뀌어 molit_stat_api.SIDO 에서 빠졌을 수 있습니다."
        )


def main() -> int:
    if not STARTS_FILE.exists():
        print(f"{STARTS_FILE} 이 없습니다. collect_supply.py 를 먼저 돌리세요.",
              file=sys.stderr)
        return 1

    starts = json.loads(STARTS_FILE.read_text(encoding="utf-8"))
    rates = json.loads(RATES_FILE.read_text(encoding="utf-8")) \
        if RATES_FILE.exists() else {}

    raw_months = starts.get("months", {})
    months = sorted(raw_months)
    if not months:
        print("착공 원본에 달이 하나도 없습니다.", file=sys.stderr)
        return 1

    merged_by_month: dict[str, dict] = {}
    for month in months:
        slot = raw_months[month]
        regions = merge_honam(slot.get("regions", {}))
        check_total(month, regions, slot.get("total"), TOLERANCE)
        merged_by_month[month] = regions

    series: list[dict] = []
    national = [0] * len(months)
    national_missing = [False] * len(months)

    for name in SIDO_ORDER:
        units = [merged_by_month[m].get(name) for m in months]
        for i, value in enumerate(units):
            if value is None:
                national_missing[i] = True
            else:
                national[i] += value
        mavg = moving_sum(units)
        series.append({
            "code": CODES[name], "name": name, "units": units, "mavg": mavg,
            "index": baseline_index(mavg, months, BASELINE_FROM, BASELINE_TO,
                                    MIN_BASELINE_MONTHS),
        })

    national_units = [None if miss else value
                      for value, miss in zip(national, national_missing)]
    national_mavg = moving_sum(national_units)
    series.insert(0, {
        "code": CODES[NATIONWIDE], "name": NATIONWIDE, "units": national_units,
        "mavg": national_mavg,
        "index": baseline_index(national_mavg, months, BASELINE_FROM,
                                BASELINE_TO, MIN_BASELINE_MONTHS),
    })

    provisional = [m for m in months if raw_months[m].get("provisional")]
    payload = {
        "generated": date.today().isoformat(),
        "latest_month": months[-1],
        "provisional_from": provisional[0] if provisional else None,
        "baseline": {"from": BASELINE_FROM, "to": BASELINE_TO,
                     "min_months": MIN_BASELINE_MONTHS},
        "months": months,
        "regions": series,
        "rates": {
            "base": [rates.get("base", {}).get(m) for m in months],
            "mortgage": [rates.get("mortgage", {}).get(m) for m in months],
        },
        "notes": [
            "광주·전남은 2026-07-01 전남광주통합특별시 출범에 따라 전 기간 합산했습니다.",
            "시군구 단위 착공 통계는 공개되지 않아 시도까지만 봅니다.",
        ],
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    size = OUT_FILE.stat().st_size / 1024
    print(f"{len(months)}개월 × {len(series)}계열 -> {OUT_FILE} ({size:.0f}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_build_supply -v`
Expected: PASS

- [ ] **Step 5: 실제 데이터로 굽고 눈으로 검산한다**

Run:
```bash
python3 scripts/build_supply.py
python3 -c "
import json,pathlib
d=json.loads(pathlib.Path('assets/realestate/supply.json').read_text(encoding='utf-8'))
print('기간',d['months'][0],'~',d['latest_month'],'· 잠정 시작',d['provisional_from'])
print('계열',len(d['regions']),[r['name'] for r in d['regions']])
i=d['months'].index('2026-06')
for r in d['regions']:
    if r['name'] in ('전국','서울','전남광주','세종'):
        print(f\"{r['name']:6s} units={r['units'][i]} mavg={r['mavg'][i]} index={r['index'][i]}\")
print('금리 2026-06: 기준',d['rates']['base'][i],'· 주담대',d['rates']['mortgage'][i])
"
```
Expected:
- 계열이 **17개**(전국 + 시도 16)
- 서울 `units` 이 `3491` (실호출로 확인한 2026-06 값)
- `전남광주` 의 index 가 `null` 이 아니다
- `세종` 의 index 가 `null` 이 아니다 (기준 구간 유효 개월 약 76 ≥ 60)
- 금리 두 값이 모두 숫자다
- 총계 대조 예외가 나지 않는다

`TotalMismatch` 가 나면 통계누리가 지역 라벨을 또 바꾼 것이다. 예외 메시지의 달을 보고 그 달의 원본 지역 라벨을 확인한 뒤 `molit_stat_api.SIDO` 에 추가한다.

- [ ] **Step 6: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_supply.py tests/test_build_supply.py assets/realestate/supply.json
git commit -m "Aggregate apartment starts into baseline index

12개월 이동합계와 평년=100 지수. 광주·전남 전 기간 합산, 총계 대조로 지역 누락을 잡는다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 차트 모듈 확장 (`charts.js`)

기존 `multiLineChart()` 를 그대로 쓰되 세 가지가 모자라다. **전부 기본값이 현재 동작과 같은 후방호환 옵션으로 추가한다** — 이 함수는 학군 페이지(`schools-app.js`)가 이미 쓰고 있어서 동작이 바뀌면 그쪽이 조용히 깨진다.

| 옵션 | 왜 필요한가 |
|---|---|
| `yMin` | 금리는 2~4% 구간인데 y축이 0부터 시작하면 변화가 납작해진다 |
| `xTicks` | 현재 x축 라벨은 `Number(c) % 5 === 0` 으로 고른다. 우리 categories 는 `"2011-01"` 이라 `NaN` 이 되어 처음·끝만 찍힌다 |
| `partialFrom` | 잠정 구간 시작에 세로 점선. `lineChart()` 에는 이미 있고 `multiLineChart()` 에는 없다 |

**Files:**
- Modify: `assets/realestate/charts.js:95-207` (`multiLineChart`)

**Interfaces:**
- Consumes: `palette.js` 의 `MUTED`, `GRID`, `AXIS`
- Produces: `multiLineChart(categories, series, { unit, decimals, height, endLabels, yMin, xTicks, partialFrom })`
  - `yMin: number` — y축 하단값. 기본 `0` (현재 동작)
  - `xTicks: number[] | null` — 라벨을 찍을 인덱스 목록. 기본 `null` (현재 동작)
  - `partialFrom: number | null` — 세로 점선을 그을 인덱스. 기본 `null`

- [ ] **Step 1: 회귀 대조군을 먼저 뜬다**

바꾸기 **전에** 학군 페이지의 현재 SVG 를 저장해 둔다. 나중에 이것과 바이트가 같아야 후방호환이 증명된다.

```bash
mkdir -p /tmp/charts-check
python3 - <<'PY'
import pathlib, re
src = pathlib.Path("assets/realestate/charts.js").read_text(encoding="utf-8")
pathlib.Path("/tmp/charts-check/before.js").write_text(src, encoding="utf-8")
print("saved", len(src), "bytes")
PY
```

- [ ] **Step 2: `yMin` 을 추가한다**

`multiLineChart` 의 옵션 구조분해와 스케일 계산을 고친다.

```javascript
export function multiLineChart(categories, series,
  { unit = '', decimals = 1, height = H, endLabels = 'full',
    yMin = 0, xTicks = null, partialFrom = null } = {}) {
  const finiteAll = series.flatMap((s) => s.values.filter((v) => v != null));
  if (!finiteAll.length) {
    return `<svg viewBox="0 0 ${W} ${height}"><text x="${W / 2}" y="${height / 2}" `
      + `text-anchor="middle" font-size="12" fill="${MUTED}">자료 없음</text></svg>`;
  }
  // yMin 은 금리처럼 0 에서 멀리 떨어진 계열을 위한 것이다. 0 부터 그리면
  // 2~4% 구간의 변화가 납작해져 아무것도 안 보인다. 기본값 0 은 기존 동작이다.
  const lo = Math.min(yMin, ...finiteAll);
  const max = Math.max(...finiteAll) * 1.15;
  const span = (max - lo) || 1;
  const n = categories.length;
  const iw = W - PAD_L - PAD_R, ih = height - PAD_T - PAD_B;
  const X = (i) => PAD_L + (n > 1 ? (i / (n - 1)) * iw : iw / 2);
  const Y = (v) => PAD_T + (1 - (v - lo) / span) * ih;
  const fmt = (v) => `${decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString()}${unit}`;
```

- [ ] **Step 3: 눈금과 바닥선이 `yMin` 을 따르게 고친다**

`niceTicks(max)` 는 0부터 시작하는 눈금만 만든다. `lo` 를 더해 옮긴다.

```javascript
  for (const t of niceTicks(max - lo).map((v) => v + lo)) {
    parts.push(`<line x1="${PAD_L}" y1="${Y(t).toFixed(1)}" x2="${W - PAD_R}" `
      + `y2="${Y(t).toFixed(1)}" stroke="${GRID}" stroke-width="1"/>`);
    parts.push(`<text x="${PAD_L - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end" `
      + `font-size="9" fill="${MUTED}">${decimals > 0 ? t.toFixed(decimals) : Math.round(t).toLocaleString()}</text>`);
  }
```

바닥 축선은 `Y(0)` 이 아니라 `Y(lo)` 여야 한다 — `yMin` 이 0 이면 결과가 같다.

```javascript
  parts.push(`<line x1="${PAD_L}" y1="${Y(lo).toFixed(1)}" x2="${W - PAD_R}" `
    + `y2="${Y(lo).toFixed(1)}" stroke="${AXIS}" stroke-width="1"/>`);
```

- [ ] **Step 4: `xTicks` 를 추가한다**

기존 x축 라벨 블록을 이렇게 바꾼다.

```javascript
  // 라벨을 다 찍으면 390px 폭에서 겹친다. 기본은 연도 categories 를 5년 단위로
  // 고르는 기존 동작이고, xTicks 를 주면 그 인덱스만 찍는다 — "2011-01" 처럼
  // Number() 로 못 읽는 categories 를 쓰는 쪽(공급 대시보드)이 이걸 쓴다.
  const shown = new Set(xTicks || [0, n - 1]);
  if (!xTicks) {
    categories.forEach((c, i) => { if (Number(c) % 5 === 0) shown.add(i); });
  }
  shown.forEach((i) => {
    if (i < 0 || i >= n) return;
    parts.push(`<text x="${X(i).toFixed(1)}" y="${height - 7}" text-anchor="middle" `
      + `font-size="9" fill="${MUTED}">${esc(categories[i])}</text>`);
  });
```

- [ ] **Step 5: `partialFrom` 을 추가한다**

`parts.push('</svg>')` 바로 앞에 넣는다. `lineChart()` 의 같은 블록과 모양을 맞춘다.

```javascript
  if (partialFrom != null && partialFrom >= 0 && partialFrom < n) {
    parts.push(`<line x1="${X(partialFrom).toFixed(1)}" y1="${PAD_T}" `
      + `x2="${X(partialFrom).toFixed(1)}" y2="${(height - PAD_B).toFixed(1)}" `
      + `stroke="${MUTED}" stroke-width="1" stroke-dasharray="3 3"/>`);
  }
```

- [ ] **Step 6: 후방호환을 증명한다**

옵션을 안 주면 바꾸기 전과 **바이트가 같은** SVG 가 나와야 한다. Node 로 직접 부른다.

```bash
node --input-type=module -e "
import { multiLineChart } from './assets/realestate/charts.js';
const cats = Array.from({length: 15}, (_, i) => String(2011 + i));
const series = [
  { label: '가', color: '#2a78d6', values: cats.map((_, i) => 10 + i * 2) },
  { label: '나', color: '#eb6834', values: cats.map((_, i) => 40 - i) },
];
const svg = multiLineChart(cats, series, { unit: '%', decimals: 1 });
console.log(svg.length);
import('fs').then(fs => fs.writeFileSync('/tmp/charts-check/after.svg', svg));
"
git stash push assets/realestate/charts.js
node --input-type=module -e "
import { multiLineChart } from './assets/realestate/charts.js';
const cats = Array.from({length: 15}, (_, i) => String(2011 + i));
const series = [
  { label: '가', color: '#2a78d6', values: cats.map((_, i) => 10 + i * 2) },
  { label: '나', color: '#eb6834', values: cats.map((_, i) => 40 - i) },
];
import('fs').then(fs => fs.writeFileSync('/tmp/charts-check/before.svg',
  multiLineChart(cats, series, { unit: '%', decimals: 1 })));
"
git stash pop
diff /tmp/charts-check/before.svg /tmp/charts-check/after.svg && echo "후방호환 확인: 출력이 같습니다"
```
Expected: `후방호환 확인: 출력이 같습니다`. 차이가 있으면 기본값이 기존 동작을 벗어난 것이므로 되돌린다.

- [ ] **Step 7: 새 옵션이 실제로 먹는지 확인한다**

```bash
node --input-type=module -e "
import { multiLineChart } from './assets/realestate/charts.js';
const cats = Array.from({length: 24}, (_, i) =>
  \`\${2011 + Math.floor(i / 12)}-\${String(i % 12 + 1).padStart(2, '0')}\`);
const series = [{ label: '기준금리', color: '#2a78d6',
                  values: cats.map((_, i) => 2 + (i % 5) * 0.25) }];
const svg = multiLineChart(cats, series,
  { unit: '%', decimals: 2, yMin: 1.5, xTicks: [0, 12, 23], partialFrom: 18 });
console.log('점선:', /stroke-dasharray=\"3 3\"/.test(svg));
console.log('x라벨 수:', (svg.match(/text-anchor=\"middle\"/g) || []).length);
console.log('최소 눈금 1.5 이상:', !/>1\.[0-4]\d</.test(svg));
"
```
Expected: `점선: true`, `x라벨 수: 3`, `최소 눈금 1.5 이상: true`

- [ ] **Step 8: 학군 페이지가 안 깨졌는지 확인한다**

메모리의 대시보드 JS 하네스 검증 방식을 쓴다 — Jekyll 빌드 없이 `schools-app.js` 가 그리는 진학률 추이 SVG 를 그대로 뜬다. 대조군은 Step 6 의 `git stash` 방식과 같다.

Run: 하네스 페이지로 `/dashboard/real-estate/schools/` 의 추이 차트를 렌더해 stash 전후 DOM 을 비교한다.
Expected: 차이 없음

- [ ] **Step 9: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add assets/realestate/charts.js
git commit -m "Add yMin, xTicks and partialFrom options to multiLineChart

금리처럼 0 에서 먼 계열과 월 단위 x축, 잠정 구간 점선을 위한 후방호환 옵션 셋.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 페이지와 화면 스크립트

**Files:**
- Create: `_pages/dashboard-supply.md`
- Create: `assets/realestate/supply-app.js`
- Create: `assets/realestate/supply.css`

**Interfaces:**
- Consumes: `assets/realestate/supply.json` (Task 4), `charts.js` 의 `multiLineChart`·`legendHtml` (Task 5), `palette.js` 의 `CATEGORICAL`·`MUTED`·`LINE`, `freshness.js` 의 `showStale`·`LIMITS`
- Produces: `/dashboard/supply/` 페이지. 상태는 `{ metric: 'index' | 'mavg' | 'units', picks: string[] }`

- [ ] **Step 1: 페이지를 만든다**

```markdown
---
layout: single
title: "시도별 아파트 착공과 금리"
permalink: /dashboard/supply/
classes: wide
author_profile: false
toc: false
description: "2011년 이후 시도별 아파트 착공 세대수를 평년 대비 지수로 바꿔 한국은행 기준금리·주택담보대출 금리와 같은 시간축에 놓았습니다. 착공은 입주보다 2~3년 앞서므로 지금 어디가 얼어붙었는지가 미래 공급의 예고편입니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/supply.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="re-app is-supply" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs" role="tablist" aria-label="지표 선택">
      <button class="re-tab is-on" data-metric="index" role="tab" aria-selected="true">평년=100 지수</button>
      <button class="re-tab" data-metric="mavg" role="tab" aria-selected="false">12개월 이동합계</button>
      <button class="re-tab" data-metric="units" role="tab" aria-selected="false">월별 착공 호수</button>
    </div>
    <div class="re-picks" role="group" aria-label="지역 선택"></div>
    <p class="re-pick-help">지역은 최대 4개까지 고를 수 있습니다. 전국은 회색으로 항상 깔립니다.</p>
  </div>

  <div class="re-panels">
    <figure class="re-panel-chart">
      <figcaption class="re-chart-title">아파트 착공</figcaption>
      <div class="re-chart" data-chart="starts"></div>
      <div class="re-legend re-chart-legend" data-legend="starts"></div>
    </figure>
    <figure class="re-panel-chart">
      <figcaption class="re-chart-title">금리</figcaption>
      <div class="re-chart" data-chart="rates"></div>
      <div class="re-legend re-chart-legend" data-legend="rates"></div>
    </figure>
  </div>

  <p class="re-footnote"></p>

  <h2>이 화면을 읽는 법</h2>
  <p class="re-note">
    <b>착공은 입주보다 2~3년 앞섭니다.</b> 그래서 지금 착공이 줄고 있으면 2~3년 뒤 입주가 마릅니다.
    이 화면이 답하려는 질문이 그것입니다.
    <br><br>
    <b>평년=100 지수</b>는 각 시도의 2011-12~2019-12 12개월 이동합계 평균을 100으로 놓은 값입니다.
    서울 3,491호와 대구 21호는 절대량으로 같은 축에서 비교되지 않기 때문입니다. 지수가 50이면
    그 시도가 평년의 절반만 짓고 있다는 뜻입니다. 2020년 이후를 기준선에서 뺀 이유는 코로나와
    금리 급등이 겹친 예외 구간이 기준 자체를 왜곡하기 때문입니다.
    <br><br>
    <b>12개월 이동합계</b>를 기본으로 쓰는 이유는 착공이 단발 대규모 사업 하나에 월값이 통째로
    흔들리기 때문입니다. 원계열은 톱니라 추세가 안 보입니다.
    <br><br>
    <b>두 패널을 위아래로 놓은 것은 의도한 선택입니다.</b> 착공과 금리를 이중 Y축 한 그림에 겹치면
    두 축의 눈금을 어떻게 잡느냐에 따라 상관관계가 있어 보이게도 없어 보이게도 만들 수 있습니다.
    같은 시간축을 공유하는 두 패널은 시점 대응을 그대로 보여주면서 그 조작 여지가 없습니다.
    <br><br>
    <b>그리고 이건 상관이지 인과가 아닙니다.</b> 금리가 오른 뒤 착공이 줄어든 것처럼 보여도,
    같은 시기에 원자재값·PF 시장·분양가 규제가 함께 움직였습니다. 이 화면은 두 선을 나란히
    놓기만 하고 상관계수나 회귀를 계산하지 않습니다.
  </p>

  <h2>자료와 한계</h2>
  <p class="re-note">
    착공은 <b>국토교통 통계누리 주택유형별 착공실적(월계)</b>, 금리는 <b>한국은행 ECOS</b>
    (기준금리, 예금은행 주택담보대출 신규취급액 가중평균)입니다.
    <br><br>
    <b>시군구 단위는 없습니다.</b> 주택 착공·준공·인허가·분양 통계가 전부 시도까지만 공개됩니다.
    시군구로 월별 제공되는 주택 통계는 미분양뿐입니다.
    <br><br>
    <b>광주와 전남은 하나로 합쳤습니다.</b> 2026년 7월 1일 전남광주통합특별시가 출범해 통계에서
    두 지역이 한 줄로 바뀌었습니다. 통합 이전 구간도 두 값을 더해 시계열 내내 같은 분모를
    유지했습니다.
    <br><br>
    <b>최근 몇 달은 잠정치</b>입니다. 세로 점선 오른쪽이며, 확정되면서 값이 바뀝니다.
    <b>세종</b>은 2012년 10월부터 착공이 잡힙니다. 그 이전은 값이 없는 것이지 0이 아닙니다.
  </p>
</div>

{% include realestate/importmap.html %}
<script type="module" src="{{ '/assets/realestate/supply-app.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
```

- [ ] **Step 2: 화면 스크립트를 만든다**

```javascript
// 시도별 아파트 착공 × 금리. 상하 2단 패널이 x축을 공유한다.
//
// 이중 Y축을 쓰지 않은 것은 의도한 선택이다 — 두 축의 눈금을 어떻게 잡느냐로
// 상관관계를 있어 보이게도 없어 보이게도 만들 수 있어서, 눈금을 고르는 순간
// 만든 사람이 결론을 정하는 셈이 된다.

import { multiLineChart, legendHtml } from './charts.js';
import { CATEGORICAL, MUTED, LINE } from './palette.js';
import { showStale, LIMITS } from './freshness.js';

const root = document.querySelector('.re-app.is-supply');
const base = (root.dataset.base || '/assets/realestate').replace(/\/$/, '');

const els = {
  tabs: root.querySelectorAll('.re-tab'),
  picks: root.querySelector('.re-picks'),
  starts: root.querySelector('[data-chart="starts"]'),
  rates: root.querySelector('[data-chart="rates"]'),
  startsLegend: root.querySelector('[data-legend="starts"]'),
  ratesLegend: root.querySelector('[data-legend="rates"]'),
  footnote: root.querySelector('.re-footnote'),
  panels: root.querySelector('.re-panels'),
};

// 색은 palette.js 의 검증된 4색 조합이다. 5개째부터는 인접쌍 대비 검증이
// 깨지므로 선택을 4개로 막는다. 전국은 회색으로 항상 깔린다.
const MAX_PICKS = CATEGORICAL.length;

const METRICS = {
  index: { unit: '평년=100', decimals: 0, key: 'index' },
  mavg: { unit: '호(최근 1년)', decimals: 0, key: 'mavg' },
  units: { unit: '호', decimals: 0, key: 'units' },
};

const state = { metric: 'index', picks: ['서울', '경기'], data: null };

async function load() {
  const res = await fetch(`${base}/supply.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`supply.json → HTTP ${res.status}`);
  return res.json();
}

// x축은 5년 단위 1월만 찍는다. 187개월을 다 찍으면 390px 에서 겹친다.
function yearTicks(months) {
  const out = [];
  months.forEach((m, i) => {
    if (m.endsWith('-01') && Number(m.slice(0, 4)) % 5 === 0) out.push(i);
  });
  if (!out.includes(months.length - 1)) out.push(months.length - 1);
  return out;
}

function byName(name) {
  return state.data.regions.find((r) => r.name === name);
}

function renderPicks() {
  const names = state.data.regions.map((r) => r.name).filter((n) => n !== '전국');
  els.picks.innerHTML = names.map((name) => {
    const on = state.picks.includes(name);
    const noIndex = state.metric === 'index'
      && byName(name).index.every((v) => v == null);
    return `<button type="button" class="re-pick${on ? ' is-on' : ''}" `
      + `data-name="${name}" aria-pressed="${on}"${noIndex ? ' disabled' : ''}>`
      + `${name}${noIndex ? ' (지수 없음)' : ''}</button>`;
  }).join('');
}

function render() {
  const { months, regions, rates } = state.data;
  const metric = METRICS[state.metric];
  const ticks = yearTicks(months);
  const partial = state.data.provisional_from
    ? months.indexOf(state.data.provisional_from) : null;

  const national = byName('전국');
  const startsSeries = [
    { label: '전국', color: MUTED, values: national[metric.key] },
    ...state.picks.map((name, i) => ({
      label: name, color: CATEGORICAL[i % CATEGORICAL.length],
      values: byName(name)[metric.key],
    })),
  ];

  els.starts.innerHTML = multiLineChart(months, startsSeries, {
    unit: metric.unit, decimals: metric.decimals, height: 200,
    endLabels: 'value', xTicks: ticks, partialFrom: partial,
  });
  els.startsLegend.innerHTML = legendHtml(startsSeries);

  // 금리는 2~4% 구간이라 0 부터 그리면 변화가 납작해진다. yMin 으로 바닥을 올린다.
  const finite = [...rates.base, ...rates.mortgage].filter((v) => v != null);
  const floor = finite.length ? Math.max(0, Math.floor(Math.min(...finite)) - 1) : 0;
  const rateSeries = [
    { label: '기준금리', color: LINE, values: rates.base },
    { label: '주담대', color: CATEGORICAL[1], values: rates.mortgage },
  ];
  els.rates.innerHTML = multiLineChart(months, rateSeries, {
    unit: '%', decimals: 2, height: 170, endLabels: 'full',
    yMin: floor, xTicks: ticks, partialFrom: partial,
  });
  els.ratesLegend.innerHTML = legendHtml(rateSeries);

  const baseline = state.data.baseline;
  els.footnote.textContent =
    `${months[0]} ~ ${state.data.latest_month} · 평년 기준 ${baseline.from}~${baseline.to}`
    + `${state.data.provisional_from ? ` · ${state.data.provisional_from}부터 잠정치(점선 오른쪽)` : ''}`;
  renderPicks();
}

function bind() {
  els.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      state.metric = tab.dataset.metric;
      els.tabs.forEach((t) => {
        const on = t === tab;
        t.classList.toggle('is-on', on);
        t.setAttribute('aria-selected', String(on));
      });
      render();
    });
  });

  els.picks.addEventListener('click', (event) => {
    const button = event.target.closest('.re-pick');
    if (!button || button.disabled) return;
    const name = button.dataset.name;
    const at = state.picks.indexOf(name);
    if (at >= 0) {
      state.picks.splice(at, 1);
    } else {
      // 색 대비가 검증된 조합이 4색이다. 넘치면 가장 오래된 선택을 밀어낸다.
      if (state.picks.length >= MAX_PICKS) state.picks.shift();
      state.picks.push(name);
    }
    render();
  });
}

load().then((data) => {
  state.data = data;
  state.picks = state.picks.filter((n) => data.regions.some((r) => r.name === n));
  bind();
  render();
  showStale(els.panels, data.generated, LIMITS.supply, '착공·금리 자료');
}).catch((err) => {
  els.starts.textContent = `자료를 불러오지 못했습니다 (${err.message}).`;
});
```

- [ ] **Step 3: 스타일을 만든다**

390px 에서 가로로 넘치지 않는 것이 요구사항이다. 패널은 세로로 쌓고, 지역 칩은 줄바꿈한다.

```css
/* 시도별 아파트 착공 × 금리. dashboard.css 의 --re-* 토큰을 그대로 쓴다. */

.re-app.is-supply .re-panels {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin: 1rem 0;
}

.re-app.is-supply .re-panel-chart {
  margin: 0;
  padding: 0.5rem 0.25rem;
  border: 1px solid var(--re-grid, #e3e5e7);
  border-radius: 6px;
  background: #fff;
}

.re-app.is-supply .re-chart-title {
  margin: 0 0 0.25rem 0.5rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--re-ink, #3d4144);
}

/* SVG 는 viewBox 로 스스로 줄어든다. 폭을 100% 로 묶어 부모를 넘지 않게 한다. */
.re-app.is-supply .re-chart svg {
  display: block;
  width: 100%;
  height: auto;
}

.re-app.is-supply .re-chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 0.75rem;
  margin: 0.25rem 0.5rem 0;
  font-size: 0.75rem;
  color: var(--re-ink2, #646769);
}

.re-app.is-supply .re-chart-legend i {
  display: inline-block;
  width: 0.65rem;
  height: 0.65rem;
  margin-right: 0.25rem;
  border-radius: 2px;
  vertical-align: baseline;
}

.re-app.is-supply .re-picks {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-top: 0.5rem;
}

.re-app.is-supply .re-pick {
  padding: 0.2rem 0.5rem;
  border: 1px solid var(--re-grid, #e3e5e7);
  border-radius: 999px;
  background: #fff;
  font-size: 0.78rem;
  color: var(--re-ink2, #646769);
  cursor: pointer;
}

.re-app.is-supply .re-pick.is-on {
  border-color: #2a78d6;
  background: #eaf2fd;
  color: #184f95;
  font-weight: 600;
}

.re-app.is-supply .re-pick[disabled] {
  opacity: 0.45;
  cursor: not-allowed;
}

.re-app.is-supply .re-pick-help,
.re-app.is-supply .re-footnote {
  margin: 0.4rem 0 0;
  font-size: 0.75rem;
  color: var(--re-muted, #7a8288);
}
```

- [ ] **Step 4: Jekyll 로 빌드해 페이지가 나오는지 확인한다**

리눅스에는 로컬 루비가 없다. Docker 를 쓰고 `jekyll-sass-converter ~> 2.0` 을 고정한다 (`sass-embedded` 가 죽는다).

```bash
mkdir -p /tmp/gemhome && sed 's/^gem "jekyll".*/&\ngem "jekyll-sass-converter", "~> 2.0"/' Gemfile > /tmp/gemhome/Gemfile
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /srv/jekyll/_site_check"
ls _site_check/dashboard/supply/index.html
```
Expected: 파일이 존재한다. 빌드 경고에 `supply` 관련 Liquid 오류가 없다.

- [ ] **Step 5: 헤드리스 크롬으로 실제로 그려지는지 본다**

```bash
python3 - <<'PY'
import subprocess, pathlib, re
page = pathlib.Path("_site_check/dashboard/supply/index.html").resolve()
out = subprocess.run([
    "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
    "--virtual-time-budget=6000", "--dump-dom", f"file://{page}",
], capture_output=True, text=True, timeout=120).stdout
print("착공 SVG:", out.count('data-chart="starts"'), "polyline:",
      len(re.findall(r"<polyline", out)))
print("금리 SVG 안 점선:", "stroke-dasharray" in out)
print("지역 칩 수:", len(re.findall(r'class="re-pick', out)))
print("오류 문구:", "불러오지 못했습니다" in out)
PY
```
Expected: `polyline` 이 여러 개, 점선 `True`, 지역 칩 16개, 오류 문구 `False`.

`file://` 로는 `fetch` 가 막힐 수 있다. 그러면 `python3 -m http.server` 로 `_site_check` 를 띄우고 `http://localhost:PORT/dashboard/supply/` 를 연다.

- [ ] **Step 6: 커밋**

```bash
rm -rf _site_check
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add _pages/dashboard-supply.md assets/realestate/supply-app.js assets/realestate/supply.css
git commit -m "Add apartment starts and interest rate dashboard page

시도별 착공과 금리를 x축 공유 2단 패널로. 이중축을 피한 이유를 화면에 적었다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 크론·신선도·허브 등록

이걸 빼면 대시보드가 만들어진 날의 데이터에서 멈추고, **아무도 그 사실을 모른다.** 이 저장소가 실제로 겪은 고장이다 (ETF 대시보드가 크론 등록 없이 "매일 갱신"을 자처했다).

**Files:**
- Create: `scripts/supply_monthly.sh`
- Modify: `scripts/freshness_api.py:44-56` (`BUILD_LIMITS`, `MONTH_LAG_LIMITS`)
- Modify: `scripts/check_freshness.py:49-55` (`TARGETS`)
- Modify: `assets/realestate/freshness.js:24-28` (`LIMITS`)
- Modify: `_pages/dashboard.md` (부동산 절에 카드 추가)

**Interfaces:**
- Consumes: `assets/realestate/supply.json` 의 `generated`·`latest_month`
- Produces: `supply` 라는 이름으로 크론·화면 양쪽 신선도 검사에 등록

- [ ] **Step 1: `freshness_api.py` 에 임계값을 넣는다**

```python
BUILD_LIMITS: dict[str, int] = {
    "trades": 3,
    "schools": 40,
    "redev": 3,
    "pokemon": 3,
    # 월 2회(5일·25일) 도는 월간 통계다. 한 번 걸러도 조용하고, 두 번 연속
    # 거르면 운다.
    "supply": 40,
}
```

```python
MONTH_LAG_LIMITS: dict[str, int] = {
    "trades": 1,
    "redev": 1,
    # 통계누리 착공 발표는 약 1.5개월 뒤진다. 9월에 7월 자료가 최신인 게 정상.
    "supply": 2,
}
```

- [ ] **Step 2: `check_freshness.py` 의 `TARGETS` 에 넣는다**

```python
    "supply": ("assets/realestate/supply.json", "착공·금리",
               ("month", ["latest_month"])),
```

- [ ] **Step 3: `freshness.js` 의 화면 임계값에 넣는다**

화면은 크론보다 이틀 넉넉하다 — 크론이 이미 따로 우니까, 화면은 헛경보를 안 내는 쪽이 낫다.

```javascript
export const LIMITS = {
  trades: 5,
  schools: 45,
  redev: 5,
  supply: 42,
};
```

- [ ] **Step 4: 등록이 실제로 먹는지 확인한다**

```bash
python3 scripts/check_freshness.py supply
python3 -c "
import sys; sys.path.insert(0,'scripts')
from freshness_api import BUILD_LIMITS, MONTH_LAG_LIMITS
print('build', BUILD_LIMITS['supply'], '· lag', MONTH_LAG_LIMITS['supply'])
"
```
Expected: `착공·금리 집계 2026-09-07(0일 전) · …` 이 찍히고 종료코드 0.

- [ ] **Step 5: 크론 스크립트를 만든다**

```bash
#!/usr/bin/env bash
# 시도별 아파트 착공과 금리를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   20 5 5,25 * * /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/supply_monthly.sh >> /home/dorumugs/.cache/realestate-supply.log 2>&1
#
# 월 2회다. 월간 통계라 매일 돌 이유가 없고, 그래도 두 번인 이유는 통계누리가
# 최근 약 10개월을 잠정치로 주고 확정하면서 값을 바꾸기 때문이다.
#
# 부동산 3종·ETF·포켓몬과 **같은 flock 을 공유한다.** 전부 git commit/push 를
# 하므로 겹치면 한쪽 커밋이 유실된다. 반드시 유지할 것.
#
# 인증키
#   ECOS_API_KEY   한국은행 ECOS. 없으면 수집이 종료코드 1 로 죽는다.
#                  통계누리(착공)는 키가 필요 없다.
#
# 환경변수
#   AUTO_COMMIT   1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH     1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   FORCE_FULL    1 이면 확정월까지 전부 다시 받는다

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"

COLLECT_ARGS=()
[ "${FORCE_FULL:-0}" = "1" ] && COLLECT_ARGS+=(--force)

echo "===== $(date '+%F %T') 착공·금리 수집 시작 ====="

FAILED=0

if ! python3 -u scripts/collect_supply.py "${COLLECT_ARGS[@]}"; then
  FAILED=1
  echo "수집이 끝까지 못 갔습니다 — 통계누리 응답과 ECOS_API_KEY 를 확인하세요." >&2
fi

# 수집이 일부 실패해도 있는 원본으로 다시 굽는다. 어제 집계본보다 낫다.
# 다만 지역 라벨이 바뀌어 총계 대조가 깨지면 여기서 죽는다 — 그건 조용히
# 넘어가면 안 되는 고장이다.
if ! python3 -u scripts/build_supply.py; then
  FAILED=1
  echo "build_supply.py 가 비정상 종료했습니다. 총계 대조 실패라면 통계누리가 지역 라벨을 바꾼 것입니다." >&2
fi

if ! python3 -u scripts/check_freshness.py supply; then
  FAILED=1
  echo "착공·금리 산출물이 낡았습니다." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

TARGETS="assets/realestate/supply.json data/supply"

if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh apartment start and interest rate data

수집 스크립트가 자동 갱신한 시도별 아파트 착공과 금리 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh apartment start and interest rate data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

git add $TARGETS
git commit -m "$COMMIT_MSG"

if [ "$AUTO_PUSH" = "1" ]; then
  export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

  # --autostash 가 없으면 수집과 무관한 미스테이징 편집 하나로도 rebase 가
  # 거부돼 push 가 통째로 막힌다. 그러면 커밋은 로컬에만 남고 사이트는 조용히
  # 멈춘다 — 로컬 파일은 최신이라 신선도 검사도 이걸 못 잡는다.
  if ! git pull --rebase --autostash -q origin gh-pages; then
    echo "pull --rebase 실패. 충돌을 수동으로 정리한 뒤 push 하세요." >&2
    exit 1
  fi
  if ! git push -q origin gh-pages; then
    echo "push 실패 — 커밋이 로컬에만 남았습니다. 사이트는 갱신되지 않습니다." >&2
    exit 1
  fi
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/gh-pages)" ]; then
    echo "push 뒤에도 origin/gh-pages 가 HEAD 와 다릅니다 — 사이트가 안 바뀝니다." >&2
    exit 1
  fi
  echo "push 완료."
fi

echo "===== $(date '+%F %T') 착공·금리 수집 종료 (FAILED=$FAILED) ====="
[ "$FAILED" = "1" ] && exit 1
exit 0
```

- [ ] **Step 6: 크론 스크립트를 커밋 없이 돌려본다**

```bash
chmod +x scripts/supply_monthly.sh
./scripts/supply_monthly.sh
```
Expected: 종료코드 0, 마지막 줄이 `AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다.`
`git status --porcelain` 에 `assets/realestate/supply.json` 만 (혹은 아무것도) 뜬다.

- [ ] **Step 7: 허브에 카드를 추가한다**

`_pages/dashboard.md` 의 `## 부동산` 절, 재개발 카드 다음에 넣는다.

```html
  <a class="rh-card is-live" href="{{ '/dashboard/supply/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">착공과 금리</h2>
    <p class="rh-desc">
      2011년 이후 <strong>시도별 아파트 착공</strong>을 평년 대비 지수로 바꿔
      기준금리·주택담보대출 금리와 같은 시간축에 놓았습니다.
      착공은 입주보다 2~3년 앞섭니다.
    </p>
    <ul class="rh-points">
      <li>평년=100 지수 · 12개월 이동합계 · 월별 착공 호수</li>
      <li>이중축을 쓰지 않은 2단 패널 — 눈금으로 상관을 만들지 않습니다</li>
      <li>시군구 단위 착공 통계는 공개되지 않아 시도까지입니다</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>
```

- [ ] **Step 8: 커밋**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/supply_monthly.sh scripts/freshness_api.py scripts/check_freshness.py assets/realestate/freshness.js _pages/dashboard.md
git commit -m "Wire supply dashboard into cron, freshness checks and hub

월 2회 크론과 신선도 검사 양쪽에 등록. 등록을 빼면 멈춘 걸 아무도 모른다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: 모바일 오버플로 검증과 마무리

저장소 규칙이다 — **모든 페이지는 390px 에서 가로로 넘치면 안 된다.**

**Files:**
- Modify: `assets/realestate/supply.css` (넘치면)
- Modify: `CLAUDE.md` (부동산 파이프라인 표에 한 줄 추가)

**Interfaces:**
- Consumes: Task 6·7 의 산출물 전부
- Produces: 없음 (검증 태스크)

- [ ] **Step 1: 사이트를 빌드한다**

```bash
mkdir -p /tmp/gemhome && sed 's/^gem "jekyll".*/&\ngem "jekyll-sass-converter", "~> 2.0"/' Gemfile > /tmp/gemhome/Gemfile
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /srv/jekyll/_site_check"
```

- [ ] **Step 2: 390px 에서 가로 넘침을 잰다**

`file://` 은 `fetch` 가 막히므로 로컬 서버로 띄운다.

```bash
(cd _site_check && python3 -m http.server 8899 >/dev/null 2>&1 &) ; sleep 2
python3 - <<'PY'
import json, subprocess
JS = """
(() => {
  const doc = document.documentElement;
  const over = [];
  document.querySelectorAll('body *').forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.right > 390.5 || r.left < -0.5) {
      over.push({ tag: el.tagName, cls: el.className && String(el.className).slice(0, 40),
                  left: Math.round(r.left), right: Math.round(r.right) });
    }
  });
  return JSON.stringify({ scrollWidth: doc.scrollWidth,
                          clientWidth: doc.clientWidth, over: over.slice(0, 10) });
})()
"""
out = subprocess.run([
    "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
    "--window-size=390,900", "--virtual-time-budget=8000",
    "--dump-dom", "http://localhost:8899/dashboard/supply/",
], capture_output=True, text=True, timeout=120).stdout
print("DOM 길이:", len(out))
print("polyline:", out.count("<polyline"))
PY
</dev/null
```

`--dump-dom` 만으로는 레이아웃 폭을 못 잰다. 메모리에 적힌 대로 **raw CDP** 로 `Runtime.evaluate` 를 써서 `document.documentElement.scrollWidth` 를 직접 읽는다 (`scripts/cdp.py` 가 이미 있으니 그걸 쓴다).

Expected: `scrollWidth <= clientWidth` (=390). `over` 배열이 비어 있다.

- [ ] **Step 3: 넘치면 고친다**

원인별 처방:

| 원인 | 처방 |
|---|---|
| SVG 가 고정 폭 | `.re-chart svg { width: 100%; height: auto }` 가 먹는지 확인 |
| 지역 칩이 한 줄로 | `.re-picks { flex-wrap: wrap }` 확인 |
| 긴 범례 텍스트 | `.re-chart-legend { flex-wrap: wrap }` 확인 |
| 탭 3개가 안 접힘 | `.re-tabs` 에 `flex-wrap: wrap` 추가 |

고친 뒤 Step 1~2 를 다시 돌린다. **넘침이 0 이 될 때까지 반복한다.**

- [ ] **Step 4: 다른 페이지가 안 깨졌는지 확인한다**

`charts.js` 를 고쳤으므로 학군 페이지를 같은 방법으로 잰다.

Run: Step 2 를 `http://localhost:8899/dashboard/real-estate/schools/` 로 반복
Expected: `scrollWidth <= 390`, 진학률 추이 차트가 그대로 그려진다

- [ ] **Step 5: 전체 테스트를 돌린다**

Run: `python3 -m unittest discover -s tests`
Expected: 전부 통과

- [ ] **Step 6: 키가 섞이지 않았는지 확인한다**

```bash
git grep -I -l "$(python3 -c "
import sys; sys.path.insert(0,'scripts')
import ecos_api
print(ecos_api.load_key()[:12])
")" || echo "키 유출 없음"
```
Expected: `키 유출 없음`

- [ ] **Step 7: `CLAUDE.md` 의 파이프라인 표에 한 줄 넣는다**

`## 부동산 데이터 파이프라인` 의 표에 추가한다.

```markdown
| `/dashboard/supply/` | 착공과 금리 — 시도별 아파트 착공 평년 지수 · 기준금리 · 주담대 |
```

크론 절의 표에도 추가한다.

```
20 5 5,25 * * supply_monthly.sh    착공·금리
```

- [ ] **Step 8: 커밋**

```bash
rm -rf _site_check
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add CLAUDE.md assets/realestate/supply.css
git commit -m "Verify supply dashboard at 390px and document the pipeline

모바일 오버플로 검증 통과. CLAUDE.md 파이프라인·크론 표에 등록.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: 사용자에게 push 를 물어본다**

`CLAUDE.md` 규칙상 push 는 명시 요청이 있을 때만 한다. 커밋 목록을 보여주고 push 여부를 묻는다.

```bash
git log --oneline origin/gh-pages..HEAD
```

---

## Self-Review

**스펙 커버리지**

| 스펙 항목 | 태스크 |
|---|---|
| 통계누리 착공 파싱 (함정 1·2·3·5) | 1 |
| ECOS 두 계열 파싱 + 키 로더 | 2 |
| 60개월 청크·잠정월 재수집·원본 보존 | 3 |
| 전남광주 합산 (함정 4) | 4 |
| 12개월 이동합계·평년 지수·유효 개월 문턱 | 4 |
| 전국 직접 합산 + `총계` 1% 대조 | 4 |
| 2단 패널·x축 공유·이중축 회피 | 5, 6 |
| 지역 4개 제한 (팔레트 검증 한계) | 6 |
| 잠정 구간 점선 | 5, 6 |
| 화면에 명시할 한계 6줄 | 6 |
| 크론 월 2회 + `flock` 공유 | 7 |
| `check_freshness` 등록 (빌드 40일 / 지연 2개월) | 7 |
| 허브 카드 | 7 |
| 390px 검증 | 8 |
| 키 유출 검사 | 8 |

빠진 항목 없음.

**주의: 스펙과 한 곳 다르다**

스펙은 `parse_starts` 를 2-튜플로 적었지만 계획은 3-튜플이다 (`(rows, totals, error)`). 스펙이 요구한 "`총계` 행과 대조" 를 하려면 총계를 실어 날라야 해서 세분화했다. Task 1 을 시작하기 전에 스펙의 해당 줄을 3-튜플로 고칠 것.

**타입 일관성**

- `parse_starts` 의 행 키 `month`/`region`/`units`/`provisional` 를 Task 3 이 그대로 읽는다 ✓
- `parse_series` 가 `list[tuple[str, float]]` 를 주고 Task 3 이 `dict(points)` 로 바꿔 저장 ✓
- `supply.json` 의 `regions[].{code,name,units,mavg,index}` 를 Task 6 이 `byName()` 으로 읽는다 ✓
- `multiLineChart` 의 새 옵션 이름 `yMin`/`xTicks`/`partialFrom` 이 Task 5 정의와 Task 6 호출에서 일치 ✓
- `LIMITS.supply` 가 Task 7 에서 정의되고 Task 6 의 `showStale` 이 쓴다 — **Task 6 이 Task 7 보다 먼저 실행되면 `LIMITS.supply` 가 `undefined`** 다. `showStale` 은 `undefined` 를 받으면 나이 비교가 `false` 가 되어 항상 경고를 띄운다. Task 6 Step 5 의 헤드리스 확인에서 경고 문단이 보여도 정상이며, Task 7 Step 3 이후 사라진다.
