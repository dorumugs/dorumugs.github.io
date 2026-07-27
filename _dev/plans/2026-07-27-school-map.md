# 학군 지도 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 서울·경기 사립초와 상위권 중학교를 지도에 올리고, 학교를 누르면 그 법정동의 아파트 실거래 시세를 보여준다.

**Architecture:** 학교 위치 API 로 받은 위경도를 실거래 지도(`map.svg`)와 **완전히 같은 투영**으로 SVG 좌표로 바꿔 한 파일(`schools.json`)로 굽는다. 프론트는 기존 `map.svg`·`map.js`·`palette.js` 를 재사용하고 점 레이어와 패널만 새로 만든다. 걸러낸 학교가 255개뿐이라 구별 lazy load 없이 첫 화면에 전부 뿌린다.

**Tech Stack:** Python 3.12 표준 라이브러리, ES 모듈 자바스크립트, SVG, Jekyll + minimal-mistakes

**설계 문서:** `_dev/specs/2026-07-27-school-map-design.md`

## Global Constraints

- **파이썬은 표준 라이브러리만.** pandas·numpy·requests 금지. 기존 `scripts/*.py` 와 같은 제약이다.
- **테스트는 stdlib `unittest`.** 이 환경에 pytest 가 없다. 실행은 `python3 -m unittest discover -s tests -v`.
- **새 파이썬 파일은** `from __future__ import annotations` 로 시작하고, 함수에 타입 힌트를 달고, docstring·주석을 한국어로 쓴다.
- **자바스크립트 외부 라이브러리 금지.** CDN·npm 모두 사용하지 않는다. 플레인 ES 모듈.
- **라이트 모드 전용.** 사이트 스킨이 `default` 다. 다크모드 스타일을 넣지 않는다.
- **색은 `assets/realestate/palette.js` 에서만 온다.** 새 hex 리터럴을 어디에도 쓰지 않는다. 학군 CSS 는 `dashboard.css` 의 `--re-*` 커스텀 프로퍼티를 쓴다.
- **CSS 선택자는 전부 `.re-` 접두사.** 다른 글에 새지 않게 한다.
- **결정론적 출력.** 같은 입력에 대해 바이트가 동일해야 한다. JSON 은 `sort_keys=True`, `separators=(",", ":")`, 좌표는 소수점 한 자리 고정.
- **390px 가로 스크롤 금지.** `_dev/tools/check_mobile.py` 의 `[click]`·`[tooltip]` 양쪽 합격.
- **테마 원본 수정 금지.** `_layouts/`, `_sass/`, `assets/css/`, 기존 `_includes/` 파일, `docs/`, `CHANGELOG.md`, `_config.yml`, `Gemfile`, `package.json`, `Rakefile`. `_includes/realestate/`, `assets/realestate/` 하위에 새 파일은 허용.
- **실거래 대시보드의 동작이 바뀌면 안 된다.** `map.svg` 는 재생성해도 **바이트가 동일해야** 하고, `/real-estate/trades/` 화면은 그대로여야 한다.
- **API 키**는 환경변수 `DATA_GO_KR_API_KEY` 우선, 없으면 `~/.claude.json` 에서 읽는다 (`collect_trades.load_api_key()` 재사용). **키를 저장소에 커밋하지 않는다.**
- **커밋 신원**은 매 커밋마다 환경변수로 지정한다 (전역 `git config` 를 바꾸지 않는다):
  ```bash
  export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
  export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
  ```
- **커밋 메시지**: 영문 한 줄 요약 + 빈 줄 + 한국어 본문 1~2줄 + 빈 줄 + `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **푸시 금지.** 사용자가 명시적으로 요청할 때만 푸시한다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `scripts/schools_api.py` | 학교 위치 API 호출·페이지네이션·필드 정규화. I/O 경계 |
| `scripts/collect_schools.py` | 전수 수집 → 서울·경기 초·중만 `data/schools.csv.gz` |
| `scripts/build_schools.py` | 필터 + 투영 + 법정동 조인 → `assets/realestate/schools.json` |
| `scripts/build_geo.py` | **수정**: 투영 파라미터를 `data/geo/projection.json` 으로 내보냄 |
| `tests/test_schools_api.py` | 응답 파싱·페이지네이션·오류코드 분기 검증 |
| `tests/test_build_schools.py` | 필터·투영·조인·결정론 검증 |
| `assets/realestate/schoolmap.js` | 점 레이어 (그리기·필터·선택·툴팁) |
| `assets/realestate/schools-app.js` | 상태·URL 파라미터·배선 |
| `assets/realestate/schools.css` | 학군 페이지 전용 스타일 |
| `assets/realestate/map.js` | **수정**: `interactive: false` 옵션 추가 |
| `_pages/real-estate-schools.md` | 페이지 |
| `assets/images/real-estate-schools/header.svg` | 표지 배너 |

### 1단계와 2단계

설계 문서대로 두 단계로 낸다. **이 계획서는 1단계(사립초)를 끝까지 다룬다.**

- **1단계 (Task 1~6)**: 사립초 41개. data.go.kr 키만 필요하고 **이미 확보돼 있다**
- **2단계**: 중학교 214개. 학교알리미 키가 나온 뒤 별도 계획서로 진행

2단계를 위해 데이터 포맷·필터 함수·점 레이어를 **학교급을 파라미터로 받게** 짜 둔다.
중학교를 켤 때 새로 만드는 것은 진학 현황 수집기 하나여야 한다.

---

## Task 1: 투영 파라미터 내보내기

**Files:**
- Modify: `scripts/build_geo.py`
- Modify: `tests/test_geo.py`
- Generates: `data/geo/projection.json`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `projection_params(rings: dict[str, list[Ring]], width: float) -> dict` — `{"min_lon","max_lat","k","span_x","span_y","width","height"}`
  - `data/geo/projection.json` — 위 딕셔너리를 결정론적 JSON 으로

지금 `project()` 는 파라미터를 내부에서 계산하고 버린다. 학교 점을 같은 좌표계에
올리려면 그 숫자가 필요하다. **파라미터 계산을 별도 함수로 빼고 `project()` 가
그것을 쓰게 한다** — 두 곳에서 각자 계산하면 조용히 어긋난다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_geo.py` 의 `class TestProject` **바로 앞에** 아래를 추가한다.

```python
class TestProjectionParams(unittest.TestCase):
    RING = [[126.0, 37.0], [127.0, 37.0], [127.0, 38.0], [126.0, 37.0]]

    def test_returns_every_key_the_consumer_needs(self) -> None:
        p = build_geo.projection_params({"11680": [self.RING]}, 1000.0)
        self.assertEqual(
            sorted(p),
            ["height", "k", "max_lat", "min_lon", "span_x", "span_y", "width"],
        )

    def test_matches_project_output(self) -> None:
        """같은 점을 params 로 직접 변환한 값과 project() 결과가 같아야 한다."""
        rings = {"11680": [self.RING]}
        projected, _, _ = build_geo.project(rings, 1000.0)
        p = build_geo.projection_params(rings, 1000.0)
        lon, lat = self.RING[1]
        x = (lon - p["min_lon"]) * p["k"] / p["span_x"] * p["width"]
        y = (p["max_lat"] - lat) / p["span_y"] * p["height"]
        got_x, got_y = projected["11680"][0][1]
        self.assertAlmostEqual(x, got_x, places=9)
        self.assertAlmostEqual(y, got_y, places=9)

    def test_height_follows_aspect_ratio(self) -> None:
        p = build_geo.projection_params({"x": [self.RING]}, 1000.0)
        self.assertAlmostEqual(p["height"], p["width"] * p["span_y"] / p["span_x"], places=9)
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_geo.TestProjectionParams -v`
Expected: FAIL — `AttributeError: module 'build_geo' has no attribute 'projection_params'`

- [ ] **Step 3: `projection_params` 를 추가하고 `project()` 가 쓰게 한다**

`scripts/build_geo.py` 의 `def project(` **바로 앞에** 넣는다.

```python
def projection_params(rings: dict[str, list[Ring]], width: float) -> dict[str, float]:
    """투영에 필요한 상수를 계산한다.

    학교 점처럼 나중에 같은 지도 위에 올릴 좌표가 이 값을 그대로 써야 한다.
    두 곳에서 각자 계산하면 경계 데이터나 --eps 를 갱신할 때 조용히 어긋난다.
    """
    pts = [pt for rs in rings.values() for r in rs for pt in r]
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    k = math.cos(math.radians((min_lat + max_lat) / 2))
    span_x = (max_lon - min_lon) * k
    span_y = max_lat - min_lat
    return {
        "min_lon": min_lon,
        "max_lat": max_lat,
        "k": k,
        "span_x": span_x,
        "span_y": span_y,
        "width": width,
        "height": width * span_y / span_x,
    }
```

그리고 기존 `project()` 의 본문을 아래로 **통째로 바꾼다** (시그니처·docstring 유지).

```python
def project(rings: dict[str, list[Ring]], width: float) -> tuple[dict[str, list[list[Point]]], float, float]:
    """등장방형 투영 + 위도 보정. 전체가 width 에 꽉 차도록 맞춘다.

    이 정도 면적(서울·경기)에서는 왜곡이 눈에 띄지 않아 별도 라이브러리가 필요 없다.
    y 는 위가 북쪽이 되도록 뒤집는다.
    """
    p = projection_params(rings, width)

    def to_xy(pt: list[float]) -> Point:
        return (
            (pt[0] - p["min_lon"]) * p["k"] / p["span_x"] * p["width"],
            (p["max_lat"] - pt[1]) / p["span_y"] * p["height"],
        )

    out = {code: [[to_xy(pt) for pt in ring] for ring in rs] for code, rs in rings.items()}
    return out, p["width"], p["height"]
```

- [ ] **Step 4: `main()` 이 파라미터 파일을 쓰게 한다**

`scripts/build_geo.py` 상단의 `SVG_FILE = ...` 아래에 추가한다.

```python
PROJECTION_FILE = GEO_DIR / "projection.json"
```

`main()` 의 `projected, w, h = project(dissolved, SVG_WIDTH)` **바로 아래**에 넣는다.

```python
    params = projection_params(dissolved, SVG_WIDTH)
```

그리고 `SVG_FILE.write_text(svg, encoding="utf-8")` **바로 아래**에 넣는다.

```python
    PROJECTION_FILE.write_text(
        json.dumps(params, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS 전부

- [ ] **Step 6: 재생성하고 `map.svg` 바이트가 그대로인지 확인한다**

이게 이 태스크의 핵심 회귀 검사다. 리팩터링이 지도를 바꾸면 안 된다.

```bash
md5sum _includes/realestate/map.svg > /tmp/map-before.md5
ls /tmp/geo/hjd.geojson || curl -sL --max-time 300 -o /tmp/geo/hjd.geojson \
  "https://raw.githubusercontent.com/vuski/admdongkor/master/ver20260701/HangJeongDong_ver20260701.geojson"
python3 scripts/build_geo.py --input /tmp/geo/hjd.geojson
md5sum -c /tmp/map-before.md5 && echo "SVG 동일 ✔"
cat data/geo/projection.json
```

Expected: `SVG 동일 ✔` 그리고 `projection.json` 에 7개 키가 보인다.
`--eps` 는 기본값(0.05)을 쓴다 — 커밋된 SVG 가 그 값으로 만들어졌다.

- [ ] **Step 7: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_geo.py tests/test_geo.py data/geo/projection.json
git commit -m "Export map projection parameters for reuse

학교 점을 같은 지도에 올리려면 투영 상수가 필요하다. 계산을 함수로 빼고
projection.json 으로 내보낸다. map.svg 는 바이트가 그대로다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: 학교 위치 API 클라이언트

**Files:**
- Create: `scripts/schools_api.py`
- Create: `tests/test_schools_api.py`

**Interfaces:**
- Consumes: `collect_trades.load_api_key() -> str` (기존)
- Produces:
  - `ENDPOINT = "https://api.data.go.kr/openapi/tn_pubr_public_elesch_mskul_lc_api"`
  - `COLUMNS: list[str]` — CSV 컬럼 순서
  - `parse_response(payload: dict) -> tuple[list[dict], int]` — (행 목록, totalCount)
  - `ApiError(Exception)` — `code: str`, `message: str` 속성
  - `normalize(item: dict) -> dict | None` — 원본 항목 → COLUMNS 딕셔너리. 서울·경기 아니거나 좌표 없으면 None
  - `fetch_page(key: str, page: int, rows: int = 1000, retries: int = 3) -> tuple[list[dict], int]`

**API 응답 형태 (실측)**

```json
{"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
              "body": {"totalCount": 12011, "items": [{...}]}}}
```

항목 필드: `schoolId, schoolNm, schoolSe, fondDate, fondType, bnhhSe, operSttus,
lnmadr, rdnmadr, cddcCode, cddcNm, edcSport, edcSportNm, creatDate, changeDate,
latitude, longitude, referenceDate, insttCode, insttNm`

오류 코드: `00` 정상, `30` 키 미등록(활용신청 필요), `12` 서비스 없음.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_schools_api.py`:

```python
"""학교 위치 API 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import schools_api  # noqa: E402


def _item(**kw) -> dict:
    base = {
        "schoolId": "B000008352", "schoolNm": "세화여자중학교", "schoolSe": "중학교",
        "fondType": "사립", "bnhhSe": "본교", "operSttus": "운영",
        "lnmadr": "서울특별시 서초구 반포동 753",
        "rdnmadr": "서울특별시 서초구 신반포로 56-7",
        "cddcNm": "서울특별시교육청", "latitude": "37.5019828", "longitude": "126.994230",
    }
    base.update(kw)
    return base


def _payload(items: list[dict], total: int) -> dict:
    return {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                         "body": {"totalCount": total, "items": items}}}


class TestParseResponse(unittest.TestCase):
    def test_returns_items_and_total(self) -> None:
        rows, total = schools_api.parse_response(_payload([_item()], 12011))
        self.assertEqual(total, 12011)
        self.assertEqual(len(rows), 1)

    def test_empty_items_is_not_an_error(self) -> None:
        rows, total = schools_api.parse_response(_payload([], 0))
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_missing_items_key_treated_as_empty(self) -> None:
        payload = {"response": {"header": {"resultCode": "00"}, "body": {"totalCount": 0}}}
        rows, _ = schools_api.parse_response(payload)
        self.assertEqual(rows, [])

    def test_unregistered_key_raises_with_code(self) -> None:
        payload = {"response": {"header": {"resultCode": "30",
                                           "resultMsg": "SERVICE KEY IS NOT REGISTERED ERROR."}}}
        with self.assertRaises(schools_api.ApiError) as ctx:
            schools_api.parse_response(payload)
        self.assertEqual(ctx.exception.code, "30")

    def test_missing_service_raises_with_code(self) -> None:
        payload = {"response": {"header": {"resultCode": "12",
                                           "resultMsg": "NO OPENAPI SERVICE ERROR."}}}
        with self.assertRaises(schools_api.ApiError) as ctx:
            schools_api.parse_response(payload)
        self.assertEqual(ctx.exception.code, "12")


class TestNormalize(unittest.TestCase):
    def test_keeps_seoul(self) -> None:
        got = schools_api.normalize(_item())
        self.assertEqual(got["school_name"], "세화여자중학교")
        self.assertEqual(got["level"], "중학교")
        self.assertEqual(got["found_type"], "사립")
        self.assertEqual(got["lat"], "37.5019828")

    def test_keeps_gyeonggi(self) -> None:
        got = schools_api.normalize(_item(lnmadr="경기도 성남시 분당구 서현동 100"))
        self.assertIsNotNone(got)

    def test_drops_other_region(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(lnmadr="부산광역시 해운대구 우동 1")))

    def test_drops_missing_coordinates(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(latitude="")))
        self.assertIsNone(schools_api.normalize(_item(longitude=None)))

    def test_drops_high_school(self) -> None:
        """고등학교는 이 도구의 범위 밖이라 수집 단계에서 버린다."""
        self.assertIsNone(schools_api.normalize(_item(schoolSe="고등학교")))

    def test_drops_closed_school(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(operSttus="폐교")))

    def test_drops_branch_school(self) -> None:
        self.assertIsNone(schools_api.normalize(_item(bnhhSe="분교")))

    def test_every_column_present(self) -> None:
        got = schools_api.normalize(_item())
        self.assertEqual(sorted(got), sorted(schools_api.COLUMNS))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_schools_api -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schools_api'`

- [ ] **Step 3: `scripts/schools_api.py` 를 구현한다**

```python
"""전국초중등학교위치표준데이터 API 클라이언트.

    https://api.data.go.kr/openapi/tn_pubr_public_elesch_mskul_lc_api

순수 파싱 함수와 네트워크 I/O 를 분리해 두어 파싱은 단독으로 검증할 수 있다.
전국 12,011건을 numOfRows=1000 으로 13번 부르면 다 온다.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://api.data.go.kr/openapi/tn_pubr_public_elesch_mskul_lc_api"

# 이 도구가 다루는 학교급. 고등학교는 배정·광역모집이라 '근처'가 성립하지 않아 뺀다.
LEVELS = ("초등학교", "중학교")
REGIONS = ("서울특별시", "경기도")

COLUMNS = [
    "school_id", "school_name", "level", "found_type",
    "addr", "road_addr", "sido_office", "lat", "lon",
]


class ApiError(Exception):
    """resultCode 가 정상(00)이 아닐 때. code 로 원인을 구분한다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def parse_response(payload: dict) -> tuple[list[dict], int]:
    """응답 본문에서 (항목 목록, 전체 건수) 를 꺼낸다. 오류면 ApiError."""
    resp = payload.get("response") or {}
    header = resp.get("header") or {}
    code = str(header.get("resultCode", "")).strip()
    if code != "00":
        raise ApiError(code, str(header.get("resultMsg", "")).strip())
    body = resp.get("body") or {}
    return list(body.get("items") or []), int(body.get("totalCount") or 0)


def normalize(item: dict) -> dict | None:
    """원본 항목을 CSV 한 행으로. 범위 밖이면 None.

    버리는 것: 서울·경기 밖, 고등학교, 분교, 운영 중이 아닌 학교, 좌표 결측.
    """
    addr = (item.get("lnmadr") or "").strip()
    if not addr.startswith(REGIONS):
        return None
    if (item.get("schoolSe") or "").strip() not in LEVELS:
        return None
    if (item.get("bnhhSe") or "").strip() not in ("", "본교"):
        return None
    if (item.get("operSttus") or "").strip() not in ("", "운영"):
        return None
    lat = (item.get("latitude") or "").strip()
    lon = (item.get("longitude") or "").strip()
    if not lat or not lon:
        return None
    return {
        "school_id": (item.get("schoolId") or "").strip(),
        "school_name": (item.get("schoolNm") or "").strip(),
        "level": (item.get("schoolSe") or "").strip(),
        "found_type": (item.get("fondType") or "").strip(),
        "addr": addr,
        "road_addr": (item.get("rdnmadr") or "").strip(),
        "sido_office": (item.get("cddcNm") or "").strip(),
        "lat": lat,
        "lon": lon,
    }


def fetch_page(key: str, page: int, rows: int = 1000, retries: int = 3) -> tuple[list[dict], int]:
    """한 페이지를 받는다. 네트워크 오류는 지수 백오프로 재시도한다."""
    query = urllib.parse.urlencode(
        {"serviceKey": key, "type": "json", "pageNo": page, "numOfRows": rows})
    url = f"{ENDPOINT}?{query}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                payload = json.loads(res.read().decode("utf-8"))
            return parse_response(payload)
        except ApiError:
            raise  # 키·서비스 문제는 재시도해도 소용없다
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"{retries}회 재시도 후에도 실패했습니다: {last}")
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS 전부

- [ ] **Step 5: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/schools_api.py tests/test_schools_api.py
git commit -m "Add school location API client

응답 파싱과 서울·경기 초중 필터를 네트워크 I/O 와 분리했다.
키 미등록(30)과 서비스 없음(12)을 구분해 올린다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 학교 데이터 수집

**Files:**
- Create: `scripts/collect_schools.py`
- Generates: `data/schools.csv.gz`

**Interfaces:**
- Consumes: `schools_api.{ENDPOINT, COLUMNS, ApiError, fetch_page, normalize}` (Task 2), `collect_trades.load_api_key()`, `rtms.gzip_bytes()`
- Produces: `data/schools.csv.gz` — `COLUMNS` 순서의 CSV, gzip

`rtms.rows_to_csv()` 는 **실거래 컬럼에 고정**돼 있어 쓸 수 없다. CSV 쓰기는 이
파일 안에서 `csv.DictWriter` 로 직접 한다. `rtms.gzip_bytes()` 는 결정론적
gzip 이라 그대로 쓴다.

- [ ] **Step 1: `scripts/collect_schools.py` 를 구현한다**

```python
"""학교 위치 데이터를 받아 서울·경기 초·중만 저장한다.

    python3 scripts/collect_schools.py

전국 12,011건을 13번에 나눠 받아 필터한 뒤 data/schools.csv.gz 로 쓴다.
실거래처럼 매일 돌릴 필요는 없다 — 학교는 자주 바뀌지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_trades  # noqa: E402
import rtms  # noqa: E402
import schools_api  # noqa: E402

OUT_FILE = ROOT / "data" / "schools.csv.gz"


def rows_to_csv(rows: list[dict]) -> str:
    """schools_api.COLUMNS 순서로 쓴다. rtms.rows_to_csv 는 실거래 컬럼 고정이라 못 쓴다."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=schools_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in schools_api.COLUMNS})
    return buf.getvalue()


def collect(key: str, page_size: int = 1000) -> list[dict]:
    """전수를 받아 서울·경기 초·중만 남긴다. 정렬해 결정론을 지킨다."""
    kept: list[dict] = []
    seen = 0
    page = 1
    while True:
        items, total = schools_api.fetch_page(key, page, page_size)
        if not items:
            break
        seen += len(items)
        for item in items:
            row = schools_api.normalize(item)
            if row is not None:
                kept.append(row)
        print(f"  {seen:,}/{total:,} 수신, 대상 {len(kept):,}건")
        if seen >= total:
            break
        page += 1
    kept.sort(key=lambda r: (r["school_id"], r["school_name"]))
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()

    key = collect_trades.load_api_key()
    try:
        rows = collect(key, args.page_size)
    except schools_api.ApiError as exc:
        if exc.code == "30":
            print("이 키로는 학교 위치 데이터를 쓸 수 없습니다. data.go.kr 에서 "
                  "'전국초중등학교위치표준데이터' 활용신청을 하세요.", file=sys.stderr)
        elif exc.code == "12":
            print(f"엔드포인트가 존재하지 않습니다: {schools_api.ENDPOINT}", file=sys.stderr)
        else:
            print(f"API 오류: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("대상 학교가 0건입니다. 필터가 과한지 확인하세요.", file=sys.stderr)
        return 1

    levels: dict[str, int] = {}
    for row in rows:
        levels[row["level"]] = levels.get(row["level"], 0) + 1
    print(f"서울·경기 대상 {len(rows):,}건 " + " / ".join(f"{k} {v:,}" for k, v in sorted(levels.items())))

    data = rtms.gzip_bytes(rows_to_csv(rows))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        print("변경 없음")
        return 0
    OUT_FILE.write_bytes(data)
    print(f"{OUT_FILE.name} 갱신 ({len(data):,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 실제로 수집한다**

Run: `python3 scripts/collect_schools.py`

Expected: 13페이지를 받고 마지막에

```
서울·경기 대상 2,943건 중학교 1,072 / 초등학교 1,871
```

정확한 숫자는 분교·폐교 필터 때문에 실측(초 1,985 / 중 1,072)보다 조금 적을 수
있다. **초등학교가 1,800건 이상, 중학교가 1,000건 이상이면 정상이다.** 그보다
훨씬 적으면 필터가 과한 것이니 `normalize()` 의 조건을 다시 본다.

- [ ] **Step 3: 결정론을 확인한다**

```bash
python3 scripts/collect_schools.py
python3 scripts/collect_schools.py
```

Expected: 두 번째 실행이 `변경 없음` 을 출력한다.

- [ ] **Step 4: 사립초가 실제로 들어있는지 확인한다**

```bash
python3 - <<'PY'
import sys, collections
sys.path.insert(0, 'scripts')
import rtms
from pathlib import Path
rows = rtms.csv_to_rows(rtms.gunzip_text(Path('data/schools.csv.gz').read_bytes()))
c = collections.Counter((r['level'], r['found_type']) for r in rows)
for k, v in sorted(c.items()):
    print(k, v)
pri = [r for r in rows if r['level'] == '초등학교' and r['found_type'] == '사립']
print('사립초', len(pri))
for r in pri[:3]:
    print(' ', r['school_name'], r['addr'], r['lat'], r['lon'])
PY
```

Expected: 사립초가 **35~41건** 나오고 각 행에 주소와 좌표가 있다.

- [ ] **Step 5: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/collect_schools.py data/schools.csv.gz
git commit -m "Collect Seoul/Gyeonggi elementary and middle school locations

전국 12,011건에서 서울·경기 초·중 본교만 남겨 저장한다.
분교·폐교·좌표 결측은 수집 단계에서 버린다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 학교 집계 빌드

**Files:**
- Create: `scripts/build_schools.py`
- Create: `tests/test_build_schools.py`
- Generates: `assets/realestate/schools.json`

**Interfaces:**
- Consumes: `data/schools.csv.gz` (Task 3), `data/geo/projection.json` (Task 1), `regions.dong_code()`, `rtms.{csv_to_rows, gunzip_text}`, `build_dashboard.write_json()`
- Produces:
  - `to_svg_xy(lat: float, lon: float, params: dict) -> tuple[float, float]`
  - `parse_addr(addr: str) -> tuple[str, str] | None` — 지번주소 → (시군구명, 읍면동명)
  - `select_private_elementary(rows: list[dict]) -> list[dict]`
  - `build(rows, params, sgg_by_name, generated) -> dict`

**핵심 난관: 주소에서 시군구 코드 찾기**

`regions.dong_code(sgg_cd, umd_nm)` 는 **시군구 5자리 코드**를 받는데, 학교
주소에는 이름만 있다 (`서울특별시 강남구 대치동 123`). `regions.sgg_codes()` 가
주는 `(코드, 정식이름)` 을 뒤집어 이름→코드 표를 만든다.

주의할 것:

- 서울은 `서울특별시 강남구` — 두 토큰
- 경기 구 있는 시는 `경기도 성남시 분당구` — 세 토큰
- 경기 구 없는 시는 `경기도 광명시` — 두 토큰

**긴 이름부터 맞춰야 한다.** `경기도 성남시` 로 먼저 맞으면 분당구를 놓친다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_build_schools.py`:

```python
"""학교 집계 빌드 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_schools  # noqa: E402

PARAMS = {"min_lon": 126.0, "max_lat": 38.0, "k": 0.8,
          "span_x": 1.6, "span_y": 1.2, "width": 1000.0, "height": 750.0}


def _school(**kw) -> dict:
    base = {
        "school_id": "S1", "school_name": "계성초등학교", "level": "초등학교",
        "found_type": "사립", "addr": "서울특별시 서초구 내곡동 1",
        "road_addr": "", "sido_office": "서울특별시교육청",
        "lat": "37.5", "lon": "127.0",
    }
    base.update(kw)
    return base


class TestToSvgXy(unittest.TestCase):
    def test_matches_projection_formula(self) -> None:
        x, y = build_schools.to_svg_xy(37.5, 127.0, PARAMS)
        self.assertAlmostEqual(x, (127.0 - 126.0) * 0.8 / 1.6 * 1000.0, places=6)
        self.assertAlmostEqual(y, (38.0 - 37.5) / 1.2 * 750.0, places=6)

    def test_north_is_up(self) -> None:
        _, y_north = build_schools.to_svg_xy(37.9, 127.0, PARAMS)
        _, y_south = build_schools.to_svg_xy(37.1, 127.0, PARAMS)
        self.assertLess(y_north, y_south)


class TestParseAddr(unittest.TestCase):
    def test_seoul_two_tokens(self) -> None:
        self.assertEqual(build_schools.parse_addr("서울특별시 강남구 대치동 123"),
                         ("서울특별시 강남구", "대치동"))

    def test_gyeonggi_with_gu_takes_longest_match(self) -> None:
        """'경기도 성남시' 로 먼저 맞으면 분당구를 놓친다. 긴 이름이 이겨야 한다."""
        self.assertEqual(build_schools.parse_addr("경기도 성남시 분당구 서현동 100"),
                         ("경기도 성남시 분당구", "서현동"))

    def test_gyeonggi_without_gu(self) -> None:
        self.assertEqual(build_schools.parse_addr("경기도 광명시 하안동 50"),
                         ("경기도 광명시", "하안동"))

    def test_eup_myeon_ri_keeps_full_suffix(self) -> None:
        """읍면 지역은 '가남읍 태평리' 처럼 두 토큰이다. 접미사 전체를 넘겨야 한다."""
        self.assertEqual(build_schools.parse_addr("경기도 여주시 가남읍 태평리 12-3"),
                         ("경기도 여주시", "가남읍 태평리"))

    def test_unknown_region_is_none(self) -> None:
        self.assertIsNone(build_schools.parse_addr("부산광역시 해운대구 우동 1"))


class TestSelectPrivateElementary(unittest.TestCase):
    def test_keeps_only_private_elementary(self) -> None:
        rows = [
            _school(school_id="A"),
            _school(school_id="B", found_type="공립"),
            _school(school_id="C", level="중학교", found_type="사립"),
            _school(school_id="D", found_type="국립"),
        ]
        got = build_schools.select_private_elementary(rows)
        self.assertEqual([r["school_id"] for r in got], ["A"])


class TestBuild(unittest.TestCase):
    def _run(self, rows: list[dict]) -> dict:
        return build_schools.build(rows, PARAMS, generated="2026-07-27")

    def test_emits_expected_shape(self) -> None:
        out = self._run([_school()])
        self.assertEqual(out["generated"], "2026-07-27")
        s = out["schools"][0]
        self.assertEqual(s["name"], "계성초등학교")
        self.assertEqual(s["lvl"], "초")
        self.assertEqual(s["found"], "사립")
        self.assertEqual(s["dong"], "내곡동")
        self.assertEqual(s["sgg"], "11650")
        self.assertEqual(s["dong_cd"], "1165010800")

    def test_coordinates_rounded_to_one_decimal(self) -> None:
        s = self._run([_school()])["schools"][0]
        self.assertEqual(s["x"], round(s["x"], 1))
        self.assertEqual(s["y"], round(s["y"], 1))

    def test_unresolvable_address_dropped(self) -> None:
        out = self._run([_school(addr="부산광역시 해운대구 우동 1")])
        self.assertEqual(out["schools"], [])

    def test_sorted_by_sgg_then_name(self) -> None:
        rows = [
            _school(school_id="A", school_name="나초등학교", addr="서울특별시 강남구 대치동 1"),
            _school(school_id="B", school_name="가초등학교", addr="서울특별시 강남구 대치동 2"),
            _school(school_id="C", school_name="다초등학교", addr="서울특별시 종로구 청운동 3"),
        ]
        got = [s["name"] for s in self._run(rows)["schools"]]
        self.assertEqual(got, ["가초등학교", "나초등학교", "다초등학교"])


class TestAgainstRealOutput(unittest.TestCase):
    OUT = ROOT / "assets" / "realestate" / "schools.json"

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_all_points_inside_viewbox(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        for s in data["schools"]:
            self.assertGreaterEqual(s["x"], 0.0, s["name"])
            self.assertLessEqual(s["x"], 1000.0, s["name"])
            self.assertGreaterEqual(s["y"], 0.0, s["name"])
            self.assertLessEqual(s["y"], 1201.0, s["name"])

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_only_private_elementary(self) -> None:
        data = json.loads(self.OUT.read_text(encoding="utf-8"))
        self.assertTrue(data["schools"])
        for s in data["schools"]:
            self.assertEqual(s["lvl"], "초")
            self.assertEqual(s["found"], "사립")

    @unittest.skipUnless(OUT.exists(), "schools.json 없음 — 먼저 빌드하세요")
    def test_within_size_budget(self) -> None:
        self.assertLess(self.OUT.stat().st_size, 100 * 1024)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python3 -m unittest tests.test_build_schools -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_schools'`

- [ ] **Step 3: `scripts/build_schools.py` 를 구현한다**

```python
"""학교 원본을 지도에 올릴 수 있는 집계 JSON 으로 굽는다.

    python3 scripts/build_schools.py

1단계는 사립초만 다룬다. 중학교는 학교알리미 진학 데이터가 붙는 2단계다.
좌표 변환은 build_geo.py 가 내보낸 projection.json 을 그대로 쓴다 —
파라미터를 각자 계산하면 지도와 점이 조용히 어긋난다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dashboard  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

SCHOOL_FILE = ROOT / "data" / "schools.csv.gz"
PROJECTION_FILE = ROOT / "data" / "geo" / "projection.json"
OUT_FILE = ROOT / "assets" / "realestate" / "schools.json"

MAX_BYTES = 100 * 1024
JOIN_FAIL_LIMIT = 0.10

# 학교급 표기를 화면용 한 글자로 줄인다. 2단계에서 '중학교' 가 추가된다.
LEVEL_SHORT = {"초등학교": "초", "중학교": "중"}


@lru_cache(maxsize=1)
def _sgg_by_name() -> list[tuple[str, str]]:
    """(정식이름, 코드) 를 이름 긴 순으로. 긴 이름이 먼저 맞아야 한다.

    '경기도 성남시' 가 '경기도 성남시 분당구' 보다 먼저 맞으면 구를 놓친다.
    """
    pairs = [(name, code) for code, name in regions.sgg_codes()]
    return sorted(pairs, key=lambda p: -len(p[0]))


def parse_addr(addr: str) -> tuple[str, str] | None:
    """지번주소를 (시군구 정식이름, 읍면동 접미사) 로 나눈다. 못 찾으면 None.

    읍면 지역은 '가남읍 태평리' 처럼 두 토큰이라 접미사 전체를 넘긴다 —
    regions.dong_code 가 접미사 전체를 1차 키로 쓰기 때문이다.
    마지막 토큰(지번)은 떼어낸다.
    """
    text = " ".join((addr or "").split())
    for name, _code in _sgg_by_name():
        if not text.startswith(name + " "):
            continue
        rest = text[len(name):].strip()
        tokens = rest.split(" ")
        # 마지막 토큰이 지번(숫자/숫자-숫자/산 12)이면 뗀다
        if tokens and (tokens[-1][:1].isdigit() or tokens[-1] == "산"):
            tokens = tokens[:-1]
        if tokens and tokens[-1] == "산":
            tokens = tokens[:-1]
        if not tokens:
            return None
        return name, " ".join(tokens)
    return None


def to_svg_xy(lat: float, lon: float, params: dict) -> tuple[float, float]:
    """build_geo.py 의 project() 와 같은 식. 파라미터는 projection.json 에서 온다."""
    x = (lon - params["min_lon"]) * params["k"] / params["span_x"] * params["width"]
    y = (params["max_lat"] - lat) / params["span_y"] * params["height"]
    return x, y


def select_private_elementary(rows: list[dict]) -> list[dict]:
    """1단계 대상: 사립 초등학교만. 국립은 요구사항이 '사립초' 라 넣지 않는다."""
    return [r for r in rows
            if r.get("level") == "초등학교" and r.get("found_type") == "사립"]


def build(rows: list[dict], params: dict, generated: str) -> dict:
    """선택된 학교를 지도 좌표와 법정동 코드가 붙은 형태로 만든다."""
    name_to_code = dict(_sgg_by_name())
    out: list[dict] = []
    failed: list[str] = []

    for row in rows:
        parsed = parse_addr(row["addr"])
        if parsed is None:
            failed.append(row["school_name"])
            continue
        sgg_name, umd = parsed
        sgg = name_to_code[sgg_name]
        dong_cd = regions.dong_code(sgg, umd)
        if dong_cd is None:
            failed.append(row["school_name"])
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (TypeError, ValueError):
            failed.append(row["school_name"])
            continue
        x, y = to_svg_xy(lat, lon, params)
        out.append({
            "name": row["school_name"],
            "lvl": LEVEL_SHORT[row["level"]],
            "found": row["found_type"],
            "sgg": sgg,
            "dong": umd.split(" ")[-1],
            "dong_cd": dong_cd,
            "addr": row["addr"],
            "x": round(x, 1),
            "y": round(y, 1),
        })

    # 완전 전순서. 동점이 흔들리면 재빌드마다 바이트가 달라진다.
    out.sort(key=lambda s: (s["sgg"], s["name"], s["dong_cd"]))
    return {"generated": generated, "schools": out, "join_failed": len(failed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not PROJECTION_FILE.exists():
        print(f"{PROJECTION_FILE} 가 없습니다. 먼저 build_geo.py 를 돌리세요.", file=sys.stderr)
        return 1
    if not SCHOOL_FILE.exists():
        print(f"{SCHOOL_FILE} 가 없습니다. 먼저 collect_schools.py 를 돌리세요.", file=sys.stderr)
        return 1

    params = json.loads(PROJECTION_FILE.read_text(encoding="utf-8"))
    rows = rtms.csv_to_rows(rtms.gunzip_text(SCHOOL_FILE.read_bytes()))
    picked = select_private_elementary(rows)
    print(f"원본 {len(rows):,}건 중 사립초 {len(picked):,}건")

    payload = build(picked, params, args.generated)
    failed = payload.pop("join_failed")
    if picked and failed / len(picked) > JOIN_FAIL_LIMIT:
        print(f"법정동 조인 실패 {failed}/{len(picked)} — 한도 "
              f"{JOIN_FAIL_LIMIT:.0%} 초과", file=sys.stderr)
        return 1
    if failed:
        print(f"법정동 조인 실패 {failed}건 (한도 안)")

    changed = build_dashboard.write_json(OUT_FILE, payload)
    size = OUT_FILE.stat().st_size
    print(f"schools.json {size:,}B {'갱신' if changed else '변경 없음'} "
          f"/ 학교 {len(payload['schools']):,}개")
    if size > MAX_BYTES:
        print(f"예산({MAX_BYTES}B) 초과", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 합성 테스트가 통과하는지 확인한다**

Run: `python3 -m unittest tests.test_build_schools -v`
Expected: `TestAgainstRealOutput` 3건은 skip, 나머지 PASS

- [ ] **Step 5: 실제로 빌드한다**

Run: `python3 scripts/build_schools.py`

Expected: `원본 2,9xx건 중 사립초 3x건` 그리고 `schools.json ...B 갱신 / 학교 3x개`

조인 실패가 한도를 넘으면 `parse_addr` 을 고친다. 실패한 학교 이름을 찍어보려면
`build()` 의 `failed` 리스트를 임시로 출력한다.

- [ ] **Step 6: 투영이 지도와 맞는지 눈으로 확인한다**

점이 진짜 제자리에 찍히는지가 이 태스크의 성패다. 좌표만 봐서는 알 수 없다.

```bash
python3 - <<'PY'
import json, pathlib, re
svg = pathlib.Path('_includes/realestate/map.svg').read_text(encoding='utf-8')
data = json.loads(pathlib.Path('assets/realestate/schools.json').read_text(encoding='utf-8'))
dots = "".join(
    f'<circle cx="{s["x"]}" cy="{s["y"]}" r="5" fill="#eb6834" stroke="#fff" stroke-width="1.5"/>'
    for s in data["schools"])
out = svg.replace('<path ', '<path fill="#eef1f3" stroke="#c9ced1" stroke-width="1" ')
out = out.replace('</svg>', dots + '</svg>')
pathlib.Path('/tmp/schools-check.svg').write_text(out, encoding='utf-8')
print("학교", len(data["schools"]))
PY
google-chrome --headless=new --disable-gpu --screenshot=/tmp/schools-check.png \
  --window-size=900,1100 file:///tmp/schools-check.svg
```

`/tmp/schools-check.png` 를 Read 로 열어 확인한다. 합격 기준:

- 점이 **전부 지도 안쪽**에 있다. 하나라도 바다·경계 밖이면 투영이 틀렸다
- 서울 강남·서초·종로 쪽에 점이 몰려 있다 (사립초 분포가 실제로 그렇다)
- 점이 한쪽 모서리에 뭉쳐 있으면 `min_lon`/`max_lat` 을 잘못 읽은 것이다

- [ ] **Step 7: 결정론과 실측 테스트를 확인한다**

```bash
cp assets/realestate/schools.json /tmp/schools-1.json
python3 scripts/build_schools.py --generated 2026-07-27
python3 scripts/build_schools.py --generated 2026-07-27
cmp /tmp/schools-1.json assets/realestate/schools.json 2>/dev/null || true
python3 -m unittest discover -s tests -v
```

Expected: 두 번째 실행이 `변경 없음`, 전체 테스트 PASS (skip 없이)

- [ ] **Step 8: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add scripts/build_schools.py tests/test_build_schools.py assets/realestate/schools.json
git commit -m "Build private elementary school map data

학교 위경도를 지도와 같은 투영으로 SVG 좌표로 바꾸고 법정동을 조인한다.
주소는 긴 시군구 이름부터 맞춰야 분당구 같은 하위 구를 놓치지 않는다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 지도에 점 얹기

**Files:**
- Modify: `assets/realestate/map.js`
- Create: `assets/realestate/schoolmap.js`
- Create: `assets/realestate/schools.css`
- Create: `assets/realestate/schools-app.js`
- Create: `_pages/real-estate-schools.md`
- Create: `assets/images/real-estate-schools/header.svg`

**Interfaces:**
- Consumes: `map.js` 의 `initMap(root, {onSelect}) -> handle`, `palette.js` 의 `SEQUENTIAL/INK/INK2/MUTED/NO_DATA`, `data.js` 의 `setBase`
- Produces:
  - `map.js`: `initMap(root, {onSelect, interactive = true})` — `interactive: false` 면 구 path 에 포커스·클릭을 걸지 않는다
  - `schoolmap.js`: `initSchoolLayer(root, {onSelect}) -> {render(schools), setSelected(name), clear()}`

**`map.js` 를 왜 고치는가**

`initMap` 은 72개 구 path 전부에 `tabindex="0"`, `role="button"`, `aria-label`,
클릭·키보드 핸들러를 건다. 학군 페이지에서는 구가 누를 대상이 아니라서, 그대로
쓰면 **아무 동작도 하지 않는 포커스 가능한 버튼 72개**가 생긴다. 실거래 때
"이름 없는 버튼 72개" 를 고쳤던 것과 같은 종류의 접근성 문제다.

옵션 하나를 추가해 막는다. 기본값은 `true` 라 실거래 대시보드는 그대로다.

- [ ] **Step 1: `map.js` 에 `interactive` 옵션을 추가한다**

`export function initMap(root, { onSelect }) {` 를 아래로 바꾼다.

```javascript
export function initMap(root, { onSelect, interactive = true }) {
```

그 아래 `for (const path of paths) {` 로 시작하는 블록 **전체**를 아래로 감싼다.
(블록 안 내용은 그대로 두고 조건만 씌운다.)

```javascript
  // 학군 페이지처럼 구가 누를 대상이 아닌 화면에서는 포커스·클릭을 걸지 않는다.
  // 걸면 아무 동작도 하지 않는 포커스 가능한 버튼이 72개 생긴다.
  if (interactive) {
    for (const path of paths) {
      // ... 기존 내용 그대로 ...
    }
  }
```

- [ ] **Step 2: 실거래 대시보드가 그대로인지 확인한다**

Docker 로 빌드해 `/real-estate/trades/` 에서 구를 눌러본다.

```bash
cat > _tmp_Gemfile <<'EOF'
source "https://rubygems.org"
gemspec
gem "jekyll-sass-converter", "~> 2.0"
gem "webrick"
EOF
docker run --rm -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll jekyll/jekyll:4.2.2 \
  sh -c "bundle install >/dev/null 2>&1 && jekyll build"
rm -f _tmp_Gemfile _tmp_Gemfile.lock
docker run -d --name kayser_serve -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll -p 4000:4000 jekyll/jekyll:4.2.2 \
  sh -c "bundle install && jekyll serve --host 0.0.0.0 --skip-initial-build --no-watch" 2>/dev/null || true
sleep 20
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:4000/real-estate/trades/
```

컨테이너가 이미 떠 있으면 `docker exec` 로 재빌드해도 된다.
`_tmp_Gemfile` 은 **커밋하지 않는다.**

- [ ] **Step 3: 점 레이어를 만든다**

`assets/realestate/schoolmap.js`:

```javascript
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
```

- [ ] **Step 4: 스타일을 만든다**

`assets/realestate/schools.css`:

```css
/* 학군 지도 전용. 색은 dashboard.css 의 --re-* 와 palette.js 를 따른다. */

.re-school-layer .re-dot {
  cursor: pointer;
  stroke: #fff;
  stroke-width: 1.5;
  transition: r 100ms linear;
}

/* 학교급 색은 palette.js 의 범주형 앞 두 칸과 같은 값이다.
   초=blue slot 1, 중=orange slot 2. 2종이라 모든 대비 판정에 여유가 있다. */
.re-school-layer .re-dot.is-초 { fill: #2a78d6; }
.re-school-layer .re-dot.is-중 { fill: #eb6834; }

.re-school-layer .re-dot:hover,
.re-school-layer .re-dot:focus-visible {
  r: 7;
  outline: none;
}

.re-school-layer .re-dot.is-selected {
  r: 8;
  stroke: var(--re-ink);
  stroke-width: 2.5;
}

/* 구는 누를 대상이 아니라 배경이다. 점이 묻히지 않게 옅게 깐다. */
.re-app.is-schools svg.re-map path { cursor: default; }
.re-app.is-schools svg.re-map path:hover { stroke: rgba(255, 255, 255, 0.85); stroke-width: 1; }

.re-legend-dots { display: flex; gap: 14px; align-items: center; font-size: 12px; color: var(--re-ink2); }
.re-legend-dots i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
.re-legend-dots .is-초 { background: #2a78d6; }
.re-legend-dots .is-중 { background: #eb6834; }

.re-caveat {
  font-size: 12px;
  color: var(--re-ink2);
  background: rgba(61, 65, 68, 0.05);
  border-radius: 6px;
  padding: 8px 11px;
  margin: 10px 0 0;
  line-height: 1.6;
}

.re-school-meta { font-size: 13px; color: var(--re-ink2); margin: 0 0 12px; }
```

**주의**: 위 CSS 의 `#2a78d6` / `#eb6834` 는 `palette.js` 의 범주형 슬롯 1·2 와
같은 값이다. CSS 에서 JS 모듈을 import 할 수 없어 불가피하게 적는다.
**두 곳이 어긋나지 않게 주석으로 묶어 뒀다** — 한쪽을 바꾸면 다른 쪽도 바꾼다.

- [ ] **Step 5: 표지 배너를 만든다**

`assets/images/real-estate-schools/header.svg` — 저장소 표준 양식이다.
`.claude/skills/check-headers/` 의 지문(1200×420, `#0e1726→#1b2942` 배경,
상단 accent 바, 한글 폰트 폴백, `role="img"`, `KayserDocs` 푸터)을 지켜야 한다.

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420" role="img" aria-label="서울·경기 학군 지도 배너">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0e1726"/>
      <stop offset="1" stop-color="#1b2942"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#2a78d6"/>
      <stop offset="1" stop-color="#eb6834"/>
    </linearGradient>
    <style>.ko { font-family: 'Apple SD Gothic Neo','Noto Sans KR','Malgun Gothic',sans-serif; }</style>
  </defs>
  <rect width="1200" height="420" fill="url(#bg)"/>
  <rect x="0" y="0" width="1200" height="6" fill="url(#accent)"/>
  <text class="ko" x="80" y="112" font-size="46" font-weight="800" fill="#f8fafc">학군 <tspan fill="#86b6ef">지도</tspan></text>
  <text class="ko" x="82" y="156" font-size="23" font-weight="500" fill="#94a3b8">학교를 누르면 그 동네 아파트가 얼마인지 봅니다</text>
  <g class="ko" text-anchor="middle">
    <rect x="90" y="236" width="250" height="118" rx="16" fill="#0e2233" stroke="#2a78d6" stroke-width="3"/>
    <text x="215" y="288" font-size="24" font-weight="800" fill="#86b6ef">사립초</text>
    <text x="215" y="320" font-size="16" font-weight="500" fill="#cbd5e1">서울·경기 41곳</text>
    <rect x="475" y="236" width="250" height="118" rx="16" fill="#2b1c14" stroke="#eb6834" stroke-width="3"/>
    <text x="600" y="288" font-size="24" font-weight="800" fill="#f0a882">중학교</text>
    <text x="600" y="320" font-size="16" font-weight="500" fill="#cbd5e1">진학 상위권 (준비 중)</text>
    <rect x="860" y="236" width="250" height="118" rx="16" fill="#16202e" stroke="#6da7ec" stroke-width="3"/>
    <text x="985" y="288" font-size="24" font-weight="800" fill="#b7d3f6">그 동네 시세</text>
    <text x="985" y="320" font-size="16" font-weight="500" fill="#cbd5e1">같은 법정동 실거래</text>
  </g>
  <text class="ko" x="80" y="398" font-size="18" fill="#64748b">KayserDocs · 학교와 집값을 같이 보기</text>
</svg>
```

- [ ] **Step 6: 페이지를 만든다**

`_pages/real-estate-schools.md`:

```markdown
---
layout: single
title: "서울·경기 학군 지도"
permalink: /real-estate/schools/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/real-estate-schools/header.svg
  teaser: /assets/images/real-estate-schools/header.svg
description: "서울·경기 사립초 위치를 지도에 올리고, 학교를 누르면 그 학교가 속한 법정동의 아파트 실거래 시세를 보여줍니다. 국토교통부 실거래가와 학교 위치 공공데이터를 겹쳐 봅니다."
---

<link rel="stylesheet" href="{{ '/assets/realestate/dashboard.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/realestate/schools.css' | relative_url }}">

<div class="re-app is-schools" data-base="{{ '/assets/realestate' | relative_url }}">
  <div class="re-controls">
    <div class="re-tabs" role="tablist" aria-label="지역 선택">
      <button class="re-tab is-on" data-view="seoul" role="tab" aria-selected="true">서울</button>
      <button class="re-tab" data-view="gyeonggi" role="tab" aria-selected="false">경기</button>
      <button class="re-tab" data-view="all" role="tab" aria-selected="false">전체</button>
    </div>
    <div class="re-legend-dots">
      <span><i class="is-초"></i>사립초</span>
    </div>
  </div>

  <div class="re-body">
    <div class="re-map-wrap">
      {% include realestate/map.svg %}
      <div class="re-tip" role="status" hidden></div>
    </div>
    <div class="re-panel">
      <h2 class="re-panel-title">학교를 선택하세요</h2>
      <p class="re-school-meta"></p>
      <h3 class="re-section-title re-rank-heading" hidden>같은 법정동 아파트</h3>
      <div class="re-chart"></div>
    </div>
  </div>

  <div class="re-table-wrap"><table class="re-table"></table></div>
  <p class="re-caveat">
    같은 법정동 기준입니다. 실제 배정 학교는 통학구역에 따라 다릅니다.
    사립초는 배정이 아니라 지원으로 가는 학교라 '근처'의 의미가 또 다릅니다.
  </p>
  <p class="re-footnote"></p>
</div>

<script type="module" src="{{ '/assets/realestate/schools-app.js' | relative_url }}"></script>
```

- [ ] **Step 7: 배선을 만든다**

`assets/realestate/schools-app.js`:

```javascript
import { setBase, loadSgg } from './data.js';
import { initMap } from './map.js';
import { initSchoolLayer } from './schoolmap.js';
import { NO_DATA, INK2 } from './palette.js';

const root = document.querySelector('.re-app');
setBase(root.dataset.base);
const BASE = root.dataset.base.replace(/\/$/, '');

const state = { view: 'seoul', school: null };
let schools = [];
let map = null;
let layer = null;

const VIEW_PREFIX = { seoul: '11', gyeonggi: '41', all: '' };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function visibleSchools() {
  const prefix = VIEW_PREFIX[state.view] ?? '';
  return schools.filter((s) => s.sgg.startsWith(prefix));
}

function paintBase() {
  // 구는 배경이다. 전부 같은 옅은 색으로 깔아 점이 묻히지 않게 한다.
  const values = new Map();
  for (const code of map.codesIn(state.view)) {
    values.set(code, { color: NO_DATA, label: '' });
  }
  map.paint(values);
}

async function selectSchool(school) {
  state.school = school;
  layer.setSelected(school.name);
  root.querySelector('.re-panel-title').textContent = school.name;
  root.querySelector('.re-school-meta').textContent =
    `${school.found} 초등학교 · ${school.addr}`;
  root.querySelector('.re-rank-heading').hidden = false;
  writeParams();

  const table = root.querySelector('.re-table');
  try {
    const detail = await loadSgg(school.sgg);
    const rows = detail.complexes
      .filter((c) => c.dong === school.dong && c.n >= 5)
      .slice(0, 30);
    const head = '<thead><tr><th>단지</th><th class="is-num">평당가(만원)</th>'
      + '<th class="is-num">세대</th><th class="is-num">거래</th></tr></thead>';
    const body = rows.map((c) => `<tr><td>${esc(c.name)}</td>`
      + `<td class="is-num">${c.med != null ? c.med.toLocaleString() : '—'}</td>`
      + `<td class="is-num is-dim">${c.hh != null ? c.hh.toLocaleString() : '—'}</td>`
      + `<td class="is-num is-dim">${c.n.toLocaleString()}</td></tr>`).join('');
    table.innerHTML = rows.length
      ? `${head}<tbody>${body}</tbody>`
      : `${head}<tbody><tr><td colspan="4">${esc(school.dong)}에 최근 12개월 거래 `
        + '5건 이상 단지가 없습니다.</td></tr></tbody>';
  } catch (err) {
    table.innerHTML = '<tbody><tr><td>시세를 불러오지 못했습니다.</td></tr></tbody>';
  }
}

function writeParams() {
  const q = new URLSearchParams();
  q.set('view', state.view);
  if (state.school) q.set('school', state.school.name);
  window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

function readParams() {
  const q = new URLSearchParams(window.location.search);
  const view = q.get('view');
  const valid = ['seoul', 'gyeonggi', 'all'].includes(view);
  if (valid) state.view = view;
  const name = q.get('school');
  if (name) {
    const hit = schools.find((s) => s.name === name);
    if (hit) {
      state.school = hit;
      // view= 가 쓸 만하지 않으면 고른 학교의 시도로 뷰를 맞춘다
      if (!valid) state.view = hit.sgg.startsWith('11') ? 'seoul' : 'gyeonggi';
    }
  }
}

function applyView() {
  root.querySelectorAll('.re-tab').forEach((b) => {
    const on = b.dataset.view === state.view;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-selected', String(on));
  });
  map.setView(state.view);
  paintBase();
  layer.render(visibleSchools());
  if (state.school) layer.setSelected(state.school.name);
}

function bind() {
  root.querySelectorAll('.re-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      applyView();
      writeParams();
    });
  });
}

async function start() {
  try {
    const res = await fetch(`${BASE}/schools.json`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    schools = data.schools;
    root.querySelector('.re-footnote').textContent =
      `학교 위치: 전국초중등학교위치표준데이터 · 시세: 국토교통부 실거래가 · 갱신 ${data.generated}`;
  } catch (err) {
    root.querySelector('.re-panel-title').textContent = '학교 자료를 불러오지 못했습니다';
    return;
  }

  readParams();
  map = initMap(root, { onSelect: () => {}, interactive: false });
  layer = initSchoolLayer(root, { onSelect: selectSchool });
  bind();
  applyView();
  if (state.school) await selectSchool(state.school);
}

start();
```

- [ ] **Step 8: 빌드하고 눈으로 확인한다**

```bash
cat > _tmp_Gemfile <<'EOF'
source "https://rubygems.org"
gemspec
gem "jekyll-sass-converter", "~> 2.0"
gem "webrick"
EOF
docker run --rm -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll jekyll/jekyll:4.2.2 \
  sh -c "bundle install >/dev/null 2>&1 && jekyll build"
rm -f _tmp_Gemfile _tmp_Gemfile.lock
google-chrome --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1280,1000 --virtual-time-budget=9000 \
  --screenshot=/tmp/schools-page.png http://127.0.0.1:4000/real-estate/schools/
```

`/tmp/schools-page.png` 를 Read 로 열어 확인한다. 합격 기준:

- 서울 지도 위에 주황 아닌 **파란 점**들이 보인다 (사립초는 `is-초`)
- 점이 지도 밖으로 나가지 않았다
- 오른쪽 패널이 "학교를 선택하세요" 이고 아래 안내 문구가 보인다
- 콘솔 오류 0

```bash
google-chrome --headless=new --disable-gpu --dump-dom --virtual-time-budget=9000 \
  http://127.0.0.1:4000/real-estate/schools/ 2>&1 | grep -iE "error|failed" | head
```

Expected: 출력 없음

- [ ] **Step 9: 학교를 눌러 시세가 뜨는지 확인한다**

```bash
python3 - <<'PY'
import json, subprocess
js = ("JSON.stringify({dots:document.querySelectorAll('.re-dot').length,"
      "title:document.querySelector('.re-panel-title').textContent})")
out = subprocess.run(["google-chrome","--headless=new","--disable-gpu",
    "--virtual-time-budget=9000","--dump-dom",
    "http://127.0.0.1:4000/real-estate/schools/"],
    capture_output=True, text=True).stdout
print("re-dot 개수(마크업):", out.count('class="re-dot'))
PY
```

점은 JS 로 그려지므로 `--dump-dom` 으로도 보인다. 0이면 `schools.json` 을 못
읽은 것이다.

클릭 검증은 실거래 때 쓴 CDP 스크립트와 같은 방식으로 한다 —
`document.querySelector('.re-dot').dispatchEvent(new MouseEvent('click',{bubbles:true}))`
후 `.re-table tbody tr` 개수와 `.re-panel-title` 을 읽는다. 표에 단지가 나오거나
"거래 5건 이상 단지가 없습니다" 가 나오면 정상이다 (사립초가 있는 동에 아파트가
없을 수 있다).

- [ ] **Step 10: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add assets/realestate/map.js assets/realestate/schoolmap.js \
  assets/realestate/schools.css assets/realestate/schools-app.js \
  _pages/real-estate-schools.md assets/images/real-estate-schools
git commit -m "Add school map page with private elementary schools

지도 SVG 위에 학교 점을 얹고, 학교를 누르면 같은 법정동 단지 시세를 보여준다.
구가 누를 대상이 아닌 화면을 위해 initMap 에 interactive 옵션을 넣었다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 허브 연결과 모바일 확인

**Files:**
- Modify: `_pages/real-estate.md`
- Modify: `scripts/daily.sh`
- Modify: `assets/realestate/schools.css` (모바일 측정 결과에 따라)

**Interfaces:**
- Consumes: Task 5 의 페이지
- Produces: 없음 (배선과 검증)

- [ ] **Step 1: 허브의 학군 카드를 살아있는 링크로 바꾼다**

`_pages/real-estate.md` 의 학군 카드는 지금 `<div class="rh-card is-planned">` 다.
아래로 바꾼다.

```html
  <a class="rh-card is-live" href="{{ '/real-estate/schools/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">학군 지도</h2>
    <p class="rh-desc">
      서울·경기 <strong>사립초 41곳</strong>을 지도에 올렸습니다.
      학교를 누르면 그 학교가 속한 법정동의 아파트 실거래 시세가 나옵니다.
    </p>
    <ul class="rh-points">
      <li>학교 위치와 그 동네 평당가를 한 화면에서</li>
      <li>진학 상위권 중학교는 준비 중</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>
```

- [ ] **Step 2: 일일 빌드에 학교 집계를 붙인다**

학교 위치는 자주 바뀌지 않지만, **경계 데이터나 투영이 바뀌면 학교 좌표도 다시
구워야 한다.** `scripts/daily.sh` 의 `python3 -u scripts/build_dashboard.py`
**바로 아래**에 넣는다.

```bash
# 학교 좌표는 지도 투영에 묶여 있다. 실거래 집계와 같이 돌려 어긋나지 않게 한다.
# 학교 원본(data/schools.csv.gz)은 collect_schools.py 로 따로 받는다 — 매일 받지 않는다.
if [ -f data/schools.csv.gz ]; then
  if ! python3 -u scripts/build_schools.py; then
    BUILD_FAILED=1
    echo "학교 집계 실패 — build_schools.py 가 비정상 종료했습니다." >&2
  fi
fi
```

`BUILD_FAILED` 는 바로 위 `build_dashboard.py` 블록에서 `0` 으로 초기화되므로
`set -u` 아래에서도 안전하다. 바로 위 블록과 같은 `if !` 형태를 쓴다 —
`|| BUILD_FAILED=1` 도 동작하지만 스타일이 갈린다. 스크립트 끝에서 이 값으로
비정상 종료하고, 집계 실패가 수집 커밋을 막지 않는 구조를 그대로 따른다.

그리고 커밋 대상 경로를 넓힌다. `git status --porcelain` 와 `git add` 두 줄에서

```
assets/realestate/summary.json assets/realestate/sgg
```

를 아래로 바꾼다.

```
assets/realestate/summary.json assets/realestate/sgg assets/realestate/schools.json
```

- [ ] **Step 3: 예행한다**

```bash
bash -n scripts/daily.sh && echo "문법 OK"
MAX_CALLS=0 AUTO_COMMIT=0 bash scripts/daily.sh
```

Expected: 집계 단계가 `schools.json ... 변경 없음` 을 포함해 지나가고 exit 0.
(`git checkout -- assets/realestate` 로 되돌린 뒤 진행한다.)

- [ ] **Step 4: 모바일을 측정한다**

```bash
docker run --rm -e BUNDLE_GEMFILE=/srv/jekyll/_tmp_Gemfile \
  -v "$PWD":/srv/jekyll -w /srv/jekyll jekyll/jekyll:4.2.2 \
  sh -c "bundle install >/dev/null 2>&1 && jekyll build"
python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/schools/
python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/
```

Expected: 양쪽 다 `[click]`·`[tooltip]` 합격, `scrollWidth=390`.

`check_mobile.py` 는 `path[data-sgg]` 를 클릭하는데 학군 페이지에서는 구가
비활성이라 아무 일도 일어나지 않는다 — 그래도 페이지 자체 오버플로는 잰다.
**표가 채워진 상태로도 재려면** 스크립트가 클릭하는 선택자를 `.re-dot` 로 바꿔
한 번 더 돌린다:

```bash
sed 's/path\[data-sgg\]/.re-dot/g' _dev/tools/check_mobile.py > /tmp/check_mobile_dots.py
python3 /tmp/check_mobile_dots.py http://127.0.0.1:4000/real-estate/schools/
rm -f /tmp/check_mobile_dots.py
```

불합격이면 원인별로 고친다. 표가 넘치면 `.re-table-wrap` 이 감싸고 있는지,
범례가 넘치면 `flex-wrap: wrap` 을 확인한다.

- [ ] **Step 5: 스크린샷을 눈으로 확인한다**

`/tmp/re-mobile.png` 를 Read 로 연다. 지도가 폭에 맞고 점을 손가락으로 누를 만한
크기인지, 안내 문구가 잘리지 않았는지 본다.

- [ ] **Step 6: 전체 테스트와 정리**

```bash
python3 -m unittest discover -s tests -v
rm -f _tmp_Gemfile _tmp_Gemfile.lock
git status --short
```

Expected: 전부 PASS, `_tmp_Gemfile` 없음.

- [ ] **Step 7: 커밋한다**

```bash
export GIT_AUTHOR_NAME="Jaehyun So" GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So" GIT_COMMITTER_EMAIL="dorumugs@gmail.com"
git add _pages/real-estate.md scripts/daily.sh assets/realestate/schools.css
git commit -m "Link school map from the hub and wire it into the daily build

학군 카드를 살아있는 링크로 바꾸고, 학교 좌표가 지도 투영과 어긋나지 않게
일일 집계에 붙였다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## 자체 검토

### 설계 문서 대비 커버리지

| 설계 항목 | 담당 |
|---|---|
| 투영 공유 (`projection.json`) | Task 1 |
| `map.svg` 바이트 동일 회귀 | Task 1 Step 6 |
| 학교 위치 API, 오류코드 30/12 구분 | Task 2 |
| 서울·경기만, 초·중만 | Task 2 (`normalize`) |
| 사립초 필터 | Task 4 (`select_private_elementary`) |
| 법정동 조인, 실패율 10% 한도 | Task 4 |
| 좌표 소수점 한 자리, 완전 전순서 정렬 | Task 4 (`build`) |
| viewBox 안에 점이 들어가는지 | Task 4 테스트 + Step 6 육안 |
| 크기 예산 100KB | Task 4 |
| 결정론 | Task 3 Step 3, Task 4 Step 7 |
| 걸러진 학교를 첫 화면에 전부 | Task 5 (`applyView`) |
| 구 채색 안 함, 배경으로만 | Task 5 (`paintBase`) |
| 점 8px 이상, 선택 시 링 | Task 5 (`schools.css`) |
| 범례 상시, 툴팁에 글자 | Task 5 (페이지, `schoolmap.js`) |
| 오해 방지 문구 | Task 5 (`re-caveat`) |
| URL 파라미터, 잘못된 값 무시 | Task 5 (`readParams`) |
| 표지 배너 | Task 5 Step 5 |
| 모바일 390px | Task 6 Step 4 |
| 허브 연결 | Task 6 Step 1 |
| 중학교(2단계) | **범위 밖** — 학교알리미 키 확보 후 별도 계획서 |

빠진 항목 없음.

### 이름 일관성

- `schools_api.{COLUMNS, normalize, parse_response, fetch_page, ApiError}` — Task 2 에서
  정의하고 Task 3 에서 같은 이름으로 쓴다
- `build_schools.{to_svg_xy, parse_addr, select_private_elementary, build}` — Task 4
  에서 정의하고 테스트가 같은 이름을 쓴다
- `build_geo.projection_params` — Task 1 에서 정의하고 Task 4 가 그 산출물
  (`projection.json`)만 읽는다. 함수를 import 하지 않는다
- `initMap(root, {onSelect, interactive})` — Task 5 Step 1 에서 옵션을 늘리고
  같은 파일의 `schools-app.js` 가 `interactive: false` 로 부른다
- `initSchoolLayer(root, {onSelect})` 의 핸들 `{render, setSelected, clear}` —
  Task 5 Step 3 에서 정의하고 `schools-app.js` 가 그 이름으로 부른다
- 학교급 문자열은 `"초"`/`"중"` 한 글자로 통일한다 (`LEVEL_SHORT`, CSS `.is-초`,
  JS `is-${school.lvl}`)

### 남는 위험

1. **`parse_addr` 의 지번 제거가 거칠다.** 마지막 토큰이 숫자로 시작하면 뗀다.
   `대치동 산 12` 처럼 '산' 이 끼면 두 번 떼도록 했지만, 실제 주소에 다른 형태가
   있을 수 있다. Task 4 Step 5 에서 조인 실패가 한도를 넘으면 여기부터 본다.
2. **CSS 와 `palette.js` 에 학교급 색이 두 번 적힌다.** CSS 에서 JS 를 import 할
   수 없어 불가피하다. 주석으로 묶어 뒀지만 한쪽만 고치면 어긋난다.
3. **사립초가 있는 법정동에 아파트가 없을 수 있다.** 표가 비는 것은 버그가
   아니다 — 빈 상태 문구가 그 경우를 덮는다. 다만 41곳 중 몇 곳이 그런지
   Task 5 Step 9 에서 세어 두면 좋다.
4. **`check_mobile.py` 가 학군 페이지에서는 구를 클릭한다.** 구가 비활성이라
   표가 빈 채로 측정된다. Task 6 Step 4 의 `sed` 우회로 점을 클릭해 한 번 더
   재도록 했지만, 근본적으로는 스크립트가 선택자를 인자로 받게 고치는 것이 낫다.
   이번 범위에서는 우회로 둔다.
