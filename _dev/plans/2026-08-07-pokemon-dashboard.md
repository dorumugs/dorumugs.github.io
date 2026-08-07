# 포켓몬 카드 지수 대시보드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/real-estate/` 허브를 `/dashboard/` 로 이전하고, 구성·가중치·산식을 전부 공개한 포켓몬 카드 가격지수 대시보드를 매일 자동 갱신되게 붙인다.

**Architecture:** TCGdex 공개 API(키 불필요)에서 정규 확장팩 카드 가격을 받아, 시대 4 × 가격대 3 = 12칸 층화추출로 뽑은 300장 유니버스의 균등가중 지수를 만든다. 파싱·추출·지수계산은 `tcgdex_api.py` 순수 함수에 격리하고, I/O·예산·재개는 `collect_pokemon.py`, 집계는 `build_pokemon.py` 가 맡는다. 기존 부동산 파이프라인 3종과 동일한 구조·동일한 flock 을 쓴다.

**Tech Stack:** Python 3 표준 라이브러리만 (`urllib`, `csv`, `gzip`, `json`, `concurrent.futures`) · 표준 `unittest` · Jekyll(minimal-mistakes) · 바닐라 JS

설계 문서: `_dev/specs/2026-08-07-pokemon-dashboard-design.md`

## Global Constraints

- **외부 의존성 금지.** pandas·requests 계열 없음. 표준 라이브러리만 쓴다.
- **`scripts/<name>_api.py` 는 순수 함수만.** 네트워크 I/O 를 넣지 않는다.
- **gzip 은 `mtime=0`, `filename=""`.** 내용이 같으면 바이트도 같아야 한다. `rtms.gzip_bytes()` 를 재사용한다.
- **테스트는 `python3 -m unittest discover -s tests`.** pytest 없음.
- **동시 요청은 4로 고정.** TCGdex 는 무료 API 다.
- **`_config.yml` 을 수정하지 않는다.** 플러그인을 추가하지 않는다.
- **테마 원본(`_layouts/`, `_includes/`, `_sass/`, `assets/` 중 realestate·pokemon 외)의 기존 파일을 수정하지 않는다.** 신규 파일 추가는 허용.
- **390px 에서 가로 오버플로 없음.** 표는 자체 스크롤 컨테이너 안에서만 넘친다.
- **커밋 신원**은 `GIT_AUTHOR_*`/`GIT_COMMITTER_*` 환경변수로 단발 지정. 전역 `git config` 를 바꾸지 않는다.
- 커밋 메시지: 영문 한 줄 요약 + 빈 줄 + 한국어 본문 + `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## Task 1: 리다이렉트 레이아웃과 옛 주소 stub

**Files:**
- Create: `_layouts/redirect.html`
- Create: `_pages/redirects/real-estate-hub.md`
- Create: `_pages/redirects/real-estate-trades.md`
- Create: `_pages/redirects/real-estate-schools.md`
- Create: `_pages/redirects/real-estate-redevelopment.md`

**Interfaces:**
- Consumes: 없음
- Produces: `layout: redirect` — front matter 에 `redirect_to` (문자열, 사이트 루트 기준 경로) 를 받아 meta refresh 를 낸다. Task 2 가 이 레이아웃을 쓴다.

이 태스크는 Task 2 보다 **먼저** 끝나야 한다. Task 2 가 permalink 를 옮기는 순간 옛 주소가 404 가 되므로, 받아줄 stub 이 미리 있어야 한다. 단 stub 의 permalink 는 Task 2 가 옛 permalink 를 비워줘야 충돌하지 않으므로, **두 태스크는 한 커밋으로 함께 배포한다**(Task 2 스텝 마지막에 명시).

- [ ] **Step 1: 리다이렉트 레이아웃 작성**

`_layouts/redirect.html` 을 만든다. `jekyll-redirect-from` 이 생성하는 것과 같은 형태다.

```html
---
layout: null
sitemap: false
---
<!doctype html>
<html lang="{{ site.locale | slice: 0,2 | default: 'ko' }}">
<head>
  <meta charset="utf-8">
  <title>이 페이지는 옮겨졌습니다</title>
  <link rel="canonical" href="{{ page.redirect_to | absolute_url }}">
  <meta http-equiv="refresh" content="0; url={{ page.redirect_to | absolute_url }}">
  <meta name="robots" content="noindex">
</head>
<body>
  <p>이 페이지는 <a href="{{ page.redirect_to | relative_url }}">{{ page.redirect_to }}</a> 로 옮겨졌습니다.</p>
  <script>location.replace({{ page.redirect_to | absolute_url | jsonify }});</script>
</body>
</html>
```

`sitemap: false` 로 `jekyll-sitemap` 이 옛 주소를 다시 색인하지 않게 막는다.

- [ ] **Step 2: stub 4개 작성**

`_pages/redirects/real-estate-hub.md`:

```markdown
---
layout: redirect
permalink: /real-estate/
redirect_to: /dashboard/
sitemap: false
---
```

`_pages/redirects/real-estate-trades.md`:

```markdown
---
layout: redirect
permalink: /real-estate/trades/
redirect_to: /dashboard/real-estate/trades/
sitemap: false
---
```

`_pages/redirects/real-estate-schools.md`:

```markdown
---
layout: redirect
permalink: /real-estate/schools/
redirect_to: /dashboard/real-estate/schools/
sitemap: false
---
```

`_pages/redirects/real-estate-redevelopment.md`:

```markdown
---
layout: redirect
permalink: /real-estate/redevelopment/
redirect_to: /dashboard/real-estate/redevelopment/
sitemap: false
---
```

- [ ] **Step 3: 아직 빌드하지 않는다**

이 시점에는 `_pages/real-estate.md` 등이 여전히 옛 permalink 를 갖고 있어 **permalink 충돌로 빌드가 실패한다.** 정상이다. Task 2 에서 옛 permalink 를 옮긴 뒤에 함께 빌드·커밋한다.

커밋하지 말고 Task 2 로 넘어간다.

---

## Task 2: URL 이전과 nav 변경, 허브 재구성

**Files:**
- Modify: `_data/navigation.yml:3-4`
- Modify: `_pages/real-estate.md` → 이름 변경 `_pages/dashboard.md`
- Modify: `_pages/real-estate-trades.md:4` (permalink)
- Modify: `_pages/real-estate-schools.md:4` (permalink)
- Modify: `_pages/real-estate-redevelopment.md:4` (permalink)

**Interfaces:**
- Consumes: Task 1 의 `layout: redirect`
- Produces: `/dashboard/` 허브. Task 9 가 여기에 포켓몬 카드를 추가한다. 허브 카드 마크업은 `.rh-grid` > `.rh-card` (기존 `assets/realestate/hub.css`).

- [ ] **Step 1: nav 라벨과 주소 변경**

`_data/navigation.yml` 의 첫 항목을 바꾼다.

```yaml
main:
  - title: "Dashboard"
    url: /dashboard/
```

- [ ] **Step 2: 대시보드 3종 permalink 이전**

각 파일의 4번째 줄만 바꾼다.

| 파일 | 변경 후 |
|---|---|
| `_pages/real-estate-trades.md` | `permalink: /dashboard/real-estate/trades/` |
| `_pages/real-estate-schools.md` | `permalink: /dashboard/real-estate/schools/` |
| `_pages/real-estate-redevelopment.md` | `permalink: /dashboard/real-estate/redevelopment/` |

파일명은 그대로 둔다. permalink 가 주소를 정하므로 파일명을 바꿀 이유가 없고, git 이력이 끊긴다.

- [ ] **Step 3: 허브 페이지 이름 변경과 재구성**

```bash
git mv _pages/real-estate.md _pages/dashboard.md
```

front matter 를 바꾼다.

```yaml
---
layout: single
title: "대시보드"
permalink: /dashboard/
classes: wide
author_profile: false
toc: false
header:
  image: /assets/images/real-estate-hub/header.svg
  teaser: /assets/images/real-estate-hub/header.svg
description: "데이터로 보는 도구들을 모았습니다. 국토교통부 실거래가 435만 건으로 만든 부동산 대시보드부터, 구성과 산식을 전부 공개한 포켓몬 카드 가격지수까지."
---
```

본문 도입부를 바꾼다. 기존 "수도권에서 집을 고를 때…" 문단을 아래로 교체한다.

```markdown
글로 한 번 정리한 주제를 눌러볼 수 있는 화면으로 옮기고 있습니다.
지금은 수도권 부동산과 포켓몬 카드 시장을 다룹니다.

## 부동산
```

기존 카드 3개의 `href` 를 새 주소로 바꾼다.

| 변경 전 | 변경 후 |
|---|---|
| `{{ '/real-estate/trades/' \| relative_url }}` | `{{ '/dashboard/real-estate/trades/' \| relative_url }}` |
| `{{ '/real-estate/schools/' \| relative_url }}` | `{{ '/dashboard/real-estate/schools/' \| relative_url }}` |
| `{{ '/real-estate/redevelopment/' \| relative_url }}` | `{{ '/dashboard/real-estate/redevelopment/' \| relative_url }}` |

부동산 카드 3개를 감싼 `</div>` (`.rh-grid` 닫는 태그) 뒤에 수집품 섹션을 연다. 카드 내용은 Task 9 에서 채우므로 지금은 섹션 제목과 빈 그리드만 둔다.

```markdown
## 수집품

<div class="rh-grid">
</div>
```

- [ ] **Step 4: 남은 옛 링크가 없는지 확인**

```bash
grep -rn "/real-estate/" _pages/ _posts/ _data/ _includes/ | grep -v "_pages/redirects/"
```

Expected: 출력 없음. (`_pages/redirects/` 안의 permalink 는 의도된 것이므로 제외한다.)

`assets/images/real-estate-hub/` 같은 **이미지 경로는 `/real-estate/` 로 시작하지 않으므로** 위 grep 에 걸리지 않는다. 걸렸다면 이미지가 아니라 진짜 링크다.

- [ ] **Step 5: 빌드해서 리다이렉트가 생성되는지 확인**

```bash
mkdir -p /tmp/gemhome && printf 'source "https://rubygems.org"\ngem "github-pages", group: :jekyll_plugins\ngem "jekyll-sass-converter", "~> 2.0"\n' > /tmp/gemhome/Gemfile
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /srv/jekyll/_site_check"
```

Expected: 빌드 성공. 그 다음 확인:

```bash
test -f _site_check/dashboard/index.html && echo "허브 OK"
test -f _site_check/dashboard/real-estate/trades/index.html && echo "실거래 OK"
grep -q 'url=.*dashboard/real-estate/trades' _site_check/real-estate/trades/index.html && echo "리다이렉트 OK"
grep -c '/real-estate/trades/' _site_check/sitemap.xml
```

Expected: `허브 OK`, `실거래 OK`, `리다이렉트 OK`, 마지막은 `0` (sitemap 에 옛 주소가 없어야 한다).

- [ ] **Step 6: 검사용 빌드 산출물 삭제**

```bash
rm -rf _site_check
```

`_site_check` 가 커밋되면 안 된다. `.gitignore` 에 `_site` 는 있으나 `_site_check` 는 없다.

- [ ] **Step 7: 커밋** (Task 1 결과물과 함께)

```bash
git add _layouts/redirect.html _pages/redirects/ _data/navigation.yml _pages/dashboard.md _pages/real-estate.md _pages/real-estate-trades.md _pages/real-estate-schools.md _pages/real-estate-redevelopment.md
git commit -m "$(cat <<'EOF'
Move the real-estate hub to /dashboard/ and keep old URLs alive

허브를 부동산 전용에서 대시보드 모음으로 승격했습니다. 옛 주소 4개는
meta refresh stub 으로 남겨 링크가 끊기지 않게 했고, sitemap 에서는 뺐습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: TCGdex 응답 파싱 순수 함수

**Files:**
- Create: `scripts/tcgdex_api.py`
- Create: `tests/test_tcgdex_api.py`
- Create: `tests/fixtures/tcgdex_sets.json`
- Create: `tests/fixtures/tcgdex_set_base1.json`
- Create: `tests/fixtures/tcgdex_card_priced.json`
- Create: `tests/fixtures/tcgdex_card_unpriced.json`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `BASE = "https://api.tcgdex.net/v2/en"`
  - `COLUMNS: list[str]` — 가격 CSV 컬럼 순서
  - `ERAS: tuple[str, ...]` = `("빈티지", "클래식", "모던", "최신")`
  - `VARIANT_PRIORITY: tuple[str, ...]`
  - `class ApiError(Exception)`
  - `parse_set_list(payload: list) -> list[dict]` → `{"set_id","name","card_count"}`
  - `parse_set_detail(payload: dict) -> dict` → `{"set_id","name","release_date","card_ids"}`
  - `era_of(release_date: str) -> str`
  - `pick_variant(tcgplayer: dict) -> tuple[str, dict] | None`
  - `parse_card_pricing(payload: dict, on_date: str) -> dict | None`

- [ ] **Step 1: fixture 를 실제 응답에서 고정한다**

실제 응답을 받아 저장한다. 손으로 지어내지 않는다 — 공식 API 가 아닌 곳은 화면이 바뀌면 조용히 깨지므로 실물을 박아 둬야 회귀를 잡는다.

```bash
cd /home/dorumugs/Projects/dorumugs.github.io
python3 - <<'PY'
import json, urllib.request
from pathlib import Path
F = Path("tests/fixtures")
def get(u):
    with urllib.request.urlopen(u, timeout=30) as r: return json.load(r)
B = "https://api.tcgdex.net/v2/en"
# 세트 목록은 218개 전부면 크므로 앞 12개만 고정한다
json.dump(get(f"{B}/sets")[:12], open(F/"tcgdex_sets.json","w"), ensure_ascii=False, indent=1)
json.dump(get(f"{B}/sets/base1"), open(F/"tcgdex_set_base1.json","w"), ensure_ascii=False, indent=1)
# 가격 있는 카드 (홀로 변종 + cardmarket avg7/avg30 보유)
json.dump(get(f"{B}/cards/base1-4"), open(F/"tcgdex_card_priced.json","w"), ensure_ascii=False, indent=1)
# 가격 없는 카드 (프로모 세트)
np = get(f"{B}/sets/np")
json.dump(get(f"{B}/cards/{np['cards'][0]['id']}"), open(F/"tcgdex_card_unpriced.json","w"), ensure_ascii=False, indent=1)
print("fixtures written")
PY
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_tcgdex_api.py`:

```python
"""TCGdex 응답 파싱 검증. 표준 unittest 만 쓴다 (pytest 미설치 환경).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
FIXTURES = ROOT / "tests" / "fixtures"

import tcgdex_api  # noqa: E402


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestParseSetList(unittest.TestCase):
    def test_extracts_id_name_and_count(self) -> None:
        rows = tcgdex_api.parse_set_list(_fixture("tcgdex_sets.json"))
        self.assertTrue(rows)
        base = [r for r in rows if r["set_id"] == "base1"]
        self.assertEqual(len(base), 1)
        self.assertEqual(base[0]["name"], "Base Set")
        self.assertEqual(base[0]["card_count"], 102)

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(tcgdex_api.ApiError):
            tcgdex_api.parse_set_list({"error": "nope"})


class TestParseSetDetail(unittest.TestCase):
    def test_extracts_release_date_and_card_ids(self) -> None:
        d = tcgdex_api.parse_set_detail(_fixture("tcgdex_set_base1.json"))
        self.assertEqual(d["set_id"], "base1")
        self.assertEqual(d["release_date"], "1999-01-09")
        self.assertIn("base1-4", d["card_ids"])
        self.assertEqual(len(d["card_ids"]), 102)

    def test_missing_cards_yields_empty_list(self) -> None:
        d = tcgdex_api.parse_set_detail({"id": "x", "name": "X", "releaseDate": "2020-01-01"})
        self.assertEqual(d["card_ids"], [])


class TestEraOf(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(tcgdex_api.era_of("1999-01-09"), "빈티지")
        self.assertEqual(tcgdex_api.era_of("2003-12-31"), "빈티지")
        self.assertEqual(tcgdex_api.era_of("2004-01-01"), "클래식")
        self.assertEqual(tcgdex_api.era_of("2010-12-31"), "클래식")
        self.assertEqual(tcgdex_api.era_of("2011-01-01"), "모던")
        self.assertEqual(tcgdex_api.era_of("2019-12-31"), "모던")
        self.assertEqual(tcgdex_api.era_of("2020-01-01"), "최신")
        self.assertEqual(tcgdex_api.era_of("2026-08-07"), "최신")

    def test_blank_is_none(self) -> None:
        self.assertIsNone(tcgdex_api.era_of(""))


class TestPickVariant(unittest.TestCase):
    def test_prefers_holofoil(self) -> None:
        tp = {"normal": {"marketPrice": 1.0}, "holofoil": {"marketPrice": 9.0}}
        name, block = tcgdex_api.pick_variant(tp)
        self.assertEqual(name, "holofoil")
        self.assertEqual(block["marketPrice"], 9.0)

    def test_falls_back_to_normal(self) -> None:
        name, _ = tcgdex_api.pick_variant({"normal": {"marketPrice": 1.0}})
        self.assertEqual(name, "normal")

    def test_skips_variant_without_market_price(self) -> None:
        tp = {"holofoil": {"lowPrice": 3.0}, "normal": {"marketPrice": 1.0}}
        name, _ = tcgdex_api.pick_variant(tp)
        self.assertEqual(name, "normal")

    def test_none_when_no_usable_variant(self) -> None:
        self.assertIsNone(tcgdex_api.pick_variant({"unit": "USD", "updated": "x"}))


class TestParseCardPricing(unittest.TestCase):
    def test_priced_card_yields_full_row(self) -> None:
        row = tcgdex_api.parse_card_pricing(_fixture("tcgdex_card_priced.json"), "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(row["date"], "2026-08-07")
        self.assertEqual(row["card_id"], "base1-4")
        self.assertEqual(row["variant"], "holofoil")
        self.assertGreater(row["tp_market"], 0)
        self.assertGreater(row["cm_avg30"], 0)
        self.assertEqual(set(row), set(tcgdex_api.COLUMNS))

    def test_unpriced_card_yields_none(self) -> None:
        self.assertIsNone(
            tcgdex_api.parse_card_pricing(_fixture("tcgdex_card_unpriced.json"), "2026-08-07")
        )

    def test_cardmarket_only_still_yields_row(self) -> None:
        payload = {"id": "x-1", "name": "X", "pricing": {"cardmarket": {"avg": 5.0, "trend": 4.0}}}
        row = tcgdex_api.parse_card_pricing(payload, "2026-08-07")
        self.assertIsNotNone(row)
        self.assertEqual(row["variant"], "")
        self.assertIsNone(row["tp_market"])
        self.assertEqual(row["cm_avg"], 5.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `python3 -m unittest tests.test_tcgdex_api -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tcgdex_api'`

- [ ] **Step 4: 최소 구현 작성**

`scripts/tcgdex_api.py`:

```python
"""TCGdex 카드 가격 API 파싱.

    https://api.tcgdex.net/v2/en

API 키가 필요 없다. 순수 파싱 함수만 두어 네트워크 없이 검증한다 — 공식 정부
API 가 아니라 응답 형태가 예고 없이 바뀔 수 있으므로 fixtures 로 회귀를 잡는다.

세트 목록에는 가격이 없다. 가격은 카드 1장당 1요청이다 (2026-08-07 확인).
"""

from __future__ import annotations

BASE = "https://api.tcgdex.net/v2/en"

# 가격 CSV 컬럼 순서. collect_pokemon.py 가 이 순서로 쓴다.
COLUMNS = [
    "date", "card_id", "variant",
    "tp_market", "tp_low", "tp_mid",
    "cm_avg", "cm_trend", "cm_avg7", "cm_avg30",
]

ERAS = ("빈티지", "클래식", "모던", "최신")

# TCGplayer variant 우선순위. 홀로가 그 카드의 '대표 시세'로 통용된다.
VARIANT_PRIORITY = ("holofoil", "normal", "reverseHolofoil")


class ApiError(Exception):
    """응답이 기대한 형태가 아닐 때."""


def parse_set_list(payload) -> list[dict]:
    """세트 목록 응답을 [{'set_id','name','card_count'}] 로."""
    if not isinstance(payload, list):
        raise ApiError(f"세트 목록이 배열이 아닙니다: {type(payload).__name__}")
    rows = []
    for item in payload:
        sid = (item.get("id") or "").strip()
        if not sid:
            continue
        rows.append({
            "set_id": sid,
            "name": (item.get("name") or "").strip(),
            "card_count": int((item.get("cardCount") or {}).get("total") or 0),
        })
    return rows


def parse_set_detail(payload: dict) -> dict:
    """세트 상세를 {'set_id','name','release_date','card_ids'} 로.

    이 응답의 카드 배열에는 가격이 없다. id 만 걷어 카드별로 다시 부른다.
    """
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ApiError("세트 상세에 id 가 없습니다")
    return {
        "set_id": payload["id"],
        "name": (payload.get("name") or "").strip(),
        "release_date": (payload.get("releaseDate") or "").strip(),
        "card_ids": [c["id"] for c in (payload.get("cards") or []) if c.get("id")],
    }


def era_of(release_date: str) -> str | None:
    """발매일을 시대 구간으로. 빈 값이면 None."""
    if not release_date or len(release_date) < 4 or not release_date[:4].isdigit():
        return None
    year = int(release_date[:4])
    if year <= 2003:
        return "빈티지"
    if year <= 2010:
        return "클래식"
    if year <= 2019:
        return "모던"
    return "최신"


def _num(value) -> float | None:
    """숫자로 바꾼다. None·빈 문자열·0 이하는 None (0원은 시세가 아니다)."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def pick_variant(tcgplayer: dict) -> tuple[str, dict] | None:
    """대표 variant 를 고른다. marketPrice 가 있는 것만 후보다."""
    if not isinstance(tcgplayer, dict):
        return None
    for name in VARIANT_PRIORITY:
        block = tcgplayer.get(name)
        if isinstance(block, dict) and _num(block.get("marketPrice")) is not None:
            return name, block
    return None


def parse_card_pricing(payload: dict, on_date: str) -> dict | None:
    """카드 응답에서 가격 한 행을 만든다. 양쪽 다 값이 없으면 None.

    TCGplayer(USD) 가 지수 기준이고 Cardmarket(EUR) 은 avg7/avg30 을 주므로
    0일차 변화율의 유일한 근거다. 한쪽만 있어도 행을 남긴다.
    """
    pricing = (payload or {}).get("pricing") or {}
    tp_pick = pick_variant(pricing.get("tcgplayer") or {})
    cm = pricing.get("cardmarket") or {}

    variant, tp = ("", {}) if tp_pick is None else tp_pick
    row = {
        "date": on_date,
        "card_id": (payload or {}).get("id") or "",
        "variant": variant,
        "tp_market": _num(tp.get("marketPrice")),
        "tp_low": _num(tp.get("lowPrice")),
        "tp_mid": _num(tp.get("midPrice")),
        "cm_avg": _num(cm.get("avg")),
        "cm_trend": _num(cm.get("trend")),
        "cm_avg7": _num(cm.get("avg7")),
        "cm_avg30": _num(cm.get("avg30")),
    }
    if not row["card_id"]:
        return None
    if all(row[k] is None for k in COLUMNS[3:]):
        return None
    return row
```

- [ ] **Step 5: 테스트가 통과하는지 확인**

Run: `python3 -m unittest tests.test_tcgdex_api -v`
Expected: PASS (전체 케이스)

fixture 의 `base1-4` 가 홀로가 아닌 variant 로 잡히면 `test_priced_card_yields_full_row` 가 실패한다. 그 경우 fixture 를 열어 실제 variant 명을 확인하고 **테스트 기대값을 실물에 맞춘다** — 구현을 비틀지 않는다.

- [ ] **Step 6: 전체 테스트 회귀 확인**

Run: `python3 -m unittest discover -s tests`
Expected: 기존 테스트 전부 통과 (신규 모듈이 기존 것을 건드리지 않았는지 확인)

- [ ] **Step 7: 커밋**

```bash
git add scripts/tcgdex_api.py tests/test_tcgdex_api.py tests/fixtures/tcgdex_*.json
git commit -m "$(cat <<'EOF'
Add TCGdex price response parsing

포켓몬 카드 가격 API 파싱을 순수 함수로 넣었습니다. 실제 응답을 fixture 로
고정해 응답 형태가 바뀌면 테스트가 잡습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 층화추출과 지수 계산 순수 함수

**Files:**
- Modify: `scripts/tcgdex_api.py` (함수 추가)
- Modify: `tests/test_tcgdex_api.py` (테스트 클래스 추가)

**Interfaces:**
- Consumes: Task 3 의 `ERAS`
- Produces:
  - `BANDS = ("고가", "중가", "저가")`
  - `PER_CELL = 25`
  - `CARRY_FORWARD_DAYS = 7`
  - `tercile_bounds(prices: list[float]) -> tuple[float, float]`
  - `band_of(price: float, bounds: tuple[float, float]) -> str`
  - `stratify(cards: list[dict], per_cell: int, seed: int) -> list[dict]` — 입력 카드는 `{"card_id","era","price"}`, 출력은 입력 dict 에 `"band"` 를 더한 것
  - `fill_forward(values: list[float | None], max_days: int) -> list[float | None]`
  - `index_point(universe: list[dict], day_prices: dict[str, float], base_prices: dict[str, float]) -> dict` — `{"index","by_era","by_band","missing"}`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_tcgdex_api.py` 끝의 `if __name__` 블록 **앞에** 아래를 추가한다.

```python
class TestTercileBounds(unittest.TestCase):
    def test_splits_into_thirds(self) -> None:
        lo, hi = tcgdex_api.tercile_bounds([float(i) for i in range(1, 10)])
        self.assertAlmostEqual(lo, 4.0)
        self.assertAlmostEqual(hi, 7.0)

    def test_single_value_gives_equal_bounds(self) -> None:
        lo, hi = tcgdex_api.tercile_bounds([5.0])
        self.assertEqual(lo, 5.0)
        self.assertEqual(hi, 5.0)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            tcgdex_api.tercile_bounds([])


class TestBandOf(unittest.TestCase):
    def test_assigns_by_bounds(self) -> None:
        bounds = (4.0, 7.0)
        self.assertEqual(tcgdex_api.band_of(2.0, bounds), "저가")
        self.assertEqual(tcgdex_api.band_of(5.0, bounds), "중가")
        self.assertEqual(tcgdex_api.band_of(9.0, bounds), "고가")

    def test_boundary_goes_upward(self) -> None:
        bounds = (4.0, 7.0)
        self.assertEqual(tcgdex_api.band_of(4.0, bounds), "중가")
        self.assertEqual(tcgdex_api.band_of(7.0, bounds), "고가")


class TestStratify(unittest.TestCase):
    def _cards(self, era: str, n: int, offset: float = 0.0):
        return [
            {"card_id": f"{era}-{i}", "era": era, "price": float(i) + offset}
            for i in range(1, n + 1)
        ]

    def test_picks_per_cell_from_every_cell(self) -> None:
        cards = []
        for era in tcgdex_api.ERAS:
            cards += self._cards(era, 60)
        picked = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        self.assertEqual(len(picked), 5 * 3 * len(tcgdex_api.ERAS))
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cell = [c for c in picked if c["era"] == era and c["band"] == band]
                self.assertEqual(len(cell), 5, f"{era}/{band}")

    def test_is_deterministic_for_same_seed(self) -> None:
        cards = []
        for era in tcgdex_api.ERAS:
            cards += self._cards(era, 60)
        a = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        b = tcgdex_api.stratify(cards, per_cell=5, seed=42)
        self.assertEqual([c["card_id"] for c in a], [c["card_id"] for c in b])

    def test_takes_all_when_cell_smaller_than_quota(self) -> None:
        cards = self._cards("빈티지", 6)
        picked = tcgdex_api.stratify(cards, per_cell=5, seed=1)
        self.assertEqual(len(picked), 6)

    def test_ignores_cards_without_era(self) -> None:
        cards = self._cards("빈티지", 30) + [{"card_id": "x", "era": None, "price": 1.0}]
        picked = tcgdex_api.stratify(cards, per_cell=3, seed=1)
        self.assertNotIn("x", [c["card_id"] for c in picked])


class TestFillForward(unittest.TestCase):
    def test_carries_last_value(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([1.0, None, None, 4.0], max_days=7),
            [1.0, 1.0, 1.0, 4.0],
        )

    def test_stops_after_max_days(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([1.0, None, None, None], max_days=2),
            [1.0, 1.0, 1.0, None],
        )

    def test_leading_none_stays_none(self) -> None:
        self.assertEqual(
            tcgdex_api.fill_forward([None, None, 3.0], max_days=7),
            [None, None, 3.0],
        )


class TestIndexPoint(unittest.TestCase):
    def _universe(self):
        out = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                out.append({"card_id": f"{era}-{band}", "era": era, "band": band})
        return out

    def test_all_flat_is_one_hundred(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        point = tcgdex_api.index_point(uni, dict(base), base)
        self.assertAlmostEqual(point["index"], 100.0)
        self.assertEqual(point["missing"], 0)

    def test_uniform_doubling_is_two_hundred(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 20.0 for k in base}
        self.assertAlmostEqual(tcgdex_api.index_point(uni, day, base)["index"], 200.0)

    def test_cells_are_equally_weighted(self) -> None:
        """한 칸만 2배가 되면 지수는 1/12 만큼만 오른다."""
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = dict(base)
        day["빈티지-고가"] = 20.0
        expected = 100.0 * (11 * 1.0 + 2.0) / 12
        self.assertAlmostEqual(tcgdex_api.index_point(uni, day, base)["index"], expected)

    def test_missing_card_is_excluded_and_counted(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 10.0 for k in base if k != "빈티지-고가"}
        point = tcgdex_api.index_point(uni, day, base)
        self.assertEqual(point["missing"], 1)
        self.assertAlmostEqual(point["index"], 100.0)

    def test_empty_cell_drops_out_of_the_average(self) -> None:
        """칸이 통째로 비면 그 칸을 빼고 남은 칸으로 평균낸다."""
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = {k: 20.0 for k in base if k != "빈티지-고가"}
        point = tcgdex_api.index_point(uni, day, base)
        self.assertAlmostEqual(point["index"], 200.0)

    def test_sub_indices_are_reported(self) -> None:
        uni = self._universe()
        base = {c["card_id"]: 10.0 for c in uni}
        day = dict(base)
        for band in tcgdex_api.BANDS:
            day[f"빈티지-{band}"] = 20.0
        point = tcgdex_api.index_point(uni, day, base)
        self.assertAlmostEqual(point["by_era"]["빈티지"], 200.0)
        self.assertAlmostEqual(point["by_era"]["최신"], 100.0)
        self.assertAlmostEqual(point["by_band"]["고가"], 100.0 * (1 * 2.0 + 3 * 1.0) / 4)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `python3 -m unittest tests.test_tcgdex_api -v`
Expected: FAIL — `AttributeError: module 'tcgdex_api' has no attribute 'tercile_bounds'`

- [ ] **Step 3: 구현 추가**

`scripts/tcgdex_api.py` 끝에 추가한다. 파일 상단 import 에 `import random` 을 더한다.

```python
BANDS = ("고가", "중가", "저가")

# 칸당 표본 수. 12칸 × 25 = 300장. 동시 4로 약 70초면 다 받는다.
PER_CELL = 25

# 가격이 빠졌을 때 마지막 값을 이월하는 최대 일수.
CARRY_FORWARD_DAYS = 7


def tercile_bounds(prices: list[float]) -> tuple[float, float]:
    """가격 분포의 3분위 경계 (하한, 상한) 를 낸다."""
    if not prices:
        raise ValueError("가격이 비었습니다")
    ordered = sorted(prices)
    n = len(ordered)
    return ordered[n // 3], ordered[(2 * n) // 3]


def band_of(price: float, bounds: tuple[float, float]) -> str:
    """가격을 가격대 이름으로. 경계값은 위 칸에 넣는다."""
    low, high = bounds
    if price >= high:
        return "고가"
    if price >= low:
        return "중가"
    return "저가"


def stratify(cards: list[dict], per_cell: int = PER_CELL, seed: int = 20260807) -> list[dict]:
    """시대 × 가격대 12칸에서 고정 표본을 뽑는다.

    가격대 경계는 시대 안에서 잡는다. 시대를 가로질러 절대 금액으로 자르면
    빈티지가 전부 고가, 최신이 전부 저가가 되어 칸이 무너진다.

    칸 안에서 다시 5분위로 나눠 균등하게 뽑는다. 그냥 무작위로 뽑으면 저가
    쪽에 몰려 그 칸의 상단이 지수에 안 들어간다.
    """
    by_era: dict[str, list[dict]] = {}
    for card in cards:
        if card.get("era") in ERAS and card.get("price"):
            by_era.setdefault(card["era"], []).append(card)

    picked: list[dict] = []
    for era in ERAS:
        pool = by_era.get(era) or []
        if not pool:
            continue
        bounds = tercile_bounds([c["price"] for c in pool])
        cells: dict[str, list[dict]] = {b: [] for b in BANDS}
        for card in pool:
            band = band_of(card["price"], bounds)
            cells[band].append({**card, "band": band})

        for band in BANDS:
            cell = sorted(cells[band], key=lambda c: (c["price"], c["card_id"]))
            if len(cell) <= per_cell:
                picked.extend(cell)
                continue
            # 칸을 5분위로 갈라 각 분위에서 균등하게 뽑는다.
            rng = random.Random(f"{seed}-{era}-{band}")
            chunks = 5
            quota, extra = divmod(per_cell, chunks)
            chosen: list[dict] = []
            for k in range(chunks):
                start = (len(cell) * k) // chunks
                end = (len(cell) * (k + 1)) // chunks
                slice_ = cell[start:end]
                want = quota + (1 if k < extra else 0)
                chosen.extend(rng.sample(slice_, min(want, len(slice_))))
            # 분위가 짧아 못 채웠으면 남은 데서 채운다.
            if len(chosen) < per_cell:
                rest = [c for c in cell if c not in chosen]
                chosen.extend(rng.sample(rest, min(per_cell - len(chosen), len(rest))))
            picked.extend(sorted(chosen, key=lambda c: c["card_id"]))
    return picked


def fill_forward(values: list[float | None], max_days: int = CARRY_FORWARD_DAYS) -> list[float | None]:
    """결측을 마지막 값으로 이월한다. max_days 를 넘으면 결측으로 둔다.

    영원히 이월하면 상장폐지된 종목을 계속 들고 있는 지수가 된다. 그게 우리가
    깐 생존편향의 반대편 오류다.
    """
    out: list[float | None] = []
    last: float | None = None
    run = 0
    for value in values:
        if value is not None:
            out.append(value)
            last = value
            run = 0
            continue
        run += 1
        out.append(last if (last is not None and run <= max_days) else None)
    return out


def _cell_relative(card_ids: list[str], day_prices: dict, base_prices: dict) -> float | None:
    """칸의 평균 상대가격. 쓸 수 있는 카드가 없으면 None."""
    rels = []
    for cid in card_ids:
        now = day_prices.get(cid)
        base = base_prices.get(cid)
        if now is not None and base:
            rels.append(now / base)
    return sum(rels) / len(rels) if rels else None


def index_point(universe: list[dict], day_prices: dict, base_prices: dict) -> dict:
    """하루치 지수와 하위지수를 낸다. 기준일 = 100.

    12칸을 균등가중한다. 시장 규모 비례가 이론적으로 낫지만 포켓몬 카드는
    유통 물량이 공개되지 않아 아무도 시장 규모를 모른다. 모르는 걸 아는 척
    가중치에 넣으면 우리가 깐 지수와 같은 짓이 된다.
    """
    cells: dict[tuple[str, str], list[str]] = {}
    for card in universe:
        cells.setdefault((card["era"], card["band"]), []).append(card["card_id"])

    rel: dict[tuple[str, str], float] = {}
    for key, ids in cells.items():
        value = _cell_relative(ids, day_prices, base_prices)
        if value is not None:
            rel[key] = value

    def mean_of(keys) -> float | None:
        vals = [rel[k] for k in keys if k in rel]
        return 100.0 * sum(vals) / len(vals) if vals else None

    return {
        "index": mean_of(list(rel)),
        "by_era": {e: mean_of([k for k in rel if k[0] == e]) for e in ERAS},
        "by_band": {b: mean_of([k for k in rel if k[1] == b]) for b in BANDS},
        "missing": sum(
            1 for c in universe
            if day_prices.get(c["card_id"]) is None or not base_prices.get(c["card_id"])
        ),
    }
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `python3 -m unittest tests.test_tcgdex_api -v`
Expected: PASS

- [ ] **Step 5: 전체 회귀 확인**

Run: `python3 -m unittest discover -s tests`
Expected: 전부 통과

- [ ] **Step 6: 커밋**

```bash
git add scripts/tcgdex_api.py tests/test_tcgdex_api.py
git commit -m "$(cat <<'EOF'
Add stratified sampling and index math for the card index

시대 4 × 가격대 3 = 12칸 층화추출과 균등가중 지수 계산을 순수 함수로
넣었습니다. 결측 이월은 7일에서 끊어 생존편향의 반대편 오류를 막습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 전수 스캔 수집기 (1회성, 예산·재개)

**Files:**
- Create: `scripts/collect_pokemon.py`
- Create: `tests/test_collect_pokemon.py`

**Interfaces:**
- Consumes: Task 3·4 의 `tcgdex_api` 전부, `rtms.gzip_bytes` / `rtms.gunzip_text`
- Produces:
  - `SCAN_STATE = ROOT / "data" / "pokemon" / "scan_state.json"`
  - `SCAN_FILE = ROOT / "data" / "pokemon" / "scan.csv.gz"`
  - `SETS_FILE = ROOT / "data" / "pokemon" / "sets.json"`
  - `PRICES_FILE = ROOT / "data" / "pokemon" / "prices.csv.gz"`
  - `UNIVERSE_FILE = ROOT / "data" / "pokemon" / "universe.json"`
  - `MIN_COVERAGE = 0.8`
  - `EXCLUDE_PATTERN` (컴파일된 정규식)
  - `is_candidate_set(name: str, release_date: str) -> bool`
  - `coverage_of(rows: list[dict], card_ids: list[str]) -> float`
  - `rows_to_csv(rows: list[dict]) -> str`
  - `merge_rows(existing: str, new_rows: list[dict]) -> str`
  - `main() -> int` — `--mode scan|daily`, `--max-calls`

**Interfaces (Task 6 이 이어받음):** Task 6 이 `--mode daily` 를 구현한다. 이 태스크는 `--mode scan` 만 만든다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_collect_pokemon.py`:

```python
"""전수 스캔 수집기의 순수 부분 검증. 네트워크는 타지 않는다.

    python3 -m unittest tests.test_collect_pokemon -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import tcgdex_api  # noqa: E402


class TestIsCandidateSet(unittest.TestCase):
    def test_accepts_regular_expansions(self) -> None:
        for name in ("Base Set", "Neo Discovery", "Flashfire", "Phantasmal Flames"):
            self.assertTrue(collect_pokemon.is_candidate_set(name, "2014-05-07"), name)

    def test_rejects_promos_and_kits(self) -> None:
        for name in ("Nintendo Black Star Promos", "DP trainer Kit (Manaphy)",
                     "Unseen Forces Unown Collection", "Miscellaneous Promos"):
            self.assertFalse(collect_pokemon.is_candidate_set(name, "2005-08-22"), name)

    def test_rejects_missing_release_date(self) -> None:
        self.assertFalse(collect_pokemon.is_candidate_set("Base Set", ""))


class TestCoverageOf(unittest.TestCase):
    def test_ratio_of_priced_cards(self) -> None:
        rows = [{"card_id": "a-1"}, {"card_id": "a-2"}]
        self.assertAlmostEqual(collect_pokemon.coverage_of(rows, ["a-1", "a-2", "a-3", "a-4"]), 0.5)

    def test_empty_set_is_zero(self) -> None:
        self.assertEqual(collect_pokemon.coverage_of([], []), 0.0)


class TestRowsToCsv(unittest.TestCase):
    def test_writes_header_and_none_as_blank(self) -> None:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "base1-4",
                    "variant": "holofoil", "tp_market": 818.65})
        out = collect_pokemon.rows_to_csv([row])
        lines = out.strip().split("\n")
        self.assertEqual(lines[0], ",".join(tcgdex_api.COLUMNS))
        self.assertIn("base1-4,holofoil,818.65", lines[1])
        self.assertTrue(lines[1].endswith(",,,,"), lines[1])


class TestMergeRows(unittest.TestCase):
    def _row(self, date: str, cid: str, market: float) -> dict:
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": date, "card_id": cid, "variant": "normal", "tp_market": market})
        return row

    def test_appends_to_existing(self) -> None:
        first = collect_pokemon.rows_to_csv([self._row("2026-08-07", "a-1", 1.0)])
        merged = collect_pokemon.merge_rows(first, [self._row("2026-08-08", "a-1", 2.0)])
        self.assertEqual(len(merged.strip().split("\n")), 3)

    def test_same_date_and_card_is_replaced_not_duplicated(self) -> None:
        first = collect_pokemon.rows_to_csv([self._row("2026-08-07", "a-1", 1.0)])
        merged = collect_pokemon.merge_rows(first, [self._row("2026-08-07", "a-1", 9.0)])
        lines = merged.strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("9.0", lines[1])

    def test_empty_existing_is_fine(self) -> None:
        merged = collect_pokemon.merge_rows("", [self._row("2026-08-07", "a-1", 1.0)])
        self.assertEqual(len(merged.strip().split("\n")), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `python3 -m unittest tests.test_collect_pokemon -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collect_pokemon'`

- [ ] **Step 3: 구현 작성**

`scripts/collect_pokemon.py`:

```python
"""포켓몬 카드 가격을 TCGdex 에서 받아 저장한다.

    python3 scripts/collect_pokemon.py --mode scan  --max-calls 6000
    python3 scripts/collect_pokemon.py --mode daily

두 단계로 나뉜다.

  scan   정규 확장팩 후보 2만여 장을 훑어 가격 분포를 만든다. 1회성이고
         --max-calls 로 하루 예산을 끊어 여러 날에 나눠 받는다.
  daily  확정된 유니버스 300장만 매일 받는다. 약 70초.

동시 요청은 4로 고정한다. 실측으로 8이면 두 배 빠르지만 무료 API 를 그렇게
두들길 이유가 없다 (4로도 2만 장에 78분).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rtms  # noqa: E402
import tcgdex_api  # noqa: E402

DATA_DIR = ROOT / "data" / "pokemon"
SCAN_STATE = DATA_DIR / "scan_state.json"
SCAN_FILE = DATA_DIR / "scan.csv.gz"
SETS_FILE = DATA_DIR / "sets.json"
PRICES_FILE = DATA_DIR / "prices.csv.gz"
UNIVERSE_FILE = DATA_DIR / "universe.json"

WORKERS = 4
TIMEOUT = 30
RETRIES = 3

# 세트가 가격 유니버스에 들어가려면 이 비율 이상이 가격을 가져야 한다.
MIN_COVERAGE = 0.8

# 1차 프리필터. 최종 판정은 가격 커버리지 실측치다 (MIN_COVERAGE).
EXCLUDE_PATTERN = re.compile(
    r"promo|trainer kit|collection|deck|tin|box|kit|misc|jumbo|energy", re.I
)


def is_candidate_set(name: str, release_date: str) -> bool:
    """정규 확장팩 후보인지. 이름 프리필터 + 발매일 유무."""
    if not release_date:
        return False
    return not EXCLUDE_PATTERN.search(name or "")


def coverage_of(rows: list[dict], card_ids: list[str]) -> float:
    """세트 카드 중 가격이 잡힌 비율."""
    if not card_ids:
        return 0.0
    priced = {r["card_id"] for r in rows}
    return len([c for c in card_ids if c in priced]) / len(card_ids)


def rows_to_csv(rows: list[dict]) -> str:
    """tcgdex_api.COLUMNS 순서로 쓴다. None 은 빈 칸."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=tcgdex_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in tcgdex_api.COLUMNS})
    return buf.getvalue()


def merge_rows(existing: str, new_rows: list[dict]) -> str:
    """기존 CSV 에 새 행을 합친다. (date, card_id) 가 같으면 새 값으로 덮는다.

    같은 날 두 번 돌려도 행이 불어나지 않아야 한다 — 수동 실행과 크론이
    겹치는 일이 실제로 생긴다.
    """
    merged: dict[tuple[str, str], dict] = {}
    if existing.strip():
        for row in csv.DictReader(io.StringIO(existing)):
            merged[(row["date"], row["card_id"])] = row
    for row in new_rows:
        merged[(row["date"], row["card_id"])] = row
    ordered = sorted(merged.values(), key=lambda r: (r["date"], r["card_id"]))
    return rows_to_csv(ordered)


def _get_json(url: str):
    """GET 후 JSON. 일시 오류는 재시도한다."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kayserdocs-pokemon/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise tcgdex_api.ApiError(f"{url} 요청 실패: {last}")


def _fetch_cards(card_ids: list[str], on_date: str) -> list[dict]:
    """카드 가격을 동시 WORKERS 개로 받는다. 실패한 카드는 조용히 빠진다."""
    def one(cid: str):
        try:
            return tcgdex_api.parse_card_pricing(_get_json(f"{tcgdex_api.BASE}/cards/{cid}"), on_date)
        except tcgdex_api.ApiError:
            return None

    with cf.ThreadPoolExecutor(WORKERS) as pool:
        return [r for r in pool.map(one, card_ids) if r]


def _read_gz(path: Path) -> str:
    return rtms.gunzip_text(path.read_bytes()) if path.exists() else ""


def _write_gz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rtms.gzip_bytes(text))


def _load_state() -> dict:
    if SCAN_STATE.exists():
        return json.loads(SCAN_STATE.read_text(encoding="utf-8"))
    return {"done_sets": [], "complete": False}


def _save_state(state: dict) -> None:
    SCAN_STATE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def run_scan(max_calls: int, on_date: str) -> int:
    """정규 확장팩 후보를 세트 단위로 훑는다. 예산이 떨어지면 중단하고 다음에 잇는다."""
    state = _load_state()
    if state.get("complete"):
        print("전수 스캔이 이미 끝났습니다. --mode daily 를 쓰세요.")
        return 0

    sets = tcgdex_api.parse_set_list(_get_json(f"{tcgdex_api.BASE}/sets"))
    done = set(state.get("done_sets") or [])
    scanned = _read_gz(SCAN_FILE)
    set_meta: dict[str, dict] = {}
    if SETS_FILE.exists():
        set_meta = json.loads(SETS_FILE.read_text(encoding="utf-8"))

    spent = 0
    for entry in sets:
        if entry["set_id"] in done:
            continue
        if spent >= max_calls:
            print(f"예산 {max_calls} 소진. 다음 실행에서 이어받습니다.")
            break

        detail = tcgdex_api.parse_set_detail(_get_json(f"{tcgdex_api.BASE}/sets/{entry['set_id']}"))
        spent += 1
        if not is_candidate_set(detail["name"], detail["release_date"]):
            set_meta[detail["set_id"]] = {
                "name": detail["name"], "release_date": detail["release_date"],
                "included": False, "reason": "이름 프리필터 제외", "coverage": None,
            }
            done.add(entry["set_id"])
            continue

        card_ids = detail["card_ids"]
        rows = _fetch_cards(card_ids, on_date)
        spent += len(card_ids)

        cov = coverage_of(rows, card_ids)
        included = cov >= MIN_COVERAGE
        set_meta[detail["set_id"]] = {
            "name": detail["name"], "release_date": detail["release_date"],
            "era": tcgdex_api.era_of(detail["release_date"]),
            "included": included,
            "reason": "" if included else f"가격 커버리지 {cov:.0%} < {MIN_COVERAGE:.0%}",
            "coverage": round(cov, 4), "card_count": len(card_ids),
        }
        if included:
            scanned = merge_rows(scanned, rows)
        done.add(entry["set_id"])
        print(f"  {detail['set_id']:<12} {detail['name'][:28]:<28} {len(rows):>4}/{len(card_ids):<4} 커버리지 {cov:.0%} {'포함' if included else '제외'}")

    _write_gz(SCAN_FILE, scanned)
    SETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETS_FILE.write_text(json.dumps(set_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    state["done_sets"] = sorted(done)
    state["complete"] = len(done) >= len(sets)
    _save_state(state)

    if state["complete"]:
        print("전수 스캔 완료. build_pokemon.py 로 유니버스를 확정하세요.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scan", "daily"), default="daily")
    parser.add_argument("--max-calls", type=int, default=6000)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    if args.mode == "scan":
        return run_scan(args.max_calls, args.date)
    return run_daily(args.date)


if __name__ == "__main__":
    raise SystemExit(main())
```

`run_daily` 는 Task 6 에서 만든다. 이 태스크에서는 아직 정의되지 않았으므로
`--mode daily` 로 부르면 `NameError` 가 난다. Task 6 이 바로 이어지므로
임시 stub 을 두지 않는다 — 죽은 코드를 남기지 않는 편이 낫다.

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `python3 -m unittest tests.test_collect_pokemon -v`
Expected: PASS

- [ ] **Step 5: 소량으로 실제 스캔이 도는지 확인**

```bash
python3 scripts/collect_pokemon.py --mode scan --max-calls 300
```

Expected: 세트 몇 개가 `포함`/`제외` 로 찍히고, 예산 소진 메시지가 나온다.

```bash
ls -la data/pokemon/
python3 -c "
import json,sys,gzip,csv,io
sys.path.insert(0,'scripts'); import rtms
print('세트 판정:', len(json.load(open('data/pokemon/sets.json'))))
rows=list(csv.DictReader(io.StringIO(rtms.gunzip_text(open('data/pokemon/scan.csv.gz','rb').read()))))
print('가격 행:', len(rows)); print('예시:', rows[0] if rows else '없음')
"
```

Expected: `sets.json`·`scan.csv.gz`·`scan_state.json` 이 생기고 행이 채워져 있다.

- [ ] **Step 6: 재개가 되는지 확인**

```bash
python3 -c "import json; s=json.load(open('data/pokemon/scan_state.json')); print('스캔 완료 세트:', len(s['done_sets']))"
python3 scripts/collect_pokemon.py --mode scan --max-calls 300
python3 -c "import json; s=json.load(open('data/pokemon/scan_state.json')); print('스캔 완료 세트:', len(s['done_sets']))"
```

Expected: 두 번째 숫자가 더 크다 (이어받았다는 뜻).

- [ ] **Step 7: gzip 재현성 확인**

```bash
md5sum data/pokemon/scan.csv.gz
python3 -c "
import sys; sys.path.insert(0,'scripts'); import rtms
from pathlib import Path
p=Path('data/pokemon/scan.csv.gz'); t=rtms.gunzip_text(p.read_bytes()); p.write_bytes(rtms.gzip_bytes(t))
"
md5sum data/pokemon/scan.csv.gz
```

Expected: 두 해시가 같다 (`mtime=0` 이 먹었다는 뜻).

- [ ] **Step 8: 커밋** (수집물은 커밋하지 않는다)

```bash
git add scripts/collect_pokemon.py tests/test_collect_pokemon.py
git commit -m "$(cat <<'EOF'
Add the full-scan collector for Pokemon card prices

정규 확장팩 후보를 세트 단위로 훑어 가격 분포를 만듭니다. --max-calls 로
하루 예산을 끊고 scan_state.json 으로 다음 실행에서 이어받습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 일일 갱신 수집기

**Files:**
- Modify: `scripts/collect_pokemon.py` (`run_daily` 추가)
- Modify: `tests/test_collect_pokemon.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 5 의 `UNIVERSE_FILE`, `PRICES_FILE`, `merge_rows`, `_fetch_cards`, `_read_gz`, `_write_gz`
- Produces:
  - `run_daily(on_date: str) -> int`
  - `load_universe_ids(universe: dict) -> list[str]`
  - `universe.json` 구조: `{"base_date": str, "seed": int, "generated": str, "cards": [{"card_id","name","set_id","set_name","era","band","base_price"}]}`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_collect_pokemon.py` 의 `if __name__` 블록 앞에 추가한다.

```python
class TestLoadUniverseIds(unittest.TestCase):
    def test_extracts_card_ids_in_order(self) -> None:
        uni = {"base_date": "2026-08-07", "cards": [
            {"card_id": "b-2"}, {"card_id": "a-1"},
        ]}
        self.assertEqual(collect_pokemon.load_universe_ids(uni), ["b-2", "a-1"])

    def test_missing_cards_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            collect_pokemon.load_universe_ids({"base_date": "2026-08-07"})
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `python3 -m unittest tests.test_collect_pokemon -v`
Expected: FAIL — `AttributeError: module 'collect_pokemon' has no attribute 'load_universe_ids'`

- [ ] **Step 3: 구현 추가**

`scripts/collect_pokemon.py` 의 `def main()` **앞에** 추가한다.

```python
def load_universe_ids(universe: dict) -> list[str]:
    """유니버스 JSON 에서 카드 id 를 순서대로 꺼낸다."""
    return [c["card_id"] for c in universe["cards"]]


def run_daily(on_date: str) -> int:
    """유니버스 카드만 받아 prices.csv.gz 에 덧쓴다."""
    if not UNIVERSE_FILE.exists():
        print(
            f"{UNIVERSE_FILE} 가 없습니다. 전수 스캔을 끝내고 build_pokemon.py 로 "
            "유니버스를 확정하세요.",
            file=sys.stderr,
        )
        return 1

    universe = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
    card_ids = load_universe_ids(universe)
    rows = _fetch_cards(card_ids, on_date)

    # 절반도 못 받았으면 API 가 이상한 것이다. 반쪽짜리 하루를 시계열에 넣으면
    # 지수가 그날만 튀고, 그 이유를 나중에 알아내기 어렵다.
    if len(rows) < len(card_ids) // 2:
        print(f"수신 {len(rows)}/{len(card_ids)} 건 — 절반 미만이라 저장하지 않습니다.", file=sys.stderr)
        return 1

    _write_gz(PRICES_FILE, merge_rows(_read_gz(PRICES_FILE), rows))
    print(f"{on_date}: {len(rows)}/{len(card_ids)} 건 저장")
    return 0
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `python3 -m unittest tests.test_collect_pokemon -v`
Expected: PASS

- [ ] **Step 5: 유니버스가 없을 때 친절히 실패하는지 확인**

```bash
python3 scripts/collect_pokemon.py --mode daily; echo "exit=$?"
```

Expected: 유니버스가 없다는 안내와 `exit=1` (Task 7 전이므로 정상)

- [ ] **Step 6: 커밋**

```bash
git add scripts/collect_pokemon.py tests/test_collect_pokemon.py
git commit -m "$(cat <<'EOF'
Add the daily universe refresh to the Pokemon collector

유니버스 300장만 매일 받아 prices.csv.gz 에 덧씁니다. 수신이 절반에 못 미치면
저장하지 않아 반쪽짜리 하루가 시계열에 들어가지 않게 했습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 집계와 유니버스 확정

**Files:**
- Create: `scripts/build_pokemon.py`
- Create: `tests/test_build_pokemon.py`

**Interfaces:**
- Consumes: Task 3·4 의 `tcgdex_api`, Task 5 의 `SCAN_FILE`/`SETS_FILE`/`PRICES_FILE`/`UNIVERSE_FILE`
- Produces:
  - `assets/pokemon/index.json` — `{"base_date","dates":[...],"index":[...],"by_era":{era:[...]},"by_band":{band:[...]},"estimated_until": str|null}`
  - `assets/pokemon/universe.json` — 화면용 구성 종목 (현재가 포함)
  - `assets/pokemon/meta.json` — `{"generated","card_count","missing","base_date","per_cell","formula","sets_included","sets_excluded"}`
  - `build_universe(scan_rows, set_meta, seed) -> dict`
  - `series_from_prices(universe, price_rows) -> dict`
  - `backcast_from_cardmarket(universe, latest_rows) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_build_pokemon.py`:

```python
"""지수 집계 검증.

    python3 -m unittest tests.test_build_pokemon -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_pokemon  # noqa: E402
import tcgdex_api  # noqa: E402


def _scan_row(cid: str, market: float) -> dict:
    row = {c: None for c in tcgdex_api.COLUMNS}
    row.update({"date": "2026-08-07", "card_id": cid, "variant": "normal", "tp_market": market})
    return row


class TestBuildUniverse(unittest.TestCase):
    def _inputs(self):
        rows, meta = [], {}
        for si, (era, rd) in enumerate(
            [("빈티지", "2001-06-01"), ("클래식", "2007-08-01"),
             ("모던", "2014-05-07"), ("최신", "2022-07-01")]
        ):
            sid = f"s{si}"
            meta[sid] = {"name": f"Set {si}", "release_date": rd, "era": era,
                         "included": True, "coverage": 1.0, "card_count": 90}
            for i in range(90):
                rows.append(_scan_row(f"{sid}-{i}", float(i + 1)))
        return rows, meta

    def test_picks_per_cell_from_every_cell(self) -> None:
        rows, meta = self._inputs()
        uni = build_pokemon.build_universe(rows, meta, seed=1, per_cell=5)
        self.assertEqual(len(uni["cards"]), 5 * 3 * 4)
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cell = [c for c in uni["cards"] if c["era"] == era and c["band"] == band]
                self.assertEqual(len(cell), 5, f"{era}/{band}")

    def test_records_base_price_and_set_name(self) -> None:
        rows, meta = self._inputs()
        uni = build_pokemon.build_universe(rows, meta, seed=1, per_cell=5)
        card = uni["cards"][0]
        self.assertGreater(card["base_price"], 0)
        self.assertTrue(card["set_name"])
        self.assertEqual(uni["seed"], 1)

    def test_excluded_sets_are_ignored(self) -> None:
        rows, meta = self._inputs()
        meta["s0"]["included"] = False
        uni = build_pokemon.build_universe(rows, meta, seed=1, per_cell=5)
        self.assertFalse([c for c in uni["cards"] if c["set_id"] == "s0"])


class TestSeriesFromPrices(unittest.TestCase):
    def _universe(self):
        cards = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                cards.append({"card_id": f"{era}-{band}", "era": era, "band": band,
                              "base_price": 10.0, "name": "x", "set_id": "s",
                              "set_name": "S"})
        return {"base_date": "2026-08-07", "seed": 1, "cards": cards}

    def _rows(self, day: str, price: float):
        out = []
        for era in tcgdex_api.ERAS:
            for band in tcgdex_api.BANDS:
                r = {c: None for c in tcgdex_api.COLUMNS}
                r.update({"date": day, "card_id": f"{era}-{band}", "tp_market": price})
                out.append(r)
        return out

    def test_base_date_is_one_hundred(self) -> None:
        uni = self._universe()
        s = build_pokemon.series_from_prices(uni, self._rows("2026-08-07", 10.0))
        self.assertEqual(s["dates"], ["2026-08-07"])
        self.assertAlmostEqual(s["index"][0], 100.0)

    def test_doubling_next_day(self) -> None:
        uni = self._universe()
        rows = self._rows("2026-08-07", 10.0) + self._rows("2026-08-08", 20.0)
        s = build_pokemon.series_from_prices(uni, rows)
        self.assertAlmostEqual(s["index"][1], 200.0)
        self.assertAlmostEqual(s["by_era"]["빈티지"][1], 200.0)

    def test_missing_day_is_carried_forward(self) -> None:
        uni = self._universe()
        rows = self._rows("2026-08-07", 10.0) + self._rows("2026-08-09", 10.0)
        # 08-08 은 통째로 결측 — 날짜 축에 없으므로 시리즈에도 없다
        s = build_pokemon.series_from_prices(uni, rows)
        self.assertEqual(s["dates"], ["2026-08-07", "2026-08-09"])


class TestBackcast(unittest.TestCase):
    def test_uses_cardmarket_averages(self) -> None:
        uni = {"base_date": "2026-08-07", "cards": [
            {"card_id": "a-1", "era": "빈티지", "band": "고가", "base_price": 10.0,
             "name": "A", "set_id": "s", "set_name": "S"},
        ]}
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "a-1", "tp_market": 10.0,
                    "cm_avg": 10.0, "cm_avg7": 8.0, "cm_avg30": 5.0})
        out = build_pokemon.backcast_from_cardmarket(uni, [row])
        self.assertEqual(len(out["dates"]), 3)
        self.assertLess(out["index"][0], out["index"][-1])
        self.assertAlmostEqual(out["index"][-1], 100.0)

    def test_no_cardmarket_yields_empty(self) -> None:
        uni = {"base_date": "2026-08-07", "cards": [
            {"card_id": "a-1", "era": "빈티지", "band": "고가", "base_price": 10.0,
             "name": "A", "set_id": "s", "set_name": "S"},
        ]}
        row = {c: None for c in tcgdex_api.COLUMNS}
        row.update({"date": "2026-08-07", "card_id": "a-1", "tp_market": 10.0})
        self.assertEqual(build_pokemon.backcast_from_cardmarket(uni, [row])["dates"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `python3 -m unittest tests.test_build_pokemon -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_pokemon'`

- [ ] **Step 3: 구현 작성**

`scripts/build_pokemon.py`:

```python
"""포켓몬 카드 가격을 지수로 집계한다.

    python3 scripts/build_pokemon.py

전수 스캔(scan.csv.gz)이 끝났는데 유니버스가 없으면 먼저 확정한다. 그 뒤
prices.csv.gz 를 읽어 지수 시계열을 굽고 assets/pokemon/*.json 을 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_pokemon  # noqa: E402
import rtms  # noqa: E402
import tcgdex_api  # noqa: E402

OUT_DIR = ROOT / "assets" / "pokemon"
SEED = 20260807

FORMULA = (
    "I_t = 100 × Σ_c w_c · (1/n_c) Σ_{i∈c} P_{i,t}/P_{i,0}   "
    "(c = 시대 4 × 가격대 3 = 12칸, w_c = 1/12 균등)"
)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = rtms.gunzip_text(path.read_bytes())
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        for col in tcgdex_api.COLUMNS[3:]:
            row[col] = float(row[col]) if row.get(col) else None
        out.append(row)
    return out


def build_universe(scan_rows: list[dict], set_meta: dict, seed: int = SEED,
                   per_cell: int = tcgdex_api.PER_CELL) -> dict:
    """스캔 결과에서 12칸 층화추출로 구성 종목을 확정한다."""
    by_set = {sid: m for sid, m in set_meta.items() if m.get("included")}
    candidates = []
    for row in scan_rows:
        sid = row["card_id"].rsplit("-", 1)[0]
        meta = by_set.get(sid)
        price = row.get("tp_market") or row.get("cm_avg")
        if not meta or not price:
            continue
        era = meta.get("era") or tcgdex_api.era_of(meta.get("release_date", ""))
        if era not in tcgdex_api.ERAS:
            continue
        candidates.append({
            "card_id": row["card_id"], "era": era, "price": price,
            "set_id": sid, "set_name": meta.get("name", ""),
        })

    picked = tcgdex_api.stratify(candidates, per_cell=per_cell, seed=seed)
    cards = [{
        "card_id": c["card_id"],
        "name": c["card_id"].rsplit("-", 1)[-1],
        "set_id": c["set_id"], "set_name": c["set_name"],
        "era": c["era"], "band": c["band"],
        "base_price": round(c["price"], 2),
    } for c in picked]

    return {
        "base_date": date.today().isoformat(),
        "seed": seed, "per_cell": per_cell,
        "generated": date.today().isoformat(),
        "cards": cards,
    }


def series_from_prices(universe: dict, price_rows: list[dict]) -> dict:
    """일자별 지수·하위지수 시계열을 만든다."""
    base = {c["card_id"]: c["base_price"] for c in universe["cards"]}
    by_date: dict[str, dict[str, float]] = {}
    for row in price_rows:
        price = row.get("tp_market") or row.get("cm_avg")
        if price:
            by_date.setdefault(row["date"], {})[row["card_id"]] = price

    dates = sorted(by_date)
    index, missing = [], []
    by_era: dict[str, list] = {e: [] for e in tcgdex_api.ERAS}
    by_band: dict[str, list] = {b: [] for b in tcgdex_api.BANDS}
    for day in dates:
        point = tcgdex_api.index_point(universe["cards"], by_date[day], base)
        index.append(point["index"])
        missing.append(point["missing"])
        for era in tcgdex_api.ERAS:
            by_era[era].append(point["by_era"][era])
        for band in tcgdex_api.BANDS:
            by_band[band].append(point["by_band"][band])

    return {"base_date": universe["base_date"], "dates": dates, "index": index,
            "by_era": by_era, "by_band": by_band, "missing": missing}


def backcast_from_cardmarket(universe: dict, latest_rows: list[dict]) -> dict:
    """Cardmarket avg7/avg30 으로 30일 소급 곡선을 추정한다.

    avg30 은 '30일 전 가격'이 아니라 '30일 평균'이다. 실제 경로와 다르므로
    화면에서 반드시 점선 + '추정' 라벨로 그린다. 이걸 실선으로 그리면 우리가
    깐 지수와 똑같은 짓이 된다.
    """
    base = {c["card_id"]: c["base_price"] for c in universe["cards"]}
    latest = {r["card_id"]: r for r in latest_rows}

    picks = [("cm_avg30", 30), ("cm_avg7", 7), ("cm_avg", 0)]
    day_maps, offsets = [], []
    for field, back in picks:
        prices = {cid: latest[cid][field] for cid in base
                  if cid in latest and latest[cid].get(field)}
        if prices:
            day_maps.append(prices)
            offsets.append(back)

    if len(day_maps) < 2:
        return {"dates": [], "index": [], "estimated": True}

    # Cardmarket 은 EUR 이고 base_price 는 USD 다. 비율만 쓰므로 통화가 상쇄되도록
    # 각 시점을 '가장 최근 Cardmarket 값' 대비 비율로 환산한 뒤 100 을 곱한다.
    anchor = day_maps[-1]
    base_date = date.fromisoformat(universe["base_date"])
    dates, index = [], []
    for prices, back in zip(day_maps, offsets):
        rels = [prices[cid] / anchor[cid] for cid in prices if anchor.get(cid)]
        if not rels:
            continue
        dates.append((base_date - timedelta(days=back)).isoformat())
        index.append(100.0 * sum(rels) / len(rels))
    return {"dates": dates, "index": index, "estimated": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    if not collect_pokemon.SETS_FILE.exists():
        print(f"{collect_pokemon.SETS_FILE} 가 없습니다. 먼저 collect_pokemon.py --mode scan 을 돌리세요.",
              file=sys.stderr)
        return 1
    set_meta = json.loads(collect_pokemon.SETS_FILE.read_text(encoding="utf-8"))

    if not collect_pokemon.UNIVERSE_FILE.exists():
        state = json.loads(collect_pokemon.SCAN_STATE.read_text(encoding="utf-8")) \
            if collect_pokemon.SCAN_STATE.exists() else {}
        if not state.get("complete"):
            print("전수 스캔이 아직 안 끝났습니다. 유니버스를 확정하지 않습니다.", file=sys.stderr)
            return 1
        universe = build_universe(_read_rows(collect_pokemon.SCAN_FILE), set_meta)
        collect_pokemon.UNIVERSE_FILE.write_text(
            json.dumps(universe, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"유니버스 확정: {len(universe['cards'])}장")
    universe = json.loads(collect_pokemon.UNIVERSE_FILE.read_text(encoding="utf-8"))

    price_rows = _read_rows(collect_pokemon.PRICES_FILE)
    series = series_from_prices(universe, price_rows)

    last_date = series["dates"][-1] if series["dates"] else None
    latest_rows = [r for r in price_rows if r["date"] == last_date] if last_date else []
    back = backcast_from_cardmarket(universe, latest_rows)

    latest_price = {r["card_id"]: (r.get("tp_market") or r.get("cm_avg")) for r in latest_rows}
    view_cards = [{**c, "price": latest_price.get(c["card_id"])} for c in universe["cards"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(json.dumps(
        {**series, "backcast": back}, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "universe.json").write_text(json.dumps(
        {**universe, "cards": view_cards}, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "generated": args.generated,
        "base_date": universe["base_date"],
        "card_count": len(universe["cards"]),
        "per_cell": universe.get("per_cell", tcgdex_api.PER_CELL),
        "seed": universe.get("seed", SEED),
        "formula": FORMULA,
        "missing": series["missing"][-1] if series["missing"] else None,
        "days": len(series["dates"]),
        "sets_included": sum(1 for m in set_meta.values() if m.get("included")),
        "sets_excluded": sum(1 for m in set_meta.values() if not m.get("included")),
        "carry_forward_days": tcgdex_api.CARRY_FORWARD_DAYS,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"지수 {len(series['dates'])}일치, 구성 {len(universe['cards'])}장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 테스트가 통과하는지 확인**

Run: `python3 -m unittest tests.test_build_pokemon -v`
Expected: PASS

- [ ] **Step 5: 전체 회귀 확인**

Run: `python3 -m unittest discover -s tests`
Expected: 전부 통과

- [ ] **Step 6: 스캔 미완 상태에서 친절히 멈추는지 확인**

```bash
python3 scripts/build_pokemon.py; echo "exit=$?"
```

Expected: 전수 스캔이 안 끝났다는 안내와 `exit=1` (Task 5 에서 300콜만 돌렸으므로 정상)

- [ ] **Step 7: 커밋**

```bash
git add scripts/build_pokemon.py tests/test_build_pokemon.py
git commit -m "$(cat <<'EOF'
Add index aggregation for the Pokemon card dashboard

전수 스캔 결과로 12칸 층화추출 유니버스를 확정하고, 일별 가격을 지수·하위지수
시계열로 굽습니다. Cardmarket avg7/avg30 소급 곡선은 추정으로 따로 표시합니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: 크론 진입점

**Files:**
- Create: `scripts/pokemon_daily.sh`

**Interfaces:**
- Consumes: Task 5·6·7 의 `collect_pokemon.py`, `build_pokemon.py`
- Produces: 크론 진입점. `AUTO_COMMIT`/`AUTO_PUSH`/`SCAN_MAX_CALLS` 환경변수를 받는다.

- [ ] **Step 1: 스크립트 작성**

`scripts/pokemon_daily.sh`:

```bash
#!/usr/bin/env bash
# 포켓몬 카드 지수 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   30 7 * * * /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/pokemon_daily.sh >> /home/dorumugs/.cache/realestate-pokemon.log 2>&1
#
# 부동산 3종과 같은 flock 을 쓴다. 넷 다 git commit/push 를 하므로 겹치면
# 한쪽 커밋이 유실된다. 07:30 은 redev_daily.sh(06:10) 가 끝난 뒤다.
#
# 스스로 갈라진다.
#   전수 스캔 미완  scan 모드로 예산만큼 훑는다 (기본 6000콜, 4일이면 끝)
#   전수 스캔 완료  daily 모드로 유니버스 300장만 받는다 (약 70초)
#
# 인증키가 필요 없다. TCGdex 는 키 없이 열려 있다.
#
# 환경변수
#   AUTO_COMMIT      1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH        1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   SCAN_MAX_CALLS   전수 스캔 1회 예산. 기본 6000

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
SCAN_MAX_CALLS="${SCAN_MAX_CALLS:-6000}"

STATE="data/pokemon/scan_state.json"

echo "===== $(date '+%F %T') 포켓몬 카드 수집 시작 ====="

FAILED=0

SCAN_DONE=0
if [ -f "$STATE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$STATE')).get('complete') else 1)"; then
  SCAN_DONE=1
fi

if [ "$SCAN_DONE" = "1" ]; then
  echo "전수 스캔 완료 상태 — 유니버스 갱신만 돌립니다."
  if ! python3 -u scripts/collect_pokemon.py --mode daily; then
    FAILED=1
    echo "일일 가격 수집 실패 — TCGdex 응답을 확인하세요." >&2
  fi
else
  echo "전수 스캔 진행 중 — 예산 ${SCAN_MAX_CALLS} 콜."
  if ! python3 -u scripts/collect_pokemon.py --mode scan --max-calls "$SCAN_MAX_CALLS"; then
    FAILED=1
    echo "전수 스캔 실패 — TCGdex 응답을 확인하세요." >&2
  fi
fi

# 집계. 수집이 일부 실패해도 있는 원본으로 다시 굽는다 — 어제 것보다 낫다.
if ! python3 -u scripts/build_pokemon.py; then
  FAILED=1
  echo "build_pokemon.py 가 비정상 종료했습니다." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

TARGETS="data/pokemon assets/pokemon"

if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 포켓몬 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh Pokemon card index data

수집 스크립트가 자동 갱신한 카드 가격과 그 지수 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh Pokemon card index data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

git add $TARGETS
git commit -m "$COMMIT_MSG"

if [ "$AUTO_PUSH" = "1" ]; then
  git pull --rebase
  git push
fi

echo "===== $(date '+%F %T') 포켓몬 카드 수집 종료 (FAILED=$FAILED) ====="
[ "$FAILED" = "1" ] && exit 1
exit 0
```

- [ ] **Step 2: 실행 권한 부여**

```bash
chmod +x scripts/pokemon_daily.sh
```

- [ ] **Step 3: 커밋 없이 한 번 돌려본다**

```bash
./scripts/pokemon_daily.sh; echo "exit=$?"
```

Expected: 스캔이 이어지고 `build_pokemon.py` 는 "전수 스캔이 아직 안 끝났습니다" 로 실패한다. `AUTO_COMMIT` 이 꺼져 있어 커밋하지 않고 `exit=1`. 정상이다.

- [ ] **Step 4: 다른 크론 스크립트와 형태가 같은지 대조**

```bash
diff <(grep -n "GIT_AUTHOR_NAME\|AUTO_PUSH\|pull --rebase" scripts/schools_monthly.sh) \
     <(grep -n "GIT_AUTHOR_NAME\|AUTO_PUSH\|pull --rebase" scripts/pokemon_daily.sh) || true
```

Expected: 줄 번호만 다르고 내용 형태가 같다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/pokemon_daily.sh
git commit -m "$(cat <<'EOF'
Add the cron entrypoint for the Pokemon card index

전수 스캔이 남았으면 스캔을, 끝났으면 일일 갱신을 돌립니다. 부동산 3종과
같은 flock 을 쓰도록 헤더에 크론 줄을 적어 두었습니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 크론 등록은 전수 스캔이 끝난 뒤에 한다**

지금 등록하지 않는다. Task 10 에서 화면까지 확인한 뒤 등록한다. 등록 명령은 Task 10 에 있다.

---

## Task 9: 대시보드 화면

**Files:**
- Create: `_pages/dashboard-pokemon.md`
- Create: `assets/pokemon/pokemon.css`
- Create: `assets/pokemon/app.js`
- Modify: `_pages/dashboard.md` (수집품 섹션 카드 추가)

**Interfaces:**
- Consumes: Task 7 의 `assets/pokemon/{index,universe,meta}.json`
- Produces: `/dashboard/pokemon/`

- [ ] **Step 1: 화면에서 쓸 JSON 을 손으로 하나 만들어 둔다**

전수 스캔이 아직 안 끝났으므로 화면 작업용 표본을 만든다. **`assets/pokemon/` 에 쓰지 않는다** — 진짜 데이터와 섞이면 안 된다.

```bash
mkdir -p /tmp/pokemon-fixture
python3 - <<'PY'
import json, math, random
from datetime import date, timedelta
from pathlib import Path
OUT = Path("/tmp/pokemon-fixture"); OUT.mkdir(exist_ok=True)
ERAS = ["빈티지","클래식","모던","최신"]; BANDS = ["고가","중가","저가"]
rng = random.Random(1)
base = date(2026,8,7); days = 40
dates = [(base + timedelta(days=i)).isoformat() for i in range(days)]
def walk(n, drift):
    v, out = 100.0, []
    for _ in range(n):
        v *= 1 + drift + rng.gauss(0, 0.01); out.append(round(v,2))
    return out
idx = walk(days, 0.001)
json.dump({"base_date": dates[0], "dates": dates, "index": idx,
  "by_era": {e: walk(days, 0.001*(i+1)) for i,e in enumerate(ERAS)},
  "by_band": {b: walk(days, 0.0005*(i+1)) for i,b in enumerate(BANDS)},
  "missing": [rng.randint(0,3) for _ in range(days)],
  "backcast": {"dates": [(base-timedelta(days=d)).isoformat() for d in (30,7,0)],
               "index": [92.0, 97.5, 100.0], "estimated": True}},
  open(OUT/"index.json","w"), ensure_ascii=False)
cards = []
for e in ERAS:
    for b in BANDS:
        for i in range(25):
            p = round(rng.uniform(1,900),2)
            cards.append({"card_id": f"{e[:2]}{b[:1]}-{i}", "name": f"카드 {i}",
                          "set_id": "s1", "set_name": f"{e} 확장팩",
                          "era": e, "band": b, "base_price": p,
                          "price": round(p*rng.uniform(0.8,1.3),2)})
json.dump({"base_date": dates[0], "seed": 20260807, "per_cell": 25, "cards": cards},
          open(OUT/"universe.json","w"), ensure_ascii=False)
json.dump({"generated": dates[-1], "base_date": dates[0], "card_count": len(cards),
  "per_cell": 25, "seed": 20260807, "days": days, "missing": 2,
  "formula": "I_t = 100 × Σ_c w_c · (1/n_c) Σ_{i∈c} P_{i,t}/P_{i,0}   (c = 시대 4 × 가격대 3 = 12칸, w_c = 1/12 균등)",
  "sets_included": 148, "sets_excluded": 70, "carry_forward_days": 7},
  open(OUT/"meta.json","w"), ensure_ascii=False)
print("표본 작성:", OUT)
PY
```

- [ ] **Step 2: 페이지 작성**

`_pages/dashboard-pokemon.md`:

```markdown
---
layout: single
title: "포켓몬 카드 가격지수"
permalink: /dashboard/pokemon/
classes: wide
author_profile: false
toc: false
description: "포켓몬 카드 시장을 시대 4 × 가격대 3 = 12칸으로 나눠 층화추출한 300장의 가격지수입니다. 구성 종목·가중치·산식을 전부 공개합니다. TCGplayer 시세를 매일 자동 갱신합니다."
---

<link rel="stylesheet" href="{{ '/assets/pokemon/pokemon.css' | relative_url }}?v={{ site.time | date: '%s' }}">

<div class="pk-app" data-base="{{ '/assets/pokemon' | relative_url }}">

  <p class="pk-lede">
    포켓몬 카드 시장을 <strong>시대 4 × 가격대 3 = 12칸</strong>으로 나누고, 각 칸에서
    25장씩 뽑은 <strong>300장</strong>의 가격을 매일 추적합니다. 칸은 균등가중입니다.
    구성 종목과 산식은 아래에 전부 적어 두었습니다.
  </p>

  <div class="pk-stats" id="pk-stats"></div>

  <div class="pk-tabs" role="tablist" aria-label="지수 선택">
    <button class="pk-tab is-on" data-series="index" role="tab" aria-selected="true">전체</button>
    <button class="pk-tab" data-series="era" role="tab" aria-selected="false">시대별</button>
    <button class="pk-tab" data-series="band" role="tab" aria-selected="false">가격대별</button>
  </div>

  <div class="pk-chart-wrap">
    <svg class="pk-chart" id="pk-chart" viewBox="0 0 720 340" preserveAspectRatio="xMidYMid meet" role="img" aria-label="가격지수 추이"></svg>
    <div class="pk-legend" id="pk-legend"></div>
  </div>

  <p class="pk-note" id="pk-note"></p>

  <h2>구성 종목</h2>
  <p class="pk-sub">지수에 들어가는 300장 전부입니다. 숨기는 종목이 없습니다.</p>

  <div class="pk-filters">
    <select id="pk-era" aria-label="시대 거르기"><option value="">시대 전체</option></select>
    <select id="pk-band" aria-label="가격대 거르기"><option value="">가격대 전체</option></select>
  </div>

  <div class="pk-table-wrap">
    <table class="pk-table" id="pk-table">
      <thead><tr>
        <th>카드</th><th>세트</th><th>시대</th><th>가격대</th>
        <th class="is-num">기준가</th><th class="is-num">현재가</th><th class="is-num">변화</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p class="pk-more"><button id="pk-more" type="button">더 보기</button></p>

  <h2>산식</h2>

  <p class="pk-formula" id="pk-formula"></p>

  <p>
    수식이 어려우면 이렇게 읽으면 됩니다. 카드마다 <strong>"기준일에 견줘 지금 몇 배냐"</strong>를
    구합니다. 기준일에 100달러였던 게 지금 120달러면 1.2배입니다. 이 배수를 칸 안에서
    평균 내면 그 칸의 성적이 나옵니다. 칸 12개 성적을 다시 평균 내고 100을 곱하면 지수입니다.
    처음이 100이고 120이 되면 시장이 20% 올랐다는 뜻입니다.
  </p>

  <h2>이 지수가 하지 않는 것</h2>

  <ul>
    <li><strong>시장 전체를 대표한다고 주장하지 않습니다.</strong> 이건 공개된 300장의 성적입니다.</li>
    <li><strong>오른 카드로 갈아타지 않습니다.</strong> 표본은 6개월 고정입니다.</li>
    <li><strong>사라진 카드를 영원히 들고 있지 않습니다.</strong> 가격이 7일 넘게 결측되면 그 카드를 빼고, 뺀 사실을 위에 표시합니다.</li>
    <li><strong>왕복 비용은 반영돼 있지 않습니다.</strong> 감정·수수료·세금은 별도입니다. <a href="{{ '/finance/포켓몬카드_투자_총비용_감정_수수료_세금/' | relative_url }}">3편</a>에서 다뤘습니다.</li>
  </ul>

  <p class="pk-source">
    출처: <a href="https://tcgdex.dev/" rel="noopener">TCGdex</a> (TCGplayer USD · Cardmarket EUR).
    이 지수는 시세 참고용이며 투자 권유가 아닙니다.
  </p>

</div>

<script src="{{ '/assets/pokemon/app.js' | relative_url }}?v={{ site.time | date: '%s' }}" defer></script>
```

- [ ] **Step 3: 스타일 작성**

`assets/pokemon/pokemon.css`. **390px 에서 가로로 넘치지 않는 것이 이 파일의 첫 요구사항이다.**

```css
/* 포켓몬 카드 지수 대시보드.
   390px 에서 가로 오버플로가 없어야 한다. 넘치는 건 .pk-table-wrap 안에서만. */

.pk-app { max-width: 100%; overflow-x: hidden; }

.pk-lede { font-size: 0.95em; line-height: 1.7; }

.pk-stats {
  display: grid; gap: 10px; margin: 1.2em 0;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}
.pk-stat {
  border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 14px;
  background: #fafafa;
}
.pk-stat b { display: block; font-size: 1.5em; line-height: 1.2; }
.pk-stat span { display: block; font-size: 0.78em; color: #666; margin-top: 4px; }
.pk-stat.is-up b { color: #c0392b; }
.pk-stat.is-down b { color: #2471a3; }

.pk-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 1em 0 0.6em; }
.pk-tab {
  border: 1px solid #d8d8d8; background: #fff; border-radius: 999px;
  padding: 6px 14px; font-size: 0.85em; cursor: pointer;
}
.pk-tab.is-on { background: #2c3e50; color: #fff; border-color: #2c3e50; }

.pk-chart-wrap { margin: 0.6em 0; }
.pk-chart { width: 100%; height: auto; display: block; }
.pk-chart .grid { stroke: #ececec; stroke-width: 1; }
.pk-chart .axis { fill: #888; font-size: 11px; }
.pk-chart .line { fill: none; stroke-width: 2; }
.pk-chart .line.is-est { stroke-dasharray: 4 4; opacity: 0.75; }

.pk-legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.8em; margin-top: 6px; }
.pk-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
.pk-legend .is-est-note { color: #888; }

.pk-note { font-size: 0.8em; color: #777; margin: 0.4em 0 1.4em; }
.pk-sub { font-size: 0.85em; color: #666; }

.pk-filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.8em 0; }
.pk-filters select { font-size: 0.85em; padding: 5px 8px; max-width: 48%; }

/* 표는 자체 스크롤 컨테이너 안에서만 넘친다. */
.pk-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.pk-table { width: 100%; min-width: 620px; border-collapse: collapse; font-size: 0.82em; }
.pk-table th, .pk-table td { padding: 7px 9px; border-bottom: 1px solid #eee; white-space: nowrap; }
.pk-table th { background: #f5f5f5; text-align: left; font-weight: 600; }
.pk-table .is-num { text-align: right; }
.pk-table .up { color: #c0392b; }
.pk-table .down { color: #2471a3; }

.pk-more { text-align: center; margin: 0.8em 0 1.6em; }
.pk-more button {
  border: 1px solid #d8d8d8; background: #fff; border-radius: 6px;
  padding: 8px 18px; font-size: 0.85em; cursor: pointer;
}

.pk-formula {
  background: #f7f7f7; border-left: 3px solid #2c3e50; padding: 12px 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78em; line-height: 1.6;
  overflow-x: auto; white-space: pre-wrap; word-break: break-word;
}

.pk-source { font-size: 0.78em; color: #777; margin-top: 1.6em; }

@media (max-width: 480px) {
  .pk-stats { grid-template-columns: repeat(2, 1fr); }
  .pk-stat b { font-size: 1.25em; }
  .pk-filters select { max-width: 100%; flex: 1 1 100%; }
}
```

- [ ] **Step 4: 스크립트 작성**

`assets/pokemon/app.js`:

```javascript
/* 포켓몬 카드 지수 대시보드.
   외부 라이브러리를 쓰지 않는다. 차트는 SVG 를 직접 그린다. */
(function () {
  'use strict';

  var app = document.querySelector('.pk-app');
  if (!app) { return; }
  var BASE = app.dataset.base;

  var COLORS = ['#2c3e50', '#c0392b', '#27ae60', '#8e44ad', '#e67e22', '#16a085'];
  var PAGE = 50;

  var state = { index: null, universe: null, meta: null, series: 'index', shown: PAGE };

  function fetchJson(name) {
    return fetch(BASE + '/' + name + '.json', { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) { throw new Error(name + ' ' + r.status); }
      return r.json();
    });
  }

  function fmt(n, digits) {
    if (n === null || n === undefined) { return '—'; }
    return Number(n).toLocaleString('ko-KR', {
      minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0
    });
  }

  function pct(now, before) {
    if (!before || now === null || now === undefined) { return null; }
    return (now / before - 1) * 100;
  }

  function lastValid(arr) {
    for (var i = arr.length - 1; i >= 0; i--) {
      if (arr[i] !== null && arr[i] !== undefined) { return arr[i]; }
    }
    return null;
  }

  function renderStats() {
    var idx = state.index, meta = state.meta;
    var series = idx.index || [];
    var now = lastValid(series);
    var d7 = series.length > 7 ? series[series.length - 8] : series[0];
    var d30 = series.length > 30 ? series[series.length - 31] : series[0];

    var cards = [
      { v: fmt(now, 1), label: '현재 지수 (기준일 ' + idx.base_date + ' = 100)', delta: null },
      { v: signed(pct(now, d7)), label: '7일 변화', delta: pct(now, d7) },
      { v: signed(pct(now, d30)), label: '30일 변화', delta: pct(now, d30) },
      { v: fmt(meta.card_count) + '장', label: '구성 종목 (12칸 × ' + meta.per_cell + ')', delta: null },
      { v: fmt(meta.days) + '일', label: '누적 관측', delta: null },
      { v: fmt(meta.missing) + '장', label: '오늘 결측', delta: null }
    ];

    document.getElementById('pk-stats').innerHTML = cards.map(function (c) {
      var cls = c.delta === null ? '' : (c.delta >= 0 ? ' is-up' : ' is-down');
      return '<div class="pk-stat' + cls + '"><b>' + c.v + '</b><span>' + c.label + '</span></div>';
    }).join('');
  }

  function signed(p) {
    if (p === null) { return '—'; }
    return (p >= 0 ? '+' : '') + p.toFixed(1) + '%';
  }

  function seriesToDraw() {
    var idx = state.index;
    if (state.series === 'era') {
      return Object.keys(idx.by_era).map(function (k, i) {
        return { name: k, values: idx.by_era[k], color: COLORS[i % COLORS.length] };
      });
    }
    if (state.series === 'band') {
      return Object.keys(idx.by_band).map(function (k, i) {
        return { name: k, values: idx.by_band[k], color: COLORS[i % COLORS.length] };
      });
    }
    return [{ name: '전체 지수', values: idx.index, color: COLORS[0] }];
  }

  function drawChart() {
    var svg = document.getElementById('pk-chart');
    var idx = state.index;
    var lines = seriesToDraw();
    var W = 720, H = 340, P = { t: 14, r: 14, b: 26, l: 44 };

    var back = idx.backcast && idx.backcast.dates && idx.backcast.dates.length
      ? idx.backcast : null;
    var allDates = (back ? back.dates.slice(0, -1) : []).concat(idx.dates);

    var vals = [];
    lines.forEach(function (l) {
      l.values.forEach(function (v) { if (v !== null && v !== undefined) { vals.push(v); } });
    });
    if (back && state.series === 'index') {
      back.index.forEach(function (v) { vals.push(v); });
    }
    if (!vals.length) { svg.innerHTML = ''; return; }

    var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
    var pad = (max - min) * 0.12 || 5;
    min -= pad; max += pad;

    function x(i) { return P.l + (W - P.l - P.r) * (allDates.length < 2 ? 0.5 : i / (allDates.length - 1)); }
    function y(v) { return P.t + (H - P.t - P.b) * (1 - (v - min) / (max - min)); }

    var parts = [];

    for (var g = 0; g <= 4; g++) {
      var gv = min + (max - min) * g / 4;
      parts.push('<line class="grid" x1="' + P.l + '" y1="' + y(gv).toFixed(1) +
        '" x2="' + (W - P.r) + '" y2="' + y(gv).toFixed(1) + '"/>');
      parts.push('<text class="axis" x="4" y="' + (y(gv) + 4).toFixed(1) + '">' + gv.toFixed(0) + '</text>');
    }

    var offset = back ? back.dates.length - 1 : 0;

    if (back && state.series === 'index') {
      var bp = back.dates.map(function (_, i) {
        return (i === 0 ? 'M' : 'L') + x(i).toFixed(1) + ' ' + y(back.index[i]).toFixed(1);
      }).join(' ');
      parts.push('<path class="line is-est" d="' + bp + '" stroke="' + COLORS[0] + '"/>');
    }

    lines.forEach(function (l) {
      var d = '', started = false;
      l.values.forEach(function (v, i) {
        if (v === null || v === undefined) { return; }
        d += (started ? ' L' : 'M') + x(i + offset).toFixed(1) + ' ' + y(v).toFixed(1);
        started = true;
      });
      if (d) { parts.push('<path class="line" d="' + d + '" stroke="' + l.color + '"/>'); }
    });

    [0, Math.floor(allDates.length / 2), allDates.length - 1].forEach(function (i, k) {
      if (i < 0 || i >= allDates.length) { return; }
      var anchor = k === 0 ? 'start' : (k === 2 ? 'end' : 'middle');
      parts.push('<text class="axis" text-anchor="' + anchor + '" x="' + x(i).toFixed(1) +
        '" y="' + (H - 8) + '">' + allDates[i].slice(5) + '</text>');
    });

    svg.innerHTML = parts.join('');

    document.getElementById('pk-legend').innerHTML =
      lines.map(function (l) {
        return '<span><i style="background:' + l.color + '"></i>' + l.name + '</span>';
      }).join('') +
      (back && state.series === 'index'
        ? '<span class="is-est-note">점선 = Cardmarket 7·30일 평균으로 만든 <strong>추정</strong> 소급 구간</span>'
        : '');

    document.getElementById('pk-note').textContent =
      '기준일 ' + state.meta.base_date + ' = 100. 실선은 실측분입니다. ' +
      '가격이 ' + state.meta.carry_forward_days + '일 넘게 결측된 카드는 지수에서 빠집니다. ' +
      '마지막 갱신 ' + state.meta.generated + '.';
  }

  function renderTable() {
    var era = document.getElementById('pk-era').value;
    var band = document.getElementById('pk-band').value;
    var rows = state.universe.cards.filter(function (c) {
      return (!era || c.era === era) && (!band || c.band === band);
    });

    var body = rows.slice(0, state.shown).map(function (c) {
      var p = pct(c.price, c.base_price);
      var cls = p === null ? '' : (p >= 0 ? 'up' : 'down');
      return '<tr>' +
        '<td>' + esc(c.name) + '</td>' +
        '<td>' + esc(c.set_name) + '</td>' +
        '<td>' + esc(c.era) + '</td>' +
        '<td>' + esc(c.band) + '</td>' +
        '<td class="is-num">$' + fmt(c.base_price, 2) + '</td>' +
        '<td class="is-num">' + (c.price ? '$' + fmt(c.price, 2) : '—') + '</td>' +
        '<td class="is-num ' + cls + '">' + signed(p) + '</td>' +
        '</tr>';
    }).join('');

    document.querySelector('#pk-table tbody').innerHTML = body ||
      '<tr><td colspan="7">해당하는 카드가 없습니다.</td></tr>';
    document.getElementById('pk-more').style.display =
      rows.length > state.shown ? '' : 'none';
    document.getElementById('pk-more').textContent =
      '더 보기 (' + state.shown + ' / ' + rows.length + ')';
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fillFilters() {
    var eras = [], bands = [];
    state.universe.cards.forEach(function (c) {
      if (eras.indexOf(c.era) < 0) { eras.push(c.era); }
      if (bands.indexOf(c.band) < 0) { bands.push(c.band); }
    });
    var es = document.getElementById('pk-era'), bs = document.getElementById('pk-band');
    eras.forEach(function (e) { es.insertAdjacentHTML('beforeend', '<option>' + esc(e) + '</option>'); });
    bands.forEach(function (b) { bs.insertAdjacentHTML('beforeend', '<option>' + esc(b) + '</option>'); });
    es.addEventListener('change', function () { state.shown = PAGE; renderTable(); });
    bs.addEventListener('change', function () { state.shown = PAGE; renderTable(); });
  }

  function bindTabs() {
    Array.prototype.forEach.call(document.querySelectorAll('.pk-tab'), function (tab) {
      tab.addEventListener('click', function () {
        Array.prototype.forEach.call(document.querySelectorAll('.pk-tab'), function (t) {
          t.classList.remove('is-on');
          t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('is-on');
        tab.setAttribute('aria-selected', 'true');
        state.series = tab.dataset.series;
        drawChart();
      });
    });
    document.getElementById('pk-more').addEventListener('click', function () {
      state.shown += PAGE;
      renderTable();
    });
  }

  Promise.all([fetchJson('index'), fetchJson('universe'), fetchJson('meta')])
    .then(function (res) {
      state.index = res[0];
      state.universe = res[1];
      state.meta = res[2];
      document.getElementById('pk-formula').textContent = state.meta.formula;
      renderStats();
      fillFilters();
      bindTabs();
      drawChart();
      renderTable();
    })
    .catch(function (err) {
      document.getElementById('pk-stats').innerHTML =
        '<p class="pk-note">데이터를 아직 불러올 수 없습니다. (' + esc(err.message) + ')</p>';
    });
})();
```

- [ ] **Step 5: 허브에 카드 추가**

`_pages/dashboard.md` 의 수집품 섹션 빈 `.rh-grid` 안을 채운다.

```markdown
  <a class="rh-card is-live" href="{{ '/dashboard/pokemon/' | relative_url }}">
    <span class="rh-badge">쓸 수 있음</span>
    <h2 class="rh-title">포켓몬 카드 가격지수</h2>
    <p class="rh-desc">
      시대 4 × 가격대 3 = <strong>12칸</strong>으로 나눠 층화추출한 <strong>300장</strong>의
      가격지수입니다. 남의 지수를 인용하지 않고 원가격에서 직접 만듭니다.
    </p>
    <ul class="rh-points">
      <li>구성 종목 300장·가중치·산식 전부 공개</li>
      <li>시대별·가격대별 하위지수</li>
      <li>매일 TCGplayer 시세 자동 갱신</li>
    </ul>
    <span class="rh-go">열어보기 →</span>
  </a>
```

- [ ] **Step 6: 표본 데이터로 화면이 그려지는지 확인**

```bash
mkdir -p assets/pokemon && cp /tmp/pokemon-fixture/*.json assets/pokemon/
mkdir -p /tmp/gemhome && printf 'source "https://rubygems.org"\ngem "github-pages", group: :jekyll_plugins\ngem "jekyll-sass-converter", "~> 2.0"\n' > /tmp/gemhome/Gemfile
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /srv/jekyll/_site_check"
test -f _site_check/dashboard/pokemon/index.html && echo "페이지 생성 OK"
```

Expected: `페이지 생성 OK`

- [ ] **Step 7: 커밋** (표본 JSON 은 커밋하지 않는다)

```bash
rm -rf _site_check
git add _pages/dashboard-pokemon.md _pages/dashboard.md assets/pokemon/pokemon.css assets/pokemon/app.js
git status --porcelain assets/pokemon/
```

Expected: `assets/pokemon/*.json` 은 스테이지에 없다. 있으면 `git reset assets/pokemon/index.json assets/pokemon/universe.json assets/pokemon/meta.json` 으로 뺀다 — 표본은 진짜 데이터가 아니다.

```bash
git commit -m "$(cat <<'EOF'
Add the Pokemon card index dashboard page

지수 차트와 구성 종목 300장 표를 그리는 화면입니다. 외부 라이브러리 없이
SVG 를 직접 그리고, 소급 추정 구간은 점선으로 구분합니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: 모바일 검증, 전수 스캔, 크론 등록

**Files:**
- 없음 (검증과 운영 등록)

**Interfaces:**
- Consumes: Task 1~9 전부

- [ ] **Step 1: 390px 가로 오버플로 검증**

표본 JSON 이 `assets/pokemon/` 에 있는 상태에서 빌드한 뒤 검사한다.

```bash
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /srv/jekyll/_site_check"
cd _site_check && python3 -m http.server 8899 &
sleep 2
```

```bash
google-chrome --headless --disable-gpu --window-size=390,900 \
  --virtual-time-budget=6000 --dump-dom http://localhost:8899/dashboard/pokemon/ > /tmp/pk.html
python3 - <<'PY'
import re
html = open('/tmp/pk.html', encoding='utf-8').read()
print('차트 SVG 렌더:', 'pk-chart' in html and '<path' in html)
print('표 행 수:', html.count('<tr>'))
PY
```

문서 폭은 CDP 로 직접 잰다.

```bash
google-chrome --headless --disable-gpu --remote-debugging-port=9222 \
  --window-size=390,900 http://localhost:8899/dashboard/pokemon/ &
sleep 4
python3 - <<'PY'
import json, urllib.request, socket, base64
tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json"))
ws = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
print("검사 대상:", ws)
PY
```

간단히 하려면 뷰포트 폭과 `scrollWidth` 를 비교하는 것으로 충분하다.

```bash
google-chrome --headless --disable-gpu --window-size=390,900 \
  --virtual-time-budget=6000 \
  --dump-dom "http://localhost:8899/dashboard/pokemon/" > /dev/null && echo "렌더 OK"
```

Expected: 아래 항목을 **눈으로** 확인한다. 스크린샷을 찍어 확인하는 게 확실하다.

```bash
google-chrome --headless --disable-gpu --window-size=390,2400 \
  --virtual-time-budget=8000 --screenshot=/tmp/pk-390.png \
  http://localhost:8899/dashboard/pokemon/
google-chrome --headless --disable-gpu --window-size=1280,2400 \
  --virtual-time-budget=8000 --screenshot=/tmp/pk-1280.png \
  http://localhost:8899/dashboard/pokemon/
echo "/tmp/pk-390.png /tmp/pk-1280.png 확인"
```

확인 항목:

| 항목 | 기대 |
|---|---|
| 페이지 전체 | 가로 스크롤바 없음 |
| 통계 타일 | 2열로 접힘, 글자 안 잘림 |
| 차트 | 폭에 맞게 줄어듦, 축 글자 겹치지 않음 |
| 구성 종목 표 | **표 안에서만** 가로 스크롤 |
| 산식 블록 | 줄바꿈되어 넘치지 않음 |
| 필터 select | 세로로 쌓임 |

넘치면 `pokemon.css` 를 고친다. `.pk-table` 의 `min-width: 620px` 은 의도된 것이고, `.pk-table-wrap` 의 `overflow-x: auto` 가 그걸 가둔다.

- [ ] **Step 2: 허브와 리다이렉트도 390px 확인**

```bash
google-chrome --headless --disable-gpu --window-size=390,2400 \
  --virtual-time-budget=6000 --screenshot=/tmp/pk-hub.png \
  http://localhost:8899/dashboard/
echo "/tmp/pk-hub.png 확인 — 부동산·수집품 두 섹션이 보이고 카드가 넘치지 않아야 한다"
```

- [ ] **Step 3: 서버 종료와 표본 제거**

```bash
kill %1 2>/dev/null || true
cd /home/dorumugs/Projects/dorumugs.github.io
rm -rf _site_check
rm -f assets/pokemon/index.json assets/pokemon/universe.json assets/pokemon/meta.json
```

표본을 반드시 지운다. 진짜 데이터가 그 자리에 들어가야 한다.

- [ ] **Step 4: 전수 스캔 완주**

이 단계는 시간이 걸린다(동시 4로 약 78분). 백그라운드로 돌린다.

```bash
nohup python3 -u scripts/collect_pokemon.py --mode scan --max-calls 30000 \
  > /tmp/pokemon-scan.log 2>&1 &
echo "PID $!"
```

진행 확인:

```bash
tail -20 /tmp/pokemon-scan.log
python3 -c "import json; s=json.load(open('data/pokemon/scan_state.json')); print('완료 세트', len(s['done_sets']), '완주', s['complete'])"
```

Expected: 마지막에 `전수 스캔 완료` 가 찍히고 `complete` 가 `True`.

- [ ] **Step 5: 유니버스 확정과 첫 집계**

```bash
python3 scripts/build_pokemon.py
```

Expected: `유니버스 확정: 300장` 이 찍힌다. (칸이 300장을 못 채우면 그보다 적을 수 있다 — 그 경우 어느 칸이 모자란지 확인한다.)

```bash
python3 -c "
import json
from collections import Counter
u = json.load(open('data/pokemon/universe.json'))
print('구성 종목:', len(u['cards']))
c = Counter((x['era'], x['band']) for x in u['cards'])
for k in sorted(c): print(' ', k, c[k])
"
```

Expected: 12칸이 각 25장. 모자란 칸이 있으면 그 칸의 모집단이 25장보다 작다는 뜻이므로 정상이다 — `sets.json` 에서 그 시대에 포함된 세트를 확인해 기록해 둔다.

- [ ] **Step 6: 첫 일일 수집**

```bash
python3 scripts/collect_pokemon.py --mode daily
python3 scripts/build_pokemon.py
python3 -c "
import json
m = json.load(open('assets/pokemon/meta.json'))
i = json.load(open('assets/pokemon/index.json'))
print('갱신', m['generated'], '구성', m['card_count'], '장, 관측', m['days'], '일')
print('지수', i['index'])
print('소급 추정', i['backcast']['dates'], i['backcast']['index'])
"
```

Expected: 지수 첫 값이 100 근처(같은 날 기준가로 잡히므로 정확히 100). 소급 추정 3점이 나온다.

- [ ] **Step 7: 실데이터로 화면 재확인**

Step 1 의 빌드·스크린샷을 실데이터로 한 번 더 돌린다. 표본과 달리 카드 이름이 id 뒷자리라 짧을 수 있다 — 표가 허전하면 `build_pokemon.py` 의 `name` 을 카드 실명으로 채우는 개선을 남긴다(이번 범위 밖, 스캔 시 이름을 저장하지 않았기 때문).

- [ ] **Step 8: 전체 테스트 회귀**

```bash
python3 -m unittest discover -s tests
```

Expected: 전부 통과

- [ ] **Step 9: 키가 섞이지 않았는지 확인**

```bash
git grep -I -lE 'sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' -- scripts/ assets/pokemon/ _pages/ || echo "키 없음"
```

Expected: `키 없음` (TCGdex 는 키를 안 쓰므로 당연하지만 규칙상 확인한다)

- [ ] **Step 10: 데이터 커밋**

```bash
git add data/pokemon assets/pokemon
git commit -m "$(cat <<'EOF'
Add the first Pokemon card index snapshot

전수 스캔으로 정규 확장팩을 가려내고 12칸 층화추출로 구성 종목 300장을
확정했습니다. 기준일 지수와 첫 집계본입니다.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11: 크론 등록**

기존 3줄 뒤에 한 줄을 더한다. **기존 줄을 건드리지 않는다.**

```bash
crontab -l > /tmp/crontab.bak
cp /tmp/crontab.bak /tmp/crontab.new
cat >> /tmp/crontab.new <<'CRON'
30 7 * * * /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/pokemon_daily.sh >> /home/dorumugs/.cache/realestate-pokemon.log 2>&1
CRON
diff /tmp/crontab.bak /tmp/crontab.new
crontab /tmp/crontab.new
crontab -l | tail -4
```

Expected: `diff` 가 추가 한 줄만 보여주고, `crontab -l` 마지막에 새 줄이 있다.

- [ ] **Step 12: 크론이 부를 형태 그대로 한 번 돌려본다** (커밋 없이)

```bash
/usr/bin/flock -w 60 /home/dorumugs/.cache/realestate.lock \
  /home/dorumugs/Projects/dorumugs.github.io/scripts/pokemon_daily.sh
echo "exit=$?"
```

Expected: `전수 스캔 완료 상태 — 유니버스 갱신만 돌립니다` → 수집·집계 성공 → `AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다` → `exit=0`

- [ ] **Step 13: 푸시**

```bash
git log --oneline -8
git push
```

배포 후 확인:

| 주소 | 기대 |
|---|---|
| `https://dorumugs.github.io/dashboard/` | 부동산·수집품 두 섹션 |
| `https://dorumugs.github.io/dashboard/pokemon/` | 지수 차트와 구성 종목 표 |
| `https://dorumugs.github.io/real-estate/` | `/dashboard/` 로 리다이렉트 |
| `https://dorumugs.github.io/real-estate/trades/` | 새 주소로 리다이렉트 |

GitHub Pages 빌드에 1~2분 걸린다.

---

## 자체 검토

**스펙 커버리지**

| 스펙 절 | 태스크 |
|---|---|
| 1. 데이터 소스 (TCGdex, 파싱) | Task 3 |
| 1. 정규 확장팩 판정 (80% 커버리지) | Task 5 (`coverage_of`, `MIN_COVERAGE`) |
| 2. 층 구조·산식·하위지수 | Task 4 |
| 2. 기준일·소급 구간 | Task 7 (`backcast_from_cardmarket`) |
| 2. 생존편향 처리 (7일 이월) | Task 4 (`fill_forward`), Task 9 (화면 표시) |
| 2. 공개 원칙 | Task 9 (구성 종목 표·산식·결측 수·갱신 시각) |
| 3. URL 이전·리다이렉트·허브 | Task 1, 2 |
| 4. 파이프라인 파일 구조 | Task 3~7 |
| 4. 저장 (prices.csv.gz, mtime=0) | Task 5 (`merge_rows`, `rtms.gzip_bytes`) |
| 4. 산출물 3종 JSON | Task 7 |
| 5. 크론 (같은 flock, 07:30) | Task 8, Task 10 Step 11 |
| 6. 테스트 | Task 3~7 각 테스트 스텝 |
| 7. 390px 검증 | Task 10 Step 1~2 |

**빠진 것 하나를 기록해 둔다.** 스펙 2절의 **6개월 리밸런싱**과
`universe_history.json` 은 이번 범위에 태스크가 없다. 첫 리밸런싱이 2027-02
이므로 지금 만들 이유가 없다(YAGNI). 대신 `universe.json` 에 `base_date` 와
`seed` 를 남겨 두었으므로 그때 이력 파일을 붙이면 된다. 스펙에 이 사실을
적어 둔다.

**타입 일관성 확인**

- `tcgdex_api.COLUMNS` 는 Task 3 에서 정의, Task 5 `rows_to_csv`·Task 7 `_read_rows` 가 같은 이름으로 쓴다. ✓
- `stratify` 는 `{"card_id","era","price"}` 를 받아 `"band"` 를 더해 돌려준다. Task 7 `build_universe` 가 그대로 쓴다. ✓
- `index_point` 는 `{"index","by_era","by_band","missing"}` 를 돌려준다. Task 7 `series_from_prices` 가 그 키로 읽는다. ✓
- `universe.json` 의 `cards[]` 키(`card_id,name,set_id,set_name,era,band,base_price`)는 Task 6 `load_universe_ids`, Task 7 `series_from_prices`, Task 9 `renderTable` 이 공유한다. Task 7 이 화면용에 `price` 를 더한다. ✓
- `meta.json` 키(`generated,base_date,card_count,per_cell,seed,formula,missing,days,sets_included,sets_excluded,carry_forward_days`)는 Task 7 이 쓰고 Task 9 `renderStats`/`drawChart` 가 읽는다. ✓
