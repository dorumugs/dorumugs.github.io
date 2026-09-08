"""통계누리 주택유형별 착공실적(formId 5387) 응답 파싱. 순수함수만 — I/O 없다.

응답은 컬럼 인덱스가 문자열 키인 평평한 배열이다.

    {"result": true,
     "data": [{"0": "2026-06 p)", "1": "서울", "2": "아파트",
               "3": "아파트", "4": "아파트", "5": "1605"}]}

컬럼 의미는 `/portal/stat/columns.do?formId=5387&styleNum=1` 이 준다.
`0`=월, `1`=지역, `2`=대분류, `5`=착공실적.

이 파일이 막는 함정 넷 (전부 실제 응답에서 확인했다):

1. **60개월 초과 오류가 HTTP 200 으로 온다.** `result` 를 안 보면 빈 데이터가 된다.
2. **잠정치 `p)`** — 최근 약 10개월이 `"2026-07 p)"` 로 오고 확정되며 값이 바뀐다.
3. **집계행 혼입** — 지역 라벨 23종 중 5종이 합계행이다. 화이트리스트로 막는다.
4. **`'-'` 결측** — 세종은 2011-01 부터 행이 있지만 값이 `'-'` 다. 0 이 아니다.

광주·전남 합산은 **여기서 하지 않는다.** 파서는 원본에 충실하고, 통합 처리는
`build_supply.py` 가 맡는다.
"""

from __future__ import annotations

from typing import NamedTuple

# 실제 시도만 통과시킨다. 블랙리스트로 짜면 나중에 합계행이 하나 늘어날 때
# 조용히 섞인다. `전남광주` 는 2026-07 통합으로 생긴 라벨이며 같은 달에
# `광주`·`전남` 과 동시에 오지 않는다.
SIDO = frozenset({
    "서울", "인천", "경기", "부산", "대구", "광주", "대전", "울산", "세종",
    "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    "전남광주",
})

# 시도가 아니지만 남긴다 — `build_supply.py` 가 16개 시도의 합과 대조해
# 화이트리스트에서 시도가 빠졌는지 잡는 데 쓴다. 개수 검사로는 못 잡는다.
TOTAL_LABEL = "총계"

CATEGORY = "아파트"

# 한 달 전국 아파트 착공 합계가 이 범위 밖이면 응답이 이상한 것으로 보고
# 그 달을 통째로 버린다. 부분적으로 틀린 시계열이 조용히 섞이는 게 더 위험하다.
SANE_MONTH_TOTAL = (1, 200000)


class Row(NamedTuple):
    month: str          # "2026-07"
    region: str         # "서울" · "전남광주" · "총계"
    units: int | None   # None 은 결측. 0 은 관측값이고 음수는 하향 정정이다
    provisional: bool


def parse_month(raw: str) -> tuple[str, bool] | None:
    """`"2026-07 p)"` → `("2026-07", True)`, `"2025-09"` → `("2025-09", False)`."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    provisional = False
    if "p)" in text:
        provisional = True
        text = text.replace("p)", "").strip()
    if len(text) != 7 or text[4] != "-":
        return None
    year, month = text[:4], text[5:]
    if not (year.isdigit() and month.isdigit()):
        return None
    if not 1 <= int(month) <= 12:
        return None
    return text, provisional


def parse_units(raw) -> int | None:
    """`'-'`·빈값·비숫자는 결측이다. 0 과 섞지 않는다.

    **음수는 결측이 아니다.** 통계누리는 하향 정정을 음수 월값으로 준다 —
    2011-12 충남이 `-1494` 이고, 그 값을 버리면 시도 합이 `총계` 와 어긋난다.
    12개월 이동합계 안에서 앞달의 과대계상을 상쇄하므로 그대로 들고 간다.
    (합계 대조가 이걸 잡아냈다. 처음엔 음수를 결측으로 버렸다.)
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace(",", "")
    if not text or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_starts(payload) -> tuple[list[Row], str | None]:
    """응답 → (행 목록, 오류 사유). 정상이면 오류는 None."""
    if not isinstance(payload, dict):
        return [], "응답이 객체가 아닙니다."
    if payload.get("result") is not True:
        msg = str(payload.get("msg") or "").strip()
        return [], msg or "result 가 false 입니다."
    data = payload.get("data")
    if not isinstance(data, list):
        return [], "data 가 배열이 아닙니다."

    rows: list[Row] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("2") != CATEGORY:
            continue
        region = item.get("1")
        if region not in SIDO and region != TOTAL_LABEL:
            continue
        month = parse_month(item.get("0"))
        if month is None:
            continue
        rows.append(Row(month[0], region, parse_units(item.get("5")), month[1]))

    return _drop_insane_months(rows), None


def _drop_insane_months(rows: list[Row]) -> list[Row]:
    """정상성 게이트. 시도 합계가 범위 밖인 달은 통째로 버린다.

    `총계` 행은 합에서 뺀다 — 넣으면 두 번 센다.
    """
    low, high = SANE_MONTH_TOTAL
    totals: dict[str, int] = {}
    for row in rows:
        if row.region == TOTAL_LABEL or row.units is None:
            continue
        totals[row.month] = totals.get(row.month, 0) + row.units

    # 값이 하나도 없는 달은 여기서 판단하지 않는다 — 게시 전이라 아직 안 채워진
    # 달과 응답이 깨진 달을 파서가 구분할 수 없다. 결측인 채로 넘겨 build 가 본다.
    bad = {m for m, total in totals.items() if not low <= total <= high}
    return [r for r in rows if r.month not in bad]
