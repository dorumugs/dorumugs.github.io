"""한국은행 ECOS 통계 응답 파싱. 순수함수만 — I/O 없다.

    GET /api/StatisticSearch/{KEY}/json/kr/1/1000/{통계표}/M/{시작}/{종료}/{항목}

쓰는 계열 둘:

| 계열 | 통계표 | 항목 | 시작 |
|---|---|---|---|
| 한국은행 기준금리 | `722Y001` | `0101000` | 1999-05 |
| 예금은행 주택담보대출 금리 | `121Y006` | `BECBLA0302` | 2001-09 |

**오류가 HTTP 200 과 함께 온다.** 키가 없거나 한도를 넘으면 데이터 대신
`{"RESULT": {"CODE": "INFO-100", …}}` 가 온다. 상태코드만 보면 실패를 놓치고
빈 시계열을 정상으로 취급한다.
"""

from __future__ import annotations

# 금리가 이 범위 밖이면 응답이 이상한 것이다. 0% 는 있을 수 있으므로 하한은 0.
SANE_RATE = (0.0, 30.0)


def parse_month(raw) -> str | None:
    """`"202601"` → `"2026-01"`."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if len(text) != 6 or not text.isdigit():
        return None
    if not 1 <= int(text[4:]) <= 12:
        return None
    return f"{text[:4]}-{text[4:]}"


def parse_series(payload) -> tuple[list[tuple[str, float]], str | None]:
    """응답 → (월 오름차순 (월, 값) 목록, 오류 사유). 정상이면 오류는 None."""
    if not isinstance(payload, dict):
        return [], "응답이 객체가 아닙니다."

    result = payload.get("RESULT")
    if isinstance(result, dict):
        code = str(result.get("CODE") or "").strip()
        message = str(result.get("MESSAGE") or "").strip()
        return [], f"{code}: {message}".strip(": ") or "ECOS 오류 응답입니다."

    container = payload.get("StatisticSearch")
    if not isinstance(container, dict):
        return [], "StatisticSearch 가 없습니다."
    rows = container.get("row")
    if not isinstance(rows, list):
        return [], "row 가 배열이 아닙니다."

    low, high = SANE_RATE
    points: dict[str, float] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        month = parse_month(item.get("TIME"))
        if month is None:
            continue
        try:
            value = float(item.get("DATA_VALUE"))
        except (TypeError, ValueError):
            continue
        if not low <= value <= high:
            continue
        points[month] = value

    return sorted(points.items()), None
