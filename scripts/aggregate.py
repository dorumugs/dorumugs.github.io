"""대시보드 집계에 쓰는 순수 함수. I/O 를 하지 않아 단독으로 검증할 수 있다."""

from __future__ import annotations

import statistics

# 1평 = 3.3058㎡. 저장소 전체에서 이 값을 쓴다.
PYEONG = 3.3058

# 월 거래가 이 미만이면 '얇은 달'로 표시하고 변화율은 이동중위로 낸다.
THIN_SAMPLE = 5

# 전용면적 구간 경계 (㎡)
AREA_EDGES = (60.0, 85.0, 135.0)


def pyeong_price(price_10k: int, area_sqm: float) -> float | None:
    """만원/평. 면적이 0 이하면 None."""
    if area_sqm <= 0:
        return None
    return price_10k / (area_sqm / PYEONG)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def pct_change(now: float | None, before: float | None) -> float | None:
    """퍼센트 변화율. 어느 쪽이든 없거나 기준이 0 이면 None."""
    if now is None or before is None or before == 0:
        return None
    return (now / before - 1) * 100


def rolling_median(series: list[float | None], window: int) -> list[float | None]:
    """창 안의 None 은 무시하고 중위값을 낸다. 창이 다 차기 전 구간은 None."""
    out: list[float | None] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = [v for v in series[i + 1 - window:i + 1] if v is not None]
        out.append(statistics.median(chunk) if chunk else None)
    return out


def from_peak(series: list[float | None], index: int) -> float | None:
    """해당 시점까지의 역대 최고 대비 몇 % 인지.

    고점은 index 이하 구간에서만 찾는다. 미래의 고점을 끌어오면
    '지금 전고점 대비 얼마'라는 뜻이 깨진다.
    """
    now = series[index]
    if now is None:
        return None
    past = [v for v in series[:index + 1] if v is not None]
    if not past:
        return None
    peak = max(past)
    if peak == 0:
        return None
    return (now / peak - 1) * 100


def turnover(trade_count: int, households: int) -> float | None:
    """세대수 대비 거래 회전율(%)."""
    if households <= 0:
        return None
    return trade_count / households * 100


def area_bucket(area_sqm: float) -> int:
    """0: ~60, 1: 60~85, 2: 85~135, 3: 135~"""
    for i, edge in enumerate(AREA_EDGES):
        if area_sqm < edge:
            return i
    return len(AREA_EDGES)
