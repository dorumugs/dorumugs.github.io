"""착공 × 금리 집계. `assets/realestate/supply.json` 을 만든다.

    python3 scripts/build_supply.py

입력은 `data/supply/starts.json`(통계누리 착공)과 `data/supply/rates.json`(ECOS 금리).

집계에서 결정한 것 넷:

**광주·전남은 전 기간 합산한다.** 2026-07 전남광주통합특별시 출범으로 그 달부터
`광주`·`전남` 행이 사라지고 `전남광주` 한 행만 온다. 15년 시계열 한가운데서 지역
축이 바뀌므로, 이전 구간도 두 값을 더해 하나로 다룬다. 그래야 평년 기준선의
분모가 시계열 내내 일관된다. 시도는 17개가 아니라 **16개**가 된다.

**12개월 이동합계로 본다.** 착공은 단발 대규모 사업 하나에 월값이 통째로 흔들려
원계열은 톱니다. 12개월 합계는 계절성을 정의상 제거하고 단위가 "최근 1년간 착공
호수"라 직관적이다. 창 안에 결측이 하나라도 있으면 그 지점은 결측이다.

**평년 기준선은 2011-12 ~ 2019-12 다.** 2020 이후는 코로나와 금리 급등이 겹친
예외 구간이라 기준에 넣으면 "평년 대비"라는 말이 무의미해진다.

**전국은 `총계` 행을 쓰지 않고 16개 시도를 직접 더한다.** `총계` 는 대조에만 쓴다.
전남광주 같은 통합이 또 일어나면 새 라벨이 화이트리스트에 없어 조용히 누락되는데,
지역 개수를 세는 검사로는 이걸 못 잡는다 — 지역 수가 줄어도 나머지는 멀쩡하기
때문이다. 합계 대조가 이걸 잡는 유일한 장치다.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from molit_stat_api import SIDO, TOTAL_LABEL, Row  # noqa: E402

STARTS_FILE = ROOT / "data" / "supply" / "starts.json"
RATES_FILE = ROOT / "data" / "supply" / "rates.json"
OUT_FILE = ROOT / "assets" / "realestate" / "supply.json"

WINDOW = 12
BASELINE_FROM = "2011-12"
BASELINE_TO = "2019-12"
BASELINE_MIN_MONTHS = 60

# 시도 하나가 화이트리스트에서 빠져도 나머지는 멀쩡하므로 개수로는 못 잡는다.
# 합계가 이 비율 넘게 어긋나면 빌드를 세운다.
TOTAL_TOLERANCE = 0.01

MERGED = ("광주", "전남")
MERGED_NAME = "전남광주"

# 행정표준코드. `전남광주` 는 아직 배정된 코드가 없어 두 코드를 이어 붙여 쓴다 —
# 어느 한쪽 코드를 빌려 쓰면 통합 이전 구간과 헷갈린다.
CODES = {
    "서울": "11", "부산": "26", "대구": "27", "인천": "28", "대전": "30",
    "울산": "31", "세종": "36", "경기": "41", "강원": "51", "충북": "43",
    "충남": "44", "전북": "52", "경북": "47", "경남": "48", "제주": "50",
    MERGED_NAME: "4629",
}

NOTES = [
    "시군구 단위 착공 통계는 공개되지 않아 시도까지만 본다",
    "광주·전남은 2026-07 통합에 따라 전 기간 합산했다",
    "최근 약 10개월은 잠정치이며 확정되면서 값이 바뀐다",
    "평년 지수의 기준선은 2011~2019 평균이다",
    "세종은 2012년 하반기부터 착공이 잡힌다",
    "두 패널의 시각적 대응은 상관이지 인과가 아니다",
]


class BuildError(Exception):
    """집계를 계속하면 안 되는 상태. 틀린 숫자를 내느니 세운다."""


def month_range(first: str, last: str) -> list[str]:
    """`"2011-01"`, `"2011-03"` → 세 달. 빈 달도 축에는 자리를 남긴다."""
    year, month = int(first[:4]), int(first[5:])
    out: list[str] = []
    while True:
        current = f"{year:04d}-{month:02d}"
        out.append(current)
        if current >= last:
            return out
        month += 1
        if month == 13:
            year, month = year + 1, 1


def rolling_sum(values: list, window: int = WINDOW) -> list:
    """창 안에 결측이 하나라도 있으면 그 지점은 결측. 0 으로 때우지 않는다."""
    out: list = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
            continue
        chunk = values[i + 1 - window:i + 1]
        out.append(None if any(v is None for v in chunk) else sum(chunk))
    return out


def merge_unification(rows: list[Row]) -> dict[tuple[str, str], int | None]:
    """(월, 지역) → 세대수. 광주·전남을 `전남광주` 하나로 합친다.

    둘 중 하나만 값이 있는 달은 합산하지 않고 결측으로 둔다. 반쪽만 더한 값은
    통합 이전 구간을 조용히 과소계상한다.
    """
    out: dict[tuple[str, str], int | None] = {}
    halves: dict[str, dict[str, int | None]] = {}

    for row in rows:
        if row.region == TOTAL_LABEL:
            continue
        if row.region in MERGED:
            halves.setdefault(row.month, {})[row.region] = row.units
        else:
            out[(row.month, row.region)] = row.units

    for month, parts in halves.items():
        key = (month, MERGED_NAME)
        if key in out:
            continue  # 통합 라벨이 이미 왔다. 그 값이 우선이다
        values = [parts.get(name) for name in MERGED]
        out[key] = None if any(v is None for v in values) else sum(values)

    return out


def fill_observed_gaps(values: list, delivered: list[bool] | None = None) -> list:
    """관측이 시작된 뒤의 결측을 0 으로 채운다. 그 앞은 결측으로 둔다.

    통계누리의 `'-'` 에는 두 가지가 섞여 있다 — "그 달 착공이 0" 과 "그때 그
    시도가 아직 없었다". 응답만 봐서는 구분이 안 된다.

    구분하는 근거는 **합계 대조**다. `'-'` 를 빼고 더한 시도 합이 187개월 전부
    `총계` 와 맞는다. 즉 관측 구간 안의 `'-'` 는 실제로 0 이다. 반면 세종의
    2011~2012 처럼 첫 관측 이전 구간은 시도 자체가 없던 때이므로 0 이 아니다 —
    0 으로 깔면 평년 기준선이 바닥으로 내려가 지수가 폭주한다.

    채우지 않으면 결측 하나가 12개월 창을 12번 죽여 대전·충북처럼 `'-'` 가
    흩뿌려진 시도는 지수를 통째로 잃는다.

    `delivered` 는 "응답이 그 달을 실제로 내려줬는가"다. 응답에 아예 없던 달은
    `'-'` 와 다르다 — 0 을 그려 넣으면 없던 관측을 만들어 내는 셈이므로 비워 둔다.
    """
    first = next((i for i, v in enumerate(values) if v is not None), None)
    if first is None:
        return list(values)
    out = list(values)
    for i, value in enumerate(out):
        if value is None and i > first and (delivered is None or delivered[i]):
            out[i] = 0
    return out


def baseline_index(mavg: list, months: list[str], start: str, end: str,
                   min_months: int) -> list:
    """평년 기준선을 100 으로 놓은 지수.

    유효 개월이 문턱 미만이면 전부 `None` 이다. 짧은 기준선으로 만든 지수는
    숫자만 그럴듯하고 뜻이 없다.
    """
    window = [v for m, v in zip(months, mavg) if start <= m <= end and v is not None]
    if len(window) < min_months:
        return [None] * len(mavg)
    base = sum(window) / len(window)
    if base <= 0:
        return [None] * len(mavg)
    return [None if v is None else round(100.0 * v / base, 1) for v in mavg]


def cross_check_total(rows: list[Row]) -> str | None:
    """시도 합과 응답의 `총계` 를 대조한다. 어긋나면 사유, 맞으면 None.

    결측은 0 으로 보고 더한다 — 세종의 `'-'` 는 실제로 착공이 없던 달이므로
    합계가 맞는다. 값이 있는데 결측으로 온 거라면 여기서 어긋나 잡힌다.
    """
    regional: dict[str, int] = {}
    totals: dict[str, int] = {}
    for row in rows:
        if row.units is None:
            continue
        if row.region == TOTAL_LABEL:
            totals[row.month] = row.units
        elif row.region in SIDO:
            regional[row.month] = regional.get(row.month, 0) + row.units

    for month, total in sorted(totals.items()):
        got = regional.get(month, 0)
        if total == 0:
            continue
        if abs(got - total) / total > TOTAL_TOLERANCE:
            return (f"{month}: 시도 합 {got:,} 이 총계 {total:,} 과 "
                    f"{abs(got - total):,} 만큼 다릅니다. "
                    f"시도 라벨이 바뀌었거나 화이트리스트에서 빠졌습니다.")
    return None


def _region_order(names: set[str]) -> list[str]:
    return sorted(names, key=lambda n: (CODES.get(n, "zz"), n))


def build(rows: list[Row], rates: dict, today: str) -> dict:
    """행 목록과 금리 두 계열 → 화면이 그대로 먹는 집계."""
    if not rows:
        raise BuildError("착공 데이터가 비어 있습니다.")

    problem = cross_check_total(rows)
    if problem:
        raise BuildError(problem)

    units_by_key = merge_unification(rows)
    months = month_range(min(r.month for r in rows), max(r.month for r in rows))
    names = _region_order({region for _, region in units_by_key})
    delivered = [m in {r.month for r in rows} for m in months]

    series: list[dict] = []
    for name in names:
        units = fill_observed_gaps([units_by_key.get((m, name)) for m in months],
                                   delivered)
        series.append({"code": CODES.get(name, "zz"), "name": name, "units": units})

    nation: list = []
    for i, _ in enumerate(months):
        present = [s["units"][i] for s in series if s["units"][i] is not None]
        nation.append(sum(present) if present else None)
    series.insert(0, {"code": "00", "name": "전국", "units": nation})

    for entry in series:
        entry["mavg"] = rolling_sum(entry["units"], WINDOW)
        entry["index"] = baseline_index(entry["mavg"], months, BASELINE_FROM,
                                        BASELINE_TO, BASELINE_MIN_MONTHS)

    provisional = sorted(r.month for r in rows if r.provisional)

    return {
        "generated": today,
        "latest_month": months[-1],
        "provisional_from": provisional[0] if provisional else None,
        "baseline": {"from": BASELINE_FROM, "to": BASELINE_TO,
                     "min_months": BASELINE_MIN_MONTHS},
        "months": months,
        "regions": series,
        "rates": {key: _align(rates.get(key) or [], months)
                  for key in ("base", "mortgage")},
        "notes": NOTES,
    }


def _align(points, months: list[str]) -> list:
    """(월, 값) 목록을 월 축에 맞춘다. 없는 달은 `None`."""
    lookup = dict(points)
    return [lookup.get(m) for m in months]


def _load_rows(path: Path) -> list[Row]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Row(m, r, u, bool(p)) for m, r, u, p in data["rows"]]


def main() -> int:
    for path in (STARTS_FILE, RATES_FILE):
        if not path.exists():
            print(f"✖ {path.relative_to(ROOT)} 이 없습니다. "
                  f"먼저 collect_supply.py 를 돌리세요.", file=sys.stderr)
            return 1

    rows = _load_rows(STARTS_FILE)
    rates_raw = json.loads(RATES_FILE.read_text(encoding="utf-8"))
    rates = {key: [tuple(p) for p in rates_raw.get(key) or []]
             for key in ("base", "mortgage")}

    try:
        out = build(rows, rates, date.today().isoformat())
    except BuildError as exc:
        print(f"✖ 집계를 세웁니다 — {exc}", file=sys.stderr)
        return 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    size = OUT_FILE.stat().st_size
    print(f"✔ {OUT_FILE.relative_to(ROOT)} — {len(out['regions'])}계열 × "
          f"{len(out['months'])}개월, {size // 1024}KB, 최신 {out['latest_month']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
