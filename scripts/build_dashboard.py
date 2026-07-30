"""수집한 원본에서 대시보드용 집계 JSON 을 만든다.

    python3 scripts/build_dashboard.py

원본은 건드리지 않는다. 지표 정의가 바뀌면 이 스크립트만 다시 돌린다.
출력은 결정론적이라 값이 바뀌지 않은 파일은 건드리지 않고 넘어간다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

TRADES_DIR = ROOT / "data" / "trades"
COMPLEX_FILE = ROOT / "data" / "complexes.csv.gz"
OUT_DIR = ROOT / "assets" / "realestate"
SUMMARY_FILE = OUT_DIR / "summary.json"

HOUSEHOLD_MIN = 300

# 중위 평당가를 낼 때 함께 묶는 개월 수. 3이면 그 달 + 직전 2개월이다.
# 창의 중심이 한 달 전이라 전환점이 그만큼 늦게 잡히는 대신, 거래 구성이
# 바뀌며 생기는 가짜 등락이 크게 줄어든다.
MEDIAN_WINDOW = 3
# 실측 347KB(gzip 전송 126KB). 247개월 × 72구 × 2필터를 온전히 담으면 이 정도다.
# 첫 로딩에서 실제로 오가는 건 gzip 크기이고, 이후 지표를 더 얹을 여유도 남겨 둔다.
MAX_SUMMARY_BYTES = 400 * 1024

_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_name(name: str) -> str:
    """단지명 2차 조인용 정규화. 괄호 꼬리표와 공백을 없앤다.

    '현대5차(71,72동)' 와 '현대5차' 가 같은 단지로 붙게 한다.
    """
    return _SUFFIX_RE.sub("", (name or "").strip()).replace(" ", "")


def load_complexes() -> tuple[dict[str, dict], dict[tuple[str, str, str], int]]:
    """단지 마스터를 PNU 인덱스와 (시군구, 법정동, 정규화 단지명) 인덱스로 읽는다."""
    rows = rtms.csv_to_rows(rtms.gunzip_text(COMPLEX_FILE.read_bytes()))
    by_pnu: dict[str, dict] = {}
    by_name: dict[tuple[str, str, str], int] = {}
    for row in rows:
        try:
            hh = int(row.get("household_count") or 0)
        except ValueError:
            continue
        pnu = row.get("pnu") or ""
        sgg = row.get("sgg_cd") or ""
        name = row.get("complex_name") or ""
        # address 는 '서울특별시 강남구 역삼동 755-1' 꼴. 법정동은 뒤에서 두 번째 토큰.
        parts = (row.get("address") or "").split()
        dong = parts[-2] if len(parts) >= 2 else ""
        if pnu:
            by_pnu[pnu] = {"name": name, "dong": dong, "hh": hh, "sgg": sgg}
        key = (sgg, dong, normalize_name(name))
        # 같은 키에 여러 단지가 오면 세대수가 큰 쪽을 남긴다 (대단지 대표값)
        if key not in by_name or by_name[key] < hh:
            by_name[key] = hh
    return by_pnu, by_name


def join_household(row: dict, by_pnu: dict[str, dict],
                   by_name: dict[tuple[str, str, str], int]) -> int | None:
    """거래 한 건의 세대수를 찾는다. PNU 완전일치 → 단지명 → 실패."""
    pnu = regions.make_pnu(row["sgg_cd"], row["umd_nm"], row["jibun"])
    if pnu:
        hit = by_pnu.get(pnu)
        if hit:
            return hit["hh"]
    dong = (row["umd_nm"] or "").split(" ")[-1]
    return by_name.get((row["sgg_cd"], dong, normalize_name(row["apt_name"])))


def month_labels() -> list[str]:
    """data/trades 에 실제로 있는 월 목록. 오름차순.

    파일명은 '2026-06.csv.gz' 꼴. Path.stem 은 접미사를 하나만 벗겨
    '2026-06.csv' 가 남으므로 removesuffix 로 두 접미사를 한 번에 뗀다.
    """
    return sorted(p.name.removesuffix(".csv.gz") for p in TRADES_DIR.glob("*/*.csv.gz"))


def read_month(ym: str) -> list[dict]:
    path = TRADES_DIR / ym[:4] / f"{ym}.csv.gz"
    if not path.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(path.read_bytes()))


class LazyMonths:
    """월 파일을 하나씩 읽어 넘긴다. 435만 행을 한꺼번에 메모리에 올리지 않기 위함이다.

    total 은 items() 로 실제 넘긴 행 수를 스트리밍 중에 누적한다. 미리
    합계를 구하려고 전체를 메모리에 올릴 필요가 없게 하기 위함이다.
    """

    def __init__(self, months: list[str]) -> None:
        self._months = months
        self.total = 0

    def items(self):
        for ym in self._months:
            rows = read_month(ym)
            self.total += len(rows)
            yield ym, rows

    def get(self, ym: str, default=None):
        """단일 월만 필요할 때 쓴다. Task 4 의 build_sgg_detail 이 이걸로
        by_month.get(ym, []) 처럼 호출해 월별 조회를 한다 — 지금은 여기서
        직접 쓰지 않지만 죽은 코드가 아니다."""
        return read_month(ym) if ym in self._months else default


def _is_better_peak(pp: float, trade_date: str, apt: str, current: dict | None) -> bool:
    """새 후보가 현재까지의 peak보다 우선하는지 판정한다.

    평당가가 더 높으면 무조건 우선. 평당가가 같으면(같은 단지·같은 면적·같은
    가격이 다른 날짜에 또 거래되는 경우가 실제로 있다) 날짜가 이른 쪽을,
    날짜까지 같으면 단지명 사전순으로 앞선 쪽을 우선한다 — 입력 행 순서와
    무관하게 항상 같은 승자를 골라야 빌드 결과가 두 번 돌려도 바이트가
    같아진다.
    """
    if current is None:
        return True
    cand = (-pp, trade_date, apt)
    cur = (-current["pp"], current["date"], current["apt"])
    return cand < cur


def build_summary(by_month: dict[str, list[dict]], months: list[str],
                  by_pnu: dict[str, dict], by_name: dict[tuple[str, str, str], int],
                  sgg_names: dict[str, str], generated: str) -> dict:
    """구 × 월 집계를 만든다. 필터 두 갈래('300', 'all')를 동시에 낸다.

    by_month 는 dict 뿐 아니라 .items() 를 한 번만 순회할 수 있는 이터러블
    (예: LazyMonths)도 받는다. 따라서 여기서는 .items() 를 정확히 한 번만
    순회한다 — 제너레이터는 두 번 돌릴 수 없다.
    """
    filters = ("300", "all")
    prices: dict[str, dict[str, dict[int, list[float]]]] = {
        f: defaultdict(lambda: defaultdict(list)) for f in filters}
    cancels: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    households: dict[str, dict[str, int]] = {f: defaultdict(int) for f in filters}
    seen_complex: dict[str, set[tuple[str, str]]] = {f: set() for f in filters}
    # 필터별 · 구별 역대 최고 평당가 거래 1건. {"pp", "date", "apt", "area", "dong"}.
    # 해제(cdeal_type == 'O') 거래는 아래 루프에서 가격 집계 전에 걸러지므로
    # 여기 후보로 들어오지 않는다.
    peaks: dict[str, dict[str, dict]] = {f: {} for f in filters}

    index = {ym: i for i, ym in enumerate(months)}

    for ym, rows in by_month.items():
        mi = index[ym]
        for row in rows:
            sgg = row["sgg_cd"]
            if sgg not in sgg_names:
                continue
            if row.get("cdeal_type") == "O":
                cancels[sgg][mi] += 1
                continue
            try:
                area = float(row["area_sqm"])
                price = int(row["price_10k"])
            except (ValueError, KeyError):
                continue
            pp = aggregate.pyeong_price(price, area)
            if pp is None:
                continue
            hh = join_household(row, by_pnu, by_name)
            targets = ["all"]
            if hh is not None and hh >= HOUSEHOLD_MIN:
                targets.append("300")
            key = (sgg, normalize_name(row["apt_name"]))
            trade_date = row["trade_date"]
            apt = row["apt_name"]
            dong = (row["umd_nm"] or "").split(" ")[-1]
            for f in targets:
                prices[f][sgg][mi].append(pp)
                if hh and key not in seen_complex[f]:
                    seen_complex[f].add(key)
                    households[f][sgg] += hh
                if _is_better_peak(pp, trade_date, apt, peaks[f].get(sgg)):
                    peaks[f][sgg] = {"pp": pp, "date": trade_date, "apt": apt,
                                     "area": area, "dong": dong}

    series: dict[str, dict[str, dict]] = {}
    for f in filters:
        series[f] = {}
        for sgg in sgg_names:
            med: list[int | None] = []
            cnt: list[int] = []
            can: list[int] = []
            for mi in range(len(months)):
                # 중위 평당가는 그 달만이 아니라 최근 MEDIAN_WINDOW 개월을 모아
                # 낸다. 한 달치만 쓰면 거래가 적은 구에서 어느 단지가 거래됐냐에
                # 따라 값이 통째로 흔들린다 — 금천구 2026-06 은 대단지 거래가
                # 빠지면서 3,186 → 2,758 로 -13% 찍었다가 다음 달 3,263 으로
                # 되돌아왔는데, 같은 단지끼리 비교하면 오히려 +4.7% 였다.
                # 서울·경기 68개 구 실측으로 전월 대비 변동폭 중위가
                # 3.50% → 1.38%, 5% 넘는 구가 16곳 → 1곳으로 줄었다.
                # n(거래 건수)은 그 달 실제 건수를 그대로 둔다 — 창을 넓힌 건
                # 가격이지 거래량이 아니다.
                vals = []
                for wj in range(max(0, mi - MEDIAN_WINDOW + 1), mi + 1):
                    vals.extend(prices[f][sgg].get(wj, []))
                m = aggregate.median(vals)
                med.append(round(m) if m is not None else None)
                cnt.append(len(prices[f][sgg].get(mi, [])))
                can.append(cancels[sgg].get(mi, 0) if f == "all" else 0)
            # cancel 은 필터와 무관하게 시군구·월 단위로 한 번만 집계한다(중복
            # 계상 방지). '300' 필터에서는 항상 0으로 채워지는데, 이는 해제
            # 건이 없어서가 아니라 '300' 쪽에 별도로 배분하지 않기 때문이다
            # — 버그가 아니라 의도된 설계다. 예산이 400KB 로 늘어난 뒤로는
            # 이 배열을 생략하지 않는다. months 와 길이가 항상 같아야
            # 프런트가 series[filter][sgg].cancel 을 조건 없이 읽을 수 있다.
            series[f][sgg] = {"med": med, "n": cnt, "cancel": can}
            # peak: 이 필터·이 구에서 역대 가장 비쌌던 평당가 거래 1건. med 와
            # 나란히 series[filter][sgg] 밑에 둔다 — 프런트가 구 상세를 열 때
            # 어차피 이 경로를 이미 읽고 있으므로 새 요청 없이 바로 붙일 수
            # 있다. 유효 거래가 한 건도 없던 구·필터는 키 자체를 생략한다
            # (명시적 null 대신 '없음'을 부재로 표현 — 프런트는 'peak' in s
            # 로 존재를 확인해야 한다).
            peak = peaks[f].get(sgg)
            if peak is not None:
                series[f][sgg]["peak"] = {
                    "pp": round(peak["pp"]), "date": peak["date"], "apt": peak["apt"],
                    "area": peak["area"], "dong": peak["dong"],
                }

    return {
        "generated": generated,
        "months": months,
        "partial": months[-1] if months else None,
        "sgg": {
            sgg: {
                "name": name,
                "sido": sgg[:2],
                "hh": {f: households[f].get(sgg, 0) for f in filters},
            }
            for sgg, name in sorted(sgg_names.items())
        },
        "series": series,
    }


SGG_DIR = OUT_DIR / "sgg"
DETAIL_WINDOW = 12          # 단지 랭킹은 최근 12개월
MAX_DETAIL_BYTES = 300 * 1024


def build_sgg_detail(by_month: dict[str, list[dict]], months: list[str], window: int,
                     by_pnu: dict[str, dict], by_name: dict[tuple[str, str, str], int],
                     sgg: str) -> dict:
    """구 하나의 단지별 집계. 창(window)은 months 의 마지막 n개월.

    1차 화면은 최근 12개월 랭킹만 쓴다. 창을 파라미터로 받아 두었으므로
    단지 시계열이 필요해지면 같은 함수를 다시 부르면 된다.
    """
    target = months[-window:] if window else months
    bucket_count = len(aggregate.AREA_EDGES) + 1

    prices: dict[tuple[str, str], list[float]] = defaultdict(list)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * bucket_count)
    households: dict[tuple[str, str], int | None] = {}
    labels: dict[tuple[str, str], str] = {}

    for ym in target:
        for row in by_month.get(ym, []):
            if row["sgg_cd"] != sgg or row.get("cdeal_type") == "O":
                continue
            try:
                area = float(row["area_sqm"])
                price = int(row["price_10k"])
            except (ValueError, KeyError):
                continue
            pp = aggregate.pyeong_price(price, area)
            if pp is None:
                continue
            dong = (row["umd_nm"] or "").split(" ")[-1]
            key = (dong, normalize_name(row["apt_name"]))
            prices[key].append(pp)
            buckets[key][aggregate.area_bucket(area)] += 1
            labels.setdefault(key, row["apt_name"])
            if key not in households:
                households[key] = join_household(row, by_pnu, by_name)

    complexes = []
    for key, vals in prices.items():
        dong, _ = key
        med = aggregate.median(vals)
        complexes.append({
            "name": labels[key],
            "dong": dong,
            "hh": households.get(key),
            "med": round(med) if med is not None else None,
            "n": len(vals),
            "bk": buckets[key],
        })
    # 비싼 순, 같으면 이름순 — 정렬이 고정돼야 출력이 결정론적이다
    complexes.sort(key=lambda c: (-(c["med"] or 0), c["name"]))

    return {
        "sgg": sgg,
        "window": [target[0], target[-1]] if target else [],
        "complexes": complexes,
    }


def write_json(path: Path, payload: dict) -> bool:
    """결정론적으로 쓴다. 내용이 같으면 건드리지 않고 False."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")) + "\n"
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default=date.today().isoformat(),
                        help="생성일. 테스트에서 고정하려면 지정한다")
    args = parser.parse_args()

    months = month_labels()
    if not months:
        print("data/trades 가 비어 있습니다.", file=sys.stderr)
        return 1

    by_pnu, by_name = load_complexes()
    sgg_names = {}
    for code, full in regions.sgg_codes():
        short = full.split(" ")[-1] if code.startswith("11") else " ".join(full.split(" ")[1:])
        sgg_names[code] = short

    # 435만 행을 한꺼번에 메모리에 올리면 이 머신(여유 메모리 ~3GB)에서
    # 실패한다. LazyMonths 는 월 파일을 하나씩 읽어 build_summary 에 넘기고,
    # build_summary 가 그 월을 다 쓰면 참조가 사라져 GC 대상이 된다.
    # build_summary 는 by_month.items() 를 정확히 한 번만 순회하므로
    # 제너레이터를 넘겨도 안전하다.
    by_month = LazyMonths(months)
    summary = build_summary(by_month, months, by_pnu, by_name, sgg_names, args.generated)
    print(f"원본 {by_month.total:,}건 / {len(months)}개월 / 시군구 {len(sgg_names)}개")

    changed = write_json(SUMMARY_FILE, summary)
    size = SUMMARY_FILE.stat().st_size
    print(f"summary.json {size:,}B {'갱신' if changed else '변경 없음'}")
    if size > MAX_SUMMARY_BYTES:
        print(f"summary.json 이 예산({MAX_SUMMARY_BYTES}B)을 넘었습니다.", file=sys.stderr)
        return 1

    # build_sgg_detail 은 시군구 72개마다 한 번씩(구별로 파일 하나) 불린다.
    # by_month 로 LazyMonths 를 그대로 넘기면 .get() 이 호출마다 파일을
    # 다시 읽어 같은 12개월 파일을 72번 재파싱하게 된다(864회 읽기). 대신
    # 창(최근 12개월)에 해당하는 파일만 딱 한 번씩 읽어 일반 dict 에 캐시해
    # 두고 재사용한다 — 247개월 전체를 올리는 게 아니라 창만 메모리에 둔다.
    window_months = months[-DETAIL_WINDOW:]
    window_cache = {ym: read_month(ym) for ym in window_months}

    SGG_DIR.mkdir(parents=True, exist_ok=True)
    updated, oversized = 0, []
    for sgg in sorted(sgg_names):
        detail = build_sgg_detail(window_cache, months, DETAIL_WINDOW,
                                  by_pnu, by_name, sgg)
        path = SGG_DIR / f"{sgg}.json"
        if write_json(path, detail):
            updated += 1
        if path.stat().st_size > MAX_DETAIL_BYTES:
            oversized.append((path.name, path.stat().st_size))
    print(f"구별 JSON {len(sgg_names)}개 중 {updated}개 갱신")
    if oversized:
        print(f"예산({MAX_DETAIL_BYTES}B) 초과: {oversized}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
