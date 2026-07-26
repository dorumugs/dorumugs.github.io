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
        return read_month(ym) if ym in self._months else default


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
            for f in targets:
                prices[f][sgg][mi].append(pp)
                if hh and key not in seen_complex[f]:
                    seen_complex[f].add(key)
                    households[f][sgg] += hh

    series: dict[str, dict[str, dict]] = {}
    for f in filters:
        series[f] = {}
        for sgg in sgg_names:
            med: list[int | None] = []
            cnt: list[int] = []
            can: list[int] = []
            for mi in range(len(months)):
                vals = prices[f][sgg].get(mi, [])
                m = aggregate.median(vals)
                med.append(round(m) if m is not None else None)
                cnt.append(len(vals))
                can.append(cancels[sgg].get(mi, 0) if f == "all" else 0)
            entry: dict = {"med": med, "n": cnt}
            # 예산 완화책: cancel 배열이 전부 0인 구는 키 자체를 생략한다.
            # '300' 필터는 해제 건수를 별도로 세지 않아(항상 0) 늘 생략된다.
            if any(can):
                entry["cancel"] = can
            series[f][sgg] = entry

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
