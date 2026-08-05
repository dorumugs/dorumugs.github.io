#!/usr/bin/env python3
"""재개발·재건축 대시보드 집계.

입력
  data/projects/projects.csv.gz   정보몽땅 사업장 목록 (서울)
  data/projects/events.csv.gz     사업장별 단계 이벤트 (일자·동의율)
  data/zones/zones.csv.gz         UPIS 정비구역 (경계 대표점·면적·추진단계코드)
  data/parcels/parcels.csv.gz     필지 대지면적·공시지가·용도지역 (서울, 지적도)
  data/bldrgst/bldrgst.csv.gz     건축물대장 총괄표제부 (전국, 세대수·연면적·대지면적)
  data/complexes.csv.gz           단지 마스터 (PNU·세대수·사용승인일)
  data/trades/                    실거래 435만 건

출력
  assets/realestate/redev.json          요약 + 자치구별 단계 집계 + 프리미엄
  assets/realestate/redev/<sgg>.json    자치구별 노후단지 표 + 사업장 목록

    python3 scripts/build_redevelopment.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate  # noqa: E402
import bldrgst_api  # noqa: E402
import build_dashboard  # noqa: E402  단지명 정규화를 한 곳에서만 정의하기 위해 빌려 쓴다
import cleanup_api  # noqa: E402
import regions  # noqa: E402
import rtms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_DIR = ROOT / "assets" / "realestate"
OUT_SUMMARY = OUT_DIR / "redev.json"
OUT_SGG_DIR = OUT_DIR / "redev"

# 노후 기준. 재건축 연한이 30년이라 그 아래는 표에서 기본으로 감춘다.
# 수집은 전량 해 두고 여기서만 자른다 — 기준을 바꿔도 다시 받을 필요가 없다.
MIN_AGE = 25

# 표에 남길 최소 세대수. 이 아래는 아파트로 등록돼 있어도 사실상 빌라다.
MIN_HOUSEHOLDS = 100

# 최근 중위 평당가를 낼 때 묶는 개월 수. build_dashboard 와 같은 값을 쓴다.
MEDIAN_WINDOW = 3

# 이벤트 스터디 창(개월). t0 직전 12개월 대 직후 12개월.
EVENT_WINDOW = 12
# 한 단계의 결과를 보여주려면 최소 이만큼의 사업장 표본이 있어야 한다.
MIN_EVENT_SAMPLE = 10
# 사업장 하나가 한 창에서 이 건수 미만이면 그 사업장은 표본에서 뺀다.
MIN_TRADES_PER_SIDE = 3

# 이벤트 스터디 대상 사업구분. 재개발은 구역 안이 빌라·단독이라
# 아파트 실거래(우리가 가진 데이터)에 잡히지 않는다. 재건축만 본다.
EVENT_BSNS_SE = {"재건축", "소규모재건축"}


# --------------------------------------------------------------------------
# 순수 함수
# --------------------------------------------------------------------------


def month_add(ym: str, delta: int) -> str:
    """'2020-08' 에 개월을 더한다."""
    year, month = int(ym[:4]), int(ym[5:7])
    total = year * 12 + (month - 1) + delta
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


# 전용면적을 연면적으로 되돌릴 때 쓰는 전용률. 계단식 아파트가 대략 이 언저리다.
# 정확한 값이 필요한 게 아니라 자릿수가 맞는지만 보면 되는 용도다.
EXCLUSIVE_RATIO = 0.75

# 역산한 용적률이 이 범위를 벗어나면 대지면적과 세대수가 서로 다른 것을 가리키고
# 있다고 본다. 아파트 단지는 저층이어도 80% 아래로 내려가기 어렵고, 초고층이어도
# 400% 를 넘기 어렵다. 등록 필지가 단지 땅이 아니거나(부속 필지·도로) 필지에
# 여러 단지가 얹혀 있는데 세대수는 일부만 잡힌 경우가 여기서 걸린다.
IMPLIED_FAR_MIN = 80.0
IMPLIED_FAR_MAX = 400.0


def implied_far(households: int, median_area_sqm: float | None, land_sqm: float) -> float | None:
    """실거래 전용면적으로 되짚은 용적률(%).

    건축물대장에 연면적이 없는 단지에만 쓰는 대비책이다. 그 단지에서 실제로
    거래된 전용면적의 중위값에 세대수를 곱하고 전용률로 나눠 연면적을 어림한다.
    어림값이므로 far_src='추정' 으로 구분해 내보낸다.
    """
    if not households or not median_area_sqm or land_sqm <= 0:
        return None
    gross = households * (median_area_sqm / EXCLUSIVE_RATIO)
    return gross / land_sqm * 100


def land_share_pyeong(
    area_sqm: float, households: int, far: float | None
) -> float | None:
    """대지지분(평) = 대지면적 ÷ 세대수. 재건축 사업성의 1순위 지표다.

    용적률(대장 실측이면 그것, 없으면 실거래 역산)이 말이 안 되면 대지면적과
    세대수가 같은 대상을 가리키지 않는 것이므로 값을 내지 않는다. 틀린 숫자를
    보여주는 것보다 낫다. 용적률을 아예 못 구하면(far is None) 검증을 건너뛰지
    않고 감춘다 — 검증 못 한 값을 검증된 값과 같은 열에 섞을 수는 없다.
    """
    if area_sqm <= 0 or households <= 0:
        return None
    if far is None or far < IMPLIED_FAR_MIN or far > IMPLIED_FAR_MAX:
        return None
    return area_sqm / households / aggregate.PYEONG


def excess_return(
    before: float | None, after: float | None,
    control_before: float | None, control_after: float | None,
) -> float | None:
    """사건 전후 변화율에서 같은 기간 대조군 변화율을 뺀 초과분(%p).

    이걸 안 빼면 시장 전체의 상승을 인가 프리미엄으로 오독한다.
    """
    own = aggregate.pct_change(after, before)
    control = aggregate.pct_change(control_after, control_before)
    if own is None or control is None:
        return None
    return own - control


def derive_propel_labels(
    zones: list[dict], projects: list[dict]
) -> dict[str, dict]:
    """UPIS 추진단계코드(PROPEL_CD)의 뜻을 정보몽땅 진행단계로 되짚는다.

    코드표가 공개되지 않아 직접 매핑할 수 없다. 같은 자치구·같은 정규화 구역명으로
    두 데이터를 붙이고, 코드마다 가장 많이 대응된 진행단계를 이름으로 삼는다.
    근거 건수를 함께 남겨 화면에서 신뢰도를 판단할 수 있게 한다.
    """
    by_key: dict[tuple[str, str], str] = {}
    for project in projects:
        norm = cleanup_api.normalize_zone_name(project.get("name", ""))
        if norm and project.get("stage"):
            by_key[(project.get("sgg_nm", ""), norm)] = project["stage"]

    sgg_name = {}
    for code, name in regions.sgg_codes():
        sgg_name[code] = name.split()[-1]

    votes: dict[str, Counter] = defaultdict(Counter)
    for zone in zones:
        code = zone.get("propel_cd")
        if not code:
            continue
        key = (sgg_name.get(zone.get("sgg_cd", ""), ""), zone.get("zone_name_norm", ""))
        stage = by_key.get(key)
        if stage:
            votes[code][stage] += 1

    out: dict[str, dict] = {}
    for code, counter in votes.items():
        label, hits = counter.most_common(1)[0]
        out[code] = {"label": label, "hits": hits, "total": sum(counter.values())}
    return out


# --------------------------------------------------------------------------
# 입출력
# --------------------------------------------------------------------------


def _read_gz(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(path.read_bytes()))


def load_complexes() -> list[dict]:
    """아파트 단지를 (시군구, 법정동, 정규화 단지명) 으로 묶어 돌려준다.

    공동주택 단지 식별정보는 한 단지를 지번마다 쪼개 등록한다. 압구정동
    현대아파트는 11조각이고, 조각의 세대수와 그 지번 필지의 면적은 서로
    대응하지 않는다 — 어떤 조각은 56세대에 63,321㎡, 어떤 조각은 144세대에
    186㎡ 다. 조각 단위로 대지지분을 내면 342평·0.4평 같은 값이 나온다.

    세대수와 필지면적을 함께 합쳐야 맞는다. 합치면 압구정 현대 25.9평,
    압구정 미성 9.2평로 알려진 값과 맞아떨어진다.
    """
    groups: dict[tuple[str, str, str], dict] = {}
    for row in _read_gz(DATA / "complexes.csv.gz"):
        if row.get("complex_type_code") != "1":
            continue
        pnu = row.get("pnu") or ""
        if not pnu:
            continue
        try:
            households = int(row.get("household_count") or 0)
        except ValueError:
            households = 0
        approval = (row.get("use_approval_date") or "").strip()
        year = int(approval[:4]) if len(approval) >= 4 and approval[:4].isdigit() else 0
        parts = (row.get("address") or "").split()
        dong = parts[-2] if len(parts) >= 2 else ""
        sgg = row.get("sgg_cd") or pnu[:5]
        name = row.get("complex_name") or ""
        key = (sgg, dong, build_dashboard.normalize_name(name))

        entry = groups.get(key)
        if entry is None:
            entry = groups[key] = {
                "key": key,
                "name": name,
                "sgg_cd": sgg,
                "dong": dong,
                "households": 0,
                "build_year": 0,
                "pnus": [],
            }
        entry["pnus"].append(pnu)
        entry["households"] += households
        # 조각마다 사용승인일이 다르면 가장 이른 것을 쓴다. 재건축 연한은
        # 단지에서 가장 오래된 동을 기준으로 따진다.
        if year and (not entry["build_year"] or year < entry["build_year"]):
            entry["build_year"] = year
        # 이름은 꼬리표가 없는 쪽을 대표로 삼는다 ('현대아파트' > '현대13차(208~211동)').
        if len(name) < len(entry["name"]):
            entry["name"] = name
    return list(groups.values())


def load_bldrgst() -> dict[str, dict]:
    """PNU → 건축물대장 총괄표제부. 단지 하나가 한 줄이다."""
    return {r["pnu"]: r for r in _read_gz(DATA / "bldrgst" / "bldrgst.csv.gz") if r.get("pnu")}


def load_parcels() -> dict[str, dict]:
    return {row["pnu"]: row for row in _read_gz(DATA / "parcels" / "parcels.csv.gz")}


def month_labels() -> list[str]:
    return sorted(p.name.removesuffix(".csv.gz") for p in (DATA / "trades").glob("*/*.csv.gz"))


def read_month(ym: str) -> list[dict]:
    path = DATA / "trades" / ym[:4] / f"{ym}.csv.gz"
    if not path.exists():
        return []
    return rtms.csv_to_rows(rtms.gunzip_text(path.read_bytes()))


def write_json(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


# --------------------------------------------------------------------------
# 실거래 스캔
# --------------------------------------------------------------------------


def scan_trades(months: list[str], target_pnus: set[str]) -> tuple[dict, dict]:
    """월 파일을 한 번만 훑어 두 가지를 만든다.

    1) by_pnu_month: 대상 단지의 (PNU, 월) → 평당가 목록  — 사건 전후 비교용
    2) by_sgg_month: (시군구, 월) → 평당가 목록            — 대조군용

    435만 행을 한꺼번에 올리지 않으려고 월 단위로 읽고 바로 접는다.
    대조군은 전 단지를 쓴다. 정비사업 대상만 빼면 표본이 얇아지는 구가 생기고,
    정비사업 단지 비중이 낮아 포함시켜도 대조군이 크게 오염되지 않는다.
    """
    by_pnu_month: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_sgg_month: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # 단지별 전용면적. 용적률 역산에 쓴다. 기간 전체를 모아 중위값을 낸다 —
    # 단지의 평형 구성은 시간이 지나도 바뀌지 않는다.
    areas_by_pnu: dict[str, list[float]] = defaultdict(list)

    for ym in months:
        for row in read_month(ym):
            # 계약해제 건은 성사되지 않은 가격이라 뺀다.
            if (row.get("cdeal_type") or "").strip() == "O":
                continue
            try:
                price = int(row["price_10k"])
                area = float(row["area_sqm"])
            except (ValueError, KeyError):
                continue
            pp = aggregate.pyeong_price(price, area)
            if pp is None:
                continue
            sgg = row.get("sgg_cd") or ""
            if sgg:
                by_sgg_month[sgg][ym].append(pp)
            pnu = regions.make_pnu(sgg, row.get("umd_nm", ""), row.get("jibun", ""))
            if pnu and pnu in target_pnus:
                by_pnu_month[pnu][ym].append(pp)
                areas_by_pnu[pnu].append(area)
    return by_pnu_month, by_sgg_month, areas_by_pnu


def window_median(series: dict[str, list[float]], start: str, end: str) -> tuple[float | None, int]:
    """[start, end] 개월 구간의 평당가 중위값과 거래 건수."""
    values: list[float] = []
    for ym, prices in series.items():
        if start <= ym <= end:
            values.extend(prices)
    if not values:
        return None, 0
    return statistics.median(values), len(values)


# --------------------------------------------------------------------------
# 이벤트 스터디
# --------------------------------------------------------------------------


def build_premium(
    projects: list[dict],
    events_by_project: dict[str, list[dict]],
    pnu_by_project: dict[str, str],
    by_pnu_month: dict,
    by_sgg_month: dict,
    latest: str,
) -> dict:
    """관문 단계 통과 전후의 평당가 초과수익을 집계한다."""
    per_stage: dict[str, list[dict]] = defaultdict(list)
    skipped = Counter()

    for project in projects:
        if project.get("bsns_se") not in EVENT_BSNS_SE:
            skipped["재건축 아님"] += 1
            continue
        cafe = project.get("cafe_url", "")
        pnu = pnu_by_project.get(cafe)
        if not pnu or pnu not in by_pnu_month:
            skipped["실거래 매칭 실패"] += 1
            continue

        milestones = cleanup_api.milestone_dates(events_by_project.get(cafe, []))
        if not milestones:
            skipped["관문 일자 없음"] += 1
            continue

        series = by_pnu_month[pnu]
        control = by_sgg_month.get(project.get("sgg_cd") or pnu[:5], {})

        for stage, day in milestones.items():
            t0 = day[:7]
            # 사건 당월은 양쪽 어디에도 넣지 않는다. 인가 시점 전후로 가격이
            # 어느 쪽에 속하는지 모호해 결과를 흐린다.
            before_lo, before_hi = month_add(t0, -EVENT_WINDOW), month_add(t0, -1)
            after_lo, after_hi = month_add(t0, 1), month_add(t0, EVENT_WINDOW)
            if after_hi > latest:
                skipped["사건 이후 기간 부족"] += 1
                continue

            before, n_before = window_median(series, before_lo, before_hi)
            after, n_after = window_median(series, after_lo, after_hi)
            if n_before < MIN_TRADES_PER_SIDE or n_after < MIN_TRADES_PER_SIDE:
                skipped["거래 부족"] += 1
                continue

            c_before, _ = window_median(control, before_lo, before_hi)
            c_after, _ = window_median(control, after_lo, after_hi)
            excess = excess_return(before, after, c_before, c_after)
            if excess is None:
                continue

            per_stage[stage].append(
                {
                    "name": project.get("name", ""),
                    "sgg": project.get("sgg_nm", ""),
                    "date": day,
                    "own": round(aggregate.pct_change(after, before) or 0, 1),
                    "control": round(aggregate.pct_change(c_after, c_before) or 0, 1),
                    "excess": round(excess, 1),
                    "trades": n_before + n_after,
                }
            )

    stages = []
    for stage in cleanup_api.MILESTONE_STAGES:
        cases = per_stage.get(stage, [])
        entry = {"stage": stage, "n": len(cases)}
        if len(cases) >= MIN_EVENT_SAMPLE:
            excesses = sorted(c["excess"] for c in cases)
            entry["median_excess"] = round(statistics.median(excesses), 1)
            entry["q1"] = round(excesses[len(excesses) // 4], 1)
            entry["q3"] = round(excesses[len(excesses) * 3 // 4], 1)
            entry["positive"] = sum(1 for e in excesses if e > 0)
            entry["cases"] = sorted(cases, key=lambda c: -c["excess"])[:20]
        stages.append(entry)

    return {
        "window_months": EVENT_WINDOW,
        "min_sample": MIN_EVENT_SAMPLE,
        "scope": "재건축·소규모재건축만. 재개발 구역은 빌라·단독이라 아파트 실거래에 잡히지 않는다.",
        "stages": stages,
        "skipped": dict(skipped),
    }


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-age", type=int, default=MIN_AGE, help="표에 남길 최소 연차")
    args = parser.parse_args()

    projects = _read_gz(DATA / "projects" / "projects.csv.gz")
    events = _read_gz(DATA / "projects" / "events.csv.gz")
    zones = _read_gz(DATA / "zones" / "zones.csv.gz")
    parcels = load_parcels()
    bldrgst = load_bldrgst()
    complexes = load_complexes()
    if not projects or not complexes:
        print("입력 데이터가 부족합니다. collect_*.py 를 먼저 돌리세요.", file=sys.stderr)
        return 1

    months = month_labels()
    if not months:
        print("실거래 월 파일이 없습니다.", file=sys.stderr)
        return 1
    latest = months[-1]
    this_year = date.today().year

    events_by_project: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        events_by_project[event["cafe_url"]].append(event)

    # 사업장 대표지번 → PNU. 실거래·단지와 붙이는 유일한 열쇠다.
    sgg_by_name = {name.split()[-1]: code for code, name in regions.sgg_codes()}
    pnu_by_project: dict[str, str] = {}
    for project in projects:
        sgg = sgg_by_name.get(project.get("sgg_nm", ""), "")
        project["sgg_cd"] = sgg
        pnu = regions.make_pnu(sgg, project.get("umd_nm", ""), project.get("jibun", ""))
        if pnu:
            pnu_by_project[project["cafe_url"]] = pnu

    project_pnus = set(pnu_by_project.values())
    old_groups = [
        g for g in complexes
        if g["build_year"] and this_year - g["build_year"] >= args.min_age
    ]
    target_pnus = project_pnus | {pnu for g in old_groups for pnu in g["pnus"]}

    print(f"사업장 {len(projects)}건 · 이벤트 {len(events)}건 · 구역 {len(zones)}건")
    print(f"대상 단지 {len(target_pnus)}곳 · 실거래 {len(months)}개월 스캔 시작")
    by_pnu_month, by_sgg_month, areas_by_pnu = scan_trades(months, target_pnus)
    print(f"  실거래 매칭된 단지 {len(by_pnu_month)}곳")

    # 자치구별 노후 단지 표
    recent_lo = month_add(latest, -(MEDIAN_WINDOW - 1))
    by_sgg_rows: dict[str, list[dict]] = defaultdict(list)
    project_stage_by_pnu = {
        pnu: p.get("stage", "")
        for p in projects
        for pnu in [pnu_by_project.get(p["cafe_url"])]
        if pnu
    }

    dropped_share = 0
    for group in old_groups:
        # 세대수가 적은 '아파트'는 사실상 빌라다. 재건축 후보로 볼 대상이 아니고,
        # 8세대짜리가 대지지분 상위를 채우면 표를 읽는 데 방해만 된다.
        if group["households"] < MIN_HOUSEHOLDS:
            continue
        # 대지면적은 건축물대장을 먼저 보고, 비어 있으면(20% 남짓) 지적도로 메운다.
        # 대장은 전국을 덮지만 platArea 가 0 인 단지가 많고, 지적도는 값이
        # 촘촘하지만 서울뿐이다. 둘을 겹쳐야 서울·경기가 다 채워진다.
        # 조각 하나라도 못 채우면 합산을 포기한다 — 분자만 모자란 채 세대수
        # 전체로 나누면 대지지분이 실제보다 작게 나온다.
        area = 0.0
        sources: set[str] = set()
        for pnu in group["pnus"]:
            ledger = bldrgst.get(pnu) or {}
            piece = bldrgst_api._num(ledger.get("plat_area"))
            if piece > 0:
                sources.add("대장")
            else:
                parcel = parcels.get(pnu) or {}
                try:
                    piece = float(parcel.get("area_sqm") or 0)
                except ValueError:
                    piece = 0.0
                if piece > 0:
                    sources.add("지적도")
            if piece <= 0:
                area = 0.0
                break
            area += piece

        # 세대수·연면적은 대장이 정확하다. 대장이 비는 단지만 마스터 세대수를 쓴다.
        ledger_hh = sum(
            int(bldrgst_api._num((bldrgst.get(pnu) or {}).get("hhld_cnt")))
            for pnu in group["pnus"]
        )
        households = ledger_hh or group["households"]
        gfa = sum(
            bldrgst_api.floor_area(bldrgst.get(pnu) or {}) or 0.0 for pnu in group["pnus"]
        )

        # 실측 용적률. 연면적을 못 구한 단지만 실거래 전용면적으로 역산한다.
        far_actual = gfa / area * 100 if (gfa > 0 and area > 0) else None
        if far_actual is None:
            areas = [a for pnu in group["pnus"] for a in areas_by_pnu.get(pnu, [])]
            median_area = statistics.median(areas) if areas else None
            far_actual = implied_far(households, median_area, area)
            far_src = "추정" if far_actual else None
        else:
            far_src = "대장"

        share = land_share_pyeong(area, households, far_actual)
        if area > 0 and share is None:
            dropped_share += 1

        # 용도지역·용적률 상한·공시지가는 지적도에만 있다 (서울 한정).
        # 조각이 여럿이면 가장 넓은 필지 것을 대표로 쓴다.
        main = None
        for m in (parcels.get(pnu) for pnu in group["pnus"]):
            if not m:
                continue
            try:
                if main is None or float(m.get("area_sqm") or 0) > float(main.get("area_sqm") or 0):
                    main = m
            except ValueError:
                continue
        main = main or {}
        try:
            far_limit = int(main.get("far_limit") or 0) or None
        except ValueError:
            far_limit = None

        # 거래는 조각 전체를 합쳐 본다. 한 단지가 여러 지번에 걸쳐 신고된다.
        merged: dict[str, list[float]] = defaultdict(list)
        for pnu in group["pnus"]:
            for ym, prices in by_pnu_month.get(pnu, {}).items():
                merged[ym].extend(prices)
        median_pp, n_trades = window_median(merged, recent_lo, latest)

        stage = next(
            (project_stage_by_pnu[p] for p in group["pnus"] if p in project_stage_by_pnu), None
        )
        by_sgg_rows[group["sgg_cd"]].append(
            {
                "name": group["name"],
                "dong": group["dong"],
                "year": group["build_year"],
                "age": this_year - group["build_year"],
                "hh": households,
                "parts": len(group["pnus"]),
                "land": round(area) if share else None,
                "land_src": "·".join(sorted(sources)) if share and sources else None,
                "share": round(share, 1) if share else None,
                # 용적률은 대지지분이 검증을 통과한 단지에만 붙인다.
                # 검증에 실패한 값은 그 자체가 신뢰할 수 없다는 뜻이다.
                "far_est": round(far_actual) if share and far_actual else None,
                "far_src": far_src if share else None,
                "zone": main.get("landuse_nm") or None,
                "far": far_limit,
                "jiga": int(float(main.get("jiga_won_sqm") or 0)) or None,
                "pp": round(median_pp) if median_pp else None,
                "n": n_trades,
                "stage": stage,
            }
        )

    # 자치구별 사업장·구역
    projects_by_sgg: dict[str, list[dict]] = defaultdict(list)
    for project in projects:
        milestones = cleanup_api.milestone_dates(events_by_project.get(project["cafe_url"], []))
        projects_by_sgg[project["sgg_cd"]].append(
            {
                "name": project.get("name", ""),
                "se": project.get("bsns_se", ""),
                "addr": project.get("jibun_addr", ""),
                "stage": project.get("stage", ""),
                "suspended": bool(project.get("suspended")),
                "milestones": milestones,
            }
        )

    propel_labels = derive_propel_labels(zones, projects)
    zones_by_sgg: dict[str, list[dict]] = defaultdict(list)
    for zone in zones:
        if not zone.get("lon"):
            continue
        zones_by_sgg[zone["sgg_cd"]].append(
            {
                "name": zone.get("zone_name", ""),
                "se": zone.get("bsns_se", ""),
                "area": round(float(zone.get("area_sqm") or 0)),
                "lon": float(zone["lon"]),
                "lat": float(zone["lat"]),
                "propel": zone.get("propel_cd", ""),
            }
        )

    premium = build_premium(
        projects, events_by_project, pnu_by_project, by_pnu_month, by_sgg_month, latest
    )

    changed = 0
    for sgg_cd, rows in by_sgg_rows.items():
        # 기본 정렬은 연차다. 대지지분으로 세우면 등록 필지가 어긋난 단지가
        # 상위를 차지해 표를 잘못 읽게 된다. 대지지분 정렬은 사용자가 직접 누른다.
        rows.sort(key=lambda r: (-(r["age"] or 0), -(r["hh"] or 0)))
        payload = {
            "sgg": sgg_cd,
            "complexes": rows,
            "projects": projects_by_sgg.get(sgg_cd, []),
            "zones": zones_by_sgg.get(sgg_cd, []),
        }
        if write_json(OUT_SGG_DIR / f"{sgg_cd}.json", payload):
            changed += 1

    stage_counts = {
        sgg: Counter(p["stage"] for p in items if p["stage"])
        for sgg, items in projects_by_sgg.items()
    }
    summary = {
        "generated": date.today().isoformat(),
        "latest_month": latest,
        "min_age": args.min_age,
        "counts": {
            "projects": len(projects),
            "projects_with_events": len(events_by_project),
            "zones": len(zones),
            "complexes": sum(len(v) for v in by_sgg_rows.values()),
            "with_land": sum(
                1 for rows in by_sgg_rows.values() for r in rows if r["share"] is not None
            ),
            "land_dropped": dropped_share,
        },
        "coverage": {
            "land_share": "건축물대장 대지면적을 먼저 쓰고 빈 곳은 서울 지적도로 메운다",
            "land_dropped": (
                f"역산 용적률이 {IMPLIED_FAR_MIN:.0f}~{IMPLIED_FAR_MAX:.0f}% 밖이라 "
                f"대지지분을 감춘 단지 {dropped_share}곳"
            ),
            "far_est": (
                "건축물대장 용적률 산정 연면적 ÷ 대지면적. 대장 연면적이 없는 단지만 "
                f"실거래 전용면적 중위값 × 세대수 ÷ 전용률 {EXCLUSIVE_RATIO:.2f} 로 어림한다 "
                "(far_src 로 구분). 용적률이 100% 아래면 대지면적이 단지 땅보다 넓게 "
                "잡혔을 수 있다."
            ),
            "stages": "서울만. 정비사업 진행 데이터는 서울시 정보몽땅이 유일한 상시 출처",
        },
        "sgg": {
            sgg: {
                "complexes": len(rows),
                "projects": len(projects_by_sgg.get(sgg, [])),
                "stages": dict(stage_counts.get(sgg, {})),
            }
            for sgg, rows in sorted(by_sgg_rows.items())
        },
        "propel_labels": propel_labels,
        "premium": premium,
    }
    if write_json(OUT_SUMMARY, summary):
        changed += 1

    print(f"\n자치구 {len(by_sgg_rows)}개 · 노후단지 {summary['counts']['complexes']}곳 "
          f"(대지지분 있음 {summary['counts']['with_land']}곳) · 파일 {changed}개 갱신")
    print("단계별 프리미엄:")
    for entry in premium["stages"]:
        if "median_excess" in entry:
            print(f"  {entry['stage']:10s} n={entry['n']:3d}  중위 초과 {entry['median_excess']:+.1f}%p "
                  f"(사분위 {entry['q1']:+.1f} ~ {entry['q3']:+.1f}, 양수 {entry['positive']}건)")
        else:
            print(f"  {entry['stage']:10s} n={entry['n']:3d}  표본 부족")
    if premium["skipped"]:
        print("  제외:", premium["skipped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
