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


def _price_of(row: dict) -> float | None:
    """지수에 쓸 가격. TCGplayer(USD) 가 기준이고 없으면 Cardmarket 으로 메운다."""
    return row.get("tp_market") or row.get("cm_avg")


def build_universe(scan_rows: list[dict], set_meta: dict, names: dict | None = None,
                   seed: int = SEED, per_cell: int = tcgdex_api.PER_CELL) -> dict:
    """스캔 결과에서 12칸 층화추출로 구성 종목을 확정한다."""
    names = names or {}
    by_set = {sid: m for sid, m in set_meta.items() if m.get("included")}
    candidates = []
    for row in scan_rows:
        sid = row["card_id"].rsplit("-", 1)[0]
        meta = by_set.get(sid)
        price = _price_of(row)
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
        "name": names.get(c["card_id"]) or c["card_id"],
        "set_id": c["set_id"], "set_name": c["set_name"],
        "era": c["era"], "band": c["band"],
        "base_price": round(c["price"], 2),
    } for c in picked]

    return {
        "base_date": date.today().isoformat(),
        "seed": seed, "per_cell": per_cell,
        "generated": date.today().isoformat(),
        # 기준가는 아직 스캔 값이다. 첫 일일 관측이 들어오면 rebase_universe 가
        # 그 값으로 다시 잡는다.
        "rebased": False,
        "cards": cards,
    }


def rebase_universe(universe: dict, price_rows: list[dict]) -> dict:
    """첫 일일 관측을 기준가로 삼는다. 한 번만 한다.

    유니버스를 확정할 때 쓰는 기준가는 전수 스캔 때 받은 값이다. 그 스캔은
    80분에 걸쳐 돌기 때문에 카드마다 찍힌 시각이 다르고, 그날 다시 받은
    값과도 어긋난다. 그대로 두면 화면에 '기준일 = 100' 이라 써 놓고 첫 점이
    98.9 로 찍힌다.

    스캔 가격은 '어느 칸에 넣을지' 를 정하는 데만 쓰고, 지수의 출발점은
    첫 관측으로 다시 잡는다.
    """
    if universe.get("rebased"):
        return universe

    by_date: dict[str, dict[str, float]] = {}
    for row in price_rows:
        price = _price_of(row)
        if price:
            by_date.setdefault(row["date"], {})[row["card_id"]] = price
    if not by_date:
        return universe

    first = min(by_date)
    prices = by_date[first]
    cards = [{**c, "base_price": round(prices[c["card_id"]], 2)}
             if c["card_id"] in prices else dict(c)
             for c in universe["cards"]]
    return {**universe, "base_date": first, "rebased": True, "cards": cards}


def series_from_prices(universe: dict, price_rows: list[dict]) -> dict:
    """일자별 지수·하위지수 시계열을 만든다."""
    base = {c["card_id"]: c["base_price"] for c in universe["cards"]}
    by_date: dict[str, dict[str, float]] = {}
    for row in price_rows:
        price = _price_of(row)
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
        print(f"{collect_pokemon.SETS_FILE} 가 없습니다. "
              "먼저 collect_pokemon.py --mode scan 을 돌리세요.", file=sys.stderr)
        return 1
    set_meta = json.loads(collect_pokemon.SETS_FILE.read_text(encoding="utf-8"))

    if not collect_pokemon.UNIVERSE_FILE.exists():
        state = json.loads(collect_pokemon.SCAN_STATE.read_text(encoding="utf-8")) \
            if collect_pokemon.SCAN_STATE.exists() else {}
        if not state.get("complete"):
            print("전수 스캔이 아직 안 끝났습니다. 유니버스를 확정하지 않습니다.", file=sys.stderr)
            return 1
        names = json.loads(collect_pokemon.NAMES_FILE.read_text(encoding="utf-8")) \
            if collect_pokemon.NAMES_FILE.exists() else {}
        universe = build_universe(_read_rows(collect_pokemon.SCAN_FILE), set_meta, names)
        collect_pokemon.UNIVERSE_FILE.write_text(
            json.dumps(universe, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"유니버스 확정: {len(universe['cards'])}장")
    universe = json.loads(collect_pokemon.UNIVERSE_FILE.read_text(encoding="utf-8"))

    price_rows = _read_rows(collect_pokemon.PRICES_FILE)

    rebased = rebase_universe(universe, price_rows)
    if rebased is not universe:
        collect_pokemon.UNIVERSE_FILE.write_text(
            json.dumps(rebased, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"기준가를 첫 관측({rebased['base_date']})으로 다시 잡았습니다.")
        universe = rebased

    series = series_from_prices(universe, price_rows)

    last_date = series["dates"][-1] if series["dates"] else None
    latest_rows = [r for r in price_rows if r["date"] == last_date] if last_date else []
    back = backcast_from_cardmarket(universe, latest_rows)

    latest_price = {r["card_id"]: _price_of(r) for r in latest_rows}
    view_cards = [{**c, "price": latest_price.get(c["card_id"])} for c in universe["cards"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.json").write_text(
        json.dumps({**series, "backcast": back}, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "universe.json").write_text(
        json.dumps({**universe, "cards": view_cards}, ensure_ascii=False), encoding="utf-8")
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
