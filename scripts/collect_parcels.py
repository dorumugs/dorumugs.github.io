#!/usr/bin/env python3
"""서울 아파트 단지의 대지면적·공시지가·용도지역을 UPIS 에서 받는다.

대지지분(대지면적 ÷ 세대수)은 재건축 사업성의 1순위 지표인데, 원래 출처인
국토교통부 건축물대장 API(BldRgstHubService)는 별도 활용신청이 필요해
현재 키로는 403 이 온다. 서울에 한해 연속지적도(LP_PA_CBND)가 같은 값을
PNU 로 바로 내주므로 여기서 받는다.

  - 대지면적: LP_PA_CBND.SPACE_AREA
  - 공시지가: LP_PA_CBND.JIGA (원/㎡)
  - 용도지역: UPIS_C_UQ111 폴리곤에 필지 대표점을 떨어뜨려 판정

경기도는 UPIS 가 서울 전용이라 이 경로로 못 받는다. 건축물대장 활용신청이
승인되면 collect_bldrgst.py 로 전국을 덮는다.

출력: data/parcels/parcels.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_api  # noqa: E402
import upis_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMPLEX_FILE = ROOT / "data" / "complexes.csv.gz"
OUT_FILE = ROOT / "data" / "parcels" / "parcels.csv.gz"

SEOUL_SIDO = "11"
APARTMENT = "1"  # complex_type_code
BATCH = 250  # PNU IN (...) 한 묶음 크기. URL 길이와 응답 크기를 함께 본 값이다.
LANDUSE_BATCH = 400  # 용도지역은 폴리곤이 커서 PNU 보다 작게 끊는다 (한도는 1000).


def _request(url: str, retries: int = 3) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KayserDocs-realestate/1.0)",
            "Referer": upis_api.REFERER,
        },
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            status = getattr(exc, "code", None)
            if status is not None and 400 <= status < 500:
                raise
            last = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise last  # type: ignore[misc]


def seoul_apartment_pnus() -> list[str]:
    text = gzip.decompress(COMPLEX_FILE.read_bytes()).decode("utf-8")
    rows = csv.DictReader(io.StringIO(text, newline=""))
    # 준공연도로 거르지 않는다. 화면에서 기준을 바꿀 때 다시 받지 않기 위해서다.
    pnus = {
        r["pnu"]
        for r in rows
        if r.get("complex_type_code") == APARTMENT and (r.get("pnu") or "").startswith(SEOUL_SIDO)
    }
    return sorted(p for p in pnus if p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=BATCH, help="PNU 묶음 크기")
    parser.add_argument("--sleep", type=float, default=0.5, help="요청 사이 대기 초")
    parser.add_argument("--skip-landuse", action="store_true", help="용도지역 판정을 건너뛴다")
    args = parser.parse_args()

    pnus = seoul_apartment_pnus()
    if not pnus:
        print("대상 PNU 가 없습니다. data/complexes.csv.gz 를 확인하세요.", file=sys.stderr)
        return 1
    print(f"서울 아파트 단지 {len(pnus)}곳 · 묶음 {args.batch}개씩 {-(-len(pnus) // args.batch)}콜")

    parcels: list[dict] = []
    for i in range(0, len(pnus), args.batch):
        chunk = pnus[i : i + args.batch]
        try:
            got = upis_api.parse_parcels(_request(upis_api.parcel_query_url(chunk)))
        except (upis_api.UpisError, urllib.error.URLError, OSError) as exc:
            print(f"  {i // args.batch + 1}번째 묶음 실패: {exc}", file=sys.stderr)
            continue
        parcels.extend(got)
        print(f"  {i + len(chunk):>5}/{len(pnus)}  누적 필지 {len(parcels)}건")
        if args.sleep:
            time.sleep(args.sleep)

    if not parcels:
        print("필지를 하나도 받지 못했습니다. 프록시가 막혔을 수 있습니다.", file=sys.stderr)
        return 1

    matched = 0
    if not args.skip_landuse:
        ids = upis_api.parse_object_ids(_request(upis_api.landuse_ids_url()))
        print(f"\n용도지역 폴리곤 {len(ids)}개 · {-(-len(ids) // LANDUSE_BATCH)}콜")
        polygons: list[dict] = []
        for i in range(0, len(ids), LANDUSE_BATCH):
            chunk = ids[i : i + LANDUSE_BATCH]
            try:
                polygons.extend(
                    upis_api.parse_landuse(_request(upis_api.landuse_query_url(chunk)))
                )
            except (upis_api.UpisError, urllib.error.URLError, OSError) as exc:
                print(f"  {i}~ 실패: {exc}", file=sys.stderr)
                continue
            if args.sleep:
                time.sleep(args.sleep)
        print(f"  폴리곤 {len(polygons)}개 확보")

        # 자치구로 못 나누는 레이어라 전량을 한 통에 두고 경계상자로 걸러 찾는다.
        for parcel in parcels:
            if not parcel["lon"]:
                continue
            hit = upis_api.find_landuse(float(parcel["lon"]), float(parcel["lat"]), polygons)
            if hit:
                parcel["landuse_cd"] = hit["code"]
                parcel["landuse_nm"] = hit["name"]
                parcel["far_limit"] = upis_api.FAR_LIMIT.get(hit["code"], "")
                matched += 1

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=upis_api.PARCEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(parcels, key=lambda r: r["pnu"]):
        writer.writerow({c: row.get(c, "") for c in upis_api.PARCEL_COLUMNS})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = cleanup_api.gzip_bytes(buf.getvalue())
    changed = not (OUT_FILE.exists() and OUT_FILE.read_bytes() == data)
    if changed:
        OUT_FILE.write_bytes(data)

    have_area = sum(1 for p in parcels if p.get("area_sqm"))
    print(
        f"\n필지 {len(parcels)}건 (요청 {len(pnus)}곳 중 {len(parcels) / len(pnus):.0%}) · "
        f"면적 있음 {have_area}건 · 용도지역 판정 {matched}건 · "
        f"{'갱신' if changed else '변화 없음'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
