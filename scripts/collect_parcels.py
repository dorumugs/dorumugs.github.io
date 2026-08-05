#!/usr/bin/env python3
"""서울 아파트 단지의 대지면적·공시지가·용도지역을 UPIS 에서 받는다.

대지지분(대지면적 ÷ 세대수)을 내려면 대지면적이 필요하다. 1순위 출처는
건축물대장(collect_bldrgst.py)이지만 platArea 가 20% 남짓 비어 있어, 빈 곳을
이 지적도로 메운다. 서울 전용이다 — 경기는 브이월드(collect_vworld.py)가 맡는다.

(건축물대장은 2026-08-05 활용신청 승인 전까지 403 이었고, 그동안은 서울
대지면적을 이 경로로만 얻었다. 지금은 보조 출처다.)

  - 대지면적: LP_PA_CBND.SPACE_AREA
  - 공시지가: LP_PA_CBND.JIGA (원/㎡)
  - 용도지역: UPIS_C_UQ111 폴리곤에 필지 대표점을 떨어뜨려 판정

한 필지의 대지면적은 세 곳에서 얻을 수 있고 build_redevelopment.py 가 이 순서로 쓴다.

  1. 건축물대장 platArea      전국. 20% 남짓 비어 있다
  2. UPIS 지적도 (이 파일)    서울만. 촘촘하다
  3. 브이월드 지적도          전국. 경기를 덮는다

은마아파트로 1·2·3 을 대조했을 때 면적 차이가 0.06% 였다.

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
