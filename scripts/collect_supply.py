"""착공(통계누리)과 금리(ECOS)를 받아 저장한다.

    python3 scripts/collect_supply.py [--from 201101] [--force]

산출은 `data/supply/starts.json` 과 `data/supply/rates.json` 이고,
집계는 `build_supply.py` 가 따로 한다.

이 수집기의 핵심은 **잠정치 재수집**이다. 통계누리는 최근 약 10개월을
`"2026-07 p)"` 로 주고 확정되면서 값이 바뀐다. 확정월은 다시 받지 않고
잠정월은 매 실행 다시 받는다. 캐시가 잠정월을 잡아먹으면 영원히 옛 값을 쓴다.

통계누리는 **1회 조회 60개월** 제한이 있어 5년 단위로 잘라 부른다. 61개월을
넘기면 `{"result": false}` 가 HTTP 200 으로 온다.

`ECOS_API_KEY` 가 없으면 명확히 실패하고 종료코드 1 을 낸다. `sample` 키로
조용히 대체하지 않는다 — 10건 제한이라 반쪽 시계열이 만들어진다.

실패하면 기존 파일을 그대로 둔다. 지난달 데이터가 빈 파일보다 낫다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ecos_api  # noqa: E402
import molit_stat_api  # noqa: E402

ENV_FILE = ROOT / ".env"
OUT_DIR = ROOT / "data" / "supply"
STARTS_FILE = OUT_DIR / "starts.json"
RATES_FILE = OUT_DIR / "rates.json"

MOLIT_URL = "https://stat.molit.go.kr/portal/stat/data.do"
MOLIT_FORM_ID = 5387          # 주택유형별 착공실적(월계)
MOLIT_CHUNK_MONTHS = 60       # 1회 조회 한도

ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
ECOS_SERIES = {
    "base": ("722Y001", "0101000"),        # 한국은행 기준금리
    "mortgage": ("121Y006", "BECBLA0302"),  # 예금은행 주택담보대출 금리
}

FIRST_MONTH = "201101"        # 착공 통계가 시작하는 달
USER_AGENT = "kayserdocs-supply/1.0"
TIMEOUT = 60


def load_ecos_key() -> str:
    key = os.environ.get("ECOS_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ECOS_API_KEY=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise SystemExit(
        "ECOS_API_KEY 를 찾지 못했습니다. ecos.bok.or.kr/api/ 에서 인증키를 받아 "
        "저장소 루트 .env 에 ECOS_API_KEY=... 로 넣으세요. "
        "sample 키는 10건 제한이라 쓰지 않습니다."
    )


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))




def fetch_starts_chunk(start: str, end: str) -> list[molit_stat_api.Row]:
    query = urllib.parse.urlencode({
        "formId": MOLIT_FORM_ID, "styleNum": 1, "apprYn": "Y",
        "startDate": start, "endDate": end,
    })
    rows, problem = molit_stat_api.parse_starts(fetch_json(f"{MOLIT_URL}?{query}"))
    if problem:
        raise RuntimeError(f"통계누리 {start}~{end}: {problem}")
    return rows


def fetch_rates(key: str, first: str, last: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for name, (stat, item) in ECOS_SERIES.items():
        url = f"{ECOS_URL}/{key}/json/kr/1/1000/{stat}/M/{first}/{last}/{item}"
        series, problem = ecos_api.parse_series(fetch_json(url))
        if problem:
            raise RuntimeError(f"ECOS {name}({stat}): {problem}")
        out[name] = [list(point) for point in series]
        print(f"  금리 {name}: {len(series)}개월 "
              f"({series[0][0]}~{series[-1][0]})" if series else f"  금리 {name}: 없음")
    return out


def _load_cached() -> tuple[dict[tuple[str, str], list], set[str]]:
    """기존 `starts.json` → (키별 행, 확정월 집합)."""
    if not STARTS_FILE.exists():
        return {}, set()
    try:
        data = json.loads(STARTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, set()
    cached: dict[tuple[str, str], list] = {}
    provisional: set[str] = set()
    for row in data.get("rows") or []:
        month, region, _units, is_provisional = row
        cached[(month, region)] = row
        if is_provisional:
            provisional.add(month)
    confirmed = {m for m, _ in cached} - provisional
    return cached, confirmed


def months_between(first: str, last: str) -> list[str]:
    """`"201111"`, `"201202"` → `["2011-11", …, "2012-02"]`. 끝을 포함한다."""
    start = int(first[:4]) * 12 + int(first[4:]) - 1
    end = int(last[:4]) * 12 + int(last[4:]) - 1
    return [f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(start, end + 1)]


def month_chunks(first: str, last: str, size: int) -> list[tuple[str, str]]:
    """`"201101"`~`"202609"` → 60개월씩 자른 (시작, 끝) 목록.

    61개월 이상을 한 번에 요청하면 통계누리가 `{"result": false}` 를 HTTP 200
    으로 준다. 한 달이라도 넘기면 그 구간이 통째로 빈다.
    """
    months = months_between(first, last)
    out: list[tuple[str, str]] = []
    for i in range(0, len(months), size):
        chunk = months[i:i + size]
        out.append((chunk[0].replace("-", ""), chunk[-1].replace("-", "")))
    return out


def chunk_needs_refetch(start: str, end: str, confirmed: set[str]) -> bool:
    """청크 안에 확정되지 않은 달이 하나라도 있으면 다시 받는다.

    잠정치는 확정되면서 값이 바뀌므로 캐시로 덮으면 안 된다.
    """
    return not set(months_between(start, end)) <= confirmed


def collect_starts(first: str, last: str, force: bool) -> list[list]:
    """확정월은 캐시를 쓰고 잠정월은 다시 받는다. (월, 지역) 키로 덮어쓴다."""
    cached, confirmed = _load_cached()
    if force:
        cached, confirmed = {}, set()

    for start, end in month_chunks(first, last, MOLIT_CHUNK_MONTHS):
        if not chunk_needs_refetch(start, end, confirmed):
            print(f"  착공 {start}~{end}: 확정분 캐시 사용")
            continue
        rows = fetch_starts_chunk(start, end)
        for row in rows:
            cached[(row.month, row.region)] = [row.month, row.region,
                                               row.units, row.provisional]
        months = sorted({r.month for r in rows})
        print(f"  착공 {start}~{end}: {len(rows)}행 ({months[0]}~{months[-1]})"
              if months else f"  착공 {start}~{end}: 빈 응답")

    return [cached[key] for key in sorted(cached)]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="착공·금리 수집")
    parser.add_argument("--from", dest="first", default=FIRST_MONTH,
                        help=f"시작 월 YYYYMM (기본 {FIRST_MONTH})")
    parser.add_argument("--force", action="store_true",
                        help="확정월 캐시를 무시하고 전부 다시 받는다")
    args = parser.parse_args()

    key = load_ecos_key()
    today = date.today()
    last = f"{today.year:04d}{today.month:02d}"

    try:
        rows = collect_starts(args.first, last, args.force)
    except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(f"✖ 착공 수집 실패 — 기존 파일을 그대로 둡니다: {exc}", file=sys.stderr)
        return 1

    try:
        rates = fetch_rates(key, args.first, last)
    except (RuntimeError, urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        print(f"✖ 금리 수집 실패 — 기존 파일을 그대로 둡니다: {exc}", file=sys.stderr)
        return 1

    stamp = today.isoformat()
    _write(STARTS_FILE, {"fetched": stamp, "rows": rows})
    _write(RATES_FILE, {"fetched": stamp, **rates})

    months = sorted({r[0] for r in rows})
    provisional = sorted({r[0] for r in rows if r[3]})
    print(f"✔ 착공 {len(rows):,}행 {months[0]}~{months[-1]} "
          f"(잠정 {len(provisional)}개월"
          f"{', ' + provisional[0] + ' 부터' if provisional else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
