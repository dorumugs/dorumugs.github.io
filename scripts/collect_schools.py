"""학교 위치 데이터를 받아 서울·경기 초·중과 특목고만 저장한다.

    python3 scripts/collect_schools.py

전국 12,011건을 13번에 나눠 받아 필터한 뒤 data/schools.csv.gz 로 쓴다.
실거래처럼 매일 돌릴 필요는 없다 — 학교는 자주 바뀌지 않는다.

고등학교는 위치 데이터만으로 특목고를 가릴 수 없어(종류 필드가 없다) NEIS
학교기본정보(neis_api.py)에서 종류·계열을 받아 (시도, 학교명) 으로 조인한다.
조인이 통째로 깨지면(이름 표기 변경 등) 특목고가 0건이 되므로 그때는 파일을
쓰지 않고 실패한다 — 조용히 사라지는 것보다 낫다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_trades  # noqa: E402
import neis_api  # noqa: E402
import rtms  # noqa: E402
import schools_api  # noqa: E402

OUT_FILE = ROOT / "data" / "schools.csv.gz"

# 서울·경기 특목고(과학·외국어·국제 계열)는 21~25곳 사이다. 이보다 적게 잡히면
# 조인이 깨진 것으로 본다 — 정확한 수를 박아 두면 학교 하나 생길 때마다 수집이
# 실패하므로 하한만 건다.
MIN_SPECIAL_HS = 15


class SpecialHighSchoolJoinFailed(Exception):
    """NEIS 조인 결과 특목고가 하한에 못 미칠 때."""

    def __init__(self, found: int) -> None:
        super().__init__(f"특목고 {found}곳 < 하한 {MIN_SPECIAL_HS}곳")
        self.found = found


class IncompleteCollection(Exception):
    """실제 수신 건수가 서버가 보고한 totalCount 보다 적을 때.

    schools_api.parse_response 는 빈 items 를 오류로 보지 않는다(Task 2 테스트로
    보장됨) — 즉 페이지 중간의 빈 응답도 resultCode 상으로는 정상이다. 그래서
    "이번 페이지가 비었다 = 다 받았다" 로 단정하면, 서버 일시 오류로 중간에 빈
    페이지가 와도 정상 종료로 오인해 잘린 파일을 그대로 커밋하게 된다. 수신량과
    totalCount 를 직접 대조해 이런 경우를 실패로 잡아낸다.
    """

    def __init__(self, received: int, total: int) -> None:
        super().__init__(f"수신 {received:,}건 < 서버 보고 {total:,}건")
        self.received = received
        self.total = total


def rows_to_csv(rows: list[dict]) -> str:
    """schools_api.COLUMNS 순서로 쓴다. rtms.rows_to_csv 는 실거래 컬럼 고정이라 못 쓴다."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=schools_api.COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in schools_api.COLUMNS})
    return buf.getvalue()


def sido_of(addr: str) -> str:
    """지번주소에서 시도만 뽑는다. NEIS 조인 키의 앞자리로 쓴다."""
    return "서울특별시" if addr.startswith("서울특별시") else "경기도"


def apply_courses(rows: list[dict], courses: dict[tuple[str, str], str]) -> list[dict]:
    """초·중은 그대로 두고, 고등학교는 대상 계열 특목고만 남기며 계열을 붙인다."""
    kept: list[dict] = []
    for row in rows:
        if row["level"] != "고등학교":
            kept.append(row)
            continue
        course = courses.get((sido_of(row["addr"]), row["school_name"]))
        if course is None:
            continue
        kept.append({**row, "course": course})
    return kept


def collect(key: str, page_size: int = 1000) -> list[dict]:
    """전수를 받아 서울·경기 초·중만 남긴다. 정렬해 결정론을 지킨다.

    Raises:
        IncompleteCollection: 수신 건수가 서버가 보고한 totalCount 에 못 미칠 때.
            페이지 중간의 빈 응답(정상 resultCode)도 '끝'이 아니라 이 실패로
            취급한다 — totalCount 가 0인 정상적인 '결과 없음' 은 seen == total(0)
            이므로 여기 해당하지 않는다.
    """
    kept: list[dict] = []
    seen = 0
    total = 0
    page = 1
    while True:
        items, total = schools_api.fetch_page(key, page, page_size)
        if not items:
            break
        seen += len(items)
        for item in items:
            row = schools_api.normalize(item)
            if row is not None:
                kept.append(row)
        print(f"  {seen:,}/{total:,} 수신, 대상 {len(kept):,}건")
        if seen >= total:
            break
        page += 1
    if seen < total:
        raise IncompleteCollection(seen, total)
    kept.sort(key=lambda r: (r["school_id"], r["school_name"]))
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=1000)
    args = parser.parse_args()

    key = collect_trades.load_api_key()
    try:
        rows = collect(key, args.page_size)
        # 조인은 수집이 끝난 뒤에 한다 — 위치 API 가 실패하면 NEIS 를 부를
        # 이유가 없고, 두 API 를 번갈아 부르면 어느 쪽이 실패했는지 흐려진다.
        courses = neis_api.fetch_courses(neis_api.load_key())
        rows = apply_courses(rows, courses)
        special = sum(1 for r in rows if r["level"] == "고등학교")
        if special < MIN_SPECIAL_HS:
            raise SpecialHighSchoolJoinFailed(special)
    except neis_api.NeisError as exc:
        print(f"NEIS 오류: {exc}. open.neis.go.kr 인증키를 확인하세요.", file=sys.stderr)
        return 1
    except SpecialHighSchoolJoinFailed as exc:
        print(f"특목고 조인이 깨졌습니다: {exc}. NEIS 학교명 표기가 바뀌었는지 "
              "확인하세요. 파일을 쓰지 않습니다.", file=sys.stderr)
        return 1
    except schools_api.ApiError as exc:
        if exc.code == "30":
            print("이 키로는 학교 위치 데이터를 쓸 수 없습니다. data.go.kr 에서 "
                  "'전국초중등학교위치표준데이터' 활용신청을 하세요.", file=sys.stderr)
        elif exc.code == "12":
            print(f"엔드포인트가 존재하지 않습니다: {schools_api.ENDPOINT}", file=sys.stderr)
        else:
            print(f"API 오류: {exc}", file=sys.stderr)
        return 1
    except IncompleteCollection as exc:
        print(
            f"수집이 완료되지 않았습니다: {exc.received:,}건 수신 / 서버 보고 "
            f"{exc.total:,}건. 일시적 서버 오류로 보고 파일을 쓰지 않습니다. 다시 "
            "실행하세요.",
            file=sys.stderr,
        )
        return 1

    if not rows:
        print("대상 학교가 0건입니다. 필터가 과한지 확인하세요.", file=sys.stderr)
        return 1

    levels: dict[str, int] = {}
    for row in rows:
        levels[row["level"]] = levels.get(row["level"], 0) + 1
    print(f"서울·경기 대상 {len(rows):,}건 " + " / ".join(f"{k} {v:,}" for k, v in sorted(levels.items())))
    by_course: dict[str, int] = {}
    for row in rows:
        if row["course"]:
            by_course[row["course"]] = by_course.get(row["course"], 0) + 1
    if by_course:
        print("  특목고 계열 " + " / ".join(f"{k} {v}" for k, v in sorted(by_course.items())))

    data = rtms.gzip_bytes(rows_to_csv(rows))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.read_bytes() == data:
        print("변경 없음")
        return 0
    OUT_FILE.write_bytes(data)
    print(f"{OUT_FILE.name} 갱신 ({len(data):,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
