"""전국 Airbnb 숙소 좌표를 쿼드트리로 훑어 모은다.

    python3 scripts/collect_airbnb.py --max-calls 1200      # 하루 예산만큼
    python3 scripts/collect_airbnb.py --box 33.6,126.99,33.1,126.14 \
        --state data/airbnb/jeju.json.gz                   # 한 지역만 따로
    python3 scripts/collect_airbnb.py --reset               # 처음부터 다시

왜 쿼드트리인가
---------------
bbox 하나로 받아지는 좌표는 약 200건이 끝이다(`airbnb_api` 의 주석 참고).
그래서 넓은 bbox 에서 시작해 "여기 상한보다 많다" 싶으면 넷으로 쪼개 다시
묻는다. 바다·산처럼 빈 곳은 첫 요청에서 바로 가지친다.

수집 예절
---------
robots.txt 는 `/s/homes` 를 막지 않지만 Airbnb 이용약관은 자동수집을 따로
금지한다. 아래는 성능 조정 항목이 아니라 지켜야 할 선이다.

  · 요청 간격 2초 고정. 병렬 요청 없음.
  · `--max-calls` 로 한 번 실행의 예산을 끊는다.
  · 연속 실패 5회면 그 실행을 접는다. 재시도로 밀어붙이지 않는다.
  · 가격·평점·후기·사진·링크는 건드리지 않는다. 좌표와 좌표 개수뿐이다.

중간에 끊겨도 되는 이유
-----------------------
아직 안 본 bbox 목록(frontier)과 지금까지 모은 좌표를 `data/airbnb/state.json.gz`
에 함께 둔다. 예산이 떨어지면 처리 중이던 bbox 를 **frontier 에 되돌려 놓고**
끝낸다 — 절반만 거둔 leaf 가 남지 않는다. 다음 실행이 거기서 이어받는다.

조용한 고장을 막는 장치
-----------------------
Airbnb 가 화면을 바꾸거나 우리를 막으면, 응답은 **빈 바다와 똑같이** 생겼다
(총건수 없음·좌표 0건). 그래서 숙소가 확실히 있는 bbox(CANARY, 강남역 일대)를
시작할 때와 `--canary-every` 호출마다 한 번씩 찔러 본다. 거기서 좌표가 0건이면
막혔거나 파싱이 깨진 것이므로 **기존 데이터를 건드리지 않고** 종료한다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import airbnb_api as api  # noqa: E402

DATA_DIR = ROOT / "data" / "airbnb"
STATE_FILE = DATA_DIR / "state.json.gz"

# 헤드리스 UA 를 걸러내는 곳이 있어 평범한 Chrome UA 를 쓴다.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/149.0.0.0 Safari/537.36")

GEO_FILE = ROOT / "data" / "geo" / "sgg_kr.geojson.gz"

# 전국을 사각형 하나로 덮을 때 쓰는 값. 규슈가 들어와 예산을 축내므로 평소에는
# 쓰지 않는다 — `seed()` 가 시도별 경계상자를 만들지 못할 때만 물러설 자리다.
KOREA = api.Box(ne_lat=38.65, ne_lng=131.90, sw_lat=32.95, sw_lng=124.50)

# 숙소가 확실히 있는 곳. 파싱이 살아 있는지 확인하는 데만 쓴다.
CANARY = api.Box(ne_lat=37.52, ne_lng=127.05, sw_lat=37.49, sw_lng=127.02)

SPLIT_AT = 180      # 실측 상한 200 보다 낮게 잡아 여유를 둔다
MIN_DEG = 0.004     # 약 400m. 더 쪼개도 의미가 없다
# 이보다 큰 bbox 가 "비었다" 고 나오면 한 번 더 묻는다. 1도는 약 110km 라
# 위쪽 두세 단계뿐이다 — 도 하나가 통째로 사라지는 것만 막고 바다의 빈 칸을
# 전부 두 번 묻지는 않는다.
CONFIRM_EMPTY_DEG = 1.0
SLEEP = 2.0
MAX_FAILURES = 5

# 이 호출 수마다 상태를 디스크에 적는다. 한 번에 1,200콜을 돌면 40분인데,
# 마지막에 한 번만 저장하면 그 사이 어떤 이유로든 프로세스가 죽을 때 모은 것이
# 통째로 날아간다 — 그리고 다시 모으려면 같은 시간만큼 남의 서버를 또 두드려야
# 한다. 저장은 몇 백 KB 쓰기라 자주 해도 싸다.
CHECKPOINT = 50


def fetch(box: api.Box, cursor: str | None = None, timeout: float = 60.0) -> str:
    """검색 HTML 한 장. 실패하면 예외를 그대로 올린다 — 호출부가 센다."""
    request = urllib.request.Request(api.search_url(box, cursor), headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Accept-Encoding": "gzip",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def load_state(path: Path) -> dict:
    if not path.exists():
        return new_state()
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def seed() -> list[api.Box]:
    """쿼드트리 시작 bbox. 시도별 경계상자를 쓰고, 없으면 전국 사각형 하나.

    전국 사각형 하나로 시작하면 규슈까지 훑는다(실측: 후쿠오카 1,694건을
    받아다 버렸다). 시군구 경계는 `build_geo.py --sido all` 이 만들어 둔다.
    """
    if GEO_FILE.exists():
        try:
            with gzip.open(GEO_FILE, "rb") as f:
                boxes = api.seed_boxes(json.loads(f.read().decode("utf-8")))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"  시도 경계를 못 읽어 전국 사각형으로 시작합니다: {exc}",
                  file=sys.stderr)
            boxes = []
        if boxes:
            return boxes
    print("  시도 경계가 없어 전국 사각형 하나로 시작합니다 "
          "(규슈까지 훑게 되므로 build_geo.py --sido all 을 먼저 돌리세요).",
          file=sys.stderr)
    return [KOREA]


def new_state(boxes: list[api.Box] | None = None) -> dict:
    return {
        "frontier": [list(b) for b in (boxes if boxes is not None else seed())],
        # 지난 바퀴를 다 돌아 확정된 스냅샷. 화면에 나가는 건 이쪽이다.
        "points": {},
        "weak": [],
        # 지금 돌고 있는 바퀴가 모으는 중인 것. 다 돌면 위를 대신한다.
        "pending": {},
        "weak_pending": [],
        "stats": {"calls": 0, "leaves": 0, "splits": 0, "empty": 0, "weak": 0},
        "started": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated": None,
        "complete": False,
        "completed_at": None,
    }


published_points = api.published_points


def begin_pass(state: dict, boxes: list[api.Box] | None = None) -> None:
    """frontier 가 비었으면 새 바퀴를 시작한다.

    **확정 스냅샷(`points`)은 건드리지 않는다.** 전국 한 바퀴에 며칠이 걸리는데
    새 바퀴를 돌 때마다 지우면 그 며칠 동안 지도가 반쯤 빈 채로 보인다.

    `pending` 이 없던 시절의 상태 파일도 여기서 손본다. 그때는 확정본과 모으는
    중인 것이 한 통에 있었으므로, **아직 돌던 중이었으면(frontier 가 남아 있으면)
    그 좌표는 확정이 아니라 `pending` 이다.** 확정으로 두면 화면이 '전국을 다
    훑었다' 고 거짓말한다.
    """
    if "pending" not in state:
        unfinished = bool(state.get("frontier"))
        state["pending"] = state.get("points") or {} if unfinished else {}
        state["weak_pending"] = (state.get("weak") or []) if unfinished else []
        if unfinished:
            state["points"] = {}
            state["weak"] = []
    state.setdefault("weak_pending", [])
    state.setdefault("weak", [])
    if state.get("frontier"):
        return
    state["frontier"] = [list(b) for b in (boxes if boxes is not None else seed())]
    state["pending"] = {}
    state["weak_pending"] = []
    state["complete"] = False


def finish_pass(state: dict) -> None:
    """다 훑었으면 이번 바퀴 결과로 확정 스냅샷을 갈아 끼운다.

    이번 바퀴가 **빈 채로** 끝났으면 갈아 끼우지 않는다 — 수집이 통째로 깨진
    바퀴가 지도를 지우는 것보다 지난주 지도가 남는 편이 낫다.
    """
    if state.get("frontier"):
        return
    if state.get("pending"):
        state["points"] = state["pending"]
        state["weak"] = state.get("weak_pending") or []
        state["pending"] = {}
        state["weak_pending"] = []
        state["completed_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["complete"] = True


def requeue(frontier: list, box: api.Box) -> None:
    """실패한 bbox 를 큐 **앞**(가장 나중에 꺼낼 자리)으로 보낸다.

    끝에 되돌리면 다음 pop 이 같은 bbox 를 다시 집는다. 영구적으로 응답이
    깨진 bbox 가 하나라도 있으면 그것만 다섯 번 두드리다 매 실행이 중단된다.
    앞으로 보내면 나머지를 다 훑은 뒤에 다시 시도한다.
    """
    frontier.insert(0, list(box))


def save_state(state: dict, path: Path) -> None:
    """mtime 을 0 으로 고정해 내용이 같으면 바이트도 같게 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(".tmp")
    with gzip.GzipFile(tmp, "wb", compresslevel=9, mtime=0) as f:
        f.write(body)
    tmp.replace(path)


def canary_alive() -> bool:
    """숙소가 확실히 있는 bbox 에서 좌표가 나오는가."""
    try:
        found = api.parse_coordinates(fetch(CANARY))
    except Exception as exc:  # noqa: BLE001
        print(f"  canary 요청 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    if not found:
        print("  canary 에서 좌표가 0건입니다 — 막혔거나 화면이 바뀌었습니다.",
              file=sys.stderr)
        return False
    return True


class Budget:
    """남은 호출 수. 요청 사이 간격도 여기서 지킨다."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._last = 0.0

    def spend(self) -> None:
        gap = SLEEP - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()
        self.used += 1

    @property
    def left(self) -> int:
        return self.limit - self.used


def collect_box(box: api.Box, page1: str, budget: Budget,
                state: dict) -> tuple[int, bool]:
    """leaf 의 좌표를 커서 끝까지 거둔다. (새로 담은 개수, 끝까지 봤나).

    예산이 도중에 떨어지면 `False` 를 돌려준다. 호출부가 이 bbox 를 frontier 에
    되돌려 다음 실행이 다시 거두게 해야 한다 — 절반만 거둔 구역을 완료로
    처리하면 그 지역 숙소가 이번 바퀴 내내 빠진 채로 남는다. 좌표는 키로
    중복이 걸러지므로 다시 거둬도 손해가 없다.
    """
    points = state["pending"]
    before = len(points)
    pages = [page1]
    complete = True
    for cursor in api.parse_cursors(page1)[1:]:
        if budget.left <= 0:
            complete = False
            break
        budget.spend()
        try:
            pages.append(fetch(box, cursor))
        except Exception as exc:  # noqa: BLE001 — 한 장 못 받아도 나머지는 살린다
            print(f"    커서 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
            complete = False
            break
    for page in pages:
        for lat, lng in api.parse_coordinates(page):
            if api.contains(box, lat, lng):
                points[f"{lat},{lng}"] = 1
    return len(points) - before, complete


def run(state: dict, budget: Budget, canary_every: int, path: Path) -> int:
    """frontier 가 빌 때까지, 또는 예산이 떨어질 때까지 훑는다."""
    frontier: list[list[float]] = state["frontier"]
    stats = state["stats"]
    failures = 0
    since_canary = 0
    saved_at = 0
    # 이번 실행 전까지 쓴 호출 수. 누계를 매번 이 값 + budget.used 로 다시
    # 계산한다 — 증분을 더해 나가면 중간에 빠져나가는 경로(canary 사망·연속
    # 실패)에서 누계가 틀어진다.
    base_calls = stats["calls"]

    def checkpoint(force: bool = False) -> None:
        nonlocal saved_at
        if force or budget.used - saved_at >= CHECKPOINT:
            saved_at = budget.used
            stats["calls"] = base_calls + budget.used
            save_state(state, path)

    while frontier and budget.left > 0:
        if canary_every and since_canary >= canary_every:
            budget.spend()
            if not canary_alive():
                print("  canary 가 죽어 이번 실행을 접습니다.", file=sys.stderr)
                stats["calls"] = base_calls + budget.used
                return 1
            since_canary = 0

        box = api.Box(*frontier.pop())
        budget.spend()
        since_canary += 1
        try:
            page1 = fetch(box)
            failures = 0
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  실패 {failures}/{MAX_FAILURES} {box}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            requeue(frontier, box)              # 큐 앞으로 미뤄 두고
            if failures >= MAX_FAILURES:
                print("  연속 실패가 한도를 넘었습니다 — 이번 실행을 접습니다.",
                      file=sys.stderr)
                stats["calls"] = base_calls + budget.used
                return 1
            time.sleep(SLEEP * failures)        # 물러섰다 다시
            continue

        total = api.parse_total(page1)
        found = api.parse_coordinates(page1)
        verdict = api.decide(total, len(found), box, SPLIT_AT, MIN_DEG)

        if verdict == "split":
            stats["splits"] += 1
            frontier.extend(list(q) for q in api.split(box))
            checkpoint()
            continue

        if api.looks_empty(total, len(found)):
            if (api.needs_empty_confirmation(box, CONFIRM_EMPTY_DEG)
                    and budget.left > 0):
                budget.spend()
                since_canary += 1
                try:
                    again = fetch(box)
                except Exception as exc:  # noqa: BLE001
                    print(f"  빈칸 재확인 실패 {box}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                    requeue(frontier, box)
                    checkpoint()
                    continue
                if not api.looks_empty(api.parse_total(again),
                                       len(api.parse_coordinates(again))):
                    # 첫 응답이 헛돈 것이다. 가지치지 않고 다시 세운다.
                    print(f"  빈칸 재확인에서 숙소가 나왔습니다 — 되돌립니다: {box}",
                          file=sys.stderr)
                    requeue(frontier, box)
                    checkpoint()
                    continue
            stats["empty"] += 1
            checkpoint()
            continue

        stats["leaves"] += 1
        added, done = collect_box(box, page1, budget, state)
        if not done:
            # 예산이 커서 도중에 떨어졌다. 완료로 치지 않고 되돌린다.
            stats["leaves"] -= 1
            requeue(frontier, box)
            print(f"  leaf total={total} +{added} — 예산이 끊겨 되돌립니다.")
            break
        if api.under_collected(total, len(found), SPLIT_AT):
            stats["weak"] += 1
            state["weak_pending"].append(list(box))
        print(f"  leaf total={total} +{added} 누적={len(state['pending'])} "
              f"남은bbox={len(frontier)} 호출={budget.used}/{budget.limit}")
        checkpoint()

    stats["calls"] = base_calls + budget.used
    finish_pass(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="전국 Airbnb 좌표 수집")
    parser.add_argument("--max-calls", type=int, default=1200,
                        help="이번 실행에서 쓸 요청 수 (기본 1200, 약 40분)")
    parser.add_argument("--box", help="전국 대신 이 bbox 만. ne_lat,ne_lng,sw_lat,sw_lng")
    parser.add_argument("--reset", action="store_true", help="frontier 를 처음부터")
    parser.add_argument("--canary-every", type=int, default=150,
                        help="이 호출 수마다 파싱이 살아 있는지 확인 (0 이면 끔)")
    parser.add_argument("--state", type=Path, default=STATE_FILE,
                        help="상태 파일. --box 로 지역만 훑을 때는 따로 주어 "
                             "전국 수집 상태를 덮어쓰지 않게 한다")
    args = parser.parse_args()

    if args.box:
        try:
            box = api.Box(*(float(v) for v in args.box.split(",")))
        except (TypeError, ValueError):
            print("--box 는 ne_lat,ne_lng,sw_lat,sw_lng 네 값입니다.", file=sys.stderr)
            return 2
        if args.state == STATE_FILE:
            print("--box 를 쓸 때는 --state 로 다른 파일을 주세요 — 전국 수집 "
                  "상태를 덮어쓰게 됩니다.", file=sys.stderr)
            return 2
        state = new_state([box])
    elif args.reset:
        state = new_state()
    else:
        state = load_state(args.state)
        begin_pass(state)

    budget = Budget(args.max_calls)
    print(f"시작: 남은bbox={len(state['frontier'])} "
          f"확정={len(state.get('points') or {})} 모으는중={len(state['pending'])} "
          f"예산={budget.limit}")

    budget.spend()
    if not canary_alive():
        print("시작 전 확인에서 막혔습니다 — 아무것도 저장하지 않고 끝냅니다.",
              file=sys.stderr)
        return 1

    code = run(state, budget, args.canary_every, args.state)
    save_state(state, args.state)
    stats = state["stats"]
    print(f"끝: 확정={len(state.get('points') or {})} 모으는중={len(state['pending'])} "
          f"leaf={stats['leaves']} 분할={stats['splits']} 빈칸={stats['empty']} "
          f"덜걷힘={stats['weak']} 호출={budget.used} "
          f"남은bbox={len(state['frontier'])} 완주={state['complete']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
