#!/usr/bin/env bash
# 전국 Airbnb 밀집 지도 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   0 21 * * 0,3 flock -w 7200 ~/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/airbnb_weekly.sh >> /home/dorumugs/.cache/realestate-airbnb.log 2>&1
#
# **flock 은 반드시 유지한다.** daily.sh · redev_daily.sh · schools_monthly.sh ·
# supply_monthly.sh 와 같은 락을 공유한다. 전부 git commit / pull --rebase / push
# 를 하므로 겹쳐 돌면 한쪽 커밋이 유실된다.
#
# 한 번에 전국이 끝나지 않는다
#   bbox 하나로 받아지는 숙소가 약 200건이라 전국을 쿼드트리로 쪼개 훑는다.
#   MAX_CALLS 로 한 번치 예산을 끊고, 못 끝낸 bbox 는 data/airbnb/state.json.gz
#   에 남아 다음 실행이 이어받는다. 한 바퀴를 다 돌면 그 결과가 화면에 나가고
#   다음 실행이 새 바퀴를 시작한다 — 새 바퀴를 도는 동안에도 화면에는 **지난
#   바퀴 결과**가 그대로 남는다(collect_airbnb.py 의 points/pending 참고).
#
#   실측(2026-09-22 첫 완주): 전국 한 바퀴에 5,481콜 · 숙소 59,265곳.
#   그때는 전국 사각형 하나로 시작해 규슈까지 훑었으니 지금은 이보다 적게 든다.
#   3,000콜을 주 2회(일·수) 돌면 한 바퀴가 대략 일주일이다 — 화면의 "낡음"
#   경고 기준(assets/realestate/freshness.js 의 airbnb: 16일)과 맞춰 둔 값이다.
#
#   21시에 도는 이유: 3,000콜이면 락을 100분쯤 잡는다. 새벽에 돌리면
#   04:30 daily.sh(실거래 9,000콜)가 그만큼 밀린다.
#
# 수집 예절 — 성능 조정 항목이 아니다
#   요청 간격 2초는 collect_airbnb.py 에 박혀 있다. MAX_CALLS 를 키우면 한 번에
#   오래 돌 뿐 초당 요청 수는 그대로다. **간격을 줄이지 말 것.**
#
# 인증
#   없다. 로그인 없이 보이는 공개 검색 화면만 읽는다.
#
# 환경변수
#   AUTO_COMMIT  1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH    1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   MAX_CALLS    이번 실행의 요청 예산. 기본 3000 (2초 간격이라 약 100분)
#   RESET        1 이면 지금까지 모은 것을 버리고 전국을 처음부터 다시 훑는다

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
MAX_CALLS="${MAX_CALLS:-3000}"
RESET="${RESET:-0}"

FAILED=0

echo "===== $(date '+%F %T') Airbnb 밀집도 수집 시작 (예산 ${MAX_CALLS}콜) ====="

COLLECT_ARGS=(--max-calls "$MAX_CALLS")
[ "$RESET" = "1" ] && COLLECT_ARGS+=(--reset)

# 수집이 실패해도 집계는 돌린다. 지난주까지 모은 좌표가 빈 화면보다 낫고,
# collect_airbnb.py 는 막혔다고 판단하면 기존 상태를 건드리지 않고 물러난다.
if ! python3 -u scripts/collect_airbnb.py "${COLLECT_ARGS[@]}"; then
  FAILED=1
  echo "수집 실패 — 지금까지 모은 좌표로 집계만 다시 굽습니다." >&2
fi

# 좌표가 하나도 없으면 build_airbnb.py 가 스스로 멈춘다 — 빈 집계로 기존
# 대시보드를 덮어쓰지 않는다.
if ! python3 -u scripts/build_airbnb.py; then
  FAILED=1
  echo "집계 실패 — assets/realestate/airbnb.json 은 이전 것 그대로입니다." >&2
fi

if ! python3 -u scripts/check_freshness.py airbnb; then
  FAILED=1
  echo "산출물이 낡았습니다 — 위의 신선도 표를 보세요." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 손으로 쓴 소스(airbnb-app.js/airbnb.css 등)가 섞이지 않도록 이 크론이 만드는
# 경로만 스테이징한다. 다른 크론과 겹치는 경로가 없다.
#
# **build_airbnb.py 가 쓰는 곳을 하나도 빠뜨리지 말 것.** assets/realestate/dong
# 이 빠져 있던 적이 있는데, 그러면 두 가지가 한꺼번에 망가진다 — 동별 숫자가
# 영영 갱신되지 않고, 커밋 안 된 파일이 작업트리에 쌓여 다른 크론의
# `pull --rebase --autostash` 가 충돌한다.
# dong/ 에는 경계(build_geo.py --dong)와 숙소 수(build_airbnb.py)가 함께 있다.
# 크론은 숫자만 바꾸지만, 경계를 다시 만들고 커밋하지 않았다면 그것도 함께
# 딸려 간다 — 둘 다 생성물이라 문제는 아니다.
TARGETS="data/airbnb assets/realestate/airbnb.json assets/realestate/airbnb assets/realestate/dong"

# shellcheck disable=SC2086
if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 Airbnb 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh Airbnb density data

수집 스크립트가 자동 갱신한 전국 Airbnb 숙소 좌표와 시군구·격자 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh Airbnb density data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

# shellcheck disable=SC2086
git add $TARGETS
git commit -q -m "$COMMIT_MSG"
echo "커밋 완료."

if [ "$AUTO_PUSH" != "1" ]; then
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# cron 에는 ssh-agent 도 tty 도 없다. 물어보는 대신 바로 실패해야 한다.
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! git pull --rebase --autostash -q origin gh-pages; then
  echo "pull --rebase 실패. 충돌을 수동으로 정리한 뒤 push 하세요." >&2
  exit 1
fi
if ! git push -q origin gh-pages; then
  echo "push 실패 — 커밋이 로컬에만 남았습니다. 사이트는 갱신되지 않습니다." >&2
  exit 1
fi
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/gh-pages)" ]; then
  echo "push 뒤에도 origin/gh-pages 가 HEAD 와 다릅니다 — 사이트가 안 바뀝니다." >&2
  exit 1
fi
echo "push 완료."

[ "$FAILED" = "1" ] && exit 1
exit 0
