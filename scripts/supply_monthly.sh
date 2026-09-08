#!/usr/bin/env bash
# 아파트 착공 × 금리 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   20 5 5,25 * * flock -w 7200 ~/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/supply_monthly.sh >> /home/dorumugs/.cache/realestate-supply.log 2>&1
#
# **flock 은 반드시 유지한다.** daily.sh · redev_daily.sh · schools_monthly.sh 와
# 같은 락을 공유한다. 넷 다 git commit / pull --rebase / push 를 하므로 겹쳐 돌면
# 한쪽 커밋이 유실된다.
#
# 월간 통계라 매일 돌 이유가 없다. **월 2회인 이유는 잠정치다** — 통계누리는
# 최근 약 10개월을 `p)` 로 주고 확정되면서 값이 바뀌므로, 한 달에 한 번만 받으면
# 갱신된 잠정치를 늦게 반영한다.
#
# 인증
#   ECOS_API_KEY  저장소 루트 .env (커밋되지 않음). 없으면 수집 자체가 실패한다.
#                 sample 키는 10건 제한이라 쓰지 않는다 — 반쪽 시계열이 만들어진다.
#   통계누리는 인증이 필요 없다.
#
# 환경변수
#   AUTO_COMMIT  1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH    1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   FORCE_FULL   1 이면 확정월 캐시를 무시하고 2011년부터 전부 다시 받는다

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
FORCE_FULL="${FORCE_FULL:-0}"

FAILED=0

echo "===== $(date '+%F %T') 착공·금리 수집 시작 ====="

COLLECT_ARGS=()
[ "$FORCE_FULL" = "1" ] && COLLECT_ARGS+=(--force)

# 수집이 실패하면 기존 원본을 그대로 두고 넘어간다. 지난달 데이터가 빈 파일보다
# 낫고, 집계는 있는 원본으로 다시 구울 수 있다.
if ! python3 -u scripts/collect_supply.py "${COLLECT_ARGS[@]}"; then
  FAILED=1
  echo "수집 실패 — 기존 원본으로 집계만 다시 굽습니다." >&2
fi

# build_supply.py 는 시도 합과 응답의 `총계` 를 대조해 1% 넘게 어긋나면 스스로
# 멈춘다. 시도 라벨이 바뀌었는데 조용히 반쪽 숫자를 내는 것보다 낫다.
if ! python3 -u scripts/build_supply.py; then
  FAILED=1
  echo "집계 실패 — assets/realestate/supply.json 은 이전 것 그대로입니다." >&2
fi

# 산출물이 실제로 최신인지 따로 묻는다. 수집이 조용히 깨져도 집계는 옛 원본으로
# 성공하므로, 여기까지 왔다는 사실만으로는 데이터가 최신이라는 보장이 없다.
if ! python3 -u scripts/check_freshness.py supply; then
  FAILED=1
  echo "산출물이 낡았습니다 — 위의 신선도 표를 보세요." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 손으로 쓴 소스(supply-app.js/supply.css 등)가 섞이지 않도록 이 크론이 만드는
# 경로만 스테이징한다. 다른 세 크론과 겹치는 경로가 없다.
TARGETS="data/supply assets/realestate/supply.json"

# shellcheck disable=SC2086
if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 착공·금리 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh apartment starts and rates data

수집 스크립트가 자동 갱신한 시도별 아파트 착공과 한국은행 금리, 그 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh apartment starts and rates data (partial)

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
