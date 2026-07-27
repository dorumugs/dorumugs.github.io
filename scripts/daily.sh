#!/usr/bin/env bash
# 매일 실거래가를 이어서 수집한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   30 4 * * * /home/dorumugs/Projects/dorumugs.github.io/scripts/daily.sh >> /home/dorumugs/.cache/realestate-collect.log 2>&1
#
# 환경변수
#   MAX_CALLS   이번 실행 최대 API 호출 수 (기본 900)
#   AUTO_COMMIT 1 이면 수집 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH   1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MAX_CALLS="${MAX_CALLS:-900}"
AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"

echo "===== $(date '+%F %T') 수집 시작 (예산 ${MAX_CALLS}콜) ====="

python3 -u scripts/collect_trades.py --max-calls "$MAX_CALLS"

# 수집이 일일 한도로 중간에 멈춘 날에도 집계는 돌린다.
# 그날까지 받은 데이터로 만든 대시보드가 어제 것보다 낫다.
echo "----- 집계 시작 -----"
python3 -u scripts/build_dashboard.py

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  exit 0
fi

if [ -z "$(git status --porcelain data assets/realestate)" ]; then
  echo "변경된 데이터 파일이 없어 커밋을 건너뜁니다."
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

MONTHS=$(git status --porcelain data/trades | wc -l | tr -d ' ')
git add data assets/realestate
git commit -q -m "Accumulate Seoul/Gyeonggi apartment trade data

수집 스크립트가 자동 갱신한 월별 실거래가 파일 ${MONTHS}개와 대시보드 집계본."

echo "커밋 완료."

if [ "$AUTO_PUSH" != "1" ]; then
  exit 0
fi

# cron 에는 ssh-agent 도 tty 도 없다. 키에 암호가 없어야 하고, 물어보는 대신 바로 실패해야 한다.
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

# 다른 곳에서 먼저 푸시했을 수 있으니 rebase 로 맞춘 뒤 올린다.
# --autostash 가 없으면 작업 중인 미스테이징 변경(수집과 무관한 편집)만으로도
# rebase 가 거부되어 push 가 통째로 막힌다. 실제로 첫 실행에서 그렇게 실패했다.
if ! git pull --rebase --autostash -q origin gh-pages; then
  echo "pull --rebase 실패. 충돌을 수동으로 정리한 뒤 push 하세요." >&2
  exit 1
fi

git push -q origin gh-pages
echo "push 완료."
