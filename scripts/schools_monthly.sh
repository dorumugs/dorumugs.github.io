#!/usr/bin/env bash
# 학군 지도 원본을 월 1회 새로 받는다. cron 에서 부르는 진입점.
#
#   crontab -e
#   40 4 3 * * /home/dorumugs/Projects/dorumugs.github.io/scripts/schools_monthly.sh >> /home/dorumugs/.cache/realestate-schools.log 2>&1
#
# 왜 매일이 아니라 월 1회인가
#   - 학교 위치·특목고 지정: 학교는 자주 안 바뀐다
#   - 졸업생 진로 현황: 연 1회(11월) 공시다. 월 1회면 공시 직후 한 달 안에 잡힌다
#   - 진학률 수집은 학교 하나에 요청 하나라 1,400여 곳 × 3년 = 4,000회를 넘는다.
#     공개 사이트에 매일 이만큼 보내는 건 예의가 아니다
#
# daily.sh(실거래)와 파일이 겹치지 않는다. 이쪽은 data/schools.csv.gz 와
# data/progression_school.csv.gz, 그리고 그 집계본만 건드린다.
#
# 환경변수
#   AUTO_COMMIT 1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH   1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   SKIP_PROGRESSION 1 이면 진학률 수집을 건너뛴다(오래 걸릴 때 임시로)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
SKIP_PROGRESSION="${SKIP_PROGRESSION:-0}"

echo "===== $(date '+%F %T') 학군 수집 시작 ====="

FAILED=0

# 1) 학교 위치 + 특목고 분류(NEIS). 키가 없으면 여기서 멈추므로 먼저 돌린다.
if ! python3 -u scripts/collect_schools.py; then
  FAILED=1
  echo "학교 위치 수집 실패 — NEIS_API_KEY(.env) 와 data.go.kr 키를 확인하세요." >&2
fi

# 2) 학교별 졸업생 진로 현황(학교알리미). 1,400여 곳 × 3년이라 20분 안팎 걸린다.
if [ "$SKIP_PROGRESSION" = "1" ]; then
  echo "SKIP_PROGRESSION=1 — 진학률 수집을 건너뜁니다."
elif ! python3 -u scripts/collect_progression_school.py; then
  FAILED=1
  echo "진학률 수집 실패 — 학교알리미 화면 구조가 바뀌었는지 확인하세요." >&2
fi

# 3) 집계. 수집이 일부 실패해도 있는 원본으로 다시 굽는다 — 어제 것보다 낫다.
for step in build_schools build_progression_school; do
  if ! python3 -u "scripts/${step}.py"; then
    FAILED=1
    echo "${step}.py 가 비정상 종료했습니다." >&2
  fi
done

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

TARGETS="data/schools.csv.gz data/progression_school.csv.gz \
assets/realestate/schools.json assets/realestate/progression_school.json"

if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 학군 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh school map and progression data

수집 스크립트가 자동 갱신한 학교 위치·특목고 분류와 학교별 특목고·자사고
진학률, 그리고 그 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh school map and progression data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

# 손으로 쓴 소스가 섞이지 않도록 이 크론이 만드는 경로만 스테이징한다.
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
git push -q origin gh-pages
echo "푸시 완료."

[ "$FAILED" = "1" ] && exit 1
exit 0
