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
#
# build_dashboard.py 는 summary.json/구별 JSON 이 예산을 넘으면 1로 종료한다.
# set -e 아래서 그냥 호출하면 그 순간 스크립트가 죽어 커밋 단계 자체를 못 가고,
# 그날 수집한 data/trades 마저 커밋되지 않은 채 유실된다 — 다음 cron 실행도
# 같은 지점에서 또 죽으므로 사람이 개입할 때까지 매일 반복된다. 집계 실패는
# 기록만 해 두고 커밋/푸시는 그대로 진행한다: 집계본 없이 수집분만 커밋하는 게
# 아무것도 커밋하지 않는 것보다 항상 낫다. 대신 스크립트 마지막에 반드시
# 비정상 종료해 cron 로그에 남긴다.
echo "----- 집계 시작 -----"
BUILD_FAILED=0
if ! python3 -u scripts/build_dashboard.py; then
  BUILD_FAILED=1
  echo "집계 실패 — build_dashboard.py 가 예산 초과 등으로 비정상 종료했습니다. 집계본 없이 수집분만 커밋합니다." >&2
fi

# 학교 좌표는 지도 투영에 묶여 있다. 실거래 집계와 같이 돌려 어긋나지 않게 한다.
# 학교 원본(data/schools.csv.gz)은 collect_schools.py 로 따로 받는다 — 매일 받지 않는다.
if [ -f data/schools.csv.gz ]; then
  if ! python3 -u scripts/build_schools.py; then
    BUILD_FAILED=1
    echo "학교 집계 실패 — build_schools.py 가 비정상 종료했습니다." >&2
  fi
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  if [ "$BUILD_FAILED" = "1" ]; then exit 1; fi
  exit 0
fi

if [ -z "$(git status --porcelain data assets/realestate/summary.json assets/realestate/sgg assets/realestate/schools.json)" ]; then
  echo "변경된 데이터 파일이 없어 커밋을 건너뜁니다."
  if [ "$BUILD_FAILED" = "1" ]; then exit 1; fi
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

MONTHS=$(git status --porcelain data/trades | wc -l | tr -d ' ')
COMMIT_MSG="Accumulate Seoul/Gyeonggi apartment trade data

수집 스크립트가 자동 갱신한 월별 실거래가 파일 ${MONTHS}개와 대시보드 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$BUILD_FAILED" = "1" ]; then
  COMMIT_MSG="Accumulate Seoul/Gyeonggi apartment trade data

수집 스크립트가 자동 갱신한 월별 실거래가 파일 ${MONTHS}개. 집계 단계는 예산 초과 등으로 실패해
대시보드 파일 상태가 최신이 아닐 수 있음.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi
# 손으로 쓴 소스(app.js/charts.js/map.js/data.js/palette.js/dashboard.css)를
# 실수로 함께 커밋하지 않도록 생성물 경로만 스테이징한다.
git add data assets/realestate/summary.json assets/realestate/sgg assets/realestate/schools.json
git commit -q -m "$COMMIT_MSG"

echo "커밋 완료."

if [ "$AUTO_PUSH" != "1" ]; then
  if [ "$BUILD_FAILED" = "1" ]; then exit 1; fi
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

# 집계가 실패했으면 커밋/푸시는 끝까지 했더라도 cron 로그·종료 코드에는 남긴다.
if [ "$BUILD_FAILED" = "1" ]; then exit 1; fi
