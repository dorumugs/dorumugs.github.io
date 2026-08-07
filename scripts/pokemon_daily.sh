#!/usr/bin/env bash
# 포켓몬 카드 지수 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   30 7 * * * /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/pokemon_daily.sh >> /home/dorumugs/.cache/realestate-pokemon.log 2>&1
#
# 부동산 3종과 같은 flock 을 쓴다. 넷 다 git commit/push 를 하므로 겹치면
# 한쪽 커밋이 유실된다. 07:30 은 redev_daily.sh(06:10) 가 끝난 뒤다.
#
# 스스로 갈라진다.
#   전수 스캔 미완  scan 모드로 예산만큼 훑는다 (기본 6000콜, 4일이면 끝)
#   전수 스캔 완료  daily 모드로 유니버스 300장만 받는다 (약 70초)
#
# 인증키가 필요 없다. TCGdex 는 키 없이 열려 있다.
#
# 환경변수
#   AUTO_COMMIT      1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH        1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   SCAN_MAX_CALLS   전수 스캔 1회 예산. 기본 6000

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
SCAN_MAX_CALLS="${SCAN_MAX_CALLS:-6000}"

STATE="data/pokemon/scan_state.json"

echo "===== $(date '+%F %T') 포켓몬 카드 수집 시작 ====="

FAILED=0

SCAN_DONE=0
if [ -f "$STATE" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$STATE')).get('complete') else 1)"; then
  SCAN_DONE=1
fi

if [ "$SCAN_DONE" = "1" ]; then
  echo "전수 스캔 완료 상태 — 유니버스 갱신만 돌립니다."
  if ! python3 -u scripts/collect_pokemon.py --mode daily; then
    FAILED=1
    echo "일일 가격 수집 실패 — TCGdex 응답을 확인하세요." >&2
  fi
else
  echo "전수 스캔 진행 중 — 예산 ${SCAN_MAX_CALLS} 콜."
  if ! python3 -u scripts/collect_pokemon.py --mode scan --max-calls "$SCAN_MAX_CALLS"; then
    FAILED=1
    echo "전수 스캔 실패 — TCGdex 응답을 확인하세요." >&2
  fi
fi

# 집계. 수집이 일부 실패해도 있는 원본으로 다시 굽는다 — 어제 것보다 낫다.
if ! python3 -u scripts/build_pokemon.py; then
  FAILED=1
  echo "build_pokemon.py 가 비정상 종료했습니다." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

TARGETS="data/pokemon assets/pokemon"

if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 포켓몬 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh Pokemon card index data

수집 스크립트가 자동 갱신한 카드 가격과 그 지수 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh Pokemon card index data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

git add $TARGETS
git commit -m "$COMMIT_MSG"

if [ "$AUTO_PUSH" = "1" ]; then
  git pull --rebase
  git push
fi

echo "===== $(date '+%F %T') 포켓몬 카드 수집 종료 (FAILED=$FAILED) ====="
[ "$FAILED" = "1" ] && exit 1
exit 0
