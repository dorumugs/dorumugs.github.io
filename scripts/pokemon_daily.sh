#!/usr/bin/env bash
# 포켓몬 카드 시세 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   30 7 * * * /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/pokemon_daily.sh >> /home/dorumugs/.cache/realestate-pokemon.log 2>&1
#
# 부동산 3종과 같은 flock 을 쓴다. 넷 다 git commit/push 를 하므로 겹치면
# 한쪽 커밋이 유실된다. 07:30 은 redev_daily.sh(06:10) 가 끝난 뒤다.
#
# 네 갈래를 돈다.
#
#   1. 글로벌 시세  TCGdex 에서 가격이 잡히는 카드 전부(약 17,700장). 동시 4로
#                   약 70분. 세트마다 저장해 중간에 죽어도 다음 실행이 이어받는다.
#   2. 대체 사진    TCGdex·TCGplayer 둘 다 사진이 없는 카드를 pokemontcg.io 로
#                   메운다. 새로 생긴 구멍만 확인하므로 평소엔 요청이 없다.
#   3. 국내 시세    KREAM 시세표. 헤드리스 Chrome 으로 페이지를 열어 받는다.
#   4. 집계         화면용 JSON 두 벌.
#
# 한글 이름(PokeAPI)은 거의 바뀌지 않아 파일이 없을 때만 받는다.
#
# 인증키가 필요 없다. 네 곳 다 키 없이 열려 있다.
#
# KREAM 은 서버가 간헐적으로 500 을 낸다. 실패해도 마지막 스냅샷이 남으므로
# 전체 실행을 실패로 치지 않는다 — 다만 한 번도 받은 적이 없으면 실패다.
#
# 환경변수
#   AUTO_COMMIT      1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH        1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   MAX_CALLS        1회 예산. 기본 30000 (전 카드가 다 들어간다)
#   SKIP_KREAM       1 이면 국내 시세 수집을 건너뛴다 (Chrome 이 없는 환경)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
MAX_CALLS="${MAX_CALLS:-30000}"

echo "===== $(date '+%F %T') 포켓몬 카드 수집 시작 ====="

FAILED=0

# 한글 이름은 거의 바뀌지 않는다. 없을 때만 받는다.
if [ ! -f data/pokemon/species_ko.json ]; then
  echo "한글 이름이 없어 먼저 받습니다."
  if ! python3 -u scripts/collect_pokemon.py --species; then
    FAILED=1
    echo "한글 이름 수집 실패 — PokeAPI 응답을 확인하세요." >&2
  fi
fi

if ! python3 -u scripts/collect_pokemon.py --max-calls "$MAX_CALLS"; then
  FAILED=1
  echo "카드 시세 수집 실패 — TCGdex 응답을 확인하세요." >&2
fi

# 사진이 빠진 카드를 pokemontcg.io 로 메운다. 새로 생긴 구멍만 두들긴다.
if ! python3 -u scripts/collect_card_art.py; then
  FAILED=1
  echo "대체 사진 수집 실패 — pokemontcg.io 응답을 확인하세요." >&2
fi

# 국내 원화 시세. KREAM 이 자주 흔들려서 실패를 다르게 다룬다.
if [ "${SKIP_KREAM:-0}" = "1" ]; then
  echo "SKIP_KREAM 이 켜져 있어 국내 시세를 건너뜁니다."
elif ! python3 -u scripts/collect_kream.py; then
  if [ -f data/pokemon/kream.json.gz ]; then
    echo "국내 시세를 못 받았습니다. 마지막 스냅샷을 그대로 씁니다." >&2
  else
    FAILED=1
    echo "국내 시세를 한 번도 받지 못했습니다 — KREAM 응답을 확인하세요." >&2
  fi
fi

# 집계. 수집이 일부 실패해도 있는 원본으로 다시 굽는다 — 어제 것보다 낫다.
if ! python3 -u scripts/build_pokemon.py; then
  FAILED=1
  echo "build_pokemon.py 가 비정상 종료했습니다." >&2
fi

if [ -f data/pokemon/kream.json.gz ]; then
  if ! python3 -u scripts/build_kream.py; then
    FAILED=1
    echo "build_kream.py 가 비정상 종료했습니다." >&2
  fi
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

COMMIT_MSG="Refresh Pokemon card price data

수집 스크립트가 자동 갱신한 글로벌·국내 카드 시세와 그 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh Pokemon card price data (partial)

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
