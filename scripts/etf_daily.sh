#!/usr/bin/env bash
# ETF·테마 20일 모멘텀 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   30 18 * * 1-5 /usr/bin/flock -w 7200 /home/dorumugs/.cache/realestate.lock env AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/etf_daily.sh >> /home/dorumugs/.cache/realestate-etf.log 2>&1
#
# 18:30 은 장 마감(15:30) 뒤 종가가 확정된 다음이다. 평일에만 돈다 — 주말엔
# 새 거래일이 없어 받아 봐야 같은 값이다.
#
# 부동산 3종·포켓몬과 **같은 flock 을 공유한다.** 다섯 다 git commit/push 를
# 하므로 겹치면 한쪽 커밋이 유실된다. 반드시 유지할 것.
#
# 두 단계다.
#
#   1. 수집   테마 265 · 업종 79 · ETF 1,160 · 일봉 4,407심볼. 전체 백필이
#             약 6,000요청에 1분이다. 주기는 collect_stocks.py 가 파일이 얼마나
#             낡았는지 보고 스스로 가른다 — 그룹·구성종목은 7일, 일봉 전체
#             재수집은 30일. 요일로 가르지 않으므로 하루 걸러도 다음 실행이 메운다.
#   2. 집계   지표와 등급을 계산해 assets/etf/*.json 을 굽는다.
#
# 인증키가 필요 없다. 네이버 세 곳 다 키 없이 열려 있다.
#
# 커밋 대상은 assets/etf 뿐이다. data/stocks 는 .gitignore 로 막혀 있다 —
# 일봉 캐시가 gzip 8MB 라 매일 커밋하면 1년에 2GB 가 쌓이는데, 2분이면 다시
# 받는 재생성 캐시라 저장소에 넣을 값어치가 없다.
#
# 환경변수
#   AUTO_COMMIT   1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH     1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   MAX_CALLS     1회 요청 예산. 기본 20000 (전체 백필이 다 들어간다)
#   FORCE_FULL    1 이면 일봉을 전체 재수집
#   FORCE_GROUPS  1 이면 테마·업종·시장구분을 강제 갱신

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
MAX_CALLS="${MAX_CALLS:-20000}"

COLLECT_ARGS=(--max-calls "$MAX_CALLS")
[ "${FORCE_FULL:-0}" = "1" ] && COLLECT_ARGS+=(--full-bars)
[ "${FORCE_GROUPS:-0}" = "1" ] && COLLECT_ARGS+=(--refresh-groups)

echo "===== $(date '+%F %T') ETF·테마 수집 시작 ====="

FAILED=0

if ! python3 -u scripts/collect_stocks.py "${COLLECT_ARGS[@]}"; then
  FAILED=1
  echo "수집이 끝까지 못 갔습니다 — 네이버 응답을 확인하세요." >&2
fi

# 수집이 일부 실패해도 있는 캐시로 다시 굽는다. 어제 집계본보다 낫다.
# 다만 캐시가 아예 없으면 집계도 못 하므로 그때는 진짜 실패다.
if ! python3 -u scripts/build_etf_theme.py; then
  FAILED=1
  echo "build_etf_theme.py 가 비정상 종료했습니다." >&2
fi

# 등급 성적표. 3.5년치를 매일 다시 채점해 최신 시장까지 반영한다. 약 20초.
# 실패해도 어제 성적표가 남으므로 전체를 죽이지 않는다 — 화면의 나머지는 멀쩡하다.
if ! python3 -u scripts/backtest_etf.py; then
  echo "backtest_etf.py 가 비정상 종료했습니다. 지난 성적표를 그대로 씁니다." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

TARGETS="assets/etf"

if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh ETF theme momentum data

수집 스크립트가 자동 갱신한 테마·업종·ETF 20일 모멘텀 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh ETF theme momentum data (partial)

수집·집계 일부가 실패해 파일 상태가 최신이 아닐 수 있음. 로그 확인 필요.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
fi

git add $TARGETS
git commit -m "$COMMIT_MSG"

if [ "$AUTO_PUSH" = "1" ]; then
  git pull --rebase
  git push
fi

echo "===== $(date '+%F %T') ETF·테마 수집 종료 (FAILED=$FAILED) ====="
[ "$FAILED" = "1" ] && exit 1
exit 0
