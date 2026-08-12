#!/usr/bin/env bash
# 재개발·재건축 대시보드 데이터를 갱신한다. cron 에서 부르는 진입점.
#
#   crontab -e
#   10 6 * * * AUTO_COMMIT=1 AUTO_PUSH=1 /home/dorumugs/Projects/dorumugs.github.io/scripts/redev_daily.sh >> /home/dorumugs/.cache/realestate-redev.log 2>&1
#
# daily.sh(실거래) 뒤에 돌려야 한다. build_redevelopment.py 가 data/trades 를 읽어
# 단지별 평당가와 이벤트 스터디를 내기 때문이다. daily.sh 가 04:30 에 시작해 길면
# 한 시간 남짓 걸리므로 06:10 으로 잡았다.
#
# 주기가 다른 수집을 한 스크립트에 모았다. 크론 줄을 여러 개 두면 어느 줄이
# 어느 파일을 건드리는지 흩어져서, 커밋 충돌이 났을 때 추적하기 어렵다.
#
#   매일        연립·다세대 실거래 (최근 3개월 갱신 + 못 받은 칸 백필)
#   월요일      정비사업장 목록·추진경과 (단계가 바뀐 곳 + 오래된 곳 일부)
#   매월 5일    건축물대장 · 브이월드 필지/용도지역 · 지자체 조례 용적률
#   항상        집계 후 커밋·푸시
#
# 각 단계는 실패해도 다음 단계를 막지 않는다. 어제 것보다 나은 상태로 끝내는 게
# 목적이라, 일부 수집이 실패해도 있는 원본으로 집계를 다시 굽고 커밋한다.
# 대신 하나라도 실패하면 마지막에 비정상 종료해 cron 로그에 남긴다.
#
# 인증
#   VWORLD_API_KEY  저장소 루트 .env (커밋되지 않음). 없으면 브이월드 단계만 건너뛴다
#   LAW_OC          없으면 'test' 로 동작 (법제처가 공개 시험용으로 열어둔 값)
#   DATA_GO_KR_API_KEY  환경변수 또는 ~/.claude.json
#
# 환경변수
#   AUTO_COMMIT  1 이면 결과를 gh-pages 에 커밋. 기본은 커밋하지 않음
#   AUTO_PUSH    1 이면 커밋 후 push. AUTO_COMMIT=1 일 때만 의미 있음
#   FORCE_MONTHLY 1 이면 날짜와 무관하게 월간 수집도 돌린다 (첫 설치·수동 점검용)
#   FORCE_WEEKLY  1 이면 요일과 무관하게 사업장 수집도 돌린다

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AUTO_COMMIT="${AUTO_COMMIT:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
FORCE_MONTHLY="${FORCE_MONTHLY:-0}"
FORCE_WEEKLY="${FORCE_WEEKLY:-0}"

DOW="$(date '+%u')"   # 1=월요일
DOM="$(date '+%d')"

FAILED=0

run_step() {
  local label="$1"; shift
  echo "----- ${label} -----"
  if ! "$@"; then
    FAILED=1
    echo "${label} 실패 — 로그를 확인하세요." >&2
    return 1
  fi
  return 0
}

echo "===== $(date '+%F %T') 재개발 수집 시작 (요일 ${DOW} · 일자 ${DOM}) ====="

# 1) 매일 — 연립·다세대 실거래.
#    재개발 구역 안은 아파트가 아니라 다세대·연립이라 이 데이터가 있어야
#    재개발 사업장의 인가 전후 가격을 볼 수 있다.
#    동시 3요청을 넘기면 800콜 언저리에서 429 를 맞는다 (일일 한도가 아니라 버스트 제한).
run_step "연립·다세대 실거래" \
  python3 -u scripts/collect_villa_trades.py --refresh 3 --max-calls 8000 --workers 3 || true

# 2) 월요일 — 정비사업장 목록과 추진경과.
#    정비사업은 몇 달 단위로 움직여 매일 전량을 다시 받을 이유가 없다.
#    목록은 매번 받고(1콜), 추진경과는 단계가 바뀐 곳과 오래된 곳 일부만 받는다.
if [ "$DOW" = "1" ] || [ "$FORCE_WEEKLY" = "1" ]; then
  run_step "정비사업장·추진경과" \
    python3 -u scripts/collect_projects.py --max-calls 600 --refresh 40 --workers 3 --sleep 0.3 || true
else
  echo "----- 정비사업장: 월요일에만 받습니다 (오늘 요일 ${DOW}) -----"
fi

# 3) 매월 5일 — 잘 안 바뀌는 것들.
#    건축물대장은 준공·증축이 있어야 바뀌고, 지적도·용도지역은 도시계획 결정이
#    있어야 바뀐다. 조례는 개정이 잦지 않다.
if [ "$DOM" = "05" ] || [ "$FORCE_MONTHLY" = "1" ]; then
  run_step "건축물대장" python3 -u scripts/collect_bldrgst.py --max-calls 2000 --refresh || true

  # 브이월드는 키가 있어야 한다. 없으면 조용히 건너뛴다 — 서울은 UPIS 지적도로
  # 이미 채워져 있어 이 단계가 빠져도 화면이 무너지지 않는다.
  if grep -qs '^VWORLD_API_KEY=' .env || [ -n "${VWORLD_API_KEY:-}" ]; then
    run_step "브이월드 필지·용도지역" python3 -u scripts/collect_vworld.py --max-calls 7000 || true
  else
    echo "----- 브이월드: VWORLD_API_KEY 가 없어 건너뜁니다 -----"
  fi

  run_step "지자체 조례 용적률" python3 -u scripts/collect_ordinance.py || true

  # 정비구역 도형은 UPIS 비공식 프록시라 막힐 수 있다. 실패해도 대표점 없이
  # 화면이 성립하도록 설계했으므로 여기서 멈추지 않는다.
  run_step "정비구역 도형" python3 -u scripts/collect_zones.py || true
else
  echo "----- 월간 수집(대장·브이월드·조례·구역): 매월 5일에만 받습니다 (오늘 ${DOM}일) -----"
fi

# 4) 집계. 수집이 일부 실패해도 있는 원본으로 다시 굽는다.
run_step "집계" python3 -u scripts/build_redevelopment.py || true

# 산출물이 실제로 최신인지 따로 묻는다. 수집이 조용히 깨져도 집계는 옛 원본으로
# 성공하므로, 여기까지 왔다는 사실만으로는 데이터가 최신이라는 보장이 없다.
if ! python3 -u scripts/check_freshness.py redev; then
  FAILED=1
  echo "산출물이 낡았습니다 — 위의 신선도 표를 보세요." >&2
fi

if [ "$AUTO_COMMIT" != "1" ]; then
  echo "AUTO_COMMIT 이 꺼져 있어 커밋하지 않습니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 손으로 쓴 소스(redev-app.js/redev.css 등)가 섞이지 않도록 이 크론이 만드는
# 경로만 스테이징한다. daily.sh·schools_monthly.sh 와 겹치는 경로가 없다.
TARGETS="data/villa_trades data/projects data/zones data/parcels data/vworld \
data/bldrgst data/ordinance data/state/villa_state.json data/state/project_state.json \
data/state/vworld_state.json data/state/bldrgst_state.json data/state/ordinance_state.json \
assets/realestate/redev.json assets/realestate/redev"

# shellcheck disable=SC2086
if [ -z "$(git status --porcelain $TARGETS)" ]; then
  echo "변경된 재개발 파일이 없어 커밋을 건너뜁니다."
  [ "$FAILED" = "1" ] && exit 1
  exit 0
fi

# 전역 git config 는 건드리지 않고 이 커밋에만 신원을 지정한다.
export GIT_AUTHOR_NAME="Jaehyun So"
export GIT_AUTHOR_EMAIL="dorumugs@gmail.com"
export GIT_COMMITTER_NAME="Jaehyun So"
export GIT_COMMITTER_EMAIL="dorumugs@gmail.com"

COMMIT_MSG="Refresh redevelopment data

수집 스크립트가 자동 갱신한 연립·다세대 실거래와 정비사업 추진경과, 그 집계본.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
if [ "$FAILED" = "1" ]; then
  COMMIT_MSG="Refresh redevelopment data (partial)

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
git push -q origin gh-pages
echo "푸시 완료."

[ "$FAILED" = "1" ] && exit 1
exit 0
