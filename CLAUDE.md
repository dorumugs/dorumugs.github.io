# CLAUDE.md

이 저장소(`dorumugs.github.io`)는 **Jekyll + minimal-mistakes 테마** 기반의 개인 기술 블로그 **KayserDocs** 입니다. 글만 있는 게 아니라 **부동산 데이터 파이프라인과 대시보드**가 함께 들어 있습니다. Claude 가 작업할 때 따라야 할 규칙을 모아둡니다.

## 저장소 기본 정보

- 사이트: `https://dorumugs.github.io`
- 테마: `minimal-mistakes-jekyll` (저장소 자체가 테마 포크. `_layouts/`, `_includes/`, `_sass/` 등이 그대로 들어있음)
- 기본 브랜치: **`gh-pages`** (GitHub Pages 가 이 브랜치를 그대로 서빙. `main` 없음)
- locale: `ko-KR`, permalink: `/:categories/:title/`
- 작업 환경: **Linux** (`/home/dorumugs/Projects/dorumugs.github.io`)

## 글 작성 규칙

톤·구조·마스킹 규칙은 메모리에 있습니다. 새 글을 쓰거나 노트를 글로 옮길 때 먼저 읽으세요.

`~/.claude/projects/-home-dorumugs-Projects-dorumugs-github-io/memory/`

핵심만 다시 짚으면

- 파일 위치: `_posts/YYYY-MM-DD-제목.md` (제목에 한글/언더스코어 OK)
- front matter: `layout: single` / `categories: coding` / `tag: [...]` / `author_profile: false` / `toc: true`
- 같은 날짜에 여러 글을 올려도 OK. 단, 제목/파일명은 겹치면 안 됨.
- 글 안에서 다른 글로 링크할 때는 permalink 규칙(`/coding/<slug>/`) 을 따른다.
- **민감정보(사설/공인 IP, 이메일 로컬파트, 토큰류)는 사용자 확인 없이 선제적으로 마스킹**해서 초안을 만든다.
- **모든 글·페이지는 390px 에서 가로로 넘치면 안 된다.** 표는 자체 스크롤 컨테이너 안에서만 넘칠 것. 푸시 전에 확인한다.
- 수식을 넣을 때는 수학이 약한 독자용 쉬운 풀이를 반드시 함께 붙인다.

## 부동산 데이터 파이프라인

`/dashboard/` 아래 도구가 붙어 있고, 전부 크론으로 자동 갱신됩니다.

| 페이지 | 내용 |
|---|---|
| `/dashboard/real-estate/trades/` | 실거래 대시보드 — 서울·경기 72개 시군구, 435만 건 |
| `/dashboard/real-estate/schools/` | 학군 지도 — 사립초·사립중·국제중·특목고 |
| `/dashboard/real-estate/redevelopment/` | 재개발·재건축 — 대지지분·진행단계·단계별 프리미엄 |
| `/dashboard/supply/` | 착공과 금리 — 시도별 아파트 착공 평년 지수 · 기준금리 · 주담대 |

설계 문서는 `_dev/specs/` 에 있습니다. 재개발 쪽을 건드린다면
`_dev/specs/2026-07-31-redevelopment-design.md` 를 먼저 읽으세요 — 데이터 출처,
조인 키, 검증 방식, 그동안 밟은 지뢰가 정리돼 있습니다.

### 구조 규칙

- `scripts/<name>_api.py` — **순수 함수만**. I/O 없음. 응답 파싱·계산이 여기 있고 테스트가 붙습니다.
- `scripts/collect_<name>.py` — I/O·예산·상태 재개. `--max-calls` 로 하루 예산을 끊고 다음 실행에서 이어받습니다.
- `scripts/build_<name>.py` — 집계. `assets/realestate/*.json` 생성.
- `scripts/*.sh` — 크론 진입점.
- **표준 라이브러리만 씁니다.** pandas·requests 계열 없음 (`urllib`, `csv`, `gzip`, `xml.etree`).
- gzip 은 `mtime=0` 으로 고정해 내용이 같으면 바이트도 같게 만듭니다. 안 그러면 매일 새 blob 이 쌓입니다.

### 테스트

```shell
python3 -m unittest discover -s tests
```

pytest 는 없습니다. 표준 `unittest` 만 씁니다. 외부 응답은 `tests/fixtures/` 에
실제 응답을 고정해 두고 파싱 회귀를 잡습니다 — 정보몽땅·조례처럼 공식 API 가
아닌 곳은 화면이 바뀌면 조용히 깨지기 때문입니다.

### 인증키

저장소에 키를 두지 마세요. `gh-pages` 는 저장소 파일을 그대로 웹에 서빙합니다 (실제로 한 번 그랬습니다).

| 키 | 용도 | 위치 |
|---|---|---|
| `DATA_GO_KR_API_KEY` | 실거래·건축물대장 | 환경변수 또는 `~/.claude.json` |
| `VWORLD_API_KEY` | 브이월드 지적도·용도지역 | `.env` (gitignore 됨) |
| `NEIS_API_KEY`, `SCHOOLINFO_API_KEY`, `ODCLOUD_APT_INFO_API_KEY` | 학교 | `.env` |
| `LAW_OC` | 법제처 조례 | 없으면 `test` 로 동작 |
| `ECOS_API_KEY` | 한국은행 ECOS 금리 | `.env` (없으면 `collect_supply.py` 가 종료코드 1) |

활용신청은 **서비스마다 따로**입니다. 같은 키라도 실거래는 되고 건축물대장은
403 일 수 있습니다. 일일 한도도 서비스마다 따로 잡힙니다.

### 크론

```
30 4 * * *     daily.sh            실거래 (+ 대시보드·학교 집계)
10 6 * * *     redev_daily.sh      재개발·재건축
30 7 * * *     pokemon_daily.sh    포켓몬 카드 시세
30 18 * * 1-5  etf_daily.sh        ETF 테마 모멘텀
40 4 3 * *     schools_monthly.sh  학군
20 5 5,25 * *  supply_monthly.sh   착공·금리
```

전부 앞에 `flock -w 7200 ~/.cache/realestate.lock` 이 붙습니다. **반드시 유지하세요.**
전부 `git commit` / `pull --rebase` / `push` 를 해서 겹쳐 돌면 한쪽 커밋이 유실됩니다.

`redev_daily.sh` 는 날짜를 보고 스스로 갈라집니다 — 매일 연립다세대 실거래,
월요일 정비사업 추진경과, 매월 5일 건축물대장·브이월드·조례·정비구역.

로그: `~/.cache/realestate-{collect,redev,pokemon,etf,schools,supply}.log`

수동 실행은 `AUTO_COMMIT` 없이 (커밋하지 않음):

```shell
./scripts/redev_daily.sh                                # 오늘 날짜 기준
FORCE_WEEKLY=1 FORCE_MONTHLY=1 ./scripts/redev_daily.sh # 전부 강제
```

## 로컬 미리보기

`./run.sh` 는 **macOS 전용**입니다 (`/opt/homebrew` 경로). 리눅스에서는 Docker 를 쓰세요.
저장소 `Gemfile` 을 그대로 쓰면 sass-embedded 가 죽어서 `jekyll-sass-converter ~> 2.0` 을 고정해야 합니다.

```shell
docker run --rm -v "$PWD":/srv/jekyll -v /tmp/gemhome:/gemhome -w /srv/jekyll \
  -e BUNDLE_GEMFILE=/gemhome/Gemfile jekyll/jekyll:4.2.2 \
  sh -c "bundle install --quiet && bundle exec jekyll build --destination /out"
```

## 커밋·푸시 규칙

- 사용자가 **명시적으로 요청할 때만** 커밋/푸시. 사전 동의 없이 자동 커밋 금지.
  (크론 스크립트는 예외 — 그건 이미 승인된 자동화입니다.)
- 기본 브랜치인 `gh-pages` 에 바로 커밋·푸시해도 됨 (단일 저자 사이트).
- author/committer 신원: `Jaehyun So <dorumugs@gmail.com>`. 전역 `git config` 변경은 금지 — `GIT_AUTHOR_*` / `GIT_COMMITTER_*` 환경변수로 단발 지정.
- 커밋 메시지: 영문 한 줄 요약 + 빈 줄 + 한국어 본문 1~2줄. 마지막에 `Co-Authored-By:` 트레일러.
- **커밋 전에 키가 섞이지 않았는지 확인**: `git grep -I -l '<키 일부>'` 가 0건이어야 합니다.

## 건드리지 말 것

- `_layouts/`, `_includes/`, `_sass/`, `docs/`, `CHANGELOG.md`, `README.md` 의 테마 절 — minimal-mistakes 원본.
- `assets/` — 테마 원본입니다. **단, 아래는 이 저장소가 직접 만든 것이라 수정해도 됩니다.**
  - `assets/realestate/` (대시보드 소스와 집계 JSON)
  - `assets/images/real-estate-*/` (배너)
- `_config.yml` — 사이트 전역 설정. 변경 전에 반드시 의도/영향 확인.
- `.github/`, `Gemfile`, `package.json`, `Rakefile` — 빌드/배포 파이프라인.
- `.env` — 커밋 금지. `.gitignore` 25~26행이 막고 있습니다.
- `data/` 아래 수집물 — 사람이 손으로 고치지 마세요. 수집 스크립트가 다시 씁니다.

## 글 작업 흐름 요약

1. 사용자가 노트/주제를 던지면 → 메모리 확인
2. 원본 텍스트를 마스킹 규칙으로 한 번 훑고
3. `_posts/YYYY-MM-DD-제목.md` 로 초안 작성
4. 셀프체크
   ```shell
   grep -nE '\b[0-9]{15,20}\b|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}' <new-post>.md
   ```
5. 390px 오버플로 확인
6. 사용자 명시 요청 시에만 커밋·푸시
