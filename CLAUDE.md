# CLAUDE.md

이 저장소(`dorumugs.github.io`)는 **Jekyll + minimal-mistakes 테마** 기반의 개인 기술 블로그 **KayserDocs** 입니다. Claude 가 작업할 때 따라야 할 프로젝트 규칙을 모아둡니다.

## 저장소 기본 정보

- 사이트: `https://dorumugs.github.io`
- 테마: `minimal-mistakes-jekyll` (저장소 자체가 테마 포크. `_layouts/`, `_includes/`, `_sass/` 등이 그대로 들어있음)
- 기본 브랜치: **`gh-pages`** (GitHub Pages 가 이 브랜치를 그대로 서빙. `main` 없음)
- locale: `ko-KR`, permalink: `/:categories/:title/`

## 글 작성 규칙 (가장 중요)

블로그 글 작업의 톤·구조·마스킹 규칙은 **메모리 파일에 정리되어 있어요. 새 글을 쓰거나 노트를 블로그 글로 변환할 때 반드시 먼저 읽으세요.**

- `~/.claude/projects/-Users-dorumugs-PycharmProjects-dorumugs-github-io/memory/blog-writing-style.md` — 톤/구조/마크다운 컨벤션
- `~/.claude/projects/-Users-dorumugs-PycharmProjects-dorumugs-github-io/memory/sensitive-data-masking.md` — 민감정보 마스킹 규칙

핵심만 다시 짚으면

- 파일 위치: `_posts/YYYY-MM-DD-제목.md` (제목에 한글/언더스코어 OK)
- front matter: `layout: single` / `categories: coding` / `tag: [...]` / `author_profile: false` / `toc: true`
- 같은 날짜에 여러 글을 올려도 OK. 단, 제목/파일명은 겹치면 안 됨.
- 글 안에서 다른 글로 링크할 때는 permalink 규칙(`/coding/<slug>/`) 을 따른다.
- **민감정보(사설/공인 IP, 이메일 로컬파트, 토큰류)는 사용자 확인 없이 선제적으로 마스킹**해서 초안을 만든다.

## 로컬 미리보기

```shell
./run.sh
# 내부적으로 bundle exec jekyll serve 실행. http://localhost:4000
```

처음이라면 의존성 설치 먼저.

```shell
bundle install
```

## 커밋·푸시 규칙

- 사용자가 **명시적으로 요청할 때만** 커밋/푸시. 사전 동의 없이 자동 커밋 금지.
- 기본 브랜치인 `gh-pages` 에 바로 커밋·푸시해도 됨 (단일 저자 사이트).
- author/committer 신원: `Jaehyun So <dorumugs@gmail.com>` (과거 커밋과 동일). 전역 `git config` 변경은 금지 — 필요 시 `GIT_AUTHOR_*` / `GIT_COMMITTER_*` 환경변수로 단발 지정.
- 커밋 메시지: 영문 한 줄 요약 + 빈 줄 + 한국어 본문 1~2줄. 마지막에 `Co-Authored-By:` 트레일러 포함.

## 건드리지 말 것

- `_layouts/`, `_includes/`, `_sass/`, `assets/`, `docs/`, `CHANGELOG.md` — minimal-mistakes 테마 원본. 사용자가 명시적으로 요청하지 않으면 수정 금지.
- `_config.yml` — 사이트 전역 설정. 변경 전에 반드시 의도/영향 확인.
- `.github/`, `Gemfile`, `package.json`, `Rakefile` — 빌드/배포 파이프라인. 함부로 건드리지 않음.

## 글 작업 흐름 요약

1. 사용자가 노트/주제를 던지면 → 메모리(`blog-writing-style.md`, `sensitive-data-masking.md`) 확인
2. 원본 텍스트를 마스킹 규칙으로 한 번 훑고
3. `_posts/YYYY-MM-DD-제목.md` 로 초안 작성 (블로그 톤 그대로 적용)
4. `grep -nE '\b[0-9]{15,20}\b|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}' <new-post>.md` 으로 셀프체크
5. 사용자 명시 요청 시에만 커밋·푸시
