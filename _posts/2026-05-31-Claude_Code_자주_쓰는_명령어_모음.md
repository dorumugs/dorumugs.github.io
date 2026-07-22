---
layout: single
title:  "(1/4) Claude Code 자주 쓰는 명령어 모음 — CLI 플래그부터 슬래시 명령어까지"
date: 2026-05-31 11:20:00 +0900
description: "매일 Claude Code 를 쓰면서 손에 익은 CLI 플래그와 세션 슬래시 명령어, 그리고 입력 prefix 단축을 예시 위주로 정리했어요."
categories: coding
tag: [ClaudeCode, CLI, AI코딩, 슬래시명령어, 워크플로우, 생산성, 개발도구, 입문]
author_profile: false
toc: true
header:
  image: /assets/images/2026-05-31-claude-code-commands/header.svg
  teaser: /assets/images/2026-05-31-claude-code-commands/header.svg
series: ai-coding-cli
series_order: 1
series_title: "🛠️ AI 코딩 CLI 명령어 모음 시리즈"
---



{% include series-ai-coding-cli.html current="1" %}

# Summary

저는 작업 종류별로 `claude --enable-auto-mode --name "infra"` 같은 식으로 Claude Code 세션을 따로따로 띄워서 굴려요. 익숙해지면 정말 편해지는데, 처음에는 "이 옵션들이 다 뭐 하는 거지?" 싶을 거예요. 이 글에서는 제가 매일 쓰는 CLI 플래그, 세션 안에서 쓰는 슬래시 명령어, 그리고 입력창의 단축 prefix(`!`, `#`, `@`)까지 한 번에 정리해볼게요.

> 💡 이 글에서 다루는 것
> - Claude Code 를 띄울 때 자주 쓰는 CLI 플래그 (`--name`, `--permission-mode`, `-c`, `-r`, `-p`, `-w` 등)
> - 세션 안에서 자주 두드리는 슬래시 명령어 (`/clear`, `/compact`, `/model`, `/agents`, `/mcp` 등)
> - 입력창의 단축 prefix — `!` 쉘 실행, `#` 메모리 추가, `@` 파일 참조
> - 제가 실제로 쓰는 조합 예시 (일상 모드, 리뷰 모드, 워크트리 격리 등)

이 글의 기준 버전은 `claude --version` 기준 **2.1.158** 이에요. 버전이 올라가면 옵션이 추가/이름 변경될 수 있어서 `claude --help` 로 한 번 확인하는 걸 추천드려요.


<br>

<br>



## 1. 세션 띄우기 — CLI 플래그

가장 많이 쓰는 옵션부터 정리할게요. 셸에서 `claude` 뒤에 붙이면 그 세션 동안만 적용돼요.


### 1-1. `--name` / `-n` — 세션에 이름 붙이기

세션을 여러 개 띄울 때 가장 먼저 손이 가는 옵션이에요. 이름을 박아두면 프롬프트 박스, `/resume` 피커, 터미널 타이틀에 다 같이 노출돼서 "어떤 작업하던 창이지?" 헷갈리지 않아요.

```shell
claude --name "infra"
claude -n "review"
```

저는 작업 종류별로 동시에 두세 개 띄우는 일이 많아서 거의 항상 박습니다.


### 1-2. `--permission-mode` — 권한 모드 고르기

Claude 가 파일 수정·셸 실행 같은 액션을 할 때 매번 물어볼지, 알아서 진행할지 정하는 옵션이에요. 모드는 다섯 가지가 있어요.

| 모드 | 동작 |
|------|------|
| `default` | 매번 권한 묻기 (기본값) |
| `acceptEdits` | 파일 편집은 자동 승인, 그 외는 물어봄 |
| `auto` | 자동으로 진행. Claude 가 알아서 판단해서 위험한 액션만 물어봄 |
| `plan` | 플랜 모드. 코드를 안 건드리고 계획만 세움 |
| `bypassPermissions` | 모든 권한 체크 무시 (위험) |

```shell
claude --permission-mode auto
claude --permission-mode plan
```

`--enable-auto-mode` 같은 단축 플래그도 같이 쓸 수 있어요 (제가 글 상단에 적은 예시처럼). 결과적으로는 `--permission-mode auto` 와 같은 효과예요.

> ⚠️ `bypassPermissions` 와 `--dangerously-skip-permissions` 는 진짜 위험할 때 — 격리된 샌드박스/인터넷 차단 환경에서만 권장합니다. 일반 작업에는 쓰지 마세요.


### 1-3. `-c, --continue` / `-r, --resume` — 이전 세션 이어가기

작업하다가 터미널을 닫아도 세션은 디스크에 남아 있어요. 다시 켜고 싶을 때 두 가지 옵션이 있어요.

```shell
claude -c                       # 현재 디렉토리의 가장 최근 세션 그대로 이어가기
claude -r                       # 세션 피커 띄워서 고르기
claude -r "blog"                # 이름/검색어로 필터링해서 피커 띄우기
```

`-c` 가 정말 자주 손이 가요. 잠깐 다른 터미널로 빠졌다 돌아올 때 한 글자로 복귀 가능.


### 1-4. `--model` — 모델 바꿔서 띄우기

세션 단위로 모델을 지정하는 옵션이에요. alias(`sonnet`, `opus`, `haiku`) 또는 풀네임 둘 다 됩니다.

```shell
claude --model opus
claude --model sonnet
claude --model claude-opus-4-7
```

저는 기본은 Opus 로 띄우고, 양이 많은 단순 변환 작업에만 Sonnet 으로 띄워요. 세션 도중에 바꾸고 싶으면 안에서 `/model` 슬래시 명령으로도 됩니다.


### 1-5. `-p, --print` — 비대화 모드 (파이프 친화적)

인터랙티브 세션 대신 결과만 한 번 뱉고 끝나는 모드. 셸 파이프나 스크립트에서 쓰기 좋아요.

```shell
claude -p "이 파일에서 TODO 가 몇 개 있나?" < src/main.py
echo "log line" | claude -p "이 로그가 의미하는 에러는?"
```

`--output-format json` 과 같이 쓰면 결과를 JSON 으로 받을 수 있어서 후처리 파이프라인에 끼워넣기도 좋습니다.

```shell
claude -p "한 줄 요약해줘" --output-format json < README.md
```


### 1-6. `-w, --worktree` — git worktree 로 격리해서 띄우기

작업 중인 브랜치를 건드리지 않고 별도 worktree 에서 실험할 때 한 줄로 만들어 줘요.

```shell
claude -w feature-x
claude -w experiment-mig --tmux
```

`--tmux` 까지 같이 박으면 tmux 세션을 자동으로 만들어 줘서 (iTerm2 면 네이티브 페인) 분리 작업이 정말 편해져요.


### 1-7. `--add-dir` — 작업 디렉토리 추가하기

기본 동작은 현재 디렉토리만 도구로 접근 가능한데, 인접한 디렉토리도 같이 열어두고 싶을 때 씁니다.

```shell
claude --add-dir ../shared-lib ../other-repo
```

저는 모노레포 한쪽에서 작업하면서 옆 폴더의 노트를 같이 참조해야 할 때 자주 씁니다.


### 1-8. `--allowedTools` / `--disallowedTools` — 도구 화이트/블랙리스트

세션 단위로 어떤 도구를 쓸지/막을지 명시적으로 지정해요. CI 나 스크립트에서 안전하게 돌릴 때 좋아요.

```shell
# 읽기만 허용
claude -p "이 코드 베이스 구조 한 줄 요약" \
  --allowedTools "Read,Grep,Glob"

# Bash 는 git 명령만 허용
claude --allowedTools "Bash(git *) Edit Read"

# 절대 쉘은 못 쓰게
claude --disallowedTools "Bash"
```


<br>

<br>



## 2. 세션 안에서 — 슬래시 명령어

세션 안에서 `/` 로 시작하는 명령어들이에요. 입력창에서 `/` 만 쳐도 자동완성 목록이 떠요. 손에 익는 순서대로 정리해볼게요.


### 2-1. `/clear` — 컨텍스트 비우기

지금까지의 대화를 통째로 비우고 새로 시작해요. 작업이 바뀌었을 때 컨텍스트 오염 막으려고 자주 누릅니다.

```text
/clear
```

저는 한 세션 안에서 "이제 다른 글 쓸게" 라고 주제가 바뀌면 무조건 한 번 비워요. 안 그러면 직전 글의 톤/소재가 새 글에 묻어 나와요.


### 2-2. `/compact` — 컨텍스트 압축

`/clear` 처럼 완전히 비우지는 않고, 지금까지의 대화를 요약본으로 압축해서 계속 이어가요. "조금 길었지만 여기까지의 흐름은 살리고 싶을 때" 쓰는 절충안.

```text
/compact
/compact 핵심 결정사항만 남기고 나머지는 정리해줘
```

뒤에 지시를 붙이면 어떤 관점으로 압축할지도 가이드 할 수 있어요.


### 2-3. `/model` — 모델 바꾸기

세션 도중에 모델 갈아끼울 때. 토큰 많이 쓰는 작업이면 Opus, 단순 반복이면 Sonnet 으로 자주 옮겨 다닙니다.

```text
/model sonnet
/model opus
```


### 2-4. `/init` — 프로젝트에 CLAUDE.md 만들기

새 저장소에서 처음 Claude Code 돌릴 때 한 번 치면, 코드베이스를 훑어보고 `CLAUDE.md` 초안을 만들어줘요. 이 파일은 이후 모든 세션의 컨텍스트로 자동 주입돼서 정말 강력해요.

```text
/init
```

저는 새 저장소 받을 때 거의 반사적으로 한 번 돌리고, 만들어진 `CLAUDE.md` 를 손으로 다듬어서 쓰는 패턴이에요.


### 2-5. `/cost` — 토큰/달러 비용 보기

이번 세션에서 토큰 얼마 썼고 달러로 얼마나 됐는지 보여줘요. Opus 로 길게 작업할 때 한 번씩 확인합니다.

```text
/cost
```


### 2-6. `/status` — 세션 상태 한눈에

지금 어떤 모델, 어떤 권한 모드, 어떤 디렉토리, 어떤 MCP 서버 붙어있는지 한 페이지로 보여줘요.

```text
/status
```


### 2-7. `/agents` — 커스텀 서브에이전트 관리

특정 작업 전용 서브에이전트를 만들거나 관리하는 명령. "코드 리뷰 전용", "테스트 작성 전용" 같은 식으로 페르소나를 따로 등록해두면 메인 컨텍스트 안 더럽히면서 위임이 가능해요.

```text
/agents
```


### 2-8. `/mcp` — MCP 서버 관리

MCP(Model Context Protocol) 서버를 붙이고 떼는 명령. Gmail, Google Drive, Notion, Linear 같은 외부 도구를 Claude 가 직접 쓰게 해주는 통로에요.

```text
/mcp
```


### 2-9. `/review` — 현재 diff 코드 리뷰

브랜치에 쌓인 변경분에 대해 한 번 훑어주는 리뷰 명령. PR 올리기 직전에 셀프 리뷰용으로 자주 씁니다.

```text
/review
```


### 2-10. `/help` — 단축키/명령어 도움말

까먹은 명령어/단축키를 빠르게 확인. `/` 자동완성도 좋지만 이쪽이 더 친절해요.

```text
/help
```


<br>

<br>



## 3. 입력창 단축 prefix — `!`, `#`, `@`

슬래시 명령어보다 더 가볍게 한 글자로 동작이 바뀌는 prefix 들이 있어요. 이게 진짜 손에 익으면 생산성이 다릅니다.


### 3-1. `!` — 쉘 명령어 즉시 실행

입력 첫 글자가 `!` 면 그 줄을 쉘로 실행해요. 결과가 대화 안에 그대로 들어와서 Claude 가 바로 다음 턴부터 활용할 수 있어요.

```text
! ls -la src/ | tail -10
! git status
! gcloud auth login
```

특히 인터랙티브 로그인(`gcloud auth login`, `aws sso login`)처럼 Claude 가 직접 못 돌리는 명령을 사용자가 직접 쳐야 할 때 이 prefix 가 답이에요.


### 3-2. `#` — 메모리에 추가

입력 첫 글자가 `#` 이면 그 내용이 메모리(또는 `CLAUDE.md`)에 저장돼요. 다음 세션부터도 계속 기억해주길 바라는 사실/규칙을 한 줄로 던질 수 있어요.

```text
# 이 프로젝트에서 PR 만들 때는 영어 제목 + 한국어 본문으로 작성
# 빌드는 ./run.sh 로만 띄울 것 (bundle exec jekyll serve 직접 호출 X)
```

저는 같은 안내를 두 번 반복하게 됐다 싶으면 그 순간 `#` 으로 박아넣어요.


### 3-3. `@` — 파일/디렉토리 참조

입력 도중에 `@` 를 치면 파일 자동완성이 떠요. 선택하면 그 파일이 컨텍스트로 첨부돼요.

```text
이 두 파일 비교해줘 @src/api/handlers.py @src/api/legacy_handlers.py
```

긴 경로를 손으로 안 쳐도 돼서 진짜 편해요.


<br>

<br>



## 4. 단축키 몇 가지

키보드만으로 자주 쓰는 단축키들도 같이 정리할게요.

| 단축키 | 동작 |
|--------|------|
| `Shift + Tab` | 권한 모드 순환 (default → acceptEdits → plan → ...) |
| `Esc` (한 번) | 현재 응답 중단 |
| `Esc Esc` (두 번) | 이전 사용자 메시지로 되돌아가서 다시 입력 |
| `Ctrl + R` | verbose 모드 토글 (도구 호출 인자 전체 보기) |
| `Ctrl + L` | 화면 클리어 (대화 컨텍스트는 유지) |

특히 `Esc Esc` 는 응답이 마음에 안 들 때 메시지 자체를 고쳐서 다시 보낼 수 있어서 정말 자주 씁니다.


<br>

<br>



## 5. 실제로 쓰는 조합 예시

플래그들을 어떻게 조합해서 쓰는지 제 워크플로우 몇 개를 옮겨볼게요.


### 5-1. 일상 작업 모드

```shell
claude --enable-auto-mode --name "daily"
```

`--name` 으로 어떤 창인지 명확히 하고, auto 모드로 띄워서 자잘한 파일 편집은 자동 진행. 매일 켜는 기본 조합이에요.


### 5-2. PR 올리기 직전 셀프 리뷰

```shell
claude --name "review" -p "/review"
```

대화형으로 안 가도 되니까 `-p` 로 한 방에 리뷰만 받고 끝. CI 에 넣으면 자동 PR 리뷰 봇처럼도 쓸 수 있어요.


### 5-3. 위험한 실험은 worktree 에 격리

```shell
claude -w mig-experiment --tmux --name "mig"
```

브랜치 일은 별도 worktree 로 빼고, tmux 페인까지 자동으로 분리. 메인 작업 환경 안 망가지면서 실험 가능.


### 5-4. 읽기만 시키는 안전 모드

```shell
claude -p "이 저장소 구조 한 단락으로 요약해줘" \
  --allowedTools "Read,Grep,Glob" \
  --model sonnet
```

`Bash` / `Edit` 를 빼고 읽기 도구만 허용. 모델도 Sonnet 으로 내려서 비용도 절약. 처음 보는 저장소 훑을 때 이 조합을 자주 씁니다.


<br>

<br>



## 마무리

여기까지 자주 쓰는 CLI 플래그, 슬래시 명령어, 입력 prefix, 단축키를 한 번에 훑었어요. 정리하면 이런 흐름이에요.

- **세션을 띄울 때**: `--name`, `--permission-mode`, `-c`, `-r`, `--model`, `-w`
- **세션 안에서**: `/clear`, `/compact`, `/model`, `/status`, `/cost`, `/agents`, `/mcp`, `/review`
- **입력창 prefix**: `!` (쉘), `#` (메모리), `@` (파일)
- **단축키**: `Shift+Tab`, `Esc Esc`, `Ctrl+R`

전부 외울 필요는 없고, **본인 워크플로우에 자주 등장하는 두세 개부터 손에 익히는 걸 추천드려요.** 저는 처음에 `--name` 하나만 박아 쓰다가, `/clear` 와 `#` 두 개를 추가로 익히면서 작업 속도가 눈에 띄게 빨라졌어요.

일단 오늘은 여기까지.....   
다음 글에서는 같은 결로 **OpenAI Codex CLI** 의 서브커맨드와 슬래시 명령어를 정리해볼게요.

---

**다음 글 →** [(2/4) Codex CLI 자주 쓰는 명령어 모음 — 서브커맨드부터 슬래시 명령어까지](/coding/Codex_CLI_자주_쓰는_명령어_모음/)
