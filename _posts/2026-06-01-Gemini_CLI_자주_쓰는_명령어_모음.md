---
layout: single
title:  "(3/3) Gemini CLI 자주 쓰는 명령어 모음 — 헤드리스 모드와 슬래시 명령어까지"
date: 2026-06-01 18:30:00 +0900
description: "Google Gemini CLI 의 자주 쓰는 플래그(--prompt, --approval-mode, --worktree, --resume)와 서브커맨드, 슬래시 명령어를 예시 위주로 정리했어요."
categories: coding
tag: [GeminiCLI, Google, Gemini, CLI, AI코딩, 슬래시명령어, 워크플로우, 생산성, 개발도구]
author_profile: false
toc: true
---



{% include series-ai-coding-cli.html current="3" %}

# Summary

[1편 Claude Code](/coding/Claude_Code_자주_쓰는_명령어_모음/), [2편 Codex CLI](/coding/Codex_CLI_자주_쓰는_명령어_모음/) 에 이어 시리즈 마지막 편은 **Google Gemini CLI** 예요. 셋 다 결은 비슷한데, Gemini 는 `--approval-mode` 와 `--prompt` 중심으로 헤드리스(비대화) 자동화 친화적인 색깔이 강해요. 이 글에선 자주 쓰는 CLI 플래그, 서브커맨드, 세션 슬래시 명령어, 그리고 헤드리스 모드 활용 예시까지 한 번에 훑어볼게요.

> 💡 이 글에서 다루는 것
> - Gemini CLI 플래그 (`--model`, `-p`, `-i`, `--approval-mode`, `--sandbox`, `--worktree`, `--resume` 등)
> - 권한 모드(`default` / `auto_edit` / `yolo` / `plan`)
> - 서브커맨드 (`gemini mcp`, `gemini extensions`, `gemini skills`, `gemini update`)
> - 슬래시 명령어 (`/help`, `/quit`, `/resume`, `/chat`, `/memory reload`, `/mcp reload`, `/settings`, `/bug`)
> - 입력 prefix `@` 와 헤드리스 자동화 예시

> ⚠️ Gemini CLI 도 업데이트 주기가 짧고 공식 docs 가 모듈별로 분산돼 있어요. 본문의 옵션이 안 보이면 `gemini --help` 와 `~/.gemini/` 설정 디렉토리를 먼저 확인하는 걸 추천드려요.


<br>

<br>



## 1. CLI 플래그 — 자주 쓰는 것부터


### 1-1. `--model` / `-m` — 모델 지정

세션 단위로 모델을 지정해요. alias 와 풀네임 둘 다 됩니다.

```shell
gemini -m gemini-2.5-flash
gemini --model gemini-2.5-pro
```

용량 큰 추론은 Pro, 단순 대량 처리는 Flash 로 갈아끼우는 흐름이 일반적이에요.


### 1-2. `--prompt` / `-p` — 헤드리스 (비대화) 실행

Claude Code 의 `-p`, Codex 의 `codex exec` 와 같은 결. 결과만 한 번 뱉고 끝나서 스크립트·파이프에서 쓰기 좋아요.

```shell
gemini -p "이 디렉토리 구조 한 단락으로 요약해줘"
cat error.log | gemini -p "이 로그가 의미하는 에러는?"
```

stdin 입력이 있으면 거기에 프롬프트가 자동으로 덧붙어요. 파이프 친화적인 구조에요.


### 1-3. `--prompt-interactive` / `-i` — 한 줄 던지고 대화로 이어가기

`-p` 와 비슷한데, 결과 뱉고 종료하지 않고 인터랙티브 세션으로 들어가요. "첫 메시지만 미리 박고 시작" 모드.

```shell
gemini -i "이 저장소 구조부터 훑어줘"
```


### 1-4. `--approval-mode` — 권한 모드 (핵심)

Gemini 의 권한 모델은 모드 네 가지로 표현돼요.

| 모드 | 동작 |
|------|------|
| `default` | 모든 액션 묻기 (기본값) |
| `auto_edit` | 파일 편집은 자동 승인, 쉘 명령은 묻기 |
| `yolo` | 모든 액션 자동 승인 |
| `plan` | 코드 안 건드리고 계획만 세움 |

```shell
gemini --approval-mode auto_edit
gemini --approval-mode plan
gemini --approval-mode yolo   # 위험
```

> 🚨 `yolo` 는 격리된 환경에서만 권장합니다. 호스트에 자유롭게 액션이 들어가서 한 번의 실수가 크게 번질 수 있어요. 일회용 컨테이너/VM 안에서만 쓰세요.

> 💡 옛 버전의 `--yolo` / `-y` 플래그는 deprecate 됐어요. 같은 동작은 `--approval-mode=yolo` 로 표현합니다.


### 1-5. `--sandbox` / `-s` — 샌드박스 실행

도구 호출을 격리된 샌드박스(컨테이너 등) 안에서 돌리는 모드. `yolo` 와 같이 박으면 안전성 + 자동화 균형이 잡혀요.

```shell
gemini -s
gemini -s --approval-mode yolo "이 마이그레이션 한 번에 끝내줘"
```


### 1-6. `--include-directories` — 추가 디렉토리 노출

작업 디렉토리 외에 다른 폴더도 컨텍스트에 같이 노출해요. 쉼표 또는 반복 지정 가능.

```shell
gemini --include-directories ../lib,../docs
```

모노레포에서 인접 패키지 같이 만질 때 자주 씁니다.


### 1-7. `--worktree` / `-w` — git worktree 격리

브랜치를 안 건드리고 별도 worktree 에서 실험할 때.

```shell
gemini -w                    # 자동 이름 (worktree-a1b2c3d4 같은 식)
gemini --worktree feature-x  # 이름 지정 시 디렉토리·브랜치 이름 동일
```

Claude Code 의 `-w` 와 동일한 결.


### 1-8. `--resume` / `-r` — 이전 세션 이어가기

이전 세션을 다시 열어요. 여러 방식 지원.

```shell
gemini --resume                                           # 가장 최근 세션
gemini --resume 1                                         # 인덱스로 지정
gemini --resume a1b2c3d4-e5f6-7890-abcd-ef1234567890      # UUID 로 지정
```

세션 목록·삭제도 같은 결로:

```shell
gemini --list-sessions
gemini --delete-session 2
```


### 1-9. `--output-format` / `-o` — 출력 포맷

`-p` 와 같이 쓸 때 결과 포맷을 바꿔요. 후처리 파이프라인에 끼워넣기 편함.

```shell
gemini -p "이 PR diff 요약" -o json
gemini -p "긴 작업 시작" -o stream-json
```

`stream-json` 은 결과를 토큰 단위로 스트리밍해서 실시간 진행 상황을 잡아낼 수 있어요.


### 1-10. `--debug` / `-d` — 디버그 모드

내부 호출·도구 실행 로그를 자세히 찍어요. 동작이 이상할 때 한 번씩 켭니다.

```shell
gemini -d
```


### 1-11. `--extensions` / `-e` — 확장만 골라 활성

설치된 확장 중 일부만 켜고 싶을 때.

```shell
gemini -e github,linear
```


<br>

<br>



## 2. 서브커맨드

`gemini` 뒤에 붙는 모드들. 자주 쓰는 것만 정리할게요.

```shell
gemini mcp           # MCP 서버 설정/관리
gemini extensions    # 확장 관리
gemini skills        # 스킬 관리 (등록/조회/리로드)
gemini update        # 최신 버전으로 업데이트
```

MCP·extension·skill 모두 외부 도구·기능을 Gemini 에 붙이는 통로에요. 새 도구 붙일 때 가장 먼저 손이 가는 명령들.


<br>

<br>



## 3. 세션 안 슬래시 명령어

세션에서 `/` 로 시작하는 명령어들. 다른 두 CLI 보다 더 모듈화돼 있어서 "리로드" 류 명령이 카테고리별로 따로 있어요.


### 3-1. `/help` / `/quit`

기본 중의 기본. 도움말 보기, 세션 종료.

```text
/help
/quit
```


### 3-2. `/memory reload` — 컨텍스트 파일 리로드

`GEMINI.md` 같은 컨텍스트 파일을 세션 도중 다시 읽어와요. 파일 수정하고 바로 반영하고 싶을 때.

```text
/memory reload
```


### 3-3. `/mcp reload` — MCP 서버 재시작

MCP 서버 설정을 바꿨거나 죽었을 때 한 번 리로드.

```text
/mcp reload
```


### 3-4. `/extensions reload` / `/skills reload` / `/agents reload` / `/commands reload`

같은 결의 리로드 명령들. 확장, 스킬, 에이전트, 커스텀 슬래시 명령을 각각 따로 다시 불러옵니다.

```text
/extensions reload
/skills reload
/agents reload
/commands reload
```

저는 새 확장이나 커스텀 명령 만들면서 테스트할 때 이 네 개를 자주 두드립니다.


### 3-5. `/resume` / `/chat` — 세션 브라우저와 체크포인트

`/resume` 은 세션 브라우저 UI 를 띄워서 이전 세션을 고르거나, 현재 세션에 이름 붙은 체크포인트를 만들어요.

```text
/resume                          # 세션 브라우저
/resume save decision-point      # 현재 시점에 체크포인트 저장
/resume list                     # 체크포인트 목록
/resume resume decision-point    # 특정 체크포인트로 복귀
```

`/chat` 은 `/resume` 의 호환 alias 에요. 작업 중간에 "여기서 가지치기 한 번 해볼까" 싶을 때 `save` 로 체크포인트 박아두는 흐름이 정말 편합니다.


### 3-6. `/settings` — 세션 설정

세션 정책(권한·도구·로깅 등) 을 인터랙티브하게 바꿔요.

```text
/settings
```


### 3-7. `/bug` — 이슈 리포트

문제 만났을 때 그 자리에서 이슈 리포트 양식을 띄워줘요.

```text
/bug
```


<br>

<br>



## 4. 입력창 prefix — `@`

Gemini 의 입력 prefix 는 `@` 이 가장 두드러져요. 단순 파일 참조가 아니라 **확장 기반 mention** 으로 동작해서, 확장(`@github`, `@linear` 등) 을 통한 외부 자원 참조가 자연스러워요.

```text
@github 내 열린 PR 목록 보여줘
@src/api/handlers.py 이 파일 리팩토링 해줘
```

> 💡 Claude Code 의 `!` (쉘) / `#` (메모리) 같은 prefix 는 Gemini CLI 공식 레퍼런스에서 명시적으로 잡혀있진 않아요. 같은 결의 기능은 권한 모드(`auto_edit`, `yolo`) 와 컨텍스트 파일(`GEMINI.md`) + `/memory reload` 조합으로 풀어요.


<br>

<br>



## 5. 헤드리스 자동화 예시 (Gemini 만의 색깔)

Gemini CLI 가 다른 두 CLI 대비 두드러지는 건 **헤드리스 모드** 의 깔끔함이에요. `-p` + `--output-format json` 조합으로 스크립트·CI 에 끼워넣기 쉬워요.


### 5-1. JSON 출력으로 후처리

```shell
gemini -p "이 저장소 라이선스 한 줄 요약" -o json > out.json
jq '.text' out.json
```


### 5-2. 스트리밍 결과 받기

```shell
gemini -p "긴 리팩토링 작업 시작" -o stream-json | tee progress.jsonl
```

`stream-json` 으로 받으면 진행 상황을 라인 단위로 받을 수 있어서 CI 로그에 그대로 흘릴 수 있어요.


### 5-3. 안전 모드로 첫 훑기

```shell
gemini -i "이 저장소 구조 한 단락으로 요약" --approval-mode plan
```

`plan` 모드는 코드를 절대 안 건드리고 계획만 세워줘서, 처음 보는 저장소 훑을 때 부담 없이 시작 가능.


### 5-4. 격리된 컨테이너에서 풀자동

```shell
gemini -s --approval-mode yolo "이 마이그레이션 끝까지 진행해줘"
```

`-s` 로 샌드박스 켜고 `yolo` 로 묻지 않는 자동 실행. 호스트는 안전, 작업은 끝까지.


### 5-5. 모노레포 멀티 디렉토리

```shell
gemini --include-directories ../shared,../types -w mono-experiment
```

인접 디렉토리 노출 + 새 worktree 까지 한 줄로.


<br>

<br>



## 6. 세 CLI 비교 한 줄 정리

시리즈 마지막인 만큼, 세 CLI 의 비슷한 결과 다른 결을 한 표로 정리할게요.

| 기능 | Claude Code | Codex | Gemini |
|------|-------------|-------|--------|
| 비대화 실행 | `-p` | `codex exec` | `-p` |
| 세션 이어가기 | `-c` / `-r` | `codex resume --last` | `-r` |
| 권한 모드 플래그 | `--permission-mode` | (세션 내 `/permissions`) | `--approval-mode` |
| 자동 편집 모드 | `acceptEdits` | `Auto` (기본) | `auto_edit` |
| 풀자동 모드 | `bypassPermissions` | `Full Access` | `yolo` |
| 플랜 모드 | `plan` | — | `plan` |
| worktree 격리 | `-w` | — (대신 `--cd`) | `-w` |
| 디렉토리 추가 | `--add-dir` | `--add-dir` | `--include-directories` |
| 컨텍스트 파일 | `CLAUDE.md` | `AGENTS.md` | `GEMINI.md` |
| 출력 포맷 | `--output-format` | `--json` | `--output-format` |

큰 그림은 정말 비슷한데, 모드 이름과 컨텍스트 파일명이 각자 다르다는 게 가장 헷갈리는 지점이에요. 셋 다 쓰다 보면 머릿속에 자연스럽게 매핑이 잡혀요.


<br>

<br>



## 마무리

여기까지 Gemini CLI 의 플래그·서브커맨드·슬래시 명령어·prefix 를 훑고, 시리즈 마무리로 세 CLI 의 비교 표까지 같이 정리해봤어요.

- **자주 쓰는 플래그**: `-m`, `-p`, `-i`, `--approval-mode`, `-s`, `-w`, `-r`, `--include-directories`, `-o`
- **권한 모드**: `default` / `auto_edit` / `yolo` / `plan` — `--approval-mode` 로 지정
- **서브커맨드**: `gemini mcp`, `extensions`, `skills`, `update`
- **슬래시 명령**: `/help`, `/quit`, `/resume`, `/chat`, `/memory reload`, `/mcp reload`, `/extensions reload`, `/skills reload`, `/agents reload`, `/commands reload`, `/settings`, `/bug`
- **입력 prefix**: `@` (확장 기반 mention)
- **시그니처**: `-p` + `-o stream-json` 조합의 깔끔한 헤드리스 모드

3편짜리 시리즈 끝났어요. 셋 다 익혀두면 작업 성격에 맞는 도구를 골라서 띄울 수 있고, 모드/플래그가 헷갈릴 때마다 이 표만 한 번 보면 됩니다.

일단 오늘은 여기까지.....   
다음 글에서는 세 CLI 의 컨텍스트 파일(`CLAUDE.md` / `AGENTS.md` / `GEMINI.md`) 을 어떻게 묶어서 같은 저장소에서 셋 다 잘 굴리게 만드는지 정리해볼게요.

---

**← 이전 글:** [(2/3) Codex CLI 자주 쓰는 명령어 모음 — 서브커맨드부터 슬래시 명령어까지](/coding/Codex_CLI_자주_쓰는_명령어_모음/) 
