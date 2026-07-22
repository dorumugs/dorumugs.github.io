---
layout: single
title:  "(3/3) AI IDE × Claude Code 하이브리드 셋업 — Antigravity / Cursor 각각에 끼워 쓰기"
date: 2026-06-02 07:00:00 +0900
categories: coding
tag: [antigravity, cursor, claude-code, hybrid-setup, mcp, agentic-ide, cursor-rules, skills, workflow]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-02-ai-ide-claude-code-hybrid/header.svg
  teaser: /assets/images/2026-06-02-ai-ide-claude-code-hybrid/header.svg
description: "Antigravity 와 Cursor 같은 AI IDE 에 Claude Code 를 같이 끼워 쓰는 하이브리드 셋업을 정리합니다. MCP, 룰/스킬 패키지화, 권한 분리, 실전 워크플로우까지. AI IDE 현업 활용 시리즈 3편(마무리)."
series: ai-ide
series_order: 3
---

{% include series-ai-ide.html current="3" %}

## Summary

[1편 — Antigravity 기능](/coding/Antigravity_현업에서_쓸_기능_정리/) 과 [2편 — Cursor 기능](/coding/Cursor_현업에서_쓸_기능_정리/) 에서 각각의 IDE 가 뭘 잘 하는지 정리했어요. 마지막 3편은 그 둘에 **Claude Code (CLI)** 를 같이 끼워 쓰는 **하이브리드 셋업** 이야기입니다. IDE 하나로 다 해결하기보다, **터미널 쪽 강한 도구(Claude Code) + IDE 쪽 강한 도구(Antigravity/Cursor)** 를 역할 나눠 쓰는 패턴이 실무에서 가장 자연스러워요.

> 💡 이 글에서 다루는 것  
> - 왜 IDE + CLI 를 같이 쓰는지 — 역할 분담  
> - 공통 준비: MCP 서버 / 토큰 / 자격증명 격리  
> - Antigravity + Claude Code 셋업 — Skill 패키지화 / Knowledge Items 활용  
> - Cursor + Claude Code 셋업 — `.cursor/rules` / Background Agent 와의 분담  
> - 실전 워크플로우 4가지 시나리오  
> - 권한 / 자격증명 분리 (보안 체크리스트)  
> - 자주 부딪히는 함정 / 트러블슈팅


<br>

<br>



## 1. 왜 IDE + CLI 를 같이 쓰는가

같은 "에이전트형 코딩 도구" 라도 IDE 쪽과 CLI 쪽이 잘 하는 영역이 분명히 갈려요.

| 축 | IDE (Antigravity / Cursor) | CLI (Claude Code) |
| --- | --- | --- |
| 강점 | UI 작업 / 브라우저 자동화 /<br>시각적 검증 / 멀티 에이전트 관제 | 터미널 친화 / 시스템·인프라 작업 /<br>스크립트화 / CI 통합 |
| 약점 | 헤드리스 환경 /<br>서버에서 직접 실행 어려움 | 시각 자료 (스크린샷, 다이어그램)<br>다루기 약함 |
| 컨텍스트 | GUI 의 패널, 룰 파일, 인덱스 | 터미널 세션 + 프로젝트 파일 |

대략 이런 분담이 잘 먹어요.

- **"눈으로 봐야 하는" 작업** (UI, 브라우저, 디자인 리뷰) → **IDE 쪽**
- **"명령으로 끝나는" 작업** (DB 마이그레이션, 인프라 설정, 리팩토링, 배포) → **Claude Code**
- **"여러 갈래를 동시에"** → IDE 의 Mission Control / Parallel Agents 로 분배 + 각 갈래 안에서 Claude Code 호출

> ✅ 한 줄 요약  
> **IDE = 관제실 + 시각 검증, Claude Code = 손 빠른 작업자.** 둘이 같은 MCP/룰/자격증명 풀을 공유하면 "한 도구처럼" 굴러갑니다.


<br>

<br>



## 2. 공통 준비 — MCP / 토큰 / 자격증명 격리

세 도구 모두 **MCP** 표준을 따르니까, MCP 서버 한 번 잘 깔아두면 IDE 와 CLI 가 같이 씁니다. 우선 공통 항목부터.

### 2.1. MCP 서버 셋업 (공통)

자주 쓰는 서버

| 서버 | 용도 |
| --- | --- |
| Postgres / MySQL | 운영/스테이지 DB 스키마, 쿼리 결과 |
| GitHub | 이슈/PR 읽기·생성, 리뷰 코멘트 |
| Sentry | 에러 트래커 — 빈도 높은 에러 분석 |
| Linear / Jira | 티켓 컨텍스트 |
| Slack | 결과 통보 / 채널 검색 |
| Filesystem | 특정 경로 노출 (보통 OFF 권장) |

설치는 보통 `npx` 또는 단일 바이너리. 예시 (Postgres MCP)

```shell
# 한 번만 깔고, 토큰만 환경변수로 분리
npm install -g @modelcontextprotocol/server-postgres
```

설정 위치

- **Claude Code**: `~/.claude.json` 또는 프로젝트의 `.mcp.json`
- **Cursor**: 설정 → MCP Servers
- **Antigravity**: 설정 → MCP / Tools

세 도구 모두 **같은 MCP 서버 정의를 공유**할 수 있으니, 한 번 정의해서 세 곳에 같이 등록하는 게 정석.

### 2.2. 토큰 / 자격증명 격리 (가장 중요)

🚨 운영 자격증명을 그대로 꽂으면 사고 납니다. 다음 원칙을 권장합니다.

| 환경 | 자격증명 |
| --- | --- |
| 로컬 개발 | 로컬 DB / 스테이지 토큰만 |
| 스테이지 | 읽기 전용 권한 위주 |
| 운영 | **MCP 에 직접 안 꽂음.** 꼭 필요하면 SELECT-only 계정 |

토큰은 환경변수로만 주입.

```text
GITHUB_TOKEN=<GITHUB_TOKEN>
POSTGRES_URL=postgres://reader:<PASSWORD>@stg-db:5432/app
SENTRY_TOKEN=<SENTRY_TOKEN>
```

세 도구 다 `${ENV_VAR}` 치환을 지원하니까 설정 파일에는 placeholder 만 두면 OK.


<br>

<br>



## 3. Antigravity + Claude Code 셋업

Antigravity 쪽 강점(브라우저 자동화 / Mission Control / Artifacts)을 그대로 두고, 시스템 쪽은 Claude Code 로 넘기는 구성이에요.

### 3.1. 역할 분담 가이드

| 작업 종류 | 어디서 |
| --- | --- |
| UI / 프론트엔드 페이지 생성 | Antigravity (Editor View + Gemini 3 Pro) |
| 브라우저 회귀 테스트 | Antigravity (Chrome 확장) |
| 멀티 레포 동시 리팩토링 | Antigravity (Mission Control) + 각 에이전트 안에서 Claude Code 호출 |
| DB 마이그레이션 / DMS / 인프라 | Claude Code |
| 배포 스크립트 / CI 설정 | Claude Code |
| 새 라이브러리 import / Dockerfile 작성 | Claude Code |

### 3.2. Antigravity Skills 로 룰 패키지화

Antigravity 의 **Skills** (progressive disclosure) 는 사내 컨벤션을 패키지로 만들기 좋아요. 예시 Skill 한 개.

```text
skills/
└── company-api-conventions/
    ├── SKILL.md      # description + 트리거 키워드 + 본문
    └── examples/
        ├── response.ts
        └── error-mapping.ts
```

`SKILL.md` 의 description 이 매칭되는 요청이 들어올 때만 컨텍스트에 로드돼서, **모든 룰을 시스템 프롬프트에 박는 낭비**를 피합니다.

> ✅ 같은 룰을 Claude Code 에서도 쓰는 법  
> Skill 본문은 그냥 마크다운이라, 같은 디렉토리를 Claude Code 의 `CLAUDE.md` / 프로젝트 룰로 심볼릭 링크 해두면 **한 소스 두 도구**가 됩니다. Skill 트리거(progressive disclosure)는 Antigravity 에서만 동작하지만, 본문 규칙 자체는 공유돼요.

### 3.3. Knowledge Items 누적

Antigravity 의 Knowledge Items 는 대화 끝날 때마다 자동으로 누적되는데, 그 결과를 주기적으로 **Claude Code 에서도 참조 가능한 형태**로 export 해두면 좋아요.

```text
docs/
├── architecture.md      # 아키텍처 결정 누적
├── api-conventions.md
└── known-issues.md      # KI 에서 추출한 함정/팁
```

이렇게 두면 Claude Code 가 프로젝트 시작할 때 자동으로 읽어 가서 두 도구 사이 지식 단절이 줄어요.


<br>

<br>



## 4. Cursor + Claude Code 셋업

Cursor 는 IDE 안에 이미 에이전트가 강하니까, **Claude Code 와의 분담**이 더 미묘해요. 핵심은 "Cursor 의 Background Agent 가 잘 하는 영역"과 "Claude Code 가 잘 하는 영역"을 구분하는 것.

### 4.1. 역할 분담 가이드

| 작업 | Cursor | Claude Code |
| --- | --- | --- |
| 코드 한 파일 인라인 편집 | ⭕ Cmd-K |  |
| 멀티파일 리팩토링 | ⭕ Composer/Agent |  |
| 정형화된 이슈 → PR | ⭕ Background Agent |  |
| PR 1차 리뷰 / 자동 수정 제안 | ⭕ BugBot |  |
| 인프라 / IaC / DB 마이그레이션 |  | ⭕ |
| 셸 위주 일회성 작업 |  | ⭕ |
| 빠른 스크립트 작성 |  | ⭕ (터미널에서 즉시) |
| CI 안에서 자동화 호출 |  | ⭕ (`claude` 명령) |

### 4.2. `.cursor/rules` 와 `CLAUDE.md` 공유

Cursor 의 `.cursor/rules/*.mdc` 와 Claude Code 의 `CLAUDE.md` 는 형식이 다르지만 **내용은 거의 같은 정보**예요. 둘을 따로 관리하면 금방 어긋납니다. 두 가지 접근.

**접근 A: `.cursor/rules` 가 단일 소스**

- `.cursor/rules/` 에 룰을 쪼개서 저장
- 프로젝트 루트의 `CLAUDE.md` 는 `.cursor/rules/` 디렉토리를 참조하는 한 줄 + 핵심만 추린 요약

**접근 B: `CLAUDE.md` 가 단일 소스**

- 큰 컨벤션은 `CLAUDE.md` 에
- `.cursor/rules/*.mdc` 는 globs / description 매칭이 필요한 경우만 따로 작성

큰 코드베이스라면 **접근 A** 가 룰을 잘게 쪼개기 좋아요. 작은 프로젝트면 **접근 B** 가 단순.

### 4.3. Background Agent vs Claude Code 의 분담

Background Agent 는 "이슈에서 PR 까지 자율 진행" 에 강한데, **결정 트리가 단순하지 않은 작업**이나 **시스템 명령이 많이 섞이는 작업**은 여전히 Claude Code 가 더 안정적이에요.

| 시그널 | 어디로 |
| --- | --- |
| 이슈 본문이 잘 정리됨 + 변경 범위 명확 | Background Agent |
| 변경 도중 사람이 끼어들 가능성 큼 | Cursor 로컬 (Composer) |
| `terraform apply`, DB 마이그 등 위험 동작 | Claude Code (사람이 같이 보고) |
| 빠르게 1회 돌릴 작업 | Claude Code 터미널 |


<br>

<br>



## 5. 실전 워크플로우 4가지

### 워크플로우 ①: 신규 기능 (백엔드 + UI)

```text
[1] Antigravity Planning Mode (Gemini 3 Pro)
    → 아키텍처 / API 스펙 / DB 모델 잡기 + Artifact 로 저장
[2] Claude Code (터미널)
    → 마이그레이션 SQL / API 핸들러 / 단위 테스트
[3] Antigravity Editor (Gemini 3 Pro)
    → 프론트엔드 페이지 생성 (Artifact 스크린샷으로 확인)
[4] Antigravity Chrome 확장
    → 회원가입 → 결제 흐름 E2E 자동 검증
```

### 워크플로우 ②: 멀티 레포 의존성 버전업

```text
[1] Cursor Background Agent 를 레포 N개에 동시 dispatch
    → 각 레포에서 PR draft 까지 생성
[2] 로컬 Claude Code 에서 각 PR 의 lint / test 일괄 검증
    → 실패한 PR 만 추려서 사람이 본격 검토
[3] Cursor BugBot 이 PR 단위로 잔여 이슈 자동 수정 제안
```

### 워크플로우 ③: 운영 사고 대응

```text
[1] Slack/Sentry 에 알람 발생
[2] Claude Code 터미널에서 MCP 로 Sentry 최근 에러 + Postgres 최근 쿼리 가져오기
    → 의심 코드 라인 추출
[3] Cursor Composer (Claude Sonnet 4.5) 로 hotfix 작성
    → 테스트 통과 확인
[4] Cursor Background Agent 로 PR 자동 생성 → 사람이 리뷰만
```

### 워크플로우 ④: 정기 점검 (cron)

```text
- 매일 새벽: Antigravity Scheduled Task 가
  "프론트엔드 회귀 시나리오 5개" 자동 실행 → 결과 Slack 보고
- 매 시간: Claude Code 가 운영 로그 요약 → 이상 패턴이면 GitHub 이슈 자동 생성
- 매 PR: Cursor BugBot 자동 1차 리뷰
```

> ✅ 핵심 패턴  
> **"사람 → IDE → CLI" 가 아니라 "사람 ↔ IDE ↔ CLI"** 의 양방향 흐름. IDE 가 만든 Artifact 를 CLI 가 입력으로 쓰고, CLI 가 만든 로그/요약을 IDE 가 다시 컨텍스트로 끌어옴.


<br>

<br>



## 6. 권한 / 자격증명 분리 — 보안 체크리스트

세 도구가 한 머신/한 계정에 모이는 만큼, 권한 분리가 엉성하면 사고 범위가 커집니다. 도입 전 체크리스트.

- 🚨 **운영 자격증명은 어디에도 안 꽂는다.** 꼭 필요하면 read-only 분리 계정.
- 🚨 **GitHub 토큰 권한 최소화.** 가능하면 fine-grained PAT 로 특정 레포 / 특정 권한만.
- 🚨 **MCP 서버는 프로젝트 단위 격리.** Postgres MCP 두 개를 동시에 띄워서 stg/local 만 노출.
- ⚠️ **Filesystem MCP 는 기본 OFF.** 켜면 홈 디렉토리 통째로 노출되는 사고 잦음.
- ⚠️ **자동 승인 화이트리스트** — Cursor / Claude Code 모두 `rm`, `git reset --hard`, `terraform apply` 같은 명령은 자동 승인 빼고 매번 묻게.
- ⚠️ **Privacy Mode / 학습 옵트아웃** — 세 도구 모두 사내 도입 시 켜는 게 기본.
- ✅ **감사 로그** — 팀 플랜이라면 누가 언제 어떤 에이전트를 돌렸는지 로그 남기는 옵션을 켤 것.


<br>

<br>



## 7. 자주 부딪히는 함정

도입하면서 한 번씩 밟게 되는 지점들.

- ⚠️ **세 도구가 같은 파일을 동시에 만짐** — Cursor Composer 가 돌고 있는데 Claude Code 터미널에서 같은 파일을 또 바꾸면 충돌. 작업 시작 전 **어느 도구가 메인인지** 정하고 들어가기.
- ⚠️ **MCP 서버 중복 정의** — 같은 Postgres 서버를 IDE 와 CLI 양쪽에 다른 이름으로 등록하면 답이 다르게 나옴. **하나의 정의 파일을 참조**하거나 이름을 통일.
- ⚠️ **룰 파일 충돌** — `.cursor/rules` 와 `CLAUDE.md` 에 서로 다른 컨벤션을 쓰면 PR 스타일이 도구별로 갈림. 5장의 단일 소스 원칙 유지.
- ⚠️ **인덱스 무거움** — 큰 모노레포는 Cursor 인덱스 + Antigravity 인덱스 둘 다 돌면 머신이 느려짐. `.cursorignore` / Antigravity 제외 패턴 잡기.
- ⚠️ **요금 폭주** — 멀티 에이전트(8 병렬 / Mission Control)는 한 번에 토큰을 많이 씁니다. **모델별 일/주 한도**를 팀 단위로 잡아두는 게 안전해요.


<br>

<br>



## 마무리

3편을 한 줄로 요약하면 **"IDE 는 관제실, Claude Code 는 손, MCP 는 공용 도구 박스"** 정도예요. 세 도구를 같은 룰/자격증명/MCP 위에 올려두면 **어디서 호출해도 일관된 결과**가 나오고, 사람이 하는 일은 "어디서 시작할지 정하는 것" 만 남습니다.

처음부터 다 깔지 말고, **현재 가장 아픈 워크플로우 하나**(예: UI 회귀 테스트, 멀티 레포 의존성 버전업, 운영 사고 대응)부터 잡아서 IDE+CLI 조합으로 옮겨보면 도입 후 ROI 가 빨리 보입니다.

일단 오늘은 여기까지.....   
시리즈는 여기서 마무리. 다음 글에서는 같은 결로 **사내 도입 시 IT/보안 검토 포인트**(SSO, 감사 로그, 데이터 이탈 경계)를 별도 글로 정리해볼게요. 

---

**← 이전 글** [(2/3) Cursor, 현업에서 진짜 쓸 기능만 골라 정리](/coding/Cursor_현업에서_쓸_기능_정리/)
