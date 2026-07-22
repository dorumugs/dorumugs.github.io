---
layout: single
title:  "(7/7) Agents Toolkit 으로 프로코드 에이전트 만들기 — VS Code 에서 코드로 정의하기"
date: 2026-06-20 08:00:00 +0900
categories: coding
tag: [Copilot, AgentsToolkit, 프로코드, VSCode, DeclarativeAgent, OpenAPI, APIplugin, 매니페스트, 에이전트, 핸즈온]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-20-agents-toolkit/header.svg
  teaser: /assets/images/2026-06-20-agents-toolkit/header.svg
description: "Copilot Studio(노코드·로우코드)로 만들던 에이전트를, 이번엔 VS Code 의 Agents Toolkit 으로 코드·매니페스트에서 직접 정의하는 프로코드 편. 프로젝트 스캐폴드·declarativeAgent.json·OpenAPI API 플러그인·프로비전/배포까지, 언제 여기까지 내려와야 하는지와 함께 정리합니다."
series: ms-copilot
series_order: 7
---

{% include series-ms-copilot.html current="7" %}


## Summary

[규정 Q&A 봇](/coding/Copilot_Studio_사내규정_QA봇_만들기/)·[휴가 신청 액션](/coding/Copilot_Studio_에이전트_액션_휴가신청/)·[자율 메일 분류](/coding/Copilot_Studio_자율에이전트_메일분류/)까지, 지금까지는 전부 **Copilot Studio** 화면에서 만들었어요. 노코드~로우코드의 편한 길이었죠. 이번 마지막 글은 사다리의 맨 끝 칸 — **프로코드** 예요. VS Code 의 **Agents Toolkit** 으로 에이전트를 **코드와 매니페스트에서 직접** 정의합니다.

미리 말하면, **대부분의 경우 여기까지 내려올 필요가 없어요.** Copilot Studio 로 충분해요. 그런데 정밀한 제어가 필요하거나(커스텀 API, 세밀한 동작), 에이전트를 **git 으로 버전 관리하고 CI 에 태우고** 싶거나, 개발팀이 코드 리뷰 흐름으로 협업하려면 — 그때 프로코드가 답이에요. 이 글은 "그 단계가 어떻게 생겼는지" 를 보여주는 지도예요.

> 💡 **이 글에서 다루는 것**
> - 노코드 → 로우코드 → 프로코드, **언제 내려오나**
> - Agents Toolkit 이란 — VS Code 확장(구 Teams Toolkit)
> - 두 종류 — **declarative** vs **custom engine** 에이전트
> - 프로젝트 스캐폴드 — manifest · declarativeAgent.json · 지시문
> - **액션을 코드로** — OpenAPI 기반 API 플러그인
> - 프로비전 · 사이드로드 · 배포
> - 버전관리 · 환경 분리 · CI
> - 프로코드를 **쓰지 말아야** 할 때

> ⚠️ 2026년 상반기 기준이고 **개발자용** 이에요. Node.js·VS Code·Microsoft 365 계정 + **사이드로딩(업로드) 권한** 이 필요하고, 일부는 관리자가 열어줘야 해요. 도구 이름(Teams Toolkit → Agents Toolkit)과 CLI 명령은 버전에 따라 달라요. 본문의 도메인·엔드포인트는 예시값(마스킹)이에요.


<br>

<br>



## 1. 왜 코드까지 내려가나

[자동화 글](/coding/M365_Copilot_대화말고_자동화/)에서 "만드는 사다리" 를 노코드 → 로우코드 → 프로코드로 그렸죠. 이번 글은 그 맨 아래 칸이에요. 셋을 다시 한 표로 비교하면 차이가 분명해요.

| | 노코드 | 로우코드 | **프로코드** |
|---|---|---|---|
| **도구** | Agent Builder | Copilot Studio | **Agents Toolkit (VS Code)** |
| **만드는 법** | 자연어 | 화면 블록 | **코드·매니페스트** |
| **버전관리** | 약함 | 환경 단위 | **git 그대로** |
| **정밀 제어** | 낮음 | 중간 | **높음(커스텀 API)** |
| **누가** | 일반 사용자 | 시민 개발자 | **개발자** |

프로코드가 주는 건 결국 **"코드가 주는 것들"** 이에요 — git 이력, PR 리뷰, CI 파이프라인, 환경 분리(dev/prod), 그리고 화면 메뉴엔 없는 동작을 **API 로 직접** 붙이는 자유. 반대로 비용도 코드의 비용 그대로예요 — 러닝커브, 빌드·배포 셋업, 유지보수.

> ✅ 그래서 순서는 항상 **"노코드로 안 되면 로우코드, 그래도 안 되면 프로코드"** 예요. 프로코드는 *필요해서 내려가는* 거지, 멋있어서 시작하는 단계가 아니에요. 4~6편을 Copilot Studio 로 끝낸 이유이기도 해요.


<br>

<br>



## 2. Agents Toolkit 이란

**Agents Toolkit** 은 VS Code 확장이에요. 예전 이름이 **Teams Toolkit** 이었는데, Teams 앱뿐 아니라 Copilot 에이전트까지 코드로 만드는 도구로 넓어지면서 이름이 바뀌었어요. VS Code 에서 프로젝트를 스캐폴드하고, 로컬에서 디버그하고, M365 에 올리는 흐름을 한 번에 제공해요. 명령줄로는 **`atk`**(Agents Toolkit CLI)를 써요.

여기서 만들 수 있는 에이전트는 크게 두 종류예요. 이 갈림길을 먼저 잡아야 해요.

| 종류 | 무엇 | 언제 |
|---|---|---|
| **Declarative agent** | 365 Copilot 의 **오케스트레이터·모델 위** 에 지식·액션만 얹음 | 대부분 — 빠르고 보안 그대로 |
| **Custom engine agent** | **내 오케스트레이터·내 모델** 로 직접 구동 | 모델·로직을 통째로 제어해야 할 때 |

이 글은 **declarative agent** 를 다뤄요. 4편에서 만든 규정 봇과 같은 종류 — 365 Copilot 본체를 그대로 쓰되, 지식과 액션만 내가 코드로 끼워넣는 방식이에요. 만들기 쉽고, 권한·보안이 본체를 따라가서 안전해요.

> 💡 **custom engine** 은 "Copilot 본체 말고 내가 만든 LLM 파이프라인을 Teams/Copilot 표면에 얹고 싶다" 일 때예요. 자유도는 최고지만 모델 호스팅·오케스트레이션·보안을 전부 직접 져야 해요. 규정 봇 같은 사내 지식 봇은 declarative 로 충분해요.


<br>

<br>



## 3. 프로젝트 스캐폴드

VS Code 에서 Agents Toolkit 의 **Create a New Agent/App** 을 누르거나, CLI 로 시작해요.

```shell
atk new
# 안내에 따라: Declarative Agent 선택 → 이름 입력 → 폴더 생성
```

만들어진 declarative agent 프로젝트는 대략 이런 모양이에요.

```text
my-policy-agent/
├─ appPackage/
│  ├─ manifest.json            # Teams/M365 앱 매니페스트(앱 메타데이터)
│  ├─ declarativeAgent.json    # 에이전트 본체 — 지시문·지식·액션
│  ├─ instruction.txt          # 에이전트 지시문(분리 보관도 가능)
│  └─ color.png / outline.png  # 아이콘
├─ env/
│  ├─ .env.dev                 # 환경별 변수(dev)
│  └─ .env.prod
└─ atk.yml                     # 프로비전·배포 파이프라인 정의
```

핵심 파일은 셋이에요. **`manifest.json`**(앱 자체의 신원), **`declarativeAgent.json`**(에이전트가 무엇을 아는지·무엇을 하는지), 그리고 지시문이에요. Copilot Studio 에서 화면으로 채우던 걸 여기선 **JSON 으로 적는다** 고 보면 돼요.

> 💡 화면이 JSON 으로 바뀌었을 뿐, 개념은 4편과 똑같아요 — **지식(무엇을 보나) + 지시문(어떻게 행동하나) + 액션(무엇을 하나).** Copilot Studio 를 거쳐왔다면 이 파일들이 낯설지 않을 거예요.


<br>

<br>



## 4. 에이전트 정의 — declarativeAgent.json

에이전트의 본체예요. 규정 봇을 코드로 옮기면 이런 모양이에요.

```json
{
  "$schema": "https://aka.ms/json-schemas/copilot/declarative-agent/v1.0/schema.json",
  "version": "v1.0",
  "name": "사내 규정 도우미",
  "description": "임직원의 인사·복리후생·경비 규정 질문에 출처를 달아 답한다.",
  "instructions": "$[file('instruction.txt')]",
  "conversation_starters": [
    { "title": "육아휴직", "text": "육아휴직은 며칠까지 쓸 수 있어?" },
    { "title": "경조사 휴가", "text": "본인 결혼 시 경조 휴가는 며칠이야?" }
  ],
  "capabilities": [
    {
      "name": "OneDriveAndSharePoint",
      "items_by_url": [
        { "url": "https://contoso.sharepoint.com/sites/HR/사내규정" }
      ]
    }
  ]
}
```

읽어보면 4편에서 화면으로 하던 게 그대로 보여요.

- **`instructions`** — 지시문. 길어지니 `instruction.txt` 로 빼서 참조했어요. 내용은 4편의 "출처 강제 · 모르면 모른다 · 금지" 지시문 그대로예요.
- **`conversation_starters`** — 사용자에게 보여줄 예시 질문 버튼.
- **`capabilities`** — 지식 소스. 여기선 SharePoint 규정 사이트를 URL 로 박았어요. 이게 4편의 "지식 추가" 에 해당해요.

지시문 파일은 평범한 텍스트예요.

```text
역할: 회사 임직원의 규정 질문에 답하는 '사내 규정 도우미'.
근거: 반드시 연결된 규정 문서에만 근거한다. 외부 법령·일반 상식으로 추측 금지.
출처: 답변 끝에 [n] 문서명 — 조항 형식으로 표기.
모르면: 근거를 못 찾으면 지어내지 말고 담당 부서로 안내한다.
금지: 개인의 급여·평가·징계 등 개인정보는 답하지 않는다.
```

> ✅ 지시문을 **별도 파일 + git 관리** 로 두면, 규정 봇의 "행동" 변경 이력이 코드 히스토리에 남아요. "언제 누가 이 폴백 문구를 바꿨나" 가 PR 로 추적되는 거예요. 화면 편집엔 없던 프로코드의 진짜 이점이에요.


<br>

<br>



## 5. 액션을 코드로 — API 플러그인

프로코드의 진짜 힘은 **액션을 커스텀 API 로 붙일 때** 나와요. 5편에서 Power Automate 플로우로 휴가 신청을 처리했죠. 프로코드에선 그 대신(또는 그와 함께) **내 HTTP API 를 OpenAPI 스펙으로 기술해서** 에이전트에 연결해요. 이걸 **API 플러그인** 이라고 해요.

먼저 API 를 OpenAPI(Swagger)로 적어요. 휴가 잔여 조회 엔드포인트라면 이런 식이에요.

```yaml
openapi: 3.0.0
info:
  title: 휴가 API
  version: 1.0.0
servers:
  - url: https://api.contoso-hr.example.com
paths:
  /leave/balance:
    get:
      operationId: getLeaveBalance
      summary: 사번으로 잔여 휴가 일수를 조회한다
      parameters:
        - name: employeeId
          in: query
          required: true
          schema: { type: string }
      responses:
        "200":
          description: 잔여 휴가
          content:
            application/json:
              schema:
                type: object
                properties:
                  remainingDays: { type: integer }
```

그리고 이 API 를 에이전트가 쓸 수 있게 **플러그인 매니페스트**(ai-plugin)로 묶어요.

```json
{
  "schema_version": "v2.1",
  "name_for_human": "휴가 API",
  "description_for_human": "잔여 휴가 조회 및 신청 처리",
  "functions": [
    {
      "name": "getLeaveBalance",
      "description": "사번으로 잔여 휴가 일수를 조회한다."
    }
  ],
  "runtimes": [
    {
      "type": "OpenApi",
      "auth": { "type": "OAuthPluginVault" },
      "spec": { "url": "apiSpecificationFile/leave.yaml" },
      "run_for_functions": ["getLeaveBalance"]
    }
  ]
}
```

마지막으로 `declarativeAgent.json` 의 `actions` 에 이 플러그인을 연결해요.

```json
"actions": [
  { "id": "leaveApi", "file": "ai-plugin.json" }
]
```

이러면 사용자가 "내 잔여 휴가 며칠 남았어?" 라고 물을 때, 에이전트가 **`getLeaveBalance` 함수를 호출** 해서 실제 API 를 때리고 결과로 답해요. 화면 커넥터에 없는 **사내 시스템 API 도 직접** 붙일 수 있는 게 프로코드의 핵심이에요.

> 🚨 **`description` 이 곧 트리거** 라는 원칙은 여기서도 같아요(5편 참고). `operationId`·`description` 을 명확히 쓸수록 에이전트가 함수를 정확히 골라 호출해요. 그리고 인증(`auth`)은 코드에 키를 박지 말고 **OAuth/시크릿 보관소** 로 빼야 해요. git 에 토큰이 올라가면 사고예요.


<br>

<br>



## 6. 프로비전 · 사이드로드 · 배포

코드가 준비되면 실제로 M365 에 올려요. Agents Toolkit 의 흐름은 보통 이래요.

| 단계 | 무엇을 | 명령(예) |
|---|---|---|
| **프로비전** | 앱 등록·리소스 준비 | `atk provision --env dev` |
| **사이드로드** | 내 계정에 올려 테스트 | VS Code 에서 F5 (디버그 실행) |
| **검증** | Copilot 에서 호출해보기 | M365 Copilot 에서 에이전트 선택 |
| **배포** | 조직에 게시 | `atk publish --env prod` |

개발 중엔 **F5(사이드로드)** 로 내 계정에만 올려 빠르게 돌려봐요. 4~6편의 "테스트 패널" 에 해당하는 단계예요. 만족스러우면 `--env prod` 로 운영 환경에 올리고, 조직 배포는 보통 **관리자 승인** 을 거쳐요.

> ⚠️ **사이드로딩 권한이 없으면 여기서 막혀요.** 많은 조직이 보안상 사용자 임의 앱 업로드를 잠가둬요. 프로코드로 가기 전에 *"내 계정에 커스텀 앱을 올릴 수 있는지"* 부터 관리자에게 확인하세요. 이게 안 되면 코드를 다 짜도 못 띄워요.


<br>

<br>



## 7. 버전관리 · 환경 · 협업

프로코드로 내려온 보람은 여기서 나와요. 에이전트가 **그냥 코드** 라서, 개발팀의 평소 흐름을 그대로 써요.

- **git 으로 관리.** `declarativeAgent.json`·지시문·OpenAPI 스펙이 전부 텍스트라 PR·코드리뷰·diff 가 자연스럽게 돼요. "지시문 한 줄 바뀐 걸 리뷰" 가 가능해져요.
- **환경 분리.** `.env.dev` / `.env.prod` 로 dev 에서 충분히 검증하고 prod 에 올려요. 규정 봇을 사내 전체에 풀기 전에 dev 에서 안전하게 굴리는 거죠.
- **CI 파이프라인.** `atk provision`·`atk publish` 를 CI 에 넣으면, 머지되면 자동 배포까지 갈 수 있어요. 단, 자동 배포는 **운영 게이트(승인)** 와 같이 두세요.
- **시크릿 분리.** API 키·토큰은 코드가 아니라 환경 변수/시크릿 보관소에. git 에 올라가면 [마스킹 규칙](/coding/Copilot_Studio_사내규정_QA봇_만들기/) 이전에 사고예요.

> 💡 정리하면 프로코드의 보상은 **"에이전트를 소프트웨어처럼 다루는 것"** 이에요. 리뷰·테스트·롤백·이력이 전부 코드 도구로 들어와요. 이게 필요 없는 봇이라면, 솔직히 Copilot Studio 가 더 빠르고 싸요.


<br>

<br>



## 8. 프로코드를 쓰지 말아야 할 때

마지막으로 균형을 잡을게요. 프로코드가 답이 **아닌** 경우가 사실 더 많아요.

| 이럴 땐 | 권장 |
|---|---|
| 사내 문서 Q&A 봇 하나 | **Copilot Studio**(4편)로 충분 |
| 트리거·승인 워크플로우 | **Copilot Studio + Power Automate**(5·6편) |
| 비개발자가 유지보수 | **노코드/로우코드** — 코드는 부담 |
| 빠른 시제품 | 화면이 빠름 — 프로코드는 셋업이 김 |
| **커스텀 API·정밀 제어·CI** | **프로코드(이 글)** |

프로코드로 가는 합당한 이유는 결국 **"화면으로는 안 되는 것"** 이 있을 때예요 — 사내 전용 API 연결, 코드 리뷰가 필요한 거버넌스, CI 자동 배포, 세밀한 동작 제어. 그게 아니라면 4~6편의 노코드/로우코드가 더 빠르고 유지보수가 싸요.

> ✅ 한 줄 기준: **"이 에이전트를 코드처럼 관리할 이유가 있나?"** 있으면 프로코드, 없으면 Copilot Studio. 멋이 아니라 필요로 고르세요.


<br>

<br>



## 9. 한 장 요약 — 그리고 시리즈 마무리

| 항목 | 핵심 한 줄 |
|---|---|
| **언제** | 화면으로 안 되는 커스텀·CI·정밀 제어가 필요할 때만 |
| **도구** | Agents Toolkit(구 Teams Toolkit) + `atk` CLI |
| **종류** | 대부분 declarative — 365 본체 위에 지식·액션만 |
| **정의** | declarativeAgent.json = 지식 + 지시문 + 액션 |
| **액션** | OpenAPI 로 사내 API 직접 연결(API 플러그인) |
| **배포** | provision → 사이드로드(F5) → publish, 사이드로딩 권한 필수 |
| **보상** | git·PR·CI·환경 분리 — 에이전트를 소프트웨어처럼 |

이걸로 **Microsoft Copilot 활용·제작 7편** 을 마무리해요. 돌아보면 한 줄로 이어져요 — Copilot 이 뭔지 지도를 그리고([1편](/coding/MS_Copilot_할수있는_작업_총정리/)), 앱에서 쓰고([2편](/coding/M365_Copilot_앱별_활용_깊게/)), 자동화 개념을 잡고([3편](/coding/M365_Copilot_대화말고_자동화/)), Copilot Studio 로 읽는 봇([4편](/coding/Copilot_Studio_사내규정_QA봇_만들기/))·쓰는 봇([5편](/coding/Copilot_Studio_에이전트_액션_휴가신청/))·무인 봇([6편](/coding/Copilot_Studio_자율에이전트_메일분류/))을 만들고, 마지막으로 코드까지 내려왔어요.

핵심은 변하지 않아요. **노코드로 안 되면 로우코드, 그래도 안 되면 프로코드.** 그리고 어느 단계든 — **권한은 좁게, 위험한 액션엔 사람을, 모든 걸 기록.** 도구가 화면이든 코드든, 안전하게 만드는 원칙은 같아요.


<br>

<br>



일단 오늘은 여기까지.....   
이걸로 Copilot 시리즈를 마칩니다. 읽어주셔서 고맙습니다. 다음엔 또 다른 주제로 찾아올게요.

---

**← 이전 글:** [(6/7) 자율 에이전트 만들기 — 스스로 도는 메일 분류봇](/coding/Copilot_Studio_자율에이전트_메일분류/)
