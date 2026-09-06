---
layout: single
title:  "(2/5) n8n MCP 서버 만들기 — Server Trigger, 도구 연결, 그리고 $fromAI()"
date: 2026-09-06 08:00:00 +0900
categories: coding
tag: [n8n, MCP, MCPServerTrigger, fromAI, 서브워크플로, AI에이전트, 자동화, 도구, ClaudeCode, 워크플로]
author_profile: false
toc: true
header:
  image: /assets/images/2026-09-06-n8n-mcp-2-build/header.svg
  teaser: /assets/images/2026-09-06-n8n-mcp-2-build/header.svg
description: "MCP Server Trigger 노드 하나로 n8n 워크플로를 MCP 서버로 바꿉니다. Path·인증·두 개의 URL 이 각각 언제 살아나는지, 도구를 붙이는 세 가지 방법이 어떻게 다른지, 그리고 $fromAI() 가 도구 입력 스키마를 만드는 방식과 그 함정까지 정확히 짚습니다."
series: n8n-mcp-server
series_order: 2
series_title: "🔌 n8n 으로 MCP 서버 만들기"
---

{% include series-n8n-mcp-server.html current="2" %}


## Summary

[1편](/coding/n8n_MCP서버_전체그림_프로토콜과_세가지자리/)에서 지형도를 그렸으니 이제 짓습니다. 이번 편은 **MCP Server Trigger 노드 하나**를 정확하게 이해하는 게 목표예요.

노드를 놓는 데는 30초 걸립니다. 그런데 그 뒤에 반드시 걸리는 게 셋 있어요.

> **① 도구가 클라이언트에 안 뜬다** — 대개 URL 을 잘못 골랐거나 워크플로를 게시하지 않은 것.
> **② 도구는 뜨는데 인자가 안 채워진다** — `$fromAI()` 를 안 썼거나 키 규칙을 어긴 것.
> **③ 테스트는 되는데 프로덕션에서 실패한다** — 서브워크플로가 게시되지 않은 것.

셋 다 원인이 정확히 정해져 있고, n8n 공식 문서에 다 적혀 있습니다. 순서대로 훑을게요.

> 💡 **이 글에서 다루는 것**
> - MCP Server Trigger 의 파라미터 — Path, 인증, 그리고 **두 개의 URL**
> - 테스트 URL 과 프로덕션 URL 이 **각각 언제 살아나는지**
> - 이 트리거가 "도구 노드만 실행한다" 는 말의 실제 의미
> - 도구를 붙이는 **세 가지 방법**과 선택 기준
> - `$fromAI()` 정확한 시그니처와 **키 규칙**, 그리고 타입
> - 서브워크플로를 도구로 쓸 때의 입력 스키마와 **게시 함정**
> - ⚠️ 커뮤니티가 보고한 "선택 인자가 필수처럼 동작한다" 문제
> - 여기까지의 체크리스트


<br>

<br>


## 1. 만들 것 — 워크플로 두 개

1편에서 정한 예제, "사내 부동산 조회 MCP" 를 짓습니다. 구조는 이래요.

| 워크플로 | 역할 | 안에 든 것 |
|---|---|---|
| **서버 워크플로** | MCP 서버 그 자체 | MCP Server Trigger + 도구 노드들 |
| **도구 워크플로** | 도구 하나의 실제 로직 | Execute Sub-workflow Trigger + 처리 노드 |

왜 나누냐면 — 서버 워크플로는 **얇게** 두는 게 좋기 때문입니다. 도구 로직이 서버 워크플로 안에 다 들어가면 캔버스가 금방 읽을 수 없게 돼요. 도구 하나당 워크플로 하나로 두면 각각 따로 테스트하고 따로 고칠 수 있습니다.

작은 도구 한두 개라면 굳이 나눌 필요는 없어요. HTTP Request Tool 하나를 서버 워크플로에 바로 붙여도 됩니다. 이번 편에서는 **둘 다** 보여드릴게요.


<br>


## 2. MCP Server Trigger 놓기 — 파라미터 정확히

캔버스에서 트리거를 추가할 때 **MCP Server Trigger** 를 고릅니다. 노드가 놓이면 파라미터가 몇 개 안 돼요.

### 2-1. Path

n8n 공식 문서 표현 그대로 — "기본값으로 **무작위 생성된 MCP URL 경로**가 들어 있으며, 이는 다른 MCP Server Trigger 노드와 충돌하지 않게 하기 위함" 입니다. 직접 지정할 수도 있고, 라우트 파라미터도 넣을 수 있어요.

| 선택 | 언제 | 대가 |
|---|---|---|
| 무작위 기본값 | 빠르게 만들 때 | 워크플로를 복제하면 URL 이 바뀜 |
| 직접 지정 | 운영에 올릴 때 | 이름 충돌을 직접 관리해야 함 |

**운영에 올릴 거라면 직접 지정하세요.** 클라이언트 설정 파일에 URL 을 박아두는데, 그 URL 이 무작위 문자열이면 나중에 워크플로를 복원하거나 다른 인스턴스로 옮길 때 클라이언트를 전부 다시 설정해야 합니다. `realestate` 처럼 의미 있는 경로를 쓰면 그 문제가 사라져요.

### 2-2. Authentication

n8n 문서가 명시하는 선택지는 둘입니다.

| 방식 | 클라이언트가 보내는 것 |
|---|---|
| **Bearer auth** | `Authorization: Bearer <토큰>` |
| **Header auth** | 지정한 이름의 커스텀 헤더 |

자격증명은 HTTP Request 계열 자격증명으로 만듭니다.

> 🚨 **인증 없이 열지 마세요.** MCP 엔드포인트는 공개 URL 이고, 붙기만 하면 그 워크플로의 도구를 **전부** 부를 수 있습니다. 도구가 읽기 전용이어도 마찬가지예요 — 사내 데이터를 조회하는 도구라면 그 자체가 유출 경로입니다. 5편에서 이 얘기를 제대로 합니다.

### 2-3. 두 개의 URL — 이게 ①번 함정

노드 패널 맨 위에 **MCP URL** 이 **두 개** 있습니다. 문서 표현은 "The MCP Server Trigger node has two MCP URLs: test and production" 이고, 패널 상단에서 **Test URL / Production URL** 을 토글해 보게 돼 있어요.

둘의 차이는 "언제 등록되는가" 입니다.

| URL | 언제 살아나나 | 데이터가 어디 보이나 |
|---|---|---|
| **Test** | **Listen for Test Event** 를 누르거나, 워크플로가 비활성일 때 실행하면 | 워크플로 에디터 안 |
| **Production** | 워크플로를 **게시(publish)** 했을 때 | 에디터엔 안 보임. **Executions** 탭에서 |

> **여기서 사람들이 넘어집니다.** 클라이언트에 **테스트 URL** 을 넣어두고, Listen 버튼을 안 누른 채 "도구가 안 보인다" 고 하는 경우가 정말 많아요. 테스트 URL 은 **버튼을 누른 그 순간부터 잠깐** 살아 있는 URL 입니다.
>
> 개발 중엔 테스트 URL + Listen 버튼, 운영은 게시 + 프로덕션 URL. 이 짝을 헷갈리지 마세요.

또 하나 — 프로덕션 실행은 에디터에 결과가 안 뜹니다. 그래서 "실행은 됐는데 아무 일도 안 일어난 것 같다" 는 착각을 하게 되는데, **Executions 탭**을 보면 다 들어와 있어요.

### 2-4. "도구 노드만 실행한다" 의 진짜 의미

1편에서 한 번 짚었지만 여기서 확실히 합니다. n8n 문서 표현은 이래요 — 이 노드는 일반 트리거와 달리 "**도구 노드에만 연결되고 도구 노드만 실행한다**". 클라이언트는 도구 목록을 조회하고 개별 도구를 호출할 수 있다고요.

그래서 캔버스 모양이 보통의 워크플로와 다릅니다.

| 일반 워크플로 | MCP 서버 워크플로 |
|---|---|
| 트리거 → 노드 → 노드 → 끝 | 트리거 아래에 도구들이 **나란히** |
| 위에서 아래로 한 번 흐름 | 호출될 때마다 **도구 하나만** 실행 |

MCP Server Trigger 뒤에 Set 이나 IF 를 일렬로 붙여봐야 **실행되지 않습니다.** 로직이 필요하면 그 로직을 **도구 안에** 넣어야 해요 — 그게 다음 섹션에서 서브워크플로를 쓰는 이유입니다.


<br>


## 3. 도구를 붙이는 세 가지 방법

트리거 아래 도구 연결점에 붙일 수 있는 건 크게 셋입니다.

| 방법 | 좋은 경우 | 한계 |
|---|---|---|
| **HTTP Request Tool** | 외부 REST API 를 그대로 노출 | 응답 가공이 어려움 |
| **앱 노드의 Tool 형태** | Gmail·Slack 등 기성 연동 | 그 서비스에 한정 |
| **Call n8n Workflow Tool** | 로직·가공·여러 단계가 필요 | 워크플로를 하나 더 관리 |

n8n 문서도 도구 중에서 **Call n8n Workflow Tool, Custom Code Tool, HTTP Request Tool** 이 셋을 특히 강력한 선택지로 꼽습니다.

**우리 예제의 선택**은 이래요.

| 도구 | 방법 | 이유 |
|---|---|---|
| `search_region_code` | Call n8n Workflow Tool | 코드 매칭 로직이 필요 |
| `get_apartment_trades` | Call n8n Workflow Tool | XML 파싱 + 요약이 필요 |
| `summarize_price_trend` | Call n8n Workflow Tool | 여러 달을 반복 호출 |

셋 다 서브워크플로예요. 이유는 3편에서 확실해지는데, 미리 한 줄로 말하면 — **공공 API 응답을 그대로 에이전트에게 던지면 토큰이 터지기 때문**입니다. 중간에 반드시 요약 단계가 필요하고, 그러려면 도구가 워크플로여야 해요.


<br>


## 4. `$fromAI()` — 인자를 모델이 채우게 하기

도구는 인자를 받습니다. 그런데 그 값은 워크플로를 만들 때 정할 수 없어요 — **호출하는 순간 모델이 정하는 값**이니까요. 그 자리를 뚫어주는 게 `$fromAI()` 입니다.

### 4-1. 시그니처

```javascript
$fromAI(key, description?, type?, defaultValue?)
```

| 인자 | 타입 | 필수 | 규칙 |
|---|---|---|---|
| `key` | string | ✅ | **1~64자**, 영문 대소문자·숫자·`_`·`-` 만 |
| `description` | string | ❌ | 모델에게 주는 설명 |
| `type` | string | ❌ | `string` `number` `boolean` `json` (기본 `string`) |
| `defaultValue` | any | ❌ | 모델이 못 채웠을 때의 대체값 |

키 규칙이 은근히 자주 걸립니다. **한글, 공백, 점(`.`), 슬래시는 못 씁니다.** 64자 제한도 있고요.

### 4-2. 쓰는 모양

가장 짧은 형태와, 실제로 권장하는 형태입니다.

```javascript
$fromAI("name")
$fromAI("name", "The commenter's name", "string", "Jane Doe")
$fromAI("numItemsInStock", "Number of items in stock", "number", 5)
```

표현식 안에 섞어 쓸 수도 있어요.

{% raw %}
```javascript
Generated by AI: {{ $fromAI("subject") }}
```
{% endraw %}

우리 예제라면 이렇게 됩니다.

{% raw %}
```javascript
{{ $fromAI("lawd_cd", "5-digit legal district code, e.g. 11680 for Gangnam-gu", "string") }}
{{ $fromAI("deal_ymd", "Target month in YYYYMM format, e.g. 202608", "string") }}
```
{% endraw %}

### 4-3. 이건 "참조" 가 아니라 "힌트" 다

여기가 개념적으로 제일 중요해요. n8n 문서가 직접 이렇게 설명합니다 — `$fromAI()` 의 인자들은 **기존 값에 대한 참조가 아니라, 모델이 올바른 데이터를 채워 넣도록 주는 힌트**라고요. 키를 `email` 로 잡으면 모델은 자기 컨텍스트·다른 도구·입력 데이터에서 이메일 주소를 찾고, 채팅 워크플로라면 **사용자에게 물어볼 수도** 있습니다.

> **평범한 말로 옮기면**: `$fromAI("lawd_cd")` 는 "어딘가에 있는 lawd_cd 변수를 가져와라" 가 **아닙니다.** "여기에 lawd_cd 라는 이름의 값이 필요하다, 네가 알아서 채워라" 라고 모델에게 말하는 거예요.
>
> 그래서 **키 이름과 description 이 곧 프롬프트**입니다. `code` 같은 모호한 이름을 쓰면 모델이 아무 코드나 넣어요. 3편의 절반이 이 얘기입니다.

### 4-4. 이게 도구의 입력 스키마가 된다

MCP 클라이언트가 `tools/list` 로 도구 목록을 받아갈 때, 각 도구에는 **입력 스키마(JSON Schema)** 가 붙어 있습니다. n8n 에서는 그 스키마가 **여러분이 쓴 `$fromAI()` 들로부터 만들어집니다.** 키가 속성 이름이 되고, `type` 이 타입이 되고, `description` 이 설명이 되는 거죠.

즉 — **`$fromAI()` 를 하나도 안 쓴 도구는 인자가 없는 도구**입니다. 클라이언트에는 뜨는데 아무 값도 못 넘기죠. 앞에서 말한 ②번 함정이 정확히 이겁니다.

### 4-5. ⚠️ 주의 두 가지

**(가) 붙일 수 있는 자리가 정해져 있습니다.** n8n 문서는 `$fromAI()` 가 "AI Agent 노드에 연결된 도구에서만 사용 가능" 하고 "**Code tool 및 다른 비-도구 클러스터 서브노드에서는 동작하지 않는다**" 고 못 박습니다. MCP Server Trigger 의 도구 자리에서는 MCP 클라이언트가 그 에이전트 역할을 대신하는 구조예요. 다만 문서 문구가 AI Agent 기준으로 쓰여 있으니, **여러분 버전에서 실제로 스키마가 잡히는지 한 번은 눈으로 확인**하시길 권합니다 — 확인 방법은 4편의 `tools/list` 절차에 있습니다.

**(나) 선택 인자가 필수처럼 동작한다는 보고가 있습니다.** n8n 커뮤니티에 "MCP 도구에서 `$fromAI()` 파라미터를 선택(optional)으로 다룰 수 있게 해달라 — 현재는 값을 빼면 스키마 검증이 실패한다" 는 요청이 올라와 있어요. 이건 공식 문서에 적힌 사양이 아니라 **커뮤니티 보고**이니 그대로 믿지는 마시고, 다만 **선택 인자를 설계에 넣기 전에 여러분 버전에서 반드시 실측**하세요. 안전한 우회는 `defaultValue` 를 주는 겁니다.


<br>


## 5. 서브워크플로를 도구로 — Call n8n Workflow Tool

도구 로직을 별도 워크플로로 빼는 방법입니다.

### 5-1. 도구 워크플로 쪽

새 워크플로를 만들고 **Execute Sub-workflow Trigger** 로 시작합니다. 여기서 **Workflow Input Schema** 를 정의해요 — 이 도구가 받을 필드 이름과 타입입니다.

우리 `get_apartment_trades` 라면 이렇게 잡습니다.

| 필드 | 타입 | 뜻 |
|---|---|---|
| `lawd_cd` | String | 법정동 코드 5자리 |
| `deal_ymd` | String | 조회 연월 (YYYYMM) |

그 뒤에 HTTP Request 로 공공 API 를 부르고, 응답을 파싱·요약해서 마지막 노드가 정리된 결과를 내놓게 하면 됩니다.

### 5-2. 서버 워크플로 쪽

MCP Server Trigger 아래에 **Call n8n Workflow Tool** 을 붙이고 이렇게 채웁니다.

| 파라미터 | 값 |
|---|---|
| **Description** | 이 도구를 **언제 써야 하는지** 모델에게 설명 |
| **Source** | `Database` (목록에서 선택하거나 워크플로 ID 입력) |
| **Workflow Inputs** | 서브워크플로 스키마 필드마다 `$fromAI()` |

Source 는 둘 중 하나예요 — 목록/ID 로 고르는 **Database**, 또는 워크플로 JSON 을 통째로 붙여넣는 **Define Below**. 운영에서는 Database 를 씁니다.

Workflow Inputs 는 고정값·표현식·`$fromAI()` 를 섞어 쓸 수 있습니다. 여기서 값을 **고정**해 버리면 모델이 못 바꾸는 인자가 되고, `$fromAI()` 로 두면 모델이 채우는 인자가 돼요. **이 구분이 곧 권한 경계**입니다 — 예를 들어 API 키나 테넌트 ID 는 고정값으로 두고 모델이 손대지 못하게 합니다.

### 5-3. 🚨 ③번 함정 — 서브워크플로도 게시해야 합니다

n8n 문서가 프로덕션 노트로 못 박는 내용이에요. Source 를 Database 로 쓸 때, **프로덕션에서는 서브워크플로가 게시(published)돼 있어야 합니다. 안 그러면 도구 호출이 실패**하고, 에이전트에게 그 취지의 오류 메시지가 돌아갑니다.

증상이 고약한 이유는 이겁니다.

| 상황 | 결과 |
|---|---|
| 에디터에서 테스트 | ✅ 잘 됨 |
| 게시 후 프로덕션 호출 | ❌ 도구 호출 실패 |

"테스트는 되는데 실배포에서 안 된다" 의 1순위 원인이에요. **서버 워크플로를 게시할 때 도구 워크플로도 같이 게시했는지** 매번 확인하세요.

### 5-4. 표현식과 다중 아이템 — 조용한 함정 하나 더

n8n 문서에 이런 주의가 붙어 있습니다 — **서브노드는 표현식으로 여러 아이템을 처리할 때 다른 노드와 다르게 동작**해서, 표현식이 항상 **첫 번째 아이템으로 해석**되고 여러 아이템을 순회하지 않습니다.

즉 도구 안에서 아이템 배열을 기대하고 표현식을 짜면 첫 개만 잡힙니다. 여러 건을 다뤄야 하면 **표현식이 아니라 워크플로 노드로** 반복을 처리하세요.


<br>


## 6. 테스트 — 도구가 실제로 뜨는지 보기

여기까지 만들었으면 두 단계로 확인합니다.

**1단계, n8n 안에서.** 워크플로 에디터에서 MCP Server Trigger 의 **Listen for Test Event** 를 누르고, 클라이언트를 **테스트 URL** 로 붙입니다. 도구 목록이 뜨면 스키마가 제대로 만들어진 거예요.

**2단계, 게시 후.** 워크플로를 게시하고 클라이언트를 **프로덕션 URL** 로 바꿔 붙입니다. 이제 결과는 에디터가 아니라 **Executions** 탭에서 봅니다.

두 단계를 다 하는 이유는 — **테스트에서만 되는 실패 모드가 실제로 존재하기 때문**입니다(5-3의 게시 함정). 게시 후 확인을 건너뛰면 그걸 못 잡아요.

클라이언트 없이 curl 로 직접 도구 목록을 찍어보는 방법은 4편에서 다룹니다. 사실 그게 제일 빠른 확인법이에요.


<br>


## 7. 여기까지의 체크리스트

2편을 끝내기 전에 이것들을 확인하세요.

| 항목 | 확인 |
|---|---|
| Path 를 의미 있는 값으로 지정했나 | 운영이면 필수 |
| 인증을 켰나 (Bearer 또는 Header) | 필수 |
| 도구마다 `$fromAI()` 로 인자를 뚫었나 | 인자 있는 도구면 필수 |
| `$fromAI()` 키가 규칙(1~64자, 영숫자·`_`·`-`)에 맞나 | 필수 |
| 고정돼야 할 값(키·테넌트)을 `$fromAI()` 로 열지 않았나 | 보안 |
| 도구 워크플로도 게시했나 | 프로덕션 필수 |
| 게시 후 프로덕션 URL 로 한 번 더 확인했나 | 필수 |

이 일곱 줄이 앞에서 말한 함정 셋을 전부 덮습니다.

3편에서는 **도구를 잘 만드는 것**으로 넘어갑니다. 동작하는 도구와 에이전트가 실제로 잘 쓰는 도구는 다른 물건이거든요.


<br>

<br>


## 참고 문서

- [n8n Docs — MCP Server Trigger](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-langchain.mcptrigger) — Path, 인증, 테스트/프로덕션 URL, 도구 노드만 실행
- [n8n Docs — Call n8n Workflow Tool](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.toolworkflow) — Description·Source·Workflow Inputs, 게시 요구사항, 다중 아이템 주의
- [n8n Docs — Let AI specify tool parameters (`$fromAI`)](https://docs.n8n.io/build/integrate-ai/ai-examples/use-ai-for-parameters) — 시그니처, 키 규칙, 타입, 힌트 개념
- [n8n Docs — How tools work](https://docs.n8n.io/build/integrate-ai/understand-ai-components/how-tools-work) — 도구의 정의와 주요 도구 노드
- [n8n Community — `$fromAI()` 선택 인자 이슈](https://community.n8n.io/t/allow-mcp-tools-to-treat-fromai-parameters-as-optional-schema-currently-fails-when-omitted/217337) — 공식 문서가 아닌 커뮤니티 보고
