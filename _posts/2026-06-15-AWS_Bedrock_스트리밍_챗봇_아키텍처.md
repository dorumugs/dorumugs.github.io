---
layout: single
title:  "AWS Bedrock 으로 스트리밍 챗봇 만들기 — 자산·라이브러리·아키텍처 정리"
date: 2026-06-15 13:00:00 +0900
categories: coding
tag: [AWS, Bedrock, 챗봇, streaming, converse_stream, FastAPI, LangChain, boto3, SSE, WebSocket, RAG]
author_profile: false
toc: true
description: "AWS Bedrock 으로 토큰 스트리밍 챗봇을 만들 때 필요한 AWS 자산, 파이썬 라이브러리, 그리고 converse_stream → SSE/WebSocket 으로 사용자에게 글자를 흘려주는 아키텍처와 기능을 한 번에 정리했습니다."
---


## Summary

챗봇에서 가장 체감이 좋은 건 **답이 한 글자씩 흘러나오는 스트리밍**이에요. ChatGPT 처럼요. AWS Bedrock 으로 이걸 만들려면 모델만 부르면 끝이 아니라, **모델 호출 → 토큰 릴레이 → 프론트 표시** 까지 한 줄로 이어줘야 합니다.

이 글은 그 한 줄을 만들기 위해 **어떤 AWS 자산이 필요한지**, **어떤 라이브러리를 쓰는지**, **무엇을 직접 만들어야 하는지**를 정리하고, 핵심인 `converse_stream` 스트리밍을 SSE/WebSocket 으로 흘려주는 아키텍처를 코드와 함께 풀었어요.

> 💡 **이 글에서 다루는 것**
> - Bedrock 챗봇 아키텍처 한 그림 + 구성요소 표
> - 준비물: Bedrock 모델 액세스·IAM·컴퓨트·세션 저장소
> - 라이브러리: `boto3`, FastAPI, `langchain-aws`, `anthropic[bedrock]`
> - 핵심: `converse_stream` 으로 토큰을 받아내는 법 (이벤트 구조 포함)
> - 스트리밍 전달 3가지 — SSE / WebSocket / Lambda 응답 스트리밍
> - 대화 상태 저장, RAG·가드레일로 기능 확장
> - 마지막에 "만들어야 하는 것" 체크리스트


<br>

<br>



## 1. 큰 그림: 어떤 조각들이 필요한가

먼저 전체 흐름부터 봐요. 사용자가 메시지를 보내면, 백엔드가 Bedrock 을 **스트리밍 모드**로 부르고, 거기서 떨어지는 토큰 조각을 그대로 사용자 화면으로 흘려보내는 구조예요.

```text
[브라우저]                [백엔드]                    [AWS Bedrock]
   │  메시지 전송            │                              │
   ├──────────────────────▶ │  converse_stream() 호출       │
   │                        ├────────────────────────────▶ │
   │                        │   contentBlockDelta 이벤트     │
   │   토큰 한 조각씩         │ ◀──────(스트림)────────────── │
   │ ◀───(SSE/WebSocket)─── │   (반복)                      │
   │  화면에 흘려서 표시      │                              │
   │                        │  대화 저장 (DynamoDB)          │
```

각 단계에 필요한 AWS 구성요소를 묶으면 다음과 같아요.

| 역할 | 구성요소 | 비고 |
|---|---|---|
| 모델 추론 | **Amazon Bedrock** | Claude 등 파운데이션 모델. 콘솔에서 모델 액세스 먼저 켜야 함 |
| 권한 | **IAM 역할/정책** | `bedrock:InvokeModelWithResponseStream` 필요 |
| 백엔드 컴퓨트 | **Lambda / ECS Fargate / App Runner** | 스트리밍을 끝까지 흘리려면 응답 스트리밍 지원이 핵심 |
| 클라이언트 연결 | **API Gateway(WebSocket) / ALB / Lambda Function URL** | 단방향이면 SSE, 양방향이면 WebSocket |
| 대화 상태 | **DynamoDB** | 세션별 메시지 히스토리 |
| 문서·검색(RAG) | **S3 + Bedrock Knowledge Bases** | 사내 문서 기반 답변 |
| 안전장치 | **Bedrock Guardrails** | 금칙어·민감정보·주제 필터 |
| 인증 | **Cognito** | 사용자 로그인/토큰 |
| 관측 | **CloudWatch / X-Ray** | 토큰 사용량·지연·에러 추적 |

> ✅ 처음 만들 때는 **Bedrock + IAM + 컴퓨트 + DynamoDB** 네 개면 충분해요. RAG·가드레일·인증은 챗봇이 돌아가는 걸 확인한 뒤에 한 겹씩 얹으면 됩니다.


<br>

<br>



## 2. 준비물: AWS 자산 체크리스트

### 2.1 Bedrock 모델 액세스부터

Bedrock 은 가입했다고 모델이 바로 열리지 않아요. **콘솔 → Bedrock → Model access** 에서 쓰려는 모델(예: Claude 계열)을 명시적으로 요청해 둬야 호출이 됩니다. 이걸 안 켜고 호출하면 `AccessDeniedException` 이 떨어져요.

또 하나, **리전**을 확인하세요. 같은 모델이라도 리전마다 가용 여부가 달라요. 서울 리전(`ap-northeast-2`)을 쓴다면, 단일 리전 모델 ID 대신 **크로스리전 추론 프로파일**(ID 앞에 `apac.` / `us.` 같은 접두사가 붙음)을 써야 하는 경우가 많습니다.

> ⚠️ 모델 ID 와 리전 가용성은 자주 바뀌어요. 글의 예시 ID(`apac.anthropic.claude-...`)는 형태를 보여주는 것이고, **실제 사용 가능한 ID 는 콘솔의 모델 목록에서 그대로 복사**하는 걸 추천드려요.


### 2.2 IAM 권한

스트리밍 챗봇이라면 **스트리밍용 액션**이 핵심이에요. 일반 호출(`InvokeModel`)과 스트리밍 호출(`InvokeModelWithResponseStream`)은 액션이 다릅니다. `Converse` / `ConverseStream` API 도 내부적으로 이 두 액션 권한으로 인가돼요.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-*",
        "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/apac.anthropic.claude-*"
      ]
    }
  ]
}
```

크로스리전 추론 프로파일을 쓰면 **추론 프로파일 ARN** 과 **그 프로파일이 라우팅하는 파운데이션 모델 ARN** 양쪽에 권한이 필요해요. 위처럼 둘 다 넣어두면 안전합니다. (`<ACCOUNT_ID>` 는 본인 계정 ID 로 치환하세요.)


### 2.3 컴퓨트 — "끝까지 흘릴 수 있는가"가 기준

챗봇 백엔드를 고를 때 일반 API 와 다른 점은 **응답을 토막토막 끝까지 내보낼 수 있느냐**예요.

| 선택지 | 스트리밍 | 언제 좋은가 |
|---|---|---|
| **Lambda Function URL** (응답 스트리밍) | ✅ `RESPONSE_STREAM` 모드 | 서버 관리 없이 빠르게. 단 실행시간·페이로드 한계 주의 |
| **ECS Fargate / App Runner** | ✅ (FastAPI 등) | 긴 세션·복잡한 로직·상시 가동 |
| **EC2** | ✅ | 완전한 제어가 필요할 때 |
| **일반 API Gateway(REST) + Lambda** | ❌ 버퍼링됨 | 스트리밍엔 부적합 (WebSocket API 로 가야 함) |

> 🚨 가장 흔한 함정: **REST API Gateway 뒤에 Lambda 를 두면 응답이 통째로 모였다가 한 번에 나가요.** 스트리밍이 죽습니다. 단방향 스트리밍은 Lambda 응답 스트리밍이나 ALB/Fargate(SSE)로, 양방향은 **API Gateway WebSocket** 으로 가야 해요.


<br>

<br>



## 3. 라이브러리 정리

파이썬 기준으로, 역할별로 이 정도만 알면 돼요.

| 라이브러리 | 역할 | 핵심 |
|---|---|---|
| `boto3` / `botocore` | Bedrock 호출의 기본 | `bedrock-runtime` 클라이언트의 `converse_stream()` |
| `aioboto3` | 비동기 호출 | FastAPI 같은 async 서버에서 이벤트 루프 안 막기 |
| `fastapi` + `uvicorn` | 백엔드 + SSE | `StreamingResponse` 로 토큰을 흘림 |
| `langchain-aws` | 추상화 계층 | `ChatBedrockConverse(...).stream()` 한 줄로 스트리밍 |
| `anthropic` (`anthropic[bedrock]`) | 대안 클라이언트 | Bedrock 위에서 Messages API 형태로 호출 |

크게 **두 갈래**예요.

- **AWS 네이티브로 가는 길** — `boto3` 의 `bedrock-runtime` 클라이언트로 `converse_stream` 을 직접 호출. 모델이 바뀌어도 호출 코드가 같아서(통합 Converse API) 유지보수가 편해요. 이 글의 메인 경로입니다.
- **프레임워크/SDK 로 감싸는 길** — `langchain-aws` 의 `ChatBedrockConverse` 로 추상화하거나, `anthropic[bedrock]` 의 `AnthropicBedrock` 클라이언트로 Messages API 스타일을 그대로 쓰는 방법. RAG 체인이나 에이전트를 얹을 거면 LangChain 쪽이 편해요.

이 글은 군더더기 없이 **`boto3` + `converse_stream`** 으로 핵심을 보여주고, 마지막에 확장 포인트를 짚을게요.


<br>

<br>



## 4. 핵심: Bedrock 에서 토큰을 받아내기

여기서부터가 진짜 핵심이에요. `converse_stream` 은 답을 한 번에 주지 않고, **이벤트 스트림**으로 토막을 흘려줘요. 우리는 그중 글자가 담긴 이벤트만 골라내면 됩니다.

```python
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="ap-northeast-2")

def stream_chat(user_text: str):
    resp = bedrock.converse_stream(
        modelId="apac.anthropic.claude-opus-4-8",   # 예시 ID — 콘솔에서 실제 ID 확인
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": 1024},
    )
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            yield event["contentBlockDelta"]["delta"]["text"]
```

실제로 어떤 값이 흘러나오는지 한 번 돌려볼게요. 제너레이터라 `for` 로 받아서 이어 붙이면 됩니다.

```python
for chunk in stream_chat("한국의 수도는 어디야? 한 문장으로."):
    print(chunk, end="", flush=True)
```

```text
한국의 수도는 서울입니다.
```

`print` 에 `end=""` 와 `flush=True` 를 준 게 포인트예요. 조각이 도착하는 즉시 화면에 찍혀서, 터미널에서도 글자가 흘러나오는 게 보입니다.

`resp["stream"]` 에서 나오는 이벤트는 종류가 몇 개 있고, 우리가 챙길 건 사실상 `contentBlockDelta` 하나예요. 나머지는 시작/끝/사용량 신호입니다.

| 이벤트 키 | 담긴 것 | 용도 |
|---|---|---|
| `messageStart` | `role` | 응답 시작 신호 |
| `contentBlockDelta` | `delta.text` | **토큰 조각** — 화면에 흘릴 대상 |
| `contentBlockStop` | — | 콘텐츠 블록 종료 |
| `messageStop` | `stopReason` | 종료 이유(`end_turn` 등) |
| `metadata` | `usage`, `metrics` | 토큰 사용량·지연시간 |

> 💡 `metadata` 이벤트의 `usage` 를 로그로 남겨두면 나중에 비용 추적이 정말 편해져요. 입력/출력 토큰 수가 여기 다 찍힙니다.

이전 대화를 이어가려면 `messages` 배열에 **지난 user/assistant 턴을 순서대로 쌓아서** 같이 보내면 돼요. Bedrock 호출 자체는 상태가 없어서(stateless), 대화 기억은 우리가 들고 있어야 합니다. 그 저장은 6장에서 다룰게요.


<br>

<br>



## 5. 스트리밍을 사용자에게 전달하는 3가지 방법

백엔드가 토큰을 받아냈으면, 이걸 브라우저까지 흘려보내야 해요. 방법은 크게 세 가지입니다.

| 방식 | 특징 | 언제 |
|---|---|---|
| **SSE** (Server-Sent Events) | 단방향(서버→클라), 구현 단순 | 일반 챗봇 답변 스트리밍 |
| **WebSocket** | 양방향, 연결 유지 | 실시간 협업·중간 취소·타이핑 표시 |
| **Lambda 응답 스트리밍** | 서버리스로 SSE 류 흘리기 | 서버 관리 없이 가고 싶을 때 |

가장 많이 쓰는 **SSE** 를 FastAPI 로 붙이면 이렇게 돼요. 4장의 제너레이터를 그대로 `StreamingResponse` 에 물리면 됩니다.

```python
import json
import boto3
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
bedrock = boto3.client("bedrock-runtime", region_name="ap-northeast-2")

@app.post("/chat")
def chat(body: dict):
    def event_stream():
        resp = bedrock.converse_stream(
            modelId="apac.anthropic.claude-opus-4-8",
            messages=[{"role": "user", "content": [{"text": body["message"]}]}],
            inferenceConfig={"maxTokens": 2048},
        )
        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                token = event["contentBlockDelta"]["delta"]["text"]
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

SSE 는 `data: ...\n\n` 형식으로 한 덩어리씩 내려주는 규약이에요. 한글이 `\uXXXX` 로 깨지지 않게 `ensure_ascii=False` 를 넣은 것도 작은 포인트입니다.

프론트에서는 받아서 화면에 이어 붙여요. 한 가지 주의할 점은, 브라우저 기본 `EventSource` 는 **GET 만** 됩니다. 위처럼 `POST /chat` 로 메시지를 실어 보내려면 `fetch` 의 `ReadableStream` 으로 직접 읽어야 해요.

```javascript
const res = await fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "안녕!" }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  // decoder.decode(value) 안의 "data: {...}" 줄을 파싱해서 화면에 이어 붙이기
  process(decoder.decode(value));
}
```

양방향이 필요하면 **API Gateway WebSocket API** 로 가요. `$connect` / `$disconnect` / 메시지 라우트를 Lambda 에 연결하고, 백엔드는 토큰을 받을 때마다 `post_to_connection` 으로 그 연결에 밀어 넣는 구조예요. 사용자가 생성 도중 "그만" 을 누르는 취소 기능이나, 여러 사용자에게 동시에 흘리는 시나리오에 잘 맞습니다.


<br>

<br>



## 6. 대화 상태와 세션 저장

Bedrock 호출은 기억이 없어서, **이전 대화를 우리가 보관**해야 멀티턴이 돼요. 보통 DynamoDB 에 세션 단위로 메시지를 쌓아둡니다.

- 파티션 키: `session_id`, 정렬 키: `timestamp` (또는 메시지 순번)
- 한 항목 = 한 메시지(`role`, `content`, 시각)
- 새 요청이 오면 해당 세션의 메시지를 시간순으로 읽어 `messages` 배열을 복원 → Bedrock 에 같이 전달

```python
# 흐름 요약 (의사코드)
history = load_messages(session_id)              # DynamoDB 에서 복원
history.append({"role": "user",
                "content": [{"text": user_text}]})

answer = ""
for token in stream_with_history(history):       # converse_stream 호출
    answer += token
    push_to_client(token)                        # SSE/WebSocket 으로 흘림

save_message(session_id, "user", user_text)      # 끝나면 양쪽 저장
save_message(session_id, "assistant", answer)
```

> ✅ 대화가 길어지면 토큰이 계속 불어나요. 오래된 턴은 **요약해서 압축**하거나 최근 N턴만 남기는 식으로 컨텍스트 길이를 관리하면 비용과 지연이 같이 줄어듭니다.


<br>

<br>



## 7. (옵션) RAG·가드레일로 기능 확장

기본 챗봇이 돌면, 여기서부터는 기능을 한 겹씩 얹는 단계예요.

- **RAG (사내 문서 기반 답변)** — **Bedrock Knowledge Bases** 를 쓰면 검색 파이프라인을 직접 안 짜도 돼요. S3 에 문서를 올리고, 벡터 저장소(OpenSearch Serverless 또는 Aurora pgvector)에 색인한 뒤, 질문이 오면 관련 청크를 찾아 프롬프트에 끼워 넣어 줍니다. "우리 회사 규정 알려줘" 같은 질문에 근거를 달아 답하게 만드는 핵심이에요.
- **가드레일** — **Bedrock Guardrails** 로 금칙 주제·욕설·개인정보(PII) 마스킹·프롬프트 인젝션 방어를 정책으로 걸 수 있어요. 모델 호출 파라미터에 가드레일 ID 를 얹는 방식이라, 코드 변경이 거의 없습니다.
- **도구 사용(툴 콜링)** — `converse_stream` 은 `toolConfig` 로 함수 호출도 지원해요. 모델이 "이 함수를 이 인자로 불러줘" 라고 요청하면, 백엔드가 실제 함수를 실행해 결과를 다시 모델에 넘기는 패턴이에요. 날씨 조회·DB 검색·주문 처리 같은 액션형 챗봇으로 확장할 때 씁니다.
- **인증·관측** — 사용자 로그인은 Cognito 로, 토큰 사용량·지연·에러는 CloudWatch 로. 4장에서 챙긴 `metadata.usage` 를 지표로 쌓아두면 비용 대시보드가 바로 나와요.


<br>

<br>



## 8. 만들어야 하는 것 — 체크리스트

지금까지 나온 걸 "직접 만들/켜야 하는 것" 기준으로 모으면 이렇게 정리돼요.

**필수 (기본 스트리밍 챗봇)**

- [ ] Bedrock 콘솔에서 모델 액세스 활성화 + 리전/모델 ID 확정
- [ ] `InvokeModelWithResponseStream` 권한이 있는 IAM 역할
- [ ] `converse_stream` 으로 토큰을 받아 yield 하는 스트리밍 핸들러
- [ ] SSE 또는 WebSocket 으로 토큰을 흘리는 백엔드 엔드포인트
- [ ] 토큰을 이어 붙여 표시하는 프론트 챗 UI
- [ ] DynamoDB 세션 저장/복원으로 멀티턴 유지
- [ ] 응답 스트리밍이 되는 컴퓨트(Lambda 응답 스트리밍 / Fargate 등)

**확장 (기능 추가)**

- [ ] Knowledge Bases + S3 + 벡터 저장소로 RAG
- [ ] Guardrails 로 안전장치
- [ ] `toolConfig` 로 도구 사용(액션형 챗봇)
- [ ] Cognito 인증 + CloudWatch 토큰 사용량 대시보드

> ✅ 이 체크리스트의 진짜 핵심은 위쪽 7개예요. **`converse_stream` 의 이벤트를 SSE/WebSocket 으로 끝까지 흘리는 한 줄**이 완성되면, 나머지는 그 위에 붙이는 장식에 가깝습니다.


<br>

<br>



정리하면, Bedrock 스트리밍 챗봇은 **모델 액세스·IAM·응답 스트리밍 컴퓨트** 를 깔고, `converse_stream` 의 `contentBlockDelta` 를 **SSE 나 WebSocket 으로 릴레이**한 뒤, DynamoDB 로 대화를 기억하게 만들면 뼈대가 끝나요. RAG·가드레일·툴 콜링은 그 뼈대 위에서 기능을 키우는 단계고요.

일단 오늘은 여기까지.....   
다음 글에서는 `converse_stream` 의 `toolConfig` 로 도구를 부르는 액션형 챗봇을 더 잘게 쪼개서 정리해볼게요.
