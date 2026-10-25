---
layout: single
title:  "(5/5) 기존 OpenAI·Anthropic API 코드를 Bedrock 으로 옮기기 — 목록 만들기부터 단계 전환까지"
date: 2026-10-25 09:40:00 +0900
categories: coding
tag: [Bedrock, AWS, OpenAI, Claude, 마이그레이션, LiteLLM, LangChain, n8n, 임베딩, RAG, 비용추적]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-migrate/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-migrate/header.svg
description: "이미 OpenAI API·Anthropic API 를 직접 호출하는 코드가 있다면 Bedrock 전환은 새로 세팅하는 것과 다른 문제예요. 호출 지점 목록을 만드는 방법, SDK 코드 전환, Bedrock 에 없는 기능과 대안, 임베딩을 바꿀 때의 재색인, LiteLLM·LangChain·n8n 설정, 비용 추적 방식 변경, 단계적 전환과 롤백을 정리했어요."
series: bedrock-gpt-claude-setup
series_order: 5
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="5" %}

## Summary

1~4편은 Bedrock 을 처음 세팅하는 경우를 다뤘어요. 이미 OpenAI API 나 Anthropic API 를 직접 호출하는 코드가 있다면, 그 코드를 Bedrock 으로 옮기는 작업이 따로 필요해요. 이 블로그에서도 [ChromaDB RAG](/coding/ChromaDB_RAG_파이프라인_LLM_답변생성/), [LLM API 비용 추적](/coding/LLM_API_비용추적_과금구조/), [모델 라우팅](/coding/LLM_API_비용추적_모델라우팅/) 같은 글이 모두 직접 API 를 기준으로 작성돼 있어요.

이미 동작하는 코드를 옮길 때는 새로 만들 때와 확인할 것이 달라요. 엔드포인트와 인증만 바꾸면 되는 호출도 있지만, Bedrock 에 없는 기능을 쓰는 호출은 대안이 필요해요. 임베딩 모델을 바꾸면 벡터를 전부 다시 만들어야 하고, 공급사 사용량 API 에 의존하던 비용 추적도 다른 방식으로 바꿔야 합니다.

이번 편은 이런 전환 작업을 순서대로 정리해요. 기본값은 1~4편과 같아요. 서울 리전에서 `bedrock-runtime` 을 쓰고, 모델 ID 는 `global.` 로 시작합니다. [2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)의 계정·권한 설정은 끝났다고 가정해요. 쿼터 증설은 승인까지 시간이 걸리니 2절의 목록을 만들 때 함께 요청하세요.

> 📅 **기준일 2026-10-25.** 공식 문서 기준이에요. Bedrock 의 OpenAI 호환 경로에서 **지원이 확인된 것**과 **문서에 없어서 확인하지 못한 것**을 구분해 적었어요.

> 💡 **이 글에서 다루는 것**
> - 전환할 때 바뀌는 것과 그대로인 것
> - 호출 지점 목록 만들기
> - OpenAI SDK·Anthropic SDK 코드 전환과 모델 ID 대응
> - Bedrock 에 없는 기능과 대안
> - 임베딩과 RAG — 재색인이 필요한 경우
> - LiteLLM·LangChain·n8n 설정 바꾸기
> - 비용 추적 바꾸기
> - 단계적 전환과 롤백

<br>

<br>



## 1. 전환할 때 바뀌는 것과 그대로인 것

| 항목 | 직접 API | Bedrock |
|---|---|---|
| 인증 | 공급사 API 키 | IAM 자격증명<br>또는 단기 토큰 |
| 주소 | 공급사 주소 | `bedrock-runtime`<br>리전 주소 |
| 모델 ID | `gpt-…`<br>`claude-…` | `global.openai.…`<br>`global.anthropic.…` |
| 요청 형식 | 그대로 | 대부분 그대로 |
| 사용량 조회 | 공급사<br>Usage API | CloudWatch·<br>Cost Explorer |
| 한도 | 공급사 등급 | AWS 서비스 쿼터 |
| 청구 | 공급사 | AWS 청구서<br>(Marketplace) |

OpenAI SDK 는 `/openai/v1` 경로로, Anthropic SDK 는 `/anthropic` 경로로 기존과 같은 요청을 보낼 수 있어요. 바뀌는 것은 주로 요청 바깥쪽입니다. 인증, 모델 ID, 사용량 조회, 한도, 청구가 모두 AWS 방식을 따르게 돼요.

데이터 처리 위치도 바뀌어요. 서울에서 `global.` 프로필로 호출하면 요청이 국외 리전에서 처리될 수 있어요([1편 5절](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)). 직접 API 도 국외에서 처리되고 있었다면 달라지는 것은 크지 않아요. 그래도 전환 문서에는 처리 위치의 변화를 적어 두세요.

<br>

<br>



## 2. 호출 지점 목록 만들기

전환은 목록에서 시작해요. 코드만 보면 놓치는 곳이 많아서, 아래 다섯 곳을 모두 확인하세요.

| 위치 | 찾을 것 |
|---|---|
| 애플리케이션 코드 | SDK 생성 코드,<br>`model=` 값 |
| 설정·비밀 저장소 | API 키 이름,<br>`base_url` |
| 게이트웨이 설정 | LiteLLM<br>`model_list` |
| 노코드 도구 | n8n 자격증명,<br>워크플로 노드 |
| 개발 도구·CI | Claude Code·Codex<br>설정, CI 비밀값 |

코드와 설정 파일은 아래 명령으로 찾아요. n8n 워크플로는 JSON 으로 내보낸 뒤 같은 명령으로 검색하면 돼요. 첫 번째 명령은 API 키와 공급사 주소를 찾아요. 두 번째 명령은 모델 이름이 적힌 곳을 찾는데, `MODEL = "gpt-…"` 같은 대문자 변수와 LiteLLM 설정의 `model: anthropic/claude-…` 형식도 잡아요.

```shell
grep -rnE "OPENAI_API_KEY|ANTHROPIC_API_KEY|sk-ant-|sk-proj-|api\.openai\.com|api\.anthropic\.com" \
  --include='*.py' --include='*.ts' --include='*.yaml' --include='*.yml' \
  --include='*.json' --include='*.toml' --include='.env*' \
  --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=.git .

grep -rniE "model(_id|_name)?\s*[=:]\s*[\"']?((openai|anthropic)/)?(gpt-|o[0-9]+[-\"' ]|claude-|text-embedding)" \
  --include='*.py' --include='*.ts' --include='*.yaml' --include='*.yml' \
  --include='*.json' --include='*.toml' --include='.env*' \
  --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=.git .
```

찾은 호출 지점마다 아래 항목을 표로 정리해요. 이 표가 이후 작업의 기준이 됩니다.

| 열 | 예시 |
|---|---|
| 위치 | `rag/answer.py:42` |
| 공급사·API | Anthropic Messages |
| 현재 모델 | `claude-opus-4-8` |
| 쓰는 기능 | 스트리밍, 도구 호출 |
| Bedrock 대응 | 그대로 / 대안 필요<br>/ 직접 API 유지 |
| 담당자 | 이름 |

"쓰는 기능" 열이 중요해요. 4절의 표와 대조해서 Bedrock 에 없는 기능을 쓰는 호출을 이 단계에서 골라내야 합니다.

<br>

<br>



## 3. SDK 코드 전환

먼저 모델 ID 를 바꿔야 해요. 직접 API 에서 쓰던 모델이 Bedrock 에 없거나 서울에서 호출되지 않을 수 있어서, 모델이 1:1 로 대응되지는 않아요.

| 직접 API 모델 | Bedrock 모델 ID(서울) |
|---|---|
| `claude-opus-…` | `global.anthropic.`<br>`claude-opus-5-5` |
| `claude-sonnet-…` | `global.anthropic.`<br>`claude-sonnet-5-5` |
| `claude-haiku-…` | `global.anthropic.`<br>`claude-haiku-5-5` |
| `gpt-…` | `global.openai.`<br>`gpt-6-astra` |

같은 버전이 Bedrock 에 있으면 그 버전을 쓰고, 없으면 가장 가까운 모델로 바꾸세요. 버전이 바뀌면 출력이 달라지므로 8절 ③의 비교 평가를 반드시 거쳐야 해요. 실제로 쓸 수 있는 ID 는 `aws bedrock list-inference-profiles --region ap-northeast-2` 로 확인하세요.

### 3-1. OpenAI SDK

OpenAI SDK 는 클라이언트를 만드는 부분과 모델 ID 만 바꾸면 돼요.

```python
# 전환 전
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY
r = client.responses.create(
    model="gpt-5.4",
    input=question,
)
```

```python
# 전환 후
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = "ap-northeast-2"
client = OpenAI(
    base_url=f"https://bedrock-runtime.{REGION}"
    ".amazonaws.com/openai/v1",
    api_key=provide_token(region=REGION),
)
r = client.responses.create(
    model="global.openai.gpt-6-astra",
    input=question,
)
```

옮길 때 확인할 점이에요.

- **Chat Completions 와 Responses 는 지원돼요.** `store` 와 `previous_response_id` 도 쓸 수 있어요. 다만 저장된 응답은 그 요청을 처리한 리전에만 있고, `background=true` 는 400 오류가 나요.
- **`store` 는 기본값이 `true` 예요.** 이 경우 입력과 출력이 30일 동안 처리 리전에 저장돼요. `global.` 프로필로 호출하면 그 리전이 국외일 수 있어요. 이전 응답을 이어 쓰는 기능이 필요 없다면 요청마다 `store=False` 를 보내세요.
- **Responses 경로는 권한 확인 대상이 더 있어요.** AWS 문서에 따르면 계정의 기본 프로젝트(default project) 리소스에도 `bedrock:InvokeModel` 을 확인하고, 저장된 응답을 조회·삭제하려면 별도 권한(`bedrock:GetInvoke` 등)이 필요해요. [2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)의 허용목록 때문에 막히면 오류 메시지에 나온 리소스와 액션을 확인해 추가하세요.
- **`client.models.list()` 는 동작하지 않아요.** `bedrock-runtime` 은 OpenAI 호환 `GET /models` 를 구현하지 않아요. 모델 목록은 `aws bedrock list-inference-profiles` 로 확인하세요.
- **지원 API 표에 없는 기능이 있어요.** Embeddings·Files·Assistants·Realtime·이미지 생성·Fine-tuning·Moderation 은 Bedrock 엔드포인트 문서의 표에 없어요. "지원하지 않는다" 는 문장을 찾지는 못했지만, 쓸 수 없다고 보고 계획하세요(4절).
- **같은 모델이 없을 수 있어요.** Bedrock 의 OpenAI 모델은 GPT-6, GPT-5.4~5.6, gpt-oss 계열이고, `gpt-4o` 같은 이전 모델은 없어요.

### 3-2. Anthropic SDK

Anthropic SDK 도 같은 방식이에요. `base_url` 을 `/anthropic` 경로로 바꾸고 단기 토큰을 넣어요.

```python
# 전환 전
from anthropic import Anthropic

client = Anthropic()  # ANTHROPIC_API_KEY
msg = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=4096,
    messages=messages,
)
```

```python
# 전환 후
from anthropic import Anthropic
from aws_bedrock_token_generator import provide_token

REGION = "ap-northeast-2"
client = Anthropic(
    base_url=f"https://bedrock-runtime.{REGION}"
    ".amazonaws.com/anthropic",
    api_key=provide_token(region=REGION),
)
msg = client.messages.create(
    model="global.anthropic.claude-opus-5-5",
    max_tokens=4096,
    messages=messages,
)
```

- 요청·응답 형식은 같아서 `msg.content`, `msg.usage` 를 읽는 코드는 그대로 쓸 수 있어요.
- 모델을 Opus 5.5 로 올리면 `temperature`·`top_p` 를 보내는 코드는 400 오류가 나요([3편 3절](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)). 응답에서 `content[0]` 을 바로 텍스트로 읽는 코드도 고쳐야 해요. 생각(thinking) 블록이 앞에 올 수 있으므로 `"".join(b.text for b in msg.content if b.type == "text")` 처럼 텍스트 블록만 골라 읽으세요.
- 기존 모델을 그대로 쓰고 싶다면 그 모델의 Bedrock 프로필이 서울에서 제공되는지 먼저 확인하세요. 예전 모델은 `anthropic.claude-sonnet-4-5-20250929-v1:0` 처럼 날짜가 붙은 ID 를 쓰고, 기본 ID 로 호출하면 400 이 나요. 프로필 ID(`global.…`)로 호출해야 합니다.

### 3-3. 토큰 만료와 롤백 스위치

위 예제는 클라이언트를 만들 때 토큰을 한 번만 발급해요. 단기 토큰은 발급에 쓴 자격증명이 만료되거나 12시간이 지나면(둘 중 이른 쪽) 만료돼요. SSO 권한 세트의 기본 세션은 1시간이라서, 오래 실행되는 서버는 토큰을 주기적으로 새로 발급해야 해요.

전환 기간에는 요청마다 직접 API 와 Bedrock 중 어디로 보낼지 고를 수 있게 해 두세요. 그러면 일부만 전환하거나 되돌리기가 쉬워요.

```python
import os
import random
import threading
import time

from anthropic import Anthropic, AuthenticationError, PermissionDeniedError
from aws_bedrock_token_generator import provide_token

REGION = "ap-northeast-2"
BASE_URL = f"https://bedrock-runtime.{REGION}.amazonaws.com/anthropic"
MODELS = {
    "direct": "claude-opus-4-8",
    "bedrock": "global.anthropic.claude-opus-5-5",
}
TOKEN_TTL = 15 * 60

_lock = threading.Lock()
_cache = {"bedrock": None, "at": 0.0, "direct": None}


def _bedrock_client(force: bool = False) -> Anthropic:
    with _lock:
        stale = time.time() - _cache["at"] > TOKEN_TTL
        if force or _cache["bedrock"] is None or stale:
            token = provide_token(region=REGION)
            _cache["bedrock"] = Anthropic(base_url=BASE_URL, api_key=token)
            _cache["at"] = time.time()
        return _cache["bedrock"]


def _direct_client() -> Anthropic:
    if _cache["direct"] is None:
        _cache["direct"] = Anthropic()
    return _cache["direct"]


def pick_provider() -> str:
    if os.environ.get("LLM_PROVIDER") == "direct":
        return "direct"
    ratio = float(os.environ.get("BEDROCK_RATIO", "1"))
    return "bedrock" if random.random() < ratio else "direct"


def create_message(messages: list, max_tokens: int = 4096):
    provider = pick_provider()
    if provider == "direct":
        return _direct_client().messages.create(
            model=MODELS["direct"], max_tokens=max_tokens, messages=messages
        )
    try:
        client = _bedrock_client()
        return client.messages.create(
            model=MODELS["bedrock"], max_tokens=max_tokens, messages=messages
        )
    except (AuthenticationError, PermissionDeniedError):
        client = _bedrock_client(force=True)
        return client.messages.create(
            model=MODELS["bedrock"], max_tokens=max_tokens, messages=messages
        )
```

- **토큰 갱신:** 토큰은 15분마다 새로 발급해요. 자격증명 만료 직전에 발급한 토큰은 15분보다 먼저 만료될 수 있으므로, 인증 오류가 나면 새 토큰으로 한 번 더 시도해요. 여러 스레드가 동시에 토큰을 발급하지 않도록 잠금(`_lock`)도 걸었어요.
- **일부 전환:** `BEDROCK_RATIO=0.1` 이면 요청의 약 10% 가 Bedrock 으로 가요. 비율을 올려 가며 전환하고, 1 이면 전부 Bedrock 으로 보내요.
- **롤백:** `LLM_PROVIDER=direct` 로 두면 모든 요청이 직접 API 로 가요. 이 요청은 Bedrock 의 IAM·로그·Guardrails 를 거치지 않으니, 롤백 경로도 1편에서 정한 데이터 경계 안에 있는지 미리 확인하세요. 환경변수는 프로세스가 시작될 때 읽히므로, 값을 바꾼 뒤에는 서비스를 다시 시작해야 해요. 재시작 없이 바꾸려면 설정 저장소(예: AWS AppConfig, Parameter Store)에서 값을 읽게 하세요. 전환이 끝나기 전까지는 직접 API 키를 지우지 마세요(8절).
- **직접 API 키 보관:** 공존 기간에는 직접 API 키가 2편 차단 정책을 우회하는 경로가 될 수 있어요. 키는 환경변수에 상시 두지 말고 Secrets Manager 같은 비밀 저장소에 두고 `direct` 경로에서만 읽게 하세요. 공급사 콘솔의 지출 한도도 낮춰 두세요.
- 서버가 EC2·ECS·Lambda 처럼 AWS 위에서 실행된다면 토큰 대신 실행 역할(IAM Role)의 자격증명을 쓰는 편이 나아요. boto3 는 역할 자격증명을 자동으로 갱신하므로 Converse([4편 3절](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/))로 호출하면 토큰을 관리할 필요가 없어요.

<br>

<br>



## 4. Bedrock 에 없는 기능과 대안

목록의 "쓰는 기능" 열을 이 표와 대조하세요.

| 기존 기능 | Bedrock | 대안 |
|---|---|---|
| Anthropic<br>Message Batches | 없음 | Bedrock 배치 추론<br>(S3 JSONL) |
| Anthropic<br>Files API·URL 입력 | 없음 | base64 로<br>본문에 포함 |
| Anthropic 웹 검색·<br>코드 실행 도구 | 없음 | 클라이언트 도구로<br>직접 구현 |
| Anthropic<br>Usage·Cost API | 없음 | CloudWatch·<br>Cost Explorer(7절) |
| Anthropic<br>서버측 fallbacks | 없음 | 클라이언트에서<br>재시도·대체 |
| OpenAI Batch | gpt-oss 만<br>(GPT-6 없음) | `/openai/v1/batches`<br>(S3 입력) |
| OpenAI Embeddings | 표에 없음 | Titan·Cohere<br>임베딩(5절) |
| OpenAI Files·<br>Assistants·Realtime | 표에 없음 | 직접 API 유지<br>또는 재설계 |
| OpenAI<br>`GET /models` | 없음 | AWS CLI 로<br>프로필 목록 조회 |

표에 덧붙일 내용이에요.

- **Bedrock 배치 추론**은 `CreateModelInvocationJob` 으로 실행해요. 입력은 S3 에 올린 JSONL 파일이고, 한 줄은 `recordId` 와 `modelInput` 으로 구성돼요. Converse 형식 잡에서는 도구 호출을 쓸 수 없고, 출력 순서도 보장되지 않아요. 특히 **배치 지원 모델 표에 Opus 5.5 와 GPT-6 계열이 없어요.** 배치로 돌리던 작업은 쓸 수 있는 모델부터 확인해야 합니다. 할인율도 모델마다 확인하세요.
- **OpenAI Batch** 는 `bedrock-runtime` 의 `/openai/v1/batches` 로 쓸 수 있지만, Files API 로 올린 파일 대신 S3 URI 를 넘기고 `X-Amzn-Bedrock-RoleArn`·`X-Amzn-Bedrock-ModelId` 헤더가 필요해요. 그래서 기존 배치 코드를 그대로 쓸 수는 없어요. 배치 지원 모델 표의 OpenAI 모델은 gpt-oss 계열뿐인데, gpt-oss 는 2편 허용목록에도 없어요. 배치를 쓰려면 배치 작업용 서비스 역할(`iam:PassRole` 범위 포함)과 입출력용 S3 버킷을 새로 만들어야 해요. 버킷에는 KMS 암호화와 보존 기간을 함께 설정하세요.
- **요청 크기 한도**를 확인하세요. Claude 를 InvokeModel 로 부를 때 요청 본문은 최대 20MB 이고, Converse 의 문서 블록은 이보다 작은 한도가 따로 있어요. Files API 대신 base64 로 넣으면 크기가 약 3분의 1 늘어나요.
- **오류 처리**도 다시 확인하세요. Bedrock 은 쿼터를 넘으면 429(`ThrottlingException`), 서비스 쪽 용량이 부족하면 503 을 돌려줘요. 기존 재시도 코드가 어떤 예외와 상태 코드를 잡는지 확인하고, 지수 백오프와 지터를 적용하세요. boto3 를 쓴다면 클라이언트의 `retries` 설정으로 재시도 방식을 정할 수 있어요.
- **프롬프트 캐싱**은 Bedrock 에서 캐시 지점을 직접 지정하는 방식으로 써야 해요. Anthropic 직접 API 의 자동 캐싱(최상위 `cache_control`)에 의존하던 코드는 캐시가 적용되는지 `usage` 로 확인하세요.

<br>

<br>



## 5. 임베딩과 RAG — 재색인이 필요한 경우

임베딩은 전환에서 비용이 가장 큰 부분일 수 있어요. 임베딩 모델을 바꾸면 기존 벡터와 새 벡터를 섞어 쓸 수 없어서, 저장된 문서를 전부 다시 임베딩해야 하기 때문이에요. [ChromaDB 1편](/coding/ChromaDB_RAG_벡터DB_제대로쓰기/)에서 설명한 "저장할 때와 검색할 때 같은 모델" 원칙입니다.

먼저 지금 어떤 임베딩을 쓰는지 확인하세요.

| 현재 임베딩 | 전환 필요 | 메모 |
|---|---|---|
| 로컬 모델<br>(SBERT 등) | 없음 | 생성 모델만<br>Bedrock 으로 |
| OpenAI<br>`text-embedding-*` | 필요 | Bedrock 에<br>OpenAI 임베딩 없음 |
| 기타 공급사 | 상황에 따라 | 같은 모델이<br>Bedrock 에 있는지 확인 |

이 블로그의 ChromaDB 시리즈처럼 로컬 임베딩 모델을 쓰고 있다면 임베딩은 그대로 두고 답변 생성 모델만 Bedrock 으로 옮기면 돼요.

OpenAI 임베딩을 쓰고 있다면 Bedrock 의 임베딩 모델로 바꿔야 해요.

| 모델 | 서울 호출 방식 | API |
|---|---|---|
| Titan Text<br>Embeddings V2 | In-Region | Invoke |
| Cohere<br>Embed v4 | Global | Invoke |

두 모델 모두 Chat Completions 경로가 아니라 InvokeModel 로 호출해요. 데이터를 서울에서만 처리해야 한다면 Titan V2 가 현실적인 선택이에요. Cohere Embed v4 는 Global 로만 호출할 수 있어서, 재색인하는 문서 전체가 국외 리전에서 처리될 수 있어요. 어느 쪽이든 [2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)의 모델 허용목록에 임베딩 모델 ARN 을 추가해야 호출할 수 있어요.

재색인은 이 순서로 진행하세요.

1. **새 컬렉션(인덱스)을 따로 만들어요.** 기존 컬렉션을 덮어쓰지 않아요.
2. 전체 문서를 새 모델로 임베딩해서 새 컬렉션에 넣어요. 비용은 대략 `청크 수 × 청크당 평균 토큰 수 × 토큰 단가` 로 먼저 계산하고, 2편의 예산 알람이 걸려 있는지 확인하세요. 문서가 많다면 배치 추론으로 돌리는 방법도 있어요. AWS 문서 기준으로 Titan V2 는 서울 리전에서 배치 추론을 지원해요.
3. 같은 질문 세트로 두 컬렉션의 검색 결과를 비교해요. 검색 품질이 떨어지면 청크 크기나 검색 개수를 조정해요.
4. 검색 경로를 새 컬렉션으로 바꾸고, 기존 컬렉션은 롤백용으로 일정 기간 남겨 둬요.

<br>

<br>



## 6. LiteLLM·LangChain·n8n 설정 바꾸기

### 6-1. LiteLLM

[모델 라우팅 편](/coding/LLM_API_비용추적_모델라우팅/)의 LiteLLM 설정은 `model` 값만 Bedrock 형식으로 바꾸면 돼요. 앱이 쓰는 논리 이름(`easy-tier`, `hard-tier`)은 그대로 둘 수 있어요.

```yaml
model_list:
  - model_name: easy-tier
    litellm_params:
      model: bedrock/converse/global.anthropic.claude-haiku-5-5
      aws_region_name: ap-northeast-2
  - model_name: hard-tier
    litellm_params:
      model: bedrock/converse/global.anthropic.claude-opus-5-5
      aws_region_name: ap-northeast-2
  - model_name: gpt-tier
    litellm_params:
      model: bedrock/global.openai.gpt-6-astra
      aws_region_name: ap-northeast-2
```

- LiteLLM 에는 mantle 엔드포인트용 공급자 설정도 있어요. 2편에서 mantle 을 차단했으므로 위처럼 `bedrock/` 접두어만 쓰세요.
- LiteLLM 의 Bedrock 경로는 `bedrock/converse/`, `bedrock/invoke/`, `bedrock/chat_completions/` 세 가지예요. 접두어를 생략하면 LiteLLM 문서 기준으로 모델에 따라 경로가 정해지고, GPT-5.6 이상 모델은 OpenAI 호환 경로(Chat Completions)를 써요. 경로를 확실히 하려면 위 예처럼 접두어를 명시하세요.
- 인증은 LiteLLM 이 실행되는 환경의 AWS 자격증명(역할 또는 환경변수)이나 `AWS_BEARER_TOKEN_BEDROCK` 을 써요. 게이트웨이가 AWS 위에서 실행된다면 실행 역할을 쓰세요.
- 전환 기간에는 같은 `model_name` 아래 직접 API 항목과 Bedrock 항목을 함께 두고, `weight` 로 비율을 조정하면서 옮길 수 있어요(8절). 아래 설정이면 `hard-tier` 요청의 약 10% 가 Bedrock 으로 가요.

```yaml
model_list:
  - model_name: hard-tier
    litellm_params:
      model: anthropic/claude-opus-4-8
      api_key: os.environ/ANTHROPIC_API_KEY
      weight: 9
  - model_name: hard-tier
    litellm_params:
      model: bedrock/converse/global.anthropic.claude-opus-5-5
      aws_region_name: ap-northeast-2
      weight: 1
```

### 6-2. LangChain

`ChatOpenAI`·`ChatAnthropic` 대신 `langchain-aws` 패키지의 `ChatBedrockConverse` 를 써요.

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model_id="global.anthropic.claude-opus-5-5",
    region_name="ap-northeast-2",
)
```

LangChain 문서는 Bedrock 의 일반 모델에 Converse 사용을 권해요. 기존 코드가 `ChatAnthropic` 을 쓰고 있다면, 같은 문서에 있는 `ChatAnthropicBedrock` 을 쓰면 바꿀 코드가 더 적어요. `ChatOpenAI` 에 Bedrock 주소를 넣는 방식은 공식 문서에서 확인하지 못했어요.

### 6-3. n8n

n8n 에는 **AWS Bedrock Chat Model** 노드가 있어요. 기존 OpenAI·Anthropic 모델 노드를 이 노드로 바꾸고, 자격증명은 AWS(IAM) 또는 AWS(Assume Role) 로 만들어요. 모델 목록에는 교차 리전 추론 프로필도 나와요. 원하는 ID 가 목록에 없으면 Expression 으로 직접 입력하면 돼요. n8n 문서에는 GPT 모델 사용 예가 없으니, GPT 를 쓸 때는 먼저 시험해 보세요.

### 6-4. Claude Code·Codex

개발 도구는 [3편](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)과 [4편](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)의 설정을 따르세요. 개인 API 키로 쓰던 설정이 남아 있으면 `ANTHROPIC_API_KEY`·`OPENAI_API_KEY` 환경변수를 지워서, 의도치 않게 직접 API 로 호출되지 않게 하세요.

<br>

<br>



## 7. 비용 추적 바꾸기

[LLM API 비용 추적 시리즈](/coding/LLM_API_비용추적_과금구조/)는 응답의 `usage` 필드, 공급사 Usage·Cost API, 키별 비용 귀속을 기반으로 했어요. Bedrock 으로 옮기면 이 세 가지가 각각 바뀌어요.

| 기존 방식 | Bedrock 에서 |
|---|---|
| 응답 `usage` 필드 | SDK 경로는 그대로,<br>Converse 는 이름이 다름 |
| 공급사 Usage·Cost API | Cost Explorer·CUR 2.0<br>(일 단위) |
| API 키별 귀속 | IAM 주체별 귀속,<br>Application inference<br>profile 태그 |
| 실시간 사용량 | CloudWatch<br>`AWS/Bedrock` |

- **`usage` 필드:** Anthropic SDK 나 OpenAI SDK 경로로 호출하면 응답 형식이 같아서 기존 수집 코드를 그대로 쓸 수 있어요. Converse 로 바꾸면 필드 이름이 `inputTokens`, `outputTokens`, `cacheReadInputTokens`, `cacheWriteInputTokens` 로 바뀌므로 수집 코드를 고쳐야 해요.
- **청구 금액 대사:** 공급사 Usage·Cost API 는 Bedrock 에서 쓸 수 없어요. 금액은 Cost Explorer 나 CUR 2.0 에서 일 단위로 확인해요. 서드파티 모델은 Marketplace 상품 이름으로 잡힌다는 점도 반영하세요([2편 7절](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)).
- **키별 귀속:** API 키 대신 호출한 IAM 주체(사용자·역할) 기준으로 귀속해요. 앱별로 나누려면 Application inference profile 에 태그를 붙여요. 다만 **`bedrock-runtime` 의 Responses API 는 Application inference profile 을 쓸 수 없어서 IAM 주체 기준으로만 귀속돼요.** 앱별 비용을 나눠야 하는 GPT 호출은 Converse 로 옮기거나 앱마다 역할을 나누세요. 단, Converse 로 옮기면 명시 캐시와 파일 입력을 쓸 수 없어요([4편 1절](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)).
- **귀속 설정:** IAM 주체별 금액을 보려면 호출자 정보를 포함하는 CUR 2.0 내보내기를 새로 만들어야 해요. 비용 할당 태그는 청구 계정에서 활성화해야 하고, 활성화하기 전의 비용에는 적용되지 않아요.
- **게이트웨이를 거치는 경우:** LiteLLM 같은 게이트웨이를 거치면 모든 호출이 게이트웨이의 역할 하나로 기록돼요. 사용자나 앱별로 나누려면 게이트웨이의 사용량 로그를 따로 쓰거나, 호출 주체별로 역할을 나눠야 해요.

<br>

<br>



## 8. 단계적 전환과 롤백

한 번에 전부 바꾸지 말고 아래 순서로 옮기세요.

| 단계 | 할 일 | 넘어가는 기준 |
|---|---|---|
| ① 목록·분류 | 2절 표 작성,<br>4절 대조 | 모든 호출 지점의<br>대응 방식 확정 |
| ② 준비 | 1~2편 계정 설정,<br>쿼터 증설 요청 | 필요한 모델의<br>`AVAILABLE` 확인 |
| ③ 비교 평가 | 같은 입력을 두 경로로<br>보내 결과 비교 | 품질·지연·비용이<br>허용 범위 안 |
| ④ 일부 전환 | 트래픽 일부를<br>Bedrock 으로 | 오류율·스로틀이<br>기준 이하 |
| ⑤ 전체 전환 | 나머지 트래픽 이동 | 일정 기간 안정 |
| ⑥ 정리 | 직접 API 키 회수,<br>롤백 코드 제거 | 기존 청구 종료 확인 |

- **③ 비교 평가:** 모델 버전이 바뀌는 경우가 많아서 이 단계가 가장 중요해요. 실제 요청에서 뽑은 질문 세트(골든셋)를 두 경로로 보내고 답변 품질, 응답 시간, 토큰 사용량을 비교하세요. 토큰 수는 모델마다 세는 방식이 달라서 같은 프롬프트라도 비용이 달라질 수 있어요.
- **② 쿼터:** 직접 API 에서 쓰던 처리량이 Bedrock 기본 쿼터보다 크면 ④ 단계에서 바로 `ThrottlingException` 이 나요. 전환 전에 Service Quotas 에서 증설을 요청하세요. GPT-6 Astra 는 출력 토큰을 10배로 계산한다는 점도 반영하세요([4편 5-2](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)).
- **④ 일부 전환:** LiteLLM 을 쓴다면 같은 `model_name` 아래 두 항목의 가중치로, 앱에서 직접 호출한다면 3-3의 `BEDROCK_RATIO` 로 비율을 조정해요.
- **롤백:** ⑥ 단계 전까지는 직접 API 키와 롤백 스위치를 남겨 두세요. 문제가 생기면 `LLM_PROVIDER=direct` 나 LiteLLM 가중치로 되돌릴 수 있어요.
- **⑥ 정리:** 전환이 끝나면 직접 API 키를 공급사 콘솔에서 폐기하고, 비밀 저장소와 CI 에서도 지우세요. 쓰지 않는 키도 유출되면 위험해요.

<br>

<br>



## 9. 옮기면서 판단이 갈린 부분

**① 전환할 때 Converse 로 통일할까**

4편에서는 새로 만드는 코드의 기본값으로 Converse 를 권했어요. 하지만 이미 동작하는 코드를 옮길 때 호출 방식까지 바꾸면 변경 범위가 커지고, 문제가 생겼을 때 원인을 가리기 어려워요. 그래서 전환 단계에서는 기존 SDK 를 그대로 쓰고 주소·인증·모델 ID 만 바꾸는 것을 권해요. Converse 통일은 전환이 안정된 뒤 별도 작업으로 진행하는 편이 좋아요. 단, Guardrails 나 앱별 비용 태깅이 필요한 GPT 호출은 처음부터 Converse 로 옮기는 편이 나아요.

**② 임베딩도 함께 옮길까**

공급사를 하나로 모으려면 임베딩도 Bedrock 으로 옮기는 편이 깔끔해요. 하지만 임베딩을 바꾸면 전체 재색인이 필요하고, 검색 품질도 다시 검증해야 해요. 로컬 임베딩 모델을 쓰고 있다면 옮길 이유가 거의 없어요. OpenAI 임베딩을 쓰는 경우에도 생성 모델 전환과 임베딩 전환은 나눠서 진행하는 것을 권해요. 두 가지를 동시에 바꾸면 품질이 달라졌을 때 어느 쪽 때문인지 구분할 수 없기 때문이에요.

**③ 직접 API 키를 언제 회수할까**

보안 관점에서는 쓰지 않는 키를 바로 폐기하는 것이 원칙이에요. 반면 운영 관점에서는 롤백할 수단이 있어야 해요. 그래서 전체 전환 후 일정 기간 안정적으로 동작하는지 확인한 다음 폐기하기를 권해요. 그 기간에는 공급사 콘솔에서 키 사용량을 지켜보세요. 사용량이 0 이 아니라면 아직 옮기지 않은 호출 지점이 있다는 뜻이니 목록을 다시 확인하세요.

<br>

<br>



## 10. 동작 확인 체크리스트

- [ ] 2절의 다섯 곳을 모두 확인해 호출 지점 목록을 만들었다
- [ ] 각 호출 지점의 대응 방식(그대로·대안·직접 API 유지)이 정해졌다
- [ ] 4절 표에 해당하는 기능을 쓰는 호출의 대안이 준비됐다
- [ ] 임베딩을 바꾼다면 새 컬렉션으로 재색인하고 검색 결과를 비교했다
- [ ] 비교 평가에서 품질·지연·비용이 허용 범위 안이다
- [ ] Bedrock 쿼터가 기존 처리량을 감당한다
- [ ] 비용 수집 코드가 Bedrock 의 사용량 필드와 Cost Explorer 기준으로 바뀌었다
- [ ] 롤백 스위치가 동작하는지 확인했다
- [ ] 전환 완료 후 직접 API 키를 폐기했다

이 시리즈는 여기까지예요.....  
Bedrock 은 지원 모델과 리전이 몇 주 단위로 바뀌어요. 새 모델을 서울 In-Region 으로 호출할 수 있게 되거나, 지금 표에 없는 OpenAI API 가 추가되면 이 시리즈의 표를 갱신할게요.

---

**← 이전 글:** [(4/5) OpenAI GPT 붙이기 — openai SDK·단일 함수·Codex 와 비용 함정](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)  
**처음으로:** [(1/5) 서울 리전에서 먼저 알아둘 것](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)
