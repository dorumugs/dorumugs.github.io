---
layout: single
title:  "(4/4) Bedrock 에 OpenAI GPT 붙이기 — openai SDK·단일 함수·Codex 와 비용 함정"
date: 2026-10-25 09:30:00 +0900
categories: coding
tag: [Bedrock, AWS, OpenAI, GPT, GPT-6, ChatGPT, Codex, openai-sdk, Converse, 비용, 쿼터, 서울리전]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-gpt/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-gpt/header.svg
description: "openai SDK 코드를 그대로 두고 base URL 과 단기 토큰만 바꿔 서울에서 global.openai.gpt-6-astra 를 부르는 법, Converse 함수 하나로 GPT 와 Claude 를 같이 부르는 법, Codex 가 서울이 아니라 미국 리전으로 가는 이유를 정리했어요. 입력이 272K 토큰을 넘으면 요청 전체 단가가 바뀌는 문턱과 출력 토큰을 10배로 세는 쿼터 계산, 첫날 관측 설정까지 마무리해요."
series: bedrock-gpt-claude-setup
series_order: 4
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="4" %}

## Summary

마지막 편이에요. [3편](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)에서 Claude 를 붙였으니 이번엔 같은 계정에 OpenAI GPT 를 붙여요. 계정·권한은 [2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)에서 이미 GPT-6 Astra 까지 열어뒀어요.

GPT 쪽은 반가운 소식부터 있어요. openai SDK 를 그대로 쓸 수 있어서 주소와 키만 바꾸면 돼요. 대신 조심할 게 두 가지 있어요. Codex 는 지금 서울이 아니라 **미국 리전**으로 가요. 그리고 GPT-6 Astra 는 입력이 272K 토큰을 넘는 순간 **요청 전체 단가가 바뀌어요.** 이걸 모르고 쓰면 데이터 경계도 청구서도 생각과 다르게 나와요.

> 💡 **이 글에서 다루는 것**
> - 경로 고르기 — openai SDK 네이티브 vs Converse
> - 첫 호출 A — openai SDK + 단기 토큰 (Responses·Chat Completions)
> - 첫 호출 B — GPT 와 Claude 를 함수 하나로
> - Codex 붙이기 — 왜 미국 리전인가, 예외는 어떻게 여나
> - 비용 함정 — 272K 문턱, 10배 쿼터 차감, Ultrafast
> - 첫날 관측 — 메트릭, CloudTrail, 호출 로그를 켤까 말까

<br>

<br>



## 1. 경로 고르기

`bedrock-runtime` 에서 GPT-6 Astra 를 부르는 길은 두 갈래예요. 차이가 생각보다 커요.

| 항목 | openai SDK<br>(Responses·Chat) | Converse<br>(boto3) |
|---|---|---|
| 코드 이전 | base URL 만 교체 | 새로 작성 |
| 인증 | 단기 토큰 | SSO 그대로 |
| Guardrails | ✗ | ✓ |
| 비용 태깅 | ✗ | ✓ |
| 명시 캐시 | ✓ | ✗ (암묵만) |
| 파일 입력 | ✓ `input_file` | ✗ |

AWS 모델 카드 기준으로, GPT-6 Astra 의 Guardrails 는 **Converse 에서만**, Application inference profile(비용 태깅)은 **Invoke·Converse 에서만** 돼요. 반대로 캐시 지점을 직접 찍는 명시 캐시와 문서 파일 입력은 Converse 에서 안 돼요.

그래서 이렇게 갈랐어요.

- **기본은 Converse.** Claude 와 같은 경로라 Guardrails·비용 태깅·로그가 한 곳에 모여요(3절).
- **이미 openai SDK 로 짠 코드가 있거나, 명시 캐시·파일 입력이 필요한 호출**만 openai SDK 로 둬요(2절).

<br>

<br>



## 2. 첫 호출 A — openai SDK + 단기 토큰

```shell
pip install -U openai aws-bedrock-token-generator
```

```python
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

REGION = "ap-northeast-2"
BASE = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1"
MODEL = "global.openai.gpt-6-astra"


def gpt() -> OpenAI:
    token = provide_token(region=REGION)
    return OpenAI(base_url=BASE, api_key=token)


r = gpt().responses.create(model=MODEL, input="안녕!")
print(r.output_text)
print(r.usage)
```

Chat Completions 로 짠 코드라면 이렇게 바뀌어요. 주소와 모델 ID 는 같아요.

```python
c = gpt().chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "안녕!"}],
)
print(c.choices[0].message.content)
```

- **`base_url` 이 `/openai/v1` 로 끝나요.** `/v1` 만 쓰면 안 돼요.
- **`bedrock-runtime` 에는 GPT-6 Astra 의 In-Region 이 없어요.** 어느 리전에서 부르든 `us.` 아니면 `global.` 프로필이어야 하고, 서울에서는 `global.` 만 돼요.
- **AWS 예제는 장기 키를 `OPENAI_API_KEY` 에 넣어요.** 그대로 따라 하지 마세요. 2편 정책이 장기 키를 막고 있고, 막혀 있지 않더라도 무기한 키를 환경변수에 두는 건 1순위 사고예요. 위처럼 SSO 자격증명에서 단기 토큰을 만들어 넣으면 돼요.
- 2편 5절에서 미뤄둔 확인이 여기예요. 이 호출이 `AccessDenied` 를 내면 메시지에 필요한 IAM 액션 이름이 찍혀요. 그 이름을 허용 정책에 더하면 돼요.

<br>

<br>



## 3. 첫 호출 B — GPT 와 Claude 를 함수 하나로

시리즈 기본값인 Converse 예제예요. 모델 ID 만 바꾸면 같은 함수가 두 회사 모델을 다 불러요. (`str | None` 표기 때문에 Python 3.10 이상이 필요해요.)

```python
import boto3
from botocore.config import Config

REGION = "ap-northeast-2"
MODELS = {
    "claude": "global.anthropic.claude-opus-5-5",
    "gpt": "global.openai.gpt-6-astra",
}

rt = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"mode": "adaptive"}),
)


def ask(model: str, prompt: str, system: str | None = None) -> str:
    kwargs = {
        "modelId": MODELS[model],
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 4096},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = rt.converse(**kwargs)
    blocks = resp["output"]["message"]["content"]
    u = resp["usage"]
    print(model, u["inputTokens"], u["outputTokens"], resp["stopReason"])
    return "".join(b["text"] for b in blocks if "text" in b)


for name in MODELS:
    print(ask(name, "한 문장으로 자기소개 해줘."))
```

- **텍스트 블록만 고르는 줄이 두 모델 모두에 필요해요.** Claude 는 생각 블록, GPT 는 추론 요약(reasoning summary) 블록이 섞일 수 있어요.
- **사용량을 찍는 줄을 지우지 마세요.** 5절의 비용 함정을 첫날 몸으로 느끼게 해주는 줄이에요.
- 공식 문서에 적힌 GPT 쪽 Converse 제약이 몇 가지 있어요. `document` 블록이나 `toolResult` 안의 이미지를 넣으면 `ValidationException` 이에요(파일은 2절의 Responses `input_file` 로). JSON 스키마 출력을 Converse 로 받으려면 `outputConfig` 에 스키마를 넣고 `additionalModelRequestFields.text.format.strict` 를 `true` 로 함께 보내야 해요.

이 함수가 있으면 "같은 질문을 두 모델에 던져 비교" 같은 일이 한 줄이 돼요. 그리고 Guardrails 를 붙이는 날에도 이 함수 한 곳에 `guardrailConfig` 만 더하면 두 모델에 같이 걸려요. 나중에 명시 캐시나 파일 입력처럼 openai SDK 로만 되는 호출이 생기면, 호출하는 쪽은 그대로 두고 `ask()` **안에서** 2절 경로로 분기하면 돼요.

<br>

<br>



## 4. Codex 붙이기 — 서울이 아니라 미국이에요

Codex(CLI·데스크톱 앱·IDE 확장)도 Bedrock 으로 붙일 수 있어요. 2026년 6월 GPT-5.5 와 함께 정식 지원이 나왔어요. 다만 이 시리즈에서 정한 기본값을 벗어나는 건 Codex 하나뿐이에요.

AWS 출시 블로그의 설정 예시와 2차 자료 기준으로, Codex 의 `amazon-bedrock` provider 는 **mantle 엔드포인트의 Responses API** 를 불러요(쓰는 버전에서 직접 확인하세요). 그리고 GPT-6 Astra 의 mantle 은 `us-east-1`·`us-west-2` 에만 있어요. 즉 **Codex 를 Bedrock 으로 쓰면 요청이 미국 리전에서 처리돼요.** 서울 출발 Global 과는 데이터 경계가 다른 결정이에요.

| 비교 | 앱(1~3절) | Codex |
|---|---|---|
| 엔드포인트 | bedrock-runtime | bedrock-mantle |
| 호출 리전 | 서울 | us-east-1 |
| 처리 위치 | 전 세계 중 한 곳 | 미국 |
| 호출 로그 | 켜면 남음 | **불가** |
| 2편 정책 | 허용 | **차단됨** |

### 4-1. 열기 전에 결정할 것

열 때는 이 순서를 권해요.

1. **"코드가 미국에서 처리돼도 되는가"** 를 먼저 결정해요. 사내 코드라면 1편의 데이터 경계 판단과 같은 문서에 한 줄 적어둬요.
2. Codex 를 쓰는 사람만 따로 묶은 SSO 권한 세트(예: `CodexUsers`)를 만들어요.
3. 2편 차단 정책의 `DenyMantleUntilNeeded` 문장에 그 역할만 예외로 두는 조건을 더해요. 2편 6절의 리전 제한 SCP 를 쓰는 조직이라면 거기에도 같은 예외가 필요해요(Codex 는 `us-east-1` 을 부르니까요).

```json
{
  "Sid": "DenyMantleUntilNeeded",
  "Effect": "Deny",
  "Action": "bedrock-mantle:*",
  "Resource": "*",
  "Condition": {
    "ArnNotLike": {
      "aws:PrincipalArn":
        "arn:aws:iam::111122223333:role/aws-reserved/sso.amazonaws.com/*AWSReservedSSO_CodexUsers_*"
    }
  }
}
```

SSO 권한 세트로 만든 역할은 `aws-reserved/sso.amazonaws.com/` 경로 아래에 `AWSReservedSSO_<권한세트>_<난수>` 이름으로 생겨서 이렇게 맞췄어요. `*CodexUsers*` 처럼 느슨하게 쓰면, 역할을 만들 수 있는 사람이 이름만 비슷하게 지어서 예외를 탈 수 있어요.

그 역할의 허용 정책은 이렇게 좁혀요. 2편의 모델 허용목록은 `bedrock:InvokeModel` 계열만 막아서 **mantle 에는 안 걸려요.** 그래서 최소한 리전이라도 `us-east-1` 로 묶어 둬요.

```json
{
  "Sid": "CodexMantleUsEast1",
  "Effect": "Allow",
  "Action": [
    "bedrock-mantle:CreateInference",
    "bedrock-mantle:CallWithBearerToken"
  ],
  "Resource": "*",
  "Condition": {
    "StringEquals": {
      "aws:RequestedRegion": "us-east-1"
    }
  }
}
```

mantle 액션을 모델 단위 리소스로 좁힐 수 있는지는 이번에 확인하지 못했어요. 확인되면 `Resource` 를 GPT-6 Astra 로 좁히는 게 맞아요.

### 4-2. 설정

`~/.codex/config.toml` 이에요. AWS 출시 블로그의 GPT-5.5 예제를 GPT-6 Astra 와 mantle 리전에 맞춘 거예요.

```toml
model = "openai.gpt-6-astra"
model_provider = "amazon-bedrock"

[model_providers.amazon-bedrock.aws]
region = "us-east-1"
```

- 모델 ID 에 `global.` 이 없어요. mantle 은 In-Region 이라 접두어 없는 ID 를 써요.
- 인증은 **`AWS_BEARER_TOKEN_BEDROCK` 이 있으면 그걸 먼저**, 없으면 AWS SDK 자격증명 체인을 써요. 단기 토큰을 쓰려면 **토큰을 만든 리전에서만 유효**하다는 점에 주의해서 `us-east-1` 로 만들어요.
- **단기 토큰은 지금 역할 자격증명이 끝나면 같이 끝나요**(SSO 권한 세트 기본 1시간). 한 번 만든 토큰으로 오래 쓰면 한 시간쯤 뒤 401·403 이 나요. 그땐 아래 명령을 다시 실행해 갈아 끼우세요.
- 토큰은 **셸 전체에 `export` 하지 말고 Codex 실행 한 번에만** 붙여요. 3편 2절에서 봤듯이 이 변수가 셸에 남아 있으면 같은 셸의 boto3·Claude Code 가 SSO 대신 이 `us-east-1` 토큰을 써서 서울 호출이 깨져요.

```shell
AWS_BEARER_TOKEN_BEDROCK="$(python3 -c \
  'from aws_bedrock_token_generator import provide_token as p; print(p(region="us-east-1"))')" \
  codex
```

- 데스크톱 앱이나 VS Code 확장은 셸 환경변수를 못 읽을 수 있어요. 그땐 `~/.codex/.env` 에 넣고 앱을 재시작해요. 이 파일은 `chmod 600` 으로 두고, 토큰이 만료되면 다시 만들어 넣어요.

설정 후 Codex 안에서 `/status` 로 provider 와 모델을 확인해요.

> ⚠️ 이 절은 AWS 출시 블로그(GPT-5.5 기준)와 모델 카드를 바탕으로 GPT-6 Astra 에 맞춰 옮긴 거예요. OpenAI 도움말 문서는 이번에 직접 열람하지 못했어요. 2차 자료에서는 mantle 이 함수·MCP 도구만 받아서 `web_search = "disabled"` 가 필요했다는 보고, `~/.aws` 프로필만으로는 실패해서 `aws configure export-credentials --format env` 로 우회했다는 보고가 있었어요. 막히면 이 두 가지부터 보세요. 앞으로 Codex 가 `bedrock-runtime` 의 Global 경로를 지원하게 되면 이 예외는 없애는 게 맞아요.

<br>

<br>



## 5. 비용 함정 세 가지

GPT-6 Astra 의 Bedrock 가격(미국 달러, 100만 토큰당, Global 기준)부터 볼게요.

| 구간 | 입력 | 캐시 읽기 | 출력 |
|---|---|---|---|
| 입력 272K 이하 | $10 | $1 | $50 |
| 입력 272K 초과 | $20 | $2 | $75 |

In-Region·Geo 는 여기에 10% 를 더한 값이에요(272K 이하 기준 $11 · $1.1 · $55).

### 5-1. 272K 문턱 — 넘으면 "전부" 비싸져요

제일 크게 데일 수 있는 부분이에요. 272K 를 넘는 부분만 비싸지는 게 아니라 **그 요청 전체**가 장기 컨텍스트 단가로 계산돼요. 같은 출력(5K 토큰)으로 입력만 늘려볼게요.

| 입력 | 입력 비용 | 출력 비용 | 합계 |
|---|---|---|---|
| 200K | 0.2 × $10<br>= $2.00 | 0.005 × $50<br>= $0.25 | **$2.25** |
| 300K | 0.3 × $20<br>= $6.00 | 0.005 × $75<br>= $0.375 | **$6.375** |

읽는 법은 간단해요. 200K 토큰은 100만 토큰의 0.2 배라서 단가 $10 에 0.2 를 곱해요. 출력 5K 는 0.005 배예요.

입력은 1.5배 늘었는데 비용은 $6.375 ÷ $2.25 ≈ **2.83배**가 됐어요. 문턱을 넘으면 입력 단가는 2배, 출력 단가는 1.5배로 한꺼번에 뛰니까요.

막는 방법 중에 제일 싼 건 앱에서 미리 자르는 거예요. 대화 이력이나 첨부 문서를 붙일 때 입력이 270K 언저리를 넘지 않게 끊어 주면 돼요. 방금 응답의 `usage` 로 입력 토큰 수를 보면서, 다음 요청에 붙일 분량을 줄이면 돼요. (Bedrock 에서 GPT-6 Astra 는 토큰 수를 미리 세는 API 를 지원하지 않아서, 응답의 사용량을 기준으로 삼는 쪽이 현실적이에요.)

### 5-2. 10배 쿼터 차감 — 돈이 아니라 속도가 막혀요

`bedrock-runtime` 에서 GPT-6 Astra 의 분당 토큰 한도(TPM)는 **출력 토큰 1개를 10개로 세요.** 요금이 10배가 되는 건 아니고, 출력 쪽이 한도를 10배 빨리 갉아먹어요.

| 요청 | 실제 토큰 | 한도에서 차감 |
|---|---|---|
| 입력 200K · 출력 5K | 205K | 200K + 5K × 10<br>= **250K** |
| 입력 10K · 출력 20K | 30K | 10K + 20K × 10<br>= **210K** |

분당 한도가 가령 200만 토큰이라고 해볼게요. 이건 계산을 위한 **가정값**이에요. 3편에서 말한 Claude 의 기본 입력 한도(2M)와는 별개고, GPT-6 Astra 의 실제 기본값은 계정·리전마다 달라서(새 계정은 더 낮게 잡히기도 해요) Service Quotas 콘솔에서 확인하세요. 첫 줄 같은 요청은 200만 ÷ 25만 = **분당 8건**에서 막혀요. 10배 차감이 없었다면 200만 ÷ 20.5만 ≈ 9.75, 즉 9건이었을 거예요.

차이가 더 크게 나는 건 두 번째 줄 같은 **출력이 긴 작업**이에요. 실제로는 3만 토큰짜리 요청이 한도에서는 21만을 써요. 요약보다 생성, 특히 코드 생성이 많은 팀이 `ThrottlingException` 을 먼저 만나요. `maxTokens` 를 필요 이상으로 크게 잡지 않는 것도 도움이 돼요.

### 5-3. Ultrafast — 6배 요금

GPT-6 Astra 에는 Ultrafast 라는 속도 등급이 있어요(`service_tier="ultrafast"`). 응답이 빨라지는 대신 **요금이 표준의 6배**예요. 272K 이하 Global 기준으로 입력 $60, 출력 $300 이에요. 기본값으로 켜둘 이유는 없고, 켜려면 비용 이유를 한 줄 남기는 걸 권해요.

<br>

<br>



## 6. 첫날 관측 — 무엇을 켜고, 무엇을 미루나

| 대상 | 첫날 | 이유 |
|---|---|---|
| CloudWatch 메트릭 | ✓ | 설정 없이 쌓임 |
| CloudTrail | ✓ | 실제 처리 리전 확인 |
| 예산 알람 | ✓ | 2편 7절 |
| 호출 로그<br>(원문) | 결정 후 | 새 민감정보<br>저장소가 생김 |

- **CloudWatch `AWS/Bedrock`** 에서 모델 ID 별로 `Invocations`, `InputTokenCount`, `OutputTokenCount`, `InvocationThrottles`, `InvocationLatency` 를 봐요. 알람은 두 개면 충분해요. 스로틀 발생, 그리고 토큰량 급증.
- **CloudTrail** 은 1편에서 말한 증거예요. 이벤트의 `additionalEventData.inferenceRegion` 에 Global 요청이 실제로 처리된 리전이 찍혀요. 보안 검토 때 이걸 보여주면 대화가 빨라져요.

### 호출 로그(Model invocation logging)를 켤까

저는 켜는 쪽이에요. 다만 민감정보 저장소를 하나 새로 만든다고 생각하고 켜야 해요. 이 로그는 요청과 응답 **본문 전체**를 남기거든요. 켜는 순간 프롬프트에 들어간 건 전부 버킷 하나에 쌓여요.

| 설정 | 권장 |
|---|---|
| 목적지 | S3 만<br>(CloudWatch 로 원문 X) |
| 암호화 | 전용 KMS 키 |
| 접근 | 보안 역할만 읽기 |
| 보존 | 30~90일 수명주기 |
| 모달리티 | 텍스트만 |

Anthropic 은 오남용 조사를 위해 최소 30일 보존을 권해요. 그럴 여유가 아직 없다면 첫날은 메트릭과 CloudTrail 만 보고, 로그는 이 표대로 설계한 다음에 켜세요. 그리고 AWS 호출 로그 문서 기준으로 **mantle 호출은 이 로그에 남지 않아요**(CloudTrail 에 어떻게 남는지는 계정에서 한 번 확인해 보세요). 4절의 Codex 예외를 "결정 후에만" 연 이유가 여기에도 있어요.

<br>

<br>



## 7. 세팅하면서 고민한 것들

**① Converse 로 통일할까**

코드를 옮기는 비용만 따지면 openai SDK 를 그대로 두는 게 제일 싸요. 기존 코드가 다 그 모양이니까요. 그런데 GPT 의 Guardrails 와 비용 태깅은 Converse 에서만 돼요. 경로가 둘이면 통제할 곳도 둘로 늘어나요. 그래서 저는 Converse 함수 하나를 기본으로 두고, 명시 캐시나 파일 입력처럼 Converse 에 없는 기능이 필요한 호출만 openai SDK 로 남겼어요. 함수 이름은 그대로 두고 안에서만 갈라지게 했어요.

**② Codex 를 시리즈에 넣을까**

"Bedrock 에서 ChatGPT" 를 찾는 분들 중에는 사실 Codex 를 쓰고 싶은 경우가 꽤 많아요. 반면 보안 쪽에서 보면 걸리는 게 많아요. mantle 이라 정책 예외가 필요하고, 호출 로그도 안 남고, 처리도 미국에서 돼요. 고민하다가 넣기로 했어요. 대신 예외라는 걸 숨기지 않으려고 서울 Global 과 비교표를 두고, 열기 전에 결정할 것과 예외 역할을 같이 실었어요.

**③ 비용 계산을 본문에 둘까**

숫자 표가 많으면 읽다가 지쳐요. 그래도 272K 문턱은 "조금 비싸진다" 가 아니라 "같은 요청이 2.83배" 인 일이라, 말로만 하면 와닿지가 않더라고요. 그래서 계산은 두 줄짜리 표로만 두고, 바로 아래에 읽는 법을 한 줄 붙였어요.

<br>

<br>



## 8. 시리즈 마무리 — 전체 체크리스트

네 편을 다 따라왔다면 아래가 전부 체크돼야 해요.

**계정 (2편)**
- [ ] 전용 계정, 루트 MFA, SSO, 결제수단
- [ ] 차단 정책 — 장기 키(runtime·mantle), 장기 키 생성, mantle, 모델 허용목록
- [ ] Claude 사용 사례 양식 `AVAILABLE`
- [ ] 계정 전체 예산 + 예측 알람

**Claude (3편)**
- [ ] Converse 와 Anthropic SDK 둘 다 `global.` ID 로 응답
- [ ] Claude Code `/status` 에 Bedrock · ap-northeast-2, 세 모델 고정

**GPT (이번 편)**
- [ ] openai SDK 가 단기 토큰으로 `global.openai.gpt-6-astra` 응답
- [ ] `ask()` 하나로 두 모델 응답, 사용량 출력 확인
- [ ] (쓴다면) Codex 예외 역할과 데이터 경계 결정이 문서로 남음
- [ ] 입력 270K 가드, 스로틀 알람

처음에 던졌던 질문으로 돌아가 볼게요. "Bedrock 에서 ChatGPT 써도 돼요?" 이제는 이렇게 답할 수 있어요. ChatGPT 앱은 아니지만 GPT 모델은 돼요. 서울에서 쓰려면 `global.` 로 부르고, 그게 국외 처리라는 사실을 결정 문서에 남기세요. 키는 단기로 쓰고, 예산은 계정 전체에 걸어 두고요.

저도 아직 답을 못 낸 게 하나 있어요. 규제 데이터는 서울 In-Region 모델이 나올 때까지 기다리는 게 맞다는 생각도 들고, 그때까지는 마스킹 게이트로 버티는 게 현실적이라는 생각도 들어요. 이건 조직마다 답이 다를 거예요. 어느 쪽을 고르든 **그 결정을 기록으로 남겨 두는 것**만은 꼭 하세요.

이 시리즈는 여기까지예요.....  
Bedrock 은 몇 주 단위로 모델과 리전 표가 바뀌어요. 새 모델을 서울 In-Region 으로 부를 수 있게 되거나 Codex 가 runtime 경로를 지원하게 되면 이 시리즈의 표를 갱신할게요.

---

**← 이전 글:** [(3/4) Claude 붙이기 — Converse·Anthropic SDK·Claude Code](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)  
**처음으로:** [(1/4) 서울 리전에서 먼저 알아둘 것](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)
