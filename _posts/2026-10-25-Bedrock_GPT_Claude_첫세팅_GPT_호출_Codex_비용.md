---
layout: single
title:  "(4/5) Bedrock 에 OpenAI GPT 붙이기 — openai SDK·단일 함수·Codex 와 비용 함정"
date: 2026-10-25 09:30:00 +0900
categories: coding
tag: [Bedrock, AWS, OpenAI, GPT, GPT-6, ChatGPT, Codex, openai-sdk, Converse, 비용, 쿼터, 서울리전]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-gpt/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-gpt/header.svg
description: "openai SDK 코드에서 base URL 과 키(단기 토큰)만 바꿔 서울 리전에서 global.openai.gpt-6-astra 를 호출하는 방법, Converse 함수 하나로 GPT 와 Claude 를 함께 호출하는 방법, Codex 가 서울이 아닌 미국 리전을 호출하는 이유를 정리했어요. 입력이 272K 토큰을 넘으면 요청 전체에 높은 단가가 적용되는 기준, 출력 토큰을 10배로 계산하는 쿼터, 첫날 켜 둘 관측 설정도 다룹니다."
series: bedrock-gpt-claude-setup
series_order: 4
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="4" %}

## Summary

[3편](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)에서는 Claude 를 연결했고, 이번 편에서는 같은 계정에서 OpenAI GPT 를 호출해요. GPT-6 Astra 호출 권한은 [2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)에서 계정·권한을 설정할 때 이미 허용해 두었어요.

GPT 는 openai SDK 를 그대로 쓸 수 있어요. 기존 코드에서 주소와 키만 바꾸면 됩니다. 다만 주의할 점이 두 가지 있어요. 첫째, Codex 는 현재 서울이 아니라 **미국 리전**을 호출해요. 둘째, GPT-6 Astra 는 입력이 272K 토큰을 넘으면 **그 요청 전체에 더 높은 단가가 적용돼요.** 이 두 가지를 모르고 쓰면 데이터가 처리되는 위치와 비용이 예상과 달라집니다.

> 💡 **이 글에서 다루는 것**
> - 호출 방식 고르기 — openai SDK vs Converse
> - 첫 호출 A — openai SDK + 단기 토큰 (Responses·Chat Completions)
> - 첫 호출 B — GPT 와 Claude 를 함수 하나로
> - Codex 연결 — 미국 리전을 쓰는 이유와 예외 정책
> - 비용 주의점 — 272K 기준, 출력 10배 쿼터 차감, Ultrafast
> - 첫날 관측 — 메트릭, CloudTrail, 호출 로그 활성화 여부

<br>

<br>



## 1. 호출 방식 고르기

`bedrock-runtime` 에서 GPT-6 Astra 를 호출하는 방식은 두 가지예요. 두 방식은 지원하는 기능이 꽤 달라요.

| 항목 | openai SDK<br>(Responses·Chat) | Converse<br>(boto3) |
|---|---|---|
| 코드 이전 | base URL 만 교체 | 새로 작성 |
| 인증 | 단기 토큰 | SSO 그대로 |
| Guardrails | ✗ | ✓ |
| 비용 태깅 | ✗ | ✓ |
| 명시 캐시 | ✓ | ✗ (암묵만) |
| 파일 입력 | ✓ `input_file` | ✗ |

AWS 모델 카드에 따르면 GPT-6 Astra 의 Guardrails 는 **Converse 에서만** 지원돼요. 비용 태깅에 쓰는 Application inference profile 은 **Invoke·Converse 에서만** 지원됩니다. 반대로 캐시 지점을 직접 지정하는 명시 캐시와 문서 파일 입력은 Converse 에서 지원되지 않아요.

그래서 다음 기준으로 나눴어요.

- **기본은 Converse 예요.** Claude 와 같은 API 를 쓰므로 Guardrails·비용 태깅·로그를 한 곳에서 관리할 수 있어요(3절).
- **이미 openai SDK 로 작성한 코드가 있거나 명시 캐시·파일 입력이 필요한 호출**에만 openai SDK 를 써요(2절).

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

Chat Completions 로 작성한 코드라면 다음과 같아요. 주소와 모델 ID 는 Responses 와 같습니다.

```python
c = gpt().chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "안녕!"}],
)
print(c.choices[0].message.content)
```

- **`base_url` 은 `/openai/v1` 로 끝나야 해요.** `/v1` 로 끝내면 안 됩니다.
- **`bedrock-runtime` 에서는 GPT-6 Astra 를 In-Region 으로 호출할 수 없어요.** 어느 리전에서 호출하든 `us.` 또는 `global.` 추론 프로필을 써야 해요. 서울에서는 `global.` 만 쓸 수 있습니다.
- **AWS 예제는 장기 키를 `OPENAI_API_KEY` 에 넣어요.** 이 방식은 권하지 않아요. 2편에서 만든 정책이 장기 키를 차단하고 있고, 차단하지 않더라도 만료 기한이 없는 키를 환경변수에 두면 유출 위험이 커요. 위 코드처럼 SSO 자격증명으로 단기 토큰을 만들어 쓰세요.
- 2편 5절에서 미뤄 둔 권한 확인을 여기서 해요. 이 호출이 `AccessDenied` 로 실패하면 오류 메시지에 필요한 IAM 액션 이름이 나와요. 그 액션을 허용 정책에 추가하세요.

<br>

<br>



## 3. 첫 호출 B — GPT 와 Claude 를 함수 하나로

이 시리즈의 기본 방식인 Converse 예제예요. 모델 ID 만 바꾸면 같은 함수로 Claude 와 GPT 를 모두 호출할 수 있어요. `str | None` 타입 표기를 쓰므로 Python 3.10 이상이 필요합니다.

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

- **텍스트 블록만 골라내는 마지막 줄은 두 모델 모두에 필요해요.** 응답에 Claude 는 생각 블록이, GPT 는 추론 요약(reasoning summary) 블록이 섞일 수 있기 때문이에요.
- **사용량을 출력하는 `print` 줄은 지우지 마세요.** 5절에서 다룰 단가 기준과 쿼터 차감을 첫날부터 숫자로 확인할 수 있어요.
- 공식 문서에는 GPT 를 Converse 로 호출할 때의 제약도 적혀 있어요. `document` 블록을 넣거나 `toolResult` 안에 이미지를 넣으면 `ValidationException` 이 나요. 파일은 2절의 Responses `input_file` 로 넣어야 해요. JSON 스키마에 맞춘 출력을 Converse 로 받으려면 `outputConfig` 에 스키마를 넣고, `additionalModelRequestFields.text.format.strict` 를 `true` 로 함께 보내야 합니다.

이 함수를 쓰면 같은 질문을 두 모델에 보내 결과를 비교하기 쉬워요. Guardrails 를 적용할 때도 이 함수에 `guardrailConfig` 만 추가하면 두 모델에 함께 적용돼요. 나중에 명시 캐시나 파일 입력처럼 openai SDK 로만 할 수 있는 호출이 필요해지면, 호출하는 코드는 그대로 두고 `ask()` 안에서 2절 방식으로 분기하면 됩니다.

<br>

<br>



## 4. Codex 연결 — 서울이 아니라 미국 리전을 써요

Codex(CLI·데스크톱 앱·IDE 확장)도 Bedrock 에 연결할 수 있어요. 2026년 6월 GPT-5.5 출시와 함께 정식 지원이 시작됐어요. 이 시리즈에서 정한 기본 설정을 따르지 않는 것은 Codex 하나뿐이에요.

AWS 출시 블로그의 설정 예시와 2차 자료에 따르면, Codex 의 `amazon-bedrock` provider 는 **mantle 엔드포인트의 Responses API** 를 호출해요. 사용하는 Codex 버전에서 직접 확인하세요. GPT-6 Astra 를 mantle 로 호출할 수 있는 리전은 `us-east-1`·`us-west-2` 뿐이에요. 따라서 **Codex 를 Bedrock 으로 쓰면 요청이 미국 리전에서 처리돼요.** 서울에서 Global 로 호출할 때와는 데이터 경계를 따로 판단해야 합니다.

| 비교 | 앱(1~3절) | Codex |
|---|---|---|
| 엔드포인트 | bedrock-runtime | bedrock-mantle |
| 호출 리전 | 서울 | us-east-1 |
| 처리 위치 | 전 세계 리전 중 하나 | 미국 |
| 호출 로그 | 켜면 남음 | **불가** |
| 2편 정책 | 허용 | **차단됨** |

### 4-1. 허용하기 전에 결정할 것

Codex 를 허용한다면 다음 순서로 진행하세요.

1. **코드가 미국에서 처리돼도 되는지** 먼저 결정해요. 사내 코드라면 이 결정을 1편에서 데이터 경계를 판단한 문서에 함께 기록해 둬요.
2. Codex 사용자만 묶은 SSO 권한 세트(예: `CodexUsers`)를 만들어요.
3. 2편 차단 정책의 `DenyMantleUntilNeeded` 문(statement)에 이 역할만 예외로 두는 조건을 추가해요. 2편 6절의 리전 제한 SCP 를 쓰는 조직이라면 SCP 에도 같은 예외를 둬야 해요. Codex 가 `us-east-1` 을 호출하기 때문이에요.

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

SSO 권한 세트로 만든 역할은 `aws-reserved/sso.amazonaws.com/` 경로 아래에 `AWSReservedSSO_<권한세트>_<난수>` 형식의 이름으로 생성돼요. 위 조건은 이 경로와 이름 형식에 맞춘 거예요. `*CodexUsers*` 처럼 넓게 쓰면 역할 생성 권한이 있는 사람이 비슷한 이름의 역할을 만들어 예외를 적용받을 수 있어요.

이 역할의 허용 정책은 다음처럼 범위를 좁혀요. 2편의 모델 허용목록은 `bedrock:InvokeModel` 계열만 제한하므로 **mantle 호출에는 적용되지 않아요.** 그래서 최소한 리전만이라도 `us-east-1` 로 제한해요.

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

mantle 액션을 모델 단위 리소스로 좁힐 수 있는지는 이번에 확인하지 못했어요. 모델 단위로 제한할 수 있다면 `Resource` 를 GPT-6 Astra 로 좁히는 것이 좋아요.

### 4-2. 설정

`~/.codex/config.toml` 설정이에요. AWS 출시 블로그의 GPT-5.5 예제를 GPT-6 Astra 와 mantle 리전에 맞게 고쳤어요.

```toml
model = "openai.gpt-6-astra"
model_provider = "amazon-bedrock"

[model_providers.amazon-bedrock.aws]
region = "us-east-1"
```

- 모델 ID 에 `global.` 접두어가 없어요. mantle 은 In-Region 호출이라 접두어 없는 ID 를 씁니다.
- 인증은 **`AWS_BEARER_TOKEN_BEDROCK` 환경변수가 있으면 그 값을 먼저** 쓰고, 없으면 AWS SDK 자격증명 체인을 써요. 단기 토큰은 **토큰을 만든 리전에서만 유효**하므로 `us-east-1` 로 만들어야 해요.
- **단기 토큰은 현재 역할 자격증명과 함께 만료돼요**(SSO 권한 세트 기본값 1시간). 같은 토큰을 계속 쓰면 약 1시간 뒤 401·403 오류가 나요. 그때는 아래 명령을 다시 실행해 새 토큰을 받으세요.
- 토큰은 **셸 전체에 `export` 하지 말고 Codex 를 실행하는 명령에만** 지정해요. 3편 2절에서 설명했듯이 이 변수가 셸에 남아 있으면 같은 셸의 boto3·Claude Code 가 SSO 대신 이 `us-east-1` 토큰을 써서 서울 리전 호출이 실패해요.

```shell
AWS_BEARER_TOKEN_BEDROCK="$(python3 -c \
  'from aws_bedrock_token_generator import provide_token as p; print(p(region="us-east-1"))')" \
  codex
```

- 데스크톱 앱이나 VS Code 확장은 셸 환경변수를 읽지 못할 수 있어요. 그럴 때는 토큰을 `~/.codex/.env` 에 넣고 앱을 다시 시작하세요. 이 파일의 권한은 `chmod 600` 으로 설정하고, 토큰이 만료되면 새 토큰으로 바꿔 넣어요.

설정을 마치면 Codex 에서 `/status` 를 실행해 provider 와 모델을 확인하세요.

> ⚠️ 이 절은 AWS 출시 블로그(GPT-5.5 기준)와 모델 카드를 바탕으로 GPT-6 Astra 에 맞게 정리한 내용이에요. OpenAI 도움말 문서는 이번에 직접 확인하지 못했어요. 2차 자료에는 두 가지 보고가 있어요. 하나는 mantle 이 함수·MCP 도구만 받기 때문에 `web_search = "disabled"` 설정이 필요했다는 보고예요. 다른 하나는 `~/.aws` 프로필만으로는 인증에 실패해 `aws configure export-credentials --format env` 로 해결했다는 보고예요. 설정이 동작하지 않으면 이 두 가지를 먼저 확인하세요. 나중에 Codex 가 `bedrock-runtime` 의 Global 추론 프로필을 지원하면 이 예외 설정은 없애는 것이 좋아요.

<br>

<br>



## 5. 비용에서 주의할 세 가지

먼저 Bedrock 에서 GPT-6 Astra 의 가격이에요(100만 토큰당 미국 달러, Global 기준).

| 구간 | 입력 | 캐시 읽기 | 출력 |
|---|---|---|---|
| 입력 272K 이하 | $10 | $1 | $50 |
| 입력 272K 초과 | $20 | $2 | $75 |

In-Region·Geo 는 이 가격에 10% 가 더해져요(272K 이하 기준 입력 $11 · 캐시 읽기 $1.1 · 출력 $55).

### 5-1. 272K 기준 — 넘으면 요청 전체 단가가 올라요

비용 차이가 가장 크게 나는 부분이에요. 입력이 272K 를 넘으면 초과분만이 아니라 **그 요청 전체**에 장기 컨텍스트 단가가 적용돼요. 출력을 5K 토큰으로 고정하고 입력만 늘려 비교해 볼게요.

| 입력 | 입력 비용 | 출력 비용 | 합계 |
|---|---|---|---|
| 200K | 0.2 × $10<br>= $2.00 | 0.005 × $50<br>= $0.25 | **$2.25** |
| 300K | 0.3 × $20<br>= $6.00 | 0.005 × $75<br>= $0.375 | **$6.375** |

표의 계산 방법은 이래요. 단가가 100만 토큰 기준이므로 토큰 수를 100만으로 나눈 값에 단가를 곱해요. 200K 토큰은 100만 토큰의 0.2배이므로 단가 $10 에 0.2 를 곱합니다. 출력 5K 토큰은 0.005배예요.

입력은 1.5배 늘었지만 비용은 $6.375 ÷ $2.25 ≈ **2.83배**가 됐어요. 272K 를 넘으면서 입력 단가는 2배, 출력 단가는 1.5배로 함께 올랐기 때문이에요.

가장 간단한 대책은 앱에서 입력을 미리 줄이는 거예요. 대화 이력이나 첨부 문서를 붙일 때 입력이 270K 정도를 넘지 않게 자르세요. 직전 응답의 `usage` 에서 입력 토큰 수를 확인하고, 다음 요청에 붙일 분량을 조절하면 돼요. Bedrock 에서 GPT-6 Astra 는 토큰 수를 미리 세는 API 를 지원하지 않으므로, 응답의 사용량을 기준으로 삼아야 해요.

### 5-2. 출력 10배 쿼터 차감 — 요금이 아니라 처리량이 줄어요

`bedrock-runtime` 에서 GPT-6 Astra 의 분당 토큰 한도(TPM)는 **출력 토큰 1개를 10개로 계산해요.** 요금이 10배가 되지는 않지만, 출력 토큰이 한도를 10배 빠르게 소진해요.

| 요청 | 실제 토큰 | 한도에서 차감 |
|---|---|---|
| 입력 200K · 출력 5K | 205K | 200K + 5K × 10<br>= **250K** |
| 입력 10K · 출력 20K | 30K | 10K + 20K × 10<br>= **210K** |

분당 한도를 200만 토큰으로 **가정**하고 계산해 볼게요. 3편에서 다룬 Claude 의 기본 입력 한도(2M)와는 관계없는 값이에요. GPT-6 Astra 의 실제 기본값은 계정·리전마다 다르고 새 계정은 더 낮게 설정되기도 하니, Service Quotas 콘솔에서 확인하세요. 표의 첫 번째 요청은 200만 ÷ 25만 = **분당 8건**까지만 처리돼요. 10배 차감이 없다면 200만 ÷ 20.5만 ≈ 9.75 이므로 분당 9건까지 처리할 수 있어요.

차이는 두 번째 요청처럼 **출력이 긴 작업**에서 더 커요. 실제 토큰은 3만 개인데 한도에서는 21만 개가 차감돼요. 그래서 요약보다 생성, 특히 코드 생성 요청이 많을수록 `ThrottlingException` 이 먼저 발생해요. `maxTokens` 를 필요 이상으로 크게 잡지 않는 것도 도움이 됩니다.

### 5-3. Ultrafast — 표준 요금의 6배

GPT-6 Astra 에는 Ultrafast 라는 속도 등급이 있어요(`service_tier="ultrafast"`). 응답은 빨라지지만 **요금이 표준의 6배**예요. 272K 이하 Global 기준으로 입력 $60, 출력 $300 입니다. 기본으로 켜 둘 필요는 없어요. 켜야 한다면 비용을 감수하는 이유를 기록해 두세요.

<br>

<br>



## 6. 첫날 관측 — 바로 켤 것과 나중에 켤 것

| 대상 | 첫날 | 이유 |
|---|---|---|
| CloudWatch 메트릭 | ✓ | 설정 없이 쌓임 |
| CloudTrail | ✓ | 실제 처리 리전 확인 |
| 예산 알람 | ✓ | 2편 7절 |
| 호출 로그<br>(원문) | 결정 후 | 새 민감정보<br>저장소가 생김 |

- **CloudWatch `AWS/Bedrock`** 에서 모델 ID 별로 `Invocations`, `InputTokenCount`, `OutputTokenCount`, `InvocationThrottles`, `InvocationLatency` 를 확인해요. 알람은 스로틀 발생과 토큰량 급증, 두 가지면 충분해요.
- **CloudTrail** 은 1편에서 말한 처리 리전의 근거 자료예요. 이벤트의 `additionalEventData.inferenceRegion` 필드에 Global 요청이 실제로 처리된 리전이 기록돼요. 보안 검토에서 데이터가 어느 리전에서 처리됐는지 이 값으로 보여 줄 수 있어요.

### 호출 로그(Model invocation logging)를 켤지

켜는 것을 권해요. 다만 민감정보 저장소를 하나 새로 만드는 일로 보고 설계해야 해요. 이 로그는 요청과 응답 **본문 전체**를 저장해요. 로그를 켜면 프롬프트에 들어간 내용이 모두 버킷 하나에 쌓입니다.

| 설정 | 권장 |
|---|---|
| 목적지 | S3 만<br>(CloudWatch 로 원문 X) |
| 암호화 | 전용 KMS 키 |
| 접근 | 보안 역할만 읽기 |
| 보존 | 30~90일 수명주기 |
| 모달리티 | 텍스트만 |

Anthropic 은 오남용 조사를 위해 최소 30일 보존을 권해요. 아직 이 표대로 구성할 준비가 안 됐다면 첫날은 메트릭과 CloudTrail 만 보고, 호출 로그는 설계를 마친 뒤에 켜세요. 또 AWS 호출 로그 문서에 따르면 **mantle 호출은 이 로그에 남지 않아요.** CloudTrail 에 어떻게 기록되는지는 직접 확인해 보세요. 4절에서 Codex 예외는 데이터 경계를 결정한 뒤에만 허용하라고 한 것도 이 때문이에요.

<br>

<br>



## 7. 세팅하면서 판단이 갈린 부분

**① Converse 로 통일할까**

코드 이전 비용만 보면 openai SDK 를 그대로 쓰는 쪽이 가장 적게 들어요. 기존 코드를 거의 고치지 않아도 되니까요. 하지만 GPT 의 Guardrails 와 비용 태깅은 Converse 에서만 지원돼요. 호출 방식이 둘이면 정책을 적용하고 점검할 곳도 둘이 됩니다. 그래서 Converse 함수 하나를 기본으로 두고, 명시 캐시나 파일 입력처럼 Converse 가 지원하지 않는 기능이 필요한 호출만 openai SDK 로 남겼어요. 호출하는 쪽은 `ask()` 를 그대로 쓰고, 분기는 함수 안에서만 하도록 했어요.

**② Codex 를 시리즈에 넣을까**

"Bedrock 에서 ChatGPT 를 쓸 수 있나요?" 라는 질문은 Codex 를 쓸 수 있느냐는 뜻일 때도 있어요. 반면 보안 측면에서는 문제가 여러 가지예요. mantle 을 쓰므로 정책 예외가 필요하고, 호출 로그가 남지 않으며, 요청이 미국에서 처리돼요. 그래도 시리즈에 넣기로 했어요. 대신 예외라는 점이 드러나도록 서울 Global 과의 비교표를 두고, 허용하기 전에 결정할 것과 예외 역할 설정을 함께 실었어요.

**③ 비용 계산을 본문에 둘까**

숫자 표가 많으면 글이 읽기 어려워져요. 하지만 272K 기준은 "조금 비싸진다" 가 아니라 "입력이 1.5배 늘면 비용이 2.83배가 된다" 는 문제라서, 계산 없이 설명하면 차이가 잘 전달되지 않아요. 그래서 계산은 두 줄짜리 표로 줄이고, 표 바로 아래에 계산 방법을 덧붙였어요.

<br>

<br>



## 8. 1~4편 전체 체크리스트

1~4편을 모두 따라 했다면 아래 항목이 모두 완료돼 있어야 해요.

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

1편 첫머리의 질문으로 돌아가 볼게요. "Bedrock 에서 ChatGPT 를 쓸 수 있나요?" 이제는 이렇게 답할 수 있어요. ChatGPT 앱은 쓸 수 없지만 GPT 모델은 호출할 수 있어요. 서울에서 쓰려면 `global.` 추론 프로필로 호출하고, 이 호출이 국외 처리라는 사실을 결정 문서에 남기세요. 키는 단기 토큰을 쓰고, 예산 알람은 계정 전체에 걸어 두세요.

아직 결론을 내리지 못한 문제도 하나 있어요. 규제 대상 데이터는 서울 In-Region 으로 호출할 수 있는 모델이 나올 때까지 기다려야 할 수도 있어요. 반대로 그때까지 마스킹 게이트를 두고 쓰는 것이 현실적일 수도 있습니다. 답은 조직마다 다를 거예요. 어느 쪽을 택하든 **그 결정은 반드시 기록으로 남겨 두세요.**

일단 오늘은 여기까지.....  
다음 글에서는 이미 OpenAI·Anthropic API 를 직접 호출하고 있는 코드를 Bedrock 으로 옮기는 방법을 다뤄요. 호출 지점 목록 만들기부터 Bedrock 에 없는 기능의 대안, 임베딩 재색인, 비용 추적 변경, 단계적 전환과 롤백까지 정리할게요.

---

**← 이전 글:** [(3/5) Claude 붙이기 — Converse·Anthropic SDK·Claude Code](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)  
**다음 글 →:** [(5/5) 기존 OpenAI·Anthropic API 코드를 Bedrock 으로 옮기기](/coding/Bedrock_GPT_Claude_첫세팅_기존코드_이전/)
