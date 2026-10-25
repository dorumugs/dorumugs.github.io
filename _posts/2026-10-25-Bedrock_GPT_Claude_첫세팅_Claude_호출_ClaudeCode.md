---
layout: single
title:  "(3/4) Bedrock 에 Claude 붙이기 — Converse·Anthropic SDK·Claude Code"
date: 2026-10-25 09:20:00 +0900
categories: coding
tag: [Bedrock, AWS, Claude, ClaudeCode, Converse, AnthropicSDK, boto3, 서울리전, 에러해결, 초기세팅]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-claude/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-claude/header.svg
description: "서울 리전에서 global.anthropic.claude-opus-5-5 로 첫 응답을 받는 두 가지 방법(boto3 Converse, Anthropic SDK + 단기 토큰)을 코드로 보여 주고, 첫 호출에서 나는 에러를 원인별 표로 정리했어요. 같은 계정에 Claude Code 를 연결할 때 서울 리전에서 생기는 apac. 접두어 문제와 모델 고정(pin)도 다뤄요."
series: bedrock-gpt-claude-setup
series_order: 3
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="3" %}

## Summary

[2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)까지 따라왔다면 계정 준비는 끝났어요. 차단 정책을 적용했고, 개발자는 SSO 로 로그인하며, Claude 사용 사례 양식의 상태는 `AVAILABLE` 이에요. 이번 편에서는 코드로 첫 응답을 받아 봅니다.

호출 코드 자체는 다섯 줄 남짓이에요. 하지만 첫 호출이 실패하는 원인은 대부분 코드 밖에 있습니다. 그래서 이번 편은 짧은 예제 코드 두 개와 에러 표 하나로 구성했어요.

> 💡 **이 글에서 다루는 것**
> - 호출 방식 고르기 — Converse vs Anthropic SDK vs Invoke
> - 첫 호출 A — boto3 Converse (SSO 자격증명 사용)
> - 첫 호출 B — Anthropic SDK + 단기 토큰
> - Opus 5.5 에서 "최소 코드" 가 실패하는 두 부분
> - 첫 호출 에러 표 — 메시지별 원인과 해결
> - Claude Code 연결 — 서울의 `apac.` 접두어 문제와 모델 고정

<br>

<br>



## 1. 호출 방식 고르기

같은 `bedrock-runtime` 에서도 Claude 를 호출하는 방식은 세 가지예요.

| 경로 | 요청 모양 | 추천 대상 |
|---|---|---|
| **Converse**<br>(boto3) | Bedrock 공통 | GPT 와 같이 쓰는 앱<br>**(이 시리즈 기본)** |
| **Messages**<br>(Anthropic SDK) | Claude 네이티브 | Claude 만 쓰는 앱,<br>직접 API 코드 이전 |
| **Invoke**<br>(boto3) | 회사별 본문 | 본문을 직접<br>구성할 때만 |

이 시리즈가 Converse 를 기본으로 정한 이유는 두 가지예요. 첫째, GPT 와 Claude 를 **함수 하나로** 호출할 수 있어서 관리할 호출 코드가 하나로 줄어요. 둘째, GPT 의 Guardrails 와 비용 태깅은 Converse 에서만 지원됩니다. 자세한 비교는 4편에서 다뤄요.

Claude 만 쓰고 Anthropic 직접 API 로 작성한 코드가 이미 있다면 Messages 방식이 편해요. 요청과 응답 형식이 같아서 `base_url` 만 바꾸면 됩니다.

> ⚠️ Anthropic 문서는 Bedrock 용으로 `AnthropicBedrockMantle` 클라이언트를 권장해요. 하지만 이 클라이언트는 mantle 엔드포인트를 사용하므로 2편의 차단 정책(`DenyMantleUntilNeeded`)에 막혀요. 그래서 이 시리즈에서는 같은 Messages API 를 `bedrock-runtime` 의 `/anthropic` 경로로 호출해요(4절).

<br>

<br>



## 2. 준비

```shell
pip install -U boto3 anthropic aws-bedrock-token-generator
aws sso login --profile bedrock-dev
export AWS_PROFILE=bedrock-dev
export AWS_REGION=ap-northeast-2
```

호출 전에 확인할 것이 하나 있어요. 셸에 `AWS_BEARER_TOKEN_BEDROCK` 환경변수가 설정돼 있으면 해제하세요. 이 변수가 있으면 SDK 와 Claude Code 는 SSO 자격증명 대신 이 변수의 토큰을 사용해요. 예전에 설정한 장기 키가 이 변수에 남아 있을 수 있습니다.

```shell
env | grep -c AWS_BEARER_TOKEN_BEDROCK
unset AWS_BEARER_TOKEN_BEDROCK
```

<br>

<br>



## 3. 첫 호출 A — boto3 Converse

boto3 는 SSO 로 받은 임시 자격증명을 그대로 사용해요. 키를 따로 관리할 필요가 없어서 가장 간단한 방법이에요.

```python
import boto3

REGION = "ap-northeast-2"
MODEL = "global.anthropic.claude-opus-5-5"

rt = boto3.client("bedrock-runtime", region_name=REGION)

resp = rt.converse(
    modelId=MODEL,
    messages=[{
        "role": "user",
        "content": [{"text": "안녕! 한 문장으로 자기소개 해줘."}],
    }],
    inferenceConfig={"maxTokens": 4096},
)

blocks = resp["output"]["message"]["content"]
print("".join(b["text"] for b in blocks if "text" in b))
print(resp["usage"], resp["stopReason"])
```

짧은 코드지만 의도적으로 작성한 부분이 세 군데 있어요.

- **`MODEL` 이 `global.` 로 시작해요.** 서울에서 `anthropic.claude-opus-5-5` 처럼 접두어 없이 호출하면 `on-demand throughput isn't supported` 400 에러가 나요(1편 3절).
- **`content[0]` 을 바로 읽지 않아요.** Opus 5.5 는 생각(thinking)이 항상 켜져 있어서 응답에 텍스트가 아닌 블록이 포함될 수 있어요. 그래서 `text` 키가 있는 블록만 골라 이어 붙입니다.
- **`maxTokens` 를 넉넉하게 잡아요.** 생각에도 출력 토큰이 쓰여요. 값이 너무 작으면 생각 단계에서 토큰이 바닥나 `stopReason` 이 `max_tokens` 가 되고, 답변은 비어 있게 됩니다.

> ⚠️ 기존 코드의 `inferenceConfig` 에 `temperature` 가 있다면 지우세요. Anthropic 문서에 따르면 Opus 5.5 는 `temperature`·`top_p` 같은 샘플링 값을 받지 않아요(400). 이 모델은 응답 성향을 샘플링 값 대신 **effort(생각 깊이)** 로 조절합니다.

<br>

<br>



## 4. 첫 호출 B — Anthropic SDK + 단기 토큰

Claude 직접 API 로 작성한 코드가 있다면 이 방법을 쓰세요. `base_url` 을 `bedrock-runtime` 의 `/anthropic` 경로로 바꾸고, API 키 대신 **단기 토큰**을 넣어요. 아래 코드는 AWS 의 Opus 5.5 모델 카드 예제를 서울 리전에 맞게 고친 거예요.

```python
from anthropic import Anthropic
from aws_bedrock_token_generator import provide_token

REGION = "ap-northeast-2"
BASE = f"https://bedrock-runtime.{REGION}.amazonaws.com/anthropic"


def claude() -> Anthropic:
    token = provide_token(region=REGION)
    return Anthropic(base_url=BASE, api_key=token)


msg = claude().messages.create(
    model="global.anthropic.claude-opus-5-5",
    max_tokens=4096,
    output_config={"effort": "low"},
    messages=[{"role": "user", "content": "안녕!"}],
)
print(next(b.text for b in msg.content if b.type == "text"))
```

- `provide_token()` 은 SSO 자격증명으로 단기 토큰을 발급해요. 토큰 만료 시각은 현재 역할 자격증명의 만료 시각(권한 세트 기본 1시간)과 12시간 중 더 이른 쪽이에요. 토큰의 권한은 SSO 역할의 권한과 같아서 장기 키를 만들 필요가 없습니다.
- **토큰은 발급한 리전에서만 유효해요.** 서울 리전(`region`)에서 발급한 토큰은 서울 엔드포인트로만 보내야 해요.
- 이 코드는 AWS 모델 카드의 예제(`Anthropic(base_url=…/anthropic, api_key=토큰)`)를 따랐어요. `output_config` 는 최신 `anthropic` 패키지에서만 지원하니 `pip install -U` 로 업데이트하세요.
- 함수를 호출할 때마다 토큰을 새로 발급하는 이유는 만료 때문이에요. AWS 는 호출마다 발급해도 될 만큼 비용이 작다고 설명합니다. 오래 실행되는 서버라면 토큰을 캐시하고 만료 전에 새로 발급하면 돼요.
- `output_config={"effort": "low"}` 는 생각 깊이를 정해요. Opus 5.5 의 기본값은 `medium` 이에요. 인사처럼 가벼운 요청은 `low` 로 두면 응답이 빠르고 비용도 줄어요.

> 💡 이 방법은 bearer 토큰을 쓰므로 `bedrock:CallWithBearerToken` 권한이 필요해요. 2편의 차단 정책은 장기 키만 막기 때문에 단기 토큰은 통과합니다. 조직이 단기 토큰까지 막았다면 3절의 Converse 방식을 쓰세요.

<br>

<br>



## 5. 첫 호출 에러 표

예제 코드를 그대로 실행해도 실패한다면 원인은 대부분 아래 표에 있어요. 에러 문구로 바로 찾을 수 있도록 첫 열은 에러 메시지 기준으로 정리했어요.

| 메시지·증상 | 원인 | 해결 |
|---|---|---|
| `on-demand throughput`<br>`isn't supported` | 접두어 없는 ID | `global.` 프로필 ID |
| `ValidationException`<br>모델 ID 잘못됨 | `apac.` 접두어,<br>오타 | `list-inference-profiles`<br>로 실제 ID 확인 |
| `AccessDenied`<br>(Claude 만, GPT 는 됨) | 사용 사례 양식<br>미제출·`PENDING` | 2편 3-2,<br>상태 조회 |
| `AccessDenied`<br>처음엔 성공 후 실패 | 자동 구독 진행 중,<br>결제수단 없음 | 15분 대기,<br>결제수단 확인 |
| `AccessDenied`<br>`bedrock:InvokeModel` | ARN 세 개 중<br>하나 누락 | 2편 4-3 의<br>세 줄 확인 |
| `AccessDenied`<br>정책은 정상 | 조직 SCP,<br>`unspecified` | 2편 6절 |
| `AccessDenied`<br>VPC 안에서만 | `aws:SourceIp` 조건<br>+ VPC 엔드포인트 | `aws:SourceVpce`<br>로 교체 |
| `AccessDenied`<br>Claude Code 의<br>백그라운드 작업만 | 고정하지 않은 모델이<br>허용목록 밖 | 6-2 의 세 모델<br>모두 고정 |
| 403<br>`bedrock-mantle:` 언급 | mantle 클라이언트<br>사용 | runtime 경로 사용<br>(1절 경고) |
| 401·403<br>(Anthropic SDK) | 토큰 만료·다른<br>리전에서 발급 | 같은 리전으로<br>재발급 |
| 400<br>`temperature` 관련 | Opus 5.5 는<br>샘플링 값 없음 | 값을 지우고<br>effort 사용 |
| 응답 텍스트가 빔 | 생각 단계에서<br>토큰 소진 | `maxTokens` 늘리기,<br>effort 낮추기 |
| `ThrottlingException` | 분당 토큰·요청<br>한도 초과 | 백오프,<br>한도 증설 요청 |

- VPC 엔드포인트와 `aws:SourceIp` 조건을 함께 쓰는 경우는 2편에서 다루지 않아서 여기서 보충해요. VPC 엔드포인트를 거친 요청에는 `aws:SourceIp` 값이 없어요. 따라서 IP 조건이 있는 정책은 이런 요청을 항상 거부합니다. VPC 안에서 호출한다면 조건을 `aws:SourceVpce` 로 바꾸세요.
- 백오프는 boto3 설정 한 줄로 켤 수 있어요. 클라이언트를 만들 때 `botocore.config.Config(retries={"mode": "adaptive"})` 를 넘기면 됩니다.
- Claude 의 기본 한도는 분당 입력 2백만 토큰이에요. 입력 5백만 · 출력 50만 토큰까지는 Anthropic 승인 없이 증설을 요청할 수 있어요. 분당 요청 수(RPM)는 AWS 가 관리하는 한도라서 AWS Support 에 요청합니다.

<br>

<br>



## 6. Claude Code 연결하기

같은 계정에 Claude Code 를 연결하면 팀원마다 API 키를 따로 발급할 필요가 없어요. 비용은 AWS 계정으로 청구되고, 권한은 SSO 로 관리됩니다.

### 6-1. 가장 쉬운 방법 — 설정 마법사

`claude` 를 실행하고 로그인 화면에서 **3rd-party platform → Amazon Bedrock** 을 선택하면 설정 마법사가 시작돼요. 이미 로그인한 상태라면 `/setup-bedrock` 을 입력하세요. 마법사는 AWS 프로필, 리전, 사용 가능한 모델을 확인한 뒤 결과를 `~/.claude/settings.json` 의 `env` 블록에 저장해요.

### 6-2. 서울 리전에서 꼭 확인할 것 — `apac.` 접두어

마법사로 설정했더라도 서울 리전에서는 아래 값을 직접 확인해야 해요.

Claude Code 는 리전에 따라 모델 ID 앞에 붙일 접두어를 정하는데, `ap-*` 리전에서는 `apac.` 을 붙여요. 하지만 1편에서 설명했듯 Opus 5.5 에는 `apac.` 추론 프로필이 없어요. 그래서 서울에서는 접두어를 `global` 로 지정해야 합니다. 또 서울에서 서브에이전트의 모델 ID 가 `us.` 로 잘못 설정되는 버그가 Claude Code 저장소에 보고된 적이 있어요(이슈 #24875). 이 글에서는 접두어 지정에 더해 **세 모델을 모두 고정(pin)** 합니다.

```json
{
  "awsAuthRefresh": "aws sso login --profile bedrock-dev",
  "env": {
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_PROFILE": "bedrock-dev",
    "AWS_REGION": "ap-northeast-2",
    "ANTHROPIC_BEDROCK_REGION_PREFIX": "global",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":
      "global.anthropic.claude-opus-5-5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL":
      "global.anthropic.claude-sonnet-5-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":
      "global.anthropic.claude-haiku-5-5"
  }
}
```

| 설정 | 하는 일 |
|---|---|
| `awsAuthRefresh` | SSO 가 만료되면<br>자동으로 다시 로그인 |
| `ANTHROPIC_BEDROCK_`<br>`REGION_PREFIX` | 기본 접두어를<br>`apac.` 대신 `global.` 로 |
| `ANTHROPIC_DEFAULT_`<br>`*_MODEL` 세 개 | 모델 버전 고정,<br>백그라운드 모델 지정 |

모델을 고정해야 하는 이유가 하나 더 있어요. **모델을 고정하지 않으면 Claude Code 는 Opus 5.5 를 기본 모델로 써요.** 가장 비싼 모델이 모든 작업을 처리하게 됩니다. 제목 생성 같은 백그라운드 작업에도 문제가 생겨요. Haiku 를 고정하지 않으면 이 작업은 Bedrock 기본값인 Sonnet 4.5 를 호출해요. 2편의 모델 허용목록은 5.5 계열만 허용하므로 이 호출은 `AccessDenied` 로 실패합니다. 세 모델을 모두 지정해 두면 작업마다 어떤 모델이 쓰이는지 명확해지고, 허용목록과 어긋나지도 않아요.

### 6-3. 확인할 것과 참고 사항

- Claude Code 에서 `/status` 를 입력하면 provider 와 실제 리전이 표시돼요. **Amazon Bedrock** 과 **ap-northeast-2** 가 보이면 정상입니다.
- 2편에서는 개발자 역할에서 Marketplace 권한을 뺐어요. 반면 Claude Code 공식 IAM 예시에는 `aws-marketplace:ViewSubscriptions`·`Subscribe` 가 포함돼 있어요. 관리자의 첫 호출로 구독이 끝났는데도 Claude Code 에서 Marketplace 관련 거부가 나면 `ViewSubscriptions` 부터 추가해 보세요.
- 2편의 허용 정책에 `ListInferenceProfiles` 를 넣은 이유도 Claude Code 때문이에요. Claude Code 는 시작할 때 사용 가능한 추론 프로필을 조회해요. 이 권한이 없으면 조회 없이 접두어를 붙여 호출하고, 그 결과 400 에러가 날 수 있어요.
- Bedrock 에서는 웹 검색 도구를 쓸 수 없어요. Claude Code 의 `WebSearch` 는 Anthropic 직접 API 기능이라 Bedrock 연결에서는 제공되지 않습니다.
- 팀에 배포할 때는 관리형 설정(managed settings)에 `availableModels` 와 `enforceAvailableModels` 를 넣어 사용할 수 있는 모델을 제한할 수 있어요. 목록은 2편의 모델 허용목록과 같게 맞추세요.

<br>

<br>



## 7. 세팅하면서 판단이 갈린 부분

**① 코드는 얼마나 줄이고, 에러 표는 어디에 둘까**

첫 응답까지의 과정이 짧을수록 따라 하기 쉬워요. 그래서 처음에는 코드 다섯 줄과 설명 세 줄로 끝낼 생각이었어요. 하지만 다섯 줄짜리 최소 코드는 Opus 5.5 에서 바로 실패해요(`content[0]`, `temperature`). 또 첫 호출 실패의 원인은 대개 IAM·양식·구독·접두어처럼 코드 밖에 있어서, 표 없이는 코드를 그대로 실행해도 해결이 안 돼요. 독자가 에러 문구로 검색해서 이 글을 찾을 수 있다는 점도 고려했어요. 그래서 코드는 최소한으로 두되 실패하는 두 부분만 보완했어요. 에러 표는 첫 열을 '메시지·증상' 으로 정하고, 접지 않은 채 본문에 넣었어요.

**② Anthropic 이 권장하는 Mantle 클라이언트를 쓸까**

공식 권장 방식이고 SDK 가 SigV4 서명까지 처리해 주니 개발 편의성은 좋아요. 하지만 보안 관점에서 mantle 은 호출 로그가 남지 않고 IAM 접두어도 달라서, 2편에서 의도적으로 막아 둔 엔드포인트예요. 예제 하나 때문에 그 차단을 풀라고 할 수는 없었어요. 그래서 같은 Messages API 를 runtime 의 `/anthropic` 경로로 호출하고, mantle 클라이언트는 경고로만 소개했어요.

**③ Claude Code 기본 모델을 Opus 로 둘까**

코딩 품질만 보면 Opus 가 나아요. 하지만 기본 모델이 Opus 라는 사실을 모르고 쓰면 예상보다 많은 비용이 청구될 수 있어요. 어떤 모델을 기본으로 쓸지는 팀마다 정하면 되지만, 이 글에서는 **세 모델을 반드시 명시적으로 고정**하는 것을 기준으로 삼았어요. 2편의 예산 알람도 함께 설정해 두세요.

<br>

<br>



## 8. 동작 확인 체크리스트

- [ ] Converse 예제가 한 줄 응답을 내고, `modelId` 가 `global.` 로 시작한다
- [ ] 15분 뒤에 다시 실행해도 성공한다 (자동 구독 확정 확인)
- [ ] Anthropic SDK 예제가 단기 토큰으로 응답한다
- [ ] 셸에 `AWS_BEARER_TOKEN_BEDROCK` 이 남아 있지 않다
- [ ] Claude Code `/status` 에 Amazon Bedrock · ap-northeast-2 가 보인다
- [ ] `settings.json` 에 세 모델이 모두 `global.` ID 로 고정돼 있다

일단 오늘은 여기까지.....  
다음 글에서는 같은 계정에 OpenAI GPT 를 연결해요. openai SDK 를 그대로 쓰는 방법, 이번 편의 Converse 호출을 함수 하나로 묶어 GPT 와 Claude 를 함께 호출하는 방법을 다룹니다. Codex 가 서울 대신 미국 리전을 호출하는 이유와 GPT-6 Astra 의 비용 함정도 정리할게요.

---

**← 이전 글:** [(2/4) 첫날 계정·권한 세팅](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)  
**다음 글 →:** [(4/4) OpenAI GPT 붙이기 — openai SDK·단일 함수·Codex 와 비용 함정](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)
