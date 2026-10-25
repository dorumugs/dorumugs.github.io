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
description: "서울 리전에서 global.anthropic.claude-opus-5-5 로 첫 응답을 받는 두 경로(boto3 Converse, Anthropic SDK + 단기 토큰)를 코드로 보여주고, 첫 호출에서 실제로 만나는 에러를 원인별 표로 모았어요. 같은 계정에 Claude Code 를 붙일 때 서울에서만 생기는 apac. 접두어 함정과 모델 고정(pin)도 다뤄요."
series: bedrock-gpt-claude-setup
series_order: 3
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="3" %}

## Summary

[2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)까지 오면 계정은 준비가 끝났어요. 차단 정책이 깔려 있고, 개발자는 SSO 로 들어오고, Claude 의 사용 사례 양식은 `AVAILABLE` 이에요. 이제 코드로 첫 응답을 받을 차례예요.

코드만 보면 다섯 줄 남짓이면 끝나요. 그런데 첫 호출이 막히는 이유는 대부분 코드 바깥에 있어요. 그래서 이번 편은 짧은 코드 두 개와 긴 에러 표 하나로 채웠어요.

> 💡 **이 글에서 다루는 것**
> - 경로 고르기 — Converse vs Anthropic SDK vs Invoke
> - 첫 호출 A — boto3 Converse (SSO 자격증명 그대로)
> - 첫 호출 B — Anthropic SDK + 단기 토큰
> - Opus 5.5 에서 "최소 코드" 가 깨지는 두 지점
> - 첫 호출 에러 표 — 메시지별 원인과 해결
> - Claude Code 붙이기 — 서울의 `apac.` 함정과 모델 고정

<br>

<br>



## 1. 경로 고르기

`bedrock-runtime` 하나 안에서도 Claude 를 부르는 길은 세 갈래예요.

| 경로 | 요청 모양 | 추천 대상 |
|---|---|---|
| **Converse**<br>(boto3) | Bedrock 공통 | GPT 와 같이 쓰는 앱<br>**(이 시리즈 기본)** |
| **Messages**<br>(Anthropic SDK) | Claude 네이티브 | Claude 만 쓰는 앱,<br>직접 API 코드 이전 |
| **Invoke**<br>(boto3) | 회사별 본문 | 직접 맞출 일이<br>있을 때만 |

이 시리즈에서 Converse 를 기본으로 잡은 이유는 4편에서 자세히 다뤄요. GPT 와 Claude 를 **같은 함수 하나로** 부를 수 있고, GPT 쪽은 Guardrails 와 비용 태깅이 Converse 에서만 돼요. 관리할 경로가 하나로 모인다는 게 제일 컸어요.

반대로 Claude 만 쓰고, 이미 Anthropic 직접 API 로 짠 코드가 있다면 Messages 경로가 편해요. 요청·응답 모양이 그대로라 `base_url` 만 바꾸면 되니까요.

> ⚠️ Anthropic 문서는 Bedrock 용으로 `AnthropicBedrockMantle` 클라이언트를 권해요. 그런데 이름 그대로 mantle 엔드포인트를 쓰는 클라이언트라서 2편의 차단 정책(`DenyMantleUntilNeeded`)에 걸려요. 그래서 이 시리즈에서는 같은 Messages API 를 `bedrock-runtime` 의 `/anthropic` 경로로 불러요(4절).

<br>

<br>



## 2. 준비

```shell
pip install -U boto3 anthropic aws-bedrock-token-generator
aws sso login --profile bedrock-dev
export AWS_PROFILE=bedrock-dev
export AWS_REGION=ap-northeast-2
```

그리고 하나만 미리 확인해요. 셸에 `AWS_BEARER_TOKEN_BEDROCK` 이 남아 있으면 지우세요. 이 변수가 있으면 SDK 와 Claude Code 가 SSO 자격증명을 무시하고 그 토큰을 써요. 예전에 넣어둔 장기 키가 여기 숨어 있는 경우가 많아요.

```shell
env | grep -c AWS_BEARER_TOKEN_BEDROCK
unset AWS_BEARER_TOKEN_BEDROCK
```

<br>

<br>



## 3. 첫 호출 A — boto3 Converse

SSO 로 받은 임시 자격증명을 boto3 가 그대로 써요. 따로 키를 다룰 일이 없어서 가장 깔끔한 길이에요.

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

짧은 코드지만 일부러 이렇게 쓴 데가 세 군데 있어요.

- **`MODEL` 이 `global.` 로 시작해요.** 서울에서 `anthropic.claude-opus-5-5` 처럼 접두어 없이 부르면 `on-demand throughput isn't supported` 400 이 나요(1편 3절).
- **`content[0]` 을 바로 꺼내지 않아요.** Opus 5.5 는 생각(thinking)이 항상 켜져 있는 모델이라 응답에 텍스트가 아닌 블록이 섞일 수 있어요. `text` 키가 있는 블록만 골라 붙이는 한 줄이 필요해요.
- **`maxTokens` 를 넉넉히 둬요.** 생각도 출력 토큰을 써요. 너무 작게 잡으면 생각만 하다가 `stopReason` 이 `max_tokens` 로 끝나고 답이 비어요.

> ⚠️ 예전 코드에서 `inferenceConfig` 에 `temperature` 를 넣던 습관이 있다면 빼세요. Anthropic 문서 기준으로 Opus 5.5 는 `temperature`·`top_p` 같은 샘플링 값을 받지 않아요(400). 응답 성향은 샘플링 값이 아니라 **effort(생각 깊이)** 로 조절하는 모델이에요.

<br>

<br>



## 4. 첫 호출 B — Anthropic SDK + 단기 토큰

Claude 직접 API 로 짠 코드가 있다면 이쪽이에요. `base_url` 을 `bedrock-runtime` 의 `/anthropic` 경로로 바꾸고, 키 자리에 **단기 토큰**을 넣어요. 형태는 AWS 의 Opus 5.5 모델 카드 예제를 서울용으로 옮긴 거예요.

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

- `provide_token()` 은 SSO 자격증명으로 단기 토큰을 만들어요. 만료는 지금 쓰는 역할 자격증명의 만료(권한 세트 기본 1시간)와 12시간 중 짧은 쪽이고, 권한도 SSO 역할 그대로예요. 장기 키를 만들 일이 없어요.
- **토큰은 만든 리전에서만 유효해요.** `region` 을 서울로 만들고 서울로 보내야 해요.
- 이 경로는 AWS 모델 카드의 예제(`Anthropic(base_url=…/anthropic, api_key=토큰)`)를 따른 거예요. `output_config` 는 최근 `anthropic` 패키지에만 있으니 `pip install -U` 로 올려 두세요.
- 토큰을 함수 안에서 매번 만드는 이유는 만료 때문이에요. AWS 는 매 호출 발급해도 될 만큼 가볍다고 설명해요. 오래 도는 서버라면 만료 전에 갈아 끼우는 캐시를 두면 돼요.
- `output_config={"effort": "low"}` 는 생각 깊이를 정해요. Opus 5.5 의 기본값은 `medium` 이에요. 인사 같은 가벼운 요청은 `low` 로 두면 빠르고 싸요.

> 💡 이 경로는 bearer 토큰을 쓰니 `bedrock:CallWithBearerToken` 이 필요해요. 2편의 차단 정책은 장기 키만 막기 때문에 단기 토큰은 통과해요. 조직이 단기 키까지 막았다면 3절의 Converse 경로를 쓰세요.

<br>

<br>



## 5. 첫 호출 에러 표

솔직히 이 글에서 제일 공들인 부분이에요. 코드를 그대로 베껴도 막히는 이유는 거의 다 여기 있어요. 에러 문구로 검색해서 들어오시는 분이 많을 것 같아 첫 열을 메시지 기준으로 잡았어요.

| 메시지·증상 | 원인 | 해결 |
|---|---|---|
| `on-demand throughput`<br>`isn't supported` | 접두어 없는 ID | `global.` 프로필 ID |
| `ValidationException`<br>모델 ID 잘못됨 | `apac.` 접두어,<br>오타 | `list-inference-profiles`<br>로 실제 ID 확인 |
| `AccessDenied`<br>(Claude 만, GPT 는 됨) | 사용 사례 양식<br>미제출·`PENDING` | 2편 3-2,<br>상태 조회 |
| `AccessDenied`<br>처음엔 되다가 실패 | 자동 구독 진행 중,<br>결제수단 없음 | 15분 대기,<br>결제수단 확인 |
| `AccessDenied`<br>`bedrock:InvokeModel` | ARN 세 겹 중<br>하나 누락 | 2편 4-3 의<br>세 줄 확인 |
| `AccessDenied`<br>정책은 멀쩡함 | 조직 SCP,<br>`unspecified` | 2편 6절 |
| `AccessDenied`<br>VPC 안에서만 | `aws:SourceIp` 조건<br>+ VPC 엔드포인트 | `aws:SourceVpce`<br>로 교체 |
| `AccessDenied`<br>Claude Code 의<br>백그라운드 작업만 | 고정 안 한 모델이<br>허용목록 밖 | 6-2 의 세 모델<br>모두 고정 |
| 403<br>`bedrock-mantle:` 언급 | mantle 클라이언트<br>사용 | runtime 경로로<br>(1절 경고) |
| 401·403<br>(Anthropic SDK) | 토큰 만료·다른<br>리전에서 발급 | 같은 리전으로<br>재발급 |
| 400<br>`temperature` 관련 | Opus 5.5 는<br>샘플링 값 없음 | 빼고 effort 로 |
| 응답 텍스트가 빔 | 생각에 토큰 소진 | `maxTokens` 늘리기,<br>effort 낮추기 |
| `ThrottlingException` | 분당 토큰·요청<br>한도 초과 | 백오프,<br>한도 증설 요청 |

- VPC 엔드포인트와 `aws:SourceIp` 조합은 2편에서 다루지 않은 함정이라 한 줄 보탤게요. VPC 엔드포인트를 거친 요청에는 `aws:SourceIp` 값이 아예 없어요. 그래서 IP 조건을 건 정책은 항상 실패해요. VPC 안에서 부를 땐 `aws:SourceVpce` 조건으로 바꿔야 해요.
- 백오프는 boto3 설정 한 줄로 켤 수 있어요. `botocore.config.Config(retries={"mode": "adaptive"})` 를 클라이언트에 넘기면 돼요.
- Claude 의 기본 한도는 분당 입력 2백만 토큰이에요. 입력 5백만 · 출력 50만까지는 Anthropic 승인 없이 늘려달라고 요청할 수 있어요. 분당 요청 수(RPM)는 AWS 쪽 한도라 AWS 지원으로 요청해요.

<br>

<br>



## 6. Claude Code 붙이기

같은 계정에 Claude Code 를 붙이면 팀원이 각자 API 키를 들고 다닐 필요가 없어요. 청구는 AWS 계정으로 모이고, 권한은 SSO 로 끝나요.

### 6-1. 가장 쉬운 길 — 마법사

`claude` 를 실행하고 로그인 화면에서 **3rd-party platform → Amazon Bedrock** 을 고르면 마법사가 나와요. 이미 로그인돼 있다면 `/setup-bedrock` 을 입력하세요. 마법사가 AWS 프로필, 리전, 쓸 수 있는 모델을 확인하고 결과를 `~/.claude/settings.json` 의 `env` 블록에 저장해요.

### 6-2. 서울이라면 꼭 손볼 것 — `apac.` 함정

마법사 결과든 직접 쓰든, 서울 사용자는 아래 값을 직접 확인해야 해요.

Claude Code 는 리전을 보고 모델 ID 앞에 붙일 접두어를 고르는데, `ap-*` 리전이면 `apac.` 을 골라요. 그런데 1편에서 봤듯 Opus 5.5 에는 `apac.` 프로필이 없어요. 서울에서는 접두어를 `global` 로 못 박아야 하는 이유예요. 서울에서 서브에이전트의 모델 ID 가 엉뚱하게 `us.` 로 잡히는 버그가 Claude Code 저장소에 보고된 적도 있어요(이슈 #24875). 그래서 저는 접두어를 지정하는 데서 그치지 않고 **세 모델을 전부 고정(pin)** 해 둬요.

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

고정해야 하는 이유가 하나 더 있어요. **고정하지 않으면 Claude Code 의 기본 모델은 Opus 5.5 예요.** 제일 비싼 모델로 하루 종일 돌게 되는 거죠. 제목 생성 같은 백그라운드 작업도 문제예요. Haiku 를 고정하지 않으면 Bedrock 기본값인 Sonnet 4.5 로 가는데, 2편의 모델 허용목록은 5.5 계열만 열어 두었으니 이 호출은 `AccessDenied` 로 끝나요. 세 개를 다 적어 두면 어디서 어떤 모델이 도는지 알 수 있고 허용목록과 어긋날 일도 없어요.

### 6-3. 확인과 알아둘 것

- Claude Code 안에서 `/status` 를 치면 provider 와 실제 리전이 나와요. **Amazon Bedrock** 과 **ap-northeast-2** 가 보이면 됩니다.
- 2편은 개발자 역할에서 Marketplace 권한을 뺐는데, Claude Code 공식 IAM 예시는 `aws-marketplace:ViewSubscriptions`·`Subscribe` 를 넣어 둬요. 관리자가 첫 호출로 구독을 끝냈는데도 Claude Code 에서 Marketplace 관련 거부가 나면 `ViewSubscriptions` 부터 더해 보세요.
- 2편의 허용 정책에 `ListInferenceProfiles` 를 넣은 이유가 여기 있어요. Claude Code 는 시작할 때 쓸 수 있는 프로필을 찾아보는데, 이 권한이 없으면 확인 없이 접두어를 붙여서 부르다 400 이 나요.
- 웹 검색 도구는 Bedrock 에서 안 돼요. Claude Code 의 `WebSearch` 는 Anthropic 직접 API 기능이라 Bedrock 에서는 빠져요.
- 팀에 배포할 때는 관리형 설정(managed settings)에 `availableModels` 와 `enforceAvailableModels` 를 넣어 쓸 수 있는 모델을 묶을 수 있어요. 2편의 모델 허용목록과 같은 목록으로 맞추면 됩니다.

<br>

<br>



## 7. 세팅하면서 고민한 것들

**① 코드를 얼마나 줄일까, 에러 표는 어디에 둘까**

첫 응답까지 가는 길이 짧아야 끝까지 따라오기 쉬워요. 그래서 처음엔 코드 다섯 줄에 설명 세 줄로 끝내려고 했어요. 그런데 다섯 줄짜리 최소 코드는 Opus 5.5 에서 바로 깨져요(`content[0]`, `temperature`). 게다가 첫 호출이 실패하는 건 대개 IAM·양식·구독·접두어처럼 코드 바깥 문제라서, 표가 없으면 코드를 그대로 베끼고도 막혀요. 에러 문구를 복사해서 검색해 들어오는 경우가 많다는 점도 생각했어요. 그래서 코드는 최소로 두되 깨지는 두 줄만 챙기고, 에러 표는 첫 열을 '메시지·증상' 으로 잡아 접지 않고 본문에 넣었어요.

**② Anthropic 이 권하는 Mantle 클라이언트를 쓸까**

공식 권장이고 SDK 가 SigV4 서명까지 해 주니 개발하는 입장에서는 끌려요. 반대로 보안 쪽에서 보면 mantle 은 호출 로그가 안 남고 IAM 접두어도 달라서 2편에서 일부러 막아 둔 경로예요. 예제 하나 때문에 그 차단을 풀게 만들 수는 없었어요. 그래서 같은 Messages API 를 runtime 의 `/anthropic` 경로로 부르고, mantle 클라이언트는 경고로만 언급했어요.

**③ Claude Code 기본 모델을 Opus 로 둘까**

코딩 품질만 보면 Opus 가 나아요. 그런데 운영해 보면 기본값이 Opus 라는 걸 모르는 팀이 많고, 청구서를 받고 나서야 놀라는 일이 제일 흔해요. 그래서 어떤 모델을 기본으로 쓸지는 팀마다 정하면 되지만, **세 모델은 반드시 명시적으로 고정**하기로 했어요. 2편의 예산 알람도 같이 걸어 두세요.

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
다음 글에서는 같은 계정에서 OpenAI GPT 를 붙여요. openai SDK 를 그대로 쓰는 법, 이번 편의 Converse 호출을 함수 하나로 묶어 GPT 와 Claude 를 같이 부르는 법, 그리고 Codex 가 서울에서 안 되는 이유와 GPT-6 Astra 의 비용 함정까지 다룰게요.

---

**← 이전 글:** [(2/4) 첫날 계정·권한 세팅](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)  
**다음 글 →:** [(4/4) OpenAI GPT 붙이기 — openai SDK·단일 함수·Codex 와 비용 함정](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/)
