---
layout: single
title:  "(2/4) Bedrock 첫날 계정·권한 세팅 — 모델 액세스, 장기 키 차단, 예산"
date: 2026-10-25 09:10:00 +0900
categories: coding
tag: [Bedrock, AWS, IAM, SCP, Marketplace, API키, 모델액세스, Budgets, 보안, 초기세팅]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-account/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-account/header.svg
description: "Bedrock 의 모델 액세스는 이제 '신청' 이 아니라 '첫 호출' 이에요. 첫 호출이 곧 계약 동의이고 Marketplace 구독이 자동으로 시작되죠. 그래서 코드보다 계정이 먼저예요. 전용 계정·SSO·결제수단, Anthropic 사용 사례 양식, 서울 Global 프로필용 허용 정책, 장기 API 키·mantle 차단 정책, 리전 제한 SCP 의 'unspecified' 함정, 예산 알람까지 첫날 기본 세트로 정리했어요."
series: bedrock-gpt-claude-setup
series_order: 2
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="2" %}

## Summary

[1편](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)에서 기본값을 정했어요. 서울에서 호출하고, `bedrock-runtime` 을 쓰고, 모델 ID 는 `global.` 로 시작한다. 이번 편은 그 호출이 **막히지 않고, 새지 않는** 계정 상태를 만드는 편이에요.

코드보다 계정이 먼저인 이유가 있어요. 2025년 10월부터 Bedrock 은 모델을 미리 "신청" 받지 않아요. **누군가 처음 호출하는 순간** Marketplace 구독이 자동으로 시작되고, 그 호출이 곧 모델 사용 약관(EULA) 동의가 돼요. 즉 권한을 열어둔 채 아무나 한 번 호출하면, 그걸로 계약이 체결돼요.

그래서 이번 편 순서는 **막아두고 → 결정하고 → 하나씩 연다** 예요.

> 💡 **이 글에서 다루는 것**
> - 첫날 기본 세트 vs 팀이 커지면 할 것
> - 계정 바닥 다지기 — 전용 계정, 루트 MFA, SSO, 결제수단
> - 모델 액세스의 새 구조 — 자동 구독, 15분, Anthropic 사용 사례 양식
> - 개발자 허용 정책 — 서울 Global 프로필의 ARN 세 겹
> - 차단 정책 — 장기 API 키, mantle, 모델 허용목록
> - 리전 제한 SCP 를 쓰는 조직의 `"unspecified"` 함정
> - 예산 알람 — "Bedrock" 으로 필터하면 안 되는 이유

<br>

<br>



## 1. 첫날 기본 세트 vs 나중에 할 것

[1편 토론 ④](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)에서 정한 기준(되돌릴 수 없거나 돈이 새는 것만 첫날에)을 실제 할 일로 옮기면 이 표예요.

| 언제 | 할 일 |
|---|---|
| **첫날** | 전용 계정·루트 MFA·SSO<br>결제수단 등록<br>차단 정책(4절) 붙이기<br>예산 알람(7절) |
| **모델 열기 직전** | 데이터 경계 결정<br>Anthropic 사용 사례 양식<br>허용 정책(5절) |
| **팀이 커지면** | Organizations·SCP<br>호출 로그·Guardrails<br>VPC 엔드포인트 |

VPC 엔드포인트나 Guardrails 는 나중에 붙여도 아무것도 잃지 않지만, 새어 나간 장기 키나 한 달 치 Opus 청구서는 되돌릴 수 없어요.

<br>

<br>



## 2. 계정 바닥 다지기

### 2-1. Bedrock 전용 계정을 하나 파세요

혼자 쓰든 다섯 명이 쓰든, 인프라 담당의 첫 권고는 **기존 계정에 섞지 말라** 였어요.

- **청구서가 갈라져요.** Claude·GPT 같은 3사 모델은 AWS Marketplace 로 청구돼요. 기존 계정 청구서에 섞이면 나중에 갈라내기 어려워요.
- **예산이 계정 단위로 걸려요.** 태그를 빼먹어도 계정 예산은 전부 잡아요.
- **사고 반경이 줄어요.** 키가 새도 피해가 그 계정 안에서 끝나요.

Claude Code 공식 문서도 같은 이유로 전용 계정을 권해요.

### 2-2. 루트 MFA, 사람은 SSO, 액세스 키는 0개

- 루트 계정에는 MFA 를 걸고 평소엔 쓰지 않아요.
- 사람은 **IAM Identity Center(SSO)** 로 로그인해요. `aws sso login` 한 번이면 로그인 세션이 유지되는 동안 boto3 와 Claude Code 가 역할의 임시 자격증명을 알아서 받아 갱신해요. 역할 자격증명 하나의 수명은 권한 세트의 세션 시간(기본 1시간, 최대 12시간)이에요.
- **IAM 사용자의 장기 액세스 키는 만들지 않아요.** 깃에 올라간 키로 남의 계정에서 LLM 을 대량 호출하는 사고(LLMjacking)는 이미 흔한 이야기예요. 청구서는 Marketplace 로 날아오고요.

```shell
aws configure sso --profile bedrock-dev
aws sso login --profile bedrock-dev
export AWS_PROFILE=bedrock-dev
export AWS_REGION=ap-northeast-2
aws sts get-caller-identity
```

마지막 줄의 결과가 **의도한 계정·역할**로 나오는지 꼭 보세요. 여러 계정을 쓰는 분은 여기서 엉뚱한 계정으로 진행하는 일이 의외로 많아요.

### 2-3. 결제수단

3사 모델은 Marketplace 구독이라 **유효한 결제수단**이 없으면 자동 구독이 실패해요. 증상은 그냥 `AccessDeniedException` 이라서 권한 문제로 착각하기 쉬워요. 첫날 확인하고 넘어가세요.

<br>

<br>



## 3. 모델 액세스는 이제 '첫 호출' 이에요

### 3-1. 바뀐 구조

예전 글들에 나오는 **"Bedrock → Model access → Modify model access"** 화면은 이제 상용 리전에 없어요(GovCloud 전용으로 남았어요). 지금 구조는 이래요.

| 단계 | 일어나는 일 |
|---|---|
| ① 누군가 첫 호출 | 백그라운드로<br>Marketplace 구독 시작 |
| ② 최대 15분 | 호출이 될 때도,<br>안 될 때도 있음 |
| ③ 구독 완료 | 계정 전체에서<br>그 모델 사용 가능 |

여기서 세 가지를 꼭 알아두세요.

- **첫 호출 = 약관 동의.** 3사 모델을 처음 호출하면 그 모델의 EULA 에 동의한 것으로 봐요. 약관 검토가 필요한 조직이라면 **호출을 먼저 막아두고**, 검토가 끝난 뒤에 여는 게 AWS 의 권고예요.
- **처음 15분은 들쭉날쭉해요.** 구독이 확정되는 동안 호출이 성공했다가 실패하기도 해요. 첫 성공을 보고 넘어갔다가 15분 뒤 실패를 보면 엉뚱한 곳을 디버깅하게 돼요.
- **한 리전에서 구독되면 다른 리전도 열려요.** "서울에서만 열었다" 는 성립하지 않아요.

자동 구독이 성공하려면 첫 호출을 하는 사람에게 Marketplace 권한이 있어야 해요. 아래 정책을 그 사람(보통 관리자)에게만 줍니다. 조건 `aws:CalledViaLast` 는 "Bedrock 이 대신 부를 때만" 이라는 뜻이라, 이 권한으로 다른 Marketplace 상품을 사지는 못해요.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAutoSubscribe",
      "Effect": "Allow",
      "Action": [
        "aws-marketplace:Subscribe",
        "aws-marketplace:ViewSubscriptions"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:CalledViaLast": "bedrock.amazonaws.com"
        }
      }
    }
  ]
}
```

> 💡 구독은 계정 단위라 **한 번만** 성공하면 돼요. 관리자가 모델마다 첫 호출을 한 번씩 해두면, 개발자 역할에는 Marketplace 권한을 줄 필요가 없어요. (AWS 문서 설명 기준이에요. Claude Code 공식 IAM 예시는 이 권한을 넣어 두니, 개발자 쪽에서 Marketplace 관련 `AccessDenied` 가 나면 `aws-marketplace:ViewSubscriptions` 부터 더해 보세요.)

### 3-2. Claude 만의 추가 단계 — 사용 사례 양식

Anthropic 모델은 `bedrock-runtime` 으로 부르기 전에 **First Time Use(FTU) 양식**을 한 번 내야 해요. 회사 이름, 웹사이트, 사용 목적을 적는 짧은 양식이에요. 개인 개발자는 회사 사이트 대신 GitHub 프로필이나 포트폴리오 주소를 적어도 된다고 AWS 문서에 나와 있어요.

- **콘솔:** Bedrock → Model catalog → Anthropic 모델 선택 → 플레이그라운드에서 양식 제출.
- **CLI:** `put-use-case-for-model-access` 에 JSON 을 base64 로 넣어요.

```shell
FORM=$(jq -nc '{
  companyName: "Example Studio",
  companyWebsite: "https://example.com",
  intendedUsers: "0",
  industryOption: "Technology",
  otherIndustryOption: "",
  useCases: "Internal coding assistant"
}' | base64 | tr -d '\n')

aws bedrock put-use-case-for-model-access \
  --form-data "$FORM"
```

`intendedUsers` 는 `0` 이 내부 사용자, `1` 이 외부, `2` 가 둘 다예요. AWS CLI v2 는 이런 바이너리 인자를 base64 로 받는 게 기본이라 위처럼 인코딩해서 넘겨요.

몇 가지 함정이 있어요.

- **Organizations 를 쓴다면 관리계정에서 한 번** 내면 멤버 계정이 물려받아요. 이때는 `bedrock:PutUseCaseForModelAccess` 권한이 필요해요.
- **바로 안 열릴 수 있어요.** 제출이 성공해도 계약이 `PENDING` 으로 남는 동안은 호출이 계속 `AccessDeniedException` 이에요. AWS 가 완료 메일을 보내요.
- **GPT 는 되는데 Claude 만 막힌다면** 십중팔구 이 양식이에요.

기다리지 말고 상태를 직접 물어보면 돼요.

```shell
aws bedrock get-foundation-model-availability \
  --model-id anthropic.claude-opus-5-5 \
  --query agreementAvailability.status \
  --output text
```

`AVAILABLE` 이면 열린 거예요. `PENDING` 은 처리 중, `NOT_AVAILABLE` 은 아직 구독이 없는 상태, `ERROR` 면 `errorMessage` 에 이유가 나와요.

<br>

<br>



## 4. 차단 정책 — 먼저 막아두기

순서상 허용보다 차단이 먼저예요. 아래 정책 하나에 첫날 막아야 할 것 다섯 가지를 담았어요. Organizations 를 쓰면 **SCP** 로, Organizations 가 없다면 같은 JSON 을 **SSO 권한 세트의 인라인 정책**(또는 권한 경계)으로 붙이면 돼요. SSO 사용자는 IAM 그룹에 속하지 않아서 IAM 그룹 정책은 적용되지 않아요. (SCP 는 관리계정 자신에게는 적용되지 않는다는 점도 기억해 두세요.)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyLongTermKeysRuntime",
      "Effect": "Deny",
      "Action": "bedrock:CallWithBearerToken",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "bedrock:bearerTokenType": "LONG_TERM"
        }
      }
    },
    {
      "Sid": "DenyLongTermKeysMantle",
      "Effect": "Deny",
      "Action": "bedrock-mantle:CallWithBearerToken",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "bedrock-mantle:bearerTokenType": "LONG_TERM"
        }
      }
    },
    {
      "Sid": "DenyCreatingBedrockLongTermKeys",
      "Effect": "Deny",
      "Action": "iam:CreateServiceSpecificCredential",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "iam:ServiceSpecificCredentialServiceName":
            "bedrock.amazonaws.com"
        }
      }
    },
    {
      "Sid": "DenyMantleUntilNeeded",
      "Effect": "Deny",
      "Action": "bedrock-mantle:*",
      "Resource": "*"
    },
    {
      "Sid": "ModelAllowlist",
      "Effect": "Deny",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "NotResource": [
        "arn:aws:bedrock:*:*:inference-profile/global.anthropic.claude-*-5-5",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*-5-5",
        "arn:aws:bedrock:::foundation-model/anthropic.claude-*-5-5",
        "arn:aws:bedrock:*:*:inference-profile/global.openai.gpt-6-astra",
        "arn:aws:bedrock:*::foundation-model/openai.gpt-6-astra",
        "arn:aws:bedrock:::foundation-model/openai.gpt-6-astra"
      ]
    }
  ]
}
```

문장마다 이유가 있어요.

| Sid | 막는 것 | 왜 |
|---|---|---|
| DenyLongTermKeys<br>Runtime·Mantle | 장기 API 키 사용 | 엔드포인트마다<br>액션이 따로라<br>**둘 다** 막아야 함 |
| DenyCreating<br>BedrockLongTermKeys | 장기 키 생성 | 서비스 이름 조건으로<br>Bedrock 만 좁힘 |
| DenyMantle<br>UntilNeeded | mantle 전체 | `bedrock:` 정책이<br>못 덮는 우회로 |
| ModelAllowlist | 목록 밖 모델 호출 | 데이터 경계를<br>정한 모델만 |

### 4-1. 장기 API 키를 왜 막나

Bedrock API 키는 두 종류예요.

| 구분 | 단기 키 | 장기 키 |
|---|---|---|
| 수명 | 최대 12시간<br>(세션이 짧으면<br>그쪽) | 1일 ~<br>**무기한** |
| 묶인 곳 | 만든 사람의 세션 | IAM 사용자 |
| AWS 권고 | 운영용 | 탐색용만 |

장기 키는 콘솔에서 버튼 하나로 만들어지고, 만료를 "없음" 으로 둘 수 있어요. 블로그 예제를 따라 만든 키가 `.zshrc` 나 메신저에 그대로 남는 게 보안 담당이 꼽은 1순위 사고예요. 만료를 없음으로 만든 키는 누가 지우기 전까지 살아 있어요.

이미 만들어 둔 장기 키가 있다면 지금 끄고 지우세요. 목록에서 `ServiceSpecificCredentialId` 를 확인한 뒤 실행해요.

```shell
aws iam list-service-specific-credentials \
  --user-name <사용자> \
  --service-name bedrock.amazonaws.com

aws iam update-service-specific-credential \
  --user-name <사용자> \
  --service-specific-credential-id <ID> \
  --status Inactive

aws iam delete-service-specific-credential \
  --user-name <사용자> \
  --service-specific-credential-id <ID>
```

그래서 이 시리즈의 예제는 **전부 단기 키나 SSO 자격증명**만 써요. 단기 키는 **생성 자체를 막을 수 없다**는 점(로컬에서 서명해 만들어서 CloudTrail 에 생성 기록도 남지 않아요)도 알아두세요. 통제는 "만든 사람의 권한" 으로 해요.

> ⚠️ 첫날은 **장기 키만** 막아요. (단기 키도 생성은 못 막지만 사용은 `bearerTokenType` 이 `SHORT_TERM` 인 Deny 로 막을 수 있어요.) 단기 키까지 막으면 3편의 Anthropic SDK 경로와 4편의 openai SDK 경로가 막혀요. 단기 키는 만든 사람의 권한과 만료를 그대로 물려받으니, SSO 권한만 잘 잡혀 있으면 충분해요.

### 4-2. "Marketplace 구독을 Deny 하면 모델이 막히지 않나요?"

**안 막혀요.** 보안 담당이 콕 집은 착각이에요. `aws-marketplace:Subscribe` 를 Deny 해도 Bedrock 이 백그라운드로 구독을 시도하기 때문에 첫 호출을 막지 못해요. 모델을 막으려면 위처럼 **`bedrock:InvokeModel` 을 모델 ARN 기준으로 Deny** 해야 해요.

gpt-oss 같은 오픈 웨이트 모델은 아예 Marketplace 상품 번호가 없어서 Marketplace 조건으로는 걸 방법도 없어요. 허용목록(`NotResource`)으로 막는 게 일관돼요.

### 4-3. 허용목록에 ARN 이 세 줄씩인 이유

`ModelAllowlist` 에 모델 하나당 ARN 이 세 줄이에요. 오타가 아니에요. 서울에서 `global.` 프로필로 호출하면 Bedrock 은 권한을 **세 번** 확인해요.

| 순서 | 확인하는 ARN |
|---|---|
| ① | 서울의 추론 프로필<br>`inference-profile/global.*` |
| ② | 서울의 기반 모델<br>`ap-northeast-2::`<br>`foundation-model/*` |
| ③ | **리전이 빈** 기반 모델<br>`arn:aws:bedrock:::`<br>`foundation-model/*` |

③이 Global 의 특징이에요. 어느 리전에서 처리될지 모르니 리전 칸이 비어 있어요. 셋 중 하나라도 빠지면 `AccessDeniedException` 이 나요. 와일드카드 `*` 가 빈 리전과도 맞는지 문서에 명확하지 않아서, ③은 따로 적어 두는 쪽을 택했어요.

이 허용목록은 의도한 대로 **목록 밖이면 다 막아요.** 그래서 나중에 이런 일이 생기면 목록에 ARN 을 더해야 해요.

- 비용 태깅용 Application inference profile(`application-inference-profile/...`)을 도입할 때
- Claude Code 가 고정하지 않은 모델(Bedrock 기본 Sonnet 4.5)이나 `us.` 프로필을 부를 때(3편 6절)
- runtime 의 `/anthropic`·`/openai/v1` 경로 호출이 다른 IAM 액션 이름으로 기록될 때 — 첫 호출 뒤 CloudTrail 의 `eventName` 을 한 번 확인해서 목록의 `Action` 을 맞춰 두세요

<br>

<br>



## 5. 허용 정책 — 개발자에게 줄 권한

차단을 깔았으면 이제 개발자 SSO 권한 세트(예: `BedrockDev`)에 줄 허용 정책이에요. 서울 출발 + `global.` 프로필 경유만 열어요. 계정 ID `111122223333` 은 예시예요.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SeoulGlobalProfiles",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:ap-northeast-2:111122223333:inference-profile/global.anthropic.claude-*-5-5",
        "arn:aws:bedrock:ap-northeast-2:111122223333:inference-profile/global.openai.gpt-6-astra"
      ]
    },
    {
      "Sid": "SeoulModelsViaProfile",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-*-5-5",
        "arn:aws:bedrock:ap-northeast-2::foundation-model/openai.gpt-6-astra"
      ],
      "Condition": {
        "ArnLike": {
          "bedrock:InferenceProfileArn":
            "arn:aws:bedrock:ap-northeast-2:111122223333:inference-profile/global.*"
        }
      }
    },
    {
      "Sid": "GlobalModelsViaProfile",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:::foundation-model/anthropic.claude-*-5-5",
        "arn:aws:bedrock:::foundation-model/openai.gpt-6-astra"
      ],
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "unspecified"
        },
        "ArnLike": {
          "bedrock:InferenceProfileArn":
            "arn:aws:bedrock:ap-northeast-2:111122223333:inference-profile/global.*"
        }
      }
    },
    {
      "Sid": "DiscoverProfiles",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListInferenceProfiles",
        "bedrock:GetInferenceProfile",
        "bedrock:GetFoundationModelAvailability"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ShortTermApiKeys",
      "Effect": "Allow",
      "Action": "bedrock:CallWithBearerToken",
      "Resource": "*"
    }
  ]
}
```

- **Converse API 도 이 정책으로 돼요.** Converse 는 따로 액션이 없고 `InvokeModel`(스트리밍은 `InvokeModelWithResponseStream`) 권한으로 동작해요.
- **`ShortTermApiKeys` 는 3편·4편의 단기 토큰 예제용이에요.** 토큰으로 부르는 경로는 이 액션도 확인해요. 장기 키는 4절 Deny 가 이기니까 여기서 열어도 단기 토큰만 통과해요.
- **`DiscoverProfiles` 는 빼지 마세요.** Claude Code 가 시작할 때 쓸 수 있는 프로필을 찾는 데 `ListInferenceProfiles` 를 써요. 없으면 확인 없이 추측한 ID 로 호출하다 400 이 나요(3편).
- GPT 를 `bedrock-runtime` 의 OpenAI 호환 경로(`/openai/v1/responses`)로 부를 때도 같은 `bedrock:InvokeModel` 계열로 인가된다고 보고 썼는데, 이 부분은 공식 문서에서 액션 이름을 명시적으로 확인하지 못했어요. **4편의 첫 호출로 검증**하세요. 막히면 에러 메시지에 필요한 액션 이름이 찍혀요.

<br>

<br>



## 6. 리전 제한 SCP 를 쓰는 조직이라면 — `"unspecified"` 함정

회사 계정에는 "서울·도쿄 말고는 아무것도 못 한다" 같은 리전 제한 SCP 가 걸려 있는 경우가 많아요. 이 상태에서 Global 프로필을 부르면 **조용히 깨져요.**

원인은 4-3의 ③이에요. 리전이 빈 ARN 을 확인할 때 `aws:RequestedRegion` 값이 실제 처리 리전이 아니라 **`"unspecified"`** 로 들어와요. 그래서 이런 SCP 가 ③에서 요청을 거부해요.

```json
{
  "Sid": "DenyOutsideAllowedRegions",
  "Effect": "Deny",
  "NotAction": ["iam:*", "sts:*", "organizations:*"],
  "Resource": "*",
  "Condition": {
    "StringNotEquals": {
      "aws:RequestedRegion": ["ap-northeast-2"]
    }
  }
}
```

고치는 방법은 두 가지예요. 위 예시는 설명용 축약본이에요. (덤으로, 리전 제한 SCP 는 `us-east-1` 로 가는 전역 API 도 막을 수 있어요. Marketplace 자동 구독·Budgets·Cost Explorer 가 그 예라, `NotAction` 예외에 들어 있는지 같이 확인하세요.)

1. 허용 리전 목록에 `"unspecified"` 를 넣는다. 간단하지만 **Bedrock 말고 다른 서비스에도** 영향이 가요.
2. **(권장)** 같은 Deny 문장에 "서울의 Global 프로필을 거친 요청은 예외" 조건을 덧붙인다. 위 SCP 의 `Condition` 블록을 아래로 통째로 바꿔요.

```json
"Condition": {
  "StringNotEquals": {
    "aws:RequestedRegion": ["ap-northeast-2"]
  },
  "ArnNotLike": {
    "bedrock:InferenceProfileArn":
      "arn:aws:bedrock:ap-northeast-2:*:inference-profile/global.*"
  }
}
```

조건이 두 개면 **둘 다 참일 때만** Deny 가 걸려요. 서울 Global 프로필을 거친 ③은 두 번째 조건이 거짓이라 통과하고, 다른 서비스 요청은 이 키가 아예 없어서(없는 키에 `ArnNotLike` 은 참) 예전처럼 막혀요. 서울 밖에서 Global 프로필을 부르는 요청도 프로필 ARN 이 서울이 아니라 막히고요.

> ⚠️ 이 조건은 AWS 문서의 교차 리전 추론 권한 설명을 바탕으로 짠 거예요. **조직 SCP 에 넣기 전에 테스트 계정에서 꼭 먼저 돌려보세요.** Control Tower 를 쓴다면 관리형 SCP 를 손으로 고치지 말고 Customizations for Control Tower 로 넣어야 해요.

반대로 "우리 조직은 Global 을 아예 금지한다" 면, `aws:RequestedRegion` 이 `unspecified` 이고 `bedrock:InferenceProfileArn` 이 `global.*` 인 요청을 Deny 하면 돼요. 1편에서 정한 대로, 이건 **규제 데이터 전용 계정**에만 거는 게 맞아요.

<br>

<br>



## 7. 예산 알람 — "Bedrock" 으로 필터하면 안 돼요

인프라 담당이 꼽은 가장 흔한 함정이에요. Claude·GPT 비용은 Marketplace 로 청구돼서 **Cost Explorer 에 "Amazon Bedrock" 이 아니라 공급자·상품 이름으로** 잡혀요. 예산을 서비스 "Bedrock" 으로 필터하면 정작 큰 돈은 안 잡혀요.

그래서 첫날 예산은 **필터 없는 계정 전체 예산**으로 걸어요. 전용 계정을 판 이유가 여기서 돌아와요.

| 설정 | 값 |
|---|---|
| 예산 유형 | Cost budget, 월간 |
| 범위 | **필터 없음**<br>(계정 전체) |
| 실제 비용 알람 | 50% · 80% · 100% |
| 예측 비용 알람 | 100% 초과 예상 |
| 추가 | Cost Anomaly Detection<br>모니터 1개 |

예측 알람이 특히 중요해요. 4편에서 다룰 GPT-6 Astra 의 272K 토큰 문턱처럼, **한 번에 단가가 두 배가 되는** 구간이 있어서 월말이 아니라 주중에 알아야 해요.

다만 예산 알람에는 한계가 둘 있어요. **비용 데이터는 몇 시간에서 하루 늦게 반영되고, 알람은 지출을 멈추지 않아요.** 새어 나간 키로 밤새 호출되는 사고를 제때 잡으려면 사용량 쪽 알람이 같이 있어야 해요. CloudWatch `AWS/Bedrock` 의 `InputTokenCount`·`OutputTokenCount` 에 "평소의 몇 배" 알람을 하나 걸어두세요. 이건 몇 분 단위로 반응해요(4편 6절).

<br>

<br>



## 8. 🗣️ 토론 노트

**① 장기 API 키를 글에서 소개할 것인가**

- **기획자:** 콘솔에서 제일 쉬운 길이다. 소개는 하되 경고 박스를 달자.
- **보안:** 블로그에 "만드는 법" 이 있으면 독자는 그대로 복사한다. 만드는 법은 빼고, **막는 법과 지우는 법**만 남겨야 한다.
- **결론:** 장기 키 생성 절차는 싣지 않는다. 차단 정책을 첫날 세트에 넣고, 예제는 전부 단기 키·SSO 로 쓴다.

**② `iam:CreateServiceSpecificCredential` 을 통째로 막을 것인가**

- **개발자:** 간단하게 그냥 다 막자.
- **보안:** 통째로 막으면 다른 서비스(예: CodeCommit 의 Git 자격증명)의 서비스 전용 자격증명까지 막힌다. AWS 문서 예시도 이 경고를 단다.
- **결론:** `iam:ServiceSpecificCredentialServiceName` 조건으로 **Bedrock 만** 막는다.

**③ Marketplace 권한을 개발자에게 줄 것인가**

- **개발자:** 첫 호출이 실패하면 막막하다. 그냥 다 주자.
- **보안:** 첫 호출이 곧 약관 동의다. 동의 권한은 결정한 사람에게만 있어야 한다.
- **결론:** 관리자가 허용 모델마다 첫 호출을 한 번 해서 구독을 끝내고, 개발자 역할에는 Marketplace 권한을 주지 않는다.

**④ 1인 개발자에게 SCP 이야기가 필요한가**

- **기획자:** 독자 대부분은 Organizations 를 안 쓴다.
- **인프라:** 회사 계정에서 시작하는 독자에게는 6절이 **첫 호출이 왜 안 되는지의 유일한 답**이다. 증상이 그냥 AccessDenied 라서 혼자서는 못 찾는다.
- **결론:** 6절은 남기되 "리전 제한 SCP 를 쓰는 조직이라면" 으로 범위를 좁혀 표시한다.

<br>

<br>



## 9. 동작 확인 체크리스트

- [ ] `aws sts get-caller-identity` 가 의도한 계정·역할로 나온다
- [ ] 결제수단이 등록돼 있다
- [ ] 차단 정책(4절)이 SCP 또는 IAM 그룹에 붙어 있다
- [ ] IAM 사용자가 있다면 `aws iam list-service-specific-credentials --user-name <사용자> --service-name bedrock.amazonaws.com` 결과가 비어 있다
- [ ] Claude 의 `get-foundation-model-availability` 가 `AVAILABLE`
- [ ] `list-inference-profiles --region ap-northeast-2` 에 `global.anthropic.claude-opus-5-5` 와 `global.openai.gpt-6-astra` 가 보인다
- [ ] 계정 전체 예산과 예측 알람이 걸려 있다

일단 오늘은 여기까지.....  
다음 글에서는 드디어 코드를 써요. 서울에서 `global.anthropic.claude-opus-5-5` 로 첫 응답을 받고, 같은 계정에 Claude Code 를 붙여볼게요. 첫 호출에서 만나는 에러는 표 하나로 모아뒀어요.

---

**← 이전 글:** [(1/4) 전체 지도와 서울 리전의 현실](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)  
**다음 글 →:** [(3/4) Claude 붙이기 — Converse·Anthropic SDK·Claude Code](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)
