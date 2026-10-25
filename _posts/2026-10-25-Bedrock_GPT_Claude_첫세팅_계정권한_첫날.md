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
description: "Bedrock 에서는 이제 모델을 미리 신청하지 않아요. 첫 호출이 약관 동의가 되고, 그 순간 Marketplace 구독이 자동으로 시작됩니다. 그래서 코드보다 계정 설정을 먼저 해야 해요. 전용 계정·SSO·결제수단, Anthropic 사용 사례 양식, 서울 Global 프로필용 허용 정책, 장기 API 키·mantle 차단 정책, 리전 제한 SCP 의 'unspecified' 문제, 예산 알람을 첫날 할 일로 정리했어요."
series: bedrock-gpt-claude-setup
series_order: 2
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="2" %}

## Summary

[1편](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)에서 기본값을 정했어요. 서울에서 호출하고, `bedrock-runtime` 을 쓰고, `global.` 로 시작하는 모델 ID 를 씁니다. 이번 편에서는 이 호출에 필요한 권한만 허용하고, 허용하지 않은 사용자·모델·비용은 막도록 계정을 설정해요.

계정 설정을 코드보다 먼저 하는 이유가 있어요. 2025년 10월부터 Bedrock 은 모델 사용 신청을 따로 받지 않습니다. **누군가 모델을 처음 호출하면** Marketplace 구독이 자동으로 시작되고, 그 호출이 모델 사용 약관(EULA)에 동의한 것으로 처리돼요. 권한이 넓게 허용돼 있으면, 누구든 한 번 호출하는 것만으로 계약이 성립합니다.

그래서 이번 편은 **먼저 차단하고 → 쓸 모델을 정하고 → 필요한 권한만 허용하는** 순서로 진행해요.

> 💡 **이 글에서 다루는 것**
> - 첫날 기본 세트 vs 팀이 커지면 할 것
> - 계정 기본 설정 — 전용 계정, 루트 MFA, SSO, 결제수단
> - 바뀐 모델 액세스 방식 — 자동 구독, 15분, Anthropic 사용 사례 양식
> - 개발자 허용 정책 — 서울 Global 프로필에서 확인하는 ARN 세 개
> - 차단 정책 — 장기 API 키, mantle, 모델 허용목록
> - 리전 제한 SCP 를 쓰는 조직의 `"unspecified"` 문제
> - 예산 알람 — "Bedrock" 으로 필터하면 안 되는 이유

<br>

<br>



## 1. 첫날 기본 세트 vs 나중에 할 것

[1편](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/) 끝에서 정한 기준은 "미루면 되돌릴 수 없거나 비용 손실로 이어지는 작업만 첫날에 한다" 였어요. 이 기준에 따라 할 일을 나누면 다음과 같아요.

| 언제 | 할 일 |
|---|---|
| **첫날** | 전용 계정·루트 MFA·SSO<br>결제수단 등록<br>차단 정책(4절) 적용<br>예산 알람(7절) |
| **모델 허용 직전** | 데이터 경계 결정<br>Anthropic 사용 사례 양식<br>허용 정책(5절) |
| **팀이 커지면** | Organizations·SCP<br>호출 로그·Guardrails<br>VPC 엔드포인트 |

VPC 엔드포인트와 Guardrails 는 나중에 추가해도 문제가 없어요. 반면 유출된 장기 키나 이미 청구된 한 달 치 Opus 비용은 되돌릴 수 없습니다.

<br>

<br>



## 2. 계정 기본 설정

### 2-1. Bedrock 전용 계정 만들기

혼자 쓰든 다섯 명이 쓰든, 기존 계정과 분리된 **Bedrock 전용 계정**을 만드는 것을 권해요.

- **청구서를 따로 볼 수 있어요.** Claude·GPT 같은 서드파티 모델은 AWS Marketplace 로 청구돼요. 기존 계정 청구서에 섞이면 나중에 분리하기 어렵습니다.
- **예산을 계정 단위로 걸 수 있어요.** 리소스에 태그를 빠뜨려도 계정 예산에는 모든 비용이 잡혀요.
- **피해 범위가 줄어요.** 키가 유출돼도 피해가 그 계정 안으로 한정됩니다.

Claude Code 공식 문서도 같은 이유로 전용 계정을 권장해요.

### 2-2. 루트 MFA, 사람은 SSO, 액세스 키는 0개

- 루트 계정에는 MFA 를 설정하고 평소에는 사용하지 않아요.
- 사람은 **IAM Identity Center(SSO)** 로 로그인해요. `aws sso login` 을 한 번 실행하면, 로그인 세션이 유지되는 동안 boto3 와 Claude Code 가 역할의 임시 자격증명을 자동으로 받아 갱신합니다. 임시 자격증명 하나의 유효 시간은 권한 세트에 설정한 세션 시간(기본 1시간, 최대 12시간)이에요.
- **IAM 사용자의 장기 액세스 키는 만들지 않아요.** Git 저장소에 노출된 키를 탈취해 남의 계정으로 LLM 을 대량 호출하는 공격(LLMjacking)이 이미 보고돼 있어요. 이때 비용은 Marketplace 청구서로 계정 주인에게 청구됩니다.

```shell
aws configure sso --profile bedrock-dev
aws sso login --profile bedrock-dev
export AWS_PROFILE=bedrock-dev
export AWS_REGION=ap-northeast-2
aws sts get-caller-identity
```

마지막 명령의 결과에 **의도한 계정과 역할**이 나오는지 확인하세요. 여러 계정을 쓰고 있다면 다른 계정으로 로그인된 상태일 수 있어요.

### 2-3. 결제수단

서드파티 모델은 Marketplace 구독으로 제공되므로, **유효한 결제수단**이 없으면 자동 구독이 실패해요. 이때 오류는 `AccessDeniedException` 으로만 나와서 권한 문제로 오해하기 쉽습니다. 첫날 결제수단이 등록돼 있는지 확인하세요.

<br>

<br>



## 3. 모델 액세스는 이제 첫 호출로 시작돼요

### 3-1. 바뀐 방식

예전 글에 나오는 **"Bedrock → Model access → Modify model access"** 화면은 이제 상용 리전에 없어요(GovCloud 에만 남아 있습니다). 지금은 다음 순서로 진행돼요.

| 단계 | 일어나는 일 |
|---|---|
| ① 누군가 첫 호출 | 백그라운드에서<br>Marketplace 구독 시작 |
| ② 최대 15분 | 호출이 성공하기도,<br>실패하기도 함 |
| ③ 구독 완료 | 계정 전체에서<br>그 모델 사용 가능 |

여기서 알아둘 점이 세 가지 있어요.

- **첫 호출이 약관 동의예요.** 서드파티 모델을 처음 호출하면 그 모델의 EULA 에 동의한 것으로 처리돼요. AWS 는 약관 검토가 필요한 조직이라면 **호출을 먼저 차단하고**, 검토가 끝난 뒤 허용하라고 권고합니다.
- **처음 15분은 결과가 일정하지 않아요.** 구독이 확정되는 동안 호출이 성공했다가 실패하기도 해요. 첫 호출의 성공만 보고 구독이 끝났다고 판단하면, 15분 안에 나는 실패를 다른 원인으로 오해하기 쉽습니다.
- **한 리전에서 구독하면 다른 리전에서도 쓸 수 있어요.** 구독을 서울 리전으로만 한정할 수는 없어요.

자동 구독이 성공하려면 첫 호출을 하는 사람에게 Marketplace 권한이 있어야 해요. 아래 정책은 첫 호출을 맡을 사람(보통 관리자)에게만 부여합니다. 조건 `aws:CalledViaLast` 는 Bedrock 이 대신 요청할 때만 허용한다는 뜻이에요. 따라서 이 권한으로 다른 Marketplace 상품은 구독할 수 없어요.

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

> 💡 구독은 계정 단위라 **한 번만** 성공하면 돼요. 관리자가 모델마다 첫 호출을 한 번씩 해 두면 개발자 역할에는 Marketplace 권한이 필요 없습니다. (AWS 문서 설명 기준이에요. Claude Code 공식 IAM 예시에는 이 권한이 들어 있어요. 개발자 역할에서 Marketplace 관련 `AccessDenied` 가 나면 `aws-marketplace:ViewSubscriptions` 부터 추가해 보세요.)

### 3-2. Claude 에만 있는 단계 — 사용 사례 양식

Anthropic 모델은 `bedrock-runtime` 으로 호출하기 전에 **First Time Use(FTU) 양식**을 한 번 제출해야 해요. 회사 이름, 웹사이트, 사용 목적을 적는 짧은 양식입니다. AWS 문서에 따르면 개인 개발자는 회사 웹사이트 대신 GitHub 프로필이나 포트폴리오 주소를 적어도 돼요.

- **콘솔:** Bedrock → Model catalog → Anthropic 모델 선택 → 플레이그라운드에서 양식 제출.
- **CLI:** `put-use-case-for-model-access` 에 base64 로 인코딩한 JSON 을 넘겨요.

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

`intendedUsers` 값은 `0` 이 내부 사용자, `1` 이 외부 사용자, `2` 가 둘 다예요. AWS CLI v2 는 바이너리 인자를 기본적으로 base64 로 받기 때문에 위처럼 인코딩해서 넘깁니다.

주의할 점은 다음과 같아요.

- **Organizations 를 쓴다면 관리 계정에서 한 번만** 제출하면 멤버 계정에도 적용돼요. 이때 `bedrock:PutUseCaseForModelAccess` 권한이 필요합니다.
- **제출 직후에는 호출이 안 될 수 있어요.** 제출에 성공해도 계약 상태가 `PENDING` 인 동안은 호출할 때마다 `AccessDeniedException` 이 나요. 처리가 끝나면 AWS 가 완료 메일을 보냅니다.
- **GPT 는 호출되는데 Claude 만 거부된다면** 이 양식이 원인일 가능성이 높아요.

메일을 기다리는 대신 CLI 로 상태를 직접 조회할 수 있어요.

```shell
aws bedrock get-foundation-model-availability \
  --model-id anthropic.claude-opus-5-5 \
  --query agreementAvailability.status \
  --output text
```

`AVAILABLE` 이면 호출할 수 있는 상태예요. `PENDING` 은 처리 중, `NOT_AVAILABLE` 은 아직 구독이 없는 상태입니다. `ERROR` 가 나오면 `errorMessage` 에 원인이 적혀 있어요.

<br>

<br>



## 4. 차단 정책 — 허용보다 먼저

허용 정책보다 차단 정책을 먼저 적용해요. 아래 정책 하나에 첫날 막아야 할 다섯 가지를 담았습니다. Organizations 를 쓴다면 **SCP** 로 적용하세요. Organizations 를 쓰지 않는다면 같은 JSON 을 **SSO 권한 세트의 인라인 정책**(또는 권한 경계)으로 붙이면 돼요. SSO 사용자는 IAM 그룹에 속하지 않으므로 IAM 그룹 정책으로는 적용되지 않습니다. (SCP 는 관리 계정 자체에는 적용되지 않는다는 점도 알아 두세요.)

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

각 문장(Statement)이 막는 대상과 이유는 다음과 같아요.

| Sid | 막는 것 | 이유 |
|---|---|---|
| DenyLongTermKeys<br>Runtime·Mantle | 장기 API 키 사용 | 엔드포인트마다<br>액션이 달라<br>**둘 다** 막아야 함 |
| DenyCreating<br>BedrockLongTermKeys | 장기 키 생성 | 서비스 이름 조건으로<br>Bedrock 키만 대상 |
| DenyMantle<br>UntilNeeded | mantle 전체 | `bedrock:` 정책이<br>적용되지 않는 경로 |
| ModelAllowlist | 목록 밖 모델 호출 | 데이터 경계를<br>정한 모델만 허용 |

### 4-1. 장기 API 키를 막는 이유

Bedrock API 키에는 두 종류가 있어요.

| 구분 | 단기 키 | 장기 키 |
|---|---|---|
| 수명 | 최대 12시간<br>(세션이 짧으면<br>그쪽) | 1일 ~<br>**무기한** |
| 묶인 곳 | 만든 사람의 세션 | IAM 사용자 |
| AWS 권고 | 운영용 | 탐색용만 |

장기 키는 콘솔에서 버튼 하나로 만들 수 있고, 만료를 "없음" 으로 설정할 수도 있어요. 이렇게 만든 키가 `.zshrc` 파일이나 메신저 대화에 남으면 유출 위험이 생깁니다. 만료가 없는 키는 누군가 삭제할 때까지 계속 유효해요.

이미 만든 장기 키가 있다면 비활성화한 뒤 삭제하세요. 먼저 목록에서 `ServiceSpecificCredentialId` 를 확인하고 아래 명령을 실행합니다.

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

이 시리즈의 예제는 **모두 단기 키나 SSO 자격증명**만 사용해요. 단기 키는 로컬에서 서명해 만들기 때문에 **생성 자체를 막을 수 없고**, CloudTrail 에 생성 기록도 남지 않습니다. 그래서 단기 키는 키를 만드는 사람의 권한으로 통제해요. 3편·4편 코드에서는 이 키를 토큰 형태로 발급해 쓰기 때문에 '단기 토큰' 이라고 부릅니다.

> ⚠️ 첫날에는 **장기 키만** 차단해요. (단기 키도 생성은 막을 수 없지만, `bearerTokenType` 이 `SHORT_TERM` 인 요청을 Deny 하면 사용은 막을 수 있어요.) 단기 키까지 차단하면 3편의 Anthropic SDK 예제와 4편의 openai SDK 예제가 동작하지 않아요. 단기 키의 권한과 만료 시각은 만든 사람의 세션을 그대로 따르므로, SSO 권한을 제대로 설정해 두면 충분합니다.

### 4-2. "Marketplace 구독을 Deny 하면 모델 호출이 막히지 않나요?"

**막히지 않아요.** `aws-marketplace:Subscribe` 를 Deny 해도 Bedrock 이 백그라운드에서 구독을 시도하므로 첫 호출은 그대로 진행돼요. 모델 호출을 막으려면 위 정책처럼 **모델 ARN 을 기준으로 `bedrock:InvokeModel` 을 Deny** 해야 합니다.

gpt-oss 같은 오픈 웨이트 모델은 Marketplace 상품 ID 가 없어서 Marketplace 조건으로는 제어할 수 없어요. 모든 모델을 허용목록(`NotResource`) 하나로 관리하는 편이 일관됩니다.

### 4-3. 허용목록에 모델마다 ARN 이 세 개인 이유

`ModelAllowlist` 에는 모델 하나당 ARN 이 세 줄씩 들어 있어요. 중복이 아니라 필요한 항목입니다. 서울에서 `global.` 프로필로 호출하면 Bedrock 은 권한을 **세 번** 확인해요.

| 순서 | 확인하는 ARN |
|---|---|
| ① | 서울의 추론 프로필<br>`inference-profile/global.*` |
| ② | 서울의 기반 모델<br>`ap-northeast-2::`<br>`foundation-model/*` |
| ③ | **리전이 빈** 기반 모델<br>`arn:aws:bedrock:::`<br>`foundation-model/*` |

③은 Global 프로필에만 있는 확인 단계예요. 요청이 어느 리전에서 처리될지 미리 알 수 없어서 ARN 의 리전 부분이 비어 있습니다. 셋 중 하나라도 빠지면 `AccessDeniedException` 이 나요. 와일드카드 `*` 가 빈 리전에도 일치하는지 문서에 명확히 나와 있지 않아서, ③은 별도로 적어 두었어요.

이 허용목록은 의도한 대로 **목록에 없는 호출을 모두 막아요.** 따라서 다음과 같은 경우에는 목록에 ARN 을 추가해야 합니다.

- 비용 태깅용 Application inference profile(`application-inference-profile/...`)을 도입할 때
- Claude Code 에 모델을 지정하지 않아 기본 모델(Bedrock 기본 Sonnet 4.5)이 호출되거나 `us.` 프로필을 호출할 때(3편 6절)
- runtime 의 `/anthropic`·`/openai/v1` 경로 호출이 다른 IAM 액션 이름으로 기록될 때 — 첫 호출 뒤 CloudTrail 의 `eventName` 을 확인하고 목록의 `Action` 을 그에 맞게 수정하세요

<br>

<br>



## 5. 허용 정책 — 개발자에게 줄 권한

차단 정책을 적용했다면, 이제 개발자 SSO 권한 세트(예: `BedrockDev`)에 허용 정책을 붙여요. 서울 리전에서 `global.` 프로필을 거치는 호출만 허용합니다. 계정 ID `111122223333` 은 예시예요.

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

- **Converse API 도 이 정책으로 호출할 수 있어요.** Converse 에는 별도 액션이 없고, `InvokeModel`(스트리밍은 `InvokeModelWithResponseStream`) 권한으로 동작합니다.
- **`ShortTermApiKeys` 는 3편·4편의 단기 키 예제에 필요해요.** 키(토큰)로 호출하면 Bedrock 이 이 액션도 확인합니다. 장기 키는 4절의 Deny 가 우선하므로, 여기서 허용해도 단기 키만 통과해요.
- **`DiscoverProfiles` 는 빼지 마세요.** Claude Code 는 시작할 때 `ListInferenceProfiles` 로 사용할 수 있는 프로필을 조회해요. 이 권한이 없으면 조회 없이 추정한 ID 로 호출하다가 400 오류가 납니다(3편).
- GPT 를 `bedrock-runtime` 의 OpenAI 호환 경로(`/openai/v1/responses`)로 호출할 때도 같은 `bedrock:InvokeModel` 계열 권한으로 인가된다고 가정했어요. 다만 이 경로의 액션 이름은 공식 문서에서 명시적으로 확인하지 못했습니다. **4편의 첫 호출로 검증**하세요. 거부되면 오류 메시지에 필요한 액션 이름이 나와요.

<br>

<br>



## 6. 리전 제한 SCP 를 쓰는 조직이라면 — `"unspecified"` 문제

회사 계정에는 "서울·도쿄 외 리전은 사용 금지" 같은 리전 제한 SCP 가 적용된 경우가 있어요. 이 상태에서 Global 프로필을 호출하면 **원인이 잘 드러나지 않는 권한 오류가 나요.**

원인은 4-3의 ③ 단계예요. 리전이 빈 ARN 을 확인할 때 `aws:RequestedRegion` 값은 실제 처리 리전이 아니라 **`"unspecified"`** 로 전달돼요. 그 결과 아래와 같은 SCP 가 ③ 단계에서 요청을 거부합니다.

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

위 예시는 설명을 위해 줄인 거예요. 참고로 리전 제한 SCP 는 `us-east-1` 로 가는 전역 API 도 막을 수 있어요. Marketplace 자동 구독·Budgets·Cost Explorer 가 여기에 해당하므로, 이 서비스들이 `NotAction` 예외에 포함돼 있는지도 확인하세요.

해결 방법은 두 가지입니다.

1. 허용 리전 목록에 `"unspecified"` 를 추가한다. 간단하지만 **Bedrock 외 다른 서비스에도** 영향을 줍니다.
2. **(권장)** 같은 Deny 문장에 "서울의 Global 프로필을 거친 요청은 제외" 조건을 추가한다. 위 SCP 의 `Condition` 블록 전체를 아래 내용으로 바꿔요.

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

한 블록에 조건이 두 개 있으면 **둘 다 참일 때만** Deny 가 적용돼요. 서울 Global 프로필을 거친 ③ 단계 요청은 두 번째 조건이 거짓이므로 통과합니다. 다른 서비스의 요청에는 이 조건 키가 없고, 키가 없으면 `ArnNotLike` 은 참이 돼요. 따라서 다른 서비스는 기존처럼 차단됩니다. 서울 밖에서 Global 프로필을 호출하는 요청도 프로필 ARN 의 리전이 서울이 아니므로 차단돼요.

> ⚠️ 이 조건은 AWS 문서의 교차 리전 추론 권한 설명을 바탕으로 작성했어요. **조직 SCP 에 적용하기 전에 반드시 테스트 계정에서 먼저 확인하세요.** Control Tower 를 쓴다면 관리형 SCP 를 직접 수정하지 말고 Customizations for Control Tower 로 적용해야 합니다.

반대로 조직에서 Global 프로필 사용을 금지하려면, `aws:RequestedRegion` 이 `unspecified` 이고 `bedrock:InferenceProfileArn` 이 `global.*` 인 요청을 Deny 하면 돼요. 1편에서 정한 대로, 이 정책은 **규제 데이터 전용 계정**에만 적용하는 것이 적절합니다.

<br>

<br>



## 7. 예산 알람 — "Bedrock" 서비스로 필터하지 마세요

Claude·GPT 비용은 Marketplace 로 청구되므로 **Cost Explorer 에 "Amazon Bedrock" 이 아니라 공급자·상품 이름으로** 표시돼요. 예산을 서비스 "Bedrock" 으로 필터하면 비용 대부분을 차지하는 모델 사용료가 집계되지 않습니다.

따라서 첫날 예산은 **필터 없이 계정 전체**에 설정해요. 2-1에서 전용 계정을 권장한 이유가 여기에 있어요.

| 설정 | 값 |
|---|---|
| 예산 유형 | Cost budget, 월간 |
| 범위 | **필터 없음**<br>(계정 전체) |
| 실제 비용 알람 | 50% · 80% · 100% |
| 예측 비용 알람 | 100% 초과 예상 |
| 추가 | Cost Anomaly Detection<br>모니터 1개 |

예측 비용 알람이 특히 중요해요. 4편에서 다룰 GPT-6 Astra 의 272K 토큰 기준처럼 **넘으면 입력 단가가 두 배가 되는** 구간이 있기 때문이에요. 이런 비용 증가는 월말이 아니라 그 주 안에 알아야 해요.

다만 예산 알람에는 한계가 두 가지 있어요. **비용 데이터는 몇 시간에서 하루 늦게 반영되고, 알람이 울려도 지출이 멈추지는 않습니다.** 유출된 키로 밤새 호출이 이어지는 상황을 빨리 알아채려면 사용량 알람도 함께 설정해야 해요. CloudWatch `AWS/Bedrock` 의 `InputTokenCount`·`OutputTokenCount` 지표에 평소 사용량보다 몇 배 많을 때 울리는 알람을 걸어 두세요. 이 알람은 몇 분 단위로 반응해요(4편 6절).

<br>

<br>



## 8. 세팅하면서 판단이 갈린 부분

**① 장기 API 키 생성 방법을 넣을까**

콘솔에서 가장 간단하게 만들 수 있는 것이 장기 키라서, 생성 방법을 소개하고 경고만 덧붙이는 방안도 생각했어요. 하지만 생성 절차가 적혀 있으면 독자가 그대로 따라 할 수 있습니다. 그래서 생성 절차는 빼고 차단 방법과 삭제 방법만 남겼어요. 차단 정책은 첫날 할 일에 넣었고, 예제도 모두 단기 키와 SSO 로 작성했어요.

**② `iam:CreateServiceSpecificCredential` 을 전부 막을까**

처음에는 액션 전체를 막는 편이 간단해 보였어요. 하지만 그러면 CodeCommit 의 Git 자격증명처럼 다른 서비스의 서비스 전용 자격증명도 함께 막힙니다. AWS 문서의 예시에도 이 점이 경고로 적혀 있어요. 그래서 `iam:ServiceSpecificCredentialServiceName` 조건으로 Bedrock 자격증명만 막았어요.

**③ 개발자에게 Marketplace 권한도 줄까**

개발자 입장에서는 첫 호출이 거부되면 원인을 찾기 어려우니 권한을 넉넉히 주고 싶을 수 있어요. 하지만 첫 호출이 곧 약관 동의이므로, 동의 권한은 그 결정을 내리는 사람에게만 있어야 합니다. 그래서 관리자가 허용할 모델마다 첫 호출을 한 번씩 해서 구독을 마쳐 두고, 개발자 역할에는 Marketplace 권한을 주지 않았어요.

**④ 개인 사용자에게도 SCP 설명이 필요할까**

이 글의 독자 상당수는 Organizations 를 쓰지 않을 거예요. 하지만 회사 계정에서 시작하는 독자에게는 6절이 첫 호출 실패의 원인을 찾는 데 필요한 정보입니다. 오류가 AccessDenied 로만 나와서 원인을 추적하기 어렵기 때문이에요. 그래서 6절은 남기고, 제목에 "리전 제한 SCP 를 쓰는 조직이라면" 이라고 대상을 밝혀 두었어요.

<br>

<br>



## 9. 동작 확인 체크리스트

- [ ] `aws sts get-caller-identity` 가 의도한 계정·역할로 나온다
- [ ] 결제수단이 등록돼 있다
- [ ] 차단 정책(4절)이 SCP 또는 SSO 권한 세트에 붙어 있다
- [ ] IAM 사용자가 있다면 `aws iam list-service-specific-credentials --user-name <사용자> --service-name bedrock.amazonaws.com` 결과가 비어 있다
- [ ] Claude 모델의 `get-foundation-model-availability` 결과가 `AVAILABLE` 이다
- [ ] `list-inference-profiles --region ap-northeast-2` 결과에 `global.anthropic.claude-opus-5-5` 와 `global.openai.gpt-6-astra` 가 있다
- [ ] 계정 전체 예산과 예측 비용 알람이 설정돼 있다

일단 오늘은 여기까지.....  
다음 글에서는 코드를 작성해요. 서울에서 `global.anthropic.claude-opus-5-5` 로 첫 응답을 받고, 같은 계정에서 Claude Code 를 설정합니다. 첫 호출에서 자주 나오는 오류는 표 하나로 정리해 두었어요.

---

**← 이전 글:** [(1/4) 서울 리전에서 먼저 알아둘 것](/coding/Bedrock_GPT_Claude_첫세팅_전체지도_서울리전/)  
**다음 글 →:** [(3/4) Claude 붙이기 — Converse·Anthropic SDK·Claude Code](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/)
