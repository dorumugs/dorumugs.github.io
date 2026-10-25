---
layout: single
title:  "(1/5) AWS Bedrock 으로 OpenAI GPT·Claude 쓰기 — 서울 리전에서 먼저 알아둘 것"
date: 2026-10-25 09:00:00 +0900
categories: coding
tag: [Bedrock, AWS, OpenAI, GPT, ChatGPT, Claude, 서울리전, CrossRegionInference, 데이터경계, 초기세팅]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-map/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-map/header.svg
description: "Bedrock 에서 'ChatGPT' 를 쓴다는 건 앱이 아니라 OpenAI GPT 모델 API 를 쓴다는 뜻이에요. 엔드포인트 두 개(bedrock-runtime·bedrock-mantle)의 차이, 서울 리전에서 최신 모델을 Global 추론 프로필로만 호출할 수 있다는 사실과 그에 따른 데이터 경계 문제를 정리하고, 첫 세팅에 쓸 기본값을 제안합니다."
series: bedrock-gpt-claude-setup
series_order: 1
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="1" %}

## Summary

"Bedrock 에서 ChatGPT 를 쓸 수 있나요?" 짧게 답하면 **ChatGPT 앱을 쓰는 건 아니고, OpenAI 의 GPT 모델을 API 로 호출할 수 있어요.** 2026년 4월 AWS 와 OpenAI 가 제휴한 뒤 6월에 GPT-5.5, 7월에 GPT-5.6, 9월에 GPT-6 Astra 가 Bedrock 에 추가됐어요. 같은 계정에서 Claude Opus 5.5 도 호출할 수 있습니다.

첫 호출까지는 확인할 것이 몇 가지 있어요. 콘솔의 "Model access" 페이지는 없어졌고, 엔드포인트는 두 개로 나뉘었습니다. 또 **서울 리전에서는 최신 모델 대부분을 Global 추론 프로필로만 호출할 수 있어요.** 이 점을 모르면 코드가 맞아도 `AccessDeniedException` 이나 `ValidationException` 이 납니다.

세팅할 때는 보안, 개발 편의, 운영 비용 가운데 무엇을 우선할지 정해야 해요. 이 시리즈에는 제가 정한 기본값을 적었고, 판단이 갈리는 부분은 각 편 끝에 따로 모았어요.

> 📅 **기준일 2026-10-25.** AWS·Anthropic·OpenAI 공식 문서(모델 카드, 사용자 가이드)를 기준으로 정리했어요. Bedrock 의 모델·리전 지원은 몇 주 단위로 바뀝니다. 표에 적힌 리전 지원은 **실행 전에 `list-inference-profiles` 로 다시 확인**하세요. 문서끼리 어긋나는 곳은 본문에 따로 표시했어요.

> 💡 **이 글에서 다루는 것**
> - "ChatGPT on Bedrock" 의 정확한 의미 — 되는 것과 안 되는 것
> - 엔드포인트 두 개: `bedrock-runtime` vs `bedrock-mantle`
> - 모델 ID 읽는 법 — 접두어 `global.` · `us.` · 없음
> - 서울 리전에서 모델별로 쓸 수 있는 호출 방식
> - Global 라우팅과 데이터 경계
> - 세팅하면서 판단이 갈린 부분과 기본값

<br>

<br>



## 1. "ChatGPT on Bedrock" 은 정확히 무엇인가

먼저 용어부터 정리할게요. "ChatGPT" 라는 이름은 앱과 모델을 함께 가리켜서 오해가 생기기 쉬워요.

| 생각하는 것 | 실제 |
|---|---|
| ChatGPT 웹·앱 화면 | **안 돼요.**<br>Bedrock 은 API 서비스예요 |
| GPT 모델을<br>코드에서 호출 | **돼요.**<br>GPT-6 Astra, GPT-5.6 등 |
| openai SDK 를<br>그대로 사용 | **돼요.**<br>base URL·키만 교체 |
| Claude 를<br>같은 계정에서 | **돼요.**<br>Opus·Sonnet·Haiku 5.5 등 |

그래서 이 시리즈 제목에는 "ChatGPT" 대신 **"OpenAI GPT"** 라고 썼어요. ChatGPT 로 검색해서 들어왔더라도, Bedrock 에서 쓰는 것은 GPT 모델 API 예요.

2026년 10월 현재 Bedrock 에서 호출할 수 있는 두 회사 모델 가운데 이 시리즈에서 다루는 것은 다음과 같아요.

| 계열 | 모델 | 비고 |
|---|---|---|
| OpenAI | GPT-6 Astra | 2026-09-08 출시<br>1,050,000 토큰 컨텍스트 |
| OpenAI | GPT-5.6<br>Sol · Terra · Luna | 2026-07 출시<br>플래그십·균형·저가 |
| OpenAI | gpt-oss-120b · 20b | 오픈 웨이트<br>Marketplace 구독 불필요 |
| Anthropic | Claude Opus 5.5 | 2026-09-22 출시<br>1M 컨텍스트 |
| Anthropic | Claude Sonnet 5.5<br>· Haiku 5.5 | 일상 작업·저가 |

> ⚠️ **"Bedrock 의 Claude" 와 "Claude Platform on AWS" 는 다른 상품이에요.** Claude Platform on AWS 는 Anthropic 이 직접 운영하는 API 를 AWS 계정으로 청구받아 쓰는 상품입니다. 모델 ID 에 접두어가 없고 엔드포인트도 달라요. 이 시리즈는 **Amazon Bedrock** 만 다룹니다. 두 상품의 예제 코드는 서로 호환되지 않아요.

Bedrock 에서는 모델 기능 가운데 **Bedrock 이 지원하는 것만** 쓸 수 있어요. 예를 들어 Bedrock 의 Claude 에는 Anthropic 직접 API 에 있는 웹 검색 도구, Files API, Anthropic 방식의 Batches, MCP 커넥터가 없습니다. 직접 API 용 예제를 Bedrock 에서 그대로 실행했을 때 400 에러가 난다면, 먼저 이 기능 차이를 의심해 보세요.

<br>

<br>



## 2. 엔드포인트가 두 개예요 — `bedrock-runtime` vs `bedrock-mantle`

예전 글([AWS Bedrock 으로 스트리밍 챗봇 만들기](/coding/AWS_Bedrock_스트리밍_챗봇_아키텍처/))을 쓸 때는 엔드포인트가 사실상 하나였어요. 지금은 두 개이고, **IAM 권한·로그·지원 리전이 서로 달라요.**

| 항목 | bedrock-runtime | bedrock-mantle |
|---|---|---|
| 주소 | `bedrock-runtime.`<br>`{리전}.amazonaws.com` | `bedrock-mantle.`<br>`{리전}.api.aws` |
| IAM 액션 | `bedrock:` | `bedrock-mantle:` |
| 서울 리전 | Global 프로필로 | 문서마다 다름 |
| Guardrails | 지원 | 미지원 |
| 호출 로그 | 지원 | 미지원 |
| 비용 태깅 | 지원 | 미지원 |

처음 시작한다면 **`bedrock-runtime` 만 쓰는 것**을 권해요. 이유는 세 가지입니다.

- mantle 의 IAM 액션 접두어는 `bedrock-mantle:` 이에요. **`bedrock:` 으로 작성한 정책은 mantle 에 적용되지 않습니다.** `bedrock:` 정책으로 모델을 차단해도 mantle 로는 여전히 호출할 수 있어요.
- Claude Opus 5.5 기준으로 mantle 은 Guardrails, 호출 로그(Invocation logging), Application inference profile(비용 태깅)을 지원하지 않아요. 운영에 필요한 세 기능을 모두 쓸 수 없습니다.
- mantle 은 지원 리전도 적어요. AWS 모델 카드에는 Opus 5.5 의 mantle 리전이 `us-east-1`·`ap-southeast-4`·`us-gov-west-1`, GPT-6 Astra 의 mantle 리전이 `us-east-1`·`us-west-2` 로 적혀 있어요. (Anthropic 문서는 서울을 Global 지원 리전으로 적고 있어서 **두 문서가 서로 어긋나요.** 확인되기 전까지는 서울에서 mantle 을 쓸 수 없다고 가정하고 설계하는 편이 안전해요.)

mantle 은 각 회사의 **네이티브 API 형식**(Claude 의 Messages API, OpenAI 의 Responses API)으로 요청을 보내기 위한 엔드포인트예요. 그런데 지금은 `bedrock-runtime` 도 같은 형식을 지원하고, 경로만 다릅니다.

| 쓰고 싶은 API | bedrock-runtime 경로 |
|---|---|
| Claude Messages | `/anthropic/v1/messages` |
| OpenAI Responses | `/openai/v1/responses` |
| OpenAI Chat Completions | `/openai/v1/chat/completions` |
| 두 회사 공통 | Converse API (boto3) |

SDK 코드는 그대로 두고 **엔드포인트 주소만 runtime 으로** 바꾸면 돼요. mantle 은 필요한 기능이 mantle 에만 있을 때 예외로 씁니다. AWS 출시 블로그에 따르면 Codex 가 대표적인 예이고, 자세한 내용은 4편에서 다뤄요.

<br>

<br>



## 3. 모델 ID 읽는 법 — 접두어가 처리 위치를 정해요

Bedrock 의 모델 ID 는 앞에 붙는 접두어에 따라 **요청이 처리되는 위치**가 달라져요. 4절의 서울 리전 내용이 이 구분을 전제로 하니 먼저 정리할게요.

| 형태 | 예시 | 처리 위치 |
|---|---|---|
| 접두어 없음<br>(In-Region) | `anthropic.`<br>`claude-opus-5-5` | 호출한 그 리전 |
| 지역 접두어<br>(Geo) | `us.openai.`<br>`gpt-6-astra` | 그 지역 안의<br>여러 리전 |
| `global.`<br>(Global) | `global.anthropic.`<br>`claude-opus-5-5` | 전 세계 상용 리전<br>중 여유 있는 곳 |

- **In-Region** 은 호출한 리전에서만 처리해요. 처리 위치가 가장 엄격하게 고정되지만 지원 모델이 적어요.
- **Geo** 는 `us.`·`eu.`·`jp.`·`au.` 처럼 지정한 지역 안의 리전에서만 처리해요. 데이터 레지던시 요건이 있을 때 AWS 가 권하는 방식이에요.
- **Global** 은 전 세계 상용 리전 가운데 여유가 있는 곳에서 처리해요. 가용성이 가장 높고 **가격도 가장 낮아요.** Claude 와 GPT-6 Astra 모두 In-Region·Geo 가 Global 보다 **10% 비싸요.**

> ⚠️ 접두어 없이 호출했을 때 `on-demand throughput isn't supported` 라는 400 에러가 나면, 이 모델은 이 리전에서 In-Region 으로 호출할 수 없다는 뜻이에요. 추론 프로필(`global.` 등)이 붙은 모델 ID 로 다시 호출하면 됩니다.

<br>

<br>



## 4. 서울 리전에서 쓸 수 있는 모델

서울(`ap-northeast-2`)에서 모델별로 쓸 수 있는 호출 방식은 다음과 같아요. `bedrock-runtime` 기준이고, 각 모델의 AWS 모델 카드에서 옮겼어요.

| 모델 | In-Region | Geo | Global |
|---|---|---|---|
| Claude Opus 5.5 | ✗ | ✗ | ✓ |
| Claude Sonnet 5.5 | ✗ | ✗ | ✓ |
| Claude Haiku 5.5 | ✗ | ✗ | ✓ |
| Claude Sonnet 5 | ✓ | ✗ | ✓ |
| GPT-6 Astra | ✗ | ✗ | ✓ |

이 표에서 알 수 있는 점은 두 가지예요.

1. **최신 모델은 서울에서 Global 추론 프로필로만 호출할 수 있어요.** Opus 5.5·Sonnet 5.5·Haiku 5.5·GPT-6 Astra 가 모두 해당합니다. 서울에서 쓰려면 모델 ID 가 `global.` 로 시작해야 해요.
2. **서울에서 쓸 수 있는 아시아 Geo 프로필은 없어요.** Opus 5.5 의 Geo 프로필은 `us.`·`eu.`·`jp.`·`au.` 네 개뿐이에요. `jp.` 는 도쿄·오사카에서 호출할 때만 쓸 수 있어서 서울에서는 쓸 수 없어요. 예전 글에서 보이던 `apac.` 접두어는 이 모델들에 없습니다.

GPT-5.6 계열은 출시 당시 미국 리전(`us-east-1`·`us-east-2`·`us-west-2`)에서만 제공됐고, 8월에 `bedrock-runtime` 과 Global CRIS(교차 리전 추론) 지원이 추가됐어요. 다만 서울에서 Global 로 호출할 수 있는지는 이번에 공식 표로 확인하지 못했어요. 그래서 이 시리즈의 GPT 예제는 **서울에서 Global 호출이 확인된 GPT-6 Astra** 로 작성했습니다.

> 💡 **표보다 계정의 실제 목록을 기준으로 삼으세요.** 리전 지원은 수시로 바뀝니다. 아래 명령을 실행하면 내 계정이 지금 서울에서 쓸 수 있는 추론 프로필 목록이 나와요. 위 표는 시간이 지나면 맞지 않을 수 있지만, 이 명령의 결과는 항상 현재 상태를 보여 줍니다.

```shell
aws bedrock list-inference-profiles \
  --region ap-northeast-2 \
  --query "inferenceProfileSummaries[].inferenceProfileId" \
  --output text | tr '\t' '\n' | grep -E 'claude|openai'
```

<br>

<br>



## 5. Global 라우팅과 데이터 경계

보안 검토에서는 이 부분을 정확하게 설명해야 해요. 위험을 과장하면 쓸 수 있는 모델도 쓰지 못하게 되고, 축소하면 규제를 받는 조직이 요건을 놓칠 수 있어요.

공식 문서 기준으로 정리하면 다음과 같아요.

> **요청은 서울 엔드포인트로 들어가지만, 모델 추론은 전 세계 AWS 상용 리전 중 어디에서든 처리될 수 있어요.** 리전 간 전송은 AWS 네트워크 안에서만 이뤄지고 공용 인터넷을 거치지 않으며, 전송 구간은 암호화돼요. CloudTrail·CloudWatch 기록은 호출한 서울 리전에 남아요.

보안 검토 문서에 쓸 표현을 이 기준으로 나누면 다음과 같아요.

| 표현 | 판정 |
|---|---|
| "데이터가 해외로 유출된다" | ✗ 과장 |
| "서울에서만 처리된다" | ✗ 사실과 다름 |
| "국외 리전에서<br>처리될 수 있다" | ✓ 정확 |

실제 처리 리전은 사후에 확인할 수 있어요. 교차 리전 추론 요청도 **호출한 리전(서울)의 CloudTrail 에 기록되고**, 이벤트의 `additionalEventData.inferenceRegion` 필드에 실제 처리 리전이 남습니다. 따라서 보안 검토에서 요청이 처리된 리전을 기록으로 제시할 수 있어요.

반면 아직 단정할 수 없는 부분도 있어요. **처리한 리전에 데이터가 저장되는지(at-rest)** 는 이번에 확인한 공식 문서에 명시돼 있지 않았어요. 그래서 이 글에서는 "저장되지 않는다" 고 쓰지 않았습니다. Anthropic 은 Bedrock 의 Claude 에 대해 "Anthropic 직원은 추론 인프라에 접근하지 않는다(zero operator access)" 고 밝히고 있어요. 다만 이는 **Anthropic 측의 설명**이라는 점을 감안하세요.

### 개인정보·신용정보·전자금융 규제가 있는 조직이라면

> ⚖️ 아래는 법률 자문이 아니에요. 최종 판단은 법무·준법감시 부서의 몫이에요.

Global 라우팅은 개인정보보호법의 국외 이전·처리위탁 조항, 신용정보법, 전자금융감독규정, 금융권 클라우드 이용 절차(중요도 평가·보고)에 해당할 수 있어요. 따라서 **검토 대상**으로 올려야 합니다. 금융권이라면 **망분리 규제**도 함께 확인해야 해요. 업무망에서 외부 생성형 AI 를 호출하려면 별도 예외 절차가 필요할 수 있어요. 검토할 수 있는 선택지는 세 가지입니다.

| 선택지 | 쓸 수 있는 모델 | 대가 |
|---|---|---|
| ① 민감정보를 빼고<br>Global 허용 | 최신 전부 | 마스킹 게이트를<br>앱에 둬야 함 |
| ② 서울 In-Region<br>모델만 | Sonnet 5 등<br>일부 | 최신 모델<br>포기 |
| ③ 다른 나라 리전의<br>In-Region | — | 역시 국외 처리라<br>해결책이 아님 |

①을 고른다면 민감정보 제외를 사람의 주의에 맡기지 말고 **코드로** 강제해야 해요. 예전에 쓴 [문서 속 개인정보 탐지 게이트](/coding/Airflow_개인정보_탐지_게이트/)처럼 모델 호출 앞단에 마스킹 단계를 두는 방식입니다.

<br>

<br>



## 6. 세팅하면서 판단이 갈린 부분

이 편의 기본값을 정할 때 판단이 갈린 네 가지를 정리했어요.

**① 제목에 'ChatGPT' 를 쓸 것인가**

검색어로는 ChatGPT 가 더 많이 쓰여요. 하지만 제목에 ChatGPT 만 쓰면 ChatGPT 앱을 Bedrock 에서 쓸 수 있다고 오해할 수 있어요. 반대로 제목에서 빼면 검색으로 찾기 어려워집니다. 그래서 'ChatGPT' 는 태그와 description 첫 문장에 넣고, 제목에는 정확한 표현인 "OpenAI GPT" 를 썼어요. 앱을 기대한 독자에게는 첫 문단에서 "앱이 아니라 모델 API" 라고 바로 밝히는 편이 낫다고 판단했습니다.

**② 서울 In-Region 을 강제할 것인가**

보안 관점에서는 데이터 경계가 불분명하면 우선 차단하는 것이 원칙이에요. 하지만 서울에는 최신 모델의 In-Region 프로필도, 아시아 Geo 프로필도 없어요. 따라서 In-Region 을 강제하면 사실상 최신 모델을 쓸 수 없습니다. 그러면 누군가 미국 리전의 mantle 로 우회할 수 있고, **데이터는 미국에서 처리되면서 Guardrails·호출 로그도 쓸 수 없는** 더 나쁜 상태가 돼요. 그래서 차단 기준을 모델이 아니라 **데이터 분류**로 정했어요. 기본값은 **서울 리전 호출 + `global.`** 로 두고, 규제 데이터를 다루는 워크로드만 별도 계정으로 분리해 그 계정에서 Global 을 SCP 로 차단합니다. 이 판단 근거는 한 쪽 분량의 문서로 남겨 두기로 했어요.

**③ mantle 을 소개할 것인가**

개발자 입장에서 mantle 은 네이티브 SDK 를 그대로 쓸 수 있다는 장점이 있어요. 하지만 IAM 접두어가 달라서 정책에 빈틈이 생기기 쉽고, 호출 로그도 남지 않아요. 게다가 runtime 도 같은 API 형식을 지원하므로 처음부터 mantle 을 쓸 이유가 거의 없어요. 그래서 이 글에는 비교표 하나만 두고 기본 엔드포인트는 runtime 으로 정했어요. mantle 은 **기본적으로 차단**하고, Codex 처럼 꼭 필요할 때만 예외로 허용합니다(2편·4편).

**④ 첫날에 어디까지 할 것인가**

VPC 엔드포인트, Guardrails, 호출 로그 파이프라인은 각각 글 한 편 분량이에요. 모두 첫날 작업에 넣으면 첫 응답을 받기까지의 절차가 지나치게 길어집니다. 반면 장기 키 차단과 예산 알람은 뺄 수 없었어요. 키가 하나만 유출돼도 짧은 시간에 큰 비용이 청구될 수 있기 때문이에요. 그래서 **"미루면 되돌릴 수 없거나 비용 손실로 이어지는 작업만 첫날에 한다"** 를 기준으로 삼았어요. 장기 키 차단·예산·모델 허용목록은 첫날 작업에 넣고, VPC 엔드포인트·Guardrails 는 나중 작업으로 미뤘어요.

<br>

<br>



## 7. 제가 정한 기본값과 로드맵

앞의 판단을 거쳐 정한 기본값을 표 하나로 모았어요. 2~4편에서는 이 기본값을 차례로 실제 설정에 반영하고, 5편에서는 기존 코드를 Bedrock 으로 옮기는 방법을 다룹니다.

| 항목 | 기본값 |
|---|---|
| 호출 리전 | 서울 `ap-northeast-2` |
| 엔드포인트 | `bedrock-runtime` |
| 모델 ID | `global.` 접두어 |
| 사람 인증 | SSO(IAM Identity Center)<br>+ 단기 자격증명 |
| 장기 API 키 | 차단 |
| mantle | 차단, 필요 시 예외 |
| 비용 | 계정 단위 예산 알람 |

- **[2편](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/) — 첫날 계정·권한.** 모델 액세스가 '신청' 이 아니라 '첫 호출' 이 된 구조, Anthropic 사용 사례 양식, 허용·차단 정책 JSON, 리전 제한 SCP 의 함정, 예산 알람.
- **[3편](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/) — Claude 붙이기.** Converse 와 Anthropic SDK 두 경로, 첫 호출 에러 표, 서울에서 Claude Code 를 연결할 때 생기는 `apac.` 접두어 문제.
- **[4편](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/) — GPT 붙이기.** openai SDK 를 그대로 쓰는 법, 두 모델을 함수 하나로 부르기, Codex 가 서울 대신 미국 리전을 호출하는 이유, 272K 기준과 출력 10배 쿼터 차감.
- **[5편](/coding/Bedrock_GPT_Claude_첫세팅_기존코드_이전/) — 기존 코드 옮기기.** OpenAI·Anthropic API 를 직접 쓰던 코드의 호출 지점 목록 만들기, SDK 전환, Bedrock 에 없는 기능과 대안, 임베딩 재색인, 비용 추적 변경, 단계적 전환과 롤백.

> 🚀 **급하면 이 순서만** — ① 결제수단 등록 → ② SSO 로그인 → ③ 2편의 차단 정책 붙이기 → ④ Claude 모델 카탈로그에서 사용 사례 양식 제출 → ⑤ 3편의 Converse 예제를 `global.` 모델 ID 로 실행.

일단 오늘은 여기까지.....  
다음 글에서는 코드를 쓰기 전에 계정부터 정리할게요. **첫 호출이 곧 계약 동의**로 처리되기 때문에, 순서를 지키지 않으면 되돌리기 번거로워요.

---

**다음 글 →:** [(2/5) 첫날 계정·권한 세팅 — 모델 액세스, 장기 키 차단, 예산](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)
