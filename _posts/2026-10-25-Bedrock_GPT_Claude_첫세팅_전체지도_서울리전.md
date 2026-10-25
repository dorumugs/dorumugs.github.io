---
layout: single
title:  "(1/4) AWS Bedrock 으로 OpenAI GPT·Claude 쓰기 — 전체 지도와 서울 리전의 현실"
date: 2026-10-25 09:00:00 +0900
categories: coding
tag: [Bedrock, AWS, OpenAI, GPT, ChatGPT, Claude, 서울리전, CrossRegionInference, 데이터경계, 초기세팅]
author_profile: false
toc: true
header:
  image: /assets/images/2026-10-25-bedrock-setup-map/header.svg
  teaser: /assets/images/2026-10-25-bedrock-setup-map/header.svg
description: "Bedrock 에서 'ChatGPT' 를 쓴다는 건 앱이 아니라 OpenAI GPT 모델 API 를 쓴다는 뜻이에요. 엔드포인트 두 개(bedrock-runtime·bedrock-mantle), 서울 리전에서 최신 모델이 Global 라우팅으로만 열리는 현실, 그게 데이터 경계에서 어떤 의미인지 정리하고, 제가 첫 세팅에 쓰기로 한 기본값까지 소개해요."
series: bedrock-gpt-claude-setup
series_order: 1
series_title: "🔐 AWS Bedrock 으로 OpenAI GPT·Claude 첫 세팅"
---

{% include series-bedrock-gpt-claude-setup.html current="1" %}

## Summary

"Bedrock 에서 ChatGPT 써도 돼요?" 요즘 제일 많이 받는 질문이에요. 짧게 답하면 **ChatGPT 앱이 들어오는 건 아니고, ChatGPT 를 만드는 회사의 GPT 모델을 API 로 부를 수 있어요.** 2026년 4월 AWS 와 OpenAI 가 손을 잡은 뒤로 6월에 GPT-5.5, 7월에 GPT-5.6, 9월에 GPT-6 Astra 가 Bedrock 에 올라왔어요. 같은 계정에서 Claude Opus 5.5 도 부를 수 있고요.

문제는 첫 호출까지 가는 길이에요. 콘솔의 "Model access" 페이지는 사라졌고, 엔드포인트는 두 개로 갈라졌고, **서울 리전에서는 최신 모델 대부분이 'Global' 라우팅으로만 열려요.** 이걸 모르고 들어가면 코드는 맞는데 `AccessDeniedException` 이나 `ValidationException` 만 보게 됩니다.

처음 세팅하는 사람은 대개 이 길에서 한참 헤매요. 세팅하다 보면 보안을 먼저 챙길지, 개발 편의를 먼저 챙길지, 운영 비용을 먼저 볼지 계속 저울질하게 돼요. 이 시리즈에는 그렇게 따져보고 정한 기본값을 적었고, 어느 쪽으로 정할지 오래 고민한 부분은 각 편 끝에 따로 모아뒀어요.

> 📅 **기준일 2026-10-25.** AWS·Anthropic·OpenAI 공식 문서(모델 카드, 사용자 가이드)를 기준으로 정리했어요. Bedrock 은 모델·리전 표가 몇 주 단위로 바뀌어요. 표에 적힌 리전 지원은 **실행 전에 꼭 `list-inference-profiles` 로 다시 확인**하세요. 문서끼리 어긋나는 곳은 본문에 따로 표시해 뒀어요.

> 💡 **이 글에서 다루는 것**
> - "ChatGPT on Bedrock" 의 정확한 의미 — 되는 것과 안 되는 것
> - 엔드포인트 두 개: `bedrock-runtime` vs `bedrock-mantle`
> - 모델 ID 읽는 법 — 접두어 `global.` · `us.` · 없음
> - 서울 리전 지도 — 어떤 모델이 어떤 방식으로 열리나
> - Global 라우팅이 데이터 경계에 뜻하는 것 (과장도 축소도 없이)
> - 세팅하면서 고민한 것들과 제가 정한 기본값

<br>

<br>



## 1. "ChatGPT on Bedrock" 은 정확히 무엇인가

먼저 단어부터 정리할게요. 이 오해 하나 때문에 검색을 헤매는 분이 많아요.

| 생각하는 것 | 실제 |
|---|---|
| ChatGPT 웹·앱 화면 | **안 들어와요.**<br>Bedrock 은 API 서비스예요 |
| GPT 모델을<br>코드에서 호출 | **돼요.**<br>GPT-6 Astra, GPT-5.6 등 |
| openai SDK 를<br>그대로 사용 | **돼요.**<br>base URL·키만 교체 |
| Claude 를<br>같은 계정에서 | **돼요.**<br>Opus·Sonnet·Haiku 5.5 등 |

그래서 이 시리즈에서는 제목을 "ChatGPT" 가 아니라 **"OpenAI GPT"** 로 씁니다. 검색은 ChatGPT 로 하셨더라도 실제로 손에 쥐는 건 GPT 모델 API 예요. 이것부터 맞춰두고 가면 뒤에서 덜 헷갈려요.

지금(2026-10) Bedrock 에서 부를 수 있는 두 회사 모델 중 이 시리즈가 다루는 것만 추리면 이래요.

| 계열 | 모델 | 비고 |
|---|---|---|
| OpenAI | GPT-6 Astra | 2026-09-08 출시<br>1,050,000 토큰 컨텍스트 |
| OpenAI | GPT-5.6<br>Sol · Terra · Luna | 2026-07 출시<br>플래그십·균형·저가 |
| OpenAI | gpt-oss-120b · 20b | 오픈 웨이트<br>Marketplace 구독 불필요 |
| Anthropic | Claude Opus 5.5 | 2026-09-22 출시<br>1M 컨텍스트 |
| Anthropic | Claude Sonnet 5.5<br>· Haiku 5.5 | 일상 작업·저가 |

> ⚠️ **"Bedrock 의 Claude" 와 "Claude Platform on AWS" 는 다른 상품이에요.** 후자는 Anthropic 이 직접 운영하는 API 를 AWS 계정·청구로 쓰는 것이라 모델 ID 에 접두어가 없고 엔드포인트도 달라요. 이 시리즈는 **Amazon Bedrock** 만 다룹니다. 검색하다 섞이면 코드가 안 맞아요.

또 하나 미리 알아둘 점이 있어요. Bedrock 은 모델을 "그대로" 주는 게 아니라 **Bedrock 이 지원하는 기능만** 줘요. 예를 들어 Claude 를 Bedrock 으로 쓰면 Anthropic 직접 API 에 있는 웹 검색 도구·Files API·Anthropic 식 Batches·MCP 커넥터는 없어요. "직접 API 예제를 붙였더니 400" 이 나오면 대개 이 경우예요.

<br>

<br>



## 2. 엔드포인트가 두 개예요 — `bedrock-runtime` vs `bedrock-mantle`

예전 글([AWS Bedrock 으로 스트리밍 챗봇 만들기](/coding/AWS_Bedrock_스트리밍_챗봇_아키텍처/))을 쓸 때는 엔드포인트가 사실상 하나였어요. 지금은 두 개고, **권한·로그·리전이 다 따로 놀아요.**

| 항목 | bedrock-runtime | bedrock-mantle |
|---|---|---|
| 주소 | `bedrock-runtime.`<br>`{리전}.amazonaws.com` | `bedrock-mantle.`<br>`{리전}.api.aws` |
| IAM 액션 | `bedrock:` | `bedrock-mantle:` |
| 서울 리전 | Global 프로필로 | 문서마다 다름 |
| Guardrails | 지원 | 미지원 |
| 호출 로그 | 지원 | 미지원 |
| 비용 태깅 | 지원 | 미지원 |

표는 길어도 할 말은 하나예요. 처음 시작한다면 **`bedrock-runtime` 하나로 가세요.**

- mantle 은 IAM 액션 접두어가 `bedrock-mantle:` 이라서 **`bedrock:` 으로 쓴 정책이 mantle 을 덮어주지 않아요.** 막았다고 생각한 길이 그대로 열려 있게 돼요.
- Claude Opus 5.5 기준으로 mantle 에서는 Guardrails·호출 로그(Invocation logging)·Application inference profile(비용 태깅)이 안 돼요. 운영에 필요한 세 가지가 다 빠집니다.
- mantle 리전도 좁아요. AWS 모델 카드는 Opus 5.5 의 mantle 을 `us-east-1`·`ap-southeast-4`·`us-gov-west-1` 로, GPT-6 Astra 의 mantle 을 `us-east-1`·`us-west-2` 로 적어요. (Anthropic 문서는 서울을 Global 로 적어서 **두 문서가 서로 어긋나요.** 확인 전까지는 서울 mantle 이 없다고 보고 설계하는 게 안전해요.)

그러면 mantle 은 왜 있을까요? 각 회사의 **네이티브 API 모양**(Claude 의 Messages API, OpenAI 의 Responses API)을 그대로 받아주는 길이에요. 그런데 지금은 `bedrock-runtime` 도 같은 모양을 받아줘요. 경로만 다릅니다.

| 쓰고 싶은 API | bedrock-runtime 경로 |
|---|---|
| Claude Messages | `/anthropic/v1/messages` |
| OpenAI Responses | `/openai/v1/responses` |
| OpenAI Chat Completions | `/openai/v1/chat/completions` |
| 두 회사 공통 | Converse API (boto3) |

즉 SDK 코드 모양은 그대로 두고 **주소만 runtime 으로** 두면 돼요. mantle 은 "그 모델이 mantle 에만 있을 때" 꺼내는 예외로 둡니다. 출시 블로그 기준으로 **Codex 가 mantle 을 쓰는** 대표적인 예외예요. 4편에서 다뤄요.

<br>

<br>



## 3. 모델 ID 읽는 법 — 접두어가 라우팅을 정해요

Bedrock 의 모델 ID 는 앞에 무엇이 붙느냐에 따라 **요청이 어디서 처리되는지**가 달라져요. 뒤에 나올 서울 리전 얘기가 전부 여기에 걸려 있어서 먼저 짚고 갈게요.

| 형태 | 예시 | 처리 위치 |
|---|---|---|
| 접두어 없음<br>(In-Region) | `anthropic.`<br>`claude-opus-5-5` | 호출한 그 리전 |
| 지역 접두어<br>(Geo) | `us.openai.`<br>`gpt-6-astra` | 그 지역 안의<br>여러 리전 |
| `global.`<br>(Global) | `global.anthropic.`<br>`claude-opus-5-5` | 전 세계 상용 리전<br>중 여유 있는 곳 |

- **In-Region** 은 호출한 리전에서만 처리해요. 가장 엄격하지만 지원 모델이 적어요.
- **Geo** 는 `us.`·`eu.`·`jp.`·`au.` 같은 지역 안에서만 돌려요. 데이터 레지던시 요건이 있을 때 AWS 가 권하는 방식이에요.
- **Global** 은 전 세계에서 여유 있는 리전으로 보내요. 가용성이 가장 좋고 **가격도 가장 싸요.** Claude 와 GPT-6 Astra 모두 In-Region·Geo 가 Global 보다 **10% 비싸요.**

> ⚠️ 접두어 없이 부르면 되는 줄 알았는데 `on-demand throughput isn't supported` 라는 400 이 나와요. "이 모델은 이 리전에서 In-Region 으로는 안 열린다, 추론 프로필(`global.` 등)로 다시 불러라" 라는 뜻이에요. 첫날 가장 많이 보는 에러예요.

<br>

<br>



## 4. 서울 리전 지도

이제 서울(`ap-northeast-2`)에서 호출했을 때 무엇이 어떻게 열리는지 볼게요. `bedrock-runtime` 기준이고, 각 모델의 AWS 모델 카드에서 옮겼어요.

| 모델 | In-Region | Geo | Global |
|---|---|---|---|
| Claude Opus 5.5 | ✗ | ✗ | ✓ |
| Claude Sonnet 5.5 | ✗ | ✗ | ✓ |
| Claude Haiku 5.5 | ✗ | ✗ | ✓ |
| Claude Sonnet 5 | ✓ | ✗ | ✓ |
| GPT-6 Astra | ✗ | ✗ | ✓ |

여기서 두 가지가 눈에 띄어요.

1. **최신 모델은 서울에서 Global 로만 열려요.** Opus 5.5·Sonnet 5.5·Haiku 5.5·GPT-6 Astra 모두 그래요. 서울에서 쓰려면 모델 ID 가 반드시 `global.` 로 시작해야 합니다.
2. **서울에는 아시아 Geo 프로필이 없어요.** Opus 5.5 의 Geo 프로필은 `us.`·`eu.`·`jp.`·`au.` 네 개뿐이에요. `jp.` 는 도쿄·오사카에서 호출할 때만 쓸 수 있어서 서울 출발로는 못 써요. 옛 글에서 보던 `apac.` 은 이 모델들에 없어요.

GPT-5.6 계열은 출시 때 미국 리전(`us-east-1`·`us-east-2`·`us-west-2`)에만 있었고, 8월에 `bedrock-runtime` 과 Global CRIS(교차 리전 추론)가 추가됐어요. 다만 서울 출발 Global 이 열렸는지는 이번에 공식 표로 확인하지 못했어요. 그래서 이 시리즈의 GPT 예제는 **서울에서 Global 로 열린 것이 확인된 GPT-6 Astra** 로 씁니다.

> 💡 **표를 믿지 말고 계정에 물어보세요.** 리전 지원은 수시로 바뀌어요. 첫날 아래 명령 하나로 "내 계정·서울에서 지금 열려 있는 프로필" 을 확인하는 습관을 들이면 위 표가 낡아도 문제가 없어요.

```shell
aws bedrock list-inference-profiles \
  --region ap-northeast-2 \
  --query "inferenceProfileSummaries[].inferenceProfileId" \
  --output text | tr '\t' '\n' | grep -E 'claude|openai'
```

<br>

<br>



## 5. Global 라우팅은 데이터 경계에 무엇을 뜻하나

이 글을 쓰면서 제일 오래 붙잡고 있던 부분이에요. 보안 쪽에서 보면 무엇보다 표현이 정확해야 해요. 겁을 주듯 과장하면 쓸 수 있는 모델도 못 쓰게 되고, 대수롭지 않게 줄여 말하면 규제를 받는 조직이 사고를 내요.

최대한 정확하게 적으면 이래요.

> **요청은 서울 엔드포인트로 들어가지만, 모델 추론(처리)은 AWS 상용 리전 전 세계 어디서든 일어날 수 있어요.** 리전 사이 이동은 AWS 네트워크 안에서만 일어나고 공용 인터넷을 지나지 않으며, 전송 구간은 암호화돼요. CloudTrail·CloudWatch 기록은 호출한 서울 리전에 남아요.

그래서 쓰면 안 되는 표현과 써야 하는 표현이 갈려요.

| 표현 | 판정 |
|---|---|
| "데이터가 해외로 유출된다" | ✗ 과장 |
| "서울에서만 처리된다" | ✗ 사실과 다름 |
| "국외 리전에서<br>처리될 수 있다" | ✓ 정확 |

실제로 어디서 처리됐는지는 사후에 볼 수 있어요. 교차 리전 추론 요청도 **출발 리전(서울)의 CloudTrail 에 남고**, 이벤트의 `additionalEventData.inferenceRegion` 필드에 실제 처리 리전이 찍혀요. 덕분에 보안 검토에서 "어디로 갔는지 모른다" 대신 "기록으로 보여드릴 수 있다" 고 말할 수 있어요.

다만 아직 단정하면 안 되는 것도 있어요. **처리한 리전에 데이터가 저장되는지(at-rest)** 는 이번에 읽은 공식 문서에 명시가 없었어요. "저장되지 않는다" 고 쓰지 않은 이유예요. Anthropic 은 Bedrock 의 Claude 를 두고 "Anthropic 직원은 추론 인프라에 접근하지 않는다(zero operator access)" 고 밝히는데, 이건 **Anthropic 측 설명**이라는 출처를 붙여서 읽으면 돼요.

### 개인정보·신용정보·전자금융 규제가 있는 조직이라면

> ⚖️ 아래는 법률 자문이 아니에요. 최종 판단은 법무·준법감시 부서의 몫이에요.

개인정보보호법의 국외 이전·처리위탁 조항, 신용정보법, 전자금융감독규정과 금융권 클라우드 이용 절차(중요도 평가·보고)에 해당할 수 있으니 **"검토 대상"** 으로 올려야 해요. 금융권이라면 **망분리 규제**도 같이 봐야 해요. 업무망에서 외부 생성형 AI 를 부르는 것 자체가 별도 예외 절차를 거쳐야 하는 경우가 있어요. 실무 선택지는 셋이에요.

| 선택지 | 쓸 수 있는 모델 | 대가 |
|---|---|---|
| ① 민감정보를 빼고<br>Global 허용 | 최신 전부 | 마스킹 게이트를<br>앱에 둬야 함 |
| ② 서울 In-Region<br>모델만 | Sonnet 5 등<br>일부 | 최신 모델<br>포기 |
| ③ 다른 나라 리전의<br>In-Region | — | 그것도 국외라<br>답이 아님 |

①을 고른다면 "민감정보를 안 넣는다" 를 사람의 주의가 아니라 **코드로** 지켜야 해요. 예전에 쓴 [문서 속 개인정보 탐지 게이트](/coding/Airflow_개인정보_탐지_게이트/)처럼 모델 호출 앞단에 마스킹을 두는 식이에요.

<br>

<br>



## 6. 세팅하면서 고민한 것들

이 편에서 정한 것들도 처음부터 답이 있지는 않았어요. 어느 쪽으로 갈지 한참 저울질한 네 가지를 적어둘게요.

**① 제목에 'ChatGPT' 를 쓸 것인가**

사람들이 실제로 검색하는 단어는 ChatGPT 예요. 그런데 제목에 ChatGPT 만 덩그러니 쓰면 앱 화면이 Bedrock 에 들어오는 줄 알고 들어오세요. 그렇다고 제목에서 빼버리면 검색으로 찾아오기가 어려워지고요. 그래서 태그와 description 첫 문장에 'ChatGPT' 를 넣어 검색은 걸리게 하고, 제목에는 오해 없는 "OpenAI GPT" 를 썼어요. 화면을 기대하고 온 분께는 차라리 첫 문단에서 "앱이 아니라 모델 API 예요" 라고 바로 말씀드리는 게 낫다고 봤어요.

**② 서울 In-Region 을 강제할 것인가**

보안 쪽에서 보면 데이터 경계가 흐릿할 때는 일단 막는 게 기본이에요. 그런데 운영해 보면 사정이 달라요. 서울에는 최신 모델의 In-Region 도, 아시아 Geo 도 없어서 In-Region 을 강제하는 건 사실상 "최신 모델 금지" 예요. 그러면 결국 누군가 미국 리전 mantle 로 돌아가게 되고, **데이터는 미국으로 가면서 Guardrails·로그는 잃는** 더 나쁜 상황이 돼요. 그래서 막는 기준을 모델이 아니라 **데이터 분류**로 옮겼어요. 기본값은 **서울 출발 + `global.`** 로 두고, 규제 데이터를 다루는 워크로드만 별도 계정으로 떼어 거기서 Global 을 SCP 로 막아요. 이 판단은 한 쪽짜리 문서로 남겨두기로 했어요.

**③ mantle 을 소개할 것인가**

개발하는 입장에서는 네이티브 SDK 를 그대로 쓸 수 있어서 끌리는 경로예요. 하지만 IAM 접두어가 달라서 정책에 구멍이 생기기 쉽고, 호출 로그도 안 남아요. 게다가 runtime 도 같은 API 모양을 받아주니까 첫날 mantle 을 써서 얻는 건 거의 없어요. 그래서 비교표 한 장만 남기고 기본은 runtime 으로 갔어요. mantle 은 **기본 차단**해 두고, Codex 처럼 꼭 필요할 때만 예외로 열어요(2편·4편).

**④ 어디까지가 "첫날" 인가**

VPC 엔드포인트, Guardrails, 호출 로그 파이프라인은 하나하나가 글 한 편 분량이에요. 다 넣으면 첫 응답까지 가는 길이 오히려 흐려져요. 반면 장기 키 차단과 예산 알람은 뺄 수가 없었어요. 키 하나만 새도 하루 만에 청구서가 터지거든요. 그래서 기준을 **"안 하면 되돌릴 수 없거나 돈이 새는 것만 첫날에 한다"** 로 잡았어요. 장기 키 차단·예산·모델 허용목록은 넣고, VPC 엔드포인트·Guardrails 는 각주로 뺐어요.

<br>

<br>



## 7. 제가 정한 기본값과 로드맵

이렇게 따져보고 남긴 기본값을 한 장으로 모았어요. 시리즈 나머지 세 편은 이 표를 하나씩 실제로 만들어 가는 과정이에요.

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
- **[3편](/coding/Bedrock_GPT_Claude_첫세팅_Claude_호출_ClaudeCode/) — Claude 붙이기.** Converse 와 Anthropic SDK 두 경로, 첫 호출 에러 표, 서울에서 Claude Code 를 붙일 때 `apac.` 함정.
- **[4편](/coding/Bedrock_GPT_Claude_첫세팅_GPT_호출_Codex_비용/) — GPT 붙이기.** openai SDK 를 그대로 쓰는 법, 두 모델을 함수 하나로 부르기, Codex 가 서울에서 안 되는 이유, 272K 문턱과 10배 쿼터 차감.

> 🚀 **급하면 이 순서만** — ① 결제수단 등록 → ② SSO 로그인 → ③ 2편의 차단 정책 붙이기 → ④ Claude 모델 카탈로그에서 사용 사례 양식 제출 → ⑤ 3편의 Converse 예제를 `global.` 모델 ID 로 실행.

일단 오늘은 여기까지.....  
다음 글에서는 코드 한 줄 쓰기 전에 계정부터 정리할게요. **첫 호출이 곧 계약 동의**가 되는 구조라, 순서를 잘못 밟으면 되돌리기가 번거로워요.

---

**다음 글 →:** [(2/4) 첫날 계정·권한 세팅 — 모델 액세스, 장기 키 차단, 예산](/coding/Bedrock_GPT_Claude_첫세팅_계정권한_첫날/)
