---
layout: single
title:  "(2/2) CloudFront Functions vs Lambda@Edge — 엣지에서 요청을 주무르기"
date: 2026-07-18 10:30:00 +0900
categories: coding
tag: [CloudFront, CloudFront-Functions, Lambda@Edge, AWS, 엣지컴퓨팅, edge, URL리라이트, 보안헤더, A/B테스트, 인증, 서버리스, 웹성능]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-18-cloudfront-functions-lambda-edge/header.svg
  teaser: /assets/images/2026-07-18-cloudfront-functions-lambda-edge/header.svg
description: "CloudFront 에서 캐싱만으론 안 되는 '요청을 엣지에서 직접 주무르는' 영역을 CloudFront Functions 와 Lambda@Edge 로 갈라 정리했어요. 요청·응답의 네 실행 지점(Viewer/Origin Request·Response), 둘의 성능·제약·비용 차이, URL 리라이트·보안 헤더·리다이렉트·인증·A/B 테스트 같은 실전 예시와 '가벼운 건 Functions, 무거운 건 Lambda@Edge' 라는 선택 기준까지 한 번에 봅니다."
---

{% include series-cdn-cloudfront.html current="2" %}

## Summary

[지난 글](/coding/CDN이란_CloudFront_엣지캐싱/)에서 CDN 과 CloudFront 의 원리 — 원본은 하나, 캐시는 전 세계 엣지에 — 를 잡았어요. 그런데 실무를 하다 보면 **캐싱만으로는 안 되는** 순간이 와요. "모바일이면 다른 페이지로 보내줘", "응답에 보안 헤더 좀 박아줘", "이 경로는 로그인 안 했으면 막아줘" 같은 요구요.

이걸 원본 서버까지 안 가고 **엣지에서 코드로 직접 처리**하는 게 CloudFront 의 두 도구, **CloudFront Functions** 와 **Lambda@Edge** 예요. 이름이 비슷해서 헷갈리는데, 사실 **쓰임새가 꽤 달라요.** 이 글에서 어디서 실행되고, 뭐가 다르고, 언제 뭘 골라야 하는지 정리해볼게요.

> 💡 **이 글에서 다루는 것**
> - 왜 엣지에서 코드를 돌리나 — 원본까지 안 가고 처리하기
> - 요청/응답의 네 실행 지점 (Viewer/Origin × Request/Response)
> - CloudFront Functions — 초경량 JS, viewer 단계 전용
> - Lambda@Edge — 무겁고 강력, 네 지점 모두
> - 한눈에 비교표 + 선택 기준
> - 실전 예시: URL 리라이트 · 보안 헤더 · 리다이렉트 · 인증 · A/B
> - 주의점 — 제약·전파 지연·비용·디버깅

<br>

<br>



## 1. 왜 엣지에서 코드를 돌리나

지난 글의 흐름을 다시 떠올려볼게요.

```text
사용자 → CloudFront 엣지 → (미스면) Origin
```

이 사이 어딘가에서 **아주 작은 코드**를 끼워넣을 수 있으면 뭐가 좋을까요?

- **원본까지 안 가도 돼요** — "http 로 오면 https 로 돌려보내기" 같은 건 원본이 할 필요가 없어요. 엣지가 사용자 코앞에서 바로 처리하면 왕복 한 번이 통째로 빠져요.
- **모든 요청에 일관되게 적용돼요** — 보안 헤더 추가 같은 걸 원본 앱마다 심는 대신 엣지 한 곳에서 처리하면 깔끔해요.
- **원본 부하가 줄어요** — 봇 차단·리다이렉트를 엣지에서 쳐내면 원본은 진짜 일만 해요.

그래서 CloudFront 는 요청이 지나가는 길목에 **코드 실행 지점**을 열어뒀어요. 이 지점이 어디냐가 두 도구를 가르는 핵심이에요.

<br>

<br>



## 2. 네 개의 실행 지점

CloudFront 를 지나는 한 번의 요청/응답에는 코드를 끼울 수 있는 **네 지점**이 있어요.

```text
사용자 ──①──▶ [엣지] ──③──▶ Origin
         ◀──④── [엣지] ◀──②── (응답)
```

| 지점 | 이름 | 언제 실행되나 |
|---|---|---|
| ① | **Viewer Request** | 사용자 요청이 엣지에 **막 도착**했을 때 (캐시 조회 전) |
| ③ | **Origin Request** | 캐시 미스라 **원본으로 보내기 직전** |
| ② | **Origin Response** | 원본이 준 응답이 엣지에 **막 도착**했을 때 (캐시 저장 전) |
| ④ | **Viewer Response** | 엣지가 사용자에게 **응답을 돌려주기 직전** |

포인트는 ①④ (**Viewer** 단계)는 **모든 요청마다** 실행되고, ②③ (**Origin** 단계)은 **캐시 미스일 때만** 실행된다는 거예요. 캐시 히트면 원본 근처엔 가지도 않으니까요.

> 💡 이 구분이 성능·비용에 직결돼요. Viewer 단계 코드는 **트래픽 전량**을 타니 아주 가벼워야 하고, Origin 단계 코드는 미스일 때만 도니 상대적으로 여유가 있어요.

<br>

<br>



## 3. CloudFront Functions — 초경량, viewer 전용

**CloudFront Functions** 는 이름 그대로 CloudFront 에 내장된 초경량 함수예요. 특징을 한 줄로 하면 **"엣지 로케이션 자체에서 도는, 극도로 빠르고 싼 JS"** 예요.

- **실행 지점**: ① Viewer Request, ④ Viewer Response — **딱 두 곳** (viewer 단계만)
- **언어**: JavaScript (ECMAScript 5.1 기반, 전용 런타임)
- **성능**: 실행 시간 **1ms 미만** 목표. 엣지 로케이션 수백 곳에서 그대로 실행
- **제약**: 네트워크 호출 불가, 파일시스템 없음, 실행 시간 극히 짧음, 메모리 2MB
- **비용**: 아주 쌈 (호출 100만 건당 몇 센트 수준)

그래서 CloudFront Functions 는 **"들어오고 나가는 순간에 헤더·URL 을 손보는"** 가벼운 일에 딱이에요. 무거운 로직은 못 담아요 — 애초에 그러라고 만든 게 아니에요.

<br>

<br>



## 4. Lambda@Edge — 무겁고 강력, 네 지점 모두

**Lambda@Edge** 는 우리가 아는 그 **AWS Lambda 를 엣지(정확히는 리전 엣지 캐시)에서 돌리는** 거예요. CloudFront Functions 보다 훨씬 무거운 일을 할 수 있어요.

- **실행 지점**: ①②③④ **네 곳 모두**
- **언어**: Node.js, Python
- **성능**: 수십~수백 ms 급. 엣지 로케이션이 아니라 **리전 엣지 캐시**에서 실행돼서 Functions 보단 사용자에게서 살짝 멀어요
- **할 수 있는 것**: **외부 네트워크 호출 가능**(DB·API·S3 조회), 더 큰 메모리, 더 긴 실행 시간
- **비용**: Functions 보다 비쌈 (Lambda 호출 + 실행 시간 과금)

즉 **"엣지에서 진짜 로직을 돌려야 할 때"** — 외부 인증 서버에 물어본다든가, 요청 내용 보고 원본을 갈아끼운다든가 — 는 Lambda@Edge 예요.

> ⚠️ Lambda@Edge 함수는 **`us-east-1`(버지니아 북부)** 에 배포해야 해요. 거기서 만들면 CloudFront 가 전 세계 엣지로 복제(replicate)해줘요. 다른 리전에서 만들면 CloudFront 에 못 붙습니다 — 처음에 다들 여기서 한 번 막혀요.

<br>

<br>



## 5. 한눈에 비교 + 선택 기준

| 기준 | CloudFront Functions | Lambda@Edge |
|---|---|---|
| 실행 지점 | Viewer Req/Res (2곳) | 네 지점 모두 |
| 언어 | JS (전용 런타임) | Node.js / Python |
| 실행 위치 | 엣지 로케이션 | 리전 엣지 캐시 |
| 실행 시간 | 1ms 미만 | 수십~수백 ms |
| 외부 네트워크 | ❌ 불가 | ✅ 가능 |
| 비용 | 매우 저렴 | 상대적으로 비쌈 |
| 적합한 일 | 헤더/URL 손보기, 리다이렉트 | 인증, 외부 조회, 콘텐츠 가공 |

고르는 기준은 사실 단순해요.

> ✅ **"헤더·URL 만 손보고, 외부 호출이 없다"** → **CloudFront Functions**.
> **"외부에 물어봐야 하거나, Origin 단계에서 개입해야 한다"** → **Lambda@Edge**.
>
> 둘 다 되면 **더 싸고 빠른 Functions** 를 먼저 고르세요. Functions 로 안 되는 순간에만 Lambda@Edge 로 올라가는 게 정석이에요.

<br>

<br>



## 6. 실전 예시 A — CloudFront Functions

### 6-1. URL 리라이트 (SPA/디렉토리 인덱스)

정적 사이트에서 `/blog/` 로 들어오면 실제 파일 `/blog/index.html` 을 찾게 경로를 고쳐주는 흔한 패턴이에요. **Viewer Request** 에 붙입니다.

```javascript
function handler(event) {
  var request = event.request;
  var uri = request.uri;

  // 디렉토리로 끝나면 index.html 붙이기
  if (uri.endsWith('/')) {
    request.uri += 'index.html';
  } else if (!uri.includes('.')) {
    // 확장자가 없으면 (예: /about) index.html 로
    request.uri += '/index.html';
  }
  return request;
}
```

### 6-2. 보안 헤더 추가

응답에 보안 헤더를 일괄로 박아요. **Viewer Response** 에 붙입니다.

```javascript
function handler(event) {
  var response = event.response;
  var headers = response.headers;

  headers['strict-transport-security'] = { value: 'max-age=63072000; includeSubDomains; preload' };
  headers['x-content-type-options']    = { value: 'nosniff' };
  headers['x-frame-options']           = { value: 'DENY' };
  headers['referrer-policy']           = { value: 'strict-origin-when-cross-origin' };
  return response;
}
```

원본 앱을 하나도 안 건드리고 모든 응답에 보안 헤더가 붙어요. 이런 게 Functions 의 진가예요.

<br>

<br>



## 7. 실전 예시 B — Lambda@Edge

### 7-1. 요청 헤더 기반 A/B 라우팅

쿠키를 보고 원본 경로를 갈아끼워 A/B 테스트를 하는 예시예요. **Origin Request**(미스일 때만 도니 비용 절약)에 붙입니다.

```python
def handler(event, context):
    request = event['Records'][0]['cf']['request']
    headers = request['headers']

    # 쿠키에 experiment=B 가 있으면 B 버전 경로로
    cookie = headers.get('cookie', [{}])[0].get('value', '')
    if 'experiment=B' in cookie:
        request['uri'] = '/b' + request['uri']

    return request
```

### 7-2. 엣지 인증 (Basic Auth / 토큰 검증)

스테이징 사이트를 막거나, 요청 토큰을 외부/자체 로직으로 검증할 때예요. **Viewer Request** 에 붙여서 통과 못 하면 바로 `401` 을 돌려줘요.

```python
def handler(event, context):
    request = event['Records'][0]['cf']['request']
    headers = request['headers']

    auth = headers.get('authorization', [{}])[0].get('value')
    if auth != 'Basic <BASE64_CREDENTIALS>':   # 실제 값은 시크릿에서 로드
        return {
            'status': '401',
            'statusText': 'Unauthorized',
            'headers': {
                'www-authenticate': [{'key': 'WWW-Authenticate', 'value': 'Basic'}]
            }
        }
    return request
```

> 🚨 자격증명·토큰·시크릿을 **함수 코드에 하드코딩하지 마세요.** 위 `<BASE64_CREDENTIALS>` 는 자리표시예요. 실제로는 Secrets Manager / SSM 에서 읽거나, 서명 검증 방식으로 처리해야 해요. Lambda@Edge 는 환경변수를 못 쓰는 제약도 있어서(전 세계 복제 때문), 시크릿 주입 방식을 미리 정해두는 게 좋아요.

이 밖에 **Origin Response** 단계에서 이미지를 리사이즈해 캐싱하거나, 404 를 예쁜 페이지로 바꾸는 것도 Lambda@Edge 의 단골 용도예요.

<br>

<br>



## 8. 주의점 — 쓰기 전에 알아둘 것

엣지 코드는 편한 만큼 함정도 있어요. 몇 개만 짚을게요.

- **전파(replicate)에 시간이 걸려요** — Lambda@Edge 는 배포/수정하면 전 세계 엣지로 복제되는 데 몇 분 걸려요. 바로 반영 안 된다고 당황하지 마세요.
- **삭제도 바로 안 돼요** — CloudFront 에 연결된 Lambda@Edge 는 복제본이 다 정리될 때까지 **삭제가 지연**돼요(수십 분~몇 시간). 연결을 먼저 끊고 기다려야 해요.
- **디버깅이 번거로워요** — 로그(CloudWatch)가 **함수가 실행된 엣지의 리전**에 남아요. `us-east-1` 만 보면 안 되고, 실제 트래픽이 탄 리전을 찾아야 해요.
- **모든 요청에 타는 코드는 무겁게 만들지 마세요** — 특히 Viewer 단계. 여기에 무거운 로직을 넣으면 트래픽 전량이 느려져요. 무거우면 Origin 단계로 내리거나 캐싱으로 실행 횟수를 줄이세요.
- **비용은 실행 횟수 × 위치** — Viewer 단계는 히트/미스 무관 전량 실행이라 호출 수가 커요. 가벼운 건 Functions 로 내려 단가를 낮추는 게 중요해요.

<br>

<br>



## 9. 정리 — 캐시 너머의 엣지

1편에서 CDN 을 **"원본은 하나, 캐시는 전 세계 엣지에"** 로 정리했다면, 2편은 그 엣지에서 **캐시를 넘어 코드까지 돌린다**는 이야기였어요.

- 요청/응답에는 코드를 끼울 **네 지점**이 있고 (Viewer/Origin × Request/Response),
- **가볍고 외부 호출 없는 일**은 CloudFront Functions,
- **무겁거나 외부에 물어봐야 하는 일**은 Lambda@Edge,
- 둘 다 되면 **더 싸고 빠른 Functions 부터** 고른다.

이걸로 CDN 의 원리(1편)부터 엣지 로직(2편)까지 한 바퀴 돌았어요. 정적 파일을 빠르게 뿌리는 것에서 시작해, 요청을 사용자 코앞에서 직접 주무르는 데까지 — CloudFront 를 "그냥 캐시 서버" 이상으로 쓸 수 있는 그림이 잡혔길 바라요.

일단 오늘은 여기까지.....  
다음엔 이 엣지 위에 WAF·Shield 를 얹어 방어 계층을 두껍게 하는 이야기로 또 찾아올게요.

---

**← 이전 글:** [(1/2) CDN 이 뭐길래 — 엣지 캐싱부터 CloudFront 실전까지](/coding/CDN이란_CloudFront_엣지캐싱/)
