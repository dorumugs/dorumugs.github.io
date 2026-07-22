---
layout: single
title:  "(1/2) CDN 이 뭐길래 — 엣지 캐싱부터 CloudFront 실전까지"
date: 2026-07-17 11:00:00 +0900
categories: coding
tag: [CDN, CloudFront, AWS, 엣지캐싱, edge, 캐시, cache-control, TTL, S3, OAC, invalidation, HTTPS, WAF, 웹성능]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-17-cdn-cloudfront/header.svg
  teaser: /assets/images/2026-07-17-cdn-cloudfront/header.svg
description: "CDN 이 대체 무엇을 푸는 기술인지 원리부터 짚고, AWS CloudFront 로 실제 어떻게 붙이는지까지 한 번에 정리했어요. 엣지 로케이션·캐시 히트/미스·TTL·캐시 키·S3 오리진 + OAC·무효화(invalidation)·서명 URL·WAF·비용까지 — '원본은 하나, 캐시는 전 세계'라는 한 문장을 실무 감각으로 풀어봤습니다."
series: cdn-cloudfront
series_order: 1
series_title: "🌐 CDN 부터 CloudFront 엣지까지"
---

{% include series-cdn-cloudfront.html current="1" %}

## Summary

웹사이트나 API 를 만들다 보면 어느 순간 이런 말을 듣게 돼요. "이거 CDN 태워." 저도 처음엔 그냥 "빠르게 해주는 거" 정도로만 알고 넘어갔는데, 막상 직접 붙이려니까 **엣지 로케이션**, **캐시 히트**, **TTL**, **무효화** 같은 단어가 우수수 쏟아지더라고요.

그래서 이 글에서는 CDN 이 **대체 무슨 문제를 푸는 기술인지** 원리부터 짚고, 그다음 AWS 의 CDN 인 **CloudFront** 를 실제로 어떻게 붙이는지까지 이어서 정리해볼게요. 개념만 붕 뜬 채로 끝내지 않고, S3 오리진에 CloudFront 를 얹는 실전 설정까지 같이 봅니다.

> 💡 **이 글에서 다루는 것**
> - CDN 이 푸는 문제 — 원본 하나로 전 세계 서빙하면 왜 느린가
> - 핵심 원리: 엣지 로케이션 · 캐시 히트/미스 · TTL · 캐시 키
> - CloudFront 가 뭔가 — AWS 의 CDN, 구성요소(Distribution / Origin / Behavior)
> - 실전: S3 를 오리진으로 CloudFront 붙이기 (OAC 로 버킷 잠그기)
> - 캐시 정책과 무효화(invalidation) — 배포했는데 옛날 파일이 계속 뜰 때
> - 보안: HTTPS · 서명 URL · Geo 제한 · WAF
> - 비용은 어떻게 나오나

<br>

<br>



## 1. CDN 이 푸는 문제 — 거리가 곧 지연이다

먼저 CDN 없이 사이트를 서빙한다고 생각해볼게요. 서울에 서버(원본, **오리진**)를 한 대 두고 전 세계에 서비스합니다.

- 서울 사용자 → 서울 서버: 가까우니까 빠릅니다.
- 브라질 상파울루 사용자 → 서울 서버: 지구 반 바퀴를 돌아야 해요.

문제는 **빛의 속도는 못 이긴다**는 거예요. 상파울루에서 서울까지 물리적 왕복만 해도 300ms 안팎이 그냥 깔립니다. 이미지 하나, JS 파일 하나 받을 때마다 이 왕복이 쌓이면 페이지가 눈에 띄게 굼떠요.

거리만 문제가 아니에요. 전 세계 트래픽이 **서버 한 대로 몰리면** 그 서버가 병목이 됩니다. 똑같은 로고 이미지를 100만 명한테 100만 번 원본에서 꺼내 보내는 건 낭비죠.

> ⚠️ 정리하면 원본 하나로 전 세계를 서빙할 때 걸리는 건 셋이에요.
> **① 물리적 거리(지연)**, **② 원본 서버 부하**, **③ 같은 파일의 반복 전송 낭비**.

CDN(Content Delivery Network)은 이 셋을 **"사용자 가까이에 복사본을 미리 깔아둔다"** 는 한 방법으로 동시에 풉니다.

<br>

<br>



## 2. 핵심 원리 — 엣지에 캐시를 둔다

CDN 회사는 전 세계 주요 도시에 **엣지 로케이션(edge location)**, 다른 말로 **PoP(Point of Presence)** 라 부르는 서버 무리를 깔아둡니다. 사용자는 원본까지 가지 않고, **자기랑 가장 가까운 엣지**에서 콘텐츠를 받아요.

그림으로 보면 이래요.

```text
[사용자(상파울루)] → [상파울루 엣지]  ── 캐시에 있으면 여기서 끝(빠름)
                          │
                          │ 캐시에 없을 때만
                          ▼
                      [서울 오리진]
```

이 흐름에서 두 가지 상황이 갈립니다.

| 상황 | 이름 | 무슨 일이 일어나나 |
|---|---|---|
| 엣지 캐시에 파일이 **있음** | **캐시 히트(Cache Hit)** | 엣지가 바로 응답. 원본까지 안 감. 제일 빠름 |
| 엣지 캐시에 파일이 **없음** | **캐시 미스(Cache Miss)** | 엣지가 원본에서 한 번 받아와 사용자에게 주고, **복사본을 저장** |

핵심은 **미스는 처음 한 번뿐**이라는 거예요. 상파울루의 첫 방문자가 미스를 내면서 원본에서 파일을 끌어오면, 그 파일이 상파울루 엣지에 저장돼요. 그다음 상파울루 사용자들은 전부 히트로 빠르게 받습니다.

> 💡 그래서 CDN 은 **정적 파일(이미지·CSS·JS·동영상·폰트)** 에 특히 효과가 커요. 한 번 캐싱해두면 수백만 명이 같은 복사본을 나눠 쓰니까요. 반대로 매 요청마다 값이 달라지는 응답(로그인한 유저별 대시보드 등)은 캐싱이 까다롭습니다 — 뒤에서 다시 봐요.

<br>

<br>



## 3. 캐시는 언제까지 유효할까 — TTL 과 캐시 키

엣지에 복사본을 저장했다면, **언제까지 그걸 믿고 쓸지**를 정해야 해요. 원본에서 파일이 바뀌었는데 엣지가 옛날 복사본을 계속 주면 곤란하잖아요.

### TTL (Time To Live)

캐시된 복사본의 **유효 기간**이에요. TTL 이 24시간이면, 엣지는 24시간 동안은 원본에 다시 묻지 않고 저장해둔 걸 그대로 줍니다. 24시간이 지나면 "이거 아직 유효해?" 하고 원본에 확인하러 가요.

TTL 은 보통 원본이 내려주는 HTTP 헤더로 정합니다.

```text
Cache-Control: public, max-age=86400
```

- `max-age=86400` → 86400초(24시간) 동안 캐시 OK.
- `public` → CDN 같은 공유 캐시도 저장해도 된다는 뜻.
- 자주 바뀌는 건 `max-age` 를 짧게, 거의 안 바뀌는 건(해시 박힌 파일명 등) 아주 길게.

> ✅ 실무 팁: 파일명에 해시를 박는 방식(`app.9f2a1c.js`)을 쓰면, 내용이 바뀔 때 **파일명 자체가 바뀌므로** TTL 을 1년으로 크게 잡아도 안전해요. 새 배포는 새 파일명 → 새 캐시. 이걸 "cache busting" 이라고 불러요.

### 캐시 키 (Cache Key)

엣지는 "이 요청이 캐시에 있는 그 요청과 같은가?" 를 **캐시 키**로 판단해요. 기본은 보통 **URL(경로 + 쿼리스트링)** 이에요.

여기서 흔한 함정이 하나 있어요. 캐시 키에 **불필요한 요소가 들어가면 히트율이 떨어져요.**

- 예를 들어 광고 추적용 `?utm_source=...` 쿼리까지 캐시 키에 넣으면, 같은 이미지인데 `utm_source` 값마다 다른 캐시로 취급돼서 미스가 폭증해요.
- 반대로 언어별로 다른 내용을 주는데 `Accept-Language` 를 캐시 키에서 빼면, 영어 사용자가 한국어 페이지를 받는 사고가 나요.

그래서 **"응답을 실제로 달라지게 하는 요소만"** 캐시 키에 넣는 게 정석이에요. CloudFront 에선 이걸 캐시 정책(Cache Policy)에서 세밀하게 고릅니다.

<br>

<br>



## 4. CloudFront 등장 — AWS 의 CDN

여기까지가 CDN 일반론이고, 이제 AWS 의 CDN 인 **CloudFront** 로 내려와 볼게요. 원리는 위에서 본 그대로예요. AWS 가 전 세계 수백 개 도시에 엣지 로케이션을 깔아뒀고, CloudFront 는 그 위에서 도는 캐싱·전송 서비스예요.

CloudFront 를 이해할 때 알아야 할 구성요소는 이 정도예요.

| 용어 | 뜻 |
|---|---|
| **Distribution(배포)** | CloudFront 설정의 최상위 단위. 도메인(`d1234.cloudfront.net`) 하나가 발급됨 |
| **Origin(오리진)** | 원본이 되는 곳. S3 버킷, ALB, EC2, 심지어 외부 HTTP 서버도 가능 |
| **Behavior(동작)** | 경로 패턴(`/images/*` 등)별로 "어떤 오리진으로 보낼지, 어떻게 캐싱할지" 규칙 |
| **Cache Policy** | 캐시 키에 뭘 넣을지 + TTL 정책 |
| **Edge Location** | 실제 캐시가 저장되고 사용자가 접속하는 전 세계 지점 |

흐름을 한 줄로 요약하면 이래요.

> 사용자 → (가장 가까운) **CloudFront 엣지** → 캐시 미스면 → **Behavior 규칙에 따라 Origin** 에서 가져와 캐싱 → 응답

<br>

<br>



## 5. 실전 — S3 를 오리진으로 CloudFront 붙이기

가장 흔한 조합인 **S3 정적 파일 + CloudFront** 를 예로 볼게요. 목표는 이거예요.

- S3 버킷은 **직접 공개하지 않는다**(퍼블릭 차단).
- 오직 **CloudFront 를 통해서만** 파일에 접근하게 한다.
- 사용자는 `https://cdn.example.com/logo.png` 같은 깔끔한 주소로 받는다.

### 5-1. 예전 방식(OAI) 말고 OAC 를 쓰세요

S3 버킷을 CloudFront 에만 열어주는 방법으로 예전엔 **OAI(Origin Access Identity)** 를 썼는데, 지금은 **OAC(Origin Access Control)** 가 권장이에요. OAC 가 SSE-KMS 암호화 버킷도 지원하고 서명 방식도 최신이라 새로 만든다면 무조건 OAC 예요.

CLI 로 OAC 를 하나 만들어 둡니다.

```shell
aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "s3-oac-example",
    "OriginAccessControlOriginType": "s3",
    "SigningBehavior": "always",
    "SigningProtocol": "sigv4"
  }'
```

### 5-2. Distribution 만들 때 핵심 설정

콘솔이든 CLI 든 배포를 만들 때 신경 쓸 포인트만 짚으면 이래요.

- **Origin domain**: 대상 S3 버킷 (반드시 REST 엔드포인트 형태 `버킷명.s3.리전.amazonaws.com` — 웹사이트 엔드포인트 아님)
- **Origin access**: 방금 만든 OAC 연결
- **Viewer protocol policy**: `Redirect HTTP to HTTPS` (HTTP 로 와도 HTTPS 로 돌려보냄)
- **Cache policy**: 정적 파일이면 AWS 관리형 `CachingOptimized` 로 시작
- **Alternate domain(CNAME)**: `cdn.example.com` 을 쓰려면 여기 등록 + ACM 인증서 필요

### 5-3. S3 버킷 정책 — CloudFront 만 허용

OAC 를 만들면 S3 버킷 정책에 "이 CloudFront 배포만 읽기 허용" 을 박아야 해요. 이렇게 생겼어요.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontOACRead",
      "Effect": "Allow",
      "Principal": { "Service": "cloudfront.amazonaws.com" },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-static-bucket/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::<ACCOUNT_ID>:distribution/<DISTRIBUTION_ID>"
        }
      }
    }
  ]
}
```

이러면 S3 버킷은 **퍼블릭 차단을 그대로 켜둔 채** 오직 그 CloudFront 배포만 파일을 읽을 수 있어요. 누가 S3 URL 을 직접 알아내도 `403` 이 떨어집니다.

> 🚨 버킷을 실수로 퍼블릭하게 열어두는 사고가 정말 흔해요. CloudFront + OAC 조합의 진짜 이점 중 하나가 **버킷을 잠근 채로 서빙**한다는 거예요. 원본 접근 경로를 하나로 좁히면 그만큼 사고 표면이 줄어요.

<br>

<br>



## 6. 배포했는데 옛날 파일이 떠요 — 무효화(Invalidation)

CloudFront 를 붙이고 나서 가장 자주 겪는 일이에요. 새 파일을 S3 에 올렸는데, 브라우저엔 **옛날 파일이 계속** 떠요. 당연해요 — 엣지가 TTL 이 살아있는 캐시를 그대로 주고 있는 거예요.

푸는 방법은 둘이에요.

**방법 A — 파일명 버저닝 (권장)**
앞에서 말한 해시 파일명(`app.9f2a1c.js`)이에요. 내용이 바뀌면 파일명이 바뀌니까 캐시 문제가 **원천적으로 없어요.** HTML 처럼 파일명을 못 바꾸는 것만 짧은 TTL 로 두면 됩니다.

**방법 B — 무효화(Invalidation)**
특정 경로의 캐시를 **강제로 버리라**고 CloudFront 에 명령해요.

```shell
aws cloudfront create-invalidation \
  --distribution-id <DISTRIBUTION_ID> \
  --paths "/index.html" "/css/*"
```

이러면 해당 경로의 엣지 캐시가 무효화되고, 다음 요청은 미스가 나서 원본에서 새로 받아와요.

> ⚠️ 무효화는 **공짜가 아니에요.** 매달 경로 1,000개까지는 무료지만 그 이상은 경로당 과금돼요. 그래서 배포할 때마다 `/*` 로 전체 무효화를 날리는 습관은 비추예요. 가능하면 **방법 A(파일명 버저닝)** 로 무효화 자체를 안 하게 설계하는 게 정석입니다.

<br>

<br>



## 7. 동적 콘텐츠와 캐시 안 하기

"CDN 은 정적 파일용" 이라고 했지만, CloudFront 는 **동적 요청도 통과**시킬 수 있어요. 이때 중요한 건 **캐싱하면 안 되는 걸 실수로 캐싱하지 않는 것** 이에요.

- 로그인 후 유저별 응답, 장바구니, API 결과 등은 **캐싱 금지 대상**이 많아요.
- 원본에서 `Cache-Control: no-store` 또는 `private` 를 내려주거나,
- CloudFront 에서 해당 경로 Behavior 에 **`CachingDisabled`** 관리형 정책을 걸어요.

> 🚨 가장 무서운 버그: **유저 A 의 로그인 응답이 캐싱돼서 유저 B 에게 그대로 전달**되는 사고. 개인정보 유출로 직결돼요. 인증이 걸린 경로는 반드시 캐시 키에 인증 요소를 넣거나 아예 캐싱을 끄세요. "정적이면 캐싱, 개인화면 no-store" 를 경로별로 명확히 가르는 게 핵심이에요.

그러면서도 동적 요청을 CloudFront 로 태우는 이유는 있어요 — **엣지까지의 구간이 AWS 백본망**이라 일반 인터넷보다 안정적이고, TLS 종료를 엣지에서 하니 핸드셰이크가 사용자 가까이서 끝나거든요. "캐시는 안 되지만 전송은 빠르게" 가 되는 거예요.

<br>

<br>



## 8. 보안 — HTTPS · 서명 URL · Geo 제한 · WAF

CloudFront 는 캐싱뿐 아니라 **경계 보안 계층**으로도 일을 많이 해요.

- **HTTPS 무료·자동** — ACM(AWS Certificate Manager)에서 인증서를 발급받아 붙이면 끝. `Redirect HTTP to HTTPS` 로 평문 접속을 막아요.
- **서명 URL / 서명 쿠키(Signed URL / Cookie)** — 유료 콘텐츠나 비공개 파일을 **시간 제한부 링크**로만 접근하게 해요. 링크에 만료 시각과 서명이 박혀서, 시간이 지나면 죽어요.
- **Geo 제한(Geo Restriction)** — 특정 국가만 허용/차단. 라이선스나 규제 대응에 써요.
- **AWS WAF 연동** — SQL 인젝션·봇·비정상 트래픽을 엣지에서 걸러내요. 원본까지 도달하기 전에 엣지에서 막으니 원본 부하도 줄고요.

> 💡 여기에 더해 CloudFront 자체가 **L3/L4 DDoS 를 흡수**하는 방패 역할도 해요(AWS Shield Standard 가 기본 포함). 전 세계 엣지에 트래픽이 분산되니 볼류메트릭 공격이 한 원본으로 몰리지 않아요.

<br>

<br>



## 9. 비용은 어떻게 나오나

CloudFront 요금은 크게 세 갈래예요. 세부 단가는 리전·구간마다 다르니 **구조만** 감 잡아두세요.

| 항목 | 과금 방식 |
|---|---|
| **데이터 전송(Out)** | 엣지 → 사용자로 나간 트래픽. 가장 큰 비중. 지역(리전 그룹)별 단가 다름 |
| **요청 수** | HTTP/HTTPS 요청 1만 건 단위 과금 |
| **무효화** | 매달 경로 1,000개 무료, 초과분 경로당 과금 |

여기서 알아둘 포인트 하나 — **S3 → CloudFront 로 나가는 트래픽(오리진 인출)은 무료**예요. 즉 S3 를 직접 공개해서 서빙하는 것보다 **CloudFront 를 앞에 두는 게 데이터 전송비가 더 싼 경우가 많아요.** 캐시 히트가 높을수록 원본 인출이 줄어드니 더 그렇고요.

> ✅ 비용 최적화 = **캐시 히트율 올리기**예요. 히트율이 높으면 원본 인출·원본 부하·전송비가 다 같이 내려가요. 캐시 키를 깔끔하게 유지하고, TTL 을 넉넉히 주고, 파일명 버저닝을 쓰는 게 결국 성능이자 비용 최적화예요.

<br>

<br>



## 10. 정리 — 한 문장으로

CDN 은 결국 **"원본은 하나, 캐시는 전 세계 엣지에"** 라는 한 문장이에요.

- 사용자는 **가까운 엣지**에서 받아 빠르고,
- **캐시 히트**면 원본까지 안 가서 원본 부하가 줄고,
- 그 위에 **HTTPS·WAF·DDoS 흡수** 같은 경계 보안까지 얹혀요.

CloudFront 는 이 원리를 AWS 위에서 구현한 것뿐이고, 실무의 90% 는 **① 캐시할 것/안 할 것을 경로별로 잘 가르기**, **② 캐시 키를 깔끔하게 유지하기**, **③ 파일명 버저닝으로 무효화를 안 하게 설계하기** 이 셋에서 갈려요. 개념만 잡혀 있으면 콘솔 설정은 그 개념에 이름표를 붙이는 일이라 금방 익숙해져요.

일단 오늘은 여기까지.....  
다음 글에서는 CloudFront Functions / Lambda@Edge 로 엣지에서 요청을 직접 주무르는 이야기를 정리해볼게요.

---

**다음 글 →:** [(2/2) CloudFront Functions vs Lambda@Edge — 엣지에서 요청을 주무르기](/coding/CloudFront_Functions_Lambda_Edge_엣지에서_요청주무르기/)
