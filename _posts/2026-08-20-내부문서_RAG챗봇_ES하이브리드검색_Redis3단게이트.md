---
layout: single
title:  "(4/5) 내부문서 RAG 챗봇 — ES 하이브리드 검색과 Redis 3단 게이트"
date: 2026-08-20 08:30:00 +0900
categories: coding
tag: [RAG, Elasticsearch, Redis, BM25, RRF, 리랭커, 하이브리드검색, 캐시, 임베딩, 챗봇]
author_profile: false
toc: true
use_math: true
header:
  image: /assets/images/2026-08-20-internal-rag-chatbot-4-serving/header.svg
  teaser: /assets/images/2026-08-20-internal-rag-chatbot-4-serving/header.svg
description: "검증된 Q&A 를 실제 답변 경로로 조립합니다. ES 에 BM25 + dense 를 RRF 로 섞고 리랭커로 다시 줄 세우기, Redis 를 exact 캐시로 되돌려 벡터 이중화 없애기, 그리고 1편 결함 ②의 처방인 3단 게이트 — 즉답·힌트 주입·순수 RAG 를 코드로 만듭니다."
series: internal-rag-chatbot
series_order: 4
series_title: "🧠 내부문서 RAG 챗봇 — Hermes·Codex·ES·Redis"
---

{% include series-internal-rag-chatbot.html current="4" %}


## Summary

재료가 다 모였어요. [2편](/coding/내부문서_RAG챗봇_QA생성_커버리지_정지조건/)에서 커버리지 0.96 까지 Q&A 를 굽고, [3편](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/)에서 81.5% 만 통과시켰고, 임베딩도 재서 골랐고, 임계값도 0.95 / 0.86 으로 확정했습니다.

이제 조립합니다. 그리고 이번 편이 [1편에서 제일 크게 지적한 결함 ②](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/) — **"Redis 를 1차 답변자로 두면 정확도 상한을 캐시가 결정한다"** — 를 실제로 고치는 자리예요.

고치는 방향을 한 줄로 요약하면 이렇습니다.

> **캐시를 검증 앞에 두지 않습니다.**
> 확실한 것만 즉답하고, 애매한 것은 **힌트로 강등**해서 RAG 에 넘깁니다.

그러면 캐시가 정확도를 **깎는** 물건에서 **보태는** 물건으로 바뀝니다.

> 💡 **이 글에서 다루는 것**
> - 서빙 경로 전체 그림과 **지연 예산**
> - ES 인덱스 두 벌 — 원문 청크와 검증된 Q&A
> - **BM25 + dense 하이브리드**, 그리고 RRF 로 섞기
> - **리랭커** — 1편에서 추천한 그 처방을 실제로 붙이기
> - Redis 를 **exact 캐시**로 되돌리기 (벡터 이중화 제거)
> - **3단 게이트** — 즉답 / 힌트 주입 / 순수 RAG
> - 힌트를 주입하되 **믿게 만들지 않는** 프롬프트
> - 규정이 개정됐을 때의 **캐시 무효화**


<br>

<br>



## 1. 서빙 경로와 지연 예산

먼저 질문 하나가 어떤 길을 걷는지 봅니다.

| 순서 | 단계 | 어디서 | 대략 지연 |
|---|---|---|---|
| 1 | 질문 정규화 → Redis exact 조회 | Redis | 1~3 ms |
| 2 | (miss) 질문 임베딩 | 상주 GPU | 10~30 ms |
| 3 | ES 에서 유사 Q&A 검색 | ES | 30~80 ms |
| 4 | **게이트 판정** (캘리브레이션된 코사인) | 앱 | ~0 ms |
| 5 | (즉답이면) 캐시 답 반환 — **여기서 끝** | — | — |
| 6 | (아니면) ES 에서 근거 청크 검색 | ES | 30~80 ms |
| 7 | 리랭킹 top-50 → top-5 | 상주 GPU | 80~200 ms |
| 8 | LLM 생성 | GPU / 관리형 | 1~5 s |

경로별로 합치면 이렇게 됩니다.

| 경로 | 거치는 단계 | 대략 지연 |
|---|---|---|
| `cache_exact` | 1 | **1~3 ms** |
| `gate_instant` | 1~5 | **40~115 ms** |
| `gate_hint` · `gate_rag` | 1~8 | 1.2~5.4 s |

여기서 두 가지가 눈에 띕니다.

**첫째, 1번의 Redis exact hit 은 진짜 빠릅니다.** 반복 질문은 여기서 끝나요. 그런데 [1편 4절](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)에서 봤듯 **exact match 는 hit rate 이 낮습니다.** 자연어 질문은 매번 표현이 다르니까요. 그래서 Redis 만으로는 부족하고, 4번의 게이트가 필요해요.

**둘째, 즉답 경로는 리랭커를 타지 않습니다.** 중요해서 따로 짚고 갑니다.

리랭커를 태우면 더 정확할 텐데 왜 안 태우냐면, [3편에서 캘리브레이션한 임계값이 **임베딩 코사인** 위에서 잡힌 값이기 때문](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/)이에요. 재는 자를 바꾸면 그 임계값은 의미를 잃습니다. 그래서 **즉답 경로의 안전은 리랭커가 아니라 "골든셋에서 오탐이 0 인 지점"이 보장합니다.**

| 원안 | 지금 |
|---|---|
| 유사하면 **바로** 반환 | **오탐 0 구간**에서만 반환 |
| 임계값이 임의의 상수 | 골든셋으로 캘리브레이션된 값 |
| 애매한 것도 즉답 | 애매하면 힌트로 강등 → RAG |

지연은 40~115ms 라 원안과 비슷해요. **느려진 게 아니라 통과 조건이 좁아진 겁니다.** 그 대가로 [1편에서 본 오답 즉답](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)이 막혀요.

다만 이 경로는 **여전히 LLM 검증을 안 거칩니다.** 구조적으로 남는 위험이라 [5편 모니터링](/coding/내부문서_RAG챗봇_운영_비용_재구축_모니터링/)에서 이 경로의 유사도 분포를 제일 위에 올려둡니다.


<br>

<br>



## 2. ES 인덱스 두 벌

[1편 결함 ④](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)에서 정한 대로, **벡터는 ES 에만** 둡니다. 인덱스는 두 개예요.

| 인덱스 | 담는 것 | 어디에 쓰나 |
|---|---|---|
| `docs_chunks` | 원문 청크 (2편의 조항 단위) | RAG 근거 회수 |
| `qa_verified` | 검증 통과 Q&A (3편의 `verified=true`) | 유사질문 매칭, 힌트 |

두 개를 나눈 이유가 있어요. **검색 대상이 다릅니다.** 청크는 "질문 → 문서" 매칭이고, Q&A 는 "질문 → 질문" 매칭이에요. 한 인덱스에 섞으면 점수 스케일이 안 맞고, 하나만 쓰고 싶을 때 못 끕니다.

```json
{
  "mappings": {
    "properties": {
      "text":       { "type": "text", "analyzer": "nori_mixed" },
      "vec":        { "type": "dense_vector", "dims": 1024,
                      "index": true, "similarity": "cosine" },
      "embedding_model": { "type": "keyword" },
      "doc":        { "type": "keyword" },
      "article":    { "type": "integer" },
      "clause":     { "type": "integer" },
      "revised_at": { "type": "date" },
      "active":     { "type": "boolean" }
    }
  }
}
```

`qa_verified` 는 여기에 `q`, `a`, `chunk_id`, `ground_score` 가 더 붙습니다. 그리고 두 인덱스에서 `text` 와 `vec` 가 **가리키는 대상이 다릅니다.**

| 인덱스 | `text` (BM25 대상) | `vec` (dense 대상) |
|---|---|---|
| `docs_chunks` | 청크 원문 | 청크 원문의 임베딩 |
| `qa_verified` | **질문 `q`** | **질문 `q` 의 임베딩** |

답(`a`)이 아니라 질문을 색인하는 게 핵심이에요. 여기서 하려는 건 "질문 → 질문" 매칭이니까요. 답을 색인하면 표현이 전혀 다른 공간에서 비교하게 됩니다.

### 한국어 분석기

BM25 쪽은 한국어 형태소 분석이 있어야 제대로 돕니다. `nori` 를 쓰되, 내부 문서에는 손을 좀 봐야 해요.

```json
{
  "analysis": {
    "tokenizer": {
      "nori_user": {
        "type": "nori_tokenizer",
        "decompound_mode": "mixed",
        "user_dictionary": "analysis/company_terms.txt"
      }
    },
    "analyzer": {
      "nori_mixed": {
        "type": "custom",
        "tokenizer": "nori_user",
        "filter": ["nori_part_of_speech", "lowercase"]
      }
    }
  }
}
```

`user_dictionary` 가 중요합니다. **사내 약어와 제도명이 여기 들어가야 해요.** 안 넣으면 "육아휴직급여"가 이상하게 쪼개지고, [1편에서 걱정한 "중도인출 vs 중간정산"](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/) 구분도 흐려집니다.

`decompound_mode: mixed` 는 복합명사를 **원형과 조각을 둘 다** 색인해요. 재현율에 유리합니다 — 어차피 순서는 리랭커가 잡을 거니까요.

### 필드 두 개가 더 있는 이유

| 필드 | 왜 |
|---|---|
| `embedding_model` | [1편의 이음매.](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/) 섞인 벡터 검출 |
| `active` | 규정 개정 시 무효화 (8절). 지우지 않고 끔 |

`active` 를 boolean 으로 둔 게 포인트예요. 삭제 대신 끄면 **되돌릴 수 있습니다.**


<br>

<br>



## 3. 하이브리드 검색 — BM25 와 dense 를 RRF 로 섞기

둘 다 씁니다. 잘하는 게 다르거든요.

| 방식 | 강한 데 | 약한 데 |
|---|---|---|
| BM25 | 정확한 용어·숫자·조항번호 | 표현이 다르면 못 찾음 |
| dense | 뜻이 같으면 표현이 달라도 찾음 | 세밀한 용어 구분이 흐림 |

"연차 15일" 같은 건 BM25 가 정확하고, "연차 며칠 쓸 수 있어" 같은 건 dense 가 낫습니다. 문제는 **두 점수를 어떻게 합치느냐**예요. BM25 점수는 범위가 안 정해져 있고 코사인은 -1~1 이라, 그냥 더하면 안 됩니다.

그래서 **점수 대신 순위를 씁니다.** RRF(Reciprocal Rank Fusion) 예요.

$$\text{RRF}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}$$

**쉽게 풀면 이렇습니다.**

- $d$ — 문서 하나. $\text{rank}_i(d)$ 는 **$i$ 번째 검색 방식에서 그 문서가 몇 등**이었는지예요.
- $\sum_i$ — "시그마". 검색 방식(BM25, dense)마다 계산해서 더하라는 뜻.
- $k$ — 보통 60 을 씁니다. 1등과 2등의 격차를 완만하게 만드는 완충 장치예요.
- 분수는 분모부터 읽으니 **"k 더하기 랭크 분의 1"**.

왜 이게 좋냐면, **점수의 단위를 안 봅니다.** 등수만 봐요. BM25 가 3등, dense 가 1등으로 준 문서는 $\frac{1}{63} + \frac{1}{61} \approx 0.0323$ 이 되고, 한쪽에서만 1등인 문서는 $\frac{1}{61} \approx 0.0164$ 가 돼요. **양쪽이 다 인정한 문서가 위로 올라옵니다.**

$k=60$ 의 역할도 직관적이에요. $k$ 가 없으면 1등은 $\frac{1}{1}=1$, 2등은 $\frac{1}{2}=0.5$ 로 격차가 너무 큽니다. 60 을 더하면 $\frac{1}{61}$ 과 $\frac{1}{62}$ 라 거의 같아져요 — **"1등이든 2등이든 상위권이면 비슷하게 쳐준다"** 는 뜻입니다.

```python
RRF_K = 60


def rrf(rank_lists, k=RRF_K):
    scores = {}
    for ranked in rank_lists:
        for pos, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(es, index, query, vec, size=50):
    bm25 = es.search(index=index, size=size, query={
        "bool": {
            "must": [{"match": {"text": {"query": query}}}],
            "filter": [{"term": {"active": True}}],
        }
    })
    dense = es.search(index=index, size=size, knn={
        "field": "vec", "query_vector": vec,
        "k": size, "num_candidates": size * 4,
        "filter": [{"term": {"active": True}}],
    })

    a = [h["_id"] for h in bm25["hits"]["hits"]]
    b = [h["_id"] for h in dense["hits"]["hits"]]
    return [doc_id for doc_id, _ in rrf([a, b])][:size]
```

`num_candidates` 를 `size * 4` 로 잡은 건 HNSW 근사 검색의 재현율을 올리기 위해서예요. 크게 잡을수록 정확하고 느립니다.

**여기서 50개를 뽑는 게 중요합니다.** [1편 7절](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)에서 말한 "1차는 놓치지만 않으면 된다"가 이거예요. [3편에서 잰 Recall@50 이 0.968](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/) 이었으니, 정답은 대체로 이 50개 안에 들어와 있습니다.


<br>

<br>



## 4. 리랭커 — 순서를 다시 잡기

50개를 뽑았지만 순서는 엉망일 수 있어요. [3편의 측정](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/)에서 Recall@50 은 0.968 인데 MRR@10 은 0.664 였죠. **정답은 들어와 있는데 뒤에 묻혀 있다**는 뜻입니다.

리랭커가 이걸 고칩니다. 임베딩과 뭐가 다르냐면

| | 임베딩 (bi-encoder) | 리랭커 (cross-encoder) |
|---|---|---|
| 방식 | 질문과 문서를 **각자** 벡터로 압축해 비교 | 질문과 문서를 **붙여서** 같이 읽음 |
| 속도 | 매우 빠름 (미리 계산 가능) | 느림 (쌍마다 계산) |
| 정확도 | 보통 | 높음 |
| 쓰는 자리 | 수백만 개에서 50개 뽑기 | 50개를 5개로 줄이기 |

미리 계산이 안 되니까 전체에는 못 쓰고, 50개에는 쓸 수 있어요. 역할 분담이 명확합니다.

```python
def rerank(reranker, query, candidates, top_k=5, batch=16):
    pairs = [(query, c["text"]) for c in candidates]

    scores = []
    for i in range(0, len(pairs), batch):
        scores.extend(reranker.score(pairs[i:i + batch]))

    ranked = sorted(zip(candidates, scores), key=lambda p: -p[1])
    return [dict(c, rerank_score=float(s)) for c, s in ranked[:top_k]]
```

배치로 나눠 돌리는 건 GPU 메모리 때문이에요. 50개를 한 번에 넣으면 긴 청크가 섞였을 때 터집니다.

그리고 [1편에서 강조한 리랭커의 이점](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)을 다시 확인하고 갈게요.

> **리랭커는 색인에 묶여 있지 않습니다.**
> 질의 때만 쓰이니까 언제든 바꿔 끼울 수 있고, 껐다 켜도 색인이 멀쩡합니다.

그래서 리랭커가 죽으면 이렇게 처리해요.

```python
def safe_rerank(reranker, query, candidates, top_k=5):
    if reranker is None or not reranker.healthy():
        # 리랭커가 없으면 RRF 순서 그대로 쓴다
        # 품질은 떨어져도 서비스는 계속
        return candidates[:top_k]
    try:
        return rerank(reranker, query, candidates, top_k)
    except Exception:
        return candidates[:top_k]
```

**우아한 성능 저하(graceful degradation)** 예요. 임베딩 모델이 죽으면 검색 자체가 안 되지만, 리랭커는 없어도 굴러갑니다. 외부 임베딩 API 를 안 쓴 이유가 이거고요.


<br>

<br>



## 5. Redis 를 Redis 답게

[1편 결함 ④](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)의 처방입니다. **Redis 에는 벡터를 두지 않습니다.**

| 원안 | 지금 |
|---|---|
| Redis Stack 벡터 인덱스로 유사질문 검색 | ES 가 유사질문 검색 |
| 벡터가 Redis + ES 두 벌 | 벡터는 **ES 에만** |
| 이중 색인, 정합성 붕괴 | 단일 소스 |

Redis 가 맡는 건 **정규화한 질문 → 완성된 응답**의 O(1) 조회 하나예요.

```python
import hashlib
import json
import unicodedata
import re

PUNCT = re.compile(r"[?!.,~\s]+")


def normalize(q):
    q = unicodedata.normalize("NFKC", q).strip().lower()
    return PUNCT.sub(" ", q).strip()


def cache_key(q, gate_version, emb_model):
    base = f"{normalize(q)}|{gate_version}|{emb_model}"
    return "qa:" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]
```

키에 `gate_version` 과 `emb_model` 을 섞은 게 핵심이에요. **설정이 바뀌면 키가 통째로 바뀌어서 옛 캐시를 자동으로 안 보게 됩니다.** 명시적으로 지울 필요가 없어요.

정규화는 **일부러 약하게** 했습니다. 조사를 떼거나 어간 추출을 하고 싶어지는데, 그러면 "연차 쓸 수 있어" 와 "연차 쓸 수 없어" 가 같은 키가 될 위험이 있어요. 여기서 무리하지 말고 **뜻 매칭은 ES 에 맡기는 게** 맞습니다.

### 휘발하지 않게

[1편 결함 ⑥](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/) — Redis 는 파생물이라는 원칙이요.

```shell
# 조용히 사라지는 게 제일 나쁘다. 넘치면 에러를 내게 한다.
maxmemory 4gb
maxmemory-policy noeviction
```

`allkeys-lru` 를 쓰면 메모리가 찰 때 **에러도 없이** 오래된 키부터 지워집니다. `noeviction` 이면 쓰기가 실패해서 **알아챌 수 있어요.** 캐시니까 쓰기 실패는 치명적이지 않습니다.

```python
def cache_put(rd, key, payload, ttl=86400 * 7):
    try:
        rd.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass  # 캐시 적재 실패는 서비스를 막지 않는다


def cache_get(rd, key):
    try:
        raw = rd.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None
```

Redis 가 통째로 죽어도 서비스는 느려질 뿐 멈추지 않습니다. **파생물이니까요.**


<br>

<br>



## 6. 3단 게이트

드디어 [1편 결함 ②의 처방](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)입니다. [3편에서 캘리브레이션한 값](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/) 0.95 / 0.86 을 씁니다.

| 구간 | 처리 | 근거 |
|---|---|---|
| $\ge 0.95$ | 캐시 답 즉답 | 3편 8절: 이 위로는 **오탐 0** |
| $0.86 \sim 0.95$ | RAG 로 보내되 그 Q&A 를 **힌트로 주입** | 양성의 90% 를 건지는 구간 |
| $< 0.86$ | 순수 RAG | 캐시가 도움 안 됨 |

여기서 유사도는 **임베딩 코사인이 아니라 리랭커 점수를 정규화한 것**을 쓰는 게 더 정확한데, 3편에서 캘리브레이션한 게 임베딩 코사인이라 일단 그걸로 갑니다. 둘을 섞지 마세요 — **캘리브레이션한 자와 실제로 재는 자가 달라지면** 그 임계값은 의미가 없습니다.

```python
GATE = {
    "embedding_model": "local-multilingual-v1",
    "top": 0.95,
    "bottom": 0.86,
    "gate_version": "g1",
}


def answer(query, es, rd, emb, reranker, llm):
    key = cache_key(query, GATE["gate_version"], GATE["embedding_model"])

    # 0단: 완전히 같은 질문이 이미 있었나
    hit = cache_get(rd, key)
    if hit:
        return dict(hit, path="cache_exact")

    vec = emb.encode(query)

    # 유사 Q&A 후보
    qa_ids = hybrid_search(es, "qa_verified", query, vec, size=50)
    qas = es.mget("qa_verified", qa_ids)
    best = max(qas, key=lambda r: emb.cos_vec(vec, r["vec"])) if qas else None
    sim = emb.cos_vec(vec, best["vec"]) if best else 0.0

    # 1단: 확실하면 즉답
    if best and sim >= GATE["top"]:
        out = {"answer": best["a"], "sources": [source_of(best)],
               "similarity": round(sim, 3), "path": "gate_instant"}
        cache_put(rd, key, out)
        return out

    # 2·3단: 근거 청크를 회수해 RAG
    chunk_ids = hybrid_search(es, "docs_chunks", query, vec, size=50)
    chunks = safe_rerank(reranker, query,
                         es.mget("docs_chunks", chunk_ids), top_k=5)

    hint = best if (best and sim >= GATE["bottom"]) else None
    out = generate(llm, query, chunks, hint)
    out["similarity"] = round(sim, 3)
    out["path"] = "gate_hint" if hint else "gate_rag"

    cache_put(rd, key, out)
    return out
```

`path` 를 응답에 넣어두는 게 운영에서 아주 유용해요. 어느 경로로 나갔는지 로그에 남아야 **경로별 정확도를 따로 잴 수 있습니다.** 5편에서 이걸 모니터링에 씁니다.


<br>

<br>



## 7. 힌트를 주되 믿게 만들지 않기

2단 구간(0.86~0.95)이 이 설계의 묘수인데, 프롬프트를 잘못 쓰면 **1단이랑 똑같아집니다.** LLM 이 힌트를 그대로 베껴버리거든요.

핵심은 힌트에 **낮은 지위**를 명시적으로 부여하는 거예요.

```text
<clauses>
{근거 청크 top-5, 출처와 함께}
</clauses>

<similar_qa confidence="medium">
A previously answered question that MAY be related. It is NOT
authoritative and MAY be about a different question entirely.
Q: {힌트 질문}
A: {힌트 답}
</similar_qa>

Rules:
1. Answer ONLY from <clauses>. They are the sole source of truth.
2. <similar_qa> is a hint about phrasing and framing.
   Do NOT copy its content unless <clauses> independently support it.
3. If <similar_qa> asks something different from the user's question,
   ignore it completely.
4. If <clauses> do not contain the answer, say so. Do not guess.
5. Cite the article number for every factual claim.
```

3번 규칙이 제일 중요해요. [1편에서 든 예](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/) — "육아휴직 급여" 를 물었는데 "육아휴직 기간" Q&A 가 힌트로 붙는 상황 — 을 정면으로 겨냥한 문장입니다.

그래서 이 구간이 실제로 뭘 얻느냐면

| 힌트가 주는 것 | 힌트가 주지 않는 것 |
|---|---|
| 답변 형식·말투의 일관성 | 사실 내용 |
| 어느 조항을 봐야 하는지 단서 | 최종 판단 |
| [1편 결함 ⑤](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)의 품질 단차 완화 | — |

세 번째가 덤이에요. Codex 가 만든 좋은 답이 **few-shot 예시처럼** 작동해서, EC2 GPU 의 작은 모델이 비슷한 톤으로 답하게 됩니다. 1편에서 "사전생성 Q&A 를 서빙 모델의 few-shot 으로 재사용" 하자고 한 게 여기서 저절로 이뤄져요.


<br>

<br>



## 8. 캐시 무효화 — 규정이 개정되면

이 시스템에서 제일 위험한 순간입니다. **규정이 바뀌었는데 옛 답이 계속 나가는 것.**

[2편에서 `revised_at` 을 메타데이터에 넣어둔 게](/coding/내부문서_RAG챗봇_QA생성_커버리지_정지조건/) 여기서 값어치를 합니다.

```python
def invalidate(es, rd, doc, article, new_revised_at):
    # 1. 해당 조항의 옛 청크를 끈다 (지우지 않는다)
    stale = es.update_by_query("docs_chunks", query={
        "bool": {"must": [
            {"term": {"doc": doc}},
            {"term": {"article": article}},
            {"range": {"revised_at": {"lt": new_revised_at}}},
        ]}
    }, script="ctx._source.active = false")

    # 2. 그 청크에 매달린 Q&A 도 함께 끈다
    es.update_by_query("qa_verified", query={
        "bool": {"must": [
            {"term": {"doc": doc}},
            {"term": {"article": article}},
            {"range": {"revised_at": {"lt": new_revised_at}}},
        ]}
    }, script="ctx._source.active = false")

    return stale
```

`active=false` 로 끄는 거라 3절의 검색 필터에서 자동으로 빠집니다. **지우지 않았으니 되돌릴 수 있고요.**

Redis 쪽은 더 간단합니다.

```python
def bump_gate_version(rd, new_version):
    # 키에 gate_version 이 섞여 있어서
    # 올리기만 하면 옛 캐시는 조회되지 않는다
    GATE["gate_version"] = new_version
```

5절에서 키에 버전을 섞어둔 이유예요. **`KEYS` 로 훑어서 지우는 짓을 안 해도 됩니다** — 운영 Redis 에서 `KEYS` 는 전체를 블로킹해서 절대 쓰면 안 되는 명령이에요. TTL 이 지나면 옛 키는 알아서 사라집니다.

전체 순서는 이렇습니다.

| 순서 | 하는 일 | 주체 |
|---|---|---|
| 1 | 개정된 조항을 다시 청킹 | 집 (2편) |
| 2 | 새 청크로 Q&A 생성 + 검증 | 집 (2·3편) |
| 3 | 새 JSONL 을 S3 로 | 집 → S3 |
| 4 | 새 청크·Q&A 색인 | AWS 로더 |
| 5 | **옛 것 `active=false`** | AWS 로더 |
| 6 | `gate_version` 올리기 | 앱 |

**5번이 4번 뒤에 오는 게 중요해요.** 순서를 바꾸면 새 게 들어오기 전에 옛 게 꺼져서 그 사이 질문에 "모르겠습니다"가 나갑니다.


<br>

<br>



## 9. 자주 밟는 지뢰

**① 즉답 경로에서 오답이 나옵니다.**
[3편의 캘리브레이션](/coding/내부문서_RAG챗봇_신뢰도게이트_그라운드니스_골든셋/)을 안 했거나, 하고 나서 임베딩 모델을 바꿨을 확률이 높습니다. `GATE["embedding_model"]` 과 실제 인코더가 같은지 확인하세요. 3편 9절의 회귀 테스트가 이걸 잡습니다.

**② RRF 를 넣었는데 결과가 오히려 나빠졌습니다.**
두 검색의 후보 수가 다를 때 그럽니다. BM25 는 200개, dense 는 50개 이런 식이면 BM25 쪽이 뒤쪽 순위까지 점수를 받아서 유리해져요. **양쪽 `size` 를 같게** 두세요.

**③ 힌트를 넣으니 답이 힌트를 그대로 베낍니다.**
7절 규칙 2·3번이 빠졌거나 약합니다. `confidence="medium"` 같은 표시도 생각보다 효과가 있어요.

**④ Redis hit rate 이 5% 밖에 안 됩니다.**
정상입니다. exact 캐시는 원래 그래요. 실제 "빠른 답"은 1단 게이트(`gate_instant`)가 담당합니다. **두 경로를 따로 세세요** — 합쳐서 보면 판단을 그르칩니다.

**⑤ 리랭킹이 800ms 씩 걸립니다.**
청크가 너무 길거나 배치가 안 나뉘어 있어요. 4절처럼 배치로 자르고, 청크 길이 상한(2편의 `MAX_CHARS`)을 확인하세요.

**⑥ 개정 후에도 옛 답이 나옵니다.**
8절 6번 단계, `gate_version` 을 안 올렸습니다. ES 만 끄고 Redis 를 잊는 게 제일 흔한 실수예요.


<br>

<br>



## 10. 정리

이번 편으로 [1편의 결함 ②와 ④](/coding/내부문서_RAG챗봇_아키텍처검증_전체그림/)가 닫혔습니다.

| 원안 | 지금 |
|---|---|
| Redis 가 1차 답변자, hit = 확정 | **3단 게이트.** 즉답은 오탐 0 구간에서만 |
| 애매한 매칭도 즉답 | 애매하면 **힌트로 강등**해서 RAG 에 |
| Redis + ES 벡터 이중화 | 벡터는 ES 에만, Redis 는 exact 캐시 |
| 캐시 무효화 대책 없음 | `active` 플래그 + `gate_version` 키 |
| 검색이 dense 단독 | BM25 + dense **RRF** + 리랭커 |

그리고 1편에서 걱정했던 것과 반대 방향으로 뒤집힌 게 하나 있어요.

> 원안에서 캐시는 **정확도를 깎는** 물건이었습니다.
> 지금 캐시는 즉답 구간에서 지연을 줄이고,
> 힌트 구간에서 **정확도와 일관성을 보태는** 물건이 됐어요.

같은 부품인데 놓는 자리를 바꿨더니 역할이 반대가 됐습니다. 1편에서 "부품이 다 표준이어도 엮는 순서가 틀리면 시스템이 틀린다"고 한 게 이 얘기였어요.

남은 건 운영입니다. 집에서 만든 걸 AWS 로 어떻게 안전하게 밀지, Redis 가 날아갔을 때 어떻게 되살릴지, GPU 를 어떻게 나눠서 비용을 잡을지, 그리고 이 모든 게 잘 돌고 있는지 어떻게 알지. [5편](/coding/내부문서_RAG챗봇_운영_비용_재구축_모니터링/)에서 마무리합니다.
