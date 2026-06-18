---
layout: single
title:  "ChromaDB 를 RAG 용 벡터 DB 로 제대로 쓰기 위해 알아야 할 것들"
date: 2026-06-18 20:00:00 +0900
categories: coding
tag: [ChromaDB, RAG, 벡터DB, embedding, vector-search, chunking, metadata-filter, cosine, retrieval, LLM]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-18-chromadb-rag/header.svg
  teaser: /assets/images/2026-06-18-chromadb-rag/header.svg
description: "ChromaDB 를 RAG 용 벡터 DB 로 제대로 쓰기 위한 핵심 — 클라이언트 모드, 컬렉션과 거리 함수, 임베딩 함수 일관성, 청킹, 메타데이터 필터, 검색 품질 끌어올리기까지 실제 코드와 함께 정리합니다."
---


## Summary

RAG 를 만들 때 가장 먼저 손이 가는 게 벡터 DB 인데, ChromaDB 는 `pip install` 한 줄로 바로 쓸 수 있어서 입문용으로 인기가 많아요. 그런데 "일단 넣고 query 하면 비슷한 게 나온다" 까지는 쉬운데, **막상 검색 결과가 영 엉뚱하게 나오는** 순간이 꼭 옵니다. 그 차이는 대부분 벡터 DB 자체가 아니라 **거리 함수·임베딩 일관성·청킹·메타데이터 필터** 를 모르고 써서 생겨요.

이 글에서는 ChromaDB 를 RAG 의 검색 엔진으로 제대로 쓰기 위해 알아야 하는 개념들을, 실제로 돌아가는 코드와 함께 정리합니다. API 레퍼런스를 전부 나열하는 게 아니라, "이걸 모르면 검색 품질이 무너진다" 수준의 핵심만 골랐어요.

> 💡 **이 글에서 다루는 것**
> - RAG 에서 벡터 DB 가 맡는 역할 — 큰 그림 먼저
> - ChromaDB 의 3가지 클라이언트 모드 (인메모리 · 영속 · 서버)
> - **컬렉션과 거리 함수** — `cosine` vs `l2` vs `ip`, 가장 흔한 함정
> - **임베딩 함수 일관성** — 넣을 때와 찾을 때가 같아야 한다
> - 청킹(chunking) — 검색 품질의 8할이 여기서 갈림
> - `add` / `query` / **메타데이터 `where` 필터**
> - 검색 품질 끌어올리기 — top-k · 필터 · 리랭킹 · MMR
> - 운영에서 자주 밟는 지뢰

벡터 DB 가 처음이면 ChromaDB 로 시작하는 걸 추천드려요. 개념을 가장 적은 마찰로 익힐 수 있거든요.


<br>

<br>



## 1. RAG 에서 벡터 DB 가 하는 일

먼저 큰 그림부터 잡고 가요. RAG(Retrieval-Augmented Generation)는 결국 **"질문과 관련된 문서 조각을 찾아서, 그걸 LLM 프롬프트에 끼워 넣고 답하게 하는"** 패턴입니다. 이때 "관련된 조각을 찾는" 단계가 벡터 DB 의 일이에요.

흐름을 한 그림으로 요약하면 다음과 같아요.

```text
[문서] → 청킹 → [작은 조각들] → 임베딩 → [벡터들] → 벡터 DB 저장
                                                          │
[질문] → 임베딩 → [질문 벡터] ──── 유사도 검색 ──────────┘
                                      │
                            [관련 조각 top-k] → LLM 프롬프트 → 답변
```

여기서 핵심은, **벡터 DB 는 "의미가 비슷한 텍스트" 를 찾아주는 도구** 라는 점이에요. 키워드가 정확히 겹치지 않아도 ("연차" 로 검색했는데 "휴가 규정" 문서가 나오는 식) 의미가 가까우면 찾아줍니다. 그 "의미" 를 숫자 벡터로 바꾸는 게 임베딩이고, 벡터끼리 얼마나 가까운지를 재는 게 거리 함수예요.

그래서 ChromaDB 를 잘 쓴다는 건 사실 **임베딩과 거리 함수를 이해하고, 검색 단계를 통제하는** 거예요. 저장·조회 API 는 정말 단순합니다.


<br>

<br>



## 2. 설치와 3가지 클라이언트 모드

설치는 한 줄이에요.

```shell
pip install chromadb
```

ChromaDB 를 쓸 때 가장 먼저 정해야 하는 건 **어떤 클라이언트로 띄울지** 예요. 모드를 헷갈리면 "분명 데이터를 넣었는데 다시 켜니 다 사라졌다" 같은 일이 생깁니다.

| 모드 | 생성 코드 | 데이터 저장 | 언제 쓰나 |
|---|---|---|---|
| 인메모리 | `chromadb.Client()` | 메모리(프로세스 끝나면 소멸) | 테스트·실험 |
| 영속(persistent) | `chromadb.PersistentClient(path=...)` | 디스크 | 로컬 앱·소규모 운영 |
| 서버(client-server) | `chromadb.HttpClient(host, port)` | 별도 서버 | 여러 앱이 공유·운영 |

가장 많이 쓰는 건 **영속 모드** 예요. `path` 에 지정한 폴더에 SQLite + 인덱스 파일로 저장돼서, 프로그램을 껐다 켜도 데이터가 남아요.

```python
import chromadb

# 디스크에 저장되는 클라이언트
client = chromadb.PersistentClient(path="./chroma_store")

print(client.heartbeat())  # 살아있는지 확인 (나노초 타임스탬프)
```

```text
1718700000000000000
```

> ⚠️ 인메모리 `chromadb.Client()` 는 프로세스가 끝나면 데이터가 전부 사라져요. "껐다 켰더니 비어있다" 의 90% 는 여기서 옵니다. 운영성 코드면 처음부터 `PersistentClient` 를 쓰세요.

큰 트래픽이나 여러 서비스가 같은 벡터 저장소를 공유해야 하면 서버 모드로 띄웁니다. 도커로 한 줄이에요.

```shell
docker run -p 8000:8000 chromadb/chroma
```

그리고 앱에서는 이렇게 붙어요.

```python
client = chromadb.HttpClient(host="localhost", port=8000)
```


<br>

<br>



## 3. 컬렉션 — 데이터를 담는 단위

ChromaDB 에서 데이터는 **컬렉션(collection)** 단위로 들어가요. 관계형 DB 의 테이블 같은 개념이에요. 컬렉션 하나에는 보통 **하나의 임베딩 함수 + 하나의 거리 함수** 가 묶입니다.

```python
collection = client.get_or_create_collection(name="docs")
```

`get_or_create_collection` 을 추천드려요. `create_collection` 은 이미 있으면 에러를 내고, `get_collection` 은 없으면 에러를 내는데, `get_or_create` 는 있으면 가져오고 없으면 만들어줘서 재실행에 안전하거든요.

컬렉션을 만들 때 가장 중요한 결정이 다음 절의 **거리 함수** 예요. 여기서 한 번 잘못 잡으면 검색 품질이 통째로 흔들립니다.


<br>

<br>



## 4. 거리 함수 — 가장 흔하게 밟는 지뢰

ChromaDB 컬렉션의 기본 거리 함수는 **`l2`(유클리드 거리 제곱)** 예요. 그런데 요즘 문장 임베딩 모델들은 대부분 **코사인 유사도(cosine)** 기준으로 학습돼 있어요. 이 둘이 안 맞으면 "비슷한 문서인데 멀다고 나오는" 일이 생깁니다.

거리 함수는 컬렉션을 만들 때 `metadata` 의 `hnsw:space` 로 지정해요.

```python
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},   # l2(기본) / cosine / ip 중 선택
)
```

세 가지를 정리하면 다음과 같아요.

| 값 | 의미 | 언제 |
|---|---|---|
| `l2` | 유클리드 거리 제곱 (기본값) | 벡터 크기까지 의미 있을 때 |
| `cosine` | 코사인 거리 (방향만 비교) | **문장 임베딩 대부분** ← 추천 |
| `ip` | 내적(inner product) | 정규화된 벡터 + 성능 튜닝 |

> 🚨 **컬렉션 생성 후에는 거리 함수를 못 바꿔요.** 나중에 바꾸려면 컬렉션을 지우고 다시 만들어서 전부 재삽입해야 합니다. 그러니 처음 만들 때 `cosine` 으로 잡는 걸 기본값으로 생각하세요. 임베딩 모델 문서에 "cosine similarity 로 비교하라" 고 적혀 있으면 거의 무조건 `cosine` 이에요.

한 가지 더 — ChromaDB 의 `query` 는 **유사도(높을수록 비슷)** 가 아니라 **거리(낮을수록 비슷)** 를 돌려줘요. `cosine` 으로 잡으면 `distance` 가 0 에 가까울수록 비슷하고, 2 에 가까울수록 정반대예요. 점수를 사람이 읽기 좋게 바꾸려면 이렇게 변환합니다.

```python
def cosine_similarity(distance: float) -> float:
    # ChromaDB 의 cosine distance(0~2) → 유사도(1 ~ -1)
    return 1 - distance

print(round(cosine_similarity(0.18), 3))   # 거리 0.18 → 꽤 비슷
print(round(cosine_similarity(1.40), 3))   # 거리 1.40 → 거의 무관
```

```text
0.82
-0.4
```


<br>

<br>



## 5. 임베딩 함수 — 넣을 때와 찾을 때가 같아야 한다

이게 ChromaDB 초보가 가장 많이 놓치는 부분이에요. **저장할 때 쓴 임베딩 모델과 검색할 때 쓰는 임베딩 모델이 반드시 같아야** 합니다. 모델이 다르면 벡터 공간 자체가 달라서, 아무리 비슷한 문장이어도 엉뚱하게 멀리 떨어져요.

ChromaDB 는 임베딩 함수를 넘기지 않으면 **기본 임베딩 함수**(`all-MiniLM-L6-v2`, 384차원)를 자동으로 씁니다. 처음 `add` 할 때 모델을 내려받아요.

```python
# 임베딩 함수를 안 넘기면 기본값(all-MiniLM-L6-v2)을 자동 사용
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)
```

기본 모델은 가볍고 빠르지만 한국어 성능이 아쉬워요. 한국어 문서가 많으면 다국어 모델로 바꾸는 걸 추천드립니다. ChromaDB 의 `embedding_functions` 로 명시적으로 지정해요.

```python
from chromadb.utils import embedding_functions

# 다국어 문장 임베딩 모델로 교체 (한국어 포함)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)
```

> 🚨 **임베딩 함수도 컬렉션에 묶여요.** 한 번 어떤 모델로 채운 컬렉션에, 나중에 다른 모델로 `query` 하면 결과가 무너집니다. ChromaDB 는 컬렉션을 다시 `get` 할 때 같은 `embedding_function` 을 안 넘기면 기본 모델로 되돌아가니, **컬렉션을 여는 모든 곳에서 같은 `embedding_function` 을 넘기는** 습관을 들이세요. 모델을 바꾸고 싶으면 새 컬렉션을 만들어 전부 재삽입하는 게 정석이에요.

OpenAI·Cohere 등 API 기반 임베딩도 `embedding_functions` 에 준비돼 있어요. 다만 외부 API 임베딩은 비용·레이트리밋·키 관리가 따라오니, 키는 환경변수로 빼고 코드에 박지 마세요.

```python
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="<OPENAI_API_KEY>",        # 실제로는 os.environ 에서 읽기
    model_name="text-embedding-3-small",
)
```


<br>

<br>



## 6. 청킹 — 검색 품질의 8할

솔직히 말하면, RAG 검색 품질을 가장 크게 좌우하는 건 벡터 DB 설정이 아니라 **문서를 어떻게 잘랐는가** 예요. 너무 크게 자르면 한 조각에 여러 주제가 섞여서 검색이 뭉툭해지고, 너무 작게 자르면 문맥이 끊겨서 LLM 이 답을 못 만들어요.

기본 감각은 이래요.

- **한 조각 = 한 주제** 가 되도록 자른다.
- 크기는 보통 **300~800 토큰** (문자로는 대략 500~1500자) 사이.
- 조각끼리 **약간 겹치게(overlap)** 잘라서 경계에서 문맥이 끊기지 않게 한다.

가장 단순한 고정 크기 + 오버랩 청커를 직접 짜보면 이렇게 돼요.

```python
def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap        # overlap 만큼 뒤로 당겨 다음 조각과 겹침
    return chunks

sample = "연차 휴가는 입사 1년 후 15일이 부여됩니다. " * 30   # 길게 만든 예시 문서
pieces = chunk_text(sample, size=120, overlap=20)

print(len(pieces))          # 몇 조각으로 잘렸나
print(repr(pieces[0][:40])) # 첫 조각 앞부분
```

```text
8
'연차 휴가는 입사 1년 후 15일이 부여됩니다. 연차 휴가는 입사 1년'
```

실전에서는 문자 수로 자르기보다 **문단·문장 경계** 를 존중하는 청커가 더 좋아요. LangChain 의 `RecursiveCharacterTextSplitter` 같은 게 대표적이에요. 핵심 원리는 "큰 구분자(문단)부터 시도하고, 너무 크면 점점 작은 구분자(문장→단어)로 쪼갠다" 입니다.

> 💡 검색이 자꾸 엉뚱하면 임베딩 모델을 바꾸기 전에 **청크 크기부터 의심** 하세요. 조각을 절반으로 줄이는 것만으로 검색 적중률이 확 오르는 경우가 많아요.


<br>

<br>



## 7. 데이터 넣기 — `add`

이제 조각들을 컬렉션에 넣어요. `add` 에는 **`documents`(원문)**, **`ids`(고유 키)** 가 필수이고, **`metadatas`(부가 정보)** 는 선택이지만 사실상 필수처럼 챙기는 게 좋아요. 임베딩은 컬렉션의 임베딩 함수가 자동으로 만들어줍니다.

```python
collection.add(
    documents=[
        "연차 휴가는 입사 1년 후 15일이 부여됩니다.",
        "재택근무는 팀장 승인 후 주 2회까지 가능합니다.",
        "경조사 휴가는 결혼 시 5일이 주어집니다.",
    ],
    ids=["hr-001", "hr-002", "hr-003"],
    metadatas=[
        {"category": "휴가", "source": "취업규칙", "year": 2026},
        {"category": "근무", "source": "취업규칙", "year": 2026},
        {"category": "휴가", "source": "취업규칙", "year": 2026},
    ],
)

print(collection.count())   # 들어간 문서 개수
```

```text
3
```

`ids` 는 직접 관리하세요. 문서 ID 를 의미 있게 잡아두면(`hr-001` 처럼) 나중에 **같은 ID 로 `add` 하면 갱신(upsert 유사)** 처럼 쓸 수 있고, 중복 삽입도 막아요. 매번 랜덤 ID 를 쓰면 같은 문서가 두 번 들어가 검색이 지저분해집니다.

> ⚠️ `metadatas` 값에는 **문자열·숫자·불리언** 같은 평평한(flat) 값만 넣을 수 있어요. 리스트나 중첩 딕셔너리는 안 들어갑니다. 태그 여러 개를 넣고 싶으면 `"tags": "휴가,복지"` 처럼 문자열로 합치거나, 키를 나눠서 표현하세요.


<br>

<br>



## 8. 검색 — `query` 와 메타데이터 `where` 필터

검색은 `query_texts` 에 질문을 넘기면 끝이에요. 질문도 같은 임베딩 함수로 벡터가 되고, 가까운 조각을 `n_results` 개 돌려줘요.

```python
res = collection.query(
    query_texts=["휴가 며칠 쓸 수 있어?"],
    n_results=2,
)

for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(round(dist, 3), doc)
```

```text
0.214 연차 휴가는 입사 1년 후 15일이 부여됩니다.
0.331 경조사 휴가는 결혼 시 5일이 주어집니다.
```

"휴가 며칠" 이라고만 물었는데 키워드가 정확히 안 겹치는 연차·경조사 문서를 의미로 찾아낸 게 보이죠. 이게 벡터 검색의 힘이에요.

여기에 **메타데이터 필터(`where`)** 를 더하면 RAG 품질이 한 단계 올라가요. 예를 들어 "휴가 카테고리 안에서만 찾기" 처럼 범위를 좁히는 거예요.

```python
res = collection.query(
    query_texts=["며칠 쉴 수 있어?"],
    n_results=2,
    where={"category": "휴가"},          # 메타데이터로 사전 필터링
)

print(res["documents"][0])
```

```text
['연차 휴가는 입사 1년 후 15일이 부여됩니다.', '경조사 휴가는 결혼 시 5일이 주어집니다.']
```

`where` 연산자도 꽤 풍부해요. 자주 쓰는 것만 추리면 다음과 같습니다.

| 연산자 | 의미 | 예시 |
|---|---|---|
| `$eq` / `$ne` | 같다 / 다르다 | `{"year": {"$eq": 2026}}` |
| `$gt` / `$gte` / `$lt` / `$lte` | 대소 비교 | `{"year": {"$gte": 2025}}` |
| `$in` / `$nin` | 목록에 포함/제외 | `{"category": {"$in": ["휴가", "근무"]}}` |
| `$and` / `$or` | 조건 결합 | `{"$and": [{"year": 2026}, {"category": "휴가"}]}` |

문서 본문에서 특정 단어가 든 조각만 찾고 싶으면 `where_document={"$contains": "재택"}` 로 본문 텍스트 필터도 같이 걸 수 있어요.

> 💡 메타데이터 필터는 RAG 의 숨은 무기예요. 사내 위키 RAG 라면 `{"department": "인사", "year": {"$gte": 2025}}` 처럼 **부서·연도·문서종류** 로 범위를 먼저 좁히면, 같은 질문이어도 훨씬 정확한 조각이 올라옵니다. 메타데이터를 처음 넣을 때 넉넉히 챙겨두세요.


<br>

<br>



## 9. 검색 품질을 한 단계 더 끌어올리기

기본 `query` 만으로 부족할 때 쓰는 카드들이에요. 순서대로 시도해보길 추천드려요.

**1) `n_results`(top-k) 를 넉넉히, 그다음 줄이기.** LLM 에 넣을 최종 조각은 3~5개라도, 검색은 일단 10~20개를 가져온 다음 추리는 게 안전해요. 한 방에 3개만 가져오면 정작 필요한 조각을 놓치기 쉽거든요.

**2) 리랭킹(re-ranking).** 벡터 검색으로 후보 20개를 가져온 뒤, **크로스 인코더(cross-encoder)** 같은 정밀 모델로 질문-조각 쌍을 다시 점수 매겨 상위 몇 개만 남기는 방법이에요. 벡터 검색은 빠르지만 거칠고, 리랭커는 느리지만 정확해서, 둘을 단계로 쓰면 속도와 정확도를 같이 챙겨요.

```text
질문 → 벡터 검색 top-20 (빠름·거침) → 리랭커로 재정렬 → top-5 (느림·정확) → LLM
```

**3) MMR(다양성 확보).** 비슷한 조각이 top-k 를 도배하면 같은 말만 반복돼요. MMR(Maximal Marginal Relevance)은 "질문과 가까우면서 + 이미 고른 것과는 다른" 조각을 뽑아 다양성을 챙겨줍니다. 긴 문서에서 여러 측면을 모아야 할 때 유용해요.

**4) 하이브리드 검색.** 의미가 아니라 **정확한 키워드**(제품 코드, 법조항 번호 같은)가 중요한 도메인이면, 벡터 검색만으론 약해요. 키워드 검색(BM25 등)과 벡터 검색 점수를 합치는 하이브리드가 더 잘 맞습니다. ChromaDB 단독으로는 키워드 쪽이 약하니, 이 요구가 강하면 `where_document` 의 `$contains` 를 보조로 쓰거나 별도 키워드 인덱스를 같이 두는 걸 고려하세요.

> ✅ 정리하면 — **top-k 넉넉히 → 메타데이터로 좁히기 → 리랭킹으로 정밀하게**. 이 세 단계만 제대로 걸어도 체감 품질이 확 달라져요.


<br>

<br>



## 10. 운영에서 자주 밟는 지뢰

마지막으로, 실제로 굴려보면 꼭 한 번씩 걸리는 것들을 모았어요.

- **모드 혼동** — 인메모리 `Client()` 로 개발하다 그대로 배포해서 재시작마다 데이터가 증발해요. 운영은 `PersistentClient` 나 서버 모드로 띄우세요.
- **임베딩 함수 불일치** — 컬렉션 채울 때와 열 때 `embedding_function` 을 다르게(또는 한쪽만) 넘겨서 검색이 무너짐. 컬렉션을 여는 모든 코드에서 같은 함수를 넘길 것.
- **거리 함수를 안 바꿈** — 기본 `l2` 인 채로 코사인 모델을 써서 결과가 미묘하게 어긋나요. 생성 시 `hnsw:space: cosine` 인지 확인하세요.
- **메타데이터에 리스트 넣기** — `metadatas` 에 리스트/딕셔너리를 넣어 에러. 평평한 값만.
- **거대한 단일 청크** — PDF 한 페이지를 통째로 한 조각으로 넣고 검색이 뭉툭하다고 함. 청크부터 쪼갤 것.
- **랜덤 ID 로 중복 삽입** — 같은 문서를 매번 새 ID 로 add 해서 검색 결과에 같은 내용이 여러 번. 의미 있는 `ids` 로 관리.
- **버전 차이** — ChromaDB 는 버전에 따라 API 시그니처가 바뀐 적이 있어요(`Client` 설정 방식 등). 튜토리얼이 안 돌면 설치된 버전부터 확인하세요.

```python
import chromadb
print(chromadb.__version__)
```

```text
0.5.x
```


<br>

<br>



## 마무리

ChromaDB 자체는 정말 단순해요. 어려운 건 벡터 DB API 가 아니라 **임베딩을 일관되게 쓰고, 거리 함수를 맞추고, 문서를 잘 자르고, 메타데이터로 검색을 좁히는** 감각이에요. 이 네 가지만 손에 익으면 어떤 벡터 DB 로 옮겨가도 RAG 검색 품질을 통제할 수 있어요.

처음엔 `PersistentClient` + `cosine` + 다국어 임베딩 + 적당한 청킹, 이 조합으로 시작해보시는 걸 추천드립니다. 거기서부터 리랭킹·하이브리드로 한 칸씩 올리면 돼요.

일단 오늘은 여기까지.....   
다음 글에서는 검색해온 조각을 LLM 프롬프트에 끼워 넣어 실제 답변을 만드는 RAG 파이프라인 쪽을 정리해볼게요.
