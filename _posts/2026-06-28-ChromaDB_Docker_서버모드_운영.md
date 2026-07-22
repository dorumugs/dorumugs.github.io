---
layout: single
title:  "(2/4) ChromaDB 를 Docker 로 띄우고 운영하기 — 볼륨·인증·백업"
date: 2026-06-28 09:30:00 +0900
categories: coding
tag: [ChromaDB, Docker, docker-compose, RAG, 벡터DB, volume, persistence, auth, backup, HttpClient]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-18-chromadb-docker/header.svg
  teaser: /assets/images/2026-06-18-chromadb-docker/header.svg
description: "ChromaDB 를 Docker 서버 모드로 띄워 운영하는 법 — 볼륨 마운트로 데이터 영속, docker-compose 로 선언적 운영, 토큰 인증, 헬스체크·리소스 제한, 그리고 백업·복구까지 실제 설정과 함께 정리합니다."
series: chromadb-rag
series_order: 2
---

{% include series-chromadb-rag.html current="2" %}


## Summary

[1편](/coding/ChromaDB_RAG_벡터DB_제대로쓰기/)에서는 ChromaDB 를 RAG 검색 엔진으로 "잘 쓰는 법"(거리 함수·임베딩 일관성·청킹·메타데이터 필터)을 정리했어요. 그때 코드는 전부 같은 프로세스 안에서 `PersistentClient` 로 디스크에 저장하는 방식이었죠. 로컬 개발이나 작은 배치 작업엔 그걸로 충분합니다.

그런데 막상 서비스로 올리려고 하면 상황이 달라져요. **여러 앱(또는 여러 워커)이 같은 벡터 저장소를 공유** 해야 하고, 앱을 배포할 때마다 벡터 DB 가 같이 죽으면 안 되거든요. 그래서 ChromaDB 를 **독립된 서버 프로세스** 로 분리하게 되고, 그 가장 쉬운 방법이 Docker 예요.

이 글에서는 ChromaDB 를 Docker 로 띄워서 **운영 가능한 상태로 만드는** 과정을 정리합니다. 단순히 `docker run` 한 줄이 아니라, 재시작해도 데이터가 살아남고, 아무나 못 들어오고, 죽었는지 알 수 있고, 날아가도 복구되는 — 그 네 가지를 챙기는 게 목표예요.

> 💡 **이 글에서 다루는 것**
> - 왜 인메모리·`PersistentClient` 를 넘어 **서버 모드** 로 가는가
> - `docker run` 으로 띄우기 — 그리고 **볼륨 마운트**(데이터 영속의 핵심)
> - `docker-compose.yml` 로 선언적으로 운영하기
> - **인증** — 토큰으로 아무나 못 들어오게 막기
> - 헬스체크 · 로그 · 리소스 제한
> - 1편 코드를 `HttpClient` 로 붙이기
> - **백업과 복구** — 볼륨 스냅샷, 그리고 마이그레이션 주의점
> - Docker 운영에서 자주 밟는 지뢰

1편을 안 읽었어도 따라올 수 있게 썼지만, 거리 함수·임베딩 일관성 얘기는 1편에 있으니 같이 보시는 걸 추천드려요.


<br>

<br>



## 1. 왜 서버 모드인가

1편에서 ChromaDB 의 클라이언트 모드를 셋 정리했어요. 다시 짚으면 이래요.

| 모드 | 생성 코드 | 데이터 | 한계 |
|---|---|---|---|
| 인메모리 | `chromadb.Client()` | 프로세스 끝나면 소멸 | 테스트 전용 |
| 영속 | `chromadb.PersistentClient(path=...)` | 디스크 파일 | **한 프로세스만** 열 수 있음 |
| 서버 | `chromadb.HttpClient(host, port)` | 별도 서버가 관리 | 서버를 따로 띄워야 함 |

여기서 운영의 발목을 잡는 건 영속 모드의 한계예요. `PersistentClient` 는 SQLite 파일을 직접 여는 방식이라, **같은 디렉터리를 두 프로세스가 동시에 열면 충돌** 나요. 웹 서버를 여러 워커로 띄우거나, 배치 인입과 검색 API 가 같은 저장소를 봐야 하면 바로 막힙니다.

서버 모드는 이걸 깔끔하게 풉니다. ChromaDB 를 **하나의 서버 프로세스** 로 띄워두고, 앱들은 HTTP 로 붙어요. 저장소를 직접 만지는 건 서버 하나뿐이라 충돌이 없고, 앱을 몇 개를 띄우든 같은 데이터를 봅니다. 그리고 그 서버를 가장 손쉽게 격리해서 띄우는 도구가 Docker 예요.

> 💡 정리하면 — **혼자 쓰는 로컬 앱이면 `PersistentClient`, 둘 이상이 공유하거나 배포 대상이면 서버 모드(Docker)** 가 기준선이에요.


<br>

<br>



## 2. `docker run` 으로 띄우기 — 그리고 볼륨

공식 이미지가 있어서 시작은 한 줄이에요.

```shell
docker run -p 8000:8000 chromadb/chroma
```

이러면 `localhost:8000` 에 ChromaDB 서버가 떠요. 그런데 이대로 운영하면 **컨테이너를 지우는 순간 데이터가 같이 사라집니다.** 컨테이너 안에 쌓인 파일은 컨테이너의 수명과 같이 가거든요. 1편에서 인메모리 모드의 "껐다 켜니 비어있다" 함정을 얘기했는데, Docker 에선 **볼륨을 안 붙이면** 똑같은 일이 컨테이너 단위로 일어나요.

그래서 운영의 첫 단추는 **볼륨 마운트** 예요. 호스트의 디렉터리를 컨테이너 안 데이터 경로에 연결해서, 컨테이너가 죽거나 새로 떠도 데이터는 호스트에 남게 합니다.

```shell
docker run -d \
  --name chroma \
  -p 8000:8000 \
  -v "$(pwd)/chroma_data:/chroma/chroma" \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  --restart unless-stopped \
  chromadb/chroma
```

옵션을 하나씩 풀면 이래요.

| 옵션 | 의미 |
|---|---|
| `-d` | 백그라운드(데몬) 실행 |
| `--name chroma` | 컨테이너 이름 고정 (로그·재시작에 편함) |
| `-p 8000:8000` | 호스트 8000 → 컨테이너 8000 |
| `-v .../chroma_data:/chroma/chroma` | **호스트 디렉터리를 데이터 경로에 마운트** |
| `-e IS_PERSISTENT=TRUE` | 디스크 영속 모드로 동작 |
| `-e ANONYMIZED_TELEMETRY=FALSE` | 익명 텔레메트리 끄기 |
| `--restart unless-stopped` | 서버 재부팅·크래시 후 자동 재기동 |

> 🚨 **컨테이너 안 데이터 경로는 버전에 따라 달라요.** 오래 쓰인 경로는 `/chroma/chroma` 인데, 일부 버전은 `PERSIST_DIRECTORY` 환경변수로 경로를 바꾸거나 기본값이 다를 수 있어요. 마운트가 헛돌면(재시작 시 비어있으면) 컨테이너 안에서 실제 저장 경로부터 확인하세요.

```shell
# 컨테이너 안에서 실제 데이터가 어디 쌓이는지 확인
docker exec chroma sh -c 'echo $PERSIST_DIRECTORY; ls -al /chroma/chroma'
```

마운트가 제대로 걸렸으면, 컨테이너를 지웠다 다시 띄워도 `chroma_data/` 안의 SQLite + 인덱스 파일이 그대로 남아서 데이터가 살아있어요. 이게 운영의 출발선이에요.


<br>

<br>



## 3. `docker-compose` 로 선언적으로 운영하기

`docker run` 옵션이 길어지면 관리가 어려워요. 포트·볼륨·환경변수·재시작 정책을 한 파일에 적어두는 `docker-compose.yml` 로 옮기는 걸 추천드립니다. 다음에 띄울 때 `docker compose up -d` 한 줄이면 끝나거든요.

```yaml
services:
  chroma:
    image: chromadb/chroma:0.5.23      # 태그를 고정 (latest 금지)
    container_name: chroma
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma    # 데이터 영속
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 5s
      retries: 3
```

띄우고 상태를 확인해요.

```shell
docker compose up -d
docker compose ps
```

```text
NAME      IMAGE                   STATUS                   PORTS
chroma    chromadb/chroma:0.5.23  Up 12 seconds (healthy)  0.0.0.0:8000->8000/tcp
```

`(healthy)` 가 뜨면 헬스체크까지 통과한 거예요.

> ⚠️ **`image: chromadb/chroma:latest` 는 운영에서 피하세요.** `latest` 는 어느 날 갑자기 메이저 버전이 올라가서 API 가 바뀌거나(1편의 "버전 차이" 지뢰 기억나시죠) 데이터 포맷이 달라질 수 있어요. **버전 태그를 박아두고**, 올릴 때는 의도적으로 올리세요.


<br>

<br>



## 4. 인증 — 아무나 못 들어오게

기본 ChromaDB 서버는 **인증이 없어요.** 8000 포트에 닿을 수 있는 사람이면 누구나 컬렉션을 읽고 지울 수 있다는 뜻이에요. 로컬 테스트면 상관없지만, 사내망이든 클라우드든 **공유 환경에 띄우는 순간 인증은 필수** 입니다.

가장 간단한 건 **토큰 인증** 이에요. 서버에 토큰을 심어두고, 클라이언트가 같은 토큰을 헤더로 보내야만 들어올 수 있게 합니다. compose 의 `environment` 에 두 줄을 더해요.

```yaml
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
      - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
      - CHROMA_SERVER_AUTHN_CREDENTIALS=<CHROMA_TOKEN>   # 실제 토큰은 .env 로 빼기
```

토큰 값은 코드·compose 에 박지 말고 `.env` 파일이나 시크릿으로 빼세요. compose 는 `${VAR}` 로 `.env` 를 자동으로 읽어요.

```yaml
      - CHROMA_SERVER_AUTHN_CREDENTIALS=${CHROMA_TOKEN}
```

```shell
# .env (git 에 올리지 말 것 — .gitignore 에 추가)
CHROMA_TOKEN=<여기에-충분히-긴-랜덤-문자열>
```

클라이언트(1편 코드)는 이렇게 토큰을 들고 붙어요.

```python
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="<CHROMA_TOKEN>",   # os.environ 에서 읽기
    ),
)

print(client.heartbeat())   # 토큰이 맞아야 응답이 옴
```

토큰이 틀리거나 없으면 `401` 이 떨어져요.

> 🚨 **인증과 네트워크 노출은 세트로 생각하세요.** 토큰을 걸어도 8000 포트를 인터넷에 그대로 열면 위험해요. 가능하면 ChromaDB 는 **내부망에서만** 닿게 두고(쿠버네티스라면 `ClusterIP`, 단일 서버라면 방화벽), 외부 노출이 꼭 필요하면 리버스 프록시(HTTPS) 뒤에 두세요. 토큰은 평문으로 헤더에 실리니 **HTTPS 가 아니면 토큰도 새어요.**

기본 인증(아이디·비밀번호)이 더 맞으면 `htpasswd` 기반 basic auth 도 지원해요. 다만 여러 앱이 붙는 서버-투-서버 구조에선 토큰 쪽이 다루기 편합니다.


<br>

<br>



## 5. 헬스체크 · 로그 · 리소스

운영은 "떠 있는지 알 수 있어야" 운영이에요. ChromaDB 는 가벼운 헬스 엔드포인트를 제공해요.

```shell
curl http://localhost:8000/api/v1/heartbeat
```

```text
{"nanosecond heartbeat": 1718700000000000000}
```

> ⚠️ 헬스 엔드포인트 경로도 버전을 타요. 신버전(v2 API)에서는 `/api/v2/heartbeat` 로 바뀌었어요. compose 의 `healthcheck` 가 자꾸 `unhealthy` 면 설치된 버전의 경로부터 확인하세요.

로그는 컨테이너 표준출력으로 나와요. 문제가 생기면 여기부터 봅니다.

```shell
docker logs -f --tail 100 chroma
```

리소스도 막아두는 게 좋아요. 임베딩을 서버에서 돌리거나 대량 인입이 들어오면 메모리를 훅 먹을 수 있거든요. compose 에 한도를 둬요.

```yaml
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4g
```

> 💡 ChromaDB 의 메모리는 **컬렉션 크기·인덱스(HNSW)·동시 쿼리** 에 비례해요. 문서가 수십만 건을 넘어가면 limit 을 넉넉히 잡고, OOM 으로 컨테이너가 자꾸 죽으면 메모리부터 올리세요.


<br>

<br>



## 6. 1편 코드를 서버에 붙이기

이제 1편에서 짠 코드를 서버 모드로 옮기면 돼요. 바뀌는 건 **클라이언트를 만드는 첫 줄뿐** 이에요. 컬렉션·`add`·`query`·`where` 필터는 1편과 100% 똑같이 동작합니다.

```python
import chromadb
from chromadb.utils import embedding_functions

# PersistentClient → HttpClient 로만 교체
client = chromadb.HttpClient(host="localhost", port=8000)

# 임베딩 함수는 1편과 동일하게 (넣을 때·찾을 때 같은 모델!)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=["연차 휴가는 입사 1년 후 15일이 부여됩니다."],
    ids=["hr-001"],
    metadatas=[{"category": "휴가", "year": 2026}],
)

res = collection.query(query_texts=["휴가 며칠 쓸 수 있어?"], n_results=1)
print(res["documents"][0])
```

```text
['연차 휴가는 입사 1년 후 15일이 부여됩니다.']
```

> 🚨 **임베딩이 어디서 도느냐를 의식하세요.** 위처럼 `SentenceTransformerEmbeddingFunction` 을 클라이언트 쪽에 두면 임베딩은 **앱(클라이언트)에서** 계산돼서 벡터만 서버로 갑니다. 즉 서버 컨테이너엔 임베딩 모델이 없어도 돼요. 반대로 서버 쪽에서 임베딩하게 구성하면 모델을 서버 이미지에 넣어야 하고요. 1편에서 강조한 **"넣을 때와 찾을 때 같은 임베딩 함수"** 규칙은 서버 모드에서도 그대로예요 — 컬렉션에 붙는 모든 클라이언트가 같은 모델을 써야 합니다.


<br>

<br>



## 7. 백업과 복구

데이터를 디스크에 영속시켰다고 끝이 아니에요. 디스크가 날아가거나, 잘못 지우거나, 버전 올리다 깨질 수 있어요. ChromaDB 의 백업은 의외로 단순해요 — **볼륨 디렉터리를 통째로 복사** 하면 됩니다. 그 안에 SQLite + 인덱스가 다 들어있거든요.

가장 안전한 건 **서버를 잠깐 멈추고** 복사하는 거예요. 쓰는 도중에 복사하면 SQLite 가 깨진 상태로 떠질 수 있어요.

```shell
# 일관성 있는 백업: 멈추고 → 압축 → 다시 띄우기
docker compose stop chroma
tar czf "chroma_backup_2026-06-18.tar.gz" chroma_data/
docker compose start chroma
```

복구는 반대로, 멈춘 상태에서 풀어 넣고 다시 띄워요.

```shell
docker compose stop chroma
rm -rf chroma_data/
tar xzf "chroma_backup_2026-06-18.tar.gz"
docker compose start chroma
```

> ⚠️ **버전 간 복구는 조심하세요.** 백업을 받을 때와 복구할 때의 ChromaDB **버전이 다르면** 데이터 포맷이 안 맞아 안 떠질 수 있어요. 백업 파일 이름이나 메모에 **그때의 이미지 태그(`chromadb/chroma:0.5.23`)를 같이 남겨두는** 걸 추천드려요. 멈출 수 없는 운영이라면 파일 복사 대신 **새 컬렉션으로 재인입(원본 문서에서 다시 `add`)** 하는 경로를 백업 전략으로 두는 게 더 안전할 때도 많아요.


<br>

<br>



## 8. Docker 운영에서 자주 밟는 지뢰

1편이 "검색 품질" 지뢰였다면, 이번엔 "운영" 지뢰예요.

- **볼륨 안 붙임** — `docker run -p 8000:8000 chromadb/chroma` 로 띄워 운영하다 컨테이너 재생성 때 데이터 전멸. `-v` 로 볼륨부터 붙이세요.
- **마운트 경로 오타** — 볼륨은 붙였는데 컨테이너 안 실제 저장 경로와 안 맞아서 여전히 휘발. `docker exec` 로 실제 경로 확인.
- **`latest` 태그 운영** — 어느 날 자동으로 메이저 버전이 올라가 API·포맷이 바뀜. 태그 고정.
- **인증 없이 공유 노출** — 토큰 없이 8000 을 내부망·인터넷에 열어 누구나 컬렉션 삭제 가능. 토큰 + 네트워크 차단 세트로.
- **토큰을 git 에 커밋** — compose·코드에 토큰 박아 푸시. `.env` + `.gitignore`, 코드는 `os.environ`.
- **헬스체크 경로 버전 불일치** — `/api/v1` vs `/api/v2` 로 `unhealthy` 가 떠서 멀쩡한 서버가 죽은 줄 앎. 설치 버전 경로 확인.
- **쓰는 중 백업** — 서버 돌아가는 채로 `chroma_data/` 복사해서 깨진 SQLite 를 백업. 멈추고 복사.
- **메모리 한도 없이 대량 인입** — OOM 으로 컨테이너가 반복 재시작. `limits.memory` 설정 + 배치 크기 조절.


<br>

<br>



## 마무리

ChromaDB 를 Docker 로 운영한다는 건 결국 네 가지를 챙기는 거였어요 — **볼륨으로 데이터 지키고, compose 로 선언적으로 띄우고, 토큰으로 막고, 스냅샷으로 백업하기.** 여기까지 하면 단일 서버에서 ChromaDB 를 "운영" 이라 부를 수 있는 상태가 됩니다.

처음엔 `docker compose up -d` + 볼륨 + 토큰, 이 조합으로 시작해보세요. 단일 서버로 충분한 규모라면 사실 이걸로 끝이에요.

일단 오늘은 여기까지.....   
다음 글에서는 이 단일 서버 한 대를 넘어서, 공식 이미지를 **ECR 로 미러링** 해서 **EKS(쿠버네티스)에 StatefulSet + PVC 로 올려 운영** 하는 그림을 정리해볼게요. 상태가 있는 벡터 DB 를 쿠버네티스에서 안전하게 굴리는 게 포인트예요.

---

**← 이전 글:** [(1/4) ChromaDB 를 RAG 용 벡터 DB 로 제대로 쓰기](/coding/ChromaDB_RAG_벡터DB_제대로쓰기/) ｜ **다음 글 →:** [(3/4) ChromaDB 를 ECR·EKS 로 올려 운영하기](/coding/ChromaDB_ECR_EKS_운영/)
