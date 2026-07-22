---
layout: single
title:  "(3/4) ChromaDB 를 ECR·EKS 로 올려 운영하기 — 공식 이미지 미러링·StatefulSet·PVC"
date: 2026-06-28 10:00:00 +0900
categories: coding
tag: [ChromaDB, EKS, ECR, Kubernetes, StatefulSet, PVC, EBS, RAG, 벡터DB, 운영]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-18-chromadb-eks/header.svg
  teaser: /assets/images/2026-06-18-chromadb-eks/header.svg
description: "ChromaDB 를 EKS 에서 운영하는 법 — 공식 이미지를 ECR 로 미러링하고, 상태 있는 벡터 DB 를 StatefulSet + PVC(EBS)로 올리고, Service·probe·토큰 시크릿까지 실제 매니페스트와 함께 정리합니다."
series: chromadb-rag
series_order: 3
---

{% include series-chromadb-rag.html current="3" %}


## Summary

[2편](/coding/ChromaDB_Docker_서버모드_운영/)에서 ChromaDB 를 Docker 서버 한 대로 띄워 볼륨·인증·백업까지 챙겼어요. 단일 서버로 충분한 규모면 사실 거기서 끝나도 됩니다. 그런데 이미 서비스를 **EKS(쿠버네티스)** 위에서 굴리고 있다면, 벡터 DB 만 서버 한 대에 따로 두는 게 어색해져요. 배포·모니터링·네트워크를 클러스터 안에서 한 번에 관리하고 싶어지죠.

이 글에서는 ChromaDB 를 EKS 에 올려 운영하는 그림을 정리합니다. 핵심은 두 가지예요. 하나는 **공식 이미지를 ECR 로 미러링** 해서 클러스터가 안정적으로 이미지를 당기게 하는 것, 다른 하나는 **상태가 있는(stateful)** 벡터 DB 를 쿠버네티스답게 **StatefulSet + PVC** 로 올려서 파드가 죽거나 재배포돼도 데이터가 살아남게 하는 거예요.

> 💡 **이 글에서 다루는 것**
> - 왜 ChromaDB 는 EKS 에서 **StatefulSet** 이어야 하나 (Deployment 와의 차이)
> - **공식 이미지 → ECR 미러링** (pull → tag → push, pull-through cache)
> - **StatefulSet + PVC(EBS)** 매니페스트 — 데이터 영속의 핵심
> - **Service(ClusterIP)** — 클러스터 안에서 부르는 주소
> - readiness · liveness **probe**
> - 토큰 인증을 **Secret** 으로
> - 클러스터 안 RAG 앱에서 `HttpClient` 로 붙기
> - EKS 운영에서 자주 밟는 지뢰

쿠버네티스 기본기(파드·서비스·kubectl)는 안다고 가정하고, ChromaDB 에 특화된 부분에 집중할게요.


<br>

<br>



## 1. 왜 StatefulSet 인가

쿠버네티스에 뭘 올릴 때 보통 `Deployment` 를 먼저 떠올려요. 무상태(stateless) 앱엔 그게 맞아요 — 파드는 언제든 갈아끼울 수 있는 일회용이니까요. 그런데 ChromaDB 는 **디스크에 데이터를 쌓는 상태 있는 서버** 예요. 2편에서 볼륨이 핵심이었던 것과 같은 이유로, EKS 에서도 **각 파드가 자기 전용 디스크를 안정적으로 물고 있어야** 합니다.

그래서 `Deployment` 보다 `StatefulSet` 이 맞아요. 둘의 차이를 ChromaDB 관점에서만 추리면 이래요.

| | Deployment | StatefulSet |
|---|---|---|
| 파드 이름 | 랜덤 | 고정(`chroma-0`) |
| 디스크(PVC) | 공유·임시 성향 | **파드별 전용 PVC** |
| 재시작 후 | 다른 디스크 물 수 있음 | **같은 디스크 다시 물음** |

ChromaDB 에 필요한 건 "파드가 죽었다 살아나도 **아까 그 디스크** 를 다시 문다" 예요. StatefulSet 의 `volumeClaimTemplates` 가 정확히 그 일을 합니다.

> 🚨 **ChromaDB(오픈소스 단일 노드)는 수평 확장으로 처리량을 늘리는 구조가 아니에요.** 여러 레플리카를 띄워 같은 데이터를 공유하는 방식이 기본 지원되지 않아서, 운영은 보통 **레플리카 1개** 로 갑니다. 즉 EKS 에 올리는 이유는 "스케일아웃" 이 아니라 **클러스터 안에서 같이 관리·배포·모니터링** 하기 위함이에요. 트래픽이 정말 커지면 ChromaDB 단일 노드의 한계를 인정하고 분산형 벡터 DB 를 검토하는 게 맞습니다.


<br>

<br>



## 2. 공식 이미지를 ECR 로 미러링

EKS 에서 이미지를 당길 때 Docker Hub 를 직접 보게 두면, **레이트리밋·외부 장애·네트워크 정책** 에 휘둘려요. 그래서 공식 `chromadb/chroma` 이미지를 **ECR 로 한 번 미러링** 해두고, 클러스터는 ECR 만 보게 하는 게 운영 정석이에요. (이 시리즈의 선택대로 커스텀 빌드 없이 **공식 이미지를 그대로** 미러링합니다.)

가장 직관적인 방법은 **pull → tag → push** 예요. 먼저 ECR 에 로그인하고 리포지터리를 만들어요.

```shell
# 변수 (실제 값으로 치환)
ACCOUNT_ID=<ACCOUNT_ID>
REGION=<REGION>
ECR="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# ECR 로그인
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR"

# 리포지터리 생성 (한 번만)
aws ecr create-repository --repository-name chromadb/chroma --region "$REGION"
```

그다음 공식 이미지를 받아서 ECR 태그를 붙여 올려요.

```shell
docker pull chromadb/chroma:0.5.23
docker tag  chromadb/chroma:0.5.23 "$ECR/chromadb/chroma:0.5.23"
docker push "$ECR/chromadb/chroma:0.5.23"
```

> 💡 **버전 태그를 그대로 들고 가세요.** 2편에서 `latest` 금지를 얘기했는데, ECR 미러링에서도 똑같아요. `chromadb/chroma:0.5.23` 처럼 받은 버전을 그대로 ECR 태그로 박아두면, 나중에 "지금 EKS 에 떠 있는 게 정확히 어느 버전인가" 가 명확해져요.

매번 손으로 미러링하기 번거로우면 ECR 의 **pull-through cache** 를 쓰는 방법도 있어요. Docker Hub(또는 공개 레지스트리)를 업스트림으로 등록해두면, 클러스터가 ECR 주소로 이미지를 요청할 때 ECR 이 알아서 한 번 당겨와 캐시해줘요. 처음 한 번만 외부를 보고, 그다음부터는 ECR 에서 빠르게 나옵니다. 자주 버전을 바꾸거나 여러 이미지를 미러링한다면 이쪽이 편해요.

마지막으로, EKS 노드가 ECR 에서 이미지를 당길 **권한** 이 있어야 해요. 보통 노드그룹의 IAM 역할에 `AmazonEC2ContainerRegistryReadOnly` 정책이 붙어 있으면 `imagePullSecrets` 없이도 같은 계정의 ECR 을 당길 수 있어요. 이게 안 되어 있으면 파드가 `ImagePullBackOff` 로 멈춥니다.


<br>

<br>



## 3. StatefulSet + PVC(EBS)

이제 본체예요. StatefulSet 으로 ChromaDB 파드를 띄우고, `volumeClaimTemplates` 로 **EBS 볼륨(PVC)** 을 파드에 물려요. 2편의 `-v` 볼륨 마운트가 쿠버네티스에서 이 형태로 바뀌는 거예요.

먼저 EBS 를 쓰는 StorageClass 가 있어야 해요. EKS 라면 보통 **EBS CSI 드라이버** 가 깔려 있고, gp3 용 StorageClass 를 이렇게 둬요.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
volumeBindingMode: WaitForFirstConsumer    # 파드가 뜰 AZ 에 맞춰 볼륨 생성
reclaimPolicy: Retain                       # PVC 지워도 데이터는 남김(안전)
```

> 🚨 **`volumeBindingMode: WaitForFirstConsumer` 가 중요해요.** EBS 볼륨은 **특정 가용영역(AZ)에 묶여요.** 볼륨을 먼저 A-AZ 에 만들어버렸는데 파드는 B-AZ 에 스케줄되면 영영 못 붙어요. `WaitForFirstConsumer` 는 파드가 어디에 뜰지 정해진 다음 그 AZ 에 볼륨을 만들어서 이 엇갈림을 막아줘요.

이제 StatefulSet 본체예요.

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: chroma
spec:
  serviceName: chroma          # 아래 headless Service 이름과 일치
  replicas: 1                  # 단일 노드 — 위 1절 참고
  selector:
    matchLabels:
      app: chroma
  template:
    metadata:
      labels:
        app: chroma
    spec:
      containers:
        - name: chroma
          image: <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/chromadb/chroma:0.5.23
          ports:
            - containerPort: 8000
          env:
            - name: IS_PERSISTENT
              value: "TRUE"
            - name: ANONYMIZED_TELEMETRY
              value: "FALSE"
            - name: CHROMA_SERVER_AUTHN_PROVIDER
              value: "chromadb.auth.token_authn.TokenAuthenticationServerProvider"
            - name: CHROMA_SERVER_AUTHN_CREDENTIALS
              valueFrom:
                secretKeyRef:
                  name: chroma-auth      # 아래 6절 Secret
                  key: token
          volumeMounts:
            - name: data
              mountPath: /chroma/chroma  # 2편과 같은 데이터 경로
          resources:
            requests:
              cpu: "500m"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 4Gi
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: ebs-gp3
        resources:
          requests:
            storage: 20Gi
```

`volumeClaimTemplates` 덕분에 `chroma-0` 파드는 자기 전용 PVC(`data-chroma-0`)를 받고, 파드가 재시작·재스케줄돼도 **같은 EBS 볼륨을 다시 물어요.** 2편의 호스트 볼륨이 EKS 에선 EBS 로 바뀐 것뿐, 데이터 영속이라는 본질은 똑같아요.


<br>

<br>



## 4. Service — 클러스터 안 주소

파드만으론 다른 앱이 부를 주소가 없어요. StatefulSet 앞에 `Service` 를 둬서 안정적인 DNS 이름을 줍니다. 외부에 열 필요가 없으니 **`ClusterIP`(클러스터 내부 전용)** 면 충분해요.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: chroma
spec:
  clusterIP: None        # headless — StatefulSet 의 안정적 파드 DNS 제공
  selector:
    app: chroma
  ports:
    - port: 8000
      targetPort: 8000
```

이러면 같은 네임스페이스의 앱은 `http://chroma:8000` 으로, 다른 네임스페이스면 `http://chroma.<namespace>.svc.cluster.local:8000` 으로 붙을 수 있어요.

> 🚨 **외부에 함부로 열지 마세요.** ChromaDB 를 `LoadBalancer` 나 Ingress 로 인터넷에 직접 노출하면, 2편에서 말한 "8000 포트 = 누구나 컬렉션 삭제" 위험이 그대로 인터넷 규모가 돼요. 기본은 `ClusterIP` 로 **내부에서만** 닿게 두고, 외부 접근이 꼭 필요하면 인증(토큰)+HTTPS+제한된 Ingress 를 따로 설계하세요.


<br>

<br>



## 5. probe — 살아있는지·받을 준비됐는지

쿠버네티스가 파드의 상태를 알려면 probe 가 필요해요. 2편의 `healthcheck` 가 쿠버네티스에선 `livenessProbe` / `readinessProbe` 로 나뉘어요. StatefulSet 의 컨테이너 스펙에 더해요.

```yaml
          livenessProbe:        # 죽었으면 재시작
            httpGet:
              path: /api/v1/heartbeat
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 30
          readinessProbe:       # 준비됐을 때만 트래픽 받음
            httpGet:
              path: /api/v1/heartbeat
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
```

- **liveness** 가 실패하면 쿠버네티스가 파드를 재시작해요. (먹통 복구)
- **readiness** 가 실패하면 Service 가 그 파드로 트래픽을 안 보내요. (뜨는 중 보호)

> ⚠️ 2편에서 짚은 것처럼 헬스 경로는 버전을 타요. 신버전이면 `/api/v2/heartbeat` 로 바꿔야 probe 가 통과해요. probe 가 계속 실패하면 파드가 무한 재시작(`CrashLoopBackOff` 비슷하게) 도니, 배포 전에 경로부터 맞추세요.


<br>

<br>



## 6. 토큰 인증을 Secret 으로

2편에서 토큰을 `.env` 로 뺐듯이, EKS 에선 **Secret** 으로 빼요. 매니페스트(StatefulSet)에는 토큰 값이 안 보이고 `secretKeyRef` 로 참조만 하니, git 에 매니페스트를 올려도 토큰은 안 새요.

```shell
kubectl create secret generic chroma-auth \
  --from-literal=token='<CHROMA_TOKEN>'
```

위 3절 StatefulSet 의 `CHROMA_SERVER_AUTHN_CREDENTIALS` 가 이 Secret 의 `token` 키를 `secretKeyRef` 로 읽어요. 서버는 이 토큰을 든 요청만 받아들입니다.

> 🚨 **Secret 은 기본적으로 base64 일 뿐 암호화가 아니에요.** etcd 저장 암호화(KMS), RBAC 로 Secret 접근 제한, git 에는 평문 Secret 매니페스트를 올리지 않기(SealedSecrets·External Secrets 등) — 이 셋을 같이 챙기세요. 토큰을 Secret 에 넣었다고 끝이 아니라 **누가 그 Secret 을 읽을 수 있는가** 까지 봐야 해요.


<br>

<br>



## 7. 클러스터 안 RAG 앱에서 붙기

이제 같은 클러스터에 떠 있는 RAG 앱(API 서버 등)에서 ChromaDB 에 붙어요. 2편의 `HttpClient` 코드에서 **`host` 만 Service 이름으로** 바꾸면 됩니다.

```python
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# host 를 쿠버네티스 Service 이름으로 (같은 네임스페이스면 "chroma")
client = chromadb.HttpClient(
    host="chroma",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.environ["CHROMA_TOKEN"],
    ),
)

# 임베딩 함수는 1·2편과 동일 (넣을 때·찾을 때 같은 모델!)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

print(collection.count())
```

```text
3
```

앱 쪽에도 같은 `CHROMA_TOKEN` 을 Secret 으로 주입하면, 앱과 ChromaDB 가 같은 토큰으로 안전하게 통신해요. 컬렉션·`add`·`query`·`where` 필터는 1편과 글자 하나 안 바뀌고 그대로 동작합니다 — **바뀐 건 ChromaDB 가 사는 위치뿐** 이에요.


<br>

<br>



## 8. EKS 운영에서 자주 밟는 지뢰

시리즈를 마무리하는 운영 체크리스트예요.

- **Deployment 로 올림** — 무상태처럼 띄워서 재배포 때 디스크가 엇갈림. StatefulSet + `volumeClaimTemplates`.
- **EBS AZ 엇갈림** — 볼륨과 파드가 다른 AZ 에 떨어져 영영 안 붙음. `WaitForFirstConsumer` 로 바인딩 지연.
- **`reclaimPolicy: Delete` 방치** — PVC 를 지우자 EBS 데이터까지 삭제. 중요한 데이터면 `Retain`.
- **레플리카를 늘림** — 처리량 늘리려 `replicas: 3` 했다가 같은 데이터 공유가 안 돼 깨짐. 단일 노드 한계 인정.
- **`latest` 미러링** — ECR 에 `latest` 로 밀어 어느 날 버전이 점프. 버전 태그 고정.
- **ECR pull 권한 없음** — 노드 IAM 에 ECR read 정책이 없어 `ImagePullBackOff`. `AmazonEC2ContainerRegistryReadOnly` 확인.
- **probe 경로 버전 불일치** — `/api/v1` vs `/api/v2` 로 파드가 무한 재시작. 설치 버전 경로 확인.
- **외부 노출** — `LoadBalancer` 로 인터넷에 직접 열어 인증 우회·삭제 위험. `ClusterIP` + 내부 전용 기본.
- **백업 전략 부재** — EBS 스냅샷·정기 백업 없이 운영하다 볼륨 사고로 전손. EBS 스냅샷 또는 원본에서 재인입 경로 확보.


<br>

<br>



## 마무리

여기까지를 한 줄로 묶으면 이래요 — **1편에서 ChromaDB 로 검색 품질을 다루는 법을 익히고, 2편에서 Docker 로 한 대를 운영하고, 3편에서 EKS 로 클러스터에 올렸어요.** 개념이 같은 임베딩·거리 함수 위에서, ChromaDB 가 사는 위치만 인메모리 → 디스크 → 컨테이너 → 쿠버네티스로 한 칸씩 올라온 거예요. 코드의 본체(컬렉션·`add`·`query`·`where`)는 끝까지 안 바뀌었다는 게 핵심이고요.

EKS 에 올릴 땐 **공식 이미지를 ECR 로 미러링 → StatefulSet + PVC(EBS) → ClusterIP Service → 토큰 Secret**, 이 순서를 기준선으로 잡으세요. 단일 노드 한계만 의식하면, 사내 RAG 정도의 규모는 이 구성으로 충분히 안정적으로 굴러가요.

일단 오늘은 여기까지.....   
이제 ChromaDB 가 좋은 조각을 돌려주는 데까지 왔어요. 마지막 4편에서는 이렇게 올린 ChromaDB 에서 검색해온 조각을 LLM 프롬프트에 끼워 넣어 **실제 답변을 만드는 RAG 파이프라인** 을 정리하면서 시리즈를 마무리할게요.

---

**← 이전 글:** [(2/4) ChromaDB 를 Docker 로 띄우고 운영하기 — 볼륨·인증·백업](/coding/ChromaDB_Docker_서버모드_운영/) ｜ **다음 글 →:** [(4/4) ChromaDB RAG 파이프라인 — 검색 조각으로 답변 만들기](/coding/ChromaDB_RAG_파이프라인_LLM_답변생성/)
