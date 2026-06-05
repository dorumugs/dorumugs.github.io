---
layout: single
title:  "(2/5) Helm 으로 Airflow 를 K8s 에 설치하기 — KubernetesExecutor 셋업"
date: 2026-06-04 21:10:00 +0900
description: "공식 apache-airflow Helm 차트로 Airflow 를 K8s 위에 처음부터 끝까지 올리는 셋업. Namespace · Secret 미리 준비, values.yaml 최소 셋, KubernetesExecutor 켜는 한 줄까지."
categories: coding
tag: [Airflow, Kubernetes, K8s, Helm, KubernetesExecutor, Postgres, values, 셋업, DevOps]
author_profile: false
toc: true
---

{% include series-airflow-k8s-helm.html current="2" %}

# Summary

(1/5) 에서 본 8~9 개 오브젝트 묶음을 실제로 K8s 에 올릴 차례. 공식 `apache-airflow` Helm 차트가 거의 다 해주고, 우리가 할 일은 두 가지뿐이에요. **(1) `Namespace` 와 두 개의 `Secret` 을 미리 만들고, (2) `values.yaml` 에 최소 옵션만 채워서 `helm install`.**

> 📚 이 글에서 만들거나 다루는 오브젝트는 전부 [K8s YAML 입문 글](/coding/K8s_YAML_오브젝트_7가지_입문/) 의 7개 안쪽이에요. 우리가 손으로 만들 건 `Namespace` 와 `Secret` 두 개뿐이고, 나머지(Airflow 3.x 의 `Deployment` × 4 — scheduler / api-server / triggerer / dag-processor — 와 `Service`, `ConfigMap`, 선택적 `Ingress`) 는 Helm 이 알아서 깔아줍니다.

> 💡 이 글에서 다루는 것
> - Helm repo 등록 / 차트 버전 확인
> - `Namespace` 만들고, Fernet 키 / API server 세션 `Secret` 두 개 미리 박기
> - `values.yaml` 최소 셋 (executor, DB, secret 참조)
> - `helm install` 한 방
> - 첫 접속 확인 + 자주 겪는 함정


<br>

<br>



## 1. Helm repo 등록과 차트 확인

```shell
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm search repo apache-airflow/airflow --versions | head -5
```

차트가 기본으로 깔아주는 Airflow 이미지 태그도 같이 확인해두면 좋아요.

```shell
helm show values apache-airflow/airflow \
  | grep -E "^(airflowVersion|defaultAirflowTag|defaultAirflowRepository):"
```

워커 이미지 커스텀은 (3/5) 에서 다루니까 일단 디폴트로 갑니다.


<br>

<br>



## 2. Namespace 와 미리 만들 Secret 두 개

먼저 K8s 의 기본 단위인 `Namespace` 부터 하나 만들어요.

```shell
kubectl create namespace airflow
```

그리고 Airflow 가 꼭 필요로 하는 두 개의 `Secret` 을 **미리** 박아둡니다. 차트가 알아서 만들어주기도 하는데, **명시적으로 만들면 `helm upgrade` 때 값이 바뀌면서 토큰이 갈리는 사고**를 막을 수 있어요.

두 키 모두 일반 `Secret` 패턴이라 `kubectl create secret` 한 줄로 처리할 수 있어요.

```shell
# 1) Fernet key — Airflow connection 암호화. 잃어버리면 기존 connection 다 못 읽음
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# 출력 예: 6q1m...XYZ=

kubectl -n airflow create secret generic airflow-fernet-key \
  --from-literal=fernet-key='<위에서_나온_키>'

# 2) API server 세션 시크릿 — Flask 세션 서명용
kubectl -n airflow create secret generic airflow-webserver-secret \
  --from-literal=webserver-secret-key="$(openssl rand -hex 32)"
```

> 🚨 두 키는 **한 번 바뀌면 복구가 까다로워요**. 운영 환경이면 Vault / Sealed Secrets 같은 곳에 백업해두세요. 잃어버리면 기존 connection 비밀번호를 전부 다시 입력해야 합니다.


<br>

<br>



## 3. values.yaml — 최소 셋

차트의 `values.yaml` 은 옵션이 천 줄 단위예요. 처음엔 **꼭 필요한 것만** 덮어쓰고 나머지는 디폴트로 갑니다.

```yaml
# values.yaml
executor: KubernetesExecutor

# 위에서 만든 Secret 을 가리키게
fernetKeySecretName: airflow-fernet-key
webserverSecretKeySecretName: airflow-webserver-secret

# 메타데이터 DB — 실험은 차트 내장 Postgres, 운영은 외부 RDS 권장
postgresql:
  enabled: true
  auth:
    postgresPassword: "change-me-postgres-admin"
    username: airflow
    password: "change-me-airflow-db"

# Airflow 가 쓸 DB 이름은 chart 의 data 블록에서 (postgresql.auth.database 가 아닌 점 주의)
data:
  metadataConnection:
    db: airflow

# KubernetesExecutor 는 Redis 안 씀
redis:
  enabled: false

# Airflow 3.x 의 상시 컴포넌트 4개 — 각각 Deployment 한 개씩
scheduler:
  replicas: 1
apiServer:           # Airflow 3.x 의 새 이름 (= 2.x 의 webserver)
  replicas: 1
  service:
    type: ClusterIP
triggerer:
  enabled: true
  replicas: 1
dagProcessor:        # 3.x 부터 standalone 필수
  enabled: true
  replicas: 1

# 초기 관리자 계정 — Airflow 3.x 의 createUserJob 블록 사용
createUserJob:
  defaultUser:
    enabled: true
    username: admin
    password: "change-me-admin"
    email: admin@example.com
    role: Admin

# 첫 설치엔 예제 DAG 끄고 시작
config:
  core:
    load_examples: "False"
```

핵심 단 한 줄은 `executor: KubernetesExecutor`. 이게 켜져 있어야 워커가 매 태스크마다 새 `Pod` 로 뜹니다.

> 💡 운영에선 거의 항상 외부 DB 를 권장합니다. `postgresql.enabled: false` 로 끄고, 차트의 `data.metadataConnection` (또는 `data.metadataSecretName`) 으로 외부 RDS 접속 정보를 주입해요.


<br>

<br>



## 4. helm install 한 방

```shell
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --values values.yaml \
  --timeout 10m
```

- `upgrade --install` 패턴: 처음이면 install, 이미 있으면 upgrade. 멱등하게 굴리기 좋아요.
- 첫 설치는 이미지 풀링 + DB 마이그레이션이 같이 도는 시간이라 5분 가까이 걸리는 경우도 있어요.

진행 상황은 다른 터미널에서 같이 봐주세요.

```shell
watch -n 2 "kubectl -n airflow get pods"
```

정상이면 `airflow-scheduler-*`, `airflow-api-server-*`, `airflow-triggerer-*`, `airflow-dag-processor-*`, `airflow-postgresql-*` 가 모두 `Running` 상태가 돼요. 첫 기동 직후엔 `airflow-run-airflow-migrations-*` job pod 와 `airflow-create-user-*` job pod 가 잠깐 떴다 사라지기도 하는데, 이것도 정상이에요.


<br>

<br>



## 5. 첫 접속 확인

`Ingress` 셋업까지 안 가도 우선 UI 는 띄워볼 수 있어요. `kubectl port-forward` 한 줄이면 충분합니다.

```shell
kubectl -n airflow port-forward svc/airflow-api-server 8080:8080
```

브라우저에서 `http://localhost:8080`. 기본 계정은 위 `createUserJob.defaultUser` 에서 잡은 `admin / change-me-admin` 이에요. 운영에선 반드시 비번을 바꾸거나 SSO 로 교체해주세요.

> ✅ 첫 접속 체크
> - [x] 좌측 사이드바에 DAG 목록 (아직 비어있어야 함 — load_examples=False)
> - [x] Admin → Connections / Variables / Pools 정상
> - [x] Browse → Triggerer 살아있는지
> - [x] Settings → "About" 의 Airflow 버전이 의도한 버전


<br>

<br>



## 6. 자주 겪는 함정

설치 자체는 깔끔하지만 첫 한두 번은 거의 한 번씩 겪어요. 자주 보이는 패턴을 정리해둡니다.

| 증상 | 원인 / 처방 |
|---|---|
| `migrations` job 이 계속 실패 | Postgres 비번 불일치. `kubectl -n airflow logs job/airflow-run-airflow-migrations` |
| Pod 가 `Pending` | 노드 리소스 부족 또는 PVC 가 바인딩 안 됨. `kubectl describe pod` 의 Events |
| Pod 가 `ImagePullBackOff` | 사설 레지스트리인데 `imagePullSecrets` 안 줌. (3/5) 의 `registry.secretName` 항목 참고 |
| DAG 가 모두 빨간색 import error | 워커 이미지에 라이브러리가 없음. (3/5) 의 커스텀 이미지로 해결 |
| `Fernet key must be 32 url-safe base64-encoded bytes` | `Fernet.generate_key()` 결과 그대로 써야 함 (끝 `=` 포함) |

`helm upgrade --install` 은 멱등하니까 values 만 고쳐서 같은 명령 다시 때리면 돼요. 처음엔 자잘하게 두세 번 돌리게 됩니다.


<br>

<br>



## 7. 정리

여기까지 오면 (1/5) 의 표에 있던 8~9 개 오브젝트가 모두 `airflow` 네임스페이스에 들어와 있는 상태예요. 다만 워커 `Pod` 가 **어떤 이미지로, 어떤 스펙으로** 뜰지는 아직 우리가 잡지 않았어요. 디폴트 이미지엔 우리 DAG 가 필요로 하는 라이브러리(`pandas`, `requests`, 사내 패키지 …) 가 없을 가능성이 큽니다.

다음 편에서 그 두 가지를 한꺼번에 잡습니다 — **워커 이미지를 직접 굽고, `pod_template_file` 로 워커 Pod 의 스펙을 우리가 결정합니다.**

일단 오늘은 여기까지.....   
다음 글에서는 커스텀 워커 이미지와 `pod_template_file` 로 워커 Pod 를 우리 입맛대로 묶어볼게요.

---

**← 이전 글:** [(1/5) Airflow on K8s 시리즈 개요 — Helm 으로 올리고 워커는 컨테이너로 띄운다](/coding/Airflow_K8s_시리즈_개요/) ｜ **다음 글 →:** [(3/5) Airflow 워커 이미지 만들고 pod_template_file 로 묶기](/coding/Airflow_워커_이미지_pod_template/)
