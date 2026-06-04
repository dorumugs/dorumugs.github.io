---
layout: single
title:  "(2/4) Helm 으로 Airflow 를 K8s 에 설치하기 — KubernetesExecutor 셋업"
date: 2026-06-04 12:00:00 +0900
description: "공식 apache-airflow Helm 차트로 Airflow 를 K8s 위에 올리는 끝-에서-끝 셋업. namespace, values.yaml, KubernetesExecutor, Fernet/Secret 까지 처음부터 짚어요."
categories: coding
tag: [Airflow, Kubernetes, K8s, Helm, KubernetesExecutor, Postgres, values, 셋업, DevOps]
author_profile: false
toc: true
---

{% include series-airflow-k8s-helm.html current="2" %}

# Summary

이제 손에 더러운 일을 시켜볼 시간이에요. 이 글에서는 공식 `apache-airflow` Helm 차트를 받아서 K8s 클러스터에 Airflow 를 한 번에 올립니다. 핵심은 `executor: KubernetesExecutor` 로 켜고, 메타데이터 DB (`Postgres`) 와 보안 키(`Fernet`/`webserverSecret`) 만 깔끔하게 잡아주는 거예요.

> 💡 이 글에서 다루는 것
> - Helm repo 등록 / 차트 버전 확인
> - 전용 namespace 만들기
> - `values.yaml` 최소 셋 — executor, DB, Fernet/secret
> - `helm install` 한 방
> - 첫 접속 확인 + 흔한 함정


<br>

<br>



## 1. Helm repo 등록과 차트 확인

먼저 공식 차트 repo 를 등록해요.

```shell
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm search repo apache-airflow/airflow --versions | head -5
```

`apache-airflow/airflow` 가 보이면 OK. 글 작성 시점 기준으로 `1.x` 대 차트 + Airflow `2.x` 이미지 조합이 표준이에요. 본인이 받은 차트가 어떤 Airflow 버전을 기본으로 쓰는지는 다음으로 빠르게 확인할 수 있어요.

```shell
helm show values apache-airflow/airflow | grep -E "^(airflowVersion|defaultAirflowTag|defaultAirflowRepository):"
```

이게 우리가 따로 안 만지면 깔리는 기본 이미지/태그예요. 일단은 **건드리지 않고** 갑니다. 워커 이미지 커스텀은 (3/4) 편에서 다뤄요.


<br>

<br>



## 2. namespace 와 시크릿 자리 잡기

전용 namespace 를 하나 만들고 거기로만 다 몰아넣어요.

```shell
kubectl create namespace airflow
```

Airflow 가 꼭 필요로 하는 두 가지 시크릿을 미리 만들어요. 차트가 알아서 생성해주기도 하지만, **명시적으로 만들어두면 helm upgrade 시 값이 갱신되며 토큰이 바뀌는 사고**를 막을 수 있어요.

```shell
# 1) Fernet key — Airflow connection 암호화에 사용. 한 번 바뀌면 기존 connection 전부 못 읽음.
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# 출력 예: 6q1m...XYZ=

kubectl -n airflow create secret generic airflow-fernet-key \
  --from-literal=fernet-key='<위에서_나온_키>'

# 2) Webserver secret key — Flask 세션 서명용. 32바이트 랜덤이면 충분.
kubectl -n airflow create secret generic airflow-webserver-secret \
  --from-literal=webserver-secret-key="$(openssl rand -hex 32)"
```

> 🚨 두 키는 **한 번 바뀌면 복구가 까다로워요**. 운영 환경이라면 Vault / Sealed Secrets 같은 곳에 백업해두세요. 잃어버리면 기존 connection 비밀번호를 전부 다시 입력해야 해요.


<br>

<br>



## 3. values.yaml 최소 셋

차트의 `values.yaml` 은 옵션이 진짜 많아요(천 줄 단위). 처음에는 **꼭 필요한 것만** 덮어쓰고 나머지는 디폴트로 갑니다.

```yaml
# values.yaml
executor: KubernetesExecutor

# Fernet / Webserver secret 을 우리가 만든 시크릿에서 가져오게
fernetKeySecretName: airflow-fernet-key
webserverSecretKeySecretName: airflow-webserver-secret

# 메타데이터 DB — 실험은 차트 내장 Postgres, 운영은 외부 RDS 권장
postgresql:
  enabled: true        # 차트가 Postgres 를 같이 띄움 (운영에서는 false)
  auth:
    postgresPassword: "change-me-postgres-admin"
    username: airflow
    password: "change-me-airflow-db"
    database: airflow

# Redis 는 KubernetesExecutor 에서는 필요 없음 — 끔
redis:
  enabled: false

# 워커는 태스크 단위 Pod 라 따로 deployment 없음. scheduler/webserver/triggerer 만 둠
scheduler:
  replicas: 1
webserver:
  replicas: 1
triggerer:
  enabled: true
  replicas: 1

# 외부 노출 — 일단은 ClusterIP + port-forward 로 확인, 나중에 ingress 로 교체
webserver:
  service:
    type: ClusterIP
```

⚠️ 위 YAML 에 `webserver:` 키가 두 번 나오는데, 실제 파일에서는 **하나로 합쳐서** 적으세요. 위/아래로 따로 두면 뒤쪽이 덮어써요. 여기선 가독성을 위해 나눠 적었어요.

> 💡 외부 Postgres 를 쓰고 싶으면 `postgresql.enabled: false` 로 끄고, `data.metadataConnection` (또는 `data.metadataSecretName`) 으로 외부 DB 접속 정보를 주입해요. 운영에서는 거의 항상 외부 DB 권장.


<br>

<br>



## 4. helm install 한 방

values 파일 준비 끝났으면 한 줄로 깔립니다.

```shell
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --values values.yaml \
  --timeout 10m
```

- `upgrade --install` 패턴: 처음이면 install, 이미 있으면 upgrade. 멱등하게 굴리기 좋아요.
- `--timeout 10m`: 첫 설치 시 이미지 풀링 + DB 마이그레이션이 같이 돌아서 5분 가까이 걸리는 경우가 있어요.

깔리는 동안 다른 터미널에서 상태를 봐주면 답답함이 줄어요.

```shell
watch -n 2 "kubectl -n airflow get pods"
```

정상이면 `airflow-scheduler-*`, `airflow-webserver-*`, `airflow-triggerer-*`, `airflow-postgresql-*` 가 모두 `Running` 으로 떠요. 첫 기동 직후엔 `airflow-run-airflow-migrations-*` 같은 job pod 가 잠깐 떴다 사라지기도 해요. 정상.


<br>

<br>



## 5. 첫 접속 확인

UI 를 잠깐 띄워서 들어가봐요.

```shell
kubectl -n airflow port-forward svc/airflow-webserver 8080:8080
```

브라우저에서 `http://localhost:8080` 접속. 기본 계정은 `admin / admin` (차트의 `webserver.defaultUser` 가 만들어주는 값). 운영에서는 반드시 `values.yaml` 의 `webserver.defaultUser.password` 를 바꾸거나, LDAP/OIDC 같은 외부 인증으로 교체하세요.

들어가면 예제 DAG 들이 같이 보일 텐데, 차트 디폴트가 `loadExamples: true` 라서 그래요. 깔끔하게 가고 싶으면 `values.yaml` 에 다음 한 줄.

```yaml
config:
  core:
    load_examples: "False"
```

> ✅ 첫 접속 후 체크할 것
> - [x] 좌측 사이드바에 DAG 목록이 뜨는가
> - [x] Admin → Connections / Variables / Pools 메뉴 정상
> - [x] Browse → Triggerer 가 살아있는지
> - [x] Settings → "About" 의 Airflow 버전이 의도한 버전인가


<br>

<br>



## 6. 자주 겪는 함정

설치 자체는 깔끔하지만, 첫 한두 번은 거의 다 한 번씩 겪어요.

| 증상 | 원인 / 처방 |
|---|---|
| `migrations` job 이 계속 실패 | Postgres 가 아직 준비 안 됐거나 비번 불일치. `kubectl -n airflow logs job/airflow-run-airflow-migrations` 로 확인 |
| Pod 가 `Pending` 으로 멈춤 | 노드 리소스 부족 또는 PVC 가 바인딩 안 됨. `kubectl -n airflow describe pod ...` 의 Events 에 다 적혀있음 |
| Pod 가 `ImagePullBackOff` | 사설 레지스트리 이미지인데 `imagePullSecrets` 안 줌. `registry` 섹션 또는 ServiceAccount 에 pull secret 붙이기 |
| 접속 후 모든 DAG 가 빨간색 import error | 워커가 라이브러리를 못 찾는 케이스가 많음. (3/4) 편의 커스텀 이미지로 해결 |
| `Fernet key must be 32 url-safe base64-encoded bytes` | Fernet key 형식 오류. `Fernet.generate_key()` 결과 그대로 써야 함. 마지막 `=` 포함 |

`helm upgrade --install` 은 멱등하니까 values 만 고쳐서 같은 명령을 다시 때리면 돼요. 처음엔 자잘하게 두세 번 돌리게 됩니다.


<br>

<br>



## 7. 정리

여기까지 오면 K8s 위에 Airflow 가 떠 있고, `KubernetesExecutor` 가 켜진 상태예요. 다만 아직 **워커 Pod 가 어떤 이미지로 어떤 스펙으로 뜨는지** 를 우리가 제어한 적은 없어요. 디폴트 이미지에는 우리 DAG 가 필요로 하는 라이브러리(`pandas`, `requests`, 사내 패키지 등) 가 없을 가능성이 큽니다.

그래서 다음 편에서는 **워커 이미지를 직접 굽고, `pod_template_file` 로 워커 Pod 스펙을 우리가 잡는** 작업을 합니다.

일단 오늘은 여기까지.....   
다음 글에서는 워커 이미지를 빌드하고 `pod_template_file` 로 워커 Pod 를 우리 입맛대로 묶어볼게요.

---

**← 이전 글:** [(1/4) Airflow on K8s 시리즈 개요 — Helm 으로 올리고 워커는 컨테이너로 띄운다](/coding/Airflow_K8s_시리즈_개요/) ｜ **다음 글 →:** [(3/4) Airflow 워커 이미지 만들고 pod_template_file 로 묶기](/coding/Airflow_워커_이미지_pod_template/)
