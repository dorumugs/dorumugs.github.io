---
layout: single
title:  "(3/5) Airflow 워커 이미지 만들고 pod_template_file 로 묶기"
date: 2026-06-04 21:15:00 +0900
description: "워커 Pod 는 K8s 의 평범한 Pod 그대로예요. 우리가 결정할 것은 두 가지 — 이미지(Dockerfile) + 스펙(pod_template_file). 빌드/푸시부터 태스크 단위 pod_override 까지."
categories: coding
tag: [Airflow, Kubernetes, K8s, KubernetesExecutor, Docker, pod_template, 워커, 이미지, DevOps]
author_profile: false
toc: true
---

{% include series-airflow-k8s-helm.html current="3" %}

# Summary

워커 `Pod` 는 [K8s YAML 입문 글](/coding/K8s_YAML_오브젝트_7가지_입문/) 에서 본 그 `Pod` 와 똑같이 생긴 오브젝트예요. 다만 KubernetesExecutor 환경에선 우리가 **딱 두 가지** 를 정해줘야 해요.

- **(1) 이미지** — `pip install` 까지 다 끝난 우리 워커용 Airflow 이미지
- **(2) 스펙** — 그 이미지가 어떤 리소스/볼륨/시크릿/노드 위에서 뜰지를 적은 `pod_template_file` (= 그냥 Pod 스펙)

> 💡 이 글은 레지스트리에 상관없이 일반적인 빌드/푸시 흐름을 다뤄요. AWS 환경에서 ECR + GitHub Actions 로 자동화하는 패턴은 (5/5) 에서 따로 정리해요.

> 💡 이 글에서 다루는 것
> - 왜 커스텀 워커 이미지가 거의 항상 필요한지
> - Airflow 베이스 이미지 위에 라이브러리 얹는 `Dockerfile`
> - 빌드 → 레지스트리 푸시 → Helm `values.yaml` 에 박기
> - `pod_template_file` 로 워커 Pod 스펙 잡기 (K8s 의 평범한 Pod 모양 그대로)
> - 태스크 단위 `pod_override` — 특정 태스크만 더 큰 메모리/다른 노드


<br>

<br>



## 1. 왜 커스텀 이미지가 필요한가

기본 `apache/airflow:3.x` 이미지엔 정말 기본만 들어있어요. 우리 파이프라인이 보통 필요로 하는 건 한참 더 많죠.

- 일반 패키지 — `pandas`, `numpy`, `pyarrow`, `requests`
- DB 드라이버 — `psycopg2-binary`, `pymssql`, `snowflake-connector-python`
- 클라우드 SDK — `boto3`, `google-cloud-storage`
- 사내 PyPI 의 사내 패키지
- 시스템 바이너리 — `git`, `unixodbc`, `libxml2`

그래서 거의 항상 **커스텀 이미지 한 장** 을 굽게 됩니다. 그리고 이 이미지는 *워커 전용* 이 아니에요. Airflow 3.x 에선 **scheduler / api-server / triggerer / dag-processor / worker 가 모두 같은 이미지** 를 씁니다. DAG 코드를 모든 컴포넌트가 같이 봐야 하기 때문이에요.


<br>

<br>



## 2. Dockerfile

공식 `apache/airflow` 베이스 위에 의존성을 얹어요.

```dockerfile
# Dockerfile
FROM apache/airflow:3.1.8-python3.11

# 1) 시스템 의존성 — root 로 잠깐
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      curl \
      unixodbc \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) 다시 airflow 유저 — pip 는 절대 root 로 깔지 말 것
USER airflow

# 3) 파이썬 의존성
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

`requirements.txt` 예시.

```text
pandas==2.2.2
pyarrow==16.1.0
boto3==1.34.140
psycopg2-binary==2.9.9
apache-airflow-providers-amazon==9.29.0
apache-airflow-providers-postgres==6.7.0
```

> ⚠️ `apache-airflow-providers-*` 는 베이스 Airflow 버전에 민감해요. 베이스보다 너무 새 버전을 박으면 import 단계에서 깨집니다.

> 💡 `USER airflow` 로 다시 돌아오는 것 까먹지 마세요. root 상태로 `pip install` 하면 권한 이슈로 컨테이너 실행 중 추가 설치/캐시 쓰기가 막혀요.


<br>

<br>



## 3. 빌드 / 푸시

레지스트리 주소는 환경에 맞게 바꿔주세요. 예시는 사내 Harbor 를 가정했어요.

```shell
REG=registry.<internal>/data-platform
TAG=3.1.8-py311-1

docker build -t $REG/airflow:$TAG .
docker push $REG/airflow:$TAG
```

운영에서는 `:latest` 같은 태그를 쓰지 말고 **불변 태그** (날짜+빌드넘버, 커밋 SHA) 를 권장합니다. K8s 가 이미지 캐시를 적극적으로 쓰는데 `:latest` 면 노드별 캐시 시점이 어긋나서, 같은 태그인데 다른 이미지가 떠 있는 사고가 나요.

사설 레지스트리면 K8s 가 이미지를 받아올 수 있게 **`Secret`** 으로 자격증명을 박아둬야 해요. 일반 `Secret` 위에 `docker-registry` 타입을 얹은 모양이에요.

```shell
kubectl -n airflow create secret docker-registry harbor-creds \
  --docker-server=registry.<internal> \
  --docker-username=<user> \
  --docker-password=<PASSWORD> \
  --docker-email=<your-email>
```


<br>

<br>



## 4. Helm values 에 이미지 박기

차트가 새 이미지를 쓰도록 `values.yaml` 업데이트.

```yaml
# values.yaml (추가)
defaultAirflowRepository: registry.<internal>/data-platform/airflow
defaultAirflowTag: "3.1.8-py311-1"
airflowVersion: "3.1.8"   # 차트 호환성 체크에 사용

# 사설 레지스트리 풀 시크릿
registry:
  secretName: harbor-creds
```

`defaultAirflowRepository` + `defaultAirflowTag` 한 쌍만 잡아도 scheduler / api-server / triggerer / dag-processor / worker 가 모두 같은 이미지를 받아요. 바로 반영해볼게요.

```shell
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --values values.yaml

kubectl -n airflow rollout status deploy/airflow-scheduler
```


<br>

<br>



## 5. pod_template_file — 워커 Pod 스펙 잡기

여기서부터가 진짜 핵심이에요. **`pod_template_file` 은 모든 워커 Pod 가 따라갈 YAML 템플릿** 이고, K8s 의 평범한 `Pod` 스펙과 **문법이 같습니다**.

```yaml
# 워커 Pod 스펙 예시 — K8s 의 평범한 Pod 와 같은 모양
apiVersion: v1
kind: Pod
metadata:
  name: airflow-worker
spec:
  containers:
    - name: base                # ← 이 이름 고정. Airflow 가 여기에 명령/이미지/env 를 자동 주입
      image: registry.<internal>/data-platform/airflow:3.1.8-py311-1
      resources:
        requests: { cpu: "500m", memory: "1Gi" }
        limits:   { cpu: "2",   memory: "4Gi" }
```

직접 ConfigMap 으로 위 YAML 을 마운트해도 되지만, Helm 차트의 `workers.*` 옵션으로 같은 결과를 만들 수 있어요. 이게 가장 깔끔해요.

```yaml
# values.yaml (추가)
workers:
  resources:
    requests: { cpu: "500m", memory: "1Gi" }
    limits:   { cpu: "2",   memory: "4Gi" }

  # 환경변수 — ConfigMap/Secret 에서 가져오기 (envFrom 표준 패턴)
  extraEnv: |
    - name: TZ
      value: Asia/Seoul
    - name: AIRFLOW__CORE__DEFAULT_TIMEZONE
      value: Asia/Seoul

  extraEnvFrom: |
    - secretRef:
        name: airflow-app-secrets   # DB 비번, API 토큰 등

  # 워커 전용 노드풀이 있을 때
  nodeSelector:
    workload: airflow-worker
  tolerations:
    - key: workload
      operator: Equal
      value: airflow-worker
      effect: NoSchedule
```

차트가 위 값들을 받아서 알아서 `pod_template_file.yaml` 을 만들고 컨테이너에 마운트해주고, Airflow config 에 `core.pod_template_file=...` 경로를 박아줘요. 우리는 그냥 values 만 채우면 끝나요.

> 💡 가장 중요한 한 줄: `containers[0].name` 은 **반드시 `base`** 여야 합니다. Airflow 가 이 이름을 기준으로 메인 컨테이너를 찾아서 명령어/이미지/환경변수를 채워요.


<br>

<br>



## 6. 태스크 단위 — pod_override

기본 스펙은 `pod_template_file` 로 잡고, **특정 태스크만** 더 큰 메모리/다른 이미지/다른 노드가 필요할 수 있어요. 이럴 땐 DAG 코드 안에서 필요한 부분만 덮어쓰면 돼요.

```python
from airflow.decorators import task
from kubernetes.client import models as k8s

@task(
    executor_config={
        "pod_override": k8s.V1Pod(
            spec=k8s.V1PodSpec(
                containers=[
                    k8s.V1Container(
                        name="base",
                        resources=k8s.V1ResourceRequirements(
                            requests={"cpu": "2", "memory": "8Gi"},
                            limits={"cpu": "4", "memory": "16Gi"},
                        ),
                    )
                ],
                node_selector={"workload": "airflow-worker-heavy"},
            )
        )
    }
)
def heavy_aggregation():
    ...
```

여기서도 `containers[0].name` 은 반드시 `"base"` 그대로예요. Airflow 가 워커 Pod 의 메인 컨테이너를 그 이름으로 잡고 override 를 머지합니다.

> ✅ 운영 팁: 기본 스펙은 보수적으로 작게, 진짜 무거운 태스크만 `pod_override` 로 키우는 패턴이 비용/안정성 둘 다 좋아요. 모든 태스크에 큰 리소스를 깔면 K8s 가 스케줄을 못 잡고 `Pending` 으로 쌓여요.


<br>

<br>



## 7. 확인 — 워커 Pod 가 떴을 때 잡아보기

가벼운 DAG 한 번 돌려놓고 워커 Pod 가 떴다 사라지는 순간을 잡아보면 적용 여부가 즉시 보여요.

```shell
# 상시 컴포넌트 (Airflow 3.x) 제외하고 일시적 워커 Pod 만 보기
kubectl -n airflow get pods -w | grep -v -E "scheduler|api-server|triggerer|dag-processor|postgresql"
```

태스크가 돌면 `<dag_id>-<task_id>-<runid>-<suffix>` 패턴 Pod 가 잠깐 뜨고 사라져요. 떠 있는 동안 한 번 `describe` 찍어봐요.

```shell
kubectl -n airflow describe pod <pod-name>
```

확인 포인트.

- [x] `Image:` 가 우리가 푸시한 레지스트리 주소 + 태그
- [x] `Requests:` / `Limits:` 가 values 의 값과 같음
- [x] `Node-Selectors:` 가 의도한 노드풀
- [x] `Volumes:` / `Mounts:` 에 Secret/ConfigMap 이 잘 붙음

여기까지 맞으면 워커 이미지 + 템플릿 셋업은 끝나요.

일단 오늘은 여기까지.....   
다음 글에서는 DAG 동기화(`git-sync`), 로그 영속화, 모니터링/스케일 같은 **운영** 쪽을 정리할게요.

---

**← 이전 글:** [(2/5) Helm 으로 Airflow 를 K8s 에 설치하기 — KubernetesExecutor 셋업](/coding/Airflow_Helm_K8s_설치_셋업/) ｜ **다음 글 →:** [(4/5) Airflow on K8s 운영 — git-sync · 로그 영속화 · 모니터링 · 스케일](/coding/Airflow_K8s_운영_DAG_동기화_모니터링/)
