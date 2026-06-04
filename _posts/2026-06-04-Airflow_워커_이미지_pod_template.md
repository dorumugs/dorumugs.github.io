---
layout: single
title:  "(3/4) Airflow 워커 이미지 만들고 pod_template_file 로 묶기"
date: 2026-06-04 13:00:00 +0900
description: "KubernetesExecutor 워커 Pod 가 어떤 리눅스 이미지로, 어떤 리소스/볼륨/시크릿으로 뜨는지를 결정하는 방법. 커스텀 Airflow 이미지 빌드부터 pod_template_file, pod_override 까지."
categories: coding
tag: [Airflow, Kubernetes, K8s, KubernetesExecutor, Docker, pod_template, 워커, 이미지, DevOps]
author_profile: false
toc: true
---

{% include series-airflow-k8s-helm.html current="3" %}

# Summary

`KubernetesExecutor` 의 핵심은 **"매 태스크마다 새 Pod 가 뜬다"** 였죠. 그럼 그 Pod 가 **어떤 이미지로, 어떤 리소스/볼륨/시크릿/노드 위에서** 뜨는지는 누가 정할까요? 그게 이번 글의 주제예요. 우리가 직접 Airflow 이미지를 빌드하고, `pod_template_file` 로 워커 Pod 의 기본 스펙을 못박은 다음, 필요하면 태스크 단위로 `pod_override` 까지.

> 💡 이 글에서 다루는 것
> - 왜 커스텀 워커 이미지가 거의 항상 필요한지
> - Airflow 베이스 이미지 위에 라이브러리 얹는 Dockerfile
> - 빌드 → 레지스트리 푸시 → Helm values 에 박기
> - `pod_template_file` 로 워커 Pod 스펙 잡기
> - 태스크 단위 `pod_override` (특정 태스크만 더 큰 메모리)


<br>

<br>



## 1. "워커 이미지" 가 뭔지 다시 정리

`KubernetesExecutor` 에서 워커 Pod 가 뜰 때 사용하는 이미지는, 사실상 **Airflow 가 깔린 리눅스 컨테이너** 예요. 그 안에 우리 DAG 코드와, 그 DAG 가 의존하는 모든 것(파이썬 패키지, 시스템 바이너리, 사내 라이브러리)이 들어있어야 해요.

차트 디폴트로 깔리는 `apache/airflow:2.x` 이미지에는 정말 기본만 들어있어요. 보통 우리 파이프라인이 필요로 하는 건 이런 것들이에요.

- `pandas`, `numpy`, `pyarrow`, `requests` 같은 일반 패키지
- DB 드라이버 — `psycopg2-binary`, `pymssql`, `cx_Oracle`, `snowflake-connector-python`
- 클라우드 SDK — `boto3`, `google-cloud-storage`, `azure-storage-blob`
- 사내 PyPI 의 사내 패키지
- 시스템 바이너리 — `git`, `unixodbc`, `curl`, `libxml2` 등

그래서 거의 항상 **커스텀 워커 이미지를 한 장 굽게 돼요**. 그리고 이 이미지는 *워커 전용*이 아니에요. Airflow 차트에서는 보통 **scheduler / webserver / triggerer / worker 가 모두 같은 이미지** 를 씁니다. DAG 코드 파싱은 스케줄러도 같이 해야 하거든요.


<br>

<br>



## 2. Dockerfile 작성

베이스 이미지로 공식 `apache/airflow` 의 특정 태그를 잡고, 그 위에 우리 의존성을 얹어요.

```dockerfile
# Dockerfile
FROM apache/airflow:2.9.2-python3.11

# 1) 시스템 의존성 — root 로 잠깐 들어감
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      curl \
      unixodbc \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) 다시 airflow 유저로 — pip 는 절대 root 로 깔지 말 것
USER airflow

# 3) 파이썬 의존성
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
```

`requirements.txt` 는 별도 파일로 빼두면 캐시 효율이 좋아져요. 예시 한 토막.

```text
pandas==2.2.2
pyarrow==16.1.0
boto3==1.34.140
psycopg2-binary==2.9.9
apache-airflow-providers-amazon==8.24.0
apache-airflow-providers-postgres==5.11.1
```

> ⚠️ **베이스 이미지의 Airflow 버전과 provider 패키지 호환성** 을 꼭 확인. `apache-airflow-providers-*` 는 Airflow 버전에 민감해서, 너무 신 버전을 박으면 import 단계에서 죽어요.

> 💡 `USER airflow` 로 다시 돌아오는 것 까먹지 마세요. root 상태로 `pip install` 하면 권한 이슈로 컨테이너 실행 중에 추가 설치/캐시 쓰기가 막혀요.


<br>

<br>



## 3. 빌드와 레지스트리 푸시

레지스트리 주소를 환경에 맞게 잡아주세요. 예시는 사내 Harbor 가정.

```shell
REG=registry.<internal>/data-platform
TAG=2.9.2-py311-1

docker build -t $REG/airflow:$TAG .
docker push $REG/airflow:$TAG
```

운영에서는 `:latest` 같은 태그 쓰지 말고 **불변 태그**(예: 날짜+빌드넘버, 커밋 SHA)를 박는 걸 강력 추천. K8s 가 이미지 캐시를 적극적으로 쓰는데 `:latest` 면 노드별로 시점이 어긋나서 같은 태그인데 다른 이미지가 떠 있는 사고가 납니다.

> 🚨 사설 레지스트리면 K8s 가 풀할 수 있게 `imagePullSecrets` 가 필요해요. namespace 에 docker-registry 시크릿을 만들고 Helm values 에 연결.

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

이제 차트가 이 이미지를 쓰도록 `values.yaml` 을 업데이트.

```yaml
# values.yaml (추가)
defaultAirflowRepository: registry.<internal>/data-platform/airflow
defaultAirflowTag: "2.9.2-py311-1"
airflowVersion: "2.9.2"   # 차트가 호환성 체크에 사용

# 사설 레지스트리 풀 시크릿
registry:
  secretName: harbor-creds

# (선택) 컴포넌트별 이미지를 따로 잡고 싶을 때
images:
  airflow:
    repository: registry.<internal>/data-platform/airflow
    tag: "2.9.2-py311-1"
    pullPolicy: IfNotPresent
```

`defaultAirflowRepository` + `defaultAirflowTag` 만 잡아도 scheduler/webserver/triggerer/worker 모두 같은 이미지를 받아요. 컴포넌트별로 다른 이미지를 굳이 쓰고 싶을 때만 `images.*` 를 따로 잡으면 돼요.

업데이트 반영.

```shell
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --values values.yaml
```

`kubectl -n airflow rollout status deploy/airflow-scheduler` 로 롤아웃 확인하고, 한 번 새로 들어가서 DAG import error 가 사라졌는지 보면 1차 완료.


<br>

<br>



## 5. pod_template_file 로 워커 Pod 스펙 잡기

여기서부터가 진짜 핵심이에요. `pod_template_file` 은 **모든 워커 Pod 가 기본적으로 따라갈 YAML 템플릿** 이에요. 이미지/리소스/볼륨/노드셀렉터/환경변수/시크릿 같은 걸 한 곳에 정의해두면, 스케줄러가 워커 Pod 를 띄울 때 이걸 베이스로 써요.

Helm 차트의 `workers.podTemplate` 또는 `podTemplate` 옵션으로 YAML 본문을 넘기는 게 가장 깔끔합니다. values.yaml 에 그대로 박을 수 있어요.

```yaml
# values.yaml (추가)
workers:
  # 워커 Pod 의 베이스 리소스/환경. pod_template_file 로 차트가 넘김
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "2"
      memory: "4Gi"

  # 추가 환경변수 / 시크릿 마운트
  extraEnv: |
    - name: TZ
      value: Asia/Seoul
    - name: AIRFLOW__CORE__DEFAULT_TIMEZONE
      value: Asia/Seoul

  extraEnvFrom: |
    - secretRef:
        name: airflow-app-secrets   # DB 비번, API 토큰 등

  # 노드 셀렉터 / tolerations — 워커 전용 노드풀에 몰고 싶을 때
  nodeSelector:
    workload: airflow-worker
  tolerations:
    - key: workload
      operator: Equal
      value: airflow-worker
      effect: NoSchedule
```

이 설정이 들어가면 차트가 다음 시점에 알아서 `pod_template_file.yaml` 을 만들어서 컨테이너 안에 마운트하고, Airflow config 에 `core.pod_template_file=/opt/airflow/pod_templates/pod_template_file.yaml` 을 박아요. 스케줄러가 새 워커 Pod 를 만들 때 이 템플릿을 베이스로 가져갑니다.

> 💡 직접 `pod_template_file.yaml` 을 한 줄 한 줄 손으로 쓸 수도 있어요. ConfigMap 으로 만들어 마운트하고 `AIRFLOW__CORE__POD_TEMPLATE_FILE` 환경변수로 경로를 잡아주는 패턴. 차트 기능을 안 쓰고 풀 커스텀하고 싶을 때 유용.


<br>

<br>



## 6. 태스크 단위로 다르게 띄우기 — pod_override

기본 스펙은 `pod_template_file` 로 잡고, **특정 태스크만 더 큰 메모리/다른 이미지/다른 노드** 가 필요할 수 있어요. 이럴 때는 DAG 코드 안에서 `executor_config` 로 부분 override.

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
    # 무거운 잡
    ...
```

여기서 `containers[0].name` 은 반드시 `"base"` 여야 해요. Airflow 가 워커 Pod 안에서 메인 컨테이너를 그 이름으로 잡고 override 를 머지합니다.

> ✅ 운영 팁: 기본 스펙은 보수적으로 작게, 진짜 무거운 태스크만 `pod_override` 로 키우는 패턴이 비용/안정성 둘 다 좋아요. 모든 태스크에 큰 리소스를 깔면 K8s 가 스케줄을 못 잡고 Pending 으로 쌓여요.


<br>

<br>



## 7. 확인하기

새 이미지 + 워커 스펙이 진짜 적용됐는지 확인하는 가장 빠른 방법은, 가벼운 DAG 한 번 돌려놓고 워커 Pod 가 뜨는 순간을 잡아보는 거예요.

```shell
# 워커 Pod 가 뜨는 걸 실시간으로 보기 (KubernetesExecutor 는 이름이 airflow-worker-* 가 아니라 태스크 기반)
kubectl -n airflow get pods -w | grep -v -E "scheduler|webserver|triggerer|postgresql"
```

태스크가 돌면 `<dag_id>-<task_id>-<runid>-<suffix>` 패턴 Pod 가 잠깐 뜨고 사라져요. 떠 있는 동안 describe 한 번.

```shell
kubectl -n airflow describe pod <pod-name>
```

확인 포인트:

- [x] `Image:` 가 우리가 푸시한 레지스트리 주소 + 태그인가
- [x] `Requests:` / `Limits:` 가 values 에 박은 값과 같은가
- [x] `Node-Selectors:` 가 의도한 노드풀인가
- [x] `Volumes:` / `Mounts:` 에 시크릿/ConfigMap 이 잘 붙었는가

여기까지 들어맞으면 워커 이미지 + 템플릿 셋업은 끝.

일단 오늘은 여기까지.....   
다음 글에서는 DAG 동기화(`git-sync`), 로그 영속화, 모니터링/스케일 같은 **운영** 쪽 이야기를 정리할게요.

---

**← 이전 글:** [(2/4) Helm 으로 Airflow 를 K8s 에 설치하기 — KubernetesExecutor 셋업](/coding/Airflow_Helm_K8s_설치_셋업/) ｜ **다음 글 →:** [(4/4) Airflow on K8s 운영 — git-sync · 로그 영속화 · 모니터링 · 스케일](/coding/Airflow_K8s_운영_DAG_동기화_모니터링/)
