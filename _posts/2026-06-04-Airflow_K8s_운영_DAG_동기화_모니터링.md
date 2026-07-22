---
layout: single
title:  "(4/5) Airflow on K8s 운영 — git-sync · 로그 영속화 · 모니터링 · 스케일"
date: 2026-06-04 21:20:00 +0900
description: "Airflow on K8s 를 운영 모드로 굴리기. git-sync 사이드카, PVC 로그 영속화, Prometheus 메트릭, 스케일과 장애 패턴까지."
categories: coding
tag: [Airflow, Kubernetes, K8s, git-sync, 로그, 모니터링, Prometheus, DevOps, 운영, 트러블슈팅]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-04-airflow-k8s-operations/header.svg
  teaser: /assets/images/2026-06-04-airflow-k8s-operations/header.svg
series: airflow-k8s-helm
series_order: 4
---

{% include series-airflow-k8s-helm.html current="4" %}

# Summary

설치와 워커 이미지까지 끝났으면 이제 운영 모드로 들어갈 차례예요. 이 글에서는 **DAG 동기화(`git-sync` 사이드카) · 로그 영속화(`PVC` 또는 객체 스토리지) · 모니터링 · 스케일 · 장애 패턴** 을 한 번에 정리합니다.

> 📚 이번 편은 [K8s YAML 입문 글](/coding/K8s_YAML_오브젝트_7가지_입문/) 에서 짚었던 **"다음 단계 친구"** (`PersistentVolumeClaim`) 영역까지 살짝 발을 들여요. 그리고 같은 글의 "Pod 에는 컨테이너가 여러 개 들어갈 수 있다" 는 사실이 `git-sync` 사이드카에서 진짜로 쓰입니다.

> 💡 이 글에서 다루는 것
> - DAG 동기화 — `git-sync` 사이드카 패턴
> - 로그 영속화 — PV 방식 vs Remote logging(S3/GCS)
> - 모니터링 — Airflow 내장 메트릭 + Prometheus exporter
> - 스케일 — scheduler / worker / DB 의 병목 위치
> - 자주 만나는 장애 패턴 + 처방
> - 운영 체크리스트


<br>

<br>



## 1. DAG 동기화 — git-sync 사이드카

DAG 를 클러스터로 어떻게 흘려보낼지 정해야 해요. 크게 세 가지가 있어요.

| 방법 | 장점 | 단점 |
|---|---|---|
| 이미지에 같이 굽기 | 가장 단순, 변경 이력 = 이미지 태그 | DAG 한 줄 고치는데 이미지 재빌드/배포 |
| PV 마운트 (NFS/EFS) | DAG 만 갈아끼우면 됨 | PV 운영 부담, 권한 이슈 |
| `git-sync` 사이드카 | Git push → 자동 반영, 변경 이력 = Git 그대로 | 사설 repo 면 SSH 키 관리 필요 |

운영에서 가장 흔한 게 `git-sync` 예요. **scheduler / api-server / dag-processor / 워커 Pod 안에 사이드카 컨테이너로 같이 떠서, 일정 주기로 `git pull`** 해서 DAG 폴더를 갱신해요.

입문 글에서 잠깐 짚었던 "한 `Pod` 안에 컨테이너가 여러 개 들어갈 수 있다" 가 여기서 진짜로 쓰입니다. 메인 컨테이너(Airflow) 옆에 `git-sync` 컨테이너가 같이 떠 있고, 둘이 같은 빈 디렉토리 볼륨을 공유해요. git-sync 가 그 폴더에 DAG 를 떨어뜨리면 Airflow 가 그걸 읽어요.

Helm 차트가 기본 지원합니다.

```yaml
# values.yaml (추가)
dags:
  gitSync:
    enabled: true
    repo: git@github.com:<org>/<airflow-dags-repo>.git
    branch: main
    rev: HEAD
    depth: 1
    wait: 60        # 60초마다 pull
    subPath: "dags" # repo 내 DAG 폴더
    sshKeySecret: airflow-git-ssh-key
```

SSH 키는 `Secret` 으로 미리 박아둬요. 일반 `Secret` 패턴 그대로입니다.

```shell
kubectl -n airflow create secret generic airflow-git-ssh-key \
  --from-file=gitSshKey=/path/to/id_ed25519
```

> ✅ git-sync 가 켜지면 scheduler / api-server / dag-processor / 워커가 같은 revision 을 봅니다. "스케줄러는 새 DAG 인데 워커는 옛 DAG 로 실행" 같은 사고가 안 나요.


<br>

<br>



## 2. 로그 영속화 — Pod 가 사라져도 로그가 남게

KubernetesExecutor 의 워커 `Pod` 는 태스크가 끝나면 사라져요. **그 안의 로그도 같이 사라진다는 뜻** 이에요. UI 에서 어제 실패한 태스크 로그를 보려는데 "log not found" 가 뜨는 건 거의 이 문제예요.

해결책은 둘 중 하나예요.

### 2-1. PVC 로 로그 폴더 영속화

입문 글의 "다음 단계 친구" 표에서 본 `PersistentVolumeClaim` 이 여기 등장해요. 차트가 옵션 한 줄로 깔아줍니다.

```yaml
# values.yaml
logs:
  persistence:
    enabled: true
    size: 50Gi
    storageClassName: standard   # 또는 nfs, gp3 등
```

> 🚨 `ReadWriteMany` 가 되는 스토리지(NFS, EFS, Azure Files, CephFS)여야 합니다. **scheduler / api-server / 워커가 같은 볼륨을 동시에 마운트** 해야 하니까요. 일반 EBS 같은 `ReadWriteOnce` 는 동작하지 않아요.

### 2-2. Remote logging (S3 / GCS / Azure Blob)

운영에선 가장 깔끔한 방식이에요. 태스크가 끝날 때 워커가 객체 스토리지에 로그를 업로드하고, UI 가 거기서 가져옵니다.

```yaml
# values.yaml
config:
  logging:
    remote_logging: "True"
    remote_base_log_folder: "s3://my-airflow-logs/airflow"
    remote_log_conn_id: "aws_default"
    encrypt_s3_logs: "False"
```

`aws_default` connection 은 Airflow UI 에서 IAM 키로 만들거나, IRSA(EKS) / Workload Identity(GKE) 같이 **클러스터 차원의 권한 위임** 으로 풀 수도 있어요. 운영이면 후자가 안전합니다.

| 방식 | 추천 상황 |
|---|---|
| PVC 영속화 | 온프레미스 K8s, NFS/EFS 가 이미 운영 중 |
| Remote logging | 클라우드 K8s, 객체 스토리지 + IAM 권한 갖춰져 있음 |


<br>

<br>



## 3. 모니터링 — 무엇을 보고 있어야 하나

운영하면서 봐야 하는 신호는 크게 세 층으로 나뉘어요.

| 층 | 지표 | 어디서 |
|---|---|---|
| Airflow 잡 단위 | DAG 성공/실패율,<br>태스크 평균 실행시간,<br>큐잉 시간 | Airflow UI +<br>statsd/prom exporter |
| 컴포넌트 단위 | scheduler heartbeat,<br>triggerer 활성,<br>api-server 응답,<br>dag-processor 파싱 시간 | `/health` 엔드포인트,<br>K8s probe |
| 클러스터 단위 | 노드 CPU/메모리,<br>Pod Pending 개수, OOM | Prometheus +<br>node-exporter |

Airflow 메트릭을 Prometheus 로 빼는 가장 간단한 길은 차트의 statsd → Prometheus exporter 를 켜는 거예요.

```yaml
# values.yaml
statsd:
  enabled: true
  extraMappings:
    - match: "airflow.dag.*.*.duration"
      name: "airflow_dag_task_duration"
      labels:
        dag_id: "$1"
        task_id: "$2"
```

Prometheus 가 ServiceMonitor 로 긁어가게 해두면 Grafana 에서 다음 같은 패널을 만들 수 있어요.

- 시간대별 태스크 실패율
- 스케줄러 heartbeat 지연
- 큐에 들어가서 워커 Pod 가 뜨기까지 걸린 시간 (= queue lag)
- DAG 별 평균 실행 시간 추이

> 💡 가장 먼저 만들 패널 두 개: **scheduler heartbeat 지연** + **queue lag**. 이 둘이 늘기 시작하면 곧 SLA 깨져요.


<br>

<br>



## 4. 스케일 — 어디가 병목인가

컴포넌트별로 병목이 다르다는 걸 기억해야 해요.

### 4-1. Scheduler

"늘리면 빨라진다" 가 아니에요. Airflow 가 멀티 스케줄러를 지원하긴 하지만, **DB 락 경합** 이 늘면 오히려 느려져요. 보통 1~3 개가 적정이고, 그 이상은 DB 튜닝(connection pool, `parsing_processes`) 부터 봐야 해요.

### 4-2. Worker

KubernetesExecutor 의 워커는 태스크당 Pod 라 자동으로 늘었다 줄어요. 우리가 조절할 건 두 가지예요.

- 동시에 띄울 수 있는 **최대 Pod 수** — `config.core.parallelism`, `config.core.max_active_tasks_per_dag`
- 워커 Pod 한 개의 **리소스** — `workers.resources` ((3/5) 참고)

> ⚠️ 노드 풀이 부족하면 워커 Pod 가 `Pending` 으로 쌓여요. **Cluster Autoscaler / Karpenter** 같은 노드 오토스케일러가 같은 노드풀에 붙어있어야 자동 확장이 진짜로 됩니다.

### 4-3. Metadata DB

운영에서 가장 자주 병목이 잡히는 곳이에요. 외부 RDS / CloudSQL 로 빼고, `pgbouncer` 같은 connection pool 을 같이 두는 게 표준이에요.

```yaml
# values.yaml
pgbouncer:
  enabled: true
  maxClientConn: 200
  poolSize: 50
```


<br>

<br>



## 5. 자주 만나는 장애 패턴

| 증상 | 원인 후보 | 처방 |
|---|---|---|
| 워커 Pod 가 `Pending` 으로<br>쌓임 | 노드 리소스 부족,<br>노드 셀렉터/toleration 불일치 | `kubectl describe pod` Events,<br>오토스케일러 / 노드풀 점검 |
| 태스크 끝나면 로그 사라짐 | remote logging 미설정 +<br>Pod 휘발 | PVC 영속화 또는<br>remote logging 적용 |
| `ImagePullBackOff` | 레지스트리 인증 / 태그 오타 | pull secret, 태그 재확인 |
| DAG 가 UI 에 안 뜸 | git-sync 실패 / 권한 X | scheduler pod 의<br>`git-sync` 사이드카 로그 확인 |
| 새 코드 배포 후<br>일부 워커는 옛 코드 | 동기화 시점 차이<br>(이미지 베이크 + PV 혼용 등) | 동기화 방식을<br>한 가지로 통일 |
| `OOMKilled` 가 빈번 | 워커 메모리 limit 작음 | `workers.resources.limits.memory`<br>상향 또는 `pod_override` |
| 스케줄러 heartbeat 지연 | DAG 파싱 시간 초과,<br>DB 락 경합 | `dag_dir_list_interval` 늘리기,<br>DAG 파일 분할,<br>pgbouncer 도입 |
| `airflow-run-airflow-migrations`<br>가 실패 | DB 비번 불일치,<br>외부 DB 권한 부족 | `kubectl logs job/...` |


<br>

<br>



## 6. 운영 체크리스트

마지막으로 한 번씩 다 짚어두면 좋은 것들이에요.

- [x] `defaultAirflowTag` 가 불변 태그(`:latest` 금지)
- [x] Fernet / API server `Secret` 이 외부에 백업
- [x] 메타데이터 DB 가 외부 관리형(RDS/CloudSQL) + 백업 정책
- [x] 로그가 `PVC` 또는 객체 스토리지로 영속화
- [x] DAG 는 git-sync 또는 이미지 베이크 **둘 중 하나로 통일**
- [x] Prometheus 로 scheduler heartbeat / queue lag 패널 존재
- [x] Cluster Autoscaler / Karpenter 가 워커 노드풀에 붙어있음
- [x] 사설 레지스트리 풀 `Secret` 이 ServiceAccount 에 잘 붙음
- [x] `createUserJob.defaultUser` 비번 교체 또는 SSO 로 대체

여기까지 들어맞으면 Airflow on K8s 운영 1차 셋업은 끝났다고 봐도 돼요.

일단 오늘은 여기까지.....   
다음 글에서는 같은 셋업을 AWS 위에 올려요. EKS + ECR + GitHub Actions 로 워커 이미지 빌드/배포를 자동화하는 흐름을 정리해볼게요.

---

**← 이전 글:** [(3/5) Airflow 워커 이미지 만들고 pod_template_file 로 묶기](/coding/Airflow_워커_이미지_pod_template/) ｜ **다음 글 →:** [(5/5) Airflow on EKS — ECR 로 이미지 굽고 GitHub Actions 로 자동화](/coding/Airflow_EKS_ECR_GitHub_Actions/)
