---
layout: single
title:  "(4/4) Airflow on K8s 운영 — git-sync · 로그 영속화 · 모니터링 · 스케일"
date: 2026-06-04 14:00:00 +0900
description: "Airflow on K8s 를 운영 모드로 굴리기. DAG 를 git-sync 로 동기화하고, 워커 Pod 가 사라져도 로그가 남게 하고, 모니터링/스케일/흔한 장애 패턴까지."
categories: coding
tag: [Airflow, Kubernetes, K8s, git-sync, 로그, 모니터링, Prometheus, DevOps, 운영, 트러블슈팅]
author_profile: false
toc: true
---

{% include series-airflow-k8s-helm.html current="4" %}

# Summary

설치도 되고 워커 이미지도 잡았으면 이제 운영 모드. 이 글에서는 DAG 를 어떻게 안전하게 클러스터에 흘려보낼지(`git-sync`), 워커 Pod 가 휘발성이라 사라져도 **로그가 남게** 하는 법, 그리고 모니터링 / 스케일 / 흔한 장애 패턴까지 정리합니다.

> 💡 이 글에서 다루는 것
> - DAG 동기화 — `git-sync` 사이드카 패턴
> - 로그 영속화 — Persistent Volume 방식 vs Remote logging(S3/GCS)
> - 모니터링 — Airflow 내장 메트릭 + Prometheus exporter
> - 스케일 — 스케줄러/워커 리소스 조정과 한계
> - 자주 만나는 장애 패턴과 처방


<br>

<br>



## 1. DAG 동기화 — git-sync 사이드카

DAG 를 어떻게 클러스터에 넣을지 정해야 해요. 크게 세 가지 방법이 있어요.

| 방법 | 장점 | 단점 |
|---|---|---|
| 이미지에 같이 굽기 | 가장 단순, 변경 이력 = 이미지 태그 | DAG 한 줄 고치는데 이미지 재빌드/배포 |
| PV 마운트 (NFS/EFS) | DAG 만 갈아끼우면 됨 | PV 운영 부담, 권한 이슈 |
| `git-sync` 사이드카 | Git push → 자동 반영, 변경 이력 = Git 그대로 | 사설 repo 면 SSH 키 관리 필요 |

운영에서 가장 자주 보이는 게 `git-sync` 예요. **사이드카 컨테이너가 일정 주기로 git pull 해서 워커/스케줄러가 보는 DAG 폴더를 갱신**해요. Helm 차트가 기본 지원합니다.

```yaml
# values.yaml (추가)
dags:
  gitSync:
    enabled: true
    repo: git@github.com:<org>/<airflow-dags-repo>.git
    branch: main
    rev: HEAD
    depth: 1
    wait: 60        # 초 단위. 60초마다 pull
    subPath: "dags" # repo 내 DAG 가 있는 폴더
    sshKeySecret: airflow-git-ssh-key
```

SSH 키는 미리 시크릿으로 만들어둬요.

```shell
kubectl -n airflow create secret generic airflow-git-ssh-key \
  --from-file=gitSshKey=/path/to/id_ed25519
```

> ✅ git-sync 가 켜지면 scheduler / webserver / 워커 Pod 모두 사이드카가 같이 떠서 같은 revision 을 봅니다. 한 곳만 늦게 따라잡혀서 "스케줄러는 새 DAG 인데 워커는 옛 DAG 로 실행" 같은 사고가 안 나요.

> 🚨 사설 repo 의 SSH 키는 위 5번(Helm 설치) 편의 Fernet 키와 같은 급으로 다뤄야 해요. 보관/회전 정책 필요.


<br>

<br>



## 2. 로그 영속화 — Pod 가 사라져도 로그가 남게

`KubernetesExecutor` 의 워커 Pod 는 태스크가 끝나면 사라져요. **Pod 안에 쌓인 로그도 같이 사라진다는 뜻** 이에요. UI 에서 어제 실패한 태스크 로그를 보려는데 "log not found" 가 뜨면 거의 이 문제예요.

해결은 둘 중 하나.

### 2-1. PV 에 로그 폴더 영속화

차트가 깔 때 옵션 한 줄.

```yaml
# values.yaml
logs:
  persistence:
    enabled: true
    size: 50Gi
    storageClassName: standard  # 또는 nfs, gp3 등
```

`ReadWriteMany` 가 되는 스토리지(NFS, EFS, Azure Files, CephFS)여야 해요. **스케줄러/웹서버/워커가 같은 볼륨을 동시에 마운트** 하기 때문이에요. 일반 EBS 같은 `ReadWriteOnce` 는 안 됨.

### 2-2. Remote logging (S3 / GCS / Azure Blob)

운영에서 가장 깔끔한 방식. 태스크가 끝날 때 워커가 객체 스토리지로 로그를 업로드하고, UI 에서 볼 때도 거기서 가져옵니다.

```yaml
# values.yaml
config:
  logging:
    remote_logging: "True"
    remote_base_log_folder: "s3://my-airflow-logs/airflow"
    remote_log_conn_id: "aws_default"
    encrypt_s3_logs: "False"
```

`aws_default` connection 은 Airflow UI 에서 IAM 키로 만들거나, IRSA(EKS) / Workload Identity(GKE) 같은 클러스터 차원의 권한으로 풀 수도 있어요. **운영에선 IAM 키 박는 것보다 클러스터 권한 위임이 안전**합니다.

| 방식 | 추천 상황 |
|---|---|
| PV 영속화 | 온프레미스 K8s, NFS/EFS 이미 운영 중 |
| Remote logging | 클라우드 K8s, 객체 스토리지 + IAM 권한이 갖춰져 있음 |


<br>

<br>



## 3. 모니터링 — 무엇을 보고 있어야 하나

운영하면서 봐야 하는 신호는 크게 세 층이에요.

| 층 | 지표 | 어디서 |
|---|---|---|
| Airflow 잡 단위 | DAG 성공/실패율, 태스크 평균 실행시간, 큐잉 시간 | Airflow UI + statsd/prom exporter |
| 컴포넌트 단위 | scheduler heartbeat, triggerer 활성, webserver 응답 | `/health` 엔드포인트, K8s probe |
| 클러스터 단위 | 노드 CPU/메모리, Pod Pending 개수, OOM 횟수 | Prometheus + node-exporter |

Airflow 메트릭을 Prometheus 로 빼는 가장 간단한 길은 차트의 statsd → Prometheus exporter 켜기.

```yaml
# values.yaml
statsd:
  enabled: true   # 차트 기본 on
  extraMappings:  # statsd → prometheus metric 이름 매핑
    - match: "airflow.dag.*.*.duration"
      name: "airflow_dag_task_duration"
      labels:
        dag_id: "$1"
        task_id: "$2"
```

이걸 Prometheus 가 ServiceMonitor 로 긁어가면 Grafana 대시보드에서 다음 같은 걸 볼 수 있어요.

- 시간대별 태스크 실패율
- 스케줄러 heartbeat 지연
- 큐에 들어가서 워커 Pod 가 뜨기까지 걸린 시간 (= queue lag)
- DAG 별 평균 실행 시간 추이

> 💡 가장 먼저 패널로 만들 것 두 개: **scheduler heartbeat 지연** 과 **queue lag**. 이 둘이 늘어나기 시작하면 곧 SLA 깨져요.


<br>

<br>



## 4. 스케일 — 어디부터 키울까

Airflow on K8s 의 스케일은 **컴포넌트별로 병목이 다르다** 는 걸 알고 가야 해요.

### 4-1. Scheduler

스케줄러는 단순히 "더 늘리면 빨라진다" 가 아니에요. Airflow 2.x 부터 멀티 스케줄러를 지원하긴 하는데, **DB 락 경합** 이 늘면 오히려 느려져요. 보통은 1 ~ 3개 사이로 굴리고, 그 이상은 DB 튜닝(Postgres connection pool, max_connections, `parsing_processes`) 먼저 봅니다.

### 4-2. Worker

`KubernetesExecutor` 의 워커는 자동으로 늘었다 줄어요(태스크 하나당 Pod 하나). 우리가 조절하는 건 두 가지.

- 동시에 띄울 수 있는 **최대 Pod 수** — `config.core.parallelism`, `config.core.max_active_tasks_per_dag`
- 워커 Pod 한 개의 **리소스** — `workers.resources` (3편 참고)

여기서 자주 빠지는 함정: K8s 노드 풀이 부족하면 Pod 가 `Pending` 으로 쌓여요. **노드 오토스케일러**(Cluster Autoscaler, Karpenter) 가 같이 켜져있어야 워커 Pod 가 진짜로 늘어요.

### 4-3. Metadata DB

운영에서 가장 자주 병목이 잡히는 곳. 태스크가 늘어나면 DB I/O 가 비례해서 늘어요. 외부 RDS(또는 CloudSQL) 로 빼고, connection pool(`pgbouncer`) 을 같이 두는 게 표준.

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

운영하다 보면 거의 다 한 번씩 만나는 것들.

| 증상 | 원인 후보 | 처방 |
|---|---|---|
| 워커 Pod 가 `Pending` 으로 쌓임 | 노드 리소스 부족, 노드 셀렉터/toleration 불일치 | `kubectl describe pod` 의 Events 확인, 오토스케일러 / 노드풀 점검 |
| 태스크 끝나면 로그 사라짐 | remote logging 미설정 + Pod 휘발 | §2 의 PV 또는 remote logging 적용 |
| `ImagePullBackOff` | 레지스트리 인증 / 태그 오타 | pull secret, 태그 재확인 |
| DAG 가 UI 에 안 뜸 | git-sync 가 실패했거나 권한 X | scheduler pod 의 `git-sync` 사이드카 로그 확인 |
| 새 코드 배포 후 일부 워커는 옛 코드로 동작 | DAG 동기화 시점 차이 (이미지 베이스 + PV 혼용 등) | 동기화 방식을 한 가지로 통일 |
| `OOMKilled` 가 빈번 | 워커 메모리 limit 작음 | `workers.resources.limits.memory` 상향 또는 무거운 태스크만 `pod_override` |
| 스케줄러 heartbeat 지연 | DAG 파싱 시간 초과, DB 락 경합 | `dag_dir_list_interval` 늘리기, DAG 파일 분할, pgbouncer 도입 |
| 시작 시 `airflow-run-airflow-migrations` 가 실패 | DB 비번 불일치, 외부 DB 권한 부족 | `kubectl logs job/...` 로 정확한 에러 확인 |


<br>

<br>



## 6. 운영 체크리스트

마지막으로 운영 모드에서 한 번씩 다 짚어두면 좋은 것들.

- [x] `defaultAirflowTag` 가 불변 태그(`:latest` 금지)
- [x] Fernet / Webserver secret 이 외부 백업되어 있음
- [x] 메타데이터 DB 가 외부 관리형(RDS/CloudSQL) + 백업 정책
- [x] 로그가 PV 또는 객체 스토리지로 영속화
- [x] DAG 는 git-sync 또는 이미지 베이크 **둘 중 하나로 통일**
- [x] Prometheus 로 scheduler heartbeat / queue lag 패널 존재
- [x] 노드 오토스케일러(Cluster Autoscaler / Karpenter) 가 워커 Pod 와 같은 노드풀에 붙어있음
- [x] 사설 레지스트리 풀 시크릿이 ServiceAccount 에 잘 붙어있음
- [x] Webserver 의 `defaultUser` 비번을 교체했거나 SSO 로 대체

여기까지 들어맞으면 Airflow on K8s 운영 1차 셋업은 완료.

일단 오늘은 여기까지.....   
다음 글에서는 이번 시리즈에서 못 다룬 외부 메타데이터 DB 분리(RDS Postgres) 와 IRSA 기반 AWS 권한 위임 패턴을 정리해볼게요.

---

**← 이전 글:** [(3/4) Airflow 워커 이미지 만들고 pod_template_file 로 묶기](/coding/Airflow_워커_이미지_pod_template/)
