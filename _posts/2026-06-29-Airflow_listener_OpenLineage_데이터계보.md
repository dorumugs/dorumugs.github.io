---
layout: single
title:  "Airflow listener 와 OpenLineage 로 데이터 계보(lineage) 자동 수집하기"
date: 2026-06-29 14:00:00 +0900
categories: coding
tag: [Airflow, listener, OpenLineage, lineage, 데이터계보, Marquez, hookimpl, data-engineering, observability, plugin]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-29-airflow-lineage/header.svg
  teaser: /assets/images/2026-06-29-airflow-lineage/header.svg
description: "Airflow 의 listener 이벤트 훅과 OpenLineage 프로바이더로 데이터 계보를 자동으로 모으는 구성을 정리합니다. listener 기본 구조, OpenLineage 가 listener 위에서 동작하는 원리, Marquez 수신 백엔드 세팅, 커스텀 오퍼레이터에 계보 붙이기까지 다뤄요."
---


## Summary

[지난 글](/coding/Airflow_플러그인_제대로_활용하기/)에서 플러그인으로 알람·멀티 스케줄을 붙였는데, 그때 표에 슬쩍 끼워뒀던 항목이 하나 있어요. 바로 **`listeners`** 예요. 이번 글의 주인공입니다.

데이터 파이프라인이 늘어나면 꼭 나오는 질문이 있어요. *"이 테이블은 어디서 왔고, 누가 쓰고 있지?"* 이걸 사람이 위키에 손으로 적어 관리하면 금세 현실과 어긋나요. 그래서 **계보(lineage)는 자동으로 수집**하는 게 정답이고, Airflow 에서는 **listener + OpenLineage** 조합이 그 일을 해줍니다.

이 글에서는 listener 가 뭔지부터 시작해서, OpenLineage 가 바로 그 listener 위에서 어떻게 동작하는지, 그리고 수집된 계보를 Marquez 로 받아 그래프로 보는 데까지 정리합니다.

> 💡 **이 글에서 다루는 것**
> - 왜 계보를 **자동으로** 모아야 하나
> - **listener** — DAG 를 안 건드리는 전역 이벤트 훅
> - listener 와 콜백(`on_failure_callback`)의 차이
> - **OpenLineage** — 계보를 위한 공용 언어, 그리고 "프로바이더가 곧 listener"
> - 수신 백엔드 **Marquez** 세우기 (docker)
> - Airflow → OpenLineage 연결 설정
> - 자동으로 잡히는 것 / **커스텀 오퍼레이터에 계보 붙이기**
> - 운영에서 자주 밟는 지뢰

지난 글과 마찬가지로 Airflow 2.7 이상을 기준으로 합니다. (OpenLineage 가 공식 프로바이더로 들어온 버전대예요.)


<br>

<br>



## 1. 왜 "자동" 계보인가

계보(lineage)는 **"이 데이터가 어디서 와서 어디로 가는지"** 의 지도예요. `orders` 테이블이 `raw_events` 에서 변환돼 나왔고, 다시 `daily_sales` 대시보드로 흘러간다 — 이런 연결 관계죠.

문제는 이걸 **손으로 관리하면 반드시 썩는다**는 거예요. DAG 하나 고치고 위키 갱신을 깜빡하는 순간부터 지도와 현실이 달라지기 시작합니다. 그래서 계보는 파이프라인이 **실제로 실행될 때 자동으로 기록**되는 게 유일하게 믿을 수 있는 방식이에요.

Airflow 에서 이걸 가능하게 하는 두 조각이 이렇습니다.

- **listener** — 태스크/DAG 가 상태를 바꿀 때마다 Airflow 가 불러주는 이벤트 훅. "무슨 일이 언제 일어났는지" 를 잡는 자리.
- **OpenLineage** — 그렇게 잡은 사건을 **표준 포맷의 계보 이벤트**로 만들어 외부 저장소로 보내는 규격.

둘을 합치면 *DAG 코드는 그대로 둔 채* 계보가 쌓여요. 하나씩 볼게요.


<br>

<br>



## 2. listener — DAG 를 안 건드리는 이벤트 훅

지난 글의 알람은 `on_failure_callback` 으로 붙였죠. 그건 **DAG(또는 태스크) 단위 설정**이에요. DAG 마다 콜백을 걸어줘야 하고, 안 걸면 안 옵니다.

**listener 는 반대예요.** 한 번 등록하면 **모든 DAG·모든 태스크**의 상태 변화에 대해 Airflow 가 알아서 불러줍니다. DAG 정의를 전혀 건드리지 않아요. 그래서 계보·감사 로그·메트릭처럼 *"전부에 일괄로 걸려야 하는"* 횡단 관심사에 딱 맞아요.

| | `on_failure_callback` (콜백) | listener |
|---|---|---|
| 등록 위치 | DAG/태스크 정의에 인자로 | 플러그인에 한 번 |
| 적용 범위 | 건 DAG 만 | 전체 DAG 일괄 |
| 잡는 이벤트 | 실패/성공/재시도 | 상태 전이 전반 + 컴포넌트 시작/종료 |
| 용도 | 그 DAG 한정 알림 | 계보·감사·메트릭 등 횡단 |

listener 가 받을 수 있는 주요 이벤트는 다음과 같아요.

| 훅 함수 | 언제 불리나 |
|---|---|
| `on_task_instance_running` | 태스크가 실행에 들어갈 때 |
| `on_task_instance_success` | 태스크 성공 |
| `on_task_instance_failed` | 태스크 실패 |
| `on_dag_run_running` / `_success` / `_failed` | DAG 런 상태 전이 |
| `on_starting` / `before_stopping` | 스케줄러·워커 컴포넌트 시작/종료 |

### 가장 작은 listener

listener 는 **`@hookimpl` 로 데코레이트한 함수들을 담은 모듈**이에요. 감사 로그를 남기는 예시를 만들어볼게요.

```python
# plugins/listeners/audit_listener.py
from airflow.listeners import hookimpl


@hookimpl
def on_task_instance_running(previous_state, task_instance, session=None):
    ti = task_instance
    print(f"[AUDIT] RUNNING {ti.dag_id}.{ti.task_id} (try {ti.try_number})")


@hookimpl
def on_task_instance_success(previous_state, task_instance, session=None):
    ti = task_instance
    print(f"[AUDIT] SUCCESS {ti.dag_id}.{ti.task_id}")


@hookimpl
def on_task_instance_failed(previous_state, task_instance, error=None, session=None):
    ti = task_instance
    print(f"[AUDIT] FAILED  {ti.dag_id}.{ti.task_id} :: {error}")
```

이 listener 가 붙은 상태에서 DAG 가 한 번 돌면, 워커/스케줄러 로그에 이렇게 찍혀요.

```text
[AUDIT] RUNNING etl_daily_sales.extract (try 1)
[AUDIT] SUCCESS etl_daily_sales.extract
[AUDIT] RUNNING etl_daily_sales.transform (try 1)
[AUDIT] FAILED  etl_daily_sales.transform :: ValueError('컬럼 amount 가 비어 있습니다')
```

`task_instance` 하나만 받아도 `dag_id` · `task_id` · `try_number` · `state` 가 다 들어 있어서, 여기에 외부 전송만 붙이면 그대로 감사/메트릭 파이프라인이 됩니다.

> ⚠️ **훅 시그니처는 버전마다 조금씩 달라요.** 예를 들어 `on_task_instance_failed` 의 `error` 인자는 비교적 최근에 추가됐어요. 시그니처가 안 맞으면 에러가 나는 게 아니라 **조용히 호출이 안 되는** 경우가 있어서, 설치된 Airflow 버전의 `airflow.listeners.spec` 문서를 한 번 맞춰보는 걸 권장드립니다.

### 플러그인으로 등록

지난 글의 플러그인 구조 그대로예요. listener 는 **모듈을 통째로** `listeners` 에 넣습니다.

```python
# plugins/my_plugin.py
from airflow.plugins_manager import AirflowPlugin
from listeners import audit_listener


class MyCompanyPlugin(AirflowPlugin):
    name = "my_company_plugin"
    listeners = [audit_listener]   # @hookimpl 함수가 든 모듈을 등록
```

> 🚨 listener 도 플러그인이라 **변경하면 프로세스 재기동이 필요**해요. 고쳤는데 안 먹으면 워커/스케줄러 재시작부터 의심하세요. (지난 글의 타임테이블과 같은 함정이에요.)


<br>

<br>



## 3. OpenLineage — 계보를 위한 공용 언어

이제 핵심. 계보 이벤트를 직접 listener 로 한 땀 한 땀 만들 수도 있지만, 그건 바퀴를 다시 발명하는 일이에요. **OpenLineage** 라는 오픈 표준이 이미 있고, Airflow 에는 이걸 구현한 공식 프로바이더가 있습니다.

여기서 중요한 한 줄.

> ✅ **OpenLineage 프로바이더는 그 자체가 listener 예요.** 패키지를 설치하면, 방금 2장에서 본 listener 메커니즘 위에서 동작하는 계보 수집기가 **자동으로 등록**됩니다. 우리가 listener 코드를 직접 쓸 필요가 없어요.

즉 2장은 "원리" 였고, 실무에서 계보가 목적이라면 listener 를 손으로 짜기보다 OpenLineage 프로바이더를 까는 게 정답입니다. 설치는 평범해요.

```shell
pip install apache-airflow-providers-openlineage
```

설치만 해두면 listener 가 등록되지만, **어디로 이벤트를 보낼지** 를 알려주기 전까지는 아무 데도 안 갑니다. 받을 곳을 먼저 세울게요.


<br>

<br>



## 4. 수신 백엔드 세우기 — Marquez

OpenLineage 이벤트를 받아서 계보 그래프로 보여주는 대표 오픈소스가 **Marquez** 예요. (OpenLineage 의 레퍼런스 구현이기도 해요.) 도커로 한 방에 띄울 수 있습니다.

```shell
git clone https://github.com/MarquezProject/marquez && cd marquez
./docker/up.sh --api-port 5000
```

뜨고 나면 포트는 이렇게 잡혀요.

| 구성요소 | 포트 | 용도 |
|---|---|---|
| Marquez API | `5000` | OpenLineage 이벤트 수신 (`POST /api/v1/lineage`) |
| Marquez Web | `3000` | 계보 그래프를 보는 웹 UI |

`http://localhost:3000` 에 들어가면 지금은 텅 비어 있을 거예요. 여기에 Airflow 를 연결하면 DAG 가 돌 때마다 노드가 하나씩 채워집니다.

> 💡 Marquez 는 계보 백엔드의 한 예일 뿐이에요. OpenLineage 는 표준이라 DataHub, Microsoft Purview 같은 다른 카탈로그로도 같은 이벤트를 보낼 수 있어요. 일단 손에 잡히는 그림을 보기엔 Marquez 가 제일 빨라서 이걸로 시작하는 걸 추천드립니다.


<br>

<br>



## 5. Airflow → OpenLineage 연결

이제 프로바이더에게 "Marquez 로 보내라" 고 알려줄 차례예요. 두 가지를 설정합니다 — **transport**(어디로 보낼지)와 **namespace**(어느 묶음으로 기록할지).

`airflow.cfg` 에 박는 방법.

```ini
[openlineage]
transport = {"type": "http", "url": "http://localhost:5000", "endpoint": "api/v1/lineage"}
namespace = airflow-prod
```

환경변수로 주는 방법(도커/쿠버네티스에서 편해요).

```shell
export AIRFLOW__OPENLINEAGE__TRANSPORT='{"type": "http", "url": "http://localhost:5000"}'
export AIRFLOW__OPENLINEAGE__NAMESPACE='airflow-prod'
```

> 🚨 `namespace` 는 **계보 그래프를 묶는 키**라서 일관성이 중요해요. 같은 환경의 잡들은 같은 namespace 로 묶어야 그래프가 한 덩어리로 이어집니다. 운영/스테이징을 섞어 같은 namespace 에 쏘면 그래프가 엉켜요.

설정을 반영하려면(역시) **재기동** 후, DAG 를 하나 돌려보세요. Marquez 웹 UI 에 잡과 데이터셋 노드가 뜨기 시작하면 연결 성공입니다.


<br>

<br>



## 6. 자동으로 잡히는 것, 그리고 커스텀 오퍼레이터

연결만 하면 계보가 다 잡힐 것 같지만, 한 가지 알아둘 게 있어요. OpenLineage 프로바이더는 **각 오퍼레이터가 "나는 이런 입력을 읽어서 이런 출력을 쓴다" 고 알려줄 때만** 정확한 계보를 그립니다.

- **자동으로 잘 잡히는 것** — SQL 계열 오퍼레이터(Postgres·Snowflake·BigQuery·Redshift 등), 주요 transfer 오퍼레이터처럼 OpenLineage 지원이 내장된 것들. 이들은 실행한 SQL 을 파싱해 입력/출력 테이블을 알아서 추출해요.
- **그냥은 안 잡히는 것** — `PythonOperator` 안의 임의 코드, 자체 제작 오퍼레이터. Airflow 입장에선 그 안에서 무슨 데이터를 만지는지 알 길이 없어요.

후자의 경우, 오퍼레이터에 **계보 힌트 메서드**를 달아주면 됩니다. 오퍼레이터가 `get_openlineage_facets_on_start` (또는 `_on_complete`) 를 구현하면, 프로바이더가 실행 시점에 그걸 불러서 입력/출력 데이터셋을 받아가요.

```python
# plugins/operators/copy_orders.py
from airflow.models import BaseOperator
from airflow.providers.openlineage.extractors import OperatorLineage


class CopyOrdersOperator(BaseOperator):
    def execute(self, context):
        # 실제 복사 로직 (DB → S3 등)
        ...

    # OpenLineage 프로바이더가 실행 시점에 이걸 호출해 계보를 가져감
    def get_openlineage_facets_on_start(self):
        from openlineage.client.run import Dataset
        return OperatorLineage(
            inputs=[Dataset(namespace="postgres://db:5432", name="public.orders")],
            outputs=[Dataset(namespace="s3://warehouse", name="orders/dt=2026-06-29")],
        )
```

여기서 `Dataset` 의 모양을 잠깐 짚으면, 두 인자로 데이터셋 하나를 가리켜요.

```python
from openlineage.client.run import Dataset

ds = Dataset(namespace="postgres://db:5432", name="public.orders")
print(ds.namespace)
print(ds.name)
```

```text
postgres://db:5432
public.orders
```

- `namespace` — 데이터가 사는 **시스템**(DB 호스트, S3 버킷 등). 5장의 잡 namespace 와는 별개로, 데이터셋 쪽 식별자예요.
- `name` — 그 시스템 안에서의 **객체 이름**(스키마.테이블, 경로 등).

이렇게 입력/출력만 알려주면, 여러 DAG·여러 잡을 거쳐도 **같은 `namespace`+`name` 을 가진 데이터셋이 자동으로 한 노드로 이어져서** 계보 그래프가 완성돼요. A 잡의 출력이 B 잡의 입력과 같은 데이터셋이면, Marquez 가 둘을 알아서 연결합니다.


<br>

<br>



## 7. 운영에서 자주 밟는 지뢰

마지막으로 제가 직접 부딪힌 것들 위주로 정리할게요.

- **listener 안에서 무거운 일을 하지 마세요.** 훅은 보통 동기로 불려서, 여기서 외부 호출을 느리게 하면 **스케줄러·워커 전체가 느려져요.** 외부 전송은 타임아웃을 짧게 주고, 가능하면 비동기로 빼는 게 안전합니다.
- **훅 시그니처 불일치 = 조용한 실패.** 2장의 ⚠️ 그대로, 인자가 안 맞으면 에러도 없이 호출이 안 돼요. 버전 문서로 맞추세요.
- **OpenLineage 전송 실패는 태스크를 죽이지 않아요.** 이건 장점이자 함정이에요. Marquez URL 이 틀려도 잡은 멀쩡히 성공하고 계보 이벤트만 조용히 유실돼요. 연결이 의심되면 OpenLineage 관련 로그를 직접 확인해야 합니다.
- **자동 수집은 "지원 오퍼레이터" 한정.** `PythonOperator` 로 다 처리하는 파이프라인은 계보가 거의 안 잡혀요. 6장처럼 힌트를 달거나, 핵심 변환을 SQL 오퍼레이터로 옮기는 걸 고려하세요.
- **namespace 일관성.** 5장 잡 namespace, 6장 데이터셋 namespace 둘 다 — 표기가 흔들리면 같은 데이터가 다른 노드로 쪼개져 그래프가 안 이어집니다. 팀 차원의 명명 규칙을 먼저 정하고 시작하세요.

계보는 처음 세팅할 때만 손이 좀 가지, 한 번 흐르기 시작하면 *"이 테이블 누가 써요?"* 같은 질문에 그래프로 답할 수 있게 돼요. 파이프라인이 열 개를 넘어가는 순간부터 체감이 확 옵니다. listener 라는 작은 훅 하나가 이렇게 큰 그림으로 이어지는 게 Airflow 의 재미있는 지점이에요.


일단 오늘은 여기까지.....  
다음 글에서는 수집한 계보 위에서 **데이터 품질 검사(Great Expectations)** 결과까지 OpenLineage 이벤트에 실어 보내는 구성을 정리해볼게요.

---

**← 이전 글:** [Airflow 플러그인 제대로 활용하기 — Slack/Teams 알람부터 멀티 스케줄까지](/coding/Airflow_플러그인_제대로_활용하기/)
