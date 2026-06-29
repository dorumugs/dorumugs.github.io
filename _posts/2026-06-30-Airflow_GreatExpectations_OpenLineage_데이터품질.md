---
layout: single
title:  "(3/4) Great Expectations 품질 검사 결과를 OpenLineage 계보에 실어 보내기"
date: 2026-06-30 06:00:00 +0900
categories: coding
tag: [Airflow, GreatExpectations, OpenLineage, 데이터품질, data-quality, lineage, Marquez, dataQualityAssertions, checkpoint, data-engineering]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-30-airflow-dataquality/header.svg
  teaser: /assets/images/2026-06-30-airflow-dataquality/header.svg
description: "Great Expectations 로 데이터 품질을 검사하고, 그 결과를 OpenLineage 의 품질 facet 으로 만들어 계보 그래프에 함께 싣는 구성을 정리합니다. GX 핵심 개념, Airflow 에서 검사 돌리기, 검증 결과를 facet 으로 매핑해 Marquez 에서 품질까지 보기."
---


{% include series-airflow-ops.html current="3" %}


## Summary

[지난 글](/coding/Airflow_listener_OpenLineage_데이터계보/)에서 listener 와 OpenLineage 로 계보 그래프를 자동으로 채웠어요. 이제 "이 데이터가 어디서 와서 어디로 가는지" 는 그래프로 보입니다. 그런데 한 가지가 빠져 있어요. **"그 데이터가 믿을 만한가?"** 예요.

계보 노드 위에 **품질(quality) 한 겹**을 더 얹으면, 그래프를 보면서 *"이 테이블은 어제 검사 통과했고, 저 테이블은 null 검사에서 깨졌네"* 까지 한눈에 알 수 있어요. 이걸 가능하게 하는 게 **Great Expectations(GX) + OpenLineage 품질 facet** 조합입니다.

이 글에서는 GX 로 품질을 검사하고, 그 결과를 OpenLineage 이벤트에 실어 지난 글에서 세운 Marquez 그래프에 품질까지 표시되게 만드는 구성을 정리합니다.

> 💡 **이 글에서 다루는 것**
> - 계보 위에 **품질** 한 겹을 더 얹는 이유
> - Great Expectations 핵심 개념 — Expectation · Suite · Checkpoint
> - Airflow 에서 GX 검사 돌리기 (`GreatExpectationsOperator`)
> - 검사 결과 → **OpenLineage 품질 facet** 매핑
> - 커스텀 오퍼레이터에서 `get_openlineage_facets_on_complete` 로 싣기
> - Marquez 에서 계보 + 품질 같이 보기
> - 운영에서 자주 밟는 지뢰 (특히 **버전 churn**)

지난 두 글과 이어지는 내용이라, 계보 파이프라인이 이미 돌고 있다는 가정에서 시작할게요.


<br>

<br>



## 1. 계보 위에 "품질" 한 겹을 더

지난 글에서 만든 계보 그래프는 **연결 관계**를 보여줘요. `raw_events` → `orders` → `daily_sales` 같은 흐름이죠. 그런데 운영에서 진짜 무서운 건 *"연결은 멀쩡한데 데이터가 틀린"* 상황이에요. 파이프라인은 초록불인데 `amount` 컬럼이 절반쯤 비어 있는 식으로요.

그래서 계보에 **품질 검사 결과를 같이 기록**해두면 강력해져요.

- 데이터셋 노드를 클릭하면 **"마지막 검사에서 뭐가 통과/실패했는지"** 가 같이 보임
- 품질이 깨진 데이터셋을 그래프에서 **빨간 노드로 추적**할 수 있음
- 검사 결과가 계보와 한 타임라인에 쌓여서, *"언제부터 깨졌나"* 를 거슬러 볼 수 있음

OpenLineage 는 이걸 위해 **품질 facet** 이라는 표준 자리를 데이터셋에 마련해뒀어요. 검사 도구로는 가장 널리 쓰이는 **Great Expectations** 를 붙여볼게요.


<br>

<br>



## 2. Great Expectations 핵심만

GX 는 **데이터에 대한 단언(assertion)을 선언적으로 적고, 그게 맞는지 검사**해주는 도구예요. 세 가지 개념만 잡으면 됩니다.

| 개념 | 무엇 |
|---|---|
| **Expectation** | 데이터에 대한 단언 하나. 예: `expect_column_values_to_not_be_null("amount")` |
| **Expectation Suite** | Expectation 들을 묶은 한 세트. "orders 테이블 검사 규칙" 같은 단위 |
| **Checkpoint** | 데이터 배치 + Suite + 후속 액션을 묶어 **실제로 검사를 돌리는** 실행 단위 |

흐름을 한 그림으로 요약하면 다음과 같아요.

```text
[데이터 배치] + [Expectation Suite] ─→ Checkpoint 실행 ─→ ValidationResult
                                                              │
                                              (성공/실패 + 통계 + 어느 규칙이 깨졌나)
```

작은 검사를 직접 돌려보면 결과가 어떻게 생겼는지 감이 와요.

```python
import great_expectations as gx

context = gx.get_context()
batch = context.sources.pandas_default.read_csv("orders.csv")

# Expectation 두 개 선언
batch.expect_column_values_to_not_be_null("amount")
batch.expect_table_row_count_to_be_between(min_value=1, max_value=1_000_000)

result = batch.validate()
print(result.success)
print(result.statistics)
```

```text
False
{'evaluated_expectations': 2, 'successful_expectations': 1,
 'unsuccessful_expectations': 1, 'success_percent': 50.0}
```

`result.success` 가 전체 통과 여부, `result.statistics` 가 몇 개 중 몇 개 통과인지예요. 우리가 facet 으로 옮길 핵심 정보가 바로 이 안에 들어 있습니다.

> ⚠️ **GX 는 버전 사이 API 변화가 유난히 커요.** 0.18 대와 1.0 대의 사용법(`get_context` 이후 배치/검증 흐름)이 꽤 다릅니다. 위 코드는 흐름을 보여주려는 의사 코드에 가까워요. 실제로는 **설치한 GX 버전의 문서에 맞춰** 메서드 이름을 확인하세요. 핵심 개념(Expectation·Suite·Checkpoint·ValidationResult)은 버전이 바뀌어도 그대로예요.


<br>

<br>



## 3. Airflow 에서 GX 검사 돌리기

Airflow 에서는 `airflow-provider-great-expectations` 의 `GreatExpectationsOperator` 로 Checkpoint 를 태스크 하나로 돌릴 수 있어요.

```shell
pip install airflow-provider-great-expectations
```

```python
# dags/quality_check.py
from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

validate_orders = GreatExpectationsOperator(
    task_id="validate_orders",
    data_context_root_dir="include/gx",     # GX 프로젝트 디렉토리
    checkpoint_name="orders_checkpoint",
    # 검사가 깨지면 이 태스크도 실패시켜 다운스트림을 막을지
    fail_task_on_validation_failure=True,
)
```

`fail_task_on_validation_failure` 가 운영에서 중요한 선택이에요.

- `True` — 검사 실패 = 태스크 실패. **나쁜 데이터가 다운스트림으로 못 흐르게** 막음. (지난 [플러그인 글](/coding/Airflow_플러그인_제대로_활용하기/)의 Slack/Teams 알람과 엮으면, 품질이 깨지는 순간 바로 알림이 와요.)
- `False` — 검사는 기록만 하고 파이프라인은 계속 진행. **일단 관측부터** 하고 싶을 때.

저는 새 검사를 도입할 때는 `False` 로 한동안 관측해서 오탐을 걷어낸 뒤, 안정되면 `True` 로 올려 게이트로 쓰는 걸 추천드려요.


<br>

<br>



## 4. 검사 결과를 OpenLineage 품질 facet 으로

이제 핵심. GX 가 만든 `ValidationResult` 를 OpenLineage 의 **품질 facet** 으로 옮기면, 지난 글의 계보 그래프에 품질이 얹힙니다. OpenLineage 가 데이터셋에 붙일 수 있는 품질 facet 은 두 가지예요.

| facet | 담는 것 |
|---|---|
| `dataQualityAssertions` | **어떤 단언이 통과/실패했나** (규칙별 성공 여부) |
| `dataQualityMetrics` | 행 수 · 컬럼별 null 수 같은 **측정값** |

이 중 `dataQualityAssertions` 가 GX 결과와 가장 자연스럽게 매핑돼요. facet 한 장의 모양은 이렇습니다.

```python
from openlineage.client.facet import (
    DataQualityAssertionsDatasetFacet,
    Assertion,
)

facet = DataQualityAssertionsDatasetFacet(
    assertions=[
        Assertion(assertion="expect_column_values_to_not_be_null",
                  success=True, column="amount"),
        Assertion(assertion="expect_table_row_count_to_be_between",
                  success=False),
    ]
)
```

`assertion` 에 규칙 이름, `success` 에 통과 여부, `column` 에 (컬럼 단위 검사면) 대상 컬럼. GX 의 `ValidationResult` 안에 있는 결과들을 이 모양으로 한 번 변환만 해주면 돼요.

> 💡 품질 검사를 OpenLineage 로 보내는 길은 두 갈래예요. (1) **GX 의 OpenLineage Action** 을 Checkpoint 에 붙여 자동 발행하는 방법과, (2) **직접 facet 을 만들어** 오퍼레이터에서 싣는 방법. (1)은 편하지만 GX·통합 패키지의 버전 궁합을 타서, 버전이 엇갈리면 잘 안 붙어요. 이 글에서는 **버전에 덜 휘둘리는 (2)번** 을 중심으로 보여드릴게요.


<br>

<br>



## 5. 커스텀 오퍼레이터에서 facet 싣기

지난 글에서 커스텀 오퍼레이터에 `get_openlineage_facets_on_start` 를 달아 입력/출력을 알려줬죠. 품질은 검사가 **끝난 뒤**에 결과가 나오니까, 이번엔 `get_openlineage_facets_on_complete` 를 씁니다. 이름 그대로 태스크 실행이 끝난 시점에 프로바이더가 불러줘요.

GX 검사를 실행하고, 그 결과를 품질 facet 으로 변환해 출력 데이터셋에 싣는 오퍼레이터예요.

```python
# plugins/operators/quality_check.py
from airflow.models import BaseOperator
from airflow.providers.openlineage.extractors import OperatorLineage


class OrdersQualityOperator(BaseOperator):
    def execute(self, context):
        # GX Checkpoint 를 실행하고 결과를 보관해둔다
        self.result = run_gx_checkpoint("orders_checkpoint")  # ValidationResult
        if not self.result.success:
            self.log.warning("품질 검사 실패 — facet 으로 기록은 남깁니다")
        return self.result.success

    # 태스크가 끝난 뒤 호출 — 검사 결과를 품질 facet 으로 싣는다
    def get_openlineage_facets_on_complete(self, task_instance):
        from openlineage.client.run import Dataset
        from openlineage.client.facet import (
            DataQualityAssertionsDatasetFacet,
            Assertion,
        )

        # ValidationResult → Assertion 리스트로 변환
        assertions = [
            Assertion(
                assertion=r["expectation_config"]["expectation_type"],
                success=r["success"],
                column=r["expectation_config"]["kwargs"].get("column"),
            )
            for r in self.result.results
        ]

        orders = Dataset(
            namespace="postgres://db:5432",
            name="public.orders",
            facets={
                "dataQualityAssertions": DataQualityAssertionsDatasetFacet(
                    assertions=assertions
                )
            },
        )
        return OperatorLineage(inputs=[orders], outputs=[orders])
```

`self.result.results` 안의 검사 하나하나를 `Assertion` 으로 펼치는 게 전부예요. 변환된 결과를 잠깐 찍어보면 이렇게 생겼어요.

```python
for a in assertions:
    print(a.assertion, "->", a.success, "(col:", a.column, ")")
```

```text
expect_column_values_to_not_be_null -> True (col: amount )
expect_table_row_count_to_be_between -> False (col: None )
```

이제 이 데이터셋이 OpenLineage 이벤트로 나가면, **같은 `namespace`+`name`(`postgres://db:5432` 의 `public.orders`)을 쓰던 지난 글의 계보 노드에 품질 facet 이 그대로 합쳐져요.** 계보를 그리던 그 노드 위에 품질이 한 겹 얹히는 거예요.


<br>

<br>



## 6. Marquez 에서 계보 + 품질 같이 보기

지난 글에서 띄운 Marquez(`http://localhost:3000`)로 가서, DAG 를 한 번 돌린 뒤 `public.orders` 데이터셋 노드를 클릭해보세요. 계보 연결은 그대로 있고, 거기에 **품질 검사 결과(통과/실패한 단언 목록)** 가 함께 붙어 있는 걸 볼 수 있어요.

확인 포인트.

- [x] 데이터셋 노드에 `dataQualityAssertions` 가 표시되는가
- [x] 실패한 단언이 구분돼 보이는가
- [x] 검사를 다시 돌리면 **이력**으로 쌓이는가 (언제부터 깨졌는지 추적용)

여기까지 오면, 지난 세 글이 하나로 모여요. **플러그인**으로 알림을 깔고([1편](/coding/Airflow_플러그인_제대로_활용하기/)), **listener/OpenLineage** 로 계보를 채우고([2편](/coding/Airflow_listener_OpenLineage_데이터계보/)), 그 위에 **품질**까지 얹은 그림입니다.


<br>

<br>



## 7. 운영에서 자주 밟는 지뢰

- **버전 궁합이 제일 큰 함정이에요.** GX, `airflow-provider-great-expectations`, OpenLineage 클라이언트 세 개가 각자 빠르게 바뀌어서, 조합이 엇갈리면 import 부터 깨지거나 facet 이 조용히 안 붙어요. 셋의 버전을 **같이 고정(pin)** 해두고, 올릴 땐 한 번에 검증하는 걸 권장드립니다.
- **facet 클래스 import 경로도 버전을 타요.** `openlineage.client.facet` 이 최신에서는 `facet_v2` 쪽으로 옮겨가고 있어요. import 에러가 나면 먼저 설치된 클라이언트의 모듈 경로를 확인하세요.
- **검사 실패 시 막을지 말지를 먼저 정하세요.** 3장의 `fail_task_on_validation_failure` 그대로 — 게이트로 쓸지 관측만 할지는 검사의 성숙도에 따라 갑니다. 처음부터 `True` 로 막으면 오탐 때문에 파이프라인이 자주 멈춰서 신뢰를 잃어요.
- **품질 검사는 비용이에요.** 매 실행마다 전체 테이블을 훑으면 느려져요. 큰 테이블은 **증분 배치**나 샘플링으로 검사 범위를 좁히는 걸 고려하세요.
- **namespace 일관성(또again).** 지난 글의 계보 노드와 **똑같은** `namespace`+`name` 을 써야 품질이 그 노드에 합쳐져요. 한 글자라도 다르면 별개 노드로 갈라져서, 계보 따로 품질 따로 떠다니게 됩니다.

데이터 품질은 "한 번 검사 붙이고 끝" 이 아니라, 계보 위에 쌓이면서 **시간축으로 신뢰를 만들어가는** 작업이에요. 어제는 통과했는데 오늘 깨졌다는 걸 그래프에서 바로 보는 순간, 데이터 사고 대응 속도가 확 달라집니다.


일단 오늘은 여기까지.....  
다음 글에서는 이 품질·계보 신호를 받아 **Slack 으로 "어느 데이터셋이 언제부터 깨졌는지"** 를 자동 리포트하는 구성을 정리해볼게요.

---

**← 이전 글:** [(2/4) Airflow listener 와 OpenLineage 로 데이터 계보(lineage) 자동 수집하기](/coding/Airflow_listener_OpenLineage_데이터계보/)
