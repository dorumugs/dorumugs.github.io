---
layout: single
title:  "(4/4) 계보·품질 신호로 Slack 데이터 품질 리포트 자동화하기"
date: 2026-06-30 06:30:00 +0900
categories: coding
tag: [Airflow, Slack, 데이터품질, OpenLineage, Marquez, 리포트, data-quality, observability, SlackWebhookHook, data-engineering]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-30-airflow-slack-report/header.svg
  teaser: /assets/images/2026-06-30-airflow-slack-report/header.svg
description: "지금까지 쌓은 계보·품질 신호를 모아 매일 Slack 으로 데이터 품질 리포트를 자동 발행하는 구성을 정리합니다. Marquez API 에서 실패한 품질 단언을 수집하고, Slack Block Kit 으로 다이제스트를 만들어 리포트 DAG 로 매일 보내기."
---

{% include series-airflow-ops.html current="4" %}


## Summary

여기까지 오느라 고생하셨어요. [1편](/coding/Airflow_플러그인_제대로_활용하기/)에서 플러그인으로 알람을 깔고, [2편](/coding/Airflow_listener_OpenLineage_데이터계보/)에서 계보를 자동 수집하고, [3편](/coding/Airflow_GreatExpectations_OpenLineage_데이터품질/)에서 그 계보 위에 품질 검사를 얹었어요. 이제 신호는 다 모였습니다. 마지막 한 조각은 **"그 신호를 사람이 보게 만드는"** 거예요.

품질 검사가 깨질 때마다 즉시 알림을 쏘는 건 3편의 콜백으로 이미 할 수 있어요. 그런데 검사가 많아지면 **개별 알림이 쏟아져서 오히려 안 보게** 돼요. 그래서 즉시 알림과 별개로, **하루 한 번 "지금 어느 데이터셋이 깨져 있고 언제부터인지" 를 요약하는 다이제스트**가 필요합니다. 이번 글의 주제예요.

> 💡 **이 글에서 다루는 것**
> - 즉시 알림 vs **다이제스트 리포트** — 왜 둘 다 필요한가
> - 신호가 모여 있는 곳 — Marquez API 에서 품질 facet 읽기
> - 실패한 품질 단언만 **추려내기**
> - Slack Block Kit 으로 **읽기 좋은 리포트** 만들기
> - 매일 보내는 **리포트 DAG** 구성
> - "언제부터 깨졌나" 까지 붙이기
> - 운영에서 자주 밟는 지뢰

시리즈의 마지막 편이라, 앞 세 편이 다 돌고 있다는 가정에서 시작할게요.


<br>

<br>



## 1. 즉시 알림 vs 다이제스트 리포트

둘은 목적이 달라요. 헷갈리면 알림 피로만 쌓이니 먼저 정리하고 갈게요.

| | 즉시 알림 | 다이제스트 리포트 |
|---|---|---|
| 언제 | 검사가 깨지는 **그 순간** | 하루 한 번 정해진 시각 |
| 무엇 | 방금 깨진 검사 하나 | **지금 깨져 있는 전체 + 언제부터** |
| 어떻게 | 3편 콜백(`on_failure_callback`) | 리포트 DAG 가 모아서 발행 |
| 문제 | 많아지면 묻힘 | (없음 — 한 장으로 요약) |

즉시 알림은 **빨리 알아채는 용도**, 다이제스트는 **현재 상태를 놓치지 않는 용도**예요. 즉시 알림만 켜두면 바쁠 때 흘려보낸 알림이 그대로 묻혀버려요. 반대로 다이제스트만 있으면 급한 사고를 하루나 늦게 알게 되고요. 그래서 **둘 다** 두는 걸 추천드립니다. 이번 글은 후자인 다이제스트를 만듭니다.


<br>

<br>



## 2. 신호는 Marquez 에 모여 있다

3편에서 품질 검사 결과를 `dataQualityAssertions` facet 으로 만들어 OpenLineage 이벤트로 보냈죠. 그 이벤트는 2편에서 띄운 Marquez 에 쌓여 있어요. 그러니 리포트는 **Marquez API 에서 데이터셋별 품질 facet 을 읽어오는** 것으로 시작합니다.

네임스페이스의 데이터셋 목록을 받아오는 호출이에요.

```python
import requests

MARQUEZ = "http://localhost:5000"
NAMESPACE = "airflow-prod"

resp = requests.get(
    f"{MARQUEZ}/api/v1/namespaces/{NAMESPACE}/datasets",
    timeout=10,
)
resp.raise_for_status()
datasets = resp.json()["datasets"]
print(len(datasets), "개 데이터셋")
print(datasets[0]["name"])
```

```text
12 개 데이터셋
public.orders
```

각 데이터셋 응답 안에 `facets` 가 들어 있고, 거기에 우리가 3편에서 실어 보낸 `dataQualityAssertions` 가 보여요.

> ⚠️ facet 이 응답의 **어느 경로에** 실리는지는 Marquez 버전마다 조금씩 달라요(데이터셋 본문 `facets` 아래일 때도, 최신 버전 facet 아래일 때도 있어요). 한 번 `print(resp.json())` 으로 실제 응답을 찍어보고 경로를 맞추는 걸 권장드려요. 아래 코드는 가장 흔한 `ds["facets"]["dataQualityAssertions"]` 형태를 가정했습니다.


<br>

<br>



## 3. 실패한 단언만 추려내기

데이터셋을 돌면서 **통과하지 못한 단언이 하나라도 있는** 데이터셋만 모읍니다.

```python
def collect_failures(namespace=NAMESPACE):
    resp = requests.get(
        f"{MARQUEZ}/api/v1/namespaces/{namespace}/datasets", timeout=10
    )
    resp.raise_for_status()

    failures = []
    for ds in resp.json()["datasets"]:
        facet = ds.get("facets", {}).get("dataQualityAssertions")
        if not facet:
            continue                       # 품질 검사가 없는 데이터셋은 건너뜀
        broken = [a for a in facet["assertions"] if not a["success"]]
        if broken:
            failures.append({
                "dataset": ds["name"],
                "broken": [a["assertion"] for a in broken],
                "updated": ds.get("updatedAt"),
            })
    return failures
```

돌려보면 이렇게 추려져요.

```python
failures = collect_failures()
for f in failures:
    print(f["dataset"], "->", f["broken"], "(", f["updated"], ")")
```

```text
public.orders -> ['expect_table_row_count_to_be_between'] ( 2026-06-30T05:00:12Z )
```

이제 이 리스트만 Slack 으로 예쁘게 보내면 됩니다.


<br>

<br>



## 4. Slack Block Kit 으로 리포트 만들기

1편에서 `SlackWebhookHook` 으로 텍스트 한 줄을 보냈는데, 리포트는 **Block Kit** 으로 구조를 잡으면 훨씬 읽기 좋아요. 헤더 + 목록으로 나눕니다. 깨진 게 없으면 초록불 한 줄로 끝내고요.

```python
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook


def build_blocks(failures):
    if not failures:
        return [{
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": ":large_green_circle: *오늘 데이터 품질 이상 없음* — 모든 검사 통과"},
        }]

    lines = [
        f"• `{f['dataset']}` — {', '.join(f['broken'])}  _(업데이트 {f['updated']})_"
        for f in failures
    ]
    return [
        {"type": "header",
         "text": {"type": "plain_text", "text": f"🔴 데이터 품질 리포트 — {len(failures)}건"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
    ]


def post_quality_report():
    failures = collect_failures()
    SlackWebhookHook(slack_webhook_conn_id="slack_alert").send(
        blocks=build_blocks(failures)
    )
```

`slack_alert` 커넥션은 1편에서 만든 그대로 재사용해요. 채널에는 이렇게 떠요.

```text
🔴 데이터 품질 리포트 — 1건
• public.orders — expect_table_row_count_to_be_between  (업데이트 2026-06-30T05:00:12Z)
```

깨진 게 없는 날은 `🟢 오늘 데이터 품질 이상 없음` 한 줄이 와요. 이 "이상 없음" 메시지도 의외로 중요해요. **리포트가 매일 온다는 사실 자체**가 파이프라인이 살아 있다는 신호거든요. (어느 날 리포트가 안 오면 리포트 DAG 가 죽은 거예요.)


<br>

<br>



## 5. 매일 보내는 리포트 DAG

마지막으로 위 함수를 매일 정해진 시각에 돌리는 DAG 하나. 1편에서 만든 실패 콜백도 같이 걸어서, **리포트 DAG 자체가 실패하면** 그것도 알림이 오게 해둬요.

```python
# dags/data_quality_report.py
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from callbacks.alerts import slack_fail_alert       # 1편의 실패 콜백
from reports.quality import post_quality_report     # 위에서 만든 함수

with DAG(
    dag_id="data_quality_report",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
    schedule="0 9 * * *",                            # 매일 아침 9시 다이제스트
    catchup=False,
    default_args={"on_failure_callback": slack_fail_alert},
    tags=["quality", "report"],
) as dag:
    PythonOperator(
        task_id="post_report",
        python_callable=post_quality_report,
    )
```

이러면 매일 아침 9시에 "지금 깨져 있는 데이터셋" 한 장이 Slack 으로 와요. 검사가 늘어나도 리포트는 항상 한 장이라 알림 피로가 안 생깁니다.

> 💡 여러 시각에 보내고 싶으면(예: 아침 9시 + 점심 1시) 1편의 **커스텀 멀티 cron Timetable** 을 그대로 끼우면 돼요. 시리즈가 한 바퀴 돌아 첫 편의 도구를 다시 쓰는 지점이에요.


<br>

<br>



## 6. "언제부터 깨졌나" 까지

현재 상태만으로도 충분히 쓸모 있지만, 사고 대응에는 **"언제부터"** 가 결정적이에요. Marquez 는 데이터셋의 **버전 이력**을 들고 있어서, 데이터셋 버전 목록을 거슬러 올라가며 품질 facet 이 **마지막으로 통과했던 버전**을 찾으면 "그 직후부터 깨졌다" 를 알 수 있어요.

```python
def first_broken_since(namespace, dataset):
    resp = requests.get(
        f"{MARQUEZ}/api/v1/namespaces/{namespace}/datasets/{dataset}/versions",
        timeout=10,
    )
    versions = resp.json()["versions"]   # 최신순 가정
    last_ok = None
    for v in versions:
        facet = v.get("facets", {}).get("dataQualityAssertions")
        passed = facet and all(a["success"] for a in facet["assertions"])
        if passed:
            last_ok = v["createdAt"]
            break
    return last_ok                       # 이 시각 직후부터 깨진 것
```

리포트 줄에 이 값을 붙이면 `public.orders — ... (정상이던 마지막: 2026-06-28 05:00)` 처럼 보여서, 어느 배포·어느 데이터부터 틀어졌는지 바로 좁혀집니다.


<br>

<br>



## 7. 운영에서 자주 밟는 지뢰

- **리포트가 조용히 죽는 게 제일 무서워요.** Marquez 가 잠깐 떠 있지 않으면 리포트 DAG 가 실패하는데, 그걸 모르면 "이상 없음" 이 아니라 "리포트가 안 온 것" 인데도 안심해버려요. 그래서 5편처럼 리포트 DAG 에도 **실패 콜백**을 꼭 걸어두세요.
- **"이상 없음" 도 매일 보내세요.** 침묵을 정상으로 해석하면 사고를 늦게 알아챕니다. 초록불이라도 매일 오는 게 살아있다는 증거예요.
- **Marquez API 응답 모양은 버전을 타요.** 2편의 ⚠️ 그대로 — facet 경로·필드명을 응답을 직접 찍어 맞추세요.
- **타임존.** Marquez 의 `updatedAt` 은 UTC(`...Z`)예요. 리포트에 그대로 박으면 한국 시각과 9시간 차이로 헷갈리니, 보낼 때 KST 로 변환하면 깔끔해요. (1편 알람에서 짚은 것과 같은 함정이에요.)
- **리포트 범위를 좁히세요.** 데이터셋이 수백 개면 전부 훑는 게 느려요. 중요 네임스페이스·핵심 테이블만 추리거나, 실패 건만 모으는 쿼리로 범위를 제한하는 걸 권장드립니다.


여기까지가 **Airflow 운영 확장 4부작**이에요. 플러그인으로 알람을 깔고, 계보를 자동으로 모으고, 그 위에 품질을 얹고, 마지막으로 그 신호를 사람이 보는 리포트로 닫았어요. DAG 만 잘 짜던 단계에서 한 발 더 나아가, **파이프라인이 스스로 자기 상태를 말하게 만드는** 그림이 완성됐습니다. 네 편을 따라오셨다면 이제 직접 한 조각씩 끼워 넣어 보세요.


일단 오늘은 여기까지.....  
긴 시리즈 따라오시느라 고생 많으셨어요. 다음엔 또 다른 주제로 찾아올게요.

---

**← 이전 글:** [(3/4) Great Expectations 품질 검사 결과를 OpenLineage 계보에 실어 보내기](/coding/Airflow_GreatExpectations_OpenLineage_데이터품질/)
