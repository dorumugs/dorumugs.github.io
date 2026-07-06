---
layout: single
title:  "(5/6) LLM API 비용 추적 — Grafana 대시보드로 비용 한눈에 보기"
date: 2026-07-07 07:20:00 +0900
categories: coding
tag: [LLM, API비용, Grafana, 대시보드, 시각화, SQLite, Docker, Infinity, 예산알림, FinOps]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-06-llm-cost/header-5.svg
  teaser: /assets/images/2026-07-06-llm-cost/header-5.svg
description: "2편 트래커가 쌓아온 SQLite 비용 데이터를 Grafana 대시보드로 올립니다. Docker Compose 셋업, SQLite 데이터소스 프로비저닝, 일별·모델별·팀별 패널 쿼리, 예산 초과 알림, 그리고 Infinity 데이터소스로 공식 cost_report 를 겹쳐 보는 법까지 — 비용의 추세가 눈에 보이는 상태를 만듭니다."
---

{% include series-llm-cost.html current="5" %}

## Summary

시리즈를 따라오셨다면 데이터는 이미 충분히 쌓여 있어요. 2편 트래커가 호출마다 SQLite 에 비용을 적고, 3편 대사가 공식 수치와 맞추고, 4편이 키·팀 단위로 귀속까지 해놨습니다. 그런데 이 데이터를 보는 방법이 아직 **Slack 텍스트와 터미널 출력**뿐이에요.

숫자는 질문에 답하지만, **추세는 그림이어야 보입니다.** "이번 주 비용이 왜 늘었지?"라는 질문에 그래프 한 장이면 "수요일부터 dev 키의 Opus 호출이 3배가 됐네"까지 10초 만에 도달해요. 마지막 편은 쌓인 데이터를 Grafana 대시보드로 올리는 이야기입니다.

> 💡 이 글에서 다루는 것
> - Slack 리포트와 대시보드의 역할 분담
> - Docker Compose 로 Grafana + SQLite 데이터소스 띄우기
> - 데이터소스 프로비저닝 — 클릭 없이 코드로
> - 패널 쿼리 5종 — 이번 달 총액 · 일별 추이 · 모델별 · 팀별 · 캐시 적중률
> - 예산 초과를 Grafana 알림으로 — Slack 연동
> - Infinity 데이터소스로 공식 cost_report 겹쳐 보기 (+ Admin 키 보안 주의)

<br>

<br>



## 1. Slack 리포트로 부족해지는 순간

3편의 아침 Slack 리포트는 여전히 유효해요. 다만 리포트와 대시보드는 답하는 질문이 다릅니다.

| | Slack 리포트 (3편) | Grafana 대시보드 (이번 편) |
|---|---|---|
| 답하는 질문 | 어제 얼마 썼나 | **언제부터, 왜** 늘었나 |
| 소비 방식 | 푸시 — 아침에 도착 | 풀 — 이상할 때 들어가서 탐색 |
| 시간 축 | 하루 스냅샷 | 기간을 자유롭게 늘였다 줄였다 |
| 드릴다운 | 없음 (고정 포맷) | 프로바이더 → 모델 → 팀으로 파고들기 |

둘은 대체 관계가 아니라 병행 관계예요. 리포트가 "이상하다"는 신호를 주면, 대시보드에서 원인을 찾는 흐름입니다.

도구로 Grafana 를 고른 이유는 단순해요. 인프라 모니터링에서 이미 사실상 표준이라 팀에 한 대쯤 떠 있을 확률이 높고, 2편의 SQLite 파일도·3편의 공식 API 도 플러그인으로 바로 붙습니다. 비용 시각화만을 위해 새 도구를 배울 필요가 없어요.

<br>

<br>



## 2. Docker Compose 로 띄우기

Grafana 본체 + 플러그인 두 개면 끝입니다. 플러그인은 컨테이너 기동 시 `GF_INSTALL_PLUGINS` 환경변수로 자동 설치돼요.

- [`frser-sqlite-datasource`](https://grafana.com/grafana/plugins/frser-sqlite-datasource/) — SQLite 파일을 데이터소스로 (2편 트래커용)
- [`yesoreyeram-infinity-datasource`](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/) — JSON API 를 데이터소스로 (7장의 공식 API 용, Grafana Labs 공식 유지)

```yaml
# docker-compose.yml
services:
  grafana:
    image: grafana/grafana-oss:latest   # 운영이라면 버전을 고정하세요
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: <GRAFANA_ADMIN_PASSWORD>
      GF_INSTALL_PLUGINS: frser-sqlite-datasource,yesoreyeram-infinity-datasource
    volumes:
      - ./data:/var/lib/grafana-data:ro          # llm_usage.db 가 들어있는 디렉토리
      - ./provisioning:/etc/grafana/provisioning  # 3장의 프로비저닝 파일
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

2편의 `llm_usage.db` 를 `./data/` 아래로 옮기고 (트래커의 `DB_PATH` 도 같이), `docker compose up -d` 후 `http://localhost:3000` 으로 들어가면 됩니다.

> ⚠️ SQLite 는 **파일 하나가 아니라 디렉토리째** 마운트하세요. 트래커 쪽에서 `PRAGMA journal_mode=WAL` 을 켰다면 `-wal`/`-shm` 동반 파일이 같은 디렉토리에 생기는데, 파일 단위 마운트면 Grafana 가 그걸 못 봐서 읽기가 깨집니다. 읽기 전용(`:ro`)이니 대시보드가 데이터를 오염시킬 걱정은 없어요.

<br>

<br>



## 3. 데이터소스 프로비저닝 — 클릭 없이 코드로

데이터소스를 UI 에서 클릭으로 등록해도 되지만, 컨테이너를 날렸다 다시 띄울 때마다 반복하게 됩니다. Grafana 는 `provisioning/` 디렉토리의 YAML 을 기동 시 자동 반영하니까, 파일로 박아두는 게 한 번에 끝나요.

```yaml
# provisioning/datasources/llm-cost.yml
apiVersion: 1
datasources:
  - name: llm-usage-sqlite
    type: frser-sqlite-datasource
    access: proxy
    isDefault: true
    jsonData:
      path: /var/lib/grafana-data/llm_usage.db
```

컨테이너를 재시작하면 Connections → Data sources 에 `llm-usage-sqlite` 가 이미 들어와 있습니다.

<br>

<br>



## 4. 패널 쿼리 5종

대시보드의 실체는 결국 SQL 쿼리 묶음이에요. 2편 스키마(`ts, provider, model, app, *_tokens, cost_usd`) 그대로 씁니다.

시간 축 처리 하나만 미리 알아두세요 — SQLite 플러그인은 쿼리 에디터에서 **time 컬럼을 지정**하면 unix 초(정수) 또는 RFC3339 문자열을 시간으로 해석합니다. 아래 쿼리들은 일 단위 버킷을 unix 초로 만들어서 `time` 이라는 이름으로 돌려줘요.

**① 이번 달 총비용 — Stat 패널**

```sql
SELECT ROUND(SUM(cost_usd), 2) AS month_usd
FROM llm_usage
WHERE ts >= strftime('%Y-%m-01', 'now');
```

대시보드 맨 위에 큰 숫자 하나. 임계값 색(예: $300 넘으면 빨강)을 걸어두면 들어오자마자 상태가 읽힙니다.

**② 일별 비용 추이 (프로바이더별) — Time series 패널**

```sql
SELECT CAST(strftime('%s', date(ts)) AS INTEGER) AS time,
       provider,
       ROUND(SUM(cost_usd), 4) AS usd
FROM llm_usage
WHERE ts >= date('now', '-30 days')
GROUP BY 1, 2
ORDER BY 1;
```

`provider` 같은 문자열 컬럼이 섞인 long 포맷이라, 패널의 Transform 에서 **Prepare time series → Multi-frame** 한 번만 걸어주면 프로바이더별 선 4개로 갈라집니다. 이 그래프가 이번 편의 주인공이에요 — "언제부터 늘었나"가 여기서 보입니다.

**③ 이번 달 모델별 비용 — Bar gauge 패널**

```sql
SELECT provider || ' / ' || model AS model,
       ROUND(SUM(cost_usd), 2) AS usd
FROM llm_usage
WHERE ts >= strftime('%Y-%m-01', 'now')
GROUP BY 1
ORDER BY 2 DESC;
```

1편에서 본 "비싼 모델로 가는 쉬운 트래픽"을 찾는 뷰예요. 상위 한두 개 모델이 전체의 80%를 차지하는 그림이 보통 나옵니다.

**④ 팀·용도별 Top 10 — Table 패널**

```sql
SELECT app,
       ROUND(SUM(cost_usd), 2) AS usd,
       COUNT(*) AS calls
FROM llm_usage
WHERE ts >= date('now', '-7 days')
GROUP BY app
ORDER BY usd DESC
LIMIT 10;
```

4편에서 정한 `app = "팀/용도"` 태그 규칙이 여기서 빛납니다. 태그만 규칙대로 달려 있으면 이 표가 곧 팀별 차지백 자료예요.

**⑤ 캐시 적중률 추이 — Time series 패널**

```sql
SELECT CAST(strftime('%s', date(ts)) AS INTEGER) AS time,
       ROUND(100.0 * SUM(cached_tokens)
             / NULLIF(SUM(cached_tokens) + SUM(input_tokens), 0), 1) AS cache_hit_pct
FROM llm_usage
WHERE ts >= date('now', '-30 days')
GROUP BY 1
ORDER BY 1;
```

3편 절감 체크리스트 1번(프롬프트 캐싱)이 실제로 작동하는지 감시하는 뷰입니다. 배포 후에 이 선이 뚝 떨어졌다면, 누군가 시스템 프롬프트 앞부분에 타임스탬프 같은 걸 끼워 넣어 캐시를 깨뜨렸다는 신호예요.

<br>

<br>



## 5. 대시보드도 코드로

패널을 UI 에서 다 만들었으면 마지막에 한 번 내보내세요. Dashboard settings → JSON Model 에서 JSON 을 복사해 저장소에 넣고, 데이터소스처럼 프로비저닝으로 물립니다.

```yaml
# provisioning/dashboards/default.yml
apiVersion: 1
providers:
  - name: llm-cost
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

JSON 파일(`llm-cost-dashboard.json`)을 같은 디렉토리에 두면, 컨테이너를 어디서 다시 띄우든 대시보드가 그대로 복원돼요. "Grafana 서버가 죽어서 대시보드를 처음부터 다시 만들었다"는 사고를 막는 보험입니다.

<br>

<br>



## 6. 예산 초과를 Grafana 알림으로

3편의 cron 리포트가 "매일 아침 요약"이라면, Grafana 알림은 **"넘는 순간"** 을 잡습니다. ①번 쿼리를 재활용해서 알림 규칙 하나를 걸어요.

1. Alerting → Alert rules → New alert rule
2. 쿼리 A — 오늘 비용:

```sql
SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage WHERE date(ts) = date('now');
```

3. 조건 — `A > 15` (3편의 일일 예산 $15 그대로), 평가 주기 5분
4. Contact point — Slack (Incoming Webhook URL 등록) 지정

이러면 오후 2시에 예산을 넘겨도 다음 날 아침이 아니라 **5분 안에** Slack 이 웁니다. cron 리포트는 요약용으로 남겨두고, 초과 감지는 이쪽으로 옮기는 걸 추천드려요.

<br>

<br>



## 7. 공식 API 겹쳐 보기 — Infinity 데이터소스

여기까지는 전부 우리 쪽 **추정치**(2편 SQLite) 기반이에요. 3편에서 추정치와 공식치를 대사했는데, 그걸 상시로 눈에 보이게 만들 수 있습니다. Infinity 데이터소스는 임의의 JSON API 를 호출해 패널에 올려주니까, Anthropic `cost_report` 를 직접 붙이면 돼요.

쿼리 설정은 이런 모양입니다.

```text
Type: JSON / Method: GET
URL: https://api.anthropic.com/v1/organizations/cost_report?starting_at=${__from:date:iso}&bucket_width=1d
Headers:
  x-api-key: <ANTHROPIC_ADMIN_KEY>     ← 데이터소스 설정에 저장 (쿼리에 직접 쓰지 않기)
  anthropic-version: 2023-06-01
```

응답의 `data[].results[]` 가 중첩 배열이라 Infinity 의 파서(UQL) 옵션으로 한 번 평탄화해주면, ②번 추정치 그래프 위에 **공식 달러 선을 겹칠** 수 있어요. 두 선의 간격이 3편에서 정한 허용 오차(2%) 안이면 트래커가 건강하다는 뜻이고, 벌어지기 시작하면 트래커가 못 보는 호출(웹 검색 과금, 트래커를 안 거친 실험)이 늘고 있다는 신호입니다.

> 🚨 **Admin 키를 Grafana 에 넣기 전에 한 번만 생각하세요.** Admin 키는 조직 전체의 사용량·키 목록·멤버 관리 권한이 있는 자격증명이에요. 사내 공용 Grafana 에 등록하면 대시보드 편집 권한이 있는 모두가 그 키로 쿼리를 만들 수 있는 셈입니다. 팀 전용 인스턴스가 아니라면 이 장은 건너뛰고, 대신 3편의 대사 스크립트가 공식치를 SQLite 의 별도 테이블에 적재하게 해서 같은 SQLite 데이터소스로 겹치는 우회를 추천드려요.

<br>

<br>



## 8. 시리즈 정리

다섯 편이 만든 전체 시스템입니다.

| 계층 | 도구 | 답하는 질문 |
|---|---|---|
| 구조 이해 (1편) | 가격표 · usage 필드 | 뭐가 얼마인가 |
| 상시 관측 (2편) | `track()` + SQLite | 지금 어디에 쓰고 있나 |
| 정산 대사 (3편) | 공식 Usage/Cost API | 청구서와 맞나 |
| 통제 (3편) | 게이트웨이 예산 | 초과를 막을 수 있나 |
| 귀속 (4편) | 키 인벤토리 · `api_key_id` 그룹핑 · 가상 키 | 누가 쓰고 있나 |
| **시각화 (5편)** | Grafana + SQLite/Infinity | **추세가 보이나** |

돌아보면 시작은 "이번 달 LLM 에 총 얼마 썼지?"라는 한 문장짜리 질문이었어요. 이제 그 질문은 아침 Slack 이 답하고, 초과는 알림이 잡고, 원인은 대시보드에서 10초 만에 찾고, 책임은 키 단위로 귀속됩니다. 비용을 "보이게 만드는" 작업은 여기서 완성이에요. 🎉

일단 오늘은 여기까지.....
다음 글에서는 이 비용 데이터를 근거로 쉬운 작업을 싼 모델로 내리는 **모델 라우팅 자동화**를 다뤄볼게요.

---

**← 이전 글:** [(4/6) LLM API 비용 추적 — 키가 난립할 때: 키 인벤토리와 키별 비용 귀속](/coding/LLM_API_비용추적_키별귀속/) ｜ **다음 글 →:** [(6/6) LLM API 비용 추적 — 모델 라우팅 자동화: 쉬운 작업은 싼 모델로](/coding/LLM_API_비용추적_모델라우팅/)
