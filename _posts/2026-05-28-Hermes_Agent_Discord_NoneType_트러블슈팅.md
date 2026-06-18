---
layout: single
title:  "(2/2) Hermes Agent + Discord: 'NoneType' object is not iterable 트러블슈팅"
date: 2026-05-28 21:11:00 +0900
description: "Discord에 연결한 Hermes Agent가 메시지/크론 작업마다 'NoneType' object is not iterable 로 죽던 이슈를 추적하고, Docker 환경에서 최신 이미지로 갈아끼워서 해결한 과정을 정리했어요."
categories: coding
tag: [hermes, discord, docker, troubleshooting, codex, openai, llm, agent, bug]
author_profile: false
toc: true
---



{% include series-hermes-agent.html current="2" %}

## Summary

[지난 글](/coding/Hermes_Agent_Docker_셋업/)에서 Docker Compose로 띄운 **Hermes Agent**에 Discord 봇을 연결해서
며칠 잘 굴리고 있었는데, 어느 순간부터 메시지를 보낼 때마다 봇이 같은 에러를 토하면서 죽기 시작했어요.   
"매일 아침 7시" 크론 작업까지 같이 망가져서 한참 들여다봤습니다. 

결론부터 말하면 제 설정 실수가 아니라 **Hermes Agent의 알려진 버그**였고,
**최신 이미지로 갈아끼우면 해결**됩니다.   
다만 Docker 환경에서는 `hermes update` 한 줄로 끝나지 않아서, 그 부분이 살짝 함정이에요. 

> 💡 **이 글에서 다루는 것**
> - `'NoneType' object is not iterable` 에러의 정체와 발생 패턴
> - Codex OAuth 스트림 처리 버그가 어떻게 봇 전체를 멈추는지
> - 일반 설치 vs Docker 환경별 업데이트 방법
> - `hermes update` 가 Docker에서 정답이 아닌 이유
> - 업데이트 후에도 불안할 때의 백업안

<br>

<br>



## 1. 증상

Discord에서 봇을 멘션하거나 크론이 돌 때마다 로그에 같은 메시지가 반복해서 찍혔어요. 

```text
Non-retryable error (HTTP None): 'NoneType' object is not iterable
Non-retryable error (HTTP None) — trying fallback...
'NoneType' object is not iterable
```

특징을 정리하면 이래요. 

- 본 요청 실패 → `trying fallback...` → 폴백 경로도 **동일 에러로** 실패
- `HTTP None` 으로 기록 — 정상 HTTP 응답 단계가 아니라 **스트림 처리 도중** 터졌다는 신호
- Hermes가 이걸 `Non-retryable` 로 분류해서 재시도조차 안 함

여기서 가장 골치 아팠던 건 인터랙티브 메시지뿐 아니라 **에이전트가 돌리는 크론 작업이 통째로 같이 망가졌다**는 점이에요.   
저는 매일 아침 7시에 도는 자동화가 며칠 조용히 누락된 다음에야 눈치챘습니다. 

> ⚠️ 한 번 죽으면 봇이 살아있긴 한데 응답만 안 와요. 컨테이너는 멀쩡히 돌고 있어서 헬스체크로도 안 잡혀요. 크론 결과를 Discord로 받는 구조라면 **결과 메시지가 안 오는 것 자체가 알람**이라는 걸 인지하고 있어야 합니다. 

<br>

<br>



## 2. 원인 — Codex 백엔드의 `output=None`

설정 실수가 아니라 Hermes Agent 쪽의 **알려진 버그**(2026-05-27 발생)였어요.   
지난 글에서 잡아둔 **OpenAI Codex OAuth(ChatGPT 구독) 프로바이더**를 쓸 때 발생합니다. 

연쇄적으로 보면 이런 흐름이에요. 

| 단계 | 일어나는 일 |
|------|------------|
| 1 | chatgpt.com Codex 백엔드가 `response.completed` 스트림 이벤트에서<br>`output = []` 가 아니라 `output = null(None)` 을 내려보냄 |
| 2 | OpenAI Python SDK는 `output` 이 **항상 iterable** 이라고 가정하고 순회 시도 |
| 3 | `TypeError: 'NoneType' object is not iterable` 로 크래시 |
| 4 | Hermes가 이 에러를 `Non-retryable` 로 분류 → 재시도 안 함 |
| 5 | 폴백 경로(`output_text` 접근)도 같은<br>`output=None` 을 건드려 또 크래시 |

폴백까지 같은 객체를 건드린다는 게 핵심이에요.   
일반적인 폴백 설계 같으면 다른 경로로 빠져야 하는데, 여기선 둘 다 같은 깨진 데이터를 보고 있어서 결과가 똑같이 나옵니다. 

> 💡 **영향 범위**는 메시지뿐이 아니에요. 에이전트가 돌리는 **모든 크론 작업**이 같은 코드패스를 타기 때문에, "매일 아침 7시" 같은 자동화도 같이 죽습니다. 

같은 날 비슷한 이슈가 트래커에 20개 넘게 올라온 걸 보면, 제 환경만의 문제는 아니었어요. 

<br>

<br>



## 3. 해결책 — 최신 버전으로 업데이트

다행히 수정 PR이 2026-05-27에 main에 머지됐어요. **`fix(agent): recover Codex Responses streams with null output`** 입니다. 

요지를 정리하면

- `codex_runtime.py` 의 스트림 순회 루프를 `try/except` 로 감싸서 null-output `TypeError` 를 잡음
- 미리 모아둔 `response.output_item.done` 항목 / 스트리밍 텍스트 델타로 **응답을 복구**
- 복구 가능한 이벤트가 전혀 없을 때에만 `create(stream=True)` 폴백 사용
- 메인 / 폴백 경로 양쪽에서 `final.output` 이 `None` 인 경우(빈 리스트뿐 아니라)도 backfill

즉, 최신 코드만 받으면 해결돼요.   
문제는 "어떻게 최신 코드를 받느냐"가 환경별로 다릅니다. 

<br>

<br>



## 4. 일반(quick installer) 설치라면

이쪽은 간단해요. 

```shell
hermes update
```

이 한 줄이면

- 최신 코드 pull
- 의존성 업데이트
- 실행 중인 게이트웨이 자동 재시작

까지 같이 처리합니다.   
재시작 동안 봇이 5~15초 잠깐 오프라인 됐다가 복귀해요. 

> ✅ 출력에 `Updating abc1234..def5678` 처럼 커밋 해시가 바뀌면 성공.   
> ✅ `Already up to date.` 면 이미 최신이라는 뜻이라 그대로 두면 돼요. 

<br>

<br>



## 5. Docker 환경이라면 — `hermes update` 는 정답이 아님

이게 이번 글에서 가장 강조하고 싶은 부분이에요.   
저는 Docker로 띄워두고 있어서 무심코 `docker exec` 로 `hermes update` 를 때리려고 했는데, 그게 함정이었습니다. 

### 5-1. 먼저 짚어야 할 두 가지

**컨테이너 이름이 `hermes-agent` 가 아니라 `hermes` 입니다.**   
지난 셋업 글에서 compose service 이름을 `hermes` 로 잡아둔 결과예요. 

```shell
docker ps --format '{{.Names}}'
# hermes
# n8n
# prefect-server
# ...
```

이걸 모르고 `docker exec -it hermes-agent ...` 를 치면 그대로 `No such container` 로 떨어집니다.   
이름 맞춰서 들어가도 — **컨테이너 안에서 돌리는 `hermes update` 는 Docker 환경에서 의미가 없어요**. 

이유는 단순합니다.   
컨테이너 안에서 `git pull` 로 코드를 받아둬도, 다음번에 컨테이너를 재생성하는 순간 그 변경분이 **이미지에 들어있는 원본으로 깨끗하게 덮여**버립니다.   
Docker에서는 코드의 출처가 "컨테이너 내부 작업 디렉토리" 가 아니라 **이미지** 거든요. 

### 5-2. 그래서 Docker는 "새 이미지 pull → 컨테이너 재생성"

다행히 데이터(설정, OAuth 세션, 스킬, 메모리)는 호스트의 `~/.hermes` → 컨테이너의 `/opt/data` 로 **바인드 마운트**돼 있고, 이미지는 stateless 입니다.   
그러니까 이미지를 갈아끼워도 설정/세션은 그대로 살아남아요. 

### 5-3. docker-compose 로 띄운 경우 (권장)

저는 이쪽이에요. compose 가 있는 디렉토리(`~/docker/hermes`)로 가서

```shell
cd ~/docker/hermes
docker compose pull hermes
docker compose up -d hermes
```

`pull` 로 최신 이미지를 받고, `up -d` 가 변경된 이미지로 컨테이너를 알아서 재생성합니다.   
볼륨/포트 옵션은 compose 파일이 들고 있으니 따로 지정할 필요 없어요. 

### 5-4. `docker run` 으로 직접 띄운 경우

compose 없이 `docker run` 으로 띄웠다면, 컨테이너를 직접 지우고 다시 만들어야 합니다.   
이때 **처음 띄울 때와 동일한 옵션**을 빠뜨리지 않는 게 핵심이에요. 

```shell
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -p 9119:9119 \
  nousresearch/hermes-agent gateway run
```

저는 게이트웨이 / 메트릭 포트로 `8642`, `9119` 를 열어두고 있어서 위에 그대로 반영했어요. 

> ⚠️ 원래 띄울 때 추가했던 `--env-file`, `--network data-pipeline-net` 같은 플래그가 있었다면 절대 빠뜨리면 안 됩니다.   
> 옵션이 누락된 채로 재생성하면 봇은 멀쩡히 떠도 정작 n8n / Prefect 와는 통신이 안 되는 상태가 돼요. 

원래 옵션이 기억나지 않으면 갈아끼우기 **전에** 현재 컨테이너 설정을 한 번 떠놓는 게 안전해요. 

```shell
docker inspect hermes > hermes.inspect.json
```

`Mounts`, `Ports`, `Env`, `HostConfig.NetworkMode` 정도만 확인해서 동일하게 맞춰주면 됩니다. 

<br>

<br>



## 6. 검증

갈아끼운 다음엔 로그를 켜놓고 한 번 흔들어봅니다. 

```shell
docker logs -f hermes
```

체크할 포인트는 세 개예요. 

- [ ] 정상 기동 로그 (`Gateway listening on :8642` 류) 가 뜨는지
- [ ] Discord에서 봇을 멘션했을 때 응답이 돌아오는지
- [ ] `'NoneType' object is not iterable` 에러가 더 이상 안 보이는지

저는 추가로 매일 아침 7시 크론 대상 작업을 수동으로 한 번 트리거해서, 결과 메시지가 Discord 채널에 도착하는지까지 확인했어요.   
크론처럼 사람이 안 보고 있는 경로일수록 **수동 trigger → 결과 도착 확인** 을 한 번 더 거쳐주는 게 마음 편합니다. 

<br>

<br>



## 7. 그래도 불안하면 — 프로바이더 교체

후속 리포트를 보면 같은 Codex OAuth 경로가 다른 케이스(서브에이전트 흐름 등)에서 또 다른 방식으로 불안정할 수 있다는 보고도 있어요.   
업데이트만으로 마음이 안 놓이거나, 크론이 가끔 또 말썽이면 **프로바이더 자체를 임시로 바꾸는 게 가장 확실한 우회책**이에요. 

- `hermes gateway setup` 을 다시 돌리거나 `~/.hermes/config.yaml` 에서 provider 변경
- 예: `gemini`, `anthropic`, 또는 일반 OpenAI API 키(Codex OAuth가 아닌 그냥 `OPENAI_API_KEY` 경로)
- 변경 후 게이트웨이(컨테이너) 한 번 재시작

```shell
docker compose restart hermes
```

ChatGPT 구독을 그대로 쓰는 매력은 잠깐 포기하지만, **자동화 파이프라인이 죽지 않는 게 우선**이라면 충분히 합리적인 선택이에요. 

<br>

<br>



## 8. 요약

| 항목 | 내용 |
|------|------|
| 증상 | Discord에서 `'NoneType' object is not iterable` 반복, 폴백도 동일하게 실패 |
| 원인 | Codex 백엔드가 `output=null` 을 내려보냄 → SDK 크래시 → Hermes가 `Non-retryable` 로 분류 (버그) |
| 해결 | 최신 이미지로 업데이트 (`fix(agent): recover Codex Responses streams with null output` 머지본) |
| Docker 핵심 | `hermes update` ❌ → **새 이미지 pull + 컨테이너 재생성** (데이터는 볼륨에 보존) |
| 주의 | 컨테이너 이름 `hermes`, 포트 `8642`/`9119`, 볼륨/네트워크 등 기존 옵션 유지 |
| 백업안 | 그래도 불안정하면 provider 를 codex 외(gemini / anthropic / openai API 키)로 교체 |

<br>

<br>



일단 오늘은 여기까지.....   
다음 글에서는 이런 자동화 봇이 또 조용히 죽었을 때 빠르게 눈치챌 수 있도록, **헬스체크 + Discord 알람** 을 어떻게 묶었는지 정리해볼게요.

---

**← 이전 글:** [(1/2) Hermes Agent Docker Compose 셋업](/coding/Hermes_Agent_Docker_셋업/)

