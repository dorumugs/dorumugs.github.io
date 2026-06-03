---
layout: single
title:  "(1/2) Hermes Agent Docker Compose 셋업"
date: 2026-05-26 22:44:00 +0900
description: "Docker Compose로 Hermes Agent를 띄우고 OpenAI Codex OAuth와 Discord 봇까지 연결하는 셋업 과정을 정리한 글이에요."
categories: coding
tag: [hermes, docker, docker-compose, llm, agent, n8n, prefect, codex, discord]
author_profile: false
toc: true
---



{% include series-hermes-agent.html current="1" %}

## Summary

집에 굴리는 미니 서버(`dorumugs@mini`, Linux)에 이미 **n8n**과 **Prefect**(server / db / worker)가
`data-pipeline-net`이라는 external 네트워크 위에서 돌고 있어요.   
여기에 **Hermes Agent**를 같은 네트워크에 묶어서 올리는 작업을 정리한 글이에요. 

LLM provider는 ChatGPT 구독을 그대로 쓸 수 있는 **OpenAI Codex OAuth**로 잡았습니다.  
한 번 셋업해두면, Hermes 안에서 `http://n8n:5678`, `http://prefect-server:4200` 같은 hostname 만으로
바로 호출할 수 있어서 자동화 파이프라인 잡기가 정말 편해져요. 

> 💡 **이 글에서 다루는 것**
> - Docker Compose로 Hermes Agent 띄우기
> - 호스트 바인드 마운트로 데이터 영구 저장
> - Codex OAuth로 ChatGPT 구독 연결
> - Discord 봇 연결 + 자주 만나는 `Unauthorized user` 이슈 해결
> - n8n / Prefect 네트워크 연결 검증

<br>

<br>



## 1. 사전 확인

### Docker 네트워크 존재 여부

먼저 같이 묶을 네트워크가 있는지 확인합니다. 

```shell
docker network ls | grep data-pipeline-net
# f7003838c0c4   data-pipeline-net   bridge    local
```

만약 없다면, 새로 만들어주세요. 

```shell
docker network create data-pipeline-net
```

이미 같은 네트워크에 어떤 친구들이 붙어있는지 궁금하면 inspect로 확인할 수 있어요. (선택)

```shell
docker network inspect data-pipeline-net \
  --format '{{range .Containers}}{{.Name}}{{"\n"}}{{end}}'
```

<br>

<br>



## 2. 디렉토리 준비

compose 파일이 살 곳과, Hermes 데이터가 살 곳을 분리합니다.  

```shell
mkdir -p ~/docker/hermes && cd ~/docker/hermes
mkdir -p ~/.hermes
```

| 경로 | 용도 |
|------|------|
| `~/docker/hermes` | docker-compose.yaml 위치 |
| `~/.hermes` | Hermes 데이터(config, auth, sessions, skills, memories) 영구 저장소 |

`~/.hermes` 는 컨테이너 안의 `/opt/data` 와 **바인드 마운트로 1:1 매핑** 됩니다.   
컨테이너를 다시 띄워도, 업그레이드 해도, 데이터는 호스트에 그대로 남아있어요. 

<br>

<br>



## 3. docker-compose.yaml 작성

파일 경로 : `~/docker/hermes/docker-compose.yaml`

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # OpenAI 호환 API 서버
      - "9119:9119"   # 웹 대시보드
    environment:
      - HERMES_HOME=/opt/data
      - HERMES_DASHBOARD=1
      - TZ=Asia/Seoul
      - GENERIC_TIMEZONE=Asia/Seoul
    volumes:
      - ~/.hermes:/opt/data
      - /home/dorumugs/docker/shared:/shared
    shm_size: 1gb
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
    networks:
      - data-pipeline-net

networks:
  data-pipeline-net:
    external: true
```

### 핵심 포인트

- **`~/.hermes:/opt/data` 호스트 바인드 마운트**  
  named volume 대신 호스트 디렉토리를 그대로 마운트합니다.   
  덕분에 뒤에서 따로 돌리는 `setup` 마법사 결과가 그대로 compose 컨테이너에 이어집니다.
- **`data-pipeline-net` external 네트워크**  
  n8n, Prefect 와 hostname으로 직접 통신할 수 있어요.  
  (`http://n8n:5678`, `http://prefect-server:4200`)
- **`shm_size: 1gb`**  
  Hermes가 Playwright/Chromium 을 띄울 때 필요한 공유 메모리예요. 깜빡하면 브라우저 도구가 죽습니다.
- **`HERMES_HOME=/opt/data`**  
  명시 안 하면 게이트웨이가 가끔 `auth.json` 위치를 헷갈리는 이슈가 있어서 박아둡니다.
- **`TZ=Asia/Seoul`**  
  로그/크론/일일 리셋 시각이 전부 한국 시간 기준으로 찍혀요.

<br>

<br>



## 4. 초기 setup (Codex OAuth) — 딱 한 번만

`compose up` 으로 띄우기 전에 **인터랙티브하게 setup 마법사**를 먼저 돌립니다.   
어차피 호스트 바인드 마운트(`~/.hermes`)에 그대로 쓰기 때문에, 이후 compose가 동일 데이터를
이어받게 돼요. 

```shell
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e TZ=Asia/Seoul \
  nousresearch/hermes-agent setup
```

setup 마법사에서 제가 고른 값들은 다음과 같아요. 

| 항목 | 선택값 | 비고 |
|------|--------|------|
| Provider | **OpenAI Codex (ChatGPT account)** | device code flow — 호스트 브라우저로 ChatGPT 로그인 후 코드 입력 |
| Model | gpt-5.5 | |
| TTS provider | Edge TTS | 기본값. 무료, 설정 불필요 |
| Terminal backend | local | Hermes 자체가 컨테이너라 굳이 더 격리 안 함 |
| Max iterations | 90 | 기본값. Codex OAuth는 ChatGPT 구독 rate limit 영향이 커서 무리하게 안 올림 |
| Tool progress | all | 기본값 |
| Compression threshold | 0.5 | 토큰 비용 절약 + rate limit 보호 |
| Session reset | Inactivity + daily reset | |
| Inactivity timeout | 1440 분 (24시간) | |
| Daily reset hour | 4 | 한국 시간 새벽 4시 |
| Messaging | Discord 활성화 | allowed users : `kayser_so` (이후 5번에서 ID로 교체) |
| Home channel ID | 빈 채로 진행 | 나중에 `/set-home` 으로 지정 |

> ✅ Codex 토큰은 `~/.hermes/auth.json` 에 저장돼요.  
> 이 파일은 곧 본인의 ChatGPT 계정과 직결되니까, **절대 외부 공유 금지**입니다. 

<br>

<br>



## 5. compose 기동

setup이 끝나면 이제 진짜로 compose로 띄웁니다. 

```shell
cd ~/docker/hermes
docker compose up -d
docker compose logs -f
```

정상적으로 올라오면 로그가 이런 식으로 흘러요. 

```text
hermes  | Dropping root privileges
hermes  | Syncing bundled skills into ~/.hermes/skills/ ...
hermes  | Done: 0 new, 0 updated, 87 unchanged. 87 total bundled.
hermes  | Starting hermes dashboard on 0.0.0.0:9119 (background)
hermes  | [dashboard]   Hermes Web UI → http://0.0.0.0:9119
hermes  | ⚕ Hermes Gateway Starting...
hermes  | [Discord] Resolving 1 username(s): kayser_so
hermes  | [Discord] Resolved 'kayser_so' -> 1196463921129865309 (kayser_so#0)
hermes  | [Discord] Updated DISCORD_ALLOWED_USERS with 1 resolved ID(s)
```

여기까지 나오면 일단 게이트웨이는 살아있는 상태예요.   
이제 대시보드는 `http://<호스트IP>:9119`, OpenAI 호환 API는 `8642` 포트로 들어옵니다. 

<br>

<br>



## 6. 상태 확인

도커 컨테이너 안에서 `hermes` 바이너리는 PATH에 노출되어 있지 않아요.   
풀 경로(`/opt/hermes/.venv/bin/hermes`) 또는 hermes 유저 셸로 호출해야 합니다. 

```shell
docker compose exec hermes /opt/hermes/.venv/bin/hermes status
```

정상 상태라면 이런 출력이 떠요. 

```text
Project:      /opt/hermes
Python:       3.13.5
.env file:    ✓ exists
Model:        gpt-5.5
Provider:     OpenAI Codex
OpenAI Codex  ✓ logged in
  Auth file:  /opt/data/auth.json
  Refreshed:  2026-05-24 22:26:26 KST
Terminal Backend:  local
Discord:           ✓ configured
Gateway Service:   ✓ running (PID 7)
```

### 편의용 alias (선택)

매번 풀 경로 치기 귀찮으니 alias 하나 박아두는 것을 추천드려요. 

```shell
echo "alias hermesx='docker compose -f ~/docker/hermes/docker-compose.yaml exec hermes /opt/hermes/.venv/bin/hermes'" >> ~/.bashrc
source ~/.bashrc

# 이후엔 짧게
hermesx status
```

<br>

<br>



## 7. 네트워크 연결 검증 (n8n / Prefect)

같은 네트워크에 묶었으니, 컨테이너 안에서 hostname으로 바로 닿는지 확인해봅니다. 

```shell
docker compose exec hermes curl -s http://n8n:5678/healthz
# {"status":"ok"}

docker compose exec hermes curl -s http://prefect-server:4200/api/health
# true
```

두 서비스 모두 hostname 기반으로 잘 닿는 걸 확인했어요.   
이제 Hermes 안에서 `http://n8n:5678/webhook/...` 같은 URL을 마음껏 쓸 수 있는 상태가 됐습니다. 

<br>

<br>



## 8. 문제 발생 — Discord 봇이 응답을 안 함

여기서 살짝 막혔습니다.   

**증상**

- 봇이 채널에 들어오면서 첫 환영 메시지("안녕! 저는 hermes에요... `/help` 로 볼 수 있습니다.")는 잘 옵니다.
- 그런데 그 이후 일반 메시지에는 **typing 표시만 뜨고 응답이 없어요**.

로그를 보니 원인이 친절하게 찍혀있었어요. 

```text
[Discord] Resolved 'kayser_so' -> 119646*******865309 (kayser_so#0)
[Discord] Updated DISCORD_ALLOWED_USERS with 1 resolved ID(s)
WARNING gateway.run: Unauthorized user: 119646*******865309 (dorumugs) on discord
WARNING gateway.run: Unauthorized user: 119646*******865309 (dorumugs) on discord
...
```

> ⚠️ 분석  
> - username(`kayser_so`) → ID(`119646*******865309`) 변환은 **성공**.  
> - allowed list에 ID도 분명히 추가됨.  
> - 그런데 **같은 ID로 들어오는 메시지가 Unauthorized로 거부**되고 있음.  
> - 변환된 in-memory 리스트와 실제 인증 체크 사이에서 매칭이 어긋나는 걸로 보임.

해결 방향은 단순합니다.   
**username 의존을 빼고, `.env`에 ID를 직접 박아서 변환 단계 자체를 제거**해버리는 거예요. 

<br>

<br>



## 9. 해결 — `DISCORD_ALLOWED_USERS` 에 ID 직접 지정

먼저 `config.yaml` 의 `discord:` 섹션을 봤는데 `allowed_users` 항목이 없었어요.   
설정이 `.env` 쪽에 들어가 있는 걸 확인합니다. 

```shell
docker compose exec hermes grep DISCORD_ALLOWED /opt/data/.env
# DISCORD_ALLOWED_USERS=kayser_so
```

이 값을 ID로 갈아끼웁니다. 호스트에서 직접 수정해도 되고, 

```shell
sudo sed -i \
  's/DISCORD_ALLOWED_USERS=kayser_so/DISCORD_ALLOWED_USERS=119646*******865309/' \
  ~/.hermes/.env
```

호스트 권한이 막혀있으면 컨테이너 안에서 실행해도 돼요. 

```shell
docker compose exec hermes sed -i \
  's/DISCORD_ALLOWED_USERS=kayser_so/DISCORD_ALLOWED_USERS=119646*******865309/' \
  /opt/data/.env
```

값이 바뀐 걸 확인하고, 

```shell
docker compose exec hermes grep DISCORD_ALLOWED /opt/data/.env
# DISCORD_ALLOWED_USERS=119646*******865309
```

재시작합니다. 

```shell
cd ~/docker/hermes
docker compose restart
docker compose logs -f
```

### 검증 포인트

- [x] `[Discord] Resolving 1 username(s):` 줄이 **더 이상 안 나옴** (이미 ID라 변환 단계 스킵)
- [x] Discord 메시지 송신 시 `Unauthorized user` 경고 **없음**
- [x] 봇이 정상적으로 응답함 🎉

<br>

<br>



## 10. Discord 봇 운용 메모

운영하면서 챙겨야 할 자잘한 것들을 같이 적어둡니다. 

- **`require_mention: true`** (`config.yaml` 의 discord 섹션 기본값)  
  → 서버 채널에서는 `@봇이름` 멘션이 필요합니다. **DM에서는 멘션 불필요**.
- **Home channel 지정**  
  봇이 들어있는 채널에서 슬래시 커맨드 `/set-home` 을 입력하면 그 채널이 홈 채널이 됩니다.   
  슬래시 커맨드 자체가 안 뜬다면, 봇 초대 시 scopes에 `applications.commands` 가 빠진 거예요.   
  Developer Portal → OAuth2 → URL Generator 에서 `bot` + `applications.commands` 로 **재초대**해주세요.
- **토큰 만료 시 갱신**

```shell
docker compose exec hermes \
  /opt/hermes/.venv/bin/hermes auth add openai-codex --type oauth
```

<br>

<br>



## 11. 자주 쓰는 관리 명령

손에 자주 닿는 명령어들 한 곳에 모아둡니다. 

```shell
cd ~/docker/hermes

docker compose ps                            # 상태
docker compose logs -f hermes                # 실시간 로그
docker compose restart                       # 재시작
docker compose down                          # 정지 (데이터 보존)
docker compose pull && docker compose up -d  # 업그레이드
```

컨테이너 내부로 들어가고 싶을 때 

```shell
docker compose exec -u hermes hermes bash
```

hermes CLI를 풀 경로로 부르고 싶을 때 

```shell
docker compose exec hermes /opt/hermes/.venv/bin/hermes status
docker compose exec hermes /opt/hermes/.venv/bin/hermes config check
docker compose exec hermes /opt/hermes/.venv/bin/hermes auth list
```

<br>

<br>



## 12. 향후 확장 아이디어

- **Hermes ↔ n8n 연동**  
  워크플로우 webhook 트리거. `http://n8n:5678/webhook/...` 를 Hermes 스킬에서 바로 호출.
- **Hermes ↔ Prefect 연동**  
  deployment trigger, flow run 상태 모니터링 (`http://prefect-server:4200/api`).
- **Telegram 게이트웨이 추가**  
  `setup` 다시 돌리거나, `.env` 에 토큰 추가 후 재시작.
- **보조(aux) 모델 분리**  
  제목 생성, 압축 같은 가벼운 작업은 별도 provider로 빼서 ChatGPT 구독 rate limit을 아껴줍니다.  
  (`config.yaml` 의 `auxiliary:` 섹션)

<br>

<br>



## 13. 주의 사항

마지막으로 보안 관련해서 꼭 챙겨야 할 부분들이에요.   

> 🚨 **Codex OAuth = 본인 ChatGPT 계정**.  
> 이 토큰이 새면 곧 본인 계정이 새는 거라고 봐야 합니다.   
> `DISCORD_ALLOWED_USERS` 를 비우거나 `GATEWAY_ALLOW_ALL_USERS=true` 같은 설정은 **절대 금지**.

- **8642 포트를 외부망에 노출**할 거면 `API_SERVER_KEY` 반드시 설정해두세요.
- 같은 `~/.hermes` 데이터 디렉토리에 **게이트웨이 컨테이너를 2개 이상 띄우지 말 것**.  
  세션 / 메모리 파일 동시 쓰기는 지원하지 않아서, 파일이 깨질 수 있어요.

<br>

<br>



일단 오늘은 여기까지.....   
다음 글에서는 Hermes에서 n8n webhook 을 직접 호출해서 workflow 트리거 거는 부분을 정리해볼게요.

---

**다음 글 →** [(2/2) Hermes Agent + Discord 'NoneType' 트러블슈팅](/coding/Hermes_Agent_Discord_NoneType_트러블슈팅/)

