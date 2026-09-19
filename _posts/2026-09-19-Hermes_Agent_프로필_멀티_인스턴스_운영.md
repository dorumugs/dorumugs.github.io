---
layout: single
title:  "(3/3) Hermes Agent 프로필 — 에이전트 한 대를 역할별로 쪼개 쓰기"
date: 2026-09-19 21:30:00 +0900
description: "hermes profile 로 완전히 격리된 Hermes 인스턴스를 여러 개 만들고, -p·래퍼·sticky 로 골라 쓰고, 게이트웨이 하나로 묶고, export 로 다른 머신에 옮기는 실전 사용법을 정리했어요."
categories: coding
tag: [hermes, agent, llm, profile, docker, cli, gateway, kanban, devops]
author_profile: false
toc: true
header:
  image: /assets/images/2026-09-19-hermes-profile/header.svg
  teaser: /assets/images/2026-09-19-hermes-profile/header.svg
series: hermes-agent
series_order: 3
---



{% include series-hermes-agent.html current="3" %}

## Summary

[1편](/coding/Hermes_Agent_Docker_셋업/)에서 Docker Compose로 Hermes Agent를 띄우고,
[2편](/coding/Hermes_Agent_Discord_NoneType_트러블슈팅/)에서 Discord 연동 버그를 잡았어요.
그 뒤로 몇 달을 굴리면서 한 가지가 계속 걸렸습니다.

**인스턴스가 하나뿐이라 모든 게 한 통에 섞인다**는 점이에요.
블로그 초안을 쓰던 세션 기억이 부동산 데이터 작업에 끼어들고,
실험용으로 켠 스킬이 운영 크론에까지 붙고,
API 키 하나를 바꾸면 무관한 작업까지 같이 흔들렸습니다.

`hermes profile` 이 정확히 이 문제를 풉니다.
**완전히 격리된 Hermes 인스턴스를 여러 개** 만들어서, 설정·키·기억·세션·스킬·크론을 각각 따로 두는 기능이에요.

> 💡 **이 글에서 다루는 것**
> - 프로필이 실제로 무엇을 가르고 무엇을 공유하는지
> - 프로필 만들기와 복제(`--clone` 계열) 옵션 차이
> - 프로필을 지목하는 세 가지 방법과 각각의 함정
> - 게이트웨이를 프로필마다 띄울지, 한 대로 묶을지
> - 역할 설명(`describe`)과 칸반 자동 라우팅
> - `export`/`import`/`install` 로 다른 머신에 옮기기
> - 이름 변경·삭제 시 남는 찌꺼기 정리

글에 나오는 출력은 전부 **Hermes Agent v0.21.3 (2026.9.14)** 컨테이너에서 실제로 실행한 결과입니다.
확인이 끝난 테스트 프로필은 전부 삭제했어요.

<br>

<br>



## 1. 프로필이 실제로 가르는 것

먼저 지금 상태부터 봅니다.

```shell
hermes profile list
```

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆default         gpt-5.5                      running      —            —
```

`◆` 가 지금 활성 프로필입니다. 아직 `default` 하나뿐이에요.

### 어디에 사는가

프로필은 **디렉토리 하나**입니다. 그게 전부예요.

| 프로필 | 경로 |
|---|---|
| `default` | `HERMES_HOME` 루트 그대로<br>(Docker면 `/opt/data`,<br>맨몸 설치면 `~/.hermes`) |
| 이름 있는<br>프로필 | `<루트>/profiles/<이름>` |

여기서 **`default` 가 특별하다**는 점을 먼저 짚고 갑니다.
`default` 는 `profiles/` 밑에 있지 않고 루트 그 자체예요.
그래서 삭제할 수 없고, 이름을 바꿔도 내부 식별자는 계속 `default` 로 남습니다.

### 프로필 안에 뭐가 들어가나

새로 만든 프로필 디렉토리는 이렇게 생겼습니다.

```shell
ls -A /opt/data/profiles/blogdemo
```

```
.env
SOUL.md
audio_cache
backups
config.yaml
cron
home
hooks
image_cache
logs
memories
pairing
plans
profile.yaml
sessions
skills
skins
workspace
```

| 항목 | 내용 | 프로필마다 따로? |
|---|---|---|
| `config.yaml` | 모델·프로바이더<br>보안·폴백 설정 | 따로 |
| `.env` | API 키·봇 토큰 | 따로 |
| `SOUL.md` | 성격·역할<br>지시문 | 따로 |
| `memories/` | 장기 기억<br>(`MEMORY.md`,<br>`USER.md`) | 따로 |
| `sessions/` | 대화 기록 | 따로 |
| `skills/` | 설치된 스킬<br>(신규 생성 시<br>58개 동기화) | 따로 |
| `cron/` | 예약 작업 | 따로 |
| `workspace/` | 작업 디렉토리 | 따로 |
| `profile.yaml` | 역할 설명(`description`) | 따로 |
| `auth.json` | Nous 계정 등 풀링된 자격증명 | **루트에서 공유** |
| `kanban.db` | 칸반 보드 | **루트에서 공유** |

정리하면 — **"이 에이전트가 누구인가"에 해당하는 건 전부 갈라지고, 계정·보드처럼 기계 단위인 건 공유**됩니다.

<br>

<br>



## 2. 첫 프로필 만들기

```shell
hermes profile create blogdemo \
  --description "블로그 초안과 자료 조사 담당"
```

```
Profile 'blogdemo' created at /opt/data/profiles/blogdemo
58 bundled skills synced.
Wrapper created: /opt/data/.local/bin/blogdemo

Next steps:
  blogdemo setup              Configure API keys and model
  blogdemo chat               Start chatting
  hermes gateway restart    Serve this profile from the running multiplexed gateway

  ⚠ This profile has no API keys yet. Run 'blogdemo setup' first,
    or it will inherit keys from your shell environment.
```

한 줄로 세 가지가 같이 일어났어요.

| 일어난 일 | 결과 |
|---|---|
| 디렉토리 생성 | `profiles/blogdemo` 와 하위 구조 |
| 스킬 동기화 | 번들 스킬 58개 복사 |
| 래퍼 스크립트 | `~/.local/bin/blogdemo` 생성 |

만들어진 결과는 `show` 로 한 번에 확인합니다.

```shell
hermes profile show blogdemo
```

```
Profile: blogdemo
Path:    /opt/data/profiles/blogdemo
Model:   gpt-5.5 (openai-codex)
Gateway: stopped
Skills:  58
.env:    exists
SOUL.md: exists
Alias:   blogdemo → hermes -p blogdemo  (/opt/data/.local/bin/blogdemo)
```

### `--description` 은 지금 주는 게 좋습니다

`--description` 은 단순한 메모가 아니라 **칸반 오케스트레이터가 작업을 배분할 때 읽는 값**입니다.
6절에서 자세히 다루는데, 나중에 붙이는 것보다 만들 때 같이 주는 편이 손이 덜 갑니다.

실제로 `profile.yaml` 에 이렇게 들어갑니다.

```shell
cat /opt/data/profiles/blogdemo/profile.yaml
```

```
description: 블로그 초안과 자료 조사 담당
description_auto: false
```

### 키 경고를 흘려듣지 마세요

출력 마지막의 경고가 중요합니다.
프로필의 `.env` 가 비어 있으면 **셸 환경변수의 키를 그대로 물려받습니다**.
"격리했으니 안전하겠지" 하고 방치하면, 실험용 프로필이 운영 키로 API를 때립니다.
새 프로필은 만들자마자 `.env` 를 채우거나 `<프로필> setup` 을 한 번 돌려주세요.

> ⚠️ `.env` 에 키를 추가할 때는 **항상 이어쓰기(`>>`)** 로 하세요.
> 덮어쓰면 그 프로필이 들고 있던 다른 키가 통째로 날아가고, 복구 경로가 사실상 없습니다.

<br>

<br>



## 3. 프로필을 지목하는 세 가지 방법

만들었으면 골라 써야죠. 방법이 세 개고, 셋 다 성격이 다릅니다.

| 방법 | 형태 | 유효 범위 | 쓰기 좋은 곳 |
|---|---|---|---|
| `-p`<br>`--profile` | `hermes -p`<br>`blogdemo chat` | 그 명령<br>한 번 | 스크립트<br>크론·CI |
| 래퍼<br>스크립트 | `blogdemo chat` | 그 명령<br>한 번 | 손으로<br>칠 때 |
| sticky<br>기본값 | `hermes profile`<br>`use blogdemo` | 바꿀 때까지<br>계속 | 한동안 한<br>프로필만<br>팔 때 |

### 함정 1 — `-p` 는 어디 붙여도 먹지만, 앞에 붙이세요

`-p` 는 argparse 가 돌기 **전에** 걷어내는 플래그입니다.
argv 전체를 훑기 때문에 서브커맨드 앞이든 뒤든 둘 다 동작해요.

```shell
hermes -p blogdemo gateway list
hermes gateway list -p blogdemo
```

둘 다 같은 결과가 나옵니다. 그런데도 **앞에 붙이는 걸 습관으로 만드는 게 낫습니다.**
틀린 값을 줬을 때 나오는 메시지가 위치에 따라 달라지거든요.

| 준 값 | 서브커맨드 **앞** | 서브커맨드 **뒤** |
|---|---|---|
| 없는 프로필<br>`nosuch` | 없다고<br>알려줌 | 같음 |
| 규칙 위반<br>`"work bot"` | 이름 규칙을<br>설명하고<br>대안 제시 | argparse<br>usage 덤프 |

앞에 붙이면 이렇게 나옵니다.

```shell
hermes -p "work bot" gateway list
```

```
Error: 'work bot' is not a valid profile name. Use lowercase letters,
numbers, '-' or '_', starting with a letter or number, up to 64 characters
(for example: work-bot). Then run `hermes profile create work-bot`.
Run `hermes profile list` to see your profiles.
```

뒤에 붙이면 같은 실수가 최상위 `usage:` 벽으로 바뀌어서, 원인을 찾는 데 시간이 더 듭니다.

### 함정 1-b — 서브커맨드가 자기 `-p` 를 가질 때

프로필 이름 규칙(`^[a-z0-9][a-z0-9_-]{0,63}$`)에 맞는 값은
**어디에 있든 프로필 플래그가 먼저 집어갑니다.**
포트 번호처럼 이름 규칙에 우연히 맞는 값이 제일 위험해요.

```shell
hermes gateway list -p 8080
```

```
Error: Profile '8080' does not exist. Create it with: hermes profile create 8080
```

서브커맨드에 `-p` 를 넘기려던 건데 프로필 플래그가 가로챈 겁니다.
이럴 땐 `--` 로 경계를 긋거나, 애초에 프로필은 앞에서 `-p` 로 지정해 두세요.

또 하나, 값은 **소문자로 정규화**됩니다. `-p NoSuch` 는 `nosuch` 를 찾습니다.

### 함정 2 — 래퍼는 PATH 에 있어야 합니다

`create` 가 만들어 주는 래퍼는 이렇게 생긴 두 줄짜리 스크립트예요.

```shell
cat /opt/data/.local/bin/blogdemo
```

```
#!/bin/sh
exec /opt/hermes/bin/hermes -p blogdemo "$@"
```

생성 위치는 항상 `~/.local/bin` 입니다.
이 디렉토리가 `PATH` 에 없으면 `blogdemo: command not found` 가 나요.
`create` 가 "Wrapper created" 라고 말해줘도 **PATH 는 대신 고쳐주지 않습니다.**

이름을 따로 주고 싶으면 `--name` 을 씁니다.

```shell
hermes profile alias blogdemo2 --name bd2
```

```
✓ Alias created: /opt/data/.local/bin/bd2
```

래퍼를 아예 안 만들려면 생성할 때 `--no-alias` 를 주면 됩니다.

### 함정 3 — sticky 는 전역 상태입니다

```shell
hermes profile use blogdemo
```

```
Switched to: blogdemo
```

이건 셸 세션 단위가 아니라 **루트에 파일 하나를 쓰는 전역 설정**입니다.

```shell
cat /opt/data/active_profile
```

```
blogdemo
```

그래서 다른 터미널, 크론, 게이트웨이가 전부 같이 영향을 받아요.
되돌릴 때는 `default` 로 되돌리면 되고, 그러면 파일 자체가 지워집니다.

```shell
hermes profile use default
```

```
Switched to: default (~/.hermes)
```

바꿔 놓은 걸 잊기 쉬우니, **자동화에는 sticky 대신 `-p` 를 쓰는 편**을 권합니다.
sticky 는 "오늘 하루 이 프로필만 판다" 같은 상황에만요.

<br>

<br>



## 4. 복제 — 어디까지 따라오는가

비슷한 프로필을 또 만들 때는 처음부터 세팅하지 말고 복제합니다.
그런데 옵션이 다섯 갈래라 헷갈려요. 뭐가 따라오는지 정리하면 이렇습니다.

| 옵션 | 따라오는 것 | 따라오지 않는 것 |
|---|---|---|
| (없음) | 번들 스킬만 | 나머지 전부 |
| `--clone` | `config.yaml`<br>`.env`<br>`SOUL.md`<br>`MEMORY.md`<br>`USER.md`<br>스킬 | 세션<br>크론<br>메신저 채널 |
| `--clone-all` | 활성 프로필<br>전체 상태 | 세션 DB<br>백업<br>체크포인트<br>크론<br>메신저 채널 |
| `--clone-from X` | `X` 를 원본으로<br>`--clone` 수행 | 위와 동일 |
| `--clone-channels` | 원본의 봇 토큰<br>허용목록<br>플랫폼 설정까지 | — |

복제원은 기본적으로 **활성 프로필**입니다. `--clone-from` 을 주면 그 프로필이 원본이 돼요.

```shell
hermes profile create blogdemo2 --clone-from blogdemo
```

```
Profile 'blogdemo2' created at /opt/data/profiles/blogdemo2
Cloned config, .env, SOUL.md, and skills from blogdemo.
Wrapper created: /opt/data/.local/bin/blogdemo2
```

출력 요약은 "config, .env, SOUL.md, and skills" 네 가지만 말하는데,
실제로는 `memories/MEMORY.md` 와 `memories/USER.md` 도 같이 복사됩니다.
에이전트의 정체성에 해당하는 파일이라 `SOUL.md` 와 한 묶음으로 취급하거든요.
**요약 줄만 믿고 "기억은 안 따라왔겠지" 하면 안 됩니다.**

### 실측 — `description` 은 복제되지 않습니다

반대로, 요약에 없으면서 실제로도 안 따라오는 게 있어요. `profile.yaml` 입니다.

```shell
cat /opt/data/profiles/blogdemo2/profile.yaml
```

```
cat: /opt/data/profiles/blogdemo2/profile.yaml: No such file or directory
```

복제로 만든 프로필은 **역할 설명이 비어 있는 상태로 시작**합니다.
칸반 라우팅을 쓸 생각이라면 복제 직후에 `describe` 로 채워 넣어야 해요.

### `--clone-channels` 는 기본적으로 쓰지 마세요

메신저 채널까지 복제한다는 건 **같은 봇 토큰을 두 프로필이 들고 있게** 된다는 뜻입니다.
둘 다 게이트웨이에 붙으면 같은 메시지를 두 번 처리하거나, 한쪽이 연결을 뺏깁니다.
Hermes 도 이걸 알아서, 원본이 살아 있는 멀티플렉스 게이트웨이에 물려 있으면 아예 거부합니다.

봇을 진짜로 하나 더 두고 싶다면, 복제가 아니라 **새 봇 토큰을 발급받아 새 프로필에 넣는 게** 맞습니다.

### 크론이 따라오지 않는 이유

`--clone-all` 조차 `cron/` 은 일부러 비웁니다.
복제본이 크론 잡을 물려받으면 게이트웨이 두 대가 **같은 잡을 각자 실행**해서,
비용이 두 배로 나가고 메시지도 두 번 갑니다. 의도된 제외예요.

<br>

<br>



## 5. 게이트웨이 — 프로필마다 한 대 vs 한 대로 통합

여기가 실전에서 제일 많이 막히는 지점입니다.

프로필은 각자 게이트웨이를 띄울 수 있고, 반대로 **기본 프로필의 게이트웨이 한 대가 전부를 서빙**할 수도 있어요.
후자를 멀티플렉스 모드라고 부릅니다.

| 모드 | 설정 | 특징 |
|---|---|---|
| 프로필별<br>독립 | `gateway.`<br>`multiplex_profiles`<br>= `false` | 프로세스가<br>프로필 수만큼.<br>격리는 강하지만<br>메모리와 포트를<br>각자 먹음 |
| 멀티플렉스 | `gateway.`<br>`multiplex_profiles`<br>= `true` | 기본 프로필<br>게이트웨이 한 대가<br>전부 서빙.<br>프로세스 하나 |

지금 설정을 확인합니다.

```shell
hermes config get gateway.multiplex_profiles
```

```
true
```

### 함정 — 프로필을 만들어도 게이트웨이는 자동으로 안 붙습니다

멀티플렉스가 켜져 있는데도, 방금 만든 프로필은 이렇게 나옵니다.

```shell
hermes gateway list
```

```
Gateways:
  ✓ default (current)        — PID 158
  ✗ blogdemo                 — not running
```

게이트웨이는 **뜰 때 한 번** 서빙할 프로필 목록을 읽습니다.
그 뒤에 생긴 프로필은 다음 재시작 전까지 인식되지 않아요.
`create` 출력이 "Serve this profile from the running multiplexed gateway" 라고 알려주는 게 이것 때문입니다.

```shell
hermes gateway restart
```

"프로필을 만들었는데 Discord 가 응답을 안 한다"의 대부분이 이겁니다.
독립 모드에서도 마찬가지로, 새 프로필은 스스로 게이트웨이를 띄우지 않습니다.

### 독립 → 멀티플렉스로 옮기기

이미 프로필마다 게이트웨이를 띄워 놨다면 전용 마이그레이션 명령이 있어요.
**먼저 `--dry-run` 으로 계획만 뽑아 보세요.**

```shell
hermes gateway migrate --multiplex --dry-run
```

이 명령은 프리플라이트를 먼저 돌립니다.
**중복 봇 토큰**이 있거나, **포트를 직접 점유하는 플랫폼에 `/p/<프로필>/` 인그레스가 없으면** 아무것도 바꾸지 않고 막아요.
되돌릴 때는 `--standalone` 을 쓰면 기록해 둔 매니페스트로 복원됩니다.

<br>

<br>



## 6. 역할 설명과 칸반 자동 배분

`hermes kanban` 은 여러 프로필이 공유하는 SQLite 작업 보드입니다.
작업을 쪼개서 프로필에 나눠줄 때, 디스패처는 **프로필 이름이 아니라 설명을 읽습니다.**

`blog-writer` 라는 이름만 보고 "글쓰기 작업을 주자"고 판단하지 않아요.
`description` 이 비어 있으면 라우팅 근거가 없습니다.

읽기와 쓰기 모두 `describe` 로 합니다.

```shell
hermes profile describe blogdemo
```

```
블로그 초안과 자료 조사 담당
```

```shell
hermes profile describe blogdemo \
  --text "부동산 수집 스크립트 유지보수와 회귀 테스트"
```

프로필이 많아져서 손으로 쓰기 귀찮다면 보조 LLM에 맡길 수도 있습니다.

| 옵션 | 동작 |
|---|---|
| `--auto` | 보조 LLM이 프로필 내용을 읽고 설명 생성 |
| `--all` | 설명이 비어 있는 모든 프로필을 한 번에 |
| `--overwrite` | 사람이 직접 쓴 설명까지 덮어씀 |

`--auto` 는 API 호출이 실제로 발생하니 프로필이 많을 때는 비용을 한 번 생각하고 돌리세요.
기본값은 **비어 있거나 이전에 자동 생성된 것만** 채웁니다 — 손으로 쓴 설명은 건드리지 않아요.

<br>

<br>



## 7. 다른 머신으로 옮기기

### 압축해서 들고 가기

```shell
hermes profile export blogdemo
```

```
✓ Exported 'blogdemo' to /opt/data/profile-exports/blogdemo-20260919-213455.tar.gz
```

아카이브는 프로필 이름 디렉토리를 최상위로 갖는 평범한 tarball 입니다.

```shell
tar -tzf /opt/data/profile-exports/blogdemo-20260919-213455.tar.gz | head
```

```
blogdemo/
blogdemo/SOUL.md
blogdemo/audio_cache/
blogdemo/backups/
blogdemo/backups/config/
blogdemo/backups/config/config.yaml.good.20260919-213319
blogdemo/backups/config/config.yaml.good.20260919-213344
blogdemo/config.yaml
blogdemo/cron/
blogdemo/home/
```

복원은 `import` 이고, `--name` 으로 다른 이름을 붙일 수 있습니다.

```shell
hermes profile import \
  /opt/data/profile-exports/blogdemo-20260919-213455.tar.gz \
  --name blogrestore
```

```
✓ Imported profile 'blogrestore' at /opt/data/profiles/blogrestore
  Wrapper created: /opt/data/.local/bin/blogrestore
```

> ⚠️ 아카이브에는 **`.env` 가 그대로 들어갑니다.** API 키와 봇 토큰이 평문으로 담긴다는 뜻이에요.
> 아카이브를 Git 저장소나 공유 드라이브에 올리지 마세요.
> 백업을 자동화할 거면 저장 위치를 먼저 정하고 시작하는 게 안전합니다.

### 배포 가능한 프로필 — distribution

팀에 같은 세팅을 나눠줄 거라면 `export`/`import` 대신 **distribution** 이 낫습니다.
Git 저장소 루트에 `distribution.yaml` 을 두면 그 자체가 설치 가능한 프로필이 돼요.

```shell
hermes profile install github.com/myteam/hermes-reviewer --alias
```

매니페스트에 들어가는 필드는 이렇습니다.

| 필드 | 용도 |
|---|---|
| `name` | 프로필 이름 (필수) |
| `version` | 배포 버전 |
| `description` | 역할 설명 |
| `hermes_requires` | 최소 Hermes 버전.<br>안 맞으면 설치 단계에서<br>즉시 실패 |
| `env_requires` | 필요한 환경변수 목록.<br>`.env.template` 생성에 쓰임 |
| `distribution_owned` | 배포본이 소유하는<br>파일·디렉토리 |

`distribution_owned` 를 지정하지 않으면 기본값이 적용됩니다 —
`SOUL.md`, `config.yaml`, `mcp.json`, `skills`, `cron`, `distribution.yaml`.

업데이트는 기록된 출처에서 다시 당겨옵니다.

```shell
hermes profile update hermes-reviewer
```

| 구분 | 대상 |
|---|---|
| 덮어씀 | 배포본이 소유한 파일, 배포본이 제공한 스킬·크론 |
| 보존 | 내가 추가한 스킬·크론, 기억, 세션, 인증, `.env` |
| 조건부 | `config.yaml` — 기본 보존, `--force-config` 를 줘야 덮어씀 |

`config.yaml` 이 기본 보존인 게 핵심이에요.
모델이나 프로바이더를 내가 바꿔 뒀다면 업데이트해도 그대로 남습니다.

<br>

<br>



## 8. 이름 바꾸기와 지우기

### rename

```shell
hermes profile rename blogdemo2 blogdemo9
```

```
✓ Renamed blogdemo2 → blogdemo9
✓ Alias updated: blogdemo9

Profile renamed: blogdemo2 → blogdemo9
Path: /opt/data/profiles/blogdemo9
```

디렉토리, 래퍼, 세션·라우팅 식별자까지 같이 옮겨줍니다.

**단, 커스텀 이름 래퍼는 고아가 됩니다.**
앞에서 `--name bd2` 로 만들어 둔 래퍼는 rename 이 정리해 주지 않아요.

```shell
cat /opt/data/.local/bin/bd2
```

```
#!/bin/sh
exec /opt/hermes/bin/hermes -p blogdemo2 "$@"
```

```shell
bd2 profile list
```

```
Error: Profile 'blogdemo2' does not exist. Create it with: hermes profile create blogdemo2
```

이름이 프로필명과 같은 래퍼만 자동 갱신되고, 별도 이름을 준 래퍼는 예전 프로필을 계속 가리킵니다.
`--name` 을 썼다면 rename 후에 직접 다시 만들어 주세요.

### delete

```shell
hermes profile delete blogdemo -y
```

```
This will permanently delete:
  • All config, API keys, memories, sessions, skills, cron jobs
  • Command alias (/opt/data/.local/bin/blogdemo)
✓ Removed /opt/data/.local/bin/blogdemo
✓ Removed /opt/data/profiles/blogdemo

Profile 'blogdemo' deleted.
```

되돌릴 수 없습니다. 아까운 게 있으면 `export` 를 먼저 하세요.

### `default` 는 예외입니다

1절에서 짚었듯 `default` 는 루트 그 자체라, 지울 수 없습니다.

```shell
hermes profile delete default -y
```

```
Error: Cannot delete the default profile (~/.hermes).
To remove everything, use: hermes uninstall
```

`rename` 도 다르게 동작해요. 디렉토리를 옮기는 대신 **표시 이름만** 붙입니다.

```shell
hermes profile rename default "메인봇"
```

```
✓ Display name set: 메인봇 (canonical id remains 'default')
```

목록에는 이렇게 보이지만, `-p` 로 지목할 때 쓰는 식별자는 여전히 `default` 입니다.

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆메인봇 (default)   gpt-5.5                      running      —            —
```

표시 이름은 루트의 `profile.yaml` 에 `display_name` 한 줄로 저장되니,
되돌리고 싶으면 그 줄을 지우면 됩니다.

### identity 재정리가 필요한 순간

`delete` 와 `rename` 은 디렉토리만 치우는 게 아니라
세션 키·라우팅 키·하트비트·전달 레코드까지 같이 정리합니다.

그런데 **게이트웨이가 돌고 있으면** 라우팅 인덱스가 메모리에 올라가 있어서,
삭제·변경이 DB에는 반영돼도 살아 있는 게이트웨이에는 안 먹을 수 있어요.
그럴 때 쓰라고 재실행용 명령이 따로 있습니다.

| 상황 | 명령 |
|---|---|
| 삭제한 프로필의<br>라우팅이 남아 있음 | `hermes profile`<br>`purge-identity`<br>`<지운이름>` |
| 이름 바꾼 프로필이<br>예전 이름으로 잡힘 | `hermes profile`<br>`migrate-identity`<br>`<옛이름> <새이름>` |

둘 다 멱등이라 여러 번 돌려도 안전합니다.
**게이트웨이를 재시작하거나 정지한 뒤에** 돌리는 게 정석이에요.

<br>

<br>



## 9. 이름 규칙 — 조용히 바뀌는 쪽을 조심

프로필 이름은 `소문자·숫자·하이픈·밑줄`, 첫 글자는 영숫자, 최대 64자입니다.
그런데 위반 처리가 두 갈래라서 이게 함정이에요.

| 입력 | 처리 |
|---|---|
| `BlogDemo` | **조용히 소문자로 정규화** → `blogdemo` |
| `blog writer` | 거부 + 대안 제시 |

대문자는 에러가 아니라 그냥 바뀝니다. `BlogDemo` 로 만든 프로필은 `blogdemo` 로 저장돼요.
그래서 이미 `blogdemo` 가 있는 상태에서 `BlogDemo` 를 만들면 이렇게 됩니다.

```shell
hermes profile create BlogDemo
```

```
Error: A profile named 'blogdemo' already exists.
```

공백은 정규화 대상이 아니라 거부됩니다. 대신 친절하게 대안을 줘요.

```shell
hermes profile create "blog writer"
```

```
Error: 'blog writer' is not a valid profile name. Use lowercase letters,
numbers, '-' or '_', starting with a letter or number, up to 64 characters
(for example: blog-writer). Then run `hermes profile create blog-writer`.
```

처음부터 소문자와 하이픈으로만 쓰면 둘 다 안 만납니다.

<br>

<br>



## 10. 명령어 한 장 정리

| 하고 싶은 것 | 명령 |
|---|---|
| 목록 보기 | `hermes profile list` |
| 상세 보기 | `hermes profile show <이름>` |
| 만들기 | `hermes profile create <이름>`<br>`--description "..."` |
| 복제해서 만들기 | `hermes profile create <새이름>`<br>`--clone-from <원본>` |
| 한 번만 지목 | `hermes -p <이름> <서브커맨드>` |
| 기본값 고정 | `hermes profile use <이름>` |
| 기본값 해제 | `hermes profile use default` |
| 래퍼 만들기 | `hermes profile alias <이름>`<br>`--name <별칭>` |
| 래퍼 지우기 | `hermes profile alias <이름>`<br>`--remove` |
| 역할 설명<br>읽기·쓰기 | `hermes profile describe <이름>`<br>`[--text "..."]` |
| 게이트웨이 상태 | `hermes gateway list` |
| 새 프로필<br>서빙 시작 | `hermes gateway restart` |
| 멀티플렉스<br>전환(계획만) | `hermes gateway migrate`<br>`--multiplex --dry-run` |
| 백업 | `hermes profile export <이름>` |
| 복원 | `hermes profile import <아카이브>`<br>`--name <이름>` |
| 배포본 설치 | `hermes profile install <git-url>`<br>`--alias` |
| 배포본 갱신 | `hermes profile update <이름>` |
| 이름 바꾸기 | `hermes profile rename <옛이름> <새이름>` |
| 지우기 | `hermes profile delete <이름> -y` |

<br>

<br>



## 11. 제가 쓰는 구성

참고삼아, 지금 굴려볼 만하다고 생각하는 구성을 적어둡니다.

| 프로필 | 역할 | 설명(`describe`) |
|---|---|---|
| `default` | 메신저 창구<br>잡다한 질문 | (비움) |
| `ops` | 크론·수집<br>스크립트<br>유지보수 | 데이터 수집<br>파이프라인<br>유지보수와<br>회귀 테스트 |
| `writer` | 블로그 초안<br>자료 조사 | 기술 블로그<br>초안 작성과<br>출처 조사 |
| `lab` | 실험용<br>언제 날려도<br>되는 곳 | 프로토타입<br>실험 전용 |

설계 원칙은 세 가지예요.

- **메신저 창구는 `default` 하나로.** 봇 토큰은 한 프로필만 들고 있어야 충돌이 안 납니다.
- **게이트웨이는 멀티플렉스 한 대로.** 프로세스를 프로필 수만큼 띄울 이유가 없어요.
- **`lab` 은 주기적으로 지우고 다시 만든다.** 실험 찌꺼기가 쌓이는 곳을 하나로 몰아두면 나머지가 깨끗합니다.

<br>

<br>



## 마무리

프로필은 기능 자체는 단순합니다. 디렉토리를 나누는 것뿐이에요.
그런데 실제로 써 보면 걸리는 곳이 정해져 있습니다.

| 자주 밟는 지뢰 | 정리 |
|---|---|
| `-p` 를<br>뒤에 붙임 | 동작은 하지만<br>에러 메시지가<br>쓸모없어짐 |
| 서브커맨드<br>`-p` 가 안 먹음 | 이름 규칙에 맞는<br>값은 프로필 플래그가<br>가로챔 |
| 프로필 만들었는데<br>봇이 무응답 | `hermes gateway restart`<br>필요 |
| 복제했는데<br>라우팅이 안 됨 | `description` 은<br>복제되지 않음 |
| 래퍼가<br>`command not found` | `~/.local/bin` 이<br>PATH 에 없음 |
| rename 후<br>커스텀 래퍼가 깨짐 | `--name` 래퍼는<br>수동 재생성 |
| 실험 프로필이<br>운영 키를 씀 | 빈 `.env` 는<br>셸 환경변수를 상속 |

여기까지가 Hermes Agent 시리즈 3편입니다.
1편에서 띄우고, 2편에서 고치고, 3편에서 역할별로 쪼갰어요.
