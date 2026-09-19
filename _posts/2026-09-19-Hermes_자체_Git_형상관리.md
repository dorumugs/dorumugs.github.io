---
layout: single
title:  "(2/2) Hermes 자체를 Git 으로 형상관리 — 632MB 에서 7.2MB 만 골라내기"
date: 2026-09-19 20:30:00 +0900
categories: coding
tag: [hermes, git, gitignore, 형상관리, 백업, 보안, 비밀관리, agent, llm, docker, pre-commit]
author_profile: false
toc: true
header:
  image: /assets/images/2026-09-19-hermes-git-config/header.svg
  teaser: /assets/images/2026-09-19-hermes-git-config/header.svg
description: "에이전트 설정 디렉토리를 Git 에 올리는 건 단순해 보이지만, 632MB 안에 되돌릴 가치가 있는 건 7.2MB 뿐이고 나머지엔 토큰이 섞여 있습니다. 화이트리스트 .gitignore 와 clean 필터를 실제 디렉토리에 돌려 검증한 기록이에요."
series: hermes-app-connect
series_order: 2
series_title: "🔗 Hermes 붙이기 — 구글과 깃"
---

{% include series-hermes-app-connect.html current="2" %}

## Summary

에이전트를 몇 달 쓰다 보면 `~/.hermes` 가 슬금슬금 자산이 됩니다.
모델·타임아웃·압축 임계값을 손으로 맞춰둔 `config.yaml`, 직접 쓴 스킬, 크론 잡, 에이전트가 쌓은 기억.
전부 **다시 만들기 귀찮고, 일부는 다시 만들 수도 없는** 것들이에요.

그런데 이 디렉토리는 백업하기가 묘하게 까다롭습니다.

> 632MB 인데 되돌릴 가치가 있는 건 **7.2MB** 뿐이고,
> 나머지 안에는 **토큰과 자격증명이 섞여** 있습니다.
> 통째로 올리면 저장소가 터지고, 대충 올리면 비밀이 샙니다.

그래서 이 글은 **"전부 무시하고 필요한 것만 되살리는"** 화이트리스트 방식으로 갑니다.
그리고 설계만 적지 않고, 실제로 돌아가는 Hermes 컨테이너에 **그대로 돌려서 확인한 숫자**를 같이 적었어요.

> 💡 **이 글에서 다루는 것**
> - 632MB 의 실측 내역 — 무엇이 자리를 차지하나
> - 🚨 함정1 **권한** — 호스트에서는 `git init` 조차 안 된다
> - 🚨 함정2 **비밀** — 어떤 파일이 절대 나가면 안 되나
> - 🚨 함정3 **`config.yaml` 은 깨끗하지 않다**
> - `.bundled_manifest` 로 내 스킬을 가르려다 실패한 기록
> - 화이트리스트 `.gitignore` 전문 + 실측 결과 (650 파일 / 7.2MB / 비밀 0)
> - `clean` 필터로 비밀 한 줄만 가리기
> - pre-commit 훅 · GitHub 프라이빗 원격 · 사고 대응 · 복원 리허설

> 📌 이 글의 수치는 전부 **제 실제 설치본**에서 측정한 값입니다.
> 버전·사용 패턴에 따라 달라지니, 숫자보다 **재는 방법**을 가져가세요.

<br>

<br>

## 1. 왜 형상관리인가

백업만으로는 부족한 이유가 있어요.

`config.yaml` 은 100줄이 넘고, 한 줄만 잘못 고쳐도 게이트웨이가 안 뜹니다.
문제는 **언제 무엇을 바꿨는지 아무 기록이 없다**는 거예요.

| 상황 | 백업만 있을 때 | Git 이 있을 때 |
|---|---|---|
| 어제까지 잘 되던 게 안 됨 | 어느 시점 백업인지 추측 | `git log -p config.yaml` |
| 스킬이 조용히 바뀜 | 모름 | `git diff` 에 뜸 |
| 설정 한 줄만 되돌리기 | 전체 복원 | `git checkout` 한 줄 |
| 다른 서버에 재현 | 통째 복사 | `git clone` |

특히 두 번째가 큽니다. Hermes 는 스킬을 **자동으로 동기화**해요.
컨테이너를 재시작할 때 번들 스킬을 밀어 넣습니다.

```text
Syncing bundled skills into ~/.hermes/skills/ ...
Done: 0 new, 0 updated, 87 unchanged. 87 total bundled.
```

여기서 `updated` 가 0 이 아니면 스킬 내용이 바뀐 거예요. **무엇이 어떻게 바뀌었는지는 안 알려줍니다.**
1편에서 본 `SKILL.md` 와 `setup.py` 의 불일치 같은 문제도, Git 이 있었다면 어느 업데이트에서 어긋났는지 바로 짚였을 겁니다.

<br>

<br>

## 2. 지형 실측 — 632MB 의 내역

설계 전에 먼저 잽니다. 짐작으로 `.gitignore` 를 쓰면 반드시 틀려요.

```shell
docker exec hermes du -sh /opt/data
# 632M    /opt/data

docker exec hermes sh -c 'du -sh /opt/data/* | sort -rh | head -8'
```

| 항목 | 용량 | 성격 | 추적? |
|---|---|---|---|
| `home/` | 117M | 에이전트 작업 홈 | ❌ |
| `bin/` | 60M | 내려받은 바이너리 | ❌ |
| `state.db` | 45M | 런타임 상태 | ❌ |
| `logs/` | 21M | 로그 | ❌ |
| `skills/` | 20M | 스킬 | ✅ 일부 |
| `cache/` | 11M | 캐시 | ❌ |
| `sessions/` | 2.8M | 대화 세션 | ❌ |
| `cron/` | 776K | 크론 잡 | ✅ 일부 |

판단 기준은 하나였어요.

> **"이 파일이 사라지면 내가 손으로 다시 만들어야 하나?"**
>
> 다시 만들어야 하면 → 추적.
> 프로그램이 다시 만들어주면 → 제외.
> 다시 만들면 안 되는 거면(=비밀) → 제외.

이 기준으로 보면 `state.db` 45MB 는 명확히 제외예요. 런타임이 다시 만듭니다.
**게다가 SQLite 는 Git 과 최악의 궁합**이에요. 바이너리라 diff 가 안 되고, 한 바이트만 바뀌어도 45MB 블롭이 통째로 새로 쌓입니다.

<br>

<br>

## 3. 🚨 함정1 — 호스트에서는 `git init` 조차 안 된다

가장 먼저 만나는 벽이고, 안 겪어보면 예상하기 어려운 벽입니다.

```shell
LC_ALL=C ls -la ~/.hermes
# ls: cannot open directory '/home/dorumugs/.hermes': Permission denied
```

제 계정인데 제 홈 디렉토리 아래를 못 읽어요. 이유는 소유자와 권한입니다.

```shell
stat -c '%u:%g %a' ~/.hermes
# 10000:10000 700
```

| 주체 | uid | `~/.hermes` 접근 |
|---|---|---|
| 호스트 `dorumugs` | 1000 | ❌ 불가 (700) |
| 컨테이너 `hermes` | 10000 | ✅ 소유자 |

바인드 마운트라 **컨테이너 안의 uid 가 호스트에 그대로 보입니다.**
컨테이너의 `hermes` 는 uid 10000 인데 호스트에 그런 사용자가 없으니, 호스트에서는 주인 없는 700 디렉토리로 보이는 거예요.

해결은 세 갈래입니다.

| 방법 | 장점 | 단점 |
|---|---|---|
| **컨테이너 안<br>에서 git** | 권한 맞음<br>소유자 안 깨짐 | git·키가<br>컨테이너에 필요 |
| 호스트에서<br>`sudo git` | 익숙함 | 🔴 `.git` 이<br>root 소유 |
| 복사 후 커밋 | 격리됨 | 복사 중<br>비밀 샐 여지 |

**컨테이너 안에서 돌리는 쪽을 권합니다.** git 은 이미 들어있어요.

```shell
docker exec hermes git --version
# git version 2.47.3
```

> ⚠️ `sudo git` 은 피하세요. `.git/` 이 root 소유로 생기는데,
> 이후 컨테이너 안 `hermes` 유저가 같은 저장소를 만지면 권한 충돌이 납니다.
> 한 저장소를 **한 주체만** 만지게 하는 게 깔끔해요.

그리고 항상 `-u hermes` 를 붙입니다. 1편의 토큰 소유권 문제와 같은 이유예요.

```shell
docker exec -u hermes hermes git status
```

<br>

<br>

## 4. 🚨 함정2 — 절대 나가면 안 되는 것들

이름만 봐도 아는 것부터, 안 보이는 것까지 있습니다.

```shell
docker exec hermes sh -c \
  'ls -la /opt/data | grep -iE "auth|token|secret|\.env"'
```

| 파일 | 내용 | 유출 시 |
|---|---|---|
| `.env` | 전 서비스 자격증명 (23KB) | 🔴 전부 털림 |
| `.env.bak-*` | 과거 `.env` 사본 | 🔴 동일 |
| `auth.json` | LLM provider OAuth 토큰 | 🔴 내 ChatGPT 계정 |
| `google_token.json` | Google 토큰 (1편) | 🔴 메일·드라이브 |
| `google_client_secret.json` | OAuth 클라이언트 | 🔴 앱 사칭 |

> 🚨 **`.env.bak-*` 를 놓치기 쉽습니다.**
> `.env` 만 무시 목록에 적고 끝내면 백업본이 그대로 올라가요.
> Hermes 는 설정을 크게 바꿀 때 타임스탬프 붙인 백업을 자동으로 남깁니다.
> 이름을 하나씩 적는 **블랙리스트 방식이 위험한 이유**가 이거예요 —
> 내가 모르는 파일은 목록에 못 적습니다.

그래서 이 글은 화이트리스트로 갑니다. **먼저 전부 막고, 아는 것만 엽니다.**
모르는 파일이 새로 생겨도 기본이 "막힘" 이에요.

비밀은 권한으로도 한 번 더 막혀 있습니다.

```shell
docker exec hermes stat -c '%a %n' /opt/data/.env /opt/data/auth.json
# 600 /opt/data/.env
# 600 /opt/data/auth.json
```

<br>

<br>

## 5. 🚨 함정3 — `config.yaml` 은 깨끗하지 않다

여기가 제가 제일 놀랐던 부분이에요.

`config.yaml` 은 가장 추적하고 싶은 파일입니다. 손으로 튜닝한 값이 다 여기 있으니까요.
그리고 "비밀은 `.env` 에 있다" 고 알려져 있어서, 마음 놓고 커밋하기 쉽습니다.

그래서 진짜 그런지 세봤어요.

```shell
docker exec hermes sh -c 'grep -nE "(key|token|secret|password)" /opt/data/config.yaml' \
  | sed -E 's/:.*/: <redacted>/'
```

값이 비었는지까지 확인해보니 결과가 이랬습니다.

| 줄 | 항목 | 값 |
|---|---|---|
| 103 | `session_key` | 비어 있음 |
| 159~323 | `api_key` × 11 | 전부 비어 있음 |
| **260** | **`password_hash`** | 🔴 **86자, 들어 있음** |

`api_key` 항목이 11개나 있는데 전부 비어 있어요. 제가 provider 키를 `.env` 로 넣었기 때문입니다.
**다른 설치본에서는 여기가 채워져 있을 수 있어요.** 그러니 이 글의 결과를 그대로 믿지 말고 각자 세어보셔야 합니다.

그리고 260줄, **대시보드 비밀번호 해시**가 들어 있습니다. 해시라도 오프라인 대입 공격의 입력이 돼요.

> 🚨 **"`config.yaml` 은 안전하다" 는 말은 틀렸습니다.**
> 파일 이름으로 안전을 판단하면 안 돼요. 커밋 전에 **내용을 직접 세야** 합니다.

그렇다고 `config.yaml` 을 통째로 포기하기는 아깝습니다. 가장 가치 있는 파일이니까요.
그래서 **파일은 추적하되 그 한 줄만 가리는** 방법을 9장에서 씁니다.

<br>

<br>

## 6. 스킬 디렉토리 — manifest 로 가르려다 실패한 기록

`skills/` 는 20MB 예요. 그냥 올리기엔 크고, 버리기엔 아깝습니다.
번들 스킬은 재설치로 복원되니, **내가 만든 것만** 고르면 될 것 같았어요.

마침 목록 파일이 있습니다.

```shell
docker exec hermes head -3 /opt/data/skills/.bundled_manifest
# airtable:3b1f4e4c0e6aac15f2fd7f55e151bda9
# apple-notes:a969ab89eaf759c3ff421709808f15b6
# apple-reminders:b38e5f2558c2842808fe85df10226598
```

`이름:해시` 형식이니, 여기 없는 스킬이 내 것이겠죠. 세어봤습니다.

| 항목 | 개수 |
|---|---|
| manifest 등재 | 58 |
| 실제 스킬 디렉토리 | 105 |
| 차집합 = "내 스킬"? | 47 |

그런데 그 47개를 열어보니 `mlops/training`, `creative/comfyui`, `github/github-auth` …
**전부 Nous 가 배포한 스킬이었어요.** 제가 만든 게 아닙니다.

> 🚨 **`.bundled_manifest` 는 완전한 목록이 아닙니다.**
> 58개만 등재돼 있는데 실제로는 105개가 깔려 있어요.
> 이걸로 자동 분류하면 **남의 스킬 47개를 "내 것" 이라고 커밋합니다.**
> 파일 이름이 그럴듯하다고 신뢰의 근거로 삼으면 안 되는 사례예요.

그래서 방향을 틀었습니다. **가르지 말고 다 가져가되, 진짜 무거운 것만 빼자.**
근거는 용량 내역이었어요.

```shell
docker exec hermes du -sh /opt/data/skills/.curator_backups
# 12M     /opt/data/skills/.curator_backups
```

`skills/` 20MB 중 **12MB 가 `.curator_backups`** 였습니다. 스킬 큐레이터가 남긴 `skills.tar.gz` 5개요.
이건 Git 이 할 일을 이미 다른 방식으로 하고 있는 것이고, 압축 파일이라 diff 도 안 됩니다. **정확히 Git 에 넣으면 안 되는 물건**이에요.

이걸 빼면 남는 8MB 의 정체는 이렇습니다.

| 확장자 | 개수 |
|---|---|
| `.md` | 485 |
| `.py` | 80 |
| `.json` | 22 |

거의 전부 텍스트예요. **Git 이 제일 잘하는 종류**입니다. diff 가 되고, 압축도 잘 되고, 히스토리가 쌓여도 완만하게 늘어요.

> ✅ 결론: 스킬은 **`.curator_backups` 만 빼고 통째로** 추적한다.
> 영리한 분류를 포기하는 대신, 틀릴 여지를 없앴습니다.

<br>

<br>

## 7. 화이트리스트 `.gitignore`

이제 설계를 옮깁니다. 3단 구조예요.

| 단계 | 하는 일 |
|---|---|
| 1 | `/*` — 최상위 전부 무시 |
| 2 | `!` — 추적할 것만 되살리기 |
| 3 | 되살린 것 안에서 다시 걷어내기 |

```gitignore
# 1) ignore everything
/*

# 2) allow back only what we track
!/.gitignore
!/.gitattributes
!/SOUL.md
!/config.yaml
!/cron/
!/hooks/
!/memories/
!/profiles/
!/scripts/
!/skills/

# 3) prune inside what we allowed back
*.lock
*.db
*.db-shm
*.db-wal
/cron/output/
/cron/ticker_*
/profiles/.deleted/
/skills/.curator_backups/
/skills/.locks/
/skills/.usage.json
```

1단계 `/*` 가 핵심이에요. 이 한 줄 덕분에 **내일 Hermes 가 새 파일을 만들어도 기본이 "무시"** 입니다.
비밀 파일 이름을 하나하나 적을 필요가 없어요 — 적지 않은 건 전부 막히니까요.

되살린 것 각각의 이유는 이렇습니다.

| 경로 | 왜 추적하나 |
|---|---|
| `config.yaml` | 손으로 맞춘 전 설정 |
| `SOUL.md` | 에이전트 성격·행동 규범 |
| `memories/` | 에이전트가 쌓은 사용자 기억 |
| `cron/` | 예약 작업 정의 |
| `hooks/` | 이벤트 훅 |
| `profiles/` | 프로필 |
| `scripts/` | 직접 쓴 스크립트 |
| `skills/` | 스킬 전체 |

3단계는 실제로 돌려보고 추가한 것들이에요. 처음 설계에는 `*.lock` 이 없었는데, 돌려보니 잠금 파일 6개가 딸려 왔습니다.

<br>

<br>

## 8. 검증 — 진짜 그렇게 되나

**여기를 건너뛰면 안 됩니다.** `.gitignore` 는 머릿속에서 맞는 것 같아도 자주 틀려요.

좋은 소식은, 커밋하지 않고도 **원본에 아무것도 쓰지 않고** 확인할 수 있다는 겁니다.
`--git-dir` 와 `--work-tree` 를 갈라두면 `.git` 이 `/opt/data` 밖에 생겨요.

```shell
docker exec hermes sh -c '
git init --bare -q /tmp/hgit
G="git --git-dir=/tmp/hgit --work-tree=/opt/data"
$G config core.bare false
$G config core.excludesFile /tmp/hermes.ignore
$G status --porcelain -uall | wc -l
'
```

> 📌 `-uall` 이 중요합니다.
> 이게 없으면 git 이 **디렉토리를 접어서** 보여줘요.
> 제가 처음 돌렸을 때 "7개" 가 나와서 놀랐는데, `skills/` 한 줄이 실제로는 644개였습니다.
> 파일 단위로 펼쳐 보지 않으면 무엇이 올라가는지 모릅니다.

실측 결과입니다.

| 항목 | 값 |
|---|---|
| 추적 파일 수 | **650** |
| 총 용량 | **7.2MB** |
| 원본 대비 | 632MB → **1.1%** |
| 비밀·잠금·DB 유출 | **0건** |

`skills/` 를 뺀 나머지는 눈으로 다 확인할 만큼 적었어요.

| 파일 | 정체 |
|---|---|
| `SOUL.md` | 에이전트 행동 규범 |
| `config.yaml` | 전체 설정 |
| `cron/jobs.json` | 크론 잡 정의 |
| `memories/USER.md` | 사용자 기억 |
| `scripts/mapo_weather_air.py` | 직접 쓴 스크립트 |
| `scripts/naver_etf_swing.py` | 직접 쓴 스크립트 |

나머지 644개가 `skills/` 입니다. **눈으로 한 번 훑을 수 있는 분량까지 줄이는 게 목표**였고, 달성됐어요.

비밀이 안 섞였는지는 따로 셉니다.

```shell
$G status --porcelain -uall | sed 's/^...//' \
  | grep -cE '(\.env|auth\.json|google_token|client_secret|\.lock$|\.db$)'
# 0
```

확인이 끝나면 흔적을 지웁니다.

```shell
docker exec hermes rm -rf /tmp/hgit /tmp/hermes.ignore
```

> ✅ 이 방식의 장점 — **실패해도 원본이 안 더러워집니다.**
> `.gitignore` 를 서너 번 고쳐가며 돌릴 수 있어요.
> 만족스러워진 뒤에 진짜 `.gitignore` 를 `/opt/data` 에 두면 됩니다.

<br>

<br>

## 9. `config.yaml` 의 한 줄만 가리기 — clean 필터

5장에서 미뤄둔 문제를 풉니다. 파일은 추적하고 싶은데 260줄만 위험한 상황이요.

Git 에는 **clean 필터**가 있어요. 스테이징할 때 내용을 한 번 거쳐가게 하는 기능입니다.

| 시점 | 필터 | 방향 |
|---|---|---|
| `git add` | `clean` | 작업파일 → 저장소 |
| `git checkout` | `smudge` | 저장소 → 작업파일 |

`clean` 만 걸면 **저장소에는 가려진 값이, 디스크에는 진짜 값이** 남습니다.

`.gitattributes` 에 대상 파일을 지정하고,

```gitattributes
config.yaml filter=hermes-redact
```

필터 본체를 등록합니다.

```shell
docker exec -u hermes hermes git -C /opt/data config \
  filter.hermes-redact.clean \
  'sed -E "s|^([[:space:]]*password_hash:).*|\1 REDACTED|"'
```

실제로 되는지 확인했습니다.

```shell
docker exec -u hermes hermes git -C /opt/data show :config.yaml | grep password_hash
# 260:    password_hash: REDACTED
```

디스크 원본은 그대로예요.

```shell
docker exec hermes grep -c password_hash /opt/data/config.yaml
# 1
```

> ✅ 스테이징된 내용은 `REDACTED`, 실제 파일은 **86자 해시 그대로**.
> 대시보드 로그인은 계속 됩니다.

### 대가 — 복원하면 그 줄은 비어 있다

공짜는 아니에요. 정확히 알고 써야 합니다.

> ⚠️ **`clean` 만 걸었으니 복원하면 `REDACTED` 가 나옵니다.**
> 저장소에서 `config.yaml` 을 되살리면 그 줄은 가려진 채예요.
> 새 서버에 복원한 뒤 **대시보드 비밀번호를 다시 설정**해야 합니다.

이건 버그가 아니라 **의도된 거래**입니다.

| 선택 | 얻는 것 | 잃는 것 |
|---|---|---|
| clean 필터 | 비밀이 절대 안 나감 | 복원 후 재설정 1회 |
| 통째 커밋 | 완전 복원 | 🔴 해시 유출 |
| 파일 제외 | 유출 없음 | 🔴 설정 전체 못 되돌림 |

한 번의 수동 재설정으로 나머지 100여 줄을 온전히 되살릴 수 있다면, 저는 그게 남는 장사라고 봅니다.

> 📌 `git config` 에 필터를 등록해야 동작해요.
> `.gitattributes` 만 커밋하고 **다른 서버에서 필터 등록을 빼먹으면 조용히 원본이 커밋됩니다.**
> 복원 절차서에 "필터 등록" 을 **첫 줄**로 적어두세요.

<br>

<br>

## 10. 첫 커밋 — 눈으로 한 번 보고

검증이 끝났으니 진짜로 만듭니다.

```shell
docker exec -u hermes hermes sh -c '
cd /opt/data
git init -q
git config user.name  "Hermes Config"
git config user.email "hermes@localhost"
'
```

`.gitignore` 와 `.gitattributes` 를 두고, 9장의 필터를 등록한 뒤 스테이징합니다.

```shell
docker exec -u hermes hermes git -C /opt/data add -A
```

**커밋 전에 목록을 눈으로 봅니다.** 이 단계를 습관으로 만드세요.

```shell
docker exec -u hermes hermes git -C /opt/data diff --cached --name-only \
  | grep -v '^skills/'
```

그리고 스테이징된 **내용**까지 한 번 훑습니다. 파일 이름만 봐서는 안 걸리는 게 있거든요.

```shell
docker exec -u hermes hermes git -C /opt/data diff --cached \
  | grep -inE 'sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|password_hash'
```

> 📌 **이름이 아니라 내용을 검사하는 게 핵심입니다.**
> 5장의 `password_hash` 는 파일명이 `config.yaml` 이라 이름 검사로는 절대 안 걸려요.
> 사고는 대개 "이 파일은 안전하다" 고 믿은 파일에서 납니다.

깨끗하면 커밋합니다.

```shell
docker exec -u hermes hermes git -C /opt/data commit -q -m "Initial Hermes configuration snapshot"
docker exec -u hermes hermes git -C /opt/data log --oneline -1
```

<br>

<br>

## 11. pre-commit 훅 — 눈검사를 자동화

10장의 검사를 매번 손으로 하면 언젠가 거릅니다. 훅으로 박아두세요.

```shell
docker exec -u hermes hermes sh -c 'cat > /opt/data/.git/hooks/pre-commit <<"HOOK"
#!/bin/sh
set -e

STAGED=$(git diff --cached)

PATTERN="sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
PATTERN="$PATTERN|-----BEGIN [A-Z ]*PRIVATE KEY-----"

HITS=$(printf "%s\n" "$STAGED" | grep -nE "$PATTERN" || true)

LEAK=$(printf "%s\n" "$STAGED" \
  | grep -nE "password_hash:[[:space:]]*[^[:space:]]" \
  | grep -v "REDACTED" || true)

if [ -n "$HITS" ] || [ -n "$LEAK" ]; then
  echo "pre-commit: secret-like content detected. Aborting."
  printf "%s\n%s\n" "$HITS" "$LEAK" | grep -v "^$" | head -5
  exit 1
fi

for f in $(git diff --cached --name-only --diff-filter=ACM); do
  [ -f "$f" ] || continue
  size=$(wc -c < "$f")
  if [ "$size" -gt 1048576 ]; then
    echo "pre-commit: $f is ${size} bytes (>1MB). Aborting."
    exit 1
  fi
done
HOOK
chmod +x /opt/data/.git/hooks/pre-commit'
```

세 가지를 봅니다.

| 검사 | 막는 것 |
|---|---|
| 토큰 패턴 | API 키·개인키가 **내용에** 섞인 경우 |
| `password_hash` | 9장 필터가 **등록 안 된** 상태 |
| 1MB 상한 | 무시 규칙 구멍으로 큰 파일이 샌 경우 |

용량 상한이 은근히 유용해요. 새 캐시 디렉토리가 생겨서 무시 규칙을 빠져나가면, **비밀 검사보다 용량 검사에 먼저 걸립니다.**

두 번째 검사가 **두 단계로 나뉜 이유**가 있습니다.

> ⚠️ `password_hash:` 뒤에 값이 있으면 막아야 하는데,
> 필터가 정상 동작했을 때 들어가는 `REDACTED` **도 "값" 입니다.**
> 한 번에 거르면 정상 커밋까지 전부 막혀요.
>
> `(?!REDACTED)` 같은 부정 전방탐색은 `grep -E` 가 지원하지 않습니다.
> 그래서 **잡고(`grep -E`) 빼는(`grep -v`)** 두 단계로 나눴어요.
> 훅을 쓰기 전에 **정상 커밋이 통과하는지 반드시 한 번 확인**하세요 —
> 늘 막는 훅은 며칠 안에 `--no-verify` 로 우회되고, 그러면 없는 것만 못합니다.

> 🚨 **훅은 저장소와 함께 복제되지 않습니다.**
> `.git/hooks/` 는 추적 대상이 아니에요.
> 다른 서버에 `clone` 하면 훅이 **없는 채로** 시작합니다.
> 훅 본체는 `scripts/` 에 넣어 추적하고, 설치는 복원 절차서에 한 줄로 적어두세요.

<br>

<br>

## 12. GitHub 프라이빗 원격 붙이기

로컬 히스토리는 디스크가 죽으면 같이 죽습니다. 원격을 붙여야 백업이 돼요.

**반드시 프라이빗으로** 만듭니다. 그리고 프라이빗이어도 앞의 마스킹을 생략하면 안 돼요.

> 🚨 저장소 공개 설정은 **클릭 한 번으로 바뀝니다.**
> 협업자 추가, 조직 이관, 포크 — 공개 범위가 넓어질 경로는 많아요.
> **내용 자체가 안전해야** 합니다. 프라이빗은 두 번째 방어선이지 첫 번째가 아니에요.

인증은 배포 키(deploy key)를 권합니다. 계정 전체가 아니라 **이 저장소 하나**에만 닿거든요.

```shell
docker exec -u hermes hermes ssh-keygen -t ed25519 -N "" \
  -f /opt/data/.ssh/id_ed25519 -C "hermes-config-backup"
docker exec -u hermes hermes cat /opt/data/.ssh/id_ed25519.pub
```

출력된 공개키를 GitHub 저장소의 **Settings → Deploy keys** 에 등록하고 쓰기 권한을 줍니다.

> ✅ 개인키를 `/opt/data/.ssh/` 에 두는 게 불안해 보이지만 괜찮아요.
> 7장의 `/*` 규칙이 화이트리스트에 없는 `.ssh/` 를 **이미 막고 있습니다.**
> 화이트리스트 설계의 덤이에요 — 새로 추가하는 비밀도 기본이 "제외" 입니다.

붙이고 올립니다.

```shell
docker exec -u hermes hermes git -C /opt/data remote add origin \
  git@github.com:<user>/hermes-config.git
docker exec -u hermes hermes git -C /opt/data push -u origin main
```

올린 뒤 **웹에서 눈으로 확인**하세요. 특히 `config.yaml` 260줄 근처가 `REDACTED` 인지요.
로컬에서 통과한 것과 원격에 실제로 올라간 것은 다를 수 있습니다.

<br>

<br>

## 13. 그래도 샜다면

사고는 납니다. 중요한 건 **순서**예요.

> 🚨 **1순위는 히스토리 정리가 아니라 키 폐기입니다.**
> 공개된 순간 그 키는 이미 죽은 키예요.
> 커밋을 지워도 **누군가 이미 받아갔을 수 있습니다.**
> GitHub 는 삭제된 커밋도 한동안 접근 가능하고, 자동 스크래퍼는 몇 분 안에 훑어갑니다.

| 순서 | 할 일 |
|---|---|
| 1 | 🔴 해당 키·토큰 **즉시 폐기·재발급** |
| 2 | 히스토리에서 제거 |
| 3 | 강제 푸시 |
| 4 | 무시 규칙·훅 보강 |

1편의 Google 토큰이 샜다면 `$GSETUP --revoke` 가 1번입니다.
`auth.json` 이면 ChatGPT 쪽 세션을 끊어야 하고요.

폐기가 끝난 뒤에 히스토리를 정리합니다.

```shell
git filter-repo --path config.yaml --invert-paths --force
```

특정 파일이 아니라 **값**을 지우려면 치환 방식이 낫습니다.

```shell
printf 'literal:AKIA0000EXAMPLE==>REDACTED\n' > /tmp/repl.txt
git filter-repo --replace-text /tmp/repl.txt --force
```

`filter-repo` 는 **모든 커밋 해시를 바꿉니다.** 클론해 둔 곳이 있으면 전부 다시 받아야 해요.
혼자 쓰는 설정 저장소라면 큰 문제는 아닙니다.

<br>

<br>

## 14. 복원 리허설 — 이게 진짜 마지막 관문

여기가 이 글에서 제일 말하고 싶은 부분이에요.

> **복원해본 적 없는 백업은 백업이 아닙니다.**

특히 이 저장소는 일부러 불완전하게 만들었어요. 비밀을 뺐고, 한 줄은 가렸고, 훅은 안 따라옵니다.
**무엇이 빠졌는지 정확히 아는 상태**여야 복원이 성립해요.

빈 디렉토리에 받아서 확인해보세요.

```shell
git clone git@github.com:<user>/hermes-config.git /tmp/restore-test
ls -a /tmp/restore-test
```

| 확인 | 기대 |
|---|---|
| `config.yaml` 있나 | ✅ |
| 260줄이 `REDACTED` 인가 | ✅ (의도대로) |
| `.env` 없나 | ✅ 없어야 정상 |
| `auth.json` 없나 | ✅ 없어야 정상 |
| `skills/` 개수 | ✅ 644 |
| `.git/hooks/pre-commit` | ❌ 없음 — 정상 |

그리고 **빠진 것을 어떻게 채울지** 적어두는 게 복원 절차서입니다.

| 빠진 것 | 채우는 법 |
|---|---|
| `.env` | 서비스별 키 재발급 |
| `auth.json` | `hermes auth add` 재인증 |
| `google_token.json` | 1편 5장 재실행 |
| `password_hash` | 대시보드에서 재설정 |
| pre-commit 훅 | `scripts/` 에서 복사 |
| clean 필터 | `git config` 재등록 |

> 📌 이 표가 **저장소에 같이 들어가야** 합니다. `RESTORE.md` 로 두세요.
> 복원이 필요한 순간은 대개 급한 순간이고, 그때 이걸 새로 떠올리긴 어렵습니다.

<br>

<br>

## 15. 자동 스냅샷

손으로 커밋하면 결국 안 하게 돼요. 크론에 겁니다.

```shell
docker exec -u hermes hermes sh -c 'cat > /opt/data/scripts/snapshot.sh <<"SH"
#!/bin/sh
set -e
cd /opt/data
git add -A
if git diff --cached --quiet; then
  exit 0
fi
git commit -q -m "Snapshot $(date +%Y-%m-%dT%H:%M:%S%z)"
git push -q origin main
SH
chmod +x /opt/data/scripts/snapshot.sh'
```

`git diff --cached --quiet` 로 **변경이 없으면 커밋하지 않습니다.** 빈 커밋이 쌓이면 히스토리가 쓸모없어져요.

호스트 크론에 겁니다.

```shell
20 5 * * * flock -w 3600 ~/.cache/hermes-snapshot.lock \
  docker exec -u hermes hermes /opt/data/scripts/snapshot.sh \
  >> ~/.cache/hermes-snapshot.log 2>&1
```

`flock` 은 꼭 넣으세요. 스냅샷이 겹쳐 돌면 인덱스 잠금이 충돌합니다.

> ⚠️ **자동 스냅샷은 pre-commit 훅이 살아있을 때만 안전합니다.**
> 사람이 안 보는 채로 `git add -A` 가 도는 구조예요.
> 훅이 없으면, 새로 생긴 비밀 파일이 무시 규칙을 빠져나갔을 때 **아무도 모르게 올라갑니다.**
> 자동화를 켜기 전에 훅부터 확인하세요.

<br>

<br>

## 16. 정리

| 항목 | 결론 |
|---|---|
| 권한 | 호스트에서 불가. 컨테이너 `-u hermes` |
| 전략 | 화이트리스트. 전부 막고 아는 것만 열기 |
| `config.yaml` | 🔴 깨끗하지 않음. clean 필터로 한 줄 마스킹 |
| `.bundled_manifest` | 🔴 불완전(58/105). 분류 근거로 쓰지 말 것 |
| `skills/` | `.curator_backups` 만 빼고 통째로 |
| 검증 | `--git-dir` 분리 + `-uall` |
| 실측 | 632MB → **7.2MB / 650 파일 / 비밀 0** |
| 사고 시 | 🔴 키 폐기가 1순위, 히스토리는 그다음 |
| 마지막 | 복원 리허설 + `RESTORE.md` |

두 편을 관통하는 게 하나 있었어요.

1편에서는 스킬 문서가 `--services` 로 스코프를 좁히라고 했지만 실제 스크립트에는 그 옵션이 없었고,
2편에서는 `.bundled_manifest` 가 번들 목록처럼 보였지만 절반만 담겨 있었습니다.
`config.yaml` 은 "비밀은 `.env` 에 있다" 는 통념과 달리 해시를 품고 있었고요.

> **문서가 약속한 것과 코드가 하는 것은 다를 수 있습니다.**
> 에이전트에게 권한을 넘기는 자리에서는, 읽은 것을 한 번씩 **재보는** 게 유일하게 확실한 방법이에요.
>
> 이 글의 숫자는 제 설치본 것입니다. 가져가실 건 숫자가 아니라 **재는 방법**입니다.

<br>

{% include series-hermes-app-connect.html current="2" %}
