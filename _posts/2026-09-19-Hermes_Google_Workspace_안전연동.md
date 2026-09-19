---
layout: single
title:  "(1/2) Hermes 에 Google Workspace 안전하게 붙이기 — 스코프는 못 좁힌다, 그래서 계정을 가른다"
date: 2026-09-19 20:00:00 +0900
categories: coding
tag: [hermes, google-workspace, oauth, gmail, google-docs, google-sheets, google-drive, agent, llm, 보안, 최소권한]
author_profile: false
toc: true
header:
  image: /assets/images/2026-09-19-hermes-google-workspace/header.svg
  teaser: /assets/images/2026-09-19-hermes-google-workspace/header.svg
description: "Hermes 에이전트에 Google Docs·Sheets 를 붙이는 전 과정입니다. 공식 스킬 문서가 약속하는 '최소 스코프로 좁히기'가 실제 스크립트에서는 동작하지 않는다는 걸 실측으로 확인하고, 그 대신 무엇으로 막아야 하는지까지 정리했어요."
series: hermes-app-connect
series_order: 1
series_title: "🔗 Hermes 붙이기 — 구글과 깃"
---

{% include series-hermes-app-connect.html current="1" %}

## Summary

Hermes 에이전트한테 "이 시트에 이번 주 실적 붙여줘" 라고 말하려면 Google 계정을 붙여야 해요.
붙이는 것 자체는 어렵지 않습니다. 번들 스킬 `google-workspace` 가 이미 들어있고, 5단계짜리 셋업이 문서화돼 있거든요.

문제는 **그 5단계를 따라 하면 실제로 어떤 권한이 넘어가는지** 입니다.

스킬 문서(`SKILL.md`, v1.2.0)는 `--services email,calendar` 처럼 **필요한 서비스만 골라서 동의 화면을 좁히라**고 안내해요.
그런데 같은 스킬에 들어있는 `setup.py` 를 실제로 열어보면 그런 옵션이 **없습니다.** 그대로 실행하면 에러가 납니다.

그래서 이 글은 두 겹이에요.

> 1. **붙이는 법** — OAuth 클라이언트 만들기부터 Docs·Sheets 실전 명령까지, 헤드리스 서버 기준으로.
> 2. **막는 법** — 스코프로 못 좁힌다는 걸 확인했으니, 그 대신 무엇으로 막아야 하는지.

> 💡 **이 글에서 다루는 것**
> - 메일만 필요한 경우엔 이 스킬을 **쓰면 안 되는** 이유
> - 🚨 문서와 실제 스크립트의 불일치 — `--services` · `--format` 은 존재하지 않는다
> - 실제로 넘어가는 8개 스코프, 그중 제일 무서운 것
> - 헤드리스 서버에서 OAuth 끝내기 (브라우저 없는 컨테이너)
> - 컨테이너에서 돌릴 때 `-u hermes` 가 필요한 이유
> - Docs · Sheets 실전 명령과 탭 문서 함정
> - 스킬이 스스로 거는 안전 규칙, 그리고 그걸 믿으면 안 되는 이유

<br>

<br>

## 1. 먼저 — 이 스킬이 정말 필요한가

시작하기 전에 갈림길이 하나 있어요. 여기서 잘못 들면 필요도 없는 Google Cloud 프로젝트를 하나 파게 됩니다.

| 필요한 것 | 써야 할 것 | 준비물 |
|---|---|---|
| 메일만 | `himalaya` | Gmail 앱<br>비밀번호 |
| 캘린더·드라이브<br>시트·문서 | `google-workspace` | Cloud 프로젝트<br>+ OAuth 클라이언트 |

스킬 문서도 같은 얘기를 먼저 합니다 — 메일만 필요하면 **이 스킬은 아예 필요 없다**고요.
앱 비밀번호는 Gmail 설정의 보안 탭에서 바로 발급되고, Cloud 콘솔을 열 일이 없어요.

이 글은 두 번째 줄, 그러니까 **Docs 와 Sheets 를 쓰려는 경우**를 다룹니다.

<br>

<br>

## 2. 🚨 문서가 약속한 '최소 권한'은 이 버전에 없다

여기가 이 글에서 제일 중요한 부분이에요.

`SKILL.md` 의 3단계에는 이렇게 하라고 적혀 있습니다 — 필요한 서비스만 골라 동의 화면을 좁히라고요.

```shell
python setup.py --auth-url --services calendar,drive,sheets,docs --format json
```

그대로 돌려봤습니다.

```text
usage: setup.py [-h] (--check | --check-live | --client-secret PATH |
                --auth-url | --auth-code CODE | --revoke | --install-deps)
setup.py: error: unrecognized arguments: --services calendar,drive,sheets,docs --format json
```

`--services` 도, `--format` 도 **존재하지 않습니다.** 스크립트가 실제로 받는 인자는 7개가 전부예요.

```shell
docker exec hermes python \
  /opt/data/skills/productivity/google-workspace/scripts/setup.py --help
```

| 인자 | 하는 일 |
|---|---|
| `--check` | 토큰이 유효한지 (종료코드 0/1) |
| `--check-live` | 실제 API 를 한 번 호출해서 확인 |
| `--client-secret PATH` | OAuth 클라이언트 JSON 저장 |
| `--auth-url` | 동의 URL 출력 |
| `--auth-code CODE` | 코드를 토큰으로 교환 |
| `--revoke` | 토큰 폐기·삭제 |
| `--install-deps` | 파이썬 의존성 설치 |

> ⚠️ 문서만 읽고 명령을 조립하면 3단계에서 막힙니다.
> 에이전트에게 "스킬 문서대로 해줘" 라고 맡기면, 에이전트도 똑같이 막힌 뒤
> 옵션을 빼고 재시도해서 **결국 전체 스코프로 동의를 받아버립니다.** 조용히요.

### 그래서 실제로 넘어가는 권한

`setup.py` 안의 스코프 목록은 분기 없이 고정입니다. 8개가 통째로 요청돼요.

| 스코프 | 의미 |
|---|---|
| `gmail.readonly` | 메일 전체 읽기 |
| `gmail.send` | 내 이름으로 메일 발송 |
| `gmail.modify` | 라벨·읽음 상태 변경 |
| `calendar` | 캘린더 읽기·쓰기 |
| `drive` | **드라이브 전체** 읽기·쓰기·삭제 |
| `contacts.readonly` | 주소록 읽기 |
| `spreadsheets` | 시트 읽기·쓰기 |
| `documents` | 문서 읽기·쓰기 |

제일 무거운 건 `drive` 입니다. `drive.file`(앱이 만든 파일만) 이 아니라 **드라이브 전체**예요.
시트 하나 쓰겠다고 붙였는데, 같은 토큰으로 10년치 개인 파일을 읽고 지울 수 있는 상태가 됩니다.

<br>

<br>

## 3. 그러면 무엇으로 막나 — 계정을 가른다

스코프로 못 좁히니, 막을 수 있는 자리는 **스코프 위쪽**뿐이에요. 권한의 범위를 못 줄이면, 그 권한이 닿는 **대상**을 줄이는 겁니다.

| 방법 | 효과 | 비용 |
|---|---|---|
| **전용 Google 계정** | 개인 메일·드라이브가 아예 시야 밖 | 계정 하나 더 |
| 공유로만 접근 | 에이전트에게 줄 문서만 그 계정에 공유 | 공유 한 번씩 |
| 전용 폴더 | 산출물이 한곳에 모여 회수·감사 쉬움 | 없음 |
| 테스트 사용자 유지 | 앱을 게시하지 않으면 그 계정만 동의 가능 | 토큰 7일 만료 |

제가 권하는 조합은 **전용 계정 + 공유로만 접근** 입니다.

원리가 단순해서 좋아요. `drive` 스코프가 전체를 열어도, **그 계정의 드라이브가 비어 있으면 열 게 없습니다.**
에이전트가 봐야 할 문서만 그 계정에 공유하면, 실질 권한이 공유한 만큼으로 줄어요.
스코프를 못 줄인 걸 소유권으로 대신 줄이는 셈입니다.

> 📌 **토큰 만료 한 가지** — 앱이 "테스트" 상태면 refresh token 이 **7일**만 살아요.
> 계속 쓰려면 OAuth 동의 화면을 "게시(In production)" 로 올려야 합니다.
> 개인 프로젝트라 심사는 필요 없지만, 미검증 앱 경고 화면은 한 번 지나가야 해요.
> 반대로 **7일마다 강제로 재인증되는 게 안전장치**라고 보고 테스트 상태로 두는 선택도 합리적입니다.

<br>

<br>

## 4. 준비 — OAuth 클라이언트 만들기

Google Cloud 콘솔에서 한 번만 하면 되는 작업이에요.

| 순서 | 할 일 |
|---|---|
| 1 | 프로젝트 생성 또는 선택 |
| 2 | API 라이브러리에서 API 활성화 |
| 3 | 사용자 인증 정보 → OAuth 2.0 클라이언트 ID |
| 4 | 애플리케이션 유형 **데스크톱 앱** |
| 5 | 테스트 상태면 내 계정을 테스트 사용자로 추가 |
| 6 | JSON 다운로드 |

활성화할 API 는 6개입니다.

| API | 왜 |
|---|---|
| Gmail API | 메일 |
| Google Calendar API | 일정 |
| Google Drive API | 파일 |
| Google Sheets API | 시트 |
| Google Docs API | 문서 |
| People API | 주소록 |

> ⚠️ 하나라도 빠뜨리면 인증은 성공하는데 **그 API 만 403** 이 납니다.
> `Access Not Configured` 가 뜨면 스코프 문제가 아니라 이 6개 중 하나를 안 켠 거예요.
> 인증을 다시 하지 말고 콘솔에서 API 부터 켜세요.

유형을 반드시 **데스크톱 앱** 으로 잡아야 합니다. 웹 애플리케이션으로 만들면 리디렉션 URI 가 달라서 뒤 단계가 어긋나요.

<br>

<br>

## 5. 셋업 — 헤드리스 서버에서 OAuth 끝내기

Hermes 는 컨테이너 안에서 돌고 브라우저가 없어요. 그래서 셋업이 **전부 비대화형**으로 설계돼 있습니다.
사람이 브라우저를 열고, 결과 URL 만 에이전트에게 돌려주는 구조예요. Discord 나 Telegram 으로도 되는 이유가 이겁니다.

명령이 길어서 먼저 줄임말을 잡습니다.

```shell
GSETUP='docker exec -u hermes hermes python
  /opt/data/skills/productivity/google-workspace/scripts/setup.py'
```

### `-u hermes` 가 필요한 이유

`docker exec` 는 기본이 root 인데, Hermes 데이터는 `hermes`(uid 10000) 소유예요.

```shell
docker exec hermes stat -c '%u:%g %U:%G' /opt/data
# 10000:10000 hermes:hermes
```

root 로 셋업을 돌리면 `google_token.json` 이 **root 소유로 생성됩니다.**
그러면 정작 게이트웨이(hermes 유저)가 그 토큰을 못 읽어요.
인증은 분명히 성공했는데 에이전트는 계속 "인증 안 됨" 이라고 말하는, 원인 찾기 고약한 상태가 됩니다.

> 📌 **`-u hermes` 를 붙이세요.** 이미 root 로 만들어버렸다면
> `docker exec hermes chown hermes:hermes /opt/data/google_token.json` 으로 고칩니다.

### 1단계 — 이미 돼 있나 확인

```shell
$GSETUP --check
```

아직이면 이렇게 나와요.

```text
NOT_AUTHENTICATED: No token at /opt/data/google_token.json
```

`AUTHENTICATED` 가 뜨면 이미 끝난 상태니 7장으로 건너뛰면 됩니다.

### 2단계 — 클라이언트 JSON 등록

다운로드한 JSON 을 컨테이너가 읽을 수 있는 자리에 두고 등록합니다.

```shell
docker cp ~/Downloads/client_secret_xxx.json hermes:/tmp/cs.json
docker exec hermes chown hermes:hermes /tmp/cs.json
$GSETUP --client-secret /tmp/cs.json
```

> ⚠️ **Hermes CLI·Discord 로 대화하며 진행할 때의 함정** —
> `/` 로 시작하는 경로를 **단독 메시지**로 보내면 슬래시 커맨드로 오인됩니다.
> `/tmp/cs.json` 만 덩그러니 보내지 말고 문장 안에 넣어 보내세요.
> 예: `파일 경로는 /tmp/cs.json 이야`

### 3단계 — 동의 URL 받기

```shell
$GSETUP --auth-url
```

출력된 URL 을 **호스트 브라우저**에서 엽니다. 같은 URL 이 파일로도 저장돼요.

```shell
docker exec hermes cat /opt/data/google_oauth_last_url.txt
```

### 4단계 — 리디렉션 URL 통째로 되돌려주기

승인하면 브라우저가 `http://localhost:1/?code=...` 로 이동하면서 **연결 실패 화면**이 뜹니다.

> ✅ **이 실패는 정상입니다.** 컨테이너에는 그 포트를 받을 서버가 없으니까요.
> 중요한 건 화면이 아니라 **주소창**이에요. 주소창의 URL 을 **통째로** 복사하세요.

```shell
$GSETUP --auth-code "http://localhost:1/?code=4/0A...&scope=..."
```

코드만 떼어 넣어도 되지만, 통째로 주는 쪽이 안전합니다. 스코프 정보까지 같이 넘어가거든요.

| 증상 | 원인 | 조치 |
|---|---|---|
| `Error 403: access_denied` | 테스트 사용자 미등록 | 콘솔에서 내 계정 추가 |
| 코드 만료·재사용 | 탭을 오래 열어둠 | 3단계부터 다시 |
| 여러 탭 혼용 | 옛 탭의 코드 사용 | **가장 최근** 리디렉션만 사용 |

코드가 만료되면 스크립트가 새 URL(`fresh_auth_url`)을 같이 돌려줘요. 그 URL 로 다시 하면 됩니다.

### 5단계 — 검증

```shell
$GSETUP --check
$GSETUP --check-live
```

`--check` 는 토큰 파일만 보고, `--check-live` 는 **실제 API 를 한 번 호출**합니다.
클라이언트가 비활성화된 경우처럼 파일만 봐서는 모르는 상태를 잡아줘요. 둘 다 돌리는 걸 권합니다.

<br>

<br>

## 6. 저장되는 파일들

셋업이 만드는 파일을 알아두면 문제 생겼을 때 빠릅니다. 그리고 **2편에서 전부 다시 만납니다** — 하나도 Git 에 올리면 안 되는 것들이거든요.

| 파일 | 내용 | 성격 |
|---|---|---|
| `google_token.json` | 액세스·리프레시 토큰 | 🔴 비밀 |
| `google_client_secret.json` | OAuth 클라이언트 자격증명 | 🔴 비밀 |
| `google_oauth_pending.json` | 교환 전 PKCE 세션 | 🟡 임시 |
| `google_oauth_last_url.txt` | 마지막 동의 URL | 🟡 임시 |

전부 `~/.hermes/` 아래에 생깁니다. 토큰은 이후 **자동 갱신**되니 다시 만질 일은 없어요.

되돌리려면 이거 하나면 됩니다.

```shell
$GSETUP --revoke
```

<br>

<br>

## 7. Sheets 실전

이제 본론이에요. 모든 호출은 `google_api.py` 하나를 지나갑니다.

```shell
GAPI='docker exec -u hermes hermes python
  /opt/data/skills/productivity/google-workspace/scripts/google_api.py'
```

서브커맨드는 6개입니다.

```shell
$GAPI --help
# usage: google_api.py [-h] {gmail,calendar,drive,contacts,sheets,docs} ...
```

### 만들기

```shell
$GAPI sheets create --title "Q4 Budget"
$GAPI sheets create --title "Inventory" --sheet-name "Stock"
```

돌아오는 JSON 에 `spreadsheetId` 가 있어요. 이후 모든 명령이 이 ID 를 씁니다.

| 필드 | 쓰임 |
|---|---|
| `spreadsheetId` | 이후 명령의 `SHEET_ID` |
| `title` | 문서 제목 |
| `spreadsheetUrl` | 브라우저로 열 주소 |

### 읽기

```shell
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"
```

결과는 2차원 배열입니다. 여기에 **함정이 하나** 있어요.

> ⚠️ **빈 칸은 배열을 짧게 만듭니다.**
> 행 끝의 빈 셀은 그냥 생략돼요. `A1:D10` 을 읽었다고 모든 행이 길이 4 로 오지 않습니다.
> 길이를 믿고 `row[3]` 으로 접근하면 인덱스 에러가 납니다.
> 에이전트에게 집계를 시킬 거라면 이 점을 프롬프트에 박아두세요.

### 쓰기와 붙이기

```shell
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" \
  --values '[["Name","Score"],["Alice","95"]]'

$GAPI sheets append SHEET_ID "Sheet1!A:C" \
  --values '[["new","row","data"]]'
```

둘의 차이가 중요합니다.

| 명령 | 동작 | 위험 |
|---|---|---|
| `update` | 지정 범위를 **덮어씀** | 🔴 기존 값 소실 |
| `append` | 표 **끝에 행 추가** | 🟢 낮음 |

> 📌 **에이전트에게 맡길 땐 `append` 를 기본으로.**
> `update` 는 범위를 한 칸 잘못 잡으면 남의 데이터를 덮습니다.
> 되돌릴 방법은 시트의 버전 기록뿐이고, 에이전트는 그걸 자동으로 확인하지 않아요.

<br>

<br>

## 8. Docs 실전 — 그리고 탭 함정

```shell
$GAPI docs create --title "Meeting Notes"
$GAPI docs create --title "Draft" --body "First paragraph..."
$GAPI docs append DOC_ID --text "Additional content"
$GAPI docs get DOC_ID
```

서브커맨드는 `get`, `create`, `append` 세 개뿐이에요. **중간 수정이나 서식 지정은 없습니다.**
할 수 있는 건 새로 만들고, 읽고, 끝에 붙이는 것까지예요. 보고서 초안을 쌓는 용도로는 충분합니다.

### 탭 문서 함정

Google Docs 에 탭 기능이 생기면서 응답 모양이 갈라졌어요.

| 문서 종류 | 응답 | `append` |
|---|---|---|
| 단일 탭·옛 문서 | `body` | 그냥 됨 |
| 탭 여러 개 | `tabs` 배열 | `--tab` **필수** |

```shell
$GAPI docs get DOC_ID --tab TAB_ID
$GAPI docs append DOC_ID --tab TAB_ID --text "..."
```

> ⚠️ 탭이 여러 개인 문서에 `--tab` 없이 `append` 하면 실패합니다.
> 그런데 이게 **읽을 때는 조용히 넘어가요** — `body` 가 없고 `tabs` 만 오니까,
> 파싱을 대충 하면 "내용이 비어 있다" 로 읽힙니다.
> 문서를 읽었는데 빈 것 같으면 탭 문서인지부터 확인하세요.

<br>

<br>

## 9. 스킬이 스스로 거는 규칙 — 그리고 믿으면 안 되는 이유

`SKILL.md` 에는 에이전트가 지켜야 할 규칙이 적혀 있어요.

| 규칙 | 내용 |
|---|---|
| 1 | 메일 발송·일정 생성/삭제·파일 삭제/공유·문서 수정 전 **사용자 확인** |
| 2 | 첫 사용 전 `--check` 로 인증 확인 |
| 3 | 복잡한 검색은 Gmail 검색 문법 레퍼런스 참조 |
| 4 | 캘린더 시각은 **반드시 타임존 포함** ISO 8601 |
| 5 | 속도 제한 존중 — 연속 호출 자제 |

1번은 실제로 잘 지켜지는 편이에요. `drive delete` 도 기본이 **휴지통**이라 되돌릴 수 있고요.

```shell
$GAPI drive delete FILE_ID              # trash (reversible)
$GAPI drive delete FILE_ID --permanent  # permanent
```

그런데 여기서 한 발 물러나 봐야 합니다.

> 🚨 **이 규칙들은 전부 프롬프트입니다.**
> 코드가 막는 게 아니라 모델에게 하지 말라고 적어둔 문장이에요.
> 컨텍스트가 길어지거나, 압축이 일어나거나, 다른 스킬 지시와 겹치면 **약해집니다.**
> 자동화(크론·웹훅)로 사람 없이 돌 때는 "확인을 받으라"는 규칙 자체가 성립하지 않고요.

그래서 정리하면 이렇습니다.

| 층 | 막는 주체 | 신뢰도 |
|---|---|---|
| 스킬 규칙 | 모델 | 🟡 보조 |
| 휴지통 기본값 | 코드 | 🟢 견고 |
| **전용 계정** | Google | 🟢 견고 |
| **공유 범위** | Google | 🟢 견고 |

**진짜 방어선은 3장의 계정 분리입니다.** 스킬 규칙은 그 위에 얹는 보조 장치로 보세요.

<br>

<br>

## 10. 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `NOT_AUTHENTICATED` | 토큰 없음 | 5장 1~5단계 |
| `REFRESH_FAILED` | 토큰 폐기·만료 | 3~5단계 재실행 |
| `403 Insufficient<br>Permission` | 스코프 부족 | `--revoke` 후<br>재인증 |
| `AUTHENTICATED (partial)` | 스코프가<br>늘어남 | `--revoke` 후<br>재인증 |
| `403 Access Not<br>Configured` | API 미활성화 | 콘솔에서<br>API 켜기 |
| `ModuleNotFoundError` | 의존성 없음 | `$GSETUP --install-deps` |
| 인증했는데 계속 실패 | 토큰이 root 소유 | `chown hermes:hermes` |
| 7일 만에 끊김 | 앱이 테스트 상태 | 동의 화면 게시 |
| 문서가 빈 것처럼 읽힘 | 탭 문서 | `--tab` 사용 |

앞의 네 줄이 헷갈리기 쉬워요. 구분이 이렇습니다.

> **`403 Insufficient Permission`** 은 내가 **토큰에 없는 권한**을 쓰려는 것 — 재인증이 답.
> **`403 Access Not Configured`** 는 **API 자체가 꺼져 있는 것** — 콘솔이 답.
> 둘 다 403 이라 같은 문제로 보이는데, 고치는 자리가 완전히 다릅니다.

<br>

<br>

## 11. 정리

| 항목 | 결론 |
|---|---|
| 메일만 필요 | `himalaya` 로. 이 스킬 불필요 |
| 문서 최신성 | `--services`·`--format` 은 **없음**. 문서가 앞서감 |
| 실제 권한 | 8개 스코프 고정. `drive` 는 **전체** |
| 진짜 방어선 | 전용 계정 + 공유 범위 |
| 컨테이너 실행 | `-u hermes` 필수 |
| 시트 기본값 | `append` 우선, `update` 는 신중히 |
| 문서 한계 | 만들기·읽기·붙이기만. 중간 수정 없음 |

한 줄로 줄이면 이렇습니다.

> **스코프로 못 줄이면 소유권으로 줄인다.**
> 권한의 크기를 못 건드릴 때는, 그 권한이 닿는 대상을 비워두는 게 유일하게 확실한 방법이에요.

그리고 이 모든 설정 — 토큰, 클라이언트 자격증명, 스킬, `config.yaml` — 은 전부 `~/.hermes` 한 디렉토리에 쌓입니다.
632MB 짜리 디렉토리에 **되돌릴 수 없는 설정과 절대 새면 안 되는 비밀이 섞여 있는** 상태예요.

2편에서는 이걸 Git 으로 형상관리합니다. 비밀은 빼고, 되돌릴 건 남기고요.
`config.yaml` 이 생각만큼 깨끗하지 않다는 것도 거기서 확인합니다.

<br>

{% include series-hermes-app-connect.html current="1" %}
