---
layout: single
title:  "(2/2) SSH 로 서버 접근하기 — config 별칭, root 차단, LocalForward"
date: 2026-05-27 08:00:00 +0900
description: "SSH config 별칭으로 내·외부망 접근을 통일하고, root 로그인 차단과 LocalForward로 원격 대시보드를 로컬 포트로 끌어오는 구성을 정리했어요."
categories: coding
tag: [ssh, sshd, port-forwarding, security, openssh, localforward]
author_profile: false
toc: true
header:
  image: /assets/images/2026-05-27-ssh-server-access/header.svg
  teaser: /assets/images/2026-05-27-ssh-server-access/header.svg
series: ssh
series_order: 2
---



{% include series-ssh.html current="2" %}

## Summary

[지난 글](/coding/GitHub_SSH_키_등록/)에서 만든 ed25519 키를 그대로 써서   
집에 있는 미니 서버 한 대에 **내부망에서도**, **외부망에서도** 같은 별칭처럼 붙는 구성을 정리합니다. 

여기에 더해서, 서버 쪽은 **root 로그인 / 패스워드 인증을 차단**해두고,   
서버 위에서 돌고 있는 n8n / Prefect / Hermes 대시보드는 **`LocalForward` 로 내 노트북의 localhost 포트에 그대로 끌어와서** 쓰는 구성이에요. 

한 번 셋업해두면 `ssh myserver` 한 방으로 서버에 붙으면서   
`http://localhost:5678` 으로 원격 n8n 대시보드까지 같이 열려서 정말 편해져요. 

> 💡 **이 글에서 다루는 것**
> - `~/.ssh/config` 로 내부망 / 외부망 별칭 만들기
> - 서버에 공개키 올리고 키 인증 확인
> - sshd 에서 root 로그인 + 패스워드 인증 차단
> - `LocalForward` 로 원격 서비스를 로컬 포트로 끌어오기
> - 셋업 끝나고 동작 검증 체크리스트

<br>

<br>



## 1. 사전 준비

이 글은 이미 ed25519 키 한 짝(`~/.ssh/id_ed25519`, `~/.ssh/id_ed25519.pub`)이 있다는 전제로 진행해요.   
아직 없다면 [GitHub SSH 키 등록 글](/coding/GitHub_SSH_키_등록/)에서 `ssh-keygen` 부분만 먼저 따라하고 오시면 돼요. 

서버 쪽 사전 조건은 아래와 같아요. 

| 항목 | 값 |
|------|-----|
| OS | Linux (예시는 Ubuntu/Debian 계열 기준) |
| 일반 유저 | `dorumugs` (sudo 가능) |
| 패키지 | `openssh-server` 설치 + 실행 중 |
| 네트워크 | 내부망 사설 IP 있음, (선택) 외부 공유기에서 포트포워딩 가능 |

<br>

<br>



## 2. `~/.ssh/config` 로 서버 별칭 만들기

한 대의 미니 서버를 **내부망에서는 빠른 사설 IP**로, **밖에 나가있을 때는 공인 IP + 다른 포트**로 접근하고 싶어요.   
매번 `ssh -i ... -p ... user@호스트` 치는 건 너무 길어서, `~/.ssh/config` 에 두 가지를 별칭으로 박아둡니다. 

파일 경로 : `~/.ssh/config`

```text
Host myserver
    HostName 192.168.x.x
    User dorumugs
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 18789 localhost:18789
    LocalForward 5678  localhost:5678
    LocalForward 4200  localhost:4200

Host myserver-ext
    HostName 222.233.x.x
    User dorumugs
    Port 1234
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 18789 localhost:18789
    LocalForward 5678  localhost:5678
    LocalForward 4200  localhost:4200
```

> 💡 위 예시의 `192.168.x.x` / `222.233.x.x` 자리에는 본인 환경의 사설/공인 IP 를 넣어주세요. (글에서는 일부 옥텟을 가렸어요)

### 항목별 의미

| 키 | 의미 |
|------|------|
| `Host` | 별칭. `ssh myserver` 처럼 부르는 이름 |
| `HostName` | 실제 IP / 도메인 |
| `User` | 원격 계정명 |
| `Port` | 원격 sshd 포트. 기본 22 외라면 명시 |
| `IdentityFile` | 사용할 개인키 경로 |
| `LocalForward` | 내 PC의 포트 → 원격에서 본 어떤 포트로 터널링 |

`config` 파일은 권한도 잡아주세요. 

```shell
chmod 600 ~/.ssh/config
```

이제 접속이 짧아졌습니다. 

```shell
# 집 와이파이일 때
ssh myserver

# 밖에서 (외부망)
ssh myserver-ext
```

비밀번호 안 물어보고 한 방에 들어가야 정상이에요.   
(키 등록을 아직 안 했다면 다음 단계에서 같이 해요.)

> 💡 키 파일을 GitHub용과 서버용으로 분리하고 싶다면 `~/.ssh/id_ed25519_server` 같이 따로 만들고 `IdentityFile` 만 바꿔주면 됩니다. 키 한 짝이 유출됐을 때 범위를 줄이는 효과가 있어요.

<br>

<br>



## 3. 원격 서버에 공개키 올리기

서버 쪽에서 인증을 받으려면 내 공개키가 서버의 `~/.ssh/authorized_keys` 에 들어있어야 합니다.   
가장 편한 방법은 `ssh-copy-id` 입니다. 

```shell
# 초기에는 아직 패스워드로 들어가야 해요 (이 단계까지만)
ssh-copy-id -i ~/.ssh/id_ed25519.pub dorumugs@192.168.x.x
```

만약 `ssh-copy-id` 가 없거나 동작이 이상하면 수동으로도 됩니다. 

```shell
cat ~/.ssh/id_ed25519.pub | ssh dorumugs@192.168.x.x \
  'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

등록이 끝나면 이제 키만으로 들어가지는지 다시 한 번 확인. 

```shell
ssh myserver "whoami && hostname"
# dorumugs
# myserver-host
```

비밀번호를 한 번도 안 물어봤다면 OK 입니다. 

<br>

<br>



## 4. 서버 보안 강화 — root 로그인 + 패스워드 인증 차단

이제 키 인증이 잘 되니까, **패스워드 인증과 root 로그인 자체를 막아둡니다**.   
공인 IP 가 열려있는 서버라면 이 단계는 사실상 필수예요. 

서버에 접속한 뒤(`ssh myserver`), sshd 설정을 엽니다. 

```shell
sudo vi /etc/ssh/sshd_config
```

다음 항목을 찾아서 값을 바꿔주세요. 주석(`#`) 이 붙어있다면 같이 풀어줍니다. 

```text
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
ChallengeResponseAuthentication no
UsePAM yes
```

| 옵션 | 권장값 | 의미 |
|------|--------|------|
| `PermitRootLogin` | `no` | root 계정으로 직접 SSH 접속 금지 |
| `PasswordAuthentication` | `no` | 비밀번호 로그인 금지 (키만 허용) |
| `PubkeyAuthentication` | `yes` | 공개키 인증 허용 |
| `ChallengeResponseAuthentication` | `no` | 챌린지/응답 기반 비밀번호 로그인 차단 |

> 🚨 **반드시 지금 열려있는 SSH 세션은 그대로 두고** 새 터미널로 다시 접속해보세요.  
> 설정이 잘못된 채로 sshd 를 재시작 + 현재 세션도 끊어버리면 서버에 못 들어가는 참사가 생깁니다.

설정 문법이 맞는지 먼저 검증합니다. 

```shell
sudo sshd -t
# (출력이 없으면 OK)
```

OK 면 적용. 

```shell
sudo systemctl restart ssh
# 배포판에 따라 sshd 일 수도 있어요
# sudo systemctl restart sshd
```

새 터미널에서 접속이 여전히 잘 되는지, 그리고 root 가 막혔는지 확인합니다. 

```shell
# 일반 유저는 여전히 OK
ssh myserver "whoami"
# dorumugs

# root 는 거부돼야 정상
ssh root@192.168.x.x
# Permission denied (publickey).
```

> ✅ 추가로 SSH 포트를 22 외(예: `1234`) 로 바꿔두면 자동화된 스캐닝이 확 줄어요.  
> 이 글의 `myserver-ext` 가 `Port 1234` 로 잡혀있는 게 그 이유입니다. 

<br>

<br>



## 5. `LocalForward` 가 진짜 편한 이유

여기가 이 글의 하이라이트예요.   
서버 위에 돌고 있는 n8n / Prefect / Hermes 대시보드들이 **외부망에 노출돼 있지 않고** 내부 서비스 포트로만 떠있다고 가정합니다.   

```text
[원격 서버]
  - n8n         → localhost:5678
  - Prefect UI  → localhost:4200
  - Hermes API  → localhost:18789
```

원래라면 이걸 브라우저로 보려면 서버 안에서 GUI 쓰거나, nginx 같은 리버스 프록시로 외부에 열어야 하잖아요.   
그런데 `LocalForward` 하나로 **내 PC의 같은 포트로 그대로 끌어올 수 있어요**. 

config 의 이 부분이 그 역할을 합니다. 

```text
LocalForward 18789 localhost:18789
LocalForward 5678  localhost:5678
LocalForward 4200  localhost:4200
```

해석하면 

> "내 PC의 `localhost:5678` 로 들어오는 트래픽을 SSH 터널을 통해 원격 서버가 보는 `localhost:5678` 로 보내라"

라는 뜻이에요. 

### 동작 순서

1. `ssh myserver` 로 접속하면 SSH 세션이 만들어집니다.
2. 동시에 내 PC에서 `localhost:5678`, `localhost:4200`, `localhost:18789` 가 listen 상태가 됩니다.
3. 브라우저에서 `http://localhost:5678` 을 열면 → SSH 터널을 타고 → 원격 서버의 `localhost:5678` 에 도달.
4. n8n 입장에서는 "localhost 에서 들어오는 정상 요청" 이라 추가 인증/방화벽 이슈 없이 그냥 응답합니다.

### 검증

서버에 접속한 상태에서 (`ssh myserver` 가 떠있는 상태) 새 브라우저 탭이나 새 터미널에서 

```shell
curl -s http://localhost:5678/healthz
# {"status":"ok"}

curl -s http://localhost:4200/api/health
# true
```

이렇게 응답이 오면 터널이 살아있는 거예요.   
브라우저에서 `http://localhost:5678` 직접 열어도 n8n UI 가 그대로 떠야 정상입니다. 

> 💡 한 가지 헷갈리는 포인트:  
> `LocalForward 5678 localhost:5678` 에서  
> - 첫 번째 `5678` = **내 PC** 의 포트  
> - `localhost:5678` 의 `localhost` = **원격 서버 입장에서의 localhost**  
> 두 번째 자리의 `localhost` 가 *내 PC가 아니라는 점* 이 핵심이에요.

<br>

<br>



## 6. 동작 검증 체크리스트

여기까지 왔으면 한 번에 같이 점검해보세요. 

- [x] `ssh myserver` 한 방으로 서버 접속 (내부망)
- [x] `ssh myserver-ext` 로 외부망에서도 접속 (필요 시)
- [x] `ssh root@<서버IP>` → `Permission denied (publickey)` 로 거부
- [x] 패스워드만으로 시도 (`ssh -o PubkeyAuthentication=no ...`) → 거부
- [x] 브라우저에서 `http://localhost:5678` → n8n UI 로딩
- [x] 브라우저에서 `http://localhost:4200` → Prefect UI 로딩
- [x] `curl http://localhost:18789/...` → Hermes API 응답 🎉

<br>

<br>



## 7. 자주 만나는 함정 & 대처

운영하면서 한 번씩 만나는 자잘한 이슈들도 같이 정리해둡니다. 

### "Could not resolve hostname myserver"

`~/.ssh/config` 를 못 읽고 있을 가능성이 큽니다.   
파일 위치(`~/.ssh/config`), 권한(`600`), 들여쓰기(스페이스 4칸, 탭 X)를 다시 봐주세요. 

### "Permissions 0644 for '~/.ssh/id_ed25519' are too open"

개인키 권한이 헐거우면 SSH 가 아예 거부합니다. 

```shell
chmod 600 ~/.ssh/id_ed25519
```

### `LocalForward` 포트가 안 열려요

이미 같은 포트를 내 PC의 다른 프로세스가 점유하고 있는 경우가 많아요.   

```shell
lsof -nP -iTCP:5678 -sTCP:LISTEN
```

겹치는 게 있으면 forward 포트를 다른 숫자로 바꿔주면 됩니다.   
예: `LocalForward 15678 localhost:5678` → 브라우저는 `http://localhost:15678` 로 접근.

### 외부망에서 접속 안 됨

공인 IP / 공유기 포트포워딩 / 서버 방화벽 / sshd 의 `Port` 설정, 이 4가지를 순서대로 확인하세요.   
보통은 공유기에서 외부 `1234` 포트를 내부 호스트의 `22` 또는 `1234` 로 매핑하는 부분을 깜빡합니다. 

<br>

<br>



## 8. 마무리 메모

- 키 인증이 안정되면 **반드시 패스워드 / root 로그인은 끄세요**. 공인 IP 환경에서는 봇이 24시간 두드립니다.
- `LocalForward` 는 노트북에 별다른 VPN 클라이언트 깔지 않고도 원격 내부 서비스를 그대로 쓰는 가장 가벼운 방법입니다.
- 한 별칭에 `LocalForward` 를 여러 줄 박아두면, `ssh myserver` 한 방으로 필요한 모든 포트가 같이 열려요.

> 🚨 **공개키는 어디에 올려도 OK, 개인키는 어디에도 올리면 안 됨.**  
> `id_ed25519` (확장자 `.pub` 없는 쪽) 은 본인 컴퓨터 밖으로 나가면 안 돼요.   
> 깃 저장소, 슬랙, 이메일, 클립보드 공유 서비스 모두 위험합니다.

<br>

<br>



일단 오늘은 여기까지.....   
다음 글에서는 같은 SSH 터널 위에 `RemoteForward` 를 얹어서 **반대 방향**(원격 서버에서 내 노트북 서비스를 호출)으로 쓰는 부분을 정리해볼게요.

---

**← 이전 글:** [(1/2) GitHub SSH 키 등록 — ed25519 한 짝으로 통일하기](/coding/GitHub_SSH_키_등록/)

