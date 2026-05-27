---
layout: single
title:  "GitHub SSH 키 등록 — ed25519 한 짝으로 통일하기"
description: "GitHub 인증을 HTTPS 토큰 대신 ed25519 SSH 키로 바꾸는 방법을 정리했어요. 키 한 짝으로 GitHub와 서버 접근까지 통일합니다."
categories: coding
tag: [github, ssh, ed25519, ssh-keygen, git]
author_profile: false
toc: true
---



## Summary

GitHub 인증을 HTTPS + Personal Access Token 대신 **SSH 키**로 바꾸는 글이에요.   
한 번 셋업해두면 `git push` 할 때마다 비밀번호/토큰을 안 물어봐서 정말 편해집니다. 

키는 **ed25519** 한 짝만 만들어서 GitHub 용과 서버 접근용으로 같이 씁니다.   
키 관리가 단순해지고, 분실/교체 시에도 신경 쓸 곳이 줄어들어요. 

> 💡 **이 글에서 다루는 것**
> - ed25519 SSH 키 생성하기
> - GitHub에 공개키 등록
> - 기존 HTTPS 리모트를 SSH 로 갈아끼우기
> - `ssh -T git@github.com` 으로 동작 검증

<br>

<br>



## 1. 사전 확인

먼저 로컬에 SSH 클라이언트가 있는지, 기존 키가 있는지 확인합니다. 

```shell
ssh -V
# OpenSSH_9.x ...

ls -al ~/.ssh
```

`id_ed25519` 또는 `id_rsa` 같은 파일이 이미 있다면, 기존 키를 그대로 써도 되고 새로 만들어도 돼요. 

| 항목 | 값 |
|------|-----|
| 키 타입 | ed25519 (짧고, 빠르고, 안전함) |
| 위치 | `~/.ssh/id_ed25519` (개인키), `~/.ssh/id_ed25519.pub` (공개키) |
| 패스프레이즈 | 강력 추천 (분실 시 안전망) |

> ✅ **왜 ed25519?**  
> RSA 4096 보다 키 길이가 훨씬 짧고(서명/검증 빠름), 보안 강도는 동등 이상이에요.  
> GitHub 도 공식 권장입니다. 새로 만든다면 ed25519 한 줄로 갑니다.

<br>

<br>



## 2. SSH 키 생성

ed25519 로 새 키를 만듭니다. 

```shell
ssh-keygen -t ed25519 -C "dorumugs@gmail.com"
```

`-C` 옆에 들어가는 코멘트는 식별용이라 본인 이메일/머신명 무엇이든 OK 입니다.   
물어보는 항목은 기본값(`Enter`)으로 넘기되, **패스프레이즈는 꼭 입력하는 걸 추천드려요**.   
키 파일이 통째로 유출돼도 패스프레이즈가 있으면 한 단계 더 막아줍니다. 

생성이 끝나면 두 파일이 생겨요. 

```shell
ls -al ~/.ssh/id_ed25519*
# -rw-------  1 dorumugs  staff   411  ...  id_ed25519
# -rw-r--r--  1 dorumugs  staff    99  ...  id_ed25519.pub
```

> ⚠️ **권한 주의**  
> 개인키(`id_ed25519`)는 반드시 `600`, `.ssh` 디렉토리는 `700` 이어야 SSH 가 동작합니다.   
> 권한이 헐겁다 싶으면 `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519` 한 번 돌려두세요.

<br>

<br>



## 3. GitHub에 공개키 등록

공개키 내용을 복사합니다. 

```shell
cat ~/.ssh/id_ed25519.pub
# ssh-ed25519 AAAAC3Nza... dorumugs@gmail.com
```

이 한 줄 전체를 그대로 복사한 다음, GitHub 에 등록합니다. 

1. GitHub → 우측 상단 프로필 → **Settings**
2. 좌측 메뉴 → **SSH and GPG keys**
3. **New SSH key** 클릭
4. Title 은 식별용(`MacBook Pro - 2026`), Key type 은 `Authentication Key`, Key 칸에 위 공개키 한 줄 붙여넣기

> 💡 여러 머신을 쓴다면 머신마다 다른 키를 만들어 각각 등록하는 게 깔끔합니다.   
> 한 머신을 분실/폐기할 때 그 키만 GitHub 에서 빼면 되니까요.

<br>

<br>



## 4. 기존 repo 의 remote 를 SSH 로 갈아끼우기

기존 repo가 HTTPS 로 클론되어 있다면 remote URL 만 SSH 로 바꿔주면 됩니다. 

```shell
# 현재 remote 확인
git remote -v
# origin  https://github.com/dorumugs/dorumugs.github.io.git (fetch)
# origin  https://github.com/dorumugs/dorumugs.github.io.git (push)

# SSH 로 변경
git remote set-url origin git@github.com:dorumugs/dorumugs.github.io.git

# 다시 확인
git remote -v
# origin  git@github.com:dorumugs/dorumugs.github.io.git (fetch)
# origin  git@github.com:dorumugs/dorumugs.github.io.git (push)
```

`https://github.com/<user>/<repo>.git` → `git@github.com:<user>/<repo>.git` 으로 바뀌는 게 포인트예요.   
처음 클론하는 경우라면 처음부터 SSH URL 로 받으면 됩니다. 

```shell
git clone git@github.com:dorumugs/dorumugs.github.io.git
```

<br>

<br>



## 5. 동작 검증

GitHub 에 키가 잘 붙었는지 확인합니다. 

```shell
ssh -T git@github.com
# The authenticity of host 'github.com (...)' can't be established.
# ...
# Hi dorumugs! You've successfully authenticated, but GitHub does not provide shell access.
```

`Hi <username>! You've successfully authenticated` 이 줄이 보이면 통과예요. 

이제 실제로 push 도 해보세요. 

```shell
git commit --allow-empty -m "test: ssh push"
git push
# Enumerating objects: ...
# To github.com:dorumugs/dorumugs.github.io.git
#    abc1234..def5678  main -> main
```

비밀번호/토큰 입력 없이 한 번에 통과해야 정상입니다. 

> ✅ 만약 `Permission denied (publickey)` 가 나오면 다음을 점검하세요.  
> - 키가 ssh-agent 에 등록돼 있는지: `ssh-add -l`  
> - GitHub 등록한 공개키가 `id_ed25519.pub` 내용과 정확히 일치하는지  
> - 권한이 `600` / `700` 으로 잘 잡혀있는지

<br>

<br>



## 6. ssh-agent 에 키 등록 (선택)

패스프레이즈를 매번 묻는 게 귀찮으면 ssh-agent 에 등록해두면 됩니다. 

```shell
# macOS
ssh-add --apple-use-keychain ~/.ssh/id_ed25519

# Linux
ssh-add ~/.ssh/id_ed25519
```

macOS 는 Keychain 과 연동돼서 로그인할 때 자동으로 ssh-agent 가 키를 들고 있어요.   
재부팅 후에도 살아있게 하려면 `~/.ssh/config` 에 다음을 추가합니다. 

```text
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

<br>

<br>



## 7. 주의 사항

> 🚨 **공개키는 어디에 올려도 OK, 개인키는 어디에도 올리면 안 됨.**  
> `id_ed25519` (확장자 `.pub` 없는 쪽) 은 본인 컴퓨터 밖으로 나가면 안 돼요.   
> 깃 저장소, 슬랙, 이메일, 클립보드 공유 서비스 모두 위험합니다.

- GitHub 에 등록한 공개키는 GitHub Settings 에서 언제든 회수할 수 있어요. 분실 의심되면 일단 회수 → 새 키 생성.
- 패스프레이즈는 키 파일 유출 시 마지막 방어선입니다. 빈 패스프레이즈는 가급적 피하세요.

<br>

<br>



일단 오늘은 여기까지.....   
다음 글에서는 같은 ed25519 키를 그대로 써서 **개인 서버에 SSH 로 접근**하고, root 로그인 차단 + LocalForward 까지 잡는 흐름을 정리해볼게요. 
