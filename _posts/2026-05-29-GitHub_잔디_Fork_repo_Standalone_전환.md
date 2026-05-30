---
layout: single
title:  "GitHub 잔디가 안 심어지던 이유 — Fork repo 에서 Standalone 으로 옮기기"
date: 2026-05-29 01:14:00 +0900
description: "블로그 글을 꾸준히 쓰는데 GitHub 잔디가 안 심어지던 이유 — fork 저장소 라서였어요. 카운팅 조건 4가지를 짚고 standalone repo 로 옮기기까지 정리했어요."
categories: coding
tag: [github, github-pages, jekyll, fork, contributions, troubleshooting, kayserdocs]
author_profile: false
toc: true
---

블로그 글을 꾸준히 쓰고 있는데, 정작 GitHub 프로필의 잔디(컨트리뷰션 그래프)는 휑한 상태였어요. "내가 잘못 푸시했나?" 하고 며칠 무시했는데, 알고 보니 **저장소 자체가 fork** 라서 커밋이 잔디로 카운트가 안 되고 있던 거였어요. 이 글은 그 원인을 찾고 standalone repo 로 옮기기까지의 전 과정을 정리한 글이에요.

> 💡 이 글에서 다루는 것
> - GitHub 잔디(컨트리뷰션 그래프) 카운팅의 4가지 조건
> - 왜 fork repo 의 커밋은 잔디로 안 잡히는지
> - GitHub Support 에 "fork 분리" 요청을 보냈더니 거절당한 이야기
> - 안전하게 standalone repo 로 옮기는 실제 절차 (rename → new repo → push → Pages)


<br>

<br>



## 1. 증상 — 잔디가 휑한데 커밋은 매일 쌓이고 있다

상황은 단순했어요.

- 블로그(`https://dorumugs.github.io`)에 글을 거의 매일 쓰고 push 도 잘 됨
- repo `dorumugs/dorumugs.github.io` 의 commit history 에는 제 커밋이 줄줄이 찍힘 (author: `Jaehyun So <do***@gmail.com>`)
- 그런데 `https://github.com/dorumugs` 프로필의 잔디는 **하나도 안 채워짐**

뭔가 한 가지가 어긋났다는 신호인데, 그 한 가지를 정확히 찍어내려면 GitHub 가 잔디 카운팅을 어떤 조건으로 하는지부터 알아야 했어요.


<br>

<br>



## 2. GitHub 잔디 카운팅의 4가지 조건

GitHub 공식 문서에 따르면 커밋이 컨트리뷰션 그래프에 잡히려면 **네 조건 전부** 만족해야 해요. 하나라도 어긋나면 무시당해요.

| 조건 | 설명 |
|---|---|
| 1️⃣ 최근 1년 이내 | 1년 넘게 묵은 커밋은 안 잡힘 |
| 2️⃣ author 이메일이 GitHub 계정에 verified | commit author email 이 본인 계정의 verified email 목록에 있어야 함 |
| 3️⃣ standalone 저장소 | **fork 면 안 잡힘** |
| 4️⃣ default branch (또는 `gh-pages` Pages 브랜치) | 그 외 feature 브랜치 커밋은 잔디로 안 잡힘 |

제 케이스에서 1, 2, 4 는 다 통과했는데, 3 이 문제였어요.


<br>

<br>



## 3. 원인 확인 — 내 repo 가 fork 였다

GitHub REST API 로 한 줄 확인해봤어요.

```shell
curl -s https://api.github.com/repos/dorumugs/dorumugs.github.io | \
  grep -E '"(fork|parent|default_branch)"'
```

결과는 이렇게 떨어졌어요.

```text
  "fork": true,
  "parent": { "full_name": "mmistakes/minimal-mistakes" }
  "default_branch": "gh-pages",
```

✅ 범인 확인 — `"fork": true`.

블로그는 [minimal-mistakes](https://github.com/mmistakes/minimal-mistakes) 테마를 기반으로 시작했는데, 그때 GitHub 의 "Fork" 버튼으로 시작해서 그대로 콘텐츠를 채워나간 거였어요. 시간이 지나면서 테마는 거의 통째로 갈아엎고 내 글이 99% 인 상태가 됐지만, GitHub 가 보기엔 여전히 mmistakes/minimal-mistakes 의 fork 였어요. 그리고 fork 의 커밋은 잔디 정책상 무조건 카운트 제외.

> ⚠️ fork 여부는 repo 페이지 상단의 "forked from mmistakes/minimal-mistakes" 표기로도 확인할 수 있어요. 평소에는 안 보고 지나치게 되니까 본인 repo 가 fork 인지 모르고 있는 경우가 의외로 많아요.


<br>

<br>



## 4. 해결 시도 1 — GitHub Support 에 detach 요청 (실패)

옛날에는 GitHub Support 에 "이 repo 를 parent 와 detach 해주세요" 라고 요청하면 처리해줬다는 글들을 봐서 support 폼을 열었어요. 라우팅은 이런 흐름이에요.

`Contributions` → `Missing contributions` → `Commit` → 상세 폼

제목/본문 다 채우고 제출 직전, GitHub 가 폼 아래에 자동 응답을 띄웠어요.

> 🚨 **Forked repositories don't count toward contribution graphs.**
> This behavior is expected and can't be overridden by Support.
> GitHub Support also can't manually "detach" a repository from its parent fork.
>
> To have future commits count toward your contributions graph, you'll need to move your site to a standalone repository.

요약하면 **정책이 바뀌어서 GitHub 가 더 이상 fork detach 안 해줘요**. 유일한 해결책은 **새 standalone repo 만들어서 옮기기**.


<br>

<br>



## 5. 해결 시도 2 — Rename + 새 repo + push

문제는 제 사이트가 **GitHub Pages user site** 라서 repo 이름이 정확히 `dorumugs.github.io` 여야 한다는 거였어요. 즉 같은 이름을 비워주고 새로 만들어야 하는데, 기존 repo 를 그냥 삭제하면 백업이 없을 때 너무 위험해요.

그래서 좀 더 안전한 순서로 진행했어요.

1. 기존 repo 를 다른 이름으로 **rename** 해서 살려둠 (아카이브)
2. 같은 이름 `dorumugs.github.io` 로 새 standalone repo 생성
3. 로컬에서 origin 갈아끼우고 push
4. 새 repo 에서 Pages 활성화
5. 이메일 verified 확인

URL 도 보존되고 (`https://dorumugs.github.io` 그대로), 잘못되면 archive 로 복구 가능. 실제 진행한 과정 정리할게요.


<br>

<br>



### 5-0. 로컬 백업 — 사고 났을 때 살아남는 보험

뭐든 시작하기 전에 로컬에서 한 번 더 백업.

```shell
git fetch --all --tags
git bundle create ~/dorumugs.github.io-backup-$(date +%Y%m%d).bundle \
  gh-pages \
  refs/remotes/origin/gh-pages \
  refs/remotes/origin/master
git bundle verify ~/dorumugs.github.io-backup-$(date +%Y%m%d).bundle
```

번들 한 파일에 전체 히스토리가 들어가요. 복구할 일이 생기면 `git clone <bundle파일> restored-repo` 한 줄로 되돌릴 수 있어요.

만든 다음에 `verify` 까지 통과시켜놓으면 안심.


<br>

<br>



### 5-1. 기존 repo 를 rename — archive 로 살려두기

GitHub 웹에서:

- `Settings` → `General` → `Repository name`
- `dorumugs.github.io` → `dorumugs.github.io-archive` 로 변경
- `Rename` 클릭

> ⚠️ rename 직후 `https://dorumugs.github.io` 가 잠시 404 가 될 수 있어요. 다음 단계들이 끝나면 다시 살아나요.

이 시점에 원래 URL `github.com/dorumugs/dorumugs.github.io` 는 GitHub 가 자동으로 archive 로 리다이렉트해줘요. 새 repo 를 만드는 순간 그 리다이렉트는 끊기고 새 repo 가 그 이름을 차지.


<br>

<br>



### 5-2. 새 standalone repo 생성

GitHub 웹 `https://github.com/new` 에서 새 repo 생성.

- **Owner**: `dorumugs`
- **Repository name**: `dorumugs.github.io` ← 정확히 같은 이름
- **Public**
- README / .gitignore / license **추가 안 함** (빈 repo 여야 push 가 깔끔)
- ⚠️ template / fork 옵션 절대 건드리지 말기

생성된 repo 페이지가 비어있고 "Quick setup" 가이드만 보이면 정상.


<br>

<br>



### 5-3. 로컬 remote 갈아끼우고 push

여기서부터는 로컬 터미널에서 진행.

```shell
# origin URL 명시적으로 새 repo 가리키게
git remote set-url origin git@github.com:dorumugs/dorumugs.github.io.git

# mmistakes 가리키던 upstream 정리
git remote remove upstream

# 확인
git remote -v
```

브랜치 push.

```shell
git push -u origin gh-pages
git push origin refs/remotes/origin/master:refs/heads/master
```

- `gh-pages` 는 현재 사이트의 본체 브랜치. 이게 본문임
- `master` 는 옛날 워크플로 잔재인데 일부 콘텐츠(이미지 등)가 살아있을 수 있어서 같이 보존
- tags 는 mmistakes 테마 태그라 푸시 안 함 (`git push --tags` 했다가는 mmistakes 의 4.x.x 릴리스 태그들이 통째로 따라옴)

✅ 푸시 후 GitHub API 로 확인했을 때 **커밋 SHA 가 그대로 보존돼야** 해요. 예를 들어 가장 최근 커밋 `798c73fa` 가 새 repo 에서도 동일한 `798c73fa` 로 들어가있어야 GitHub 가 그 커밋을 잔디로 인식해줘요.

```shell
curl -s https://api.github.com/repos/dorumugs/dorumugs.github.io | \
  grep -E '"(fork|default_branch)"'
# "fork": false,
# "default_branch": "gh-pages",
```

`fork: false` 가 나오면 standalone 전환 성공.


<br>

<br>



### 5-4. Pages 활성화

새 repo 의 GitHub 웹에서:

- `Settings` → `Pages`
- **Source**: `Deploy from a branch`
- **Branch**: `gh-pages` / `/(root)`
- `Save`

1~5분 뒤 Actions 탭에서 "pages build and deployment" 가 success 로 끝나면 `https://dorumugs.github.io` 가 다시 살아나요.

확인 포인트.

- [ ] `https://dorumugs.github.io` HTTP 200
- [ ] 글 상세 페이지 한두 개 정상 렌더링
- [ ] 커스텀 도메인 쓰고 있었다면 도메인 설정 + `CNAME` 파일 재확인


<br>

<br>



### 5-5. 이메일 verified 확인 — 자주 빼먹는 단계

`Settings` → `Emails` 에서 commit author 로 쓰는 이메일이 verified 상태인지 확인.

> ⚠️ "Keep my email addresses private" 가 켜져 있으면 push 할 때 실제 이메일 대신 `*****+dorumugs@users.noreply.github.com` 으로 마스킹돼서 들어가요. 그러면 잔디 안 잡혀요. 끄거나, git config 를 noreply 이메일로 통일하거나, 둘 중 하나로 정리해두는 게 좋아요.

```shell
# 내 최근 커밋의 author email 확인
git log --format='%ae' -10
```

여기서 나오는 이메일이 GitHub `Settings` → `Emails` 의 verified 목록에 있어야 해요.


<br>

<br>



## 6. 검증 — 잔디가 다시 자라기까지

push 끝나고 Pages 도 잘 떴다면 이제 GitHub 가 백필을 돌려요.

| 항목 | 반영 시점 |
|---|---|
| 새 push 커밋 | 거의 즉시 ~ 몇 분 |
| 과거 커밋 백필 | 보통 24시간 이내 |
| 프로필 페이지 캐시 | 새로고침 / 강제 리로드로 빠르게 갱신 가능 |

🎉 fork 가 아닌 standalone repo 가 되는 순간부터 과거 커밋도 사후 카운팅 대상으로 풀려요. 

만약 48시간 지나도 잔디가 안 잡힌다면 이 두 가지를 다시 확인.

- [ ] commit author email 이 정말 verified email 인지
- [ ] repo 가 정말 `fork: false` 인지


<br>

<br>



## 7. 정리

- GitHub 잔디는 **fork repo 의 커밋은 무조건 제외**. 정책이고, 정책상 detach 도 더 이상 안 해줘요.
- 해결책은 standalone repo 로 옮기는 것 한 가지뿐.
- user site (`<username>.github.io`) 인 경우 rename → new repo (같은 이름) → push 순서로 가면 URL 도 보존돼요.
- 가장 자주 빼먹는 함정은 **commit author email 이 verified email 인지**, **Keep email private 설정 여부**.

블로그 시작할 때 "Fork" 버튼을 누른 게 1년 넘게 잔디를 가렸던 셈이에요. 새 프로젝트 시작할 때는 가능하면 `Use this template` 가 있으면 그걸 쓰거나, 그게 없으면 fork 후 가능한 빨리 standalone 으로 옮겨두는 게 좋아요.

일단 오늘은 여기까지.....   
다음 글에서는 새 repo 환경에서 이어가는 minimal-mistakes 커스터마이즈 작업 정리해볼게요.
