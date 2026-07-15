---
layout: single
title:  "worktree 브랜치를 PR 로 묶고 자동 리뷰까지 — 병렬 세션 작업을 안전하게 통합하기"
date: 2026-07-15 14:00:00 +0900
categories: coding
tag: [claude-code, git, worktree, pull-request, gh-cli, code-review, github-actions, merge-queue, ci, 병렬작업]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-15-worktree-pr-review/header.svg
  teaser: /assets/images/2026-07-15-worktree-pr-review/header.svg
description: "worktree 로 세션마다 격리한 브랜치를 손으로 순차 merge 하는 대신, gh CLI 로 각 브랜치를 PR 로 올리고 CI·코드 리뷰를 자동으로 태운 뒤 merge queue 로 안전하게 합치는 흐름. 각 Claude 세션이 자기 PR 을 관리하게 만드는 실전 워크플로우와 함정까지 정리했습니다."
---

## Summary

[이전 글](/coding/여러_Claude_세션_git_worktree_격리/)에서 같은 폴더에서 여러 Claude 세션이 서로 코드를 덮어쓰던 문제를, `git worktree` 로 세션마다 폴더·브랜치를 격리해서 풀었어요. 이제 각 세션은 자기 브랜치(`feature-auth`, `feature-api`, `feature-docs`)에 커밋을 착착 쌓고 있습니다.

그다음이 통합인데, 이전 글에서는 메인 폴더에서 손으로 `git merge` 를 하나씩 돌렸죠. 이게 안전하긴 한데, **아무 검토 없이 그냥 합쳐진다**는 게 마음에 걸려요. 에이전트 세 개가 각자 만든 코드를 사람 눈 한 번 안 거치고 `main` 에 밀어넣는 거니까요. 그래서 이번 글은 그 통합 단계를 **PR(Pull Request)** 로 바꿔서, 각 브랜치를 CI·코드 리뷰에 태운 뒤 안전한 순서로 합치는 흐름을 정리해볼게요.

> 💡 **이 글에서 다루는 것**
> - 왜 손 merge 대신 PR 로 합치는 게 나은지
> - `gh` CLI 로 worktree 브랜치를 PR 로 올리기
> - PR 에 CI·자동 코드 리뷰를 붙이는 법 (GitHub Actions)
> - 여러 PR 을 충돌 없이 안전한 순서로 합치기 (rebase·merge queue·auto-merge)
> - 각 Claude 세션이 자기 PR 을 스스로 관리하게 만들기
> - worktree 별 "커밋 → push → PR" 을 한 번에 처리하는 스크립트
> - 자주 부딪히는 함정과 체크리스트

<br>

<br>



## 1. 왜 손 merge 대신 PR 인가

이전 글의 순차 merge 는 "코드 유실을 막는다" 는 목표는 완벽히 달성해요. 격리된 브랜치를 하나씩 합치니까 서로 밟을 일이 없죠. 그런데 병렬 에이전트를 상시로 굴리기 시작하면 목표가 하나 더 생깁니다. **합쳐지는 코드가 맞는 코드인지 검증**하는 거예요.

손 merge 의 아쉬운 점을 짚어보면 이래요.

- **검토 지점이 없다.** `git merge feature-auth` 는 그 브랜치가 테스트를 통과하는지, 스타일이 맞는지 안 보고 그냥 합칩니다. 세 세션이 각자 회귀(regression)를 심어놨어도 그대로 `main` 에 들어가요.
- **통합 순서를 사람이 기억해야 한다.** 어느 브랜치를 먼저 합치고, 합친 뒤 다음 브랜치를 rebase 해야 하는지를 머리로 관리해야 합니다. 브랜치가 3개를 넘어가면 금세 헷갈려요.
- **기록이 안 남는다.** "이 변경이 왜 들어왔는지" 가 커밋 메시지 한 줄로만 남아요. PR 이면 설명·리뷰 코멘트·CI 결과가 한 곳에 묶여 남습니다.

PR 로 바꾸면 이 셋이 다 해결돼요. 각 브랜치가 PR 이 되고, 거기에 **CI(빌드·테스트)와 코드 리뷰가 자동으로 붙고**, 통과한 것만 순서대로 `main` 에 들어갑니다. 격리(worktree)가 "안 부딪히게" 만든 거라면, PR 은 "검증하고 합치게" 만드는 단계예요.

> 💡 격리와 검증은 별개의 목표예요. worktree 로 물리적 충돌을 막았어도, 합치기 전에 "이게 맞나" 를 보는 관문은 따로 필요합니다. PR 이 딱 그 관문이에요.

<br>

<br>



## 2. worktree 브랜치를 PR 로 올리기

준비물은 GitHub 공식 CLI 인 **`gh`** 입니다. 먼저 로그인 상태만 확인해요.

```shell
gh auth status
```

로그인이 안 돼 있으면 아래로 한 번 붙여둡니다(브라우저로 인증 흐름이 떠요).

```shell
gh auth login
```

이제 각 worktree 폴더로 들어가서 브랜치를 원격에 밀고 PR 을 만듭니다. `gh pr create` 는 **현재 체크아웃된 브랜치를 head 로** 잡기 때문에, worktree 폴더 안에서 실행하는 게 자연스러워요.

```shell
cd ~/Projects/myapp-auth

# 1) 브랜치를 원격에 올리고 (최초 1회는 -u 로 업스트림 연결)
git push -u origin feature-auth

# 2) 그 브랜치로 PR 생성 (base 는 통합 대상 브랜치)
gh pr create --base main --head feature-auth \
  --title "로그인 기능 추가" \
  --body "auth 세션 작업. 이메일/비밀번호 로그인 + 세션 발급."
```

정상이면 방금 만들어진 PR 의 URL 이 찍혀요. 나머지 두 세션도 각자 폴더에서 똑같이 하면 브랜치 3개가 각각 PR 3개로 올라갑니다.

```text
#41  로그인 기능 추가        feature-auth  → main
#42  API 엔드포인트 추가     feature-api   → main
#43  문서 정리              feature-docs  → main
```

한 번에 상태를 보고 싶으면 이렇게 확인해요.

```shell
gh pr list
```

> ✅ `--title`·`--body` 를 생략하면 `gh` 가 에디터를 띄워 대화식으로 물어봐요. 스크립트로 자동화할 거면 위처럼 인자로 넘기는 게 편합니다.

<br>

<br>



## 3. PR 에 CI·자동 리뷰 붙이기

PR 이 관문 역할을 하려면 **PR 이 열릴 때마다 자동으로 검사가 도는** 장치가 있어야 해요. GitHub Actions 워크플로우 하나면 됩니다. `.github/workflows/ci.yml` 예시예요(파이썬 프로젝트 기준, 언어만 바꾸면 그대로 적용).

```yaml
name: CI
on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest -q          # 테스트
      - run: ruff check .       # 린트
```

이렇게 두면 PR #41·#42·#43 각각에 대해 테스트·린트가 자동으로 돌고, 결과가 PR 화면에 ✅/❌ 로 붙어요. 실패한 PR 은 합치기 전에 걸러집니다.

여기에 **코드 리뷰** 관문을 더 얹을 수 있어요. 두 가지를 같이 쓰면 좋아요.

- **사람 리뷰 강제** — 저장소 설정에서 `main` 을 보호 브랜치로 잡고 "머지 전 승인 1건 필수" 를 켜둡니다. 누가 봐야 하는지는 `CODEOWNERS` 파일로 지정할 수 있어요.
- **자동 코드 리뷰** — 로컬에서 Claude Code 로 PR 을 리뷰하거나(터미널에서 해당 PR 을 리뷰하는 명령), GitHub Actions 에 코드 리뷰 잡을 하나 더 붙여 PR 코멘트로 지적을 남기게 합니다. 병렬로 찍어낸 코드일수록 이 자동 리뷰 한 겹이 회귀를 잘 걸러줘요.

핵심은 **PR 을 열기만 하면 검사가 알아서 돈다**는 거예요. 세션이 몇 개든, 각 PR 이 같은 관문을 통과해야 `main` 에 들어갈 수 있습니다.

> 🚨 보호 브랜치 설정 없이 PR 만 만들면 관문이 "장식" 이 돼요. CI 가 빨간불이어도 그냥 머지가 됩니다. **"검사 통과 + 승인" 을 머지 필수 조건으로 잡는 브랜치 보호 규칙**을 꼭 같이 켜세요.

<br>

<br>



## 4. 여러 PR 을 안전한 순서로 합치기

PR 이 셋 다 초록불이 됐다고 아무 순서로 막 머지하면 문제가 생겨요. #41 을 `main` 에 합치는 순간 `main` 이 바뀌는데, #42·#43 은 여전히 **옛날 `main` 기준으로 검사를 통과한 상태**거든요. 합쳐놓고 보면 실제로는 충돌하거나 깨질 수 있습니다. 이걸 "머지 순서 문제" 라고 해요.

해결책은 상황에 따라 셋 중 하나예요.

**(1) 소규모 — 하나 합치고 나머지 rebase.** PR 이 2~3개면 손으로도 충분해요. 하나 머지한 뒤, 나머지 브랜치를 최신 `main` 위로 다시 올려(rebase) CI 를 재실행합니다.

```shell
# #41 머지 후, api 브랜치를 최신 main 에 맞춰 재정렬
cd ~/Projects/myapp-api
git fetch origin
git rebase origin/main
git push --force-with-lease        # rebase 했으니 갱신 필요
```

여기서 `-f`(`--force`) 대신 **`--force-with-lease`** 를 쓰는 게 중요해요. 내가 마지막으로 본 이후 남이 그 브랜치를 건드렸으면 푸시를 거부해줘서, 남의 작업을 날리는 사고를 막아줍니다.

**(2) 자동화 — auto-merge.** 각 PR 에 "검사 다 통과하면 알아서 머지해줘" 를 걸어둘 수 있어요.

```shell
gh pr merge 41 --auto --squash
```

CI 가 초록이 되는 순간 GitHub 이 자동으로 합쳐요. 사람이 지켜볼 필요가 없어집니다.

**(3) 다수 — merge queue.** PR 이 여러 개 밀려 있으면 GitHub 의 **merge queue** 를 켜는 게 정답이에요. 큐에 들어온 PR 을 하나씩, **각각 최신 base 기준으로 다시 검사한 뒤** 순서대로 합쳐줍니다. 방금 말한 "옛날 main 기준 통과" 문제를 근본적으로 없애줘요. 저장소 설정의 브랜치 보호 규칙에서 "Require merge queue" 를 켜면 됩니다.

> 💡 규모별로 고르면 돼요. **2~3개는 rebase + auto-merge**, **상시로 여러 PR 이 쏟아지면 merge queue**. 어느 쪽이든 핵심은 "합치기 직전의 최신 base 로 검사했는가" 예요.

<br>

<br>



## 5. 각 세션이 자기 PR 을 스스로 관리하게

여기까지 오면 자연스러운 다음 수순이 있어요. **각 Claude 세션한테 자기 브랜치의 PR 까지 맡기는** 거예요. 어차피 세션은 자기 worktree 폴더 안에 격리돼 있으니, 자기 PR 을 만들고 리뷰 코멘트를 반영하는 것까지 자기 영역에서 하면 딱 맞아요.

세션별로 이런 식으로 역할을 줄 수 있어요.

- 작업이 끝나면 **커밋 → `git push` → `gh pr create`** 까지 스스로.
- CI 가 실패하면 로그를 `gh run view` / `gh pr checks` 로 읽고 **자기 브랜치에서 고쳐서 다시 push.**
- 리뷰 코멘트가 달리면 `gh pr view --comments` 로 읽고 **반영 커밋을 자기 브랜치에 추가.**

한 가지 규칙만 지키면 돼요.

> 🚨 **각 세션은 자기 PR·자기 브랜치만 건드린다.** 남의 PR 을 머지하거나 남의 브랜치를 rebase 하는 건 금지. 그건 "통합 담당" 한 명(사람이든, 메인 폴더의 세션 하나든)만 합니다.

이 경계만 지키면, 이전 글에서 세운 "논리적 분리를 물리적 분리로 뒷받침한다" 는 원칙이 PR 단계까지 그대로 이어져요. 세션은 자기 PR 이라는 자기 영역 안에서만 움직이고, 합치는 결정은 관문(CI·리뷰) + 통합 담당이 내립니다.

<br>

<br>



## 6. worktree 하나를 "커밋 → PR" 로 한 번에

매번 push 하고 `gh pr create` 치는 게 귀찮으면 작은 스크립트로 묶어두면 편해요. worktree 폴더 안에서 실행하면 현재 브랜치를 그대로 PR 로 올려줍니다.

```shell
#!/usr/bin/env bash
# open-pr.sh — 현재 worktree 브랜치를 커밋하고 PR 로 올린다
set -euo pipefail

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ]; then
  echo "main 에서는 실행 금지 (작업 브랜치에서 실행하세요)"; exit 1
fi

# 남은 변경이 있으면 커밋 (메시지는 인자로, 없으면 기본값)
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "${1:-작업 커밋: $branch}"
fi

git push -u origin "$branch"

# 이미 PR 이 있으면 새로 안 만들고 건너뜀
if gh pr view "$branch" >/dev/null 2>&1; then
  echo "이미 PR 존재 — push 만 반영됨"
else
  gh pr create --base main --head "$branch" --fill
fi
```

`--fill` 은 커밋 메시지로 PR 제목·본문을 자동으로 채워줘요. 실행은 이렇게.

```shell
cd ~/Projects/myapp-auth
./open-pr.sh "로그인 기능 1차 구현"
```

각 세션한테 "작업 끝나면 `open-pr.sh` 돌려" 라고만 하면, 커밋부터 PR 생성·재푸시까지 한 줄로 정리됩니다.

> ⚠️ `set -euo pipefail` 은 중간에 실패하면 바로 멈추게 해줘요. 자동화 스크립트엔 거의 항상 넣는 게 좋아요. 반쯤 진행되다 애매하게 성공한 척하는 걸 막아줍니다.

<br>

<br>



## 7. 자주 부딪히는 함정

| 증상 | 원인 | 대응 |
| --- | --- | --- |
| `gh pr create` 가 head 를 엉뚱하게 잡음 | 메인 폴더(`main`)에서 실행 | **작업 worktree 폴더 안**에서 실행 |
| CI 초록인데 머지하니 깨짐 | 옛날 base 기준으로 통과 | rebase 후 재검사 or **merge queue** |
| `git push` 가 거부됨(non-fast-forward) | rebase 로 히스토리가 갈림 | `--force-with-lease` (그냥 `-f` 말고) |
| 자동 리뷰가 매번 같은 지적 반복 | 리뷰 결과를 브랜치에 반영 안 함 | 세션이 코멘트 읽고 **반영 커밋** 추가 |
| PR 은 많은데 뭐가 통과했는지 모름 | 상태를 한눈에 안 봄 | `gh pr checks` / `gh pr list` 로 확인 |
| 보호 규칙 없이 관문만 만듦 | 브랜치 보호 미설정 | `main` 에 "검사+승인 필수" 규칙 켜기 |

특히 두 번째(옛날 base 통과)와 마지막(보호 규칙 없음)이 실무에서 제일 자주 사고로 이어져요. **"머지 직전 최신 base 로 재검사"** 와 **"검사·승인을 머지 필수 조건으로"** 이 두 개만 지켜도 병렬 세션 통합의 대형 사고는 거의 안 납니다.

<br>

<br>



## 8. 최종 체크리스트

- [x] 각 worktree 브랜치를 **폴더 안에서** `git push -u` → `gh pr create` 로 PR 화
- [x] PR 마다 **CI(테스트·린트)** 가 자동으로 돌게 GitHub Actions 설정
- [x] `main` 을 **보호 브랜치**로 — "검사 통과 + 승인" 을 머지 필수 조건으로
- [x] 통합 순서는 **rebase + auto-merge**(소규모) 또는 **merge queue**(다수)
- [x] rebase 후 push 는 **`--force-with-lease`** (그냥 `-f` 금지)
- [x] 각 세션은 **자기 PR·자기 브랜치만** — 남의 PR 머지/rebase 는 통합 담당만
- [x] "커밋 → push → PR" 은 작은 스크립트(`open-pr.sh`)로 묶어 세션에 위임

두 편에 걸친 핵심을 한 줄로 줄이면, **격리(worktree)로 안 부딪히게 만들고, 관문(PR·CI·리뷰)으로 검증하고 합쳐라** 예요. 이 두 겹이 있으면 Claude 세션을 몇 개를 띄우든 코드가 사라지지도, 검증 안 된 코드가 슬쩍 들어가지도 않습니다.

<br>

<br>



일단 오늘은 여기까지.....  
다음 글에서는 이 흐름을 한 발 더 밀어서, 통합 담당 세션이 여러 PR 의 충돌을 스스로 감지하고 정리하는 오케스트레이션 패턴을 정리해볼게요.

---

**← 이전 글:** [여러 Claude 세션이 서로 덮어쓸 때 — git worktree 로 격리하기](/coding/여러_Claude_세션_git_worktree_격리/)
