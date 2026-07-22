---
layout: single
title:  "(1/4) 여러 Claude 세션이 서로 덮어쓸 때 — 한 폴더 동시 작업을 git worktree 로 격리하기"
date: 2026-07-15 10:30:00 +0900
categories: coding
tag: [claude-code, git, worktree, 병렬작업, 멀티에이전트, merge, 브랜치전략, workflow, 충돌해결, 협업]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-15-claude-sessions-worktree/header.svg
  teaser: /assets/images/2026-07-15-claude-sessions-worktree/header.svg
description: "같은 폴더에서 Claude Code 세션을 여러 개 띄워 각자 작업하다 git 으로 합치면 과거 코드가 덮어써지는 문제. 왜 한 워킹트리에서의 동시 편집은 merge 로도 못 살리는지, git worktree 로 세션마다 폴더·브랜치를 격리하고 한 곳에서 순차 통합하는 실전 워크플로우를 정리했습니다."
series: parallel-sessions
series_order: 1
series_title: "📦 병렬 Claude 세션 작업 안전하게 합치기"
---

{% include series-parallel-sessions.html current="1" %}

## Summary

Claude Code 를 여러 개 띄워놓고 병렬로 굴리기 시작하면 누구나 한 번쯤 겪는 사고가 있어요. 같은 폴더에서 세션 A, B, C 를 각각 다른 이름으로 띄워 각자 작업을 시키고, 나중에 git 으로 합치려는데 **어떤 세션이 방금 만든 코드가 다른 세션 커밋에 통째로 덮어써져서 사라지는** 일이죠. 저도 딱 이걸 겪고 나서야 "아, 한 폴더에서 여러 에이전트를 굴리는 건 애초에 구조가 틀렸구나" 하고 정리하게 됐어요.

결론부터 말하면, 문제의 뿌리는 git merge 를 잘못 써서가 아니라 **여러 세션이 물리적으로 같은 워킹트리(working tree) 하나를 공유**한다는 데 있습니다. 이 글에서는 왜 그런 일이 벌어지는지, 그리고 `git worktree` 로 세션마다 독립된 폴더·브랜치를 주고 한 곳에서만 순차적으로 합치는 워크플로우를 정리해볼게요.

> 💡 **이 글에서 다루는 것**
> - 같은 폴더 동시 작업이 왜 코드를 유실시키는지 (lost update 의 구조)
> - "그냥 git merge 하면 되잖아" 가 왜 이 경우엔 안 통하는지
> - `git worktree` 로 세션마다 폴더+브랜치를 격리하는 법
> - 각 세션 작업을 한 곳에서 순차적으로 안전하게 합치는 통합 워크플로우
> - worktree 를 못 쓰는 상황에서의 차선책(파일 영역 분리·자주 커밋)
> - 자주 부딪히는 함정과 최종 체크리스트

<br>

<br>



## 1. 무슨 일이 벌어졌나

상황을 재현해볼게요. 프로젝트 폴더 하나(`~/Projects/myapp`)에서 터미널 탭을 세 개 열고, 각각 Claude Code 를 띄웁니다. 세션 이름만 다를 뿐 **작업 디렉토리는 셋 다 똑같아요.**

```text
탭1: cd ~/Projects/myapp && claude   # 세션 "auth"  — 로그인 기능
탭2: cd ~/Projects/myapp && claude   # 세션 "api"   — API 엔드포인트
탭3: cd ~/Projects/myapp && claude   # 세션 "docs"  — 문서 정리
```

셋 다 열심히 파일을 고칩니다. 그리고 여기서 사고가 나요. 대표적인 두 갈래가 있습니다.

**(1) 저장이 겹치는 경우.** 세션 `auth` 가 `config.py` 를 읽어서 A 버전으로 고쳐 저장합니다. 거의 동시에 세션 `api` 도 같은 `config.py` 를 (아직 A 수정이 반영 안 된 상태로) 읽어놨다가 B 버전으로 저장해요. 디스크에는 마지막에 쓴 B 만 남고, A 의 수정은 흔적도 없이 날아갑니다. git 이 개입할 틈도 없어요. 그냥 파일시스템 레벨에서 덮어쓴 거예요.

**(2) 커밋·머지가 겹치는 경우.** 세션마다 각자 `git add . && git commit` 을 합니다. 문제는 셋 다 같은 브랜치(`main` 이든 `gh-pages` 든) 위에 서 있다는 거예요. `api` 세션이 커밋할 때 워킹트리에는 `auth` 세션이 아직 커밋 안 한 변경분이 섞여 있거나, 반대로 누군가 `git checkout .` / `git reset --hard` 로 워킹트리를 되돌리면서 다른 세션의 미커밋 작업을 쓸어버립니다. 여기에 `git push -f` 까지 끼면 원격의 히스토리마저 서로 덮어써요.

핵심은 이거예요. **세 세션이 브랜치는 "논리적으로 다른 일" 을 한다고 생각하지만, 물리적으로는 파일 하나·인덱스 하나·HEAD 하나를 공유**하고 있다는 것. git 은 이걸 세 갈래로 인식하지 못합니다. 그냥 한 사람이 정신없이 왔다갔다 편집하는 걸로 볼 뿐이에요.

> ⚠️ 이건 Claude Code 만의 문제가 아니라, **한 워킹트리에서 여러 주체가 동시에 편집하면 무조건 생기는** 고전적인 lost update 입니다. 사람 둘이 같은 폴더를 SSH 로 붙어 동시에 고쳐도 똑같이 나요. 에이전트는 손이 빨라서 더 자주 부딪힐 뿐이에요.

<br>

<br>



## 2. "그냥 git merge 하면 되잖아" 가 안 되는 이유

여기서 많이들 "브랜치 나눠서 merge 하면 git 이 알아서 합쳐주잖아" 라고 생각해요. 맞는 말인데, **전제가 하나 빠졌습니다.** git 의 3-way merge 가 작동하려면 **합치려는 두 변경이 각자 독립된 커밋 히스토리로 존재**해야 해요. 그런데 한 폴더에서 세션을 돌리면 그 전제가 깨집니다.

git 워킹트리의 상태는 크게 세 층이에요.

| 층 | 정체 | 한 폴더에서 세션 3개면 |
| --- | --- | --- |
| 워킹 디렉토리 | 실제 파일들 | **1개를 공유** — 마지막 저장이 이김 |
| 인덱스(staging) | `git add` 한 스냅샷 | **1개를 공유** — 서로의 add 가 섞임 |
| HEAD / 현재 브랜치 | 지금 체크아웃된 지점 | **1개를 공유** — 브랜치 전환도 전역 |

세 세션이 브랜치를 각자 만들어 `git checkout -b feature-auth` 하는 것조차 위험해요. 브랜치 체크아웃은 **폴더 전체의 전역 상태**라서, `auth` 세션이 브랜치를 바꾸는 순간 `api` 세션이 보고 있던 파일들도 그 브랜치 기준으로 싹 바뀝니다. 한 세션이 밟고 선 바닥을 다른 세션이 갈아끼우는 셈이에요.

즉 merge 가 나빠서가 아니라, **애초에 merge 에 넣을 깨끗한 두 갈래를 만들지 못하는 구조**가 문제입니다. 두 변경이 같은 워킹트리에서 서로를 밟고 지나간 뒤라, git 손에 들어올 때는 이미 한쪽이 사라진 상태예요. 합칠 재료 자체가 유실된 거죠.

> 💡 정리하면 필요한 건 "더 똑똑한 merge" 가 아니라 **세션마다 독립된 워킹트리·인덱스·HEAD** 입니다. 그래야 git 이 진짜로 세 갈래를 인식하고, 나중에 3-way merge 로 제대로 합쳐줘요.

<br>

<br>



## 3. 해결의 핵심 — `git worktree`

브랜치마다 폴더를 따로 `git clone` 하는 방법도 있지만, 그러면 저장소 히스토리가 통째로 복제돼서 무겁고 원격 설정도 따로 놀아요. 그래서 정답은 **`git worktree`** 입니다.

`git worktree` 는 **하나의 `.git` 저장소(오브젝트·히스토리)를 공유하면서, 워킹 디렉토리와 인덱스·HEAD 만 여러 벌로 늘려주는** 기능이에요. 딱 우리가 원하던 그림이죠.

- 히스토리·오브젝트는 한 벌만 → 디스크 절약, `git fetch` 한 번이면 모든 worktree 가 최신 커밋 공유
- 워킹 디렉토리·인덱스·HEAD 는 worktree 마다 독립 → 세션끼리 파일이 안 섞임
- **같은 브랜치를 두 worktree 에 동시에 체크아웃하는 걸 git 이 막아줌** → 실수로 겹치는 걸 원천 차단

그림으로 보면 이렇게 바뀝니다.

```text
Before ── 한 폴더, 세 세션이 공유 (충돌)
  ~/Projects/myapp   ← 세션 auth·api·docs 가 전부 여기

After ── worktree 로 세션마다 폴더·브랜치 분리
  ~/Projects/myapp          (main, 통합 담당)
  ~/Projects/myapp-auth     (feature-auth)   ← 세션 auth
  ~/Projects/myapp-api      (feature-api)    ← 세션 api
  ~/Projects/myapp-docs     (feature-docs)   ← 세션 docs
```

이제 세션 `auth` 가 브랜치를 바꾸든 파일을 고치든, 그건 `myapp-auth` 폴더 안에서만 일어나요. `api` 세션은 자기 폴더에서 태평하게 자기 일만 합니다. **물리적으로 격리됐기 때문에 서로 덮어쓸 방법이 없어요.**

> 💡 Claude Code 자체에도 worktree 격리를 돕는 기능들이 있어요(작업을 별도 worktree 에서 실행하는 옵션 등). 다만 원리를 알아야 어떤 상황에 뭘 쓸지 판단이 서니까, 이 글은 순수 `git worktree` 명령 기준으로 풀어갈게요.

<br>

<br>



## 4. 실전 셋업 — 세션마다 worktree 하나씩

메인 저장소 폴더(`~/Projects/myapp`)에서 시작합니다. 이 폴더는 앞으로 **통합(merge) 전용**으로만 쓸 거예요. 여기서 각 작업용 worktree 를 만듭니다.

```shell
cd ~/Projects/myapp

# 형식: git worktree add <새 폴더 경로> -b <새 브랜치명>
git worktree add ../myapp-auth -b feature-auth
git worktree add ../myapp-api  -b feature-api
git worktree add ../myapp-docs -b feature-docs
```

`-b feature-auth` 는 "새 브랜치를 만들면서 그 브랜치를 이 폴더에 체크아웃하라" 는 뜻이에요. 이미 있는 브랜치를 붙이고 싶으면 `-b` 를 빼고 브랜치명만 적으면 됩니다.

```shell
# 이미 존재하는 브랜치를 worktree 로 꺼내기
git worktree add ../myapp-hotfix hotfix-login
```

현재 worktree 목록은 이렇게 확인해요.

```shell
git worktree list
```

정상이면 이런 출력이 떠요. 각 폴더가 어느 브랜치를 물고 있는지 한눈에 보입니다.

```text
/home/me/Projects/myapp        a1b2c3d [main]
/home/me/Projects/myapp-auth   a1b2c3d [feature-auth]
/home/me/Projects/myapp-api    a1b2c3d [feature-api]
/home/me/Projects/myapp-docs   a1b2c3d [feature-docs]
```

이제 **Claude Code 를 각 worktree 폴더 안에서 띄웁니다.** 이게 핵심이에요. 같은 폴더에서 세 번 띄우는 대신, 폴더를 갈라서 하나씩.

```shell
# 탭1
cd ~/Projects/myapp-auth && claude
# 탭2
cd ~/Projects/myapp-api  && claude
# 탭3
cd ~/Projects/myapp-docs && claude
```

각 세션 입장에서는 그냥 평범한 git 저장소 하나가 통째로 자기 것이에요. 다른 세션의 존재를 몰라도 되고, 알 필요도 없습니다. 자기 폴더에서 마음껏 고치고 `git add`·`git commit` 하면 그게 자기 브랜치에 차곡차곡 쌓여요.

사실 이 "`worktree add` → `cd` → `claude`" 3단계는 **Claude Code 의 `--worktree` 플래그로 한 줄로** 줄일 수 있어요. 세션을 띄울 때 워크트리를 알아서 만들어 그 안에서 시작합니다.

```shell
# git worktree add + cd + claude 를 한 번에
claude --worktree auth      # .claude/worktrees/auth/ 에 worktree-auth 브랜치로 시작
claude --worktree api       # 다른 탭에서
claude --worktree docs
```

`--worktree` (짧게 `-w`) 는 워크트리를 `.claude/worktrees/<이름>/` 에 만들고 `worktree-<이름>` 브랜치를 붙여줘요(기본 브랜치에서 분기). `--tmux` 를 같이 주면 tmux 패널로도 띄워집니다. 다만 **세션마다 이 플래그를 붙여줘야** 하고, "여러 세션을 자동으로 각자 워크트리에" 몰아주는 전역 설정은 없어요(그건 데스크톱 앱이 세션마다 자동으로 해줍니다). 그래도 손으로 `git worktree add` 를 치는 것보다 훨씬 간편하죠.

여기서 꼭 짚을 게 하나 있어요. 세션에 이름을 다르게 주는 **`--name`(짧게 `-n`) 은 격리가 아니에요.** `--name` 은 프롬프트 박스·`/resume` 피커·터미널 제목에 뜨는 **표시용 이름표**일 뿐, 작업 폴더와 파일은 그대로 공유합니다. 그래서 세션 이름만 갈라놓고 같은 폴더에서 돌리면 1장에서 본 덮어쓰기가 그대로 재현돼요. 이름이 다른 것과 작업 공간이 다른 것은 완전히 별개입니다.

```shell
claude --name auth          # 이름표만 다름 → 여전히 같은 폴더 공유 → 덮어쓰기 O
claude --worktree auth      # 폴더 자체가 갈림 → 진짜 격리 → 덮어쓰기 X
```

> 🚨 이게 "세션명은 다 다르게 했는데 왜 코드가 덮어써지지?" 의 정확한 원인이에요. **`--name` 은 라벨, `--worktree` 는 격리.** 병렬로 돌릴 거면 이름이 아니라 워크트리(또는 폴더)를 갈라야 합니다. 둘을 같이 써서 `claude --worktree auth --name auth` 처럼 이름표까지 붙이면 보기도 편해요.

> ✅ 팁: worktree 폴더 이름(또는 `--worktree <이름>` 의 이름)에 작업 성격을 박아두면 터미널 탭이 여러 개일 때 "지금 내가 어느 세션인지" 헷갈리지 않아요. 프롬프트에 git 브랜치를 표시하는 셸 설정도 같이 켜두면 더 안전합니다.

<br>

<br>



## 5. 각 세션 작업을 한 곳에서 합치기

이제 세션들이 각자 커밋을 쌓았어요. 합칠 차례입니다. 여기서 규칙이 딱 하나 있어요.

> 🚨 **통합(merge)은 메인 폴더 한 곳에서, 한 번에 하나씩, 순차적으로만 한다.**

작업용 worktree 안에서 다른 브랜치를 merge 하려 들면 다시 여러 세션이 통합 브랜치를 동시에 건드리는 원점으로 돌아가요. 통합은 반드시 통합 전용 폴더(`~/Projects/myapp`, `main`)에서만.

```shell
cd ~/Projects/myapp
git checkout main

# 각 브랜치를 순서대로 하나씩 병합
git merge feature-auth
git merge feature-api
git merge feature-docs
```

서로 다른 파일을 건드렸다면 git 이 알아서 3-way merge 로 조용히 합쳐줍니다. **1장에서 유실됐던 A 수정이 이번엔 안 사라지는 이유가 바로 이거예요.** 각 변경이 독립된 커밋으로 살아있으니 git 이 양쪽을 다 반영합니다.

같은 파일의 같은 줄을 두 세션이 고쳤다면 그때는 정직하게 충돌(conflict)이 납니다. 이건 사고가 아니라 **git 이 "여기 둘이 겹쳤으니 사람이 판단해라" 하고 정상적으로 알려주는** 거예요. 조용히 한쪽을 날리는 것보다 백 배 낫습니다.

```shell
# 충돌 파일 확인
git status

# 파일 열어서 <<<<<<< / ======= / >>>>>>> 마커 기준으로 정리한 뒤
git add <해결한 파일>
git merge --continue
```

충돌 해결 자체를 또 다른 Claude 세션한테 맡길 수도 있어요. 다만 그것도 **통합 폴더에서 도는 세션 하나**한테만 시키세요. 여러 세션이 동시에 충돌을 풀면 그게 또 새로운 충돌의 씨앗이 됩니다.

통합이 끝났으면 원격에 올립니다. 이때 **`git push` 만, 절대 `git push -f` 는 쓰지 마세요.** force push 는 방금 순서대로 쌓은 통합 히스토리를 남의 커밋째로 날려버릴 수 있어요.

```shell
git push origin main
```

<br>

<br>



## 6. worktree 를 다 쓰고 정리하기

작업이 끝난 worktree 는 남겨두면 헷갈리니 깔끔하게 치웁니다. 폴더를 `rm -rf` 로 그냥 지우지 말고 **`git worktree remove`** 를 쓰세요. git 이 내부 메타데이터까지 같이 정리해줘요.

```shell
cd ~/Projects/myapp

# 병합 끝난 worktree 제거
git worktree remove ../myapp-auth
git worktree remove ../myapp-api
git worktree remove ../myapp-docs

# 합쳐진 브랜치도 정리
git branch -d feature-auth feature-api feature-docs
```

폴더를 손으로 지워버려서 목록에 유령이 남았다면 이걸로 청소해요.

```shell
git worktree prune
```

> ⚠️ 커밋 안 한 변경이 남은 worktree 는 `git worktree remove` 가 거부합니다(안전장치). 진짜 버릴 거면 `--force` 를 붙이되, 정말 날려도 되는지 한 번 더 확인하고요.

<br>

<br>



## 7. worktree 를 못 쓰는 상황이라면 (차선책)

가끔 worktree 로 못 가는 사정이 있어요. 빌드 산출물이 폴더에 종속적이거나, `.env`·거대한 `node_modules` 를 폴더마다 복제하기 부담스럽거나. 그럴 땐 한 폴더를 유지하되 **충돌 확률을 최대한 낮추는** 쪽으로 갑니다. 완벽한 격리는 아니지만 사고를 크게 줄여줘요.

- **파일 영역을 아예 안 겹치게 나눈다.** 세션 A 는 `src/auth/` 만, 세션 B 는 `src/api/` 만. 같은 파일을 두 세션이 만질 일을 애초에 없앱니다. 겹치는 공통 파일(`config`, 라우터 등록부 등)은 한 세션이 전담.
- **작게, 자주 커밋한다.** 한 세션이 한 덩어리 작업을 끝내면 바로 커밋. 워킹트리에 미커밋 변경이 오래 떠 있을수록 다른 세션이 덮어쓸 창이 넓어져요.
- **작업 전 `git pull`/`git status` 로 현재 상태를 먼저 확인**하고 시작. 남이 방금 바꾼 걸 모르고 옛날 버전 위에 얹으면 그게 lost update 예요.
- **`git checkout .` / `git reset --hard` / `git stash` 를 함부로 쓰지 않는다.** 이 셋은 워킹트리를 통째로 되돌려서 다른 세션의 미커밋 작업을 소리 없이 지웁니다. 여러 세션이 도는 폴더에서는 특히 조심.
- **`git push -f` 금지.** 원격 히스토리를 덮어써서 남의 커밋을 날리는 가장 흔한 사고 경로예요.

그래도 이건 어디까지나 임시방편입니다. 병렬 세션을 상시로 굴릴 거면 결국 worktree 로 가는 게 맞아요.

<br>

<br>



## 8. 자주 부딪히는 함정

worktree 를 쓰기 시작하면 새로 만나는 자잘한 벽들이 있어요. 미리 알아두면 안 당합니다.

| 증상 | 원인 | 대응 |
| --- | --- | --- |
| `fatal: '...'`<br>`already checked out` | 같은 브랜치를 두 worktree 에<br>체크아웃 시도 | 브랜치는 worktree 당 하나.<br>새 브랜치를 `-b` 로 파거나<br>기존 걸 옮기기 |
| 로컬 서버 포트 충돌 | 여러 worktree 에서 같은 포트로<br>동시에 `serve` 실행 | 포트를 다르게<br>(`-P 4001`, `--port 3001` 등) |
| `node_modules`·`.env` 가 없음 | worktree 는 추적 파일만 복제,<br>untracked 는 안 따라옴 | worktree 마다 `npm install` /<br>`.env` 복사 한 번씩 |
| 빌드 산출물이 섞임 | `_site/`·`dist/` 를 커밋에 포함 | `.gitignore` 에 넣어 추적 제외 |
| 메인 폴더에서 브랜치 삭제 거부 | 그 브랜치가 다른 worktree 에서<br>체크아웃 중 | 먼저 해당 worktree 를 `remove` 후 삭제 |

특히 이 블로그(`dorumugs.github.io`)처럼 Jekyll 로컬 프리뷰를 돌리는 저장소라면, worktree 두 곳에서 `bundle exec jekyll serve` 를 동시에 켜면 4000 포트가 겹쳐요. 한쪽은 `--port 4001` 로 띄우면 됩니다.

<br>

<br>



## 9. 최종 체크리스트

병렬 세션을 안전하게 굴리기 위한 최소 규칙을 정리하면 이래요.

- [x] 세션마다 **별도 worktree 폴더 + 별도 브랜치** 를 준다 (`git worktree add ../x -b feature-x`)
- [x] Claude Code 는 **각 worktree 폴더 안에서** 띄운다 (같은 폴더 다중 실행 금지)
- [x] 통합은 **메인 폴더 한 곳에서, 한 번에 하나씩** 순차 merge
- [x] 충돌은 사고가 아니라 정상 신호 — 조용히 한쪽 날리지 말고 손으로 해결
- [x] `git push -f`, 무분별한 `git reset --hard`/`checkout .` 금지
- [x] 끝난 worktree 는 `git worktree remove` 로 정리
- [x] worktree 를 못 쓰면 최소한 **파일 영역 분리 + 작게 자주 커밋**

핵심을 한 줄로 줄이면, **"세션의 논리적 분리(브랜치)를 물리적 분리(폴더)로 뒷받침해라"** 예요. git worktree 가 그 다리를 놔줍니다. 한 폴더에서 여러 에이전트를 굴리며 코드가 사라지던 문제가, 폴더를 가르는 순간 거짓말처럼 없어질 거예요.

<br>

<br>



일단 오늘은 여기까지.....  
다음 글에서는 이렇게 나눈 worktree 브랜치들을 자동으로 PR 로 묶어 리뷰까지 태우는 흐름을 정리해볼게요.

---

**다음 글 →:** [(2/4) worktree 브랜치를 PR 로 묶고 자동 리뷰까지 통합하기](/coding/worktree_브랜치_PR_자동리뷰_통합/)
