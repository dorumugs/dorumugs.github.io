---
layout: single
title:  "(2/2) CodeCommit 에서 GitHub 으로 리포 옮기기 — 미러 마이그레이션 실전"
date: 2026-06-26 12:40:00 +0900
categories: coding
tag: [AWS, CodeCommit, GitHub, git, migration, 마이그레이션, mirror, GitHubActions, CodePipeline, DevOps]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-26-codecommit-migration/header.svg
  teaser: /assets/images/2026-06-26-codecommit-migration/header.svg
description: "CodeCommit 리포를 GitHub 으로 옮기는 실전 가이드. git clone --mirror / push --mirror 로 전체 히스토리·브랜치·태그를 통째 이관하고, CI/CD(CodePipeline→Actions)·보호 규칙·컷오버까지 체크리스트로 정리했어요."
series: codecommit-github
series_order: 2
---

{% include series-codecommit-github.html current="2" %}

## Summary

[지난 글](/coding/AWS_CodeCommit_vs_GitHub_사용법/)에서 CodeCommit 과 GitHub 을 같은 작업끼리 나란히 비교했어요. 결론은 "새로 시작한다면 GitHub" 이었죠. 그렇다면 **이미 CodeCommit 에 쌓여 있는 리포는 어떻게 옮기느냐**가 다음 숙제입니다.

다행히 git 저장소를 옮기는 일 자체는 단순해요. `git clone --mirror` 로 통째 받아서 `git push --mirror` 로 새 원격에 밀어넣으면 **전체 커밋 히스토리·모든 브랜치·태그**가 그대로 따라옵니다. 진짜 손이 가는 건 git 바깥, 즉 **CI/CD·권한·PR** 같은 주변 장치를 다시 세우는 일이에요. 이 글에서 둘 다 정리합니다.

> 💡 **이 글에서 다루는 것**
> - 마이그레이션의 큰 그림 — 무엇이 따라오고 무엇이 안 따라오나
> - `git clone --mirror` → `git push --mirror` 핵심 2단계
> - Git LFS 가 있을 때 추가로 할 일
> - 이관 후 검증 — 브랜치·태그·커밋 수 대조
> - CI/CD 재구성 — CodePipeline 을 GitHub Actions 로
> - 보호 규칙·권한 재구성 (Approval rule → Branch protection)
> - 컷오버(전환) 체크리스트

<br>

<br>



## 1. 큰 그림 — 무엇이 따라오고, 무엇이 안 따라오나

가장 먼저 짚어야 할 건 **"git 이 아는 것만 따라온다"** 는 사실이에요. 커밋·브랜치·태그는 git 객체라 그대로 옮겨지지만, **PR·코드리뷰 댓글·승인 규칙·웹훅·CI 설정**은 git 바깥의 서비스 데이터라 자동으로 안 넘어옵니다.

| 항목 | 미러 push 로 따라오나 | 비고 |
|---|---|---|
| 커밋 히스토리 전체 | ✅ | SHA 그대로 보존 |
| 모든 브랜치 | ✅ | `--mirror` 가 refs 전부 복제 |
| 태그 | ✅ | 릴리스 태그 포함 |
| Git LFS 대용량 파일 | ⚠️ | 별도 명령 필요 (5장) |
| Pull Request·리뷰 댓글 | ❌ | 서비스 메타데이터, 수동 |
| Approval rule·보호 규칙 | ❌ | GitHub 에서 새로 설정 |
| CI/CD 파이프라인 | ❌ | Actions 로 재작성 |
| 트리거(EventBridge)·웹훅 | ❌ | 새 리포에 다시 연결 |

> 💡 핵심은 **"코드 히스토리는 그대로, 자동화는 다시"** 예요. 옮기는 데 며칠씩 걸리는 게 아니라, git 부분은 몇 분이면 끝나고 나머지 절반의 시간을 CI/CD·권한 재구성에 씁니다.

<br>

<br>



## 2. 사전 준비

세 가지를 미리 맞춰둡니다.

1. **CodeCommit 읽기 권한** — 1편에서 본 `git-remote-codecommit(grc)` 또는 HTTPS Git credentials 로 소스 리포를 clone 할 수 있어야 해요.
2. **GitHub 빈 리포** — 대상 리포를 **README·.gitignore 없이 완전히 비워서** 만듭니다. 초기 커밋이 하나라도 있으면 미러 push 가 충돌해요.
3. **GitHub 인증** — PAT 또는 SSH 키.

```shell
# 대상 GitHub 리포를 비운 상태로 생성 (gh CLI)
gh repo create my-org/my-demo-repo --private
```

> ⚠️ GitHub 웹에서 리포를 만들 때 "Add a README" 체크를 **끄세요**. 빈 리포여야 `push --mirror` 가 히스토리를 깔끔하게 올립니다. 실수로 커밋이 생겼다면 push 전에 비우거나 새 리포를 다시 만드는 게 빠릅니다.

<br>

<br>



## 3. 핵심 2단계 — clone --mirror → push --mirror

여기가 마이그레이션의 심장이에요. 딱 두 명령입니다.

```shell
# 1) 소스(CodeCommit)를 mirror 로 통째 복제 — 베어 저장소가 생김
git clone --mirror codecommit::ap-northeast-2://MyDemoRepo

cd MyDemoRepo.git

# 2) 대상(GitHub)을 새 원격으로 걸고 mirror 로 push
git remote add github https://github.com/my-org/my-demo-repo.git
git push github --mirror
```

`--mirror` 가 일반 clone 과 다른 점은 **모든 ref(브랜치·태그·노트)를 1:1로 복제**한다는 거예요. 작업 트리 없는 베어 저장소(`*.git` 폴더)로 받아지고, 그걸 그대로 다른 원격에 밀어넣으면 끝입니다.

> 🚨 `push --mirror` 는 **대상 리포의 ref 를 소스와 똑같이 강제로 맞춰요.** 대상에 이미 다른 브랜치가 있으면 **지워집니다.** 그래서 2장에서 "빈 리포" 를 그렇게 강조한 거예요. 운영 중인 리포에 미러를 쏘지 마세요.

remote URL 만 익숙해지면 됩니다. 소스가 grc 스킴(`codecommit::`)이고 대상이 평범한 GitHub HTTPS URL 이라는 것만 다르고, 나머지는 표준 git 이에요.

<br>

<br>



## 4. 이관 후 검증 — 숫자로 대조

"옮겼다" 고 끝내지 말고, **소스와 대상의 브랜치·태그·커밋 수가 같은지** 숫자로 맞춰봅니다.

```shell
# 브랜치 수 (소스 mirror 폴더 안에서)
git branch -a | wc -l

# 태그 수
git tag | wc -l

# 특정 브랜치의 커밋 수 — 소스/대상에서 각각 찍어 비교
git rev-list --count main
```

대상 GitHub 리포를 따로 clone 해서 같은 명령으로 찍은 값이 **소스와 일치하면** 이관 성공이에요.

```shell
# 대상을 평범하게 clone 해서 동일 검증
git clone https://github.com/my-org/my-demo-repo.git verify
cd verify
git branch -a | wc -l
git tag | wc -l
```

> ✅ 체크 포인트: 브랜치 수 · 태그 수 · `main`(또는 기본 브랜치) 최신 커밋 SHA 가 양쪽에서 동일한지. SHA 까지 같으면 히스토리가 한 비트도 안 틀어졌다는 뜻이에요.

<br>

<br>



## 5. Git LFS 가 있을 때

리포가 **Git LFS**(대용량 파일)를 쓰고 있었다면, 미러 push 만으로는 LFS 객체가 안 따라와요. 포인터만 옮겨지고 실제 파일이 비어요. LFS 객체는 따로 밀어줍니다.

```shell
# 소스에서 LFS 객체까지 받아오기
git lfs fetch --all

# 대상으로 LFS 객체 push
git lfs push --all github
```

> ⚠️ LFS 를 안 썼다면 이 단계는 건너뛰면 돼요. 쓰는지 모르겠으면 리포 루트에 `.gitattributes` 가 있고 거기 `filter=lfs` 가 보이는지 확인하세요.

<br>

<br>



## 6. CI/CD 재구성 — CodePipeline 을 Actions 로

코드는 옮겼으니 이제 자동화 차례예요. 1편에서 본 것처럼 CodeCommit 은 **CodePipeline + CodeBuild(`buildspec.yml`)** 조합이었죠. GitHub 으로 오면 보통 **GitHub Actions** 로 합칩니다.

`buildspec.yml` 의 빌드 단계는 거의 그대로 Actions step 으로 옮길 수 있어요.

```yaml
# CodeBuild buildspec.yml (기존)
version: 0.2
phases:
  build:
    commands:
      - npm ci
      - npm test
```

```yaml
# .github/workflows/ci.yml (옮긴 뒤)
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
```

배포까지 있었다면 AWS 리소스 접근이 필요한데, 여기서 권장은 **장기 액세스 키 대신 OIDC** 예요. GitHub Actions 가 AWS IAM 역할을 **단기 토큰**으로 assume 하게 해서, 비밀 키를 리포에 안 박아도 됩니다.

```yaml
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/github-actions-deploy
          aws-region: ap-northeast-2
```

> 🚨 위 `111122223333` 은 **AWS 계정 ID 자리**예요(예시값으로 마스킹). 그리고 마이그레이션을 핑계로 AWS 키를 리포 Secret 에 그냥 넣지 마세요. **OIDC + 역할 신뢰관계**로 가면 GitHub 에 영구 자격증명을 두지 않아 훨씬 안전합니다.

<br>

<br>



## 7. 보호 규칙·권한 재구성

CodeCommit 의 **Approval rule template**(예: "머지하려면 2명 승인")은 GitHub 의 **Branch protection rule** 로 다시 세웁니다. 개념은 1:1로 대응돼요.

| CodeCommit | GitHub |
|---|---|
| Approval rule template | Branch protection rule |
| 승인 N명 요구 | Require N approvals |
| IAM 주체 기반 통제 | 팀·CODEOWNERS 기반 |
| IAM 정책으로 push 제한 | 팀 권한 + 보호 규칙 |

```shell
# gh CLI 로 기본 브랜치 보호 규칙 걸기 (승인 1명 필수 예시)
gh api -X PUT repos/my-org/my-demo-repo/branches/main/protection \
  -F required_pull_request_reviews.required_approving_review_count=1 \
  -F enforce_admins=true \
  -F required_status_checks.strict=true \
  -f required_status_checks.contexts[]="build"
```

권한도 이 시점에 다시 설계해요. CodeCommit 에선 IAM 정책의 Resource 로 리포를 좁혔다면, GitHub 에선 **팀에 리포 역할(Read/Write/Admin)을 부여**하는 모델로 바뀝니다. 1편의 권한 비교표를 그대로 뒤집어 적용하면 됩니다.

<br>

<br>



## 8. 컷오버(전환) 체크리스트

마지막으로 **언제 소스를 닫고 GitHub 으로 완전히 넘어가느냐** 입니다. 한 번에 끊지 말고 아래 순서로 합니다.

- [x] 미러 push 완료 + 브랜치·태그·SHA 검증 (3~4장)
- [x] LFS 객체 이관 (필요 시, 5장)
- [x] GitHub Actions 빌드/테스트 녹색 확인
- [x] 보호 규칙·팀 권한 설정 (7장)
- [ ] 팀원 로컬 remote 갈아끼우기 안내
- [ ] CodeCommit 리포를 **읽기 전용**으로 전환 (IAM 에서 push 권한 회수)
- [ ] 일정 기간 후 CodeCommit 리포 보관/삭제

팀원들은 로컬에서 원격만 바꿔 끼우면 그대로 이어서 작업할 수 있어요.

```shell
# 각자 로컬에서 origin 을 새 GitHub 로 교체
git remote set-url origin https://github.com/my-org/my-demo-repo.git
git remote -v   # 바뀌었는지 확인
```

> 🚨 컷오버 직전에 **소스(CodeCommit)에 새 커밋이 더 들어오지 않도록** 먼저 읽기 전용으로 막고 마지막 미러 push 를 한 번 더 돌리는 게 안전해요. 안 그러면 전환 중에 들어온 커밋이 새 리포에 누락됩니다.

<br>

<br>



## 9. 마무리

정리하면, CodeCommit → GitHub 이관은 **git 부분(몇 분)과 주변 자동화 부분(나머지 대부분)** 으로 나뉩니다.

- **git**: `clone --mirror` → `push --mirror`, 그리고 숫자로 검증. 여기까진 거의 항상 매끄럽게 끝나요.
- **자동화**: CI/CD(Actions)·보호 규칙·권한·웹훅을 새로 세우기. PR 댓글 같은 히스토리는 안 따라오니, 중요하면 별도로 보존하세요.
- **컷오버**: 소스를 읽기 전용으로 막고 → 마지막 동기화 → 팀 원격 교체 → 보관.

1편의 비교가 "두 서비스가 어떻게 다른가" 였다면, 이 글은 "그 차이를 실제로 건너는 다리" 였어요. 두 편을 같이 읽으면 CodeCommit 환경을 운영하다 GitHub 으로 넘어가는 길이 한눈에 그려질 거예요.

<br>

<br>



일단 오늘은 여기까지.....  
두 편짜리 CodeCommit·GitHub 시리즈를 여기서 마무리할게요. 다음엔 GitHub Actions 로 AWS 배포를 OIDC 로 깔끔하게 묶는 이야기로 찾아올게요.

---

**← 이전 글:** [(1/2) AWS CodeCommit 사용법 — GitHub 과 나란히 놓고 비교하기](/coding/AWS_CodeCommit_vs_GitHub_사용법/)
