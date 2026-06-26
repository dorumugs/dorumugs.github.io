---
layout: single
title:  "(1/2) AWS CodeCommit 사용법 — GitHub 과 나란히 놓고 비교하기"
date: 2026-06-26 12:00:00 +0900
categories: coding
tag: [AWS, CodeCommit, GitHub, git, IAM, CodePipeline, GitHubActions, DevOps, 형상관리, 인증]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-26-codecommit-github/header.svg
  teaser: /assets/images/2026-06-26-codecommit-github/header.svg
description: "AWS CodeCommit 을 GitHub 사용법과 1:1로 비교합니다. 리포 생성·clone·push·PR·CI/CD 연결을 같은 작업끼리 나란히 놓고, 가장 다른 인증(IAM vs PAT)과 권한 모델까지 명령어로 정리했어요."
---

{% include series-codecommit-github.html current="1" %}

## Summary

git 자체는 어디서 쓰든 똑같습니다. `add`, `commit`, `push` 는 한 글자도 안 바뀌어요. 그런데 **그 git 저장소를 어디에 두느냐**에 따라 인증하는 방법, 권한을 주는 방법, CI/CD 를 붙이는 방법이 꽤 달라집니다.

이 글은 **AWS CodeCommit** 을 평소 익숙한 **GitHub** 와 같은 작업끼리 나란히 놓고 비교하는 글이에요. "GitHub 에선 이렇게 했는데 CodeCommit 에선 어떻게 하지?" 를 표와 명령어로 한 번에 정리했습니다.

> 💡 **이 글에서 다루는 것**
> - CodeCommit 과 GitHub 의 근본적인 차이 (호스팅 위치·인증 주체)
> - 리포 만들기 / clone / push / pull 을 양쪽 명령어로 비교
> - 가장 헷갈리는 **인증 방식** 3가지 (grc · HTTPS Git credentials · SSH)
> - PR(풀 리퀘스트)·코드리뷰 흐름 비교
> - CI/CD 연결 — CodePipeline vs GitHub Actions
> - 권한 관리 — IAM 정책 vs GitHub 조직/팀
> - ⚠️ CodeCommit 신규 가입 중단 이슈와, 그래도 알아둬야 하는 이유

> ⚠️ 먼저 짚고 갈게요. AWS 는 **2024년 7월 25일부로 CodeCommit 신규 고객 가입을 중단**했어요. 그 전부터 쓰던 계정은 계속 쓸 수 있지만, 새로 시작하는 프로젝트라면 사실상 GitHub·GitLab 쪽으로 가는 게 맞습니다. 그래도 사내에 이미 CodeCommit 으로 굴러가는 파이프라인이 많아서, 운영·이관 관점에서 사용법을 알아둘 가치는 충분해요. 이 글은 그 관점으로 정리했습니다.

<br>

<br>



## 1. 근본적인 차이 — 무엇이 같고 무엇이 다른가

둘 다 git 리포지토리를 호스팅한다는 점은 같아요. 하지만 **"저장소가 어디에 살고, 누가 접근을 허락하느냐"** 가 완전히 다릅니다.

| 구분 | AWS CodeCommit | GitHub |
|---|---|---|
| 호스팅 위치 | 내 **AWS 계정 안**의 리전 | GitHub 의 클라우드(github.com) |
| 인증 주체 | **IAM** (사용자·역할·정책) | GitHub 계정 + PAT/SSH |
| 기본 공개 범위 | 항상 **비공개** (계정 내부) | 공개/비공개 선택 |
| 권한 모델 | IAM 정책 (JSON) | 조직·팀·Role, 리포 단위 권한 |
| 코드리뷰 | Pull Request (approval rule) | Pull Request (review·필수승인) |
| CI/CD | CodePipeline · CodeBuild | GitHub Actions |
| 생태계 | AWS 서비스와 깊게 결합 | Marketplace·오픈소스 표준 |
| 비용 | 활성 사용자당 과금(프리티어 5명) | 무료/유료 플랜 |

한 줄로 요약하면 이래요. **GitHub 는 "개발자 협업 생태계의 표준"** 이고, **CodeCommit 은 "AWS 계정 경계 안에 git 을 가둬두는 서비스"** 예요. 그래서 보안·규제 때문에 코드가 AWS 밖으로 나가면 안 되는 환경에서 CodeCommit 을 선택해 왔습니다.

가장 큰 실무 차이는 **인증**이에요. GitHub 는 GitHub 계정으로 로그인하고 토큰을 발급받지만, CodeCommit 은 별도 계정 개념이 없습니다. **IAM 사용자/역할이 곧 git 사용자**예요. 이 부분만 이해하면 나머지는 거의 똑같습니다.

<br>

<br>



## 2. 사전 준비 — 접근 권한부터

### GitHub

GitHub 는 직관적이에요. 계정 만들고 → **Personal Access Token(PAT)** 발급하거나 **SSH 키**를 등록하면 끝.

```text
GitHub → Settings → Developer settings → Personal access tokens
  → Fine-grained token 생성 → 리포 권한(Contents: Read/Write) 부여
```

### CodeCommit

CodeCommit 은 IAM 에서 시작합니다. IAM 사용자(또는 역할)에 **CodeCommit 접근 정책**을 붙여야 해요. AWS 가 관리형 정책을 미리 만들어 뒀습니다.

```shell
# IAM 사용자에게 PowerUser 정책 연결 (리포 생성·삭제 빼고 거의 다 가능)
aws iam attach-user-policy \
  --user-name my-dev \
  --policy-arn arn:aws:iam::aws:policy/AWSCodeCommitPowerUser
```

대표 관리형 정책 3종을 비교하면 다음과 같아요.

| 정책 | 할 수 있는 것 |
|---|---|
| `AWSCodeCommitReadOnly` | clone·pull·읽기만 |
| `AWSCodeCommitPowerUser` | push·PR·브랜치 등 일상 작업 (리포 생성/삭제 제외) |
| `AWSCodeCommitFullAccess` | 리포 생성·삭제 포함 전체 |

> 💡 GitHub 의 "이 토큰에 어떤 리포 권한을 줄까"가 CodeCommit 에선 "이 IAM 주체에 어떤 정책을 붙일까"로 바뀐 거예요. 권한을 코드(JSON)로 관리한다는 게 가장 큰 차이입니다.

<br>

<br>



## 3. 인증 방식 — 여기가 제일 다릅니다

CodeCommit 에 git 으로 접속하는 방법은 **3가지**예요. 이 중 뭘 고르느냐가 체감 난이도를 좌우합니다. 저는 **git-remote-codecommit(grc)** 을 가장 추천드려요.

### 방법 A. git-remote-codecommit (grc) — 추천

별도 git 자격증명을 안 만들고, **이미 설정된 AWS CLI 자격증명을 그대로** 빌려 쓰는 방식이에요. MFA·임시자격·SSO 와도 잘 맞아서 제일 깔끔합니다.

```shell
# 설치 (pip)
pip install git-remote-codecommit

# clone — URL 이 git-codecommit... 가 아니라 codecommit:: 스킴
git clone codecommit::ap-northeast-2://MyDemoRepo

# 특정 AWS 프로파일을 쓰고 싶으면 profile@ 를 앞에 붙임
git clone codecommit::ap-northeast-2://my-profile@MyDemoRepo
```

clone 만 grc 로 받으면 그 다음부턴 평소처럼 `git push` / `git pull` 만 하면 돼요. AWS CLI 가 알아서 서명을 붙여줍니다.

### 방법 B. HTTPS + Git credentials

IAM 사용자에 전용 **Git 사용자명/비밀번호**를 따로 발급받아 쓰는 방식이에요. GitHub 의 PAT 와 가장 비슷한 감각입니다.

```shell
# IAM 콘솔: 사용자 → 보안 자격 증명 → "AWS CodeCommit에 대한 HTTPS Git 자격 증명" 생성
#  → 한 번만 보여주는 username/password 를 안전하게 저장

git clone https://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/MyDemoRepo
# username / password 입력 (위에서 발급받은 Git credentials)
```

> ⚠️ 이 Git credentials 는 IAM 로그인 비밀번호와 **다른** 별도 자격증명이에요. SSO·임시자격 기반 계정에선 이 메뉴 자체가 안 보일 수 있어서, 그럴 땐 방법 A(grc)로 가야 합니다.

### 방법 C. SSH

IAM 사용자에 **SSH 공개키**를 업로드하고, 발급된 **SSH Key ID** 를 사용자명으로 쓰는 방식이에요.

```shell
# 공개키를 IAM 사용자에 업로드하면 "SSH Key ID"(APKA...로 시작)가 나옴
# ~/.ssh/config 에 등록
```

```text
Host git-codecommit.*.amazonaws.com
  User APKAEXAMPLE12345
  IdentityFile ~/.ssh/codecommit_rsa
```

```shell
git clone ssh://git-codecommit.ap-northeast-2.amazonaws.com/v1/repos/MyDemoRepo
```

세 방법을 GitHub 와 묶어 비교하면 이렇게 정리돼요.

| 작업 | GitHub | CodeCommit |
|---|---|---|
| 토큰/비번 방식 | PAT (HTTPS) | Git credentials (HTTPS, 방법 B) |
| 키 방식 | SSH 키 등록 | SSH 키 IAM 업로드 (방법 C) |
| CLI 자격 재활용 | `gh auth login` | **grc** (방법 A) |

> 💡 SSO·MFA 를 쓰는 회사 계정이라면 고민하지 말고 **grc**. 단순 IAM 사용자 한 명으로 쓰는 환경이면 **HTTPS Git credentials** 가 GitHub PAT 처럼 익숙하실 거예요.

<br>

<br>



## 4. 리포 만들기

### GitHub

```shell
# gh CLI 로 비공개 리포 생성하면서 현재 폴더를 올림
gh repo create my-demo-repo --private --source=. --remote=origin --push
```

### CodeCommit

CodeCommit 은 빈 리포를 먼저 만들고, 거기에 로컬을 연결하는 흐름이에요.

```shell
# 1) 리포 생성
aws codecommit create-repository \
  --repository-name MyDemoRepo \
  --repository-description "데모용 리포"

# 2) 로컬에서 remote 연결 후 첫 push (grc 기준)
cd my-project
git init
git add .
git commit -m "first commit"
git remote add origin codecommit::ap-northeast-2://MyDemoRepo
git push -u origin main
```

리포가 잘 만들어졌는지는 CLI 로 바로 확인돼요.

```shell
aws codecommit list-repositories
aws codecommit get-repository --repository-name MyDemoRepo
```

정상이면 clone URL(HTTPS/SSH)과 리포 ARN 이 친절하게 찍혀 나옵니다.

<br>

<br>



## 5. 일상 작업 — clone · push · pull

여기서부턴 **거의 차이가 없어요.** remote 만 한 번 잡아두면 `add → commit → push` 는 동일합니다.

| 작업 | GitHub | CodeCommit |
|---|---|---|
| clone | `git clone https://github.com/me/repo.git` | `git clone codecommit::ap-northeast-2://MyDemoRepo` |
| 브랜치 생성 | `git switch -c feature/login` | `git switch -c feature/login` (동일) |
| push | `git push -u origin feature/login` | `git push -u origin feature/login` (동일) |
| pull | `git pull` | `git pull` (동일) |

즉 **다른 건 첫 remote URL 하나뿐**이고, 일단 연결되면 그 뒤 git 명령어는 토씨 하나 안 바뀝니다. 이게 "git 은 어디서나 똑같다"는 말의 실제 의미예요.

<br>

<br>



## 6. Pull Request — 코드리뷰 흐름

둘 다 PR(풀 리퀘스트) 개념이 있어요. 브랜치를 만들어 push 하고, 메인으로 합치기 전에 리뷰받는 흐름이 똑같습니다. 다만 CodeCommit 은 콘솔/CLI 중심이고, GitHub 은 웹 UI 가 훨씬 풍부해요.

### GitHub

```shell
gh pr create --base main --head feature/login --title "로그인 추가" --body "설명"
gh pr merge --squash
```

### CodeCommit

```shell
# PR 생성
aws codecommit create-pull-request \
  --title "로그인 추가" \
  --description "설명" \
  --targets repositoryName=MyDemoRepo,sourceReference=feature/login,destinationReference=main

# 머지 (squash)
aws codecommit merge-pull-request-by-squash \
  --pull-request-id 17 \
  --repository-name MyDemoRepo
```

리뷰 승인 규칙도 비교해두면 좋아요.

| 항목 | GitHub | CodeCommit |
|---|---|---|
| 필수 승인 | Branch protection rule | **Approval rule template** |
| 승인 지정 | 리뷰어 지정·CODEOWNERS | IAM 주체/숫자 기준 룰 |
| 댓글 리뷰 | 라인 코멘트 풍부 | 코멘트 가능하나 UI 단순 |

> 💡 CodeCommit 의 **Approval rule template** 은 "이 리포에 머지하려면 최소 2명 승인" 같은 규칙을 템플릿으로 만들어 여러 리포에 일괄 적용하는 기능이에요. GitHub 의 branch protection 을 조직 차원에서 찍어내는 느낌입니다.

<br>

<br>



## 7. CI/CD 연결

코드를 올린 다음 자동으로 빌드·테스트·배포하는 부분이에요. 여기서 두 서비스의 **생태계 차이**가 가장 선명하게 드러납니다.

### GitHub — Actions

리포 안에 YAML 한 장 넣으면 끝이에요. Marketplace 에 액션이 널려 있어서 조립이 빠릅니다.

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test
```

### CodeCommit — CodePipeline + CodeBuild

CodeCommit 은 push 를 **트리거(EventBridge)** 로 잡아 **CodePipeline** 을 돌리고, 실제 빌드는 **CodeBuild** 가 `buildspec.yml` 을 보고 수행해요. AWS 서비스를 조합하는 구조라 처음엔 손이 더 갑니다.

```yaml
# buildspec.yml (CodeBuild 가 읽음)
version: 0.2
phases:
  install:
    runtime-versions:
      nodejs: 20
  build:
    commands:
      - npm ci
      - npm test
```

| 항목 | GitHub Actions | CodeCommit + CodePipeline |
|---|---|---|
| 정의 위치 | `.github/workflows/*.yml` | 파이프라인(콘솔/IaC) + `buildspec.yml` |
| 실행 주체 | GitHub Runner | CodeBuild |
| 트리거 | `on: [push]` | EventBridge 이벤트 |
| 강점 | Marketplace·즉시성 | IAM·VPC·AWS 리소스 통합 |

> 💡 한쪽이 더 좋다기보단 결이 달라요. **GitHub Actions** 는 "리포 안에서 YAML 로 다 끝내는" 빠른 맛이고, **CodePipeline** 은 "AWS 권한·네트워크 경계 안에서 배포까지 통제하는" 무게감이에요. 사내망 안에서 IAM·VPC 로 배포를 통제해야 하는 환경이면 후자가 유리합니다.

<br>

<br>



## 8. 권한 관리 — IAM vs 조직/팀

마지막으로 "누구에게 뭘 허용할까"의 모델 차이예요.

GitHub 은 **조직 → 팀 → 리포** 로 권한을 묶고, 리포마다 Read/Write/Admin 같은 역할을 부여합니다. 사람이 보기 편한 모델이에요.

CodeCommit 은 전부 **IAM 정책(JSON)** 으로 내려가요. 예를 들어 "특정 리포에만 push 허용" 은 이렇게 씁니다.

```json
{
  "Effect": "Allow",
  "Action": [
    "codecommit:GitPull",
    "codecommit:GitPush"
  ],
  "Resource": "arn:aws:codecommit:ap-northeast-2:111122223333:MyDemoRepo"
}
```

| 항목 | GitHub | CodeCommit |
|---|---|---|
| 권한 단위 | 조직·팀·리포 역할 | IAM 정책의 Action·Resource |
| 세밀함 | 역할 프리셋 위주 | 액션 단위로 매우 세밀 |
| 자동화 | API/Terraform | IAM 정책(JSON)·IaC 친화 |
| 학습 곡선 | 낮음 | IAM 이해 필요 |

> 🚨 위 JSON 의 `111122223333` 은 **AWS 계정 ID 자리**예요(예시값으로 마스킹). 실제 정책엔 본인 계정 ID 가 들어갑니다. CodeCommit 권한은 git 권한이자 곧 AWS 보안 경계라서, 리포 ARN 을 정확히 좁혀 최소권한으로 주는 걸 추천드려요.

<br>

<br>



## 9. 그래서, 지금 무엇을 골라야 하나

정리하면 이렇습니다.

- **새 프로젝트** → 솔직히 **GitHub**(또는 GitLab). CodeCommit 은 2024년 7월부터 신규 가입을 안 받아서 길게 보면 선택지가 아니에요.
- **이미 CodeCommit 으로 굴러가는 사내 환경** → 사용법·인증(grc)·파이프라인 구조를 알아둬야 운영·이관이 됩니다. 이 글이 그 지도예요.
- **AWS 안에서 코드가 절대 못 나가는 규제 환경** → CodeCommit 의 IAM·VPC 통합이 여전히 강점이지만, 신규 도입은 GitLab 셀프호스팅 등 대안을 같이 검토하세요.

결국 핵심은 하나예요. **git 명령어는 똑같고, 달라지는 건 "인증·권한·CI/CD 를 누가 쥐고 있느냐"** 입니다. GitHub 은 GitHub 계정이, CodeCommit 은 IAM 이 그걸 쥐고 있어요. 이 한 축만 머리에 넣어두면 두 서비스를 오가는 게 어렵지 않습니다.

<br>

<br>



일단 오늘은 여기까지.....  
다음 글에서는 CodeCommit 에서 GitHub 으로 실제 리포를 옮기는 마이그레이션(미러 push·파이프라인 재구성) 과정을 정리해볼게요.

---

**다음 글 →:** [(2/2) CodeCommit 에서 GitHub 으로 리포 옮기기 — 미러 마이그레이션 실전](/coding/CodeCommit_GitHub_마이그레이션/)
