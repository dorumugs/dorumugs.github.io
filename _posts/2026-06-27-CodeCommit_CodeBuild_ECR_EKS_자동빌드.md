---
layout: single
title:  "CodeCommit main 푸시 → CodeBuild 자동 빌드 → ECR(자동생성·latest) → EKS, 초보자용"
date: 2026-06-27 10:00:00 +0900
categories: coding
tag: [AWS, CodeCommit, CodeBuild, ECR, EKS, EventBridge, buildspec, Docker, CICD, 초보자]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-27-codecommit-codebuild-ecr-eks/header.svg
  teaser: /assets/images/2026-06-27-codecommit-codebuild-ecr-eks/header.svg
description: "CodeCommit main 브랜치에 push 하면 EventBridge 가 CodeBuild 를 깨워 도커 이미지를 빌드하고, ECR 리포를 없으면 자동 생성한 뒤 latest 태그로 push, EKS 까지 연동하는 흐름을 초보자 눈높이로 처음부터 설정합니다."
---

## Summary

직접 손으로 `docker build` 하고 `docker push` 하고 `kubectl set image` 하는 거, 처음 몇 번은 괜찮은데 금방 지겨워져요. 빠뜨리기도 하고요. 그래서 이번 글에서는 **"main 에 push 만 하면 알아서 이미지가 빌드되고 ECR 에 올라가는"** 파이프라인을 **초보자 눈높이로 처음부터** 만들어 봅니다.

목표 흐름은 이래요. CodeCommit `main` 브랜치에 push → **EventBridge** 가 변경을 감지해 **CodeBuild** 를 깨움 → CodeBuild 가 도커 이미지를 빌드하고, **ECR 리포가 없으면 자동으로 만든 다음** `latest` 태그로 push → 그 이미지를 **EKS** 가 받아 굴림. 한 번 세팅해두면 그 뒤론 `git push` 가 곧 배포 준비예요.

> 💡 **이 글에서 다루는 것**
> - 전체 그림과 등장 인물(CodeCommit·EventBridge·CodeBuild·ECR·EKS)
> - CodeBuild 가 쓸 IAM 권한(서비스 역할) 세팅 — 초보자가 제일 많이 막히는 곳
> - `buildspec.yml` 작성 — ECR 자동 생성 + `latest` 태그 push 까지
> - main 푸시를 빌드 트리거로 거는 EventBridge 규칙
> - EKS 가 새 이미지를 받게 하는 마지막 연결
> - `latest` 태그만 쓸 때의 함정과 안전장치

> 💡 이 글은 CodeCommit 을 다룬 [지난 비교 글](/coding/AWS_CodeCommit_vs_GitHub_사용법/)에서 한 발 더 들어가는 실습편이에요. CodeCommit 접근·인증이 아직 낯설면 그 글을 먼저 보고 오시면 수월합니다.

<br>

<br>



## 1. 전체 그림부터 — 누가 무슨 일을 하나

자동화라고 하면 막연한데, 등장 인물 5명이 **바통을 넘기는 릴레이**라고 보면 쉬워요.

| 순서 | 누가 | 무슨 일을 하나 |
|---|---|---|
| 1 | **CodeCommit** | 코드 저장소. `main` 에 push 가 들어옴 |
| 2 | **EventBridge** | "main 이 바뀌었다"는 이벤트를 듣고 다음 주자를 깨움 |
| 3 | **CodeBuild** | 도커 이미지를 빌드하는 일꾼. `buildspec.yml` 대로 움직임 |
| 4 | **ECR** | 빌드된 이미지를 보관하는 창고. 없으면 자동 생성 |
| 5 | **EKS** | 그 이미지를 받아 실제로 굴리는 쿠버네티스 |

핵심 두 개만 기억하면 돼요. **트리거**는 EventBridge 가 잡고, **실제 빌드**는 CodeBuild 가 `buildspec.yml` 을 보고 합니다. 나머지(ECR 생성·태그·push)는 전부 그 `buildspec.yml` 안에 적어둘 거예요.

<br>

<br>



## 2. 사전 준비 — 무엇이 이미 있어야 하나

시작 전에 이 정도는 갖춰져 있다고 가정할게요.

- ✅ CodeCommit 에 리포가 하나 있고, 루트에 **`Dockerfile`** 이 있다.
- ✅ EKS 클러스터가 이미 떠 있고 `kubectl` 로 접속된다.
- ✅ AWS CLI 가 설치돼 있고 적절한 권한으로 로그인돼 있다.

리포 루트에 가장 단순한 `Dockerfile` 이 이런 모양이라고 해볼게요.

```dockerfile
FROM nginx:1.27-alpine
COPY ./html /usr/share/nginx/html
```

별거 없어요. 이걸 이미지로 구워 ECR 에 올리는 게 이번 목표입니다.

<br>

<br>



## 3. CodeBuild 권한(IAM 서비스 역할) — 여기서 제일 많이 막혀요

초보자가 가장 자주 걸리는 곳이 **권한**이에요. CodeBuild 는 자기 이름으로 ECR 을 만들고 이미지를 push 해야 하는데, 그러려면 **서비스 역할(service role)** 에 권한이 붙어 있어야 합니다.

CodeBuild 프로젝트를 만들면 보통 역할이 자동 생성되는데, 거기에 **ECR 권한을 추가로** 붙여줘야 해요. 아래 정책을 서비스 역할에 연결합니다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository",
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

> 🚨 `ecr:GetAuthorizationToken` 은 리소스를 특정할 수 없어 `Resource: "*"` 가 맞지만, 나머지 ECR 액션은 실서비스에선 **특정 리포 ARN 으로 좁히는** 걸 추천드려요. 처음 연습할 땐 위 형태로 시작하고, 동작을 확인한 뒤 최소권한으로 조여가세요.

CodeBuild 가 CodeCommit 소스를 받아오는 권한(`codecommit:GitPull`)과 로그 권한(`logs:*`)은 프로젝트 생성 시 기본 역할에 대개 포함돼요. ECR 만 빠져서 막히는 경우가 대부분입니다.

<br>

<br>



## 4. buildspec.yml — 이 파일이 진짜 핵심

CodeBuild 가 실제로 하는 일은 전부 리포 루트의 **`buildspec.yml`** 에 적혀요. "ECR 자동 생성 + `latest` 태그 push" 도 여기서 처리합니다. 한 줄씩 뜯어볼게요.

```yaml
version: 0.2

env:
  variables:
    IMAGE_REPO_NAME: "my-app"     # ECR 리포 이름 (없으면 만들 거예요)

phases:
  pre_build:
    commands:
      - echo "계정/레지스트리 주소 계산"
      - ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
      - REGISTRY=$ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
      - echo "ECR 리포가 없으면 자동 생성"
      - aws ecr describe-repositories --repository-names $IMAGE_REPO_NAME || aws ecr create-repository --repository-name $IMAGE_REPO_NAME
      - echo "ECR 로그인"
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $REGISTRY
      - echo "커밋 해시 앞 7자리를 추가 태그로"
      - IMAGE_TAG=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c1-7)
  build:
    commands:
      - echo "이미지 빌드 — 기본 태그가 latest 가 됩니다"
      - docker build -t $IMAGE_REPO_NAME .
      - docker tag $IMAGE_REPO_NAME:latest $REGISTRY/$IMAGE_REPO_NAME:latest
      - docker tag $IMAGE_REPO_NAME:latest $REGISTRY/$IMAGE_REPO_NAME:$IMAGE_TAG
  post_build:
    commands:
      - echo "ECR 로 push (latest + 커밋해시 둘 다)"
      - docker push $REGISTRY/$IMAGE_REPO_NAME:latest
      - docker push $REGISTRY/$IMAGE_REPO_NAME:$IMAGE_TAG
```

초보자가 헷갈리기 쉬운 지점만 짚을게요.

- **ECR 자동 생성**은 마법이 아니라 한 줄이에요. `describe-repositories` 로 있는지 보고, 없어서 에러가 나면 `||` 뒤의 `create-repository` 가 실행됩니다. 이미 있으면 그냥 통과.
- **`docker build -t my-app .`** 는 태그를 안 붙이면 자동으로 `my-app:latest` 가 돼요. 그래서 "latest 태그 달기" 가 따로 필요 없이 기본으로 생깁니다.
- **`CODEBUILD_RESOLVED_SOURCE_VERSION`** 은 CodeBuild 가 자동으로 채워주는 환경변수로, 이번 빌드의 커밋 해시예요. `latest` 와 함께 **커밋 해시 태그**도 같이 올리면 "지금 도는 게 정확히 어느 커밋인지" 추적할 수 있어요.

> 💡 `latest` 만 올리면 편하지만 "어느 버전인지" 가 사라져요. 그래서 위처럼 **`latest` + 커밋해시** 두 개를 같이 push 하는 패턴을 추천드립니다. 운영 배포는 커밋해시 태그로 고정하고, `latest` 는 "가장 최근" 표식으로만 쓰는 식이에요.

<br>

<br>



## 5. CodeBuild 프로젝트 만들기

이제 위 `buildspec.yml` 을 실행할 CodeBuild 프로젝트를 만듭니다. 콘솔에서 만드는 게 초보자에겐 편하지만, 어떤 값이 중요한지 알아야 하니 핵심만 정리할게요.

| 설정 | 값 | 왜 |
|---|---|---|
| Source | CodeCommit + 해당 리포 | 빌드할 코드 출처 |
| Environment image | `aws/codebuild/standard` 최신 | 빌드 OS·도구 |
| **Privileged** | **켜기(ON)** | 도커 빌드하려면 필수 |
| Service role | 3장에서 ECR 권한 붙인 역할 | push 권한 |
| Buildspec | `buildspec.yml` 사용 | 4장 파일 |

> 🚨 **Privileged mode 를 안 켜면** CodeBuild 안에서 `docker build` 가 "Cannot connect to the Docker daemon" 으로 죽어요. 도커 인 도커가 필요해서 그래요. 초보자 단골 실수라 꼭 체크하세요.

잘 만들었는지는 한 번 수동 실행으로 확인해요.

```shell
# 프로젝트를 수동으로 한 번 돌려보기
aws codebuild start-build --project-name my-app-build
```

빌드 로그 끝에 ECR push 가 성공으로 찍히고, 콘솔의 ECR 에 `my-app` 리포와 `latest` 태그가 생겼으면 절반은 끝난 거예요.

<br>

<br>



## 6. 트리거 — main 푸시에 자동으로 반응하게

지금까진 손으로 `start-build` 했죠. 이걸 **`main` 에 push 가 들어오면 자동으로** 돌게 만드는 게 EventBridge 예요.

규칙은 "이 CodeCommit 리포의 `main` 브랜치가 업데이트되면 → CodeBuild 를 시작하라" 입니다.

```shell
# 1) main 브랜치 변경을 잡는 이벤트 규칙
aws events put-rule \
  --name codecommit-main-to-codebuild \
  --event-pattern '{
    "source": ["aws.codecommit"],
    "detail-type": ["CodeCommit Repository State Change"],
    "resources": ["arn:aws:codecommit:ap-northeast-2:111122223333:my-app"],
    "detail": {
      "event": ["referenceUpdated"],
      "referenceType": ["branch"],
      "referenceName": ["main"]
    }
  }'

# 2) 그 규칙의 타깃으로 CodeBuild 프로젝트를 연결
aws events put-targets \
  --rule codecommit-main-to-codebuild \
  --targets 'Id=1,Arn=arn:aws:codebuild:ap-northeast-2:111122223333:project/my-app-build,RoleArn=arn:aws:iam::111122223333:role/eventbridge-start-build'
```

> 🚨 위 `111122223333` 은 **AWS 계정 ID 자리**(예시값으로 마스킹)이고, `arn:...` 들도 본인 환경 값으로 바꿔야 해요. 그리고 EventBridge 가 CodeBuild 를 깨우려면 **`eventbridge-start-build` 역할에 `codebuild:StartBuild` 권한**이 있어야 합니다. 이 역할 권한을 빠뜨리면 push 해도 빌드가 안 돌아서 "왜 안 되지" 로 한참 헤매요.

이제 진짜로 테스트해봅니다.

```shell
git commit -m "trigger test" --allow-empty
git push origin main
```

CodeBuild 콘솔에서 빌드가 **자동으로** 시작되면 트리거 성공이에요. 🎉

<br>

<br>



## 7. EKS 가 새 이미지를 받게 하기

이미지는 ECR 에 올라갔으니, 마지막은 **EKS 가 그 새 이미지를 가져가게** 하는 거예요. 배포 매니페스트가 ECR 이미지를 바라보게 해둡니다.

```yaml
# deployment.yaml (일부)
spec:
  template:
    spec:
      containers:
        - name: my-app
          image: 111122223333.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
          imagePullPolicy: Always
```

여기서 `latest` 태그의 특성이 중요해요. 태그 이름이 `latest` 로 **그대로**라서, 쿠버네티스 입장에선 "이미지가 바뀐 걸" 못 알아챕니다. 그래서 새 이미지를 받게 하려면 **롤아웃을 한 번 흔들어줘야** 해요.

```shell
# 같은 latest 태그라도 파드를 새로 띄워 최신 이미지를 당겨오게
kubectl rollout restart deployment my-app
```

> ⚠️ `latest` + `imagePullPolicy: Always` + `rollout restart` 조합은 초보 단계에서 가장 간단한 방법이에요. 다만 "지금 도는 게 정확히 어느 빌드인지" 가 흐려지는 단점이 있어요. 한 단계 성장하면 4장의 **커밋해시 태그**로 배포해서 (`kubectl set image ... my-app:<해시>`) 버전을 못 박는 방식으로 넘어가는 걸 추천드립니다.

이 마지막 `rollout restart` 까지 자동화하려면 `buildspec.yml` 의 `post_build` 에 `aws eks update-kubeconfig` 와 `kubectl rollout restart` 를 더하면 되는데, 그러려면 CodeBuild 역할에 EKS 접근 권한과 클러스터 RBAC 설정이 더 필요해요. 이건 다음 단계 주제로 남겨둘게요.

<br>

<br>



## 8. 정리 — 한 번 세팅하면 남는 것

처음 세팅이 손이 좀 가지만, 한 번 만들어두면 다음부터는 단순해져요.

- [x] CodeBuild 서비스 역할에 ECR 권한 (3장) — 제일 많이 막히는 곳
- [x] `buildspec.yml` 에 ECR 자동생성 + `latest`·커밋해시 push (4장)
- [x] CodeBuild 프로젝트 Privileged mode ON (5장)
- [x] EventBridge 로 main 푸시 → 빌드 트리거 (6장)
- [x] EKS 배포 + `rollout restart` 로 최신 이미지 반영 (7장)

이제 코드를 고치고 `git push origin main` 만 하면, 그 뒤는 파이프라인이 알아서 이미지를 굽고 ECR 에 올려둡니다. **"push 가 곧 빌드"** 가 된 거예요.

<br>

<br>



일단 오늘은 여기까지.....  
다음 글에서는 `latest` 를 졸업하고 **커밋해시 태그로 EKS 까지 자동 배포**(CodeBuild 에서 `kubectl` 권한 잡기)를 정리해볼게요.
