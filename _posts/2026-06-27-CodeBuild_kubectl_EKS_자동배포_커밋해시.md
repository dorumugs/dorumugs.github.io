---
layout: single
title:  "(2/3) latest 졸업 — 커밋해시 태그로 EKS 자동 배포하기 (CodeBuild 에서 kubectl 권한 잡기), 초보자용"
date: 2026-06-27 10:40:00 +0900
categories: coding
tag: [AWS, CodeBuild, EKS, kubectl, ECR, IAM, AccessEntry, awsauth, CICD, 초보자]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-27-codebuild-kubectl-eks-deploy/header.svg
  teaser: /assets/images/2026-06-27-codebuild-kubectl-eks-deploy/header.svg
description: "latest 태그를 졸업하고 커밋해시 태그로 EKS 에 자동 배포합니다. CodeBuild 안에서 kubectl 을 깔고, IAM 역할을 EKS 클러스터 권한(Access Entry/aws-auth)으로 잇고, 버전을 못 박아 롤백까지 가능하게 만드는 과정을 초보자 눈높이로 정리했어요."
---

{% include series-codebuild-eks.html current="2" %}

## Summary

[지난 글](/coding/CodeCommit_CodeBuild_ECR_EKS_자동빌드/)에서 main 에 push 하면 이미지가 빌드돼 ECR 에 `latest` 로 올라가는 데까지 만들었어요. 편하긴 한데 `latest` 에는 약점이 있었죠. **"지금 EKS 에서 도는 게 정확히 어느 빌드인지"** 를 알 수가 없고, 그래서 **되돌리기(롤백)** 도 애매합니다.

이번 글은 그 `latest` 를 졸업하는 편이에요. 빌드할 때 **커밋해시 태그**를 같이 올리고, CodeBuild 가 **직접 EKS 에 그 해시로 배포**하게 만듭니다. 여기서 초보자가 거의 100% 막히는 관문이 하나 있어요. **"CodeBuild 의 IAM 역할을 EKS 클러스터 안에서도 권한 있는 사용자로 인정받게 하는 일"** — 이걸 차근차근 풀어볼게요.

> 💡 **이 글에서 다루는 것**
> - 왜 `latest` 를 졸업해야 하나 — 버전 추적·롤백
> - CodeBuild 안에 `kubectl` 깔기
> - 🔑 IAM 역할 ↔ EKS 권한 잇기 — Access Entry(요즘 방식) vs aws-auth(전통 방식)
> - `buildspec.yml` 에 배포 단계 추가 — 커밋해시로 `kubectl set image`
> - 배포 확인과 롤백
> - `imagePullPolicy` 를 바꿔도 되는 이유

<br>

<br>



## 1. 왜 latest 를 졸업하나

`latest` 는 "가장 최근" 이라는 **이름표일 뿐** 내용이 계속 바뀝니다. 그래서 생기는 문제가 이거예요.

| 상황 | latest 만 쓸 때 | 커밋해시 태그를 쓸 때 |
|---|---|---|
| 지금 도는 버전 확인 | "그냥 최신"… 어느 커밋인지 모름 | `:a1b2c3d` 처럼 커밋이 그대로 보임 |
| 롤백 | 이전 이미지를 특정 못 함 | 이전 해시로 한 줄에 되돌림 |
| 같은 태그 재배포 | 캐시 때문에 안 바뀌기도 | 태그가 달라 항상 새로 받음 |
| 장애 추적 | "언제 뭐가 바뀌었지?" | 해시→커밋→변경내역 추적 |

핵심은 **불변(immutable) 태그**예요. 커밋해시는 커밋마다 유일하니까, 태그 하나가 이미지 하나를 영원히 가리킵니다. 그래서 "이 버전으로 돌려줘" 가 가능해져요.

> 💡 `latest` 를 아예 버리는 건 아니에요. `latest` 는 "가장 최근" 표식으로 같이 남겨두고, **실제 배포는 커밋해시 태그로** 못 박는 게 실무에서 흔한 절충입니다. 지난 글의 `buildspec.yml` 이 이미 둘 다 push 하고 있었어요.

<br>

<br>



## 2. CodeBuild 안에 kubectl 깔기

CodeBuild 가 EKS 에 배포하려면 `kubectl` 이 필요한데, 기본 빌드 이미지엔 없을 수 있어요. 그래서 빌드 초반에 설치해줍니다. `buildspec.yml` 의 `install` 단계에 한 토막 넣으면 돼요.

```yaml
phases:
  install:
    commands:
      - echo "kubectl 설치"
      - curl -sLO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
      - chmod +x kubectl && mv kubectl /usr/local/bin/kubectl
      - kubectl version --client
```

`kubectl version --client` 가 버전을 찍으면 설치 성공이에요. (AWS CLI 는 CodeBuild 표준 이미지에 이미 있어서 따로 안 깔아도 됩니다.)

<br>

<br>



## 3. 🔑 제일 중요한 관문 — IAM 역할을 EKS 권한으로 잇기

여기가 이 글의 핵심이에요. 많은 분들이 "권한 다 줬는데 왜 `kubectl` 이 거부당하지?" 로 한참 헤맵니다.

이유는 **EKS 에 권한이 두 겹**이라서예요.

1. **IAM 권한** — "이 역할이 EKS API(`eks:DescribeCluster` 등)를 호출해도 되나" (AWS 차원)
2. **쿠버네티스 RBAC** — "이 사용자가 클러스터 *안에서* `Deployment` 를 바꿔도 되나" (클러스터 차원)

CodeBuild 역할에 IAM 권한만 주고 끝내면, AWS 문은 통과해도 **클러스터 안에서 낯선 사람** 취급을 받아요. 그래서 **IAM 역할을 클러스터 안 사용자로 매핑**해줘야 합니다.

### 먼저 IAM 쪽 (가벼움)

CodeBuild 서비스 역할에 EKS 접근용 권한을 더해요. `update-kubeconfig` 에 필요합니다.

```json
{
  "Effect": "Allow",
  "Action": ["eks:DescribeCluster"],
  "Resource": "arn:aws:eks:ap-northeast-2:111122223333:cluster/my-cluster"
}
```

### 그다음 클러스터 쪽 (진짜 관문)

두 가지 방법이 있어요. **요즘은 Access Entry 가 권장**됩니다.

**방법 A. EKS Access Entry (요즘 방식, 추천)** — `aws-auth` 를 직접 안 건드려서 실수 위험이 적어요.

```shell
# 1) CodeBuild 역할을 클러스터 접근 주체로 등록
aws eks create-access-entry \
  --cluster-name my-cluster \
  --principal-arn arn:aws:iam::111122223333:role/codebuild-deploy-role \
  --type STANDARD

# 2) 그 주체에 권한 정책 연결 (default 네임스페이스 편집 권한 예시)
aws eks associate-access-policy \
  --cluster-name my-cluster \
  --principal-arn arn:aws:iam::111122223333:role/codebuild-deploy-role \
  --access-policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSEditPolicy \
  --access-scope type=namespace,namespaces=default
```

**방법 B. aws-auth ConfigMap (전통 방식)** — 예전부터 쓰던 방법. 클러스터의 `aws-auth` 를 편집해 역할을 매핑해요.

```yaml
# kubectl edit configmap aws-auth -n kube-system 안의 mapRoles 에 추가
- rolearn: arn:aws:iam::111122223333:role/codebuild-deploy-role
  username: codebuild-deployer
  groups:
    - eks-deployer        # 미리 RoleBinding 으로 권한을 묶어둔 그룹
```

> 🚨 `aws-auth` 매핑에 `system:masters` 그룹을 넣으면 **클러스터 관리자 전권**이 가요. 편하다고 그렇게 두지 말고, 배포에 필요한 만큼만 묶은 그룹(또는 Access Entry 의 네임스페이스 한정 정책)으로 **최소권한**을 주세요. CI 역할이 관리자 권한을 들고 있는 건 사고의 지름길이에요.

> 💡 둘 중 헷갈리면 **Access Entry(방법 A)** 로 가세요. `aws-auth` 는 한 글자만 틀려도 클러스터 접근이 통째로 막히는 위험한 파일이라, 콘솔/CLI 로 깔끔하게 다루는 Access Entry 가 초보자에게 훨씬 안전합니다.

<br>

<br>



## 4. buildspec.yml 에 배포 단계 추가

이제 빌드 끝에 **배포**를 붙입니다. 지난 글의 `buildspec.yml` 에서 이미지 push 까지는 그대로 두고, `post_build` 에 EKS 배포를 더해요.

```yaml
  post_build:
    commands:
      - echo "ECR push (latest + 커밋해시)"
      - docker push $REGISTRY/$IMAGE_REPO_NAME:latest
      - docker push $REGISTRY/$IMAGE_REPO_NAME:$IMAGE_TAG
      - echo "EKS 접속 설정"
      - aws eks update-kubeconfig --name my-cluster --region $AWS_DEFAULT_REGION
      - echo "커밋해시 태그로 배포 — 여기가 latest 졸업의 핵심"
      - kubectl set image deployment/my-app my-app=$REGISTRY/$IMAGE_REPO_NAME:$IMAGE_TAG
      - echo "배포가 끝날 때까지 기다리고 결과 확인"
      - kubectl rollout status deployment/my-app --timeout=120s
```

지난 글과 달라진 건 마지막 세 줄이에요.

- **`aws eks update-kubeconfig`** — CodeBuild 안에서 우리 클러스터에 접속할 설정을 만들어요. (3장의 IAM·클러스터 권한이 여기서 효력을 냅니다.)
- **`kubectl set image ... :$IMAGE_TAG`** — `latest` 가 아니라 **이번 커밋해시**로 이미지를 콕 집어 바꿔요. 이게 "버전 못 박기" 예요.
- **`kubectl rollout status`** — 새 파드가 정상으로 뜰 때까지 기다렸다가, 실패하면 빌드도 실패로 끝내줘요. "배포했다 치고 넘어가는" 사고를 막습니다.

> 💡 `IMAGE_TAG` 는 지난 글에서 `CODEBUILD_RESOLVED_SOURCE_VERSION` 의 앞 7자리로 만들어 둔 그 값이에요. 빌드와 배포가 **같은 해시**를 쓰니, 방금 구운 이미지가 그대로 배포됩니다.

<br>

<br>



## 5. 배포 확인과 롤백

배포가 잘 됐는지, 그리고 문제가 생겼을 때 어떻게 되돌리는지가 `latest` 졸업의 진짜 보상이에요.

```shell
# 지금 무슨 이미지가 도는지 — 해시가 그대로 보임
kubectl get deployment my-app -o jsonpath='{.spec.template.spec.containers[0].image}'

# 배포 이력 보기
kubectl rollout history deployment/my-app
```

문제가 생기면 되돌리는 방법은 두 가지예요.

```shell
# 방법 1) 바로 직전 버전으로 되돌리기
kubectl rollout undo deployment/my-app

# 방법 2) 특정 커밋해시로 콕 집어 되돌리기 (불변 태그라 가능)
kubectl set image deployment/my-app my-app=$REGISTRY/my-app:<예전_해시>
```

> ✅ `latest` 만 쓸 땐 "예전 이미지" 를 가리킬 방법이 없어서 방법 2 가 불가능했어요. 커밋해시 태그로 졸업하면 **"그때 그 버전으로"** 가 한 줄로 됩니다. 이게 불변 태그를 쓰는 가장 큰 이유예요.

<br>

<br>



## 6. imagePullPolicy 도 바꿔도 돼요

지난 글에서는 `latest` 라서 `imagePullPolicy: Always` 가 필요했죠. 같은 태그라도 매번 새로 당겨와야 했으니까요. 이제 태그가 **커밋마다 달라지니까**, 굳이 매번 안 당겨도 됩니다.

```yaml
# deployment.yaml
        - name: my-app
          image: 111122223333.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:a1b2c3d
          imagePullPolicy: IfNotPresent
```

태그가 바뀌면 쿠버네티스가 "새 이미지네" 하고 알아서 당겨와요. `kubectl rollout restart` 로 억지로 흔들 필요도 없어졌습니다. **태그를 바꾸는 것 자체가 곧 배포 신호**가 되는 거예요.

> 🚨 위 `111122223333` 은 **AWS 계정 ID 자리**(예시값으로 마스킹)예요. 그리고 ECR 리포를 **태그 불변(immutable)** 으로 설정해두면, 같은 해시 태그를 실수로 덮어쓰는 일까지 막아줘서 더 안전합니다.

<br>

<br>



## 7. 정리 — latest 졸업으로 얻은 것

지난 글이 "push 하면 빌드되고 ECR 에 올라간다" 였다면, 이번 글로 **"push 하면 그 커밋이 EKS 까지 자동으로 배포된다"** 가 됐어요.

- [x] 커밋해시 태그로 배포 — 어느 버전인지 항상 추적 (1장)
- [x] CodeBuild 에 `kubectl` 설치 (2장)
- [x] 🔑 IAM 역할 ↔ EKS 권한 잇기 (Access Entry 추천, 3장) — 제일 많이 막히는 곳
- [x] `buildspec.yml` 에 `update-kubeconfig` + `set image` + `rollout status` (4장)
- [x] 해시로 콕 집는 롤백 (5장)
- [x] `imagePullPolicy: IfNotPresent` 로 전환 (6장)

가장 강조하고 싶은 한 가지는 3장이에요. **"IAM 권한과 쿠버네티스 RBAC 은 별개"** 라는 것만 머리에 박아두면, EKS 자동화에서 만나는 권한 문제의 절반은 미리 피할 수 있어요.

<br>

<br>



일단 오늘은 여기까지.....  
이번 편으로 CodeCommit 한 번의 push 가 EKS 배포까지 이어지는 길이 완성됐어요. 다음엔 배포 전에 자동으로 테스트를 돌리고, 실패하면 배포를 멈추는 **품질 게이트**를 붙이는 이야기로 찾아올게요.

---

**← 이전 글:** [(1/3) CodeCommit main 푸시 → CodeBuild 자동 빌드 → ECR(자동생성·latest) → EKS](/coding/CodeCommit_CodeBuild_ECR_EKS_자동빌드/) ｜ **다음 글 →:** [(3/3) 배포 전 자동 테스트 품질 게이트 — 실패하면 배포를 멈추기](/coding/배포전_자동테스트_품질게이트/)
