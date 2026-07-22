---
layout: single
title:  "(5/5) Airflow on EKS — ECR 로 이미지 굽고 GitHub Actions 로 자동화"
date: 2026-06-05 10:00:00 +0900
description: "K8s 시리즈의 AWS 응용편. EKS 위에 올린 Airflow 가 ECR 의 워커 이미지를 자동으로 쓰게 만들고, 그 이미지를 GitHub Actions 가 OIDC 로 ECR 에 푸시하게 잇는 흐름."
categories: coding
tag: [Airflow, EKS, ECR, GitHub Actions, AWS, OIDC, IRSA, KubernetesExecutor, CI/CD, DevOps]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-05-airflow-eks-ecr-actions/header.svg
  teaser: /assets/images/2026-06-05-airflow-eks-ecr-actions/header.svg
series: airflow-k8s-helm
series_order: 5
---

{% include series-airflow-k8s-helm.html current="5" %}

# Summary

(1/5) ~ (4/5) 가 일반 K8s 위에서의 Airflow 였다면, 이번 편은 같은 셋업을 **AWS 위**로 옮겨요. 세 조각을 하나로 잇는 게 목표예요.

- **EKS** — Airflow 가 돌아가는 K8s 클러스터
- **ECR** — 워커가 쓸 Airflow 이미지를 보관할 사설 레지스트리
- **GitHub Actions** — 코드 푸시 한 번에 이미지가 ECR 로 올라가게 하는 CI

(3/5) 의 `Dockerfile` 과 (4/5) 의 운영 체크리스트는 그대로 살아있어요. 여기선 **레지스트리/권한/자동화** 만 AWS 식으로 교체해요.

> 💡 이 글에서 다루는 것
> - 큰 그림 — GitHub Actions → ECR → EKS Airflow 한 그림
> - ECR 레포 만들기
> - EKS 가 ECR 이미지를 받아올 수 있게 하기 (노드 IAM 정책 한 줄)
> - GitHub Actions 가 ECR 로 푸시하게 하기 (OIDC + IAM Role)
> - Helm values 에 ECR 이미지 박고 `helm upgrade`
> - DAG 한 번 굴려서 워커 Pod 가 ECR 이미지로 뜨는지 확인



<br>

<br>



## 1. 전체 흐름 한 그림

```text
[GitHub Repo]
   │  git push (main)
   ▼
[GitHub Actions]
   │  OIDC 로 IAM Role assume
   │  aws ecr login → docker build → docker push
   ▼
[Amazon ECR]  123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/airflow:<sha>
   │
   │  EKS 노드가 ECR 에서 이미지 풀
   ▼
[EKS Cluster — Namespace: airflow]
   ├─ scheduler / api-server / triggerer / dag-processor (Deployment)
   └─ Worker Pod (KubernetesExecutor 가 태스크마다 새로 띄움)
        ← 같은 ECR 태그 이미지로 뜸
```

핵심은 **이미지 태그가 한 줄로 흐른다** 는 거예요. GitHub 의 커밋 SHA 가 그대로 ECR 태그가 되고, 그 태그를 Helm `values.yaml` 의 `defaultAirflowTag` 가 받아요. 코드와 워커 이미지의 버전이 항상 1:1 로 묶입니다.



<br>

<br>



## 2. 전제 — EKS 클러스터와 Namespace

이 글은 EKS 클러스터가 이미 있다고 가정해요. 새로 만들 거면 가장 흔한 패턴은 `eksctl` 한 줄이에요.

```shell
eksctl create cluster \
  --name data-platform \
  --region ap-northeast-2 \
  --version 1.30 \
  --nodegroup-name workers \
  --node-type m6i.xlarge \
  --nodes 2 --nodes-min 2 --nodes-max 6 \
  --managed
```

(2/5) 와 똑같이 Airflow 용 `Namespace` 와 두 개의 `Secret` 부터 박아둬요.

```shell
aws eks update-kubeconfig --name data-platform --region ap-northeast-2
kubectl create namespace airflow

# Fernet / API server 시크릿은 (2/5) 의 명령 그대로
```

> 💡 EKS 도 그냥 K8s 라서 (2/5) ~ (4/5) 의 명령이 그대로 먹어요. AWS 색이 진해지는 건 다음 섹션부터예요.



<br>

<br>



## 3. ECR 레포 만들기

워커 이미지를 보관할 사설 레지스트리예요. 한 번만 만들면 돼요.

```shell
aws ecr create-repository \
  --repository-name airflow \
  --region ap-northeast-2 \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability IMMUTABLE
```

옵션 두 개 짚어둘게요.

- `scanOnPush=true` — 푸시할 때마다 ECR 이 알아서 취약점 스캔.
- `IMMUTABLE` — **같은 태그를 두 번 못 쓰게** 강제. (3/5) 에서 봤듯 `:latest` 같이 가변 태그를 쓰면 노드별 캐시가 어긋나서 같은 태그인데 다른 이미지가 떠있는 사고가 나요. IMMUTABLE 이면 처음부터 그런 사고가 불가능해요.

레지스트리 호스트는 `<aws-account-id>.dkr.ecr.<region>.amazonaws.com` 모양으로 고정이에요. 예시는 이렇게 가정할게요.

```text
123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/airflow
```



<br>

<br>



## 4. EKS 가 ECR 이미지를 받아올 수 있게

ECR 은 사설 레지스트리지만 EKS 노드는 **노드 IAM 역할에 정책 한 줄만 있으면** 자격증명 없이 풀해요. 사설 레지스트리인데 `imagePullSecrets` 가 필요 없는 게 ECR + EKS 조합의 큰 장점이에요.

확인 포인트는 노드 그룹의 IAM Role 에 `AmazonEC2ContainerRegistryReadOnly` 정책이 붙어있는지예요. `eksctl` 로 만든 클러스터는 기본으로 붙어있어요.

```shell
# 노드 그룹의 IAM Role 확인
aws eks describe-nodegroup \
  --cluster-name data-platform \
  --nodegroup-name workers \
  --query 'nodegroup.nodeRole' --output text
# 출력 예: arn:aws:iam::123456789012:role/eksctl-data-platform-nodegroup-...

# 그 Role 에 정책이 붙어있나
aws iam list-attached-role-policies \
  --role-name <위에서_나온_role_name> \
  --query 'AttachedPolicies[].PolicyName'
# 출력에 AmazonEC2ContainerRegistryReadOnly 가 있으면 OK
```

없으면 한 줄 추가.

```shell
aws iam attach-role-policy \
  --role-name <node-role-name> \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
```

> ✅ 이게 끝이에요. EKS Pod 의 `imagePullSecrets` 안 채워도 ECR 풀이 됩니다. (3/5) 에서 사설 Harbor 쓸 때 `docker-registry` 시크릿을 박았던 단계가 통째로 사라져요.



<br>

<br>



## 5. GitHub Actions 가 ECR 로 푸시하게

여기가 이번 글의 진짜 본편이에요. 두 단계로 나뉘어요.

1. **AWS 쪽** — GitHub OIDC 를 신뢰하는 IAM Role 만들기 (영구 액세스 키 박지 않기 위해서)
2. **GitHub Actions 쪽** — 그 Role 을 assume → ECR 로그인 → 빌드/푸시

### 5-1. GitHub OIDC Provider 등록

처음 한 번만.

```shell
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 5-2. ECR 푸시 권한을 가진 IAM Role

`trust-policy.json` 으로 GitHub Actions 만 이 Role 을 assume 할 수 있게 잠가요.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:dorumugs/airflow-images:ref:refs/heads/main"
      }
    }
  }]
}
```

> 🚨 `sub` 의 `repo:<owner>/<repo>:ref:refs/heads/<branch>` 가 가장 중요한 보안 줄이에요. 이 줄 없이 `*` 로 풀어두면 **누구의 GitHub Actions 든** 이 Role 을 assume 할 수 있게 돼요. 반드시 본인 repo + 브랜치로 고정.

Role 을 만들고 ECR 푸시 정책을 붙여요.

```shell
aws iam create-role \
  --role-name github-actions-airflow-ecr \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name github-actions-airflow-ecr \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
```

생성된 Role ARN 을 메모해둬요. 예: `arn:aws:iam::123456789012:role/github-actions-airflow-ecr`.

### 5-3. GitHub Actions 워크플로

레포 안 `.github/workflows/build-airflow-image.yml`.

```yaml
name: Build & Push Airflow Image

on:
  push:
    branches: [main]
    paths:
      - 'Dockerfile'
      - 'requirements.txt'
      - '.github/workflows/build-airflow-image.yml'

permissions:
  id-token: write    # OIDC 토큰 발급에 필수
  contents: read

env:
  AWS_REGION: ap-northeast-2
  ECR_REPOSITORY: airflow
  ROLE_ARN: arn:aws:iam::123456789012:role/github-actions-airflow-ecr

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ env.ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & push
        env:
          REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          TAG: 3.1.8-py311-${{ github.sha }}
        run: |
          docker build -t $REGISTRY/$ECR_REPOSITORY:$TAG .
          docker push $REGISTRY/$ECR_REPOSITORY:$TAG
          echo "image=$REGISTRY/$ECR_REPOSITORY:$TAG" >> $GITHUB_OUTPUT
```

핵심 한 줄은 `id-token: write` 권한이에요. 이게 없으면 OIDC 토큰이 안 발급돼서 `configure-aws-credentials` 가 실패해요.

태그는 **베이스 Airflow 버전 + 커밋 SHA** 패턴이 깔끔해요. 어떤 베이스 위에서 어떤 코드로 만들어진 이미지인지 한눈에 잡혀요.

> ✅ 푸시 후 ECR 콘솔에서 새 태그가 보이면 1차 성공. `aws ecr describe-images --repository-name airflow` 로도 확인 가능.



<br>

<br>



## 6. Helm values 에 ECR 이미지 박기

(3/5) 의 `values.yaml` 에서 레지스트리 주소만 ECR 로 바꾸면 끝이에요. `imagePullSecrets` 가 필요 없는 게 ECR + EKS 의 미덕.

```yaml
# values.yaml (변경)
defaultAirflowRepository: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/airflow
defaultAirflowTag: "3.1.8-py311-<sha>"   # GHA 가 푸시한 태그 그대로
airflowVersion: "3.1.8"

# (3/5) 의 registry.secretName 블록은 통째로 삭제 — ECR 은 노드 IAM 으로 풀
```

배포는 (2/5) 와 같아요.

```shell
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow \
  --values values.yaml \
  --timeout 10m

kubectl -n airflow rollout status deploy/airflow-scheduler
```

> 💡 **자동 helm upgrade** 까지 잇고 싶으면 GitHub Actions 잡 끝에 한 단계 더 붙이면 돼요. 흔한 패턴 두 가지.
> - **values 파일을 같은 repo 에 두고** GHA 가 `sed -i "s/defaultAirflowTag:.*/defaultAirflowTag: \"$TAG\"/" values.yaml` 한 뒤 자동 커밋 → Argo CD 가 감지해서 배포.
> - **GHA 가 직접 `helm upgrade`** — kubeconfig 발급용 IAM 권한과 `aws eks update-kubeconfig` 단계를 추가. 단순한 환경엔 충분하지만 클러스터 자격증명을 CI 가 직접 들고 다닌다는 부담이 있어요.



<br>

<br>



## 7. 워커 Pod 가 ECR 이미지로 뜨는지 확인

가벼운 DAG 한 개 굴려놓고 (3/5) 의 확인 절차를 그대로 돌려요.

```shell
kubectl -n airflow get pods -w \
  | grep -v -E "scheduler|api-server|triggerer|dag-processor|postgresql"
```

태스크가 도는 동안 워커 Pod 를 잡아서 `describe` 찍어봐요.

```shell
kubectl -n airflow describe pod <worker-pod-name>
```

확인 포인트.

- [x] `Image:` 가 `123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/airflow:3.1.8-py311-<sha>`
- [x] `Successfully pulled image` 이벤트가 떴고 `ImagePullBackOff` 가 아님
- [x] (3/5) 의 리소스/노드셀렉터 항목도 그대로 적용됨

여기까지 맞으면 GitHub push → ECR → EKS 워커 Pod 까지 한 줄로 흐르는 거예요.

> ⚠️ `ImagePullBackOff` 가 뜨면 거의 항상 노드 IAM 의 `AmazonEC2ContainerRegistryReadOnly` 가 안 붙어있는 거예요. 4번 섹션으로 다시.



<br>

<br>



## 8. 정리 — 시리즈를 닫으며

5편을 통과하면서 Airflow on K8s 의 풀스택이 한 번 그려졌어요.

| 편 | 핵심 |
|---|---|
| (1/5) 개요 | 큰 그림, KubernetesExecutor 가 워커 Pod 를 어떻게 다루는가 |
| (2/5) 설치 | Helm 차트 한 방으로 7~8 개 K8s 오브젝트 묶음 깔기 |
| (3/5) 워커 이미지 | 커스텀 이미지 빌드 + `pod_template_file` 로 워커 Pod 스펙 결정 |
| (4/5) 운영 | git-sync, 로그 영속화, 모니터링, 스케일, 장애 패턴 |
| (5/5) AWS 응용 *(지금 글)* | EKS + ECR + GitHub Actions 로 이미지 흐름 자동화 |

다음 단계로 자연스럽게 이어지는 주제들도 적어둘게요.

- **IRSA (IAM Roles for Service Accounts)** — Airflow 의 ServiceAccount 에 IAM Role 을 매핑해서 S3/RDS/Secrets Manager 등 AWS 리소스를 키 없이 접근. (4/5) 의 remote logging 을 IAM 키 없이 풀어내는 가장 깔끔한 방식.
- **RDS Postgres + pgbouncer** — 메타데이터 DB 를 EKS 밖으로 빼고 connection pool 까지 같이.
- **ALB Ingress / 사내 SSO** — Airflow UI 노출과 인증.

일단 오늘은 여기까지.....   
시리즈 끝까지 따라와주셔서 고맙습니다. 다음엔 위 IRSA 패턴으로 다시 돌아올게요.

---

**← 이전 글:** [(4/5) Airflow on K8s 운영 — git-sync · 로그 영속화 · 모니터링 · 스케일](/coding/Airflow_K8s_운영_DAG_동기화_모니터링/)
