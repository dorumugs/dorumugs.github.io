---
layout: single
title:  "(2/5) AWS RDS for SQL Server Native Backup/Restore 실전 — 옵션 그룹부터 컷오버까지"
date: 2026-05-30 09:09:00 +0900
description: "AWS RDS for SQL Server 의 Native Backup/Restore 를 옵션 그룹 세팅부터 S3 업로드, 복원, 컷오버 시 차등 백업까지 한 번에 따라가는 실전 가이드입니다."
categories: coding
tag: [mssql, aws, rds, backup, restore, s3, migration, native-backup, kayserdocs]
author_profile: false
toc: true
---

{% include series-mssql-rds.html current="2" %}

[지난 글](/coding/내부망_MSSQL_AWS_RDS_마이그레이션/)에서 내부망 MSSQL → AWS RDS for SQL Server 마이그레이션 방법 6가지를 비교하고, **Native Backup/Restore** 를 1순위로 추천드렸어요. 이번 글은 그 실전편이에요. RDS 옵션 그룹 세팅부터 시작해서, 풀백업 → S3 업로드 → 복원, 마지막으로 컷오버 시점의 **차등 백업(differential)** 으로 다운타임을 짧게 끊는 데까지 한 번에 따라가봅니다.

> 💡 이 글에서 다루는 것
> - RDS 옵션 그룹(`SQLSERVER_BACKUP_RESTORE`) + IAM Role + S3 버킷 사전 세팅
> - 원본에서 `BACKUP DATABASE` 로 풀백업 (STRIPE / COMPRESSION / CHECKSUM)
> - S3 업로드 시 주의점 (KMS, 멀티파트, 리전 매칭)
> - RDS 에서 `rds_restore_database` 로 복원하고 `rds_task_status` 로 진행 따라가기
> - 컷오버 시점의 차등 백업 + NORECOVERY 흐름
> - 복원 후 반드시 챙겨야 하는 로그인/사용자 매핑


<br>

<br>



## 1. 사전 그림 — 무엇이 어디에 있어야 하나

먼저 그림 한 번 그려두면 머리가 정리돼요.

```text
[내부망 MSSQL]
    │
    │ (1) BACKUP DATABASE → .bak (stripe N개, 압축)
    ▼
[내부망 파일 서버 / jump 서버]
    │
    │ (2) aws s3 cp  (S2S VPN 또는 인터넷)
    ▼
[S3 버킷 (AWS, 같은 리전)]
    │
    │ (3) EXEC msdb.dbo.rds_restore_database
    ▼
[AWS RDS for SQL Server]
```

세 가지를 사전에 세팅해두면 본 작업은 사실상 명령어 몇 줄이에요.

- **S3 버킷** — RDS 인스턴스와 **같은 리전**.
- **IAM Role** — RDS 가 S3 에 접근할 수 있도록.
- **RDS Option Group** — `SQLSERVER_BACKUP_RESTORE` 옵션을 붙이고, 위 IAM Role 을 지정.

이 셋 중 하나라도 어긋나면 `rds_restore_database` 가 곧장 권한 에러를 뱉어요. 순서대로 해두면 한 번에 통과합니다.


<br>

<br>



## 2. S3 버킷 + IAM Role 만들기

### 2-1. S3 버킷

리전만 잘 맞춰주세요. 그리고 버킷 정책은 처음엔 **굳이 손대지 않아도** 동작해요. IAM Role 권한으로 처리되니까요.

```shell
aws s3 mb s3://my-mssql-migration --region ap-northeast-2
```

> ⚠️ 인스턴스 리전과 버킷 리전이 다르면 RDS 가 접근 자체를 거부해요. 가장 흔히 깨지는 포인트 1위.

### 2-2. IAM Role

RDS 의 SQL Server 서비스가 S3 에 접근하도록 신뢰관계를 잡아줘야 해요. trust policy 와 권한 정책 둘 다 필요.

**Trust policy** (`rds-s3-trust.json`)

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "rds.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

**권한 정책** (`rds-s3-policy.json`)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::my-mssql-migration"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListMultipartUploadParts",
        "s3:AbortMultipartUpload"
      ],
      "Resource": "arn:aws:s3:::my-mssql-migration/*"
    }
  ]
}
```

```shell
aws iam create-role \
  --role-name rds-sqlserver-s3-role \
  --assume-role-policy-document file://rds-s3-trust.json

aws iam put-role-policy \
  --role-name rds-sqlserver-s3-role \
  --policy-name rds-sqlserver-s3-policy \
  --policy-document file://rds-s3-policy.json
```

> 💡 KMS 로 암호화된 버킷이면 위 정책에 `kms:Decrypt`, `kms:GenerateDataKey` 권한도 추가해주세요. 이거 빼먹고 한참 헤매는 분 많아요.


<br>

<br>



## 3. RDS Option Group 만들고 붙이기

이제 만든 IAM Role 을 RDS 인스턴스가 쓰도록 옵션 그룹을 만들고 attach 해요.

```shell
# 1. 옵션 그룹 생성 (SQL Server 버전/에디션 맞춰서)
aws rds create-option-group \
  --option-group-name mssql-backup-restore-og \
  --engine-name sqlserver-se \
  --major-engine-version 15.00 \
  --option-group-description "Native backup/restore for migration"

# 2. SQLSERVER_BACKUP_RESTORE 옵션 추가 + IAM Role 연결
aws rds add-option-to-option-group \
  --option-group-name mssql-backup-restore-og \
  --options "OptionName=SQLSERVER_BACKUP_RESTORE,OptionSettings=[{Name=IAM_ROLE_ARN,Value=arn:aws:iam::123456789012:role/rds-sqlserver-s3-role}]" \
  --apply-immediately

# 3. 인스턴스에 옵션 그룹 적용
aws rds modify-db-instance \
  --db-instance-identifier my-mssql-rds \
  --option-group-name mssql-backup-restore-og \
  --apply-immediately
```

옵션 그룹은 **동적(dynamic)** 옵션이라 인스턴스 재부팅 없이도 붙어요. 단, modify-db-instance 가 완전히 끝날 때까지(상태 `available`) 기다린 다음에 다음 단계로 가세요.

✅ 검증: SSMS / sqlcmd 로 인스턴스에 붙어서 아래 한 줄 실행해 보세요.

```sql
EXEC msdb.dbo.rds_show_configuration;
```

옵션이 잘 붙었으면 `S3 ARN access for backup/restore` 같은 라인이 보여요.


<br>

<br>



## 4. 원본에서 풀백업 만들기 (STRIPE + COMPRESSION)

이제 원본 MSSQL 에서 백업을 떠요. 한 파일로 통째로 떨어뜨리지 말고 **STRIPE 으로 4~8 분할** 하는 걸 추천드려요. 백업/복원 둘 다 병렬화돼서 훨씬 빠릅니다.

```sql
BACKUP DATABASE [MyDB]
TO DISK = N'D:\backup\MyDB_full_01.bak',
   DISK = N'D:\backup\MyDB_full_02.bak',
   DISK = N'D:\backup\MyDB_full_03.bak',
   DISK = N'D:\backup\MyDB_full_04.bak'
WITH 
  COMPRESSION,
  CHECKSUM,
  MAXTRANSFERSIZE = 4194304,  -- 4MB
  BUFFERCOUNT = 50,
  STATS = 5;
```

옵션 의미 한 줄씩.

| 옵션 | 의미 |
|---|---|
| `STRIPE (DISK 여러 개)` | 병렬 IO. 보통 4~8 분할이 가성비 좋아요. |
| `COMPRESSION` | .bak 크기 1/4~1/8 로 축소. 전송 시간 단축. |
| `CHECKSUM` | 백업 시점에 페이지 체크섬 검증 + .bak 자체 체크섬 기록. 손상 사전 감지. |
| `MAXTRANSFERSIZE = 4MB` | IO 단위 크게 잡아서 처리량 ↑ |
| `BUFFERCOUNT = 50` | 백업 IO 버퍼 수. 메모리 여유 있으면 늘려도 OK. |
| `STATS = 5` | 진행률 5% 단위로 출력. 모니터링 편함. |

> ⚠️ `BUFFERCOUNT` 너무 크게 잡으면 백업 세션이 메모리를 많이 먹어요. 운영 DB 라면 50 정도가 무난.

백업 끝나면 파일 크기와 무결성을 한 번 더 확인하세요.

```sql
RESTORE VERIFYONLY 
FROM DISK = N'D:\backup\MyDB_full_01.bak',
     DISK = N'D:\backup\MyDB_full_02.bak',
     DISK = N'D:\backup\MyDB_full_03.bak',
     DISK = N'D:\backup\MyDB_full_04.bak'
WITH CHECKSUM;
```

🎉 `The backup set on file 1 is valid.` 가 떨어지면 OK.


<br>

<br>



## 5. S3 로 업로드

내부망 → S3 업로드는 S2S VPN 으로 가도 되고, 인터넷 게이트웨이로 가도 돼요. **VPC endpoint(S3 Gateway Endpoint)** 가 있으면 RDS 가 복원할 때도 더 안정적이고 빨라요.

```shell
aws s3 cp D:\backup\ s3://my-mssql-migration/mssql/ \
  --recursive \
  --include "MyDB_full_*.bak" \
  --expected-size 50000000000  # 대용량이면 명시
```

큰 파일은 CLI 가 자동으로 멀티파트 업로드로 끊어 올려요. 멀티파트 기본 chunk 가 작으면 업로드 객체 수가 너무 많아져서 느려질 수 있는데, 이때는 chunk 를 키워주세요.

```shell
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_concurrent_requests 16
```

업로드 끝나면 객체 리스트 한 번 확인하세요.

```shell
aws s3 ls s3://my-mssql-migration/mssql/
```

> 💡 업로드 중간에 끊겨도 멀티파트는 **이어 올라가요**. `aws s3 cp` 를 다시 실행하면 이미 업로드된 파일은 skip 됩니다 (`--no-progress` 떼고 보면 보임).


<br>

<br>



## 6. RDS 에서 복원 — `rds_restore_database`

이제 본 게임. RDS 의 `msdb.dbo.rds_restore_database` 저장 프로시저를 호출해요. STRIPE 백업은 콤마로 ARN 을 이어 붙여요.

```sql
EXEC msdb.dbo.rds_restore_database
  @restore_db_name = 'MyDB',
  @s3_arn_to_restore_from = 
    'arn:aws:s3:::my-mssql-migration/mssql/MyDB_full_01.bak,
     arn:aws:s3:::my-mssql-migration/mssql/MyDB_full_02.bak,
     arn:aws:s3:::my-mssql-migration/mssql/MyDB_full_03.bak,
     arn:aws:s3:::my-mssql-migration/mssql/MyDB_full_04.bak',
  @with_norecovery = 1;  -- 차등 백업을 뒤이어 적용할 거라면 1
```

`@with_norecovery = 1` 로 두면 DB 가 **RESTORING** 상태로 남아서, 이어서 차등 백업을 추가 복원할 수 있어요. 마지막 복원에만 `0` 으로 두면 DB 가 ONLINE 으로 올라옵니다.

복원은 비동기로 큐에 들어가요. 진행 상황은 따로 조회.

```sql
EXEC msdb.dbo.rds_task_status @db_name = 'MyDB';
```

진행 단계는 `lifecycle` 컬럼으로 봐요.

| lifecycle | 의미 |
|---|---|
| `CREATED` | 큐에 등록됨 |
| `IN_PROGRESS` | 복원 중 |
| `SUCCESS` | 성공 |
| `ERROR` | 실패 (에러 메시지는 `task_info` 컬럼) |
| `CANCEL_REQUESTED` / `CANCELLED` | 취소 요청/완료 |

✅ `SUCCESS` 가 뜨면 풀백업 복원이 끝나요.


<br>

<br>



## 7. 컷오버 — 차등 백업으로 다운타임 줄이기

풀백업이 큰 DB 라면, **풀백업을 먼저 옮겨놓고** + **컷오버 직전에 차등 백업만 다시 옮기는** 방식으로 다운타임을 분 단위로 줄일 수 있어요.

### 7-1. 원본에서 차등 백업

```sql
BACKUP DATABASE [MyDB]
TO DISK = N'D:\backup\MyDB_diff_01.bak',
   DISK = N'D:\backup\MyDB_diff_02.bak'
WITH 
  DIFFERENTIAL,
  COMPRESSION,
  CHECKSUM,
  STATS = 5;
```

차등 백업은 풀백업 이후 변경된 페이지만 떠요. 보통 풀의 1~10% 크기로 끝납니다.

### 7-2. S3 업로드 후 RDS 에 차등 복원

```sql
EXEC msdb.dbo.rds_restore_database
  @restore_db_name = 'MyDB',
  @s3_arn_to_restore_from = 
    'arn:aws:s3:::my-mssql-migration/mssql/MyDB_diff_01.bak,
     arn:aws:s3:::my-mssql-migration/mssql/MyDB_diff_02.bak',
  @type = 'DIFFERENTIAL',
  @with_norecovery = 0;  -- 마지막 복원이므로 ONLINE 으로 올림
```

`@type = 'DIFFERENTIAL'` 가 핵심. `@with_norecovery = 0` 으로 끝내면 DB 가 `ONLINE` 으로 올라와요.

> 💡 RDS for SQL Server 의 차등 복원은 비교적 최근에 추가된 기능이에요. 인스턴스의 **엔진 버전** 이 차등 복원을 지원하는지 사전 확인하세요. 안 되면 풀백업만 반복하거나 컷오버 시간을 길게 잡아야 해요.


<br>

<br>



## 8. 복원 후 반드시 챙길 것 — 로그인 / 사용자 매핑

`.bak` 으로 복원하면 **DB 사용자(users)는 들어오지만**, 서버 레벨의 **로그인(logins)** 은 안 들어와요. 원본 서버에 있던 로그인을 RDS 마스터에 똑같이 만들고, DB 사용자와 다시 매핑해줘야 해요.

```sql
-- 1. RDS 에 로그인 생성 (원본에 있던 SQL 로그인 그대로)
CREATE LOGIN [app_user] WITH PASSWORD = '<APP_USER_PASSWORD>';

-- 2. 복원된 DB 의 user 를 새 로그인에 다시 묶기 (orphaned user 정리)
USE [MyDB];
EXEC sp_change_users_login 'Update_One', 'app_user', 'app_user';

-- 또는 ALTER USER ... WITH LOGIN = ...
ALTER USER [app_user] WITH LOGIN = [app_user];
```

✅ 검증: `EXEC sp_change_users_login 'Report';` 로 고아 사용자가 남아있는지 한 번 더 확인하세요.

> ⚠️ Windows 인증 기반 로그인은 RDS 가 Active Directory 연동(`SQLSERVER_AUDIT` / `SQLSERVER_AD` 옵션)되어 있어야 매핑 가능해요. 단순 SQL 로그인이면 위 절차로 충분해요.


<br>

<br>



## 9. 데이터 일관성 검증 — 한 번 더

복원이 끝나면 마지막으로 데이터 검증 한 사이클 돌려요.

```sql
-- 무결성 검사
DBCC CHECKDB ([MyDB]) WITH NO_INFOMSGS;

-- 테이블별 row count 스냅샷 비교용
SELECT s.name + '.' + t.name AS table_name, p.rows
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id
WHERE p.index_id IN (0,1)
ORDER BY p.rows DESC;
```

같은 쿼리를 원본 / 대상 양쪽에서 돌려서 결과를 비교해요. 큰 테이블 몇 개만이라도 row count 와 `MAX(updated_at)`, `CHECKSUM_AGG()` 정도는 맞춰보고 컷오버를 결정하세요.


<br>

<br>



## 10. 자주 깨지는 포인트 — 체크리스트

이번 작업하면서 한 번씩 다 밟아봤던 함정들이에요. 미리 알면 안 밟습니다.

- [x] S3 버킷 리전과 RDS 인스턴스 리전 **일치** 했는가
- [x] IAM Role 의 trust principal 이 `rds.amazonaws.com` 인가
- [x] KMS 암호화 버킷이면 IAM Role 에 `kms:Decrypt` 권한 있는가
- [x] 옵션 그룹의 `IAM_ROLE_ARN` 값이 위 Role 의 ARN 과 정확히 일치하는가
- [x] STRIPE 백업의 모든 파일이 S3 에 다 올라갔는가 (하나라도 빠지면 복원 실패)
- [x] `rds_task_status` 의 `lifecycle` 이 `SUCCESS` 인지 확인했는가
- [x] 차등 복원 전에 풀 복원이 `NORECOVERY` 로 끝났는가
- [x] 복원 후 로그인 재생성 + 고아 사용자 매핑까지 했는가
- [x] `DBCC CHECKDB` 클린인가
- [x] 원본/대상 row count 스냅샷 비교 완료했는가


<br>

<br>



## 11. 정리

Native Backup/Restore 는 단계가 많아 보이지만, 한 번 셋업해두면 명령어 몇 줄로 끝나는 깔끔한 흐름이에요. 핵심만 다시 짚으면

- 옵션 그룹 + IAM Role + S3 버킷 → **한 번만** 세팅
- 풀백업은 **STRIPE + COMPRESSION + CHECKSUM** 으로
- 컷오버는 **NORECOVERY 풀백업 → 차등 백업** 패턴으로 다운타임 단축
- 복원 후 **로그인 재생성 + DBCC + row count 비교** 까지 해야 끝난 거예요

일단 오늘은 여기까지.....   
다음 글에서는 같은 마이그레이션을 **AWS DMS + CDC** 로 풀어서, "다운타임 거의 0" 컷오버를 어떻게 구성하는지 정리해볼게요.

---

**← 이전 글:** [(1/5) 방법 비교 — 6가지 중에서 고르기](/coding/내부망_MSSQL_AWS_RDS_마이그레이션/) ｜ **다음 글 →:** [(3/5) DMS + CDC 무중단 컷오버 — 풀로드 후 변경분 따라잡기](/coding/MSSQL_AWS_DMS_CDC_무중단_컷오버/)

