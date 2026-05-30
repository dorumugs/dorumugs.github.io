---
layout: single
title:  "AWS DMS + CDC 로 MSSQL 무중단 컷오버 — 풀로드 후 변경분 따라잡기"
categories: coding
tag: [mssql, aws, dms, cdc, migration, rds, zero-downtime, kayserdocs]
author_profile: false
toc: true
---

[지난 글](/coding/RDS_SQLServer_Native_Backup_Restore_실전/)에서 Native Backup/Restore 로 풀카피 → 컷오버 순서를 정리했어요. 그런데 운영 DB 라 **단 몇 분의 다운타임도 허용 안 되는** 케이스가 있죠. 이번 글은 그런 상황에서 쓰는 **AWS DMS + CDC** 구성법이에요. 원본을 운영 중 그대로 두고 풀로드 → 변경분 실시간 따라잡기 → 컷오버 순서로 다운타임을 분 단위 이하로 줄이는 흐름을 정리합니다.

> 💡 이 글에서 다루는 것
> - 원본 MSSQL 에서 CDC 활성화 (DB 레벨 + 테이블 레벨)
> - DMS Replication Instance / Endpoints / Task 세팅
> - Full Load + CDC 동시 모드 vs 분리 모드 차이
> - 스키마/인덱스/IDENTITY 는 DMS 가 안 가져온다는 점과 대응
> - CloudWatch 로 latency 추적해서 안전하게 컷오버하는 절차
> - 마이그레이션 끝난 뒤 CDC/DMS 정리(cleanup)


<br>

<br>



## 1. 큰 그림 — 무엇이 어디서 도는가

```text
[내부망 MSSQL]
   │  (1) CDC 로 변경분 캡처 (트랜잭션 로그 기반)
   │
   ▼
[DMS Replication Instance (VPC)]
   │  (2) Full Load + 캡처된 변경분을 타깃에 적용
   │
   ▼
[AWS RDS for SQL Server]
```

DMS 는 **Replication Instance** 라는 EC2 같은 워커를 하나 띄워두고, 거기서 소스의 **트랜잭션 로그** 를 읽어와 타깃에 흘려줘요. 원본은 CDC 만 켜두면 되고, DMS 가 직접 원본을 폴링하면서 변경을 잡아갑니다.

> 💡 풀로드 + CDC 를 **같은 task** 에서 켜면 풀로드 진행 중에도 변경을 캐시해두고, 풀로드 끝나는 순간부터 따라잡기를 시작해요. "다운타임 거의 0" 의 핵심.


<br>

<br>



## 2. 원본 MSSQL 준비 — CDC 활성화

DMS 가 MSSQL 의 변경분을 잡아가려면 **원본에서 CDC 가 켜져 있어야** 해요. SQL Server 의 CDC 는 트랜잭션 로그를 읽어서 변경을 별도 시스템 테이블에 기록하는 기능이에요.

### 2-1. DB 레벨 CDC 켜기

```sql
USE [MyDB];
EXEC sys.sp_cdc_enable_db;

-- 확인
SELECT name, is_cdc_enabled 
FROM sys.databases 
WHERE name = 'MyDB';
```

`is_cdc_enabled = 1` 이면 OK.

### 2-2. 테이블 레벨 CDC 켜기

CDC 는 **테이블 단위로 한 번 더 켜줘야** 해요. 마이그레이션 대상 테이블 전부에 대해 돌립니다.

```sql
USE [MyDB];

EXEC sys.sp_cdc_enable_table
  @source_schema = N'dbo',
  @source_name   = N'Orders',
  @role_name     = NULL,           -- 별도 role 안 쓰면 NULL
  @supports_net_changes = 1;
```

테이블 많으면 동적 SQL 로 한 번에 돌려요.

```sql
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql = @sql + 
  N'EXEC sys.sp_cdc_enable_table @source_schema=N''' + s.name 
  + N''', @source_name=N''' + t.name 
  + N''', @role_name=NULL, @supports_net_changes=1;' + CHAR(13)
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
WHERE s.name = 'dbo';

EXEC sp_executesql @sql;
```

### 2-3. SQL Server Agent 가 떠 있어야 함

CDC 캡처 잡(`cdc.MyDB_capture`) 은 SQL Server Agent 가 돌려요. Agent 가 죽어있으면 변경이 안 잡혀요.

```sql
EXEC msdb.dbo.sp_help_job @job_name = N'cdc.MyDB_capture';
```

`current_execution_status` 가 `1` (idle 아닌 실행 중) 또는 정상 스케줄로 도는지 확인.

> ⚠️ RDS for SQL Server **소스** 였다면 CDC 활성화 절차가 조금 달라요(`rds_cdc_enable_db`). 이 글은 **온프레미스/EC2 소스** 기준.


<br>

<br>



## 3. AWS 측 사전 세팅 — Replication Instance / Endpoints

### 3-1. Replication Subnet Group + Instance

```shell
aws dms create-replication-subnet-group \
  --replication-subnet-group-identifier dms-subnet-grp \
  --replication-subnet-group-description "DMS subnet group" \
  --subnet-ids subnet-aaaa subnet-bbbb

aws dms create-replication-instance \
  --replication-instance-identifier dms-mssql-mig \
  --replication-instance-class dms.r5.large \
  --allocated-storage 100 \
  --vpc-security-group-ids sg-xxxxxxxx \
  --replication-subnet-group-identifier dms-subnet-grp \
  --no-publicly-accessible \
  --multi-az
```

인스턴스 사이즈는 풀로드 데이터량/병렬도/CDC 부하 따라 골라요. 보통 **dms.r5.large** 부터 시작해서 latency 보면서 키워요.

### 3-2. Source / Target Endpoints

**Source (내부망 MSSQL)** — S2S VPN 으로 접근.

```shell
aws dms create-endpoint \
  --endpoint-identifier mssql-source \
  --endpoint-type source \
  --engine-name sqlserver \
  --server-name 10.x.x.x \
  --port 1433 \
  --database-name MyDB \
  --username dms_user \
  --password '<DMS_USER_PASSWORD>'
```

**Target (AWS RDS for SQL Server)**

```shell
aws dms create-endpoint \
  --endpoint-identifier rds-target \
  --endpoint-type target \
  --engine-name sqlserver \
  --server-name my-mssql-rds.xxxx.ap-northeast-2.rds.amazonaws.com \
  --port 1433 \
  --database-name MyDB \
  --username admin \
  --password '<RDS_ADMIN_PASSWORD>'
```

### 3-3. Connection 테스트

엔드포인트를 만들면 반드시 한 번 테스트.

```shell
aws dms test-connection \
  --replication-instance-arn arn:aws:dms:ap-northeast-2:123456789012:rep:DMS-MSSQL-MIG \
  --endpoint-arn arn:aws:dms:ap-northeast-2:123456789012:endpoint:MSSQL-SOURCE
```

✅ 결과가 `successful` 이어야 다음 단계로 갈 수 있어요. 실패 사유는 콘솔의 `Last failure message` 에 떨어져요.

### 3-4. 소스 계정 권한

`dms_user` 에는 최소 권한만. 흔히 깨지는 권한 세트는 다음 세 가지에요.

```sql
USE master;
GRANT VIEW SERVER STATE TO dms_user;

USE MyDB;
EXEC sp_addrolemember 'db_owner', 'dms_user';   
-- 또는 SELECT + VIEW DATABASE STATE + db_datareader 조합
```

DMS 가 CDC 함수(`fn_cdc_get_all_changes_*`) 를 호출하기 때문에 단순 `db_datareader` 만으론 부족해요. `db_owner` 가 가장 깔끔하지만, 보안상 줄이고 싶다면 `EXECUTE` 권한을 CDC 함수에 별도 부여하는 방식이 있어요.


<br>

<br>



## 4. 스키마는 따로 — DMS 가 안 챙기는 것들

🚨 **여기서 가장 많이 헷갈려요.** DMS 는 **데이터 위주** 로 옮기는 서비스라, 다음을 직접 챙겨야 해요.

| 항목 | DMS 가 옮겨주나? |
|---|---|
| 테이블 구조 (CREATE TABLE) | △ (기본 타입 변환만, 추천 X) |
| 인덱스 (clustered/nonclustered) | ❌ |
| 외래키 제약 | ❌ (풀로드 중 자동 disable 권장) |
| 트리거 / 저장 프로시저 / 함수 | ❌ |
| 시퀀스 / IDENTITY 시드 | ❌ |
| 사용자/로그인/권한 | ❌ |

**일반적인 대응 패턴**

1. 원본에서 `SqlPackage /Action:Extract` 로 **DACPAC** 만들기 (스키마만, 데이터 X)
2. 타깃 RDS 에 `SqlPackage /Action:Publish` 로 스키마 먼저 적용
3. 그 다음 DMS 로 데이터만 채우기
4. DMS 끝난 후 외래키 / 트리거 / 시퀀스 시드 보정

또는 SSMS 의 "Generate Scripts" 로 **스키마 + 보조 객체** 만 추출해도 OK.

> 💡 **풀로드 중에는 타깃의 외래키와 인덱스를 잠시 끄거나 비활성화** 하는 게 빨라요. Task 설정의 `Target metadata` 에서 `BatchApplyEnabled` + 인덱스/제약 처리를 조정할 수 있어요.


<br>

<br>



## 5. Migration Task 만들기

이제 본 task. **Full Load + CDC** 모드로 만들고, table mapping 으로 옮길 테이블을 지정해요.

### 5-1. Table mappings (`table-mappings.json`)

```json
{
  "rules": [
    {
      "rule-type": "selection",
      "rule-id": "1",
      "rule-name": "select-dbo",
      "object-locator": {
        "schema-name": "dbo",
        "table-name": "%"
      },
      "rule-action": "include"
    }
  ]
}
```

`dbo` 스키마의 모든 테이블을 선택. 일부만 옮길 거면 `table-name` 을 구체적으로.

### 5-2. Task settings (`task-settings.json`)

```json
{
  "TargetMetadata": {
    "BatchApplyEnabled": true,
    "ParallelLoadThreads": 8,
    "ParallelLoadBufferSize": 500
  },
  "FullLoadSettings": {
    "TargetTablePrepMode": "TRUNCATE_BEFORE_LOAD",
    "MaxFullLoadSubTasks": 8,
    "CommitRate": 10000
  },
  "Logging": {
    "EnableLogging": true,
    "LogComponents": [
      { "Id": "SOURCE_CAPTURE", "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "SOURCE_UNLOAD",  "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "TARGET_APPLY",   "Severity": "LOGGER_SEVERITY_DEFAULT" },
      { "Id": "TARGET_LOAD",    "Severity": "LOGGER_SEVERITY_DEFAULT" }
    ]
  },
  "ChangeProcessingTuning": {
    "BatchApplyPreserveTransaction": true,
    "BatchApplyTimeoutMin": 1,
    "BatchApplyTimeoutMax": 30,
    "MinTransactionSize": 1000
  }
}
```

| 항목 | 의미 |
|---|---|
| `BatchApplyEnabled` | 변경을 배치로 묶어 타깃에 적용. CDC 단계에서 처리량 ↑ |
| `ParallelLoadThreads` | 테이블당 병렬 스레드 |
| `MaxFullLoadSubTasks` | 동시에 풀로드 돌릴 테이블 수 |
| `TargetTablePrepMode = TRUNCATE_BEFORE_LOAD` | 풀로드 전 타깃 테이블 비우기. 단, 외래키 있으면 실패할 수 있어 사전 작업 필요 |
| `CommitRate` | 풀로드 커밋 단위. 트랜잭션 로그 부담 조절 |

### 5-3. Task 생성

```shell
aws dms create-replication-task \
  --replication-task-identifier mssql-fullload-cdc \
  --source-endpoint-arn arn:aws:dms:...:MSSQL-SOURCE \
  --target-endpoint-arn arn:aws:dms:...:RDS-TARGET \
  --replication-instance-arn arn:aws:dms:...:DMS-MSSQL-MIG \
  --migration-type full-load-and-cdc \
  --table-mappings file://table-mappings.json \
  --replication-task-settings file://task-settings.json

# 시작
aws dms start-replication-task \
  --replication-task-arn arn:aws:dms:...:task:MSSQL-FULLLOAD-CDC \
  --start-replication-task-type start-replication
```

> 💡 `full-load-and-cdc` 가 핵심. `full-load` 만이면 풀로드 끝난 시점부터의 변경이 누락돼요.


<br>

<br>



## 6. 진행 모니터링 — 무엇을 봐야 하는가

### 6-1. Task 통계

콘솔에서 Task 의 **Table statistics** 탭이 가장 직관적이에요. 테이블별로

- `Full load rows` / `Total`
- `Inserts / Updates / Deletes` (CDC 단계)
- `Load state` (`Table completed`, `Before load`, `Full load`, `Table error`)

### 6-2. CloudWatch 핵심 지표

| 지표 | 의미 | 컷오버 기준 |
|---|---|---|
| `CDCLatencySource` | 소스 트랜잭션 로그 → DMS 까지 지연 (초) | 1초 이내 안정 |
| `CDCLatencyTarget` | DMS → 타깃 반영까지 지연 (초) | 1~3초 이내 안정 |
| `CDCIncomingChanges` | 캡처 중인 변경 수 | 누적 안 쌓이고 평탄 |
| `CDCChangesMemorySource` | 메모리에 쌓인 변경 | 안정적으로 낮게 |
| `CDCChangesDiskSource` | 디스크로 떨어진 변경 | 0 근처 유지 |

✅ 컷오버 직전 체크 — `CDCLatencyTarget` 이 **연속 5~10분간 1~3초 이내** 유지되면 따라잡기 안정 상태로 봐도 됩니다.

> ⚠️ `CDCLatencyTarget` 이 점점 커지면 DMS 인스턴스 사양 / 타깃 디스크 IOPS / 타깃 인덱스가 병목. 이 상태로 컷오버하면 안 됩니다.


<br>

<br>



## 7. 컷오버 절차 — 다운타임 분 단위 이하로

여기서부터가 본 게임이에요. 절차는 한 줄도 빼먹지 마세요.

### Step 1. 사전 정렬

- [x] CDC latency 가 안정적으로 작음 (5~10분 연속)
- [x] Table statistics 의 에러 0
- [x] 타깃에 인덱스/외래키 보조 객체 사전 생성 완료 (또는 사전 비활성)
- [x] 애플리케이션 측 새 접속 정보(RDS 엔드포인트) 준비 완료

### Step 2. 원본 쓰기 차단

가장 안전한 방법은 애플리케이션 측에서 **새 트랜잭션을 막는** 거예요. DB 단에서 막으려면

```sql
-- 원본 MSSQL
ALTER DATABASE [MyDB] SET READ_ONLY WITH ROLLBACK IMMEDIATE;
```

또는 앱 계정의 권한을 일시적으로 회수.

### Step 3. 마지막 변경분 따라잡기 대기

```text
CDCLatencyTarget → 0 으로 수렴
CDCIncomingChanges → 0 으로 수렴
```

10~30초 정도면 보통 마지막 트랜잭션까지 다 따라잡아요.

### Step 4. 검증

```sql
-- 양쪽 row count 비교
SELECT COUNT(*) FROM dbo.Orders;       -- 원본
SELECT COUNT(*) FROM dbo.Orders;       -- 타깃

-- 최신 데이터 비교
SELECT MAX(updated_at) FROM dbo.Orders;
```

핵심 테이블 몇 개만 빠르게 비교.

### Step 5. 외래키/시퀀스/IDENTITY 복구

```sql
-- 외래키 다시 enable
ALTER TABLE dbo.Orders WITH CHECK CHECK CONSTRAINT ALL;

-- IDENTITY 시드 재설정 (원본의 최댓값으로)
DBCC CHECKIDENT('dbo.Orders', RESEED, <원본 MAX(id)>);
```

### Step 6. 앱 connection string 전환

DNS 변경 or config 핫스왑. 새 쿼리는 RDS 로 흐름.

### Step 7. Task 중지

```shell
aws dms stop-replication-task --replication-task-arn arn:aws:dms:...:task:MSSQL-FULLLOAD-CDC
```

🎉 컷오버 완료. 다운타임은 보통 **1~3분** 안에 끝나요.


<br>

<br>



## 8. 사후 정리 — 비싸게 두지 말기

DMS Replication Instance 는 시간당 요금이 꾸준히 나와요. 컷오버 끝나면 빠르게 정리.

```shell
# 1. Task 삭제
aws dms delete-replication-task --replication-task-arn ...

# 2. Endpoints 삭제
aws dms delete-endpoint --endpoint-arn arn:aws:dms:...:endpoint:MSSQL-SOURCE
aws dms delete-endpoint --endpoint-arn arn:aws:dms:...:endpoint:RDS-TARGET

# 3. Replication Instance 삭제
aws dms delete-replication-instance --replication-instance-arn arn:aws:dms:...:rep:DMS-MSSQL-MIG
```

원본 MSSQL 의 CDC 도 더 이상 필요 없으면 끄세요. 트랜잭션 로그 보존 부담이 줄어요.

```sql
USE [MyDB];

-- 테이블별 CDC 끄기
EXEC sys.sp_cdc_disable_table 
  @source_schema = N'dbo', 
  @source_name = N'Orders', 
  @capture_instance = N'all';

-- DB 레벨 CDC 끄기
EXEC sys.sp_cdc_disable_db;
```


<br>

<br>



## 9. 자주 깨지는 함정

저도 한 번씩 다 밟아본 함정들이에요.

- 🚨 **CDC 안 켜고 task 시작** → `full-load-and-cdc` 가 `CDC source endpoint not configured` 에러로 죽음
- 🚨 **SQL Server Agent 가 꺼져있음** → CDC 캡처 잡이 안 돌아 변경이 누적되다 폭주
- 🚨 **타깃에 외래키가 켜진 채 풀로드** → 부모/자식 로드 순서 때문에 실패. 풀로드 전 disable
- 🚨 **IDENTITY 시드 미보정** → 컷오버 후 새 INSERT 가 키 충돌
- 🚨 **권한 부족** (`VIEW SERVER STATE`, CDC 함수 `EXECUTE`) → CDC 단계에서 silent 하게 latency 만 늘어남
- 🚨 **`CDCLatencyTarget` 이 점점 커짐** → DMS 인스턴스 / 타깃 IOPS / 인덱스 병목. 이 상태로 컷오버 절대 금지
- 🚨 **트랜잭션 로그 폭증** → CDC 가 안 따라잡고 있거나, 풀백업/로그백업 주기가 너무 길어서. log_reuse_wait_desc 확인


<br>

<br>



## 10. Native Backup/Restore vs DMS + CDC — 다시 정리

이전 글과 함께 의사결정 표 한 장으로 정리해드릴게요.

| 기준 | Native Backup/Restore | DMS + CDC |
|---|---|---|
| 원본 그대로 보존(인덱스/제약/IDENTITY) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (수동 보정 필요) |
| 다운타임 | 분~시간 단위 | 분 단위 이하 |
| 운영 부담/비용 | 낮음 (일회성) | 중간 (Task 가동 시간 비용) |
| 셋업 난이도 | 낮음 | 중간~높음 |
| 추천 시나리오 | "그대로 + 짧은 유지보수 창" | "절대 다운 불가 + 운영 중 이전" |

저희 사내 케이스 기준으로는

- **유지보수 창이 1~2시간 허락** → Native Backup/Restore 단독
- **유지보수 창이 거의 없음** → Native Backup/Restore 로 풀카피 후 **DMS 의 CDC-only 모드** 로 변경분만 흘려 컷오버

두 번째 패턴이 사실상 베스트에요. 풀로드 비용은 백업/복원으로 절약하고, CDC 만 DMS 로 처리해서 latency 안정화 후 컷오버.

일단 오늘은 여기까지.....   
다음 글에서는 이 두 번째 패턴 — **백업/복원 + CDC-only DMS** 의 결합을 단계별로 풀어볼게요. 
