---
layout: single
title:  "(4/5) Native Backup/Restore 뒤에 변경분 계속 쌓기 — NORECOVERY 체인과 CDC-only 결합"
date: 2026-05-30 10:26:00 +0900
description: "Native Backup/Restore 로 풀카피한 뒤 원본의 변경분을 계속 따라잡는 방법. NORECOVERY 체인과 DMS CDC-only 결합 패턴을 RDS for SQL Server 제약과 함께 풀어봅니다."
categories: coding
header:
  image: /assets/images/2026-05-30-rds-restore-cdc-incremental/header.svg
  teaser: /assets/images/2026-05-30-rds-restore-cdc-incremental/header.svg
tag: [mssql, aws, rds, backup, restore, dms, cdc, incremental, kayserdocs]
author_profile: false
toc: true
---

{% include series-mssql-rds.html current="4" %}

[Native Backup/Restore 실전편](/coding/RDS_SQLServer_Native_Backup_Restore_실전/)을 따라 풀백업 복원까지 끝내고 나면, 다음 질문이 바로 나와요. **"이제 운영 중인 원본에서 새로 들어오는 데이터는 어떻게 따라잡지?"** 풀백업이 끝난 시점 이후 원본은 계속 INSERT/UPDATE 가 들어오는 중이니까요. 이번 글은 그 "복원 위에 변경분 계속 쌓기" 문제를 RDS for SQL Server 의 제약을 짚어가며 정리해볼게요.

> 💡 이 글에서 다루는 것
> - 왜 "복원이 끝난 DB 위에 다시 백업을 못 올리는가" — RECOVERY/NORECOVERY 차이
> - 옵션 A: 차등 + 트랜잭션 로그 백업 체인 (NORECOVERY 유지)
> - 옵션 B: 한 번 ONLINE 된 후에는 **DMS CDC-only** 로 흘리기
> - 옵션 C: 앱 레벨 dual-write / 쿼리 기반 증분 ETL
> - 베스트 패턴: 풀백업/복원 + CDC-only 결합으로 비용/속도 모두 잡기


<br>

<br>



## 1. 먼저 짚어야 하는 제약 — RECOVERY vs NORECOVERY

가장 많이 헷갈리는 부분부터 정리할게요. SQL Server 의 복원은 두 가지 상태로 끝낼 수 있어요.

| 상태 | 의미 | 다음에 백업 더 적용 가능? |
|---|---|---|
| `RECOVERY` (기본) | DB 가 ONLINE 으로 올라옴. 사용자/쿼리 접근 OK | ❌ 추가 백업 못 적용 |
| `NORECOVERY` | DB 가 `RESTORING` 상태로 남음. 쿼리 접근 X | ✅ 차등/로그 백업 이어서 적용 가능 |

🚨 **여기가 함정입니다.** 풀백업 복원을 `WITH RECOVERY` (기본값) 로 끝내면, **그 위에 차등/로그 백업을 더 못 올려요.** RDS for SQL Server 의 `rds_restore_database` 도 마찬가지로, 한 번 ONLINE 으로 올라온 DB 에 또 백업 적용을 시도하면 LSN 체인 에러가 떨어집니다.

> 💡 그래서 이전 글에서 풀백업 복원에 `@with_norecovery = 1` 을 강조했던 거예요. 차등 백업까지 다 적용하고, **마지막 복원에만** `@with_norecovery = 0` 으로 ONLINE 으로 올리는 패턴.


<br>

<br>



## 2. 옵션 A — 차등 + 로그 백업 체인 (NORECOVERY 유지)

가장 정공법. 컷오버 직전까지 타깃 DB 를 `RESTORING` 상태로 두고, 원본에서 떠오는 차등/로그 백업을 계속 적용해요.

### 2-1. 흐름

```text
[원본]                                  [타깃 RDS]
  1) FULL backup --------- 업로드 ----> rds_restore_database (NORECOVERY)
  2) LOG backup #1 ------- 업로드 ----> rds_restore_log     (NORECOVERY)
  3) LOG backup #2 ------- 업로드 ----> rds_restore_log     (NORECOVERY)
  ...
  N) 컷오버 시점 LOG ----- 업로드 ----> rds_restore_log     (RECOVERY=ONLINE)
```

### 2-2. 원본 — 로그 백업 만들기

차등은 이전 글에서 다뤘으니, 여기선 **트랜잭션 로그 백업** 위주로.

```sql
-- 원본 MSSQL 의 복구 모델이 FULL 또는 BULK_LOGGED 여야 로그 백업이 의미 있음
ALTER DATABASE [MyDB] SET RECOVERY FULL;

-- 10분마다 로그 백업 (예시)
BACKUP LOG [MyDB]
TO DISK = N'D:\backup\log\MyDB_log_20260530_1000.trn'
WITH COMPRESSION, CHECKSUM, STATS = 10;
```

> ⚠️ 복구 모델이 `SIMPLE` 이면 로그 백업 자체가 불가능해요. 운영 DB 가 SIMPLE 이면 옵션 A 는 사용 불가 → 옵션 B(CDC) 로.

### 2-3. 타깃 RDS — 로그 백업 적용

```sql
EXEC msdb.dbo.rds_restore_log
  @restore_db_name = 'MyDB',
  @s3_arn_to_restore_from = 'arn:aws:s3:::my-mssql-migration/log/MyDB_log_20260530_1000.trn',
  @with_norecovery = 1;
```

진행은 `rds_task_status` 로 확인. 로그 백업은 보통 풀백업보다 훨씬 작아서 1~5분 단위로 빠르게 흘릴 수 있어요.

### 2-4. 컷오버 — 마지막 로그만 RECOVERY 로

```sql
-- 원본: 쓰기 차단
ALTER DATABASE [MyDB] SET READ_ONLY WITH ROLLBACK IMMEDIATE;

-- 원본: 마지막 tail-log 백업
BACKUP LOG [MyDB]
TO DISK = N'D:\backup\log\MyDB_tail.trn'
WITH NORECOVERY, COMPRESSION, CHECKSUM;

-- 타깃: 마지막 로그를 RECOVERY 로 적용 → DB ONLINE
EXEC msdb.dbo.rds_restore_log
  @restore_db_name = 'MyDB',
  @s3_arn_to_restore_from = 'arn:aws:s3:::my-mssql-migration/log/MyDB_tail.trn',
  @with_norecovery = 0;
```

🎉 이 시점에 타깃 DB 가 ONLINE 으로 올라오면서 컷오버가 완료돼요.

### 2-5. 옵션 A 의 장단점

**장점**

- **순정 MSSQL 방식**. 추가 매니지드 서비스 비용 없음
- 데이터 손실 0 (트랜잭션 로그 단위로 정확히 따라잡힘)
- "그대로" 보존이 가장 잘 됨

**단점**

- 타깃 DB 가 컷오버 전까지 **계속 RESTORING 상태** → 쿼리/검증 불가
- 로그 백업 주기 / 업로드 / 복원이 모두 **수동 파이프라인**. 자동화 필요
- 원본 복구 모델이 SIMPLE 이면 사용 불가


<br>

<br>



## 3. 옵션 B — DMS CDC-only 모드 (이미 ONLINE 된 DB 에 흘리기)

만약 이미 `WITH RECOVERY` 로 ONLINE 시켜버렸거나, 검증을 위해 타깃에서 쿼리를 돌려보고 싶다면 옵션 A 는 쓸 수 없어요. 이때 쓰는 게 **DMS 의 CDC-only 모드**.

### 3-1. 핵심 아이디어

- 풀로드는 이미 끝났다 (Native Backup/Restore 로)
- DMS 한테 **"풀로드 건너뛰고 변경분만 잡아 흘려"** 라고 시킴
- 시작 시점(timestamp 또는 LSN)을 명시해서, 풀백업 시점 이후의 변경을 잡아냄

### 3-2. Migration Type

```shell
aws dms create-replication-task \
  --replication-task-identifier mssql-cdc-only \
  --source-endpoint-arn arn:aws:dms:...:MSSQL-SOURCE \
  --target-endpoint-arn arn:aws:dms:...:RDS-TARGET \
  --replication-instance-arn arn:aws:dms:...:DMS-MSSQL-MIG \
  --migration-type cdc \
  --cdc-start-time 2026-05-30T10:00:00Z \
  --table-mappings file://table-mappings.json \
  --replication-task-settings file://task-settings.json
```

핵심 옵션 두 개.

| 옵션 | 의미 |
|---|---|
| `--migration-type cdc` | 풀로드 생략, 변경분만 |
| `--cdc-start-time` | 변경 캡처 시작 시점. **풀백업 시작 시각** 또는 그 직전으로 |

### 3-3. 시작 시점을 어떻게 정하나

가장 안전한 방법은 **풀백업을 시작하기 직전의 시각** 을 기록해두는 거예요.

```sql
-- 원본에서 풀백업 시작 전에
SELECT GETUTCDATE() AS backup_start_utc;
-- => 2026-05-30 09:55:00

-- 이 값을 cdc-start-time 으로 넣음. backup 진행 중 변경은 중복 적용되겠지만,
-- DMS 의 idempotent 동작과 PK 충돌 처리로 보통 안전하게 흡수됨.
```

> 💡 LSN 기준으로 더 정확하게 잡고 싶다면 `--cdc-start-position` 에 `LSN:xxxxx:xxxxxxxx` 형태로 줄 수 있어요. 풀백업 직전 LSN 을 `sys.fn_dblog` 로 따와서 넣는 패턴.

### 3-4. 옵션 B 의 장단점

**장점**

- 풀로드를 DMS 로 안 돌려도 됨 → **비용 절감 + 시간 단축**
- 타깃 DB 를 ONLINE 으로 두고도 변경 따라잡기 가능 → 검증/대시보드 미리 붙여볼 수 있음
- CDC latency 를 보면서 컷오버 타이밍을 잡을 수 있음 (이전 글의 7단계 컷오버 그대로 적용)

**단점**

- 원본에 **CDC 활성화** 가 필요 (이전 DMS 글 참고)
- 풀백업 시점 ~ CDC 시작 시점 사이의 변경이 **중복 적용** 될 수 있음. PK/UPSERT 동작 사전 점검
- DMS Replication Instance 비용은 발생


<br>

<br>



## 4. 옵션 C — 앱 레벨 dual-write / 쿼리 기반 증분 ETL

DB 단 메커니즘(백업 체인, CDC) 을 못 쓰는 환경에선 마지막 카드로.

### 4-1. Dual-write

애플리케이션이 INSERT/UPDATE 를 칠 때 **원본 + 타깃 양쪽에 동시 기록**.

- 장점: DB 메커니즘 의존 없음. 앱이 통제 가능
- 단점: 코드 수정 필요. 실패/재시도 시 양쪽 일관성 보장이 까다로움. 두 DB 가 잠시라도 다르면 디버깅 지옥

### 4-2. 쿼리 기반 증분 ETL

타임스탬프나 버전 컬럼이 있는 테이블에 한해, 주기적으로

```sql
SELECT * FROM dbo.Orders 
WHERE updated_at > @last_sync_ts;
```

식으로 끌어와 타깃에 MERGE.

- 장점: 구현 간단. 추가 서비스 불필요
- 단점: **DELETE 를 못 잡음**(soft delete 면 OK). `updated_at` 이 없는 테이블엔 적용 불가. PK 충돌 처리 직접 구현

🚨 옵션 C 는 "다른 게 다 안 될 때" 의 카드예요. 실무에선 잔불 정도로만 쓰고, 본진은 A 또는 B 로.


<br>

<br>



## 5. 베스트 패턴 — 풀백업/복원 + CDC-only 결합

이게 [이전 DMS 글](/coding/MSSQL_AWS_DMS_CDC_무중단_컷오버/) 마지막에 예고한 패턴이에요. 두 방법의 장점만 가져와요.

```text
시점         원본                             타깃 RDS
─────────────────────────────────────────────────────────────
T0           ┌─ CDC 활성화                     
T1           │  GETUTCDATE() = T1 기록
T1+ε         │  FULL BACKUP 시작                                       
T2           │  FULL BACKUP 완료 → S3 업로드                            
T3           │                                rds_restore_database
T4           │                                (WITH RECOVERY=ONLINE) 🎉
T5           │  DMS CDC-only Task 시작
             │   --cdc-start-time = T1
T5+          │  CDC latency 안정화 추적
컷오버       │  앱 쓰기 차단 → latency 0 대기
             └─ 앱 connection string 전환
```

### 5-1. 왜 이게 좋은가

| 비교축 | 옵션 A (백업 체인 단독) | 옵션 B (DMS Full+CDC) | **A+B 결합** |
|---|---|---|---|
| 초기 풀로드 속도 | 빠름 (백업/복원) | 느림 (DMS 풀로드) | **빠름** |
| 초기 비용 | 낮음 | 높음 (Replication 시간 길어짐) | **낮음** |
| 컷오버 다운타임 | 짧음(분 단위) | 매우 짧음(초 단위) | **매우 짧음** |
| 타깃 DB 사전 검증 | 불가(RESTORING) | 가능 | **가능** |
| 운영 부담 | 로그 백업 파이프라인 | DMS 모니터링 | **DMS 모니터링** |

핵심은 "**풀카피의 무거운 일은 백업/복원 으로, 변경분 추적의 가벼운 일은 DMS 로**" 분담시키는 것. DMS 가 풀로드까지 하면 시간/비용이 다 무거워지는데, 그 부분을 백업/복원으로 떼어내요.

### 5-2. 실전 순서 (정리)

- [x] **T0**: 원본에 CDC 활성화 (DB + 모든 대상 테이블)
- [x] **T1**: 풀백업 시작 시각 기록 (`SELECT GETUTCDATE()`)
- [x] **T1+ε**: 원본 풀백업 → S3 업로드
- [x] **T3**: RDS 에서 `rds_restore_database` (`@with_norecovery = 0` → ONLINE)
- [x] **T4**: 타깃 검증 (DBCC CHECKDB, row count 스냅샷)
- [x] **T5**: DMS CDC-only task 시작 (`--migration-type cdc`, `--cdc-start-time T1`)
- [x] **T5+**: CloudWatch 의 `CDCLatencyTarget` 안정화 확인 (5~10분 연속 < 3초)
- [x] **컷오버**: 앱 쓰기 차단 → latency 0 수렴 대기 → 앱 connection string 전환
- [x] **사후**: DMS Task / Endpoints / Replication Instance 삭제, 원본 CDC 비활성화

> 💡 풀백업 시작 ~ CDC 시작 사이의 변경은 **재적용** 돼요. PK 충돌 발생 시 DMS 가 어떻게 처리하는지 사전에 확인하세요. Task 설정의 `ErrorBehavior` 에서 duplicate key 를 `IGNORE_RECORD` 로 두면 보통 안전하게 흡수됩니다.


<br>

<br>



## 6. 자주 깨지는 함정

이 패턴에서 한 번씩 다 밟아본 함정들이에요.

- 🚨 **풀백업 복원을 `WITH RECOVERY` 로 끝낸 뒤 차등 백업 더 올리려고 시도** → LSN 체인 에러. 옵션 A 를 끝까지 가려면 마지막까지 NORECOVERY 유지
- 🚨 **CDC 활성화 시점을 풀백업 시작 뒤로 미룸** → 두 시각 사이의 변경이 캡처 안 됨 → 영원히 누락
- 🚨 **`cdc-start-time` 을 풀백업 완료 시각으로 넣음** → 풀백업 진행 도중의 변경 누락
- 🚨 **로그 백업과 차등 백업을 섞었는데 LSN 체인이 어긋남** → 복원 거부. 차등을 적용했다면 그 뒤로는 로그 백업 체인만 쭉 가야 함
- 🚨 **타깃 DB 의 외래키/트리거가 활성화된 상태에서 CDC 적용 시작** → 적용 실패. 컷오버 직전까지 트리거 disable 권장
- 🚨 **원본 트랜잭션 로그가 폭증** → 로그 백업 주기가 너무 길거나 CDC 캡처 잡이 멈춤. `log_reuse_wait_desc` 확인


<br>

<br>



## 7. 정리 — 상황별 선택 가이드

| 상황 | 추천 옵션 |
|---|---|
| 운영 다운타임 1~2시간 OK, 단순함 우선 | **옵션 A 단독** (NORECOVERY 체인 + 마지막 RECOVERY) |
| 다운타임 거의 0, 타깃 사전 검증 필요 | **옵션 A+B 결합 (베스트)** |
| 이미 타깃 DB 가 ONLINE 으로 올라가 버림 | **옵션 B 단독** (CDC-only) |
| 원본이 SIMPLE 복구 모델이고 변경 못 함 | **옵션 B 또는 C** |
| 앱이 통제 가능하고 DB 메커니즘 못 씀 | **옵션 C** (dual-write) |

저희 사내 케이스에서 가장 자주 쓰는 건 **옵션 A+B 결합** 이에요. 풀카피는 백업/복원으로 한 방에 끝내고, 변경분은 DMS CDC-only 로 가볍게 따라잡는 구성. 비용/속도/다운타임 세 마리 토끼를 잡아요.

일단 오늘은 여기까지.....   
다음 글에서는 이 결합 패턴에서 **실제로 자주 만나는 데이터 정합성 문제** — 중복 키 처리, NULL 컬럼, IDENTITY 시드 어긋남, 외래키 적용 순서 — 를 케이스별로 풀어볼게요.

---

**← 이전 글:** [(3/5) DMS + CDC 무중단 컷오버 — 풀로드 후 변경분 따라잡기](/coding/MSSQL_AWS_DMS_CDC_무중단_컷오버/) ｜ **다음 글 →:** [(5/5) 정합성 트러블슈팅 — 케이스별 8가지 함정](/coding/MSSQL_마이그레이션_정합성_트러블슈팅/)

