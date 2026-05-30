---
layout: single
title:  "(1/5) 내부망 MSSQL → AWS RDS for SQL Server 옮기기 — 부하 안 주고 그대로 넣는 법"
date: 2026-05-30 08:30:00 +0900
description: "내부망 MSSQL 을 AWS RDS for SQL Server 로 옮기는 6가지 방법을 비교하고, 원본 부하 최소화 관점에서 Native Backup/Restore 를 1순위로 추천하는 글이에요."
categories: coding
tag: [mssql, aws, rds, dms, bcp, migration, s2s-vpn, database, kayserdocs]
author_profile: false
toc: true
---

{% include series-mssql-rds.html current="1" %}

내부망에 있던 MSSQL DB 한 덩어리를 **AWS RDS for SQL Server** 로 그대로 옮겨야 했어요. 다행히 양쪽은 **S2S VPN** 으로 이미 통신이 되는 상태였고, 조건은 두 가지였어요. 첫째 **원본을 그대로 옮길 것**(스키마/식별자/인덱스 보존), 둘째 **운영 중인 원본 DB에 부하를 주지 말 것**. 이 글은 그때 검토했던 방법들과 최종 선택, 그리고 건수별 소요 시간 추정을 정리한 글이에요.

> 💡 이 글에서 다루는 것
> - 내부망 → AWS RDS for SQL Server 마이그레이션 방법 6가지 비교
> - "부하 방지" 관점에서 각 방법이 원본에 주는 영향
> - 1만 건 / 100만 건 / 1억 건 기준 예상 소요 시간
> - 최종 추천: 어떤 상황엔 무엇을 골라야 하는가


<br>

<br>



## 1. 전제 — 우리 환경 정리

먼저 작업 시작 전에 짚어둔 전제예요.

| 항목 | 값 |
|---|---|
| 원본 | 내부망 MSSQL (온프레미스, SQL Server 2019 기준) |
| 대상 | AWS RDS for SQL Server (같은 메이저 버전 또는 상위) |
| 네트워크 | S2S VPN 으로 양방향 통신 가능 (대역폭은 보통 1Gbps 미만) |
| 목표 | 원본 그대로 복제 (one-shot 풀 카피) |
| 제약 | 원본 DB 부하 최소화, 서비스는 운영 중 |

여기서 가장 중요한 두 가지는 **"그대로"** 와 **"부하 없이"** 예요. 이 두 단어가 방법 선택의 거의 모든 기준이 돼요.


<br>

<br>



## 2. 옵션 정리 — 6가지 방법

먼저 후보를 다 펼쳐놓고 시작할게요. 표 한 번 보고 본문으로 들어가요.

| # | 방법 | 그대로 보존 | 원본 부하 | 네트워크 부담 | 작업 난이도 |
|---|---|---|---|---|---|
| 1 | Native Backup / Restore (.bak + S3) | ⭐⭐⭐⭐⭐ | ⭐ (낮음) | 한 번에 큰 파일 | 낮음 |
| 2 | AWS DMS (Full Load) | ⭐⭐⭐ | ⭐⭐ | 균등 분산 | 중간 |
| 3 | bcp (Bulk Copy) | ⭐⭐⭐⭐ | ⭐⭐ | 테이블 단위 | 중간 |
| 4 | SSIS (Integration Services) | ⭐⭐⭐⭐ | ⭐⭐⭐ | 균등 분산 | 중간~높음 |
| 5 | SqlPackage (BACPAC) | ⭐⭐⭐⭐ | ⭐⭐ | 한 번에 큰 파일 | 낮음 |
| 6 | Linked Server `INSERT … SELECT` | ⭐⭐ | ⭐⭐⭐⭐⭐ (높음) | 가장 비효율 | 가장 낮음 |

⚠️ **표의 별 개수 의미** — "보존" 은 많을수록 좋음(원본 그대로), "부하" 와 "네트워크 부담" 은 많을수록 나쁨(원본/네트워크에 부담). 별 1개가 좋은 것일 수도, 나쁜 것일 수도 있어요. 헷갈리지 않게 ⭐ 옆에 (낮음)/(높음) 으로도 표시했어요.


<br>

<br>



## 3. 방법 1 — Native Backup/Restore (.bak → S3 → RDS)

**MSSQL 의 정공법**. 원본에서 `BACKUP DATABASE` 로 `.bak` 파일을 만들고, S3 에 올린 다음, RDS 의 저장 프로시저 `msdb.dbo.rds_restore_database` 로 복원하는 방법이에요.

```sql
-- 원본 (내부망 MSSQL)
BACKUP DATABASE [MyDB]
TO DISK = N'D:\backup\MyDB_full.bak'
WITH COMPRESSION, CHECKSUM, STATS = 10;
```

```shell
# S3 업로드 (내부망 jump 서버에서 → S2S VPN 통해 S3 endpoint)
aws s3 cp D:\backup\MyDB_full.bak s3://my-mssql-migration/MyDB_full.bak
```

```sql
-- AWS RDS for SQL Server 에서 실행
EXEC msdb.dbo.rds_restore_database
  @restore_db_name = 'MyDB',
  @s3_arn_to_restore_from = 'arn:aws:s3:::my-mssql-migration/MyDB_full.bak';
```

진행상황은 `EXEC msdb.dbo.rds_task_status @db_name='MyDB';` 로 따라갈 수 있어요.

**장점**

- 스키마/인덱스/제약/식별자 시드/CLR 어셈블리 전부 **그대로** 보존돼요. "그대로 넣는다" 의 끝판왕.
- 원본 부하는 **풀백업 시점의 디스크 I/O 만**. row-by-row 쿼리가 없으니 운영 쿼리에 영향이 작아요.
- 큰 DB 도 압축 백업이면 보통 1/4~1/8 로 줄어들어 전송도 빨라요.

**단점**

- RDS 가 **Native Backup/Restore 옵션 그룹** 활성화 + S3 권한이 사전에 세팅되어 있어야 해요.
- AWS RDS for SQL Server 에서 **`differential` 복원은 일정 버전 이상부터** 지원돼요. 풀백업 + 차등은 미리 RDS 버전 확인.
- 한 파일이 큼 → 전송 중 끊기면 처음부터 다시 (멀티파트 업로드로 어느 정도 보완 가능).

> ✅ "그대로" 가 가장 잘 지켜지는 방법. one-shot 마이그레이션이라면 1순위 후보.


<br>

<br>



## 4. 방법 2 — AWS DMS (Database Migration Service)

AWS 가 만든 **마이그레이션 전용 매니지드 서비스**. Replication Instance 를 띄우고, 소스(내부망 MSSQL) / 타깃(RDS for SQL Server) 엔드포인트를 등록한 뒤, **Full Load** 또는 **Full Load + CDC** 로 돌려요.

```text
[내부망 MSSQL] --(S2S VPN)--> [DMS Replication Instance] --> [AWS RDS for SQL Server]
```

**장점**

- 테이블/스키마 단위로 **병렬 로드** 가능. 큰 DB 를 시간 안에 끝내고 싶을 때 좋아요.
- **CDC(Change Data Capture)** 를 켜면 풀 로드 후 변경분만 따라잡아 줘서 **다운타임 거의 0** 컷오버 가능.
- 진행률/에러/지표가 콘솔에 다 보임 → 운영하기 편함.

**단점**

- 보존 측면에서 **인덱스/식별자 시드/제약을 별도 챙겨야** 해요. DMS 는 데이터 위주로 옮기고, 보조 인덱스는 옵션으로 따로 만들거나 컷오버 후 생성하는 게 일반적이에요. "그대로" 라는 관점에선 백업/복원만 못해요.
- 풀로드 도중 원본에 **장기 SELECT** 가 걸려요. 트랜잭션 격리 수준/락 옵션을 잘못 두면 오히려 원본에 부담.
- DMS Replication Instance 비용이 따로 들어요.

> 💡 "운영 중인 DB 를 무중단에 가깝게 옮기고 싶다" 가 강한 요건이면 DMS + CDC 가 최선.


<br>

<br>



## 5. 방법 3 — bcp (Bulk Copy Program)

MSSQL 의 **고전 명령어**. 테이블 단위로 원본에서 파일로 추출(`bcp out`)하고, 대상에 일괄 적재(`bcp in` 또는 `BULK INSERT`)해요.

```shell
# 원본에서 추출 (네이티브 포맷이 가장 빠르고 안전)
bcp MyDB.dbo.Orders out Orders.dat ^
  -S 192.168.x.x -U sa -P <PASSWORD> -n -b 10000

# AWS RDS 로 적재
bcp MyDB.dbo.Orders in Orders.dat ^
  -S mydb.xxxx.ap-northeast-2.rds.amazonaws.com -U admin -P <PASSWORD> -n -b 10000 -h "TABLOCK"
```

옵션 의미는 짧게.

- `-n` : 네이티브(바이너리) 포맷. 파싱 비용이 적어서 빠르고 타입 오차도 없음.
- `-b 10000` : 배치 사이즈. 트랜잭션을 잘게 끊어서 로그 폭주를 막아요.
- `-h "TABLOCK"` : 대상 테이블에 락을 잡고 적재 → minimally logged operation 으로 빨라져요.

**장점**

- **테이블별로 골라서** 옮길 수 있음. "이 테이블만 빨리" 가 가능.
- 배치 사이즈 / 병렬 처리(여러 bcp 동시 실행) 로 **부하 조절** 이 비교적 쉬워요.
- AWS 서비스 의존성 없음. 사내 jump 서버에서 그냥 돌리면 끝.

**단점**

- 풀 DB 옮기려면 테이블/제약/인덱스 스크립트는 별도로 준비해야 함.
- 외래키/식별자 시드/순서 의존성을 직접 관리해야 함 (적재 순서, IDENTITY_INSERT 등).

> ⚠️ bcp 에서 부하 방지의 핵심은 **`-b` 배치 사이즈** 와 **동시 실행 수** 둘이에요. 무작정 `-b 0` 으로 한 트랜잭션에 다 밀어넣지 마세요. 트랜잭션 로그가 폭주합니다.


<br>

<br>



## 6. 방법 4 — SSIS / 방법 5 — SqlPackage(BACPAC)

두 개는 묶어서 짧게.

**SSIS (SQL Server Integration Services)**

- ETL GUI 가 익숙하다면 좋아요. 패키지 한 번 만들어두면 재실행도 편함.
- 내부적으로는 결국 bulk insert + 데이터 흐름이라 속도는 bcp 와 비슷한 급.
- 단점: 패키지를 어디서 돌릴지(SSIS 카탈로그 서버) 가 필요. 일회성 마이그레이션엔 좀 무거워요.

**SqlPackage (BACPAC / DACPAC)**

- `SqlPackage.exe /Action:Export ...` → `.bacpac` 파일 생성 → S3 업로드 → 대상에 Import.
- 스키마 + 데이터를 한 덩어리로 들고 다니기에 좋아요. 작은 DB 에 잘 맞음.
- 큰 DB(수십 GB 이상)는 내보내기/들여오기 둘 다 느려서 `.bak` 보다 불리해요.

이 둘은 **원본 부하** 관점에선 bcp 와 비슷하거나 살짝 더 무거워요(스키마 메타데이터 추출이 추가됨).


<br>

<br>



## 7. 방법 6 — Linked Server `INSERT ... SELECT` (비추)

가장 쉬워 보이지만, 실제론 **운영 환경에서 거의 안 쓰는 방법**.

```sql
-- 대상(RDS) 에서 원본을 Linked Server 로 등록 후
INSERT INTO MyDB.dbo.Orders
SELECT * FROM [SRC_LINK].[MyDB].[dbo].[Orders];
```

**왜 비추인가**

- 원본에서 **한 줄 한 줄 네트워크로 끌어옴**. S2S VPN 왕복 지연이 매 row 마다 발생해요.
- 원본 트랜잭션 로그/락 부담이 큼. 부하 방지가 요건인데 정반대로 가요.
- 도중에 끊기면 일관성 회복도 까다로움.

🚨 이 방법은 **테스트용 작은 테이블** 외엔 쓰지 마세요. 100만 건만 돼도 시간이 끝없이 늘어집니다.


<br>

<br>



## 8. 추천 — 우리 케이스에서의 베스트

요건이 "그대로" + "부하 최소" + "S2S VPN 있음" 이라면 결론은 명확해요.

**🎉 1순위 — Native Backup/Restore (.bak → S3 → RDS)**

이유.

- 원본에서 발생하는 부하는 **백업 작업 한 번뿐**. 그것도 디스크 I/O 라 운영 쿼리(메모리/CPU 위주)와 겹치는 영역이 작아요.
- 풀백업 시점에 원본을 **점적(point-in-time)** 으로 떠오므로 데이터 일관성 보장.
- 인덱스/제약/IDENTITY/통계까지 **그대로** 복원. 사용자 요건 그대로.

**🎉 2순위 — DMS Full Load + CDC (다운타임 거의 0 이 필요한 경우)**

- 컷오버 시간을 분 단위로 줄여야 한다면 DMS.
- 단, "그대로" 가 아니라 "내용은 같지만 인덱스 등은 새로 만들어도 됨" 이 허용돼야 가성비가 좋아요.

**🎉 3순위 — bcp (특정 테이블만 빠르게)**

- "이 테이블 하나만 옮기면 돼요" 같은 부분 마이그레이션에 최적.

> ✅ 사내 케이스에서 저는 1순위(Native Backup/Restore) 로 풀카피 → 컷오버 시점에 차등 백업 한 번 더 로 정리하는 패턴을 가장 선호해요.


<br>

<br>



## 9. 소요 시간 추정 — 1만 건 / 100만 건 / 1억 건

이 부분은 **환경마다 차이가 큰** 영역이에요. 그래도 기준선이 있어야 계획이 가능하니, **평균 행 크기 1KB, 인덱스 보통 수준, S2S VPN 실효 200~500Mbps** 가정으로 정리할게요.

### 9-1. 방법별 처리량(throughput) 감

| 방법 | 평균 처리량 (1KB 행 기준) | 비고 |
|---|---|---|
| Native Backup/Restore | **네트워크 대역폭에 수렴** | row 단위 처리량 개념이 아님 |
| AWS DMS (Full Load, 단일 task) | ~5,000 ~ 20,000 rows/sec | 병렬 task 로 N배 확장 |
| bcp (네이티브 + TABLOCK + 적절한 배치) | ~20,000 ~ 100,000 rows/sec | LAN 이면 더 빨라짐 |
| SSIS | ~10,000 ~ 50,000 rows/sec | 패키지 튜닝 의존 |
| BACPAC (Import) | ~3,000 ~ 15,000 rows/sec | 인덱스 재생성 비용 큼 |
| Linked Server | **~100 ~ 1,000 rows/sec** | WAN 왕복 지연으로 매우 느림 |

### 9-2. 건수별 예상 시간

⚠️ 아래 숫자는 **단일 task / 단일 connection** 기준 추정치예요. 실제로는 병렬화로 단축돼요.

| 방법 | 1만 건 | 100만 건 | 1억 건 |
|---|---|---|---|
| Native Backup/Restore | 의미 없음 (한 번에 처리) | 약 1~3분 (.bak 1~5GB) | 약 30분~2시간 (.bak 100~500GB, 압축 시) |
| AWS DMS (단일) | ~1~3초 | **~1~3분** | ~3~10시간 (병렬 시 1/N) |
| bcp (튜닝 후) | <1초 | **~10~50초** | ~30분~2시간 |
| SSIS | ~1초 | ~30초~2분 | ~1~3시간 |
| BACPAC Import | ~2~5초 | ~1~5분 | ~3~10시간 |
| Linked Server | ~10~60초 | **~30분~3시간** | ❌ 사실상 비현실적 |

**핵심 포인트**

- 1만 건 수준은 **어떤 방법이든 1분 안에 끝남**. 시간 차이가 거의 없음.
- 100만 건부터는 방법별로 **분~시간 단위** 차이가 벌어져요. bcp 와 .bak 가 압도적으로 빠름.
- 1억 건 이상은 단일 task 로 끌고 가지 말고, **테이블/파티션 단위 병렬화** 가 필수.

> 💡 정확한 수치는 결국 행 크기, 인덱스 개수, 트랜잭션 로그 설정, 네트워크 실효 대역폭에 좌우돼요. 본격 작업 전엔 **샘플 테이블 100만 건으로 한 번 실측** 해두는 걸 추천해요.


<br>

<br>



## 10. 부하 방지 체크리스트

방법을 골랐다면, 실행 단계에서 챙겨야 할 부하 방지 포인트들이에요.

- [x] 작업 시간대를 **운영 트래픽 적은 시간** 으로 잡기 (보통 새벽 2~5시)
- [x] 원본 쿼리는 가능하면 **`READ UNCOMMITTED`** 또는 **스냅샷 격리** 로 잡아서 락 충돌 회피
- [x] bcp/DMS 사용 시 **배치 사이즈** 를 작게 시작해서 점진적으로 키우기
- [x] 트랜잭션 로그 백업 주기 짧게 유지 (단순 복구 모델이면 더 좋음)
- [x] **Resource Governor** 로 마이그레이션 세션 CPU/IO 상한 걸기
- [x] 백업 파일은 **원본 디스크와 다른 볼륨** 에 떨어뜨리기 (IO 분리)
- [x] 작업 중 원본 DB **모니터링 대시보드** 띄워두기 (CPU, IO, lock waits)


<br>

<br>



## 11. 정리 — 어떤 상황엔 무엇

마지막으로 상황별 추천을 한 번 더 정리해드릴게요.

| 상황 | 추천 방법 |
|---|---|
| 운영 중지 시간을 1~2시간 줄 수 있다 + 그대로 옮기고 싶다 | **Native Backup/Restore** |
| 다운타임을 거의 0 으로 가야 한다 | **DMS Full Load + CDC** |
| 특정 큰 테이블 몇 개만 빠르게 옮기면 된다 | **bcp** |
| DB 가 작고(수 GB) 한 덩어리로 들고 다니고 싶다 | **BACPAC** |
| GUI ETL 익숙하고 재실행 자주 한다 | **SSIS** |
| Linked Server INSERT...SELECT | **하지 마세요** |

저희 사내 케이스처럼 **"그대로" + "부하 최소" + "S2S 가능"** 이면 **Native Backup/Restore** 가 거의 항상 1순위예요. DMS 는 컷오버 단축이 필요할 때 더해서 쓰는 식.

일단 오늘은 여기까지.....   
다음 글에서는 실제로 Native Backup/Restore 를 RDS 옵션 그룹 세팅부터 끝까지 따라가는 실전편을 정리해볼게요.

---

**다음 글 →** [(2/5) Native Backup/Restore 실전 — 옵션 그룹부터 컷오버까지](/coding/RDS_SQLServer_Native_Backup_Restore_실전/)

