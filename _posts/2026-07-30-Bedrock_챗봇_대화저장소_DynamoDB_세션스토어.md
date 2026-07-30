---
layout: single
title:  "(2/4) 대화 저장소 설계 — DynamoDB 세션 스토어와 메시지 스키마"
date: 2026-07-30 07:10:00 +0900
categories: coding
tag: [Bedrock, AWS, DynamoDB, 챗봇, 세션스토어, 메시지스키마, 툴콜, S3, Lambda, 스트리밍, Claude]
author_profile: false
toc: true
header:
  image: /assets/images/2026-07-30-bedrock-chat-session-store/header.svg
  teaser: /assets/images/2026-07-30-bedrock-chat-session-store/header.svg
description: "Bedrock 챗봇 맥락 유지의 가장 아래층인 세션 스토어를 DynamoDB 로 짭니다. 응답에서 텍스트만 뽑아 저장하면 다음 턴이 왜 400 으로 거절당하는지, 도구 호출 짝을 어떻게 지키는지, 400KB 항목 한계를 넘는 도구 결과를 S3 로 어떻게 밀어내는지, 스트리밍이 중간에 끊겼을 때 무엇을 저장해야 하는지까지 실제 코드로 정리했어요."
series: bedrock-chatbot-context
series_order: 2
---

{% include series-bedrock-chatbot-context.html current="2" %}

## Summary

[1편](/coding/Bedrock_챗봇_맥락유지_전체설계/)에서 맥락 유지를 4층으로 나눴어요. 이번 글은 그 **가장 아래층인 세션 스토어**예요. 여기서 스키마를 잘못 잡으면 위층이 전부 흔들리니까, 제일 공들여야 하는 부분입니다.

그리고 이 층에는 **정말 흔하고 정말 아픈 함정**이 하나 있어요. 응답에서 텍스트만 뽑아서 저장하는 건데, 처음엔 잘 돌아가다가 도구(툴)를 붙이는 순간 다음 턴 요청이 통째로 거절당해요. 그 이유부터 짚고 시작할게요.

> 💡 **이 글에서 다루는 것**
> - 왜 프로세스 메모리로는 안 되는가
> - 저장 단위는 "문자열"이 아니라 "블록 배열"
> - 텍스트만 뽑아 저장하면 다음 턴이 깨지는 이유
> - DynamoDB 스키마 — PK/SK, seq, TTL, 사용자별 목록
> - 도구 호출 짝(`tool_use` ↔ `tool_result`) 지키기
> - 400KB 항목 한계와 S3 밀어내기
> - 스트리밍이 끊겼을 때 무엇을 저장하나
> - 재조립 함수 전체 코드

<br>

<br>



## 1. 왜 외부 저장소가 필요한가

로컬에서 실험할 때는 이렇게 써도 잘 돌아가요.

```python
# 로컬 실험용 — 운영에서는 반드시 깨집니다
SESSIONS = {}

def load_messages(session_id):
    return SESSIONS.get(session_id, [])
```

이게 운영에서 깨지는 이유는 세 가지예요.

- **인스턴스가 여러 대예요.** ALB 뒤에 서버가 두 대만 돼도, 사용자의 다음 요청이 다른 서버로 가면 대화가 통째로 없어요. 스티키 세션으로 붙여둘 수도 있지만, 그 서버가 배포로 재시작하면 또 날아갑니다.
- **Lambda 는 그냥 안 돼요.** 실행 환경이 재사용될 때도 있어서 **간헐적으로 되는 것처럼 보이는 게 더 위험**해요. "가끔 기억을 못 한다"는 버그로 리포트가 들어오기 시작합니다.
- **대화는 자산이에요.** 나중에 품질 평가, 프롬프트 개선, 고객 문의 대응에 다 필요해요. 메모리에만 있으면 전부 사라집니다.

그래서 대화는 처음부터 **외부에 append-only 로 쌓는다**고 정하고 시작하는 게 맞아요. 저는 DynamoDB 를 기본으로 씁니다. 이유는 단순해요 — 세션 스토어가 필요한 접근 패턴이 딱 두 개인데, 둘 다 DynamoDB 가 가장 잘 하는 모양이에요.

```text
패턴 A: "이 세션의 메시지를 순서대로 전부"       → PK 하나로 정렬 조회
패턴 B: "이 사용자의 세션 목록을 최근 순으로"     → GSI 하나로 조회
```

<br>

<br>



## 2. 저장 단위는 문자열이 아니라 블록 배열

여기가 이 글의 핵심이에요. **한 턴을 문자열 하나로 저장하지 마세요.**

Messages API 에서 한 턴의 `content` 는 사실 **블록들의 배열**이에요. 단순한 대화에서는 텍스트 블록 하나뿐이라서 문자열처럼 보이지만, 실제로는 이렇게 여러 종류가 섞여 나옵니다.

| 블록 종류 | 언제 나오나 | 저장 필수? |
|---|---|---|
| `text` | 항상 | ✅ |
| `thinking` | 사고가<br>켜져 있을 때 | ✅ 필수 |
| `tool_use` | 모델이 도구를<br>부를 때 | ✅ 필수 |
| `tool_result` | 우리가 도구 결과를<br>돌려줄 때 | ✅ 필수 |
| `compaction` | 서버가 히스토리를<br>압축했을 때 | ✅ 필수 |

그래서 이렇게 저장하면 안 돼요.

```python
# ❌ 이렇게 하면 도구를 붙이는 순간 무너집니다
text = response.content[0].text
save(session_id, "assistant", text)
```

이렇게 해야 합니다.

```python
# ✅ content 배열을 통째로, 받은 모양 그대로
save(session_id, "assistant", response.content)
```

### 2-1. 텍스트만 저장하면 왜 깨지나

세 가지가 동시에 무너져요.

**첫째, 도구 호출이 미아가 돼요.** 모델이 도구를 부르면 응답에 `tool_use` 블록이 담기고, 우리는 다음 요청에서 그에 대응하는 `tool_result` 를 넣어줘야 해요. 그런데 `tool_use` 를 저장하지 않았으면 히스토리에 **결과만 덩그러니 남습니다.** API 는 짝이 안 맞는 `tool_result` 를 그냥 거절해요.

```text
저장된 히스토리:
  어시스턴트: "날씨를 확인해볼게요"          ← tool_use 가 사라졌음
  사용자:     [tool_result: "맑음, 24도"]    ← 짝이 없는 결과

다음 요청 결과: 400 invalid_request_error
```

**둘째, 사고(thinking) 블록이 서명 검증에 걸려요.** 사고 블록은 같은 모델로 대화를 이어갈 때 **받은 그대로 되돌려줘야** 합니다. 여기서 특히 헷갈리는 게 있는데, 최신 모델들은 기본 설정에서 **사고 내용을 빈 문자열로 돌려줘요.** 그래서 "내용이 없으니 안 중요한 블록이네" 하고 버리기 쉬운데, **블록 자체는 반드시 보존해야** 합니다. 지우면 순서·서명 검증에서 400 이 날 수 있어요.

**셋째, 압축 블록이 사라져요.** 4편에서 다룰 압축 기능은 서버가 오래된 히스토리를 요약해서 `compaction` 블록으로 돌려주는 방식이에요. 이 블록을 다시 넣어주지 않으면 **압축 상태가 통째로 사라지고**, 다음 요청에서 원본 히스토리를 다시 요구하게 됩니다.

> ⚠️ 정리하면 원칙은 하나예요. **모델이 준 `content` 는 우리가 해석하지 말고 보관용으로는 그대로 둔다.** 화면에 뿌릴 텍스트는 거기서 별도로 뽑아 쓰면 됩니다. 저장본과 표시본을 분리하세요.

```python
# 저장본은 원문, 표시본은 따로 추출
blocks = response.content                                  # 저장용
shown  = "".join(b.text for b in blocks if b.type == "text")  # 화면용
```

<br>

<br>



## 3. DynamoDB 스키마

접근 패턴 두 개에 맞춰 단일 테이블로 잡습니다.

### 3-1. 테이블 설계

| 항목 | 값 | 설명 |
|---|---|---|
| PK | `SESSION#<sid>` | 세션 단위 묶음 |
| SK | `MSG#000042` | 0 채운 순번<br>(사전순 = 시간순) |
| SK | `META` | 세션 메타 1건 |

메시지 항목의 속성은 이렇게 둡니다.

```json
{
  "PK":         "SESSION#01JQ8Z...",
  "SK":         "MSG#000042",
  "seq":        42,
  "role":       "assistant",
  "blocks":     "[{\"type\":\"text\",\"text\":\"...\"}]",
  "blocks_ref": null,
  "created_at": "2026-07-30T07:12:03.412Z",
  "model":      "anthropic.claude-opus-5",
  "usage":      { "input": 12043, "output": 512, "cache_read": 11800 },
  "expire_at":  1793404323
}
```

몇 가지 선택에 이유가 있어요.

- **`SK` 를 0 채운 문자열로** — DynamoDB 는 문자열 정렬키를 사전순으로 정렬해요. `MSG#42` 로 두면 `MSG#100` 보다 뒤로 가버리니까, `MSG#000042` 처럼 폭을 고정해야 순번이 곧 시간순이 됩니다.
- **`blocks` 를 JSON 문자열로** — 맵 타입으로 넣어도 되지만, 문자열로 두면 SDK 타입 변환(특히 숫자가 `Decimal` 로 바뀌는 문제)을 안 겪어요. 우리는 이 값을 **읽어서 그대로 API 에 넘기기만** 하니까 문자열이 오히려 안전합니다.
- **`blocks_ref` 는 S3 탈출구** — 항목이 400KB 한계에 닿을 때 쓰는 필드예요. 5절에서 다뤄요.
- **`usage` 를 같이 기록** — 나중에 "어느 대화에서 캐시가 안 먹었나"를 추적할 수 있어요. 이거 없으면 3편의 캐싱 튜닝을 눈 감고 하게 됩니다.
- **`expire_at` 은 TTL** — 유닉스 초 단위예요. 보존 기간 정책을 코드가 아니라 DynamoDB 가 집행해줍니다.

### 3-2. 사용자별 세션 목록 (GSI)

패턴 B 를 위해 GSI 하나를 더 둡니다. `META` 항목에만 GSI 키를 넣으면 **인덱스에 메시지가 안 올라가서** 저장 비용이 훨씬 줄어요. 이게 스파스 인덱스(sparse index) 라고 하는 기법이에요.

| GSI 키 | 값 |
|---|---|
| GSI1PK | `USER#<uid>` |
| GSI1SK | `2026-07-30T07:12:03Z` |

<br>

<br>



## 4. 쓰기 — 순번과 원자성

append-only 라서 쓰기는 단순한데, 두 가지만 조심하면 돼요.

### 4-1. 순번은 메타 항목에서 원자적으로 발급

여러 요청이 동시에 들어와도 순번이 겹치면 안 되니까, `META` 항목의 카운터를 원자적으로 올려서 받아옵니다.

```python
import boto3, json, time
from decimal import Decimal

ddb   = boto3.resource("dynamodb", region_name="us-west-2")
table = ddb.Table("chat_sessions")

TTL_DAYS = 90

def next_seq(session_id: str) -> int:
    """META 항목의 카운터를 원자적으로 +1 하고 그 값을 받아옵니다."""
    res = table.update_item(
        Key={"PK": f"SESSION#{session_id}", "SK": "META"},
        UpdateExpression="SET last_seq = if_not_exists(last_seq, :zero) + :one",
        ExpressionAttributeValues={":zero": Decimal(0), ":one": Decimal(1)},
        ReturnValues="UPDATED_NEW",
    )
    return int(res["Attributes"]["last_seq"])


def append_message(session_id: str, role: str, blocks: list, **meta) -> int:
    """한 턴을 블록 배열 원문 그대로 저장합니다."""
    seq  = next_seq(session_id)
    body = json.dumps(to_plain(blocks), ensure_ascii=False)

    item = {
        "PK":         f"SESSION#{session_id}",
        "SK":         f"MSG#{seq:06d}",
        "seq":        seq,
        "role":       role,
        "blocks":     body,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expire_at":  int(time.time()) + TTL_DAYS * 86400,
        **meta,
    }

    # 같은 SK 가 이미 있으면 실패 — 재시도가 중복 저장되는 걸 막아줍니다
    table.put_item(
        Item=item,
        ConditionExpression="attribute_not_exists(SK)",
    )
    return seq
```

`ConditionExpression` 이 있는 이유는 **재시도 때문**이에요. 네트워크 타임아웃으로 클라이언트가 재시도하면 같은 턴이 두 번 저장될 수 있는데, 그러면 히스토리에 어시스턴트 턴이 두 번 들어가서 대화가 이상해져요. 조건부 쓰기로 막습니다.

### 4-2. SDK 객체를 평범한 dict 로 바꾸기

SDK 가 돌려주는 `response.content` 는 Pydantic 모델 객체예요. 그대로 `json.dumps` 하면 실패하니까 한 번 변환해줍니다.

```python
def to_plain(blocks) -> list:
    """SDK 블록 객체를 저장 가능한 dict 리스트로 변환."""
    out = []
    for b in blocks:
        if isinstance(b, dict):
            out.append(b)
        elif hasattr(b, "model_dump"):
            # None 필드는 빼둡니다 — 다시 넣을 때 잡음이 됩니다
            out.append(b.model_dump(exclude_none=True))
        else:
            raise TypeError(f"저장할 수 없는 블록: {type(b)}")
    return out
```

> ⚠️ `exclude_none=True` 를 빼면 `{"cache_control": null}` 같은 필드가 같이 저장돼요. 그 자체로 에러는 안 나지만, 3편에서 다룰 캐싱에서 **프롬프트 바이트가 미묘하게 달라져 캐시가 안 먹는** 원인이 될 수 있어요. 저장할 때부터 깔끔하게 두는 게 낫습니다.

<br>

<br>



## 5. 400KB 한계와 S3 밀어내기

DynamoDB 항목 하나는 **최대 400KB** 예요. 평범한 대화 턴은 몇 KB 수준이라 신경 쓸 일이 없지만, **도구 결과가 이 한계를 자주 넘습니다.** DB 조회 결과, 문서 전문, API 응답 JSON 같은 것들이요.

그래서 임계값을 두고 넘으면 S3 로 밀어냅니다.

```python
import boto3

s3       = boto3.client("s3")
BUCKET   = "my-chat-blocks"
MAX_INLINE = 300 * 1024   # 400KB 한계에 여유를 둡니다

def append_message(session_id: str, role: str, blocks: list, **meta) -> int:
    seq  = next_seq(session_id)
    body = json.dumps(to_plain(blocks), ensure_ascii=False)
    encoded = body.encode("utf-8")

    item = {
        "PK": f"SESSION#{session_id}",
        "SK": f"MSG#{seq:06d}",
        "seq": seq,
        "role": role,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expire_at": int(time.time()) + TTL_DAYS * 86400,
        **meta,
    }

    if len(encoded) <= MAX_INLINE:
        item["blocks"] = body
    else:
        key = f"sessions/{session_id}/{seq:06d}.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=encoded,
                      ContentType="application/json")
        item["blocks_ref"] = key      # DynamoDB 에는 포인터만

    table.put_item(Item=item, ConditionExpression="attribute_not_exists(SK)")
    return seq
```

`MAX_INLINE` 을 400KB 가 아니라 300KB 로 잡은 이유는, **항목 크기는 blocks 만이 아니라 속성 이름과 다른 값까지 다 합친 것**이기 때문이에요. 한계에 딱 붙여두면 어느 날 갑자기 실패합니다.

> 💡 사실 더 좋은 방법이 하나 더 있어요. **애초에 거대한 도구 결과를 히스토리에 넣지 않는 것**입니다. 도구가 100KB JSON 을 돌려준다면, 그걸 그대로 모델에 먹이는 대신 "S3 에 저장했고 요약은 이렇다" 형태로 줄여서 넣는 게 비용과 품질 양쪽에 다 좋아요. 저장 문제를 저장으로 풀기 전에 설계로 풀 수 있는지 먼저 보세요.

<br>

<br>



## 6. 읽기 — 재조립

이제 읽는 쪽이에요. 순서대로 꺼내서, S3 포인터를 채우고, API 가 받을 모양으로 만듭니다.

```python
from boto3.dynamodb.conditions import Key

def load_messages(session_id: str, limit: int | None = None) -> list:
    """세션의 메시지를 순서대로 꺼내 messages 배열로 재조립합니다."""
    items, start_key = [], None

    while True:
        kwargs = {
            "KeyConditionExpression": (
                Key("PK").eq(f"SESSION#{session_id}")
                & Key("SK").begins_with("MSG#")
            ),
            "ScanIndexForward": True,          # 오래된 것부터
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key

        page = table.query(**kwargs)
        items.extend(page["Items"])
        start_key = page.get("LastEvaluatedKey")
        if not start_key:
            break

    messages = []
    for it in items:
        if it.get("blocks_ref"):
            obj = s3.get_object(Bucket=BUCKET, Key=it["blocks_ref"])
            blocks = json.loads(obj["Body"].read())
        else:
            blocks = json.loads(it["blocks"])

        messages.append({"role": it["role"], "content": blocks})

    if limit:
        messages = trim_to_valid_window(messages, limit)

    return messages
```

두 가지 짚을 게 있어요.

- **페이지네이션을 반드시 돌리세요.** DynamoDB 의 `query` 는 응답 1MB 에서 잘려요. 긴 대화에서는 그 이상이 나오는데, `LastEvaluatedKey` 를 안 따라가면 **히스토리 앞부분이 조용히 사라져요.** 에러도 안 나고요. 실무에서 자주 보는 버그예요.
- **`begins_with("MSG#")`** 로 `META` 항목을 걸러냅니다. 이거 없으면 메타 항목이 메시지처럼 섞여 들어와요.

<br>

<br>



## 7. 도구 호출 짝 지키기

`limit` 으로 최근 N턴만 남기려면 **아무 데서나 자를 수 없어요.** `tool_use` 와 `tool_result` 사이에서 자르면 짝이 깨져서 요청이 거절당합니다.

```text
❌ 잘못 자른 히스토리
   어시스턴트: [tool_use id=abc "날씨조회"]      ← 여기서 자름
   ────────── 자른 지점 ──────────
   사용자:     [tool_result id=abc "맑음"]        ← 짝이 없는 결과 → 400
```

규칙은 간단해요. **자르는 지점은 항상 "온전한 사용자 턴"이어야 합니다.** 도구 결과로만 이루어진 사용자 턴은 시작점이 될 수 없어요.

```python
def trim_to_valid_window(messages: list, keep: int) -> list:
    """최근 keep 개 정도만 남기되, 도구 호출 짝이 깨지지 않는 지점에서 자릅니다."""
    if len(messages) <= keep:
        return messages

    start = len(messages) - keep

    # 안전한 시작점까지 뒤로 밀기: 도구 결과만 든 사용자 턴은 시작점이 못 됨
    while start < len(messages):
        m = messages[start]
        if m["role"] != "user":
            start += 1
            continue
        blocks = m["content"]
        if isinstance(blocks, list) and any(
            b.get("type") == "tool_result" for b in blocks if isinstance(b, dict)
        ):
            start += 1          # 이 턴은 앞 턴의 tool_use 에 딸린 결과
            continue
        break

    return messages[start:]
```

저장할 때 검증을 한 번 더 걸어두면 더 좋아요. 히스토리에 짝 안 맞는 블록이 들어가는 순간을 바로 잡을 수 있습니다.

```python
def assert_pairs_ok(messages: list) -> None:
    """tool_use 와 tool_result 의 id 짝이 맞는지 확인합니다."""
    pending = set()
    for m in messages:
        blocks = m["content"] if isinstance(m["content"], list) else []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                pending.add(b["id"])
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid not in pending:
                    raise ValueError(f"짝 없는 tool_result: {tid}")
                pending.discard(tid)
    if pending:
        raise ValueError(f"결과가 없는 tool_use: {pending}")
```

> ✅ 이 검증 함수를 요청 직전에 한 줄 넣어두면, 400 이 났을 때 **API 응답을 보고 추측하는 대신 우리 코드에서 정확한 원인을 알려줘요.** 개발 단계에서만 켜도 시간을 많이 아껴줍니다.

<br>

<br>



## 8. 스트리밍이 중간에 끊겼을 때

챗봇은 거의 항상 스트리밍으로 만드니까, 이 경우를 정해두어야 해요. 사용자가 브라우저를 닫거나 네트워크가 끊기면 응답이 절반만 온 상태가 됩니다. (토큰을 SSE·WebSocket 으로 흘려주는 쪽 구조는 [스트리밍 챗봇 아키텍처 글](/coding/AWS_Bedrock_스트리밍_챗봇_아키텍처/)에 정리해뒀어요. 여기서는 **저장 정책**만 봅니다.)

선택지는 셋이고, 저는 **B** 를 기본으로 권해요.

| 방식 | 저장 내용 | 성격 |
|---|---|---|
| A. 버리기 | 아무것도<br>저장 안 함 | 사용자 질문만<br>남아 어색해짐 |
| B. 부분 저장 | 받은 만큼<br>+ 중단 표시 | 이어가기 자연스러움<br>(권장) |
| C. 전부 저장 | 끊긴 뒤에도<br>서버가 끝까지 받음 | 비용은 어차피<br>다 나감 |

A 를 고르면 히스토리에 사용자 질문 하나가 답 없이 남아요. 모델은 그걸 보고 "아직 안 답했구나" 하고 다시 답하려 하는데, 사용자 화면에는 이미 절반이 떠 있었으니 대화가 어긋납니다.

C 도 나름 합리적이에요. **어차피 토큰 비용은 발생한 상태**니까, 서버에서 스트림을 끝까지 받아 온전한 턴으로 저장하면 히스토리가 가장 깨끗해요. 백그라운드 태스크를 붙일 수 있는 구조라면 이게 제일 좋습니다.

B 는 그 중간이에요.

```python
def stream_answer(session_id: str, user_blocks: list):
    messages = load_messages(session_id) + [
        {"role": "user", "content": user_blocks}
    ]
    append_message(session_id, "user", user_blocks)

    collected, interrupted = [], False
    try:
        with client.messages.stream(
            model="anthropic.claude-opus-5",
            max_tokens=16000,
            system=SYSTEM_BLOCKS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            collected = to_plain(final.content)
    except (GeneratorExit, ConnectionError):
        interrupted = True
        # 여기까지 받은 텍스트를 블록으로 만들어 둡니다
        collected = partial_blocks()
        raise
    finally:
        if collected:
            if interrupted:
                collected.append({
                    "type": "text",
                    "text": "\n\n(응답이 중간에 끊겼습니다)",
                })
            append_message(session_id, "assistant", collected)
```

> ⚠️ 부분 저장에서 한 가지만 조심하세요. **끝나지 않은 `tool_use` 블록은 저장하지 마세요.** 도구 인자가 절반만 스트리밍된 상태일 수 있고, 그러면 다음 턴에 짝 없는 도구 호출이 히스토리에 박혀서 그 세션이 계속 400 을 냅니다. 부분 저장할 때는 완성된 `text` 블록만 남기는 게 안전해요.

<br>

<br>



## 9. 정리 — 이 층의 체크리스트

세션 스토어를 짤 때 확인할 항목을 모았어요.

- [ ] 응답 `content` 를 **통째로** 저장하나요? (텍스트만 뽑고 있지 않나요)
- [ ] `thinking` 블록을 **내용이 비어 있어도** 보존하나요?
- [ ] 정렬키를 **0 채운 고정폭**으로 두었나요? (`MSG#000042`)
- [ ] 순번을 **원자적으로** 발급하나요? (동시 요청에서 겹치지 않게)
- [ ] 조건부 쓰기로 **재시도 중복**을 막았나요?
- [ ] `query` **페이지네이션**을 돌리나요? (1MB 에서 잘려요)
- [ ] 히스토리를 자를 때 **도구 짝**을 지키나요?
- [ ] 400KB 넘는 도구 결과를 **S3 로 밀어내나요?**
- [ ] `usage` 를 함께 기록하나요? (3편의 캐싱 튜닝에 필요해요)
- [ ] TTL 로 **보존 기간**을 집행하나요?

여기까지 하면 대화가 안 사라지고, 다음 턴 요청이 거절당하지 않아요. 그런데 아직 **비용 문제는 그대로**예요. 20턴이 되면 앞의 19턴을 매번 다시 계산하고 있으니까요.

일단 오늘은 여기까지.....  
다음 글에서는 그 재전송 비용을 걷어내는 **프롬프트 캐싱**을 파봅니다. 규칙은 딱 하나뿐인데, 그 하나를 몰라서 캐시가 조용히 안 먹는 경우가 정말 많아요.

---

**이전 글 ←:** [(1/4) AWS Bedrock 챗봇, 대화 맥락은 어떻게 유지하나 — 전체 설계도](/coding/Bedrock_챗봇_맥락유지_전체설계/)

**다음 글 →:** [(3/4) 프롬프트 캐싱 — 같은 히스토리를 매번 보내면서 돈 아끼기](/coding/Bedrock_챗봇_프롬프트캐싱_비용절감/)
