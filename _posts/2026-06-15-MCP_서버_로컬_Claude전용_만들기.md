---
layout: single
title:  "(1/2) MCP 서버를 로컬·Claude 전용으로 만들기 — Python(FastMCP) stdio 완벽 가이드"
date: 2026-06-15 07:30:00 +0900
categories: coding
tag: [mcp, model-context-protocol, fastmcp, python, claude, claude-desktop, claude-code, stdio, local-server, tool]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-15-mcp-local-stdio/header.svg
  teaser: /assets/images/2026-06-15-mcp-local-stdio/header.svg
description: "외부에 포트 하나 열지 않고, 내 컴퓨터의 Claude(Desktop·Code)만 붙는 MCP 서버를 Python(FastMCP)으로 만드는 법을 처음부터 끝까지 정리합니다. stdio 트랜스포트, 도구·리소스·프롬프트, 등록·디버깅, 그리고 로컬 전용 방식의 장단점까지."
---

{% include series-mcp.html current="1" %}

## Summary

MCP(Model Context Protocol) 서버를 만들 때 가장 먼저 갈리는 갈림길이 **"이걸 네트워크에 띄울 거냐, 내 컴퓨터 안에서만 돌릴 거냐"** 예요. 이 글은 후자 — **포트를 하나도 열지 않고, 내 머신의 Claude 만 붙는 로컬 전용 MCP 서버** 를 Python 으로 만드는 방법을 처음부터 끝까지 정리합니다.

핵심은 **stdio 트랜스포트** 입니다. 서버가 HTTP 포트를 여는 게 아니라, Claude 가 서버 프로세스를 직접 띄우고 표준입출력 파이프로 대화해요. 그래서 외부에서 닿을 방법 자체가 없습니다. 사내 스크립트·로컬 DB·내 파일을 Claude 에 물려주고 싶은데 그걸 인터넷에 노출하긴 싫을 때 딱 맞는 구조예요.

> 💡 이 글에서 다루는 것  
> - "로컬 전용 / Claude 전용" 이 트랜스포트 관점에서 정확히 무슨 뜻인지  
> - Python(FastMCP)으로 가장 작은 MCP 서버 만들기  
> - 도구(tool) · 리소스(resource) · 프롬프트(prompt) 세 가지 기본 요소  
> - MCP Inspector 로 로컬 디버깅  
> - Claude Desktop / Claude Code 에 각각 등록하는 법  
> - 조금 쓸모 있는 예시 (로컬 SQLite 조회 서버)  
> - 로컬·Claude 전용 방식의 **장단점** 과 원격(HTTP) 방식과의 비교  
> - 보안 체크리스트 / 자주 밟는 함정


<br>

<br>



## 1. "로컬 전용 / Claude 전용" 이 정확히 무슨 뜻인가

MCP 를 처음 보면 "서버" 라는 단어 때문에 웹서버처럼 포트를 열어야 할 것 같지만, 꼭 그렇지 않아요. MCP 는 **호스트(host) ↔ 서버(server)** 가 어떻게 메시지를 주고받는지를 **트랜스포트** 로 정의하는데, 크게 둘입니다.

| 트랜스포트 | 동작 | 노출 범위 |
| --- | --- | --- |
| **stdio** | 호스트가 서버 프로세스를 직접 실행하고 stdin/stdout 파이프로 통신 | **같은 머신, 그 프로세스를 띄운 사용자만** |
| **Streamable HTTP** | 서버가 HTTP 엔드포인트를 열고 호스트가 네트워크로 접속 | 포트가 닿는 모든 곳 (인증·방화벽 필요) |

여기서 말하는 "로컬 전용 / Claude 전용" 은 **stdio 트랜스포트** 를 쓰겠다는 뜻이에요. 정리하면 이런 그림입니다.

- **로컬 전용** — 서버가 포트를 안 엽니다. 외부에서 접속할 소켓 자체가 없어요. 같은 컴퓨터 안에서 호스트가 자식 프로세스로 띄울 뿐.
- **Claude 전용** — 그 서버를 등록한 호스트(여기선 **Claude Desktop** 또는 **Claude Code**)만 그 서버를 띄워서 씁니다. 다른 앱·다른 사람은 등록 정보가 없으니 붙을 수 없어요.

> 💡 호스트 / 클라이언트 / 서버 용어 정리  
> **호스트** 는 Claude Desktop·Claude Code 같은 LLM 앱이고, 그 안의 **클라이언트** 가 **서버**(우리가 만들 것)와 1:1 로 연결돼요. 우리는 이 글에서 "서버" 만 만들면 됩니다.

stdio 의 수명 관리도 호스트가 합니다. **Claude 가 켜질 때 서버를 spawn 하고, 꺼질 때 같이 종료** 시켜요. 우리가 데몬을 직접 띄워둘 필요가 없습니다.


<br>

<br>



## 2. 사전 확인

세 가지만 보면 됩니다.

```shell
python3 --version   # 3.10 이상 권장 (타입힌트로 도구 스키마를 자동 생성)
uv --version        # 없으면 pip 로도 가능. 있으면 의존성 관리가 훨씬 편함
```

> ✅ Python 3.10+ 를 권장하는 이유 — FastMCP 는 함수의 **타입힌트** 를 읽어서 도구의 입력 스키마(JSON Schema)를 자동으로 만들어요. 신형 타입 표기(`list[int]`, `X | None`)를 그대로 쓰려면 3.10 이상이 편합니다.

`uv` 를 추천드려요. Claude Desktop 은 우리가 쓰는 로그인 셸의 가상환경을 모르기 때문에, "이 디렉토리에서 이 의존성으로 실행" 을 한 줄로 표현할 수 있는 `uv run` 이 등록할 때 사고를 크게 줄여줍니다. (이유는 12장 함정 부분에서 다시 짚을게요.)


<br>

<br>



## 3. 설치

프로젝트 폴더를 하나 만들고 공식 Python SDK 를 깝니다. `[cli]` 추가분이 디버깅·설치용 `mcp` 커맨드를 같이 깔아줘요.

```shell
# uv 를 쓰는 경우 (권장)
uv init mcp-local-demo
cd mcp-local-demo
uv add "mcp[cli]"
```

```shell
# 그냥 pip 를 쓰는 경우
mkdir mcp-local-demo && cd mcp-local-demo
python3 -m venv .venv && source .venv/bin/activate
pip install "mcp[cli]"
```

설치가 되면 `mcp` 커맨드가 생깁니다. 확인해볼게요.

```shell
mcp version
```


<br>

<br>



## 4. 가장 작은 MCP 서버

`server.py` 한 파일이면 됩니다. **FastMCP** 는 데코레이터로 함수를 도구로 노출해주는 고수준 래퍼예요.

```python
# server.py
from mcp.server.fastmcp import FastMCP

# 서버 이름 — Claude 쪽 UI 에 이 이름으로 보입니다.
mcp = FastMCP("local-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더해서 돌려준다."""
    return a + b


if __name__ == "__main__":
    # 인자를 안 주면 기본이 stdio 입니다. 즉 "로컬 전용" 이 기본값이에요.
    mcp.run()
```

데코레이터 안쪽은 그냥 평범한 파이썬 함수라, 따로 호출해보면 동작이 바로 보여요.

```python
# 같은 함수를 그냥 파이썬에서 불러보면
result = add(3, 4)
print(result)
```

```text
7
```

여기서 두 가지가 중요해요.

- `mcp.run()` 은 **인자가 없으면 stdio 모드** 로 돕니다. 포트를 안 열어요. 이게 우리가 원하는 "로컬 전용" 그 자체입니다.
- 함수의 타입힌트(`a: int, b: int`)와 docstring(`"""두 정수를 ..."""`)이 그대로 **도구 설명** 으로 Claude 에 전달돼요. 그래서 docstring 을 사람이 읽듯 또박또박 적어주는 게 곧 도구 품질입니다.

> ⚠️ stdio 모드에서는 `print()` 가 금지입니다 — 표준출력이 곧 프로토콜 채널이라, 우리가 `print` 로 뭔가 찍으면 JSON-RPC 메시지가 오염돼서 서버가 안 떠요. 위 `print(result)` 는 "그냥 파이썬에서 함수를 호출해본" 예시일 뿐, **서버 코드 안에는 `print` 를 두지 않습니다.** (로깅은 11장에서 다룹니다.)


<br>

<br>



## 5. 도구 · 리소스 · 프롬프트 — 세 가지 기본 요소

MCP 서버가 호스트에 제공할 수 있는 건 크게 셋이에요. 역할이 분명히 다릅니다.

| 요소 | 누가 호출하나 | 비유 |
| --- | --- | --- |
| **도구(tool)** | 모델이 자율적으로 호출 (함수 실행) | POST — 동작을 일으킴 |
| **리소스(resource)** | 앱/사용자가 컨텍스트로 끌어옴 (읽기) | GET — 데이터를 읽음 |
| **프롬프트(prompt)** | 사용자가 명시적으로 선택 (템플릿) | 슬래시 명령 템플릿 |

셋을 한 파일에 다 넣어볼게요.

```python
# server.py
import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-demo")


# ── 도구: 모델이 필요할 때 실행 ───────────────────────────
@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더해서 돌려준다."""
    return a + b


@mcp.tool()
def days_until(target: str) -> int:
    """오늘부터 target 날짜(YYYY-MM-DD)까지 남은 일수를 돌려준다."""
    d = datetime.date.fromisoformat(target)
    return (d - datetime.date.today()).days


# ── 리소스: 읽기 전용 데이터. URI 로 식별 ──────────────────
@mcp.resource("config://app-version")
def app_version() -> str:
    """앱 버전 문자열."""
    return "1.4.2"


# URI 에 {} 자리표시자를 두면 파라미터가 있는 리소스가 됩니다.
@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """이름을 받아 인사말을 돌려준다."""
    return f"안녕하세요, {name}님!"


# ── 프롬프트: 사용자가 고르는 재사용 템플릿 ────────────────
@mcp.prompt()
def summarize_3lines(text: str) -> str:
    """긴 글을 3줄로 요약시키는 프롬프트."""
    return f"다음 글을 한국어 3줄로 요약해줘:\n\n{text}"


if __name__ == "__main__":
    mcp.run()
```

`days_until` 도 그냥 함수라서, 동작은 직접 불러보면 바로 확인됩니다.

```python
print(days_until("2026-12-31"))
```

```text
199
```

이 정도면 "내 컴퓨터의 작은 기능들" 을 Claude 가 손처럼 쓰게 만드는 뼈대는 끝이에요.


<br>

<br>



## 6. 등록 전에 — MCP Inspector 로 먼저 돌려보기

Claude 에 붙이기 전에, 서버가 멀쩡한지 **Inspector** 로 확인하는 습관을 강하게 추천드려요. 등록부터 하면 "안 뜨는데 왜인지 모르는" 상태로 빠지기 쉽거든요.

```shell
mcp dev server.py
```

이러면 로컬에 Inspector 웹 UI 가 뜨고(브라우저로 열려요), 거기서 우리 서버에 붙어서

- 노출된 **도구 목록** 이 보이는지
- 도구를 직접 호출해서 **반환값** 이 맞는지
- 리소스·프롬프트가 잡히는지

를 클릭으로 확인할 수 있어요. 여기서 도구가 다 보이고 호출이 되면, Claude 쪽 등록은 거의 형식적인 단계가 됩니다.

> ✅ Inspector 에서 도구가 안 보이면 거의 항상 (1) `@mcp.tool()` 데코레이터 빠짐, (2) 타입힌트 누락으로 스키마 생성 실패, (3) import 단계 예외 — 셋 중 하나예요.


<br>

<br>



## 7. Claude Desktop 에 등록하기

방법은 두 가지예요. **자동 설치** 와 **수동 설정**.

### 7-1. 자동 설치 (`mcp install`)

가장 간단한 길. `mcp` 커맨드가 Claude Desktop 설정 파일을 알아서 고쳐줍니다.

```shell
mcp install server.py --name "local-demo"
```

서버가 환경변수를 필요로 하면 같이 넘길 수 있어요. (값은 설정 파일에 평문으로 들어가니, 토큰류는 13장 보안 규칙을 같이 보세요.)

```shell
mcp install server.py --name "local-demo" -v API_BASE=http://127.0.0.1:8080
```

### 7-2. 수동 설정 (설정 JSON 직접 편집)

자동 설치가 내부적으로 건드리는 파일을 직접 열어서 적어도 됩니다. 어떤 구조인지 알아두면 디버깅이 쉬워져요. 위치는 OS 별로 다릅니다.

| OS | 경로 |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

이 파일에 `mcpServers` 항목을 추가합니다. `uv run` 으로 "이 디렉토리에서 이 의존성으로 실행" 을 못박는 형태를 권장해요.

```json
{
  "mcpServers": {
    "local-demo": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/Users/me/projects/mcp-local-demo",
        "mcp", "run", "server.py"
      ]
    }
  }
}
```

`uv` 없이 그냥 파이썬으로 띄우려면 이렇게요. 단, 이때 `command` 와 스크립트 경로는 **둘 다 절대경로** 여야 합니다.

```json
{
  "mcpServers": {
    "local-demo": {
      "command": "/Users/me/projects/mcp-local-demo/.venv/bin/python",
      "args": ["/Users/me/projects/mcp-local-demo/server.py"]
    }
  }
}
```

> 🚨 설정을 고쳤으면 **Claude Desktop 을 완전히 재시작** 해야 반영돼요. 그냥 창만 닫으면 백그라운드 프로세스가 남아 옛 설정으로 돕니다. macOS 라면 `Cmd+Q` 로 완전히 끄고 다시 켜세요.

재시작 후, 입력창 근처의 도구(🔨) 아이콘에 우리 `local-demo` 의 도구들이 잡히면 성공입니다.


<br>

<br>



## 8. Claude Code 에 등록하기

CLI 쪽은 더 간단해요. 설정 파일을 직접 안 만지고 `claude mcp add` 한 줄이면 됩니다. `--` 뒤가 "서버를 띄우는 실제 명령" 이에요.

```shell
# 가장 단순한 형태
claude mcp add local-demo -- uv run --directory /Users/me/projects/mcp-local-demo mcp run server.py
```

등록 범위(scope)를 고를 수 있는데, 이게 "누구한테까지 보일까" 를 정하는 부분이라 중요해요.

| 스코프 | 플래그 | 저장 위치 | 의미 |
| --- | --- | --- | --- |
| local | (기본) | 내 사용자 설정 (프로젝트별) | **나만**, 이 프로젝트에서만 |
| project | `-s project` | 저장소의 `.mcp.json` (커밋됨) | 이 저장소를 받은 **모두**에게 공유 |
| user | `-s user` | 내 사용자 설정 (전역) | **나만**, 모든 프로젝트에서 |

"로컬·내 전용" 의도를 그대로 살리려면 기본(local) 또는 `user` 가 맞아요. `project` 는 `.mcp.json` 이 커밋돼서 협업자에게 퍼지니, 의도와 다르면 주의하세요.

```shell
# 모든 프로젝트에서 나만 쓰게
claude mcp add local-demo -s user -- uv run --directory /Users/me/projects/mcp-local-demo mcp run server.py

# 등록 확인 / 상세 / 삭제
claude mcp list
claude mcp get local-demo
claude mcp remove local-demo
```

붙었는지는 세션 안에서 `/mcp` 로 확인할 수 있어요. 도구가 `local-demo` 이름 아래 뜨면 됩니다.


<br>

<br>



## 9. 조금 쓸모 있는 예시 — 로컬 SQLite 조회 서버

기능 하나짜리 데모를 넘어서, **로컬에만 있는 SQLite 파일을 Claude 가 안전하게 조회** 하게 만들어볼게요. 로컬 전용 MCP 의 전형적인 쓰임새예요 — 데이터는 내 디스크에만 있고, 인터넷엔 한 바이트도 안 나갑니다.

```python
# sqlite_server.py
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-sqlite")

# 접근 가능한 DB 를 명시적으로 한 곳으로 못박는다 (경로 화이트리스트).
DB_PATH = Path.home() / "data" / "app.db"


@mcp.tool()
def list_tables() -> list[str]:
    """DB 안의 테이블 이름 목록을 돌려준다."""
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@mcp.tool()
def run_select(query: str, limit: int = 50) -> list[dict]:
    """SELECT 한정으로 쿼리를 실행하고 행을 dict 목록으로 돌려준다.

    안전을 위해 SELECT 로 시작하는 쿼리만 허용한다.
    """
    if not query.strip().lower().startswith("select"):
        raise ValueError("이 도구는 SELECT 쿼리만 허용해요.")

    # mode=ro: 읽기 전용으로 열어서 실수로도 쓰기가 안 되게 한다.
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(query).fetchmany(limit)
        return [dict(r) for r in rows]
    finally:
        con.close()


if __name__ == "__main__":
    mcp.run()
```

`run_select` 의 동작도 작은 인메모리 DB 로 바로 확인할 수 있어요. (아래는 도구 로직을 그대로 떼어 손으로 돌려본 예시입니다.)

```python
import sqlite3
con = sqlite3.connect(":memory:")
con.row_factory = sqlite3.Row
con.execute("CREATE TABLE city(name TEXT, pop INT)")
con.executemany("INSERT INTO city VALUES (?,?)",
                [("서울", 939), ("부산", 332), ("인천", 300)])

rows = con.execute("SELECT name, pop FROM city ORDER BY pop DESC").fetchmany(2)
print([dict(r) for r in rows])
```

```text
[{'name': '서울', 'pop': 939}, {'name': '부산', 'pop': 332}]
```

여기서 잡아둔 안전장치 두 개가 핵심이에요.

- **`mode=ro` (읽기 전용 연결)** — 모델이 어떤 쿼리를 만들든 DB 에 쓰기가 물리적으로 안 됩니다.
- **`SELECT` 화이트리스트** — `DROP`/`DELETE` 같은 구문을 입구에서 막아요.

로컬 전용이라고 안전한 게 아니라, **"모델이 자율적으로 호출하는 함수" 라는 점** 때문에 도구 안쪽에서 우리가 한계를 그어줘야 합니다.


<br>

<br>



## 10. 로컬·Claude 전용 방식의 장단점

이 방식(stdio · 로컬 등록)을 원격(HTTP) 방식과 견줘서 정리하면 이렇습니다.

### 장점

| 장점 | 설명 |
| --- | --- |
| **네트워크 노출이 0** | 포트를 안 엽니다. 외부에서 닿을 소켓 자체가 없어 인증·방화벽·TLS 가 아예 필요 없어요. |
| **설정이 단순** | `command` + `args` 한 덩어리가 전부. 인증 토큰·HTTPS 인증서·리버스 프록시가 없습니다. |
| **로컬 자원에 바로 접근** | 내 파일·로컬 DB·사내 스크립트를 그대로 씁니다. 자격증명도 로컬 환경변수로 끝나요. |
| **지연이 거의 없음** | 같은 머신의 stdin/stdout 파이프라 네트워크 왕복이 없습니다. |
| **수명 관리를 호스트가 대신** | Claude 가 켜질 때 띄우고 꺼질 때 종료. 데몬을 직접 띄워둘 필요가 없어요. |

### 단점

| 단점 | 설명 |
| --- | --- |
| **그 머신, 그 사용자만** | 같은 서버를 동료와 공유할 수 없어요.<br>각자 자기 컴퓨터에 따로 깔아야 합니다. |
| **호스트마다 따로 등록** | Claude Desktop 과 Claude Code 에<br>각각 등록해야 해요. |
| **런타임이 클라이언트에 있어야** | `python`·`uv` 가 그 컴퓨터에<br>깔려 있어야 떠요. |
| **중앙 관리·관측성이 약함** | 로그가 로컬에 흩어지고,<br>버전을 한 번에 올리거나<br>모니터링하기 어렵습니다. |
| **`stdout` 오염에 취약** | 표준출력이 프로토콜 채널이라,<br>코드 한 줄의 `print` 나<br>라이브러리 배너가<br>서버를 통째로 죽일 수 있어요. |

### 한 줄 결론

> 💡 **"내 컴퓨터 안의 것을, 나만, 인터넷에 안 내보내고 Claude 에 물려준다"** 면 로컬·stdio 가 정답에 가깝습니다. **여러 사람이 같이 쓰거나, 항상 떠 있어야 하거나, 중앙에서 관리** 해야 하면 그때 원격(Streamable HTTP) 으로 넘어가세요.


<br>

<br>



## 11. 로깅 — `stdout` 대신 `stderr`

stdio 모드에서 표준출력은 손대면 안 된다고 했죠. 그럼 디버깅 로그는 어디로 보낼까요? **표준에러(stderr)** 또는 **파일** 입니다.

```python
import logging
import sys

# stream=sys.stderr 가 핵심. stdout 으로 가면 프로토콜이 깨져요.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("local-demo")

log.info("서버 기동")   # 안전 — stderr 로 나감
```

Claude Desktop 이 서버 stderr 를 모아두는 로그 위치도 알아두면 좋아요.

```shell
# macOS — 서버별 로그가 mcp-server-<이름>.log 로 쌓입니다.
tail -f ~/Library/Logs/Claude/mcp*.log
```

"서버가 안 뜬다" 싶을 때 거의 항상 이 로그에 원인이 또박또박 찍혀 있어요.


<br>

<br>



## 12. 자주 밟는 함정

직접 밟아본 것들 위주로 정리합니다. 등록이 안 되거나 서버가 안 뜰 때 이 표부터 보세요.

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 서버가 떴다 바로 죽음 | 코드에 `print()` 가 있어<br>stdout 오염 | 모든 출력을<br>`stderr`/파일 로깅으로.<br>라이브러리 배너도 점검 |
| `command not found` / 안 뜸 | Claude Desktop 이<br>로그인 셸 PATH 를<br>안 물려받음 | `command` 를<br>`uv`/`python` 의<br>**절대경로** 로 |
| 파일을 못 찾음 | 호스트의 작업 디렉토리(cwd)가<br>내 폴더가 아님 | 스크립트·데이터 경로를<br>**절대경로** 로.<br>`uv run --directory` 활용 |
| 고쳤는데 그대로 | Claude Desktop 미재시작 | `Cmd+Q` 로<br>완전 종료 후 재실행 |
| 도구는 보이는데 스키마가 이상 | 타입힌트 누락 | 인자·반환에<br>타입힌트를 빠짐없이 |
| 의존성 못 찾음 | 시스템 파이썬으로 떠서<br>venv 패키지가 안 보임 | `uv run --directory`,<br>또는 venv 의<br>`python` 절대경로 지정 |

> 🚨 단 하나만 기억한다면 — **로컬 stdio 서버의 8할은 "절대경로 + stdout 청결" 로 해결됩니다.** 등록이 안 될 때 이 두 가지부터 의심하세요.


<br>

<br>



## 13. 보안 체크리스트

로컬 전용이라 외부 공격면은 거의 없지만, **모델이 도구를 자율 호출** 한다는 성질 때문에 안쪽에서 막아야 할 게 있어요.

- [ ] **위험한 도구는 읽기 전용으로** — 9장처럼 `mode=ro` + `SELECT` 화이트리스트. 삭제·쉘 실행 도구는 가능하면 안 만들거나 명시적으로 분리.
- [ ] **파일/경로는 화이트리스트** — `Path.home() / "data"` 처럼 접근 가능한 루트를 못박고, 들어온 경로가 그 안인지 검사.
- [ ] **토큰은 환경변수로, 코드·설정에 하드코딩 금지** — 값을 평문으로 두지 말 것.

```python
import os
API_TOKEN = os.environ["MY_API_TOKEN"]   # 코드에 값을 적지 않는다
```

- [ ] **설정 JSON 에 비밀 평문 주의** — `mcp install -v KEY=value` 나 설정 파일의 `env` 에 넣은 값은 평문으로 저장돼요. 진짜 비밀은 OS 키체인/별도 시크릿 매니저에서 읽어오는 형태로.
- [ ] **로그에 비밀 안 찍기** — `stderr` 로깅에 토큰이 새지 않게. (예: `Authorization=<TOKEN>` 처럼 가려서)

> ✅ 로컬 전용의 가장 큰 보안 이점은 "공격자가 네트워크로 닿을 길이 없다" 는 거예요. 대신 위협 모델이 **"모델이 실수로 위험한 도구를 부르는 것"** 으로 옮겨가니, 방어선도 도구 함수 안쪽에 그어야 합니다.


<br>

<br>



## 14. 마무리

정리하면, **로컬·Claude 전용 MCP 서버의 본질은 "stdio 트랜스포트 + 호스트 등록"** 이에요.

- 서버는 포트를 안 엽니다. `mcp.run()` 기본값이 stdio 라, 별다른 설정 없이 이미 로컬 전용입니다.
- Python(FastMCP)으로 함수에 `@mcp.tool()` 만 붙이면 도구가 되고, 타입힌트·docstring 이 곧 스펙입니다.
- Inspector(`mcp dev`)로 먼저 검증하고, Claude Desktop(`mcp install` / 설정 JSON)·Claude Code(`claude mcp add`)에 등록하면 끝이에요.
- 장점은 **노출 0 · 설정 단순 · 로컬 자원 직결**, 단점은 **공유 불가 · 중앙 관리 약함 · stdout 취약** 입니다.

내 컴퓨터 안의 작은 기능들을 모델의 손에 쥐여주되 인터넷엔 한 발짝도 안 내보내고 싶을 때, 이 구조가 가장 군더더기 없는 선택이에요.

일단 오늘은 여기까지.....   
다음 글에서는 같은 서버를 **원격(Streamable HTTP) + 인증** 으로 띄워 여러 사람이 같이 쓰게 하는 방법을 정리해볼게요. 

---

**다음 글 →:** [(2/2) MCP 서버를 원격 HTTP + 인증으로 띄우기 — 여러 사람이 같이 쓰기](/coding/MCP_서버_원격_HTTP_인증_만들기/)
