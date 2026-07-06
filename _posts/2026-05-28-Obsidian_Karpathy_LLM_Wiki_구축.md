---
layout: single
title:  "Obsidian + Karpathy 식 LLM WIKI 체계적으로 구축하기"
date: 2026-05-28 07:59:00 +0900
description: "Obsidian Web Clipper로 모은 RAW 노트를 Karpathy 가 말한 LLM WIKI 방식으로 정리하고, Claude Code 로 자동화하는 워크플로우를 정리했어요."
categories: coding
tag: [Obsidian, Karpathy, LLM, Wiki, ZettelKasten, ClaudeCode, WebClipper, PKM, 지식관리]
author_profile: false
toc: true
header:
  image: /assets/images/2026-05-28-obsidian-llm-wiki/header.svg
  teaser: /assets/images/2026-05-28-obsidian-llm-wiki/header.svg
---



## Summary

요즘 정보가 쌓이는 속도가 너무 빨라요. 블로그 글, 논문, 영상 캡션, 회사 위키, Slack 스크롤 한참 내리다 보면 "어제 본 그거 어디 갔지" 하는 순간이 꼭 옵니다.  
Andrej Karpathy 가 평소 흘리는 PKM 얘기 — **원본을 모아두는 inbox 와 내가 다시 쓴 wiki 를 명확히 분리한다** — 에서 힌트를 얻어, 저는 옵시디언 구조를 통째로 갈아엎었어요. 그러고 나니 검색이 너무 편해져서, 이 글에서 그 구조와 운영 흐름을 정리해서 공유드립니다.

> 💡 이 글에서 다루는 것
> - Karpathy 식 RAW / WIKI 2단 구조의 핵심
> - Chrome 확장 **Obsidian Web Clipper** 로 RAW 채워넣기
> - RAW 폴더 어떻게 나누고 어떤 메타데이터를 박을지
> - WIKI 폴더 구조화 (개념 / MOC / 프로젝트 노트)
> - RAW → WIKI 로 끌어올리는 작업 흐름
> - 그래프 뷰가 의미 있게 보이도록 태그/링크 짜는 법
> - Claude Code 로 이 모든 흐름을 자동화하는 포인트


<br>

<br>



## 1. 왜 "RAW / WIKI" 두 폴더로 나누나

지식 관리(PKM) 도구를 처음 쓰면 누구나 같은 함정에 빠져요. **수집한 자료가 곧 노트인 줄 안다는 것.** 웹에서 긁어온 글, 영상 캡션, 논문 PDF 메모를 그냥 한 폴더에 쌓다 보면, 1년쯤 지나면 검색이 안 되는 거대한 쓰레기통이 됩니다. 제가 이걸 두 번이나 했어요.

Karpathy 식 분리의 핵심은 단순합니다.

> 💡 "원본 텍스트"와 "내가 이해한 텍스트"는 같은 폴더에 들어가면 안 된다.

- **RAW** — 외부에서 들어온 원본. 가공하지 않음. 양이 많아도 OK.
- **WIKI** — 내가 한 번이라도 읽고, 내 언어로 다시 쓴 노트. 양은 적지만 밀도가 높음.

이걸 분리하면 LLM 한테 시킬 때도 명확해져요. "RAW 의 이 노트를 요약해서 WIKI 에 새 항목으로 만들어줘" 처럼 입력/출력이 깔끔하게 갈립니다. 같은 폴더에 섞여 있으면 LLM 도, 나도 헷갈립니다.


<br>

<br>



## 2. Obsidian Web Clipper 셋업

Web Clipper 는 Obsidian 공식에서 내놓은 Chrome / Firefox 확장이에요. 보고 있는 페이지를 마크다운으로 변환해서 Vault 의 지정된 폴더에 바로 넣어줍니다. 별도 서버나 동기화 설정 없이 로컬에 떨어져요.

### 2-1. 설치

크롬 웹 스토어에서 `Obsidian Web Clipper` 검색 → 설치.  
설치하면 주소창 우측에 보라색 옵시디언 아이콘이 생깁니다.

### 2-2. Vault 연결

처음 클릭하면 옵시디언이 자동으로 떠서 권한 요청해요. 허용하면 끝.  
여러 Vault 가 있다면 확장 설정의 `General → Default vault` 에서 어디로 떨어뜨릴지 지정합니다.

### 2-3. 핵심: Template 만들기

여기서 한 번에 시간을 들여야 나중에 편해져요. 확장 설정의 `Templates` 탭에서 새 템플릿을 만듭니다. 저는 RAW 전용으로 하나 둡니다.

| 필드 | 값 |
|---|---|
| Template name | `RAW - Web Clip` |
| Note location | `RAW/web/` |
| Note name | `{% raw %}{{date}}-{{title|safe_name}}{% endraw %}` |
| Behavior | `Create new note` |

`{% raw %}{{title|safe_name}}{% endraw %}` 는 파일명에 못 쓰는 문자(`/`, `:`, `?` 등)를 자동으로 치환해주는 필터예요. 안 걸어두면 페이지 제목에 슬래시 하나만 있어도 파일 생성이 실패해요.

그리고 Properties 섹션에 메타데이터를 박아둡니다.

{% raw %}
```yaml
type: raw
source: web
url: "{{url}}"
title: "{{title}}"
author: "{{author}}"
published: "{{published}}"
clipped: "{{date}}"
tags:
  - raw/web
status: inbox
```
{% endraw %}

이 `status: inbox` 가 나중에 WIKI 로 끌어올릴 때 필터링 기준이 돼요. 한 번 가공해서 WIKI 노트가 만들어지면 `status: processed` 로 바꿔둡니다.

자주 쓰는 변수와 필터는 이 정도면 충분해요.

| 변수 | 의미 |
|---|---|
| `{% raw %}{{title}}{% endraw %}` | 페이지 제목 (`<title>` 태그) |
| `{% raw %}{{url}}{% endraw %}` | 현재 URL |
| `{% raw %}{{date}}{% endraw %}` | 클립한 날짜 (YYYY-MM-DD) |
| `{% raw %}{{time}}{% endraw %}` | 클립한 시각 |
| `{% raw %}{{author}}{% endraw %}` | `<meta name="author">` 값 |
| `{% raw %}{{published}}{% endraw %}` | `<meta property="article:published_time">` |
| `{% raw %}{{description}}{% endraw %}` | OG description |
| `{% raw %}{{content}}{% endraw %}` | 본문 (자동 추출, 마크다운) |
| `{% raw %}{{selection}}{% endraw %}` | 클릭 전 선택한 텍스트만 |

필터는 `|safe_name`, `|lower`, `|trim`, `|replace("a","b")` 정도가 자주 쓰여요. 파이프로 체이닝됩니다 (`{% raw %}{{title|lower|safe_name}}{% endraw %}`).

> ✅ 본문 영역에 `{% raw %}{{content}}{% endraw %}` 를 그대로 박아두면 됩니다. 본문 자동 추출이 Mozilla Readability 기반이라 광고/네비/푸터는 거의 알아서 잘라줘요. 원본 HTML 이 꼭 필요한 경우만 `{% raw %}{{contentHtml}}{% endraw %}` 를 쓰세요.

### 2-4. 영상/논문용 템플릿도 따로

YouTube, arXiv, 회사 위키처럼 자주 가는 출처는 각각 템플릿을 따로 두는 걸 추천드려요. 확장 설정 `Templates` 탭에서 각 템플릿에 **URL pattern** 을 걸어두면, 해당 도메인에서 클립할 때 자동으로 그 템플릿이 골라집니다.

| 템플릿 | URL pattern | Note location |
|---|---|---|
| `RAW - YouTube` | `https://www.youtube.com/watch*` | `RAW/video/` |
| `RAW - arXiv` | `https://arxiv.org/abs/*` | `RAW/paper/` |
| `RAW - Notion` | `https://www.notion.so/*` | `RAW/work/` |
| `RAW - Web Clip` | (기본값) | `RAW/web/` |

이렇게 출처별로 RAW 안을 한 번 더 갈라두면 나중에 그래프 뷰에서 "어떤 출처의 노트가 얼마나 쌓였나" 가 한눈에 보입니다.

### 2-5. 트러블슈팅 (제가 한 번씩 다 겪은 것들)

| 증상 | 원인 / 해결 |
|---|---|
| 파일이 안 떨어진다 | Obsidian 이 실행 중이어야 함. 백그라운드라도 OK |
| 제목이 `unsafe-filename.md` 가 됨 | Note name 에 `safe_name` 필터 안 걸음.<br>`{% raw %}{{title|safe_name}}{% endraw %}` 로 수정 |
| 본문이 광고/메뉴까지 다 들어옴 | `content` 가 아니라 `contentHtml` 을 썼을 때 그래요.<br>`content` 로 바꾸면 Readability 가 본문만 뽑아냄 |
| 같은 글이 두 번 들어감 | `Behavior` 가 `Create new note` 면 매번 새 파일.<br>`Append to existing` 으로 바꾸면 기존 파일 뒤에 붙음 |
| 한글 파일명이 깨짐 | 운영체제 인코딩 문제. iCloud 동기화 중인 Vault 에서 자주 나옴.<br>Vault 를 로컬 디스크로 옮기면 해결 |


<br>

<br>



## 3. RAW 폴더 구조

저는 이렇게 잡고 있어요. 너무 깊게 가지 말고 **2단까지**가 한계라고 생각해요. 그 이상 들어가면 결국 안 들어갑니다.

```text
RAW/
├── web/          # 일반 블로그/문서 클립
├── video/        # YouTube, 강연 캡션
├── paper/        # arXiv, 논문 PDF 메모
├── work/         # 회사 내부 자료, Slack 발췌
├── chat/         # ChatGPT/Claude 대화 export
└── voice/        # 음성 메모 transcribe 결과
```

RAW 안의 노트는 다음 두 가지 원칙만 지킵니다.

> 💡 **원칙 1**: RAW 노트는 *절대 수정하지 않는다*. 오타가 있어도 그대로 둠. 출처와 시점을 보존하는 게 목적.

> 💡 **원칙 2**: RAW 노트끼리는 직접 링크하지 않는다. 링크는 WIKI 노트가 해야 할 일.

이 두 원칙을 어기면 RAW 와 WIKI 경계가 흐려지고, 그래프 뷰가 다시 진흙탕이 됩니다. 처음 한두 달은 손이 근질거리는데, 참아야 해요.

| 메타필드 | 의미 | 예시 |
|---|---|---|
| `type` | 항상 `raw` | `raw` |
| `source` | 출처 카테고리 | `web`, `video`, `paper` |
| `url` | 원본 링크 | `https://...` |
| `clipped` | 수집 일자 | `2026-05-28` |
| `status` | 처리 상태 | `inbox`, `reading`, `processed`, `archived` |
| `tags` | 자유 태그 | `raw/web`, `topic/rag` |

실제로 들어가는 RAW 노트는 이런 모양이에요. Web Clipper 가 자동으로 만들어주는 결과물입니다.

```markdown
---
type: raw
source: web
url: "https://example.com/posts/speculative-decoding"
title: "Speculative Decoding 정리"
author: "Some Person"
published: "2026-03-14"
clipped: "2026-05-28"
tags:
  - raw/web
status: inbox
---

# Speculative Decoding 정리

Speculative decoding 은 작은 draft 모델이 토큰을 미리 뽑고,
큰 target 모델이 검증하는 ...

(원문 그대로, 손대지 않음)
```

여기서 핵심은 frontmatter 가 풍부하다는 점이에요. 나중에 Dataview 로 "지난주에 클립한 paper 중 status 가 inbox 인 것만" 같은 쿼리를 던질 수 있게 됩니다.


<br>

<br>



## 4. WIKI 폴더 구조

WIKI 는 내가 이해한 만큼만 들어가는 공간이에요. 한 노트 = 한 개념이 원칙이고, **Zettelkasten** 의 atomic note 개념과 거의 같습니다.

```text
WIKI/
├── concept/      # 단일 개념 노트 (atomic)
├── moc/          # Map of Content (목차 노트)
├── project/      # 진행 중인 프로젝트
├── people/       # 인물 (저자, 동료)
└── template/     # 노트 템플릿 모음
```

### 4-1. concept/ — atomic note

한 파일에 한 개념만 담아요. `concept/RAG.md`, `concept/LoRA.md`, `concept/Speculative_Decoding.md` 같은 식입니다.  
길이는 짧아도 돼요. 200~500 자 정도면 충분합니다. 중요한 건 **다른 concept 노트들과 어떻게 연결되는가**.

```markdown
---
type: concept
status: living
tags: [topic/llm, topic/inference]
related: ["[[KV_Cache]]", "[[Draft_Model]]"]
---

# Speculative Decoding

작은 draft 모델이 토큰 여러 개를 미리 뽑고, 큰 target 모델이
검증만 하는 방식. ...

## 관련
- [[KV_Cache]] — 검증 단계에서 재사용
- [[Draft_Model]] — 어떤 모델을 쓰는지가 성능에 직결
```

### 4-2. moc/ — Map of Content

MOC 는 "이 주제 들어가는 입구" 역할 노트예요. concept 노트들을 묶어주는 목차 같은 거예요. 예: `moc/LLM_Inference.md` 에서 KV cache, batching, speculative decoding 같은 concept 들을 링크로 모아둡니다.

> 💡 MOC 가 있으면 그래프 뷰에서 "허브" 노드가 만들어지고, 시각적으로 흐름이 잡혀요. concept 만 잔뜩 있으면 그래프가 평평하게 퍼져서 보기 힘듭니다.

### 4-3. project/ — 진행 중인 작업

업무/사이드 프로젝트 노트. concept 와 다른 점은 **마감과 상태**가 있다는 것.

```yaml
type: project
status: active   # active / paused / done
deadline: 2026-06-30
```

project 노트는 concept 와 자유롭게 링크합니다. 거꾸로 concept 노트는 특정 project 를 직접 가리키지 않아요(시간이 지나면 의미가 죽으니까).


<br>

<br>



## 5. RAW → WIKI 끌어올리기 (가장 중요)

이게 이 워크플로우의 본체예요. 수집은 누구나 하는데, **가공 단계**가 빠지면 결국 RAW 가 쓰레기통이 됩니다. 저는 일주일에 한 번, 보통 일요일 오전에 30분~1시간씩 잡아요.

### 5-1. 후보 추려내기

먼저 RAW 에서 `status: inbox` 인 노트들을 다 띄웁니다. Obsidian 의 Dataview 플러그인이 있으면 한 줄로 끝나요.

```dataview
TABLE source, clipped
FROM "RAW"
WHERE status = "inbox"
SORT clipped DESC
```

이 중에서 **다시 안 볼 것 같은 건 과감히 status 를 `archived` 로 바꿔서 뺍니다.** 모든 RAW 가 WIKI 가 될 필요는 없어요. 보통 10개 중 2~3개만 살아남아요.

### 5-2. 한 노트 = 한 개념 추출

살아남은 RAW 노트를 읽으면서, "여기서 내가 새로 알게 된 개념이 뭐냐" 를 한두 줄로 적어봅니다.  
이미 있는 concept 노트라면 거기로 합치고, 새로운 거면 `WIKI/concept/` 에 새 파일을 만들어요.

> ⚠️ RAW 의 텍스트를 그대로 복붙하지 마세요. 무조건 **내 언어로 다시 씁니다.** 복붙하면 RAW 가 늘어난 것과 같아요. 이해가 부족해서 못 쓰겠다 싶으면 그건 아직 WIKI 로 갈 준비가 안 된 거예요. RAW 에 `status: reading` 으로 두고 다음 주에 다시 봅니다.

### 5-3. 링크 박기

새로 만든 concept 노트에서 관련 개념을 `[[...]]` 로 링크해요. **2~5개 정도**가 적당해요. 이게 그래프 뷰의 밀도를 결정합니다.

### 5-4. RAW 노트 닫기

WIKI 에 흡수되면 RAW 노트의 `status` 를 `processed` 로 바꾸고, 본문 맨 위에 어떤 WIKI 노트로 갔는지 한 줄 적어둡니다.

```yaml
status: processed
processed_into: ["[[concept/Speculative_Decoding]]"]
```

이러면 나중에 그 RAW 가 다시 검색에 걸려도, "아 이건 이미 빨아먹은 거구나" 가 한 번에 보여요.


<br>

<br>



## 6. WIKI 는 어떻게 자라는가

이건 1년쯤 굴려보면 자연스럽게 보이는 패턴인데, WIKI 의 성장은 3단계로 갑니다.

### 6-1. 1단계: concept 들이 점처럼 떠 있는 시기

처음 몇 주는 노트들이 다 외딴 섬이에요. 링크가 거의 없어요. 그래프 뷰가 진짜 별자리처럼 점만 흩어져 있습니다. 정상이에요. 여기서 포기하지 않는 게 중요해요.

### 6-2. 2단계: 같은 주제 노트들이 작은 클러스터로 묶이는 시기

3~6개월쯤 되면 비슷한 주제끼리 자동으로 뭉치기 시작해요. "어, 이 노트랑 저 노트는 같은 얘기네" 가 보이기 시작하면, 그때 **MOC 노트를 만듭니다.** 위에서 말한 `moc/LLM_Inference.md` 같은 거.

### 6-3. 3단계: MOC 끼리 연결되며 거대한 지도가 되는 시기

1년쯤 지나면 MOC 들이 서로 링크되기 시작해요. 예를 들어 `moc/LLM_Inference.md` 와 `moc/Quantization.md` 가 KV cache 압축 얘기로 만나는 식. 이 단계가 되면 그래프 뷰를 켜는 게 진짜 재밌어집니다. **새 주제를 공부할 때 "어디에 붙일지" 가 보이거든요.**

> 🎉 이 3단계까지 오면, 새 글을 읽을 때 "이거 내 위키에 어디 들어가야 하지" 가 자동으로 떠올라요. 그게 이 시스템이 효자가 되는 신호입니다.


<br>

<br>



## 7. 그래프 뷰를 의미 있게 만드는 법

Obsidian 의 그래프 뷰는 예쁘긴 한데, 아무 생각 없이 쓰면 그냥 점이 잔뜩 있는 화면이 돼요. 의미 있게 보이게 하려면 몇 가지를 신경 써야 해요.

### 7-1. 태그 계층화

태그를 평면적으로 쓰지 말고, 슬래시로 계층을 만듭니다.

| ❌ 평면 | ✅ 계층 |
|---|---|
| `llm`, `rag`, `inference` | `topic/llm`, `topic/llm/rag`, `topic/llm/inference` |
| `raw`, `web`, `paper` | `raw/web`, `raw/paper` |
| `done`, `wip`, `todo` | `status/done`, `status/wip` |

이렇게 하면 그래프 뷰의 색깔 그룹(Color groups) 으로 깔끔하게 묶을 수 있어요. `topic/llm` 으로 시작하는 노트는 다 한 색깔, `raw/*` 는 다 다른 색깔 식으로요.

### 7-2. RAW 와 WIKI 시각적으로 분리

그래프 뷰의 `Filters` 와 `Groups` 에서 다음처럼 설정해요.

```text
Group 1: path:"WIKI/concept"   → 파랑
Group 2: path:"WIKI/moc"       → 빨강 (허브로 강조)
Group 3: path:"WIKI/project"   → 초록
Group 4: path:"RAW"            → 회색 (배경처럼)
```

이렇게 두면 그래프를 켰을 때 빨간 MOC 가 허브로 보이고, 파란 concept 들이 그 주변에 모여있고, 회색 RAW 가 외곽에 깔리는 그림이 나옵니다. **눈으로 시스템 구조가 보이는 게 핵심**이에요.

### 7-3. 고아 노트(Orphan) 관리

링크가 하나도 없는 노트는 그래프에서 외톨이로 떠다녀요. Obsidian 의 `Outgoing links` / `Backlinks` 패널에서 확인 가능. 정기적으로(저는 월 1회) 고아를 찾아서 둘 중 하나로 처리합니다.

- 다른 노트랑 연결해서 살리거나
- WIKI 에 있을 자격이 없는 노트면 RAW 로 내리거나 삭제

> ⚠️ 고아를 방치하면 "다 모아둔다" 의 유혹에 다시 빠져요. 위키는 정원이지 창고가 아니에요.

### 7-4. 링크 수 적정선

한 노트가 너무 많이 링크되면 허브가 너무 두꺼워지고, 너무 적으면 외톨이가 돼요. 저는 이런 기준을 씁니다.

| 노트 종류 | 적정 outgoing 링크 수 |
|---|---|
| concept | 2~5 |
| MOC | 10~30 |
| project | 5~15 |
| RAW | 0 (원칙적으로) |


<br>

<br>



## 8. Claude Code 를 어떻게 끼워넣나

여기가 이 워크플로우의 진짜 가속 페달이에요. Claude Code 의 강점은 **로컬 파일을 직접 읽고 쓰면서 grep/find 같은 도구를 자유롭게 쓴다는 것**이에요. 옵시디언 Vault 도 결국은 마크다운 파일 더미니까, Claude Code 입장에서는 그냥 코드베이스랑 똑같아요.

### 8-1. Vault 를 작업 디렉토리로 열기

Claude Code 를 옵시디언 Vault 루트에서 띄웁니다.

```shell
cd ~/Documents/MyVault
claude
```

이러면 RAW/, WIKI/ 둘 다 한 번에 시야에 들어옵니다. 처음 띄울 때 한 번만 해두면 좋은 두 가지가 있어요.

- **`CLAUDE.md`** 를 Vault 루트에 만들어 규칙을 적어둡니다. "RAW 본문은 절대 수정 금지", "concept 노트 톤은 친근한 존댓말", "링크는 2~5개" 같은 것. 매번 안 시켜도 알아서 지켜요.
- **`.claude/settings.local.json`** 에 `permissions.deny` 로 `Bash(rm:*)` 같은 위험 명령 금지. Vault 가 통째로 날아가면 답이 없어요.

### 8-2. 주간 정리: RAW 트리아지 자동화

매주 일요일에 제가 손으로 하던 작업을 Claude Code 가 1차로 해줘요. 슬래시 커맨드로 만들어두면 편합니다.

`.claude/commands/triage-raw.md`:

```markdown
RAW/ 폴더에서 `status: inbox` 인 노트들을 모두 찾아서,
각 노트마다 다음을 한 줄씩 보고해줘:

1. 핵심 주제 (한 줄)
2. 이미 WIKI/concept/ 에 비슷한 노트가 있는지 (있다면 파일명)
3. 새로 만들 가치가 있는지 (yes / no / maybe)

WIKI 에는 절대 쓰지 말고, 보고만 해줘.
```

이러면 50개 RAW 가 쌓여있어도 5분이면 어떤 걸 살릴지 1차 분류가 끝나요. 실제 가공은 제가 해요. (남이 정리해주면 학습이 안 됨)

### 8-3. concept 노트 초안 생성

특정 RAW 를 골라서 "이걸 WIKI 로 만들어줘" 라고 시킬 수 있어요. 단, **초안만**.

```text
RAW/paper/2026-03-Speculative_Decoding.md 를 읽고,
WIKI/concept/ 아래에 한 개념 노트 초안을 만들어줘.
- 본문은 500자 이내, 내 톤으로 (친근한 존댓말)
- WIKI/concept/ 안에서 관련 있을 만한 기존 노트를 grep 으로 찾아서 [[]] 로 링크
- frontmatter 형식은 WIKI/template/concept.md 참고
- 작성 후 RAW 노트의 status 를 processed 로 바꾸고 processed_into 에 새 노트 경로 추가
```

이 작업은 Claude Code 가 정말 잘해요. 파일 여러 개를 동시에 읽고, grep 으로 기존 concept 검색하고, 링크 박는 것까지 한 흐름으로 갑니다.  
다만 **초안은 반드시 사람이 한 번 손을 거쳐야** 합니다. 그래야 WIKI 가 "내 언어" 로 유지돼요.

### 8-4. 그래프 위생 점검

월 1회 정도, 그래프 뷰 위생을 Claude Code 한테 점검시켜요.

```text
WIKI/concept/ 에서 outgoing 링크가 0개인 노트(고아) 다 찾아줘.
그리고 각 노트마다 WIKI 안에서 가장 비슷한 노트 2개를 추천해줘
(파일 내용 기반, 단순 파일명 매칭 말고).
```

이 결과를 보면서 제가 직접 링크를 박거나, 고아 노트를 RAW 로 내리거나 합니다.

### 8-5. MOC 후보 발굴

이게 가장 강력해요. concept 가 50개를 넘어가면 어떤 게 MOC 으로 묶일지 사람 눈으로 안 보여요.

```text
WIKI/concept/ 의 모든 노트를 훑어서,
서로 자주 같은 단어/주제로 묶이는 노트 그룹을 찾아줘.
그룹마다 "MOC 후보 제목" 과 묶이는 노트 파일 목록을 제시해.
```

저는 이걸로 `moc/LLM_Inference.md`, `moc/Prompt_Engineering.md` 같은 MOC 의 절반 정도를 만들었어요.

### 8-6. 주간 회고 자동 초안

매주 금요일에 그 주에 무슨 노트가 새로 들어왔고, 어떤 concept 가 자랐는지 한 장으로 정리하면 진짜 좋아요. 사람이 매주 하긴 귀찮으니까 초안만 LLM 한테 받습니다.

```text
지난 7일간 RAW/ 에 새로 들어온 노트와
WIKI/concept/ 에서 수정된(파일 mtime 기준) 노트를 모아서,
"2026-W22 주간 리뷰" 초안을 만들어줘.
- 출처별 새 RAW 개수
- WIKI 로 흡수된 수 (processed 로 바뀐 RAW)
- 이번 주에 새로 만든 concept 노트 목록과 한 줄 요약
- 다음 주에 처리해야 할 inbox 잔량
파일은 `WIKI/review/YYYY-WNN.md` 에 저장.
```

이 노트가 쌓이면 1년 후에 "내가 올해 무슨 공부를 했는가" 를 한 폴더에서 확인할 수 있어요. 회사 평가용으로도 쓸만합니다.

### 8-7. 자주 까먹는 가드레일

| 시키지 말 것 | 이유 |
|---|---|
| RAW 노트 본문 자동 수정 | RAW 의 원본 보존 원칙 |
| WIKI 노트를 자동 commit | 내 언어로 정리되지 않은 글이 들어갈 수 있음 |
| 링크를 너무 많이 박기 | concept 당 2~5개 원칙. LLM 은 보통 과하게 박음 |
| 태그를 새로 만들기 | 태그 체계는 사람이 관리. LLM 은 기존 태그만 쓰게 |


<br>

<br>



## 9. 흔한 실수 모음

제가 1년 동안 굴리면서 직접 밟은 지뢰들이에요. 처음 셋업할 때 한 번씩 떠올려보면 도움이 됩니다.

- **🚨 RAW 를 검색 안 되는 압축 폴더에 둔다** — iCloud 가 "최적화" 한답시고 마크다운을 클라우드로 올려버리면 grep 이 안 먹어요. Vault 는 로컬 디스크에 두는 게 정신건강에 좋습니다.
- **⚠️ 폴더를 3단 이상 판다** — `RAW/web/blog/llm/inference/2026/05/...` 같은 구조 만들면 한 달 만에 못 들어갑니다. 2단까지가 한계.
- **⚠️ WIKI 노트 첫 문장에 "이 글은" 이라고 쓴다** — 그런 메타 설명은 빼고 본론으로 바로 들어가요. 노트는 글이 아니라 단편이에요.
- **⚠️ MOC 를 처음부터 만든다** — concept 가 10개 이하일 때 MOC 만들면 의미 없어요. 30~50개 쌓이고 자연스럽게 묶임이 보일 때 만들어도 늦지 않아요.
- **⚠️ 매일 정리하려고 한다** — 매일 RAW 비우려고 들면 곧 지칩니다. 주 1회 30분이 가장 지속 가능해요.
- **⚠️ 모든 글을 클립한다** — Web Clipper 가 너무 편해서 무지성 클립을 하기 쉬워요. "이건 한 달 뒤에도 다시 볼까?" 를 한 번 생각하고 누르면 RAW 가 30% 가벼워집니다.


<br>

<br>



## 10. 정리

| 단계 | 도구 | 사람이 하는 일 |
|---|---|---|
| 수집 | Obsidian Web Clipper | 그냥 클릭 |
| 분류 | 자동 (템플릿) | 가끔 출처별 템플릿 다듬기 |
| 가공 | Claude Code + 사람 | 초안은 LLM, 최종은 사람 |
| 연결 | 사람 | `[[...]]` 로 링크 박기 |
| 점검 | Claude Code | 고아/MOC 후보 발굴 |
| 활용 | 사람 + 그래프 뷰 | 새 주제 공부할 때 어디 붙일지 보기 |

핵심은 결국 **"RAW 는 자동, WIKI 는 수작업, 점검은 LLM"** 이에요. 이 경계를 흐리지 않아야 시스템이 1년 이상 굴러가요.



일단 오늘은 여기까지.....   
다음 글에서는 Dataview 와 Templater 플러그인으로 RAW 트리아지를 옵시디언 안에서 끝내는 방법을 정리해볼게요. 
