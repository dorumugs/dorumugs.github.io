---
layout: single
title:  "(3/3) Claude Code Channels — 텔레그램·CI 이벤트를 세션에 흘려보내기"
date: 2026-06-12 21:50:00 +0900
categories: coding
tag: [claude-code, channels, telegram, discord, imessage, webhook, MCP, AI코딩]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-12-claude-code-channels/header.svg
  teaser: /assets/images/2026-06-12-claude-code-channels/header.svg
description: "Claude Code 의 Channels 로 텔레그램·디스코드·CI 웹훅 같은 외부 이벤트를 돌고 있는 세션에 실시간으로 흘려보내는 법을 정리했어요. 텔레그램 셋업 명령어, 권한 릴레이, 커스텀 웹훅, Remote Control 과의 차이까지."
series: claude-code-remote
series_order: 3
---

{% include series-claude-code-remote.html current="3" %}


## Summary

[1편 Remote Control](/coding/Claude_Code_Remote_Control_완벽_가이드/)이 "내가 세션을 조종", [2편 Dispatch](/coding/Claude_Code_Dispatch_폰에서_작업_맡기기/)가 "내가 일을 던지기" 였다면, **Channels** 는 방향이 반대예요. **세상이 내 세션으로 이벤트를 밀어 넣는** 방식입니다. CI 가 깨지거나, 텔레그램으로 메시지가 오거나, 모니터링 시스템이 알람을 쏘면 — 그게 **이미 돌고 있는** Claude Code 세션 안으로 들어와서 Claude 가 바로 반응해요.

이번 글에서는 Channels 의 구조, 텔레그램을 예로 든 셋업 명령어, 권한을 폰에서 승인하는 릴레이, 그리고 커스텀 웹훅까지 풀어볼게요. 3종 시리즈의 마지막 편입니다.

> 💡 **이 글에서 다루는 것**
> - Channels 가 뭐고 어떤 구조로 동작하는지 (MCP 기반 이벤트 푸시)
> - 텔레그램·디스코드·iMessage·Fakechat 셋업 명령어
> - 실전 워크플로우 (CI 알림, 채팅 브리지, 권한 릴레이)
> - 한계와 보안 (리서치 프리뷰, 발신자 허용목록)
> - Remote Control·Slack 과의 차이

<br>

<br>



## 1. Channels 가 뭔가요?

한 문장으로 요약하면, **"외부 시스템이 보낸 이벤트를 돌고 있는 세션에 실시간으로 밀어 넣는"** 기능이에요.

핵심은 **"이미 떠 있는 세션"** 이라는 점이에요. 새 세션을 새로 띄우거나(웹·Dispatch), 폴링을 기다리는 게 아니라, 지금 작업 중인 그 세션에 이벤트가 바로 도착합니다. 그래서 Claude 가 하던 작업의 맥락을 그대로 유지한 채 반응할 수 있어요.

동작 구조를 그림으로 그리면 다음과 같아요.

```text
외부 시스템 (텔레그램 / 디스코드 / CI / 웹훅)
        ↓  이벤트 푸시
로컬 채널 서버 (플러그인 또는 커스텀 MCP 서버)
        ↓  notifications/claude/channel
Claude Code 세션 (돌고 있는 그 세션)
        ↓
Claude 가 읽고 반응 (필요하면 같은 채널로 답장)
```

**채널(channel)** 은 Claude Code 옆에서 같이 도는 일종의 MCP 서버예요. 외부 시스템이 이 채널로 이벤트를 보내면, 채널이 그걸 Claude Code 로 전달합니다. CI 알림처럼 **한 방향** 으로 받기만 할 수도 있고, 텔레그램 채팅처럼 **양방향** 으로 Claude 가 다시 답장을 보낼 수도 있어요.

> ✅ 핵심: **외부 이벤트 → 로컬 채널 서버 → 돌고 있는 세션.** 새 세션을 만드는 게 아니라 기존 세션에 끼워 넣는 거예요.

<br>

<br>



## 2. 시작 전 준비물

채널 종류와 무관하게 공통으로 필요한 건 이렇습니다.

- **Claude Code `v2.1.80` 이상** — `claude --version` 으로 확인하세요.
- **Bun 설치** — 공식 플러그인들이 Bun 위에서 돌아요. `bun --version` 으로 확인하고, 없으면 [bun.sh](https://bun.sh) 에서 설치.
- **claude.ai 계정 또는 Console API 키 인증** — Bedrock·Vertex·Foundry 같은 서드파티 프로바이더에서는 동작하지 않아요.
- **`--channels` 플래그** — 세션을 띄울 때마다 명시적으로 켜는 방식(opt-in)이에요.

> 🚨 **Team·Enterprise 라면** 관리자가 채널을 켜둬야 합니다. 관리자 설정의 `channelsEnabled: true` 가 필요하고, `allowedChannelPlugins` 로 허용 플러그인을 제한할 수도 있어요. 그리고 Channels 는 아직 **리서치 프리뷰** 라, 문법·프로토콜이 바뀔 수 있다는 점도 감안하세요.

<br>

<br>



## 3. 텔레그램 채널 셋업 (대표 예시)

가장 많이 쓰는 텔레그램을 예로 처음부터 끝까지 따라가 볼게요.

**① 텔레그램에서 봇 만들기** — 텔레그램에서 [@BotFather](https://t.me/BotFather) 를 열고 `/newbot` 을 보냅니다. 표시 이름과 (반드시 `bot` 으로 끝나는) 고유 username 을 정하면, BotFather 가 **토큰** 을 줘요. 이 토큰을 복사해둡니다.

> 🚨 이 봇 토큰은 민감정보예요. 실제 글·메모·커밋에 그대로 박지 말고, 환경변수나 설정 파일로만 다루세요.

<br>

**② 플러그인 설치** — Claude Code 안에서:

```text
/plugin install telegram@claude-plugins-official
```

만약 플러그인을 못 찾는다고 나오면 마켓플레이스를 갱신하고 다시 설치합니다.

```text
/plugin marketplace update claude-plugins-official
```

<br>

**③ 플러그인 리로드 후 토큰 설정**

```text
/reload-plugins
/telegram:configure <BotFather에서_받은_토큰>
```

설정값은 `~/.claude/channels/telegram/.env` 에 저장돼요.

<br>

**④ 채널 플래그를 붙여 재시작** — 일단 Claude Code 를 나갔다가, 이번엔 `--channels` 를 붙여 다시 띄웁니다.

```shell
claude --channels plugin:telegram@claude-plugins-official
```

<br>

**⑤ 내 텔레그램 계정 페어링** — 텔레그램에서 내 봇에게 아무 메시지나 하나 보내면, 봇이 **페어링 코드** 를 답장해줘요. 그 코드를 터미널에서 입력합니다.

```text
/telegram:access pair <봇이_준_코드>
```

<br>

**⑥ 접근 잠그기 (권장)** — 나만 메시지를 보낼 수 있게 허용목록 정책으로 바꿉니다.

```text
/telegram:access policy allowlist
```

이제 텔레그램에서 봇에게 보낸 메시지가 돌고 있는 Claude Code 세션으로 들어가고, Claude 가 같은 채널로 답장까지 보내요.

> 💡 디스코드도 흐름은 똑같아요. [Discord 개발자 포털](https://discord.com/developers/applications)에서 봇을 만들고 **Message Content Intent** 를 켠 뒤 서버에 초대하고 나면, `/plugin install discord@claude-plugins-official` → `/discord:configure <토큰>` → `claude --channels plugin:discord@claude-plugins-official` → `/discord:access pair <코드>` 순서로 동일합니다.

<br>

<br>



## 4. 더 간단한 채널들

봇 토큰 발급이 번거롭다면, 더 가벼운 선택지도 있어요.

**iMessage (macOS 전용)** — 봇 토큰 없이 내 Mac 의 메시지 앱을 그대로 씁니다. 먼저 **시스템 설정 → 개인정보 보호 및 보안 → 전체 디스크 접근 권한** 에 터미널 앱을 추가해야 해요 (메시지 DB 를 읽어야 하거든요). 그다음:

```shell
claude --channels plugin:imessage@claude-plugins-official
```

내 자신에게 메시지를 보내면 바로 세션에 도착해요 (셀프 채팅은 접근 제어를 건너뜁니다). 다른 연락처를 허용하려면:

```text
/imessage:access allow +15551234567
```

<br>

**Fakechat (테스트용)** — 진짜 채팅앱을 붙이기 전에 채널이 뭔지 감만 잡고 싶을 때 제일 좋아요.

```shell
claude --channels plugin:fakechat@claude-plugins-official
```

띄운 뒤 브라우저로 `http://localhost:8787` 을 열고 메시지를 치면, 그게 곧바로 Claude Code 로 들어옵니다. 외부 의존성이 전혀 없어서 첫 실험용으로 딱이에요.

> 💡 여러 채널을 한 세션에 동시에 붙일 수도 있어요. `claude --channels plugin:telegram@claude-plugins-official plugin:discord@claude-plugins-official` 처럼 나열하면 됩니다.

<br>

<br>



## 5. 실전 워크플로우

채널이 실제로 어떻게 쓰이는지 그림 몇 개로 그려볼게요.

**예시 ① CI 실패를 텔레그램으로 받아 바로 고치기**

```text
GitHub CI 가 main 에서 실패
   ↓ 웹훅 → 텔레그램 채널
돌고 있는 세션에 이벤트 도착:
   <channel source="telegram"> build failed on main: <빌드 로그 링크> </channel>
   ↓
Claude 가 로그를 보고 원인 코드를 고친 뒤 CI 재실행
   ↓
Claude 가 텔레그램으로 답장: "빌드 고쳤어요, 테스트 다시 돌리는 중"
```

자리에 없어도 폰으로 실시간으로 상황이 오가는 셈이에요.

<br>

**예시 ② 텔레그램을 원격 CLI 처럼 쓰기**

```text
폰에서 봇에게: "지금 작업 디렉토리에 뭐 있어?"
   ↓ 텔레그램 채널 → 세션
Claude 가 ls 를 돌리고 결과를 텔레그램으로 답장
```

코드 작업은 전부 내 로컬 머신에서 돌고, 나는 텔레그램으로 그걸 조종하는 모양이 돼요.

<br>

**예시 ③ 권한을 폰에서 승인 (권한 릴레이)**

오래 걸리는 작업을 걸어두고 자리를 떴는데, Claude 가 승인이 필요한 명령을 만나 멈춰버리면 답답하죠. 채널이 그 승인 요청을 채팅으로 넘겨줄 수 있어요.

```text
Claude 가 Bash 실행 승인 필요 → 터미널엔 다이얼로그가 떠 있지만 나는 자리에 없음
   ↓ 채널이 승인 요청을 텔레그램으로 전달
"Claude 가 `apt-get update` 를 실행하려 해요. 'yes abcde' 또는 'no abcde' 로 답하세요"
   ↓ 폰에서: "yes abcde"
   ↓
Claude 가 명령 실행 (터미널 앞 사람이 먼저 답하면 그게 우선)
```

> ⚠️ 단, 권한 릴레이는 채널이 그 기능을 선언한 경우에만 동작해요. 그리고 자리를 완전히 비운 채 무인으로 돌리려고 `--dangerously-skip-permissions` 를 쓰는 건 위험하니 신중하게.

<br>

<br>



## 6. Remote Control·Slack 과 뭐가 다른가요?

같은 "터미널 밖에서 쓰기" 라도 트리거와 실행 위치가 달라요.

| | Channels | Remote Control | Slack |
|---|---|---|---|
| **무엇이 들어오나** | 외부 이벤트 (텔레그램·CI·웹훅) | 내 명령 (claude.ai/code·앱) | 채널에서 `@Claude` 멘션 |
| **트리거 성격** | 반응형 (이벤트 구동) | 인터랙티브 (내가 조종) | 팀 채팅 멘션 |
| **Claude 실행 위치** | 내 머신 (로컬 세션) | 내 머신 (로컬 세션) | Anthropic 클라우드 |
| **세션 상태** | 이미 떠 있어야 함 | 이미 떠 있어야 함 | 클라우드에서 새로 |
| **양방향?** | 예 (같은 채널로 답장) | 예 | 예 |

쉽게 가르면, **외부 이벤트(CI·채팅 알림)가 내 세션으로 흘러들어오게** 하고 싶으면 Channels, **내가 직접 진행 중인 세션을 다른 기기에서 끌고** 가고 싶으면 Remote Control, **팀 채널에서 PR·리뷰** 를 돌리고 싶으면 Slack 이에요.

<br>

<br>



## 7. 한계와 보안

마지막으로 주의점을 정리할게요.

- **세션이 떠 있어야 이벤트가 도착해요.** 스케줄 작업과 달리 채널은 세션이 열려 있는 동안에만 이벤트를 받습니다. 세션이 꺼져 있으면 그동안 온 메시지는 따로 큐잉되지 않아요.
- **리서치 프리뷰예요.** 문법·프로토콜이 바뀔 수 있고, 커스텀 채널을 개발 중 테스트하려면 `--dangerously-load-development-channels` 플래그가 필요합니다.
- **발신자 허용목록이 사실상 필수예요.** 텔레그램·디스코드는 페어링, iMessage 는 `/imessage:access allow` 로 허용된 발신자만 메시지를 밀어 넣을 수 있어요. 신뢰할 수 없는 네트워크에선 더 조심해야 합니다.
- **iMessage 는 macOS 전용** 이에요. 텔레그램·디스코드는 macOS·Windows 양쪽에서 됩니다.
- **커스텀 웹훅은 손이 많이 가요.** 모니터링·내부 시스템을 붙이려면 직접 채널 서버를 만들어야 하는데, MCP 와 채널 프로토콜 이해가 필요합니다. 자세한 건 [Channels 레퍼런스 문서](https://code.claude.com/docs/en/channels-reference)를 참고하세요.

<br>

<br>



## 8. 장점·단점 한눈 정리

| 👍 장점 | 👎 단점 |
|---|---|
| 외부 이벤트가 돌고 있는 세션에 실시간 도착 | 세션이 떠 있는 동안만 이벤트 수신 |
| 새 세션 안 띄우고 기존 환경·맥락 재사용 | 리서치 프리뷰 — 문법·프로토콜 변경 가능 |
| 양방향 — 같은 채널로 답장 가능 | 채팅앱마다 봇 토큰·페어링·허용목록 셋업 필요 |
| 커스텀 웹훅으로 CI·모니터링 등 무엇이든 연결 | 커스텀 채널은 MCP·Bun 이해가 필요 |
| 권한 릴레이로 폰에서 원격 승인 | iMessage 는 macOS 전용 |
| Pro·Max·Team·Enterprise 다 지원 | 발신자 허용목록 관리가 필요 |

<br>

<br>



3종 시리즈를 마무리하면, 세 기능은 결국 "Claude Code 를 터미널 앞에만 묶어두지 않는" 서로 다른 길이에요. **[조종]은 Remote Control, [위임]은 Dispatch, [반응]은 Channels.** 내 작업 패턴에 맞는 걸 하나만 익혀둬도 자리에 묶여 있을 이유가 확 줄어듭니다.

<br>

일단 오늘은 여기까지.....   
세 편을 끝까지 봐주셔서 고마워요. 다음엔 또 다른 주제로 찾아올게요.

---

**← 이전 글:** [(2/3) Claude Code Dispatch — 폰에서 작업 맡기고 결과만 받기](/coding/Claude_Code_Dispatch_폰에서_작업_맡기기/)
