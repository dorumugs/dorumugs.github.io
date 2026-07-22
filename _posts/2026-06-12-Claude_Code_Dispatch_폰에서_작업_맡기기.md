---
layout: single
title:  "(2/3) Claude Code Dispatch — 폰에서 작업 맡기고 결과만 받기"
date: 2026-06-12 21:40:00 +0900
categories: coding
tag: [claude-code, dispatch, cowork, 모바일, AI코딩, 데스크톱앱, 푸시알림, 원격작업]
author_profile: false
toc: true
header:
  image: /assets/images/2026-06-12-claude-code-dispatch/header.svg
  teaser: /assets/images/2026-06-12-claude-code-dispatch/header.svg
description: "Claude Code 의 Dispatch 로 폰에서 \"이 버그 고쳐줘\" 한 줄 보내면 내 데스크톱이 알아서 작업하고 결과를 푸시로 알려줍니다. 셋업·페어링·실전 예시와 Remote Control 과의 차이를 초보자 눈높이로 정리했어요."
series: claude-code-remote
series_order: 2
---

{% include series-claude-code-remote.html current="2" %}


## Summary

[1편 Remote Control](/coding/Claude_Code_Remote_Control_완벽_가이드/)이 "내 PC 세션을 폰으로 **조종**" 하는 거였다면, **Dispatch** 는 한 발 더 나가서 "폰에서 일을 **던져주면** 내 데스크톱이 알아서 처리" 하는 방식이에요. 카페에 앉아 폰으로 "로그인 버그 고치고 PR 올려줘" 한 줄 보내면, 집에 켜둔 데스크톱에서 Claude Code 세션이 새로 떠서 일을 하고, 끝나면 폰으로 알림이 옵니다.

이번 글에서는 Dispatch 가 어떤 구조인지, 어떻게 셋업하고 쓰는지, 그리고 1편의 Remote Control 과 정확히 뭐가 다른지를 풀어볼게요.

> 💡 **이 글에서 다루는 것**
> - Dispatch 가 뭐고 어떤 흐름으로 동작하는지 (Cowork → Code 세션)
> - 데스크톱 ↔ 모바일 페어링 셋업
> - 실제로 작업을 던지는 예시
> - 알아둬야 할 한계 (Pro·Max 전용, 데스크톱 켜짐, 30분 승인 만료)
> - Remote Control 과의 차이 (조종 vs 위임)

<br>

<br>



## 1. Dispatch 가 뭔가요?

한 문장으로 요약하면, **"폰의 Cowork 탭에서 일을 던지면, Claude 가 알아서 처리하거나 내 데스크톱에 Code 세션을 띄워서 시키는"** 기능이에요.

Claude 데스크톱 앱에는 **Cowork** 탭이 있어요. 여기서 Claude 와 나누는 대화가 일종의 "비서 창구" 라고 보면 됩니다. 폰에서 이 Cowork 으로 할 일을 보내면, Dispatch 가 그 일의 성격을 보고 두 갈래로 나눠요.

- **개발 작업** (버그 수정, 의존성 업데이트, 테스트 실행, PR 올리기 등) → 내 **데스크톱에 Code 세션을 새로 띄워서** 처리
- **그 외 작업** (조사, 문서 작성, 스프레드시트 등) → Cowork 안에서 Claude 가 직접 처리

개발 작업으로 분류되면 데스크톱 앱의 **Code 탭** 사이드바에 **Dispatch 배지** 가 붙은 세션이 새로 생겨요. 그 세션은 내 로컬 데스크톱에서 도니까, 평소처럼 내 파일·도구·MCP 서버를 그대로 씁니다. 일이 끝나거나 승인이 필요하면 폰으로 푸시 알림이 와요.

> ✅ 핵심: **폰에서 일을 던진다 → Dispatch 가 성격을 보고 분류한다 → 개발 작업이면 데스크톱에 세션을 새로 띄운다 → 결과를 폰으로 알린다.**

<br>

<br>



## 2. Remote Control 과 뭐가 다른가요?

1편을 읽었다면 "이거 Remote Control 이랑 뭐가 다르지?" 싶을 거예요. 가장 큰 차이는 **"이미 도는 세션을 조종" vs "새 세션을 던져서 위임"** 입니다.

| | Dispatch | Remote Control |
|---|---|---|
| **내가 하는 일** | 폰에서 새 작업을 메시지로 던짐 | 이미 도는 세션을 실시간으로 조종 |
| **언제 쓰나** | 작업 시작 *전* — 비동기 위임 | 작업이 도는 *중* — 라이브 제어 |
| **세션 생성** | Dispatch 가 필요하면 새로 띄움 | 내가 `claude remote-control` 로 띄움 |
| **상호작용 방식** | 일 던지고 → 결과 알림 받기 | 작업 보면서 한 줄씩 같이 진행 |
| **데스크톱 켜짐 필요** | 예 (세션이 거기 뜨니까) | 예 (세션이 거기서 도니까) |
| **지원 플랜** | **Pro·Max 만** | Pro·Max·Team·Enterprise |

쉽게 가르면 이래요. **"일 시켜놓고 나중에 확인" 하고 싶으면 Dispatch**, **"지금 돌고 있는 작업을 옆에서 같이 끌고 가고" 싶으면 Remote Control** 입니다. Dispatch 는 폰에서 데스크톱으로 일을 보내는 한 방향이라, 이미 도는 세션을 폰으로 끌고 오려면 그건 Remote Control 의 몫이에요.

<br>

<br>



## 3. 시작 전 준비물

Dispatch 는 명령어로 켜는 게 아니라 **앱끼리 페어링** 으로 동작해요. 네 가지를 갖춰두면 됩니다.

1. **Claude 데스크톱 앱** (macOS·Windows) 설치 — Code 세션이 여기 뜹니다.
2. **Pro 또는 Max 플랜** 계정으로 데스크톱 앱 로그인. (Team·Enterprise 는 Dispatch 미지원이에요.)
3. **Claude 모바일 앱** (iOS·Android)에 **같은 계정** 으로 로그인.
4. **모바일 앱 ↔ 데스크톱 페어링** — 이게 핵심이에요. 자세한 단계는 [Anthropic 페어링 안내](https://support.claude.com/en/articles/13947068) 문서를 따라가면 됩니다.

> 🚨 페어링이 Dispatch 의 전부라고 봐도 돼요. 폰에서 던진 일이 "어느 데스크톱" 으로 갈지를 이 페어링이 정해주거든요. 페어링이 안 돼 있으면 Cowork 에서 메시지를 보내도 데스크톱에 Code 세션이 안 뜹니다.

<br>

<br>



## 4. 실제로 작업 던지기

셋업이 끝났으면 쓰는 건 정말 간단해요. 별도 명령어나 플래그가 없고, 전부 앱 UI 로 돌아갑니다.

1. 폰에서 Claude 앱을 엽니다.
2. **Cowork** 탭으로 갑니다.
3. 할 일을 메시지로 씁니다. 예: `로그인 페이지 오타 고치고 PR 올려줘`
4. 전송하면 Dispatch 가 성격을 판단해요.
   - 개발 작업이면 → 데스크톱에 Code 세션을 자동으로 띄움 (Code 탭에 **Dispatch 배지** 표시)
   - 아니면 → Cowork 안에서 Claude 가 바로 처리
5. 진행 상황을 받습니다.
   - 세션이 끝나거나 승인이 필요하면 **폰으로 푸시 알림**
   - 데스크톱 앱을 열어 실시간으로 지켜봐도 됩니다.

예시로 어떤 메시지가 어디로 가는지 감을 잡아볼게요.

| 폰에서 던진 메시지 | Dispatch 의 판단 |
|---|---|
| "결제 모듈 테스트 돌려보고 깨지면 고쳐줘" | 개발 작업 → 데스크톱 Code 세션 |
| "npm 패키지 전부 최신으로 올리고 테스트 통과하면 PR" | 개발 작업 → 데스크톱 Code 세션 |
| "우리 코드베이스 아키텍처 요약해줘" | 조사 → Cowork 에서 바로 처리 |

> 💡 1편의 Remote Control 처럼, 프롬프트에 "끝나면 알려줘" 를 굳이 안 붙여도 Dispatch 가 띄운 세션은 완료·승인 시점에 알아서 푸시를 보냅니다.

<br>

<br>



## 5. 한계와 주의점

편한 만큼 전제 조건도 분명해요. 미리 알아두면 덜 당황합니다.

- **Pro·Max 전용입니다.** Team·Enterprise 플랜에서는 Dispatch 를 못 써요. (이 부분이 1편 Remote Control 과 갈리는 지점이에요. Remote Control 은 Team·Enterprise 도 관리자가 켜면 됩니다.)
- **데스크톱이 켜져 있고 네트워크에 붙어 있어야 합니다.** 앱을 닫거나 컴퓨터를 꺼두면, 다시 온라인이 될 때까지 새 세션이 도착하지 못해요.
- **앱 승인은 30분 만료예요.** Dispatch 가 띄운 Code 세션이 GUI 앱(컴퓨터 사용 기능) 승인을 받았어도, 그 승인은 30분 뒤 만료돼 다시 물어봅니다. 일반 Code 세션의 무기한 승인과 다른 점이에요.
- **한 방향입니다.** 폰 → 데스크톱으로 일을 보낼 수는 있어도, 이미 도는 세션을 폰으로 끌고 와 조종하는 건 안 돼요. 그건 1편 Remote Control 입니다.
- **트리거가 Cowork·모바일 뿐이에요.** Slack·Discord·Telegram 같은 외부 채팅에서는 Dispatch 를 못 띄웁니다. (외부 이벤트로 띄우고 싶다면 3편 Channels 를 보세요.)

<br>

<br>



## 6. 장점·단점 한눈 정리

| 👍 장점 | 👎 단점 |
|---|---|
| 폰에서 일 던지면 내 머신이 알아서 처리 | Pro·Max 전용 (Team·Enterprise 불가) |
| 작업 성격을 보고 자동 라우팅 (Code vs Cowork) | 데스크톱이 켜져 있어야 세션이 뜸 |
| 띄워진 세션은 내 로컬 파일·도구·MCP 그대로 | 컴퓨터 사용 승인이 30분이면 만료 |
| 완료·승인 시 폰 푸시 알림 | 한 방향 — 도는 세션 조종은 불가 |
| 터미널·CLI 필요 없이 앱만으로 | 외부 채팅(슬랙 등)으론 못 띄움 |

<br>

<br>



이미 평소에 Claude 데스크톱 앱을 쓰고 있다면, Dispatch 는 페어링 한 번으로 "이동 중 비서" 가 생기는 셈이에요. 출근길에 떠오른 자잘한 수정을 폰으로 툭 던져놓으면, 자리에 앉기 전에 PR 이 올라와 있는 경험을 하게 됩니다.

<br>

일단 오늘은 여기까지.....   
다음 글에서는 텔레그램 메시지나 CI 실패 같은 **외부 이벤트를 돌고 있는 세션에 흘려보내는 Channels** 를 정리해볼게요.

---

**← 이전 글:** [(1/3) Claude Code Remote Control — 폰으로 내 PC 세션 이어받기](/coding/Claude_Code_Remote_Control_완벽_가이드/) ｜ **다음 글 →:** [(3/3) Claude Code Channels — 텔레그램·CI 이벤트를 세션에 흘려보내기](/coding/Claude_Code_Channels_외부이벤트_연결/)
