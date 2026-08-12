"""브라우저로만 열리는 주소를 받아 온다 — 표준 라이브러리 + 헤드리스 크롬.

**웬만하면 쓰지 마세요.** urllib 로 되는 곳은 urllib 로 받습니다. 이건 서버가
헤더가 아니라 **TLS 지문**으로 막을 때만 쓰는 마지막 수단입니다.

Invesco(dng-api.invesco.com)가 그렇습니다. User-Agent·Accept·Origin·Referer·
sec-ch-ua·Sec-Fetch-* 를 크롬과 똑같이 맞춰도 406 이고, curl 과 urllib 이 똑같이
막히는데 같은 기계의 크롬은 200 을 받습니다. 헤더로 넘을 수 있는 벽이 아닙니다.
그런데 QQQ 는 이 유니버스에서 거래대금 1위(하루 285억 달러)로 2위의 6배라,
"못 받는다" 로 두기엔 너무 큽니다.

지켜야 할 선:

  - 크롬이 없거나 어디서든 실패하면 **빈 딕셔너리**를 돌려줍니다. 예외를 밖으로
    던지지 않습니다 — 수집기 전체가 이것 때문에 죽으면 안 됩니다. 못 받은 종목은
    화면에 "구성종목을 받지 못했습니다" 로 남고, 그건 지금과 같은 상태입니다.
  - 브라우저는 **한 번만** 띄우고 주소를 전부 훑습니다. 종목마다 새로 띄우면
    크론에서 몇 분씩 걸립니다.
  - 페이지를 렌더해 화면을 긁는 게 아니라, 그 출처에서 fetch() 를 불러 **응답
    본문만** 가져옵니다. 화면 구조가 바뀌어도 안 깨집니다.

cdp.py 도 표준 라이브러리만 씁니다(raw 소켓 위의 WebSocket). 외부 의존성은
크롬 실행 파일 하나뿐이고, 그건 이 저장소의 모바일 오버플로 검사에서 이미
쓰고 있는 것입니다.
"""

from __future__ import annotations

import json
import sys
import time

# 응답 상태와 본문을 한 문자열로 실어 나를 때 쓰는 구분자. 본문에 섞일 일이
# 없는 값이어야 한다.
_SEP = ""


def fetch_json_via_browser(
    urls: dict[str, str],
    warmup_url: str,
    timeout: float = 90.0,
    settle: float = 8.0,
) -> dict[str, bytes]:
    """urls 를 브라우저 안에서 fetch 해 본문을 돌려준다.

    urls    : {키: 주소}. 키는 호출부가 알아보는 이름(보통 티커).
    warmup  : 먼저 열어 둘 페이지. 그 출처의 쿠키·세션을 얻기 위한 것이다.

    돌려주는 건 {키: 본문 bytes} 이고, 200 이 아니거나 실패한 키는 빠진다.
    크롬을 못 띄우면 빈 딕셔너리다.
    """
    try:
        from cdp import Browser  # noqa: PLC0415 — 크롬 없는 환경에서도 임포트는 되게
    except ImportError:
        print("    cdp.py 를 못 읽었습니다 — 브라우저 수집을 건너뜁니다.", file=sys.stderr)
        return {}

    out: dict[str, bytes] = {}
    try:
        with Browser() as browser:
            page = browser.page()
            page.call("Page.enable")
            page.call("Page.navigate", {"url": warmup_url}, timeout=timeout)
            time.sleep(settle)
            for key, url in urls.items():
                body = _fetch_one(page, url, timeout)
                if body is not None:
                    out[key] = body
    except Exception as exc:  # noqa: BLE001 — 크롬은 어떤 식으로든 실패할 수 있다
        print(f"    브라우저 수집 실패({type(exc).__name__}: {exc}) — {len(out)}개만 받았습니다.",
              file=sys.stderr)
    return out


def _fetch_one(page, url: str, timeout: float) -> bytes | None:
    """페이지 안에서 fetch 한 번. 실패는 None."""
    expression = (
        "(function(){return fetch(" + json.dumps(url) + ",{credentials:'include'})"
        ".then(function(r){return r.text().then(function(t){"
        "return r.status+" + json.dumps(_SEP) + "+t;});})"
        ".catch(function(e){return 'ERR'+" + json.dumps(_SEP) + "+e;});})()"
    )
    try:
        result = page.call(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        return None
    value = (result.get("result") or {}).get("value")
    if not isinstance(value, str):
        return None
    status, _, body = value.partition(_SEP)
    if status != "200" or not body:
        return None
    return body.encode("utf-8")
