"""390px 뷰포트에서 가로 오버플로가 있는지 잰다.

    python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/

이 환경에는 puppeteer/selenium 이 없다. 원시 소켓으로 CDP WebSocket 을 직접 쓴다.
스크린샷은 뷰포트 폭으로 잘려 오버플로가 안 보이므로 scrollWidth 를 재는 게 핵심이다.

두 가지 상태를 각각 잰다.

1. click — 구를 하나 눌러 표까지 채운 상태. 빈 표는 넘칠 수가 없다.
2. tooltip — 화면에 보이는 것 중 가장 오른쪽 구에 mousemove 를 쏴서 지도 툴팁
   (.re-tip)을 실제로 띄운 상태. .re-tip 은 mousemove/focus 이벤트로만 나타나는데
   click 한 번으로는 절대 렌더링되지 않으므로, click 상태만 재면 이 요소는 검사
   대상에 아예 들어오지 않는다 — 실제로 이 경로에서 화면 오른쪽 끝 근처 구를
   누르면 페이지가 넘치는 버그가 있었고, click 만 보내는 이전 버전은 이를 놓쳤다.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request

WIDTH = 390
HEIGHT = 844
PORT = 9334

# .re-app 안에서 뷰포트보다 넓은 요소를 찾되, 스크롤 가능한 조상(.re-table-wrap 같은)
# 안에 있으면 정상으로 보고 건너뛴다. click/tooltip 두 상태에서 그대로 재사용한다.
MEASURE_JS = """
  (() => {
    const doc = document.documentElement.scrollWidth;
    const bad = [];
    for (const el of document.querySelectorAll('.re-app *')) {
      const r = el.getBoundingClientRect();
      if (r.width <= window.innerWidth + 1) continue;
      let p = el.parentElement, scrollable = false;
      while (p) {
        if (getComputedStyle(p).overflowX === 'auto'
            || getComputedStyle(p).overflowX === 'scroll') { scrollable = true; break; }
        p = p.parentElement;
      }
      if (!scrollable) bad.push(el.className + ' w=' + Math.round(r.width));
    }
    return JSON.stringify({doc, inner: window.innerWidth, bad: bad.slice(0, 8)});
  })()
"""

# 지도에서 화면에 보이는(.style.display !== 'none') 구 중 오른쪽 끝이 가장 먼 것을
# 골라 mousemove 를 쏴 툴팁을 띄운다. 지도가 없는 페이지(일반 블로그 글)에서는
# best 가 null 이라 조용히 실패하는데, 그런 페이지에는 애초에 이 시나리오가 없으므로
# 문제 없다.
TRIGGER_TOOLTIP_JS = """
  (() => {
    const paths = Array.from(document.querySelectorAll('svg.re-map path[data-sgg]'))
      .filter((p) => p.style.display !== 'none');
    let best = null, bestRight = -Infinity;
    for (const p of paths) {
      const r = p.getBoundingClientRect();
      if (r.right > bestRight) { bestRight = r.right; best = p; }
    }
    if (!best) return;
    const r = best.getBoundingClientRect();
    best.dispatchEvent(new MouseEvent('mousemove', {
      bubbles: true, clientX: r.right - 2, clientY: r.top + r.height / 2,
    }));
  })()
"""


class CDP:
    def __init__(self, ws_url: str) -> None:
        host, rest = ws_url.split("://")[1].split("/", 1)
        h, p = host.split(":")
        self.sock = socket.create_connection((h, int(p)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.send((
            f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        self.mid = 0

    def _send(self, method: str, params: dict) -> int:
        self.mid += 1
        payload = json.dumps({"id": self.mid, "method": method,
                              "params": params}).encode()
        mask = os.urandom(4)
        n = len(payload)
        hdr = b"\x81"
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.send(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        return self.mid

    def _read(self) -> dict:
        def rd(n: int) -> bytes:
            out = b""
            while len(out) < n:
                out += self.sock.recv(n - len(out))
            return out
        _, b2 = rd(2)
        ln = b2 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", rd(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", rd(8))[0]
        return json.loads(rd(ln))

    def call(self, method: str, params: dict | None = None) -> dict:
        mid = self._send(method, params or {})
        while True:
            msg = self._read()
            if msg.get("id") == mid:
                return msg


def measure(cdp: CDP) -> dict:
    """현재 상태에서 document.scrollWidth 와 넘치는 요소 목록을 잰다."""
    res = cdp.call("Runtime.evaluate", {"returnByValue": True, "expression": MEASURE_JS})
    return json.loads(res["result"]["result"]["value"])


def report(state: str, out: dict) -> bool:
    """한 상태(state)의 측정 결과를 출력하고 합격 여부를 돌려준다."""
    ok = out["doc"] <= out["inner"] and not out["bad"]
    print(f"[{state}] scrollWidth={out['doc']} innerWidth={out['inner']}")
    if out["bad"]:
        print(f"[{state}] 스크롤 컨테이너 밖에서 넘치는 요소:")
        for b in out["bad"]:
            print(f"  - {b}")
    print(f"[{state}] " + ("합격" if ok else "불합격"))
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: check_mobile.py <URL>", file=sys.stderr)
        return 2
    url = sys.argv[1]

    proc = subprocess.Popen(
        ["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={PORT}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3)
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        cdp = CDP(ws)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride",
                 {"width": WIDTH, "height": HEIGHT, "deviceScaleFactor": 2, "mobile": True})
        cdp.call("Page.navigate", {"url": url})
        time.sleep(7)

        # 1) 구를 하나 눌러 표까지 채운 상태로 잰다.
        cdp.call("Runtime.evaluate", {"expression":
            "(document.querySelector('path[data-sgg=\"11680\"]')"
            "||document.querySelector('path[data-sgg]'))"
            ".dispatchEvent(new MouseEvent('click',{bubbles:true}))"})
        time.sleep(3)
        ok_click = report("click", measure(cdp))

        # 2) 가장 오른쪽에 보이는 구에 mousemove 를 쏴 지도 툴팁을 띄운 채로 다시 잰다.
        cdp.call("Runtime.evaluate", {"expression": TRIGGER_TOOLTIP_JS})
        time.sleep(1)
        ok_tip = report("tooltip", measure(cdp))

        shot = cdp.call("Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": True})
        open("/tmp/re-mobile.png", "wb").write(base64.b64decode(shot["result"]["data"]))
        print("스크린샷: /tmp/re-mobile.png")

        ok = ok_click and ok_tip
        print("전체 결과: " + ("합격" if ok else "불합격"))
        return 0 if ok else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
