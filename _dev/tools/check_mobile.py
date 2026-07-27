"""390px 뷰포트에서 가로 오버플로가 있는지 잰다.

    python3 _dev/tools/check_mobile.py http://127.0.0.1:4000/real-estate/

이 환경에는 puppeteer/selenium 이 없다. 원시 소켓으로 CDP WebSocket 을 직접 쓴다.
스크린샷은 뷰포트 폭으로 잘려 오버플로가 안 보이므로 scrollWidth 를 재는 게 핵심이다.
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
        # 구를 하나 눌러 표까지 채운 상태로 잰다. 빈 표는 넘칠 수가 없다.
        cdp.call("Runtime.evaluate", {"expression":
            "(document.querySelector('path[data-sgg=\"11680\"]')"
            "||document.querySelector('path[data-sgg]'))"
            ".dispatchEvent(new MouseEvent('click',{bubbles:true}))"})
        time.sleep(3)

        res = cdp.call("Runtime.evaluate", {"returnByValue": True, "expression": """
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
        """})
        out = json.loads(res["result"]["result"]["value"])
        shot = cdp.call("Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": True})
        open("/tmp/re-mobile.png", "wb").write(base64.b64decode(shot["result"]["data"]))

        ok = out["doc"] <= out["inner"] and not out["bad"]
        print(f"scrollWidth={out['doc']} innerWidth={out['inner']}")
        if out["bad"]:
            print("스크롤 컨테이너 밖에서 넘치는 요소:")
            for b in out["bad"]:
                print(f"  - {b}")
        print("합격" if ok else "불합격")
        print("스크린샷: /tmp/re-mobile.png")
        return 0 if ok else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
