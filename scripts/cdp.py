"""헤드리스 Chrome 을 직접 모는 최소 CDP 클라이언트. 표준 라이브러리만 쓴다.

`websocket-client` 를 안 쓰는 이유는 이 저장소가 표준 라이브러리만 쓰기
때문이다. 필요한 건 프레임 만들기·풀기뿐이라 100줄이면 된다.

쓰는 곳
  scripts/collect_kream.py   KREAM 시세를 브라우저로 받아온다
  검증 스크립트              390px 오버플로·화면 동작 확인
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import time
import urllib.request

# 헤드리스 기본 UA 에는 'HeadlessChrome' 이 박혀 있다. 그것만 보고 걸러내는
# 곳이 있어서(KREAM 이 그렇다) 평범한 Chrome UA 로 바꿔 준다.
DEFAULT_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/149.0.0.0 Safari/537.36")


class Timeout(Exception):
    pass


class Connection:
    """CDP 웹소켓 한 개. 요청/응답과 이벤트를 다룬다."""

    def __init__(self, ws_url: str, timeout: float = 30.0):
        _, rest = ws_url.split("://", 1)
        hostport, path = rest.split("/", 1)
        host, port = hostport.split(":")
        self._sock = socket.create_connection((host, int(port)), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall((
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("CDP 핸드셰이크가 끊겼습니다")
            buf += chunk
        self._buf = buf.split(b"\r\n\r\n", 1)[1]
        self._next_id = 0
        self.events: list[dict] = []

    def _read(self, count: int) -> bytes:
        while len(self._buf) < count:
            chunk = self._sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("CDP 소켓이 닫혔습니다")
            self._buf += chunk
        out, self._buf = self._buf[:count], self._buf[count:]
        return out

    def _send_frame(self, payload: bytes) -> None:
        head = bytearray([0x81])
        size = len(payload)
        if size < 126:
            head.append(0x80 | size)
        elif size < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", size)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", size)
        mask = os.urandom(4)
        head += mask
        self._sock.sendall(bytes(head) + bytes(
            byte ^ mask[i % 4] for i, byte in enumerate(payload)))

    def _recv_message(self) -> dict:
        """조각난 프레임을 이어 붙여 메시지 하나를 만든다."""
        data = b""
        while True:
            first, second = self._read(2)
            size = second & 0x7F
            if size == 126:
                size = struct.unpack(">H", self._read(2))[0]
            elif size == 127:
                size = struct.unpack(">Q", self._read(8))[0]
            data += self._read(size)
            if first & 0x80:
                return json.loads(data.decode("utf-8", "replace"))

    def call(self, method: str, params: dict | None = None, timeout: float = 60.0):
        """명령 하나를 보내고 그 응답을 기다린다. 도중의 이벤트는 모아 둔다."""
        self._next_id += 1
        request_id = self._next_id
        self._send_frame(json.dumps(
            {"id": request_id, "method": method, "params": params or {}}).encode())
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._sock.settimeout(max(1.0, deadline - time.time()))
            try:
                message = self._recv_message()
            except socket.timeout:
                break
            if message.get("id") == request_id:
                if message.get("error"):
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            if message.get("method"):
                self.events.append(message)
        raise Timeout(method)

    def evaluate(self, expression: str, timeout: float = 60.0):
        """페이지 안에서 식을 계산해 값을 돌려받는다."""
        result = self.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True}, timeout)
        return result.get("result", {}).get("value")

    def read_big_string(self, expression: str, step: int = 500_000) -> str:
        """큰 문자열을 조각내 가져온다. 한 번에 받으면 CDP 가 버거워한다."""
        size = self.evaluate(f"({expression}) ? ({expression}).length : 0") or 0
        parts = []
        for offset in range(0, int(size), step):
            parts.append(self.evaluate(
                f"({expression}).slice({offset},{offset + step})") or "")
        return "".join(parts)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class Browser:
    """헤드리스 Chrome 프로세스 하나. with 문으로 쓴다."""

    def __init__(self, port: int = 9222, user_agent: str = DEFAULT_UA,
                 window: str = "1440,900", profile: str | None = None,
                 binary: str = "google-chrome"):
        self.port = port
        self._args = [
            binary, "--headless=new", "--disable-gpu", "--no-first-run",
            f"--window-size={window}", f"--user-agent={user_agent}",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile or f'/tmp/cdp-profile-{port}'}",
            "about:blank",
        ]
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> "Browser":
        self._process = subprocess.Popen(
            self._args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(80):
            try:
                self._http("/json/version")
                return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("헤드리스 Chrome 이 뜨지 않았습니다")

    def __exit__(self, *_exc) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _http(self, path: str):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=10) as response:
            return json.load(response)

    def page(self, timeout: float = 30.0) -> Connection:
        """첫 탭에 붙는다. /json/new 는 요즘 Chrome 에서 PUT 만 받는다."""
        for _ in range(20):
            pages = [t for t in self._http("/json/list")
                     if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            if pages:
                return Connection(pages[0]["webSocketDebuggerUrl"], timeout)
            time.sleep(0.25)
        raise RuntimeError("붙을 탭이 없습니다")
