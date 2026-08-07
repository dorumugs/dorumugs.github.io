"""KREAM 포켓몬 카드 시세표(원화)를 받아 저장한다.

    python3 scripts/collect_kream.py                  # 브라우저로 직접 받기
    python3 scripts/collect_kream.py --from-file x.json   # 받아둔 응답 넣기

왜 브라우저를 띄우는가
----------------------
시세는 `api.kream.co.kr/api/data_mart/serving/trading_pokemon_card` 하나에만
있다. 페이지의 프리렌더 페이로드는 비어 있어서 HTML 만 받아서는 못 얻는다.
그 API 는 요청마다 페이지 JS 가 만드는 서명 헤더를 요구하므로, 서명을
흉내내는 대신 **페이지를 그대로 열어 그 결과를 구경한다.**

실측으로 알아낸 것 (2026-08-08)
  · curl 은 전부 500 이다. Chrome 이 아닌 TLS 지문을 걷어낸다.
  · 헤드리스도 기본 UA('HeadlessChrome')면 500 이다. 평범한 Chrome UA 를
    주면 통과한다. 관문은 UA 하나였다.
  · 응답을 Fetch 도메인으로 붙잡으면 요청이 지연돼 서명이 만료되고 500 이
    된다. 그래서 가로채지 않고, 페이지 JS 보다 먼저 fetch 를 감싸 응답을
    페이지 안에 쟁여 두었다가 나중에 꺼내온다.
  · 서버가 간헐적으로 500 을 낸다. 그때는 KREAM 자기 화면도 오류를 띄운다.
    실패하면 **기존 파일을 건드리지 않고** 물러난다. 마지막 정상 스냅샷이
    남는 편이 빈 화면보다 낫다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cdp  # noqa: E402
import kream_api  # noqa: E402
import rtms  # noqa: E402

DATA_DIR = ROOT / "data" / "pokemon"
KREAM_FILE = DATA_DIR / "kream.json.gz"

CHART_URL = "https://content.kream.co.kr/pokemon-tcg-chart"
WANTED = "trading_pokemon_card"

# 페이지 JS 가 돌기 전에 심는다. fetch 를 감싸 응답 본문만 챙겨 둔다.
# 요청을 붙잡거나 늦추지 않는다 — 늦추면 서명이 만료된다.
HOOK = """
(function () {
  window.__kream = null;
  var want = '%s';
  function keep(text) { if (text && text.length > 1000) { window.__kream = text; } }
  var origFetch = window.fetch;
  window.fetch = function () {
    var args = arguments;
    // window 에 고정해서 부른다. 페이지가 `fetch(...)` 로 그냥 부르면 strict
    // mode 에서 this 가 undefined 라 'Illegal invocation' 으로 요청이 죽는다.
    return origFetch.apply(window, args).then(function (res) {
      try {
        var url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '';
        if (url.indexOf(want) >= 0 && res.ok) {
          res.clone().text().then(keep);
        }
      } catch (e) {}
      return res;
    });
  };
  var origOpen = XMLHttpRequest.prototype.open;
  var origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url) {
    this.__url = url;
    return origOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function () {
    var xhr = this;
    xhr.addEventListener('load', function () {
      try {
        if (String(xhr.__url).indexOf(want) >= 0 && xhr.status === 200) {
          keep(xhr.responseText);
        }
      } catch (e) {}
    });
    return origSend.apply(this, arguments);
  };
})();
""" % WANTED


def capture_once(page, wait_seconds: int = 60) -> str:
    """페이지를 열고 응답이 쟁여지기를 기다린다. 못 받으면 빈 문자열."""
    page.call("Page.navigate", {"url": CHART_URL})
    for _ in range(wait_seconds):
        time.sleep(1)
        try:
            size = page.evaluate("window.__kream ? window.__kream.length : 0") or 0
        except Exception:
            size = 0
        if size:
            return page.read_big_string("window.__kream")
    return ""


def page_message(page) -> str:
    try:
        text = page.evaluate("document.body.innerText.slice(0,160)") or ""
    except Exception:
        return ""
    return " ".join(text.split())


def capture(attempts: int = 3, gap_seconds: int = 45, port: int = 9222) -> dict | None:
    """브라우저로 시세를 받아온다. 서버가 간헐적으로 죽어 몇 번 다시 시도한다."""
    with cdp.Browser(port=port) as browser:
        page = browser.page()
        page.call("Page.enable")
        page.call("Runtime.enable")
        page.call("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})

        for attempt in range(1, attempts + 1):
            text = capture_once(page)
            if text:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as err:
                    print(f"  시도 {attempt}: JSON 이 아닙니다 ({err})", file=sys.stderr)
                    payload = None
                if payload is not None:
                    if kream_api.is_valid(payload):
                        return payload
                    print(f"  시도 {attempt}: 응답이 비어 있습니다", file=sys.stderr)
            else:
                print(f"  시도 {attempt}: 못 받았습니다 — {page_message(page)}",
                      file=sys.stderr)
            if attempt < attempts:
                time.sleep(gap_seconds)
    return None


def save(payload: dict) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    KREAM_FILE.write_bytes(rtms.gzip_bytes(text))
    dates = kream_api.history_dates(payload)
    market = kream_api.parse_market(payload)
    products = payload.get("product", {}).get("rows", [])
    print(f"상품 {len(products):,}종 · 시장 {len(market)}일 "
          f"({dates[0] if dates else '?'} ~ {dates[-1] if dates else '?'}) "
          f"· 버전 {payload.get('version', '?')}")
    print(f"{KREAM_FILE} ({KREAM_FILE.stat().st_size / 1024:.0f}KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path,
                        help="브라우저에서 받아둔 응답 JSON. 서버가 죽었을 때 쓴다")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--gap", type=int, default=45, help="재시도 간격(초)")
    parser.add_argument("--port", type=int, default=9222)
    args = parser.parse_args()

    if args.from_file:
        payload = json.loads(args.from_file.read_text(encoding="utf-8"))
        if not kream_api.is_valid(payload):
            print(f"{args.from_file} 은 시세 응답이 아닙니다.", file=sys.stderr)
            return 1
        return save(payload)

    payload = capture(attempts=args.attempts, gap_seconds=args.gap, port=args.port)
    if payload is None:
        kept = "기존 파일을 그대로 둡니다." if KREAM_FILE.exists() else "저장된 것이 없습니다."
        print(f"KREAM 시세를 받지 못했습니다. {kept}", file=sys.stderr)
        return 1
    return save(payload)


if __name__ == "__main__":
    raise SystemExit(main())
