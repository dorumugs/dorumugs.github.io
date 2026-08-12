"""집계 산출물이 얼마나 낡았는지 판정한다 — 순수 함수만. I/O 없음.

이 저장소의 대시보드는 전부 크론이 매일(학군은 매월) 채워 준다. 그래서 가장
위험한 고장은 "에러가 난다" 가 아니라 **아무 일도 안 일어나는 것**이다.
크론이 등록에서 빠지거나, 수집원 화면이 바뀌어 파서가 빈 결과를 내면, 어제
파일이 그대로 남아 오늘 값인 척한다. 개수를 세는 검사는 이걸 절대 못 잡는다 —
개수는 그대로이기 때문이다. 실제로 ETF 대시보드가 크론 등록이 빠진 채로
"매일 갱신" 을 자처하고 있었다.

**빌드 날짜와 데이터 날짜는 다르다.** 이게 이 모듈의 핵심이다.

  generated(빌드 날짜)  집계 스크립트가 돈 날. 크론이 멈추면 이게 굳는다.
  months[-1]·latest_month(데이터 날짜)  실제로 담긴 자료가 어디까지인지.

수집이 깨져도 **집계는 옛 원본으로 성공한다.** 그러면 generated 는 오늘로
갱신되고 데이터는 지난달에 멈춘다 — generated 만 보면 이 고장을 못 잡는다.
그래서 둘을 따로 본다. 데이터 날짜가 없는 산출물은 없다고 표시한다(있는
척하면 검사했다는 착각만 남는다).

판정은 두 군데서 같은 규칙으로 쓴다.

  크론   집계 뒤에 check_freshness.py 가 불러 종료코드를 정한다
  화면   각 대시보드 JS 가 같은 임계값으로 배너를 띄운다

화면 쪽이 특히 중요하다. 크론이 아예 안 돌면 로그조차 안 생기므로, 낡았다는
사실을 알릴 수 있는 건 페이지뿐이다. 그래서 화면은 나이를 **보는 사람의
시계로** 잰다 — 빌드 때 계산해 박아 두면 크론이 멈추는 순간 그 숫자도 같이
멈춰서 낡음 자체가 안 보이게 된다.
"""

from __future__ import annotations

from datetime import date, datetime

# 빌드 날짜의 허용 나이(일). 갱신 주기에 여유를 더한 값이다.
#
#   매일 도는 것   주말·공휴일에 원본이 안 바뀌는 경우가 있어 3일
#   매월 도는 것   학군은 매월 3일에 한 번이라 한 달 + 여유로 40일
#
# 여유를 너무 크게 잡으면 경보가 늦고, 너무 좁으면 멀쩡한 날에 울려서 아무도
# 안 보게 된다. 후자가 더 위험하다.
BUILD_LIMITS: dict[str, int] = {
    "trades": 3,
    "schools": 40,
    "redev": 3,
    "pokemon": 3,
}

# 데이터 날짜(YYYY-MM)가 이번 달로부터 몇 달까지 뒤처져도 되나.
#
# 1 로 둔 이유: 국토부 실거래는 신고 기한이 있어 달이 바뀐 직후에는 지난달이
# 아직 최신일 수 있다. 2달 이상 뒤처지면 그건 수집이 멈춘 것이다.
MONTH_LAG_LIMITS: dict[str, int] = {
    "trades": 1,
    "redev": 1,
}

# 영어 월 이름. 포켓몬 환율이 "11 Aug 2026" 으로 온다. %b 로 파싱하면 크론의
# LC_TIME 에 따라 조용히 실패하므로 직접 매핑한다.
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def days_since(iso: str, today: date) -> int | None:
    """ISO 날짜(YYYY-MM-DD 또는 YYYYMMDD)가 며칠 지났나. 못 읽으면 None."""
    text = (iso or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return (today - datetime.strptime(text, fmt).date()).days
        except ValueError:
            continue
    return None


def days_since_english(text: str, today: date) -> int | None:
    """'11 Aug 2026' 처럼 영어 월 이름이 낀 날짜. 로케일에 안 기댄다."""
    parts = (text or "").replace(",", " ").split()
    if len(parts) != 3:
        return None
    day, month, year = parts
    index = _MONTHS.get(month[:3].lower())
    if index is None:
        return None
    try:
        return (today - date(int(year), index, int(day))).days
    except ValueError:
        return None


def months_behind(month: str, today: date) -> int | None:
    """YYYY-MM 이 이번 달로부터 몇 달 뒤처졌나. 못 읽으면 None."""
    try:
        year, mon = (int(x) for x in (month or "").strip().split("-"))
    except (ValueError, TypeError):
        return None
    if not 1 <= mon <= 12:
        return None
    return (today.year - year) * 12 + (today.month - mon)


def is_stale(iso: str, limit_days: int, today: date) -> bool:
    """허용 나이를 넘겼나.

    **날짜를 못 읽으면 낡은 것으로 본다.** 빠진 값을 '최신' 으로 처리하면
    필드 이름이 바뀌는 순간 검사기가 통째로 무력화되는데, 그게 바로 이 모듈이
    막으려는 조용한 붕괴다.
    """
    age = days_since(iso, today)
    return age is None or age > limit_days


def is_month_stale(month: str, limit_months: int, today: date) -> bool:
    """데이터 달이 너무 뒤처졌나. 못 읽으면 낡은 것으로 본다."""
    lag = months_behind(month, today)
    return lag is None or lag > limit_months
