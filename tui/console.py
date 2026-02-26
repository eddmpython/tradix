"""
Rich console output for backtest results with multiple display styles.

Renders styled metric cards, trade tables, sparklines, monthly heatmaps,
strategy DNA gauges, health dashboards, and black swan defense reports.
Supports modern (TradingView), bloomberg (dense), and minimal (hedge fund) styles.

백테스트 결과를 위한 Rich 콘솔 출력 (다중 스타일 지원).
메트릭 카드, 거래 테이블, 스파크라인, 월별 히트맵, 전략 DNA 게이지,
건강 대시보드, 블랙스완 방어 보고서를 렌더링합니다.
modern(TradingView), bloomberg(고밀도), minimal(헤지펀드) 스타일을 지원합니다.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os
import sys
import math

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout
from rich.rule import Rule
from rich import box

console = Console()

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_SPARK_FALLBACK = " .:;|"
_BAR_FULL = "\u2588"
_BAR_EMPTY = "\u2591"
_BAR_MED = "\u2593"


def _canUnicode() -> bool:
    """
    Check if the terminal supports Unicode sparkline characters.
    터미널이 유니코드 스파크라인 문자를 지원하는지 확인합니다.
    """
    try:
        "\u2581".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _sparkline(values: list, width: int = 20) -> str:
    """
    Generate a Unicode sparkline string from a list of numeric values.
    숫자 리스트로부터 유니코드 스파크라인 문자열을 생성합니다.

    Args:
        values: List of numeric values to visualize. 시각화할 숫자 리스트.
        width: Maximum character width of the sparkline. 스파크라인의 최대 문자 폭.

    Returns:
        str: Unicode sparkline string, or empty string if insufficient data.
            유니코드 스파크라인 문자열, 데이터 부족 시 빈 문자열.
    """
    if not values or len(values) < 2:
        return ""
    chars = _SPARK if _canUnicode() else _SPARK_FALLBACK
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    mxIdx = len(chars) - 1
    return "".join(chars[min(int((v - mn) / rng * mxIdx), mxIdx)] for v in sampled)


def _cv(value: float, fmt: str = "+.2f", suffix: str = "%") -> Text:
    """
    Color a numeric value green if positive, red if negative.
    양수면 녹색, 음수면 빨간색으로 숫자 값을 색칠합니다.

    Args:
        value: Numeric value to format. 포맷할 숫자 값.
        fmt: Python format spec string. 포맷 명세 문자열.
        suffix: String appended after the formatted number. 숫자 뒤에 붙는 접미사.

    Returns:
        Text: Rich Text object with appropriate color styling.
            적절한 색상이 적용된 Rich Text 객체.
    """
    text = f"{value:{fmt}}{suffix}"
    if value > 0:
        return Text(text, style="bold green")
    elif value < 0:
        return Text(text, style="bold red")
    return Text(text, style="dim")


def _cm(value: float, fmt: str = ".2f", suffix: str = "", threshold: float = 0) -> Text:
    """
    Color a metric green if above threshold, red if below.
    임계값 초과 시 녹색, 미만 시 빨간색으로 지표를 색칠합니다.

    Args:
        value: Metric value to format. 포맷할 지표 값.
        fmt: Python format spec string. 포맷 명세 문자열.
        suffix: String appended after the formatted number. 접미사.
        threshold: Boundary value for color decision. 색상 결정을 위한 경계값.

    Returns:
        Text: Rich Text object with appropriate color styling.
            적절한 색상이 적용된 Rich Text 객체.
    """
    text = f"{value:{fmt}}{suffix}"
    if value > threshold:
        return Text(text, style="green")
    elif value < threshold:
        return Text(text, style="red")
    return Text(text, style="dim")


def _metric(r: Any, name: str, default: float = 0.0) -> float:
    """
    Extract a metric value from a result object by attribute or metrics dict.
    결과 객체의 속성 또는 metrics 딕셔너리에서 지표 값을 추출합니다.

    Args:
        r: Result object (BacktestResult, VectorizedResult, etc.).
            결과 객체.
        name: Metric name to look up. 조회할 지표 이름.
        default: Fallback value if not found. 미발견 시 기본값.

    Returns:
        float: Extracted metric value or default. 추출된 지표 값 또는 기본값.
    """
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    m = getattr(r, "metrics", {})
    return float(m.get(name, default)) if isinstance(m, dict) else default


def _unwrap(result: Any) -> Any:
    """
    Unwrap a result proxy to its underlying result object.
    결과 프록시를 내부 결과 객체로 언래핑합니다.

    Args:
        result: Possibly wrapped result object. 래핑된 결과 객체.

    Returns:
        Any: The underlying result object. 내부 결과 객체.
    """
    if hasattr(result, "_result"):
        return result._result
    return result


def _equityData(r: Any) -> list:
    """
    Extract equity curve as a plain Python list from a result object.
    결과 객체에서 자산 곡선을 Python 리스트로 추출합니다.

    Args:
        r: Result object containing equityCurve attribute. equityCurve 속성을 포함하는 결과 객체.

    Returns:
        list: Equity values as a list of floats. float 리스트로 된 자산 값.
    """
    ec = getattr(r, "equityCurve", None)
    if ec is None:
        return []
    try:
        import numpy as np
        if isinstance(ec, np.ndarray):
            return ec.tolist()
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(ec, pd.Series):
            return ec.tolist()
    except ImportError:
        pass
    if isinstance(ec, list):
        return ec
    return list(ec) if hasattr(ec, "__iter__") else []


def _horizontalBar(value: float, maxVal: float, width: int = 20) -> str:
    """
    Render a Unicode horizontal bar gauge.
    유니코드 수평 바 게이지를 렌더링합니다.

    Args:
        value: Current value to represent. 표현할 현재 값.
        maxVal: Maximum value for the scale. 스케일의 최대 값.
        width: Total character width of the bar. 바의 총 문자 폭.

    Returns:
        str: String of filled and empty block characters representing the ratio.
            비율을 나타내는 채움/빈 블록 문자 문자열.
    """
    if maxVal <= 0:
        return _BAR_EMPTY * width
    ratio = max(0.0, min(1.0, value / maxVal))
    filled = int(ratio * width)
    return _BAR_FULL * filled + _BAR_EMPTY * (width - filled)


def _terminalWidth() -> str:
    """
    Classify the terminal width into one of three categories.
    터미널 폭을 세 가지 범주 중 하나로 분류합니다.

    Returns:
        str: 'wide' (>=120 cols), 'standard' (>=80 cols), or 'narrow' (<80 cols).
            'wide' (>=120열), 'standard' (>=80열), 또는 'narrow' (<80열).
    """
    try:
        cols = os.get_terminal_size().columns
    except (OSError, ValueError):
        cols = 80
    if cols >= 120:
        return "wide"
    elif cols >= 80:
        return "standard"
    return "narrow"


def _monthlyReturns(r: Any) -> Dict[int, Dict[int, float]]:
    """
    Extract monthly returns from the equity curve, grouped by year and month.
    자산 곡선에서 월별 수익률을 추출하여 연도 및 월별로 그룹화합니다.

    Args:
        r: Result object with equityCurve attribute. equityCurve 속성을 가진 결과 객체.

    Returns:
        Dict[int, Dict[int, float]]: Nested dict {year: {month: return_pct}}.
            중첩 딕셔너리 {연도: {월: 수익률_퍼센트}}.
    """
    ec = getattr(r, "equityCurve", None)
    if ec is None:
        return {}
    try:
        import pandas as pd
        if not isinstance(ec, pd.Series):
            return {}
        if len(ec) < 2:
            return {}
        monthly = ec.resample("ME").last()
        returns = monthly.pct_change().dropna()
        result = {}
        for dt, val in returns.items():
            year = dt.year
            month = dt.month
            if year not in result:
                result[year] = {}
            result[year][month] = float(val) * 100.0
        return result
    except Exception:
        return {}


def _drawdownData(r: Any) -> list:
    """
    Compute drawdown series as a plain list from the equity curve.
    자산 곡선으로부터 낙폭 시리즈를 플레인 리스트로 계산합니다.

    Args:
        r: Result object with equityCurve attribute. equityCurve 속성을 가진 결과 객체.

    Returns:
        list: Drawdown percentages as a list. 낙폭 퍼센트 리스트.
    """
    ec = getattr(r, "equityCurve", None)
    if ec is None:
        return []
    try:
        import pandas as pd
        if isinstance(ec, pd.Series) and len(ec) > 1:
            cumMax = ec.cummax()
            dd = ((ec - cumMax) / cumMax * 100)
            return dd.tolist()
    except Exception:
        pass
    return []


def _formatDate(dateStr: Any) -> str:
    """
    Format a date value to YYYY-MM-DD string.
    날짜 값을 YYYY-MM-DD 문자열로 포맷합니다.

    Args:
        dateStr: Date string, datetime, or Timestamp. 날짜 문자열, datetime, 또는 Timestamp.

    Returns:
        str: Formatted date string. 포맷된 날짜 문자열.
    """
    s = str(dateStr)[:10]
    return s if len(s) >= 8 else s


def _qualityIndicator(value: float, threshold: float, reverse: bool = False) -> str:
    """
    Return a text quality indicator (^ for good, v for bad).
    텍스트 품질 지표를 반환합니다 (^ 좋음, v 나쁨).

    Args:
        value: Metric value to evaluate. 평가할 지표 값.
        threshold: Boundary for good/bad. 좋음/나쁨 경계.
        reverse: If True, lower is better. True면 낮을수록 좋음.

    Returns:
        str: '^' for good, 'v' for bad, '-' for neutral. 좋음/나쁨/중립 지표.
    """
    if reverse:
        return "^" if value <= threshold else "v"
    return "^" if value >= threshold else "v"


def _colorBarText(value: float, maxVal: float, width: int = 16) -> Text:
    """
    Render a colored horizontal bar as Rich Text.
    색상이 적용된 수평 바를 Rich Text로 렌더링합니다.

    Args:
        value: Current value. 현재 값.
        maxVal: Maximum value for full bar. 전체 바에 해당하는 최대 값.
        width: Bar width in characters. 바의 문자 폭.

    Returns:
        Text: Rich Text object with colored bar characters. 색상 바 문자가 포함된 Rich Text.
    """
    ratio = max(0.0, min(1.0, value / maxVal)) if maxVal > 0 else 0.0
    filled = int(ratio * width)
    empty = width - filled

    if ratio >= 0.7:
        style = "green"
    elif ratio >= 0.4:
        style = "yellow"
    else:
        style = "red"

    t = Text()
    t.append(_BAR_FULL * filled, style=style)
    t.append(_BAR_EMPTY * empty, style="dim")
    return t


def _heatmapColor(value: float, minClamp: float = -5.0, maxClamp: float = 5.0) -> str:
    """
    Map a return percentage to an RGB background color for heatmap rendering.
    수익률 퍼센트를 히트맵 렌더링용 RGB 배경색으로 매핑합니다.

    Uses a gradient from red (negative) through a muted neutral (zero) to green (positive).

    Args:
        value: Return percentage value. 수익률 퍼센트 값.
        minClamp: Minimum value clamp for color scaling. 색상 스케일링의 최소 클램프.
        maxClamp: Maximum value clamp for color scaling. 색상 스케일링의 최대 클램프.

    Returns:
        str: Rich style string with rgb background. rgb 배경이 포함된 Rich 스타일 문자열.
    """
    clamped = max(minClamp, min(maxClamp, value))

    if clamped >= 0:
        ratio = clamped / maxClamp if maxClamp != 0 else 0
        r = int(200 - ratio * 160)
        g = int(200 + ratio * 55)
        b = int(200 - ratio * 160)
    else:
        ratio = abs(clamped) / abs(minClamp) if minClamp != 0 else 0
        r = int(200 + ratio * 55)
        g = int(200 - ratio * 160)
        b = int(200 - ratio * 160)

    return f"on rgb({r},{g},{b})"


def _l(ko: bool, en: str, kr: str) -> str:
    """
    Choose between English and Korean label.
    영어와 한국어 레이블 중 선택합니다.

    Args:
        ko: If True, use Korean. True면 한국어 사용.
        en: English label. 영어 레이블.
        kr: Korean label. 한국어 레이블.

    Returns:
        str: Selected label string. 선택된 레이블 문자열.
    """
    return kr if ko else en


def printResult(result: Any, lang: str = "en", style: str = "modern") -> None:
    """
    Print backtest result summary to the console in the chosen visual style.
    선택된 시각 스타일로 백테스트 결과 요약을 콘솔에 출력합니다.

    Supports three display styles:
    - 'modern': TradingView-inspired cards and detail table.
    - 'bloomberg': Dense, information-rich, 4-quadrant layout with monthly heatmap.
    - 'minimal': Clean, border-free, hedge fund report style.

    세 가지 출력 스타일을 지원합니다:
    - 'modern': TradingView 스타일 카드 및 상세 테이블.
    - 'bloomberg': 고밀도 4분면 레이아웃 + 월별 히트맵.
    - 'minimal': 깔끔한 무테두리 헤지펀드 보고서 스타일.

    Args:
        result: BacktestResult, VectorizedResult, or EasyResult object.
            BacktestResult, VectorizedResult, 또는 EasyResult 객체.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
        style: Display style ('modern', 'bloomberg', or 'minimal').
            표시 스타일 ('modern', 'bloomberg', 또는 'minimal').
    """
    r = _unwrap(result)
    ko = lang == "ko"

    strategy = getattr(r, "strategy", "Strategy")
    symbol = getattr(r, "symbol", "")
    start = _formatDate(getattr(r, "startDate", ""))
    end = _formatDate(getattr(r, "endDate", ""))
    initial = getattr(r, "initialCash", 0)
    final = getattr(r, "finalEquity", 0)
    totalRet = getattr(r, "totalReturn", 0)
    annual = _metric(r, "annualReturn")
    vol = _metric(r, "volatility")
    sharpe = _metric(r, "sharpeRatio")
    sortino = _metric(r, "sortinoRatio")
    calmar = _metric(r, "calmarRatio")
    mdd = _metric(r, "maxDrawdown")
    mddDur = _metric(r, "maxDrawdownDuration", 0)
    pf = _metric(r, "profitFactor")
    trades = getattr(r, "totalTrades", 0)
    wr = getattr(r, "winRate", 0)
    avgW = _metric(r, "avgWin")
    avgL = _metric(r, "avgLoss")
    avgWPct = _metric(r, "avgWinPercent")
    avgLPct = _metric(r, "avgLossPercent")
    expectancy = _metric(r, "expectancy")
    avgHold = _metric(r, "avgHoldingDays")
    maxConW = _metric(r, "maxConsecutiveWins", 0)
    maxConL = _metric(r, "maxConsecutiveLosses", 0)
    winningTrades = _metric(r, "winningTrades", 0)
    losingTrades = _metric(r, "losingTrades", 0)
    bestMonth = _metric(r, "bestMonth")
    worstMonth = _metric(r, "worstMonth")

    eq = _equityData(r)
    dd = _drawdownData(r)

    if style == "bloomberg":
        _printResultBloomberg(
            r, ko, strategy, symbol, start, end, initial, final,
            totalRet, annual, vol, sharpe, sortino, calmar,
            mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
            expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
            bestMonth, worstMonth, eq, dd,
        )
    elif style == "minimal":
        _printResultMinimal(
            r, ko, strategy, symbol, start, end, initial, final,
            totalRet, annual, vol, sharpe, sortino, calmar,
            mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
            expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
            bestMonth, worstMonth, eq, dd,
        )
    else:
        _printResultModern(
            r, ko, strategy, symbol, start, end, initial, final,
            totalRet, annual, vol, sharpe, sortino, calmar,
            mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
            expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
            bestMonth, worstMonth, eq, dd,
        )


def _printResultModern(
    r, ko, strategy, symbol, start, end, initial, final,
    totalRet, annual, vol, sharpe, sortino, calmar,
    mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
    expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
    bestMonth, worstMonth, eq, dd,
) -> None:
    """
    Render the modern (TradingView) style backtest result.
    모던(TradingView) 스타일 백테스트 결과를 렌더링합니다.
    """
    spark = _sparkline(eq, width=30) if eq else ""

    hdr = Text()
    hdr.append(f" {strategy} ", style="bold white on blue")
    hdr.append(f"  {symbol}", style="bold cyan")
    hdr.append(f"  {start} ~ {end}", style="dim")
    if spark:
        hdr.append(f"  {spark}", style="cyan")

    console.print()
    console.print(Panel(
        hdr,
        title="[bold]TRADEX[/bold]",
        subtitle=f"{initial:,.0f} \u2192 {final:,.0f}",
        border_style="blue",
    ))

    cards = []

    def _card(label: str, value: Text, width: int = 18) -> Panel:
        content = Text()
        content.append(f"{label}\n", style="dim")
        content.append_text(value)
        return Panel(content, width=width, border_style="dim", padding=(0, 1))

    cards.append(_card(
        _l(ko, "Return", "\uc218\uc775\ub960"),
        _cv(totalRet),
    ))
    cards.append(_card(
        _l(ko, "Sharpe", "\uc0e4\ud504"),
        _cm(sharpe, threshold=1.0),
    ))
    cards.append(_card(
        _l(ko, "MDD", "\ucd5c\ub300\ub099\ud3ed"),
        _cv(mdd, fmt=".2f"),
    ))
    cards.append(_card(
        _l(ko, "Win Rate", "\uc2b9\ub960"),
        _cm(wr, suffix="%", threshold=50),
    ))
    cards.append(_card(
        _l(ko, "Trades", "\uac70\ub798\uc218"),
        Text(str(trades), style="bold white"),
    ))

    console.print(Columns(cards, padding=(0, 0), equal=True))

    wrBarStr = _horizontalBar(wr, 100.0, width=16)
    wrBar = Text()
    wrBar.append(f"{wr:.1f}% ", style="green" if wr >= 50 else "red")
    wrBar.append(wrBarStr, style="green" if wr >= 50 else "red")

    detail = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    detail.add_column(min_width=18)
    detail.add_column(justify="right", min_width=14)
    detail.add_column(min_width=18)
    detail.add_column(justify="right", min_width=14)

    detail.add_row(
        _l(ko, "Initial", "\ucd08\uae30\uc790\uae08"),
        Text(f"{initial:,.0f}", style="white"),
        _l(ko, "Annual Return", "\uc5f0\uc218\uc775"),
        _cv(annual),
    )
    detail.add_row(
        _l(ko, "Final", "\ucd5c\uc885\uc790\uc0b0"),
        Text(f"{final:,.0f}", style="bold white"),
        _l(ko, "Volatility", "\ubcc0\ub3d9\uc131"),
        _cm(vol, suffix="%"),
    )
    detail.add_row(
        _l(ko, "Sharpe Ratio", "\uc0e4\ud504 \ube44\uc728"),
        _cm(sharpe, threshold=1.0),
        _l(ko, "Sortino Ratio", "\uc18c\ub974\ud2f0\ub178 \ube44\uc728"),
        _cm(sortino, threshold=1.0),
    )
    detail.add_row(
        _l(ko, "Calmar Ratio", "\uce7c\ub9c8 \ube44\uc728"),
        _cm(calmar, threshold=1.0),
        _l(ko, "Profit Factor", "\uc190\uc775\ube44"),
        _cm(pf, threshold=1.0),
    )
    detail.add_row(
        _l(ko, "Max Drawdown", "\ucd5c\ub300\ub099\ud3ed"),
        _cv(mdd, fmt=".2f"),
        _l(ko, "MDD Duration", "\ub099\ud3ed\uae30\uac04"),
        Text(
            f"{int(mddDur)}{'d' if not ko else '일'}",
            style="dim",
        ),
    )
    detail.add_row(
        _l(ko, "Win Rate (bar)", "\uc2b9\ub960 (\ubc14)"),
        wrBar,
        _l(ko, "Avg Win", "\ud3c9\uade0\uc774\uc775"),
        Text(f"{avgW:,.0f}", style="green") if avgW else Text("-", style="dim"),
    )
    detail.add_row(
        _l(ko, "Avg Holding", "\ud3c9\uade0\ubcf4\uc720"),
        Text(
            f"{avgHold:.1f}{'d' if not ko else '일'}",
            style="white",
        ),
        _l(ko, "Avg Loss", "\ud3c9\uade0\uc190\uc2e4"),
        Text(f"{avgL:,.0f}", style="red") if avgL else Text("-", style="dim"),
    )
    detail.add_row(
        _l(ko, "Expectancy", "\uae30\ub300\uc218\uc775"),
        _cv(expectancy, fmt="+.4f", suffix="%"),
        _l(ko, "Best Month", "\ucd5c\uace0\uc6d4"),
        _cv(bestMonth) if bestMonth else Text("-", style="dim"),
    )
    detail.add_row(
        _l(ko, "Consec. Wins", "\uc5f0\uc18d\uc2b9"),
        Text(str(int(maxConW)), style="green"),
        _l(ko, "Worst Month", "\ucd5c\uc800\uc6d4"),
        _cv(worstMonth) if worstMonth else Text("-", style="dim"),
    )
    detail.add_row(
        _l(ko, "Consec. Losses", "\uc5f0\uc18d\ud328"),
        Text(str(int(maxConL)), style="red"),
        "", "",
    )

    console.print(detail)
    console.print()


def _printResultBloomberg(
    r, ko, strategy, symbol, start, end, initial, final,
    totalRet, annual, vol, sharpe, sortino, calmar,
    mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
    expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
    bestMonth, worstMonth, eq, dd,
) -> None:
    """
    Render the Bloomberg-style dense backtest result.
    블룸버그 스타일 고밀도 백테스트 결과를 렌더링합니다.
    """
    spark = _sparkline(eq, width=24) if eq else ""
    ddSpark = _sparkline(dd, width=24) if dd else ""

    try:
        import pandas as pd
        ecSeries = getattr(r, "equityCurve", None)
        if ecSeries is not None and isinstance(ecSeries, pd.Series) and len(ecSeries) > 20:
            returns = ecSeries.pct_change().dropna()
            from scipy import stats as scipyStats
            skewVal = float(scipyStats.skew(returns.values))
            kurtVal = float(scipyStats.kurtosis(returns.values, fisher=True))
            var95 = float(returns.quantile(0.05)) * 100
            kellyPct = 0.0
            if wr > 0 and avgWPct != 0 and avgLPct != 0:
                winProb = wr / 100.0
                lossProb = 1.0 - winProb
                if avgLPct != 0:
                    kellyPct = winProb - (lossProb / (abs(avgWPct) / abs(avgLPct))) if abs(avgLPct) > 0 else 0.0
                kellyPct = max(0.0, kellyPct) * 100
            ddArr = returns.rolling(20).std().dropna()
            ulcerIndex = 0.0
            if len(dd) > 2:
                import numpy as np
                ddNp = np.array(dd)
                ddSq = ddNp ** 2
                ulcerIndex = float(np.sqrt(np.mean(ddSq)))
        else:
            skewVal = 0.0
            kurtVal = 0.0
            var95 = 0.0
            kellyPct = 0.0
            ulcerIndex = 0.0
    except Exception:
        skewVal = 0.0
        kurtVal = 0.0
        var95 = 0.0
        kellyPct = 0.0
        ulcerIndex = 0.0

    title = Text()
    title.append(f" {strategy} ", style="bold white on dark_blue")
    title.append(f" {symbol} ", style="bold yellow")
    title.append(f" {start} \u2192 {end} ", style="dim white")

    console.print()

    perfTable = Table(
        box=None, show_header=True, padding=(0, 1),
        header_style="bold white",
    )
    perfTable.add_column(
        _l(ko, "PERFORMANCE", "\uc131\uacfc"), min_width=16,
    )
    perfTable.add_column("", justify="right", min_width=12)

    perfTable.add_row(
        _l(ko, "Total Return", "\ucd1d\uc218\uc775\ub960"), _cv(totalRet),
    )
    perfTable.add_row(
        _l(ko, "Annual Return", "\uc5f0\uc218\uc775\ub960"), _cv(annual),
    )
    perfTable.add_row(
        _l(ko, "Volatility", "\ubcc0\ub3d9\uc131"), _cm(vol, suffix="%"),
    )
    perfTable.add_row(
        _l(ko, "Best Month", "\ucd5c\uace0\uc6d4"), _cv(bestMonth),
    )
    perfTable.add_row(
        _l(ko, "Worst Month", "\ucd5c\uc800\uc6d4"), _cv(worstMonth),
    )
    if spark:
        perfTable.add_row(
            _l(ko, "Equity", "\uc790\uc0b0\uace1\uc120"),
            Text(spark, style="cyan"),
        )

    riskTable = Table(
        box=None, show_header=True, padding=(0, 1),
        header_style="bold white",
    )
    riskTable.add_column(
        _l(ko, "RISK-ADJUSTED", "\uc704\ud5d8\uc870\uc815"), min_width=16,
    )
    riskTable.add_column("", justify="right", min_width=12)

    riskTable.add_row(
        _l(ko, "Sharpe", "\uc0e4\ud504"), _cm(sharpe, threshold=1.0),
    )
    riskTable.add_row(
        _l(ko, "Sortino", "\uc18c\ub974\ud2f0\ub178"), _cm(sortino, threshold=1.0),
    )
    riskTable.add_row(
        _l(ko, "Calmar", "\uce7c\ub9c8"), _cm(calmar, threshold=1.0),
    )
    riskTable.add_row(
        _l(ko, "VaR (95%)", "VaR (95%)"),
        _cv(var95, fmt=".3f"),
    )
    riskTable.add_row(
        _l(ko, "Skewness", "\uc654\ub3c4"),
        _cm(skewVal, fmt=".3f", threshold=0),
    )
    riskTable.add_row(
        _l(ko, "Kurtosis", "\ucca8\ub3c4"),
        _cm(kurtVal, fmt=".2f", threshold=3),
    )

    ddTable = Table(
        box=None, show_header=True, padding=(0, 1),
        header_style="bold white",
    )
    ddTable.add_column(
        _l(ko, "DRAWDOWN", "\ub099\ud3ed"), min_width=16,
    )
    ddTable.add_column("", justify="right", min_width=12)

    ddTable.add_row(
        _l(ko, "Max Drawdown", "\ucd5c\ub300\ub099\ud3ed"), _cv(mdd, fmt=".2f"),
    )
    ddTable.add_row(
        _l(ko, "MDD Duration", "\ub099\ud3ed\uae30\uac04"),
        Text(f"{int(mddDur)}{'d' if not ko else '일'}", style="dim"),
    )
    ddTable.add_row(
        _l(ko, "Ulcer Index", "\uc5bc\uc11c \uc9c0\uc218"),
        _cm(ulcerIndex, fmt=".3f", threshold=5),
    )
    ddTable.add_row(
        _l(ko, "Kelly %", "\ucf08\ub9ac %"),
        _cm(kellyPct, fmt=".1f", suffix="%", threshold=0),
    )
    if ddSpark:
        ddTable.add_row(
            _l(ko, "DD Curve", "\ub099\ud3ed\uace1\uc120"),
            Text(ddSpark, style="red"),
        )

    tradeTable = Table(
        box=None, show_header=True, padding=(0, 1),
        header_style="bold white",
    )
    tradeTable.add_column(
        _l(ko, "TRADE STATS", "\uac70\ub798\ud1b5\uacc4"), min_width=16,
    )
    tradeTable.add_column("", justify="right", min_width=12)

    tradeTable.add_row(
        _l(ko, "Total Trades", "\ucd1d\uac70\ub798\uc218"),
        Text(str(trades), style="bold white"),
    )
    tradeTable.add_row(
        _l(ko, "Win Rate", "\uc2b9\ub960"),
        _cm(wr, suffix="%", threshold=50),
    )
    tradeTable.add_row(
        _l(ko, "Profit Factor", "\uc190\uc775\ube44"), _cm(pf, threshold=1.0),
    )
    tradeTable.add_row(
        _l(ko, "Expectancy", "\uae30\ub300\uc218\uc775"),
        _cv(expectancy, fmt="+.4f", suffix="%"),
    )
    tradeTable.add_row(
        _l(ko, "Avg Hold", "\ud3c9\uade0\ubcf4\uc720"),
        Text(f"{avgHold:.1f}{'d' if not ko else '일'}", style="white"),
    )
    tradeTable.add_row(
        _l(ko, "W/L Streak", "\uc5f0\uc18d \uc2b9/\ud328"),
        Text(f"{int(maxConW)}W / {int(maxConL)}L", style="white"),
    )

    topGrid = Table.grid(padding=(0, 2))
    topGrid.add_column()
    topGrid.add_column()
    topGrid.add_row(perfTable, riskTable)

    bottomGrid = Table.grid(padding=(0, 2))
    bottomGrid.add_column()
    bottomGrid.add_column()
    bottomGrid.add_row(ddTable, tradeTable)

    innerContent = Table.grid()
    innerContent.add_column()
    innerContent.add_row(title)
    innerContent.add_row(Text(""))
    innerContent.add_row(topGrid)
    innerContent.add_row(Text(""))
    innerContent.add_row(bottomGrid)

    monthData = _monthlyReturns(r)
    if monthData:
        innerContent.add_row(Text(""))
        heatmapTable = _buildMonthlyHeatmapTable(monthData, ko)
        innerContent.add_row(heatmapTable)

    console.print(Panel(
        innerContent,
        title="[bold white]TRADEX BLOOMBERG[/bold white]",
        subtitle=f"{initial:,.0f} \u2192 {final:,.0f}",
        border_style="blue",
        box=box.DOUBLE,
        padding=(1, 2),
    ))
    console.print()


def _printResultMinimal(
    r, ko, strategy, symbol, start, end, initial, final,
    totalRet, annual, vol, sharpe, sortino, calmar,
    mdd, mddDur, pf, trades, wr, avgW, avgL, avgWPct, avgLPct,
    expectancy, avgHold, maxConW, maxConL, winningTrades, losingTrades,
    bestMonth, worstMonth, eq, dd,
) -> None:
    """
    Render the minimal (hedge fund) style backtest result.
    미니멀(헤지펀드) 스타일 백테스트 결과를 렌더링합니다.
    """
    spark = _sparkline(eq, width=30) if eq else ""
    ddSpark = _sparkline(dd, width=30) if dd else ""

    console.print()
    console.print(Rule(
        f"[bold]{strategy}[/bold]  {symbol}  {start} ~ {end}",
        style="blue",
    ))
    console.print()

    grid = Table.grid(padding=(0, 4))
    grid.add_column(min_width=24)
    grid.add_column(min_width=24)

    qRet = _qualityIndicator(totalRet, 0)
    qSharpe = _qualityIndicator(sharpe, 1.0)
    qSortino = _qualityIndicator(sortino, 1.0)
    qCalmar = _qualityIndicator(calmar, 1.0)
    qMdd = _qualityIndicator(abs(mdd), 20, reverse=True)
    qWr = _qualityIndicator(wr, 50)
    qPf = _qualityIndicator(pf, 1.0)
    qVol = _qualityIndicator(vol, 30, reverse=True)

    def _minMetric(label: str, value: str, indicator: str) -> Text:
        t = Text()
        iStyle = "green" if indicator == "^" else ("red" if indicator == "v" else "dim")
        t.append(f"{indicator} ", style=iStyle)
        t.append(f"{label}: ", style="dim")
        t.append(value, style="white")
        return t

    grid.add_row(
        _minMetric(
            _l(ko, "Total Return", "\ucd1d\uc218\uc775\ub960"),
            f"{totalRet:+.2f}%", qRet,
        ),
        _minMetric(
            _l(ko, "Annual Return", "\uc5f0\uc218\uc775\ub960"),
            f"{annual:+.2f}%", qRet,
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Sharpe", "\uc0e4\ud504"),
            f"{sharpe:.2f}", qSharpe,
        ),
        _minMetric(
            _l(ko, "Sortino", "\uc18c\ub974\ud2f0\ub178"),
            f"{sortino:.2f}", qSortino,
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Calmar", "\uce7c\ub9c8"),
            f"{calmar:.2f}", qCalmar,
        ),
        _minMetric(
            _l(ko, "Volatility", "\ubcc0\ub3d9\uc131"),
            f"{vol:.2f}%", qVol,
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Max Drawdown", "\ucd5c\ub300\ub099\ud3ed"),
            f"{mdd:.2f}%", qMdd,
        ),
        _minMetric(
            _l(ko, "MDD Duration", "\ub099\ud3ed\uae30\uac04"),
            f"{int(mddDur)}{'d' if not ko else '일'}", "-",
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Win Rate", "\uc2b9\ub960"),
            f"{wr:.1f}%", qWr,
        ),
        _minMetric(
            _l(ko, "Profit Factor", "\uc190\uc775\ube44"),
            f"{pf:.2f}", qPf,
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Trades", "\uac70\ub798\uc218"),
            str(trades), "-",
        ),
        _minMetric(
            _l(ko, "Avg Holding", "\ud3c9\uade0\ubcf4\uc720"),
            f"{avgHold:.1f}{'d' if not ko else '일'}", "-",
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Expectancy", "\uae30\ub300\uc218\uc775"),
            f"{expectancy:+.4f}%", _qualityIndicator(expectancy, 0),
        ),
        _minMetric(
            _l(ko, "W/L Streak", "\uc5f0\uc18d \uc2b9/\ud328"),
            f"{int(maxConW)}W / {int(maxConL)}L", "-",
        ),
    )
    grid.add_row(
        _minMetric(
            _l(ko, "Initial", "\ucd08\uae30\uc790\uae08"),
            f"{initial:,.0f}", "-",
        ),
        _minMetric(
            _l(ko, "Final", "\ucd5c\uc885\uc790\uc0b0"),
            f"{final:,.0f}", "-",
        ),
    )

    console.print(grid)

    if spark:
        console.print()
        eqLine = Text()
        eqLine.append(
            _l(ko, "  Equity:    ", "  \uc790\uc0b0\uace1\uc120: "),
            style="dim",
        )
        eqLine.append(spark, style="cyan")
        console.print(eqLine)

    if ddSpark:
        ddLine = Text()
        ddLine.append(
            _l(ko, "  Drawdown:  ", "  \ub099\ud3ed\uace1\uc120: "),
            style="dim",
        )
        ddLine.append(ddSpark, style="red")
        console.print(ddLine)

    console.print()
    console.print(Rule(style="dim"))
    console.print()


def printComparison(results: List[Any], lang: str = "en") -> None:
    """
    Print a comparison table of multiple backtest strategy results.
    여러 백테스트 전략 결과의 비교 테이블을 출력합니다.

    Includes Sortino and Profit Factor columns, row highlighting for
    best/worst values per metric, and sparkline equity curves.

    소르티노, 손익비 열을 포함하며, 지표별 최상/최악 값을 강조 표시하고
    스파크라인 자산 곡선을 표시합니다.

    Args:
        results: List of result objects to compare. 비교할 결과 객체 리스트.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    ko = lang == "ko"

    if not results:
        console.print("[dim]No results to compare.[/dim]")
        return

    rows = []
    for result in results:
        r = _unwrap(result)
        eq = _equityData(r)
        sp = _sparkline(eq, width=15) if eq else ""
        rows.append({
            "strategy": getattr(r, "strategy", "?"),
            "totalReturn": getattr(r, "totalReturn", 0),
            "annual": _metric(r, "annualReturn"),
            "sharpe": _metric(r, "sharpeRatio"),
            "sortino": _metric(r, "sortinoRatio"),
            "mdd": _metric(r, "maxDrawdown"),
            "wr": getattr(r, "winRate", 0),
            "pf": _metric(r, "profitFactor"),
            "trades": getattr(r, "totalTrades", 0),
            "spark": sp,
        })

    bestReturn = max(row["totalReturn"] for row in rows)
    worstReturn = min(row["totalReturn"] for row in rows)
    bestSharpe = max(row["sharpe"] for row in rows)
    worstSharpe = min(row["sharpe"] for row in rows)
    bestSortino = max(row["sortino"] for row in rows)
    bestMdd = max(row["mdd"] for row in rows)
    worstMdd = min(row["mdd"] for row in rows)
    bestWr = max(row["wr"] for row in rows)
    bestPf = max(row["pf"] for row in rows)

    table = Table(
        title=_l(ko, "Strategy Comparison", "\uc804\ub7b5 \ube44\uad50"),
        box=box.ROUNDED,
        header_style="bold cyan",
        padding=(0, 1),
    )

    table.add_column(_l(ko, "Strategy", "\uc804\ub7b5"), style="bold white")
    table.add_column(_l(ko, "Return", "\uc218\uc775\ub960"), justify="right")
    table.add_column(_l(ko, "Annual", "\uc5f0\uc218\uc775"), justify="right")
    table.add_column(_l(ko, "Sharpe", "\uc0e4\ud504"), justify="right")
    table.add_column(_l(ko, "Sortino", "\uc18c\ub974\ud2f0\ub178"), justify="right")
    table.add_column("MDD", justify="right")
    table.add_column(_l(ko, "Win%", "\uc2b9\ub960"), justify="right")
    table.add_column(_l(ko, "P.Factor", "\uc190\uc775\ube44"), justify="right")
    table.add_column(_l(ko, "Trades", "\uac70\ub798"), justify="right")
    table.add_column(_l(ko, "Equity", "\uc790\uc0b0\uace1\uc120"), justify="left")

    for row in rows:
        retText = _cv(row["totalReturn"])
        if len(rows) > 1 and row["totalReturn"] == bestReturn:
            retText = Text(f"{row['totalReturn']:+.2f}%", style="bold green underline")
        elif len(rows) > 1 and row["totalReturn"] == worstReturn:
            retText = Text(f"{row['totalReturn']:+.2f}%", style="bold red underline")

        sharpeText = _cm(row["sharpe"], threshold=1.0)
        if len(rows) > 1 and row["sharpe"] == bestSharpe:
            sharpeText = Text(f"{row['sharpe']:.2f}", style="green underline")
        elif len(rows) > 1 and row["sharpe"] == worstSharpe:
            sharpeText = Text(f"{row['sharpe']:.2f}", style="red underline")

        sortinoText = _cm(row["sortino"], threshold=1.0)
        if len(rows) > 1 and row["sortino"] == bestSortino:
            sortinoText = Text(f"{row['sortino']:.2f}", style="green underline")

        mddText = _cv(row["mdd"], fmt=".2f")
        if len(rows) > 1 and row["mdd"] == bestMdd:
            mddText = Text(f"{row['mdd']:.2f}%", style="green underline")
        elif len(rows) > 1 and row["mdd"] == worstMdd:
            mddText = Text(f"{row['mdd']:.2f}%", style="red underline")

        wrText = _cm(row["wr"], suffix="%", threshold=50)
        if len(rows) > 1 and row["wr"] == bestWr:
            wrText = Text(f"{row['wr']:.2f}%", style="green underline")

        pfText = _cm(row["pf"], threshold=1.0)
        if len(rows) > 1 and row["pf"] == bestPf:
            pfText = Text(f"{row['pf']:.2f}", style="green underline")

        table.add_row(
            row["strategy"],
            retText,
            _cv(row["annual"]),
            sharpeText,
            sortinoText,
            mddText,
            wrText,
            pfText,
            str(row["trades"]),
            Text(row["spark"], style="cyan"),
        )

    console.print()
    console.print(table)
    console.print()


def printTrades(result: Any, limit: int = 20, lang: str = "en") -> None:
    """
    Print trade history table with holding period, colored rows, and summary.
    보유 기간, 색상 행, 요약이 포함된 거래 내역 테이블을 출력합니다.

    Winning trades are tinted green and losing trades are tinted red.
    A summary row at the bottom shows total P&L, average return, and win rate.

    수익 거래는 녹색 배경, 손실 거래는 빨간색 배경으로 표시됩니다.
    하단 요약 행에 총 손익, 평균 수익률, 승률을 표시합니다.

    Args:
        result: Result object containing trade history. 거래 내역이 포함된 결과 객체.
        limit: Maximum number of recent trades to display. 표시할 최근 거래 최대 수.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    r = _unwrap(result)
    tradeList = getattr(r, "trades", [])
    if not tradeList:
        console.print("[dim]No trades.[/dim]")
        return

    ko = lang == "ko"

    table = Table(
        title=_l(
            ko,
            f"Trade History (last {limit})",
            f"\uac70\ub798 \ub0b4\uc5ed (\ucd5c\uadfc {limit}\uac74)",
        ),
        box=box.SIMPLE,
        header_style="bold white",
        padding=(0, 1),
    )

    table.add_column("#", style="dim", justify="right")
    table.add_column(_l(ko, "Side", "\ubc29\ud5a5"), justify="center")
    table.add_column(_l(ko, "Entry", "\uc9c4\uc785\uc77c"))
    table.add_column(_l(ko, "Exit", "\uccad\uc0b0\uc77c"))
    table.add_column(_l(ko, "Entry$", "\uc9c4\uc785\uac00"), justify="right")
    table.add_column(_l(ko, "Exit$", "\uccad\uc0b0\uac00"), justify="right")
    table.add_column(_l(ko, "P&L", "\uc190\uc775"), justify="right")
    table.add_column(_l(ko, "Return", "\uc218\uc775\ub960"), justify="right")
    table.add_column(
        _l(ko, "Hold", "\ubcf4\uc720"),
        justify="right",
    )

    displayTrades = tradeList[-limit:]
    totalPnl = 0.0
    totalRetPct = 0.0
    winCount = 0
    tradeCount = 0

    for i, t in enumerate(displayTrades, 1):
        if isinstance(t, dict):
            side = t.get("side", "buy")
            ed = _formatDate(t.get("entryDate", ""))
            xd = _formatDate(t.get("exitDate", ""))
            ep = t.get("entryPrice", 0)
            xp = t.get("exitPrice", 0)
            pnl = t.get("pnl", xp - ep)
            ret = t.get("returnPct", t.get("pnlPercent", (xp / ep - 1) * 100 if ep else 0))
            holdDays = t.get("holdingDays", 0)
        else:
            side = getattr(t, "side", "buy")
            ed = _formatDate(getattr(t, "entryDate", ""))
            xd = _formatDate(getattr(t, "exitDate", ""))
            ep = getattr(t, "entryPrice", 0)
            xp = getattr(t, "exitPrice", 0)
            pnl = getattr(t, "pnl", xp - ep)
            ret = getattr(t, "returnPct", getattr(t, "pnlPercent", (xp / ep - 1) * 100 if ep else 0))
            holdDays = getattr(t, "holdingDays", 0)

        totalPnl += pnl
        totalRetPct += ret
        tradeCount += 1
        if pnl > 0:
            winCount += 1

        isBuy = "buy" in str(side).lower()
        sideText = Text("BUY" if isBuy else "SELL", style="green" if isBuy else "red")

        rowStyle = ""
        if pnl > 0:
            rowStyle = "on rgb(0,40,0)"
        elif pnl < 0:
            rowStyle = "on rgb(40,0,0)"

        holdStr = f"{holdDays}{'d' if not ko else '일'}" if holdDays else "-"

        table.add_row(
            str(i), sideText, ed, xd,
            f"{ep:,.0f}", f"{xp:,.0f}",
            _cv(pnl, fmt=",.0f", suffix=""),
            _cv(ret),
            Text(holdStr, style="dim"),
            style=rowStyle,
        )

    if tradeCount > 0:
        avgRetPct = totalRetPct / tradeCount
        tradeWr = winCount / tradeCount * 100

        table.add_row(
            "", "",
            Text(
                _l(ko, "SUMMARY", "\uc694\uc57d"),
                style="bold white",
            ),
            "", "", "",
            _cv(totalPnl, fmt=",.0f", suffix=""),
            _cv(avgRetPct, fmt="+.2f", suffix="% avg"),
            Text(
                f"{tradeWr:.0f}%{'win' if not ko else '승'}",
                style="bold green" if tradeWr >= 50 else "bold red",
            ),
            style="on rgb(30,30,50)",
        )

    console.print()
    console.print(table)
    console.print()


def _buildMonthlyHeatmapTable(monthData: Dict[int, Dict[int, float]], ko: bool) -> Table:
    """
    Build a Rich Table representing a monthly returns heatmap.
    월별 수익률 히트맵을 나타내는 Rich Table을 생성합니다.

    Rows are years, columns are months (Jan-Dec) plus a Year total column.
    Cell background color uses an RGB gradient from red through white to green.

    행은 연도, 열은 월(1-12월) + 연간 합계 열입니다.
    셀 배경색은 빨강에서 흰색을 거쳐 녹색까지의 RGB 그래디언트를 사용합니다.

    Args:
        monthData: Nested dict {year: {month: return_pct}}. 중첩 딕셔너리.
        ko: Use Korean labels if True. True면 한국어 레이블 사용.

    Returns:
        Table: Rich Table with heatmap cell coloring. 히트맵 셀 색상이 적용된 Rich Table.
    """
    monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if ko:
        monthNames = ["1\uc6d4", "2\uc6d4", "3\uc6d4", "4\uc6d4", "5\uc6d4", "6\uc6d4",
                       "7\uc6d4", "8\uc6d4", "9\uc6d4", "10\uc6d4", "11\uc6d4", "12\uc6d4"]

    table = Table(
        title=_l(ko, "Monthly Returns (%)", "\uc6d4\ubcc4 \uc218\uc775\ub960 (%)"),
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        padding=(0, 1),
    )

    table.add_column(
        _l(ko, "Year", "\uc5f0\ub3c4"),
        style="bold white", justify="right",
    )
    for mn in monthNames:
        table.add_column(mn, justify="right", min_width=6)
    table.add_column(
        _l(ko, "Year", "\uc5f0\uac04"),
        justify="right", min_width=7, style="bold",
    )

    for year in sorted(monthData.keys()):
        rowCells = [str(year)]
        yearTotal = 0.0
        monthCount = 0

        for month in range(1, 13):
            val = monthData[year].get(month)
            if val is not None:
                bgStyle = _heatmapColor(val)
                cell = Text(f"{val:+.1f}", style=f"bold {bgStyle}")
                rowCells.append(cell)
                yearTotal += val
                monthCount += 1
            else:
                rowCells.append(Text("-", style="dim"))

        yearBg = _heatmapColor(yearTotal)
        rowCells.append(Text(f"{yearTotal:+.1f}", style=f"bold {yearBg}"))
        table.add_row(*rowCells)

    return table


def printMonthlyHeatmap(result: Any, lang: str = "en") -> None:
    """
    Print a standalone monthly returns heatmap table.
    독립 실행형 월별 수익률 히트맵 테이블을 출력합니다.

    Rows represent years, columns represent months (Jan-Dec) plus an annual total.
    Each cell has a background color gradient: red for negative returns,
    white for zero, green for positive returns.

    행은 연도, 열은 월(1-12월) + 연간 합계를 나타냅니다.
    각 셀은 배경색 그래디언트를 가집니다: 음수 수익률은 빨강,
    0은 흰색, 양수 수익률은 녹색.

    Args:
        result: Result object with equityCurve. equityCurve가 포함된 결과 객체.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    r = _unwrap(result)
    ko = lang == "ko"
    monthData = _monthlyReturns(r)

    if not monthData:
        console.print(
            _l(ko, "[dim]No monthly return data available.[/dim]",
               "[dim]\uc6d4\ubcc4 \uc218\uc775\ub960 \ub370\uc774\ud130\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.[/dim]"),
        )
        return

    table = _buildMonthlyHeatmapTable(monthData, ko)
    console.print()
    console.print(table)
    console.print()


def printStrategyDna(dna: Any, lang: str = "en") -> None:
    """
    Display Strategy DNA as horizontal bar gauges with classification.
    전략 DNA를 수평 바 게이지와 분류로 표시합니다.

    Renders each of the 12 DNA dimensions as a labeled horizontal bar,
    color-coded green (high), yellow (mid), or red (low).
    Shows dominant traits and strategy archetype classification.

    12개 DNA 차원 각각을 라벨이 있는 수평 바로 렌더링하며,
    높음(녹색), 중간(노란색), 낮음(빨간색)으로 색상 코딩합니다.
    지배적 특성과 전략 원형 분류를 표시합니다.

    Args:
        dna: StrategyDNA dataclass instance. StrategyDNA 데이터클래스 인스턴스.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    ko = lang == "ko"

    DIMENSION_NAMES = [
        "trendSensitivity",
        "meanReversionAffinity",
        "volatilityPreference",
        "holdingPeriodProfile",
        "drawdownTolerance",
        "winRateProfile",
        "marketRegimeDependence",
        "concentrationLevel",
        "tradingFrequency",
        "riskRewardRatio",
        "momentumExposure",
        "defensiveScore",
    ]

    LABELS_EN = {
        "trendSensitivity": "Trend Sensitivity",
        "meanReversionAffinity": "Mean Reversion",
        "volatilityPreference": "Volatility Pref",
        "holdingPeriodProfile": "Holding Period",
        "drawdownTolerance": "DD Tolerance",
        "winRateProfile": "Win Rate Profile",
        "marketRegimeDependence": "Regime Depend.",
        "concentrationLevel": "Concentration",
        "tradingFrequency": "Trade Frequency",
        "riskRewardRatio": "Risk/Reward",
        "momentumExposure": "Momentum Exp.",
        "defensiveScore": "Defensive Score",
    }

    LABELS_KO = {
        "trendSensitivity": "\ucd94\uc138 \ubbfc\uac10\ub3c4",
        "meanReversionAffinity": "\ud3c9\uade0\ud68c\uadc0 \uc131\ud5a5",
        "volatilityPreference": "\ubcc0\ub3d9\uc131 \uc120\ud638\ub3c4",
        "holdingPeriodProfile": "\ubcf4\uc720 \uae30\uac04",
        "drawdownTolerance": "\ub099\ud3ed \ud5c8\uc6a9\ub3c4",
        "winRateProfile": "\uc2b9\ub960 \ud504\ub85c\ud30c\uc77c",
        "marketRegimeDependence": "\ub808\uc9d0 \uc758\uc874\ub3c4",
        "concentrationLevel": "\uc9d1\uc911\ub3c4",
        "tradingFrequency": "\uac70\ub798 \ube48\ub3c4",
        "riskRewardRatio": "\ub9ac\uc2a4\ud06c-\ub9ac\uc6cc\ub4dc",
        "momentumExposure": "\ubaa8\uba58\ud140 \uc775\uc2a4\ud3ec\uc838",
        "defensiveScore": "\ubc29\uc5b4\uc801 \uc810\uc218",
    }

    labels = LABELS_KO if ko else LABELS_EN

    table = Table(
        title=_l(ko, "Strategy DNA Profile", "\uc804\ub7b5 DNA \ud504\ub85c\ud30c\uc77c"),
        box=box.ROUNDED,
        header_style="bold cyan",
        padding=(0, 1),
    )

    table.add_column(
        _l(ko, "Dimension", "\ucc28\uc6d0"),
        min_width=18, style="white",
    )
    table.add_column(
        _l(ko, "Gauge", "\uac8c\uc774\uc9c0"),
        min_width=20,
    )
    table.add_column(
        _l(ko, "Value", "\uac12"),
        justify="right", min_width=6,
    )

    for name in DIMENSION_NAMES:
        val = getattr(dna, name, 0.0)
        barText = _colorBarText(val, 1.0, width=20)

        if val >= 0.7:
            valStyle = "bold green"
        elif val >= 0.4:
            valStyle = "bold yellow"
        else:
            valStyle = "bold red"

        table.add_row(
            labels.get(name, name),
            barText,
            Text(f"{val:.3f}", style=valStyle),
        )

    console.print()
    console.print(table)

    dominantTraits = []
    if hasattr(dna, "dominantTraits"):
        dominantTraits = dna.dominantTraits(3)

    if dominantTraits:
        traitText = Text()
        traitText.append(
            _l(ko, "  Dominant Traits: ", "  \uc9c0\ubc30\uc801 \ud2b9\uc131: "),
            style="bold white",
        )
        for i, (name, val) in enumerate(dominantTraits):
            if i > 0:
                traitText.append(", ", style="dim")
            label = labels.get(name, name)
            traitText.append(f"{label}", style="cyan")
            traitText.append(f" ({val:.2f})", style="green")
        console.print(traitText)

    try:
        from tradex.analytics.strategyDna import StrategyDnaAnalyzer
        analyzer = StrategyDnaAnalyzer()
        classification = analyzer.classify(dna)
        classText = Text()
        classText.append(
            _l(ko, "  Classification: ", "  \ubd84\ub958: "),
            style="bold white",
        )
        classText.append(classification, style="bold yellow")
        console.print(classText)
    except Exception:
        pass

    console.print()


def printHealthScore(score: Any, lang: str = "en") -> None:
    """
    Display Strategy Health Score as a comprehensive dashboard.
    전략 건강 점수를 종합 대시보드로 표시합니다.

    Shows the overall score with letter grade, five sub-scores as horizontal bars,
    warnings in yellow, and recommendations in cyan.

    전체 점수와 등급, 다섯 개 하위 점수를 수평 바로 표시하고,
    경고를 노란색으로, 권고 사항을 시안색으로 표시합니다.

    Args:
        score: StrategyHealthScore dataclass instance. StrategyHealthScore 인스턴스.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    ko = lang == "ko"

    overall = getattr(score, "overallHealth", 0.0)
    grade = getattr(score, "grade", "F")
    overfitting = getattr(score, "overfittingRisk", 0.0)
    stability = getattr(score, "parameterStability", 0.0)
    consistency = getattr(score, "performanceConsistency", 0.0)
    quality = getattr(score, "tradeQuality", 0.0)
    adaptability = getattr(score, "marketAdaptability", 0.0)
    warnings = getattr(score, "warnings", [])
    recommendations = getattr(score, "recommendations", [])

    if overall >= 80:
        gradeStyle = "bold green"
        overallStyle = "green"
    elif overall >= 60:
        gradeStyle = "bold cyan"
        overallStyle = "cyan"
    elif overall >= 40:
        gradeStyle = "bold yellow"
        overallStyle = "yellow"
    else:
        gradeStyle = "bold red"
        overallStyle = "red"

    headerText = Text()
    headerText.append(
        _l(ko, " Strategy Health Score ", " \uc804\ub7b5 \uac74\uac15 \uc810\uc218 "),
        style="bold white on dark_blue",
    )
    headerText.append("  ", style="")
    headerText.append(f"{overall:.1f}", style=f"bold {overallStyle}")
    headerText.append("/100 ", style="dim")
    headerText.append(f"[{grade}]", style=gradeStyle)

    subScores = [
        (
            _l(ko, "Overfitting Risk", "\uacfc\uc801\ud569 \uc800\ud56d"),
            overfitting,
        ),
        (
            _l(ko, "Param Stability", "\ud30c\ub77c\ubbf8\ud130 \uc548\uc815\uc131"),
            stability,
        ),
        (
            _l(ko, "Perf Consistency", "\uc131\uacfc \uc77c\uad00\uc131"),
            consistency,
        ),
        (
            _l(ko, "Trade Quality", "\uac70\ub798 \ud488\uc9c8"),
            quality,
        ),
        (
            _l(ko, "Market Adapt.", "\uc2dc\uc7a5 \uc801\uc751\ub825"),
            adaptability,
        ),
    ]

    scoreTable = Table(
        box=box.SIMPLE, show_header=False, padding=(0, 1),
    )
    scoreTable.add_column(min_width=20)
    scoreTable.add_column(min_width=22)
    scoreTable.add_column(justify="right", min_width=10)

    for label, val in subScores:
        barText = _colorBarText(val, 100.0, width=20)
        if val >= 70:
            valStyle = "bold green"
        elif val >= 50:
            valStyle = "bold yellow"
        else:
            valStyle = "bold red"
        scoreTable.add_row(
            Text(label, style="white"),
            barText,
            Text(f"{val:.1f}/100", style=valStyle),
        )

    innerContent = Table.grid()
    innerContent.add_column()
    innerContent.add_row(headerText)
    innerContent.add_row(Text(""))
    innerContent.add_row(scoreTable)

    if warnings:
        innerContent.add_row(Text(""))
        warnHeader = Text()
        warnHeader.append(
            _l(ko, f"  Warnings ({len(warnings)})", f"  \uacbd\uace0 ({len(warnings)}\uac74)"),
            style="bold yellow",
        )
        innerContent.add_row(warnHeader)
        for w in warnings:
            warnText = Text()
            warnText.append("    \u26a0 ", style="yellow")
            warnText.append(w, style="yellow")
            innerContent.add_row(warnText)

    if recommendations:
        innerContent.add_row(Text(""))
        recHeader = Text()
        recHeader.append(
            _l(
                ko,
                f"  Recommendations ({len(recommendations)})",
                f"  \uad8c\uace0 \uc0ac\ud56d ({len(recommendations)}\uac74)",
            ),
            style="bold cyan",
        )
        innerContent.add_row(recHeader)
        for rec in recommendations:
            recText = Text()
            recText.append("    \u2192 ", style="cyan")
            recText.append(rec, style="cyan")
            innerContent.add_row(recText)

    console.print()
    console.print(Panel(
        innerContent,
        border_style=overallStyle,
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()


def printBlackSwanScore(score: Any, lang: str = "en") -> None:
    """
    Display Black Swan Defense Score as a detailed dashboard.
    블랙스완 방어 점수를 상세 대시보드로 표시합니다.

    Shows the overall 0-100 score with letter grade, five sub-scores
    as horizontal bars, and a crisis performance breakdown if available.

    전체 0-100 점수와 등급, 다섯 개 하위 점수를 수평 바로 표시하고,
    가능한 경우 위기 기간 성과 분해를 표시합니다.

    Args:
        score: BlackSwanScore dataclass instance. BlackSwanScore 인스턴스.
        lang: Language code ('en' or 'ko'). 언어 코드 ('en' 또는 'ko').
    """
    ko = lang == "ko"

    overall = getattr(score, "overallScore", 0.0)
    grade = getattr(score, "grade", "F")
    tailRisk = getattr(score, "tailRiskScore", 0.0)
    recovery = getattr(score, "recoverySpeed", 0.0)
    ddResilience = getattr(score, "drawdownResilience", 0.0)
    volAdapt = getattr(score, "volatilityAdaptation", 0.0)
    crisisPerf = getattr(score, "crisisPerformance", 0.0)
    details = getattr(score, "details", {})

    if overall >= 80:
        gradeStyle = "bold green"
        overallStyle = "green"
    elif overall >= 60:
        gradeStyle = "bold cyan"
        overallStyle = "cyan"
    elif overall >= 40:
        gradeStyle = "bold yellow"
        overallStyle = "yellow"
    else:
        gradeStyle = "bold red"
        overallStyle = "red"

    headerText = Text()
    headerText.append(
        _l(
            ko,
            " Black Swan Defense Score ",
            " \ube14\ub799\uc2a4\uc640 \ubc29\uc5b4 \uc810\uc218 ",
        ),
        style="bold white on dark_red",
    )
    headerText.append("  ", style="")
    headerText.append(f"{overall:.1f}", style=f"bold {overallStyle}")
    headerText.append("/100 ", style="dim")
    headerText.append(f"[{grade}]", style=gradeStyle)

    subScores = [
        (
            _l(ko, "Tail Risk", "\uaf2c\ub9ac \uc704\ud5d8 \ub300\uc751"),
            tailRisk,
        ),
        (
            _l(ko, "Recovery Speed", "\ud68c\ubcf5 \uc18d\ub3c4"),
            recovery,
        ),
        (
            _l(ko, "DD Resilience", "\ub099\ud3ed \uc800\ud56d\ub825"),
            ddResilience,
        ),
        (
            _l(ko, "Vol Adaptation", "\ubcc0\ub3d9\uc131 \uc801\uc751"),
            volAdapt,
        ),
        (
            _l(ko, "Crisis Perf.", "\uc704\uae30 \uc131\uacfc"),
            crisisPerf,
        ),
    ]

    scoreTable = Table(
        box=box.SIMPLE, show_header=False, padding=(0, 1),
    )
    scoreTable.add_column(min_width=20)
    scoreTable.add_column(min_width=22)
    scoreTable.add_column(justify="right", min_width=10)

    for label, val in subScores:
        barText = _colorBarText(val, 100.0, width=20)
        if val >= 70:
            valStyle = "bold green"
        elif val >= 50:
            valStyle = "bold yellow"
        else:
            valStyle = "bold red"
        scoreTable.add_row(
            Text(label, style="white"),
            barText,
            Text(f"{val:.1f}/100", style=valStyle),
        )

    innerContent = Table.grid()
    innerContent.add_column()
    innerContent.add_row(headerText)
    innerContent.add_row(Text(""))
    innerContent.add_row(scoreTable)

    crisisBreakdown = details.get("crisisPerformance", {}).get("crisisBreakdown", {})
    if crisisBreakdown:
        innerContent.add_row(Text(""))
        crisisHeader = Text()
        crisisHeader.append(
            _l(ko, "  Crisis Period Breakdown", "  \uc704\uae30 \uae30\uac04 \uc131\uacfc"),
            style="bold white",
        )
        innerContent.add_row(crisisHeader)

        crisisTable = Table(
            box=box.SIMPLE, show_header=True, padding=(0, 1),
            header_style="bold dim",
        )
        crisisTable.add_column(
            _l(ko, "Crisis", "\uc704\uae30"), min_width=24,
        )
        crisisTable.add_column(
            _l(ko, "Return", "\uc218\uc775\ub960"),
            justify="right", min_width=10,
        )
        crisisTable.add_column(
            _l(ko, "Score", "\uc810\uc218"),
            justify="right", min_width=10,
        )
        crisisTable.add_column(
            _l(ko, "Days", "\uac70\ub798\uc77c"),
            justify="right", min_width=6,
        )

        for crisisName, crisisData in crisisBreakdown.items():
            crisisLabel = crisisData.get("label", crisisName)
            crisisRet = crisisData.get("returnPct", 0.0)
            crisisScore = crisisData.get("score", 0.0)
            crisisDays = crisisData.get("tradingDays", 0)

            crisisTable.add_row(
                crisisLabel,
                _cv(crisisRet),
                _cm(crisisScore, threshold=50),
                str(crisisDays),
            )

        innerContent.add_row(crisisTable)

    tailDetails = details.get("tailRisk", {})
    if tailDetails and "var95" in tailDetails:
        innerContent.add_row(Text(""))
        tailHeader = Text()
        tailHeader.append(
            _l(ko, "  Tail Risk Metrics", "  \uaf2c\ub9ac \uc704\ud5d8 \uc9c0\ud45c"),
            style="bold white",
        )
        innerContent.add_row(tailHeader)

        tailGrid = Table.grid(padding=(0, 3))
        tailGrid.add_column()
        tailGrid.add_column()
        tailGrid.add_column()

        tailGrid.add_row(
            Text(f"  VaR(95%): {tailDetails.get('var95', 0):.3f}%", style="white"),
            Text(f"VaR(99%): {tailDetails.get('var99', 0):.3f}%", style="white"),
            Text(f"CVaR(99%): {tailDetails.get('cvar99', 0):.3f}%", style="white"),
        )
        tailGrid.add_row(
            Text(
                f"  {_l(ko, 'Skewness', '\uc654\ub3c4')}: {tailDetails.get('skewness', 0):.3f}",
                style="white",
            ),
            Text(
                f"{_l(ko, 'Kurtosis', '\ucca8\ub3c4')}: {tailDetails.get('excessKurtosis', 0):.3f}",
                style="white",
            ),
            Text(
                f"{_l(ko, 'Tail Ratio', '\uaf2c\ub9ac\ube44')}: {tailDetails.get('tailRatio', 0):.3f}",
                style="white",
            ),
        )

        innerContent.add_row(tailGrid)

    console.print()
    console.print(Panel(
        innerContent,
        border_style=overallStyle,
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()
