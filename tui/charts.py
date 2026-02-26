"""
Terminal charts using Plotext.

Renders equity curves, candlestick charts, drawdown, and return distributions
directly in the terminal.

Plotext 기반 터미널 차트. 자산곡선, 캔들스틱, 낙폭, 수익분포를 터미널에 직접 렌더링합니다.
"""

from typing import Any, List, Optional
import sys
import io

# Windows cp949 encoding fix for plotext unicode output
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import plotext as plt


def _unwrap(result: Any) -> Any:
    if hasattr(result, '_result'):
        return result._result
    return result


def _getEquityList(r: Any) -> list:
    ec = getattr(r, 'equityCurve', None)
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
    return list(ec) if hasattr(ec, '__iter__') else []


def _getDateLabels(r: Any) -> list:
    ec = getattr(r, 'equityCurve', None)
    try:
        import pandas as pd
        if isinstance(ec, pd.Series) and hasattr(ec.index, 'strftime'):
            dates = ec.index.strftime('%Y-%m-%d').tolist()
            return dates
    except (ImportError, AttributeError):
        pass
    return []


def plotEquityCurve(result: Any, title: str = "Equity Curve", height: int = 20, width: Optional[int] = None) -> None:
    """Plot equity curve in terminal.

    Args:
        result: Backtest result object.
        title: Chart title.
        height: Chart height in rows.
        width: Chart width in columns (auto if None).
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data:
        print("[No equity data]")
        return

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    dates = _getDateLabels(r)
    if dates and len(dates) == len(data):
        plt.plot(data, label="Equity")
        n = len(dates)
        step = max(1, n // 6)
        ticks = list(range(0, n, step))
        plt.xticks(ticks, [dates[i] for i in ticks])
    else:
        plt.plot(data, label="Equity")
        plt.xlabel("Days")

    plt.ylabel("Value")

    initial = getattr(r, 'initialCash', data[0] if data else 0)
    if initial:
        plt.hline(initial, color="gray")

    plt.show()
    plt.clear_figure()


def plotDrawdown(result: Any, title: str = "Drawdown", height: int = 15, width: Optional[int] = None) -> None:
    """Plot drawdown curve in terminal.

    Args:
        result: Backtest result object.
        title: Chart title.
        height: Chart height.
        width: Chart width.
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data or len(data) < 2:
        print("[No equity data for drawdown]")
        return

    peak = data[0]
    dd = []
    for v in data:
        if v > peak:
            peak = v
        dd.append((v / peak - 1) * 100 if peak else 0)

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    plt.plot(dd, color="red", label="Drawdown %")
    plt.hline(0, color="gray")
    plt.ylabel("%")
    plt.xlabel("Days")

    plt.show()
    plt.clear_figure()


def plotCandlestick(
    df: Any,
    title: str = "Price",
    height: int = 20,
    width: Optional[int] = None,
    last_n: int = 100,
) -> None:
    """Plot candlestick chart in terminal.

    Args:
        df: DataFrame with Open, High, Low, Close columns.
        title: Chart title.
        height: Chart height.
        width: Chart width.
        last_n: Number of recent bars to show.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        print("[Candlestick requires DataFrame with OHLC columns]")
        return

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('open', 'o'):
            col_map['open'] = col
        elif cl in ('high', 'h'):
            col_map['high'] = col
        elif cl in ('low', 'l'):
            col_map['low'] = col
        elif cl in ('close', 'c'):
            col_map['close'] = col

    if len(col_map) < 4:
        print(f"[Need OHLC columns, found: {list(df.columns)}]")
        return

    df = df.tail(last_n).copy()
    opens = df[col_map['open']].tolist()
    highs = df[col_map['high']].tolist()
    lows = df[col_map['low']].tolist()
    closes = df[col_map['close']].tolist()

    dates = None
    if hasattr(df.index, 'strftime'):
        dates = df.index.strftime('%Y-%m-%d').tolist()

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")
    plt.candlestick(opens, closes, highs, lows)

    if dates:
        n = len(dates)
        step = max(1, n // 6)
        ticks = list(range(0, n, step))
        plt.xticks(ticks, [dates[i] for i in ticks])

    plt.show()
    plt.clear_figure()


def plotReturns(result: Any, title: str = "Daily Returns Distribution", height: int = 15, width: Optional[int] = None) -> None:
    """Plot return distribution histogram in terminal.

    Args:
        result: Backtest result object.
        title: Chart title.
        height: Chart height.
        width: Chart width.
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data or len(data) < 3:
        print("[Not enough equity data]")
        return

    returns = [(data[i] / data[i - 1] - 1) * 100 for i in range(1, len(data)) if data[i - 1] != 0]
    if not returns:
        return

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")
    plt.hist(returns, bins=40, color="cyan")
    plt.vline(0, color="gray")
    plt.xlabel("Return %")
    plt.ylabel("Frequency")
    plt.show()
    plt.clear_figure()


def plotDashboard(result: Any, lang: str = "en") -> None:
    """Print full dashboard with charts and metrics.

    Combines equity curve, drawdown, and return distribution in one view.

    Args:
        result: Backtest result object.
        lang: Language ("en" or "ko").
    """
    from tradex.tui.console import printResult

    ko = lang == "ko"

    printResult(result, lang=lang)

    r = _unwrap(result)
    data = _getEquityList(r)
    if not data:
        return

    half_w = max(40, plt.terminal_width() // 2 - 2)

    # Equity Curve
    plotEquityCurve(
        result,
        title="Equity Curve" if not ko else "자산 곡선",
        height=15,
    )

    # Drawdown
    plotDrawdown(
        result,
        title="Drawdown" if not ko else "낙폭",
        height=12,
    )

    # Return distribution
    plotReturns(
        result,
        title="Return Distribution" if not ko else "수익률 분포",
        height=12,
    )
