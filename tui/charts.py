"""
TradingView-inspired terminal charts using Plotext.

Renders candlestick + volume + indicator overlays + trade markers + drawdown
directly in the terminal. Designed to mimic TradingView's layout.

TradingView 스타일 터미널 차트. 캔들스틱+거래량+지표 오버레이+매매 마커+낙폭을
터미널에 직접 렌더링합니다.
"""

from typing import Any, List, Optional, Dict
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import plotext as plt


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

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
            return ec.index.strftime('%Y-%m-%d').tolist()
    except (ImportError, AttributeError):
        pass
    return []


def _sma(data: list, period: int) -> list:
    """Simple moving average."""
    result = [None] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result


def _dateTicks(dates: list, n_ticks: int = 6) -> tuple:
    if not dates:
        return [], []
    n = len(dates)
    step = max(1, n // n_ticks)
    indices = list(range(0, n, step))
    labels = [dates[i] for i in indices]
    return indices, labels


def _getMetric(r: Any, name: str, default: float = 0.0) -> float:
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    metrics = getattr(r, 'metrics', {})
    if isinstance(metrics, dict):
        return float(metrics.get(name, default))
    return default


# ──────────────────────────────────────────────────────────────
# Core Chart Functions
# ──────────────────────────────────────────────────────────────

def plotEquityCurve(
    result: Any,
    title: str = "Equity Curve",
    height: int = 18,
    width: Optional[int] = None,
    sma_periods: List[int] = None,
) -> None:
    """Plot equity curve with optional SMA overlays.

    Args:
        result: Backtest result object.
        title: Chart title.
        height: Chart height in rows.
        width: Chart width in columns.
        sma_periods: SMA periods to overlay (e.g., [20, 60]).
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data:
        print("[No equity data]")
        return

    plt.clear_figure()
    w = width or plt.terminal_width()
    plt.plotsize(w, height)
    plt.title(title)
    plt.theme("dark")

    plt.plot(data, label="Equity", color="cyan")

    initial = getattr(r, 'initialCash', data[0] if data else 0)
    if initial:
        plt.hline(initial, color="gray")

    if sma_periods:
        colors = ["yellow", "magenta", "white"]
        for i, period in enumerate(sma_periods):
            ma = _sma(data, period)
            valid = [(j, v) for j, v in enumerate(ma) if v is not None]
            if valid:
                plt.plot(
                    [j for j, _ in valid],
                    [v for _, v in valid],
                    label=f"SMA{period}",
                    color=colors[i % len(colors)],
                )

    dates = _getDateLabels(r)
    if dates and len(dates) == len(data):
        idx, lbl = _dateTicks(dates)
        plt.xticks(idx, lbl)

    plt.ylabel("Value")
    plt.show()
    plt.clear_figure()


def plotDrawdown(
    result: Any,
    title: str = "Drawdown",
    height: int = 10,
    width: Optional[int] = None,
) -> None:
    """Plot drawdown curve in terminal."""
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

    dates = _getDateLabels(r)
    if dates and len(dates) == len(dd):
        idx, lbl = _dateTicks(dates)
        plt.xticks(idx, lbl)

    plt.show()
    plt.clear_figure()


def plotCandlestick(
    df: Any,
    title: str = "Price",
    height: int = 18,
    width: Optional[int] = None,
    last_n: int = 100,
    sma_periods: List[int] = None,
    volume: bool = True,
) -> None:
    """Plot candlestick chart with optional SMA overlay and volume bars.

    TradingView style: candlestick on top, volume bars below.

    Args:
        df: DataFrame with OHLCV columns.
        title: Chart title.
        height: Chart height.
        width: Chart width.
        last_n: Number of recent bars to show.
        sma_periods: SMA periods to overlay (e.g., [5, 20, 60]).
        volume: Show volume subplot below.
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
        elif cl in ('volume', 'v', 'vol'):
            col_map['volume'] = col

    if len(col_map) < 4:
        print(f"[Need OHLC columns, found: {list(df.columns)}]")
        return

    df = df.tail(last_n).copy().reset_index(drop=False)
    opens = df[col_map['open']].tolist()
    highs = df[col_map['high']].tolist()
    lows = df[col_map['low']].tolist()
    closes = df[col_map['close']].tolist()

    dates = None
    idx_col = df.columns[0]
    try:
        if hasattr(df[idx_col], 'dt'):
            dates = df[idx_col].dt.strftime('%Y-%m-%d').tolist()
        elif hasattr(df[idx_col].iloc[0], 'strftime'):
            dates = [d.strftime('%Y-%m-%d') for d in df[idx_col]]
    except Exception:
        pass

    has_vol = 'volume' in col_map and volume
    w = width or plt.terminal_width()

    if has_vol:
        vol_data = df[col_map['volume']].tolist()
        plt.clear_figure()
        plt.subplots(2, 1)

        # --- Candlestick subplot ---
        plt.subplot(1, 1)
        plt.plotsize(w, height)
        plt.title(title)
        plt.theme("dark")
        plt.candlestick(opens, closes, highs, lows)

        if sma_periods:
            colors = ["yellow", "magenta", "white"]
            for i, period in enumerate(sma_periods or []):
                ma = _sma(closes, period)
                valid = [(j, v) for j, v in enumerate(ma) if v is not None]
                if valid:
                    plt.plot(
                        [j for j, _ in valid],
                        [v for _, v in valid],
                        label=f"SMA{period}",
                        color=colors[i % len(colors)],
                    )

        if dates:
            idx, lbl = _dateTicks(dates)
            plt.xticks(idx, lbl)

        # --- Volume subplot ---
        plt.subplot(2, 1)
        plt.plotsize(w, 6)
        plt.theme("dark")

        vol_colors = ["green" if closes[i] >= opens[i] else "red" for i in range(len(closes))]
        plt.bar(list(range(len(vol_data))), vol_data, color="gray")
        plt.title("Volume")

        if dates:
            idx, lbl = _dateTicks(dates)
            plt.xticks(idx, lbl)

        plt.show()
        plt.clear_figure()
    else:
        plt.clear_figure()
        plt.plotsize(w, height)
        plt.title(title)
        plt.theme("dark")
        plt.candlestick(opens, closes, highs, lows)

        if sma_periods:
            colors = ["yellow", "magenta", "white"]
            for i, period in enumerate(sma_periods or []):
                ma = _sma(closes, period)
                valid = [(j, v) for j, v in enumerate(ma) if v is not None]
                if valid:
                    plt.plot(
                        [j for j, _ in valid],
                        [v for _, v in valid],
                        label=f"SMA{period}",
                        color=colors[i % len(colors)],
                    )

        if dates:
            idx, lbl = _dateTicks(dates)
            plt.xticks(idx, lbl)

        plt.show()
        plt.clear_figure()


def plotReturns(
    result: Any,
    title: str = "Daily Returns",
    height: int = 10,
    width: Optional[int] = None,
) -> None:
    """Plot return distribution histogram in terminal."""
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
    plt.ylabel("Count")
    plt.show()
    plt.clear_figure()


def plotTradeMarkers(
    result: Any,
    title: str = "Equity + Trades",
    height: int = 18,
    width: Optional[int] = None,
) -> None:
    """Plot equity curve with buy/sell trade markers.

    Green dots = entry, Red dots = exit.
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    trades = getattr(r, 'trades', [])
    if not data:
        print("[No equity data]")
        return

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    plt.plot(data, label="Equity", color="cyan")

    initial = getattr(r, 'initialCash', data[0])
    if initial:
        plt.hline(initial, color="gray")

    # Mark trades on equity curve
    dates = _getDateLabels(r)
    if trades and dates:
        date_to_idx = {d: i for i, d in enumerate(dates)}
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for t in trades:
            if isinstance(t, dict):
                entry_d = str(t.get('entryDate', ''))[:10]
                exit_d = str(t.get('exitDate', ''))[:10]
            else:
                entry_d = str(getattr(t, 'entryDate', ''))[:10]
                exit_d = str(getattr(t, 'exitDate', ''))[:10]

            if entry_d in date_to_idx:
                idx = date_to_idx[entry_d]
                buy_x.append(idx)
                buy_y.append(data[idx])
            if exit_d in date_to_idx:
                idx = date_to_idx[exit_d]
                sell_x.append(idx)
                sell_y.append(data[idx])

        if buy_x:
            plt.scatter(buy_x, buy_y, color="green", label="Buy", marker="dot")
        if sell_x:
            plt.scatter(sell_x, sell_y, color="red", label="Sell", marker="dot")

    if dates:
        idx, lbl = _dateTicks(dates)
        plt.xticks(idx, lbl)

    plt.ylabel("Value")
    plt.show()
    plt.clear_figure()


# ──────────────────────────────────────────────────────────────
# Dashboard - TradingView Style
# ──────────────────────────────────────────────────────────────

def plotDashboard(result: Any, lang: str = "en") -> None:
    """Print TradingView-inspired dashboard.

    Layout:
        [Header] Strategy / Symbol / Period / Key Metrics
        [Chart 1] Equity Curve + Trade Markers + SMA overlays
        [Chart 2] Drawdown
        [Chart 3] Return Distribution

    Args:
        result: Backtest result object.
        lang: Language ("en" or "ko").
    """
    from tradex.tui.console import printResult

    ko = lang == "ko"
    r = _unwrap(result)
    data = _getEquityList(r)

    # 1. Rich metrics panel
    printResult(result, lang=lang)

    if not data:
        return

    # 2. Equity + Trade Markers
    plotTradeMarkers(
        result,
        title="Equity Curve + Trades" if not ko else "자산곡선 + 매매",
        height=16,
    )

    # 3. Drawdown
    plotDrawdown(
        result,
        title="Drawdown" if not ko else "낙폭",
        height=10,
    )

    # 4. Return Distribution
    plotReturns(
        result,
        title="Daily Returns" if not ko else "일간 수익률 분포",
        height=10,
    )
