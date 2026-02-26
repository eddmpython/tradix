"""
Rich-based console output for backtest results.

Renders styled tables, panels, and sparklines in the terminal.
Rich 기반 백테스트 결과 터미널 출력.
"""

from typing import Any, Dict, List, Optional, Union
import sys

# Windows cp949 encoding fix for unicode output
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
from rich import box

console = Console()

_SPARKLINE_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_SPARKLINE_FALLBACK = " .:;|"


def _canUnicode() -> bool:
    """Check if terminal supports unicode sparkline characters."""
    import sys
    try:
        "\u2581".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _sparkline(values: list, width: int = 20) -> str:
    if not values or len(values) < 2:
        return ""
    chars = _SPARKLINE_CHARS if _canUnicode() else _SPARKLINE_FALLBACK
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    max_idx = len(chars) - 1
    return "".join(
        chars[min(int((v - mn) / rng * max_idx), max_idx)] for v in sampled
    )


def _colorVal(value: float, fmt: str = "+.2f", suffix: str = "%") -> Text:
    text = f"{value:{fmt}}{suffix}"
    if value > 0:
        return Text(text, style="bold green")
    elif value < 0:
        return Text(text, style="bold red")
    return Text(text, style="dim")


def _colorMetric(value: float, fmt: str = ".2f", suffix: str = "", good_above: float = 0) -> Text:
    text = f"{value:{fmt}}{suffix}"
    if value > good_above:
        return Text(text, style="green")
    elif value < good_above:
        return Text(text, style="red")
    return Text(text, style="dim")


def printResult(result: Any, lang: str = "en") -> None:
    """Print backtest result as Rich styled tables.

    Args:
        result: BacktestResult, VectorizedResult, or EasyResult.
        lang: Language ("en" or "ko").
    """
    r = _unwrap(result)
    equity_data = _getEquityData(r)
    spark = _sparkline(equity_data) if equity_data else ""

    ko = lang == "ko"

    # --- Header ---
    strategy_name = getattr(r, 'strategy', 'Strategy')
    symbol = getattr(r, 'symbol', '')
    start = getattr(r, 'startDate', '')
    end = getattr(r, 'endDate', '')

    header = Text()
    header.append(f"  {strategy_name}", style="bold cyan")
    header.append(f"  {symbol}", style="bold white")
    header.append(f"  {start} ~ {end}", style="dim")

    console.print()
    console.print(Panel(header, title="TRADEX" if not ko else "트레이덱스", border_style="cyan"))

    # --- Return & Risk Panel ---
    ret_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        padding=(0, 2),
    )
    ret_table.add_column("Return" if not ko else "수익", justify="left", min_width=16)
    ret_table.add_column("Value" if not ko else "값", justify="right", min_width=12)
    ret_table.add_column("Risk" if not ko else "리스크", justify="left", min_width=16)
    ret_table.add_column("Value" if not ko else "값", justify="right", min_width=12)

    initial = getattr(r, 'initialCash', 0)
    final = getattr(r, 'finalEquity', 0)
    total_ret = getattr(r, 'totalReturn', 0)
    annual_ret = _getMetric(r, 'annualReturn')
    volatility = _getMetric(r, 'volatility')
    sharpe = _getMetric(r, 'sharpeRatio')
    max_dd = _getMetric(r, 'maxDrawdown')
    mdd_dur = _getMetric(r, 'maxDrawdownDuration', 0)
    sortino = _getMetric(r, 'sortinoRatio')
    calmar = _getMetric(r, 'calmarRatio')

    lbl = {
        "initial": "Initial Capital" if not ko else "초기 자금",
        "final": "Final Equity" if not ko else "최종 자산",
        "total": "Total Return" if not ko else "총 수익률",
        "annual": "Annual Return" if not ko else "연 수익률",
        "vol": "Volatility" if not ko else "변동성",
        "sharpe": "Sharpe Ratio" if not ko else "샤프 비율",
        "mdd": "Max Drawdown" if not ko else "최대 낙폭",
        "mdd_dur": "MDD Duration" if not ko else "낙폭 기간",
        "sortino": "Sortino Ratio" if not ko else "소르티노",
        "calmar": "Calmar Ratio" if not ko else "칼마 비율",
    }

    ret_table.add_row(
        lbl["initial"], Text(f"{initial:,.0f}", style="white"),
        lbl["vol"], _colorMetric(volatility, suffix="%"),
    )
    ret_table.add_row(
        lbl["final"], Text(f"{final:,.0f}", style="bold white"),
        lbl["sharpe"], _colorMetric(sharpe, good_above=1.0),
    )
    ret_table.add_row(
        lbl["total"], _colorVal(total_ret),
        lbl["mdd"], _colorVal(max_dd, fmt=".2f"),
    )
    ret_table.add_row(
        lbl["annual"], _colorVal(annual_ret),
        lbl["mdd_dur"], Text(f"{int(mdd_dur)} days" if not ko else f"{int(mdd_dur)}일", style="dim"),
    )

    console.print(ret_table)

    # --- Trade Stats ---
    trades_total = getattr(r, 'totalTrades', 0)
    win_rate = getattr(r, 'winRate', 0)
    pf = _getMetric(r, 'profitFactor')
    avg_win = _getMetric(r, 'avgWin')
    avg_loss = _getMetric(r, 'avgLoss')

    trade_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        padding=(0, 2),
    )
    lbl_t = {
        "trades": "Total Trades" if not ko else "총 거래",
        "winrate": "Win Rate" if not ko else "승률",
        "pf": "Profit Factor" if not ko else "손익비",
        "avgwin": "Avg Win" if not ko else "평균 이익",
        "avgloss": "Avg Loss" if not ko else "평균 손실",
    }
    trade_table.add_column("Trade Stats" if not ko else "거래 통계", justify="left", min_width=16)
    trade_table.add_column("", justify="right", min_width=12)
    trade_table.add_column("", justify="left", min_width=16)
    trade_table.add_column("", justify="right", min_width=12)

    trade_table.add_row(
        lbl_t["trades"], Text(str(trades_total), style="bold white"),
        lbl_t["pf"], _colorMetric(pf, good_above=1.0),
    )
    trade_table.add_row(
        lbl_t["winrate"], _colorMetric(win_rate, suffix="%", good_above=50),
        lbl_t["avgwin"], Text(f"{avg_win:,.0f}", style="green") if avg_win else Text("-", style="dim"),
    )
    trade_table.add_row(
        "", "",
        lbl_t["avgloss"], Text(f"{avg_loss:,.0f}", style="red") if avg_loss else Text("-", style="dim"),
    )

    console.print(trade_table)

    # --- Equity Sparkline ---
    if spark:
        equity_panel = Panel(
            Text(spark, style="cyan"),
            title="Equity Curve" if not ko else "자산 곡선",
            border_style="dim",
            width=min(60, console.width),
        )
        console.print(equity_panel)

    console.print()


def printComparison(results: List[Any], lang: str = "en") -> None:
    """Print comparison table for multiple backtest results.

    Args:
        results: List of backtest results.
        lang: Language ("en" or "ko").
    """
    ko = lang == "ko"
    table = Table(
        title="Strategy Comparison" if not ko else "전략 비교",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )

    table.add_column("Strategy" if not ko else "전략", style="bold white")
    table.add_column("Return" if not ko else "수익률", justify="right")
    table.add_column("Sharpe" if not ko else "샤프", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("Win%" if not ko else "승률", justify="right")
    table.add_column("Trades" if not ko else "거래", justify="right")
    table.add_column("Equity" if not ko else "자산곡선", justify="left")

    for result in results:
        r = _unwrap(result)
        equity = _getEquityData(r)
        spark = _sparkline(equity, width=15) if equity else ""

        table.add_row(
            getattr(r, 'strategy', '?'),
            _colorVal(getattr(r, 'totalReturn', 0)),
            _colorMetric(_getMetric(r, 'sharpeRatio'), good_above=1.0),
            _colorVal(_getMetric(r, 'maxDrawdown'), fmt=".2f"),
            _colorMetric(getattr(r, 'winRate', 0), suffix="%", good_above=50),
            str(getattr(r, 'totalTrades', 0)),
            Text(spark, style="cyan"),
        )

    console.print()
    console.print(table)
    console.print()


def printTrades(result: Any, limit: int = 20, lang: str = "en") -> None:
    """Print trade history table.

    Args:
        result: Backtest result.
        limit: Max trades to show.
        lang: Language ("en" or "ko").
    """
    r = _unwrap(result)
    trades = getattr(r, 'trades', [])
    if not trades:
        console.print("[dim]No trades.[/dim]")
        return

    ko = lang == "ko"
    table = Table(
        title=f"Trade History (last {limit})" if not ko else f"거래 내역 (최근 {limit}건)",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        padding=(0, 1),
    )

    table.add_column("#", style="dim", justify="right")
    table.add_column("Side" if not ko else "방향", justify="center")
    table.add_column("Entry" if not ko else "진입일", justify="left")
    table.add_column("Exit" if not ko else "청산일", justify="left")
    table.add_column("Entry $" if not ko else "진입가", justify="right")
    table.add_column("Exit $" if not ko else "청산가", justify="right")
    table.add_column("P&L" if not ko else "손익", justify="right")
    table.add_column("Return" if not ko else "수익률", justify="right")

    display_trades = trades[-limit:]
    for i, t in enumerate(display_trades, 1):
        if isinstance(t, dict):
            side = t.get('side', 'buy')
            entry_date = str(t.get('entryDate', ''))[:10]
            exit_date = str(t.get('exitDate', ''))[:10]
            entry_price = t.get('entryPrice', 0)
            exit_price = t.get('exitPrice', 0)
            pnl = t.get('pnl', exit_price - entry_price)
            ret = t.get('returnPct', (exit_price / entry_price - 1) * 100 if entry_price else 0)
        else:
            side = getattr(t, 'side', 'buy')
            entry_date = str(getattr(t, 'entryDate', ''))[:10]
            exit_date = str(getattr(t, 'exitDate', ''))[:10]
            entry_price = getattr(t, 'entryPrice', 0)
            exit_price = getattr(t, 'exitPrice', 0)
            pnl = getattr(t, 'pnl', exit_price - entry_price)
            ret = getattr(t, 'returnPct', (exit_price / entry_price - 1) * 100 if entry_price else 0)

        side_text = Text("BUY" if 'buy' in str(side).lower() else "SELL",
                         style="green" if 'buy' in str(side).lower() else "red")

        table.add_row(
            str(i),
            side_text,
            entry_date,
            exit_date,
            f"{entry_price:,.0f}",
            f"{exit_price:,.0f}",
            _colorVal(pnl, fmt=",.0f", suffix=""),
            _colorVal(ret),
        )

    console.print()
    console.print(table)
    console.print()


def _unwrap(result: Any) -> Any:
    """Unwrap EasyResult to underlying BacktestResult."""
    if hasattr(result, '_result'):
        return result._result
    return result


def _getEquityData(r: Any) -> list:
    """Extract equity curve as plain list."""
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


def _getMetric(r: Any, name: str, default: float = 0.0) -> float:
    """Get metric value from result or its metrics dict."""
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    metrics = getattr(r, 'metrics', {})
    if isinstance(metrics, dict):
        return float(metrics.get(name, default))
    return default
