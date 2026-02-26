"""
TradingView-inspired Rich console output for backtest results.

Renders styled metric cards, trade tables, and sparklines.
TradingView 스타일 Rich 콘솔 출력.
"""

from typing import Any, Dict, List, Optional, Union
import sys

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

_SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_SPARK_FALLBACK = " .:;|"


def _canUnicode() -> bool:
    try:
        "\u2581".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _sparkline(values: list, width: int = 20) -> str:
    if not values or len(values) < 2:
        return ""
    chars = _SPARK if _canUnicode() else _SPARK_FALLBACK
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    mx_idx = len(chars) - 1
    return "".join(chars[min(int((v - mn) / rng * mx_idx), mx_idx)] for v in sampled)


def _cv(value: float, fmt: str = "+.2f", suffix: str = "%") -> Text:
    """Color value: green if positive, red if negative."""
    text = f"{value:{fmt}}{suffix}"
    if value > 0:
        return Text(text, style="bold green")
    elif value < 0:
        return Text(text, style="bold red")
    return Text(text, style="dim")


def _cm(value: float, fmt: str = ".2f", suffix: str = "", threshold: float = 0) -> Text:
    """Color metric: green if above threshold, red if below."""
    text = f"{value:{fmt}}{suffix}"
    if value > threshold:
        return Text(text, style="green")
    elif value < threshold:
        return Text(text, style="red")
    return Text(text, style="dim")


def _metric(r: Any, name: str, default: float = 0.0) -> float:
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    m = getattr(r, 'metrics', {})
    return float(m.get(name, default)) if isinstance(m, dict) else default


def _unwrap(result: Any) -> Any:
    if hasattr(result, '_result'):
        return result._result
    return result


def _equityData(r: Any) -> list:
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


# ──────────────────────────────────────────────────────────────
# Main Result Printer
# ──────────────────────────────────────────────────────────────

def printResult(result: Any, lang: str = "en") -> None:
    """Print TradingView-style backtest summary.

    Args:
        result: BacktestResult, VectorizedResult, or EasyResult.
        lang: "en" or "ko".
    """
    r = _unwrap(result)
    ko = lang == "ko"
    eq = _equityData(r)
    spark = _sparkline(eq, width=30) if eq else ""

    strategy = getattr(r, 'strategy', 'Strategy')
    symbol = getattr(r, 'symbol', '')
    start = getattr(r, 'startDate', '')
    end = getattr(r, 'endDate', '')

    initial = getattr(r, 'initialCash', 0)
    final = getattr(r, 'finalEquity', 0)
    total_ret = getattr(r, 'totalReturn', 0)
    annual = _metric(r, 'annualReturn')
    vol = _metric(r, 'volatility')
    sharpe = _metric(r, 'sharpeRatio')
    mdd = _metric(r, 'maxDrawdown')
    mdd_dur = _metric(r, 'maxDrawdownDuration', 0)
    pf = _metric(r, 'profitFactor')
    trades = getattr(r, 'totalTrades', 0)
    wr = getattr(r, 'winRate', 0)
    avg_w = _metric(r, 'avgWin')
    avg_l = _metric(r, 'avgLoss')

    # ── Header ──
    hdr = Text()
    hdr.append(f" {strategy} ", style="bold white on blue")
    hdr.append(f"  {symbol}", style="bold cyan")
    hdr.append(f"  {start} ~ {end}", style="dim")
    if spark:
        hdr.append(f"  {spark}", style="cyan")

    console.print()
    console.print(Panel(
        hdr,
        title="[bold]TRADEX[/bold]" if not ko else "[bold]TRADEX[/bold]",
        subtitle=f"{initial:,.0f} -> {final:,.0f}" if not ko else f"{initial:,.0f} -> {final:,.0f}",
        border_style="blue",
    ))

    # ── Metric Cards (TradingView style: key metrics in a row) ──
    cards = []

    def _card(label: str, value: Text, width: int = 18) -> Panel:
        content = Text()
        content.append(f"{label}\n", style="dim")
        content.append_text(value)
        return Panel(content, width=width, border_style="dim", padding=(0, 1))

    cards.append(_card(
        "Return" if not ko else "수익률",
        _cv(total_ret),
    ))
    cards.append(_card(
        "Annual" if not ko else "연수익",
        _cv(annual),
    ))
    cards.append(_card(
        "Sharpe" if not ko else "샤프",
        _cm(sharpe, threshold=1.0),
    ))
    cards.append(_card(
        "MDD" if not ko else "최대낙폭",
        _cv(mdd, fmt=".2f"),
    ))
    cards.append(_card(
        "Win Rate" if not ko else "승률",
        _cm(wr, suffix="%", threshold=50),
    ))
    cards.append(_card(
        "Trades" if not ko else "거래수",
        Text(str(trades), style="bold white"),
    ))

    console.print(Columns(cards, padding=(0, 0), equal=True))

    # ── Detail Table ──
    detail = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    detail.add_column(min_width=16)
    detail.add_column(justify="right", min_width=14)
    detail.add_column(min_width=16)
    detail.add_column(justify="right", min_width=14)

    l = lambda en, kr: kr if ko else en

    detail.add_row(
        l("Initial", "초기자금"), Text(f"{initial:,.0f}", style="white"),
        l("Volatility", "변동성"), _cm(vol, suffix="%"),
    )
    detail.add_row(
        l("Final", "최종자산"), Text(f"{final:,.0f}", style="bold white"),
        l("Profit Factor", "손익비"), _cm(pf, threshold=1.0),
    )
    detail.add_row(
        l("MDD Duration", "낙폭기간"),
        Text(f"{int(mdd_dur)}d" if not ko else f"{int(mdd_dur)}일", style="dim"),
        l("Avg Win", "평균이익"),
        Text(f"{avg_w:,.0f}", style="green") if avg_w else Text("-", style="dim"),
    )
    detail.add_row(
        "", "",
        l("Avg Loss", "평균손실"),
        Text(f"{avg_l:,.0f}", style="red") if avg_l else Text("-", style="dim"),
    )

    console.print(detail)
    console.print()


# ──────────────────────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────────────────────

def printComparison(results: List[Any], lang: str = "en") -> None:
    """Print strategy comparison table."""
    ko = lang == "ko"
    table = Table(
        title="Strategy Comparison" if not ko else "전략 비교",
        box=box.ROUNDED,
        header_style="bold cyan",
        padding=(0, 1),
    )

    l = lambda en, kr: kr if ko else en
    table.add_column(l("Strategy", "전략"), style="bold white")
    table.add_column(l("Return", "수익률"), justify="right")
    table.add_column(l("Annual", "연수익"), justify="right")
    table.add_column(l("Sharpe", "샤프"), justify="right")
    table.add_column("MDD", justify="right")
    table.add_column(l("Win%", "승률"), justify="right")
    table.add_column(l("Trades", "거래"), justify="right")
    table.add_column(l("Equity", "자산곡선"), justify="left")

    for result in results:
        r = _unwrap(result)
        eq = _equityData(r)
        sp = _sparkline(eq, width=15) if eq else ""

        table.add_row(
            getattr(r, 'strategy', '?'),
            _cv(getattr(r, 'totalReturn', 0)),
            _cv(_metric(r, 'annualReturn')),
            _cm(_metric(r, 'sharpeRatio'), threshold=1.0),
            _cv(_metric(r, 'maxDrawdown'), fmt=".2f"),
            _cm(getattr(r, 'winRate', 0), suffix="%", threshold=50),
            str(getattr(r, 'totalTrades', 0)),
            Text(sp, style="cyan"),
        )

    console.print()
    console.print(table)
    console.print()


# ──────────────────────────────────────────────────────────────
# Trade History
# ──────────────────────────────────────────────────────────────

def printTrades(result: Any, limit: int = 20, lang: str = "en") -> None:
    """Print trade history table."""
    r = _unwrap(result)
    trades = getattr(r, 'trades', [])
    if not trades:
        console.print("[dim]No trades.[/dim]")
        return

    ko = lang == "ko"
    l = lambda en, kr: kr if ko else en

    table = Table(
        title=l(f"Trade History (last {limit})", f"거래 내역 (최근 {limit}건)"),
        box=box.SIMPLE,
        header_style="bold white",
        padding=(0, 1),
    )

    table.add_column("#", style="dim", justify="right")
    table.add_column(l("Side", "방향"), justify="center")
    table.add_column(l("Entry", "진입일"))
    table.add_column(l("Exit", "청산일"))
    table.add_column(l("Entry$", "진입가"), justify="right")
    table.add_column(l("Exit$", "청산가"), justify="right")
    table.add_column(l("P&L", "손익"), justify="right")
    table.add_column(l("Return", "수익률"), justify="right")

    for i, t in enumerate(trades[-limit:], 1):
        if isinstance(t, dict):
            side = t.get('side', 'buy')
            ed = str(t.get('entryDate', ''))[:10]
            xd = str(t.get('exitDate', ''))[:10]
            ep = t.get('entryPrice', 0)
            xp = t.get('exitPrice', 0)
            pnl = t.get('pnl', xp - ep)
            ret = t.get('returnPct', (xp / ep - 1) * 100 if ep else 0)
        else:
            side = getattr(t, 'side', 'buy')
            ed = str(getattr(t, 'entryDate', ''))[:10]
            xd = str(getattr(t, 'exitDate', ''))[:10]
            ep = getattr(t, 'entryPrice', 0)
            xp = getattr(t, 'exitPrice', 0)
            pnl = getattr(t, 'pnl', xp - ep)
            ret = getattr(t, 'returnPct', (xp / ep - 1) * 100 if ep else 0)

        is_buy = 'buy' in str(side).lower()
        side_text = Text("BUY" if is_buy else "SELL", style="green" if is_buy else "red")

        table.add_row(
            str(i), side_text, ed, xd,
            f"{ep:,.0f}", f"{xp:,.0f}",
            _cv(pnl, fmt=",.0f", suffix=""),
            _cv(ret),
        )

    console.print()
    console.print(table)
    console.print()
