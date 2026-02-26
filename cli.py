"""
Tradex CLI - Terminal command-line interface.

Provides `tradex` command with subcommands for backtesting, optimization,
and chart visualization directly from the terminal.

Usage:
    tradex backtest AAPL --strategy goldenCross
    tradex backtest 삼성전자 --strategy rsiOversold --period 5년
    tradex optimize AAPL --strategy goldenCross --fast 5-20 --slow 20-60
    tradex chart AAPL --period 1년
    tradex compare AAPL --strategies goldenCross,rsiOversold,macdCross

Tradex CLI - 터미널 명령줄 인터페이스.
"""

from typing import Optional, List
import typer
from rich.console import Console

app = typer.Typer(
    name="tradex",
    help="Blazing-fast backtesting engine for quantitative trading.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

STRATEGY_MAP = {
    "goldenCross": "goldenCross",
    "rsiOversold": "rsiOversold",
    "bollingerBreakout": "bollingerBreakout",
    "macdCross": "macdCross",
    "breakout": "breakout",
    "meanReversion": "meanReversion",
    "trendFollowing": "trendFollowing",
    "골든크로스": "goldenCross",
    "RSI과매도": "rsiOversold",
    "볼린저돌파": "bollingerBreakout",
    "MACD크로스": "macdCross",
    "돌파전략": "breakout",
    "평균회귀": "meanReversion",
    "추세추종": "trendFollowing",
}


def _getPreset(name: str):
    from tradex.easy.presets import (
        goldenCross, rsiOversold, bollingerBreakout,
        macdCross, breakout, meanReversion, trendFollowing,
    )
    presets = {
        "goldenCross": goldenCross,
        "rsiOversold": rsiOversold,
        "bollingerBreakout": bollingerBreakout,
        "macdCross": macdCross,
        "breakout": breakout,
        "meanReversion": meanReversion,
        "trendFollowing": trendFollowing,
    }
    resolved = STRATEGY_MAP.get(name, name)
    fn = presets.get(resolved)
    if fn is None:
        console.print(f"[red]Unknown strategy: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(presets.keys())}[/dim]")
        raise typer.Exit(1)
    return fn


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Ticker symbol or Korean name (e.g., AAPL, 삼성전자)"),
    strategy: str = typer.Option("goldenCross", "--strategy", "-s", help="Strategy name"),
    period: str = typer.Option("3년", "--period", "-p", help="Backtest period (e.g., 3년, 1년, 2020-01-01~2024-12-31)"),
    cash: int = typer.Option(10_000_000, "--cash", "-c", help="Initial capital"),
    chart: bool = typer.Option(False, "--chart", help="Show equity curve chart"),
    dashboard: bool = typer.Option(False, "--dashboard", "-d", help="Show full dashboard with charts"),
    lang: str = typer.Option("en", "--lang", "-l", help="Language (en/ko)"),
):
    """Run a backtest with a preset strategy.

    \b
    Examples:
        tradex backtest AAPL
        tradex backtest 삼성전자 -s goldenCross -p 5년 --chart
        tradex backtest AAPL -s rsiOversold --dashboard
    """
    from tradex.easy import backtest as _backtest
    from tradex.tui.console import printResult
    from tradex.tui.charts import plotEquityCurve, plotDashboard

    preset_fn = _getPreset(strategy)
    strat = preset_fn()

    console.print(f"\n[bold cyan]Running backtest...[/bold cyan] {symbol} / {strategy} / {period}\n")

    result = _backtest(symbol, strat, period=period, initialCash=cash)

    if dashboard:
        plotDashboard(result, lang=lang)
    else:
        printResult(result, lang=lang)
        if chart:
            plotEquityCurve(result, title=f"{symbol} Equity Curve")


@app.command()
def optimize(
    symbol: str = typer.Argument(..., help="Ticker symbol or Korean name"),
    strategy: str = typer.Option("goldenCross", "--strategy", "-s", help="Strategy name"),
    period: str = typer.Option("3년", "--period", "-p", help="Backtest period"),
    metric: str = typer.Option("sharpeRatio", "--metric", "-m", help="Optimization metric"),
    lang: str = typer.Option("en", "--lang", "-l", help="Language (en/ko)"),
):
    """Optimize strategy parameters via grid search.

    \b
    Examples:
        tradex optimize AAPL -s goldenCross
        tradex optimize 삼성전자 -s macdCross -m totalReturn
    """
    from tradex.easy import optimize as _optimize
    from tradex.tui.console import printResult, printComparison
    from tradex.tui.progress import optimizeProgress
    from rich.table import Table
    from rich import box

    preset_fn = _getPreset(strategy)

    console.print(f"\n[bold cyan]Optimizing...[/bold cyan] {symbol} / {strategy} / metric={metric}\n")

    best = _optimize(symbol, preset_fn, period=period, metric=metric)

    if best and best.get('best', {}).get('result'):
        console.print("[bold green]Best Parameters:[/bold green]")
        params_table = Table(box=box.SIMPLE)
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="white")
        for k, v in (best['best'].get('params') or {}).items():
            params_table.add_row(k, str(v))
        console.print(params_table)
        console.print()

        printResult(best['best']['result'], lang=lang)

        if best.get('all'):
            top_table = Table(title="Top Results", box=box.ROUNDED, header_style="bold cyan")
            top_table.add_column("#", style="dim")
            top_table.add_column("Params")
            top_table.add_column(metric, justify="right", style="green")
            for i, r in enumerate(best['all'][:10], 1):
                top_table.add_row(str(i), str(r.get('params', {})), f"{r.get('metric', 0):.4f}")
            console.print(top_table)
    else:
        console.print("[yellow]No valid optimization results.[/yellow]")


@app.command(name="chart")
def chart_cmd(
    symbol: str = typer.Argument(..., help="Ticker symbol or Korean name"),
    period: str = typer.Option("1년", "--period", "-p", help="Period"),
    chart_type: str = typer.Option("candle", "--type", "-t", help="Chart type (candle/line)"),
    bars: int = typer.Option(100, "--bars", "-n", help="Number of bars to display"),
):
    """Show price chart in terminal.

    \b
    Examples:
        tradex chart AAPL
        tradex chart 삼성전자 -p 6개월 -n 60
    """
    from tradex.easy.api import _resolveSymbol, _resolvePeriod
    from tradex.datafeed.fdr import FinanceDataReaderFeed
    from tradex.tui.charts import plotCandlestick

    ticker = _resolveSymbol(symbol)
    start, end = _resolvePeriod(period)

    console.print(f"\n[bold cyan]Loading price data...[/bold cyan] {symbol} ({ticker})\n")

    feed = FinanceDataReaderFeed(ticker, start, end)
    df = feed.getData()

    if df is None or df.empty:
        console.print("[red]Failed to load data.[/red]")
        raise typer.Exit(1)

    plotCandlestick(df, title=f"{symbol} ({ticker})", lastN=bars)


@app.command()
def compare(
    symbol: str = typer.Argument(..., help="Ticker symbol or Korean name"),
    strategies: str = typer.Option(
        "goldenCross,rsiOversold,macdCross,bollingerBreakout,meanReversion,trendFollowing",
        "--strategies", "-s",
        help="Comma-separated strategy names",
    ),
    period: str = typer.Option("3년", "--period", "-p", help="Backtest period"),
    lang: str = typer.Option("en", "--lang", "-l", help="Language (en/ko)"),
):
    """Compare multiple strategies on one symbol.

    \b
    Examples:
        tradex compare AAPL
        tradex compare 삼성전자 -s goldenCross,rsiOversold,macdCross -l ko
    """
    from tradex.easy import backtest as _backtest
    from tradex.tui.console import printComparison

    strat_names = [s.strip() for s in strategies.split(",")]
    results = []

    for name in strat_names:
        try:
            preset_fn = _getPreset(name)
            strat = preset_fn()
            console.print(f"  [dim]Running {name}...[/dim]")
            result = _backtest(symbol, strat, period=period)
            results.append(result)
        except Exception as e:
            console.print(f"  [red]{name}: {e}[/red]")

    if results:
        printComparison(results, lang=lang)
    else:
        console.print("[yellow]No results to compare.[/yellow]")


@app.command()
def version():
    """Show Tradex version."""
    from tradex.version import CURRENT_VERSION
    console.print(f"[bold cyan]Tradex[/bold cyan] v{CURRENT_VERSION}")


@app.command(name="list")
def list_strategies():
    """List all available preset strategies."""
    from rich.table import Table
    from rich import box

    table = Table(title="Available Strategies", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Name", style="bold white")
    table.add_column("Korean", style="cyan")
    table.add_column("Description")

    table.add_row("goldenCross", "골든크로스", "SMA crossover (fast crosses above slow)")
    table.add_row("rsiOversold", "RSI과매도", "Buy when RSI < 30, sell when RSI > 70")
    table.add_row("bollingerBreakout", "볼린저돌파", "Buy on lower band touch, sell on upper")
    table.add_row("macdCross", "MACD크로스", "Buy/sell on MACD signal line crossover")
    table.add_row("breakout", "돌파전략", "Channel breakout (Turtle Trading)")
    table.add_row("meanReversion", "평균회귀", "Bollinger Band mean reversion")
    table.add_row("trendFollowing", "추세추종", "ADX-filtered trend following with trailing stop")

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
