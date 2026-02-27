"""
Tradix CLI - Terminal command-line interface.

Provides `tradix` command with subcommands for backtesting, optimization,
and chart visualization directly from the terminal.

Usage:
    tradix backtest AAPL --strategy goldenCross
    tradix backtest 삼성전자 --strategy rsiOversold --period 5년
    tradix optimize AAPL --strategy goldenCross --fast 5-20 --slow 20-60
    tradix chart AAPL --period 1년
    tradix compare AAPL --strategies goldenCross,rsiOversold,macdCross

Tradix CLI - 터미널 명령줄 인터페이스.
"""

from typing import Optional, List
import typer
from rich.console import Console

app = typer.Typer(
    name="tradix",
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
    "emaCross": "emaCross",
    "tripleScreen": "tripleScreen",
    "dualMomentum": "dualMomentum",
    "momentumCross": "momentumCross",
    "rocBreakout": "rocBreakout",
    "stochasticCross": "stochasticCross",
    "williamsReversal": "williamsReversal",
    "cciBreakout": "cciBreakout",
    "rsiDivergence": "rsiDivergence",
    "volatilityBreakout": "volatilityBreakout",
    "keltnerChannel": "keltnerChannel",
    "bollingerSqueeze": "bollingerSqueeze",
    "superTrend": "superTrend",
    "ichimokuCloud": "ichimokuCloud",
    "parabolicSar": "parabolicSar",
    "donchianBreakout": "donchianBreakout",
    "tripleEma": "tripleEma",
    "macdRsiCombo": "macdRsiCombo",
    "trendMomentum": "trendMomentum",
    "bollingerRsi": "bollingerRsi",
    "gapTrading": "gapTrading",
    "pyramiding": "pyramiding",
    "swingTrading": "swingTrading",
    "scalpingMomentum": "scalpingMomentum",
    "buyAndHold": "buyAndHold",
    "dollarCostAverage": "dollarCostAverage",
    "골든크로스": "goldenCross",
    "RSI과매도": "rsiOversold",
    "볼린저돌파": "bollingerBreakout",
    "MACD크로스": "macdCross",
    "돌파전략": "breakout",
    "평균회귀": "meanReversion",
    "추세추종": "trendFollowing",
    "EMA크로스": "emaCross",
    "삼중스크린": "tripleScreen",
    "듀얼모멘텀": "dualMomentum",
    "모멘텀크로스": "momentumCross",
    "ROC돌파": "rocBreakout",
    "스토캐스틱크로스": "stochasticCross",
    "윌리엄스반전": "williamsReversal",
    "CCI돌파": "cciBreakout",
    "RSI다이버전스": "rsiDivergence",
    "변동성돌파": "volatilityBreakout",
    "켈트너채널": "keltnerChannel",
    "볼린저스퀴즈": "bollingerSqueeze",
    "슈퍼트렌드": "superTrend",
    "일목균형표": "ichimokuCloud",
    "파라볼릭SAR": "parabolicSar",
    "돈치안돌파": "donchianBreakout",
    "삼중EMA": "tripleEma",
    "MACD_RSI콤보": "macdRsiCombo",
    "추세모멘텀": "trendMomentum",
    "볼린저RSI": "bollingerRsi",
    "갭트레이딩": "gapTrading",
    "피라미딩": "pyramiding",
    "스윙트레이딩": "swingTrading",
    "스캘핑모멘텀": "scalpingMomentum",
    "바이앤홀드": "buyAndHold",
    "적립식투자": "dollarCostAverage",
}


def _getPreset(name: str):
    from tradix.easy.presets import (
        goldenCross, rsiOversold, bollingerBreakout,
        macdCross, breakout, meanReversion, trendFollowing,
        emaCross, tripleScreen, dualMomentum,
        momentumCross, rocBreakout, stochasticCross,
        williamsReversal, cciBreakout, rsiDivergence,
        volatilityBreakout, keltnerChannel, bollingerSqueeze,
        superTrend, ichimokuCloud, parabolicSar,
        donchianBreakout, tripleEma, macdRsiCombo,
        trendMomentum, bollingerRsi, gapTrading,
        pyramiding, swingTrading, scalpingMomentum,
        buyAndHold, dollarCostAverage,
    )
    presets = {
        "goldenCross": goldenCross,
        "rsiOversold": rsiOversold,
        "bollingerBreakout": bollingerBreakout,
        "macdCross": macdCross,
        "breakout": breakout,
        "meanReversion": meanReversion,
        "trendFollowing": trendFollowing,
        "emaCross": emaCross,
        "tripleScreen": tripleScreen,
        "dualMomentum": dualMomentum,
        "momentumCross": momentumCross,
        "rocBreakout": rocBreakout,
        "stochasticCross": stochasticCross,
        "williamsReversal": williamsReversal,
        "cciBreakout": cciBreakout,
        "rsiDivergence": rsiDivergence,
        "volatilityBreakout": volatilityBreakout,
        "keltnerChannel": keltnerChannel,
        "bollingerSqueeze": bollingerSqueeze,
        "superTrend": superTrend,
        "ichimokuCloud": ichimokuCloud,
        "parabolicSar": parabolicSar,
        "donchianBreakout": donchianBreakout,
        "tripleEma": tripleEma,
        "macdRsiCombo": macdRsiCombo,
        "trendMomentum": trendMomentum,
        "bollingerRsi": bollingerRsi,
        "gapTrading": gapTrading,
        "pyramiding": pyramiding,
        "swingTrading": swingTrading,
        "scalpingMomentum": scalpingMomentum,
        "buyAndHold": buyAndHold,
        "dollarCostAverage": dollarCostAverage,
    }
    resolved = STRATEGY_MAP.get(name, name)
    fn = presets.get(resolved)
    if fn is None:
        console.print(f"[red]Unknown strategy: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(sorted(presets.keys()))}[/dim]")
        console.print("[dim]Run 'tradix list' to see all strategies.[/dim]")
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
    style: str = typer.Option("modern", "--style", help="Display style (modern/bloomberg/minimal)"),
):
    """Run a backtest with a preset strategy.

    \b
    Examples:
        tradix backtest AAPL
        tradix backtest 삼성전자 -s goldenCross -p 5년 --chart
        tradix backtest AAPL -s rsiOversold --dashboard
        tradix backtest AAPL -s bollingerSqueeze --style bloomberg
    """
    from tradix.easy import backtest as _backtest
    from tradix.tui.console import printResult
    from tradix.tui.charts import plotEquityCurve, plotDashboard

    preset_fn = _getPreset(strategy)
    strat = preset_fn()

    console.print(f"\n[bold cyan]Running backtest...[/bold cyan] {symbol} / {strategy} / {period}\n")

    result = _backtest(symbol, strat, period=period, initialCash=cash)

    if dashboard:
        plotDashboard(result, lang=lang)
    else:
        printResult(result, lang=lang, style=style)
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
        tradix optimize AAPL -s goldenCross
        tradix optimize 삼성전자 -s macdCross -m totalReturn
    """
    from tradix.easy import optimize as _optimize
    from tradix.tui.console import printResult, printComparison
    from tradix.tui.progress import optimizeProgress
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
        tradix chart AAPL
        tradix chart 삼성전자 -p 6개월 -n 60
    """
    from tradix.easy.api import _resolveSymbol, _resolvePeriod
    from tradix.datafeed.fdr import FinanceDataReaderFeed
    from tradix.tui.charts import plotCandlestick

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
        tradix compare AAPL
        tradix compare 삼성전자 -s goldenCross,rsiOversold,macdCross -l ko
    """
    from tradix.easy import backtest as _backtest
    from tradix.tui.console import printComparison

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
    """Show Tradix version."""
    from tradix.version import CURRENT_VERSION
    console.print(f"[bold cyan]Tradix[/bold cyan] v{CURRENT_VERSION}")


@app.command(name="list")
def list_strategies():
    """List all 33 available preset strategies."""
    from rich.table import Table
    from rich import box

    table = Table(title="Available Strategies (33)", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Name", style="bold white", min_width=20)
    table.add_column("Category", style="dim")
    table.add_column("Description")

    table.add_row("goldenCross", "Trend", "SMA crossover (fast/slow)")
    table.add_row("emaCross", "Trend", "EMA crossover")
    table.add_row("tripleEma", "Trend", "Triple EMA crossover")
    table.add_row("trendFollowing", "Trend", "ADX-filtered trend with trailing stop")
    table.add_row("superTrend", "Trend", "Supertrend indicator reversal")
    table.add_row("ichimokuCloud", "Trend", "Ichimoku cloud breakout")
    table.add_row("parabolicSar", "Trend", "Parabolic SAR reversal")
    table.add_row("donchianBreakout", "Trend", "Donchian channel breakout")
    table.add_row("breakout", "Trend", "Channel breakout (Turtle Trading)")

    table.add_row("rsiOversold", "Momentum", "RSI oversold/overbought reversal")
    table.add_row("macdCross", "Momentum", "MACD signal line crossover")
    table.add_row("stochasticCross", "Momentum", "Stochastic K/D crossover")
    table.add_row("williamsReversal", "Momentum", "Williams %R reversal")
    table.add_row("cciBreakout", "Momentum", "CCI overbought/oversold breakout")
    table.add_row("rsiDivergence", "Momentum", "RSI divergence detection")
    table.add_row("momentumCross", "Momentum", "Momentum zero-line crossover")
    table.add_row("rocBreakout", "Momentum", "Rate of Change breakout")

    table.add_row("bollingerBreakout", "Volatility", "Bollinger band breakout")
    table.add_row("bollingerSqueeze", "Volatility", "Bollinger squeeze expansion")
    table.add_row("keltnerChannel", "Volatility", "Keltner channel breakout")
    table.add_row("volatilityBreakout", "Volatility", "ATR-based volatility breakout")
    table.add_row("meanReversion", "Volatility", "Bollinger mean reversion")

    table.add_row("tripleScreen", "Combo", "Elder's triple screen system")
    table.add_row("dualMomentum", "Combo", "Absolute + relative momentum")
    table.add_row("macdRsiCombo", "Combo", "MACD + RSI combined signal")
    table.add_row("trendMomentum", "Combo", "Trend + momentum filter")
    table.add_row("bollingerRsi", "Combo", "Bollinger + RSI combined")

    table.add_row("gapTrading", "Special", "Gap up/down trading")
    table.add_row("pyramiding", "Special", "Pyramiding position building")
    table.add_row("swingTrading", "Special", "Swing high/low trading")
    table.add_row("scalpingMomentum", "Special", "Short-term momentum scalping")
    table.add_row("buyAndHold", "Special", "Passive buy and hold")
    table.add_row("dollarCostAverage", "Special", "Dollar cost averaging")

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
