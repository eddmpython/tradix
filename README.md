<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero.svg">
  <img alt="Tradix — Backtest the Market" src=".github/assets/hero.svg" width="100%">
</picture>

<br>

<h3>Blazing-Fast Vectorized Backtesting Engine</h3>

<p>
<img src="https://img.shields.io/badge/33-Strategies-00d4aa?style=for-the-badge&labelColor=0f172a" alt="Strategies">
<img src="https://img.shields.io/badge/60+-Indicators-00b4d8?style=for-the-badge&labelColor=0f172a" alt="Indicators">
<img src="https://img.shields.io/badge/2-Lines%20to%20Backtest-0096c7?style=for-the-badge&labelColor=0f172a" alt="Two Lines">
</p>

<p>
<a href="https://pypi.org/project/tradix/"><img src="https://img.shields.io/pypi/v/tradix?style=for-the-badge&color=00d4aa&labelColor=0f172a&logo=pypi&logoColor=white" alt="PyPI"></a>
<a href="https://pypi.org/project/tradix/"><img src="https://img.shields.io/pypi/pyversions/tradix?style=for-the-badge&labelColor=0f172a&logo=python&logoColor=white" alt="Python"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-00d4aa?style=for-the-badge&labelColor=0f172a" alt="License"></a>
<img src="https://img.shields.io/badge/Tests-87%20passed-00d4aa?style=for-the-badge&labelColor=0f172a&logo=pytest&logoColor=white" alt="Tests">
</p>

<br>

<a href="#-quick-start">Quick Start</a> ·
<a href="#-why-tradix">Why Tradix?</a> ·
<a href="#-features">Features</a> ·
<a href="#-installation">Installation</a> ·
<a href="#-api-reference">API Reference</a> ·
<a href="README_KR.md">한국어</a>

</div>

<br>

Tradix is a high-performance backtesting engine built for quantitative traders and researchers. Powered by pure NumPy vectorization, it runs 1,000 parameter optimizations in under a second. No boilerplate, no configuration files, no data pipeline setup — just install and backtest.

The library ships with 33 battle-tested strategies, 60+ technical indicators, and a built-in data feed that auto-downloads from global exchanges including KRX (Korea Exchange). Whether you're prototyping a Golden Cross on Samsung Electronics or stress-testing a multi-factor portfolio against Black Swan scenarios, Tradix handles the entire pipeline: data → signals → execution → analytics → visualization.

> **Two lines.** `backtest` · `show` — that's the entire workflow.

```python
from tradix import backtest, goldenCross

result = backtest("AAPL", goldenCross())
result.show()
```

<br>

## ◈ Quick Start

```bash
pip install tradix
```

```python
from tradix import backtest, goldenCross

result = backtest("005930", goldenCross())
result.show()
result.show(style="bloomberg")
```

`backtest()` auto-downloads price data via FinanceDataReader, applies the strategy with realistic commission and slippage modeling, and returns a comprehensive result object. Pass any ticker symbol — US stocks, Korean stocks, ETFs, indices, and crypto are all supported out of the box.

```
=== Backtest Result ===
Strategy: GoldenCross
Period: 2020-01-02 ~ 2024-12-30
Initial: 10,000,000 KRW
Final:   14,230,000 KRW
Return:  +42.30%
Sharpe:  1.23
Max DD:  -12.45%
Trades:  18
Win Rate: 61.1%
```

<br>

## ◈ Why Tradix?

Most backtesting frameworks require hundreds of lines of boilerplate — data loaders, strategy classes, broker configs, result parsers. Tradix eliminates all of that. Here's how it compares:

<table>
<tr><td>

| Dimension | Tradix | Backtrader | Zipline | bt |
|:--|:--:|:--:|:--:|:--:|
| **Zero-config backtest** | ✅ | ❌ | ❌ | ❌ |
| **2-line API** | ✅ | ❌ | ❌ | ❌ |
| **Vectorized engine** | ✅ | ❌ | ❌ | ✅ |
| **1K optimizations < 1s** | ✅ | ❌ | ❌ | ❌ |
| **Built-in data feed** | ✅ | ❌ | ❌ | ❌ |
| **33 preset strategies** | ✅ | ❌ | ❌ | ❌ |
| **Terminal dashboard** | ✅ | ❌ | ❌ | ❌ |
| **Korean market native** | ✅ | ❌ | ❌ | ❌ |
| **Walk-forward analysis** | ✅ | ❌ | ❌ | ❌ |
| **Strategy DNA analysis** | ✅ | ❌ | ❌ | ❌ |

</td></tr>
</table>

> **Three pillars.** `numpy` · `pandas` · `scipy` — that's the entire foundation. No C extensions, no binary dependencies, no compilation step.

<br>

## ◈ Features

<details open>
<summary><b>Core Engine</b></summary>

<br>

The heart of Tradix is a dual-engine architecture: an event-driven engine for complex strategies with bar-by-bar logic, and a vectorized engine for lightning-fast batch operations. Both engines share the same indicator library and produce identical results.

| Feature | Description |
|:--------|:------------|
| **Vectorized Engine** | NumPy-powered core, 100x faster than event-driven loops |
| **60+ Indicators** | SMA, EMA, RSI, MACD, Bollinger, ATR, Ichimoku, Supertrend, StochRSI, KDJ, and more |
| **33 Preset Strategies** | Trend, momentum, oscillator, volatility, multi-indicator, buy & hold, DCA |
| **Strategy Builder** | Declarative method chaining — no subclassing needed |
| **Walk-Forward** | Built-in overfitting prevention with time-series cross-validation |
| **Optimization** | Grid search and random search with any metric |
| **Multi-Asset** | Backtest across multiple symbols with rebalancing |
| **Realistic Simulation** | Commission, slippage, fill logic, position sizing |
| **OCO / Bracket / Trailing Stop** | Advanced order types for automated stop-loss and take-profit management |

</details>

<details>
<summary><b>Multi-Source Data Feeds</b></summary>

<br>

Connect to global markets without writing a single data loader. Tradix ships with three data feed backends — FinanceDataReader for Korean/US/JP equities, Yahoo Finance for global ETFs and indices, and CCXT for 100+ cryptocurrency exchanges. All feeds share the same interface and include Parquet caching for offline replay.

| Feature | Description |
|:--------|:------------|
| **FinanceDataReader** | KRX, US, JP stocks with auto-caching |
| **Yahoo Finance** | Global equities, ETFs, indices, crypto pairs via yfinance |
| **CCXT (100+ Exchanges)** | Binance, Bybit, Upbit, Coinbase — 1m to 1w candles with auto-pagination |
| **Parquet Cache** | Offline-first with configurable expiration |
| **Unified Interface** | All feeds implement the same DataFeed API |

</details>

<details>
<summary><b>Advanced Analytics</b></summary>

<br>

Go beyond simple returns. Tradix provides institutional-grade analytics that dissect every aspect of your strategy — from 12-dimensional DNA fingerprinting to Black Swan resilience scoring. Understand not just *how much* your strategy made, but *why* it works and *when* it might fail.

| Feature | Description |
|:--------|:------------|
| **Strategy DNA** | 12-dimensional strategy fingerprinting |
| **Black Swan Defense** | Extreme event resilience scoring (0-100) |
| **Health Score** | Overfitting risk, parameter stability diagnostics |
| **What-If Simulator** | Commission, slippage, capital sensitivity analysis |
| **Drawdown Simulator** | Historical worst-case scenario generation |
| **Seasonality Analyzer** | Monthly, weekday, quarterly pattern discovery |
| **Correlation Matrix** | Multi-strategy correlation and clustering |
| **Trading Journal** | Auto trade diary with MFE/MAE analytics |
| **Strategy Leaderboard** | Multi-strategy ranking with badge system |

</details>

<details>
<summary><b>Innovative Analytics</b></summary>

<br>

Tradix pushes the boundary of what a backtesting library can do. Monte Carlo simulations with 10,000 paths estimate your ruin probability. Fractal analysis reveals whether your market is trending or mean-reverting. Regime detection identifies hidden market states so you can adapt your strategy dynamically.

| Feature | Description |
|:--------|:------------|
| **Monte Carlo** | 10K-path bootstrap with ruin probability and confidence bands |
| **Fractal Analysis** | Hurst exponent for market character classification |
| **Regime Detector** | GMM-based probabilistic regime detection with transition matrix |
| **Information Theory** | Shannon entropy, mutual information for signal quality |
| **Portfolio Stress** | 6 crisis scenarios (crash, volatility spike, rate shock, etc.) |

</details>

<details>
<summary><b>Terminal UI — TradingView-Inspired</b></summary>

<br>

No browser needed. Tradix renders professional trading dashboards directly in your terminal using Rich and Plotext. Choose from three display styles inspired by TradingView, Bloomberg Terminal, and hedge fund reports. The interactive Textual dashboard provides a full 5-view experience with keyboard navigation.

| Feature | Description |
|:--------|:------------|
| **3 Display Styles** | Modern (TradingView), Bloomberg (dense 4-quadrant), Minimal (hedge fund) |
| **14 Chart Types** | Equity, drawdown, candlestick, returns, seasonality, heatmap, DNA, rolling metrics, and more |
| **Plotly Candlestick** | Interactive OHLCV chart with volume, SMA/EMA overlays, and trade markers |
| **Rolling Metrics** | Time-varying Sharpe, Sortino, and volatility visualization |
| **Interactive Dashboard** | 5-view Textual app with keyboard navigation |
| **CLI** | `tradix backtest`, `tradix chart`, `tradix compare`, `tradix optimize`, `tradix list` |

</details>

<details>
<summary><b>Korean Market</b></summary>

<br>

Tradix is the first backtesting engine built with native Korean market support. Transaction tax (0.18%), brokerage fees, and KRX stock name mapping are all built-in. You can even write your entire strategy in Korean — function names, variable names, and output are all available in 한국어.

| Feature | Description |
|:--------|:------------|
| **Native Support** | Built-in transaction tax (0.18%), brokerage fees, KRX stock mapping |
| **Korean API** | Full Korean function names: `백테스트("삼성전자", 골든크로스())` |

</details>

<br>

## ◈ Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager.

```bash
uv init my-backtest && cd my-backtest
uv add tradix

uv add "tradix[full]"       # + Plotly, statsmodels, scikit-learn
uv add "tradix[yahoo]"      # + Yahoo Finance global data
uv add "tradix[crypto]"     # + CCXT 100+ crypto exchanges
uv add "tradix[tui]"        # + Textual interactive dashboard
```

### Using pip

```bash
pip install tradix            # Core (NumPy + Pandas + Rich)
pip install "tradix[full]"    # + Plotly, statsmodels, scikit-learn
pip install "tradix[yahoo]"   # + Yahoo Finance global data
pip install "tradix[crypto]"  # + CCXT 100+ crypto exchanges
pip install "tradix[tui]"     # + Textual interactive dashboard
```

### From source

```bash
git clone https://github.com/eddmpython/tradix.git
cd tradix
uv sync --dev
uv run pytest
```

**Requirements:** Python 3.9+ · NumPy · Pandas · Rich · Plotext · Typer

<br>

## ◈ Usage

Tradix offers multiple levels of abstraction. Start with the 2-line easy API, graduate to the declarative strategy builder for custom logic, or drop down to the full event-driven engine when you need bar-by-bar control.

### Declarative Strategy Builder

```python
from tradix import QuickStrategy, backtest, sma, rsi, crossover, crossunder

strategy = (
    QuickStrategy("MomentumRSI")
    .buyWhen(crossover(sma(10), sma(30)))
    .buyWhen(rsi(14) < 30)
    .sellWhen(crossunder(sma(10), sma(30)))
    .sellWhen(rsi(14) > 70)
    .stopLoss(5)
    .takeProfit(15)
)

result = backtest("AAPL", strategy)
result.show()
```

### Parameter Optimization

Find optimal strategy parameters by sweeping across a parameter grid. Tradix runs all combinations using the vectorized engine — 1,000 combinations finish in under a second.

```python
from tradix import voptimize

best = voptimize(
    "005930",
    "goldenCross",
    fast=(5, 20, 5),
    slow=(20, 60, 10),
    metric="sharpeRatio"
)

print(f"Best: fast={best['best']['params']['fast']}, slow={best['best']['params']['slow']}")
print(f"Sharpe: {best['best']['metric']:.2f}")
```

### Vectorized Mode (100x Faster)

Skip the event loop entirely. The vectorized engine processes the entire price history as NumPy arrays, delivering results in microseconds.

```python
from tradix import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)
print(f"Return: {result.totalReturn:+.2f}%")
print(f"Sharpe: {result.sharpeRatio:.2f}")
print(f"Max DD: {result.maxDrawdown:.2f}%")
```

### Korean API (한국어 API)

Every function, every strategy, every output — available in native Korean. No wrappers, no translation layers. First-class Korean market citizen.

```python
from tradix import 백테스트, 골든크로스

결과 = 백테스트("삼성전자", 골든크로스())
결과.보기()
결과.차트()
```

<details>
<summary><b>More Examples</b> — Class-based strategy, walk-forward, multi-asset, risk</summary>

### Class-Based Strategy

```python
from tradix import Strategy, Bar, BacktestEngine
from tradix.datafeed import FinanceDataReaderFeed

class DualMomentum(Strategy):
    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30

    def onBar(self, bar: Bar):
        fast = self.sma(self.fastPeriod)
        slow = self.sma(self.slowPeriod)

        if fast is None or slow is None:
            return

        if fast > slow and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)
        elif fast < slow and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)

data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")
engine = BacktestEngine(data, DualMomentum(), initialCash=10_000_000)
result = engine.run()
result.show()
```

### Walk-Forward Analysis

```python
from tradix import Strategy, Bar, WalkForwardAnalyzer, ParameterSpace
from tradix.datafeed import FinanceDataReaderFeed

def createStrategy(params):
    class MySma(Strategy):
        def initialize(self):
            self.fast = params["fast"]
            self.slow = params["slow"]
        def onBar(self, bar: Bar):
            f, s = self.sma(self.fast), self.sma(self.slow)
            if f and s:
                if f > s and not self.hasPosition(bar.symbol):
                    self.buy(bar.symbol)
                elif f < s and self.hasPosition(bar.symbol):
                    self.closePosition(bar.symbol)
    return MySma()

data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")
space = ParameterSpace()
space.addInt("fast", 5, 20, step=5)
space.addInt("slow", 20, 60, step=10)

wfa = WalkForwardAnalyzer(
    data=data,
    strategyFactory=createStrategy,
    parameterSpace=space,
    inSampleMonths=12,
    outOfSampleMonths=3,
)

result = wfa.run()
print(f"Robustness: {result.robustnessRatio:.1%}")
```

### Multi-Asset Portfolio

```python
from tradix import MultiAssetEngine, MultiAssetStrategy

class EqualWeight(MultiAssetStrategy):
    def onBars(self, bars):
        self.rebalance({
            "005930": 0.4,
            "000660": 0.3,
            "035420": 0.3,
        })

engine = MultiAssetEngine(strategy=EqualWeight())
result = engine.run()
result.show()
```

### Risk Simulation

```python
import pandas as pd
from tradix import backtest, goldenCross
from tradix.risk import RiskSimulator, VaRMethod

result = backtest("005930", goldenCross())
returns = pd.DataFrame({"returns": pd.Series(result.equityCurve).pct_change().dropna()})

simulator = RiskSimulator()
simulator.fit(returns)

var95, cvar95 = simulator.calcVaR(confidence=0.95, method=VaRMethod.HISTORICAL)
print(f"95% VaR: {var95:.2%}")
print(f"CVaR: {cvar95:.2%}")

mc = simulator.monteCarloSimulation(horizon=252, nSim=10000)
```

</details>

<br>

## ◈ Strategies

33 production-ready strategies organized by trading style. Each strategy is a callable that returns a configured strategy object — no subclassing, no boilerplate. Pass parameters to customize, or use the defaults.

<details open>
<summary><b>Trend Following (9)</b></summary>

| Strategy | Description |
|:---------|:------------|
| **`goldenCross()`** | SMA crossover (fast/slow) |
| **`emaCross()`** | EMA crossover |
| **`tripleEma()`** | Triple EMA crossover |
| **`trendFollowing()`** | ADX-filtered trend following with trailing stop |
| **`superTrend()`** | Supertrend indicator reversal |
| **`ichimokuCloud()`** | Ichimoku cloud breakout |
| **`parabolicSar()`** | Parabolic SAR reversal |
| **`donchianBreakout()`** | Donchian channel breakout |
| **`breakout()`** | Channel breakout (Turtle Trading) |

</details>

<details>
<summary><b>Momentum & Oscillator (8)</b></summary>

| Strategy | Description |
|:---------|:------------|
| **`rsiOversold()`** | RSI reversal (oversold/overbought) |
| **`macdCross()`** | MACD histogram crossover |
| **`stochasticCross()`** | Stochastic K/D crossover |
| **`williamsReversal()`** | Williams %R reversal |
| **`cciBreakout()`** | CCI overbought/oversold breakout |
| **`rsiDivergence()`** | RSI divergence detection |
| **`momentumCross()`** | Momentum zero-line crossover |
| **`rocBreakout()`** | Rate of Change breakout |

</details>

<details>
<summary><b>Volatility (5)</b></summary>

| Strategy | Description |
|:---------|:------------|
| **`bollingerBreakout()`** | Bollinger band breakout |
| **`bollingerSqueeze()`** | Bollinger squeeze expansion |
| **`keltnerChannel()`** | Keltner channel breakout |
| **`volatilityBreakout()`** | ATR-based volatility breakout |
| **`meanReversion()`** | Bollinger mean reversion |

</details>

<details>
<summary><b>Multi-Indicator (5)</b></summary>

| Strategy | Description |
|:---------|:------------|
| **`tripleScreen()`** | Elder's triple screen system |
| **`dualMomentum()`** | Absolute + relative momentum |
| **`macdRsiCombo()`** | MACD + RSI combined signal |
| **`trendMomentum()`** | Trend + momentum filter |
| **`bollingerRsi()`** | Bollinger + RSI combined |

</details>

<details>
<summary><b>Special (6)</b></summary>

| Strategy | Description |
|:---------|:------------|
| **`gapTrading()`** | Gap up/down trading |
| **`pyramiding()`** | Pyramiding position building |
| **`swingTrading()`** | Swing high/low trading |
| **`scalpingMomentum()`** | Short-term momentum scalping |
| **`buyAndHold()`** | Passive buy and hold |
| **`dollarCostAverage()`** | Dollar cost averaging |

</details>

<br>

## ◈ Indicators

60+ technical indicators implemented in pure NumPy. Each indicator is available as both a standalone function and a strategy-builder condition. All calculations are vectorized — even complex indicators like Ichimoku Cloud run in microseconds.

| Category | Indicators |
|:---------|:-----------|
| **Moving Averages** | `sma` `ema` `wma` `hma` `tema` `dema` `vwma` `alma` |
| **Momentum** | `rsi` `macd` `stochastic` `roc` `momentum` `cci` `williamsR` `cmo` `stochasticRsi` `kdj` `awesomeOscillator` `ultimateOscillator` |
| **Volatility** | `atr` `bollinger` `keltner` `donchian` `bollingerPercentB` `bollingerWidth` |
| **Volume** | `obv` `vwap` `mfi` `adl` `chaikin` `emv` `forceIndex` `nvi` `pvi` `vroc` `pvt` `klingerOscillator` |
| **Trend** | `adx` `supertrend` `psar` `ichimoku` `trix` `dpo` `linearRegression` |
| **Price** | `pivotPoints` `fibonacciRetracement` `zigzag` `elderRay` `twap` |
| **Other** | `ulcer` `percentChange` `highest` `lowest` |

<br>

## ◈ Performance

Tradix is built for speed. The vectorized engine processes entire price histories as contiguous NumPy arrays, eliminating Python loop overhead. Benchmarked on 10 years of daily data (2,458 bars):

| Operation | Time |
|:----------|:-----|
| **SMA calculation** | 0.006ms |
| **RSI calculation** | 0.009ms |
| **MACD calculation** | 0.040ms |
| **Full backtest (single)** | 0.132ms |
| **1,000 param optimization** | 0.02s |

<br>

## ◈ Terminal UI

Professional trading dashboards rendered directly in your terminal. No browser, no GUI toolkit, no Jupyter dependency. Tradix uses Rich for styled tables and panels, Plotext for ASCII charts, and optionally Textual for a full interactive experience.

### Display Styles

```python
result.show()                     # Modern — TradingView metric cards
result.show(style="bloomberg")    # Bloomberg — dense 4-quadrant layout
result.show(style="minimal")      # Minimal — clean hedge fund report
```

### CLI

```bash
tradix backtest AAPL -s goldenCross --dashboard
tradix backtest AAPL -s bollingerSqueeze --style bloomberg
tradix chart AAPL -n 60
tradix compare AAPL -s goldenCross,rsiOversold
tradix optimize AAPL -s goldenCross
tradix list
```

### Charts (12 Types)

```python
from tradix.tui.charts import (
    plotEquityCurve, plotDrawdown, plotCandlestick, plotReturns,
    plotSeasonality, plotMonthlyHeatmap, plotRollingMetrics,
    plotTradeScatter, plotTradeMarkers, plotCorrelationBars,
    plotStrategyDna, plotDashboard,
)

plotEquityCurve(result, smaPeriods=[20, 60])
plotDrawdown(result)
plotCandlestick(df, smaPeriods=[5, 20])
plotDashboard(result, lang="ko")
```

### Interactive Dashboard

```bash
pip install tradix[tui]
```

```python
from tradix.tui.dashboard import launchDashboard
launchDashboard(result)
```

<br>

## ◈ Advanced Analytics

Tradix includes a full suite of institutional-grade analytics. Each analyzer takes a backtest result and returns a structured report with scores, metrics, and actionable insights.

```python
from tradix import (
    StrategyDnaAnalyzer, BlackSwanAnalyzer, StrategyHealthAnalyzer,
    WhatIfSimulator, DrawdownSimulator, SeasonalityAnalyzer,
    CorrelationAnalyzer, TradingJournal, StrategyLeaderboard,
)

result = backtest("005930", goldenCross())

dna = StrategyDnaAnalyzer().analyze(result)
printStrategyDna(dna)

health = StrategyHealthAnalyzer().analyze(result)
printHealthScore(health)

blackSwan = BlackSwanAnalyzer().analyze(result)
printBlackSwanScore(blackSwan)
```

```python
from tradix import (
    MonteCarloStressAnalyzer, FractalAnalyzer, RegimeDetector,
    InformationTheoryAnalyzer, PortfolioStressAnalyzer,
)

mc = MonteCarloStressAnalyzer().analyze(result, paths=10000)
print(f"Ruin probability: {mc.ruinProbability:.2%}")

fractal = FractalAnalyzer().analyze(result)
print(f"Hurst: {fractal.hurstExponent:.3f} → {fractal.marketCharacter}")

regime = RegimeDetector().analyze(result)
print(f"Current regime: {regime.currentRegime}")
```

<br>

## ◈ API Reference

<details>
<summary><b>Core Functions</b></summary>

| Function | Description |
|:---------|:------------|
| **`backtest(symbol, strategy)`** | Run a backtest with a preset or custom strategy |
| **`vbacktest(symbol, strategy, **params)`** | Run a vectorized backtest |
| **`voptimize(symbol, strategy, **ranges)`** | Grid search parameter optimization |
| **`BacktestEngine(data, strategy)`** | Event-driven backtest engine |
| **`VectorizedEngine(initialCash)`** | Vectorized backtest engine |
| **`QuickStrategy(name)`** | Declarative strategy builder |

</details>

<details>
<summary><b>Strategy Builder Methods</b></summary>

| Method | Description |
|:-------|:------------|
| **`.buyWhen(condition)`** | Add buy condition |
| **`.sellWhen(condition)`** | Add sell condition |
| **`.stopLoss(pct)`** | Set stop loss percentage |
| **`.takeProfit(pct)`** | Set take profit percentage |
| **`.trailingStop(pct)`** | Set trailing stop percentage |

</details>

<details>
<summary><b>Condition Builders</b></summary>

| Function | Returns |
|:---------|:--------|
| **`sma(period)`** | SMA indicator |
| **`ema(period)`** | EMA indicator |
| **`rsi(period)`** | RSI indicator |
| **`macd(fast, slow, signal)`** | MACD indicator |
| **`bollinger(period, std)`** | Bollinger Bands |
| **`atr(period)`** | ATR indicator |
| **`price()`** | Current price |
| **`crossover(fast, slow)`** | Crossover condition |
| **`crossunder(fast, slow)`** | Crossunder condition |

Indicators support comparison operators: `sma(10) > sma(30)`, `rsi(14) < 30`

</details>

<br>

## ◈ Architecture

Tradix follows a modular architecture. Each subsystem is independently importable, so you can use just the indicators, just the risk engine, or the full pipeline.

```
tradix/
├── engine.py              # Core backtest engine
├── multiAssetEngine.py    # Multi-asset portfolio engine
├── strategy/              # Strategy base + 60+ indicators + ensemble
├── easy/                  # 2-line API, presets, Korean API
├── vectorized/            # Vectorized engine + indicators (Pure NumPy)
├── datafeed/              # Data feeds (FinanceDataReader + Parquet cache)
├── broker/                # Commission, slippage, fill simulation
├── risk/                  # Position sizing, VaR, Monte Carlo
├── optimize/              # Grid / random search optimizer
├── walkforward/           # Walk-forward analysis
├── analytics/             # Strategy DNA, Black Swan, Health Score, etc.
├── portfolio/             # Portfolio tracking + optimization
├── quant/                 # Factor analysis, statistical arbitrage
├── signals/               # Signal prediction + adaptive signals
├── advisor/               # Market regime + strategy recommendation
├── entities/              # Bar, Order, Position, Trade
├── events/                # Event system
├── tui/                   # Terminal UI (Rich + Plotext + Textual)
├── cli.py                 # Typer CLI
└── tests/                 # 87 tests
```

<br>

## ◈ Contributing

Contributions are welcome. Fork the repo, create a feature branch, and submit a pull request.

```bash
git clone https://github.com/eddmpython/tradix.git
cd tradix
uv sync --dev
uv run pytest
```

<br>

## ◈ Support

If Tradix helps your trading research, consider supporting the project:

<p>
  <a href="https://buymeacoffee.com/eddmpython" target="_blank">
    <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
  </a>
</p>

<br>

## ◈ License

[MIT](LICENSE) — Use freely in personal and commercial projects.

<div align="center">

<br>

*Trade smarter. Backtest faster.*

<br>

</div>
