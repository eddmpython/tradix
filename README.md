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

> `backtest()` auto-downloads data via FinanceDataReader, runs the strategy with commission and slippage, and returns a full result object.

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

<br>

## ◈ Features

<details open>
<summary><b>Core Engine</b></summary>

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

</details>

<details>
<summary><b>Advanced Analytics</b></summary>

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

| Feature | Description |
|:--------|:------------|
| **3 Display Styles** | Modern (TradingView), Bloomberg (dense 4-quadrant), Minimal (hedge fund) |
| **12 Chart Types** | Equity, drawdown, candlestick, returns, seasonality, heatmap, DNA, and more |
| **Interactive Dashboard** | 5-view Textual app with keyboard navigation |
| **CLI** | `tradix backtest`, `tradix chart`, `tradix compare`, `tradix optimize`, `tradix list` |

</details>

<details>
<summary><b>Korean Market</b></summary>

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

uv add "tradix[full]"
```

### Using pip

```bash
pip install tradix

pip install "tradix[full]"
```

### From source

```bash
git clone https://github.com/eddmpython/tradix.git
cd tradix
uv sync --dev
```

**Requirements:** Python 3.9+, NumPy, Pandas

<br>

## ◈ Usage

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

```python
from tradix import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)
print(f"Return: {result.totalReturn:+.2f}%")
print(f"Sharpe: {result.sharpeRatio:.2f}")
print(f"Max DD: {result.maxDrawdown:.2f}%")
```

### Korean API

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

Benchmarked on 10 years of daily data (2,458 bars):

| Operation | Time |
|:----------|:-----|
| **SMA calculation** | 0.006ms |
| **RSI calculation** | 0.009ms |
| **MACD calculation** | 0.040ms |
| **Full backtest (single)** | 0.132ms |
| **1,000 param optimization** | 0.02s |

<br>

## ◈ Terminal UI

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

MIT License. Use it freely in personal and commercial projects.

<div align="center">

<br>

*Trade smarter. Backtest faster.*

<br>

</div>
