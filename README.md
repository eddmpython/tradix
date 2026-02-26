<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:161b22,100:1f6feb&height=220&text=TradeX&fontSize=80&fontColor=58a6ff&animation=fadeIn&fontAlignY=35&desc=Blazing-fast%20backtesting%20engine%20for%20quantitative%20trading&descSize=18&descColor=8b949e&descAlignY=55" width="100%"/>

<p>
  <a href="https://pypi.org/project/tradex-backtest/"><img src="https://img.shields.io/pypi/v/tradex-backtest?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"></a>
  <a href="https://buymeacoffee.com/eddmpython"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=flat-square&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p>
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#terminal-ui">Terminal UI</a> &middot;
  <a href="#api-reference">API Reference</a> &middot;
  <a href="README_KR.md">한국어</a>
</p>

</div>

---

Tradex is a high-performance backtesting library built from the ground up for speed and simplicity. Powered by NumPy vectorized operations, it runs **1,000 parameter optimizations in 0.02 seconds** — no Numba, no C extensions, just pure Python speed.

```python
from tradex import backtest, goldenCross

result = backtest("AAPL", goldenCross())
print(result.summary())
```

That's it. Two lines to a full backtest with commission, slippage, and performance metrics.

## Why Tradex?

| Library | Issue |
|---------|-------|
| **VectorBT** | Free version discontinued, Pro is $29/mo |
| **Backtesting.py** | AGPL license, limited portfolio support |
| **Lumibot** | Heavy, hard to customize |
| **LEAN** | C#-based, requires Docker, steep learning curve |
| **bt** | No position sizing, no transaction costs |

Tradex gives you all of the above — **for free, under MIT license** — with an API designed to get out of your way.

## Features

### Core Engine
- **Vectorized Engine** — NumPy-powered core, 100x faster than event-driven loops
- **60+ Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger, ATR, Ichimoku, Supertrend, Stochastic RSI, KDJ, and more
- **33 Preset Strategies** — Trend, momentum, oscillator, volatility, multi-indicator, buy & hold, DCA
- **Declarative Strategy Builder** — Build strategies with method chaining, no subclassing needed
- **Walk-Forward Analysis** — Built-in overfitting prevention with time-series cross-validation
- **Parameter Optimization** — Grid search and random search with any metric
- **Multi-Asset Portfolios** — Backtest across multiple symbols with rebalancing
- **Realistic Simulation** — Commission models, slippage, fill logic, position sizing

### Advanced Analytics
- **Strategy DNA** — 12-dimensional strategy fingerprinting (trend sensitivity, mean reversion, volatility preference, etc.)
- **Black Swan Defense** — Extreme event resilience scoring (0-100) with crisis period breakdown
- **Strategy Health Score** — Overfitting risk, parameter stability, performance consistency diagnostics
- **What-If Simulator** — Commission, slippage, capital, and timing sensitivity analysis
- **Drawdown Simulator** — Historical worst-case drawdown scenario generation
- **Seasonality Analyzer** — Monthly, weekday, and quarterly pattern discovery
- **Correlation Matrix** — Multi-strategy correlation analysis and clustering
- **Trading Journal** — Automatic trade diary with MFE/MAE analytics
- **Strategy Leaderboard** — Multi-strategy ranking with composite scoring and badge system

### Innovative Analytics (World-First)
- **Monte Carlo Stress Test** — 10,000-path bootstrap simulation with ruin probability, confidence bands, Sharpe/MDD distributions
- **Fractal Analysis** — Hurst Exponent and fractal dimension for market character classification (trending/random/mean-reverting)
- **Regime Detector** — GMM-based probabilistic regime detection with transition matrix and per-regime performance decomposition
- **Information Theory** — Shannon entropy, mutual information, and transfer entropy for signal quality measurement
- **Portfolio Stress Test** — 6 hypothetical crisis scenarios (market crash, volatility spike, rate shock, liquidity crisis, flash crash, correlation breakdown)

### Risk & Quant
- **Risk Analytics** — VaR, CVaR, Monte Carlo simulation, Sharpe, Sortino, Calmar
- **Factor Analysis** — Multi-factor models, statistical arbitrage, pair trading

### Terminal UI (TradingView-inspired)
- **3 Display Styles** — Modern (TradingView cards), Bloomberg (dense 4-quadrant), Minimal (hedge fund report)
- **12 Chart Types** — Equity curve, drawdown, candlestick+volume, return distribution, seasonality, rolling metrics, trade scatter, correlation bars, strategy DNA, monthly heatmap, trade markers, equity overlay
- **Interactive Dashboard** — 5-view Textual app (Overview, Metrics, Trades, Charts, Compare) with keyboard navigation
- **CLI** — `tradex backtest`, `tradex chart`, `tradex compare`, `tradex optimize`, `tradex list`

### Korean Market
- **Korean Market Native** — Built-in transaction tax (0.18%), brokerage fees, KRX stock mapping
- **Korean Language API** — Full Korean function names: `백테스트("삼성전자", 골든크로스())`

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the fastest Python package manager. If you don't have it yet:

```bash
# Install uv
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install Tradex:

```bash
# Create a new project and add tradex
uv init my-backtest && cd my-backtest
uv add tradex-backtest

# Or with all optional dependencies (scipy, plotly, statsmodels, scikit-learn)
uv add "tradex-backtest[full]"
```

### Using pip

```bash
pip install tradex-backtest

# With full features
pip install "tradex-backtest[full]"
```

### From source (development)

```bash
git clone https://github.com/eddmpython/tradex.git
cd tradex

# Using uv
uv sync --dev

# Using pip
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, NumPy, Pandas

## Quick Start

### 1. Two-Line Backtest

The simplest way to run a backtest with a preset strategy:

```python
from tradex import backtest, goldenCross

result = backtest("005930", goldenCross())  # Samsung Electronics
print(result.summary())
```

Output:
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

### 2. Vectorized Mode (100x Faster)

For batch computation over entire price histories:

```python
from tradex import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)
print(f"Return: {result.totalReturn:+.2f}%")
print(f"Sharpe: {result.sharpeRatio:.2f}")
print(f"Max DD: {result.maxDrawdown:.2f}%")
```

### 3. Declarative Strategy Builder

Build strategies with method chaining — zero boilerplate:

```python
from tradex import QuickStrategy, backtest, sma, rsi, crossover, crossunder

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
print(result.summary())
```

### 4. Class-Based Strategy (Full Control)

For maximum flexibility, write your own strategy class:

```python
from tradex import Strategy, Bar, BacktestEngine
from tradex.datafeed import FinanceDataReaderFeed

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
print(result.summary())
```

### 5. Parameter Optimization

Find the best parameters in seconds:

```python
from tradex import voptimize

best = voptimize(
    "005930",
    "goldenCross",
    fast=(5, 20, 5),     # (start, end, step)
    slow=(20, 60, 10),
    metric="sharpeRatio"
)

print(f"Best: fast={best['best']['params']['fast']}, slow={best['best']['params']['slow']}")
print(f"Sharpe: {best['best']['metric']:.2f}")
```

### 6. Walk-Forward Analysis

Prevent overfitting with out-of-sample validation:

```python
from tradex import WalkForwardAnalyzer, ParameterSpace

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

### 7. Multi-Asset Portfolio

```python
from tradex import MultiAssetEngine, MultiAssetStrategy

class EqualWeight(MultiAssetStrategy):
    def onBars(self, bars):
        self.rebalance({
            "005930": 0.4,  # Samsung
            "000660": 0.3,  # SK Hynix
            "035420": 0.3,  # Naver
        })

engine = MultiAssetEngine(strategy=EqualWeight())
result = engine.run()
print(result.summary())
```

### 8. Risk Simulation

```python
from tradex.risk import RiskSimulator, VaRMethod

simulator = RiskSimulator()
simulator.fit(returns)

var = simulator.calcVaR(confidence=0.95, method=VaRMethod.HISTORICAL)
print(f"95% VaR: {var.var:.2%}")
print(f"CVaR: {var.cvar:.2%}")

mc = simulator.monteCarloSimulation(horizon=252, nSim=10000)
```

### 9. Korean Language API

For Korean developers — write strategies entirely in Korean:

```python
from tradex import 백테스트, 골든크로스

결과 = 백테스트("삼성전자", 골든크로스())
print(결과.요약())
```

## Preset Strategies (33)

### Trend Following
| Strategy | Description |
|----------|-------------|
| `goldenCross()` | SMA crossover (fast/slow) |
| `emaCross()` | EMA crossover |
| `tripleEma()` | Triple EMA crossover |
| `trendFollowing()` | ADX-filtered trend following with trailing stop |
| `superTrend()` | Supertrend indicator reversal |
| `ichimokuCloud()` | Ichimoku cloud breakout |
| `parabolicSar()` | Parabolic SAR reversal |
| `donchianBreakout()` | Donchian channel breakout |
| `breakout()` | Channel breakout (Turtle Trading) |

### Momentum & Oscillator
| Strategy | Description |
|----------|-------------|
| `rsiOversold()` | RSI reversal (oversold/overbought) |
| `macdCross()` | MACD histogram crossover |
| `stochasticCross()` | Stochastic K/D crossover |
| `williamsReversal()` | Williams %R reversal |
| `cciBreakout()` | CCI overbought/oversold breakout |
| `rsiDivergence()` | RSI divergence detection |
| `momentumCross()` | Momentum zero-line crossover |
| `rocBreakout()` | Rate of Change breakout |

### Volatility
| Strategy | Description |
|----------|-------------|
| `bollingerBreakout()` | Bollinger band breakout |
| `bollingerSqueeze()` | Bollinger squeeze expansion |
| `keltnerChannel()` | Keltner channel breakout |
| `volatilityBreakout()` | ATR-based volatility breakout |
| `meanReversion()` | Bollinger mean reversion |

### Multi-Indicator & Combo
| Strategy | Description |
|----------|-------------|
| `tripleScreen()` | Elder's triple screen system |
| `dualMomentum()` | Absolute + relative momentum |
| `macdRsiCombo()` | MACD + RSI combined signal |
| `trendMomentum()` | Trend + momentum filter |
| `bollingerRsi()` | Bollinger + RSI combined |

### Special
| Strategy | Description |
|----------|-------------|
| `gapTrading()` | Gap up/down trading |
| `pyramiding()` | Pyramiding position building |
| `swingTrading()` | Swing high/low trading |
| `scalpingMomentum()` | Short-term momentum scalping |
| `buyAndHold()` | Passive buy and hold |
| `dollarCostAverage()` | Dollar cost averaging |

## Indicators (60+)

| Category | Indicators |
|----------|-----------|
| **Moving Averages** | `sma` `ema` `wma` `hma` `tema` `dema` `vwma` `alma` |
| **Momentum** | `rsi` `macd` `stochastic` `roc` `momentum` `cci` `williamsR` `cmo` `stochasticRsi` `kdj` `awesomeOscillator` `ultimateOscillator` |
| **Volatility** | `atr` `bollinger` `keltner` `donchian` `bollingerPercentB` `bollingerWidth` |
| **Volume** | `obv` `vwap` `mfi` `adl` `chaikin` `emv` `forceIndex` `nvi` `pvi` `vroc` `pvt` `klingerOscillator` |
| **Trend** | `adx` `supertrend` `psar` `ichimoku` `trix` `dpo` `linearRegression` |
| **Price** | `pivotPoints` `fibonacciRetracement` `zigzag` `elderRay` `twap` |
| **Other** | `ulcer` `percentChange` `highest` `lowest` |

## Performance

Benchmarked on 10 years of daily data (2,458 bars):

| Operation | Time |
|-----------|------|
| SMA calculation | **0.006ms** |
| RSI calculation | **0.009ms** |
| MACD calculation | **0.040ms** |
| Full backtest (single) | **0.132ms** |
| 1,000 param optimization | **0.02s** |

## API Reference

### Core

| Class/Function | Description |
|---------------|-------------|
| `backtest(symbol, strategy)` | Run a backtest with a preset or custom strategy |
| `vbacktest(symbol, strategy, **params)` | Run a vectorized backtest |
| `voptimize(symbol, strategy, **ranges)` | Grid search parameter optimization |
| `BacktestEngine(data, strategy)` | Event-driven backtest engine |
| `VectorizedEngine(initialCash)` | Vectorized backtest engine |
| `QuickStrategy(name)` | Declarative strategy builder |

### Strategy Builder Methods

| Method | Description |
|--------|-------------|
| `.buyWhen(condition)` | Add buy condition |
| `.sellWhen(condition)` | Add sell condition |
| `.stopLoss(pct)` | Set stop loss percentage |
| `.takeProfit(pct)` | Set take profit percentage |
| `.trailingStop(pct)` | Set trailing stop percentage |

### Condition Builders

| Function | Returns |
|----------|---------|
| `sma(period)` | SMA indicator |
| `ema(period)` | EMA indicator |
| `rsi(period)` | RSI indicator |
| `macd(fast, slow, signal)` | MACD indicator |
| `atr(period)` | ATR indicator |
| `crossover(fast, slow)` | Crossover condition |
| `crossunder(fast, slow)` | Crossunder condition |

Indicators support comparison operators: `sma(10) > sma(30)`, `rsi(14) < 30`

## Terminal UI

Tradex includes a TradingView-inspired terminal dashboard. No browser needed.

### Result Visualization

```python
result = backtest("AAPL", goldenCross())

result.show()    # Rich metric cards (Return, Sharpe, MDD, Win Rate, Trades)
result.chart()   # Full dashboard: Equity + Drawdown + Returns histogram
```

### CLI Commands

```bash
tradex backtest AAPL -s goldenCross --dashboard   # Full dashboard
tradex chart AAPL -n 60                            # Candlestick chart
tradex compare AAPL -s goldenCross,rsiOversold     # Strategy comparison
tradex optimize AAPL -s goldenCross                # Parameter optimization
tradex list                                        # Available strategies
```

### Individual Charts

```python
from tradex.tui.charts import plotEquityCurve, plotCandlestick, plotDrawdown

plotEquityCurve(result, sma_periods=[20, 60])  # Equity + SMA overlay
plotCandlestick(df, sma_periods=[5, 20])       # Candlestick + Volume
plotDrawdown(result)                            # Drawdown chart
```

### Interactive Dashboard (Optional)

```bash
pip install tradex-backtest[tui]
```

```python
from tradex.tui.dashboard import launchDashboard
launchDashboard(result)  # Full-screen Textual app (q=quit, d=theme)
```

## Advanced Analytics

```python
from tradex import (
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

whatIf = WhatIfSimulator().analyze(result)

drawdown = DrawdownSimulator().simulate(result)

seasonality = SeasonalityAnalyzer().analyze(result)

journal = TradingJournal()
journal.record(result)

leaderboard = StrategyLeaderboard()
leaderboard.addResult("goldenCross", result)
leaderboard.addResult("rsiOversold", result2)
leaderboard.printRanking()
```

### Innovative Analytics

```python
from tradex import (
    MonteCarloStressAnalyzer, FractalAnalyzer, RegimeDetector,
    InformationTheoryAnalyzer, PortfolioStressAnalyzer,
)

result = backtest("005930", goldenCross())

# Monte Carlo: 10K path stress test
mc = MonteCarloStressAnalyzer().analyze(result, paths=10000)
print(f"Ruin probability: {mc.ruinProbability:.2%}")
print(f"99% confidence band: {mc.confidenceBands['99%']}")

# Fractal: market character classification
fractal = FractalAnalyzer().analyze(result)
print(f"Hurst: {fractal.hurstExponent:.3f} → {fractal.marketCharacter}")

# Regime: probabilistic market state detection
regime = RegimeDetector().analyze(result)
print(f"Current regime: {regime.currentRegime}")
print(f"Transition matrix: {regime.transitionMatrix}")

# Information Theory: signal quality measurement
info = InformationTheoryAnalyzer().analyze(result)
print(f"Signal quality: {info.signalQuality}")
print(f"Mutual information: {info.mutualInformation:.4f} bits")

# Portfolio Stress: crisis scenario simulation
stress = PortfolioStressAnalyzer().analyze(result)
print(f"Overall grade: {stress.overallGrade}")
print(f"Survival rate: {stress.survivalRate:.0%}")
```

## Terminal UI Styles

### Modern (TradingView)
```python
result.show()                              # Default: metric cards + detail table
result.show(style="modern")                # Same as above
```

### Bloomberg (Dense)
```python
result.show(style="bloomberg")             # 4-quadrant layout + monthly heatmap
```

### Minimal (Hedge Fund)
```python
result.show(style="minimal")               # Clean report with quality indicators
```

### Chart Types (12)
```python
from tradex.tui.charts import *

plotEquityCurve(result, smaPeriods=[20, 60])  # Equity + SMA + trade markers
plotDrawdown(result)                           # Drawdown with MDD annotation
plotCandlestick(df, smaPeriods=[5, 20])        # OHLCV candlestick + volume subplot
plotReturns(result)                            # Return distribution (skew/kurtosis)
plotSeasonality(result)                        # Monthly seasonality bars
plotMonthlyHeatmap(result)                     # Year x Month heatmap
plotRollingMetrics(result)                     # Rolling Sharpe + volatility
plotTradeScatter(result)                       # Holding days vs return scatter
plotTradeMarkers(result)                       # Equity + buy/sell markers
plotCorrelationBars(names, matrix)             # Strategy correlation horizontal bars
plotStrategyDna(dna)                           # 12-dimension DNA horizontal bars
plotDashboard(result, lang="ko")               # All-in-one dashboard
```

## Architecture

```
tradex/
├── engine.py              # Core backtest engine
├── multiAssetEngine.py    # Multi-asset portfolio engine
├── strategy/              # Strategy base + 60+ indicators + ensemble combiner
├── easy/                  # 2-line API, presets, Korean API
├── vectorized/            # Vectorized engine + indicators (Pure NumPy)
├── datafeed/              # Data feeds (FinanceDataReader + Parquet cache)
├── broker/                # Commission, slippage, fill, execution simulation
├── risk/                  # Position sizing, VaR, Monte Carlo
├── optimize/              # Grid / random search optimizer
├── walkforward/           # Walk-forward analysis
├── analytics/             # Strategy DNA, Black Swan, Health Score, What-If, Seasonality, etc.
├── portfolio/             # Portfolio tracking + optimization
├── quant/                 # Factor analysis, statistical arbitrage
├── signals/               # Signal prediction + adaptive signals
├── advisor/               # Market regime + strategy recommendation
├── entities/              # Bar, Order, Position, Trade
├── events/                # Event system (Market, Signal, Order, Fill)
├── tui/                   # Terminal UI (Rich + Plotext + Textual), 3 styles, 12 chart types
├── cli.py                 # Typer CLI (backtest, optimize, chart, compare, list)
└── tests/                 # Unit + integration tests (87 tests)
```

## Running Tests

```bash
# Using uv
uv run pytest

# Using pip
pytest
```

## Contributing

Contributions are welcome! Feel free to open issues and pull requests.

```bash
git clone https://github.com/eddmpython/tradex.git
cd tradex

# Using uv (recommended)
uv sync --dev
uv run pytest

# Using pip
pip install -e ".[dev]"
pytest
```

## Support

If Tradex helps your trading research, consider supporting the project:

<p>
  <a href="https://buymeacoffee.com/eddmpython" target="_blank">
    <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
  </a>
</p>

Your support keeps this project free and actively maintained. Every coffee fuels a new feature.

## License

MIT License. Use it freely in personal and commercial projects.
