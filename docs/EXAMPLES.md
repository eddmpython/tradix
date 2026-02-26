# Tradex Examples

Practical examples from simple to advanced usage.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Preset Strategies](#preset-strategies)
3. [Custom Strategies](#custom-strategies)
4. [Vectorized Mode](#vectorized-mode)
5. [Parameter Optimization](#parameter-optimization)
6. [Walk-Forward Analysis](#walk-forward-analysis)
7. [Risk Management](#risk-management)
8. [Multi-Asset Portfolio](#multi-asset-portfolio)

---

## Quick Start

### 2-Line Backtest

```python
from tradex import backtest, goldenCross

result = backtest("005930", goldenCross())
print(result.summary())
```

Output:
```
=== Backtest Result ===
Total Return: +45.32%
Annual Return: +12.85%
Sharpe Ratio: 1.24
Max Drawdown: -15.67%
Win Rate: 58.33%
Total Trades: 24
```

### Using Korean Stock Names

```python
from tradex import backtest, rsiOversold

result = backtest("삼성전자", rsiOversold())
print(f"Return: {result.totalReturn:+.2f}%")
```

---

## Preset Strategies

### Golden Cross (SMA Crossover)

```python
from tradex import backtest, goldenCross

# Default: fast=10, slow=30
result = backtest("005930", goldenCross())

# Custom parameters
result = backtest("005930", goldenCross(fast=5, slow=20, stopLoss=5, takeProfit=15))
```

### RSI Oversold

```python
from tradex import backtest, rsiOversold

# Buy when RSI crosses above oversold, sell when crosses below overbought
result = backtest("005930", rsiOversold(period=14, oversold=30, overbought=70))
```

### Bollinger Breakout

```python
from tradex import backtest, bollingerBreakout

# Buy on lower band bounce, sell at upper band
result = backtest("005930", bollingerBreakout(period=20, std=2.0))
```

### MACD Cross

```python
from tradex import backtest, macdCross

result = backtest("005930", macdCross(fast=12, slow=26, signal=9))
```

### Trend Following

```python
from tradex import backtest, trendFollowing

# SMA crossover filtered by ADX
result = backtest("005930", trendFollowing(fast=10, slow=30, adxThreshold=25))
```

### Mean Reversion

```python
from tradex import backtest, meanReversion

result = backtest("005930", meanReversion(period=20, threshold=2.0))
```

---

## Custom Strategies

### Using QuickStrategy (Declarative)

```python
from tradex import QuickStrategy, backtest, sma, rsi, crossover, crossunder

strategy = (
    QuickStrategy("MyStrategy")
    .buyWhen(crossover(sma(10), sma(30)))
    .buyWhen(rsi(14) < 30)
    .sellWhen(crossunder(sma(10), sma(30)))
    .sellWhen(rsi(14) > 70)
    .stopLoss(5)
    .takeProfit(15)
    .positionSize(50)
)

result = backtest("005930", strategy)
print(result.summary())
```

### Multiple Conditions

```python
from tradex import QuickStrategy, backtest, sma, ema, rsi, macd, crossover

# All conditions must be met (AND logic)
strategy = (
    QuickStrategy("MultiCondition")
    .buyWhen(crossover(ema(10), ema(30)))  # EMA crossover
    .buyWhen(rsi(14) < 40)                  # RSI not overbought
    .buyWhen(macd().histogram > 0)          # MACD positive
    .sellWhen(rsi(14) > 70)
    .stopLoss(7)
)

result = backtest("005930", strategy, period="5년")
```

### Class-Based Strategy (Advanced)

```python
from tradex import Strategy, Bar, BacktestEngine
from tradex.datafeed import FinanceDataReaderFeed

class TripleScreenStrategy(Strategy):
    """Elder's Triple Screen Trading System"""

    def initialize(self):
        self.weeklyTrendUp = False

    def onBar(self, bar: Bar):
        # Screen 1: Weekly trend (using 26-day EMA as proxy)
        ema26 = self.ema(26)
        ema13 = self.ema(13)

        if ema26 is None or ema13 is None:
            return

        self.weeklyTrendUp = ema13 > ema26

        # Screen 2: Daily oscillator
        rsi = self.rsi(14)
        if rsi is None:
            return

        # Screen 3: Entry
        if self.weeklyTrendUp and rsi < 30:
            if not self.hasPosition(bar.symbol):
                self.buy(bar.symbol)
        elif not self.weeklyTrendUp and rsi > 70:
            if self.hasPosition(bar.symbol):
                self.closePosition(bar.symbol)

# Run backtest
data = FinanceDataReaderFeed("005930", "2019-01-01", "2024-12-31")
engine = BacktestEngine(data, TripleScreenStrategy(), initialCash=10_000_000)
result = engine.run()
print(result.summary())
```

---

## Vectorized Mode

100x faster using Numba acceleration.

### Basic Vectorized Backtest

```python
from tradex import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)

print(f"Return: {result.totalReturn:+.2f}%")
print(f"Sharpe: {result.sharpeRatio:.2f}")
print(f"Max DD: {result.maxDrawdown:.2f}%")
print(f"Trades: {result.totalTrades}")
```

### Using Stop Loss / Take Profit

```python
from tradex import vbacktest

result = vbacktest(
    "005930",
    "goldenCross",
    fast=10,
    slow=30,
    stopLoss=5.0,      # 5% stop loss
    takeProfit=15.0,   # 15% take profit
)
```

### Vectorized Indicators Directly

```python
import numpy as np
from tradex import vsma, vema, vrsi, vmacd, vbollinger

# Generate sample data
close = np.random.randn(2500).cumsum() + 100

# Calculate indicators
sma20 = vsma(close, 20)
ema12 = vema(close, 12)
rsi14 = vrsi(close, 14)
macd_line, signal_line, histogram = vmacd(close, 12, 26, 9)
upper, middle, lower = vbollinger(close, 20, 2.0)

print(f"Current SMA20: {sma20[-1]:.2f}")
print(f"Current RSI: {rsi14[-1]:.2f}")
print(f"MACD: {macd_line[-1]:.4f}")
```

### Custom Signal Generation

```python
from tradex import VectorizedEngine
from tradex.datafeed import FinanceDataReaderFeed
import numpy as np

# Load data
data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")

# Create engine
engine = VectorizedEngine(data)

# Generate custom signals
close = engine.close
signals = np.zeros(len(close), dtype=np.int8)

# Custom logic: Buy on consecutive up days
for i in range(5, len(close)):
    upDays = sum(1 for j in range(5) if close[i-j] > close[i-j-1])
    if upDays >= 4:
        signals[i] = 1
    elif upDays <= 1:
        signals[i] = -1

# Run backtest with custom signals
result = engine.run(signals)
print(result.summary())
```

---

## Parameter Optimization

### Grid Search

```python
from tradex import voptimize

# Test all combinations (45 * 40 = 1,800 combinations)
best = voptimize(
    "005930",
    "goldenCross",
    fast=(5, 50, 1),     # 5 to 50, step 1
    slow=(20, 100, 2),   # 20 to 100, step 2
    metric="sharpeRatio"
)

print(f"Best parameters: {best['best']['params']}")
print(f"Best Sharpe: {best['best']['metric']:.3f}")

# Top 5 results
for i, r in enumerate(best['top5'], 1):
    print(f"{i}. fast={r['params']['fast']}, slow={r['params']['slow']}, "
          f"Sharpe={r['metric']:.3f}")
```

### Multi-Metric Optimization

```python
from tradex import voptimize

# Optimize for different metrics
metrics = ["sharpeRatio", "totalReturn", "winRate", "calmarRatio"]

for metric in metrics:
    best = voptimize(
        "005930", "goldenCross",
        fast=(5, 30, 5),
        slow=(20, 60, 10),
        metric=metric
    )
    print(f"{metric}: fast={best['best']['params']['fast']}, "
          f"slow={best['best']['params']['slow']}, "
          f"value={best['best']['metric']:.3f}")
```

### Robustness Check

```python
from tradex import voptimize

# Optimize on multiple symbols
symbols = ["005930", "000660", "035420"]
results = {}

for symbol in symbols:
    best = voptimize(
        symbol, "goldenCross",
        fast=(5, 25, 5),
        slow=(20, 50, 10),
        metric="sharpeRatio"
    )
    results[symbol] = best['best']['params']
    print(f"{symbol}: {best['best']['params']}")

# Find robust parameters (common across stocks)
from collections import Counter
fast_votes = Counter(r['fast'] for r in results.values())
slow_votes = Counter(r['slow'] for r in results.values())
print(f"Most robust: fast={fast_votes.most_common(1)[0][0]}, "
      f"slow={slow_votes.most_common(1)[0][0]}")
```

---

## Walk-Forward Analysis

Prevent overfitting with rolling out-of-sample testing.

```python
from tradex import WalkForwardAnalyzer, ParameterSpace
from tradex.datafeed import FinanceDataReaderFeed
from tradex.easy import goldenCross

def createStrategy(params):
    return goldenCross(fast=params['fast'], slow=params['slow'])

# Define parameter space
space = ParameterSpace()
space.addInt("fast", 5, 25, step=5)
space.addInt("slow", 20, 60, step=10)

# Load data
data = FinanceDataReaderFeed("005930", "2015-01-01", "2024-12-31")

# Walk-forward analysis
wfa = WalkForwardAnalyzer(
    data=data,
    strategyFactory=createStrategy,
    parameterSpace=space,
    inSampleMonths=12,    # Optimize on 12 months
    outOfSampleMonths=3,  # Test on next 3 months
)

result = wfa.run()

print(f"=== Walk-Forward Results ===")
print(f"Total Periods: {result.totalPeriods}")
print(f"Robustness Ratio: {result.robustnessRatio:.1%}")
print(f"In-Sample Sharpe: {result.inSampleSharpe:.2f}")
print(f"Out-of-Sample Sharpe: {result.outOfSampleSharpe:.2f}")
print(f"Stability: {result.stability:.2f}")

# Period-by-period breakdown
for period in result.periods:
    print(f"Period {period.index}: "
          f"IS={period.inSampleReturn:+.1f}%, "
          f"OOS={period.outOfSampleReturn:+.1f}%")
```

---

## Risk Management

### VaR and CVaR

```python
from tradex import vbacktest
from tradex.risk import RiskSimulator, VaRMethod
import numpy as np

# Run backtest
result = vbacktest("005930", "goldenCross", fast=10, slow=30)

# Calculate daily returns
equity = result.equity
returns = np.diff(equity) / equity[:-1]

# Risk analysis
simulator = RiskSimulator()
simulator.fit(returns)

# Historical VaR
var = simulator.calcVaR(confidence=0.95, method=VaRMethod.HISTORICAL)
print(f"95% VaR: {var.var:.2%}")
print(f"CVaR (Expected Shortfall): {var.cvar:.2%}")

# Parametric VaR
var_param = simulator.calcVaR(confidence=0.95, method=VaRMethod.PARAMETRIC)
print(f"Parametric VaR: {var_param.var:.2%}")

# Monte Carlo VaR
var_mc = simulator.calcVaR(confidence=0.95, method=VaRMethod.MONTE_CARLO, nSim=10000)
print(f"Monte Carlo VaR: {var_mc.var:.2%}")
```

### Monte Carlo Simulation

```python
from tradex.risk import RiskSimulator
import numpy as np

# Historical returns
returns = np.random.normal(0.0005, 0.02, 252)  # Example

simulator = RiskSimulator()
simulator.fit(returns)

# Simulate future scenarios
mc = simulator.monteCarloSimulation(
    horizon=252,      # 1 year forward
    nSim=10000,       # 10,000 scenarios
    initialValue=10_000_000
)

print(f"Expected Value: {mc.expectedValue:,.0f}")
print(f"5th Percentile: {mc.percentile5:,.0f}")
print(f"95th Percentile: {mc.percentile95:,.0f}")
print(f"Probability of Loss: {mc.probLoss:.1%}")
```

### Position Sizing

```python
from tradex import Strategy, Bar
from tradex.risk import PercentEquitySizer, KellySizer

class SizedStrategy(Strategy):
    def initialize(self):
        # Use 5% of equity per trade
        self.sizer = PercentEquitySizer(percent=5)

        # Or use Kelly criterion
        # self.sizer = KellySizer(winRate=0.55, avgWin=2.0, avgLoss=1.0)

    def onBar(self, bar: Bar):
        if self.shouldBuy():
            size = self.sizer.calculate(
                equity=self.portfolio.equity,
                price=bar.close,
                atr=self.atr(14)
            )
            self.buy(bar.symbol, quantity=size)
```

---

## Multi-Asset Portfolio

### Basic Multi-Asset

```python
from tradex import MultiAssetEngine, MultiAssetStrategy

class EqualWeightStrategy(MultiAssetStrategy):
    """Equal weight rebalancing"""

    def initialize(self):
        self.symbols = ["005930", "000660", "035420"]
        self.rebalanceFreq = 20  # Days

    def onBars(self, bars):
        if self.dayCount % self.rebalanceFreq == 0:
            weight = 1.0 / len(self.symbols)
            self.rebalance({s: weight for s in self.symbols})

engine = MultiAssetEngine(strategy=EqualWeightStrategy())
result = engine.run()
print(result.summary())
```

### Momentum Portfolio

```python
from tradex import MultiAssetEngine, MultiAssetStrategy
import numpy as np

class MomentumPortfolio(MultiAssetStrategy):
    """Long top momentum stocks"""

    def initialize(self):
        self.symbols = ["005930", "000660", "035420", "005380", "051910"]
        self.lookback = 60
        self.topN = 2

    def onBars(self, bars):
        if self.dayCount < self.lookback:
            return

        if self.dayCount % 20 != 0:  # Monthly rebalance
            return

        # Calculate momentum
        momentum = {}
        for symbol in self.symbols:
            returns = self.getReturns(symbol, self.lookback)
            momentum[symbol] = np.prod(1 + returns) - 1

        # Select top N
        ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        topSymbols = [s for s, _ in ranked[:self.topN]]

        # Equal weight top performers
        weight = 1.0 / self.topN
        allocation = {s: weight if s in topSymbols else 0 for s in self.symbols}
        self.rebalance(allocation)

engine = MultiAssetEngine(strategy=MomentumPortfolio())
result = engine.run()
```

---

## Complete Example: Production Strategy

```python
"""
Production-ready trading strategy with:
- Multiple entry conditions
- Stop loss / Take profit
- Position sizing
- Walk-forward validation
"""

from tradex import (
    vbacktest, voptimize, WalkForwardAnalyzer, ParameterSpace,
    QuickStrategy, backtest, sma, ema, rsi, atr, crossover, crossunder
)
from tradex.datafeed import FinanceDataReaderFeed
from tradex.risk import RiskSimulator, VaRMethod
import numpy as np

# 1. Define strategy
def createStrategy(params):
    return (
        QuickStrategy("ProductionStrategy")
        .buyWhen(crossover(ema(params['fast']), ema(params['slow'])))
        .buyWhen(rsi(14) < params['rsiEntry'])
        .sellWhen(crossunder(ema(params['fast']), ema(params['slow'])))
        .sellWhen(rsi(14) > params['rsiExit'])
        .stopLoss(params['stopLoss'])
        .takeProfit(params['takeProfit'])
    )

# 2. Parameter optimization
print("=== Optimization ===")
best = voptimize(
    "005930",
    "goldenCross",  # Use as base
    fast=(5, 20, 5),
    slow=(20, 50, 10),
    metric="sharpeRatio"
)
print(f"Best params: {best['best']['params']}")

# 3. Walk-forward validation
print("\n=== Walk-Forward Analysis ===")
space = ParameterSpace()
space.addInt("fast", 5, 20, step=5)
space.addInt("slow", 20, 50, step=10)
space.addInt("rsiEntry", 25, 40, step=5)
space.addInt("rsiExit", 60, 80, step=5)
space.addFloat("stopLoss", 3, 7, step=1)
space.addFloat("takeProfit", 10, 20, step=5)

data = FinanceDataReaderFeed("005930", "2018-01-01", "2024-12-31")

wfa = WalkForwardAnalyzer(
    data=data,
    strategyFactory=createStrategy,
    parameterSpace=space,
    inSampleMonths=12,
    outOfSampleMonths=3
)

wfResult = wfa.run()
print(f"Robustness: {wfResult.robustnessRatio:.1%}")
print(f"OOS Sharpe: {wfResult.outOfSampleSharpe:.2f}")

# 4. Final backtest
print("\n=== Final Backtest ===")
finalParams = wfResult.bestParams
strategy = createStrategy(finalParams)
result = backtest("005930", strategy, period="3년")
print(result.summary())

# 5. Risk analysis
print("\n=== Risk Analysis ===")
returns = np.diff(result.equity) / result.equity[:-1]
simulator = RiskSimulator()
simulator.fit(returns)

var = simulator.calcVaR(0.95, VaRMethod.HISTORICAL)
print(f"95% VaR: {var.var:.2%}")
print(f"CVaR: {var.cvar:.2%}")

mc = simulator.monteCarloSimulation(252, 10000, 10_000_000)
print(f"1Y Expected: {mc.expectedValue:,.0f}")
print(f"1Y 95% CI: [{mc.percentile5:,.0f}, {mc.percentile95:,.0f}]")
```

---

## Performance Tips

1. **Use vectorized mode** for optimization:
   ```python
   # Slow (event-driven)
   result = backtest("005930", goldenCross())

   # Fast (vectorized)
   result = vbacktest("005930", "goldenCross")
   ```

2. **Cache data** for repeated backtests:
   ```python
   data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31", cache=True)
   ```

3. **Parallelize optimization** for multiple symbols:
   ```python
   from concurrent.futures import ProcessPoolExecutor

   def optimizeSymbol(symbol):
       return voptimize(symbol, "goldenCross", fast=(5, 30, 5), slow=(20, 60, 10))

   with ProcessPoolExecutor() as executor:
       results = list(executor.map(optimizeSymbol, symbols))
   ```

4. **Use appropriate period** - longer isn't always better:
   ```python
   # Too short: insufficient data
   result = backtest("005930", strategy, period="6개월")

   # Recommended: 3-5 years
   result = backtest("005930", strategy, period="3년")
   ```
