# Tradix API Reference

Complete API documentation for Tradix v2.1.0.

## Table of Contents

- [Easy API](#easy-api)
  - [backtest()](#backtest)
  - [vbacktest()](#vbacktest)
  - [optimize()](#optimize)
  - [voptimize()](#voptimize)
  - [QuickStrategy](#quickstrategy)
- [Preset Strategies](#preset-strategies)
- [Condition Builders](#condition-builders)
- [Vectorized Indicators](#vectorized-indicators)
- [Vectorized Signals](#vectorized-signals)
- [Core Engine](#core-engine)
- [Data Feeds](#data-feeds)
- [Risk Management](#risk-management)
- [Walk-Forward Analysis](#walk-forward-analysis)

---

## Easy API

### backtest()

Run a backtest with minimal configuration.

```python
def backtest(
    symbol: str,
    strategy: Union[Strategy, QuickStrategy, str] = None,
    period: str = "3년",
    initialCash: int = 10_000_000,
    mode: str = "auto",
    **kwargs
) -> EasyResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | required | Stock symbol (e.g., "005930") or Korean name ("삼성전자") |
| `strategy` | Strategy/QuickStrategy/str | None | Strategy to test. If None, uses goldenCross |
| `period` | str | "3년" | Backtest period ("1년", "3년", "5년", "10년") |
| `initialCash` | int | 10,000,000 | Initial capital in KRW |
| `mode` | str | "auto" | Execution mode: "auto", "vectorized", "event" |

**Returns:** `EasyResult` object with performance metrics.

**Example:**

```python
from tradix import backtest, goldenCross

# Simple usage
result = backtest("005930", goldenCross())
print(f"Return: {result.totalReturn:+.2f}%")
print(f"Sharpe: {result.sharpeRatio:.2f}")
print(f"Max DD: {result.maxDrawdown:.2f}%")

# With custom parameters
result = backtest("005930", goldenCross(fast=5, slow=20), period="5년")
```

---

### vbacktest()

Run a vectorized backtest (100x faster).

```python
def vbacktest(
    symbol: str,
    strategy: str = "goldenCross",
    period: str = "3년",
    initialCash: float = 10_000_000.0,
    commission: float = 0.00015,
    tax: float = 0.0018,
    slippage: float = 0.001,
    stopLoss: float = 0.0,
    takeProfit: float = 0.0,
    **params
) -> VectorizedResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | required | Stock symbol |
| `strategy` | str | "goldenCross" | Strategy name |
| `period` | str | "3년" | Backtest period |
| `initialCash` | float | 10,000,000 | Initial capital |
| `commission` | float | 0.00015 | Commission rate (0.015%) |
| `tax` | float | 0.0018 | Tax rate (0.18% for Korean stocks) |
| `slippage` | float | 0.001 | Slippage rate |
| `stopLoss` | float | 0.0 | Stop loss percentage (0 = disabled) |
| `takeProfit` | float | 0.0 | Take profit percentage (0 = disabled) |
| `**params` | | | Strategy-specific parameters |

**Returns:** `VectorizedResult` object.

**Example:**

```python
from tradix import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)
print(f"Return: {result.totalReturn:+.2f}%")
print(f"Trades: {result.totalTrades}")
```

---

### optimize()

Find optimal strategy parameters.

```python
def optimize(
    symbol: str,
    strategy: str,
    metric: str = "sharpeRatio",
    period: str = "3년",
    **paramRanges
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | required | Stock symbol |
| `strategy` | str | required | Strategy name |
| `metric` | str | "sharpeRatio" | Optimization target metric |
| `period` | str | "3년" | Backtest period |
| `**paramRanges` | tuple | | Parameter ranges as (start, end, step) |

**Returns:** Dictionary with best parameters and results.

**Example:**

```python
from tradix import optimize

best = optimize(
    "005930",
    "goldenCross",
    fast=(5, 20, 5),
    slow=(20, 60, 10),
    metric="sharpeRatio"
)

print(f"Best params: {best['params']}")
print(f"Best Sharpe: {best['metric']:.2f}")
```

---

### voptimize()

Vectorized parameter optimization (much faster).

```python
def voptimize(
    symbol: str,
    strategy: str = "goldenCross",
    metric: str = "sharpeRatio",
    period: str = "3년",
    **paramRanges
) -> dict
```

**Example:**

```python
from tradix import voptimize

# Optimize 1000 parameter combinations in ~0.02s
best = voptimize(
    "005930",
    "goldenCross",
    fast=(5, 50, 1),     # 45 values
    slow=(20, 100, 2),   # 40 values
    metric="sharpeRatio"
)
```

---

### QuickStrategy

Declarative strategy builder for custom strategies.

```python
class QuickStrategy:
    def __init__(self, name: str = "QuickStrategy")
    def buyWhen(self, condition: Condition) -> QuickStrategy
    def sellWhen(self, condition: Condition) -> QuickStrategy
    def stopLoss(self, percent: float) -> QuickStrategy
    def takeProfit(self, percent: float) -> QuickStrategy
    def maxPositions(self, n: int) -> QuickStrategy
    def positionSize(self, percent: float) -> QuickStrategy
```

**Example:**

```python
from tradix import QuickStrategy, backtest, sma, rsi, crossover

strategy = (
    QuickStrategy("MyStrategy")
    .buyWhen(crossover(sma(10), sma(30)))
    .buyWhen(rsi(14) < 30)
    .sellWhen(rsi(14) > 70)
    .stopLoss(5)
    .takeProfit(15)
    .positionSize(50)
)

result = backtest("005930", strategy)
```

---

## Preset Strategies

Ready-to-use trading strategies.

| Strategy | Description | Parameters |
|----------|-------------|------------|
| `goldenCross(fast, slow)` | SMA crossover | fast=10, slow=30 |
| `rsiOversold(period, oversold, overbought)` | RSI reversal | period=14, oversold=30, overbought=70 |
| `bollingerBreakout(period, std)` | Bollinger band breakout | period=20, std=2.0 |
| `macdCross(fast, slow, signal)` | MACD crossover | fast=12, slow=26, signal=9 |
| `breakout(period)` | Channel breakout (Turtle) | period=20 |
| `meanReversion(period, threshold)` | Mean reversion | period=20, threshold=2.0 |
| `trendFollowing(fast, slow, adxThreshold)` | Trend following + ADX | fast=10, slow=30, adxThreshold=25 |

**Example:**

```python
from tradix import backtest, rsiOversold, bollingerBreakout

# RSI strategy with custom parameters
result = backtest("005930", rsiOversold(period=14, oversold=25, overbought=75))

# Bollinger breakout
result = backtest("005930", bollingerBreakout(period=20, std=2.5))
```

---

## Condition Builders

Build custom conditions for QuickStrategy.

### Indicators

```python
sma(period: int) -> Indicator         # Simple Moving Average
ema(period: int) -> Indicator         # Exponential Moving Average
rsi(period: int = 14) -> Indicator    # Relative Strength Index
macd(fast=12, slow=26, signal=9)      # MACD
bollinger(period=20, std=2.0)         # Bollinger Bands
atr(period: int = 14) -> Indicator    # Average True Range
price() -> Indicator                  # Current price
```

### Crossover Functions

```python
crossover(ind1, ind2) -> Condition    # ind1 crosses above ind2
crossunder(ind1, ind2) -> Condition   # ind1 crosses below ind2
```

### Comparison Operators

Indicators support comparison operators:

```python
sma(10) > sma(30)      # SMA10 above SMA30
rsi(14) < 30           # RSI below 30
price() > bollinger(20).upper  # Price above upper band
```

**Example:**

```python
from tradix import QuickStrategy, sma, ema, rsi, crossover, crossunder

strategy = (
    QuickStrategy("MultiCondition")
    .buyWhen(crossover(ema(10), ema(30)))
    .buyWhen(rsi(14) < 35)
    .sellWhen(crossunder(ema(10), ema(30)))
    .sellWhen(rsi(14) > 65)
)
```

---

## Vectorized Indicators

Numba-accelerated indicators returning NumPy arrays.

```python
vsma(close: np.ndarray, period: int) -> np.ndarray
vema(close: np.ndarray, period: int) -> np.ndarray
vrsi(close: np.ndarray, period: int = 14) -> np.ndarray
vmacd(close, fast=12, slow=26, signal=9) -> tuple[np.ndarray, np.ndarray, np.ndarray]
vbollinger(close, period=20, std=2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]
vatr(high, low, close, period=14) -> np.ndarray
vstochastic(high, low, close, kPeriod=14, dPeriod=3) -> tuple[np.ndarray, np.ndarray]
vadx(high, low, close, period=14) -> np.ndarray
```

**Performance:**

| Indicator | Time (10-year data) |
|-----------|---------------------|
| vsma | 0.006ms |
| vema | 0.012ms |
| vrsi | 0.009ms |
| vmacd | 0.040ms |

**Example:**

```python
import numpy as np
from tradix import vsma, vrsi, vmacd

close = np.random.randn(2500).cumsum() + 100

# Calculate indicators
sma20 = vsma(close, 20)
rsi14 = vrsi(close, 14)
macd_line, signal_line, histogram = vmacd(close)
```

---

## Vectorized Signals

Signal generators returning arrays of buy/sell signals.

```python
vcrossover(fast, slow) -> np.ndarray   # Returns 1 for crossover, 0 otherwise
vcrossunder(fast, slow) -> np.ndarray  # Returns -1 for crossunder, 0 otherwise
vgoldenCross(close, fast=10, slow=30) -> np.ndarray
vrsiSignal(rsi, oversold=30, overbought=70) -> np.ndarray
vmacdSignal(close, fast=12, slow=26, signal=9) -> np.ndarray
vbollingerSignal(close, period=20, std=2.0) -> np.ndarray
```

Signal values: `1` = buy, `-1` = sell, `0` = hold

**Example:**

```python
from tradix import vgoldenCross, vrsiSignal, vrsi

close = data['close'].values

# Golden cross signals
signals = vgoldenCross(close, fast=10, slow=30)
buy_days = np.where(signals == 1)[0]

# RSI signals
rsi_values = vrsi(close, 14)
signals = vrsiSignal(rsi_values, oversold=30, overbought=70)
```

---

## Core Engine

For advanced users requiring full control.

### BacktestEngine

```python
class BacktestEngine:
    def __init__(
        self,
        data: DataFeed,
        strategy: Strategy,
        initialCash: float = 10_000_000,
        commission: float = 0.00015,
        tax: float = 0.0018,
        slippage: float = 0.001
    )

    def run(self) -> BacktestResult
```

### Strategy Base Class

```python
class Strategy:
    def initialize(self):
        """Called once before backtest starts."""
        pass

    def onBar(self, bar: Bar):
        """Called for each bar. Override this."""
        pass

    # Helper methods
    def sma(self, period: int) -> float
    def ema(self, period: int) -> float
    def rsi(self, period: int = 14) -> float
    def buy(self, symbol: str, quantity: int = None)
    def sell(self, symbol: str, quantity: int = None)
    def closePosition(self, symbol: str)
    def hasPosition(self, symbol: str) -> bool
```

**Example:**

```python
from tradix import Strategy, Bar, BacktestEngine
from tradix.datafeed import FinanceDataReaderFeed

class MyStrategy(Strategy):
    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30

    def onBar(self, bar: Bar):
        fastSma = self.sma(self.fastPeriod)
        slowSma = self.sma(self.slowPeriod)

        if fastSma is None or slowSma is None:
            return

        if fastSma > slowSma and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)
        elif fastSma < slowSma and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)

data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")
engine = BacktestEngine(data, MyStrategy(), initialCash=10_000_000)
result = engine.run()
```

---

## Data Feeds

### FinanceDataReaderFeed

Load data from Korean/US/Japan markets.

```python
from tradix.datafeed import FinanceDataReaderFeed

# Korean stock
data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")

# US stock
data = FinanceDataReaderFeed("AAPL", "2020-01-01", "2024-12-31")

# With caching
data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31", cache=True)
```

### MultiDataFeed

For multi-asset strategies.

```python
from tradix.datafeed import MultiDataFeed

data = MultiDataFeed()
data.addSymbol("005930", "2020-01-01", "2024-12-31")
data.addSymbol("000660", "2020-01-01", "2024-12-31")
```

---

## Risk Management

### RiskSimulator

```python
from tradix.risk import RiskSimulator, VaRMethod

simulator = RiskSimulator()
simulator.fit(returns)  # Daily returns array

# VaR calculation
var = simulator.calcVaR(confidence=0.95, method=VaRMethod.HISTORICAL)
print(f"95% VaR: {var.var:.2%}")
print(f"CVaR: {var.cvar:.2%}")

# Monte Carlo simulation
mc = simulator.monteCarloSimulation(horizon=252, nSim=10000)
print(f"Expected return: {mc.expectedReturn:.2%}")
print(f"95% CI: [{mc.percentile5:.2%}, {mc.percentile95:.2%}]")
```

### Position Sizing

```python
from tradix.risk import (
    FixedQuantitySizer,
    PercentEquitySizer,
    FixedRiskSizer,
    KellySizer
)

# Fixed 100 shares per trade
sizer = FixedQuantitySizer(quantity=100)

# 10% of equity per trade
sizer = PercentEquitySizer(percent=10)

# Risk 2% of equity per trade
sizer = FixedRiskSizer(riskPercent=2, atrMultiple=2)

# Kelly criterion
sizer = KellySizer(winRate=0.55, avgWin=2.0, avgLoss=1.0)
```

---

## Walk-Forward Analysis

Prevent overfitting with out-of-sample validation.

```python
from tradix import WalkForwardAnalyzer, ParameterSpace

def createStrategy(params):
    return goldenCross(fast=params['fast'], slow=params['slow'])

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
print(f"Robustness Ratio: {result.robustnessRatio:.1%}")
print(f"Out-of-sample Sharpe: {result.outOfSampleSharpe:.2f}")
```

---

## Result Objects

### VectorizedResult

```python
@dataclass
class VectorizedResult:
    totalReturn: float      # Total return percentage
    annualReturn: float     # Annualized return
    sharpeRatio: float      # Sharpe ratio
    maxDrawdown: float      # Maximum drawdown percentage
    winRate: float          # Win rate percentage
    totalTrades: int        # Total number of trades
    equity: np.ndarray      # Equity curve

    def summary(self) -> str
    def plot(self)
```

### BacktestResult

```python
@dataclass
class BacktestResult:
    totalReturn: float
    annualReturn: float
    sharpeRatio: float
    sortinoRatio: float
    maxDrawdown: float
    calmarRatio: float
    winRate: float
    profitFactor: float
    totalTrades: int
    avgTradeDuration: float
    trades: List[Trade]
    equity: pd.Series

    def summary(self) -> str
    def plot(self)
```

---

## Korean Market Support

Tradix includes built-in support for Korean markets:

- **Commission**: 0.015% (both buy and sell)
- **Tax**: 0.18% (sell only, securities transaction tax)
- **Korean stock names**: "삼성전자" → "005930" auto-conversion

```python
# Korean stock name support
result = backtest("삼성전자", goldenCross())

# Commission/tax are automatically applied
# Buy: 0.015% commission
# Sell: 0.015% commission + 0.18% tax
```

---

## Performance Benchmarks

Tested on 10-year daily data (2,458 bars):

| Operation | Time |
|-----------|------|
| SMA calculation | 0.006ms |
| EMA calculation | 0.012ms |
| RSI calculation | 0.009ms |
| MACD calculation | 0.040ms |
| Full backtest | 0.132ms |
| 1000 param optimization | 0.02s |

---

## Error Handling

```python
from tradix import backtest
from tradix.exceptions import (
    DataNotFoundError,
    InvalidSymbolError,
    InvalidParameterError
)

try:
    result = backtest("INVALID", goldenCross())
except InvalidSymbolError as e:
    print(f"Invalid symbol: {e}")
except DataNotFoundError as e:
    print(f"Data not found: {e}")
```

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
