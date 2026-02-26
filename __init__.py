"""
Tradex - Blazing-fast backtesting engine for quantitative trading.

A comprehensive backtesting framework built from the ground up for speed
and simplicity. Powered by NumPy vectorized operations, it runs 1,000
parameter optimizations in 0.02 seconds.

Features:
    - Vectorized engine: 100x faster than event-driven loops
    - 50+ technical indicators: SMA, EMA, RSI, MACD, Bollinger, ATR, Ichimoku, etc.
    - Declarative strategy builder: Method chaining, no subclassing needed
    - Walk-forward analysis: Built-in overfitting prevention
    - Parameter optimization: Grid search / random search
    - Multi-asset portfolios: Backtest across multiple symbols with rebalancing
    - Realistic simulation: Commission, slippage, fill logic, position sizing
    - Risk analytics: VaR, CVaR, Monte Carlo, Sharpe, Sortino, Calmar
    - Factor analysis: Multi-factor models, statistical arbitrage
    - Korean market native: Transaction tax (0.18%), brokerage fees, KRX mapping
    - Korean language API: Full Korean function names supported

Quick Start:
    >>> from tradex import backtest, goldenCross
    >>> result = backtest("005930", goldenCross())
    >>> print(result.summary())

    >>> from tradex import vbacktest
    >>> result = vbacktest("005930", "goldenCross", fast=10, slow=30)

    >>> from tradex import QuickStrategy, backtest, sma, crossover, crossunder
    >>> strategy = (
    ...     QuickStrategy("MyStrategy")
    ...     .buyWhen(crossover(sma(10), sma(30)))
    ...     .sellWhen(crossunder(sma(10), sma(30)))
    ...     .stopLoss(5)
    ... )
    >>> result = backtest("005930", strategy)

Modules:
    - easy: 2-line API, preset strategies, Korean API
    - vectorized: Pure NumPy vectorized engine and indicators
    - strategy: Strategy base class, 50+ indicators, ensemble combiner
    - datafeed: Data feeds (FinanceDataReader, Parquet cache)
    - broker: Commission, slippage, fill, execution simulation
    - risk: Position sizing, VaR, Monte Carlo simulation
    - optimize: Grid / random search parameter optimization
    - walkforward: Walk-forward analysis for overfitting prevention
    - analytics: Performance metrics, charts, tearsheet, report generation
    - portfolio: Portfolio tracking and optimization
    - quant: Factor analysis, statistical arbitrage
    - signal: Signal prediction, adaptive signals, forecasting
    - advisor: Market regime classification, strategy recommendation
"""

from tradex.entities import Order, OrderSide, OrderType, TimeInForce, Position, Trade, Bar
from tradex.events import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent
from tradex.strategy import Strategy, Indicators
from tradex.portfolio import Portfolio
from tradex.engine import BacktestEngine, BacktestResult
from tradex.optimize import Optimizer, ParameterSpace, OptimizeResult
from tradex.walkforward import WalkForwardAnalyzer, WalkForwardResult, PeriodSplitter
from tradex.multiAssetEngine import MultiAssetEngine, MultiAssetStrategy, MultiAssetResult
from tradex.datafeed import MultiDataFeed
from tradex.analytics.charts import BacktestChart
from tradex.advisor import StrategyAdvisor, MarketClassifier, MarketRegime
from tradex.version import CURRENT_VERSION, getVersion, checkVersion, VersionManager
from tradex.signals import (
    SignalPredictor,
    SignalResult,
    SignalConfig,
    SignalType,
    MultiSignalAnalyzer,
    SignalScanner,
    SymbolSignal,
    MarketSignal,
)

from tradex.easy import (
    backtest,
    optimize,
    quickTest,
    QuickStrategy,
    goldenCross,
    rsiOversold,
    bollingerBreakout,
    macdCross,
    breakout,
    meanReversion,
    trendFollowing,
    백테스트,
    최적화,
    전략,
    골든크로스,
    RSI과매도,
    볼린저돌파,
    MACD크로스,
    돌파전략,
    평균회귀,
    추세추종,
    sma,
    ema,
    rsi,
    macd,
    bollinger,
    atr,
    price,
    crossover,
    crossunder,
)

from tradex.vectorized import (
    vsma,
    vema,
    vrsi,
    vmacd,
    vbollinger,
    vatr,
    vstochastic,
    vadx,
    vroc,
    vmomentum,
    vcrossover,
    vcrossunder,
    vcross,
    vgoldenCross,
    vrsiSignal,
    vmacdSignal,
    vbollingerSignal,
    vbreakoutSignal,
    vTrendFilter,
)

from tradex.vectorized import (
    VectorizedEngine,
    VectorizedResult,
    vbacktest,
    voptimize,
)

from tradex.tui import (
    printResult,
    printComparison,
    printTrades,
    plotEquityCurve,
    plotDrawdown,
    plotCandlestick,
    plotReturns,
    plotDashboard,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",

    # === Easy API (Primary Interface) ===
    "backtest",
    "optimize",
    "quickTest",
    "QuickStrategy",

    # Preset Strategies
    "goldenCross",
    "rsiOversold",
    "bollingerBreakout",
    "macdCross",
    "breakout",
    "meanReversion",
    "trendFollowing",

    # Condition Builders
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "price",
    "crossover",
    "crossunder",

    # === Vectorized Engine (High Performance) ===
    "VectorizedEngine",
    "VectorizedResult",
    "vbacktest",
    "voptimize",

    # Vectorized Indicators (v2 - Pure NumPy)
    "vsma",
    "vema",
    "vrsi",
    "vmacd",
    "vbollinger",
    "vatr",
    "vstochastic",
    "vadx",
    "vroc",
    "vmomentum",

    # Vectorized Signals (v2 - Pure NumPy)
    "vcrossover",
    "vcrossunder",
    "vcross",
    "vgoldenCross",
    "vrsiSignal",
    "vmacdSignal",
    "vbollingerSignal",
    "vbreakoutSignal",
    "vTrendFilter",

    # === Core Engine (Advanced) ===
    "BacktestEngine",
    "BacktestResult",
    "Strategy",
    "Indicators",
    "Portfolio",

    # Entities
    "Order",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Position",
    "Trade",
    "Bar",

    # Events
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",

    # === Optimization ===
    "Optimizer",
    "ParameterSpace",
    "OptimizeResult",

    # === Walk-Forward Analysis ===
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "PeriodSplitter",

    # === Multi-Asset ===
    "MultiAssetEngine",
    "MultiAssetStrategy",
    "MultiAssetResult",
    "MultiDataFeed",

    # === Visualization ===
    "BacktestChart",

    # === Strategy Advisor ===
    "StrategyAdvisor",
    "MarketClassifier",
    "MarketRegime",

    # === Version Management ===
    "CURRENT_VERSION",
    "getVersion",
    "checkVersion",
    "VersionManager",

    # === Korean API ===
    "백테스트",
    "최적화",
    "전략",
    "골든크로스",
    "RSI과매도",
    "볼린저돌파",
    "MACD크로스",
    "돌파전략",
    "평균회귀",
    "추세추종",

    # === Terminal UI ===
    "printResult",
    "printComparison",
    "printTrades",
    "plotEquityCurve",
    "plotDrawdown",
    "plotCandlestick",
    "plotReturns",
    "plotDashboard",

    # === Signal Predictor ===
    "SignalPredictor",
    "SignalResult",
    "SignalConfig",
    "SignalType",
    "MultiSignalAnalyzer",
    "SignalScanner",
    "SymbolSignal",
    "MarketSignal",
]
