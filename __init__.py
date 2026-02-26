"""
Tradex - Blazing-fast backtesting engine for quantitative trading.

A comprehensive backtesting framework built from the ground up for speed
and simplicity. Powered by NumPy vectorized operations, it runs 1,000
parameter optimizations in 0.02 seconds.

Features:
    - Vectorized engine: 100x faster than event-driven loops
    - 50+ technical indicators: SMA, EMA, RSI, MACD, Bollinger, ATR, Ichimoku, etc.
    - 33 preset strategies: Trend, momentum, oscillator, volatility, multi-indicator
    - Declarative strategy builder: Method chaining, no subclassing needed
    - Walk-forward analysis: Built-in overfitting prevention
    - Parameter optimization: Grid search / random search
    - Multi-asset portfolios: Backtest across multiple symbols with rebalancing
    - Realistic simulation: Commission, slippage, fill logic, position sizing
    - Risk analytics: VaR, CVaR, Monte Carlo, Sharpe, Sortino, Calmar
    - Factor analysis: Multi-factor models, statistical arbitrage
    - Strategy DNA: 12-dimensional strategy fingerprinting
    - Black Swan Defense: Extreme event resilience scoring (0-100)
    - Strategy Health: Overfitting/stability/consistency diagnostics
    - What-If Simulator: Commission/slippage/capital sensitivity analysis
    - Drawdown Simulator: Historical worst-case drawdown scenarios
    - Seasonality Analyzer: Monthly/weekday/quarterly pattern discovery
    - Correlation Matrix: Multi-strategy correlation and clustering
    - Trading Journal: Automatic trade diary with MFE/MAE analytics
    - Strategy Leaderboard: Multi-strategy ranking and badge system
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
    - easy: 2-line API, 33 preset strategies, Korean API
    - vectorized: Pure NumPy vectorized engine and indicators
    - strategy: Strategy base class, 50+ indicators, ensemble combiner
    - datafeed: Data feeds (FinanceDataReader, Parquet cache)
    - broker: Commission, slippage, fill, execution simulation
    - risk: Position sizing, VaR, Monte Carlo simulation
    - optimize: Grid / random search parameter optimization
    - walkforward: Walk-forward analysis for overfitting prevention
    - analytics: Strategy DNA, Black Swan, Health Score, What-If, Seasonality, etc.
    - portfolio: Portfolio tracking and optimization
    - quant: Factor analysis, statistical arbitrage
    - signals: Signal prediction, adaptive signals, forecasting
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
from tradex.analytics.metrics import PerformanceMetrics
from tradex.analytics import (
    StrategyDNA,
    StrategyDnaAnalyzer,
    BlackSwanScore,
    BlackSwanAnalyzer,
    StrategyHealthScore,
    StrategyHealthAnalyzer,
    WhatIfResult,
    WhatIfSimulator,
    DrawdownScenario,
    DrawdownSimulator,
    SeasonalPattern,
    SeasonalityAnalyzer,
    CorrelationResult,
    CorrelationAnalyzer,
    JournalEntry,
    TradingJournal,
    LeaderboardEntry,
    StrategyLeaderboard,
    MonteCarloStressResult,
    MonteCarloStressAnalyzer,
    FractalAnalysisResult,
    FractalAnalyzer,
    RegimeAnalysisResult,
    RegimeDetector,
    InformationTheoryResult,
    InformationTheoryAnalyzer,
    StressScenario,
    PortfolioStressResult,
    PortfolioStressAnalyzer,
)
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
    emaCross,
    tripleScreen,
    dualMomentum,
    momentumCross,
    rocBreakout,
    stochasticCross,
    williamsReversal,
    cciBreakout,
    rsiDivergence,
    volatilityBreakout,
    keltnerChannel,
    bollingerSqueeze,
    superTrend,
    ichimokuCloud,
    parabolicSar,
    donchianBreakout,
    tripleEma,
    macdRsiCombo,
    trendMomentum,
    bollingerRsi,
    gapTrading,
    pyramiding,
    swingTrading,
    scalpingMomentum,
    buyAndHold,
    dollarCostAverage,
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
    printMonthlyHeatmap,
    printStrategyDna,
    printHealthScore,
    printBlackSwanScore,
    plotEquityCurve,
    plotDrawdown,
    plotCandlestick,
    plotReturns,
    plotTradeMarkers,
    plotDashboard,
    plotMonthlyHeatmap,
    plotRollingMetrics,
    plotTradeScatter,
    plotCorrelationBars,
    plotStrategyDna,
    plotSeasonality,
)

__version__ = "1.2.0"

__all__ = [
    # Version
    "__version__",

    # === Easy API (Primary Interface) ===
    "backtest",
    "optimize",
    "quickTest",
    "QuickStrategy",

    # Preset Strategies (33)
    "goldenCross",
    "rsiOversold",
    "bollingerBreakout",
    "macdCross",
    "breakout",
    "meanReversion",
    "trendFollowing",
    "emaCross",
    "tripleScreen",
    "dualMomentum",
    "momentumCross",
    "rocBreakout",
    "stochasticCross",
    "williamsReversal",
    "cciBreakout",
    "rsiDivergence",
    "volatilityBreakout",
    "keltnerChannel",
    "bollingerSqueeze",
    "superTrend",
    "ichimokuCloud",
    "parabolicSar",
    "donchianBreakout",
    "tripleEma",
    "macdRsiCombo",
    "trendMomentum",
    "bollingerRsi",
    "gapTrading",
    "pyramiding",
    "swingTrading",
    "scalpingMomentum",
    "buyAndHold",
    "dollarCostAverage",

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
    "PerformanceMetrics",

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
    "printMonthlyHeatmap",
    "printStrategyDna",
    "printHealthScore",
    "printBlackSwanScore",
    "plotEquityCurve",
    "plotDrawdown",
    "plotCandlestick",
    "plotReturns",
    "plotTradeMarkers",
    "plotDashboard",
    "plotMonthlyHeatmap",
    "plotRollingMetrics",
    "plotTradeScatter",
    "plotCorrelationBars",
    "plotStrategyDna",
    "plotSeasonality",

    # === Signal Predictor ===
    "SignalPredictor",
    "SignalResult",
    "SignalConfig",
    "SignalType",
    "MultiSignalAnalyzer",
    "SignalScanner",
    "SymbolSignal",
    "MarketSignal",

    # === Advanced Analytics ===
    "StrategyDNA",
    "StrategyDnaAnalyzer",
    "BlackSwanScore",
    "BlackSwanAnalyzer",
    "StrategyHealthScore",
    "StrategyHealthAnalyzer",
    "WhatIfResult",
    "WhatIfSimulator",
    "DrawdownScenario",
    "DrawdownSimulator",
    "SeasonalPattern",
    "SeasonalityAnalyzer",
    "CorrelationResult",
    "CorrelationAnalyzer",
    "JournalEntry",
    "TradingJournal",
    "LeaderboardEntry",
    "StrategyLeaderboard",

    # === Innovative Analytics (World-First) ===
    "MonteCarloStressResult",
    "MonteCarloStressAnalyzer",
    "FractalAnalysisResult",
    "FractalAnalyzer",
    "RegimeAnalysisResult",
    "RegimeDetector",
    "InformationTheoryResult",
    "InformationTheoryAnalyzer",
    "StressScenario",
    "PortfolioStressResult",
    "PortfolioStressAnalyzer",
]
