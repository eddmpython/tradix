"""Tradix Signal Analysis Package.

Provides technical-indicator-based trading signal prediction, multi-symbol
analysis, adaptive regime-aware strategy selection, time-series forecasting,
and signal quality backtesting.

기술 지표 기반 매매 신호 예측 패키지 - 단일/다중 종목 분석, 적응형 전략 선택,
시계열 예측, 신호 백테스트 기능을 제공합니다.

Features:
    - SignalPredictor: Single-symbol consensus signal prediction (70+ indicators)
    - MultiSignalAnalyzer: Multi-symbol parallel analysis with market consensus
    - SignalScanner: Condition-based symbol screening
    - AdaptiveSignalPredictor: Regime-aware automatic strategy selection
    - PriceForecast / TrendAnalyzer: Ensemble time-series forecasting
    - SignalBacktester: Signal quality verification vs. Buy&Hold / Random

Usage:
    from tradix.signals import SignalPredictor, MultiSignalAnalyzer

    predictor = SignalPredictor(df)
    result = predictor.predict()
    if result.isBuy:
        print(f"Buy signal! Strength: {result.strength:.2f}")

    analyzer = MultiSignalAnalyzer()
    analyzer.addSymbol("005930", samsung_df)
    market = analyzer.analyze()
    print(market.summary)
"""

from tradix.signals.predictor import (
    SignalPredictor,
    SignalResult,
    SignalConfig,
    SignalType,
)

from tradix.signals.analyzer import (
    MultiSignalAnalyzer,
    SignalScanner,
    SymbolSignal,
    MarketSignal,
)

from tradix.signals.forecast import (
    PriceForecast,
    TrendAnalyzer,
    ForecastResult,
)

from tradix.signals.backtest import (
    SignalBacktester,
    SignalBacktestResult,
    BacktestMetrics,
    BenchmarkComparison,
    quickEvaluate,
)

from tradix.signals.adaptive import (
    AdaptiveSignalPredictor,
    AdaptiveSignalResult,
    RegimeDetector,
    MarketRegime,
    quickAdaptivePredict,
)

__all__ = [
    "SignalPredictor",
    "SignalResult",
    "SignalConfig",
    "SignalType",
    "MultiSignalAnalyzer",
    "SignalScanner",
    "SymbolSignal",
    "MarketSignal",
    "PriceForecast",
    "TrendAnalyzer",
    "ForecastResult",
    "AdaptiveSignalPredictor",
    "AdaptiveSignalResult",
    "RegimeDetector",
    "MarketRegime",
    "quickAdaptivePredict",
]
