"""Tradex Advisor Package.

Provides market regime classification, strategy recommendation, regime
forecasting, and empirically learned pattern mappings for intelligent
trading strategy selection and parameter tuning.

전략 추천 및 시장 예측 패키지 - 시장 레짐 분류, 전략 추천, 레짐 예측,
학습된 패턴 매핑을 통해 지능적인 전략 선택과 파라미터 튜닝을 지원합니다.

Features:
    - MarketClassifier: Six-regime classification using multi-indicator analysis
    - StrategyAdvisor: Regime-aware strategy recommendation with scoring
    - RegimeForecaster: Markov Chain-based regime transition forecasting
    - Learned Patterns: Empirical strategy-regime mappings from 10-stock x 3-period backtests
    - Convenience functions: getRecommendedStrategies, getAdjustedParams, getBenchmark

Usage:
    from tradex.advisor import MarketClassifier, StrategyAdvisor, RegimeForecaster

    classifier = MarketClassifier()
    analysis = classifier.analyze(df)

    advisor = StrategyAdvisor()
    result = advisor.recommend(df)
    print(result.topStrategy.name)
"""

from tradex.advisor.marketClassifier import MarketClassifier, MarketRegime, MarketAnalysis
from tradex.advisor.strategyAdvisor import (
    StrategyAdvisor,
    StrategyProfile,
    StrategyRecommendation,
    AdvisorResult,
)
from tradex.advisor.learnedPatterns import (
    REGIME_STRATEGY_MAP,
    STRATEGY_PARAM_ADJUSTMENTS,
    PERFORMANCE_BENCHMARKS,
    KEY_INSIGHTS,
    getRecommendedStrategies,
    getAdjustedParams,
    getBenchmark,
)
from tradex.advisor.regimeForecaster import (
    RegimeForecaster,
    RegimeForecast,
    RegimeTransition,
    TransitionSignal,
)

__all__ = [
    "MarketClassifier",
    "MarketRegime",
    "MarketAnalysis",
    "StrategyAdvisor",
    "StrategyProfile",
    "StrategyRecommendation",
    "AdvisorResult",
    "REGIME_STRATEGY_MAP",
    "STRATEGY_PARAM_ADJUSTMENTS",
    "PERFORMANCE_BENCHMARKS",
    "KEY_INSIGHTS",
    "getRecommendedStrategies",
    "getAdjustedParams",
    "getBenchmark",
    "RegimeForecaster",
    "RegimeForecast",
    "RegimeTransition",
    "TransitionSignal",
]
