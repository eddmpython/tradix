"""Tradex Quantitative Extension Package.

Extends the core Tradex backtesting engine with quantitative finance tools,
building on top of Tradex's regime forecasting and learned pattern infrastructure.

Tradex 퀀트 확장 패키지 - 기존 백테스트 엔진과 학습된 패턴 위에
팩터 분석, 통계적 차익거래, 레짐-팩터 통합 기능을 제공합니다.

Features:
    - Factor Analysis: exposure analysis, rolling analysis, return attribution
    - Factor Model: multi-factor risk/return decomposition (Fama-French style)
    - Statistical Arbitrage: cointegration testing, pair discovery
    - Pair Trading: Z-score signal generation, backtesting
    - Regime-Factor Integration: regime-conditional factor optimization

Usage:
    from tradex.quant import FactorAnalyzer, StatArbAnalyzer, RegimeFactorIntegration

    analyzer = FactorAnalyzer()
    analyzer.setFactors(factor_returns)
    result = analyzer.analyze(portfolio_returns)

    arb = StatArbAnalyzer()
    pairs = arb.findPairs(price_data, topN=5)
"""

from tradex.quant.factor import (
    FactorAnalyzer,
    FactorModel,
    Factor,
    FactorExposure,
    FactorResult,
)
from tradex.quant.statarb import (
    StatArbAnalyzer,
    PairTrading,
    CointegrationResult,
    SpreadSignal,
)
from tradex.quant.integration import (
    RegimeFactorIntegration,
    AdaptiveFactorStrategy,
    IntegratedSignal,
)

__all__ = [
    # Factor Analysis
    "FactorAnalyzer",
    "FactorModel",
    "Factor",
    "FactorExposure",
    "FactorResult",
    # Statistical Arbitrage
    "StatArbAnalyzer",
    "PairTrading",
    "CointegrationResult",
    "SpreadSignal",
    # Integration
    "RegimeFactorIntegration",
    "AdaptiveFactorStrategy",
    "IntegratedSignal",
]
