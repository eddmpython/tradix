"""
Tradix Analytics Package - Comprehensive backtest performance analysis.

Provides a full suite of analysis tools for backtest results, including
quantitative metrics, HTML tearsheet generation, strategy DNA fingerprinting,
Black Swan resilience scoring, strategy health diagnostics, what-if simulation,
drawdown analysis, seasonality detection, correlation analysis, trading journals,
and strategy leaderboards.

백테스트 결과에 대한 종합 분석 도구를 제공하는 패키지입니다.
성과 지표, HTML 리포트, 전략 DNA 지문, 블랙스완 방어 점수, 전략 건강 점수,
What-If 시뮬레이션, 낙폭 분석, 계절성 분석, 상관관계, 트레이딩 일지,
전략 리더보드 기능을 포함합니다.

Features:
    - PerformanceMetrics: Comprehensive backtest performance statistics
    - Tearsheet: QuantStats-style HTML performance report generator
    - StrategyDNA: 12-dimensional strategy fingerprinting (WORLD-FIRST)
    - BlackSwanAnalyzer: Extreme event resilience scoring (WORLD-FIRST)
    - StrategyHealthAnalyzer: Overfitting/stability diagnostics (WORLD-FIRST)
    - WhatIfSimulator: Commission/slippage/capital sensitivity analysis
    - DrawdownSimulator: Historical worst-case drawdown scenarios
    - SeasonalityAnalyzer: Monthly/weekday/quarterly pattern discovery
    - CorrelationAnalyzer: Multi-strategy correlation and clustering
    - TradingJournal: Automatic trade diary with MFE/MAE analytics
    - StrategyLeaderboard: Multi-strategy ranking and badge system
    - MonteCarloStressAnalyzer: 10,000-path bootstrap stress testing (WORLD-FIRST)
    - FractalAnalyzer: Hurst exponent & fractal dimension analysis (WORLD-FIRST)
    - RegimeDetector: HMM-based market regime detection & decomposition (WORLD-FIRST)
    - InformationTheoryAnalyzer: Entropy & mutual information signal analysis (WORLD-FIRST)
    - PortfolioStressAnalyzer: Hypothetical crisis scenario simulation (WORLD-FIRST)

Usage:
    from tradix.analytics import PerformanceMetrics, Tearsheet
    from tradix.analytics import StrategyDnaAnalyzer, BlackSwanAnalyzer
    from tradix.analytics import StrategyHealthAnalyzer, WhatIfSimulator
    from tradix.analytics import DrawdownSimulator, SeasonalityAnalyzer
    from tradix.analytics import CorrelationAnalyzer, TradingJournal
    from tradix.analytics import StrategyLeaderboard
"""

from tradix.analytics.metrics import PerformanceMetrics
from tradix.analytics.tearsheet import Tearsheet
from tradix.analytics.strategyDna import StrategyDNA, StrategyDnaAnalyzer
from tradix.analytics.blackSwan import BlackSwanScore, BlackSwanAnalyzer
from tradix.analytics.strategyHealth import StrategyHealthScore, StrategyHealthAnalyzer
from tradix.analytics.whatIf import WhatIfResult, WhatIfSimulator
from tradix.analytics.drawdownSimulator import DrawdownScenario, DrawdownSimulator
from tradix.analytics.seasonality import SeasonalPattern, SeasonalityAnalyzer
from tradix.analytics.correlation import CorrelationResult, CorrelationAnalyzer
from tradix.analytics.tradingJournal import JournalEntry, TradingJournal
from tradix.analytics.leaderboard import LeaderboardEntry, StrategyLeaderboard
from tradix.analytics.monteCarloStress import MonteCarloStressResult, MonteCarloStressAnalyzer
from tradix.analytics.fractalAnalysis import FractalAnalysisResult, FractalAnalyzer
from tradix.analytics.regimeDetector import RegimeAnalysisResult, RegimeDetector
from tradix.analytics.informationTheory import InformationTheoryResult, InformationTheoryAnalyzer
from tradix.analytics.portfolioStress import StressScenario, PortfolioStressResult, PortfolioStressAnalyzer

__all__ = [
    "PerformanceMetrics",
    "Tearsheet",

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
