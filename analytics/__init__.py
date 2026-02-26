"""
Tradex Analytics Package - Comprehensive backtest performance analysis.

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

Usage:
    from tradex.analytics import PerformanceMetrics, Tearsheet
    from tradex.analytics import StrategyDnaAnalyzer, BlackSwanAnalyzer
    from tradex.analytics import StrategyHealthAnalyzer, WhatIfSimulator
    from tradex.analytics import DrawdownSimulator, SeasonalityAnalyzer
    from tradex.analytics import CorrelationAnalyzer, TradingJournal
    from tradex.analytics import StrategyLeaderboard
"""

from tradex.analytics.metrics import PerformanceMetrics
from tradex.analytics.tearsheet import Tearsheet
from tradex.analytics.strategyDna import StrategyDNA, StrategyDnaAnalyzer
from tradex.analytics.blackSwan import BlackSwanScore, BlackSwanAnalyzer
from tradex.analytics.strategyHealth import StrategyHealthScore, StrategyHealthAnalyzer
from tradex.analytics.whatIf import WhatIfResult, WhatIfSimulator
from tradex.analytics.drawdownSimulator import DrawdownScenario, DrawdownSimulator
from tradex.analytics.seasonality import SeasonalPattern, SeasonalityAnalyzer
from tradex.analytics.correlation import CorrelationResult, CorrelationAnalyzer
from tradex.analytics.tradingJournal import JournalEntry, TradingJournal
from tradex.analytics.leaderboard import LeaderboardEntry, StrategyLeaderboard

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
]
