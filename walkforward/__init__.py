"""
Tradex Walk-Forward Analysis Package.

Provides rolling-window out-of-sample validation to detect and prevent
overfitting in strategy parameter optimization. Includes period splitting
utilities and comprehensive robustness analysis.

과적합 방지를 위한 롤링 윈도우 기반 Out-of-Sample 검증 패키지입니다.
기간 분할 유틸리티와 종합적인 견고성 분석을 제공합니다.

Features:
    - WalkForwardAnalyzer: Full walk-forward optimization pipeline
    - PeriodSplitter: Rolling, anchored, and K-fold period splitting
    - WalkForwardResult: Robustness ratio and parameter stability analysis
    - FoldResult: Individual fold in-sample / out-of-sample results

Usage:
    from tradex.walkforward import WalkForwardAnalyzer, PeriodSplitter

    wfa = WalkForwardAnalyzer(
        data=data,
        strategyFactory=createStrategy,
        parameterSpace=space,
        metric='sharpeRatio',
        inSampleMonths=12,
        outOfSampleMonths=3,
    )

    result = wfa.run()
    print(result.summary())
"""

from tradex.walkforward.splitter import PeriodSplitter
from tradex.walkforward.analyzer import WalkForwardAnalyzer, WalkForwardResult, FoldResult

__all__ = [
    'PeriodSplitter',
    'WalkForwardAnalyzer',
    'WalkForwardResult',
    'FoldResult',
]
