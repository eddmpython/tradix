"""
Tradex Analytics Package.

Provides performance analysis tools for backtest results, including
quantitative metrics computation and HTML tearsheet generation.

백테스트 결과에 대한 성과 분석 도구를 제공하는 패키지입니다.
정량적 지표 계산 및 HTML 리포트(Tearsheet) 생성 기능을 포함합니다.

Features:
    - PerformanceMetrics: Comprehensive backtest performance statistics
    - Tearsheet: QuantStats-style HTML performance report generator

Usage:
    from tradex.analytics import PerformanceMetrics, Tearsheet

    metrics = PerformanceMetrics.calculate(trades, equityCurve)
    tearsheet = Tearsheet(
        strategyName="SMA Cross",
        symbol="005930",
        metrics=metrics,
        trades=trades,
        equityCurve=equityCurve,
    )
    tearsheet.save("report.html")
"""

from tradex.analytics.metrics import PerformanceMetrics
from tradex.analytics.tearsheet import Tearsheet

__all__ = [
    "PerformanceMetrics",
    "Tearsheet",
]
