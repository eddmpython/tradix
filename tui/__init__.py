"""
Tradex TUI - Rich-based terminal user interface for backtest results.

Provides beautiful terminal output including styled tables, progress bars,
and terminal charts using Rich and Plotext.

Tradex TUI - Rich 기반 터미널 UI. 백테스트 결과를 테이블, 프로그레스 바,
터미널 차트로 출력합니다.
"""

from tradex.tui.console import (
    console,
    printResult,
    printComparison,
    printTrades,
    printMonthlyHeatmap,
    printStrategyDna,
    printHealthScore,
    printBlackSwanScore,
)
from tradex.tui.charts import (
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
from tradex.tui.progress import optimizeProgress, backtestProgress

__all__ = [
    "console",
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
    "optimizeProgress",
    "backtestProgress",
]
