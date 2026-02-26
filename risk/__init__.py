"""
Tradex Risk Management Package.

Provides position sizing, pre-trade risk validation, and advanced risk
simulation tools for portfolio risk assessment during backtesting.

리스크 관리 패키지 - 포지션 사이징, 주문 검증, VaR, 몬테카를로 시뮬레이션
등 리스크 분석 도구를 제공합니다.

Features:
    - Position sizing: fixed, percent equity, fixed risk, Kelly, volatility-based
    - Risk manager with position limits, drawdown protection, and daily loss caps
    - Risk simulator with VaR (Historical/Parametric/Monte Carlo), stress testing
    - Tail risk analysis and drawdown statistics

Usage:
    >>> from tradex.risk import RiskManager, PercentEquitySizer, RiskSimulator
    >>> manager = RiskManager.moderate()
    >>> sizer = PercentEquitySizer(percent=0.1)
    >>> simulator = RiskSimulator()
"""

from tradex.risk.sizing import (
    PositionSizer,
    FixedQuantitySizer,
    FixedAmountSizer,
    PercentEquitySizer,
    FixedRiskSizer,
    KellySizer,
)
from tradex.risk.manager import RiskManager
from tradex.risk.simulator import (
    RiskSimulator,
    VaRResult,
    VaRMethod,
    MonteCarloResult,
    StressTestResult,
)

__all__ = [
    "PositionSizer",
    "FixedQuantitySizer",
    "FixedAmountSizer",
    "PercentEquitySizer",
    "FixedRiskSizer",
    "KellySizer",
    "RiskManager",
    "RiskSimulator",
    "VaRResult",
    "VaRMethod",
    "MonteCarloResult",
    "StressTestResult",
]
