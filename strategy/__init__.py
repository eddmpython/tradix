"""
Tradex Strategy Package - Core strategy framework for backtesting.

Provides the abstract Strategy base class for implementing custom trading
logic and the Indicators engine for technical analysis computation.

전략 패키지 - 백테스팅을 위한 핵심 전략 프레임워크.
커스텀 트레이딩 로직 구현을 위한 Strategy 기반 클래스와
기술 분석 계산을 위한 Indicators 엔진을 제공합니다.

Usage:
    >>> from tradex.strategy import Strategy, Indicators
    >>>
    >>> class MyStrategy(Strategy):
    ...     def onBar(self, bar):
    ...         if self.sma(10) > self.sma(30):
    ...             self.buy(bar.symbol)
"""

from tradex.strategy.base import Strategy
from tradex.strategy.indicators import Indicators

__all__ = [
    "Strategy",
    "Indicators",
]
