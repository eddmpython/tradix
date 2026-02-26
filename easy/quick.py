"""
Tradex QuickStrategy Module - Declarative strategy builder with method chaining.

Build complete trading strategies without subclassing by chaining declarative
condition methods. Supports indicator-based conditions, lambda functions,
stop-loss, take-profit, and trailing stop. Also provides Korean-language
aliases for all methods.

선언형 전략 빌더 모듈 - 클래스 상속 없이 메서드 체이닝으로 전략을 구성합니다.
지표 기반 조건, 람다 함수, 손절/익절/추적손절을 지원하며
모든 메서드에 한글 별칭을 제공합니다.

Features:
    - Method chaining: .buyWhen().sellWhen().stopLoss().takeProfit()
    - Indicator conditions: sma, ema, rsi, macd, bollinger, etc.
    - Lambda conditions: lambda s, bar: bar.close > s.sma(20)
    - Risk management: stopLoss, takeProfit, trailingStop
    - Korean aliases: 매수조건, 매도조건, 손절, 익절, 추적손절

Usage:
    >>> from tradex.easy.quick import QuickStrategy
    >>> from tradex.easy.conditions import sma, rsi, crossover, crossunder
    >>>
    >>> strategy = (
    ...     QuickStrategy("MyStrategy")
    ...     .buyWhen(crossover(sma(10), sma(30)))
    ...     .sellWhen(crossunder(sma(10), sma(30)))
    ...     .stopLoss(5)
    ...     .takeProfit(15)
    ... )
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

from tradex.easy.conditions import (
    Condition,
    ConditionEvaluator,
    CrossCondition,
    Indicator,
)
from tradex.entities.bar import Bar
from tradex.strategy.base import Strategy


class QuickStrategy(Strategy):
    """
    Declarative strategy builder with fluent method chaining.

    Build complete trading strategies without subclassing Strategy.
    Chain conditions using .buyWhen(), .sellWhen(), and risk management
    methods. Supports both Indicator-based conditions and lambda functions.

    클래스 상속 없이 메서드 체이닝으로 전략을 구성하는 선언형 전략 빌더입니다.

    Attributes:
        name (str): Strategy name for identification and display.

    Example:
        >>> strategy = (
        ...     QuickStrategy("GoldenCross")
        ...     .buyWhen(crossover(sma(10), sma(30)))
        ...     .sellWhen(rsi(14) > 70)
        ...     .stopLoss(5)
        ...     .takeProfit(15)
        ... )
    """

    def __init__(self, name: str = "QuickStrategy"):
        super().__init__()
        self.name = name

        self._buyConditions: List[Union[Condition, Callable]] = []
        self._sellConditions: List[Union[Condition, Callable]] = []
        self._stopLossPct: Optional[float] = None
        self._takeProfitPct: Optional[float] = None
        self._trailingStopPct: Optional[float] = None

        self._entryPrice: float = 0.0
        self._highestPrice: float = 0.0

        self._evaluator: Optional[ConditionEvaluator] = None

    def buyWhen(self, condition: Union[Condition, Callable]) -> 'QuickStrategy':
        """
        Add a buy entry condition. / 매수 조건 추가.

        Args:
            condition: Condition object or lambda (strategy, bar) -> bool. / 조건 객체 또는 람다 함수.

        Returns:
            QuickStrategy: self for method chaining.

        Example:
            >>> strategy.buyWhen(crossover(sma(10), sma(30)))
            >>> strategy.buyWhen(rsi(14) < 30)
            >>> strategy.buyWhen(lambda s, bar: bar.close > s.sma(20))
        """
        self._buyConditions.append(condition)
        return self

    def sellWhen(self, condition: Union[Condition, Callable]) -> 'QuickStrategy':
        """
        Add a sell exit condition. / 매도 조건 추가.

        Args:
            condition: Condition object or lambda (strategy, bar) -> bool. / 조건 객체 또는 람다 함수.

        Returns:
            QuickStrategy: self for method chaining.

        Example:
            >>> strategy.sellWhen(crossunder(sma(10), sma(30)))
            >>> strategy.sellWhen(rsi(14) > 70)
        """
        self._sellConditions.append(condition)
        return self

    def stopLoss(self, percent: float) -> 'QuickStrategy':
        """
        Set stop-loss percentage. / 손절 비율 설정.

        Args:
            percent: Stop-loss percent (e.g., 5 = exit at 5% loss). / 손절 퍼센트.

        Returns:
            QuickStrategy: self for method chaining.
        """
        self._stopLossPct = percent / 100
        return self

    def takeProfit(self, percent: float) -> 'QuickStrategy':
        """
        Set take-profit percentage. / 익절 비율 설정.

        Args:
            percent: Take-profit percent (e.g., 15 = exit at 15% gain). / 익절 퍼센트.

        Returns:
            QuickStrategy: self for method chaining.
        """
        self._takeProfitPct = percent / 100
        return self

    def trailingStop(self, percent: float) -> 'QuickStrategy':
        """
        Set trailing stop percentage. / 추적손절 비율 설정.

        Args:
            percent: Trailing stop percent from peak (e.g., 10 = exit on 10% drop from high). / 고점 대비 하락 퍼센트.

        Returns:
            QuickStrategy: self for method chaining.
        """
        self._trailingStopPct = percent / 100
        return self

    def 매수조건(self, condition: Union[Condition, Callable]) -> 'QuickStrategy':
        """Korean alias for buyWhen(). / buyWhen()의 한글 별칭."""
        return self.buyWhen(condition)

    def 매도조건(self, condition: Union[Condition, Callable]) -> 'QuickStrategy':
        """Korean alias for sellWhen(). / sellWhen()의 한글 별칭."""
        return self.sellWhen(condition)

    def 손절(self, percent: float) -> 'QuickStrategy':
        """Korean alias for stopLoss(). / stopLoss()의 한글 별칭."""
        return self.stopLoss(percent)

    def 익절(self, percent: float) -> 'QuickStrategy':
        """Korean alias for takeProfit(). / takeProfit()의 한글 별칭."""
        return self.takeProfit(percent)

    def 추적손절(self, percent: float) -> 'QuickStrategy':
        """Korean alias for trailingStop(). / trailingStop()의 한글 별칭."""
        return self.trailingStop(percent)

    def initialize(self):
        """Initialize condition evaluator and position tracking state. / 조건 평가기 및 포지션 추적 상태 초기화."""
        self._evaluator = ConditionEvaluator(self)
        self._entryPrice = 0.0
        self._highestPrice = 0.0

    def onBar(self, bar: Bar):
        """Process each bar: check entry/exit conditions and manage positions. / 각 바 처리: 진입/청산 조건 확인 및 포지션 관리."""
        if self.hasPosition(bar.symbol):
            self._updatePosition(bar)
            if self._checkExitConditions(bar):
                return
            if self._checkSellConditions(bar):
                self._exitPosition(bar)
        else:
            if self._checkBuyConditions(bar):
                self._enterPosition(bar)

    def _checkBuyConditions(self, bar: Bar) -> bool:
        """Evaluate all buy conditions (OR logic). / 모든 매수 조건 평가 (OR 논리)."""
        if not self._buyConditions:
            return False

        for condition in self._buyConditions:
            if callable(condition) and not isinstance(condition, (Condition, CrossCondition)):
                if condition(self, bar):
                    return True
            elif self._evaluator.evaluate(condition, bar):
                return True

        return False

    def _checkSellConditions(self, bar: Bar) -> bool:
        """Evaluate all sell conditions (OR logic). / 모든 매도 조건 평가 (OR 논리)."""
        if not self._sellConditions:
            return False

        for condition in self._sellConditions:
            if callable(condition) and not isinstance(condition, (Condition, CrossCondition)):
                if condition(self, bar):
                    return True
            elif self._evaluator.evaluate(condition, bar):
                return True

        return False

    def _checkExitConditions(self, bar: Bar) -> bool:
        """Check stop-loss, take-profit, and trailing stop conditions. / 손절/익절/추적손절 조건 확인."""
        if self._entryPrice <= 0:
            return False

        currentReturn = (bar.close - self._entryPrice) / self._entryPrice

        if self._stopLossPct and currentReturn <= -self._stopLossPct:
            self._exitPosition(bar, reason="손절")
            return True

        if self._takeProfitPct and currentReturn >= self._takeProfitPct:
            self._exitPosition(bar, reason="익절")
            return True

        if self._trailingStopPct and self._highestPrice > 0:
            dropFromHigh = (self._highestPrice - bar.close) / self._highestPrice
            if dropFromHigh >= self._trailingStopPct:
                self._exitPosition(bar, reason="추적손절")
                return True

        return False

    def _updatePosition(self, bar: Bar):
        """Update highest price for trailing stop tracking. / 추적손절을 위한 최고가 업데이트."""
        if bar.high > self._highestPrice:
            self._highestPrice = bar.high

    def _enterPosition(self, bar: Bar):
        """Enter a new long position. / 신규 매수 포지션 진입."""
        self.buy(bar.symbol)
        self._entryPrice = bar.close
        self._highestPrice = bar.high

    def _exitPosition(self, bar: Bar, reason: str = ""):
        """Close the current position and reset tracking state. / 포지션 청산 및 추적 상태 초기화."""
        self.closePosition(bar.symbol)
        self._entryPrice = 0.0
        self._highestPrice = 0.0

    def clone(self) -> 'QuickStrategy':
        """Create a deep copy of this strategy for optimization. / 최적화를 위한 전략 복제."""
        new = QuickStrategy(self.name)
        new._buyConditions = self._buyConditions.copy()
        new._sellConditions = self._sellConditions.copy()
        new._stopLossPct = self._stopLossPct
        new._takeProfitPct = self._takeProfitPct
        new._trailingStopPct = self._trailingStopPct
        return new

    def __repr__(self) -> str:
        parts = [f"QuickStrategy('{self.name}')"]
        if self._buyConditions:
            parts.append(f".buyWhen({len(self._buyConditions)} conditions)")
        if self._sellConditions:
            parts.append(f".sellWhen({len(self._sellConditions)} conditions)")
        if self._stopLossPct:
            parts.append(f".stopLoss({self._stopLossPct*100:.0f}%)")
        if self._takeProfitPct:
            parts.append(f".takeProfit({self._takeProfitPct*100:.0f}%)")
        return "".join(parts)
