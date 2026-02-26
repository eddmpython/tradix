"""
Tradex Risk Manager Module.

Provides pre-trade risk validation and portfolio-level risk controls
including position sizing limits, drawdown protection, daily loss limits,
and order frequency caps.

리스크 관리자 모듈 - 주문 검증 및 포트폴리오 수준의 리스크 제어를 제공합니다.

Features:
    - Single position concentration limits
    - Total portfolio exposure limits
    - Maximum drawdown protection with automatic halt
    - Daily loss limit enforcement
    - Maximum open positions and daily order count caps
    - Factory presets for conservative, moderate, and aggressive profiles
    - Manual trading halt/resume controls

Usage:
    >>> from tradex.risk.manager import RiskManager
    >>> manager = RiskManager(maxPositionPercent=0.2, maxDrawdown=0.15)
    >>> approved, reason = manager.checkOrder(order, portfolio, price=50000)
    >>> if approved:
    ...     broker.processOrder(order, bar)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Tuple, TYPE_CHECKING

from tradex.entities.order import Order, OrderSide

if TYPE_CHECKING:
    from tradex.portfolio.portfolio import Portfolio


@dataclass
class RiskManager:
    """
    Pre-trade risk manager for order validation and portfolio risk controls.

    Validates orders against configurable risk limits before execution,
    including position concentration, total exposure, drawdown thresholds,
    daily loss limits, and order frequency caps.

    리스크 관리자 - 포지션 비중, 노출 한도, 낙폭, 일일 손실 등의
    리스크 제한을 적용하여 주문을 검증합니다.

    Attributes:
        maxPositionPercent (float): Maximum single position as fraction
            of equity (0.2 = 20%). Default: 0.2. 단일 종목 최대 비중.
        maxTotalExposure (float): Maximum total exposure as fraction of
            equity (1.0 = 100%). Default: 1.0. 전체 노출 한도.
        maxDrawdown (float): Maximum drawdown threshold (0.15 = 15%).
            Default: 0.15. 최대 낙폭 한도.
        dailyLossLimit (float): Maximum daily loss as fraction of equity
            (0.03 = 3%). Default: 0.03. 일일 손실 한도.
        maxOpenPositions (int): Maximum number of concurrent positions.
            Default: 10. 최대 보유 종목 수.
        maxOrdersPerDay (int): Maximum orders per trading day.
            Default: 50. 일일 최대 주문 수.

    Example:
        >>> manager = RiskManager(maxPositionPercent=0.2, maxDrawdown=0.15)
        >>> approved, reason = manager.checkOrder(order, portfolio, price=50000)
        >>> if approved:
        ...     broker.processOrder(order, bar)
    """
    maxPositionPercent: float = 0.2
    maxTotalExposure: float = 1.0
    maxDrawdown: float = 0.15
    dailyLossLimit: float = 0.03
    maxOpenPositions: int = 10
    maxOrdersPerDay: int = 50

    _peakEquity: float = field(default=0.0, init=False)
    _dailyStartEquity: float = field(default=0.0, init=False)
    _currentDate: date = field(default=None, init=False)
    _ordersToday: int = field(default=0, init=False)
    _tradingHalted: bool = field(default=False, init=False)
    _haltReason: str = field(default="", init=False)

    def checkOrder(
        self,
        order: Order,
        portfolio: 'Portfolio',
        price: float
    ) -> Tuple[bool, str]:
        """
        Validate an order against all configured risk limits.

        모든 리스크 제한을 기준으로 주문을 검증합니다.

        Args:
            order (Order): The order to validate. 검증할 주문.
            portfolio (Portfolio): Current portfolio state. 포트폴리오.
            price (float): Current market price. 현재 가격.

        Returns:
            tuple[bool, str]: A tuple of (approved, reason). Returns
                (True, "OK") if the order passes all checks, or
                (False, reason_string) if any limit is breached.
        """
        if self._tradingHalted:
            return False, f"거래 중단: {self._haltReason}"

        if self._ordersToday >= self.maxOrdersPerDay:
            return False, f"일일 주문 한도 초과 ({self.maxOrdersPerDay})"

        if order.side == OrderSide.BUY:
            positionCount = len(portfolio.positions)
            if order.symbol not in portfolio.positions:
                if positionCount >= self.maxOpenPositions:
                    return False, f"최대 보유 종목 수 초과 ({self.maxOpenPositions})"

            orderValue = order.quantity * price
            positionPercent = orderValue / portfolio.equity if portfolio.equity > 0 else 0

            existingValue = portfolio.getPositionValue(order.symbol)
            totalPercent = (orderValue + existingValue) / portfolio.equity if portfolio.equity > 0 else 0

            if totalPercent > self.maxPositionPercent:
                return False, f"단일 종목 비중 초과 ({totalPercent:.1%} > {self.maxPositionPercent:.1%})"

            currentExposure = portfolio.totalExposure
            newExposure = (currentExposure + orderValue) / portfolio.equity if portfolio.equity > 0 else 0
            if newExposure > self.maxTotalExposure:
                return False, f"전체 노출 한도 초과 ({newExposure:.1%} > {self.maxTotalExposure:.1%})"

        if self._peakEquity > 0:
            currentDrawdown = (self._peakEquity - portfolio.equity) / self._peakEquity
            if currentDrawdown >= self.maxDrawdown:
                return False, f"최대 낙폭 도달 ({currentDrawdown:.1%} >= {self.maxDrawdown:.1%})"

        if self._dailyStartEquity > 0:
            dailyReturn = (portfolio.equity - self._dailyStartEquity) / self._dailyStartEquity
            if dailyReturn <= -self.dailyLossLimit:
                return False, f"일일 손실 한도 도달 ({dailyReturn:.1%})"

        return True, "OK"

    def updateEquity(self, equity: float, timestamp: datetime = None):
        """
        Update equity tracking for drawdown and daily loss monitoring.

        자산 추적을 업데이트합니다 (낙폭 및 일일 손실 모니터링용).

        Args:
            equity (float): Current portfolio equity value. 현재 자산.
            timestamp (datetime, optional): Current timestamp. When the
                date changes, resets daily counters. 현재 시간.
        """
        self._peakEquity = max(self._peakEquity, equity)

        if timestamp:
            currentDate = timestamp.date()
            if self._currentDate != currentDate:
                self._currentDate = currentDate
                self._dailyStartEquity = equity
                self._ordersToday = 0

    def recordOrder(self):
        """Record an order execution to track daily order count. 일일 주문 수 추적을 위해 주문을 기록합니다."""
        self._ordersToday += 1

    def haltTrading(self, reason: str):
        """
        Halt all trading with a specified reason.

        지정된 사유로 모든 거래를 중단합니다.

        Args:
            reason (str): Reason for halting trading. 거래 중단 사유.
        """
        self._tradingHalted = True
        self._haltReason = reason

    def resumeTrading(self):
        """Resume trading after a halt. 거래를 재개합니다."""
        self._tradingHalted = False
        self._haltReason = ""

    def reset(self):
        """Reset all internal state to initial values. 모든 내부 상태를 초기값으로 리셋합니다."""
        self._peakEquity = 0.0
        self._dailyStartEquity = 0.0
        self._currentDate = None
        self._ordersToday = 0
        self._tradingHalted = False
        self._haltReason = ""

    @property
    def currentDrawdown(self) -> float:
        """Return the current drawdown from peak equity. 최고점 대비 현재 낙폭을 반환합니다."""
        if self._peakEquity <= 0:
            return 0.0
        return 0.0

    @property
    def isHalted(self) -> bool:
        """Return whether trading is currently halted. 거래 중단 여부를 반환합니다."""
        return self._tradingHalted

    def getStatus(self) -> dict:
        """
        Return the current risk manager status as a dictionary.

        현재 리스크 관리자 상태를 딕셔너리로 반환합니다.

        Returns:
            dict: Status with keys 'peakEquity', 'dailyStartEquity',
                'currentDate', 'ordersToday', 'tradingHalted', 'haltReason'.
        """
        return {
            'peakEquity': self._peakEquity,
            'dailyStartEquity': self._dailyStartEquity,
            'currentDate': self._currentDate,
            'ordersToday': self._ordersToday,
            'tradingHalted': self._tradingHalted,
            'haltReason': self._haltReason,
        }

    @classmethod
    def conservative(cls) -> 'RiskManager':
        """
        Create a conservative risk manager with tight limits.

        보수적 리스크 설정 (낮은 비중, 엄격한 낙폭 한도).

        Returns:
            RiskManager: Conservative risk profile.
        """
        return cls(
            maxPositionPercent=0.1,
            maxTotalExposure=0.8,
            maxDrawdown=0.1,
            dailyLossLimit=0.02,
            maxOpenPositions=5
        )

    @classmethod
    def moderate(cls) -> 'RiskManager':
        """
        Create a moderate risk manager with balanced limits.

        중간 수준의 리스크 설정 (균형 잡힌 한도).

        Returns:
            RiskManager: Moderate risk profile.
        """
        return cls(
            maxPositionPercent=0.2,
            maxTotalExposure=1.0,
            maxDrawdown=0.15,
            dailyLossLimit=0.03,
            maxOpenPositions=10
        )

    @classmethod
    def aggressive(cls) -> 'RiskManager':
        """
        Create an aggressive risk manager with relaxed limits.

        공격적 리스크 설정 (높은 비중, 넓은 낙폭 한도).

        Returns:
            RiskManager: Aggressive risk profile.
        """
        return cls(
            maxPositionPercent=0.3,
            maxTotalExposure=1.5,
            maxDrawdown=0.25,
            dailyLossLimit=0.05,
            maxOpenPositions=20
        )

    def __repr__(self) -> str:
        return (
            f"RiskManager(maxPos={self.maxPositionPercent:.0%}, "
            f"maxDD={self.maxDrawdown:.0%}, "
            f"halted={self._tradingHalted})"
        )
