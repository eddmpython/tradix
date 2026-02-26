"""
Tradex Position Entity - Portfolio position tracking and P&L calculation.
Tradex 포지션(Position) 엔티티 - 포트폴리오 포지션 추적 및 손익 계산.

This module defines the Position dataclass for tracking open trading positions
within the Tradex backtesting engine. It manages average cost basis, market
valuation, and both realized and unrealized profit/loss calculations.

Features:
    - Average price recalculation on position additions
    - Realized P&L tracking on partial or full position reduction
    - Unrealized P&L computation with current market price
    - Long/short position detection
    - Full position closure with automatic P&L settlement

Usage:
    >>> from tradex.entities.position import Position
    >>> pos = Position(symbol="AAPL", quantity=100, avgPrice=150.0)
    >>> pos.updatePrice(160.0)
    >>> pos.unrealizedPnl
    1000.0
    >>> pos.unrealizedPnlPercent
    6.666...
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    """
    Trading position data container with P&L tracking.
    거래 포지션 데이터 컨테이너 - 손익 추적 기능 포함.

    Represents an open position in a single instrument, tracking the average
    cost basis, current market valuation, and cumulative realized profit/loss
    from partial reductions. Supports both long and short positions.

    단일 종목의 오픈 포지션을 표현하며, 평균 원가, 현재 시가 평가액,
    부분 청산에 따른 누적 실현 손익을 추적합니다.
    롱 및 숏 포지션 모두 지원합니다.

    Attributes:
        symbol (str): Ticker symbol identifier. 종목 코드.
        quantity (float): Current position quantity (positive=long, negative=short).
            현재 보유 수량 (양수=롱, 음수=숏).
        avgPrice (float): Volume-weighted average entry price. 평균 매수가.
        currentPrice (float): Latest market price for valuation (defaults to avgPrice).
            현재가 (기본값은 avgPrice).
        openedAt (Optional[datetime]): Timestamp when position was opened (auto-set to now).
            포지션 개시 시간 (자동으로 현재 시간 설정).
        realizedPnl (float): Cumulative realized P&L from partial closures.
            부분 청산에 의한 누적 실현 손익.

    Example:
        >>> pos = Position(symbol="005930", quantity=100, avgPrice=70000)
        >>> pos.updatePrice(73000)
        >>> pos.unrealizedPnl
        300000.0
    """
    symbol: str
    quantity: float
    avgPrice: float
    currentPrice: float = 0.0
    openedAt: Optional[datetime] = None
    realizedPnl: float = 0.0

    def __post_init__(self):
        if self.openedAt is None:
            self.openedAt = datetime.now()
        if self.currentPrice == 0.0:
            self.currentPrice = self.avgPrice

    @property
    def marketValue(self) -> float:
        """Compute current market value of the position. 시장 가치 (현재가 기준)."""
        return self.quantity * self.currentPrice

    @property
    def costBasis(self) -> float:
        """Compute total cost basis of the position. 취득 원가."""
        return self.quantity * self.avgPrice

    @property
    def unrealizedPnl(self) -> float:
        """Compute unrealized profit/loss. 미실현 손익."""
        return self.marketValue - self.costBasis

    @property
    def unrealizedPnlPercent(self) -> float:
        """Compute unrealized P&L as a percentage of cost basis. 미실현 손익률(%)."""
        if self.costBasis == 0:
            return 0.0
        return (self.unrealizedPnl / self.costBasis) * 100

    @property
    def totalPnl(self) -> float:
        """Compute total P&L (realized + unrealized). 총 손익 (실현 + 미실현)."""
        return self.realizedPnl + self.unrealizedPnl

    @property
    def isLong(self) -> bool:
        """Check if the position is long (positive quantity). 롱 포지션 여부."""
        return self.quantity > 0

    @property
    def isShort(self) -> bool:
        """Check if the position is short (negative quantity). 숏 포지션 여부."""
        return self.quantity < 0

    def updatePrice(self, price: float):
        """Update the current market price for valuation. 현재가 업데이트.

        Args:
            price: New market price. 새로운 시장 가격.
        """
        self.currentPrice = price

    def addQuantity(self, quantity: float, price: float):
        """Add to the position and recalculate the average entry price. 포지션 추가 (평균 단가 재계산).

        Args:
            quantity: Quantity to add (positive for buys). 추가 수량.
            price: Execution price for the addition. 추가 매수 가격.
        """
        if quantity == 0:
            return

        totalCost = self.costBasis + (quantity * price)
        self.quantity += quantity

        if self.quantity != 0:
            self.avgPrice = totalCost / self.quantity
        else:
            self.avgPrice = 0

    def reduceQuantity(self, quantity: float, price: float) -> float:
        """Reduce the position and compute realized P&L. 포지션 감소 (실현 손익 계산).

        Args:
            quantity: Quantity to reduce (capped at current position size). 감소 수량.
            price: Execution price for the reduction. 매도 가격.

        Returns:
            float: Realized profit/loss from the reduction. 실현 손익.
        """
        if quantity > abs(self.quantity):
            quantity = abs(self.quantity)

        pnl = (price - self.avgPrice) * quantity
        self.realizedPnl += pnl
        self.quantity -= quantity

        if self.quantity == 0:
            self.avgPrice = 0

        return pnl

    def close(self, price: float) -> float:
        """Close the entire position. 포지션 전량 청산.

        Args:
            price: Execution price for the closure. 청산 가격.

        Returns:
            float: Realized profit/loss from closing the position. 실현 손익.
        """
        return self.reduceQuantity(abs(self.quantity), price)

    def __repr__(self) -> str:
        direction = "LONG" if self.isLong else "SHORT" if self.isShort else "FLAT"
        return (
            f"Position({self.symbol} {direction} {abs(self.quantity)} "
            f"@ {self.avgPrice:,.0f}, P&L: {self.unrealizedPnl:+,.0f} ({self.unrealizedPnlPercent:+.2f}%))"
        )
