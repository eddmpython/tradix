"""
Tradix Trade Entity - Completed round-trip trade record.
Tradix 거래(Trade) 엔티티 - 완료된 왕복 거래 기록.

This module defines the Trade dataclass for recording completed round-trip
trades within the Tradix backtesting engine. A Trade is created when a
position is opened and completed when the position is closed, capturing
entry/exit prices, P&L, commissions, and holding period metrics.

Features:
    - Round-trip trade lifecycle (open to close)
    - Gross and net P&L calculation with commission/slippage deduction
    - Percentage return computation
    - Holding period tracking in both days and minutes
    - Win/loss classification for performance analysis

Usage:
    >>> from tradix.entities.trade import Trade
    >>> from tradix.entities.order import OrderSide
    >>> from datetime import datetime
    >>> trade = Trade(
    ...     id="t001", symbol="AAPL", side=OrderSide.BUY,
    ...     entryDate=datetime(2025, 1, 1), entryPrice=150.0, quantity=100
    ... )
    >>> trade.close(exitDate=datetime(2025, 1, 10), exitPrice=160.0, commission=20.0)
    >>> trade.pnl
    980.0
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tradix.entities.order import OrderSide


@dataclass
class Trade:
    """
    Completed round-trip trade record.
    완료된 왕복 거래 기록.

    Created when a position is opened and finalized when the position is
    closed. Tracks the full lifecycle including entry/exit timing, prices,
    transaction costs, and performance metrics such as P&L and holding period.

    포지션이 개시될 때 생성되고 청산될 때 최종 확정됩니다.
    진입/청산 시점, 가격, 거래 비용, 손익 및 보유 기간 등의
    전체 수명주기를 추적합니다.

    Attributes:
        id (str): Unique trade identifier. 거래 고유 ID.
        symbol (str): Ticker symbol identifier. 종목 코드.
        side (OrderSide): Entry direction (BUY or SELL). 진입 방향 (매수/매도).
        entryDate (datetime): Entry timestamp. 진입 일시.
        entryPrice (float): Entry execution price. 진입 가격.
        quantity (float): Trade quantity. 거래 수량.
        exitDate (Optional[datetime]): Exit timestamp (None if still open).
            청산 일시 (미청산 시 None).
        exitPrice (Optional[float]): Exit execution price (None if still open).
            청산 가격 (미청산 시 None).
        commission (float): Total commission charges (entry + exit). 총 수수료.
        slippage (float): Total slippage cost (entry + exit). 총 슬리피지.
        entryOrderId (Optional[str]): Associated entry order ID. 진입 주문 ID.
        exitOrderId (Optional[str]): Associated exit order ID. 청산 주문 ID.

    Example:
        >>> trade = Trade(
        ...     id="t001", symbol="AAPL", side=OrderSide.BUY,
        ...     entryDate=datetime(2025, 1, 1), entryPrice=150.0, quantity=100
        ... )
        >>> trade.isOpen
        True
        >>> trade.close(exitDate=datetime(2025, 1, 10), exitPrice=160.0)
        >>> trade.pnlPercent
        6.666...
    """
    id: str
    symbol: str
    side: OrderSide
    entryDate: datetime
    entryPrice: float
    quantity: float
    exitDate: Optional[datetime] = None
    exitPrice: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    entryOrderId: Optional[str] = None
    exitOrderId: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)

    @property
    def isOpen(self) -> bool:
        """Check if the trade is still open (not yet closed). 거래 미청산 여부."""
        return self.exitDate is None

    @property
    def isClosed(self) -> bool:
        """Check if the trade has been closed. 거래 청산 완료 여부."""
        return self.exitDate is not None

    @property
    def pnl(self) -> float:
        """Compute net P&L after commission and slippage. 순손익 (수수료/슬리피지 차감)."""
        if self.exitPrice is None:
            return 0.0

        if self.side == OrderSide.BUY:
            grossPnl = (self.exitPrice - self.entryPrice) * self.quantity
        else:
            grossPnl = (self.entryPrice - self.exitPrice) * self.quantity

        return grossPnl - self.commission - self.slippage

    @property
    def grossPnl(self) -> float:
        """Compute gross P&L before commission and slippage. 총손익 (수수료/슬리피지 미차감)."""
        if self.exitPrice is None:
            return 0.0

        if self.side == OrderSide.BUY:
            return (self.exitPrice - self.entryPrice) * self.quantity
        else:
            return (self.entryPrice - self.exitPrice) * self.quantity

    @property
    def pnlPercent(self) -> float:
        """Compute net P&L as a percentage of entry value. 순손익률(%)."""
        cost = self.entryPrice * self.quantity
        if cost == 0:
            return 0.0
        return (self.pnl / cost) * 100

    @property
    def grossPnlPercent(self) -> float:
        """Compute gross P&L as a percentage of entry value. 총손익률(%)."""
        cost = self.entryPrice * self.quantity
        if cost == 0:
            return 0.0
        return (self.grossPnl / cost) * 100

    @property
    def holdingDays(self) -> int:
        """Compute holding period in calendar days. 보유 기간 (일)."""
        if self.exitDate is None:
            return 0
        return (self.exitDate - self.entryDate).days

    @property
    def holdingMinutes(self) -> int:
        """Compute holding period in minutes. 보유 기간 (분)."""
        if self.exitDate is None:
            return 0
        delta = self.exitDate - self.entryDate
        return int(delta.total_seconds() / 60)

    @property
    def isWin(self) -> bool:
        """Check if this is a winning trade (positive P&L). 수익 거래 여부."""
        return self.pnl > 0

    @property
    def isLoss(self) -> bool:
        """Check if this is a losing trade (negative P&L). 손실 거래 여부."""
        return self.pnl < 0

    @property
    def entryValue(self) -> float:
        """Compute entry notional value. 진입 금액."""
        return self.entryPrice * self.quantity

    @property
    def exitValue(self) -> float:
        """Compute exit notional value. 청산 금액."""
        if self.exitPrice is None:
            return 0.0
        return self.exitPrice * self.quantity

    def close(self, exitDate: datetime, exitPrice: float, commission: float = 0.0, slippage: float = 0.0, orderId: str = None):
        """Close the trade with exit details. 거래 청산.

        Finalizes the trade by recording exit timing, price, and transaction
        costs. Commission and slippage are added to existing values (from entry).

        Args:
            exitDate: Exit timestamp. 청산 일시.
            exitPrice: Exit execution price. 청산 가격.
            commission: Commission charged on exit. 청산 수수료.
            slippage: Slippage incurred on exit. 청산 슬리피지.
            orderId: Associated exit order ID. 청산 주문 ID.
        """
        self.exitDate = exitDate
        self.exitPrice = exitPrice
        self.commission += commission
        self.slippage += slippage
        self.exitOrderId = orderId

    def __repr__(self) -> str:
        status = "OPEN" if self.isOpen else "CLOSED"
        pnlStr = f"P&L: {self.pnl:+,.0f} ({self.pnlPercent:+.2f}%)" if self.isClosed else ""
        return (
            f"Trade({self.symbol} {self.side.value.upper()} {self.quantity} "
            f"@ {self.entryPrice:,.0f} -> {self.exitPrice or '?':,.0f} [{status}] {pnlStr})"
        )
