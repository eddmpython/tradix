"""
Tradix Order Entity - Order management and execution tracking.
Tradix 주문(Order) 엔티티 - 주문 관리 및 체결 추적.

This module defines the Order dataclass and its associated enumerations for
representing and managing trading orders within the Tradix backtesting engine.
It supports market, limit, stop, and stop-limit order types with full lifecycle
tracking from creation through fill or cancellation.

Features:
    - Multiple order types: market, limit, stop, stop-limit
    - Order lifecycle management: pending, submitted, partial, filled, cancelled, rejected
    - Automatic fill tracking with weighted average price calculation
    - Commission and slippage accounting
    - Time-in-force policies: DAY, GTC, IOC

Usage:
    >>> from tradix.entities.order import Order, OrderSide, OrderType
    >>> order = Order(
    ...     symbol="AAPL",
    ...     side=OrderSide.BUY,
    ...     orderType=OrderType.MARKET,
    ...     quantity=100,
    ... )
    >>> order.fill(price=150.0, quantity=100, commission=1.5)
    >>> order.isFilled
    True
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderSide(Enum):
    """Order direction enumeration. 주문 방향 열거형."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration. 주문 유형 열거형."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stopLimit"
    TRAILING_STOP = "trailingStop"


class TimeInForce(Enum):
    """Order time-in-force policy enumeration. 주문 유효 기간 열거형."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(Enum):
    """Order status enumeration. 주문 상태 열거형."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Trading order data container with full lifecycle management.
    주문 데이터 컨테이너 - 전체 주문 수명주기 관리.

    Represents a single trading order from creation through execution or
    cancellation. Supports partial fills with weighted average price
    recalculation, commission/slippage tracking, and automatic status
    transitions. String values for enum fields are auto-converted in __post_init__.

    단일 거래 주문을 생성부터 체결 또는 취소까지 표현합니다.
    가중 평균 가격 재계산을 통한 부분 체결, 수수료/슬리피지 추적,
    자동 상태 전환을 지원합니다.

    Attributes:
        symbol (str): Ticker symbol identifier. 종목 코드.
        side (OrderSide): Order direction (BUY or SELL). 주문 방향 (매수/매도).
        orderType (OrderType): Order type (market, limit, stop, stop-limit).
            주문 유형 (시장가, 지정가, 스톱, 스톱리밋).
        quantity (float): Requested order quantity. 주문 수량.
        price (Optional[float]): Limit price for limit orders. 지정가 (limit 주문용).
        stopPrice (Optional[float]): Trigger price for stop orders. 손절가 (stop 주문용).
        timeInForce (TimeInForce): Order validity policy (defaults to DAY).
            주문 유효 기간 (기본값 DAY).
        id (str): Unique order identifier (auto-generated 8-char UUID).
            주문 고유 ID (자동 생성 8자 UUID).
        createdAt (Optional[datetime]): Order creation timestamp (auto-set to now).
            주문 생성 시간 (자동으로 현재 시간 설정).
        filledQuantity (float): Cumulative filled quantity. 체결된 총 수량.
        filledPrice (float): Volume-weighted average fill price. 가중 평균 체결 가격.
        commission (float): Cumulative commission charges. 누적 수수료.
        slippage (float): Cumulative slippage cost. 누적 슬리피지.
        status (OrderStatus): Current order status. 현재 주문 상태.
        rejectReason (Optional[str]): Rejection reason if order was rejected.
            주문 거부 사유.

    Example:
        >>> order = Order(
        ...     symbol="005930", side=OrderSide.BUY,
        ...     orderType=OrderType.LIMIT, quantity=10, price=70000
        ... )
        >>> order.fill(price=70000, quantity=5, commission=350)
        >>> order.remainingQuantity
        5.0
    """
    symbol: str
    side: OrderSide
    orderType: OrderType
    quantity: float
    price: Optional[float] = None
    stopPrice: Optional[float] = None
    trailingPercent: Optional[float] = None
    parentId: Optional[str] = None
    timeInForce: TimeInForce = TimeInForce.DAY

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    createdAt: Optional[datetime] = None

    filledQuantity: float = 0.0
    filledPrice: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    rejectReason: Optional[str] = None

    def __post_init__(self):
        if self.createdAt is None:
            self.createdAt = datetime.now()

        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.orderType, str):
            self.orderType = OrderType(self.orderType)
        if isinstance(self.timeInForce, str):
            self.timeInForce = TimeInForce(self.timeInForce)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)

    @property
    def isBuy(self) -> bool:
        """Check if this is a buy order. 매수 주문 여부."""
        return self.side == OrderSide.BUY

    @property
    def isSell(self) -> bool:
        """Check if this is a sell order. 매도 주문 여부."""
        return self.side == OrderSide.SELL

    @property
    def isMarket(self) -> bool:
        """Check if this is a market order. 시장가 주문 여부."""
        return self.orderType == OrderType.MARKET

    @property
    def isLimit(self) -> bool:
        """Check if this is a limit order. 지정가 주문 여부."""
        return self.orderType == OrderType.LIMIT

    @property
    def isFilled(self) -> bool:
        """Check if the order is fully filled. 전량 체결 여부."""
        return self.status == OrderStatus.FILLED

    @property
    def isPending(self) -> bool:
        """Check if the order is still active (pending, submitted, or partial). 주문 활성 여부."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)

    @property
    def remainingQuantity(self) -> float:
        """Compute unfilled quantity remaining. 미체결 잔량."""
        return self.quantity - self.filledQuantity

    @property
    def filledValue(self) -> float:
        """Compute total filled notional value. 체결 금액."""
        return self.filledQuantity * self.filledPrice

    @property
    def totalCost(self) -> float:
        """Compute total cost including commission and slippage. 총 비용 (체결금액 + 수수료 + 슬리피지)."""
        return self.filledValue + self.commission + self.slippage

    def fill(self, price: float, quantity: float, commission: float = 0.0, slippage: float = 0.0):
        """Process a fill execution against this order. 주문 체결 처리.

        Handles partial and full fills. Recalculates the volume-weighted
        average fill price and updates the order status accordingly.

        Args:
            price: Execution price for this fill. 체결 가격.
            quantity: Execution quantity for this fill. 체결 수량.
            commission: Commission charged for this fill. 수수료.
            slippage: Slippage incurred for this fill. 슬리피지.
        """
        if quantity > self.remainingQuantity:
            quantity = self.remainingQuantity

        totalFilledValue = (self.filledPrice * self.filledQuantity) + (price * quantity)
        self.filledQuantity += quantity
        self.filledPrice = totalFilledValue / self.filledQuantity if self.filledQuantity > 0 else 0
        self.commission += commission
        self.slippage += slippage

        if self.filledQuantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def cancel(self):
        """Cancel the order if it is still pending. 미체결 주문 취소."""
        if self.isPending:
            self.status = OrderStatus.CANCELLED

    def reject(self, reason: str):
        """Reject the order with a specified reason. 주문 거부 처리.

        Args:
            reason: Human-readable rejection reason. 거부 사유.
        """
        self.status = OrderStatus.REJECTED
        self.rejectReason = reason

    def __repr__(self) -> str:
        return (
            f"Order(id={self.id}, {self.side.value} {self.quantity} {self.symbol} "
            f"@ {self.orderType.value}, status={self.status.value})"
        )
