"""
Tradix Fill Event - Order execution confirmation.
Tradix 체결 이벤트 - 주문 체결 확인.

This module defines the FillEvent class, which is emitted when an order is
executed (filled) in the Tradix backtesting engine. FillEvents are the final
stage in the event pipeline and carry execution details including fill price,
quantity, commission, and slippage for portfolio and position updates.

Features:
    - Complete fill execution details (price, quantity, costs)
    - Commission and slippage percentage calculations
    - Total cost computation (fill value + commission + slippage)
    - Part of the event pipeline: OrderEvent -> FillEvent -> Portfolio update

Usage:
    >>> from tradix.events.fill import FillEvent
    >>> from tradix.entities.order import Order, OrderSide, OrderType
    >>> from datetime import datetime
    >>> order = Order(symbol="AAPL", side=OrderSide.BUY, orderType=OrderType.MARKET, quantity=100)
    >>> fill = FillEvent(
    ...     timestamp=datetime.now(), order=order,
    ...     fillPrice=150.0, fillQuantity=100, commission=1.5
    ... )
    >>> fill.fillValue
    15000.0
"""

from dataclasses import dataclass, field
from datetime import datetime

from tradix.events.base import Event, EventType
from tradix.entities.order import Order


@dataclass
class FillEvent(Event):
    """
    Order fill (execution) event emitted when an order is executed.
    주문이 체결되었을 때 발생하는 체결 이벤트.

    The final event in the Tradix processing pipeline. Contains execution
    details including fill price, quantity, and transaction costs. Used by
    the portfolio manager to update positions and track performance.

    Tradix 처리 파이프라인의 마지막 이벤트입니다. 체결 가격, 수량, 거래 비용
    등의 체결 세부 정보를 포함합니다. 포트폴리오 매니저가 포지션 업데이트 및
    성과 추적에 사용합니다.

    Attributes:
        order (Order): The filled order entity. 체결된 주문 객체.
        fillPrice (float): Execution price. 체결 가격.
        fillQuantity (float): Executed quantity. 체결 수량.
        commission (float): Commission charged for the fill. 수수료.
        slippage (float): Slippage cost incurred. 슬리피지 금액.

    Example:
        >>> fill = FillEvent(
        ...     timestamp=datetime.now(), order=order,
        ...     fillPrice=150.0, fillQuantity=100,
        ...     commission=1.5, slippage=0.5
        ... )
        >>> fill.totalCost
        15002.0
    """
    order: Order = None
    fillPrice: float = 0.0
    fillQuantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    eventType: EventType = field(default=EventType.FILL, init=False)

    @property
    def symbol(self) -> str:
        """Return the ticker symbol from the wrapped order. 종목 코드."""
        return self.order.symbol if self.order else ""

    @property
    def orderId(self) -> str:
        """Return the order ID from the wrapped order. 주문 ID."""
        return self.order.id if self.order else ""

    @property
    def side(self):
        """Return the order side (buy/sell) from the wrapped order. 주문 방향."""
        return self.order.side if self.order else None

    @property
    def fillValue(self) -> float:
        """Compute fill notional value (price * quantity). 체결 금액."""
        return self.fillPrice * self.fillQuantity

    @property
    def totalCost(self) -> float:
        """Compute total cost (fill value + commission + slippage). 총 비용."""
        return self.fillValue + self.commission + self.slippage

    @property
    def slippagePercent(self) -> float:
        """Compute slippage as a percentage of fill value. 슬리피지 비율(%)."""
        if self.fillValue == 0:
            return 0.0
        return (self.slippage / self.fillValue) * 100

    @property
    def commissionPercent(self) -> float:
        """Compute commission as a percentage of fill value. 수수료 비율(%)."""
        if self.fillValue == 0:
            return 0.0
        return (self.commission / self.fillValue) * 100

    def __repr__(self) -> str:
        if self.order:
            return (
                f"FillEvent({self.order.side.value.upper()} {self.fillQuantity} {self.symbol} "
                f"@ {self.fillPrice:,.0f}, commission={self.commission:,.0f}, slippage={self.slippage:,.0f})"
            )
        return f"FillEvent(empty)"
