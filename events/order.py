"""
Tradix Order Event - Order submission notification.
Tradix 주문 이벤트 - 주문 제출 알림.

This module defines the OrderEvent class, which is emitted when a trading
order is created and submitted to the broker within the Tradix backtesting
engine. It wraps an Order entity and serves as the bridge between signal
processing and order execution.

Features:
    - Wraps an Order entity with event metadata (timestamp, ID)
    - Convenience properties for symbol and order ID access
    - Part of the event pipeline: SignalEvent -> OrderEvent -> FillEvent

Usage:
    >>> from tradix.events.order import OrderEvent
    >>> from tradix.entities.order import Order, OrderSide, OrderType
    >>> from datetime import datetime
    >>> order = Order(symbol="AAPL", side=OrderSide.BUY, orderType=OrderType.MARKET, quantity=100)
    >>> event = OrderEvent(timestamp=datetime.now(), order=order)
    >>> event.symbol
    'AAPL'
"""

from dataclasses import dataclass, field
from datetime import datetime

from tradix.events.base import Event, EventType
from tradix.entities.order import Order


@dataclass
class OrderEvent(Event):
    """
    Order submission event emitted when an order is sent to the broker.
    주문이 생성되어 브로커로 전송될 때 발생하는 주문 이벤트.

    Bridges signal processing and order execution in the event pipeline.
    Wraps an Order entity with event metadata for the event queue.

    시그널 처리와 주문 체결 사이의 이벤트 파이프라인을 연결합니다.
    이벤트 큐를 위해 Order 엔티티를 이벤트 메타데이터로 감쌉니다.

    Attributes:
        order (Order): The order entity being submitted. 제출되는 주문 객체.

    Example:
        >>> order = Order(symbol="AAPL", side=OrderSide.BUY, orderType=OrderType.MARKET, quantity=100)
        >>> event = OrderEvent(timestamp=datetime.now(), order=order)
        >>> event.symbol
        'AAPL'
    """
    order: Order = None
    eventType: EventType = field(default=EventType.ORDER, init=False)

    @property
    def symbol(self) -> str:
        """Return the ticker symbol from the wrapped order. 종목 코드."""
        return self.order.symbol if self.order else ""

    @property
    def orderId(self) -> str:
        """Return the order ID from the wrapped order. 주문 ID."""
        return self.order.id if self.order else ""

    def __repr__(self) -> str:
        if self.order:
            return (
                f"OrderEvent({self.order.side.value.upper()} {self.order.quantity} "
                f"{self.order.symbol} @ {self.order.orderType.value})"
            )
        return f"OrderEvent(empty)"
