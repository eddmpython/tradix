"""
Tradix Advanced Order Types Module.

Provides OCO (One-Cancels-Other), Bracket, and Trailing Stop order
management for the backtesting engine.

고급 주문 유형 모듈 - OCO, 브라켓, 트레일링 스탑 주문 관리를 제공합니다.

Features:
    - OCO: Two orders linked; when one fills, the other is cancelled
    - Bracket: Entry order with automatic stop-loss and take-profit
    - Trailing Stop: Dynamic stop price that follows favorable movement

Usage:
    >>> from tradix.broker.advancedOrders import OcoOrder, BracketOrder, TrailingStopOrder
    >>> oco = OcoOrder(takeProfitOrder, stopLossOrder)
    >>> bracket = BracketOrder.create('005930', quantity=10,
    ...     entryPrice=70000, stopLoss=68000, takeProfit=74000)
    >>> trailing = TrailingStopOrder('005930', quantity=10, trailingPercent=3.0)
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import uuid

from tradix.entities.order import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from tradix.entities.bar import Bar


@dataclass
class OcoOrder:
    """
    One-Cancels-Other order pair.

    Links two orders together so that when one is filled, the other is
    automatically cancelled.

    OCO 주문 - 두 주문을 연결하여 하나가 체결되면 다른 하나가 자동 취소됩니다.

    Attributes:
        orderA (Order): First leg of the OCO pair. 첫 번째 주문.
        orderB (Order): Second leg of the OCO pair. 두 번째 주문.
        id (str): Unique OCO group identifier. OCO 그룹 ID.

    Example:
        >>> tp = Order(symbol='005930', side=OrderSide.SELL,
        ...     orderType=OrderType.LIMIT, quantity=10, price=74000)
        >>> sl = Order(symbol='005930', side=OrderSide.SELL,
        ...     orderType=OrderType.STOP, quantity=10, stopPrice=68000)
        >>> oco = OcoOrder(tp, sl)
        >>> oco.update(bar)
    """
    orderA: Order
    orderB: Order
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        self.orderA.parentId = self.id
        self.orderB.parentId = self.id

    @property
    def isComplete(self) -> bool:
        return (
            self.orderA.isFilled or self.orderB.isFilled
            or (self.orderA.status == OrderStatus.CANCELLED and self.orderB.status == OrderStatus.CANCELLED)
        )

    @property
    def filledOrder(self) -> Optional[Order]:
        if self.orderA.isFilled:
            return self.orderA
        if self.orderB.isFilled:
            return self.orderB
        return None

    def update(self, bar: Bar):
        """
        Check if either order should trigger and handle OCO cancellation.

        바 데이터를 기반으로 주문 트리거 여부를 확인하고 OCO 취소를 처리합니다.

        Args:
            bar: Current bar data. 현재 바 데이터.

        Returns:
            Order or None: The triggered order, or None if neither triggered.
        """
        if self.isComplete:
            return None

        triggered = self._checkTrigger(self.orderA, bar)
        if triggered:
            self.orderB.cancel()
            return self.orderA

        triggered = self._checkTrigger(self.orderB, bar)
        if triggered:
            self.orderA.cancel()
            return self.orderB

        return None

    def _checkTrigger(self, order: Order, bar: Bar) -> bool:
        if not order.isPending:
            return False

        if order.orderType == OrderType.LIMIT:
            if order.side == OrderSide.SELL and bar.high >= order.price:
                return True
            if order.side == OrderSide.BUY and bar.low <= order.price:
                return True

        elif order.orderType == OrderType.STOP:
            if order.side == OrderSide.SELL and bar.low <= order.stopPrice:
                return True
            if order.side == OrderSide.BUY and bar.high >= order.stopPrice:
                return True

        return False

    def cancel(self):
        self.orderA.cancel()
        self.orderB.cancel()


@dataclass
class BracketOrder:
    """
    Bracket order: entry + stop-loss + take-profit.

    Automatically creates a market/limit entry order with attached
    stop-loss and take-profit exit orders as an OCO pair.

    브라켓 주문 - 진입 주문과 함께 손절/익절 주문을 OCO로 묶어 관리합니다.

    Attributes:
        entry (Order): Entry order. 진입 주문.
        stopLoss (Order): Stop-loss exit order. 손절 주문.
        takeProfit (Order): Take-profit exit order. 익절 주문.
        exitOco (OcoOrder): OCO pair linking stop-loss and take-profit.
            손절/익절을 묶는 OCO 쌍.

    Example:
        >>> bracket = BracketOrder.create('005930', quantity=10,
        ...     entryPrice=70000, stopLoss=68000, takeProfit=74000)
        >>> bracket.update(bar)
    """
    entry: Order
    stopLoss: Order
    takeProfit: Order
    exitOco: OcoOrder = field(init=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        self.exitOco = OcoOrder(self.takeProfit, self.stopLoss)
        self.entry.parentId = self.id

    @classmethod
    def create(
        cls,
        symbol: str,
        quantity: float,
        entryPrice: float,
        stopLoss: float,
        takeProfit: float,
        side: OrderSide = OrderSide.BUY,
        entryType: OrderType = OrderType.LIMIT,
    ) -> 'BracketOrder':
        """
        Factory method to create a complete bracket order.

        브라켓 주문을 생성하는 팩토리 메서드.

        Args:
            symbol: Ticker symbol. 종목 코드.
            quantity: Order quantity. 주문 수량.
            entryPrice: Entry limit price. 진입 가격.
            stopLoss: Stop-loss price. 손절 가격.
            takeProfit: Take-profit price. 익절 가격.
            side: Entry direction (BUY or SELL). Default: BUY. 진입 방향.
            entryType: Entry order type. Default: LIMIT. 진입 주문 유형.

        Returns:
            BracketOrder: Fully configured bracket order.
        """
        exitSide = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        entryOrder = Order(
            symbol=symbol,
            side=side,
            orderType=entryType,
            quantity=quantity,
            price=entryPrice,
        )

        slOrder = Order(
            symbol=symbol,
            side=exitSide,
            orderType=OrderType.STOP,
            quantity=quantity,
            stopPrice=stopLoss,
        )

        tpOrder = Order(
            symbol=symbol,
            side=exitSide,
            orderType=OrderType.LIMIT,
            quantity=quantity,
            price=takeProfit,
        )

        return cls(entry=entryOrder, stopLoss=slOrder, takeProfit=tpOrder)

    @property
    def isEntryFilled(self) -> bool:
        return self.entry.isFilled

    @property
    def isComplete(self) -> bool:
        return self.isEntryFilled and self.exitOco.isComplete

    def update(self, bar: Bar) -> Optional[Order]:
        """
        Update bracket state based on new bar data.

        새로운 바 데이터로 브라켓 상태를 업데이트합니다.

        Args:
            bar: Current bar data. 현재 바 데이터.

        Returns:
            Order or None: Triggered order (entry or exit), or None.
        """
        if not self.isEntryFilled:
            if self.exitOco._checkTrigger(self.entry, bar):
                return self.entry
            return None

        return self.exitOco.update(bar)

    def cancel(self):
        self.entry.cancel()
        self.exitOco.cancel()


@dataclass
class TrailingStopOrder:
    """
    Trailing stop order with dynamic stop price.

    Maintains a stop price that follows favorable price movement by a
    fixed percentage, locking in gains as the market moves in favor.

    트레일링 스탑 주문 - 가격이 유리한 방향으로 움직이면 스탑 가격이
    일정 비율로 따라가며, 수익을 보호합니다.

    Attributes:
        symbol (str): Ticker symbol. 종목 코드.
        quantity (float): Order quantity. 주문 수량.
        side (OrderSide): Exit side (SELL for long, BUY for short). 주문 방향.
        trailingPercent (float): Trailing distance in percent. 추적 비율 (%).
        activationPrice (Optional[float]): Price at which trailing activates.
            추적 활성화 가격 (None이면 즉시 활성).
        currentStop (float): Current computed stop price. 현재 스탑 가격.
        highWaterMark (float): Peak favorable price observed. 관측된 최고/최저가.
        triggered (bool): Whether the stop has been triggered. 트리거 여부.

    Example:
        >>> ts = TrailingStopOrder('005930', quantity=10, trailingPercent=3.0)
        >>> ts.activate(70000)
        >>> ts.update(bar)
    """
    symbol: str
    quantity: float
    side: OrderSide = OrderSide.SELL
    trailingPercent: float = 3.0
    activationPrice: Optional[float] = None
    currentStop: float = 0.0
    highWaterMark: float = 0.0
    triggered: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def activate(self, price: float):
        """
        Activate the trailing stop at a given price.

        주어진 가격에서 트레일링 스탑을 활성화합니다.

        Args:
            price: Initial reference price. 초기 기준 가격.
        """
        self.highWaterMark = price
        if self.side == OrderSide.SELL:
            self.currentStop = price * (1 - self.trailingPercent / 100)
        else:
            self.currentStop = price * (1 + self.trailingPercent / 100)

    def update(self, bar: Bar) -> bool:
        """
        Update the trailing stop with new bar data.

        새로운 바 데이터로 트레일링 스탑을 업데이트합니다.

        Args:
            bar: Current bar data. 현재 바 데이터.

        Returns:
            bool: True if the stop was triggered on this bar. 트리거 여부.
        """
        if self.triggered:
            return False

        if self.activationPrice is not None:
            if self.side == OrderSide.SELL and bar.high < self.activationPrice:
                return False
            if self.side == OrderSide.BUY and bar.low > self.activationPrice:
                return False
            if self.highWaterMark == 0:
                self.activate(bar.close)

        if self.highWaterMark == 0:
            return False

        if self.side == OrderSide.SELL:
            if bar.high > self.highWaterMark:
                self.highWaterMark = bar.high
                self.currentStop = self.highWaterMark * (1 - self.trailingPercent / 100)

            if bar.low <= self.currentStop:
                self.triggered = True
                return True
        else:
            if bar.low < self.highWaterMark:
                self.highWaterMark = bar.low
                self.currentStop = self.highWaterMark * (1 + self.trailingPercent / 100)

            if bar.high >= self.currentStop:
                self.triggered = True
                return True

        return False

    def toOrder(self) -> Order:
        """
        Convert to a standard Order for execution.

        실행을 위해 표준 Order로 변환합니다.

        Returns:
            Order: Market sell/buy order at current stop price.
        """
        return Order(
            symbol=self.symbol,
            side=self.side,
            orderType=OrderType.TRAILING_STOP,
            quantity=self.quantity,
            stopPrice=self.currentStop,
            trailingPercent=self.trailingPercent,
        )
