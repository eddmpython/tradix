"""
Tradix Events Package - Event-driven architecture for the backtesting engine.
Tradix 이벤트 패키지 - 백테스팅 엔진의 이벤트 기반 아키텍처.

This package implements the event system that drives the Tradix backtesting
engine. Events flow through the system in a well-defined pipeline:
MarketEvent -> SignalEvent -> OrderEvent -> FillEvent.

Features:
    - Event: Abstract base class with timestamp-based ordering
    - MarketEvent: New market data (OHLCV bar) arrival notifications
    - SignalEvent: Trading signal generation from strategy logic
    - OrderEvent: Order submission to the execution broker
    - FillEvent: Order execution confirmation with cost details

Usage:
    >>> from tradix.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
"""

from tradix.events.base import Event, EventType
from tradix.events.market import MarketEvent
from tradix.events.signal import SignalEvent, SignalType
from tradix.events.order import OrderEvent
from tradix.events.fill import FillEvent

__all__ = [
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "SignalType",
    "OrderEvent",
    "FillEvent",
]
