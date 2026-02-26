"""
Tradex Entities Package - Core data objects for the backtesting engine.
Tradex 엔티티 패키지 - 백테스팅 엔진의 핵심 데이터 객체.

This package provides the fundamental data structures used throughout the
Tradex backtesting framework, including market data bars, trading orders,
portfolio positions, and completed trade records.

Features:
    - Bar: OHLCV candlestick data with technical analysis properties
    - Order: Full-lifecycle order management with fill tracking
    - Position: Open position tracking with P&L computation
    - Trade: Completed round-trip trade records with performance metrics

Usage:
    >>> from tradex.entities import Bar, Order, OrderSide, OrderType, Position, Trade
"""

from tradex.entities.order import Order, OrderSide, OrderType, TimeInForce
from tradex.entities.position import Position
from tradex.entities.trade import Trade
from tradex.entities.bar import Bar

__all__ = [
    "Order",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Position",
    "Trade",
    "Bar",
]
