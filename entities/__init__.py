"""
Tradix Entities Package - Core data objects for the backtesting engine.
Tradix 엔티티 패키지 - 백테스팅 엔진의 핵심 데이터 객체.

This package provides the fundamental data structures used throughout the
Tradix backtesting framework, including market data bars, trading orders,
portfolio positions, and completed trade records.

Features:
    - Bar: OHLCV candlestick data with technical analysis properties
    - Order: Full-lifecycle order management with fill tracking
    - Position: Open position tracking with P&L computation
    - Trade: Completed round-trip trade records with performance metrics

Usage:
    >>> from tradix.entities import Bar, Order, OrderSide, OrderType, Position, Trade
"""

from tradix.entities.order import Order, OrderSide, OrderType, TimeInForce
from tradix.entities.position import Position
from tradix.entities.trade import Trade
from tradix.entities.bar import Bar

__all__ = [
    "Order",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Position",
    "Trade",
    "Bar",
]
