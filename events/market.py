"""
Tradex Market Event - Market data arrival notification.
Tradex 시장 이벤트 - 시장 데이터 도착 알림.

This module defines the MarketEvent class, which is emitted when new OHLCV
bar data arrives in the Tradex backtesting engine. MarketEvents are the first
stage in the event pipeline, triggering strategy evaluation and subsequent
signal generation.

Features:
    - Wraps a Bar entity with event metadata (timestamp, ID)
    - Convenience properties for direct price/volume access
    - Seamless integration with the event-driven pipeline

Usage:
    >>> from tradex.events.market import MarketEvent
    >>> from tradex.entities.bar import Bar
    >>> from datetime import datetime
    >>> bar = Bar("AAPL", datetime(2025, 1, 15), 150, 155, 148, 153, 1000000)
    >>> event = MarketEvent(timestamp=bar.datetime, symbol="AAPL", bar=bar)
    >>> event.close
    153
"""

from dataclasses import dataclass, field
from datetime import datetime

from tradex.events.base import Event, EventType
from tradex.entities.bar import Bar


@dataclass
class MarketEvent(Event):
    """
    Market data event emitted when a new OHLCV bar arrives.
    새로운 OHLCV 바 데이터가 도착했을 때 발생하는 시장 데이터 이벤트.

    The first event in the Tradex processing pipeline. Contains a Bar entity
    with price and volume data for a specific symbol. Strategies consume
    MarketEvents to evaluate trading conditions and generate signals.

    Tradex 처리 파이프라인의 첫 번째 이벤트입니다. 특정 종목의 가격 및
    거래량 데이터를 담은 Bar 엔티티를 포함합니다.

    Attributes:
        symbol (str): Ticker symbol identifier. 종목 코드.
        bar (Bar): OHLCV bar data entity. OHLCV 바 데이터 엔티티.

    Example:
        >>> event = MarketEvent(timestamp=bar.datetime, symbol="AAPL", bar=bar)
        >>> event.close
        153.0
    """
    symbol: str = ""
    bar: Bar = None
    eventType: EventType = field(default=EventType.MARKET, init=False)

    @property
    def close(self) -> float:
        """Return the closing price from the bar. 종가."""
        return self.bar.close if self.bar else 0.0

    @property
    def open(self) -> float:
        """Return the opening price from the bar. 시가."""
        return self.bar.open if self.bar else 0.0

    @property
    def high(self) -> float:
        """Return the highest price from the bar. 고가."""
        return self.bar.high if self.bar else 0.0

    @property
    def low(self) -> float:
        """Return the lowest price from the bar. 저가."""
        return self.bar.low if self.bar else 0.0

    @property
    def volume(self) -> float:
        """Return the trading volume from the bar. 거래량."""
        return self.bar.volume if self.bar else 0.0

    def __repr__(self) -> str:
        if self.bar:
            return (
                f"MarketEvent({self.symbol} {self.timestamp.strftime('%Y-%m-%d')} "
                f"C:{self.bar.close:,.0f})"
            )
        return f"MarketEvent({self.symbol} {self.timestamp})"
