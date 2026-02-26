"""
Tradex Bar Entity - OHLCV candlestick data representation.
Tradex 바(Bar) 엔티티 - OHLCV 캔들스틱 데이터 표현.

This module defines the Bar dataclass, which encapsulates a single OHLCV
(Open, High, Low, Close, Volume) candlestick bar used throughout the Tradex
backtesting engine for market data representation and price analysis.

Features:
    - Immutable OHLCV data storage via Python dataclass
    - Derived price calculations (typical, median, weighted)
    - Candlestick pattern analysis (bullish, bearish, doji)
    - Shadow and body size computations
    - Dictionary serialization and deserialization

Usage:
    >>> from tradex.entities.bar import Bar
    >>> from datetime import datetime
    >>> bar = Bar(
    ...     symbol="AAPL",
    ...     datetime=datetime(2025, 1, 15, 9, 30),
    ...     open=150.0, high=155.0, low=148.0, close=153.0,
    ...     volume=1000000.0
    ... )
    >>> bar.typical
    152.0
    >>> bar.isBullish
    True
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass
class Bar:
    """
    OHLCV candlestick bar data container.
    OHLCV 캔들스틱 바 데이터 컨테이너.

    Represents a single price bar with open, high, low, close prices and volume.
    Provides computed properties for common technical analysis calculations
    including typical price, candlestick body/shadow analysis, and directional indicators.

    하나의 가격 바를 시가, 고가, 저가, 종가 및 거래량과 함께 표현합니다.
    전형적 가격, 캔들스틱 몸통/꼬리 분석, 방향 지표 등 일반적인 기술적 분석
    계산을 위한 파생 속성을 제공합니다.

    Attributes:
        symbol (str): Ticker symbol identifier. 종목 코드.
        datetime (datetime): Timestamp of the bar. 바의 타임스탬프.
        open (float): Opening price. 시가.
        high (float): Highest price. 고가.
        low (float): Lowest price. 저가.
        close (float): Closing price. 종가.
        volume (float): Trading volume. 거래량.
        amount (float): Trading value/turnover (optional, defaults to 0.0).
            거래대금 (선택사항, 기본값 0.0).

    Example:
        >>> bar = Bar(
        ...     symbol="005930",
        ...     datetime=datetime(2025, 1, 15),
        ...     open=70000, high=72000, low=69500, close=71500,
        ...     volume=5000000
        ... )
        >>> bar.range
        2500
        >>> bar.isBullish
        True
    """
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0

    @property
    def ohlc(self) -> tuple:
        """Return OHLC prices as a tuple. OHLC 가격을 튜플로 반환."""
        return (self.open, self.high, self.low, self.close)

    @property
    def ohlcv(self) -> tuple:
        """Return OHLCV prices and volume as a tuple. OHLCV 가격과 거래량을 튜플로 반환."""
        return (self.open, self.high, self.low, self.close, self.volume)

    @property
    def typical(self) -> float:
        """Compute typical price: (high + low + close) / 3. 전형적 가격 (TP)."""
        return (self.high + self.low + self.close) / 3

    @property
    def median(self) -> float:
        """Compute median price: (high + low) / 2. 중간 가격."""
        return (self.high + self.low) / 2

    @property
    def weighted(self) -> float:
        """Compute weighted close price: (high + low + 2*close) / 4. 가중 종가."""
        return (self.high + self.low + 2 * self.close) / 4

    @property
    def range(self) -> float:
        """Compute price range: high - low. 가격 범위."""
        return self.high - self.low

    @property
    def bodySize(self) -> float:
        """Compute candlestick body size: |close - open|. 캔들 몸통 크기."""
        return abs(self.close - self.open)

    @property
    def upperShadow(self) -> float:
        """Compute upper shadow length. 윗꼬리 길이."""
        return self.high - max(self.open, self.close)

    @property
    def lowerShadow(self) -> float:
        """Compute lower shadow length. 아래꼬리 길이."""
        return min(self.open, self.close) - self.low

    @property
    def isBullish(self) -> bool:
        """Check if the bar is bullish (close > open). 양봉 여부."""
        return self.close > self.open

    @property
    def isBearish(self) -> bool:
        """Check if the bar is bearish (close < open). 음봉 여부."""
        return self.close < self.open

    @property
    def isDoji(self) -> bool:
        """Check if the bar is a doji candle (body < 10% of range). 도지 캔들 여부."""
        if self.range == 0:
            return True
        return self.bodySize / self.range < 0.1

    @property
    def changePercent(self) -> float:
        """Compute price change percentage: (close - open) / open * 100. 변동률(%)."""
        if self.open == 0:
            return 0.0
        return ((self.close - self.open) / self.open) * 100

    @property
    def date(self) -> datetime:
        """Return the date component only (without time). 날짜만 반환 (시간 제외)."""
        return self.datetime.date()

    def toDict(self) -> Dict[str, Any]:
        """Convert the bar to a dictionary representation. 딕셔너리로 변환.

        Returns:
            Dict[str, Any]: Dictionary containing all bar fields.
        """
        return {
            "symbol": self.symbol,
            "datetime": self.datetime,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
        }

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "Bar":
        """Create a Bar instance from a dictionary. 딕셔너리에서 Bar 인스턴스 생성.

        Args:
            data: Dictionary containing bar field values.

        Returns:
            Bar: A new Bar instance populated from the dictionary.
        """
        return cls(
            symbol=data.get("symbol", ""),
            datetime=data.get("datetime"),
            open=data.get("open", 0),
            high=data.get("high", 0),
            low=data.get("low", 0),
            close=data.get("close", 0),
            volume=data.get("volume", 0),
            amount=data.get("amount", 0),
        )

    def __repr__(self) -> str:
        direction = "▲" if self.isBullish else "▼" if self.isBearish else "─"
        return (
            f"Bar({self.symbol} {self.datetime.strftime('%Y-%m-%d')} "
            f"O:{self.open:,.0f} H:{self.high:,.0f} L:{self.low:,.0f} C:{self.close:,.0f} "
            f"V:{self.volume:,.0f} {direction}{self.changePercent:+.2f}%)"
        )
