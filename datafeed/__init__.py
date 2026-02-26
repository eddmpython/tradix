"""
Tradex Data Feed Package.

Provides data feed abstractions and implementations for loading OHLCV
market data from various sources into the Tradex backtesting engine.

데이터 피드 패키지 - 다양한 소스에서 OHLCV 시장 데이터를 로드하는
데이터 피드 추상화 및 구현을 제공합니다.

Features:
    - Abstract DataFeed base class with iterator protocol
    - FinanceDataReader integration for KRX, US, JP markets
    - Parquet-based price caching with expiration management
    - MultiDataFeed for synchronized multi-symbol backtesting

Usage:
    >>> from tradex.datafeed import FinanceDataReaderFeed, MultiDataFeed
    >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
    >>> for bar in feed:
    ...     print(bar.close)
"""

from tradex.datafeed.feed import DataFeed
from tradex.datafeed.fdr import FinanceDataReaderFeed
from tradex.datafeed.cache import PriceCache
from tradex.datafeed.multiDataFeed import MultiDataFeed

__all__ = [
    "DataFeed",
    "FinanceDataReaderFeed",
    "PriceCache",
    "MultiDataFeed",
]
