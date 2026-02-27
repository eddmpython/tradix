"""
Tradix Data Feed Package.

Provides data feed abstractions and implementations for loading OHLCV
market data from various sources into the Tradix backtesting engine.

데이터 피드 패키지 - 다양한 소스에서 OHLCV 시장 데이터를 로드하는
데이터 피드 추상화 및 구현을 제공합니다.

Features:
    - Abstract DataFeed base class with iterator protocol
    - FinanceDataReader integration for KRX, US, JP markets
    - Yahoo Finance integration for global markets, ETFs, crypto
    - CCXT integration for 100+ crypto exchanges
    - Parquet-based price caching with expiration management
    - MultiDataFeed for synchronized multi-symbol backtesting

Usage:
    >>> from tradix.datafeed import FinanceDataReaderFeed, YahooFinanceFeed, CcxtFeed
    >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
    >>> feed = YahooFinanceFeed('AAPL', '2020-01-01', '2024-12-31')
    >>> feed = CcxtFeed('BTC/USDT', '2023-01-01', '2024-12-31', exchange='binance')
"""

from tradix.datafeed.feed import DataFeed
from tradix.datafeed.fdr import FinanceDataReaderFeed
from tradix.datafeed.yahoo import YahooFinanceFeed
from tradix.datafeed.ccxt import CcxtFeed
from tradix.datafeed.cache import PriceCache
from tradix.datafeed.multiDataFeed import MultiDataFeed

__all__ = [
    "DataFeed",
    "FinanceDataReaderFeed",
    "YahooFinanceFeed",
    "CcxtFeed",
    "PriceCache",
    "MultiDataFeed",
]
