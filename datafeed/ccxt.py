"""
Tradix CCXT Crypto Feed Module.

Provides a data feed implementation backed by the CCXT library,
supporting 100+ cryptocurrency exchanges with OHLCV candlestick data.

CCXT 암호화폐 데이터 피드 모듈 - 100개 이상의 거래소에서
OHLCV 캔들스틱 데이터를 로드합니다.

Features:
    - 100+ exchange support (Binance, Bybit, Upbit, Coinbase, etc.)
    - Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d, 1w)
    - Automatic pagination for large date ranges
    - Parquet file caching with configurable expiration

Usage:
    >>> from tradix.datafeed.ccxt import CcxtFeed
    >>> feed = CcxtFeed('BTC/USDT', '2023-01-01', '2024-12-31', exchange='binance')
    >>> for bar in feed:
    ...     print(f"{bar.datetime}: {bar.close}")
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from tradix.datafeed.feed import DataFeed
from tradix.datafeed.cache import PriceCache

TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    'D': '1d',
    '1d': '1d',
    'W': '1w',
    '1w': '1w',
}

TIMEFRAME_MS = {
    '1m': 60_000,
    '5m': 300_000,
    '15m': 900_000,
    '30m': 1_800_000,
    '1h': 3_600_000,
    '4h': 14_400_000,
    '1d': 86_400_000,
    '1w': 604_800_000,
}


class CcxtFeed(DataFeed):
    """
    Data feed backed by the CCXT library for cryptocurrency markets.

    Loads OHLCV data from any CCXT-supported exchange with Parquet-based
    caching. Supports multiple timeframes from 1-minute to weekly bars.

    CCXT 기반 암호화폐 데이터 피드 - 100개 이상의 거래소에서
    OHLCV 데이터를 로드합니다.

    Attributes:
        exchange (str): Exchange name ('binance', 'bybit', 'upbit', etc.). 거래소.
        timeframe (str): Candle timeframe ('1m', '5m', '1h', '1d', etc.). 타임프레임.
        useCache (bool): Whether to use Parquet caching. 캐시 사용 여부.
        cacheMaxAge (int): Cache expiration in days. 캐시 유효 기간 (일).

    Example:
        >>> feed = CcxtFeed('BTC/USDT', '2023-01-01', '2024-12-31')
        >>> feed = CcxtFeed('ETH/USDT', '2023-01-01', '2024-12-31', exchange='bybit')
        >>> feed = CcxtFeed('BTC/USDT', '2024-01-01', '2024-12-31', timeframe='1h')
    """

    def __init__(
        self,
        symbol: str,
        startDate: str,
        endDate: str,
        exchange: str = 'binance',
        timeframe: str = 'D',
        useCache: bool = True,
        cacheMaxAge: int = 1,
        limit: int = 1000,
    ):
        """
        Initialize the CCXT crypto feed.

        CCXT 암호화폐 피드를 초기화합니다.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/BTC').
                거래 쌍.
            startDate: Start date in YYYY-MM-DD format. 시작일.
            endDate: End date in YYYY-MM-DD format. 종료일.
            exchange: CCXT exchange ID ('binance', 'bybit', 'upbit', 'coinbase').
                Default: 'binance'. 거래소 ID.
            timeframe: Candle timeframe ('1m', '5m', '15m', '30m', '1h',
                '4h', 'D'/'1d', 'W'/'1w'). Default: 'D'. 타임프레임.
            useCache: Enable Parquet caching. Default: True. 캐시 사용 여부.
            cacheMaxAge: Cache expiration in days. Default: 1. 캐시 유효 기간.
            limit: Max candles per API request. Default: 1000. API 요청당 최대 캔들 수.
        """
        super().__init__(symbol, startDate, endDate)
        self.exchange = exchange.lower()
        self.timeframe = timeframe
        self.useCache = useCache
        self.cacheMaxAge = cacheMaxAge
        self.limit = limit

        self._ccxtTf = TIMEFRAME_MAP.get(timeframe, '1d')
        cacheKey = f"CCXT_{self.exchange.upper()}"
        self._cache = PriceCache(market=cacheKey, maxAgeDays=cacheMaxAge)

    def load(self) -> pd.DataFrame:
        """
        Load data with cache-first strategy.

        캐시 우선 전략으로 데이터를 로드합니다.

        Returns:
            pd.DataFrame: OHLCV DataFrame.
        """
        if self._loaded and self._data is not None:
            return self._data

        cacheSymbol = f"{self.symbol.replace('/', '_')}_{self._ccxtTf}"

        if self.useCache:
            cached = self._cache.get(cacheSymbol, self.startDate, self.endDate)
            if cached is not None and self._isCacheComplete(cached):
                self._data = cached
                self._buildNumpyArrays()
                self._loaded = True
                return self._data

        df = self._loadFromApi()

        if df is not None and len(df) > 0:
            if self.useCache:
                self._cache.save(cacheSymbol, df, merge=True)

            self._data = df
            self._buildNumpyArrays()
            self._loaded = True
        else:
            self._data = pd.DataFrame()
            self._loaded = True

        return self._data

    def _isCacheComplete(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) == 0:
            return False

        cacheStart = df.index[0]
        cacheEnd = df.index[-1]

        requestStart = pd.Timestamp(self.startDate)
        requestEnd = pd.Timestamp(self.endDate)

        today = pd.Timestamp(datetime.now().date())
        if requestEnd > today:
            requestEnd = today

        return cacheStart <= requestStart and cacheEnd >= requestEnd - timedelta(days=5)

    def _loadFromApi(self) -> Optional[pd.DataFrame]:
        """
        Load data from a CCXT exchange with automatic pagination.

        CCXT 거래소에서 자동 페이지네이션으로 데이터를 로드합니다.

        Returns:
            pd.DataFrame or None: OHLCV DataFrame, or None on failure.
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt가 설치되어 있지 않습니다. "
                "pip install ccxt 로 설치해주세요."
            )

        exchangeClass = getattr(ccxt, self.exchange, None)
        if exchangeClass is None:
            raise ValueError(f"지원하지 않는 거래소: {self.exchange}")

        ex = exchangeClass({'enableRateLimit': True})

        since = int(pd.Timestamp(self.startDate).timestamp() * 1000)
        endMs = int(pd.Timestamp(self.endDate).timestamp() * 1000)
        tfMs = TIMEFRAME_MS.get(self._ccxtTf, 86_400_000)

        allCandles = []
        current = since

        while current < endMs:
            candles = ex.fetch_ohlcv(
                self.symbol,
                timeframe=self._ccxtTf,
                since=current,
                limit=self.limit,
            )

            if not candles:
                break

            allCandles.extend(candles)
            lastTs = candles[-1][0]

            if lastTs <= current:
                break

            current = lastTs + tfMs

        if not allCandles:
            return None

        df = pd.DataFrame(allCandles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        mask = (df.index >= pd.Timestamp(self.startDate)) & (df.index <= pd.Timestamp(self.endDate))
        df = df.loc[mask]

        return df if len(df) > 0 else None

    def __repr__(self) -> str:
        return (
            f"CcxtFeed({self.symbol}, "
            f"{self.exchange}, "
            f"{self.startDate} ~ {self.endDate}, "
            f"{self._ccxtTf}, "
            f"bars={self.totalBars})"
        )
