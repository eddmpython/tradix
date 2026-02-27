"""
Tradix Yahoo Finance Feed Module.

Provides a data feed implementation backed by the yfinance library,
supporting global stock markets, ETFs, indices, and cryptocurrency pairs.

Yahoo Finance 데이터 피드 모듈 - yfinance를 통해 글로벌 주식, ETF,
지수, 암호화폐 데이터를 로드합니다.

Features:
    - Global market coverage (US, EU, Asia, crypto)
    - Daily, weekly, monthly timeframes
    - Adjusted close price support
    - Parquet file caching with configurable expiration

Usage:
    >>> from tradix.datafeed.yahoo import YahooFinanceFeed
    >>> feed = YahooFinanceFeed('AAPL', '2020-01-01', '2024-12-31')
    >>> for bar in feed:
    ...     print(f"{bar.datetime}: {bar.close}")
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from tradix.datafeed.feed import DataFeed
from tradix.datafeed.cache import PriceCache


class YahooFinanceFeed(DataFeed):
    """
    Data feed backed by the yfinance library.

    Loads OHLCV data from Yahoo Finance with Parquet-based caching.
    Supports global equities, ETFs, indices, and crypto pairs.

    yfinance 기반 데이터 피드 - 글로벌 주식/ETF/지수/암호화폐 데이터를
    Parquet 캐시와 함께 로드합니다.

    Attributes:
        timeframe (str): Bar timeframe ('D', 'W', 'M'). 타임프레임.
        useCache (bool): Whether to use Parquet caching. 캐시 사용 여부.
        cacheMaxAge (int): Cache expiration in days. 캐시 유효 기간 (일).
        autoAdjust (bool): Use adjusted prices. 수정주가 사용 여부.

    Example:
        >>> feed = YahooFinanceFeed('AAPL', '2020-01-01', '2024-12-31')
        >>> feed = YahooFinanceFeed('BTC-USD', '2020-01-01', '2024-12-31')
        >>> feed = YahooFinanceFeed('SPY', '2020-01-01', '2024-12-31', timeframe='W')
    """

    def __init__(
        self,
        symbol: str,
        startDate: str,
        endDate: str,
        timeframe: str = 'D',
        useCache: bool = True,
        cacheMaxAge: int = 1,
        autoAdjust: bool = True,
    ):
        """
        Initialize the Yahoo Finance feed.

        Yahoo Finance 피드를 초기화합니다.

        Args:
            symbol: Yahoo Finance ticker (e.g., 'AAPL', 'BTC-USD', '^GSPC').
                Yahoo Finance 티커.
            startDate: Start date in YYYY-MM-DD format. 시작일.
            endDate: End date in YYYY-MM-DD format. 종료일.
            timeframe: Bar timeframe ('D'=daily, 'W'=weekly, 'M'=monthly).
                Default: 'D'. 타임프레임.
            useCache: Enable Parquet caching. Default: True. 캐시 사용 여부.
            cacheMaxAge: Cache expiration in days. Default: 1. 캐시 유효 기간.
            autoAdjust: Use adjusted close prices. Default: True. 수정주가 사용.
        """
        super().__init__(symbol, startDate, endDate)
        self.timeframe = timeframe.upper()
        self.useCache = useCache
        self.cacheMaxAge = cacheMaxAge
        self.autoAdjust = autoAdjust

        self._cache = PriceCache(market='YAHOO', maxAgeDays=cacheMaxAge)

    def load(self) -> pd.DataFrame:
        """
        Load data with cache-first strategy.

        캐시 우선 전략으로 데이터를 로드합니다.

        Returns:
            pd.DataFrame: OHLCV DataFrame.
        """
        if self._loaded and self._data is not None:
            return self._data

        if self.useCache:
            cached = self._cache.get(self.symbol, self.startDate, self.endDate)
            if cached is not None and self._isCacheComplete(cached):
                if self.timeframe != 'D':
                    cached = self._resampleTimeframe(cached)
                self._data = cached
                self._buildNumpyArrays()
                self._loaded = True
                return self._data

        df = self._loadFromApi()

        if df is not None and len(df) > 0:
            if self.useCache:
                self._cache.save(self.symbol, df, merge=True)

            if self.timeframe != 'D':
                df = self._resampleTimeframe(df)

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

    def _resampleTimeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.timeframe == 'D':
            return df

        ruleMap = {'W': 'W', 'M': 'ME'}
        rule = ruleMap.get(self.timeframe, 'D')

        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        return resampled

    def _loadFromApi(self) -> Optional[pd.DataFrame]:
        """
        Load data from the Yahoo Finance API via yfinance.

        yfinance API에서 데이터를 로드합니다.

        Returns:
            pd.DataFrame or None: OHLCV DataFrame, or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance가 설치되어 있지 않습니다. "
                "pip install yfinance 로 설치해주세요."
            )

        ticker = yf.Ticker(self.symbol)
        df = ticker.history(
            start=self.startDate,
            end=self.endDate,
            auto_adjust=self.autoAdjust,
        )

        if df is None or len(df) == 0:
            return None

        df.columns = df.columns.str.lower()

        result = pd.DataFrame(index=df.index)
        result['open'] = df['open'] if 'open' in df.columns else df.get('close', 0)
        result['high'] = df['high'] if 'high' in df.columns else df.get('close', 0)
        result['low'] = df['low'] if 'low' in df.columns else df.get('close', 0)
        result['close'] = df['close'] if 'close' in df.columns else 0
        result['volume'] = df['volume'] if 'volume' in df.columns else 0

        result = result.dropna()
        result.index = pd.to_datetime(result.index)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        return result

    def __repr__(self) -> str:
        tfName = {'D': '일봉', 'W': '주봉', 'M': '월봉'}.get(self.timeframe, self.timeframe)
        return (
            f"YahooFinanceFeed({self.symbol}, "
            f"{self.startDate} ~ {self.endDate}, "
            f"{tfName}, "
            f"bars={self.totalBars})"
        )
