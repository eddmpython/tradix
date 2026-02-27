"""
Tradix FinanceDataReader Feed Module.

Provides a data feed implementation backed by the FinanceDataReader
library, supporting Korean (KRX), US, and Japanese stock markets with
Parquet-based caching and timeframe resampling.

FinanceDataReader 데이터 피드 모듈 - 한국, 미국, 일본 주식 데이터를
FinanceDataReader에서 로드하며, Parquet 캐시와 타임프레임 리샘플링을
지원합니다.

Features:
    - Automatic data loading from FinanceDataReader API
    - Parquet file caching with configurable expiration
    - Timeframe resampling (daily to weekly/monthly)
    - Multi-market support (KRX, US, JP)
    - Cache management (update, info, clear)

Usage:
    >>> from tradix.datafeed.fdr import FinanceDataReaderFeed
    >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
    >>> for bar in feed:
    ...     print(f"{bar.datetime}: {bar.close}")
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from tradix.datafeed.feed import DataFeed
from tradix.datafeed.cache import PriceCache


class FinanceDataReaderFeed(DataFeed):
    """
    Data feed backed by the FinanceDataReader library.

    Loads OHLCV data from FinanceDataReader with Parquet-based caching
    to minimize API calls. Supports Korean (KRX), US, and Japanese
    stock markets, with optional timeframe resampling from daily to
    weekly or monthly bars.

    FinanceDataReader 데이터 소스 - 한국/미국/일본 주식 데이터를
    Parquet 캐시와 함께 로드합니다.

    Attributes:
        market (str): Market code ('KRX', 'US', 'JP'). 시장 코드.
        timeframe (str): Bar timeframe ('D', 'W', 'M'). 타임프레임.
        useCache (bool): Whether to use Parquet caching. 캐시 사용 여부.
        cacheMaxAge (int): Cache expiration in days. 캐시 유효 기간 (일).

    Example:
        >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
        >>> feed = FinanceDataReaderFeed('AAPL', '2020-01-01', '2024-12-31', market='US')
        >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31', timeframe='W')
        >>> for bar in feed:
        ...     print(f"{bar.datetime}: {bar.close}")
    """

    def __init__(
        self,
        symbol: str,
        startDate: str,
        endDate: str,
        market: str = 'KRX',
        timeframe: str = 'D',
        useCache: bool = True,
        cacheMaxAge: int = 1,
    ):
        """
        Initialize the FinanceDataReader feed.

        FinanceDataReader 피드를 초기화합니다.

        Args:
            symbol (str): Ticker symbol (e.g., '005930', 'AAPL', '7203.T').
                종목 코드.
            startDate (str): Start date in YYYY-MM-DD format. 시작일.
            endDate (str): End date in YYYY-MM-DD format. 종료일.
            market (str): Market code ('KRX', 'US', 'JP').
                Default: 'KRX'. 시장 코드.
            timeframe (str): Bar timeframe ('D'=daily, 'W'=weekly,
                'M'=monthly). Default: 'D'. 타임프레임.
            useCache (bool): Enable Parquet caching. Default: True.
                캐시 사용 여부.
            cacheMaxAge (int): Cache expiration in days. Default: 1.
                캐시 최대 유효 기간.
        """
        super().__init__(symbol, startDate, endDate)
        self.market = market.upper()
        self.timeframe = timeframe.upper()
        self.useCache = useCache
        self.cacheMaxAge = cacheMaxAge

        self._cache = PriceCache(market=self.market, maxAgeDays=cacheMaxAge)

    def load(self) -> pd.DataFrame:
        """
        Load data with cache-first strategy.

        캐시 우선 전략으로 데이터를 로드합니다. 캐시에 완전한 데이터가
        있으면 캐시를 사용하고, 없으면 API에서 로드합니다.

        Returns:
            pd.DataFrame: OHLCV DataFrame.
        """
        if self._loaded and self._data is not None:
            return self._data

        if self.useCache:
            cached = self._loadFromCache()
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
                self._saveToCache(df)

            if self.timeframe != 'D':
                df = self._resampleTimeframe(df)

            self._data = df
            self._buildNumpyArrays()
            self._loaded = True
        else:
            self._data = pd.DataFrame()
            self._loaded = True

        return self._data

    def _loadFromCache(self) -> Optional[pd.DataFrame]:
        """Load data from the Parquet cache. 캐시에서 데이터를 로드합니다."""
        return self._cache.get(self.symbol, self.startDate, self.endDate)

    def _saveToCache(self, df: pd.DataFrame):
        """Save data to the Parquet cache. 캐시에 데이터를 저장합니다."""
        self._cache.save(self.symbol, df, merge=True)

    def _resampleTimeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily bars to weekly or monthly timeframe.

        일봉 데이터를 주봉 또는 월봉으로 리샘플링합니다.

        Args:
            df (pd.DataFrame): Daily OHLCV DataFrame. 일봉 DataFrame.

        Returns:
            pd.DataFrame: Resampled OHLCV DataFrame.
        """
        if self.timeframe == 'D':
            return df

        ruleMap = {
            'W': 'W',
            'M': 'ME',
        }

        rule = ruleMap.get(self.timeframe, 'D')

        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def _isCacheComplete(self, df: pd.DataFrame) -> bool:
        """Check if cached data covers the full requested date range. 캐시가 요청 기간을 완전히 포함하는지 확인합니다."""
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
        Load data from the FinanceDataReader API.

        FinanceDataReader API에서 데이터를 로드합니다.

        Returns:
            pd.DataFrame or None: OHLCV DataFrame, or None on failure.

        Raises:
            ImportError: If FinanceDataReader is not installed.
        """
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader(self.symbol, self.startDate, self.endDate)

            if df is None or len(df) == 0:
                return None

            df.columns = df.columns.str.lower()

            columnMap = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'adj close': 'adjClose',
                'change': 'change',
            }

            df = df.rename(columns={k: v for k, v in columnMap.items() if k in df.columns})

            requiredCols = ['open', 'high', 'low', 'close', 'volume']
            for col in requiredCols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0
                    else:
                        df[col] = df['close'] if 'close' in df.columns else 0

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.dropna()
            df.index = pd.to_datetime(df.index)

            return df

        except ImportError:
            raise ImportError(
                "FinanceDataReader가 설치되어 있지 않습니다. "
                "pip install finance-datareader 로 설치해주세요."
            )
        except Exception as e:
            print(f"[FinanceDataReaderFeed] 데이터 로드 실패: {self.symbol} - {e}")
            return None

    def updateCache(self) -> bool:
        """
        Update the cache with the latest available data.

        캐시에 최신 데이터를 추가합니다.

        Returns:
            bool: True if update succeeded or no update needed.
        """
        lastDate = self._cache.getLastDate(self.symbol)

        if lastDate is None:
            startDate = self.startDate
        else:
            startDate = (lastDate + timedelta(days=1)).strftime('%Y-%m-%d')

        today = datetime.now().strftime('%Y-%m-%d')

        if startDate > today:
            return True

        try:
            import FinanceDataReader as fdr
            df = fdr.DataReader(self.symbol, startDate, today)

            if df is not None and len(df) > 0:
                df.columns = df.columns.str.lower()
                df = df.rename(columns={
                    'open': 'open', 'high': 'high', 'low': 'low',
                    'close': 'close', 'volume': 'volume'
                })
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.index = pd.to_datetime(df.index)
                self._cache.save(self.symbol, df, merge=True)

            return True

        except Exception as e:
            print(f"[FinanceDataReaderFeed] 캐시 업데이트 실패: {self.symbol} - {e}")
            return False

    def getCacheInfo(self) -> dict:
        """
        Return cache metadata for this symbol.

        이 종목의 캐시 메타데이터를 반환합니다.

        Returns:
            dict: Cache information including existence, size, dates.
        """
        return self._cache.getCacheInfo(self.symbol)

    def clearCache(self):
        """Delete the cached data for this symbol. 이 종목의 캐시를 삭제합니다."""
        self._cache.delete(self.symbol)

    def __repr__(self) -> str:
        tfName = {'D': '일봉', 'W': '주봉', 'M': '월봉'}.get(self.timeframe, self.timeframe)
        return (
            f"FinanceDataReaderFeed({self.symbol}, "
            f"{self.startDate} ~ {self.endDate}, "
            f"{tfName}, "
            f"bars={self.totalBars})"
        )
