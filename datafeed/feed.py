"""
Tradix Data Feed Base Module.

Provides the abstract base class for all data feed implementations in
Tradix. Defines the common interface for loading, iterating, and
accessing OHLCV bar data from various data sources.

데이터 피드 추상 기반 모듈 - 모든 데이터 소스의 공통 인터페이스를
정의합니다.

Features:
    - Abstract load interface for subclass implementations
    - Iterator protocol for bar-by-bar backtesting
    - NumPy array caching for high-performance data access
    - Historical data slicing with configurable lookback
    - Progress tracking and bar count properties

Usage:
    >>> from tradix.datafeed.fdr import FinanceDataReaderFeed
    >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
    >>> for bar in feed:
    ...     print(bar.close)
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, List
import pandas as pd
import numpy as np

from tradix.entities.bar import Bar


class DataFeed(ABC):
    """
    Abstract base class for all data feed implementations.

    Provides the common interface and shared functionality for loading
    OHLCV data, iterating bar-by-bar, and accessing historical data.
    Supports FinanceDataReader, CSV, API, and other data sources
    through subclassing.

    데이터 피드 추상 클래스 - 모든 데이터 소스의 부모 클래스로,
    OHLCV 데이터 로딩, 바별 이터레이션, 과거 데이터 접근 기능을
    제공합니다.

    Attributes:
        symbol (str): Ticker symbol. 종목 코드.
        startDate (str): Start date in YYYY-MM-DD format. 시작일.
        endDate (str): End date in YYYY-MM-DD format. 종료일.

    Example:
        >>> feed = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')
        >>> for bar in feed:
        ...     print(bar.close)
        >>> df = feed.toDataFrame()
    """

    def __init__(
        self,
        symbol: str,
        startDate: str,
        endDate: str,
        **kwargs
    ):
        self.symbol = symbol
        self.startDate = startDate
        self.endDate = endDate

        self._data: Optional[pd.DataFrame] = None
        self._index: int = 0
        self._loaded: bool = False

        self._npOpen: Optional[np.ndarray] = None
        self._npHigh: Optional[np.ndarray] = None
        self._npLow: Optional[np.ndarray] = None
        self._npClose: Optional[np.ndarray] = None
        self._npVolume: Optional[np.ndarray] = None
        self._npDatetime: Optional[np.ndarray] = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load OHLCV data from the data source.

        데이터 소스에서 OHLCV 데이터를 로드합니다.

        Returns:
            pd.DataFrame: OHLCV DataFrame with DatetimeIndex and columns
                'open', 'high', 'low', 'close', 'volume'.
        """
        pass

    def reload(self) -> pd.DataFrame:
        """
        Reload data from source, bypassing any cache.

        캐시를 무시하고 데이터를 재로드합니다.

        Returns:
            pd.DataFrame: Freshly loaded OHLCV DataFrame.
        """
        self._loaded = False
        return self.load()

    def _buildNumpyArrays(self):
        """Build NumPy arrays from DataFrame for high-performance access. 성능 최적화를 위한 NumPy 배열 생성."""
        if self._data is not None:
            self._npOpen = self._data['open'].values
            self._npHigh = self._data['high'].values
            self._npLow = self._data['low'].values
            self._npClose = self._data['close'].values
            self._npVolume = self._data['volume'].values
            self._npDatetime = self._data.index.to_pydatetime()

    def toDataFrame(self) -> pd.DataFrame:
        """
        Return the loaded data as a pandas DataFrame copy.

        로드된 데이터를 DataFrame 복사본으로 반환합니다.

        Returns:
            pd.DataFrame: Copy of the OHLCV data, or empty DataFrame
                if not loaded.
        """
        if not self._loaded:
            self.load()
        return self._data.copy() if self._data is not None else pd.DataFrame()

    def getHistory(self, periods: Optional[int] = None, copy: bool = True) -> pd.DataFrame:
        """
        Return historical data up to the current iteration position.

        현재 이터레이션 위치까지의 과거 데이터를 반환합니다.

        Args:
            periods (int, optional): Number of bars to retrieve. If
                None, returns all history up to current position.
                가져올 기간 수.
            copy (bool): If True, returns a copy. Set to False for
                better performance when mutation is not needed.
                Default: True. 복사본 반환 여부.

        Returns:
            pd.DataFrame: Historical OHLCV data.
        """
        if self._data is None:
            return pd.DataFrame()

        if periods is None:
            result = self._data.iloc[:self._index]
        else:
            startIdx = max(0, self._index - periods)
            result = self._data.iloc[startIdx:self._index]

        return result.copy() if copy else result

    def getBar(self, index: int) -> Optional[Bar]:
        """
        Return a Bar object at a specific index.

        특정 인덱스의 바를 반환합니다.

        Args:
            index (int): Zero-based bar index. 바 인덱스.

        Returns:
            Bar or None: Bar at the specified index, or None if
                out of range.
        """
        if self._npClose is None or index < 0 or index >= len(self._npClose):
            return None

        return Bar(
            symbol=self.symbol,
            datetime=self._npDatetime[index],
            open=float(self._npOpen[index]),
            high=float(self._npHigh[index]),
            low=float(self._npLow[index]),
            close=float(self._npClose[index]),
            volume=float(self._npVolume[index]),
            amount=0.0,
        )

    def getCurrentBar(self) -> Optional[Bar]:
        """Return the most recently iterated bar. 가장 최근에 이터레이션된 바를 반환합니다."""
        if self._index > 0:
            return self.getBar(self._index - 1)
        return None

    @property
    def currentIndex(self) -> int:
        """Return the current iteration index. 현재 이터레이션 인덱스를 반환합니다."""
        return self._index

    @property
    def totalBars(self) -> int:
        """Return the total number of bars in the dataset. 전체 바 수를 반환합니다."""
        return len(self._data) if self._data is not None else 0

    @property
    def remainingBars(self) -> int:
        """Return the number of remaining bars to iterate. 남은 바 수를 반환합니다."""
        return self.totalBars - self._index

    @property
    def progress(self) -> float:
        """Return iteration progress as a fraction (0.0 to 1.0). 진행률을 반환합니다."""
        if self.totalBars == 0:
            return 0.0
        return self._index / self.totalBars

    @property
    def isLoaded(self) -> bool:
        """Return whether data has been loaded. 데이터 로드 여부를 반환합니다."""
        return self._loaded

    @property
    def isEmpty(self) -> bool:
        """Return whether the dataset is empty or unloaded. 데이터가 비어있는지 여부를 반환합니다."""
        return self._data is None or len(self._data) == 0

    @property
    def firstDate(self) -> Optional[pd.Timestamp]:
        """Return the first date in the dataset. 데이터의 첫 번째 날짜를 반환합니다."""
        if self._data is not None and len(self._data) > 0:
            return self._data.index[0]
        return None

    @property
    def lastDate(self) -> Optional[pd.Timestamp]:
        """Return the last date in the dataset. 데이터의 마지막 날짜를 반환합니다."""
        if self._data is not None and len(self._data) > 0:
            return self._data.index[-1]
        return None

    def reset(self):
        """Reset the iteration index to the beginning. 이터레이션 인덱스를 초기화합니다."""
        self._index = 0

    def __iter__(self) -> Iterator[Bar]:
        if not self._loaded:
            self.load()
        self._index = 0
        return self

    def __next__(self) -> Bar:
        if self._data is None or self._index >= len(self._data):
            raise StopIteration

        bar = self.getBar(self._index)
        self._index += 1
        return bar

    def __len__(self) -> int:
        return self.totalBars

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.symbol}, "
            f"{self.startDate} ~ {self.endDate}, "
            f"{self.totalBars} bars)"
        )
