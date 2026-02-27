"""
Tradix Multi-Data Feed Module.

Manages synchronized multi-symbol data feeds for portfolio-level
backtesting. Aligns multiple data sources by date and provides
unified iteration across all symbols simultaneously.

복수 종목 데이터 피드 모듈 - 포트폴리오 수준의 백테스트를 위해
여러 종목의 데이터를 날짜 기준으로 동기화하여 관리합니다.

Features:
    - Date alignment across multiple symbols (inner/outer/left join)
    - Missing data handling (forward fill, drop, zero fill)
    - Synchronized bar-by-bar iteration across all symbols
    - NumPy array caching for high-performance access
    - Per-symbol historical data retrieval

Usage:
    >>> from tradix.datafeed import FinanceDataReaderFeed, MultiDataFeed
    >>> feeds = {
    ...     '005930': FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
    ...     '000660': FinanceDataReaderFeed('000660', '2020-01-01', '2024-12-31'),
    ... }
    >>> multi = MultiDataFeed(feeds)
    >>> multi.load()
    >>> for bars in multi:
    ...     for symbol, bar in bars.items():
    ...         print(f"{bar.datetime}: {symbol} = {bar.close}")
"""

from typing import Dict, List, Iterator, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from tradix.datafeed.feed import DataFeed
from tradix.entities.bar import Bar


class MultiDataFeed:
    """
    Multi-symbol data feed manager with date synchronization.

    Loads, aligns, and iterates multiple data feeds simultaneously,
    providing synchronized bar data across all symbols for each date.
    Supports configurable alignment methods and missing data handling.

    복수 종목 데이터 관리자 - 여러 데이터 피드를 날짜 기준으로
    동기화하여 동시에 이터레이션합니다.

    Attributes:
        feeds (dict[str, DataFeed]): Symbol-to-feed mapping. 종목별 피드.
        alignMethod (str): Date alignment method ('inner', 'outer', 'left').
            날짜 정렬 방법.
        fillMethod (str): Missing data fill method ('ffill', 'drop', 'zero').
            결측 데이터 처리 방법.

    Example:
        >>> feeds = {
        ...     '005930': FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
        ...     '000660': FinanceDataReaderFeed('000660', '2020-01-01', '2024-12-31'),
        ... }
        >>> multi = MultiDataFeed(feeds)
        >>> multi.load()
        >>> for bars in multi:
        ...     for symbol, bar in bars.items():
        ...         print(f"{symbol}: {bar.close}")
    """

    def __init__(
        self,
        feeds: Dict[str, DataFeed],
        alignMethod: str = 'inner',
        fillMethod: str = 'ffill',
    ):
        """
        Initialize the multi-data feed.

        복수 종목 데이터 피드를 초기화합니다.

        Args:
            feeds (dict[str, DataFeed]): Mapping of symbol to DataFeed
                instance. 종목별 데이터 피드 딕셔너리.
            alignMethod (str): Date alignment strategy. Default: 'inner'.
                - 'inner': Only dates where all symbols have data.
                - 'outer': All dates where any symbol has data.
                - 'left': Dates from the first symbol only.
                날짜 정렬 방법.
            fillMethod (str): Missing data handling. Default: 'ffill'.
                - 'ffill': Forward fill from previous values.
                - 'drop': Drop rows with any missing data.
                - 'zero': Fill missing values with 0.
                결측 데이터 처리 방법.
        """
        self.feeds = feeds
        self.alignMethod = alignMethod
        self.fillMethod = fillMethod

        self._alignedData: Dict[str, pd.DataFrame] = {}
        self._dates: pd.DatetimeIndex = None
        self._index: int = 0
        self._loaded: bool = False

        self._npData: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load all feeds and synchronize dates.

        모든 피드를 로드하고 날짜를 동기화합니다.

        Returns:
            dict[str, pd.DataFrame]: Aligned data for each symbol.
        """
        if self._loaded:
            return self._alignedData

        for symbol, feed in self.feeds.items():
            feed.load()

        self._alignDates()

        self._buildNumpyArrays()

        self._loaded = True

        return self._alignedData

    def _alignDates(self):
        """Align dates across all feeds using the configured method. 설정된 방법으로 모든 피드의 날짜를 동기화합니다."""
        dataFrames = {}
        for symbol, feed in self.feeds.items():
            df = feed.toDataFrame()
            df = df.add_prefix(f'{symbol}_')
            dataFrames[symbol] = df

        if not dataFrames:
            self._dates = pd.DatetimeIndex([])
            return

        symbols = list(dataFrames.keys())
        firstSymbol = symbols[0]

        if self.alignMethod == 'inner':
            combined = dataFrames[firstSymbol]
            for symbol in symbols[1:]:
                combined = combined.join(dataFrames[symbol], how='inner')

        elif self.alignMethod == 'outer':
            combined = dataFrames[firstSymbol]
            for symbol in symbols[1:]:
                combined = combined.join(dataFrames[symbol], how='outer')

        else:
            combined = dataFrames[firstSymbol]
            for symbol in symbols[1:]:
                combined = combined.join(dataFrames[symbol], how='left')

        if self.fillMethod == 'ffill':
            combined = combined.ffill()
        elif self.fillMethod == 'drop':
            combined = combined.dropna()
        elif self.fillMethod == 'zero':
            combined = combined.fillna(0)

        self._dates = combined.index

        for symbol in symbols:
            cols = [c for c in combined.columns if c.startswith(f'{symbol}_')]
            df = combined[cols].copy()
            df.columns = [c.replace(f'{symbol}_', '') for c in df.columns]
            self._alignedData[symbol] = df

    def _buildNumpyArrays(self):
        """Build NumPy arrays from aligned DataFrames for fast access. 빠른 접근을 위해 정렬된 DataFrame에서 NumPy 배열을 생성합니다."""
        for symbol, df in self._alignedData.items():
            self._npData[symbol] = {
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'volume': df['volume'].values,
            }

    def reset(self):
        """Reset the iterator to the beginning. 이터레이터를 처음으로 리셋합니다."""
        self._index = 0

    def __iter__(self) -> Iterator[Dict[str, Bar]]:
        """Iterate yielding {symbol: Bar} dict for each date. 날짜별 {종목: 바} 딕셔너리를 반환합니다."""
        if not self._loaded:
            self.load()

        self._index = 0

        for i, date in enumerate(self._dates):
            self._index = i + 1
            yield self._getBarsAtIndex(i, date)

    def _getBarsAtIndex(self, index: int, date: datetime) -> Dict[str, Bar]:
        """Return bars for all symbols at a specific index. 특정 인덱스의 모든 종목 바를 반환합니다."""
        bars = {}

        for symbol in self.symbols:
            npData = self._npData.get(symbol)
            if npData and index < len(npData['close']):
                bars[symbol] = Bar(
                    symbol=symbol,
                    datetime=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                    open=float(npData['open'][index]),
                    high=float(npData['high'][index]),
                    low=float(npData['low'][index]),
                    close=float(npData['close'][index]),
                    volume=float(npData['volume'][index]),
                    amount=0.0,
                )

        return bars

    def getBar(self, symbol: str, index: int) -> Optional[Bar]:
        """
        Return a Bar for a specific symbol at a specific index.

        특정 종목의 특정 인덱스 바를 반환합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.
            index (int): Zero-based bar index. 바 인덱스.

        Returns:
            Bar or None: Bar at the specified position, or None if
                symbol or index is invalid.
        """
        if symbol not in self._npData:
            return None

        npData = self._npData[symbol]
        if index < 0 or index >= len(npData['close']):
            return None

        date = self._dates[index]

        return Bar(
            symbol=symbol,
            datetime=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
            open=float(npData['open'][index]),
            high=float(npData['high'][index]),
            low=float(npData['low'][index]),
            close=float(npData['close'][index]),
            volume=float(npData['volume'][index]),
            amount=0.0,
        )

    def getDataFrame(self, symbol: str) -> pd.DataFrame:
        """
        Return a copy of the aligned DataFrame for a specific symbol.

        특정 종목의 정렬된 DataFrame 복사본을 반환합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.

        Returns:
            pd.DataFrame: Aligned OHLCV data for the symbol.
        """
        return self._alignedData.get(symbol, pd.DataFrame()).copy()

    def getAllDataFrames(self) -> Dict[str, pd.DataFrame]:
        """
        Return copies of aligned DataFrames for all symbols.

        모든 종목의 정렬된 DataFrame 복사본을 반환합니다.

        Returns:
            dict[str, pd.DataFrame]: Symbol-to-DataFrame mapping.
        """
        return {s: df.copy() for s, df in self._alignedData.items()}

    def getHistory(self, symbol: str, periods: int = None) -> pd.DataFrame:
        """
        Return historical data for a symbol up to the current index.

        현재 인덱스까지 특정 종목의 과거 데이터를 반환합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.
            periods (int, optional): Number of bars to look back.
                If None, returns all history. 조회할 기간 수.

        Returns:
            pd.DataFrame: Historical OHLCV data.
        """
        if symbol not in self._alignedData:
            return pd.DataFrame()

        df = self._alignedData[symbol]

        if periods is None:
            return df.iloc[:self._index].copy()
        else:
            startIdx = max(0, self._index - periods)
            return df.iloc[startIdx:self._index].copy()

    def getClose(self, symbol: str, offset: int = 0) -> Optional[float]:
        """
        Return the close price for a symbol at a relative offset.

        상대 오프셋에서 특정 종목의 종가를 반환합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.
            offset (int): Bars back from current position (0 = latest).
                Default: 0. 현재 위치로부터의 오프셋.

        Returns:
            float or None: Close price, or None if unavailable.
        """
        if symbol not in self._npData:
            return None

        idx = self._index - 1 - offset
        if idx < 0 or idx >= len(self._npData[symbol]['close']):
            return None

        return float(self._npData[symbol]['close'][idx])

    @property
    def symbols(self) -> List[str]:
        """Return the list of symbol names. 종목 목록을 반환합니다."""
        return list(self.feeds.keys())

    @property
    def totalBars(self) -> int:
        """Return the total number of synchronized bars. 동기화된 총 바 수를 반환합니다."""
        return len(self._dates) if self._dates is not None else 0

    @property
    def currentIndex(self) -> int:
        """Return the current iteration index. 현재 이터레이션 인덱스를 반환합니다."""
        return self._index

    @property
    def startDate(self) -> str:
        """Return the start date of synchronized data. 동기화된 데이터의 시작일을 반환합니다."""
        if self._dates is not None and len(self._dates) > 0:
            return self._dates[0].strftime('%Y-%m-%d')
        return ''

    @property
    def endDate(self) -> str:
        """Return the end date of synchronized data. 동기화된 데이터의 종료일을 반환합니다."""
        if self._dates is not None and len(self._dates) > 0:
            return self._dates[-1].strftime('%Y-%m-%d')
        return ''

    def __len__(self) -> int:
        return self.totalBars

    def __repr__(self) -> str:
        return (
            f"MultiDataFeed({self.symbols}, "
            f"{self.startDate} ~ {self.endDate}, "
            f"{self.totalBars} bars)"
        )
