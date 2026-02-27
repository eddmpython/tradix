"""
Tradix Price Cache Module.

Manages local Parquet-based caching of OHLCV price data to minimize
redundant API calls. Supports per-market cache directories, automatic
merging of new data with existing cache, and cache expiration management.

가격 캐시 모듈 - OHLCV 데이터를 Parquet 형식으로 로컬 캐시하여
API 호출을 최소화합니다.

Features:
    - Parquet format for efficient storage and fast I/O
    - Automatic merge with existing cached data (deduplication)
    - Per-market cache directories (KRX, US, custom)
    - Cache expiration based on file modification time
    - Cache info, listing, and size reporting
    - Date range slicing on retrieval

Usage:
    >>> from tradix.datafeed.cache import PriceCache
    >>> cache = PriceCache(market='KRX', maxAgeDays=1)
    >>> df = cache.get('005930', '2020-01-01', '2024-12-31')
    >>> cache.save('005930', new_data, merge=True)
    >>> if cache.needsUpdate('005930'):
    ...     fresh_data = fetch_from_api('005930')
    ...     cache.save('005930', fresh_data)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

_CACHE_ROOT = Path.home() / '.tradix' / 'cache'
KRX_PRICE_DIR = _CACHE_ROOT / 'krx'
US_PRICE_DIR = _CACHE_ROOT / 'us'
BACKTEST_CACHE_DIR = _CACHE_ROOT


class PriceCache:
    """
    Parquet-based price data cache manager.

    Stores and retrieves OHLCV data in Parquet format with per-market
    directory organization, automatic data merging, and expiration
    management.

    주가 데이터 Parquet 캐시 관리자 - 시장별 디렉토리에 Parquet
    형식으로 데이터를 저장/조회하며, 자동 병합 및 만료 관리를 지원합니다.

    Attributes:
        market (str): Market code ('KRX', 'US', etc.). 시장 코드.
        maxAgeDays (int): Maximum cache age in days before considered
            stale. Default: 1. 캐시 최대 유효 기간 (일).
        cacheDir (Path): Directory path for cache files. 캐시 디렉토리.

    Example:
        >>> cache = PriceCache(market='KRX')
        >>> df = cache.get('005930', '2020-01-01', '2024-12-31')
        >>> cache.save('005930', df, merge=True)
    """

    def __init__(self, market: str = 'KRX', maxAgeDays: int = 1):
        """
        Initialize the price cache.

        가격 캐시를 초기화합니다.

        Args:
            market (str): Market code ('KRX', 'US', or custom).
                Default: 'KRX'. 시장 코드.
            maxAgeDays (int): Cache expiration in days. Default: 1.
                캐시 최대 유효 기간.
        """
        self.market = market.upper()
        self.maxAgeDays = maxAgeDays

        if self.market == 'KRX':
            self.cacheDir = KRX_PRICE_DIR
        elif self.market == 'US':
            self.cacheDir = US_PRICE_DIR
        else:
            self.cacheDir = BACKTEST_CACHE_DIR / market.lower()

        self.cacheDir.mkdir(parents=True, exist_ok=True)

    def _getCachePath(self, symbol: str) -> Path:
        """Return the Parquet file path for a symbol. 종목의 캐시 파일 경로를 반환합니다."""
        safeSymbol = symbol.replace('/', '_').replace('\\', '_')
        return self.cacheDir / f"{safeSymbol}.parquet"

    def exists(self, symbol: str) -> bool:
        """Check if a cache file exists for the given symbol. 캐시 파일 존재 여부를 확인합니다."""
        return self._getCachePath(symbol).exists()

    def get(
        self,
        symbol: str,
        startDate: Optional[str] = None,
        endDate: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load cached data for a symbol, optionally filtered by date range.

        종목의 캐시 데이터를 로드합니다 (날짜 범위 필터 선택적 적용).

        Args:
            symbol (str): Ticker symbol. 종목 코드.
            startDate (str, optional): Start date filter (YYYY-MM-DD).
                시작일.
            endDate (str, optional): End date filter (YYYY-MM-DD).
                종료일.

        Returns:
            pd.DataFrame or None: Cached OHLCV data, or None if no
                cache exists or data is empty.
        """
        path = self._getCachePath(symbol)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)

            if startDate or endDate:
                if startDate:
                    df = df[df.index >= startDate]
                if endDate:
                    df = df[df.index <= endDate]

            return df if len(df) > 0 else None

        except Exception:
            return None

    def save(self, symbol: str, df: pd.DataFrame, merge: bool = True):
        """
        Save OHLCV data to the cache.

        OHLCV 데이터를 캐시에 저장합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.
            df (pd.DataFrame): OHLCV DataFrame to save. 저장할 데이터.
            merge (bool): If True, merge with existing cache data,
                keeping the latest entry for duplicate dates.
                Default: True. 기존 캐시와 병합 여부.
        """
        if df is None or len(df) == 0:
            return

        path = self._getCachePath(symbol)

        if merge and path.exists():
            try:
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
            except Exception:
                pass

        df.to_parquet(path)

    def delete(self, symbol: str) -> bool:
        """
        Delete the cache file for a symbol.

        종목의 캐시 파일을 삭제합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.

        Returns:
            bool: True if cache was deleted, False if it did not exist.
        """
        path = self._getCachePath(symbol)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self):
        """Delete all cached files in this market's cache directory. 이 시장의 모든 캐시 파일을 삭제합니다."""
        for path in self.cacheDir.glob('*.parquet'):
            path.unlink()

    def needsUpdate(self, symbol: str) -> bool:
        """
        Check if the cache needs updating based on file age.

        파일 수정 시간을 기준으로 캐시 갱신 필요 여부를 확인합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.

        Returns:
            bool: True if cache does not exist or has exceeded
                maxAgeDays since last modification.
        """
        path = self._getCachePath(symbol)
        if not path.exists():
            return True

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime

        return age > timedelta(days=self.maxAgeDays)

    def getLastDate(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the last date in the cached data. 캐시된 마지막 날짜를 반환합니다."""
        df = self.get(symbol)
        if df is not None and len(df) > 0:
            return df.index[-1]
        return None

    def getFirstDate(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the first date in the cached data. 캐시된 첫 번째 날짜를 반환합니다."""
        df = self.get(symbol)
        if df is not None and len(df) > 0:
            return df.index[0]
        return None

    def getDateRange(self, symbol: str) -> tuple:
        """Return the cached date range as (first_date, last_date). 캐시된 날짜 범위를 반환합니다."""
        return (self.getFirstDate(symbol), self.getLastDate(symbol))

    def getCacheInfo(self, symbol: str) -> dict:
        """
        Return detailed cache metadata for a symbol.

        종목의 캐시 상세 정보를 반환합니다.

        Args:
            symbol (str): Ticker symbol. 종목 코드.

        Returns:
            dict: Cache info with keys 'exists', 'symbol', 'path',
                'sizeBytes', 'sizeMb', 'modifiedAt', 'rowCount',
                'firstDate', 'lastDate', 'needsUpdate'.
        """
        path = self._getCachePath(symbol)
        if not path.exists():
            return {
                'exists': False,
                'symbol': symbol,
            }

        stat = path.stat()
        df = self.get(symbol)
        rowCount = len(df) if df is not None else 0

        return {
            'exists': True,
            'symbol': symbol,
            'path': str(path),
            'sizeBytes': stat.st_size,
            'sizeMb': round(stat.st_size / (1024 * 1024), 2),
            'modifiedAt': datetime.fromtimestamp(stat.st_mtime),
            'rowCount': rowCount,
            'firstDate': self.getFirstDate(symbol),
            'lastDate': self.getLastDate(symbol),
            'needsUpdate': self.needsUpdate(symbol),
        }

    def listCached(self) -> list:
        """
        Return a sorted list of all cached symbol names.

        캐시된 모든 종목 이름의 정렬된 목록을 반환합니다.

        Returns:
            list[str]: Sorted list of cached symbols.
        """
        symbols = []
        for path in self.cacheDir.glob('*.parquet'):
            symbols.append(path.stem)
        return sorted(symbols)

    def getTotalSize(self) -> int:
        """
        Return the total size of all cache files in bytes.

        모든 캐시 파일의 총 크기를 바이트 단위로 반환합니다.

        Returns:
            int: Total cache size in bytes.
        """
        total = 0
        for path in self.cacheDir.glob('*.parquet'):
            total += path.stat().st_size
        return total

    def __repr__(self) -> str:
        cachedCount = len(self.listCached())
        totalSize = self.getTotalSize() / (1024 * 1024)
        return (
            f"PriceCache(market={self.market}, "
            f"cached={cachedCount}, size={totalSize:.1f}MB)"
        )
