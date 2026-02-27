"""
Tradix Indicators Module - Technical indicator computation engine.

Provides the Indicators class used internally by Strategy to compute and cache
40+ technical indicators. All indicators are pre-computed on the full dataset
and retrieved by index for maximum performance during backtests.

기술 지표 계산 엔진 모듈.
Strategy 클래스에서 내부적으로 사용하며, 전체 데이터셋에 대해 지표를
사전 계산하고 인덱스로 조회하여 백테스트 성능을 극대화합니다.

Features:
    - 40+ built-in indicators (trend, momentum, volatility, volume)
    - Pre-computation with full-dataset caching for O(1) lookback
    - Series-returning variants for crossover detection
    - Offset parameter for historical value access

Usage:
    >>> from tradix.strategy.indicators import Indicators
    >>> indicators = Indicators()
    >>> indicators.setFullData(df)
    >>> indicators.setIndex(100)
    >>> sma_val = indicators.sma(20)
    >>> rsi_val = indicators.rsi(14)
    >>> macd_val, signal, hist = indicators.macd()
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


class Indicators:
    """
    Technical indicator computation engine with pre-computation caching.

    Computes and caches indicator series on the full dataset, then retrieves
    values by index for O(1) lookback during bar-by-bar iteration. This
    design avoids redundant rolling-window calculations and dramatically
    improves backtest throughput.

    전략에서 사용하는 기술 지표들을 사전 계산하고 캐싱하여 O(1) 조회를 제공합니다.

    Attributes:
        data (pd.DataFrame): Current windowed OHLCV data (legacy).
        _fullData (pd.DataFrame): Full pre-loaded dataset for caching.
        _seriesCache (dict): Cache of pre-computed indicator arrays.
        _lastIndex (int): Current bar index for value retrieval.

    Example:
        >>> indicators = Indicators()
        >>> indicators.setFullData(ohlcv_df)
        >>> indicators.setIndex(50)
        >>> print(indicators.sma(20))
        >>> print(indicators.rsi(14))
    """

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._fullData: Optional[pd.DataFrame] = None
        self._seriesCache: dict = {}
        self._lastIndex: int = 0

    def setData(self, data: pd.DataFrame):
        """Set windowed data for legacy compatibility. / 레거시 호환용 윈도우 데이터 설정."""
        self._data = data
        self._lastIndex = len(data) if data is not None else 0

    def setIndex(self, index: int):
        """Update current bar index for optimized lookback. / 최적화된 조회를 위한 인덱스 업데이트."""
        self._lastIndex = index

    def setFullData(self, fullData: pd.DataFrame):
        """Set full dataset and clear cache for indicator pre-computation. / 전체 데이터 설정 (지표 사전 계산용)."""
        self._fullData = fullData
        self._seriesCache.clear()

    @property
    def data(self) -> pd.DataFrame:
        """Return current windowed data. / 현재 윈도우 데이터 반환."""
        return self._data if self._data is not None else pd.DataFrame()

    def _getColumn(self, column: str) -> pd.Series:
        """Retrieve a column from the windowed data as a Series. / 윈도우 데이터에서 컬럼 시리즈 추출."""
        if self._data is None or column not in self._data.columns:
            return pd.Series(dtype=float)
        return self._data[column]

    def sma(self, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """
        Compute Simple Moving Average (SMA). / 단순 이동평균 계산.

        Args:
            period: Lookback window size. / 이동평균 기간.
            column: Target column name. / 대상 컬럼명.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스 (0=현재, 1=1봉 전).

        Returns:
            float or None: Current SMA value, or None if insufficient data. / 현재 SMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'sma_{period}_{column}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            self._seriesCache[cacheKey] = self._fullData[column].rolling(window=period).mean().values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        series = self._getColumn(column)
        if len(series) < period + offset:
            return None
        return series.iloc[-(period + offset):-offset if offset > 0 else None].mean() if offset > 0 else series.tail(period).mean()

    def smaSeries(self, period: int, column: str = 'close') -> pd.Series:
        """Return full SMA Series for crossover detection. / 크로스오버 감지용 SMA 시리즈 반환."""
        series = self._getColumn(column)
        return series.rolling(window=period).mean()

    def ema(self, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """
        Compute Exponential Moving Average (EMA). / 지수 이동평균 계산.

        Args:
            period: Lookback window size. / 이동평균 기간.
            column: Target column name. / 대상 컬럼명.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Current EMA value, or None if insufficient data. / 현재 EMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'ema_{period}_{column}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            self._seriesCache[cacheKey] = self._fullData[column].ewm(span=period, adjust=False).mean().values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        series = self._getColumn(column)
        if len(series) < period:
            return None
        ema = series.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]

    def emaSeries(self, period: int, column: str = 'close') -> pd.Series:
        """Return full EMA Series for crossover detection. / 크로스오버 감지용 EMA 시리즈 반환."""
        series = self._getColumn(column)
        return series.ewm(span=period, adjust=False).mean()

    def rsi(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Relative Strength Index (RSI). / RSI (상대 강도 지수) 계산.

        Args:
            period: RSI period (default 14). / RSI 기간 (기본 14).
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: RSI value in range 0-100, or None. / RSI 값 (0~100).
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'rsi_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            series = self._fullData['close']
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avgGain = gain.rolling(window=period).mean()
            avgLoss = loss.rolling(window=period).mean()
            rs = avgGain / avgLoss
            self._seriesCache[cacheKey] = (100 - (100 / (1 + rs))).values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        series = self._getColumn('close')
        if len(series) < period + 1:
            return None

        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avgGain = gain.rolling(window=period).mean()
        avgLoss = loss.rolling(window=period).mean()

        rs = avgGain / avgLoss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def rsiSeries(self, period: int = 14) -> pd.Series:
        """Return full RSI Series. / RSI 시리즈 반환."""
        series = self._getColumn('close')
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avgGain = gain.rolling(window=period).mean()
        avgLoss = loss.rolling(window=period).mean()

        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        offset: int = 0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute MACD (Moving Average Convergence Divergence). / MACD 계산.

        Args:
            fast: Fast EMA period (default 12). / 단기 EMA 기간.
            slow: Slow EMA period (default 26). / 장기 EMA 기간.
            signal: Signal line period (default 9). / 시그널 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            tuple: (MACD line, Signal line, Histogram), any may be None.
        """
        if self._lastIndex < slow + signal + offset:
            return (None, None, None)

        cacheKey = f'macd_{fast}_{slow}_{signal}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            series = self._fullData['close']
            fastEma = series.ewm(span=fast, adjust=False).mean()
            slowEma = series.ewm(span=slow, adjust=False).mean()
            macdLine = fastEma - slowEma
            signalLine = macdLine.ewm(span=signal, adjust=False).mean()
            histogram = macdLine - signalLine
            self._seriesCache[cacheKey] = (macdLine.values, signalLine.values, histogram.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached[0]):
                m, s, h = cached[0][idx], cached[1][idx], cached[2][idx]
                return (
                    m if not pd.isna(m) else None,
                    s if not pd.isna(s) else None,
                    h if not pd.isna(h) else None
                )

        series = self._getColumn('close')
        if len(series) < slow + signal:
            return (None, None, None)

        fastEma = series.ewm(span=fast, adjust=False).mean()
        slowEma = series.ewm(span=slow, adjust=False).mean()
        macdLine = fastEma - slowEma
        signalLine = macdLine.ewm(span=signal, adjust=False).mean()
        histogram = macdLine - signalLine

        return (
            macdLine.iloc[-1],
            signalLine.iloc[-1],
            histogram.iloc[-1]
        )

    def macdSeries(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return full MACD Series tuple (macd, signal, histogram). / MACD 시리즈 반환."""
        series = self._getColumn('close')
        fastEma = series.ewm(span=fast, adjust=False).mean()
        slowEma = series.ewm(span=slow, adjust=False).mean()
        macdLine = fastEma - slowEma
        signalLine = macdLine.ewm(span=signal, adjust=False).mean()
        histogram = macdLine - signalLine
        return (macdLine, signalLine, histogram)

    def bollinger(
        self,
        period: int = 20,
        std: float = 2.0,
        offset: int = 0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute Bollinger Bands. / 볼린저 밴드 계산.

        Args:
            period: Moving average period. / 이동평균 기간.
            std: Standard deviation multiplier. / 표준편차 배수.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            tuple: (Upper band, Middle band, Lower band), any may be None.
        """
        if self._lastIndex < period + offset:
            return (None, None, None)

        cacheKey = f'bollinger_{period}_{std}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            series = self._fullData['close']
            middle = series.rolling(window=period).mean()
            stdDev = series.rolling(window=period).std()
            upper = middle + (stdDev * std)
            lower = middle - (stdDev * std)
            self._seriesCache[cacheKey] = (upper.values, middle.values, lower.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached[0]):
                u, m, l = cached[0][idx], cached[1][idx], cached[2][idx]
                return (
                    u if not pd.isna(u) else None,
                    m if not pd.isna(m) else None,
                    l if not pd.isna(l) else None
                )

        series = self._getColumn('close')
        if len(series) < period:
            return (None, None, None)

        middle = series.rolling(window=period).mean()
        stdDev = series.rolling(window=period).std()
        upper = middle + (stdDev * std)
        lower = middle - (stdDev * std)

        return (
            upper.iloc[-1],
            middle.iloc[-1],
            lower.iloc[-1]
        )

    def bollingerSeries(
        self,
        period: int = 20,
        std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return full Bollinger Bands Series tuple (upper, middle, lower). / 볼린저 밴드 시리즈 반환."""
        series = self._getColumn('close')
        middle = series.rolling(window=period).mean()
        stdDev = series.rolling(window=period).std()
        upper = middle + (stdDev * std)
        lower = middle - (stdDev * std)
        return (upper, middle, lower)

    def atr(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Average True Range (ATR). / ATR (평균 실제 범위) 계산.

        Args:
            period: ATR period (default 14). / ATR 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Current ATR value, or None if insufficient data. / 현재 ATR 값.
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'atr_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self._seriesCache[cacheKey] = tr.rolling(window=period).mean().values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        if self._data is None:
            return None

        high = self._getColumn('high')
        low = self._getColumn('low')
        close = self._getColumn('close')

        if len(close) < period + 1:
            return None

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None

    def atrSeries(self, period: int = 14) -> pd.Series:
        """Return full ATR Series. / ATR 시리즈 반환."""
        if self._data is None:
            return pd.Series(dtype=float)

        high = self._getColumn('high')
        low = self._getColumn('low')
        close = self._getColumn('close')

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def stochastic(
        self,
        kPeriod: int = 14,
        dPeriod: int = 3
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Stochastic Oscillator. / 스토캐스틱 오실레이터 계산.

        Args:
            kPeriod: %K lookback period. / %K 기간.
            dPeriod: %D smoothing period. / %D 스무딩 기간.

        Returns:
            tuple: (%K, %D) values, either may be None.
        """
        if self._data is None:
            return (None, None)

        high = self._getColumn('high')
        low = self._getColumn('low')
        close = self._getColumn('close')

        if len(close) < kPeriod + dPeriod:
            return (None, None)

        lowestLow = low.rolling(window=kPeriod).min()
        highestHigh = high.rolling(window=kPeriod).max()

        k = 100 * (close - lowestLow) / (highestHigh - lowestLow)
        d = k.rolling(window=dPeriod).mean()

        return (
            k.iloc[-1] if not pd.isna(k.iloc[-1]) else None,
            d.iloc[-1] if not pd.isna(d.iloc[-1]) else None
        )

    def stochasticSeries(
        self,
        kPeriod: int = 14,
        dPeriod: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Return full Stochastic Series (%K, %D). / 스토캐스틱 시리즈 반환."""
        if self._data is None:
            return (pd.Series(dtype=float), pd.Series(dtype=float))

        high = self._getColumn('high')
        low = self._getColumn('low')
        close = self._getColumn('close')

        lowestLow = low.rolling(window=kPeriod).min()
        highestHigh = high.rolling(window=kPeriod).max()

        k = 100 * (close - lowestLow) / (highestHigh - lowestLow)
        d = k.rolling(window=dPeriod).mean()

        return (k, d)

    def crossover(self, series1: pd.Series, series2: pd.Series) -> bool:
        """
        Detect if series1 crossed above series2 on the latest bar. / series1이 series2를 상향 돌파했는지 감지.

        Args:
            series1: First data series (e.g., fast MA). / 첫 번째 시리즈.
            series2: Second data series (e.g., slow MA). / 두 번째 시리즈.

        Returns:
            bool: True if crossover occurred. / 상향 돌파 시 True.
        """
        if len(series1) < 2 or len(series2) < 2:
            return False

        prevDiff = series1.iloc[-2] - series2.iloc[-2]
        currDiff = series1.iloc[-1] - series2.iloc[-1]

        return prevDiff <= 0 and currDiff > 0

    def crossunder(self, series1: pd.Series, series2: pd.Series) -> bool:
        """
        Detect if series1 crossed below series2 on the latest bar. / series1이 series2를 하향 돌파했는지 감지.

        Args:
            series1: First data series. / 첫 번째 시리즈.
            series2: Second data series. / 두 번째 시리즈.

        Returns:
            bool: True if crossunder occurred. / 하향 돌파 시 True.
        """
        if len(series1) < 2 or len(series2) < 2:
            return False

        prevDiff = series1.iloc[-2] - series2.iloc[-2]
        currDiff = series1.iloc[-1] - series2.iloc[-1]

        return prevDiff >= 0 and currDiff < 0

    def highest(self, period: int, column: str = 'high') -> Optional[float]:
        """Return the highest value within the lookback period. / 기간 내 최고값 반환."""
        series = self._getColumn(column)
        if len(series) < period:
            return None
        return series.tail(period).max()

    def lowest(self, period: int, column: str = 'low') -> Optional[float]:
        """Return the lowest value within the lookback period. / 기간 내 최저값 반환."""
        series = self._getColumn(column)
        if len(series) < period:
            return None
        return series.tail(period).min()

    def percentChange(self, period: int = 1, column: str = 'close') -> Optional[float]:
        """Return percentage change over N periods. / N기간 대비 변화율(%) 반환."""
        series = self._getColumn(column)
        if len(series) < period + 1:
            return None
        current = series.iloc[-1]
        past = series.iloc[-(period + 1)]
        if past == 0:
            return None
        return ((current - past) / past) * 100

    def volatility(self, period: int = 20, column: str = 'close') -> Optional[float]:
        """Compute annualized volatility (standard deviation of returns). / 연환산 변동성 계산."""
        series = self._getColumn(column)
        if len(series) < period:
            return None
        returns = series.pct_change().tail(period)
        return returns.std() * np.sqrt(252)

    def adx(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Average Directional Index (ADX) for trend strength. / ADX (추세 강도 지표) 계산.

        Args:
            period: ADX period (default 14). / ADX 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: ADX value (0-100, 25+ indicates strong trend). / ADX 값.
        """
        if self._lastIndex < period * 2 + offset:
            return None

        cacheKey = f'adx_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            plusDm = high.diff()
            minusDm = -low.diff()
            plusDm = plusDm.where((plusDm > minusDm) & (plusDm > 0), 0)
            minusDm = minusDm.where((minusDm > plusDm) & (minusDm > 0), 0)

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.ewm(span=period, adjust=False).mean()
            plusDi = 100 * (plusDm.ewm(span=period, adjust=False).mean() / atr)
            minusDi = 100 * (minusDm.ewm(span=period, adjust=False).mean() / atr)

            dx = 100 * abs(plusDi - minusDi) / (plusDi + minusDi)
            adxVal = dx.ewm(span=period, adjust=False).mean()
            self._seriesCache[cacheKey] = adxVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def adxWithDi(self, period: int = 14, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute ADX with directional indicators. / ADX와 방향 지표 계산.

        Args:
            period: ADX period (default 14). / ADX 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            tuple: (ADX, +DI, -DI), any may be None.
        """
        if self._lastIndex < period * 2 + offset:
            return (None, None, None)

        cacheKey = f'adx_di_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            plusDm = high.diff()
            minusDm = -low.diff()
            plusDm = plusDm.where((plusDm > minusDm) & (plusDm > 0), 0)
            minusDm = minusDm.where((minusDm > plusDm) & (minusDm > 0), 0)

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.ewm(span=period, adjust=False).mean()
            plusDi = 100 * (plusDm.ewm(span=period, adjust=False).mean() / atr)
            minusDi = 100 * (minusDm.ewm(span=period, adjust=False).mean() / atr)

            dx = 100 * abs(plusDi - minusDi) / (plusDi + minusDi)
            adxVal = dx.ewm(span=period, adjust=False).mean()
            self._seriesCache[cacheKey] = (adxVal.values, plusDi.values, minusDi.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached[0]):
                a, p, m = cached[0][idx], cached[1][idx], cached[2][idx]
                return (
                    a if not pd.isna(a) else None,
                    p if not pd.isna(p) else None,
                    m if not pd.isna(m) else None
                )

        return (None, None, None)

    def obv(self, offset: int = 0) -> Optional[float]:
        """
        Compute On Balance Volume (OBV). / OBV (거래량 기반 추세 지표) 계산.

        Args:
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Current OBV value. / 현재 OBV 값.
        """
        if self._lastIndex < 2 + offset:
            return None

        cacheKey = 'obv'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            volume = self._fullData['volume']
            direction = np.sign(close.diff())
            obvVal = (volume * direction).cumsum()
            self._seriesCache[cacheKey] = obvVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def williamsR(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Williams %R oscillator. / Williams %R 오실레이터 계산.

        Args:
            period: Lookback period (default 14). / 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Value in range -100 to 0 (-80-=oversold, -20+=overbought). / Williams %R 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'williams_r_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            highestHigh = high.rolling(window=period).max()
            lowestLow = low.rolling(window=period).min()

            wr = -100 * (highestHigh - close) / (highestHigh - lowestLow)
            self._seriesCache[cacheKey] = wr.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def cci(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Commodity Channel Index (CCI). / CCI (상품 채널 지수) 계산.

        Args:
            period: CCI period (default 20). / CCI 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: CCI value (+100+=overbought, -100-=oversold). / CCI 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'cci_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            tp = (high + low + close) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

            cciVal = (tp - sma) / (0.015 * mad)
            self._seriesCache[cacheKey] = cciVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def vwap(self, offset: int = 0) -> Optional[float]:
        """
        Compute Volume Weighted Average Price (VWAP). / 거래량 가중 평균가 계산.

        Args:
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Current VWAP value. / 현재 VWAP 값.
        """
        if self._lastIndex < 1 + offset:
            return None

        cacheKey = 'vwap'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']
            volume = self._fullData['volume']

            tp = (high + low + close) / 3
            cumTpVol = (tp * volume).cumsum()
            cumVol = volume.cumsum()

            vwapVal = cumTpVol / cumVol
            self._seriesCache[cacheKey] = vwapVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def mfi(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Money Flow Index (MFI), a volume-weighted RSI. / MFI (거래량 기반 RSI) 계산.

        Args:
            period: MFI period (default 14). / MFI 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: MFI value (0-100, 80+=overbought, 20-=oversold). / MFI 값.
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'mfi_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']
            volume = self._fullData['volume']

            tp = (high + low + close) / 3
            mf = tp * volume

            tpChange = tp.diff()
            posMf = mf.where(tpChange > 0, 0)
            negMf = mf.where(tpChange < 0, 0)

            posSum = posMf.rolling(window=period).sum()
            negSum = negMf.rolling(window=period).sum()

            mfiVal = 100 - (100 / (1 + posSum / negSum))
            self._seriesCache[cacheKey] = mfiVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def roc(self, period: int = 12, offset: int = 0) -> Optional[float]:
        """
        Compute Rate of Change (ROC) as percentage. / 변화율(%) 계산.

        Args:
            period: ROC period (default 12). / ROC 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: ROC value in percent. / ROC 값(%).
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'roc_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            rocVal = ((close - close.shift(period)) / close.shift(period)) * 100
            self._seriesCache[cacheKey] = rocVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def momentum(self, period: int = 10, offset: int = 0) -> Optional[float]:
        """
        Compute price Momentum (close - close[N periods ago]). / 모멘텀 계산.

        Args:
            period: Lookback period (default 10). / 모멘텀 기간.
            offset: Bars ago (0=current, 1=previous, ...). / 과거 인덱스.

        Returns:
            float or None: Momentum value. / 모멘텀 값.
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'momentum_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            momVal = close - close.shift(period)
            self._seriesCache[cacheKey] = momVal.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if idx >= 0 and idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def psar(self, afStart: float = 0.02, afStep: float = 0.02, afMax: float = 0.2, offset: int = 0) -> Optional[float]:
        """
        Compute Parabolic SAR for trend following and reversal detection. / 파라볼릭 SAR 계산.

        Args:
            afStart: Initial acceleration factor (default 0.02). / 가속인자 시작값.
            afStep: Acceleration factor step (default 0.02). / 가속인자 증가폭.
            afMax: Maximum acceleration factor (default 0.2). / 가속인자 최대값.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current Parabolic SAR value. / 현재 PSAR 값.
        """
        if self._lastIndex < 2 + offset:
            return None

        cacheKey = f'psar_{afStart}_{afStep}_{afMax}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            close = self._fullData['close'].values
            n = len(close)

            psar = np.zeros(n)
            psarBull = np.ones(n, dtype=bool)
            af = np.full(n, afStart)
            ep = np.zeros(n)

            psar[0] = close[0]
            psarBull[0] = True
            ep[0] = high[0]

            for i in range(1, n):
                if psarBull[i-1]:
                    psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                    psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])

                    if low[i] < psar[i]:
                        psarBull[i] = False
                        psar[i] = ep[i-1]
                        ep[i] = low[i]
                        af[i] = afStart
                    else:
                        psarBull[i] = True
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + afStep, afMax)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:
                    psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
                    psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])

                    if high[i] > psar[i]:
                        psarBull[i] = True
                        psar[i] = ep[i-1]
                        ep[i] = high[i]
                        af[i] = afStart
                    else:
                        psarBull[i] = False
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + afStep, afMax)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]

            self._seriesCache[cacheKey] = psar

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                return cached[idx]

        return None

    def supertrend(self, period: int = 10, multiplier: float = 3.0, offset: int = 0) -> Tuple[Optional[float], Optional[int]]:
        """
        Compute Supertrend indicator. / 슈퍼트렌드 계산.

        Args:
            period: ATR period (default 10). / ATR 기간.
            multiplier: ATR multiplier (default 3.0). / ATR 배수.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Supertrend value, direction: 1=bullish, -1=bearish).
        """
        if self._lastIndex < period + 1 + offset:
            return (None, None)

        cacheKey = f'supertrend_{period}_{multiplier}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            close = self._fullData['close'].values
            n = len(close)

            tr = np.maximum(high - low,
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                     np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]

            atr = np.zeros(n)
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

            hl2 = (high + low) / 2
            upperBand = hl2 + multiplier * atr
            lowerBand = hl2 - multiplier * atr

            supertrend = np.zeros(n)
            direction = np.ones(n, dtype=int)

            supertrend[period-1] = upperBand[period-1]
            direction[period-1] = -1

            for i in range(period, n):
                if close[i-1] <= supertrend[i-1]:
                    supertrend[i] = min(upperBand[i], supertrend[i-1]) if upperBand[i] < supertrend[i-1] or close[i-1] > supertrend[i-1] else supertrend[i-1]
                    if close[i] > supertrend[i]:
                        direction[i] = 1
                        supertrend[i] = lowerBand[i]
                    else:
                        direction[i] = -1
                        supertrend[i] = min(upperBand[i], supertrend[i-1])
                else:
                    supertrend[i] = max(lowerBand[i], supertrend[i-1]) if lowerBand[i] > supertrend[i-1] or close[i-1] < supertrend[i-1] else supertrend[i-1]
                    if close[i] < supertrend[i]:
                        direction[i] = -1
                        supertrend[i] = upperBand[i]
                    else:
                        direction[i] = 1
                        supertrend[i] = max(lowerBand[i], supertrend[i-1])

            self._seriesCache[cacheKey] = (supertrend, direction)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                return (cached[0][idx], cached[1][idx])

        return (None, None)

    def ichimoku(self, tenkanPeriod: int = 9, kijunPeriod: int = 26, senkouBPeriod: int = 52, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute Ichimoku Cloud (Ichimoku Kinko Hyo). / 일목균형표 계산.

        Args:
            tenkanPeriod: Tenkan-sen (conversion line) period (default 9). / 전환선 기간.
            kijunPeriod: Kijun-sen (base line) period (default 26). / 기준선 기간.
            senkouBPeriod: Senkou Span B period (default 52). / 선행스팬B 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Tenkan, Kijun, Senkou A, Senkou B, Chikou), any may be None.
        """
        if self._lastIndex < senkouBPeriod + offset:
            return (None, None, None, None, None)

        cacheKey = f'ichimoku_{tenkanPeriod}_{kijunPeriod}_{senkouBPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            close = self._fullData['close'].values
            n = len(close)

            def donchian(period, idx):
                start = max(0, idx - period + 1)
                return (np.max(high[start:idx+1]) + np.min(low[start:idx+1])) / 2

            tenkan = np.array([donchian(tenkanPeriod, i) if i >= tenkanPeriod - 1 else np.nan for i in range(n)])
            kijun = np.array([donchian(kijunPeriod, i) if i >= kijunPeriod - 1 else np.nan for i in range(n)])

            senkouA = np.full(n, np.nan)
            senkouB = np.full(n, np.nan)
            for i in range(kijunPeriod - 1, n):
                if i + kijunPeriod < n:
                    senkouA[i + kijunPeriod] = (tenkan[i] + kijun[i]) / 2
            for i in range(senkouBPeriod - 1, n):
                if i + kijunPeriod < n:
                    senkouB[i + kijunPeriod] = donchian(senkouBPeriod, i)

            chikou = np.full(n, np.nan)
            for i in range(kijunPeriod, n):
                chikou[i - kijunPeriod] = close[i]

            self._seriesCache[cacheKey] = (tenkan, kijun, senkouA, senkouB, chikou)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                return (
                    cached[0][idx] if not np.isnan(cached[0][idx]) else None,
                    cached[1][idx] if not np.isnan(cached[1][idx]) else None,
                    cached[2][idx] if not np.isnan(cached[2][idx]) else None,
                    cached[3][idx] if not np.isnan(cached[3][idx]) else None,
                    cached[4][idx] if not np.isnan(cached[4][idx]) else None,
                )

        return (None, None, None, None, None)

    def keltner(self, period: int = 20, atrPeriod: int = 10, multiplier: float = 2.0, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute Keltner Channel (volatility-based envelope). / 켈트너 채널 계산.

        Args:
            period: EMA period for middle band (default 20). / EMA 기간.
            atrPeriod: ATR period (default 10). / ATR 기간.
            multiplier: ATR multiplier (default 2.0). / ATR 배수.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Upper, Middle, Lower), any may be None.
        """
        if self._lastIndex < max(period, atrPeriod) + offset:
            return (None, None, None)

        cacheKey = f'keltner_{period}_{atrPeriod}_{multiplier}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            middle = close.ewm(span=period, adjust=False).mean()

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atrPeriod).mean()

            upper = middle + multiplier * atr
            lower = middle - multiplier * atr

            self._seriesCache[cacheKey] = (upper.values, middle.values, lower.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                u, m, l = cached[0][idx], cached[1][idx], cached[2][idx]
                return (
                    u if not pd.isna(u) else None,
                    m if not pd.isna(m) else None,
                    l if not pd.isna(l) else None
                )

        return (None, None, None)

    def donchian(self, period: int = 20, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute Donchian Channel for breakout strategies. / 돈치안 채널 (돌파 전략용) 계산.

        Args:
            period: Lookback period (default 20). / 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Upper, Middle, Lower), any may be None.
        """
        if self._lastIndex < period + offset:
            return (None, None, None)

        cacheKey = f'donchian_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']

            upper = high.rolling(window=period).max()
            lower = low.rolling(window=period).min()
            middle = (upper + lower) / 2

            self._seriesCache[cacheKey] = (upper.values, middle.values, lower.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                u, m, l = cached[0][idx], cached[1][idx], cached[2][idx]
                return (
                    u if not pd.isna(u) else None,
                    m if not pd.isna(m) else None,
                    l if not pd.isna(l) else None
                )

        return (None, None, None)

    def trix(self, period: int = 15, signalPeriod: int = 9, offset: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute TRIX (Triple Exponential Average). / TRIX (삼중 지수 평균) 계산.

        Args:
            period: EMA period (default 15). / EMA 기간.
            signalPeriod: Signal line period (default 9). / 시그널 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (TRIX, Signal), either may be None.
        """
        if self._lastIndex < period * 3 + signalPeriod + offset:
            return (None, None)

        cacheKey = f'trix_{period}_{signalPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']

            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()

            trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
            signal = trix.ewm(span=signalPeriod, adjust=False).mean()

            self._seriesCache[cacheKey] = (trix.values, signal.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                t, s = cached[0][idx], cached[1][idx]
                return (
                    t if not pd.isna(t) else None,
                    s if not pd.isna(s) else None
                )

        return (None, None)

    def dpo(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Detrended Price Oscillator (DPO). / 추세 제거 가격 오실레이터 계산.

        Args:
            period: DPO period (default 20). / DPO 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current DPO value. / 현재 DPO 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'dpo_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            sma = close.rolling(window=period).mean()
            shift = period // 2 + 1
            dpo = close.shift(shift) - sma
            self._seriesCache[cacheKey] = dpo.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def cmo(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Chande Momentum Oscillator (CMO). / CMO 계산.

        Args:
            period: CMO period (default 14). / CMO 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: CMO value in range -100 to 100. / CMO 값 (-100~100).
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'cmo_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            delta = close.diff()

            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            sumGain = gain.rolling(window=period).sum()
            sumLoss = loss.rolling(window=period).sum()

            cmo = 100 * (sumGain - sumLoss) / (sumGain + sumLoss)
            self._seriesCache[cacheKey] = cmo.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def ulcer(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Ulcer Index for downside risk measurement. / 울서 인덱스 (하락 리스크) 계산.

        Args:
            period: Lookback period (default 14). / 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current Ulcer Index value. / 현재 Ulcer Index 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'ulcer_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            highest = close.rolling(window=period).max()
            drawdown = 100 * (close - highest) / highest
            ulcer = np.sqrt((drawdown ** 2).rolling(window=period).mean())
            self._seriesCache[cacheKey] = ulcer.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def elderRay(self, period: int = 13, offset: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Elder Ray Index (Bull Power, Bear Power). / 엘더 레이 지수 계산.

        Args:
            period: EMA period (default 13). / EMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Bull Power, Bear Power), either may be None.
        """
        if self._lastIndex < period + offset:
            return (None, None)

        cacheKey = f'elder_ray_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            ema = close.ewm(span=period, adjust=False).mean()
            bullPower = high - ema
            bearPower = low - ema

            self._seriesCache[cacheKey] = (bullPower.values, bearPower.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                b, br = cached[0][idx], cached[1][idx]
                return (
                    b if not pd.isna(b) else None,
                    br if not pd.isna(br) else None
                )

        return (None, None)

    def chaikin(self, fastPeriod: int = 3, slowPeriod: int = 10, offset: int = 0) -> Optional[float]:
        """
        Compute Chaikin Oscillator (A/D Line based). / 차이킨 오실레이터 계산.

        Args:
            fastPeriod: Fast EMA period (default 3). / 단기 EMA 기간.
            slowPeriod: Slow EMA period (default 10). / 장기 EMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current Chaikin Oscillator value. / 차이킨 오실레이터 값.
        """
        if self._lastIndex < slowPeriod + offset:
            return None

        cacheKey = f'chaikin_{fastPeriod}_{slowPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']
            volume = self._fullData['volume']

            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            adl = mfv.cumsum()

            fastEma = adl.ewm(span=fastPeriod, adjust=False).mean()
            slowEma = adl.ewm(span=slowPeriod, adjust=False).mean()
            chaikin = fastEma - slowEma

            self._seriesCache[cacheKey] = chaikin.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def adl(self, offset: int = 0) -> Optional[float]:
        """
        Compute Accumulation/Distribution Line (A/D Line). / 누적/분배 라인 계산.

        Args:
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current A/D Line value. / 현재 A/D 라인 값.
        """
        if self._lastIndex < 1 + offset:
            return None

        cacheKey = 'adl'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']
            volume = self._fullData['volume']

            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            adl = mfv.cumsum()

            self._seriesCache[cacheKey] = adl.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def emv(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Ease of Movement (EMV). / EMV (가격 변화 용이성) 계산.

        Args:
            period: Smoothing period (default 14). / 평균 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current smoothed EMV value. / 현재 EMV 값.
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'emv_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            volume = self._fullData['volume']

            hl = (high + low) / 2
            distance = hl.diff()
            boxRatio = (volume / 1e8) / (high - low)
            emv = distance / boxRatio
            emvMa = emv.rolling(window=period).mean()

            self._seriesCache[cacheKey] = emvMa.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def forceIndex(self, period: int = 13, offset: int = 0) -> Optional[float]:
        """
        Compute Force Index (price change * volume). / 포스 인덱스 계산.

        Args:
            period: EMA smoothing period (default 13). / EMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current Force Index value. / 현재 포스 인덱스 값.
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'force_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            volume = self._fullData['volume']

            force = close.diff() * volume
            forceEma = force.ewm(span=period, adjust=False).mean()

            self._seriesCache[cacheKey] = forceEma.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def nvi(self, offset: int = 0) -> Optional[float]:
        """
        Compute Negative Volume Index (NVI). / NVI (음의 거래량 지수) 계산.

        Args:
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current NVI value (starts at 1000). / 현재 NVI 값.
        """
        if self._lastIndex < 2 + offset:
            return None

        cacheKey = 'nvi'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close'].values
            volume = self._fullData['volume'].values
            n = len(close)

            nvi = np.zeros(n)
            nvi[0] = 1000

            for i in range(1, n):
                if volume[i] < volume[i-1]:
                    nvi[i] = nvi[i-1] + (close[i] - close[i-1]) / close[i-1] * nvi[i-1]
                else:
                    nvi[i] = nvi[i-1]

            self._seriesCache[cacheKey] = nvi

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                return cached[idx]

        return None

    def pvi(self, offset: int = 0) -> Optional[float]:
        """
        Compute Positive Volume Index (PVI). / PVI (양의 거래량 지수) 계산.

        Args:
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current PVI value (starts at 1000). / 현재 PVI 값.
        """
        if self._lastIndex < 2 + offset:
            return None

        cacheKey = 'pvi'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close'].values
            volume = self._fullData['volume'].values
            n = len(close)

            pvi = np.zeros(n)
            pvi[0] = 1000

            for i in range(1, n):
                if volume[i] > volume[i-1]:
                    pvi[i] = pvi[i-1] + (close[i] - close[i-1]) / close[i-1] * pvi[i-1]
                else:
                    pvi[i] = pvi[i-1]

            self._seriesCache[cacheKey] = pvi

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                return cached[idx]

        return None

    def vroc(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """
        Compute Volume Rate of Change (VROC). / 거래량 변화율 계산.

        Args:
            period: Lookback period (default 14). / 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: VROC value in percent. / VROC 값(%).
        """
        if self._lastIndex < period + 1 + offset:
            return None

        cacheKey = f'vroc_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            volume = self._fullData['volume']
            vroc = ((volume - volume.shift(period)) / volume.shift(period)) * 100
            self._seriesCache[cacheKey] = vroc.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def pvt(self, offset: int = 0) -> Optional[float]:
        """
        Compute Price Volume Trend (PVT). / 가격 거래량 추세 계산.

        Args:
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current PVT value. / 현재 PVT 값.
        """
        if self._lastIndex < 2 + offset:
            return None

        cacheKey = 'pvt'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            volume = self._fullData['volume']
            pvt = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
            self._seriesCache[cacheKey] = pvt.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def zigzag(self, threshold: float = 5.0, offset: int = 0) -> Optional[float]:
        """
        Compute ZigZag indicator for trend reversal identification. / 지그재그 추세 전환점 식별.

        Args:
            threshold: Reversal threshold in percent (default 5.0). / 전환 임계값(%).
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Pivot point value, or None if not a pivot. / 피봇 포인트 값.
        """
        if self._lastIndex < 3 + offset:
            return None

        cacheKey = f'zigzag_{threshold}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close'].values
            n = len(close)

            zigzag = np.full(n, np.nan)
            trend = 1
            lastPivot = close[0]
            lastPivotIdx = 0
            zigzag[0] = close[0]

            for i in range(1, n):
                change = (close[i] - lastPivot) / lastPivot * 100

                if trend == 1:
                    if close[i] > lastPivot:
                        lastPivot = close[i]
                        lastPivotIdx = i
                    elif change <= -threshold:
                        zigzag[lastPivotIdx] = lastPivot
                        trend = -1
                        lastPivot = close[i]
                        lastPivotIdx = i
                else:
                    if close[i] < lastPivot:
                        lastPivot = close[i]
                        lastPivotIdx = i
                    elif change >= threshold:
                        zigzag[lastPivotIdx] = lastPivot
                        trend = 1
                        lastPivot = close[i]
                        lastPivotIdx = i

            zigzag[lastPivotIdx] = lastPivot
            self._seriesCache[cacheKey] = zigzag

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not np.isnan(val) else None

        return None

    def hma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Hull Moving Average (HMA) for reduced lag. / HMA (헐 이동평균) 계산.

        Args:
            period: HMA period (default 20). / HMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current HMA value. / 현재 HMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'hma_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']

            halfPeriod = int(period / 2)
            sqrtPeriod = int(np.sqrt(period))

            wma1 = close.rolling(window=halfPeriod).apply(
                lambda x: np.sum(x * np.arange(1, halfPeriod + 1)) / np.sum(np.arange(1, halfPeriod + 1)), raw=True)
            wma2 = close.rolling(window=period).apply(
                lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True)

            rawHma = 2 * wma1 - wma2
            hma = rawHma.rolling(window=sqrtPeriod).apply(
                lambda x: np.sum(x * np.arange(1, sqrtPeriod + 1)) / np.sum(np.arange(1, sqrtPeriod + 1)), raw=True)

            self._seriesCache[cacheKey] = hma.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def tema(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Triple Exponential Moving Average (TEMA). / TEMA 계산.

        Args:
            period: TEMA period (default 20). / TEMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current TEMA value. / 현재 TEMA 값.
        """
        if self._lastIndex < period * 3 + offset:
            return None

        cacheKey = f'tema_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']

            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()

            tema = 3 * ema1 - 3 * ema2 + ema3
            self._seriesCache[cacheKey] = tema.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def dema(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Double Exponential Moving Average (DEMA). / DEMA 계산.

        Args:
            period: DEMA period (default 20). / DEMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current DEMA value. / 현재 DEMA 값.
        """
        if self._lastIndex < period * 2 + offset:
            return None

        cacheKey = f'dema_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']

            ema1 = close.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()

            dema = 2 * ema1 - ema2
            self._seriesCache[cacheKey] = dema.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def wma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Weighted Moving Average (WMA). / 가중 이동평균 계산.

        Args:
            period: WMA period (default 20). / WMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current WMA value. / 현재 WMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'wma_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            weights = np.arange(1, period + 1)
            wma = close.rolling(window=period).apply(
                lambda x: np.sum(x * weights) / np.sum(weights), raw=True)
            self._seriesCache[cacheKey] = wma.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def vwma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Volume Weighted Moving Average (VWMA). / 거래량 가중 이동평균 계산.

        Args:
            period: VWMA period (default 20). / VWMA 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current VWMA value. / 현재 VWMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'vwma_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            volume = self._fullData['volume']

            vwma = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            self._seriesCache[cacheKey] = vwma.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None

        return None

    def alma(self, period: int = 20, sigma: float = 6.0, offset_factor: float = 0.85, offset: int = 0) -> Optional[float]:
        """
        Compute Arnaud Legoux Moving Average (ALMA). / ALMA 계산.

        Args:
            period: ALMA period (default 20). / ALMA 기간.
            sigma: Gaussian distribution width (default 6.0). / 가우시안 분포 너비.
            offset_factor: Offset factor (default 0.85). / 오프셋 팩터.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            float or None: Current ALMA value. / 현재 ALMA 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'alma_{period}_{sigma}_{offset_factor}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            n = len(close)

            m = offset_factor * (period - 1)
            s = period / sigma
            weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
            weights /= weights.sum()

            alma = np.full(n, np.nan)
            for i in range(period - 1, n):
                alma[i] = np.sum(close.values[i - period + 1:i + 1] * weights)

            self._seriesCache[cacheKey] = alma

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not np.isnan(val) else None

        return None

    def pivotPoints(self, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute Pivot Points for support/resistance levels. / 피봇 포인트 (지지/저항) 계산.

        Args:
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (Pivot, R1, R2, R3, S1, S2, S3).
        """
        if self._lastIndex < 2 + offset:
            return (None, None, None, None, None, None, None)

        cacheKey = 'pivot_points'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            close = self._fullData['close'].values
            n = len(close)

            pivot = np.zeros(n)
            r1 = np.zeros(n)
            r2 = np.zeros(n)
            r3 = np.zeros(n)
            s1 = np.zeros(n)
            s2 = np.zeros(n)
            s3 = np.zeros(n)

            for i in range(1, n):
                pp = (high[i-1] + low[i-1] + close[i-1]) / 3
                pivot[i] = pp
                r1[i] = 2 * pp - low[i-1]
                s1[i] = 2 * pp - high[i-1]
                r2[i] = pp + (high[i-1] - low[i-1])
                s2[i] = pp - (high[i-1] - low[i-1])
                r3[i] = high[i-1] + 2 * (pp - low[i-1])
                s3[i] = low[i-1] - 2 * (high[i-1] - pp)

            self._seriesCache[cacheKey] = (pivot, r1, r2, r3, s1, s2, s3)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                return tuple(cached[i][idx] for i in range(7))

        return (None, None, None, None, None, None, None)

    def fibonacciRetracement(self, period: int = 50, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute Fibonacci Retracement levels. / 피보나치 되돌림 레벨 계산.

        Args:
            period: Lookback period for high/low detection (default 50). / 고점/저점 탐색 기간.
            offset: Bars ago (0=current). / 과거 인덱스.

        Returns:
            tuple: (0%, 23.6%, 38.2%, 50%, 61.8%) retracement levels.
        """
        if self._lastIndex < period + offset:
            return (None, None, None, None, None)

        cacheKey = f'fib_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            n = len(high)

            fib0 = np.zeros(n)
            fib236 = np.zeros(n)
            fib382 = np.zeros(n)
            fib50 = np.zeros(n)
            fib618 = np.zeros(n)

            for i in range(period - 1, n):
                periodHigh = np.max(high[i - period + 1:i + 1])
                periodLow = np.min(low[i - period + 1:i + 1])
                diff = periodHigh - periodLow

                fib0[i] = periodLow
                fib236[i] = periodLow + 0.236 * diff
                fib382[i] = periodLow + 0.382 * diff
                fib50[i] = periodLow + 0.5 * diff
                fib618[i] = periodLow + 0.618 * diff

            self._seriesCache[cacheKey] = (fib0, fib236, fib382, fib50, fib618)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]) and cached[0][idx] != 0:
                return tuple(cached[i][idx] for i in range(5))

        return (None, None, None, None, None)

    def stochasticRsi(self, rsiPeriod: int = 14, stochPeriod: int = 14, kPeriod: int = 3, dPeriod: int = 3, offset: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Stochastic RSI (%K, %D). / 스토캐스틱 RSI 계산.

        Args:
            rsiPeriod: RSI period. / RSI 기간.
            stochPeriod: Stochastic lookback. / 스토캐스틱 기간.
            kPeriod: %K smoothing. / %K 스무딩 기간.
            dPeriod: %D smoothing. / %D 스무딩 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            Tuple of (%K, %D) values in 0-100 range, or (None, None). / (%K, %D) 값.
        """
        cacheKey = f'stochRsi_{rsiPeriod}_{stochPeriod}_{kPeriod}_{dPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            series = self._fullData['close']
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avgGain = gain.rolling(window=rsiPeriod).mean()
            avgLoss = loss.rolling(window=rsiPeriod).mean()
            rs = avgGain / avgLoss
            rsiValues = 100 - (100 / (1 + rs))

            rsiMin = rsiValues.rolling(window=stochPeriod).min()
            rsiMax = rsiValues.rolling(window=stochPeriod).max()
            rsiRange = rsiMax - rsiMin
            stochRsi = np.where(rsiRange != 0, (rsiValues - rsiMin) / rsiRange * 100, 50)
            k = pd.Series(stochRsi, index=series.index).rolling(window=kPeriod).mean().values
            d = pd.Series(k, index=series.index).rolling(window=dPeriod).mean().values
            self._seriesCache[cacheKey] = (k, d)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                kVal = cached[0][idx]
                dVal = cached[1][idx]
                kVal = kVal if not pd.isna(kVal) else None
                dVal = dVal if not pd.isna(dVal) else None
                return (kVal, dVal)
        return (None, None)

    def kdj(self, period: int = 9, kPeriod: int = 3, dPeriod: int = 3, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute KDJ indicator. / KDJ 지표 계산.

        Args:
            period: Stochastic lookback. / 스토캐스틱 기간.
            kPeriod: %K smoothing. / %K 스무딩 기간.
            dPeriod: %D smoothing. / %D 스무딩 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            Tuple of (K, D, J) values, or (None, None, None). / (K, D, J) 값.
        """
        cacheKey = f'kdj_{period}_{kPeriod}_{dPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high']
            low = self._fullData['low']
            close = self._fullData['close']

            lowestLow = low.rolling(window=period).min()
            highestHigh = high.rolling(window=period).max()
            priceRange = highestHigh - lowestLow
            rsv = np.where(priceRange != 0, (close - lowestLow) / priceRange * 100, 50)
            rsvSeries = pd.Series(rsv, index=close.index)

            k = rsvSeries.ewm(com=kPeriod - 1, adjust=False).mean().values
            d = pd.Series(k, index=close.index).ewm(com=dPeriod - 1, adjust=False).mean().values
            j = 3 * k - 2 * d
            self._seriesCache[cacheKey] = (k, d, j)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                kVal = cached[0][idx]
                dVal = cached[1][idx]
                jVal = cached[2][idx]
                kVal = kVal if not pd.isna(kVal) else None
                dVal = dVal if not pd.isna(dVal) else None
                jVal = jVal if not pd.isna(jVal) else None
                return (kVal, dVal, jVal)
        return (None, None, None)

    def awesomeOscillator(self, fastPeriod: int = 5, slowPeriod: int = 34, offset: int = 0) -> Optional[float]:
        """
        Compute Awesome Oscillator (AO). / 어썸 오실레이터 계산.

        Args:
            fastPeriod: Fast SMA period (default 5). / 빠른 SMA 기간.
            slowPeriod: Slow SMA period (default 34). / 느린 SMA 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            float or None: AO value. / AO 값.
        """
        if self._lastIndex < slowPeriod + offset:
            return None

        cacheKey = f'ao_{fastPeriod}_{slowPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            midPrice = (self._fullData['high'] + self._fullData['low']) / 2
            fastSma = midPrice.rolling(window=fastPeriod).mean()
            slowSma = midPrice.rolling(window=slowPeriod).mean()
            self._seriesCache[cacheKey] = (fastSma - slowSma).values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None
        return None

    def ultimateOscillator(self, short: int = 7, medium: int = 14, long: int = 28, offset: int = 0) -> Optional[float]:
        """
        Compute Ultimate Oscillator. / 얼티밋 오실레이터 계산.

        Args:
            short: Short period (default 7). / 단기 기간.
            medium: Medium period (default 14). / 중기 기간.
            long: Long period (default 28). / 장기 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            float or None: UO value in 0-100 range. / UO 값 (0~100).
        """
        if self._lastIndex < long + 1 + offset:
            return None

        cacheKey = f'uo_{short}_{medium}_{long}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close'].values
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            n = len(close)

            bp = np.zeros(n)
            tr = np.zeros(n)
            for i in range(1, n):
                trueLow = min(low[i], close[i - 1])
                trueHigh = max(high[i], close[i - 1])
                bp[i] = close[i] - trueLow
                tr[i] = trueHigh - trueLow

            bpSeries = pd.Series(bp)
            trSeries = pd.Series(tr)

            avg7 = bpSeries.rolling(short).sum() / trSeries.rolling(short).sum()
            avg14 = bpSeries.rolling(medium).sum() / trSeries.rolling(medium).sum()
            avg28 = bpSeries.rolling(long).sum() / trSeries.rolling(long).sum()

            uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            self._seriesCache[cacheKey] = uo.values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None
        return None

    def klingerOscillator(self, fastPeriod: int = 34, slowPeriod: int = 55, signalPeriod: int = 13, offset: int = 0) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Klinger Volume Oscillator. / 클링거 거래량 오실레이터 계산.

        Args:
            fastPeriod: Fast EMA period. / 빠른 EMA 기간.
            slowPeriod: Slow EMA period. / 느린 EMA 기간.
            signalPeriod: Signal line EMA period. / 시그널 라인 EMA 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            Tuple of (KVO, Signal) values, or (None, None). / (KVO, 시그널) 값.
        """
        cacheKey = f'klinger_{fastPeriod}_{slowPeriod}_{signalPeriod}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            high = self._fullData['high'].values
            low = self._fullData['low'].values
            close = self._fullData['close'].values
            volume = self._fullData['volume'].values
            n = len(close)

            hlc = (high + low + close) / 3
            trend = np.ones(n)
            for i in range(1, n):
                trend[i] = 1 if hlc[i] > hlc[i - 1] else -1

            dm = high - low
            vf = volume * abs(2 * dm / np.where(dm != 0, dm, 1) - 1) * trend

            vfSeries = pd.Series(vf)
            kvo = vfSeries.ewm(span=fastPeriod, adjust=False).mean() - vfSeries.ewm(span=slowPeriod, adjust=False).mean()
            signal = kvo.ewm(span=signalPeriod, adjust=False).mean()
            self._seriesCache[cacheKey] = (kvo.values, signal.values)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                kvoVal = cached[0][idx]
                sigVal = cached[1][idx]
                kvoVal = kvoVal if not pd.isna(kvoVal) else None
                sigVal = sigVal if not pd.isna(sigVal) else None
                return (kvoVal, sigVal)
        return (None, None)

    def bollingerPercentB(self, period: int = 20, numStd: float = 2.0, offset: int = 0) -> Optional[float]:
        """
        Compute Bollinger Band %B (position within bands). / 볼린저 밴드 %B (밴드 내 위치) 계산.

        Args:
            period: Bollinger period. / 볼린저 기간.
            numStd: Standard deviation multiplier. / 표준편차 배수.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            float or None: %B value (0=lower, 1=upper). / %B 값 (0=하단, 1=상단).
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'bbPctB_{period}_{numStd}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            middle = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = middle + numStd * std
            lower = middle - numStd * std
            bandwidth = upper - lower
            pctB = np.where(bandwidth != 0, (close - lower) / bandwidth, 0.5)
            self._seriesCache[cacheKey] = pctB

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None
        return None

    def bollingerWidth(self, period: int = 20, numStd: float = 2.0, offset: int = 0) -> Optional[float]:
        """
        Compute Bollinger Band Width (normalized). / 볼린저 밴드 폭 (정규화) 계산.

        Args:
            period: Bollinger period. / 볼린저 기간.
            numStd: Standard deviation multiplier. / 표준편차 배수.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            float or None: Bandwidth percentage. / 밴드 폭 (%).
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'bbWidth_{period}_{numStd}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close']
            middle = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            upper = middle + numStd * std
            lower = middle - numStd * std
            width = np.where(middle != 0, (upper - lower) / middle * 100, 0)
            self._seriesCache[cacheKey] = width

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None
        return None

    def twap(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """
        Compute Time-Weighted Average Price (TWAP). / 시간 가중 평균가 계산.

        Args:
            period: TWAP period. / TWAP 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            float or None: TWAP value. / TWAP 값.
        """
        if self._lastIndex < period + offset:
            return None

        cacheKey = f'twap_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            ohlc = (self._fullData['open'] + self._fullData['high'] + self._fullData['low'] + self._fullData['close']) / 4
            self._seriesCache[cacheKey] = ohlc.rolling(window=period).mean().values

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached):
                val = cached[idx]
                return val if not pd.isna(val) else None
        return None

    def linearRegression(self, period: int = 20, offset: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute Linear Regression (value, slope, R-squared). / 선형회귀 (값, 기울기, R제곱) 계산.

        Args:
            period: Regression lookback. / 회귀 기간.
            offset: Bars ago. / 과거 인덱스.

        Returns:
            Tuple of (regression value, slope, r_squared), or (None, None, None). / (회귀값, 기울기, R제곱).
        """
        if self._lastIndex < period + offset:
            return (None, None, None)

        cacheKey = f'linreg_{period}'
        if cacheKey not in self._seriesCache and self._fullData is not None:
            close = self._fullData['close'].values
            n = len(close)
            x = np.arange(period, dtype=float)
            xMean = x.mean()
            xVar = np.sum((x - xMean) ** 2)

            regVal = np.full(n, np.nan)
            slope = np.full(n, np.nan)
            rSq = np.full(n, np.nan)

            for i in range(period - 1, n):
                y = close[i - period + 1:i + 1]
                yMean = y.mean()
                covXY = np.sum((x - xMean) * (y - yMean))
                b = covXY / xVar if xVar != 0 else 0
                a = yMean - b * xMean
                regVal[i] = a + b * (period - 1)
                slope[i] = b
                yPred = a + b * x
                ssTot = np.sum((y - yMean) ** 2)
                ssRes = np.sum((y - yPred) ** 2)
                rSq[i] = 1 - ssRes / ssTot if ssTot != 0 else 0

            self._seriesCache[cacheKey] = (regVal, slope, rSq)

        cached = self._seriesCache.get(cacheKey)
        if cached is not None:
            idx = self._lastIndex - 1 - offset
            if 0 <= idx < len(cached[0]):
                v = cached[0][idx]
                s = cached[1][idx]
                r = cached[2][idx]
                v = v if not pd.isna(v) else None
                s = s if not pd.isna(s) else None
                r = r if not pd.isna(r) else None
                return (v, s, r)
        return (None, None, None)

    def __repr__(self) -> str:
        rows = len(self._data) if self._data is not None else 0
        return f"Indicators(rows={rows})"
