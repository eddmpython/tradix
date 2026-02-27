"""
Vectorized Technical Indicators - Pure NumPy implementation.

High-performance technical indicator calculations using NumPy vectorized
operations. Each function takes NumPy arrays and returns NumPy arrays,
enabling batch computation over entire price histories at once.

Available Indicators:
    - vsma: Simple Moving Average (cumsum optimized, 0.006ms)
    - vema: Exponential Moving Average
    - vrsi: Relative Strength Index (Wilder's smoothing, 0.009ms)
    - vmacd: MACD line, signal line, histogram (0.040ms)
    - vbollinger: Bollinger Bands (upper, middle, lower)
    - vatr: Average True Range
    - vstochastic: Stochastic Oscillator (%K, %D)
    - vadx: Average Directional Index
    - vroc: Rate of Change (fully vectorized)
    - vmomentum: Price Momentum (fully vectorized)

Usage:
    >>> import numpy as np
    >>> from tradix.vectorized.indicators import vsma, vrsi, vmacd
    >>> close = np.array([100, 102, 101, 103, 105, 104, 106])
    >>> sma = vsma(close, period=3)
    >>> rsi = vrsi(close, period=14)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def vsma(close: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Compute Simple Moving Average using cumsum optimization.

    Args:
        close: Array of closing prices.
        period: Lookback window size.

    Returns:
        Array with SMA values. First (period-1) elements are NaN.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    cumsum = np.cumsum(close)
    result[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period

    return result


def vema(close: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Compute Exponential Moving Average.

    Args:
        close: Array of closing prices.
        period: Lookback window size. Smoothing factor = 2/(period+1).

    Returns:
        Array with EMA values. First (period-1) elements are NaN.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    alpha = 2.0 / (period + 1)
    result[period - 1] = np.mean(close[:period])

    for i in range(period, n):
        result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]

    return result


def vrsi(close: NDArray[np.float64], period: int = 14) -> NDArray[np.float64]:
    """Compute Relative Strength Index using Wilder's smoothing.

    Args:
        close: Array of closing prices.
        period: RSI lookback period (default: 14).

    Returns:
        Array with RSI values (0-100). First `period` elements are NaN.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avgGain = np.mean(gains[1:period + 1])
    avgLoss = np.mean(losses[1:period + 1])

    if avgLoss == 0:
        result[period] = 100.0
    else:
        rs = avgGain / avgLoss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avgGain = (avgGain * (period - 1) + gains[i]) / period
        avgLoss = (avgLoss * (period - 1) + losses[i]) / period

        if avgLoss == 0:
            result[i] = 100.0
        else:
            rs = avgGain / avgLoss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


def vmacd(
    close: NDArray[np.float64],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute MACD (Moving Average Convergence Divergence).

    Args:
        close: Array of closing prices.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) arrays.
    """
    fastEma = vema(close, fast)
    slowEma = vema(close, slow)

    macdLine = fastEma - slowEma

    n = len(close)
    signalLine = np.full(n, np.nan, dtype=np.float64)

    alpha = 2.0 / (signal + 1)
    startIdx = slow - 1 + signal - 1

    if startIdx < n:
        validMacd = macdLine[slow-1:startIdx+1]
        validMacd = validMacd[~np.isnan(validMacd)]
        if len(validMacd) > 0:
            signalLine[startIdx] = np.mean(validMacd)

        for i in range(startIdx + 1, n):
            if not np.isnan(macdLine[i]) and not np.isnan(signalLine[i - 1]):
                signalLine[i] = alpha * macdLine[i] + (1 - alpha) * signalLine[i - 1]

    histogram = macdLine - signalLine

    return macdLine, signalLine, histogram


def vbollinger(
    close: NDArray[np.float64],
    period: int = 20,
    std: float = 2.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bollinger Bands (upper, middle, lower).

    Args:
        close: Array of closing prices.
        period: SMA period for middle band (default: 20).
        std: Standard deviation multiplier (default: 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band) arrays.
    """
    n = len(close)
    middle = vsma(close, period)

    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        stdDev = np.std(window, ddof=0)
        upper[i] = middle[i] + std * stdDev
        lower[i] = middle[i] - std * stdDev

    return upper, middle, lower


def vatr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """Compute Average True Range using Wilder's smoothing.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        period: ATR lookback period (default: 14).

    Returns:
        Array with ATR values. First (period-1) elements are NaN.
    """
    n = len(close)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)

    atr = np.full(n, np.nan, dtype=np.float64)
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def vstochastic(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    kPeriod: int = 14,
    dPeriod: int = 3
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Stochastic Oscillator (%K and %D lines).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        kPeriod: %K lookback period (default: 14).
        dPeriod: %D smoothing period (default: 3).

    Returns:
        Tuple of (%K, %D) arrays. Values range 0-100.
    """
    n = len(close)
    k = np.full(n, np.nan, dtype=np.float64)

    for i in range(kPeriod - 1, n):
        highestHigh = np.max(high[i - kPeriod + 1:i + 1])
        lowestLow = np.min(low[i - kPeriod + 1:i + 1])

        if highestHigh != lowestLow:
            k[i] = 100.0 * (close[i] - lowestLow) / (highestHigh - lowestLow)
        else:
            k[i] = 50.0

    d = vsma(k, dPeriod)

    return k, d


def vadx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14
) -> NDArray[np.float64]:
    """Compute Average Directional Index (ADX).

    Measures trend strength regardless of direction. Values above 25
    indicate a strong trend, below 20 indicate a weak/no trend.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        period: ADX lookback period (default: 14).

    Returns:
        Array with ADX values (0-100). First (2*period-1) elements are NaN.
    """
    n = len(close)

    upMove = np.diff(high, prepend=high[0])
    downMove = -np.diff(low, prepend=low[0])

    plusDm = np.where((upMove > downMove) & (upMove > 0), upMove, 0)
    minusDm = np.where((downMove > upMove) & (downMove > 0), downMove, 0)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)

    smoothedPlusDm = np.zeros(n, dtype=np.float64)
    smoothedMinusDm = np.zeros(n, dtype=np.float64)
    smoothedTr = np.zeros(n, dtype=np.float64)

    smoothedPlusDm[period] = np.sum(plusDm[1:period + 1])
    smoothedMinusDm[period] = np.sum(minusDm[1:period + 1])
    smoothedTr[period] = np.sum(tr[1:period + 1])

    for i in range(period + 1, n):
        smoothedPlusDm[i] = smoothedPlusDm[i - 1] - smoothedPlusDm[i - 1] / period + plusDm[i]
        smoothedMinusDm[i] = smoothedMinusDm[i - 1] - smoothedMinusDm[i - 1] / period + minusDm[i]
        smoothedTr[i] = smoothedTr[i - 1] - smoothedTr[i - 1] / period + tr[i]

    plusDi = np.zeros(n, dtype=np.float64)
    minusDi = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)

    mask = smoothedTr[period:] != 0
    plusDi[period:][mask] = 100.0 * smoothedPlusDm[period:][mask] / smoothedTr[period:][mask]
    minusDi[period:][mask] = 100.0 * smoothedMinusDm[period:][mask] / smoothedTr[period:][mask]

    diSum = plusDi + minusDi
    diSumMask = diSum != 0
    dx[diSumMask] = 100.0 * np.abs(plusDi[diSumMask] - minusDi[diSumMask]) / diSum[diSumMask]

    adx = np.full(n, np.nan, dtype=np.float64)
    adx[2 * period - 1] = np.mean(dx[period:2 * period])

    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def vroc(close: NDArray[np.float64], period: int = 12) -> NDArray[np.float64]:
    """Compute Rate of Change (percentage price change over N periods).

    Args:
        close: Array of closing prices.
        period: Lookback period (default: 12).

    Returns:
        Array with ROC values in percentage. First `period` elements are NaN.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    prevClose = close[:-period]
    mask = prevClose != 0
    result[period:][mask] = ((close[period:][mask] - prevClose[mask]) / prevClose[mask]) * 100.0

    return result


def vmomentum(close: NDArray[np.float64], period: int = 10) -> NDArray[np.float64]:
    """Compute Price Momentum (absolute price change over N periods).

    Args:
        close: Array of closing prices.
        period: Lookback period (default: 10).

    Returns:
        Array with momentum values. First `period` elements are NaN.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    result[period:] = close[period:] - close[:-period]
    return result
