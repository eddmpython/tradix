"""
Vectorized Signal Generators - Pure NumPy implementation.

Trading signal generators that operate on entire price arrays at once.
Each function returns an int8 array where 1 = buy, -1 = sell, 0 = no signal.

Available Signals:
    - vcrossover: Detect upward crossovers between two series
    - vcrossunder: Detect downward crossovers between two series
    - vcross: Combined crossover/crossunder detection
    - vgoldenCross: SMA golden cross / death cross signals
    - vrsiSignal: RSI oversold/overbought reversal signals
    - vmacdSignal: MACD line / signal line crossover signals
    - vbollingerSignal: Bollinger Band breakout signals
    - vbreakoutSignal: N-period channel breakout signals (Turtle Trading)
    - vTrendFilter: ADX-based trend filter for existing signals

Usage:
    >>> from tradex.vectorized.signals import vgoldenCross, vrsiSignal
    >>> signals = vgoldenCross(close, fast=10, slow=30)
    >>> buy_dates = np.where(signals == 1)[0]
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from tradex.vectorized.indicators import vsma, vmacd, vbollinger


def vcrossover(
    fast: NDArray[np.float64],
    slow: NDArray[np.float64]
) -> NDArray[np.int8]:
    """Detect upward crossovers where fast crosses above slow.

    Args:
        fast: Fast indicator array.
        slow: Slow indicator array.

    Returns:
        Int8 array: 1 at crossover points, 0 elsewhere.
    """
    n = len(fast)
    signals = np.zeros(n, dtype=np.int8)

    prevFast = np.roll(fast, 1)
    prevSlow = np.roll(slow, 1)

    crossover = (prevFast <= prevSlow) & (fast > slow)
    crossover[0] = False
    crossover = crossover & ~np.isnan(fast) & ~np.isnan(slow) & ~np.isnan(prevFast) & ~np.isnan(prevSlow)

    signals[crossover] = 1

    return signals


def vcrossunder(
    fast: NDArray[np.float64],
    slow: NDArray[np.float64]
) -> NDArray[np.int8]:
    """Detect downward crossovers where fast crosses below slow.

    Args:
        fast: Fast indicator array.
        slow: Slow indicator array.

    Returns:
        Int8 array: -1 at crossunder points, 0 elsewhere.
    """
    n = len(fast)
    signals = np.zeros(n, dtype=np.int8)

    prevFast = np.roll(fast, 1)
    prevSlow = np.roll(slow, 1)

    crossunder = (prevFast >= prevSlow) & (fast < slow)
    crossunder[0] = False
    crossunder = crossunder & ~np.isnan(fast) & ~np.isnan(slow) & ~np.isnan(prevFast) & ~np.isnan(prevSlow)

    signals[crossunder] = -1

    return signals


def vcross(
    fast: NDArray[np.float64],
    slow: NDArray[np.float64]
) -> NDArray[np.int8]:
    """Detect both crossovers (+1) and crossunders (-1) in one pass.

    Args:
        fast: Fast indicator array.
        slow: Slow indicator array.

    Returns:
        Int8 array: 1 = crossover, -1 = crossunder, 0 = no signal.
    """
    n = len(fast)
    signals = np.zeros(n, dtype=np.int8)

    prevFast = np.roll(fast, 1)
    prevSlow = np.roll(slow, 1)

    valid = ~np.isnan(fast) & ~np.isnan(slow) & ~np.isnan(prevFast) & ~np.isnan(prevSlow)
    valid[0] = False

    crossover = valid & (prevFast <= prevSlow) & (fast > slow)
    crossunder = valid & (prevFast >= prevSlow) & (fast < slow)

    signals[crossover] = 1
    signals[crossunder] = -1

    return signals


def vgoldenCross(
    close: NDArray[np.float64],
    fast: int = 10,
    slow: int = 30
) -> NDArray[np.int8]:
    """Generate Golden Cross (buy) / Death Cross (sell) signals from SMA crossovers.

    Args:
        close: Array of closing prices.
        fast: Fast SMA period (default: 10).
        slow: Slow SMA period (default: 30).

    Returns:
        Int8 array: 1 = golden cross, -1 = death cross, 0 = no signal.
    """
    fastSma = vsma(close, fast)
    slowSma = vsma(close, slow)
    return vcross(fastSma, slowSma)


def vrsiSignal(
    rsi: NDArray[np.float64],
    oversold: float = 30.0,
    overbought: float = 70.0
) -> NDArray[np.int8]:
    """Generate buy/sell signals from RSI oversold/overbought levels.

    Buy when RSI crosses above oversold level, sell when crosses below overbought.

    Args:
        rsi: Pre-computed RSI array.
        oversold: Oversold threshold (default: 30.0).
        overbought: Overbought threshold (default: 70.0).

    Returns:
        Int8 array: 1 = buy (oversold recovery), -1 = sell (overbought reversal).
    """
    n = len(rsi)
    signals = np.zeros(n, dtype=np.int8)

    prevRsi = np.roll(rsi, 1)

    valid = ~np.isnan(rsi) & ~np.isnan(prevRsi)
    valid[0] = False

    buySignal = valid & (prevRsi <= oversold) & (rsi > oversold)
    sellSignal = valid & (prevRsi >= overbought) & (rsi < overbought)

    signals[buySignal] = 1
    signals[sellSignal] = -1

    return signals


def vmacdSignal(
    close: NDArray[np.float64],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> NDArray[np.int8]:
    """Generate buy/sell signals from MACD / signal line crossovers.

    Args:
        close: Array of closing prices.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        Int8 array: 1 = MACD crosses above signal, -1 = below.
    """
    macdLine, signalLine, _ = vmacd(close, fast, slow, signal)
    return vcross(macdLine, signalLine)


def vbollingerSignal(
    close: NDArray[np.float64],
    period: int = 20,
    std: float = 2.0
) -> NDArray[np.int8]:
    """Generate buy/sell signals from Bollinger Band breakouts.

    Buy when price rebounds from below lower band, sell when price hits upper band.

    Args:
        close: Array of closing prices.
        period: Bollinger Band period (default: 20).
        std: Standard deviation multiplier (default: 2.0).

    Returns:
        Int8 array: 1 = lower band rebound, -1 = upper band hit.
    """
    upper, middle, lower = vbollinger(close, period, std)

    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    prevClose = np.roll(close, 1)
    prevLower = np.roll(lower, 1)

    valid = ~np.isnan(upper) & ~np.isnan(lower)
    valid[0] = False

    buySignal = valid & (prevClose <= prevLower) & (close > lower)
    sellSignal = valid & (close >= upper)

    signals[buySignal] = 1
    signals[sellSignal] = -1

    return signals


def vbreakoutSignal(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 20
) -> NDArray[np.int8]:
    """Generate channel breakout signals (Turtle Trading style).

    Buy when price breaks above N-period highest high,
    sell when price breaks below N-period lowest low.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        period: Lookback period for channel (default: 20).

    Returns:
        Int8 array: 1 = upside breakout, -1 = downside breakout.
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    for i in range(period, n):
        highestHigh = np.max(high[i - period:i])
        lowestLow = np.min(low[i - period:i])

        if close[i] > highestHigh:
            signals[i] = 1
        elif close[i] < lowestLow:
            signals[i] = -1

    return signals


def vTrendFilter(
    close: NDArray[np.float64],
    sma: NDArray[np.float64],
    adx: NDArray[np.float64],
    signals: NDArray[np.int8],
    adxThreshold: float = 25.0
) -> NDArray[np.int8]:
    """Filter existing signals by ADX trend strength and price position.

    Only passes buy signals when ADX >= threshold and price > SMA (uptrend),
    and sell signals when ADX >= threshold and price < SMA (downtrend).

    Args:
        close: Array of closing prices.
        sma: Pre-computed SMA array for trend direction.
        adx: Pre-computed ADX array for trend strength.
        signals: Existing signal array to filter.
        adxThreshold: Minimum ADX for strong trend (default: 25.0).

    Returns:
        Int8 array: Filtered signals (weak trends removed).
    """
    n = len(close)
    filtered = np.zeros(n, dtype=np.int8)

    valid = ~np.isnan(sma) & ~np.isnan(adx)
    strongTrend = valid & (adx >= adxThreshold)

    buyFilter = strongTrend & (signals == 1) & (close > sma)
    sellFilter = strongTrend & (signals == -1) & (close < sma)

    filtered[buyFilter] = 1
    filtered[sellFilter] = -1

    return filtered
