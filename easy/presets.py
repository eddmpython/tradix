"""
Tradex Preset Strategies Module - Ready-to-use one-liner strategy presets.

Provides factory functions that return pre-configured QuickStrategy instances
for popular trading strategies. Each preset accepts optional parameters for
customization and risk management (stop-loss, take-profit).

프리셋 전략 모듈 - 1줄로 사용 가능한 검증된 트레이딩 전략들을 제공합니다.
각 프리셋은 커스터마이징 및 리스크 관리 파라미터를 선택적으로 지원합니다.

Features:
    - goldenCross: SMA golden/death cross strategy
    - rsiOversold: RSI oversold bounce strategy
    - bollingerBreakout: Bollinger Band breakout strategy
    - macdCross: MACD signal line crossover strategy
    - breakout: Channel breakout (Turtle Trading) strategy
    - meanReversion: Mean reversion with Bollinger Bands
    - trendFollowing: ADX-filtered trend following strategy
    - emaCross: EMA crossover strategy
    - tripleScreen: Alexander Elder's Triple Screen strategy

Usage:
    >>> from tradex.easy import backtest, goldenCross, rsiOversold
    >>>
    >>> result = backtest("005930", goldenCross())
    >>> result = backtest("삼성전자", rsiOversold(period=14, oversold=30))
"""

from tradex.easy.quick import QuickStrategy
from tradex.easy.conditions import (
    sma,
    ema,
    rsi,
    macd,
    macdSignal,
    bollingerUpper,
    bollingerLower,
    bollingerMiddle,
    atr,
    price,
    crossover,
    crossunder,
)


def goldenCross(fast: int = 10, slow: int = 30, stopLoss: float = None, takeProfit: float = None) -> QuickStrategy:
    """
    Golden Cross / Death Cross strategy using SMA crossovers.

    Buys when fast SMA crosses above slow SMA; sells on the reverse.
    단기 SMA가 장기 SMA를 상향 돌파하면 매수, 하향 돌파하면 매도합니다.

    Args:
        fast: Fast SMA period (default 10). / 단기 이동평균 기간.
        slow: Slow SMA period (default 30). / 장기 이동평균 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured golden cross strategy.

    Example:
        >>> result = backtest("005930", goldenCross())
        >>> result = backtest("005930", goldenCross(fast=5, slow=20, stopLoss=5))
    """
    strategy = (
        QuickStrategy(f"골든크로스({fast}/{slow})")
        .buyWhen(crossover(sma(fast), sma(slow)))
        .sellWhen(crossunder(sma(fast), sma(slow)))
    )

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def rsiOversold(
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    RSI oversold bounce strategy.

    Buys when RSI crosses above the oversold level; sells when RSI crosses
    below the overbought level.
    RSI가 과매도 수준을 상향 돌파하면 매수, 과매수 수준을 하향 돌파하면 매도합니다.

    Args:
        period: RSI period (default 14). / RSI 기간.
        oversold: Oversold threshold (default 30). / 과매도 기준.
        overbought: Overbought threshold (default 70). / 과매수 기준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured RSI oversold strategy.

    Example:
        >>> result = backtest("005930", rsiOversold())
        >>> result = backtest("005930", rsiOversold(oversold=25, overbought=75))
    """
    strategy = (
        QuickStrategy(f"RSI과매도({period})")
        .buyWhen(crossover(rsi(period), oversold))
        .sellWhen(crossunder(rsi(period), overbought))
    )

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def bollingerBreakout(
    period: int = 20,
    std: float = 2.0,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Bollinger Band bounce strategy.

    Buys when price crosses above lower band (bounce); sells when price
    exceeds upper band.
    가격이 하단 밴드를 상향 돌파하면 매수, 상단 밴드를 초과하면 매도합니다.

    Args:
        period: Bollinger Band period (default 20). / 볼린저 밴드 기간.
        std: Standard deviation multiplier (default 2.0). / 표준편차 배수.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Bollinger breakout strategy.

    Example:
        >>> result = backtest("005930", bollingerBreakout())
    """
    strategy = (
        QuickStrategy(f"볼린저돌파({period})")
        .buyWhen(crossover(price, bollingerLower(period, std)))
        .sellWhen(price > bollingerUpper(period, std))
    )

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def macdCross(
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    MACD signal line crossover strategy.

    Buys when MACD crosses above signal line; sells on the reverse.
    MACD가 시그널선을 상향 돌파하면 매수, 하향 돌파하면 매도합니다.

    Args:
        fast: Fast EMA period (default 12). / 빠른 EMA 기간.
        slow: Slow EMA period (default 26). / 느린 EMA 기간.
        signal: Signal line period (default 9). / 시그널 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured MACD crossover strategy.

    Example:
        >>> result = backtest("005930", macdCross())
    """
    strategy = (
        QuickStrategy(f"MACD크로스({fast}/{slow}/{signal})")
        .buyWhen(crossover(macd(fast, slow, signal), macdSignal(fast, slow, signal)))
        .sellWhen(crossunder(macd(fast, slow, signal), macdSignal(fast, slow, signal)))
    )

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def breakout(
    period: int = 20,
    stopLoss: float = None,
    takeProfit: float = None,
    trailingStop: float = None
) -> QuickStrategy:
    """
    Channel breakout strategy (Turtle Trading).

    Buys when price breaks above the N-period highest high; sells when
    price drops below the N-period lowest low. Supports trailing stop
    for trend-riding profit protection.
    가격이 N일 최고가를 돌파하면 매수, N일 최저가를 하회하면 매도합니다.
    추적손절로 추세 수익을 보호할 수 있습니다.

    Args:
        period: Lookback period for channel (default 20). / 돌파 기준 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.
        trailingStop: Trailing stop percent (optional). / 추적손절%.

    Returns:
        QuickStrategy: Configured channel breakout strategy.

    Example:
        >>> result = backtest("005930", breakout(period=20))
        >>> result = backtest("005930", breakout(trailingStop=10))
    """
    strategy = QuickStrategy(f"돌파전략({period})")

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < period:
            return False
        highestHigh = s._history['high'].iloc[-period:].max()
        return bar.close > highestHigh

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < period:
            return False
        lowestLow = s._history['low'].iloc[-period:].min()
        return bar.close < lowestLow

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)
    if trailingStop:
        strategy.trailingStop(trailingStop)

    return strategy


def meanReversion(
    period: int = 20,
    threshold: float = 2.0,
    stopLoss: float = 5,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Mean reversion strategy using Bollinger Bands.

    Buys when price drops below the lower Bollinger Band (oversold);
    sells when price rises above the upper band (overbought). Includes
    a default stop-loss for risk management.
    가격이 볼린저 하단 밴드 아래로 하락하면 매수, 상단 밴드 위로 상승하면 매도합니다.
    기본 손절이 포함되어 있습니다.

    Args:
        period: Moving average period (default 20). / 이동평균 기간.
        threshold: Standard deviation multiplier for bands (default 2.0). / 진입 기준 표준편차 배수.
        stopLoss: Stop-loss percent (default 5). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured mean reversion strategy.

    Example:
        >>> result = backtest("005930", meanReversion())
        >>> result = backtest("005930", meanReversion(threshold=1.5, stopLoss=3))
    """
    strategy = QuickStrategy(f"평균회귀({period})")

    def buyCondition(s, bar):
        smaVal = s.sma(period)
        if smaVal is None:
            return False
        upper, _, lower = s.bollinger(period, threshold)
        if lower is None:
            return False
        return bar.close < lower

    def sellCondition(s, bar):
        smaVal = s.sma(period)
        if smaVal is None:
            return False
        upper, _, lower = s.bollinger(period, threshold)
        if upper is None:
            return False
        return bar.close > upper

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)
    strategy.stopLoss(stopLoss)

    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def trendFollowing(
    fastPeriod: int = 10,
    slowPeriod: int = 30,
    adxPeriod: int = 14,
    adxThreshold: float = 25,
    trailingStop: float = 10
) -> QuickStrategy:
    """
    ADX-filtered trend following strategy.

    Buys on a golden cross (fast SMA crosses above slow SMA) only when
    ADX confirms a strong trend (above threshold). Uses trailing stop
    to protect profits during trend reversals.
    ADX가 임계값 이상(강한 추세)이고 골든크로스 발생 시 매수하며,
    추적손절로 수익을 보호합니다.

    Args:
        fastPeriod: Fast SMA period (default 10). / 단기 이동평균 기간.
        slowPeriod: Slow SMA period (default 30). / 장기 이동평균 기간.
        adxPeriod: ADX calculation period (default 14). / ADX 기간.
        adxThreshold: Minimum ADX for trend confirmation (default 25). / ADX 임계값.
        trailingStop: Trailing stop percent from peak (default 10). / 추적손절%.

    Returns:
        QuickStrategy: Configured trend following strategy.

    Example:
        >>> result = backtest("005930", trendFollowing())
        >>> result = backtest("005930", trendFollowing(adxThreshold=30, trailingStop=8))
    """
    strategy = QuickStrategy(f"추세추종({fastPeriod}/{slowPeriod})")

    def buyCondition(s, bar):
        fastSma = s.sma(fastPeriod)
        slowSma = s.sma(slowPeriod)
        prevFast = s.sma(fastPeriod, offset=1)
        prevSlow = s.sma(slowPeriod, offset=1)
        adxVal = s.adx(adxPeriod)

        if None in (fastSma, slowSma, prevFast, prevSlow, adxVal):
            return False

        goldenCross = prevFast <= prevSlow and fastSma > slowSma
        strongTrend = adxVal >= adxThreshold

        return goldenCross and strongTrend

    def sellCondition(s, bar):
        fastSma = s.sma(fastPeriod)
        slowSma = s.sma(slowPeriod)
        prevFast = s.sma(fastPeriod, offset=1)
        prevSlow = s.sma(slowPeriod, offset=1)

        if None in (fastSma, slowSma, prevFast, prevSlow):
            return False

        deadCross = prevFast >= prevSlow and fastSma < slowSma
        return deadCross

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)
    strategy.trailingStop(trailingStop)

    return strategy


def emaCross(fast: int = 12, slow: int = 26, stopLoss: float = None, takeProfit: float = None) -> QuickStrategy:
    """
    EMA crossover strategy.

    Buys when fast EMA crosses above slow EMA; sells on the reverse.
    Similar to goldenCross but uses EMA for faster trend response.
    단기 EMA가 장기 EMA를 상향 돌파하면 매수, 하향 돌파하면 매도합니다.

    Args:
        fast: Fast EMA period (default 12). / 단기 EMA 기간.
        slow: Slow EMA period (default 26). / 장기 EMA 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured EMA crossover strategy.

    Example:
        >>> result = backtest("005930", emaCross())
        >>> result = backtest("005930", emaCross(fast=8, slow=21, stopLoss=5))
    """
    strategy = (
        QuickStrategy(f"EMA크로스({fast}/{slow})")
        .buyWhen(crossover(ema(fast), ema(slow)))
        .sellWhen(crossunder(ema(fast), ema(slow)))
    )

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def tripleScreen(
    longPeriod: int = 50,
    mediumPeriod: int = 20,
    rsiPeriod: int = 14,
    rsiOversold: float = 30,
    stopLoss: float = 5
) -> QuickStrategy:
    """
    Alexander Elder's Triple Screen trading strategy.

    Combines three timeframe filters for high-probability entries:
    1. Long-term trend confirmation (price above long SMA = uptrend)
    2. Medium-term oscillator (RSI oversold = pullback opportunity)
    3. Short-term entry (price above medium SMA = bounce confirmed)
    Sells when downtrend detected or RSI reaches overbought levels.
    알렉산더 엘더의 삼중 스크린 전략입니다. 장기 추세, 중기 RSI 과매도,
    단기 반등을 조합하여 높은 확률의 진입 시점을 찾습니다.

    Args:
        longPeriod: Long-term trend SMA period (default 50). / 장기 추세 확인 기간.
        mediumPeriod: Medium-term SMA period (default 20). / 중기 기간.
        rsiPeriod: RSI calculation period (default 14). / RSI 기간.
        rsiOversold: RSI oversold level for entry (default 30). / RSI 과매도 수준.
        stopLoss: Stop-loss percent (default 5). / 손절%.

    Returns:
        QuickStrategy: Configured triple screen strategy.

    Example:
        >>> result = backtest("005930", tripleScreen())
        >>> result = backtest("005930", tripleScreen(longPeriod=60, rsiOversold=25))
    """
    strategy = QuickStrategy(f"삼중스크린({longPeriod}/{mediumPeriod})")

    def buyCondition(s, bar):
        longSma = s.sma(longPeriod)
        medSma = s.sma(mediumPeriod)
        rsiVal = s.rsi(rsiPeriod)

        if None in (longSma, medSma, rsiVal):
            return False

        uptrend = bar.close > longSma
        oversold = rsiVal < rsiOversold
        aboveMedium = bar.close > medSma

        return uptrend and oversold and aboveMedium

    def sellCondition(s, bar):
        longSma = s.sma(longPeriod)
        rsiVal = s.rsi(rsiPeriod)

        if None in (longSma, rsiVal):
            return False

        downtrend = bar.close < longSma
        overbought = rsiVal > 70

        return downtrend or overbought

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)
    strategy.stopLoss(stopLoss)

    return strategy
