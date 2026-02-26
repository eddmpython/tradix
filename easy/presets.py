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
    - dualMomentum: Dual Momentum (absolute + relative) strategy
    - momentumCross: Momentum crossover strategy
    - rocBreakout: Rate of Change breakout strategy
    - stochasticCross: Stochastic %K/%D crossover strategy
    - williamsReversal: Williams %R reversal strategy
    - cciBreakout: CCI breakout strategy
    - rsiDivergence: RSI oversold with momentum confirmation strategy
    - volatilityBreakout: ATR-based volatility breakout strategy
    - keltnerChannel: Keltner Channel strategy
    - bollingerSqueeze: Bollinger Band squeeze breakout strategy
    - superTrend: SuperTrend indicator strategy
    - ichimokuCloud: Ichimoku Cloud strategy
    - parabolicSar: Parabolic SAR trend reversal strategy
    - donchianBreakout: Donchian Channel breakout (Turtle Trading variant)
    - tripleEma: Triple EMA crossover strategy
    - macdRsiCombo: MACD + RSI combination strategy
    - trendMomentum: Trend + momentum combination strategy
    - bollingerRsi: Bollinger Band + RSI filter strategy
    - gapTrading: Gap trading strategy
    - pyramiding: Pyramid entry on multi-period confirmation strategy
    - swingTrading: Swing trading with ATR stops strategy
    - scalpingMomentum: Fast scalping momentum strategy
    - buyAndHold: Simple buy and hold benchmark strategy
    - dollarCostAverage: Dollar Cost Averaging (DCA) strategy

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


def dualMomentum(
    lookback: int = 252,
    holdPeriod: int = 21,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Dual Momentum strategy (absolute + relative momentum).

    Buys when both absolute momentum (positive return over lookback) and
    relative momentum (rate of change above zero) confirm upward trend.
    Sells after the hold period expires or momentum turns negative.
    절대 모멘텀(양의 수익률)과 상대 모멘텀(변화율)이 동시에 상승 추세를
    확인하면 매수합니다. 보유 기간 만료 또는 모멘텀 전환 시 매도합니다.

    Args:
        lookback: Lookback period for momentum (default 252). / 모멘텀 산출 기간.
        holdPeriod: Hold period in bars after entry (default 21). / 보유 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured dual momentum strategy.

    Example:
        >>> result = backtest("005930", dualMomentum())
        >>> result = backtest("005930", dualMomentum(lookback=126, holdPeriod=10))
    """
    strategy = QuickStrategy(f"듀얼모멘텀({lookback}/{holdPeriod})")

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < lookback:
            return False
        pastClose = s._history['close'].iloc[-lookback]
        absoluteMomentum = bar.close > pastClose
        rocVal = s.roc(lookback)
        if rocVal is None:
            return False
        relativeMomentum = rocVal > 0
        return absoluteMomentum and relativeMomentum

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < holdPeriod:
            return False
        rocVal = s.roc(lookback)
        if rocVal is None:
            return False
        return rocVal < 0

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def momentumCross(
    fast: int = 10,
    slow: int = 30,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Momentum crossover strategy.

    Buys when fast-period momentum crosses above slow-period momentum,
    indicating accelerating price movement. Sells on the reverse crossunder.
    단기 모멘텀이 장기 모멘텀을 상향 돌파하면 가속 구간으로 판단하여 매수,
    하향 돌파 시 매도합니다.

    Args:
        fast: Fast momentum period (default 10). / 단기 모멘텀 기간.
        slow: Slow momentum period (default 30). / 장기 모멘텀 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured momentum crossover strategy.

    Example:
        >>> result = backtest("005930", momentumCross())
        >>> result = backtest("005930", momentumCross(fast=5, slow=20))
    """
    strategy = QuickStrategy(f"모멘텀크로스({fast}/{slow})")

    def buyCondition(s, bar):
        fastMom = s.momentum(fast)
        slowMom = s.momentum(slow)
        prevFastMom = s.momentum(fast, offset=1)
        prevSlowMom = s.momentum(slow, offset=1)
        if None in (fastMom, slowMom, prevFastMom, prevSlowMom):
            return False
        return prevFastMom <= prevSlowMom and fastMom > slowMom

    def sellCondition(s, bar):
        fastMom = s.momentum(fast)
        slowMom = s.momentum(slow)
        prevFastMom = s.momentum(fast, offset=1)
        prevSlowMom = s.momentum(slow, offset=1)
        if None in (fastMom, slowMom, prevFastMom, prevSlowMom):
            return False
        return prevFastMom >= prevSlowMom and fastMom < slowMom

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def rocBreakout(
    period: int = 14,
    threshold: float = 5,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Rate of Change (ROC) breakout strategy.

    Buys when ROC exceeds the positive threshold (strong upward momentum);
    sells when ROC drops below the negative threshold (strong downward momentum).
    ROC가 양의 임계값을 초과하면 강한 상승 모멘텀으로 매수,
    음의 임계값 이하로 하락하면 매도합니다.

    Args:
        period: ROC calculation period (default 14). / ROC 기간.
        threshold: Breakout threshold in percent (default 5). / 돌파 임계값(%).
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured ROC breakout strategy.

    Example:
        >>> result = backtest("005930", rocBreakout())
        >>> result = backtest("005930", rocBreakout(period=20, threshold=3))
    """
    strategy = QuickStrategy(f"ROC돌파({period}/{threshold})")

    def buyCondition(s, bar):
        rocVal = s.roc(period)
        if rocVal is None:
            return False
        return rocVal > threshold

    def sellCondition(s, bar):
        rocVal = s.roc(period)
        if rocVal is None:
            return False
        return rocVal < -threshold

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def stochasticCross(
    kPeriod: int = 14,
    dPeriod: int = 3,
    oversold: float = 20,
    overbought: float = 80,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Stochastic %K/%D crossover strategy.

    Buys when %K crosses above %D in the oversold zone (below oversold level);
    sells when %K crosses below %D in the overbought zone (above overbought level).
    과매도 구간에서 %K가 %D를 상향 돌파하면 매수,
    과매수 구간에서 %K가 %D를 하향 돌파하면 매도합니다.

    Args:
        kPeriod: %K period (default 14). / %K 기간.
        dPeriod: %D period (default 3). / %D 기간.
        oversold: Oversold level (default 20). / 과매도 수준.
        overbought: Overbought level (default 80). / 과매수 수준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured stochastic crossover strategy.

    Example:
        >>> result = backtest("005930", stochasticCross())
        >>> result = backtest("005930", stochasticCross(kPeriod=9, dPeriod=3))
    """
    strategy = QuickStrategy(f"스토캐스틱크로스({kPeriod}/{dPeriod})")

    def buyCondition(s, bar):
        result = s.stochastic(kPeriod, dPeriod)
        prevResult = s.stochastic(kPeriod, dPeriod, offset=1)
        if result is None or prevResult is None:
            return False
        k, d = result
        prevK, prevD = prevResult
        if None in (k, d, prevK, prevD):
            return False
        return prevK <= prevD and k > d and k < oversold

    def sellCondition(s, bar):
        result = s.stochastic(kPeriod, dPeriod)
        prevResult = s.stochastic(kPeriod, dPeriod, offset=1)
        if result is None or prevResult is None:
            return False
        k, d = result
        prevK, prevD = prevResult
        if None in (k, d, prevK, prevD):
            return False
        return prevK >= prevD and k < d and k > overbought

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def williamsReversal(
    period: int = 14,
    oversold: float = -80,
    overbought: float = -20,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Williams %R reversal strategy.

    Buys when Williams %R crosses above the oversold level (reversal from
    extreme low); sells when it crosses below the overbought level.
    Williams %R이 과매도 수준을 상향 돌파하면 반전 매수,
    과매수 수준을 하향 돌파하면 매도합니다.

    Args:
        period: Williams %R period (default 14). / Williams %R 기간.
        oversold: Oversold level (default -80). / 과매도 수준.
        overbought: Overbought level (default -20). / 과매수 수준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Williams %R reversal strategy.

    Example:
        >>> result = backtest("005930", williamsReversal())
        >>> result = backtest("005930", williamsReversal(oversold=-85, overbought=-15))
    """
    strategy = QuickStrategy(f"윌리엄스%R반전({period})")

    def buyCondition(s, bar):
        wrVal = s.williamsR(period)
        prevWr = s.williamsR(period, offset=1)
        if None in (wrVal, prevWr):
            return False
        return prevWr <= oversold and wrVal > oversold

    def sellCondition(s, bar):
        wrVal = s.williamsR(period)
        prevWr = s.williamsR(period, offset=1)
        if None in (wrVal, prevWr):
            return False
        return prevWr >= overbought and wrVal < overbought

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def cciBreakout(
    period: int = 20,
    threshold: float = 100,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Commodity Channel Index (CCI) breakout strategy.

    Buys when CCI crosses above the positive threshold (strong bullish
    momentum); sells when CCI crosses below the negative threshold.
    CCI가 양의 임계값을 상향 돌파하면 강한 상승 신호로 매수,
    음의 임계값을 하향 돌파하면 매도합니다.

    Args:
        period: CCI period (default 20). / CCI 기간.
        threshold: Breakout threshold (default 100). / 돌파 임계값.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured CCI breakout strategy.

    Example:
        >>> result = backtest("005930", cciBreakout())
        >>> result = backtest("005930", cciBreakout(period=14, threshold=150))
    """
    strategy = QuickStrategy(f"CCI돌파({period}/{threshold})")

    def buyCondition(s, bar):
        cciVal = s.cci(period)
        prevCci = s.cci(period, offset=1)
        if None in (cciVal, prevCci):
            return False
        return prevCci <= threshold and cciVal > threshold

    def sellCondition(s, bar):
        cciVal = s.cci(period)
        prevCci = s.cci(period, offset=1)
        if None in (cciVal, prevCci):
            return False
        return prevCci >= -threshold and cciVal < -threshold

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def rsiDivergence(
    period: int = 14,
    oversold: float = 30,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    RSI oversold with momentum confirmation strategy.

    Buys when RSI enters oversold territory and positive momentum confirms
    a potential reversal (momentum turns from negative to positive). Sells
    when RSI reaches overbought or momentum turns decisively negative.
    RSI가 과매도 구간에 진입하고 모멘텀이 양전환하여 반전을 확인하면 매수,
    RSI 과매수 또는 모멘텀 음전환 시 매도합니다.

    Args:
        period: RSI period (default 14). / RSI 기간.
        oversold: RSI oversold level (default 30). / RSI 과매도 수준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured RSI divergence strategy.

    Example:
        >>> result = backtest("005930", rsiDivergence())
        >>> result = backtest("005930", rsiDivergence(period=10, oversold=25))
    """
    strategy = QuickStrategy(f"RSI다이버전스({period})")

    def buyCondition(s, bar):
        rsiVal = s.rsi(period)
        momVal = s.momentum(10)
        prevMom = s.momentum(10, offset=1)
        if None in (rsiVal, momVal, prevMom):
            return False
        return rsiVal < oversold and prevMom < 0 and momVal > 0

    def sellCondition(s, bar):
        rsiVal = s.rsi(period)
        momVal = s.momentum(10)
        if None in (rsiVal, momVal):
            return False
        return rsiVal > (100 - oversold) or momVal < 0

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def volatilityBreakout(
    atrPeriod: int = 14,
    multiplier: float = 2.0,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    ATR-based volatility breakout strategy.

    Buys when price closes above the previous close plus ATR multiplied by
    the multiplier (Larry Williams-style breakout). Sells when price drops
    below the previous close minus ATR times multiplier.
    종가가 전일 종가 + ATR x 배수를 초과하면 매수 (래리 윌리엄스 스타일),
    전일 종가 - ATR x 배수 미만이면 매도합니다.

    Args:
        atrPeriod: ATR period (default 14). / ATR 기간.
        multiplier: ATR multiplier for breakout threshold (default 2.0). / ATR 배수.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured volatility breakout strategy.

    Example:
        >>> result = backtest("005930", volatilityBreakout())
        >>> result = backtest("005930", volatilityBreakout(multiplier=1.5))
    """
    strategy = QuickStrategy(f"변동성돌파({atrPeriod}x{multiplier})")

    def buyCondition(s, bar):
        atrVal = s.atr(atrPeriod)
        if atrVal is None or s._history is None or len(s._history) < 2:
            return False
        prevClose = s._history['close'].iloc[-1]
        return bar.close > prevClose + atrVal * multiplier

    def sellCondition(s, bar):
        atrVal = s.atr(atrPeriod)
        if atrVal is None or s._history is None or len(s._history) < 2:
            return False
        prevClose = s._history['close'].iloc[-1]
        return bar.close < prevClose - atrVal * multiplier

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def keltnerChannel(
    period: int = 20,
    multiplier: float = 2.0,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Keltner Channel strategy.

    Uses EMA as the center line and ATR for channel width. Buys when price
    closes above the upper channel (EMA + ATR x multiplier); sells when
    price drops below the lower channel (EMA - ATR x multiplier).
    EMA를 중심선, ATR을 채널 폭으로 사용합니다. 상단 채널 돌파 시 매수,
    하단 채널 이탈 시 매도합니다.

    Args:
        period: EMA and ATR period (default 20). / EMA 및 ATR 기간.
        multiplier: ATR multiplier for channel width (default 2.0). / ATR 배수.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Keltner Channel strategy.

    Example:
        >>> result = backtest("005930", keltnerChannel())
        >>> result = backtest("005930", keltnerChannel(period=15, multiplier=1.5))
    """
    strategy = QuickStrategy(f"켈트너채널({period}x{multiplier})")

    def buyCondition(s, bar):
        emaVal = s.ema(period)
        atrVal = s.atr(period)
        if None in (emaVal, atrVal):
            return False
        upperChannel = emaVal + atrVal * multiplier
        return bar.close > upperChannel

    def sellCondition(s, bar):
        emaVal = s.ema(period)
        atrVal = s.atr(period)
        if None in (emaVal, atrVal):
            return False
        lowerChannel = emaVal - atrVal * multiplier
        return bar.close < lowerChannel

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def bollingerSqueeze(
    period: int = 20,
    std: float = 2.0,
    squeezeThreshold: float = 0.04,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Bollinger Band squeeze breakout strategy.

    Detects low-volatility squeeze (narrow bandwidth) and enters when price
    breaks out above the upper band after the squeeze. Sells when price
    drops below the middle band.
    볼린저 밴드가 좁아진 스퀴즈 상태에서 상단 밴드 돌파 시 매수,
    중간 밴드 이하로 하락 시 매도합니다.

    Args:
        period: Bollinger Band period (default 20). / 볼린저 밴드 기간.
        std: Standard deviation multiplier (default 2.0). / 표준편차 배수.
        squeezeThreshold: Bandwidth threshold for squeeze detection (default 0.04). / 스퀴즈 감지 대역폭 임계값.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Bollinger squeeze strategy.

    Example:
        >>> result = backtest("005930", bollingerSqueeze())
        >>> result = backtest("005930", bollingerSqueeze(squeezeThreshold=0.03))
    """
    strategy = QuickStrategy(f"볼린저스퀴즈({period})")

    def buyCondition(s, bar):
        result = s.bollinger(period, std)
        if result is None:
            return False
        upper, middle, lower = result
        if None in (upper, middle, lower) or middle == 0:
            return False
        bandwidth = (upper - lower) / middle
        return bandwidth < squeezeThreshold and bar.close > upper

    def sellCondition(s, bar):
        result = s.bollinger(period, std)
        if result is None:
            return False
        upper, middle, lower = result
        if middle is None:
            return False
        return bar.close < middle

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def superTrend(
    atrPeriod: int = 10,
    multiplier: float = 3.0,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    SuperTrend indicator strategy.

    Calculates dynamic support/resistance using ATR. Buys when price
    crosses above the SuperTrend line (uptrend); sells when price crosses
    below (downtrend). The SuperTrend line acts as a trailing stop level.
    ATR 기반 동적 지지/저항선을 산출합니다. 가격이 슈퍼트렌드 선을 상향 돌파하면
    매수, 하향 돌파하면 매도합니다.

    Args:
        atrPeriod: ATR period (default 10). / ATR 기간.
        multiplier: ATR multiplier (default 3.0). / ATR 배수.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured SuperTrend strategy.

    Example:
        >>> result = backtest("005930", superTrend())
        >>> result = backtest("005930", superTrend(atrPeriod=14, multiplier=2.0))
    """
    strategy = QuickStrategy(f"슈퍼트렌드({atrPeriod}x{multiplier})")

    def buyCondition(s, bar):
        atrVal = s.atr(atrPeriod)
        prevAtr = s.atr(atrPeriod, offset=1)
        if None in (atrVal, prevAtr) or s._history is None or len(s._history) < 2:
            return False
        hl2 = (bar.high + bar.low) / 2
        prevHl2 = (s._history['high'].iloc[-1] + s._history['low'].iloc[-1]) / 2
        upperBand = hl2 + multiplier * atrVal
        lowerBand = hl2 - multiplier * atrVal
        prevLowerBand = prevHl2 - multiplier * prevAtr
        prevClose = s._history['close'].iloc[-1]
        return prevClose <= prevLowerBand and bar.close > lowerBand

    def sellCondition(s, bar):
        atrVal = s.atr(atrPeriod)
        prevAtr = s.atr(atrPeriod, offset=1)
        if None in (atrVal, prevAtr) or s._history is None or len(s._history) < 2:
            return False
        hl2 = (bar.high + bar.low) / 2
        prevHl2 = (s._history['high'].iloc[-1] + s._history['low'].iloc[-1]) / 2
        upperBand = hl2 + multiplier * atrVal
        prevUpperBand = prevHl2 + multiplier * prevAtr
        prevClose = s._history['close'].iloc[-1]
        return prevClose >= prevUpperBand and bar.close < upperBand

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def ichimokuCloud(
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Ichimoku Cloud strategy.

    Calculates Tenkan-sen (conversion line) and Kijun-sen (base line) from
    highest high and lowest low. Buys when Tenkan crosses above Kijun and
    price is above the cloud (Senkou Span). Sells on the reverse signal.
    전환선(텐칸센)이 기준선(키준센)을 상향 돌파하고 가격이 구름대 위에 있으면
    매수, 반대 신호 시 매도합니다.

    Args:
        tenkan: Tenkan-sen (conversion line) period (default 9). / 전환선 기간.
        kijun: Kijun-sen (base line) period (default 26). / 기준선 기간.
        senkou: Senkou Span B period (default 52). / 선행스팬B 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Ichimoku Cloud strategy.

    Example:
        >>> result = backtest("005930", ichimokuCloud())
        >>> result = backtest("005930", ichimokuCloud(tenkan=7, kijun=22))
    """
    strategy = QuickStrategy(f"일목균형표({tenkan}/{kijun}/{senkou})")

    def _calcMidpoint(history, period):
        if len(history) < period:
            return None
        highVal = history['high'].iloc[-period:].max()
        lowVal = history['low'].iloc[-period:].min()
        return (highVal + lowVal) / 2

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < senkou:
            return False
        tenkanSen = _calcMidpoint(s._history, tenkan)
        kijunSen = _calcMidpoint(s._history, kijun)
        senkouB = _calcMidpoint(s._history, senkou)
        if None in (tenkanSen, kijunSen, senkouB):
            return False
        senkouA = (tenkanSen + kijunSen) / 2
        cloudTop = max(senkouA, senkouB)
        prevHistory = s._history.iloc[:-1]
        if len(prevHistory) < kijun:
            return False
        prevTenkan = _calcMidpoint(prevHistory, tenkan)
        prevKijun = _calcMidpoint(prevHistory, kijun)
        if None in (prevTenkan, prevKijun):
            return False
        return prevTenkan <= prevKijun and tenkanSen > kijunSen and bar.close > cloudTop

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < senkou:
            return False
        tenkanSen = _calcMidpoint(s._history, tenkan)
        kijunSen = _calcMidpoint(s._history, kijun)
        senkouB = _calcMidpoint(s._history, senkou)
        if None in (tenkanSen, kijunSen, senkouB):
            return False
        senkouA = (tenkanSen + kijunSen) / 2
        cloudBottom = min(senkouA, senkouB)
        return tenkanSen < kijunSen or bar.close < cloudBottom

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def parabolicSar(
    step: float = 0.02,
    maxStep: float = 0.2,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Parabolic SAR trend reversal strategy.

    Tracks the Parabolic Stop and Reverse (SAR) indicator. Buys when price
    crosses above the SAR (trend turns bullish); sells when price crosses
    below the SAR (trend turns bearish).
    가격이 파라볼릭 SAR을 상향 돌파하면 상승 추세 전환으로 매수,
    하향 돌파하면 매도합니다.

    Args:
        step: Acceleration factor step (default 0.02). / 가속 계수 단위.
        maxStep: Maximum acceleration factor (default 0.2). / 최대 가속 계수.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Parabolic SAR strategy.

    Example:
        >>> result = backtest("005930", parabolicSar())
        >>> result = backtest("005930", parabolicSar(step=0.01, maxStep=0.1))
    """
    strategy = QuickStrategy(f"파라볼릭SAR({step}/{maxStep})")

    def _calcSar(history):
        if len(history) < 3:
            return None, None
        highs = history['high'].values
        lows = history['low'].values
        closes = history['close'].values
        isUptrend = closes[-2] > closes[-3]
        af = step
        if isUptrend:
            sar = lows[-3]
            ep = highs[-2]
        else:
            sar = highs[-3]
            ep = lows[-2]
        for i in range(len(history) - 2, len(history)):
            if isUptrend:
                sar = sar + af * (ep - sar)
                sar = min(sar, lows[i - 1], lows[i - 2] if i >= 2 else lows[i - 1])
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step, maxStep)
                if lows[i] < sar:
                    isUptrend = False
                    sar = ep
                    ep = lows[i]
                    af = step
            else:
                sar = sar + af * (ep - sar)
                sar = max(sar, highs[i - 1], highs[i - 2] if i >= 2 else highs[i - 1])
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step, maxStep)
                if highs[i] > sar:
                    isUptrend = True
                    sar = ep
                    ep = highs[i]
                    af = step
        return sar, isUptrend

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < 5:
            return False
        sar, isUptrend = _calcSar(s._history)
        if sar is None:
            return False
        return isUptrend and bar.close > sar

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < 5:
            return False
        sar, isUptrend = _calcSar(s._history)
        if sar is None:
            return False
        return not isUptrend and bar.close < sar

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def donchianBreakout(
    period: int = 20,
    exitPeriod: int = 10,
    stopLoss: float = None,
    takeProfit: float = None,
    trailingStop: float = None
) -> QuickStrategy:
    """
    Donchian Channel breakout strategy (Turtle Trading variant).

    Buys when price breaks above the N-period highest high (entry channel);
    sells when price breaks below the exit-period lowest low (shorter
    exit channel). Uses asymmetric entry/exit channels for trend riding.
    N일 최고가를 돌파하면 매수, 청산 기간의 최저가를 이탈하면 매도합니다.
    비대칭 진입/청산 채널로 추세를 최대한 추종합니다.

    Args:
        period: Entry channel period (default 20). / 진입 채널 기간.
        exitPeriod: Exit channel period (default 10). / 청산 채널 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.
        trailingStop: Trailing stop percent (optional). / 추적손절%.

    Returns:
        QuickStrategy: Configured Donchian breakout strategy.

    Example:
        >>> result = backtest("005930", donchianBreakout())
        >>> result = backtest("005930", donchianBreakout(period=55, exitPeriod=20))
    """
    strategy = QuickStrategy(f"돈치안돌파({period}/{exitPeriod})")

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < period:
            return False
        highestHigh = s._history['high'].iloc[-period:].max()
        return bar.close > highestHigh

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < exitPeriod:
            return False
        lowestLow = s._history['low'].iloc[-exitPeriod:].min()
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


def tripleEma(
    fast: int = 5,
    medium: int = 13,
    slow: int = 34,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Triple EMA crossover strategy.

    Uses three EMAs (fast, medium, slow) for high-confidence trend signals.
    Buys when all three EMAs are aligned bullishly (fast > medium > slow)
    and a fresh crossover occurs. Sells on bearish realignment.
    세 개의 EMA(단기, 중기, 장기)가 정배열(단기>중기>장기)이고 신규 크로스오버
    발생 시 매수, 역배열 전환 시 매도합니다.

    Args:
        fast: Fast EMA period (default 5). / 단기 EMA 기간.
        medium: Medium EMA period (default 13). / 중기 EMA 기간.
        slow: Slow EMA period (default 34). / 장기 EMA 기간.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured triple EMA crossover strategy.

    Example:
        >>> result = backtest("005930", tripleEma())
        >>> result = backtest("005930", tripleEma(fast=8, medium=21, slow=55))
    """
    strategy = QuickStrategy(f"삼중EMA({fast}/{medium}/{slow})")

    def buyCondition(s, bar):
        fastEma = s.ema(fast)
        medEma = s.ema(medium)
        slowEma = s.ema(slow)
        prevFast = s.ema(fast, offset=1)
        prevMed = s.ema(medium, offset=1)
        if None in (fastEma, medEma, slowEma, prevFast, prevMed):
            return False
        aligned = fastEma > medEma > slowEma
        crossover = prevFast <= prevMed and fastEma > medEma
        return aligned and crossover

    def sellCondition(s, bar):
        fastEma = s.ema(fast)
        medEma = s.ema(medium)
        slowEma = s.ema(slow)
        if None in (fastEma, medEma, slowEma):
            return False
        return fastEma < medEma < slowEma

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def macdRsiCombo(
    rsiPeriod: int = 14,
    macdFast: int = 12,
    macdSlow: int = 26,
    macdSignalPeriod: int = 9,
    rsiOversold: float = 40,
    rsiOverbought: float = 60,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    MACD + RSI combination strategy.

    Combines MACD crossover with RSI filter for confirmation. Buys when MACD
    crosses above signal line AND RSI is in oversold/neutral zone (below
    overbought). Sells on MACD bearish crossunder AND RSI confirmation.
    MACD 골든크로스와 RSI 필터를 결합합니다. MACD 상향 돌파 + RSI 확인 시 매수,
    MACD 하향 돌파 + RSI 확인 시 매도합니다.

    Args:
        rsiPeriod: RSI period (default 14). / RSI 기간.
        macdFast: MACD fast EMA period (default 12). / MACD 빠른 EMA 기간.
        macdSlow: MACD slow EMA period (default 26). / MACD 느린 EMA 기간.
        macdSignalPeriod: MACD signal period (default 9). / MACD 시그널 기간.
        rsiOversold: RSI buy filter level (default 40). / RSI 매수 필터 수준.
        rsiOverbought: RSI sell filter level (default 60). / RSI 매도 필터 수준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured MACD + RSI combination strategy.

    Example:
        >>> result = backtest("005930", macdRsiCombo())
        >>> result = backtest("005930", macdRsiCombo(rsiPeriod=10, macdFast=8))
    """
    strategy = QuickStrategy(f"MACD+RSI({macdFast}/{macdSlow}/{rsiPeriod})")

    def buyCondition(s, bar):
        macdResult = s.macd(macdFast, macdSlow, macdSignalPeriod)
        prevMacdResult = s.macd(macdFast, macdSlow, macdSignalPeriod, offset=1)
        rsiVal = s.rsi(rsiPeriod)
        if macdResult is None or prevMacdResult is None or rsiVal is None:
            return False
        macdLine, signalLine, _ = macdResult
        prevMacdLine, prevSignalLine, _ = prevMacdResult
        if None in (macdLine, signalLine, prevMacdLine, prevSignalLine):
            return False
        macdCrossover = prevMacdLine <= prevSignalLine and macdLine > signalLine
        return macdCrossover and rsiVal < rsiOverbought

    def sellCondition(s, bar):
        macdResult = s.macd(macdFast, macdSlow, macdSignalPeriod)
        prevMacdResult = s.macd(macdFast, macdSlow, macdSignalPeriod, offset=1)
        rsiVal = s.rsi(rsiPeriod)
        if macdResult is None or prevMacdResult is None or rsiVal is None:
            return False
        macdLine, signalLine, _ = macdResult
        prevMacdLine, prevSignalLine, _ = prevMacdResult
        if None in (macdLine, signalLine, prevMacdLine, prevSignalLine):
            return False
        macdCrossunder = prevMacdLine >= prevSignalLine and macdLine < signalLine
        return macdCrossunder and rsiVal > rsiOversold

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def trendMomentum(
    smaPeriod: int = 50,
    rsiPeriod: int = 14,
    adxPeriod: int = 14,
    adxThreshold: float = 25,
    rsiOversold: float = 40,
    stopLoss: float = None,
    takeProfit: float = None,
    trailingStop: float = None
) -> QuickStrategy:
    """
    Trend + momentum combination strategy.

    Triple confirmation: trend direction (price above SMA), trend strength
    (ADX above threshold), and momentum entry timing (RSI not overbought).
    Sells when trend weakens or momentum reverses.
    추세 방향(SMA 위), 추세 강도(ADX 임계값 이상), 모멘텀 타이밍(RSI 과매수 아님)
    삼중 확인 후 매수합니다. 추세 약화 또는 모멘텀 반전 시 매도합니다.

    Args:
        smaPeriod: Trend SMA period (default 50). / 추세 SMA 기간.
        rsiPeriod: RSI period (default 14). / RSI 기간.
        adxPeriod: ADX period (default 14). / ADX 기간.
        adxThreshold: Minimum ADX for trend (default 25). / ADX 임계값.
        rsiOversold: RSI buy zone upper limit (default 40). / RSI 매수 구간 상한.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.
        trailingStop: Trailing stop percent (optional). / 추적손절%.

    Returns:
        QuickStrategy: Configured trend + momentum strategy.

    Example:
        >>> result = backtest("005930", trendMomentum())
        >>> result = backtest("005930", trendMomentum(smaPeriod=100, adxThreshold=20))
    """
    strategy = QuickStrategy(f"추세모멘텀({smaPeriod}/{rsiPeriod}/{adxPeriod})")

    def buyCondition(s, bar):
        smaVal = s.sma(smaPeriod)
        rsiVal = s.rsi(rsiPeriod)
        adxVal = s.adx(adxPeriod)
        if None in (smaVal, rsiVal, adxVal):
            return False
        uptrend = bar.close > smaVal
        strongTrend = adxVal >= adxThreshold
        notOverbought = rsiVal < (100 - rsiOversold)
        return uptrend and strongTrend and notOverbought

    def sellCondition(s, bar):
        smaVal = s.sma(smaPeriod)
        adxVal = s.adx(adxPeriod)
        rsiVal = s.rsi(rsiPeriod)
        if None in (smaVal, adxVal, rsiVal):
            return False
        downtrend = bar.close < smaVal
        weakTrend = adxVal < adxThreshold
        overbought = rsiVal > (100 - rsiOversold)
        return downtrend or (weakTrend and overbought)

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)
    if trailingStop:
        strategy.trailingStop(trailingStop)

    return strategy


def bollingerRsi(
    bbPeriod: int = 20,
    rsiPeriod: int = 14,
    bbStd: float = 2.0,
    rsiOversold: float = 30,
    rsiOverbought: float = 70,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Bollinger Band + RSI filter strategy.

    Combines Bollinger Band mean reversion with RSI momentum confirmation.
    Buys when price touches the lower band AND RSI confirms oversold.
    Sells when price reaches the upper band AND RSI is overbought.
    볼린저 하단 밴드 터치 + RSI 과매도 확인 시 매수,
    상단 밴드 도달 + RSI 과매수 확인 시 매도합니다.

    Args:
        bbPeriod: Bollinger Band period (default 20). / 볼린저 밴드 기간.
        rsiPeriod: RSI period (default 14). / RSI 기간.
        bbStd: Bollinger Band standard deviation (default 2.0). / 볼린저 표준편차.
        rsiOversold: RSI oversold threshold (default 30). / RSI 과매도 기준.
        rsiOverbought: RSI overbought threshold (default 70). / RSI 과매수 기준.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Bollinger + RSI strategy.

    Example:
        >>> result = backtest("005930", bollingerRsi())
        >>> result = backtest("005930", bollingerRsi(rsiOversold=25, rsiOverbought=75))
    """
    strategy = QuickStrategy(f"볼린저+RSI({bbPeriod}/{rsiPeriod})")

    def buyCondition(s, bar):
        result = s.bollinger(bbPeriod, bbStd)
        rsiVal = s.rsi(rsiPeriod)
        if result is None or rsiVal is None:
            return False
        upper, middle, lower = result
        if lower is None:
            return False
        return bar.close <= lower and rsiVal < rsiOversold

    def sellCondition(s, bar):
        result = s.bollinger(bbPeriod, bbStd)
        rsiVal = s.rsi(rsiPeriod)
        if result is None or rsiVal is None:
            return False
        upper, middle, lower = result
        if upper is None:
            return False
        return bar.close >= upper and rsiVal > rsiOverbought

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def gapTrading(
    gapPercent: float = 2.0,
    stopLoss: float = 3,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Gap trading strategy.

    Trades opening gaps by entering in the direction of the gap with
    expectation of continuation. Buys when today's open gaps up by more
    than gapPercent above yesterday's close. Sells when a down gap occurs
    or price falls below the gap fill level.
    금일 시가가 전일 종가 대비 갭 상승하면 갭 방향 매수,
    갭 하락 발생 또는 갭 메우기 수준 이하 시 매도합니다.

    Args:
        gapPercent: Minimum gap size in percent (default 2.0). / 최소 갭 크기(%).
        stopLoss: Stop-loss percent (default 3). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured gap trading strategy.

    Example:
        >>> result = backtest("005930", gapTrading())
        >>> result = backtest("005930", gapTrading(gapPercent=1.5, stopLoss=2))
    """
    strategy = QuickStrategy(f"갭트레이딩({gapPercent}%)")

    def buyCondition(s, bar):
        if s._history is None or len(s._history) < 1:
            return False
        prevClose = s._history['close'].iloc[-1]
        if prevClose == 0:
            return False
        gapSize = (bar.open - prevClose) / prevClose * 100
        return gapSize >= gapPercent

    def sellCondition(s, bar):
        if s._history is None or len(s._history) < 1:
            return False
        prevClose = s._history['close'].iloc[-1]
        if prevClose == 0:
            return False
        gapSize = (bar.open - prevClose) / prevClose * 100
        priceDropped = bar.close < prevClose
        return gapSize <= -gapPercent or priceDropped

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)
    strategy.stopLoss(stopLoss)

    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def pyramiding(
    periods: list = None,
    stopLoss: float = None,
    takeProfit: float = None,
    trailingStop: float = 10
) -> QuickStrategy:
    """
    Pyramid entry on multi-period SMA confirmation strategy.

    Enters when price is confirmed above multiple SMA periods simultaneously,
    indicating strong multi-timeframe alignment. Sells when price drops
    below the longest SMA (trend reversal confirmed).
    여러 기간의 SMA가 동시에 가격 아래에 위치하면 다중 시간대 정배열로 매수,
    가장 긴 SMA 아래로 하락하면 추세 전환으로 매도합니다.

    Args:
        periods: List of SMA periods to confirm (default [10, 20, 30]). / SMA 기간 목록.
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.
        trailingStop: Trailing stop percent (default 10). / 추적손절%.

    Returns:
        QuickStrategy: Configured pyramiding strategy.

    Example:
        >>> result = backtest("005930", pyramiding())
        >>> result = backtest("005930", pyramiding(periods=[5, 15, 30, 60]))
    """
    if periods is None:
        periods = [10, 20, 30]
    periodStr = "/".join(str(p) for p in periods)
    strategy = QuickStrategy(f"피라미딩({periodStr})")
    sortedPeriods = sorted(periods)

    def buyCondition(s, bar):
        smaValues = []
        for p in sortedPeriods:
            val = s.sma(p)
            if val is None:
                return False
            smaValues.append(val)
        for val in smaValues:
            if bar.close <= val:
                return False
        for i in range(len(smaValues) - 1):
            if smaValues[i] <= smaValues[i + 1]:
                return False
        prevShortSma = s.sma(sortedPeriods[0], offset=1)
        prevMedSma = s.sma(sortedPeriods[1], offset=1)
        if None in (prevShortSma, prevMedSma):
            return False
        return prevShortSma <= prevMedSma

    def sellCondition(s, bar):
        longestSma = s.sma(sortedPeriods[-1])
        if longestSma is None:
            return False
        return bar.close < longestSma

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)
    if trailingStop:
        strategy.trailingStop(trailingStop)

    return strategy


def swingTrading(
    atrPeriod: int = 14,
    rsiPeriod: int = 14,
    rsiOversold: float = 30,
    rsiOverbought: float = 70,
    atrMultiplier: float = 2.0,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Swing trading with ATR-based dynamic stops strategy.

    Combines RSI oversold entries with ATR-based position sizing awareness.
    Buys when RSI is oversold and volatility is within acceptable range.
    Sells when RSI is overbought. Uses ATR-derived stop loss if no
    explicit stop loss is provided.
    RSI 과매도 + 적정 변동성 구간에서 매수, RSI 과매수 시 매도합니다.
    명시적 손절이 없으면 ATR 기반 동적 손절을 적용합니다.

    Args:
        atrPeriod: ATR period (default 14). / ATR 기간.
        rsiPeriod: RSI period (default 14). / RSI 기간.
        rsiOversold: RSI oversold level (default 30). / RSI 과매도 수준.
        rsiOverbought: RSI overbought level (default 70). / RSI 과매수 수준.
        atrMultiplier: ATR multiplier for stop calculation (default 2.0). / ATR 손절 배수.
        stopLoss: Stop-loss percent (optional, overrides ATR stop). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured swing trading strategy.

    Example:
        >>> result = backtest("005930", swingTrading())
        >>> result = backtest("005930", swingTrading(rsiOversold=25, atrMultiplier=1.5))
    """
    strategy = QuickStrategy(f"스윙트레이딩(ATR{atrPeriod}/RSI{rsiPeriod})")

    def buyCondition(s, bar):
        rsiVal = s.rsi(rsiPeriod)
        atrVal = s.atr(atrPeriod)
        smaVal = s.sma(20)
        if None in (rsiVal, atrVal, smaVal):
            return False
        if bar.close == 0:
            return False
        atrPercent = atrVal / bar.close * 100
        return rsiVal < rsiOversold and atrPercent < 10 and bar.close > smaVal

    def sellCondition(s, bar):
        rsiVal = s.rsi(rsiPeriod)
        atrVal = s.atr(atrPeriod)
        if None in (rsiVal, atrVal):
            return False
        overbought = rsiVal > rsiOverbought
        if s._entryPrice > 0 and not stopLoss:
            atrStop = s._entryPrice - atrVal * atrMultiplier
            if bar.close < atrStop:
                return True
        return overbought

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def scalpingMomentum(
    emaPeriod: int = 5,
    rsiPeriod: int = 7,
    rsiEntry: float = 50,
    stopLoss: float = 2,
    takeProfit: float = 3
) -> QuickStrategy:
    """
    Fast scalping momentum strategy.

    Designed for short-term trades using fast EMA and short RSI. Buys
    when price crosses above fast EMA with RSI momentum confirmation.
    Tight stop-loss and take-profit for quick captures.
    빠른 EMA와 단기 RSI를 사용한 단타 전략입니다. 가격이 빠른 EMA를
    상향 돌파하고 RSI 모멘텀이 확인되면 매수합니다.

    Args:
        emaPeriod: Fast EMA period (default 5). / 빠른 EMA 기간.
        rsiPeriod: Short RSI period (default 7). / 단기 RSI 기간.
        rsiEntry: RSI threshold for entry (default 50). / RSI 진입 기준.
        stopLoss: Stop-loss percent (default 2). / 손절%.
        takeProfit: Take-profit percent (default 3). / 익절%.

    Returns:
        QuickStrategy: Configured scalping momentum strategy.

    Example:
        >>> result = backtest("005930", scalpingMomentum())
        >>> result = backtest("005930", scalpingMomentum(emaPeriod=3, rsiPeriod=5))
    """
    strategy = QuickStrategy(f"스캘핑모멘텀(EMA{emaPeriod}/RSI{rsiPeriod})")

    def buyCondition(s, bar):
        emaVal = s.ema(emaPeriod)
        prevEma = s.ema(emaPeriod, offset=1)
        rsiVal = s.rsi(rsiPeriod)
        prevRsi = s.rsi(rsiPeriod, offset=1)
        if None in (emaVal, prevEma, rsiVal, prevRsi):
            return False
        priceCross = bar.close > emaVal and s._history['close'].iloc[-1] <= prevEma
        rsiRising = rsiVal > rsiEntry and prevRsi <= rsiEntry
        return priceCross and rsiRising

    def sellCondition(s, bar):
        emaVal = s.ema(emaPeriod)
        rsiVal = s.rsi(rsiPeriod)
        if None in (emaVal, rsiVal):
            return False
        return bar.close < emaVal or rsiVal < (100 - rsiEntry)

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)
    strategy.stopLoss(stopLoss)
    strategy.takeProfit(takeProfit)

    return strategy


def buyAndHold(
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Simple buy and hold benchmark strategy.

    Buys on the first available bar and holds indefinitely. Useful as a
    benchmark to compare active strategies against passive investing.
    No sell condition is applied unless stop-loss or take-profit is set.
    첫 번째 가용 바에서 매수하고 계속 보유합니다. 능동적 전략 대비
    패시브 투자 벤치마크로 유용합니다.

    Args:
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured buy and hold strategy.

    Example:
        >>> result = backtest("005930", buyAndHold())
        >>> result = backtest("005930", buyAndHold(stopLoss=20))
    """
    strategy = QuickStrategy("바이앤홀드")

    def buyCondition(s, bar):
        return not s.hasPosition(bar.symbol)

    def sellCondition(s, bar):
        return False

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy


def dollarCostAverage(
    interval: int = 21,
    stopLoss: float = None,
    takeProfit: float = None
) -> QuickStrategy:
    """
    Dollar Cost Averaging (DCA) strategy.

    Buys at regular intervals (every N bars) regardless of price.
    Never sells unless stop-loss or take-profit is triggered.
    Implements systematic periodic investment for long-term accumulation.
    가격에 관계없이 N봉마다 정기적으로 매수합니다.
    손절/익절 설정 시에만 매도합니다. 장기 적립식 투자를 구현합니다.

    Args:
        interval: Number of bars between purchases (default 21). / 매수 간격(봉 수).
        stopLoss: Stop-loss percent (optional). / 손절%.
        takeProfit: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured DCA strategy.

    Example:
        >>> result = backtest("005930", dollarCostAverage())
        >>> result = backtest("005930", dollarCostAverage(interval=5))
    """
    strategy = QuickStrategy(f"적립식투자({interval}봉)")

    def buyCondition(s, bar):
        if s._history is None:
            return True
        return len(s._history) % interval == 0

    def sellCondition(s, bar):
        return False

    strategy.buyWhen(buyCondition)
    strategy.sellWhen(sellCondition)

    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return strategy
