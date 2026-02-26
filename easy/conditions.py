"""
Tradex Condition Builders Module - Declarative indicator and condition definitions.

Provides a TradingView Pine Script-like interface for defining trading conditions.
Indicators are composable with comparison operators (>, <, >=, <=, ==) and
logical operators (&, |, ~). Crossover/crossunder detection is built in.

조건 빌더 모듈 - TradingView Pine Script 스타일의 선언형 지표/조건 정의.
비교 연산자와 논리 연산자를 통해 지표를 조합하고 크로스오버/크로스언더를 감지합니다.

Features:
    - Indicator factories: sma(), ema(), rsi(), macd(), bollinger(), atr()
    - Comparison operators: indicator > value, indicator < indicator
    - Logical operators: condition & condition, condition | condition, ~condition
    - Cross detection: crossover(fast, slow), crossunder(fast, slow)
    - Historical access: indicator.ago(N) for N bars ago
    - Price accessors: price, open_, high, low, close, volume

Usage:
    >>> from tradex.easy.conditions import sma, rsi, crossover
    >>>
    >>> strategy = (
    ...     QuickStrategy("MyStrategy")
    ...     .buyWhen(crossover(sma(10), sma(30)))
    ...     .buyWhen(rsi(14) < 30)
    ...     .sellWhen(rsi(14) > 70)
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union


class IndicatorType(Enum):
    """Enumeration of supported indicator types. / 지원되는 지표 유형 열거형."""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    MACD_SIGNAL = "macdSignal"
    MACD_HIST = "macdHist"
    BOLLINGER_UPPER = "bollingerUpper"
    BOLLINGER_MIDDLE = "bollingerMiddle"
    BOLLINGER_LOWER = "bollingerLower"
    ATR = "atr"
    PRICE = "price"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"


@dataclass
class Indicator:
    """
    Indicator definition for declarative condition building.

    Supports comparison operators to create conditions and .ago(N) for
    historical value access. Used as building blocks for QuickStrategy.

    선언형 조건 빌더를 위한 지표 정의 클래스.
    비교 연산자와 .ago(N)을 통한 과거 값 접근을 지원합니다.

    Attributes:
        indicatorType (IndicatorType): Type of indicator (SMA, EMA, RSI, etc.).
        params (dict): Indicator parameters (e.g., {"period": 20}).
        offset (int): Bar offset for historical values (0=current, 1=previous).

    Example:
        >>> sma20 = Indicator(IndicatorType.SMA, {"period": 20})
        >>> sma20_prev = sma20.ago(1)
        >>> condition = sma20 > sma20.ago(1)
    """
    indicatorType: IndicatorType
    params: dict
    offset: int = 0

    @property
    def type(self) -> IndicatorType:
        """Alias for indicatorType. / indicatorType의 별칭."""
        return self.indicatorType

    def __gt__(self, other: Union[Indicator, float, int]) -> CompareCondition:
        """Create greater-than condition (>). / 초과 조건 생성."""
        return CompareCondition(self, ">", other)

    def __lt__(self, other: Union[Indicator, float, int]) -> CompareCondition:
        """Create less-than condition (<). / 미만 조건 생성."""
        return CompareCondition(self, "<", other)

    def __ge__(self, other: Union[Indicator, float, int]) -> CompareCondition:
        """Create greater-or-equal condition (>=). / 이상 조건 생성."""
        return CompareCondition(self, ">=", other)

    def __le__(self, other: Union[Indicator, float, int]) -> CompareCondition:
        """Create less-or-equal condition (<=). / 이하 조건 생성."""
        return CompareCondition(self, "<=", other)

    def __eq__(self, other: Union[Indicator, float, int]) -> CompareCondition:
        """Create equality condition (==). / 동등 조건 생성."""
        return CompareCondition(self, "==", other)

    def ago(self, bars: int) -> Indicator:
        """
        Get indicator value from N bars ago. / N봉 전 지표 값 가져오기.

        Args:
            bars: Number of bars to look back. / 과거 봉 수.

        Returns:
            Indicator: New Indicator with offset applied. / 오프셋이 적용된 새 Indicator.

        Example:
            >>> sma(20).ago(1)  # Yesterday's SMA20
        """
        return Indicator(self.indicatorType, self.params, offset=bars)


@dataclass
class Condition:
    """
    Base condition class for comparison and logical operations.

    Supports AND (&), OR (|), and NOT (~) composition to build
    complex condition trees from simple comparisons.

    비교 및 논리 연산을 위한 기본 조건 클래스.
    AND, OR, NOT 합성으로 복합 조건 트리를 구성합니다.

    Attributes:
        left: Left operand (Indicator, Condition, or numeric value).
        operator (str): Comparison (>, <, >=, <=, ==) or logical (and, or, not) operator.
        right: Right operand (may be None for unary NOT).
    """
    left: Union[Indicator, Condition, float]
    operator: str
    right: Union[Indicator, Condition, float, None]

    def __and__(self, other: Condition) -> Condition:
        """Combine conditions with AND logic. / AND 논리로 조건 결합."""
        return Condition(self, "and", other)

    def __or__(self, other: Condition) -> Condition:
        """Combine conditions with OR logic. / OR 논리로 조건 결합."""
        return Condition(self, "or", other)

    def __invert__(self) -> Condition:
        """Negate the condition with NOT logic. / NOT 논리로 조건 반전."""
        return Condition(self, "not", None)


@dataclass
class CompareCondition(Condition):
    """Comparison condition (>, <, >=, <=, ==). / 비교 조건."""
    pass


@dataclass
class CrossCondition(Condition):
    """
    Crossover/crossunder detection condition.

    크로스오버/크로스언더 감지 조건.

    Attributes:
        crossType (str): "over" for bullish crossover, "under" for bearish crossunder.
    """
    crossType: str = "over"


def sma(period: int) -> Indicator:
    """
    Create Simple Moving Average (SMA) indicator. / SMA 지표 생성.

    Args:
        period: Lookback period for averaging. / 평균 기간.

    Returns:
        Indicator: SMA indicator instance.

    Example:
        >>> sma(20)           # 20-period SMA
        >>> sma(10) > sma(30) # Golden cross condition
        >>> sma(20).ago(1)    # Yesterday's SMA
    """
    return Indicator(IndicatorType.SMA, {"period": period})


def ema(period: int) -> Indicator:
    """
    Create Exponential Moving Average (EMA) indicator. / EMA 지표 생성.

    Args:
        period: Lookback period. / EMA 기간.

    Returns:
        Indicator: EMA indicator instance.

    Example:
        >>> ema(12)           # 12-period EMA
        >>> ema(12) > ema(26) # EMA crossover condition
    """
    return Indicator(IndicatorType.EMA, {"period": period})


def rsi(period: int = 14) -> Indicator:
    """
    Create Relative Strength Index (RSI) indicator. / RSI 지표 생성.

    Args:
        period: RSI period (default 14). / RSI 기간.

    Returns:
        Indicator: RSI indicator instance.

    Example:
        >>> rsi(14) < 30  # Oversold condition
        >>> rsi(14) > 70  # Overbought condition
    """
    return Indicator(IndicatorType.RSI, {"period": period})


def macd(fast: int = 12, slow: int = 26, signal: int = 9) -> Indicator:
    """
    Create MACD line indicator. / MACD 라인 지표 생성.

    Args:
        fast: Fast EMA period (default 12). / 단기 EMA 기간.
        slow: Slow EMA period (default 26). / 장기 EMA 기간.
        signal: Signal line period (default 9). / 시그널 기간.

    Returns:
        Indicator: MACD line indicator instance.

    Example:
        >>> crossover(macd(), macdSignal())  # MACD bullish crossover
    """
    return Indicator(IndicatorType.MACD, {"fast": fast, "slow": slow, "signal": signal})


def macdSignal(fast: int = 12, slow: int = 26, signal: int = 9) -> Indicator:
    """Create MACD signal line indicator. / MACD 시그널선 지표 생성."""
    return Indicator(IndicatorType.MACD_SIGNAL, {"fast": fast, "slow": slow, "signal": signal})


def macdHist(fast: int = 12, slow: int = 26, signal: int = 9) -> Indicator:
    """Create MACD histogram indicator. / MACD 히스토그램 지표 생성."""
    return Indicator(IndicatorType.MACD_HIST, {"fast": fast, "slow": slow, "signal": signal})


def bollinger(period: int = 20, std: float = 2.0) -> Tuple[Indicator, Indicator, Indicator]:
    """
    Create Bollinger Bands indicator tuple. / 볼린저 밴드 지표 튜플 생성.

    Args:
        period: Moving average period (default 20). / 이동평균 기간.
        std: Standard deviation multiplier (default 2.0). / 표준편차 배수.

    Returns:
        tuple: (upper, middle, lower) band Indicator instances.

    Example:
        >>> upper, middle, lower = bollinger(20, 2.0)
        >>> price < lower  # Price below lower band
    """
    upper = Indicator(IndicatorType.BOLLINGER_UPPER, {"period": period, "std": std})
    middle = Indicator(IndicatorType.BOLLINGER_MIDDLE, {"period": period, "std": std})
    lower = Indicator(IndicatorType.BOLLINGER_LOWER, {"period": period, "std": std})
    return upper, middle, lower


def bollingerUpper(period: int = 20, std: float = 2.0) -> Indicator:
    """Create Bollinger upper band indicator. / 볼린저 상단 밴드 지표 생성."""
    return Indicator(IndicatorType.BOLLINGER_UPPER, {"period": period, "std": std})


def bollingerLower(period: int = 20, std: float = 2.0) -> Indicator:
    """Create Bollinger lower band indicator. / 볼린저 하단 밴드 지표 생성."""
    return Indicator(IndicatorType.BOLLINGER_LOWER, {"period": period, "std": std})


def bollingerMiddle(period: int = 20, std: float = 2.0) -> Indicator:
    """Create Bollinger middle band (SMA) indicator. / 볼린저 중간 밴드 (SMA) 지표 생성."""
    return Indicator(IndicatorType.BOLLINGER_MIDDLE, {"period": period, "std": std})


def atr(period: int = 14) -> Indicator:
    """
    Create Average True Range (ATR) indicator. / ATR 지표 생성.

    Args:
        period: ATR period (default 14). / ATR 기간.

    Returns:
        Indicator: ATR indicator instance.

    Example:
        >>> atr(14)  # Volatility measurement
    """
    return Indicator(IndicatorType.ATR, {"period": period})


def price() -> Indicator:
    """Create current price (close) indicator."""
    return Indicator(IndicatorType.CLOSE, {})


price = Indicator(IndicatorType.CLOSE, {})
open_ = Indicator(IndicatorType.OPEN, {})
high = Indicator(IndicatorType.HIGH, {})
low = Indicator(IndicatorType.LOW, {})
close = Indicator(IndicatorType.CLOSE, {})
volume = Indicator(IndicatorType.VOLUME, {})


def crossover(indicator1: Indicator, indicator2: Union[Indicator, float]) -> CrossCondition:
    """
    Create crossover (golden cross) condition. / 골든크로스 조건 생성.

    Detects when indicator1 crosses above indicator2.
    indicator1이 indicator2를 상향 돌파하는 시점을 감지합니다.

    Args:
        indicator1: Fast indicator or value. / 빠른 지표 또는 값.
        indicator2: Slow indicator or threshold. / 느린 지표 또는 임계값.

    Returns:
        CrossCondition: Crossover condition instance.

    Example:
        >>> crossover(sma(10), sma(30))  # Golden cross
        >>> crossover(rsi(14), 30)       # RSI crosses above 30
    """
    return CrossCondition(indicator1, "crossover", indicator2, crossType="over")


def crossunder(indicator1: Indicator, indicator2: Union[Indicator, float]) -> CrossCondition:
    """
    Create crossunder (death cross) condition. / 데드크로스 조건 생성.

    Detects when indicator1 crosses below indicator2.
    indicator1이 indicator2를 하향 돌파하는 시점을 감지합니다.

    Args:
        indicator1: Fast indicator or value. / 빠른 지표 또는 값.
        indicator2: Slow indicator or threshold. / 느린 지표 또는 임계값.

    Returns:
        CrossCondition: Crossunder condition instance.

    Example:
        >>> crossunder(sma(10), sma(30))  # Death cross
        >>> crossunder(rsi(14), 70)       # RSI crosses below 70
    """
    return CrossCondition(indicator1, "crossunder", indicator2, crossType="under")


class ConditionEvaluator:
    """
    Runtime condition evaluator for QuickStrategy.

    Traverses condition trees and evaluates them against current bar data
    by delegating indicator computations to the parent strategy instance.

    QuickStrategy의 런타임 조건 평가기.
    조건 트리를 순회하며 부모 전략의 지표 계산을 활용하여 현재 바에 대해 평가합니다.

    Attributes:
        strategy: Parent strategy instance providing indicator methods.
    """

    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy

    def evaluate(self, condition: Union[Condition, CrossCondition], bar: Any) -> bool:
        """
        Evaluate a condition tree against the current bar. / 현재 바에 대해 조건 트리를 평가.

        Args:
            condition: Condition or CrossCondition to evaluate. / 평가할 조건.
            bar: Current OHLCV price bar. / 현재 가격 바.

        Returns:
            bool: True if the condition is met. / 조건 충족 시 True.
        """
        if isinstance(condition, CrossCondition):
            return self._evaluateCross(condition, bar)
        return self._evaluateCondition(condition, bar)

    def _evaluateCondition(self, condition: Condition, bar: Any) -> bool:
        """Evaluate comparison or logical condition node. / 비교 또는 논리 조건 노드 평가."""
        if condition.operator == "and":
            return self.evaluate(condition.left, bar) and self.evaluate(condition.right, bar)
        if condition.operator == "or":
            return self.evaluate(condition.left, bar) or self.evaluate(condition.right, bar)
        if condition.operator == "not":
            return not self.evaluate(condition.left, bar)

        leftVal = self._getValue(condition.left, bar)
        rightVal = self._getValue(condition.right, bar)

        if leftVal is None or rightVal is None:
            return False

        if condition.operator == ">":
            return leftVal > rightVal
        if condition.operator == "<":
            return leftVal < rightVal
        if condition.operator == ">=":
            return leftVal >= rightVal
        if condition.operator == "<=":
            return leftVal <= rightVal
        if condition.operator == "==":
            return leftVal == rightVal

        return False

    def _evaluateCross(self, condition: CrossCondition, bar: Any) -> bool:
        """Evaluate crossover/crossunder condition using current and previous values. / 크로스오버/크로스언더 조건 평가."""
        currLeft = self._getValue(condition.left, bar, offset=0)
        prevLeft = self._getValue(condition.left, bar, offset=1)
        currRight = self._getValue(condition.right, bar, offset=0)
        prevRight = self._getValue(condition.right, bar, offset=1)

        if None in (currLeft, prevLeft, currRight, prevRight):
            return False

        if condition.crossType == "over":
            return prevLeft <= prevRight and currLeft > currRight
        else:
            return prevLeft >= prevRight and currLeft < currRight

    def _getValue(
        self,
        item: Union[Indicator, float, int],
        bar: Any,
        offset: Optional[int] = None
    ) -> Optional[float]:
        """Resolve a value from an Indicator instance or numeric constant. / 지표 또는 상수에서 값 추출."""
        if isinstance(item, (int, float)):
            return float(item)

        if isinstance(item, Indicator):
            actualOffset = offset if offset is not None else item.offset
            return self._getIndicatorValue(item, actualOffset)

        return None

    def _getIndicatorValue(self, indicator: Indicator, offset: int) -> Optional[float]:
        """Compute indicator value by dispatching to the parent strategy. / 부모 전략에 위임하여 지표 값 계산."""
        s = self.strategy
        p = indicator.params
        t = indicator.indicatorType

        if t == IndicatorType.SMA:
            return s.sma(p["period"], offset=offset)
        elif t == IndicatorType.EMA:
            return s.ema(p["period"], offset=offset)
        elif t == IndicatorType.RSI:
            return s.rsi(p["period"], offset=offset)
        elif t == IndicatorType.MACD:
            result = s.macd(p["fast"], p["slow"], p["signal"], offset=offset)
            return result[0] if result else None
        elif t == IndicatorType.MACD_SIGNAL:
            result = s.macd(p["fast"], p["slow"], p["signal"], offset=offset)
            return result[1] if result else None
        elif t == IndicatorType.MACD_HIST:
            result = s.macd(p["fast"], p["slow"], p["signal"], offset=offset)
            return result[2] if result else None
        elif t == IndicatorType.BOLLINGER_UPPER:
            result = s.bollinger(p["period"], p["std"], offset=offset)
            return result[0] if result else None
        elif t == IndicatorType.BOLLINGER_MIDDLE:
            result = s.bollinger(p["period"], p["std"], offset=offset)
            return result[1] if result else None
        elif t == IndicatorType.BOLLINGER_LOWER:
            result = s.bollinger(p["period"], p["std"], offset=offset)
            return result[2] if result else None
        elif t == IndicatorType.ATR:
            return s.atr(p["period"], offset=offset)
        elif t == IndicatorType.CLOSE:
            return s._currentBar.close if s._currentBar else None
        elif t == IndicatorType.OPEN:
            return s._currentBar.open if s._currentBar else None
        elif t == IndicatorType.HIGH:
            return s._currentBar.high if s._currentBar else None
        elif t == IndicatorType.LOW:
            return s._currentBar.low if s._currentBar else None
        elif t == IndicatorType.VOLUME:
            return s._currentBar.volume if s._currentBar else None

        return None
