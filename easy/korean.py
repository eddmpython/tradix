"""
Tradex Korean API Module - Full Korean-language wrappers for backtesting.

Provides Korean-named functions that delegate to the English Easy API,
enabling users to write complete backtesting code entirely in Korean.
All parameters, function names, and return values use Korean identifiers.

한글 API 모듈 - 영어 없이 순수 한글만으로 백테스팅을 할 수 있습니다.
모든 함수명, 파라미터, 반환값이 한글로 제공됩니다.

Features:
    - 백테스트: Run a backtest with Korean parameters
    - 최적화: Parameter optimization with Korean metric names
    - 빠른테스트: Quick lambda-based testing in Korean
    - 전략: Create QuickStrategy with Korean name
    - 골든크로스, RSI과매도, 볼린저돌파, MACD크로스: Preset strategies
    - 돌파전략, 평균회귀, 추세추종: Advanced preset strategies

Usage:
    >>> from tradex.easy import 백테스트, 골든크로스, 전략
    >>>
    >>> 결과 = 백테스트("삼성전자", 골든크로스())
    >>> print(결과.수익률)
    >>> print(결과.요약())
    >>>
    >>> 내전략 = 전략("내전략").매수조건(crossover(sma(10), sma(30))).손절(5)
    >>> 결과 = 백테스트("삼성전자", 내전략)
"""

from typing import Union, Callable
from tradex.easy.api import backtest, optimize, EasyResult, quickTest
from tradex.easy.quick import QuickStrategy
from tradex.easy.presets import (
    goldenCross,
    rsiOversold,
    bollingerBreakout,
    macdCross,
    breakout,
    meanReversion,
    trendFollowing,
)
from tradex.strategy.base import Strategy


def 백테스트(
    종목: str,
    전략: Union[Strategy, QuickStrategy],
    기간: str = "3년",
    초기자금: Union[int, float, str] = "1000만원",
    수수료: float = 0.00015,
    상세출력: bool = False,
) -> EasyResult:
    """
    Run a backtest using Korean parameters. / 한글 파라미터로 백테스트를 실행합니다.

    Korean-language wrapper for the ``backtest()`` function. Accepts
    stock names in Korean (e.g., "삼성전자") and Korean currency
    expressions (e.g., "1000만원", "1억").

    Args:
        종목: Stock code or Korean name (e.g., "005930", "삼성전자"). / 종목 코드 또는 한글명.
        전략: Strategy object (preset or custom). / 전략 객체.
        기간: Period expression (e.g., "3년", "6개월", "2020-01-01~2024-12-31"). / 기간.
        초기자금: Initial capital (e.g., "1000만원", "1억", 10000000). / 초기 자금.
        수수료: Commission rate (default 0.015%). / 수수료율.
        상세출력: Enable verbose logging. / 상세 로그 출력 여부.

    Returns:
        EasyResult: Result object with Korean property accessors. / 한글 속성을 지원하는 결과 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 골든크로스())
        >>> print(결과.수익률)
        >>> print(결과.요약())
    """
    return backtest(
        symbol=종목,
        strategy=전략,
        period=기간,
        initialCash=초기자금,
        commission=수수료,
        verbose=상세출력,
    )


def 최적화(
    종목: str,
    전략생성함수: Callable[..., Strategy],
    기간: str = "3년",
    지표: str = "샤프비율",
    **파라미터범위
):
    """
    Run parameter optimization using Korean parameters. / 한글 파라미터로 최적화를 실행합니다.

    Korean-language wrapper for the ``optimize()`` function. Accepts
    Korean metric names (샤프비율, 수익률, 최대낙폭, 승률) and
    Korean parameter names (단기, 장기, 기간, 손절, 익절).

    Args:
        종목: Stock code or Korean name. / 종목 코드 또는 한글명.
        전략생성함수: Strategy factory function. / 전략 생성 함수.
        기간: Period expression. / 기간.
        지표: Optimization metric ("샤프비율", "수익률", "최대낙폭", "승률"). / 최적화 지표.
        **파라미터범위: Parameter ranges as (start, stop, step) tuples. / 파라미터 범위.

    Returns:
        dict: Optimization results with best parameters and all results. / 최적 파라미터와 전체 결과.

    Example:
        >>> 최적 = 최적화(
        ...     "삼성전자",
        ...     골든크로스,
        ...     단기=(5, 20, 5),
        ...     장기=(20, 60, 10),
        ...     지표="샤프비율"
        ... )
    """
    metricMap = {
        "샤프비율": "sharpeRatio",
        "수익률": "totalReturn",
        "최대낙폭": "maxDrawdown",
        "승률": "winRate",
    }

    metric = metricMap.get(지표, 지표)

    paramMapping = {
        "단기": "fast",
        "장기": "slow",
        "기간": "period",
        "손절": "stopLoss",
        "익절": "takeProfit",
    }

    englishParams = {}
    for k, v in 파라미터범위.items():
        engKey = paramMapping.get(k, k)
        englishParams[engKey] = v

    return optimize(
        symbol=종목,
        strategyFactory=전략생성함수,
        period=기간,
        metric=metric,
        **englishParams
    )


def 빠른테스트(
    종목: str,
    매수조건: Callable,
    매도조건: Callable = None,
    기간: str = "3년",
    손절: float = None,
    익절: float = None,
) -> EasyResult:
    """
    Quick idea testing with lambda conditions in Korean. / 한글로 람다 조건을 사용한 빠른 아이디어 테스트.

    Korean-language wrapper for the ``quickTest()`` function. Test
    trading ideas quickly using lambda functions without building
    a full strategy object.

    Args:
        종목: Stock code or Korean name. / 종목 코드 또는 한글명.
        매수조건: Buy condition as lambda(strategy, bar) -> bool. / 매수 조건 람다 함수.
        매도조건: Sell condition as lambda(strategy, bar) -> bool (optional). / 매도 조건 람다 함수.
        기간: Period expression (default "3년"). / 기간.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        EasyResult: Result object with Korean property accessors. / 한글 속성을 지원하는 결과 객체.

    Example:
        >>> 결과 = 빠른테스트(
        ...     "삼성전자",
        ...     매수조건=lambda s, bar: bar.close > s.sma(20),
        ...     매도조건=lambda s, bar: bar.close < s.sma(20)
        ... )
    """
    return quickTest(
        symbol=종목,
        buyCondition=매수조건,
        sellCondition=매도조건,
        period=기간,
        stopLoss=손절,
        takeProfit=익절,
    )


def 전략(이름: str = "내전략") -> QuickStrategy:
    """
    Create a declarative QuickStrategy with a Korean name. / 한글 이름으로 선언형 전략을 생성합니다.

    Korean-language wrapper that creates a QuickStrategy instance.
    The returned strategy supports Korean method aliases (매수조건,
    매도조건, 손절, 익절, 추적손절) for full Korean chaining.

    Args:
        이름: Strategy name (default "내전략"). / 전략 이름.

    Returns:
        QuickStrategy: New strategy instance with Korean method support. / 한글 메서드를 지원하는 전략 인스턴스.

    Example:
        >>> 내전략 = 전략("골든크로스전략")
        >>> 내전략.매수조건(crossover(sma(10), sma(30)))
        >>> 내전략.매도조건(crossunder(sma(10), sma(30)))
        >>> 내전략.손절(5).익절(15)
        >>> 결과 = 백테스트("삼성전자", 내전략)
    """
    return QuickStrategy(이름)


def 골든크로스(단기: int = 10, 장기: int = 30, 손절: float = None, 익절: float = None) -> QuickStrategy:
    """
    Golden Cross / Death Cross strategy (Korean wrapper). / 골든크로스/데드크로스 전략 (한글 래퍼).

    Korean-language wrapper for the ``goldenCross()`` preset.
    Delegates to ``goldenCross(fast, slow, stopLoss, takeProfit)``.

    Args:
        단기: Fast SMA period (default 10). / 단기 이동평균 기간.
        장기: Slow SMA period (default 30). / 장기 이동평균 기간.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured golden cross strategy. / 골든크로스 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 골든크로스())
        >>> 결과 = 백테스트("삼성전자", 골든크로스(단기=5, 장기=20))
    """
    return goldenCross(fast=단기, slow=장기, stopLoss=손절, takeProfit=익절)


def RSI과매도(
    기간: int = 14,
    과매도: float = 30,
    과매수: float = 70,
    손절: float = None,
    익절: float = None
) -> QuickStrategy:
    """
    RSI oversold bounce strategy (Korean wrapper). / RSI 과매도 반등 전략 (한글 래퍼).

    Korean-language wrapper for the ``rsiOversold()`` preset.
    Delegates to ``rsiOversold(period, oversold, overbought, stopLoss, takeProfit)``.

    Args:
        기간: RSI calculation period (default 14). / RSI 기간.
        과매도: Oversold threshold (default 30). / 과매도 기준.
        과매수: Overbought threshold (default 70). / 과매수 기준.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured RSI oversold strategy. / RSI 과매도 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", RSI과매도())
        >>> 결과 = 백테스트("삼성전자", RSI과매도(과매도=25))
    """
    return rsiOversold(period=기간, oversold=과매도, overbought=과매수, stopLoss=손절, takeProfit=익절)


def 볼린저돌파(
    기간: int = 20,
    표준편차: float = 2.0,
    손절: float = None,
    익절: float = None
) -> QuickStrategy:
    """
    Bollinger Band breakout strategy (Korean wrapper). / 볼린저 밴드 돌파 전략 (한글 래퍼).

    Korean-language wrapper for the ``bollingerBreakout()`` preset.
    Delegates to ``bollingerBreakout(period, std, stopLoss, takeProfit)``.

    Args:
        기간: Bollinger Band period (default 20). / 볼린저 밴드 기간.
        표준편차: Standard deviation multiplier (default 2.0). / 표준편차 배수.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured Bollinger breakout strategy. / 볼린저 돌파 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 볼린저돌파())
    """
    return bollingerBreakout(period=기간, std=표준편차, stopLoss=손절, takeProfit=익절)


def MACD크로스(
    빠른기간: int = 12,
    느린기간: int = 26,
    신호기간: int = 9,
    손절: float = None,
    익절: float = None
) -> QuickStrategy:
    """
    MACD signal line crossover strategy (Korean wrapper). / MACD 크로스 전략 (한글 래퍼).

    Korean-language wrapper for the ``macdCross()`` preset.
    Delegates to ``macdCross(fast, slow, signal, stopLoss, takeProfit)``.

    Args:
        빠른기간: Fast EMA period (default 12). / 빠른 EMA 기간.
        느린기간: Slow EMA period (default 26). / 느린 EMA 기간.
        신호기간: Signal line period (default 9). / 시그널 기간.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured MACD crossover strategy. / MACD 크로스 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", MACD크로스())
    """
    return macdCross(fast=빠른기간, slow=느린기간, signal=신호기간, stopLoss=손절, takeProfit=익절)


def 돌파전략(
    기간: int = 20,
    손절: float = None,
    익절: float = None,
    추적손절: float = None
) -> QuickStrategy:
    """
    Channel breakout / Turtle Trading strategy (Korean wrapper). / 채널 돌파 전략 (한글 래퍼).

    Korean-language wrapper for the ``breakout()`` preset.
    Delegates to ``breakout(period, stopLoss, takeProfit, trailingStop)``.

    Args:
        기간: Channel lookback period (default 20). / 돌파 기준 기간.
        손절: Stop-loss percent (optional). / 손절%.
        익절: Take-profit percent (optional). / 익절%.
        추적손절: Trailing stop percent (optional). / 추적손절%.

    Returns:
        QuickStrategy: Configured channel breakout strategy. / 돌파 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 돌파전략())
        >>> 결과 = 백테스트("삼성전자", 돌파전략(추적손절=10))
    """
    return breakout(period=기간, stopLoss=손절, takeProfit=익절, trailingStop=추적손절)


def 평균회귀(
    기간: int = 20,
    임계값: float = 2.0,
    손절: float = 5,
    익절: float = None
) -> QuickStrategy:
    """
    Mean reversion strategy with Bollinger Bands (Korean wrapper). / 평균 회귀 전략 (한글 래퍼).

    Korean-language wrapper for the ``meanReversion()`` preset.
    Delegates to ``meanReversion(period, threshold, stopLoss, takeProfit)``.

    Args:
        기간: Moving average period (default 20). / 이동평균 기간.
        임계값: Standard deviation multiplier for bands (default 2.0). / 진입 기준 표준편차 배수.
        손절: Stop-loss percent (default 5). / 손절%.
        익절: Take-profit percent (optional). / 익절%.

    Returns:
        QuickStrategy: Configured mean reversion strategy. / 평균회귀 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 평균회귀())
    """
    return meanReversion(period=기간, threshold=임계값, stopLoss=손절, takeProfit=익절)


def 추세추종(
    단기: int = 10,
    장기: int = 30,
    ADX기간: int = 14,
    ADX임계값: float = 25,
    추적손절: float = 10
) -> QuickStrategy:
    """
    ADX-filtered trend following strategy (Korean wrapper). / 추세 추종 전략 (한글 래퍼).

    Korean-language wrapper for the ``trendFollowing()`` preset.
    Delegates to ``trendFollowing(fastPeriod, slowPeriod, adxPeriod, adxThreshold, trailingStop)``.

    Args:
        단기: Fast SMA period (default 10). / 단기 이동평균 기간.
        장기: Slow SMA period (default 30). / 장기 이동평균 기간.
        ADX기간: ADX calculation period (default 14). / ADX 기간.
        ADX임계값: Minimum ADX for trend confirmation (default 25). / ADX 임계값.
        추적손절: Trailing stop percent from peak (default 10). / 추적손절%.

    Returns:
        QuickStrategy: Configured trend following strategy. / 추세추종 전략 객체.

    Example:
        >>> 결과 = 백테스트("삼성전자", 추세추종())
    """
    return trendFollowing(
        fastPeriod=단기,
        slowPeriod=장기,
        adxPeriod=ADX기간,
        adxThreshold=ADX임계값,
        trailingStop=추적손절
    )
