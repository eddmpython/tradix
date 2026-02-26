"""
Tradex Easy API Package - Simple, accessible backtesting for everyone.

Provides a unified entry point for Tradex's simplified backtesting API
with three skill levels: Level 1 (one-liner presets for beginners),
Level 2 (declarative strategy builder for intermediate users), and
Level 3 (full Strategy subclassing for advanced users). Supports both
English and Korean interfaces with full bilingual parity.

간편 API 패키지 - 3단계 난이도로 누구나 쉽게 백테스팅할 수 있습니다.
영어와 한글 인터페이스를 모두 지원합니다.

Features:
    - backtest / 백테스트: Run backtests with one function call
    - optimize / 최적화: Grid search parameter optimization
    - quickTest / 빠른테스트: Lambda-based quick idea testing
    - QuickStrategy / 전략: Declarative strategy builder with chaining
    - Preset strategies: goldenCross, rsiOversold, bollingerBreakout, etc.
    - Indicator conditions: sma, ema, rsi, macd, crossover, crossunder
    - Full Korean API: 골든크로스, RSI과매도, 볼린저돌파, MACD크로스, etc.

Usage:
    >>> # Level 1: One-liner preset backtest
    >>> from tradex.easy import backtest, goldenCross
    >>> result = backtest("005930", goldenCross())
    >>> print(result.totalReturn)
    >>>
    >>> # Level 1: Korean API
    >>> from tradex.easy import 백테스트, 골든크로스
    >>> 결과 = 백테스트("삼성전자", 골든크로스())
    >>> print(결과.수익률)
    >>>
    >>> # Level 2: Declarative strategy builder
    >>> from tradex.easy import backtest, QuickStrategy, sma, crossover
    >>> strategy = QuickStrategy("My Strategy").buyWhen(crossover(sma(10), sma(30))).stopLoss(5)
    >>> result = backtest("005930", strategy)
"""

from tradex.easy.api import backtest, optimize, quickTest
from tradex.easy.quick import QuickStrategy
from tradex.easy.presets import (
    goldenCross,
    rsiOversold,
    bollingerBreakout,
    macdCross,
    breakout,
    meanReversion,
    trendFollowing,
    emaCross,
    tripleScreen,
    dualMomentum,
    momentumCross,
    rocBreakout,
    stochasticCross,
    williamsReversal,
    cciBreakout,
    rsiDivergence,
    volatilityBreakout,
    keltnerChannel,
    bollingerSqueeze,
    superTrend,
    ichimokuCloud,
    parabolicSar,
    donchianBreakout,
    tripleEma,
    macdRsiCombo,
    trendMomentum,
    bollingerRsi,
    gapTrading,
    pyramiding,
    swingTrading,
    scalpingMomentum,
    buyAndHold,
    dollarCostAverage,
)
from tradex.easy.korean import (
    백테스트, 최적화, 빠른테스트, 전략,
    골든크로스, RSI과매도, 볼린저돌파, MACD크로스,
    돌파전략, 평균회귀, 추세추종, EMA크로스,
    삼중스크린, 듀얼모멘텀, 모멘텀크로스, ROC돌파,
    스토캐스틱크로스, 윌리엄스반전, CCI돌파, RSI다이버전스,
    변동성돌파, 켈트너채널, 볼린저스퀴즈, 슈퍼트렌드,
    일목균형표, 파라볼릭SAR, 돈치안돌파, 삼중EMA,
    MACD_RSI콤보, 추세모멘텀, 볼린저RSI, 갭트레이딩,
    피라미딩, 스윙트레이딩, 스캘핑모멘텀, 바이앤홀드, 적립식투자,
)
from tradex.easy.conditions import (
    sma,
    ema,
    rsi,
    macd,
    bollinger,
    atr,
    price,
    crossover,
    crossunder,
)

__all__ = [
    "backtest",
    "optimize",
    "quickTest",
    "QuickStrategy",
    "goldenCross",
    "rsiOversold",
    "bollingerBreakout",
    "macdCross",
    "breakout",
    "meanReversion",
    "trendFollowing",
    "emaCross",
    "tripleScreen",
    "dualMomentum",
    "momentumCross",
    "rocBreakout",
    "stochasticCross",
    "williamsReversal",
    "cciBreakout",
    "rsiDivergence",
    "volatilityBreakout",
    "keltnerChannel",
    "bollingerSqueeze",
    "superTrend",
    "ichimokuCloud",
    "parabolicSar",
    "donchianBreakout",
    "tripleEma",
    "macdRsiCombo",
    "trendMomentum",
    "bollingerRsi",
    "gapTrading",
    "pyramiding",
    "swingTrading",
    "scalpingMomentum",
    "buyAndHold",
    "dollarCostAverage",
    "백테스트", "최적화", "빠른테스트", "전략",
    "골든크로스", "RSI과매도", "볼린저돌파", "MACD크로스",
    "돌파전략", "평균회귀", "추세추종", "EMA크로스",
    "삼중스크린", "듀얼모멘텀", "모멘텀크로스", "ROC돌파",
    "스토캐스틱크로스", "윌리엄스반전", "CCI돌파", "RSI다이버전스",
    "변동성돌파", "켈트너채널", "볼린저스퀴즈", "슈퍼트렌드",
    "일목균형표", "파라볼릭SAR", "돈치안돌파", "삼중EMA",
    "MACD_RSI콤보", "추세모멘텀", "볼린저RSI", "갭트레이딩",
    "피라미딩", "스윙트레이딩", "스캘핑모멘텀", "바이앤홀드", "적립식투자",
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "price",
    "crossover",
    "crossunder",
]
