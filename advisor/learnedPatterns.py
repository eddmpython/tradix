"""Tradix Learned Patterns Module.

Contains empirically derived strategy-regime mappings, parameter adjustments,
and performance benchmarks from large-scale backtesting (10 stocks x 3 periods).

학습된 전략 패턴 모듈 - 대규모 백테스트(10종목 x 3기간) 결과에서 도출된
최적 전략 매핑, 파라미터 조정, 성과 벤치마크를 제공합니다.

Features:
    - REGIME_STRATEGY_MAP: Per-regime ranked strategy recommendations with confidence
    - STRATEGY_PARAM_ADJUSTMENTS: Regime-specific parameter tuning per strategy
    - PERFORMANCE_BENCHMARKS: Expected Sharpe, return, and MDD per regime
    - KEY_INSIGHTS: Human-readable summary of learned patterns
    - Convenience functions: getRecommendedStrategies, getAdjustedParams, getBenchmark

Usage:
    from tradix.advisor.learnedPatterns import (
        getRecommendedStrategies, getAdjustedParams, getBenchmark
    )

    strategies = getRecommendedStrategies(MarketRegime.UPTREND)
    params = getAdjustedParams("SMA Cross", MarketRegime.UPTREND)
    benchmark = getBenchmark(MarketRegime.UPTREND)
"""

from typing import Dict, List, Tuple
from tradix.advisor.marketClassifier import MarketRegime


REGIME_STRATEGY_MAP: Dict[MarketRegime, List[Tuple[str, float, str]]] = {
    MarketRegime.STRONG_UPTREND: [
        ("MACD", 0.95, "강한 상승추세에서 모멘텀 포착에 효과적"),
        ("SMA Cross", 0.85, "골든크로스 신호가 잘 작동"),
        ("Breakout", 0.80, "신고가 돌파 전략 유효"),
    ],

    MarketRegime.UPTREND: [
        ("SMA Cross", 0.90, "상승장에서 가장 안정적인 성과 (43%)"),
        ("Trend Filter", 0.85, "200MA 필터로 잘못된 진입 방지"),
        ("Mean Reversion", 0.75, "조정 시 매수 기회 포착"),
    ],

    MarketRegime.SIDEWAYS: [
        ("SMA Cross", 0.85, "횡보장에서도 60% 최고 성과"),
        ("MACD", 0.75, "단기 추세 포착"),
        ("Breakout", 0.70, "돌파 시 추세 시작 포착"),
    ],

    MarketRegime.DOWNTREND: [
        ("Trend Filter", 0.95, "하락장에서 50% 최고 성과, 손실 방어"),
        ("Mean Reversion", 0.80, "과매도 구간 반등 포착"),
        ("Breakout", 0.70, "바닥 돌파 시 진입"),
    ],

    MarketRegime.STRONG_DOWNTREND: [
        ("Mean Reversion", 0.95, "강한 하락장에서 유일한 양수 샤프"),
        ("Trend Filter", 0.85, "매수 금지로 손실 방어"),
        ("Breakout", 0.60, "바닥 확인 후 진입"),
    ],

    MarketRegime.HIGH_VOLATILITY: [
        ("MACD", 0.85, "고변동성에서 33% 최고 성과"),
        ("SMA Cross", 0.80, "추세 전환 포착"),
        ("Mean Reversion", 0.75, "극단적 움직임 후 회귀"),
        ("Breakout", 0.70, "변동성 돌파"),
    ],
}


STRATEGY_PARAM_ADJUSTMENTS: Dict[str, Dict[MarketRegime, Dict[str, any]]] = {
    "SMA Cross": {
        MarketRegime.STRONG_UPTREND: {"fastPeriod": 5, "slowPeriod": 15},
        MarketRegime.UPTREND: {"fastPeriod": 10, "slowPeriod": 30},
        MarketRegime.SIDEWAYS: {"fastPeriod": 10, "slowPeriod": 20},
        MarketRegime.DOWNTREND: {"fastPeriod": 15, "slowPeriod": 40},
        MarketRegime.HIGH_VOLATILITY: {"fastPeriod": 5, "slowPeriod": 20},
    },

    "MACD": {
        MarketRegime.STRONG_UPTREND: {"fastPeriod": 8, "slowPeriod": 17, "signalPeriod": 9},
        MarketRegime.UPTREND: {"fastPeriod": 12, "slowPeriod": 26, "signalPeriod": 9},
        MarketRegime.HIGH_VOLATILITY: {"fastPeriod": 10, "slowPeriod": 20, "signalPeriod": 7},
    },

    "Breakout": {
        MarketRegime.STRONG_UPTREND: {"entryPeriod": 10, "exitPeriod": 5},
        MarketRegime.DOWNTREND: {"entryPeriod": 20, "exitPeriod": 10},
        MarketRegime.HIGH_VOLATILITY: {"entryPeriod": 15, "exitPeriod": 7},
    },

    "Mean Reversion": {
        MarketRegime.SIDEWAYS: {"bbPeriod": 20, "bbStd": 2.0, "rsiOversold": 30},
        MarketRegime.DOWNTREND: {"bbPeriod": 20, "bbStd": 2.5, "rsiOversold": 25},
        MarketRegime.HIGH_VOLATILITY: {"bbPeriod": 15, "bbStd": 2.5, "rsiOversold": 20},
    },

    "Trend Filter": {
        MarketRegime.UPTREND: {"trendPeriod": 200, "useAdxFilter": True},
        MarketRegime.DOWNTREND: {"trendPeriod": 200, "useAdxFilter": False},
    },
}


PERFORMANCE_BENCHMARKS = {
    "strong_uptrend": {
        "expected_sharpe": 1.5,
        "expected_return": 50.0,
        "max_acceptable_mdd": -30.0,
    },
    "uptrend": {
        "expected_sharpe": 0.8,
        "expected_return": 20.0,
        "max_acceptable_mdd": -25.0,
    },
    "sideways": {
        "expected_sharpe": 0.4,
        "expected_return": 5.0,
        "max_acceptable_mdd": -15.0,
    },
    "downtrend": {
        "expected_sharpe": 0.0,
        "expected_return": 0.0,
        "max_acceptable_mdd": -20.0,
    },
    "strong_downtrend": {
        "expected_sharpe": -0.2,
        "expected_return": -10.0,
        "max_acceptable_mdd": -25.0,
    },
    "high_volatility": {
        "expected_sharpe": 0.5,
        "expected_return": 15.0,
        "max_acceptable_mdd": -35.0,
    },
}


KEY_INSIGHTS = """
=== 학습된 핵심 인사이트 ===

1. 상승장 (Uptrend)
   - SMA Cross가 43%로 가장 빈번하게 최고 성과
   - 단순한 골든크로스 전략이 의외로 강력함
   - Trend Filter는 안정성 제공

2. 하락장 (Downtrend)
   - Trend Filter가 50%로 압도적 (손실 방어)
   - 하락장에서는 "안 잃는 것"이 최고 전략
   - Mean Reversion은 반등 포착에 유용

3. 횡보장 (Sideways)
   - 역시 SMA Cross가 60%로 우세
   - 횡보장에서도 작은 추세는 있음
   - 과도한 복잡한 전략보다 단순함이 유리

4. 고변동성 (High Volatility)
   - MACD가 33%로 가장 효과적
   - 모멘텀 지표가 변동성 환경에서 강점
   - Tesla 같은 종목에서 SMA Cross도 2.71 샤프 달성

5. 강한 상승장 (Strong Uptrend)
   - MACD가 100% 최고 성과
   - 강한 추세에서는 모멘텀 추종이 핵심

6. 강한 하락장 (Strong Downtrend)
   - Mean Reversion이 유일한 양수 샤프 (0.43)
   - 과매도 반등 전략만이 수익 가능

=== 실전 적용 가이드 ===

1. 먼저 MarketClassifier로 현재 시장 상황 파악
2. REGIME_STRATEGY_MAP에서 추천 전략 확인
3. STRATEGY_PARAM_ADJUSTMENTS로 파라미터 조정
4. Walk-Forward로 과적합 검증 후 실전 적용
"""


def getRecommendedStrategies(regime: MarketRegime) -> List[Tuple[str, float, str]]:
    """Retrieve ranked strategy recommendations for a given market regime.

    Returns a list of (strategy_name, confidence, rationale) tuples ordered
    by empirically observed effectiveness from large-scale backtesting.

    주어진 시장 레짐에 대해 백테스트 결과 기반으로 순위화된 전략 추천 목록을 반환합니다.

    Args:
        regime: Target MarketRegime to look up recommendations for
            (추천을 조회할 대상 시장 레짐).

    Returns:
        List of (strategy_name, confidence, rationale) tuples sorted by
        confidence descending. Returns an empty list if the regime is not
        found in the mapping (신뢰도 내림차순 정렬된 튜플 리스트).

    Example:
        >>> strategies = getRecommendedStrategies(MarketRegime.UPTREND)
        >>> top_strategy, confidence, reason = strategies[0]
        >>> print(f"{top_strategy}: {confidence:.0%}")
        SMA Cross: 90%
    """
    return REGIME_STRATEGY_MAP.get(regime, [])


def getAdjustedParams(strategyName: str, regime: MarketRegime) -> Dict[str, any]:
    """Retrieve regime-adjusted strategy parameters for optimal performance.

    Returns parameter overrides tuned for the specific strategy-regime
    combination based on empirical backtesting results.

    백테스트 결과 기반으로 특정 전략-레짐 조합에 최적화된 파라미터 오버라이드를 반환합니다.

    Args:
        strategyName: Strategy name to look up (e.g., "SMA Cross", "MACD")
            (조회할 전략 이름).
        regime: Current MarketRegime for parameter adjustment
            (파라미터 조정 대상 시장 레짐).

    Returns:
        Dict of parameter overrides for the given strategy-regime pair.
        Returns an empty dict if no adjustments are defined
        (전략-레짐 조합의 파라미터 오버라이드 딕셔너리).

    Example:
        >>> params = getAdjustedParams("SMA Cross", MarketRegime.UPTREND)
        >>> print(params)
        {'fastPeriod': 10, 'slowPeriod': 30}
    """
    strategyParams = STRATEGY_PARAM_ADJUSTMENTS.get(strategyName, {})
    return strategyParams.get(regime, {})


def getBenchmark(regime: MarketRegime) -> Dict[str, float]:
    """Retrieve expected performance benchmarks for a given market regime.

    Returns expected Sharpe ratio, annualized return, and maximum acceptable
    drawdown derived from large-scale backtesting across 10 stocks and
    3 time periods.

    대규모 백테스트(10종목 x 3기간)에서 도출된 기대 샤프비율, 연율 수익률,
    최대 허용 낙폭을 반환합니다.

    Args:
        regime: Target MarketRegime for benchmark lookup
            (벤치마크를 조회할 대상 시장 레짐).

    Returns:
        Dict with keys 'expected_sharpe', 'expected_return', and
        'max_acceptable_mdd'. Returns an empty dict if the regime is
        not found (기대 성과 지표 딕셔너리).

    Example:
        >>> benchmark = getBenchmark(MarketRegime.UPTREND)
        >>> print(f"Expected Sharpe: {benchmark['expected_sharpe']}")
        Expected Sharpe: 0.8
    """
    return PERFORMANCE_BENCHMARKS.get(regime.value, {})
