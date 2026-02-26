"""Tradex Strategy Advisor Module.

Recommends optimal trading strategies for the current market regime based on
large-scale backtest-derived learned patterns, with optional automatic
backtesting and regime forecast integration.

전략 추천 엔진 - 대규모 백테스트 결과에서 학습된 패턴을 활용하여
시장 상황에 최적화된 전략을 추천하며, 미래 레짐 예측 기반 선제적 추천을 지원합니다.

Features:
    - Market regime classification and strategy scoring
    - Learned-pattern-based strategy-regime mapping (REGIME_STRATEGY_MAP)
    - Automatic parameter adjustment per regime (STRATEGY_PARAM_ADJUSTMENTS)
    - Optional in-line backtesting of recommended strategies
    - Regime forecast integration for proactive strategy switching
    - Benchmark-aware performance expectation

Usage:
    from tradex.advisor import StrategyAdvisor

    advisor = StrategyAdvisor()
    result = advisor.recommend(data)
    print(result.summary())

    top = result.topStrategy()
    strategy = top.profile.createStrategy()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Callable, Any
import pandas as pd
import numpy as np

from tradex.advisor.marketClassifier import MarketClassifier, MarketRegime, MarketAnalysis
from tradex.advisor.learnedPatterns import (
    REGIME_STRATEGY_MAP,
    STRATEGY_PARAM_ADJUSTMENTS,
    PERFORMANCE_BENCHMARKS,
    getRecommendedStrategies,
    getAdjustedParams,
    getBenchmark,
)
from tradex.strategy.base import Strategy
from tradex.engine import BacktestEngine, BacktestResult
from tradex.datafeed.feed import DataFeed


@dataclass
class StrategyProfile:
    """Profile describing a trading strategy and its characteristics.

    전략의 특성을 기술하는 프로파일.

    Attributes:
        name: Human-readable strategy name (전략 이름).
        strategyClass: Strategy class reference (전략 클래스).
        defaultParams: Default parameter dict (기본 파라미터).
        suitableRegimes: List of regimes where this strategy performs well
                         (적합한 시장 레짐 목록).
        riskLevel: Risk level description (리스크 수준).
        description: Brief description of the strategy (전략 설명).
    """
    name: str
    strategyClass: Type[Strategy]
    defaultParams: Dict[str, Any]
    suitableRegimes: List[MarketRegime]
    riskLevel: str
    description: str

    def createStrategy(self, params: Dict[str, Any] = None) -> Strategy:
        """Create a Strategy instance with the given or default parameters.

        Args:
            params: Override parameters. Merged with defaultParams (파라미터 오버라이드).

        Returns:
            Configured Strategy instance.
        """
        strategy = self.strategyClass()
        useParams = {**self.defaultParams, **(params or {})}
        for key, value in useParams.items():
            setattr(strategy, key, value)
        return strategy


@dataclass
class StrategyRecommendation:
    """A single strategy recommendation with score and expected metrics.

    개별 전략 추천 결과.

    Attributes:
        profile: StrategyProfile for the recommended strategy (전략 프로파일).
        score: Recommendation score 0-1 (추천 점수).
        reason: Explanation for the recommendation (추천 사유).
        suggestedParams: Regime-adjusted parameters (레짐 조정 파라미터).
        expectedMetrics: Expected backtest metrics if available (기대 성과 지표).
        benchmark: Regime benchmark expectations (레짐 벤치마크).
    """
    profile: StrategyProfile
    score: float
    reason: str
    suggestedParams: Dict[str, Any]
    expectedMetrics: Dict[str, float] = field(default_factory=dict)
    benchmark: Dict[str, float] = field(default_factory=dict)

    def __repr__(self):
        return f"Recommendation({self.profile.name}, score={self.score:.2f})"

    def compareWithBenchmark(self) -> str:
        """Compare actual/expected metrics against the regime benchmark.

        Returns:
            Formatted comparison string or "No comparison data" if unavailable.
        """
        if not self.expectedMetrics or not self.benchmark:
            return "비교 데이터 없음"

        lines = []
        actualReturn = self.expectedMetrics.get('totalReturn', 0)
        expectedReturn = self.benchmark.get('expected_return', 0)
        returnDiff = actualReturn - expectedReturn

        actualSharpe = self.expectedMetrics.get('sharpeRatio', 0)
        expectedSharpe = self.benchmark.get('expected_sharpe', 0)
        sharpeDiff = actualSharpe - expectedSharpe

        lines.append(f"수익률: {actualReturn:.1f}% (기대: {expectedReturn:.1f}%, 차이: {returnDiff:+.1f}%)")
        lines.append(f"샤프비: {actualSharpe:.2f} (기대: {expectedSharpe:.2f}, 차이: {sharpeDiff:+.2f})")

        return "\n".join(lines)


@dataclass
class AdvisorResult:
    """Complete advisor result with market analysis and strategy recommendations.

    전체 전략 추천 결과 (시장 분석 및 백테스트 결과 포함).

    Attributes:
        marketAnalysis: MarketAnalysis of current market conditions (시장 분석 결과).
        recommendations: Sorted list of StrategyRecommendation (추천 전략 리스트).
        backtestResults: Dict of strategy name to BacktestResult (백테스트 결과).
    """
    marketAnalysis: MarketAnalysis
    recommendations: List[StrategyRecommendation]
    backtestResults: Dict[str, BacktestResult] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a formatted summary of market analysis and top 3 recommendations.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            "전략 추천 결과",
            "=" * 60,
            "",
            self.marketAnalysis.summary(),
            "",
            "-" * 60,
            "추천 전략 (상위 3개)",
            "-" * 60,
        ]

        for i, rec in enumerate(self.recommendations[:3], 1):
            lines.append(f"\n{i}. {rec.profile.name} (점수: {rec.score:.2f})")
            lines.append(f"   리스크: {rec.profile.riskLevel}")
            lines.append(f"   사유: {rec.reason}")

            if rec.profile.name in self.backtestResults:
                result = self.backtestResults[rec.profile.name]
                m = result.metrics
                lines.append(f"   백테스트: 수익률 {m.get('totalReturn', 0):.1f}%, "
                           f"샤프 {m.get('sharpeRatio', 0):.2f}, "
                           f"MDD {m.get('maxDrawdown', 0):.1f}%")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def topStrategy(self) -> Optional[StrategyRecommendation]:
        """Return the highest-scored strategy recommendation.

        Returns:
            Top StrategyRecommendation, or None if no recommendations exist.
        """
        return self.recommendations[0] if self.recommendations else None


class StrategyAdvisor:
    """Strategy recommendation engine based on learned patterns and market analysis.

    Analyzes current market conditions, scores registered strategies using
    learned regime-strategy mappings, optionally backtests top recommendations,
    and provides regime-forecast-aware proactive recommendations.

    전략 추천 엔진 - 시장 상황을 분석하고, 학습된 레짐-전략 매핑을 기반으로
    전략 점수를 산출하며, 필요 시 자동 백테스트를 수행합니다.

    Attributes:
        classifier: MarketClassifier instance (시장 분류기).
        profiles: List of registered StrategyProfile instances (등록된 전략 목록).

    Example:
        >>> advisor = StrategyAdvisor()
        >>> result = advisor.recommend(data)
        >>> print(result.summary())
        >>> strategy = result.topStrategy().profile.createStrategy()
    """

    def __init__(self):
        self.classifier = MarketClassifier()
        self.profiles: List[StrategyProfile] = []
        self._registerDefaultStrategies()

    def _registerDefaultStrategies(self):
        """Register the default set of built-in strategies."""
        from tradex.examples import (
            SmaCrossStrategy,
            MacdStrategy,
            BreakoutStrategy,
            MeanReversionStrategy,
            TrendFilterStrategy,
        )

        self.registerStrategy(StrategyProfile(
            name="SMA Cross",
            strategyClass=SmaCrossStrategy,
            defaultParams={'fastPeriod': 10, 'slowPeriod': 30},
            suitableRegimes=[MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND],
            riskLevel="중간",
            description="골든크로스/데드크로스 기반 추세추종",
        ))

        self.registerStrategy(StrategyProfile(
            name="MACD",
            strategyClass=MacdStrategy,
            defaultParams={'fastPeriod': 12, 'slowPeriod': 26, 'signalPeriod': 9},
            suitableRegimes=[MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND],
            riskLevel="중간",
            description="MACD 시그널 교차 기반",
        ))

        self.registerStrategy(StrategyProfile(
            name="Breakout",
            strategyClass=BreakoutStrategy,
            defaultParams={'entryPeriod': 20, 'exitPeriod': 10},
            suitableRegimes=[MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND],
            riskLevel="높음",
            description="Donchian 채널 돌파 전략",
        ))

        self.registerStrategy(StrategyProfile(
            name="Mean Reversion",
            strategyClass=MeanReversionStrategy,
            defaultParams={'bbPeriod': 20, 'rsiPeriod': 14},
            suitableRegimes=[MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY],
            riskLevel="낮음",
            description="볼린저밴드 + RSI 평균회귀",
        ))

        self.registerStrategy(StrategyProfile(
            name="Trend Filter",
            strategyClass=TrendFilterStrategy,
            defaultParams={'trendPeriod': 200, 'fastPeriod': 10, 'slowPeriod': 30},
            suitableRegimes=[MarketRegime.UPTREND, MarketRegime.SIDEWAYS],
            riskLevel="낮음",
            description="200MA 트렌드 필터 + SMA 교차",
        ))

    def registerStrategy(self, profile: StrategyProfile):
        """Register a strategy profile for recommendation consideration.

        Args:
            profile: StrategyProfile to register (등록할 전략 프로파일).
        """
        self.profiles.append(profile)

    def recommend(
        self,
        data: DataFeed,
        runBacktest: bool = True,
        initialCash: float = 10_000_000,
    ) -> AdvisorResult:
        """Recommend strategies for the current market conditions.

        Args:
            data: DataFeed instance with loaded data (데이터 피드).
            runBacktest: Whether to backtest top 3 recommendations (백테스트 실행 여부).
            initialCash: Initial capital for backtesting (초기 자본금).

        Returns:
            AdvisorResult with market analysis, recommendations, and optional
            backtest results.
        """
        if not data._loaded:
            data.load()

        df = data.toDataFrame()
        analysis = self.classifier.analyze(df)

        recommendations = self._scoreStrategies(analysis)
        recommendations.sort(key=lambda x: x.score, reverse=True)

        backtestResults = {}
        if runBacktest:
            for rec in recommendations[:3]:
                data._index = 0

                strategy = rec.profile.createStrategy(rec.suggestedParams)
                engine = BacktestEngine(
                    data=data,
                    strategy=strategy,
                    initialCash=initialCash,
                )
                result = engine.run(verbose=False)
                backtestResults[rec.profile.name] = result

                rec.expectedMetrics = {
                    'totalReturn': result.metrics.get('totalReturn', 0),
                    'sharpeRatio': result.metrics.get('sharpeRatio', 0),
                    'maxDrawdown': result.metrics.get('maxDrawdown', 0),
                }

        return AdvisorResult(
            marketAnalysis=analysis,
            recommendations=recommendations,
            backtestResults=backtestResults,
        )

    def _scoreStrategies(self, analysis: MarketAnalysis) -> List[StrategyRecommendation]:
        """Score all registered strategies against the current market analysis.

        Args:
            analysis: MarketAnalysis of current conditions.

        Returns:
            List of StrategyRecommendation with scores and benchmarks.
        """
        recommendations = []
        benchmark = getBenchmark(analysis.regime)

        for profile in self.profiles:
            score, reason, params = self._calcScore(profile, analysis)

            recommendations.append(StrategyRecommendation(
                profile=profile,
                score=score,
                reason=reason,
                suggestedParams=params,
                benchmark=benchmark,
            ))

        return recommendations

    def _calcScore(
        self,
        profile: StrategyProfile,
        analysis: MarketAnalysis
    ) -> tuple:
        """Calculate score for a single strategy using learned patterns.

        Uses REGIME_STRATEGY_MAP from large-scale backtest results to score
        strategy-regime fitness, then adjusts for trend strength and risk level.

        Args:
            profile: Strategy profile to score.
            analysis: Current market analysis.

        Returns:
            Tuple of (score, reason_string, adjusted_params).
        """
        score = 0.3
        reasons = []
        params = profile.defaultParams.copy()

        learnedStrategies = getRecommendedStrategies(analysis.regime)
        learnedStrategyNames = [s[0] for s in learnedStrategies]

        for strategyName, confidence, reason in learnedStrategies:
            if profile.name == strategyName:
                score += confidence * 0.5
                reasons.append(reason)
                break

        adjustedParams = getAdjustedParams(profile.name, analysis.regime)
        if adjustedParams:
            params.update(adjustedParams)
            reasons.append("학습된 최적 파라미터 적용")

        if analysis.regime in profile.suitableRegimes:
            score += 0.1
            reasons.append(f"{analysis.regime.value} 시장에 등록된 적합 전략")

        if analysis.adx > 30:
            if profile.name in ["SMA Cross", "MACD", "Breakout"]:
                score += 0.05
                reasons.append(f"강한 추세 (ADX={analysis.adx:.0f})")
        elif analysis.adx < 20:
            if profile.name == "Mean Reversion":
                score += 0.05
                reasons.append(f"약한 추세 (ADX={analysis.adx:.0f})")

        if profile.riskLevel == "낮음":
            score += 0.02
        elif profile.riskLevel == "높음":
            score -= 0.02

        score = max(0, min(1, score))
        reason = "; ".join(reasons) if reasons else "기본 점수"

        return score, reason, params

    def analyzeAndCompare(
        self,
        data: DataFeed,
        initialCash: float = 10_000_000,
    ) -> pd.DataFrame:
        """Backtest all registered strategies and produce a comparison DataFrame.

        Args:
            data: DataFeed instance (데이터 피드).
            initialCash: Initial capital for each backtest (초기 자본금).

        Returns:
            DataFrame sorted by Sharpe ratio with columns: strategy, riskLevel,
            totalReturn, sharpeRatio, maxDrawdown, winRate, totalTrades
            (전략별 비교 결과 DataFrame).
        """
        if not data._loaded:
            data.load()

        results = []

        for profile in self.profiles:
            data._index = 0
            strategy = profile.createStrategy()

            engine = BacktestEngine(
                data=data,
                strategy=strategy,
                initialCash=initialCash,
            )

            result = engine.run(verbose=False)
            m = result.metrics

            results.append({
                'strategy': profile.name,
                'riskLevel': profile.riskLevel,
                'totalReturn': m.get('totalReturn', 0),
                'sharpeRatio': m.get('sharpeRatio', 0),
                'maxDrawdown': m.get('maxDrawdown', 0),
                'winRate': m.get('winRate', 0),
                'totalTrades': m.get('totalTrades', 0),
            })

        df = pd.DataFrame(results)
        df = df.sort_values('sharpeRatio', ascending=False).reset_index(drop=True)

        return df

    def recommendWithForecast(
        self,
        data: DataFeed,
        horizonDays: List[int] = None,
        runBacktest: bool = True,
        initialCash: float = 10_000_000,
    ) -> Dict[str, any]:
        """Recommend strategies using both current analysis and regime forecasting.

        Combines current market analysis with future regime prediction to provide
        proactive strategy switching recommendations.

        Args:
            data: DataFeed instance (데이터 피드).
            horizonDays: Forecast horizon list in days (default [5, 10, 20])
                         (예측 기간 리스트).
            runBacktest: Whether to backtest current recommendations (백테스트 실행 여부).
            initialCash: Initial capital for backtesting (초기 자본금).

        Returns:
            Dict with keys: 'current' (regime, top strategy, recommendations),
            'forecast' (summary, risk, transition signals), 'futureRecommendations'
            (per-horizon strategy suggestions), 'transitionMatrix', 'regimeStats'.
        """
        from tradex.advisor.regimeForecaster import RegimeForecaster

        if horizonDays is None:
            horizonDays = [5, 10, 20]

        if not data._loaded:
            data.load()

        df = data.toDataFrame()

        forecaster = RegimeForecaster(classifier=self.classifier)
        forecaster.fit(df)
        forecast = forecaster.predict(df, horizonDays=horizonDays)

        currentResult = self.recommend(data, runBacktest=runBacktest, initialCash=initialCash)

        futureRecommendations = {}
        for days, regime, confidence in forecast.forecasts:
            strategies = getRecommendedStrategies(regime)
            benchmark = getBenchmark(regime)

            futureRecommendations[days] = {
                'regime': regime,
                'regimeConfidence': confidence,
                'strategies': [
                    {
                        'name': name,
                        'score': score,
                        'reason': reason,
                        'params': getAdjustedParams(name, regime),
                    }
                    for name, score, reason in strategies[:3]
                ],
                'benchmark': benchmark,
            }

        regimeChange = (
            forecast.forecasts[-1][1] != forecast.currentRegime
            if forecast.forecasts else False
        )

        return {
            'current': {
                'regime': currentResult.marketAnalysis.regime,
                'confidence': currentResult.marketAnalysis.confidence,
                'topStrategy': currentResult.topStrategy().profile.name if currentResult.topStrategy() else None,
                'recommendations': currentResult.recommendations[:3],
            },
            'forecast': {
                'summary': forecast.summary,
                'riskLevel': forecast.riskLevel,
                'transitionSignals': forecast.transitionSignals,
                'regimeChangeExpected': regimeChange,
            },
            'futureRecommendations': futureRecommendations,
            'transitionMatrix': forecaster.getTransitionMatrix(),
            'regimeStats': forecaster.getRegimeStats(),
        }
