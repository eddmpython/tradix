"""Tradex Regime-Factor Integration Module.

Bridges Tradex's learned patterns and regime forecasting with quantitative
factor analysis, enabling regime-conditional factor exposure optimization
and adaptive rebalancing.

레짐-팩터 통합 모듈 - Tradex의 학습된 패턴과 레짐 예측을 팩터 분석과 연결하여,
레짐별 팩터 효과를 분석하고 적응형 팩터 전략을 제공합니다.

Features:
    - Per-regime factor premium analysis
    - Optimal factor exposure recommendation by market regime
    - Regime-transition-based rebalancing suggestions
    - Adaptive factor strategy with built-in backtesting
    - Integration with Tradex's REGIME_STRATEGY_MAP for synergy

Usage:
    from tradex.quant.integration import RegimeFactorIntegration

    integration = RegimeFactorIntegration()
    integration.fit(price_data, factor_returns)
    exposures = integration.getOptimalExposures(current_regime)
    rebalance = integration.suggestRebalancing(forecast)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np

from tradex.advisor.marketClassifier import MarketClassifier, MarketRegime, MarketAnalysis
from tradex.advisor.regimeForecaster import RegimeForecaster, RegimeForecast
from tradex.advisor.learnedPatterns import REGIME_STRATEGY_MAP
from tradex.quant.factor import FactorAnalyzer, FactorResult, Factor


@dataclass
class RegimeFactorProfile:
    """Factor profile for a specific market regime.

    특정 시장 레짐에 대한 팩터 프로파일.

    Attributes:
        regime: The market regime this profile describes (시장 레짐).
        effectiveFactors: List of factor names that are effective in this regime
                          (해당 레짐에서 유효한 팩터 목록).
        factorPremiums: Dict of factor name to annualized premium (팩터별 연율 프리미엄).
        recommendedExposures: Normalized factor weights for this regime
                              (레짐별 권장 팩터 노출 비중).
        confidence: R-squared of the factor model in this regime (모델 신뢰도).
        notes: Human-readable notes about this regime-factor relationship (참고 사항).
    """
    regime: MarketRegime
    effectiveFactors: List[str]
    factorPremiums: Dict[str, float]
    recommendedExposures: Dict[str, float]
    confidence: float
    notes: str


@dataclass
class IntegratedSignal:
    """Integrated signal combining regime prediction and factor exposure.

    레짐 예측과 팩터 노출을 결합한 통합 시그널.

    Attributes:
        date: Signal timestamp (시그널 시점).
        currentRegime: Currently detected market regime (현재 감지된 레짐).
        predictedRegime: Forecasted future regime (예측된 미래 레짐).
        regimeConfidence: Confidence of regime prediction 0-1 (레짐 예측 신뢰도).
        factorExposures: Recommended factor exposures (권장 팩터 노출).
        strategyWeight: Win probability from learned strategy map (전략 승률).
        overallSignal: Combined signal score 0-1 (종합 시그널 점수).
        recommendation: Human-readable recommendation text (추천 텍스트).
    """
    date: pd.Timestamp
    currentRegime: MarketRegime
    predictedRegime: MarketRegime
    regimeConfidence: float
    factorExposures: Dict[str, float]
    strategyWeight: float
    overallSignal: float
    recommendation: str


@dataclass
class AdaptiveFactorResult:
    """Result of an adaptive factor strategy backtest.

    적응형 팩터 전략 백테스트 결과.

    Attributes:
        signals: List of generated IntegratedSignal instances (생성된 통합 시그널 리스트).
        regimeFactorMap: Mapping of regime to its factor profile (레짐별 팩터 프로파일 맵).
        performanceByRegime: Per-regime performance statistics (레짐별 성과 통계).
        summary: Formatted summary string (결과 요약 문자열).
    """
    signals: List[IntegratedSignal]
    regimeFactorMap: Dict[MarketRegime, RegimeFactorProfile]
    performanceByRegime: Dict[MarketRegime, Dict[str, float]]
    summary: str


class RegimeFactorIntegration:
    """Integrates Tradex regime forecasting with quantitative factor analysis.

    Learns per-regime factor effectiveness from historical data, recommends
    optimal factor exposures for the current or predicted regime, and suggests
    rebalancing actions when regime transitions are forecasted.

    Tradex의 학습된 패턴과 레짐 예측을 팩터 분석과 연결하여,
    레짐별 최적 팩터 노출을 제안하고 리밸런싱을 안내합니다.

    Attributes:
        classifier: MarketClassifier instance (시장 분류기).
        forecaster: RegimeForecaster instance (레짐 예측기).
        factorAnalyzer: FactorAnalyzer instance (팩터 분석기).
        regimeFactorProfiles: Learned per-regime factor profiles (레짐별 팩터 프로파일).
        REGIME_FACTOR_INSIGHTS: Domain-knowledge mapping of effective/avoid factors
                                per regime (레짐별 팩터 인사이트).

    Example:
        >>> integration = RegimeFactorIntegration()
        >>> integration.fit(price_data, factor_returns)
        >>> exposures = integration.getOptimalExposures(MarketRegime.UPTREND)
        >>> rebalance = integration.suggestRebalancing(forecast)
    """

    REGIME_FACTOR_INSIGHTS = {
        MarketRegime.STRONG_UPTREND: {
            'effective': ['momentum', 'quality'],
            'avoid': ['volatility'],
            'note': '강한 상승장에서는 모멘텀이 가장 효과적',
        },
        MarketRegime.UPTREND: {
            'effective': ['momentum', 'size', 'quality'],
            'avoid': [],
            'note': '상승장에서는 소형주 + 모멘텀 조합이 유효',
        },
        MarketRegime.SIDEWAYS: {
            'effective': ['value', 'quality'],
            'avoid': ['momentum'],
            'note': '횡보장에서는 가치주가 상대적 우위',
        },
        MarketRegime.DOWNTREND: {
            'effective': ['quality', 'volatility'],
            'avoid': ['size', 'momentum'],
            'note': '하락장에서는 저변동성 + 우량주 방어',
        },
        MarketRegime.STRONG_DOWNTREND: {
            'effective': ['volatility'],
            'avoid': ['momentum', 'size'],
            'note': '강한 하락장에서는 저변동성만 유효',
        },
        MarketRegime.HIGH_VOLATILITY: {
            'effective': ['quality'],
            'avoid': ['size', 'momentum'],
            'note': '고변동성에서는 우량주 집중',
        },
    }

    def __init__(
        self,
        classifier: MarketClassifier = None,
        forecaster: RegimeForecaster = None,
        factorAnalyzer: FactorAnalyzer = None,
    ):
        """Initialize the regime-factor integration engine.

        Args:
            classifier: MarketClassifier instance. Defaults to a new instance
                        (시장 분류기, 기본값은 새 인스턴스).
            forecaster: RegimeForecaster instance. Defaults to a new instance
                        (레짐 예측기, 기본값은 새 인스턴스).
            factorAnalyzer: FactorAnalyzer instance. Defaults to a new instance
                            (팩터 분석기, 기본값은 새 인스턴스).
        """
        self.classifier = classifier or MarketClassifier()
        self.forecaster = forecaster or RegimeForecaster()
        self.factorAnalyzer = factorAnalyzer or FactorAnalyzer()

        self.regimeFactorProfiles: Dict[MarketRegime, RegimeFactorProfile] = {}
        self._fitted = False

    def fit(
        self,
        priceData: pd.DataFrame,
        factorReturns: pd.DataFrame,
        assetReturns: pd.Series = None,
    ) -> 'RegimeFactorIntegration':
        """Learn per-regime factor effectiveness from historical data.

        Args:
            priceData: OHLCV DataFrame for regime classification and forecaster fitting
                       (OHLCV 데이터).
            factorReturns: DataFrame of daily factor returns (팩터 수익률 DataFrame).
            assetReturns: Daily asset return series. If None, computed from priceData
                          close prices (자산 수익률, None이면 종가에서 계산).

        Returns:
            Self for method chaining.
        """
        self.forecaster.fit(priceData)
        self.factorAnalyzer.setFactors(factorReturns)

        if assetReturns is None:
            assetReturns = priceData['close'].pct_change().dropna()

        historyDf = self.classifier.analyzeHistory(priceData)

        for regime in MarketRegime:
            regimeDates = historyDf[historyDf['regime'] == regime.value]['date']

            if len(regimeDates) < 20:
                self.regimeFactorProfiles[regime] = self._getDefaultProfile(regime)
                continue

            regimeReturns = assetReturns[assetReturns.index.isin(regimeDates)]
            regimeFactors = factorReturns[factorReturns.index.isin(regimeDates)]

            commonIdx = regimeReturns.index.intersection(regimeFactors.index)
            if len(commonIdx) < 20:
                self.regimeFactorProfiles[regime] = self._getDefaultProfile(regime)
                continue

            result = self.factorAnalyzer.analyze(
                regimeReturns.loc[commonIdx],
                regimeFactors.loc[commonIdx]
            )

            effectiveFactors = [
                name for name, exp in result.exposures.items()
                if exp.isSignificant() and exp.contribution > 0
            ]

            premiums = {
                name: exp.contribution
                for name, exp in result.exposures.items()
            }

            exposures = self._calcOptimalExposures(result, regime)

            insight = self.REGIME_FACTOR_INSIGHTS.get(regime, {})

            self.regimeFactorProfiles[regime] = RegimeFactorProfile(
                regime=regime,
                effectiveFactors=effectiveFactors or insight.get('effective', []),
                factorPremiums=premiums,
                recommendedExposures=exposures,
                confidence=result.rSquared,
                notes=insight.get('note', ''),
            )

        self._fitted = True
        return self

    def _getDefaultProfile(self, regime: MarketRegime) -> RegimeFactorProfile:
        """Return a default factor profile when insufficient data is available.

        Args:
            regime: The market regime.

        Returns:
            RegimeFactorProfile with domain-knowledge-based defaults.
        """
        insight = self.REGIME_FACTOR_INSIGHTS.get(regime, {})

        effectiveFactors = insight.get('effective', ['market'])
        exposures = {f: 0.2 for f in effectiveFactors}

        return RegimeFactorProfile(
            regime=regime,
            effectiveFactors=effectiveFactors,
            factorPremiums={f: 0.05 for f in effectiveFactors},
            recommendedExposures=exposures,
            confidence=0.5,
            notes=insight.get('note', '학습 데이터 부족'),
        )

    def _calcOptimalExposures(
        self,
        factorResult: FactorResult,
        regime: MarketRegime
    ) -> Dict[str, float]:
        """Compute optimal normalized factor exposures for a given regime.

        Args:
            factorResult: Factor analysis result for this regime's data.
            regime: The market regime to optimize for.

        Returns:
            Dict of factor name to normalized weight (0-1 range, summing to 1).
        """
        insight = self.REGIME_FACTOR_INSIGHTS.get(regime, {})
        avoidFactors = insight.get('avoid', [])
        effectiveFactors = insight.get('effective', [])

        exposures = {}

        for name, exp in factorResult.exposures.items():
            if name in avoidFactors:
                exposures[name] = max(0, exp.beta * 0.3)
            elif name in effectiveFactors:
                if exp.contribution > 0:
                    exposures[name] = min(1.0, abs(exp.beta) * 1.2)
                else:
                    exposures[name] = abs(exp.beta) * 0.5
            else:
                exposures[name] = abs(exp.beta) * 0.8

        total = sum(exposures.values())
        if total > 0:
            exposures = {k: v / total for k, v in exposures.items()}

        return exposures

    def getOptimalExposures(
        self,
        regime: MarketRegime
    ) -> Dict[str, float]:
        """Retrieve the recommended factor exposures for the given regime.

        Args:
            regime: The market regime to query (조회할 시장 레짐).

        Returns:
            Dict of factor name to recommended weight (권장 팩터 노출 비중).
        """
        if regime in self.regimeFactorProfiles:
            return self.regimeFactorProfiles[regime].recommendedExposures
        return self._getDefaultProfile(regime).recommendedExposures

    def getRegimeProfile(
        self,
        regime: MarketRegime
    ) -> RegimeFactorProfile:
        """Retrieve the factor profile for the given regime.

        Args:
            regime: The market regime to query (조회할 시장 레짐).

        Returns:
            RegimeFactorProfile for the specified regime (해당 레짐의 팩터 프로파일).
        """
        if regime in self.regimeFactorProfiles:
            return self.regimeFactorProfiles[regime]
        return self._getDefaultProfile(regime)

    def suggestRebalancing(
        self,
        forecast: RegimeForecast,
        currentExposures: Dict[str, float] = None,
        rebalanceThreshold: float = 0.1,
    ) -> Dict[str, any]:
        """Suggest factor exposure rebalancing based on regime forecast.

        Args:
            forecast: RegimeForecast from the regime forecaster (레짐 예측 결과).
            currentExposures: Current factor exposures. If None, uses the current
                              regime's recommended exposures (현재 팩터 노출).
            rebalanceThreshold: Minimum exposure difference to trigger rebalancing
                                (리밸런싱 임계값).

        Returns:
            Dict with keys: needRebalance, reason, currentRegime, predictedRegime,
            predictionConfidence, currentExposures, targetExposures,
            exposureChanges, riskLevel (리밸런싱 제안 딕셔너리).
        """
        currentProfile = self.getRegimeProfile(forecast.currentRegime)

        if not forecast.forecasts:
            return {
                'needRebalance': False,
                'reason': '예측 데이터 없음',
                'currentExposures': currentProfile.recommendedExposures,
                'targetExposures': currentProfile.recommendedExposures,
            }

        shortTermForecast = forecast.forecasts[0]
        predictedRegime = shortTermForecast[1]
        predictionConfidence = shortTermForecast[2]

        targetProfile = self.getRegimeProfile(predictedRegime)

        if currentExposures is None:
            currentExposures = currentProfile.recommendedExposures

        exposureDiffs = {}
        for factor in set(list(currentExposures.keys()) + list(targetProfile.recommendedExposures.keys())):
            current = currentExposures.get(factor, 0)
            target = targetProfile.recommendedExposures.get(factor, 0)
            exposureDiffs[factor] = target - current

        maxDiff = max(abs(d) for d in exposureDiffs.values()) if exposureDiffs else 0
        needRebalance = maxDiff > rebalanceThreshold and predictionConfidence > 0.4

        if forecast.currentRegime != predictedRegime:
            reason = f"레짐 전환 예상: {forecast.currentRegime.value} → {predictedRegime.value}"
        elif maxDiff > rebalanceThreshold:
            reason = f"팩터 노출 차이 과다: {maxDiff:.1%}"
        else:
            reason = "리밸런싱 불필요"

        return {
            'needRebalance': needRebalance,
            'reason': reason,
            'currentRegime': forecast.currentRegime.value,
            'predictedRegime': predictedRegime.value,
            'predictionConfidence': predictionConfidence,
            'currentExposures': currentExposures,
            'targetExposures': targetProfile.recommendedExposures,
            'exposureChanges': exposureDiffs,
            'riskLevel': forecast.riskLevel,
        }

    def generateSignals(
        self,
        priceData: pd.DataFrame,
        factorReturns: pd.DataFrame,
        lookbackPeriod: int = 60,
    ) -> List[IntegratedSignal]:
        """Generate integrated signals combining regime prediction and factor exposure.

        Args:
            priceData: OHLCV DataFrame (OHLCV 데이터).
            factorReturns: DataFrame of daily factor returns (팩터 수익률).
            lookbackPeriod: Minimum lookback period before signal generation starts
                            (시그널 생성 시작 전 최소 룩백 기간).

        Returns:
            List of IntegratedSignal instances (통합 시그널 리스트).
        """
        signals = []

        for i in range(lookbackPeriod, len(priceData) - 1):
            windowData = priceData.iloc[:i + 1]

            analysis = self.classifier.analyze(windowData)
            forecast = self.forecaster.predict(windowData, horizonDays=[5])

            currentRegime = analysis.regime
            predictedRegime = forecast.forecasts[0][1] if forecast.forecasts else currentRegime
            regimeConfidence = forecast.forecasts[0][2] if forecast.forecasts else 0.5

            exposures = self.getOptimalExposures(predictedRegime)

            strategyMap = REGIME_STRATEGY_MAP.get(currentRegime.value, {})
            if strategyMap:
                topStrategy = max(strategyMap.items(), key=lambda x: x[1]['winProbability'])
                strategyWeight = topStrategy[1]['winProbability']
            else:
                strategyWeight = 0.5

            bullishScore = 0
            if predictedRegime in [MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND]:
                bullishScore = 0.5 + regimeConfidence * 0.5
            elif predictedRegime in [MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND]:
                bullishScore = 0.5 - regimeConfidence * 0.5
            else:
                bullishScore = 0.5

            momentumExposure = exposures.get('momentum', 0)
            overallSignal = bullishScore * 0.6 + momentumExposure * 0.4

            if overallSignal > 0.65:
                recommendation = "강한 매수"
            elif overallSignal > 0.55:
                recommendation = "매수"
            elif overallSignal < 0.35:
                recommendation = "강한 매도"
            elif overallSignal < 0.45:
                recommendation = "매도"
            else:
                recommendation = "관망"

            signals.append(IntegratedSignal(
                date=priceData.index[i],
                currentRegime=currentRegime,
                predictedRegime=predictedRegime,
                regimeConfidence=regimeConfidence,
                factorExposures=exposures,
                strategyWeight=strategyWeight,
                overallSignal=overallSignal,
                recommendation=recommendation,
            ))

        return signals

    def summary(self) -> str:
        """Generate a human-readable summary of the regime-factor integration analysis.

        Returns:
            Formatted multi-line string with per-regime factor profiles and
            learned pattern connections (통합 분석 요약 문자열).
        """
        lines = [
            "=" * 60,
            "레짐-팩터 통합 분석",
            "=" * 60,
            "",
        ]

        for regime in MarketRegime:
            profile = self.getRegimeProfile(regime)
            lines.extend([
                f"[{regime.value}]",
                f"  효과적 팩터: {', '.join(profile.effectiveFactors)}",
                f"  신뢰도: {profile.confidence:.1%}",
                f"  노트: {profile.notes}",
                "",
            ])

        strategyConnection = [
            "",
            "-" * 60,
            "학습된 패턴과의 연결",
            "-" * 60,
        ]

        for regime, strategies in REGIME_STRATEGY_MAP.items():
            if strategies:
                topStrategy = max(strategies.items(), key=lambda x: x[1].get('winProbability', 0))
                strategyConnection.append(
                    f"{regime}: {topStrategy[0]} (승률 {topStrategy[1].get('winProbability', 0):.0%})"
                )

        lines.extend(strategyConnection)
        lines.append("=" * 60)

        return "\n".join(lines)


class AdaptiveFactorStrategy:
    """Adaptive factor strategy that dynamically adjusts factor exposures by regime.

    Uses RegimeFactorIntegration to predict regime transitions and rebalance
    factor exposures accordingly. Includes backtesting capability and real-time
    position recommendation.

    적응형 팩터 전략 - 레짐 예측에 따라 팩터 노출을 동적으로 조정하며,
    백테스트 및 실시간 포지션 제안 기능을 제공합니다.

    Attributes:
        integration: RegimeFactorIntegration instance (레짐-팩터 통합 객체).
        rebalanceFreq: Rebalancing frequency in trading days (리밸런싱 주기, 거래일).
        maxExposure: Maximum allowed factor exposure (최대 팩터 노출).

    Example:
        >>> strategy = AdaptiveFactorStrategy(integration, rebalanceFreq=20)
        >>> result = strategy.backtest(price_data, factor_returns)
        >>> print(result.summary)
        >>> position = strategy.getCurrentPosition(price_data)
    """

    def __init__(
        self,
        integration: RegimeFactorIntegration,
        rebalanceFreq: int = 20,
        maxExposure: float = 1.5,
    ):
        """Initialize the adaptive factor strategy.

        Args:
            integration: RegimeFactorIntegration instance (레짐-팩터 통합 객체).
            rebalanceFreq: Rebalancing frequency in trading days (리밸런싱 주기).
            maxExposure: Maximum factor exposure cap (최대 팩터 노출).
        """
        self.integration = integration
        self.rebalanceFreq = rebalanceFreq
        self.maxExposure = maxExposure

    def backtest(
        self,
        priceData: pd.DataFrame,
        factorReturns: pd.DataFrame,
        initialCapital: float = 10_000_000,
    ) -> AdaptiveFactorResult:
        """Backtest the adaptive factor strategy over historical data.

        Args:
            priceData: OHLCV DataFrame (OHLCV 데이터).
            factorReturns: DataFrame of daily factor returns (팩터 수익률).
            initialCapital: Initial capital for the backtest (초기 자본금).

        Returns:
            AdaptiveFactorResult with signals, per-regime performance, and summary.
        """
        signals = self.integration.generateSignals(priceData, factorReturns)

        performanceByRegime: Dict[MarketRegime, Dict[str, List[float]]] = {
            r: {'returns': [], 'signals': []}
            for r in MarketRegime
        }

        for signal in signals:
            nextDayReturn = priceData['close'].pct_change().shift(-1).loc[signal.date]
            if pd.notna(nextDayReturn):
                positionReturn = (signal.overallSignal - 0.5) * 2 * nextDayReturn

                performanceByRegime[signal.currentRegime]['returns'].append(positionReturn)
                performanceByRegime[signal.currentRegime]['signals'].append(signal.overallSignal)

        regimePerformance = {}
        for regime, data in performanceByRegime.items():
            if data['returns']:
                returns = np.array(data['returns'])
                regimePerformance[regime] = {
                    'avgReturn': np.mean(returns),
                    'totalReturn': np.sum(returns),
                    'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                    'winRate': np.mean([r > 0 for r in returns]),
                    'nSignals': len(returns),
                }

        summary = self._buildSummary(signals, regimePerformance)

        return AdaptiveFactorResult(
            signals=signals,
            regimeFactorMap=self.integration.regimeFactorProfiles,
            performanceByRegime=regimePerformance,
            summary=summary,
        )

    def _buildSummary(
        self,
        signals: List[IntegratedSignal],
        regimePerformance: Dict[MarketRegime, Dict[str, float]]
    ) -> str:
        """Build a formatted backtest result summary string.

        Args:
            signals: List of generated integrated signals.
            regimePerformance: Per-regime performance statistics dict.

        Returns:
            Multi-line formatted summary string.
        """
        lines = [
            "=" * 60,
            "적응형 팩터 전략 백테스트 결과",
            "=" * 60,
            f"총 시그널 수: {len(signals)}",
            "",
            "-" * 60,
            "레짐별 성과",
            "-" * 60,
            f"{'레짐':<20} {'평균수익':>10} {'승률':>10} {'샤프':>10}",
            "-" * 60,
        ]

        for regime, perf in regimePerformance.items():
            if perf['nSignals'] > 0:
                lines.append(
                    f"{regime.value:<20} {perf['avgReturn']:>9.2%} "
                    f"{perf['winRate']:>9.1%} {perf['sharpe']:>10.2f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def getCurrentPosition(
        self,
        priceData: pd.DataFrame,
    ) -> Dict[str, any]:
        """Get current position recommendation based on the latest data.

        Args:
            priceData: Latest OHLCV DataFrame (최신 OHLCV 데이터).

        Returns:
            Dict with regime, regimeConfidence, predictedRegime, factorExposures,
            needRebalance, rebalanceReason, recommendedStrategies, riskLevel
            (포지션 제안 딕셔너리).
        """
        forecast = self.integration.forecaster.predict(priceData)
        exposures = self.integration.getOptimalExposures(forecast.currentRegime)
        rebalance = self.integration.suggestRebalancing(forecast)

        strategyMap = REGIME_STRATEGY_MAP.get(forecast.currentRegime.value, {})

        return {
            'regime': forecast.currentRegime.value,
            'regimeConfidence': forecast.currentConfidence,
            'predictedRegime': forecast.forecasts[0][1].value if forecast.forecasts else None,
            'factorExposures': exposures,
            'needRebalance': rebalance['needRebalance'],
            'rebalanceReason': rebalance['reason'],
            'recommendedStrategies': list(strategyMap.keys())[:3],
            'riskLevel': forecast.riskLevel,
        }
