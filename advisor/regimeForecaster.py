"""Tradix Regime Forecaster Module.

Learns regime transition patterns from historical data using Markov Chain modeling,
detects transition leading signals, and forecasts future market regimes with
confidence scores.

시장 레짐 예측기 모듈 - 과거 레짐 전이 패턴을 Markov Chain으로 학습하고,
전환 선행 신호를 감지하여 미래 시장 상황을 예측합니다.

Features:
    - Regime transition probability matrix (Markov Chain)
    - Transition leading signal detection (SMA convergence, RSI extremes, etc.)
    - N-day-ahead regime prediction with confidence scoring
    - Per-regime duration statistics
    - Risk level assessment based on regime and transition signals

Usage:
    from tradix.advisor.regimeForecaster import RegimeForecaster

    forecaster = RegimeForecaster()
    forecaster.fit(df)
    forecast = forecaster.predict(df, horizonDays=[5, 10, 20])
    print(forecast.summary)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np

from tradix.advisor.marketClassifier import MarketClassifier, MarketRegime, MarketAnalysis


@dataclass
class RegimeTransition:
    """Information about a specific regime transition.

    특정 레짐 전이에 대한 정보.

    Attributes:
        fromRegime: Source regime (출발 레짐).
        toRegime: Destination regime (도착 레짐).
        probability: Transition probability (전이 확률).
        avgDuration: Average duration of source regime in steps (출발 레짐 평균 지속 기간).
        sampleCount: Number of observed transitions (관측된 전이 횟수).
    """
    fromRegime: MarketRegime
    toRegime: MarketRegime
    probability: float
    avgDuration: float
    sampleCount: int


@dataclass
class RegimeForecast:
    """Result of a regime forecast.

    레짐 예측 결과.

    Attributes:
        currentRegime: Currently detected regime (현재 감지된 레짐).
        currentConfidence: Confidence of current classification (현재 분류 신뢰도).
        forecasts: List of (days_ahead, predicted_regime, probability) tuples
                   (예측 결과 리스트: (일수, 예측 레짐, 확률)).
        transitionSignals: Human-readable transition signal descriptions
                           (전환 신호 설명 목록).
        riskLevel: Risk assessment string (리스크 수준 문자열).
        summary: Formatted summary string (요약 문자열).
    """
    currentRegime: MarketRegime
    currentConfidence: float
    forecasts: List[Tuple[int, MarketRegime, float]]
    transitionSignals: List[str]
    riskLevel: str
    summary: str

    def getForecast(self, daysAhead: int) -> Tuple[MarketRegime, float]:
        """Retrieve forecast for a specific number of days ahead.

        Args:
            daysAhead: Number of days into the future (예측 일수).

        Returns:
            Tuple of (predicted_regime, probability). Falls back to current regime
            with 0.5 probability if no matching forecast exists.
        """
        for days, regime, prob in self.forecasts:
            if days == daysAhead:
                return regime, prob
        return self.currentRegime, 0.5


@dataclass
class TransitionSignal:
    """A detected regime transition signal.

    감지된 레짐 전환 신호.

    Attributes:
        signalType: Signal identifier string (신호 유형 식별자).
        strength: Signal strength 0-1 (신호 강도).
        direction: Direction hint ('bullish', 'bearish', 'neutral', 'uncertain', 'continuation')
                   (방향 힌트).
        description: Human-readable description (설명 문자열).
    """
    signalType: str
    strength: float
    direction: str
    description: str


class RegimeForecaster:
    """Market regime forecaster using Markov Chain transition modeling.

    Learns regime transition probabilities from historical data, detects
    transition leading signals from current indicators, and produces
    probabilistic forecasts for multiple time horizons.

    시장 레짐 예측기 - 과거 데이터에서 레짐 전이 패턴을 Markov Chain으로 학습하고,
    현재 지표 기반 선행 신호를 결합하여 미래 레짐을 예측합니다.

    Attributes:
        classifier: MarketClassifier instance (시장 분류기).
        windowSize: Regime analysis window size (레짐 분석 윈도우).
        stepSize: Analysis step size (분석 스텝 크기).
        transitionMatrix: Learned regime transition probabilities (전이 확률 매트릭스).
        regimeDurations: Per-regime duration observations (레짐별 지속 기간 관측치).
        regimeHistory: Chronological list of (date, regime) tuples (레짐 히스토리).
        REGIME_ORDER: Canonical ordering of regimes (레짐 정렬 순서).

    Example:
        >>> forecaster = RegimeForecaster()
        >>> forecaster.fit(df)
        >>> forecast = forecaster.predict(df, horizonDays=[5, 10, 20])
        >>> print(forecast.summary)
    """

    REGIME_ORDER = [
        MarketRegime.STRONG_DOWNTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.SIDEWAYS,
        MarketRegime.UPTREND,
        MarketRegime.STRONG_UPTREND,
        MarketRegime.HIGH_VOLATILITY,
    ]

    def __init__(
        self,
        classifier: MarketClassifier = None,
        windowSize: int = 60,
        stepSize: int = 5,
    ):
        """Initialize the regime forecaster.

        Args:
            classifier: MarketClassifier instance. Defaults to a new instance
                        (시장 분류기, 기본값은 새 인스턴스).
            windowSize: Regime analysis window size in rows (레짐 분석 윈도우 크기).
            stepSize: Step size between analyses (분석 스텝 크기).
        """
        self.classifier = classifier or MarketClassifier()
        self.windowSize = windowSize
        self.stepSize = stepSize

        self.transitionMatrix: Dict[MarketRegime, Dict[MarketRegime, float]] = {}
        self.regimeDurations: Dict[MarketRegime, List[int]] = {}
        self.regimeHistory: List[Tuple[pd.Timestamp, MarketRegime]] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'RegimeForecaster':
        """Learn regime transition patterns from historical OHLCV data.

        Args:
            df: OHLCV DataFrame (OHLCV 데이터).

        Returns:
            Self for method chaining.
        """
        historyDf = self.classifier.analyzeHistory(
            df,
            windowSize=self.windowSize,
            stepSize=self.stepSize
        )

        self.regimeHistory = [
            (row['date'], MarketRegime(row['regime']))
            for _, row in historyDf.iterrows()
        ]

        self._buildTransitionMatrix()
        self._calcRegimeDurations()
        self._fitted = True

        return self

    def _buildTransitionMatrix(self):
        """Build the regime transition probability matrix from observed history."""
        for regime in MarketRegime:
            self.transitionMatrix[regime] = {r: 0.0 for r in MarketRegime}

        transitionCounts: Dict[MarketRegime, Dict[MarketRegime, int]] = {
            r: {r2: 0 for r2 in MarketRegime} for r in MarketRegime
        }

        for i in range(len(self.regimeHistory) - 1):
            currentRegime = self.regimeHistory[i][1]
            nextRegime = self.regimeHistory[i + 1][1]
            transitionCounts[currentRegime][nextRegime] += 1

        for fromRegime in MarketRegime:
            total = sum(transitionCounts[fromRegime].values())
            if total > 0:
                for toRegime in MarketRegime:
                    self.transitionMatrix[fromRegime][toRegime] = (
                        transitionCounts[fromRegime][toRegime] / total
                    )
            else:
                self.transitionMatrix[fromRegime][fromRegime] = 1.0

    def _calcRegimeDurations(self):
        """Calculate per-regime duration statistics from observed history."""
        for regime in MarketRegime:
            self.regimeDurations[regime] = []

        if not self.regimeHistory:
            return

        currentRegime = self.regimeHistory[0][1]
        duration = 1

        for i in range(1, len(self.regimeHistory)):
            if self.regimeHistory[i][1] == currentRegime:
                duration += 1
            else:
                self.regimeDurations[currentRegime].append(duration)
                currentRegime = self.regimeHistory[i][1]
                duration = 1

        self.regimeDurations[currentRegime].append(duration)

    def getTransitionProbability(
        self,
        fromRegime: MarketRegime,
        toRegime: MarketRegime
    ) -> float:
        """Get the transition probability between two specific regimes.

        Args:
            fromRegime: Source regime (출발 레짐).
            toRegime: Destination regime (도착 레짐).

        Returns:
            Transition probability (0-1).
        """
        return self.transitionMatrix.get(fromRegime, {}).get(toRegime, 0.0)

    def getAvgDuration(self, regime: MarketRegime) -> float:
        """Get the average duration of a regime in step units.

        Args:
            regime: The market regime to query (조회할 레짐).

        Returns:
            Average duration in steps (default 10.0 if no data).
        """
        durations = self.regimeDurations.get(regime, [])
        if not durations:
            return 10.0
        return np.mean(durations)

    def predict(
        self,
        df: pd.DataFrame,
        horizonDays: List[int] = None,
    ) -> RegimeForecast:
        """Forecast future market regimes for multiple time horizons.

        Args:
            df: Latest OHLCV DataFrame (최신 OHLCV DataFrame).
            horizonDays: List of forecast horizons in days (default [5, 10, 20, 60])
                         (예측 기간 리스트).

        Returns:
            RegimeForecast with per-horizon predictions, transition signals,
            risk level, and summary.
        """
        if horizonDays is None:
            horizonDays = [5, 10, 20, 60]

        currentAnalysis = self.classifier.analyze(df)
        currentRegime = currentAnalysis.regime

        transitionSignals = self._detectTransitionSignals(df, currentAnalysis)

        forecasts = []
        for days in horizonDays:
            regime, prob = self._forecastRegime(
                currentRegime, days, transitionSignals
            )
            forecasts.append((days, regime, prob))

        riskLevel = self._assessRisk(currentRegime, transitionSignals)

        summary = self._buildSummary(
            currentRegime, currentAnalysis.confidence,
            forecasts, transitionSignals, riskLevel
        )

        return RegimeForecast(
            currentRegime=currentRegime,
            currentConfidence=currentAnalysis.confidence,
            forecasts=forecasts,
            transitionSignals=[s.description for s in transitionSignals],
            riskLevel=riskLevel,
            summary=summary,
        )

    def _detectTransitionSignals(
        self,
        df: pd.DataFrame,
        analysis: MarketAnalysis
    ) -> List[TransitionSignal]:
        """Detect leading signals that indicate potential regime transitions.

        Args:
            df: OHLCV DataFrame.
            analysis: Current MarketAnalysis.

        Returns:
            List of detected TransitionSignal instances.
        """
        signals = []
        close = df['close'].values
        n = len(close)

        if n < 50:
            return signals

        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
        currentPrice = close[-1]

        smaGap = (sma20 - sma50) / sma50 * 100

        if analysis.regime in [MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND]:
            if currentPrice < sma20 and smaGap > 0 and smaGap < 2:
                signals.append(TransitionSignal(
                    signalType="price_below_sma20",
                    strength=0.6,
                    direction="bearish",
                    description="가격이 20일선 하회 - 상승 추세 약화 신호"
                ))

            if smaGap < 1 and smaGap > 0:
                signals.append(TransitionSignal(
                    signalType="sma_convergence",
                    strength=0.5,
                    direction="neutral",
                    description="이평선 수렴 중 - 횡보 전환 가능성"
                ))

        elif analysis.regime in [MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND]:
            if currentPrice > sma20 and smaGap < 0 and smaGap > -2:
                signals.append(TransitionSignal(
                    signalType="price_above_sma20",
                    strength=0.6,
                    direction="bullish",
                    description="가격이 20일선 상회 - 하락 추세 약화 신호"
                ))

        if analysis.adx < 20:
            signals.append(TransitionSignal(
                signalType="weak_trend",
                strength=0.4,
                direction="neutral",
                description=f"추세 약화 (ADX={analysis.adx:.0f}) - 레짐 전환 가능성 높음"
            ))
        elif analysis.adx > 40:
            signals.append(TransitionSignal(
                signalType="strong_trend",
                strength=0.3,
                direction="continuation",
                description=f"강한 추세 (ADX={analysis.adx:.0f}) - 현재 레짐 유지 가능성"
            ))

        if analysis.rsi > 70:
            signals.append(TransitionSignal(
                signalType="overbought",
                strength=0.7,
                direction="bearish",
                description=f"과매수 (RSI={analysis.rsi:.0f}) - 조정 가능성"
            ))
        elif analysis.rsi < 30:
            signals.append(TransitionSignal(
                signalType="oversold",
                strength=0.7,
                direction="bullish",
                description=f"과매도 (RSI={analysis.rsi:.0f}) - 반등 가능성"
            ))

        if analysis.volatility > 0.4:
            signals.append(TransitionSignal(
                signalType="high_volatility",
                strength=0.8,
                direction="uncertain",
                description=f"고변동성 ({analysis.volatility:.1%}) - 급격한 레짐 전환 가능"
            ))

        recentReturns = pd.Series(close).pct_change().tail(5)
        if (recentReturns > 0).all() and analysis.regime != MarketRegime.STRONG_UPTREND:
            signals.append(TransitionSignal(
                signalType="consecutive_up",
                strength=0.5,
                direction="bullish",
                description="5일 연속 상승 - 상승 추세 강화 신호"
            ))
        elif (recentReturns < 0).all() and analysis.regime != MarketRegime.STRONG_DOWNTREND:
            signals.append(TransitionSignal(
                signalType="consecutive_down",
                strength=0.5,
                direction="bearish",
                description="5일 연속 하락 - 하락 추세 강화 신호"
            ))

        return signals

    def _forecastRegime(
        self,
        currentRegime: MarketRegime,
        daysAhead: int,
        signals: List[TransitionSignal]
    ) -> Tuple[MarketRegime, float]:
        """Forecast the regime N days ahead using Markov Chain propagation.

        Args:
            currentRegime: Current regime starting point.
            daysAhead: Number of days to forecast.
            signals: Detected transition signals for probability adjustment.

        Returns:
            Tuple of (most_probable_regime, probability).
        """
        stepsAhead = max(1, daysAhead // self.stepSize)

        probVector = {r: 0.0 for r in MarketRegime}
        probVector[currentRegime] = 1.0

        for _ in range(stepsAhead):
            newProbVector = {r: 0.0 for r in MarketRegime}
            for fromRegime in MarketRegime:
                if probVector[fromRegime] > 0.001:
                    for toRegime in MarketRegime:
                        transProb = self.transitionMatrix.get(fromRegime, {}).get(toRegime, 0)
                        newProbVector[toRegime] += probVector[fromRegime] * transProb
            probVector = newProbVector

        signalAdjustment = self._calcSignalAdjustment(signals, currentRegime)
        for regime, adjustment in signalAdjustment.items():
            probVector[regime] = max(0, min(1, probVector[regime] + adjustment))

        total = sum(probVector.values())
        if total > 0:
            probVector = {r: p / total for r, p in probVector.items()}

        predictedRegime = max(probVector.keys(), key=lambda r: probVector[r])
        confidence = probVector[predictedRegime]

        return predictedRegime, confidence

    def _calcSignalAdjustment(
        self,
        signals: List[TransitionSignal],
        currentRegime: MarketRegime
    ) -> Dict[MarketRegime, float]:
        """Calculate probability adjustments based on detected transition signals.

        Args:
            signals: List of transition signals.
            currentRegime: Current market regime.

        Returns:
            Dict mapping each regime to its probability adjustment value.
        """
        adjustment = {r: 0.0 for r in MarketRegime}

        for signal in signals:
            strength = signal.strength * 0.1

            if signal.direction == "bullish":
                if currentRegime in [MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND]:
                    adjustment[MarketRegime.SIDEWAYS] += strength
                    adjustment[MarketRegime.DOWNTREND] -= strength * 0.5
                elif currentRegime == MarketRegime.SIDEWAYS:
                    adjustment[MarketRegime.UPTREND] += strength
                elif currentRegime == MarketRegime.UPTREND:
                    adjustment[MarketRegime.STRONG_UPTREND] += strength * 0.5

            elif signal.direction == "bearish":
                if currentRegime in [MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND]:
                    adjustment[MarketRegime.SIDEWAYS] += strength
                    adjustment[MarketRegime.UPTREND] -= strength * 0.5
                elif currentRegime == MarketRegime.SIDEWAYS:
                    adjustment[MarketRegime.DOWNTREND] += strength
                elif currentRegime == MarketRegime.DOWNTREND:
                    adjustment[MarketRegime.STRONG_DOWNTREND] += strength * 0.5

            elif signal.direction == "neutral":
                adjustment[MarketRegime.SIDEWAYS] += strength * 0.5

            elif signal.direction == "uncertain":
                adjustment[MarketRegime.HIGH_VOLATILITY] += strength

        return adjustment

    def _assessRisk(
        self,
        currentRegime: MarketRegime,
        signals: List[TransitionSignal]
    ) -> str:
        """Assess the overall risk level based on regime and transition signals.

        Args:
            currentRegime: Current market regime.
            signals: Detected transition signals.

        Returns:
            Risk level string: 'high', 'medium', or 'low' (in Korean).
        """
        riskScore = 0

        if currentRegime == MarketRegime.HIGH_VOLATILITY:
            riskScore += 3
        elif currentRegime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND]:
            riskScore += 2
        elif currentRegime == MarketRegime.SIDEWAYS:
            riskScore += 1

        for signal in signals:
            if signal.direction == "bearish":
                riskScore += signal.strength
            elif signal.direction == "uncertain":
                riskScore += signal.strength * 1.5

        if riskScore >= 4:
            return "높음"
        elif riskScore >= 2:
            return "중간"
        else:
            return "낮음"

    def _buildSummary(
        self,
        currentRegime: MarketRegime,
        confidence: float,
        forecasts: List[Tuple[int, MarketRegime, float]],
        signals: List[TransitionSignal],
        riskLevel: str
    ) -> str:
        """Build a formatted forecast summary string.

        Args:
            currentRegime: Current regime.
            confidence: Current regime confidence.
            forecasts: List of (days, regime, probability) tuples.
            signals: Detected transition signals.
            riskLevel: Assessed risk level.

        Returns:
            Multi-line formatted summary string.
        """
        regimeKr = {
            MarketRegime.STRONG_UPTREND: "강한 상승장",
            MarketRegime.UPTREND: "상승장",
            MarketRegime.SIDEWAYS: "횡보장",
            MarketRegime.DOWNTREND: "하락장",
            MarketRegime.STRONG_DOWNTREND: "강한 하락장",
            MarketRegime.HIGH_VOLATILITY: "고변동성",
        }

        lines = [
            "=" * 50,
            "시장 레짐 예측",
            "=" * 50,
            "",
            f"현재 레짐: {regimeKr.get(currentRegime, currentRegime.value)}",
            f"신뢰도: {confidence:.1%}",
            f"리스크: {riskLevel}",
            "",
            "-" * 50,
            "미래 예측",
            "-" * 50,
        ]

        for days, regime, prob in forecasts:
            lines.append(f"  {days:2d}일 후: {regimeKr.get(regime, regime.value)} ({prob:.1%})")

        if signals:
            lines.append("")
            lines.append("-" * 50)
            lines.append("전환 신호")
            lines.append("-" * 50)
            for signal in signals[:5]:
                lines.append(f"  • {signal.description}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    def getTransitionMatrix(self) -> pd.DataFrame:
        """Export the transition probability matrix as a DataFrame.

        Returns:
            DataFrame with regime names as both index and columns (전이 확률 DataFrame).
        """
        data = []
        for fromRegime in self.REGIME_ORDER:
            row = {'from': fromRegime.value}
            for toRegime in self.REGIME_ORDER:
                row[toRegime.value] = self.transitionMatrix.get(fromRegime, {}).get(toRegime, 0)
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index('from')
        return df

    def getRegimeStats(self) -> pd.DataFrame:
        """Export per-regime statistics as a DataFrame.

        Returns:
            DataFrame with columns: regime, occurrences, frequency, avgDuration,
            maxDuration, minDuration (레짐별 통계 DataFrame).
        """
        data = []
        regimeKr = {
            MarketRegime.STRONG_UPTREND: "강한 상승장",
            MarketRegime.UPTREND: "상승장",
            MarketRegime.SIDEWAYS: "횡보장",
            MarketRegime.DOWNTREND: "하락장",
            MarketRegime.STRONG_DOWNTREND: "강한 하락장",
            MarketRegime.HIGH_VOLATILITY: "고변동성",
        }

        totalOccurrences = sum(len(d) for d in self.regimeDurations.values())

        for regime in self.REGIME_ORDER:
            durations = self.regimeDurations.get(regime, [])
            occurrences = len(durations)

            data.append({
                'regime': regimeKr.get(regime, regime.value),
                'occurrences': occurrences,
                'frequency': occurrences / totalOccurrences if totalOccurrences > 0 else 0,
                'avgDuration': np.mean(durations) if durations else 0,
                'maxDuration': max(durations) if durations else 0,
                'minDuration': min(durations) if durations else 0,
            })

        return pd.DataFrame(data)
