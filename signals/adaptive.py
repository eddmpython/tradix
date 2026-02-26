"""Tradex Adaptive Signal Predictor Module.

Automatically detects the current market regime and selects the optimal
trading strategy based on empirical experiment results (E004, E007).

적응형 전략 선택기 - E004, E007 실험 결과를 기반으로 시장 레짐을 자동 감지하고
최적 전략을 선택하는 신호 예측기입니다.

Features:
    - Market regime detection (uptrend, downtrend, sideways, high-volatility,
      strong-uptrend) using trend, volatility, and momentum analysis
    - Experiment-backed strategy mapping (E004 results):
      uptrend -> trend, downtrend -> meanReversion, sideways -> trend+momentum,
      high-volatility -> all strategies, strong-uptrend -> buy-and-hold
    - Regime history tracking for lookback analysis
    - Quick one-call prediction via quickAdaptivePredict()

Usage:
    from tradex.signals.adaptive import AdaptiveSignalPredictor

    predictor = AdaptiveSignalPredictor(df)
    result = predictor.predict()
    print(result.regime, result.strategyName, result.signal)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from tradex.signals.predictor import SignalPredictor, SignalResult, SignalConfig


class MarketRegime(Enum):
    """Market regime classification for adaptive strategy selection.

    적응형 전략 선택을 위한 시장 레짐 분류.

    Attributes:
        UPTREND: Moderate uptrend (상승장).
        DOWNTREND: Moderate downtrend (하락장).
        SIDEWAYS: Range-bound market (횡보장).
        HIGH_VOLATILITY: High-volatility environment (고변동장).
        STRONG_UPTREND: Strong uptrend where buy-and-hold is recommended (강한 상승장).
    """
    UPTREND = "상승장"
    DOWNTREND = "하락장"
    SIDEWAYS = "횡보장"
    HIGH_VOLATILITY = "고변동장"
    STRONG_UPTREND = "강한상승장"


@dataclass
class AdaptiveSignalResult:
    """Result of adaptive signal prediction including regime context.

    적응형 신호 예측 결과 - 레짐 컨텍스트를 포함합니다.

    Attributes:
        signal: Final signal (1=buy, -1=sell, 0=hold) (최종 신호).
        strength: Signal strength 0.0-1.0 (신호 강도).
        confidence: Signal confidence (신호 신뢰도).
        reasons: List of reasons for the signal (신호 사유 목록).
        regime: Detected market regime (감지된 시장 레짐).
        regimeStrength: Strength of the regime detection 0.0-1.0 (레짐 감지 강도).
        strategy: List of strategy names used (사용된 전략 이름 목록).
        strategyName: Human-readable strategy name (전략 표시 이름).
        baseResult: Underlying SignalResult from SignalPredictor (기반 예측 결과).
        regimeDetails: Detailed regime detection metrics (레짐 감지 상세 지표).
    """
    signal: int
    strength: float
    confidence: float
    reasons: List[str]
    regime: MarketRegime
    regimeStrength: float
    strategy: List[str]
    strategyName: str
    baseResult: SignalResult
    regimeDetails: Dict

    @property
    def isBuy(self) -> bool:
        return self.signal == 1

    @property
    def isSell(self) -> bool:
        return self.signal == -1

    @property
    def isHold(self) -> bool:
        return self.signal == 0

    def __repr__(self) -> str:
        signalStr = {1: "BUY", -1: "SELL", 0: "HOLD"}[self.signal]
        return f"AdaptiveSignalResult({signalStr}, regime={self.regime.value}, strategy={self.strategyName})"


class RegimeDetector:
    """Market regime detector using trend, volatility, and momentum analysis.

    Combines multiple technical indicators to classify the current market
    environment into one of five regime categories.

    여러 기술 지표를 종합하여 현재 시장 상황을 5개 레짐 중 하나로 분류합니다.

    Attributes:
        df: OHLCV DataFrame (OHLCV 데이터).
        closes: Numpy array of close prices (종가 배열).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closes = df['close'].values

    def detect(self, window: int = 20) -> Tuple[MarketRegime, float, Dict]:
        """Detect the current market regime.

        Args:
            window: Lookback window for analysis (분석 룩백 윈도우, 거래일).

        Returns:
            Tuple of (MarketRegime, strength 0-1, details dict with trend/volatility/momentum).
        """
        if len(self.closes) < window + 10:
            return MarketRegime.SIDEWAYS, 0.5, {}

        details = {}

        trendScore, trendDetails = self._analyzeTrend(window)
        details['trend'] = trendDetails

        volatility, volDetails = self._analyzeVolatility(window)
        details['volatility'] = volDetails

        momentum, momDetails = self._analyzeMomentum(window)
        details['momentum'] = momDetails

        if volatility > 0.4:
            regime = MarketRegime.HIGH_VOLATILITY
            strength = min(1.0, volatility / 0.5)
        elif trendScore > 0.6 and momentum > 0.5:
            regime = MarketRegime.STRONG_UPTREND
            strength = (trendScore + momentum) / 2
        elif trendScore > 0.3:
            regime = MarketRegime.UPTREND
            strength = trendScore
        elif trendScore < -0.3:
            regime = MarketRegime.DOWNTREND
            strength = abs(trendScore)
        else:
            regime = MarketRegime.SIDEWAYS
            strength = 1.0 - abs(trendScore) * 2

        return regime, strength, details

    def _analyzeTrend(self, window: int) -> Tuple[float, Dict]:
        """Analyze trend strength using SMA positioning and return momentum.

        Args:
            window: Lookback window.

        Returns:
            Tuple of (trend_score from -1 to +1, details dict).
        """
        closes = self.closes

        totalReturn = (closes[-1] / closes[-window]) - 1

        sma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        sma200 = np.mean(closes[-200:]) if len(closes) >= 200 else closes[-1]

        currentPrice = closes[-1]

        smaScore = 0
        if currentPrice > sma20:
            smaScore += 0.3
        if currentPrice > sma50:
            smaScore += 0.3
        if currentPrice > sma200:
            smaScore += 0.4

        if currentPrice < sma20:
            smaScore -= 0.3
        if currentPrice < sma50:
            smaScore -= 0.3
        if currentPrice < sma200:
            smaScore -= 0.4

        returnScore = np.clip(totalReturn * 10, -1, 1)

        trendScore = (smaScore * 0.6 + returnScore * 0.4)

        return trendScore, {
            'totalReturn': totalReturn,
            'smaScore': smaScore,
            'returnScore': returnScore,
            'priceVsSma20': currentPrice / sma20 - 1,
            'priceVsSma50': currentPrice / sma50 - 1,
        }

    def _analyzeVolatility(self, window: int) -> Tuple[float, Dict]:
        """Analyze annualized volatility and ATR-based volatility.

        Args:
            window: Lookback window.

        Returns:
            Tuple of (annualized_volatility, details dict).
        """
        if len(self.closes) < window + 1:
            return 0.2, {}

        returns = np.diff(self.closes[-window-1:]) / self.closes[-window-1:-1]
        dailyVol = np.std(returns)
        annualizedVol = dailyVol * np.sqrt(252)

        atr = self._calculateATR(window)
        atrPercent = atr / self.closes[-1] if self.closes[-1] > 0 else 0

        return annualizedVol, {
            'dailyVol': dailyVol,
            'annualizedVol': annualizedVol,
            'atrPercent': atrPercent,
        }

    def _analyzeMomentum(self, window: int) -> Tuple[float, Dict]:
        """Analyze short-term and long-term momentum.

        Args:
            window: Lookback window.

        Returns:
            Tuple of (momentum_score from -1 to +1, details dict).
        """
        if len(self.closes) < window:
            return 0, {}

        roc = (self.closes[-1] - self.closes[-window]) / self.closes[-window]

        shortMom = (self.closes[-1] - self.closes[-5]) / self.closes[-5] if len(self.closes) >= 5 else 0
        longMom = (self.closes[-1] - self.closes[-20]) / self.closes[-20] if len(self.closes) >= 20 else 0

        momScore = np.clip((shortMom * 0.4 + longMom * 0.6) * 10, -1, 1)

        return momScore, {
            'roc': roc,
            'shortMom': shortMom,
            'longMom': longMom,
        }

    def _calculateATR(self, period: int = 14) -> float:
        """Calculate Average True Range.

        Args:
            period: ATR period (ATR 기간).

        Returns:
            Average True Range value.
        """
        if len(self.df) < period + 1:
            return 0

        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values

        tr = np.zeros(len(self.df) - 1)
        for i in range(1, len(self.df)):
            tr[i-1] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        return np.mean(tr[-period:])


class AdaptiveSignalPredictor:
    """Adaptive signal predictor that auto-selects strategies by market regime.

    Detects the current market regime and maps it to the empirically optimal
    trading strategy based on E004 experiment results:
      - Uptrend -> trend (profit factor 4.01)
      - Downtrend -> meanReversion (profit factor 1.51)
      - Sideways -> trend+momentum (profit factor 1.64)
      - High volatility -> all strategies (profit factor 5.77)
      - Strong uptrend -> buy-and-hold recommended (signals disabled)

    적응형 신호 예측기 - 시장 레짐을 자동 감지하고 E004 실험 결과에 기반하여
    최적 전략을 선택합니다.

    Attributes:
        df: OHLCV DataFrame (OHLCV 데이터).
        config: SignalConfig instance (신호 설정).
        regimeWindow: Window size for regime detection (레짐 감지 윈도우).
        regimeDetector: RegimeDetector instance (레짐 감지기).
        STRATEGY_MAP: Mapping of regime to (strategies, threshold, name) (전략 매핑).

    Example:
        >>> predictor = AdaptiveSignalPredictor(df)
        >>> result = predictor.predict()
        >>> if result.regime == MarketRegime.STRONG_UPTREND:
        ...     print("Strong uptrend - Buy & Hold recommended")
        >>> elif result.isBuy:
        ...     print(f"Buy signal ({result.strategyName})")
    """

    STRATEGY_MAP = {
        MarketRegime.UPTREND: (['trend'], 0.5, 'trend'),
        MarketRegime.DOWNTREND: (['meanReversion'], 0.5, 'meanReversion'),
        MarketRegime.SIDEWAYS: (['trend', 'momentum'], 0.5, 'trend+momentum'),
        MarketRegime.HIGH_VOLATILITY: (None, 0.4, 'all@40%'),
        MarketRegime.STRONG_UPTREND: (None, 0.6, 'buyhold_recommended'),
    }

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[SignalConfig] = None,
        regimeWindow: int = 20,
    ):
        """Initialize the adaptive signal predictor.

        Args:
            df: OHLCV DataFrame with required columns: open, high, low, close, volume.
                OHLCV DataFrame (필수 컬럼: open, high, low, close, volume).
            config: SignalConfig instance. Defaults to SignalConfig() (신호 설정).
            regimeWindow: Window size for regime detection in days (레짐 감지 윈도우).
        """
        self.df = df
        self.config = config or SignalConfig()
        self.regimeWindow = regimeWindow
        self.regimeDetector = RegimeDetector(df)

    def predict(
        self,
        forceStrategy: Optional[List[str]] = None,
        strengthFilter: float = 0.0,
    ) -> AdaptiveSignalResult:
        """Generate an adaptive signal by detecting the regime and selecting the best strategy.

        Args:
            forceStrategy: Override strategy list, ignoring regime detection
                           (강제 전략, 레짐 무시).
            strengthFilter: Minimum strength threshold; signals below this are
                            filtered to HOLD. E005 recommends 0.6
                            (최소 강도 필터, E005 권장값 0.6).

        Returns:
            AdaptiveSignalResult with regime context, strategy info, and signal.
        """
        regime, regimeStrength, regimeDetails = self.regimeDetector.detect(self.regimeWindow)

        if forceStrategy is not None:
            strategies = forceStrategy
            threshold = 0.5
            strategyName = 'forced'
        else:
            strategies, threshold, strategyName = self.STRATEGY_MAP[regime]

        if regime == MarketRegime.STRONG_UPTREND and forceStrategy is None:
            return AdaptiveSignalResult(
                signal=0,
                strength=0.0,
                confidence=regimeStrength,
                reasons=["강한 상승장 감지 - Buy & Hold 권장", "신호 전략보다 보유가 유리할 가능성 높음"],
                regime=regime,
                regimeStrength=regimeStrength,
                strategy=[],
                strategyName='buyhold_recommended',
                baseResult=None,
                regimeDetails=regimeDetails,
            )

        basePredictor = SignalPredictor(self.df, self.config)
        baseResult = basePredictor.predict(strategies=strategies, consensusThreshold=threshold)

        if strengthFilter > 0 and baseResult.strength < strengthFilter:
            return AdaptiveSignalResult(
                signal=0,
                strength=baseResult.strength,
                confidence=baseResult.confidence,
                reasons=[f"신호 강도({baseResult.strength:.2f}) < 필터({strengthFilter})"],
                regime=regime,
                regimeStrength=regimeStrength,
                strategy=strategies or [],
                strategyName=strategyName,
                baseResult=baseResult,
                regimeDetails=regimeDetails,
            )

        reasons = [f"[{regime.value}] {strategyName} 전략 적용"] + baseResult.reasons[:3]

        return AdaptiveSignalResult(
            signal=baseResult.signal,
            strength=baseResult.strength,
            confidence=baseResult.confidence,
            reasons=reasons,
            regime=regime,
            regimeStrength=regimeStrength,
            strategy=strategies or [],
            strategyName=strategyName,
            baseResult=baseResult,
            regimeDetails=regimeDetails,
        )

    def getRegimeHistory(self, lookback: int = 60) -> pd.DataFrame:
        """Generate regime classification history over the past N days.

        Args:
            lookback: Number of days to look back (조회 기간, 일).

        Returns:
            DataFrame with columns: date, regime, strength, close (레짐 히스토리 DataFrame).
        """
        results = []

        for i in range(max(self.regimeWindow + 10, len(self.df) - lookback), len(self.df)):
            subDf = self.df.iloc[:i+1]
            detector = RegimeDetector(subDf)
            regime, strength, _ = detector.detect(self.regimeWindow)

            results.append({
                'date': subDf.index[-1] if hasattr(subDf.index, '__getitem__') else i,
                'regime': regime.value,
                'strength': strength,
                'close': subDf['close'].iloc[-1],
            })

        return pd.DataFrame(results)


def quickAdaptivePredict(df: pd.DataFrame) -> AdaptiveSignalResult:
    """One-call adaptive prediction with default settings and 0.6 strength filter.

    Convenience function that creates an AdaptiveSignalPredictor and runs
    prediction with the recommended E005 strength filter of 0.6.

    간편 적응형 예측 - 기본 설정과 E005 권장 강도 필터(0.6)로 즉시 예측합니다.

    Args:
        df: OHLCV DataFrame (OHLCV 데이터).

    Returns:
        AdaptiveSignalResult with regime-aware signal.
    """
    predictor = AdaptiveSignalPredictor(df)
    return predictor.predict(strengthFilter=0.6)
