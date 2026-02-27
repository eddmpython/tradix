"""Tradix Market Classifier Module.

Classifies current market conditions into one of six regimes (strong uptrend,
uptrend, sideways, downtrend, strong downtrend, high volatility) using
moving averages, volatility, momentum, ADX, and RSI analysis.

시장 상황 분류기 모듈 - 이동평균, 변동성, 모멘텀, ADX, RSI 분석을 통해
현재 시장을 6개 레짐 중 하나로 분류합니다.

Features:
    - Six-regime classification with confidence scoring
    - Trend analysis via SMA50/SMA200 positioning
    - Annualized volatility computation
    - ADX-based trend strength measurement
    - RSI-based momentum assessment
    - Historical regime analysis with sliding windows

Usage:
    from tradix.advisor.marketClassifier import MarketClassifier

    classifier = MarketClassifier()
    analysis = classifier.analyze(df)
    print(analysis.regime, analysis.confidence)
    print(analysis.summary())
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class MarketRegime(Enum):
    """Market regime classification enumeration.

    시장 상황 분류 열거형.

    Attributes:
        STRONG_UPTREND: Strong uptrend with high ADX (강한 상승장).
        UPTREND: Moderate uptrend (상승장).
        SIDEWAYS: Range-bound / consolidation (횡보장).
        DOWNTREND: Moderate downtrend (하락장).
        STRONG_DOWNTREND: Strong downtrend with high ADX (강한 하락장).
        HIGH_VOLATILITY: High-volatility environment (고변동성).
    """
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class MarketAnalysis:
    """Result of market regime analysis.

    시장 레짐 분석 결과.

    Attributes:
        regime: Classified MarketRegime (분류된 시장 레짐).
        confidence: Classification confidence 0-1 (분류 신뢰도).
        trend: Trend strength from -1 to +1 (추세 강도).
        volatility: Annualized volatility (연율 변동성).
        momentum: RSI-based momentum from -50 to +50 (RSI 기반 모멘텀).
        adx: ADX value (ADX 값).
        rsi: RSI value (RSI 값).
        details: Additional detail metrics (추가 상세 지표).
    """
    regime: MarketRegime
    confidence: float
    trend: float
    volatility: float
    momentum: float
    adx: float
    rsi: float
    details: Dict[str, float]

    def summary(self) -> str:
        """Generate a human-readable summary of the market analysis.

        Returns:
            Formatted string with regime, confidence, trend, volatility, etc.
        """
        regimeKr = {
            MarketRegime.STRONG_UPTREND: "강한 상승장",
            MarketRegime.UPTREND: "상승장",
            MarketRegime.SIDEWAYS: "횡보장",
            MarketRegime.DOWNTREND: "하락장",
            MarketRegime.STRONG_DOWNTREND: "강한 하락장",
            MarketRegime.HIGH_VOLATILITY: "고변동성",
        }

        return (
            f"시장 상황: {regimeKr.get(self.regime, self.regime.value)}\n"
            f"신뢰도: {self.confidence:.1%}\n"
            f"추세 강도: {self.trend:+.1%}\n"
            f"변동성: {self.volatility:.1%}\n"
            f"모멘텀: {self.momentum:+.1f}\n"
            f"ADX: {self.adx:.1f}\n"
            f"RSI: {self.rsi:.1f}"
        )


class MarketClassifier:
    """Market regime classifier using multi-indicator analysis.

    Analyzes OHLCV price data using trend, volatility, momentum, ADX, and RSI
    indicators to classify the current market into one of six regimes.

    다중 지표 분석 기반 시장 레짐 분류기 - OHLCV 데이터를 분석하여
    현재 시장을 6개 레짐 중 하나로 분류합니다.

    Attributes:
        trendPeriod: SMA period for trend calculation (추세 계산 SMA 기간).
        volatilityPeriod: Period for volatility calculation (변동성 계산 기간).
        adxPeriod: ADX calculation period (ADX 계산 기간).
        rsiPeriod: RSI calculation period (RSI 계산 기간).

    Example:
        >>> classifier = MarketClassifier()
        >>> analysis = classifier.analyze(df)
        >>> print(analysis.regime, analysis.confidence)
    """

    def __init__(
        self,
        trendPeriod: int = 50,
        volatilityPeriod: int = 20,
        adxPeriod: int = 14,
        rsiPeriod: int = 14,
    ):
        self.trendPeriod = trendPeriod
        self.volatilityPeriod = volatilityPeriod
        self.adxPeriod = adxPeriod
        self.rsiPeriod = rsiPeriod

    def analyze(self, df: pd.DataFrame) -> MarketAnalysis:
        """Analyze the current market regime from OHLCV data.

        Args:
            df: OHLCV DataFrame (minimum 200 rows required)
                (OHLCV DataFrame, 최소 200행 필요).

        Returns:
            MarketAnalysis with regime classification and indicator values.

        Raises:
            ValueError: If insufficient data is provided.
        """
        if len(df) < max(self.trendPeriod, 200):
            raise ValueError(f"데이터가 부족합니다. 최소 {max(self.trendPeriod, 200)}개 필요")

        trend = self._calcTrend(df)
        volatility = self._calcVolatility(df)
        momentum = self._calcMomentum(df)
        adx = self._calcAdx(df)
        rsi = self._calcRsi(df)

        regime, confidence = self._classifyRegime(trend, volatility, momentum, adx, rsi)

        return MarketAnalysis(
            regime=regime,
            confidence=confidence,
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            adx=adx,
            rsi=rsi,
            details={
                'sma50': df['close'].rolling(50).mean().iloc[-1],
                'sma200': df['close'].rolling(200).mean().iloc[-1],
                'currentPrice': df['close'].iloc[-1],
            }
        )

    def _calcTrend(self, df: pd.DataFrame) -> float:
        """Calculate trend strength from -1 to +1 using SMA50/SMA200 positioning.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Trend score clipped to [-1, +1].
        """
        close = df['close']

        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]
        currentPrice = close.iloc[-1]

        trendVsSma50 = (currentPrice - sma50) / sma50
        trendVsSma200 = (currentPrice - sma200) / sma200
        smaCross = (sma50 - sma200) / sma200

        trend = (trendVsSma50 * 0.3 + trendVsSma200 * 0.3 + smaCross * 0.4)

        return np.clip(trend, -1, 1)

    def _calcVolatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility from recent returns.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Annualized volatility.
        """
        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(self.volatilityPeriod).std() * np.sqrt(252)
        return volatility

    def _calcMomentum(self, df: pd.DataFrame) -> float:
        """Calculate RSI-based momentum from -50 to +50.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Momentum value (RSI - 50).
        """
        rsi = self._calcRsi(df)
        return rsi - 50

    def _calcAdx(self, df: pd.DataFrame) -> float:
        """Calculate the Average Directional Index (ADX).

        Args:
            df: OHLCV DataFrame.

        Returns:
            ADX value indicating trend strength.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        period = self.adxPeriod
        n = len(close)

        tr = np.zeros(n)
        plusDm = np.zeros(n)
        minusDm = np.zeros(n)

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            upMove = high[i] - high[i-1]
            downMove = low[i-1] - low[i]

            plusDm[i] = upMove if upMove > downMove and upMove > 0 else 0
            minusDm[i] = downMove if downMove > upMove and downMove > 0 else 0

        atr = pd.Series(tr).rolling(period).mean().values
        plusDi = 100 * pd.Series(plusDm).rolling(period).mean().values / (atr + 1e-10)
        minusDi = 100 * pd.Series(minusDm).rolling(period).mean().values / (atr + 1e-10)

        dx = 100 * np.abs(plusDi - minusDi) / (plusDi + minusDi + 1e-10)
        adx = pd.Series(dx).rolling(period).mean().iloc[-1]

        return adx if not np.isnan(adx) else 0

    def _calcRsi(self, df: pd.DataFrame) -> float:
        """Calculate the Relative Strength Index (RSI).

        Args:
            df: OHLCV DataFrame.

        Returns:
            RSI value from 0 to 100.
        """
        close = df['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avgGain = gain.rolling(self.rsiPeriod).mean().iloc[-1]
        avgLoss = loss.rolling(self.rsiPeriod).mean().iloc[-1]

        if avgLoss == 0:
            return 100

        rs = avgGain / avgLoss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _classifyRegime(
        self,
        trend: float,
        volatility: float,
        momentum: float,
        adx: float,
        rsi: float
    ) -> Tuple[MarketRegime, float]:
        """Classify the market regime from computed indicators.

        Args:
            trend: Trend score (-1 to +1).
            volatility: Annualized volatility.
            momentum: RSI-based momentum (-50 to +50).
            adx: ADX value.
            rsi: RSI value.

        Returns:
            Tuple of (MarketRegime, confidence).
        """

        if volatility > 0.4:
            return MarketRegime.HIGH_VOLATILITY, 0.8

        if trend > 0.15 and adx > 25:
            return MarketRegime.STRONG_UPTREND, min(0.9, 0.5 + trend + adx/100)

        if trend > 0.05:
            confidence = 0.6 + trend * 2
            return MarketRegime.UPTREND, min(0.85, confidence)

        if trend < -0.15 and adx > 25:
            return MarketRegime.STRONG_DOWNTREND, min(0.9, 0.5 + abs(trend) + adx/100)

        if trend < -0.05:
            confidence = 0.6 + abs(trend) * 2
            return MarketRegime.DOWNTREND, min(0.85, confidence)

        return MarketRegime.SIDEWAYS, 0.7

    def analyzeHistory(
        self,
        df: pd.DataFrame,
        windowSize: int = 60,
        stepSize: int = 20
    ) -> pd.DataFrame:
        """Analyze historical market regimes using a sliding window approach.

        Args:
            df: OHLCV DataFrame (OHLCV 데이터).
            windowSize: Analysis window size in rows (분석 윈도우 크기).
            stepSize: Step size between windows (스텝 크기).

        Returns:
            DataFrame with columns: date, regime, trend, volatility, adx, rsi
            (시장 상황 히스토리 DataFrame).

        Raises:
            ValueError: If insufficient data for the window configuration.
        """
        results = []
        minRequired = max(self.trendPeriod, 200) + windowSize

        if len(df) < minRequired:
            raise ValueError(f"데이터 부족: 최소 {minRequired}개 필요")

        for i in range(200, len(df) - windowSize, stepSize):
            windowDf = df.iloc[:i + windowSize]

            try:
                analysis = self.analyze(windowDf)
                results.append({
                    'date': df.index[i + windowSize - 1],
                    'regime': analysis.regime.value,
                    'trend': analysis.trend,
                    'volatility': analysis.volatility,
                    'adx': analysis.adx,
                    'rsi': analysis.rsi,
                })
            except Exception:
                continue

        return pd.DataFrame(results)
