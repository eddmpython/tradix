"""Tradex Price Forecast Module.

Standalone time-series forecasting engine for price direction prediction,
built with zero external dependencies beyond numpy and pandas.

주가 예측 모듈 - numpy와 pandas만으로 구현된 시계열 예측 엔진으로,
Tradex 신호 예측에 특화된 기능을 제공합니다.

Features:
    - Simple Exponential Smoothing (SES) with grid-search alpha optimization
    - Holt's Linear Trend Method (Double Exponential Smoothing)
    - Theta Method (Assimakopoulos & Nikolopoulos, 2000)
    - Random Walk with Drift
    - Moving Average Forecast
    - Weighted ensemble of all models for robust prediction
    - EWMA-based volatility forecasting
    - Multi-method trend analysis (linear, MA, momentum, ADX-style)

Usage:
    from tradex.signals.forecast import PriceForecast, TrendAnalyzer

    forecast = PriceForecast(close_prices)
    result = forecast.predict(steps=5)
    print(result.direction, result.expectedReturn)

    analyzer = TrendAnalyzer(close_prices)
    trend = analyzer.analyze()
    print(trend['direction'], trend['strength'])
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Result of a price forecast.

    가격 예측 결과.

    Attributes:
        predictions: Array of predicted prices for each step (단계별 예측 가격 배열).
        direction: Predicted direction (1=up, -1=down, 0=flat) (예측 방향).
        expectedReturn: Expected return from current price to final prediction (기대 수익률).
        confidence: Forecast confidence 0-1 based on model agreement (예측 신뢰도).
        model: Name of the model or ensemble used (사용된 모델명).
    """
    predictions: np.ndarray
    direction: int
    expectedReturn: float
    confidence: float
    model: str


class PriceForecast:
    """Ensemble price forecasting engine combining multiple time-series models.

    Produces weighted-average predictions from SES, Holt, Theta, Drift, and
    Moving Average models. Confidence is based on inter-model agreement.

    여러 예측 모델을 앙상블하여 미래 가격 방향을 예측합니다.
    모델 간 일치도를 기반으로 신뢰도를 산출합니다.

    Attributes:
        y: Input price time series as float64 array (입력 가격 시계열).
        n: Length of the input series (시계열 길이).
        seasonalPeriod: Seasonal period (default 5 = 1 trading week) (계절성 주기).

    Example:
        >>> forecast = PriceForecast(close_prices)
        >>> result = forecast.predict(steps=5)
        >>> print(result.direction, result.expectedReturn)
    """

    def __init__(self, y: np.ndarray, seasonalPeriod: int = 5):
        """Initialize the price forecast engine.

        Args:
            y: Close price time series as numpy array (종가 시계열 배열).
            seasonalPeriod: Seasonal period in trading days (default 5 = 1 week)
                            (계절성 주기, 기본 5일 = 1주).
        """
        self.y = np.asarray(y, dtype=np.float64)
        self.n = len(self.y)
        self.seasonalPeriod = seasonalPeriod

    def predict(self, steps: int = 5) -> ForecastResult:
        """Generate an ensemble forecast by weighted-averaging multiple models.

        Args:
            steps: Number of trading days to forecast (예측 기간, 거래일).

        Returns:
            ForecastResult with ensemble predictions, direction, expected return,
            and confidence.
        """
        if self.n < 30:
            return ForecastResult(
                predictions=np.full(steps, self.y[-1]),
                direction=0,
                expectedReturn=0.0,
                confidence=0.0,
                model='insufficient_data',
            )

        predictions = {}
        weights = {}

        try:
            sesPred = self._ses(steps)
            predictions['ses'] = sesPred
            weights['ses'] = 0.15
        except Exception:
            pass

        try:
            holtPred = self._holt(steps)
            predictions['holt'] = holtPred
            weights['holt'] = 0.25
        except Exception:
            pass

        try:
            thetaPred = self._theta(steps)
            predictions['theta'] = thetaPred
            weights['theta'] = 0.30
        except Exception:
            pass

        try:
            driftPred = self._drift(steps)
            predictions['drift'] = driftPred
            weights['drift'] = 0.20
        except Exception:
            pass

        try:
            maPred = self._movingAverage(steps)
            predictions['ma'] = maPred
            weights['ma'] = 0.10
        except Exception:
            pass

        if not predictions:
            return ForecastResult(
                predictions=np.full(steps, self.y[-1]),
                direction=0,
                expectedReturn=0.0,
                confidence=0.0,
                model='all_failed',
            )

        totalWeight = sum(weights[k] for k in predictions.keys())
        ensemble = np.zeros(steps)

        for model, pred in predictions.items():
            ensemble += pred * (weights[model] / totalWeight)

        currentPrice = self.y[-1]
        futurePrice = ensemble[-1]
        expectedReturn = (futurePrice - currentPrice) / currentPrice

        threshold = 0.005
        if expectedReturn > threshold:
            direction = 1
        elif expectedReturn < -threshold:
            direction = -1
        else:
            direction = 0

        stdDev = np.std([p[-1] for p in predictions.values()])
        avgPred = np.mean([p[-1] for p in predictions.values()])
        cv = stdDev / avgPred if avgPred != 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - cv * 2))

        bestModel = max(predictions.keys(), key=lambda k: weights[k])

        return ForecastResult(
            predictions=ensemble,
            direction=direction,
            expectedReturn=expectedReturn,
            confidence=confidence,
            model=f'ensemble({len(predictions)})',
        )

    def _ses(self, steps: int, alpha: float = None) -> np.ndarray:
        """Simple Exponential Smoothing forecast.

        Args:
            steps: Number of steps to forecast.
            alpha: Smoothing parameter. If None, optimized via grid search.

        Returns:
            Array of predicted values (constant level for all steps).
        """
        if alpha is None:
            alpha = self._optimizeAlpha()

        level = self.y[0]
        for i in range(1, self.n):
            level = alpha * self.y[i] + (1 - alpha) * level

        return np.full(steps, level)

    def _holt(self, steps: int, alpha: float = None, beta: float = None) -> np.ndarray:
        """Holt's Linear Trend Method (Double Exponential Smoothing) forecast.

        Args:
            steps: Number of steps to forecast.
            alpha: Level smoothing parameter (default 0.3).
            beta: Trend smoothing parameter (default 0.1).

        Returns:
            Array of predicted values incorporating level and trend.
        """
        if alpha is None:
            alpha = 0.3
        if beta is None:
            beta = 0.1

        level = self.y[0]
        trend = (self.y[min(5, self.n-1)] - self.y[0]) / min(5, self.n-1)

        for i in range(1, self.n):
            prevLevel = level
            level = alpha * self.y[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prevLevel) + (1 - beta) * trend

        predictions = np.zeros(steps)
        for h in range(steps):
            predictions[h] = level + (h + 1) * trend

        return predictions

    def _theta(self, steps: int) -> np.ndarray:
        """Theta Method forecast (Assimakopoulos & Nikolopoulos, 2000).

        Args:
            steps: Number of steps to forecast.

        Returns:
            Array of predicted values combining detrended SES with linear trend.
        """
        n = self.n
        y = self.y

        x = np.arange(1, n + 1)
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)

        detrended = y - (intercept + slope * x)

        alpha = self._optimizeAlpha(detrended)
        level = detrended[0]
        for i in range(1, n):
            level = alpha * detrended[i] + (1 - alpha) * level

        predictions = np.zeros(steps)
        for h in range(steps):
            trendComponent = intercept + slope * (n + h + 1)
            predictions[h] = level + trendComponent

        return predictions

    def _drift(self, steps: int) -> np.ndarray:
        """Random Walk with Drift forecast.

        Args:
            steps: Number of steps to forecast.

        Returns:
            Array of predicted values with constant drift from historical average.
        """
        n = self.n
        drift = (self.y[-1] - self.y[0]) / (n - 1)

        predictions = np.zeros(steps)
        for h in range(steps):
            predictions[h] = self.y[-1] + drift * (h + 1)

        return predictions

    def _movingAverage(self, steps: int, window: int = 10) -> np.ndarray:
        """Simple Moving Average flat forecast.

        Args:
            steps: Number of steps to forecast.
            window: Moving average window (default 10).

        Returns:
            Array of constant predicted values equal to the trailing MA.
        """
        window = min(window, self.n)
        ma = np.mean(self.y[-window:])
        return np.full(steps, ma)

    def _optimizeAlpha(self, y: np.ndarray = None) -> float:
        """Find optimal SES alpha via grid search minimizing MSE.

        Args:
            y: Time series to optimize on. Defaults to self.y.

        Returns:
            Optimal alpha value from {0.1, 0.2, ..., 0.9}.
        """
        if y is None:
            y = self.y

        bestAlpha = 0.3
        bestMse = float('inf')

        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mse = self._sesMse(y, alpha)
            if mse < bestMse:
                bestMse = mse
                bestAlpha = alpha

        return bestAlpha

    def _sesMse(self, y: np.ndarray, alpha: float) -> float:
        """Compute the Mean Squared Error of SES with the given alpha.

        Args:
            y: Time series array.
            alpha: SES smoothing parameter.

        Returns:
            MSE value.
        """
        n = len(y)
        if n < 2:
            return float('inf')

        level = y[0]
        sse = 0.0

        for i in range(1, n):
            error = y[i] - level
            sse += error ** 2
            level = alpha * y[i] + (1 - alpha) * level

        return sse / (n - 1)

    def predictDirection(self, steps: int = 5) -> Tuple[int, float]:
        """Predict price direction only (convenience interface).

        Args:
            steps: Number of days to forecast (예측 기간, 일).

        Returns:
            Tuple of (direction, confidence) where direction is 1=up, -1=down, 0=flat.
        """
        result = self.predict(steps)
        return result.direction, result.confidence

    def getVolatilityForecast(self, steps: int = 5) -> float:
        """Forecast daily volatility using EWMA (Exponentially Weighted Moving Average).

        Args:
            steps: Forecast horizon (not used in computation; reserved for future use).

        Returns:
            Estimated daily volatility (standard deviation) (예상 일간 변동성).
        """
        returns = np.diff(self.y) / self.y[:-1]

        if len(returns) < 5:
            return 0.02

        lambda_ = 0.94
        variance = returns[0] ** 2

        for r in returns[1:]:
            variance = lambda_ * variance + (1 - lambda_) * (r ** 2)

        return np.sqrt(variance)


class TrendAnalyzer:
    """Multi-method trend analyzer combining linear, MA, momentum, and ADX approaches.

    Synthesizes multiple trend detection methods into a single direction and
    strength assessment, weighted by ADX-style trend quality.

    여러 추세 분석 방법을 종합하여 방향과 강도를 판단합니다.

    Attributes:
        y: Input price series as float64 array (입력 가격 시계열).
        n: Length of the input series (시계열 길이).

    Example:
        >>> analyzer = TrendAnalyzer(close_prices)
        >>> trend = analyzer.analyze()
        >>> print(trend['direction'], trend['strength'])
    """

    def __init__(self, y: np.ndarray):
        self.y = np.asarray(y, dtype=np.float64)
        self.n = len(self.y)

    def analyze(self) -> dict:
        """Run comprehensive trend analysis combining all methods.

        Returns:
            Dict with 'direction' (1=up, -1=down, 0=flat), 'strength' (0-1),
            and 'details' with per-method results (종합 추세 분석 결과).
        """
        results = {}

        results['linear'] = self._linearTrend()
        results['ma'] = self._maTrend()
        results['momentum'] = self._momentumTrend()
        results['adx'] = self._adxStrength()

        directions = [
            results['linear']['direction'],
            results['ma']['direction'],
            results['momentum']['direction'],
        ]

        strengths = [
            results['linear']['strength'],
            results['ma']['strength'],
            results['momentum']['strength'],
        ]

        avgDirection = np.mean(directions)
        if avgDirection > 0.3:
            finalDirection = 1
        elif avgDirection < -0.3:
            finalDirection = -1
        else:
            finalDirection = 0

        finalStrength = np.mean(strengths)

        adxStrength = results['adx']['strength']
        finalStrength = finalStrength * (0.5 + 0.5 * adxStrength)

        return {
            'direction': finalDirection,
            'strength': min(1.0, finalStrength),
            'details': results,
        }

    def _linearTrend(self) -> dict:
        """Assess trend via linear regression slope normalized by average price.

        Returns:
            Dict with 'direction', 'strength', and 'slope'.
        """
        x = np.arange(self.n)
        slope = np.cov(x, self.y)[0, 1] / np.var(x)

        avgPrice = np.mean(self.y)
        normalizedSlope = slope / avgPrice * 100

        if normalizedSlope > 0.1:
            direction = 1
        elif normalizedSlope < -0.1:
            direction = -1
        else:
            direction = 0

        strength = min(1.0, abs(normalizedSlope) / 0.5)

        return {'direction': direction, 'strength': strength, 'slope': normalizedSlope}

    def _maTrend(self) -> dict:
        """Assess trend via 5-day vs 20-day moving average crossover.

        Returns:
            Dict with 'direction' and 'strength'.
        """
        if self.n < 20:
            return {'direction': 0, 'strength': 0.0}

        ma5 = np.mean(self.y[-5:])
        ma20 = np.mean(self.y[-20:])

        diff = (ma5 - ma20) / ma20

        if diff > 0.01:
            direction = 1
        elif diff < -0.01:
            direction = -1
        else:
            direction = 0

        strength = min(1.0, abs(diff) / 0.05)

        return {'direction': direction, 'strength': strength}

    def _momentumTrend(self) -> dict:
        """Assess trend via 10-day rate of change (ROC).

        Returns:
            Dict with 'direction', 'strength', and 'roc'.
        """
        if self.n < 10:
            return {'direction': 0, 'strength': 0.0}

        roc = (self.y[-1] - self.y[-10]) / self.y[-10]

        if roc > 0.02:
            direction = 1
        elif roc < -0.02:
            direction = -1
        else:
            direction = 0

        strength = min(1.0, abs(roc) / 0.1)

        return {'direction': direction, 'strength': strength, 'roc': roc}

    def _adxStrength(self) -> dict:
        """Estimate trend quality using a simplified ADX-style metric.

        Returns:
            Dict with 'strength' (0-1).
        """
        if self.n < 14:
            return {'strength': 0.5}

        changes = np.abs(np.diff(self.y))
        avgChange = np.mean(changes[-14:])
        totalRange = np.max(self.y[-14:]) - np.min(self.y[-14:])

        if totalRange == 0:
            strength = 0.0
        else:
            strength = avgChange / totalRange * 14

        strength = min(1.0, max(0.0, strength))

        return {'strength': strength}
