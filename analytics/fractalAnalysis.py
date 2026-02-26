"""
Fractal Market Analysis - Hurst Exponent and Fractal Dimension computation.
프랙탈 시장 분석 - 허스트 지수 및 프랙탈 차원 계산.

Computes the Hurst Exponent and Fractal Dimension from equity curve returns
to determine whether the market/strategy exhibits trending, random walk, or
mean-reverting behavior. Uses Rescaled Range (R/S) Analysis as the primary
method and Detrended Fluctuation Analysis (DFA) for cross-validation.

자산 곡선 수익률로부터 허스트 지수와 프랙탈 차원을 계산하여 시장/전략이
추세 추종, 랜덤 워크, 평균 회귀 행동 중 어떤 것을 보이는지 판별합니다.
주요 방법으로 재조정 범위(R/S) 분석을 사용하고, 비추세 변동 분석(DFA)으로
교차 검증합니다.

Features:
    - Rescaled Range (R/S) Analysis for Hurst Exponent estimation
    - Detrended Fluctuation Analysis (DFA) for validation
    - Fractal Dimension computation (D = 2 - H)
    - Rolling Hurst Exponent for sub-period analysis (1m, 3m, 6m, 1y)
    - Market character classification (trending / random / meanReverting)
    - Strategy fitness recommendations based on Hurst regime
    - Confidence measurement via R-squared of log-log regression

Usage:
    from tradex.analytics.fractalAnalysis import FractalAnalyzer

    analyzer = FractalAnalyzer()
    result = analyzer.analyze(backtestResult)
    print(result.summary())
    print(result.summary(ko=True))
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from tradex.engine import BacktestResult


HURST_TRENDING_THRESHOLD = 0.55
HURST_MEAN_REVERTING_THRESHOLD = 0.45

CHARACTER_LABELS = {
    "trending": "Trending",
    "random": "Random Walk",
    "meanReverting": "Mean-Reverting",
}

CHARACTER_LABELS_KO = {
    "trending": "추세 지속형",
    "random": "랜덤 워크",
    "meanReverting": "평균 회귀형",
}

PERIOD_WINDOWS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

STRATEGY_TYPES = [
    "trendFollowing",
    "meanReversion",
    "momentum",
    "breakout",
    "statistical",
]

STRATEGY_LABELS_KO = {
    "trendFollowing": "추세 추종",
    "meanReversion": "평균 회귀",
    "momentum": "모멘텀",
    "breakout": "돌파",
    "statistical": "통계적 차익",
}


@dataclass
class FractalAnalysisResult:
    """
    Result container for Fractal Market Analysis.
    프랙탈 시장 분석 결과 컨테이너.

    Encapsulates the Hurst Exponent, Fractal Dimension, market character
    classification, rolling Hurst by period, strategy fitness scores, and
    statistical confidence of the estimation.

    허스트 지수, 프랙탈 차원, 시장 성격 분류, 기간별 롤링 허스트,
    전략 적합도 점수, 추정의 통계적 신뢰도를 캡슐화합니다.

    Attributes:
        hurstExponent (float): Estimated Hurst Exponent.
            H > 0.5 = trending, H ~ 0.5 = random, H < 0.5 = mean-reverting.
            허스트 지수. H > 0.5 추세, H ~ 0.5 랜덤, H < 0.5 평균 회귀.
        fractalDimension (float): Fractal Dimension = 2 - H.
            Ranges 1.0 (smooth trend) to 2.0 (space-filling, complex).
            프랙탈 차원 = 2 - H. 1.0(매끈한 추세)~2.0(공간 충전, 복잡).
        marketCharacter (str): "trending", "random", or "meanReverting".
            시장 성격 분류 문자열.
        hurstByPeriod (Dict[str, float]): Rolling Hurst for sub-periods.
            Keys: "1m", "3m", "6m", "1y". 기간별 롤링 허스트 지수.
        strategyFit (Dict[str, float]): Fitness scores (0-100) for strategy types.
            전략 유형별 적합도 점수 (0-100).
        confidence (float): R-squared of the log-log regression (0-1).
            로그-로그 회귀의 결정계수 (0-1).
        details (Dict): Detailed internal metrics for debugging/reporting.
            디버깅/보고용 상세 내부 지표.

    Example:
        >>> analyzer = FractalAnalyzer()
        >>> fractal = analyzer.analyze(backtestResult)
        >>> print(f"Hurst: {fractal.hurstExponent:.4f}")
        >>> print(f"Character: {fractal.marketCharacter}")
    """
    hurstExponent: float = 0.5
    fractalDimension: float = 1.5
    marketCharacter: str = "random"
    hurstByPeriod: Dict[str, float] = field(default_factory=dict)
    strategyFit: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)

    def summary(self, ko: bool = False) -> str:
        """
        Generate a formatted summary report of the fractal analysis.
        프랙탈 분석 결과의 포맷된 요약 보고서를 생성합니다.

        Args:
            ko (bool): If True, generate Korean-language summary.
                True이면 한국어 요약을 생성합니다. Defaults to False.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 구성된 포맷 요약 문자열.
        """
        if ko:
            return self._summaryKo()
        return self._summaryEn()

    def _summaryEn(self) -> str:
        """
        Generate English-language summary.
        영어 요약을 생성합니다.

        Returns:
            str: English formatted summary string.
        """
        characterLabel = CHARACTER_LABELS.get(self.marketCharacter, self.marketCharacter)
        confidencePct = self.confidence * 100

        lines = [
            "",
            "=" * 55,
            "  Fractal Market Analysis",
            "=" * 55,
            f"  Hurst Exponent:     {self.hurstExponent:.4f}",
            f"  Fractal Dimension:  {self.fractalDimension:.4f}",
            f"  Market Character:   {characterLabel}",
            f"  Confidence (R2):    {confidencePct:.1f}%",
        ]

        if self.hurstByPeriod:
            lines.append("-" * 55)
            lines.append("  Rolling Hurst by Period:")
            for period, hurst in self.hurstByPeriod.items():
                lines.append(f"    {period:>4s}: {hurst:.4f}")

        if self.strategyFit:
            lines.append("-" * 55)
            lines.append("  Strategy Fitness Scores:")
            sortedFit = sorted(self.strategyFit.items(), key=lambda x: x[1], reverse=True)
            for sType, score in sortedFit:
                lines.append(f"    {sType:<20s} {score:.1f}/100")

        dfaExponent = self.details.get("dfaExponent")
        if dfaExponent is not None:
            lines.append("-" * 55)
            lines.append(f"  DFA Exponent (validation): {dfaExponent:.4f}")

        lines.append("=" * 55)
        return "\n".join(lines)

    def _summaryKo(self) -> str:
        """
        Generate Korean-language summary.
        한국어 요약을 생성합니다.

        Returns:
            str: Korean formatted summary string.
        """
        characterLabel = CHARACTER_LABELS_KO.get(self.marketCharacter, self.marketCharacter)
        confidencePct = self.confidence * 100

        lines = [
            "",
            "=" * 55,
            "  프랙탈 시장 분석 (Fractal Market Analysis)",
            "=" * 55,
            f"  허스트 지수:        {self.hurstExponent:.4f}",
            f"  프랙탈 차원:        {self.fractalDimension:.4f}",
            f"  시장 성격:          {characterLabel}",
            f"  신뢰도 (R2):        {confidencePct:.1f}%",
        ]

        if self.hurstByPeriod:
            lines.append("\u2500" * 55)
            lines.append("  기간별 롤링 허스트:")
            periodLabelsKo = {"1m": "1개월", "3m": "3개월", "6m": "6개월", "1y": "1년"}
            for period, hurst in self.hurstByPeriod.items():
                pLabel = periodLabelsKo.get(period, period)
                lines.append(f"    {pLabel:>6s}: {hurst:.4f}")

        if self.strategyFit:
            lines.append("\u2500" * 55)
            lines.append("  전략 적합도 점수:")
            sortedFit = sorted(self.strategyFit.items(), key=lambda x: x[1], reverse=True)
            for sType, score in sortedFit:
                sLabel = STRATEGY_LABELS_KO.get(sType, sType)
                lines.append(f"    {sLabel:<12s} {score:.1f}/100")

        dfaExponent = self.details.get("dfaExponent")
        if dfaExponent is not None:
            lines.append("\u2500" * 55)
            lines.append(f"  DFA 지수 (검증용): {dfaExponent:.4f}")

        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FractalAnalysisResult(H={self.hurstExponent:.4f}, "
            f"D={self.fractalDimension:.4f}, "
            f"character={self.marketCharacter}, "
            f"confidence={self.confidence:.4f})"
        )


class FractalAnalyzer:
    """
    Analyzer for computing Fractal Market Analysis from backtest results.
    백테스트 결과로부터 프랙탈 시장 분석을 수행하는 분석기.

    Uses Rescaled Range (R/S) Analysis to estimate the Hurst Exponent,
    computes the Fractal Dimension, classifies the market character, and
    generates strategy fitness recommendations. Detrended Fluctuation
    Analysis (DFA) is used as an independent validation method.

    재조정 범위(R/S) 분석으로 허스트 지수를 추정하고, 프랙탈 차원을 계산하며,
    시장 성격을 분류하고, 전략 적합도 권고를 생성합니다. 비추세 변동 분석(DFA)은
    독립적인 검증 방법으로 사용됩니다.

    Example:
        >>> analyzer = FractalAnalyzer()
        >>> fractal = analyzer.analyze(backtestResult)
        >>> print(fractal.summary(ko=True))
        >>> print(fractal.strategyFit)
    """

    MIN_DATA_POINTS = 50
    RS_MIN_WINDOW = 20

    def analyze(self, result: BacktestResult, window: int = 252) -> FractalAnalysisResult:
        """
        Perform full fractal market analysis on a backtest result.
        백테스트 결과에 대한 전체 프랙탈 시장 분석을 수행합니다.

        Extracts daily returns from the equity curve, computes Hurst Exponent
        via R/S analysis, derives Fractal Dimension, classifies the market
        character, computes rolling Hurst for sub-periods, generates strategy
        fitness scores, and validates via DFA.

        자산 곡선에서 일별 수익률을 추출하고, R/S 분석으로 허스트 지수를 계산하며,
        프랙탈 차원을 도출하고, 시장 성격을 분류하고, 하위 기간에 대한 롤링 허스트를
        계산하고, 전략 적합도 점수를 생성하고, DFA로 검증합니다.

        Args:
            result (BacktestResult): Completed backtest result containing an
                equity curve. 자산 곡선이 포함된 백테스트 결과.
            window (int): Primary analysis window size in trading days.
                주 분석 윈도우 크기(거래일 수). Defaults to 252.

        Returns:
            FractalAnalysisResult: Complete fractal analysis with Hurst, fractal
                dimension, market character, rolling analysis, and strategy fitness.
                허스트, 프랙탈 차원, 시장 성격, 롤링 분석, 전략 적합도가 포함된
                프랙탈 분석 결과.
        """
        equityCurve = result.equityCurve
        if equityCurve is None or len(equityCurve) < 2:
            return FractalAnalysisResult(details={"error": "no_equity_curve"})

        returns = equityCurve.pct_change().dropna()
        if len(returns) < self.MIN_DATA_POINTS:
            return FractalAnalysisResult(
                details={"error": "insufficient_data", "dataPoints": len(returns)}
            )

        returnsArr = returns.values.astype(np.float64)

        analysisReturns = returnsArr
        if len(returnsArr) > window:
            analysisReturns = returnsArr[-window:]

        hurstExponent, rSquared = self._rescaledRangeHurst(analysisReturns)
        fractalDim = self._fractalDimension(hurstExponent)
        marketChar = self._classifyMarket(hurstExponent)
        hurstByPeriod = self._rollingHurst(returnsArr, window)
        strategyFit = self._strategyFitness(hurstExponent)
        dfaExponent = self._dfa(analysisReturns)

        details = {
            "totalDataPoints": len(returnsArr),
            "analysisWindow": len(analysisReturns),
            "rsHurst": round(hurstExponent, 6),
            "rsRSquared": round(rSquared, 6),
            "dfaExponent": round(dfaExponent, 6),
            "hurstDfaDelta": round(abs(hurstExponent - dfaExponent), 6),
        }

        return FractalAnalysisResult(
            hurstExponent=round(hurstExponent, 6),
            fractalDimension=round(fractalDim, 6),
            marketCharacter=marketChar,
            hurstByPeriod=hurstByPeriod,
            strategyFit=strategyFit,
            confidence=round(rSquared, 6),
            details=details,
        )

    def _rescaledRangeHurst(
        self,
        series: np.ndarray,
        minWindow: int = 20,
    ) -> Tuple[float, float]:
        """
        Estimate Hurst Exponent using Rescaled Range (R/S) Analysis.
        재조정 범위(R/S) 분석을 사용하여 허스트 지수를 추정합니다.

        For each window size n, divides the series into non-overlapping
        segments, computes the R/S statistic for each segment, and averages.
        The Hurst exponent is the slope of log(R/S) vs log(n).

        각 윈도우 크기 n에 대해 시리즈를 비중첩 세그먼트로 나누고, 각 세그먼트의
        R/S 통계량을 계산하여 평균을 구합니다. 허스트 지수는 log(R/S) 대 log(n)의
        기울기입니다.

        Args:
            series (np.ndarray): 1D array of returns. 수익률 1차원 배열.
            minWindow (int): Minimum window size for R/S computation.
                R/S 계산의 최소 윈도우 크기. Defaults to 20.

        Returns:
            Tuple[float, float]: (hurstExponent, rSquared).
                hurstExponent in [0, 1], rSquared of the log-log fit.
                허스트 지수 [0, 1], 로그-로그 적합의 결정계수.
        """
        n = len(series)
        if n < minWindow * 2:
            return 0.5, 0.0

        maxWindow = n // 2
        windowSizes = []
        currentSize = minWindow
        while currentSize <= maxWindow:
            windowSizes.append(currentSize)
            nextSize = int(currentSize * 1.5)
            if nextSize == currentSize:
                nextSize = currentSize + 1
            currentSize = nextSize

        if len(windowSizes) < 3:
            return 0.5, 0.0

        logN = []
        logRS = []

        for wSize in windowSizes:
            numSegments = n // wSize
            if numSegments < 1:
                continue

            rsValues = []
            for seg in range(numSegments):
                startIdx = seg * wSize
                endIdx = startIdx + wSize
                segment = series[startIdx:endIdx]

                segMean = np.mean(segment)
                deviations = segment - segMean
                cumulativeDeviation = np.cumsum(deviations)

                rangeVal = np.max(cumulativeDeviation) - np.min(cumulativeDeviation)
                stdVal = np.std(segment, ddof=1)

                if stdVal > 0:
                    rsValues.append(rangeVal / stdVal)

            if len(rsValues) > 0:
                avgRS = np.mean(rsValues)
                if avgRS > 0:
                    logN.append(np.log(wSize))
                    logRS.append(np.log(avgRS))

        if len(logN) < 3:
            return 0.5, 0.0

        logNArr = np.array(logN, dtype=np.float64)
        logRSArr = np.array(logRS, dtype=np.float64)

        nPts = len(logNArr)
        sumX = np.sum(logNArr)
        sumY = np.sum(logRSArr)
        sumXY = np.sum(logNArr * logRSArr)
        sumX2 = np.sum(logNArr ** 2)

        denominator = nPts * sumX2 - sumX ** 2
        if abs(denominator) < 1e-15:
            return 0.5, 0.0

        slope = (nPts * sumXY - sumX * sumY) / denominator

        yMean = np.mean(logRSArr)
        yPred = slope * logNArr + (sumY - slope * sumX) / nPts
        ssRes = np.sum((logRSArr - yPred) ** 2)
        ssTot = np.sum((logRSArr - yMean) ** 2)

        rSquared = 1.0 - (ssRes / ssTot) if ssTot > 0 else 0.0
        rSquared = float(np.clip(rSquared, 0.0, 1.0))

        hurstExponent = float(np.clip(slope, 0.0, 1.0))

        return hurstExponent, rSquared

    def _dfa(
        self,
        series: np.ndarray,
        minWindow: int = 10,
        maxWindow: Optional[int] = None,
    ) -> float:
        """
        Estimate scaling exponent using Detrended Fluctuation Analysis (DFA).
        비추세 변동 분석(DFA)을 사용하여 스케일링 지수를 추정합니다.

        DFA measures the relationship between the fluctuation function F(n)
        and the window size n. The DFA exponent alpha is related to Hurst:
        alpha ~ H for fractional Brownian motion.

        DFA는 변동 함수 F(n)과 윈도우 크기 n 사이의 관계를 측정합니다.
        DFA 지수 alpha는 분수 브라운 운동에서 H와 유사합니다.

        Steps:
        1. Compute the cumulative sum of the mean-subtracted series (profile)
        2. For each window size n, divide profile into segments
        3. In each segment, fit a linear trend and compute RMS of residuals
        4. F(n) = RMS over all segments
        5. alpha = slope of log(F(n)) vs log(n)

        Args:
            series (np.ndarray): 1D array of returns. 수익률 1차원 배열.
            minWindow (int): Minimum DFA window. 최소 DFA 윈도우. Defaults to 10.
            maxWindow (int, optional): Maximum DFA window.
                최대 DFA 윈도우. None이면 len(series)//4.

        Returns:
            float: DFA exponent (alpha). DFA 지수. ~0.5=random, >0.5=persistent,
                <0.5=anti-persistent.
        """
        n = len(series)
        if n < minWindow * 4:
            return 0.5

        if maxWindow is None:
            maxWindow = n // 4

        if maxWindow < minWindow:
            return 0.5

        profile = np.cumsum(series - np.mean(series))

        windowSizes = []
        currentSize = minWindow
        while currentSize <= maxWindow:
            windowSizes.append(currentSize)
            nextSize = int(currentSize * 1.3)
            if nextSize == currentSize:
                nextSize = currentSize + 1
            currentSize = nextSize

        if len(windowSizes) < 3:
            return 0.5

        logN = []
        logF = []

        for wSize in windowSizes:
            numSegments = n // wSize
            if numSegments < 1:
                continue

            fluctuations = []

            for seg in range(numSegments):
                startIdx = seg * wSize
                endIdx = startIdx + wSize
                segment = profile[startIdx:endIdx]

                xVals = np.arange(wSize, dtype=np.float64)
                sumXl = np.sum(xVals)
                sumYl = np.sum(segment)
                sumXYl = np.sum(xVals * segment)
                sumX2l = np.sum(xVals ** 2)

                denomLocal = wSize * sumX2l - sumXl ** 2
                if abs(denomLocal) < 1e-15:
                    continue

                slopeLocal = (wSize * sumXYl - sumXl * sumYl) / denomLocal
                interceptLocal = (sumYl - slopeLocal * sumXl) / wSize
                trendLocal = slopeLocal * xVals + interceptLocal

                residuals = segment - trendLocal
                rmsLocal = np.sqrt(np.mean(residuals ** 2))
                fluctuations.append(rmsLocal)

            for seg in range(numSegments):
                startIdx = n - (seg + 1) * wSize
                if startIdx < 0:
                    break
                endIdx = startIdx + wSize
                segment = profile[startIdx:endIdx]

                xVals = np.arange(wSize, dtype=np.float64)
                sumXl = np.sum(xVals)
                sumYl = np.sum(segment)
                sumXYl = np.sum(xVals * segment)
                sumX2l = np.sum(xVals ** 2)

                denomLocal = wSize * sumX2l - sumXl ** 2
                if abs(denomLocal) < 1e-15:
                    continue

                slopeLocal = (wSize * sumXYl - sumXl * sumYl) / denomLocal
                interceptLocal = (sumYl - slopeLocal * sumXl) / wSize
                trendLocal = slopeLocal * xVals + interceptLocal

                residuals = segment - trendLocal
                rmsLocal = np.sqrt(np.mean(residuals ** 2))
                fluctuations.append(rmsLocal)

            if len(fluctuations) > 0:
                fN = np.sqrt(np.mean(np.array(fluctuations) ** 2))
                if fN > 0:
                    logN.append(np.log(wSize))
                    logF.append(np.log(fN))

        if len(logN) < 3:
            return 0.5

        logNArr = np.array(logN, dtype=np.float64)
        logFArr = np.array(logF, dtype=np.float64)

        nPts = len(logNArr)
        sumX = np.sum(logNArr)
        sumY = np.sum(logFArr)
        sumXY = np.sum(logNArr * logFArr)
        sumX2 = np.sum(logNArr ** 2)

        denominator = nPts * sumX2 - sumX ** 2
        if abs(denominator) < 1e-15:
            return 0.5

        alpha = (nPts * sumXY - sumX * sumY) / denominator
        alpha = float(np.clip(alpha, 0.0, 2.0))

        return alpha

    def _fractalDimension(self, hurst: float) -> float:
        """
        Compute fractal dimension from Hurst Exponent.
        허스트 지수로부터 프랙탈 차원을 계산합니다.

        The fractal dimension D = 2 - H for a self-affine time series.
        D = 1.0 indicates a smooth line (strong trend), D = 1.5 is Brownian
        motion (random walk), D = 2.0 is space-filling (highly complex).

        자기 유사 시계열에서 프랙탈 차원 D = 2 - H 입니다.
        D = 1.0은 매끈한 선(강한 추세), D = 1.5는 브라운 운동(랜덤 워크),
        D = 2.0은 공간 충전(매우 복잡)을 나타냅니다.

        Args:
            hurst (float): Hurst Exponent in [0, 1]. 허스트 지수 [0, 1].

        Returns:
            float: Fractal dimension in [1.0, 2.0]. 프랙탈 차원 [1.0, 2.0].
        """
        return float(np.clip(2.0 - hurst, 1.0, 2.0))

    def _rollingHurst(
        self,
        series: np.ndarray,
        window: int,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute Hurst Exponent for standard sub-periods (1m, 3m, 6m, 1y).
        표준 하위 기간(1개월, 3개월, 6개월, 1년)에 대한 허스트 지수를 계산합니다.

        For each period, takes the most recent slice of the return series
        matching the period window and computes the R/S Hurst Exponent.
        Periods requiring more data than available are skipped.

        각 기간에 대해, 수익률 시리즈에서 해당 기간 윈도우에 맞는 가장 최근
        슬라이스를 취하고 R/S 허스트 지수를 계산합니다. 가용 데이터보다
        더 많은 데이터가 필요한 기간은 건너뜁니다.

        Args:
            series (np.ndarray): Full return series. 전체 수익률 시리즈.
            window (int): Primary window (unused in this method but kept for API
                consistency). 주 윈도우 (이 메서드에서는 미사용, API 일관성 유지).
            step (int, optional): Step size for rolling (not used in period mode).
                롤링 단계 크기 (기간 모드에서는 미사용).

        Returns:
            Dict[str, float]: Period labels mapped to Hurst Exponent values.
                기간 레이블을 허스트 지수 값에 매핑한 딕셔너리.
        """
        n = len(series)
        result = {}

        for periodLabel, periodWindow in PERIOD_WINDOWS.items():
            if n < periodWindow:
                continue

            periodSlice = series[-periodWindow:]

            if len(periodSlice) < self.RS_MIN_WINDOW * 2:
                continue

            hurst, _ = self._rescaledRangeHurst(periodSlice, minWindow=max(10, periodWindow // 10))
            result[periodLabel] = round(hurst, 4)

        return result

    def _classifyMarket(self, hurst: float) -> str:
        """
        Classify market character based on Hurst Exponent value.
        허스트 지수 값을 기반으로 시장 성격을 분류합니다.

        Classification thresholds:
            - H > 0.55: trending (persistent, long memory)
            - 0.45 <= H <= 0.55: random walk (no memory)
            - H < 0.45: mean-reverting (anti-persistent)

        분류 기준:
            - H > 0.55: 추세 지속형 (장기 기억)
            - 0.45 <= H <= 0.55: 랜덤 워크 (기억 없음)
            - H < 0.45: 평균 회귀형 (반지속형)

        Args:
            hurst (float): Hurst Exponent. 허스트 지수.

        Returns:
            str: "trending", "random", or "meanReverting".
        """
        if hurst > HURST_TRENDING_THRESHOLD:
            return "trending"
        if hurst < HURST_MEAN_REVERTING_THRESHOLD:
            return "meanReverting"
        return "random"

    def _strategyFitness(self, hurst: float) -> Dict[str, float]:
        """
        Generate strategy fitness scores based on the Hurst Exponent.
        허스트 지수를 기반으로 전략 적합도 점수를 생성합니다.

        Maps the Hurst value to fitness scores (0-100) for each strategy type:
            - trendFollowing: scores highest when H >> 0.5
            - meanReversion: scores highest when H << 0.5
            - momentum: similar to trend following but emphasizes persistence
            - breakout: benefits from trending regimes
            - statistical: benefits from mean reversion, penalized by trends

        허스트 값을 각 전략 유형에 대한 적합도 점수(0-100)로 매핑합니다:
            - 추세 추종: H >> 0.5일 때 최고 점수
            - 평균 회귀: H << 0.5일 때 최고 점수
            - 모멘텀: 추세 추종과 유사, 지속성 강조
            - 돌파: 추세 체제에서 유리
            - 통계적 차익: 평균 회귀에서 유리, 추세에서 불리

        Args:
            hurst (float): Hurst Exponent. 허스트 지수.

        Returns:
            Dict[str, float]: Strategy type -> fitness score (0-100).
                전략 유형 -> 적합도 점수 (0-100) 딕셔너리.
        """
        hurstDeviation = hurst - 0.5

        trendFollowingScore = float(np.clip(50.0 + hurstDeviation * 200.0, 0.0, 100.0))

        meanReversionScore = float(np.clip(50.0 - hurstDeviation * 200.0, 0.0, 100.0))

        momentumScore = float(np.clip(40.0 + hurstDeviation * 180.0, 0.0, 100.0))

        breakoutScore = float(np.clip(45.0 + hurstDeviation * 160.0, 0.0, 100.0))

        statisticalScore = float(np.clip(45.0 - hurstDeviation * 150.0, 0.0, 100.0))

        randomPenalty = 1.0 - 2.0 * abs(hurstDeviation)
        randomPenalty = float(np.clip(randomPenalty, 0.0, 1.0))
        if randomPenalty > 0.8:
            penaltyFactor = 0.6 + 0.4 * (1.0 - randomPenalty)
            trendFollowingScore *= penaltyFactor
            meanReversionScore *= penaltyFactor
            momentumScore *= penaltyFactor
            breakoutScore *= penaltyFactor
            statisticalScore *= penaltyFactor

        return {
            "trendFollowing": round(trendFollowingScore, 1),
            "meanReversion": round(meanReversionScore, 1),
            "momentum": round(momentumScore, 1),
            "breakout": round(breakoutScore, 1),
            "statistical": round(statisticalScore, 1),
        }
