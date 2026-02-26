"""Tradex Statistical Arbitrage Module.

Provides cointegration-based pair trading and statistical arbitrage tools,
including spread analysis, Z-score signal generation, and simple backtesting.

통계적 차익거래 모듈 - 공적분 기반 페어 트레이딩 및 스프레드 분석 도구를
제공하며, RegimeForecaster와 연동하여 레짐별 최적 페어 선택을 지원합니다.

Features:
    - Cointegration testing with simplified ADF (no statsmodels dependency)
    - Half-life estimation for mean-reversion speed assessment
    - Pair discovery across a universe of assets
    - Z-score-based spread signal generation (entry / exit / stop-loss)
    - Position sizing and simple pair trading backtest
    - Regime-aware entry/exit threshold adjustment

Usage:
    from tradex.quant.statarb import StatArbAnalyzer, PairTrading

    analyzer = StatArbAnalyzer()
    result = analyzer.testCointegration(prices_a, prices_b)
    pairs = analyzer.findPairs(price_df, topN=5)

    pair = PairTrading(hedgeRatio=result.hedgeRatio)
    signals = pair.generateSignals(prices_a, prices_b)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class CointegrationResult:
    """Result of a cointegration test between two assets.

    두 자산 간 공적분 검정 결과를 담는 데이터 클래스.

    Attributes:
        asset1: Name of the first asset (첫 번째 자산 이름).
        asset2: Name of the second asset (두 번째 자산 이름).
        isCointegrated: Whether the pair passes the cointegration test (공적분 여부).
        adfStat: ADF test statistic (ADF 검정 통계량).
        pValue: Approximate p-value of the ADF test (근사 p-값).
        criticalValues: Critical values at 1%, 5%, 10% levels (임계값).
        hedgeRatio: OLS hedge ratio (log-price regression slope) (헷지 비율).
        halfLife: Estimated mean-reversion half-life in days (평균 회귀 반감기, 일).
        correlation: Pearson correlation of the two price series (가격 상관계수).
    """
    asset1: str
    asset2: str
    isCointegrated: bool
    adfStat: float
    pValue: float
    criticalValues: Dict[str, float]
    hedgeRatio: float
    halfLife: float
    correlation: float

    def summary(self) -> str:
        """Generate a formatted summary of the cointegration test result.

        Returns:
            Multi-line string with test outcome, ADF statistic, hedge ratio, etc.
        """
        status = "공적분 O" if self.isCointegrated else "공적분 X"
        return (
            f"=== 공적분 검정: {self.asset1} - {self.asset2} ===\n"
            f"결과: {status}\n"
            f"ADF 통계량: {self.adfStat:.4f}\n"
            f"p-value: {self.pValue:.4f}\n"
            f"헷지 비율: {self.hedgeRatio:.4f}\n"
            f"반감기: {self.halfLife:.1f}일\n"
            f"상관계수: {self.correlation:.4f}"
        )


class SignalType(Enum):
    """Spread trading signal types.

    스프레드 트레이딩 시그널 유형.

    Attributes:
        LONG_SPREAD: Go long the spread (스프레드 매수).
        SHORT_SPREAD: Go short the spread (스프레드 매도).
        CLOSE: Close the current position (포지션 청산).
        HOLD: No action (관망).
    """
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class SpreadSignal:
    """A single spread trading signal at a specific point in time.

    특정 시점의 스프레드 트레이딩 시그널.

    Attributes:
        date: Signal timestamp (시그널 시점).
        signalType: Type of signal action (시그널 유형).
        spread: Current spread value (현재 스프레드 값).
        zScore: Z-score of the spread (스프레드 Z-점수).
        confidence: Signal confidence level 0-1 (시그널 신뢰도).
        asset1Weight: Portfolio weight for asset 1 (자산1 비중).
        asset2Weight: Portfolio weight for asset 2 (자산2 비중).
    """
    date: pd.Timestamp
    signalType: SignalType
    spread: float
    zScore: float
    confidence: float
    asset1Weight: float
    asset2Weight: float

    def summary(self) -> str:
        """Generate a formatted summary of this spread signal.

        Returns:
            Multi-line string with date, signal type, spread, Z-score, and weights.
        """
        return (
            f"[{self.date}] {self.signalType.value}\n"
            f"  스프레드: {self.spread:.4f}, Z-Score: {self.zScore:.2f}\n"
            f"  비중: {self.asset1Weight:.2%} / {self.asset2Weight:.2%}"
        )


class StatArbAnalyzer:
    """Statistical arbitrage analyzer for cointegration testing and pair discovery.

    Tests cointegration relationships between asset pairs, discovers candidate
    pairs from a price universe, and provides spread analysis with Bollinger-style
    bands around the rolling Z-score.

    통계적 차익거래 분석기 - 공적분 검정, 페어 후보 탐색, 스프레드 분석을 수행합니다.

    Attributes:
        significance: Significance level for ADF test (유의 수준).
        minHalfLife: Minimum acceptable half-life in days (최소 반감기).
        maxHalfLife: Maximum acceptable half-life in days (최대 반감기).

    Example:
        >>> analyzer = StatArbAnalyzer(significance=0.05)
        >>> result = analyzer.testCointegration(prices_a, prices_b)
        >>> pairs = analyzer.findPairs(price_df, topN=5)
        >>> spread_df = analyzer.analyzeSpread(prices_a, prices_b)
    """

    def __init__(
        self,
        significance: float = 0.05,
        minHalfLife: int = 5,
        maxHalfLife: int = 60,
    ):
        """Initialize the statistical arbitrage analyzer.

        Args:
            significance: Significance level for cointegration test (유의 수준).
            minHalfLife: Minimum acceptable half-life in days (최소 반감기, 일).
            maxHalfLife: Maximum acceptable half-life in days (최대 반감기, 일).
        """
        self.significance = significance
        self.minHalfLife = minHalfLife
        self.maxHalfLife = maxHalfLife

    def testCointegration(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        asset1Name: str = "Asset1",
        asset2Name: str = "Asset2",
    ) -> CointegrationResult:
        """Test for cointegration between two price series.

        Performs log-price OLS regression to obtain the hedge ratio, then runs
        a simplified ADF test on the spread residuals.

        Args:
            prices1: Price series for asset 1 (자산1 가격 시리즈).
            prices2: Price series for asset 2 (자산2 가격 시리즈).
            asset1Name: Display name for asset 1 (자산1 이름).
            asset2Name: Display name for asset 2 (자산2 이름).

        Returns:
            CointegrationResult with test outcome, hedge ratio, and half-life.
        """
        commonIdx = prices1.index.intersection(prices2.index)
        p1 = prices1.loc[commonIdx].values
        p2 = prices2.loc[commonIdx].values

        log1 = np.log(p1)
        log2 = np.log(p2)

        X = np.column_stack([np.ones(len(log2)), log2])
        beta, _, _, _ = np.linalg.lstsq(X, log1, rcond=None)
        hedgeRatio = beta[1]

        spread = log1 - hedgeRatio * log2

        adfStat, pValue, critVals = self._adfTest(spread)

        halfLife = self._calcHalfLife(spread)

        correlation = np.corrcoef(p1, p2)[0, 1]

        isCointegrated = (
            pValue < self.significance
            and self.minHalfLife <= halfLife <= self.maxHalfLife
        )

        return CointegrationResult(
            asset1=asset1Name,
            asset2=asset2Name,
            isCointegrated=isCointegrated,
            adfStat=adfStat,
            pValue=pValue,
            criticalValues=critVals,
            hedgeRatio=hedgeRatio,
            halfLife=halfLife,
            correlation=correlation,
        )

    def _adfTest(self, series: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """Run a simplified Augmented Dickey-Fuller test (no statsmodels dependency).

        Args:
            series: Spread series to test for stationarity.

        Returns:
            Tuple of (t-statistic, approximate p-value, critical values dict).
        """
        y = series[1:]
        yLag = series[:-1]
        dy = y - yLag

        X = np.column_stack([np.ones(len(yLag)), yLag])
        beta, residuals, _, _ = np.linalg.lstsq(X, dy, rcond=None)

        gamma = beta[1]

        n = len(dy)
        mse = np.sum((dy - X @ beta) ** 2) / (n - 2)
        se = np.sqrt(mse * np.linalg.inv(X.T @ X)[1, 1])
        tStat = gamma / se if se > 0 else 0

        critVals = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57,
        }

        if tStat < critVals['1%']:
            pValue = 0.001
        elif tStat < critVals['5%']:
            pValue = 0.03
        elif tStat < critVals['10%']:
            pValue = 0.07
        else:
            pValue = 0.5 + 0.5 * (1 - stats.norm.cdf(abs(tStat)))

        return tStat, pValue, critVals

    def _calcHalfLife(self, spread: np.ndarray) -> float:
        """Estimate the mean-reversion half-life from an AR(1) regression on the spread.

        Args:
            spread: Spread series array.

        Returns:
            Half-life in number of observations (infinity if non-mean-reverting).
        """
        y = spread[1:]
        yLag = spread[:-1]
        dy = y - yLag

        X = np.column_stack([np.ones(len(yLag)), yLag])
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)

        theta = -beta[1]
        if theta <= 0:
            return float('inf')

        halfLife = np.log(2) / theta
        return halfLife

    def findPairs(
        self,
        priceData: pd.DataFrame,
        threshold: float = None,
        topN: int = 10,
    ) -> List[CointegrationResult]:
        """Discover cointegrated pairs from a universe of asset prices.

        Tests all pairwise combinations and returns the top candidates sorted
        by p-value.

        Args:
            priceData: DataFrame where each column is an asset's price series
                       (자산별 가격 DataFrame).
            threshold: p-value threshold. Defaults to self.significance
                       (p-value 임계값).
            topN: Maximum number of pairs to return (상위 N개 페어 반환).

        Returns:
            List of CointegrationResult for pairs that pass the test, sorted by
            p-value ascending (공적분 통과 페어 리스트).
        """
        if threshold is None:
            threshold = self.significance

        assets = list(priceData.columns)
        results = []

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                result = self.testCointegration(
                    priceData[assets[i]],
                    priceData[assets[j]],
                    assets[i],
                    assets[j],
                )

                if result.isCointegrated:
                    results.append(result)

        results.sort(key=lambda x: x.pValue)
        return results[:topN]

    def analyzeSpread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedgeRatio: float = None,
    ) -> pd.DataFrame:
        """Analyze the log-price spread between two assets with rolling Z-score bands.

        Args:
            prices1: Price series for asset 1 (자산1 가격 시리즈).
            prices2: Price series for asset 2 (자산2 가격 시리즈).
            hedgeRatio: Hedge ratio to use. If None, estimated via OLS
                        (헷지 비율, None이면 OLS로 추정).

        Returns:
            DataFrame with columns: spread, zScore, mean, upperBand, lowerBand
            (스프레드 분석 DataFrame).
        """
        commonIdx = prices1.index.intersection(prices2.index)
        p1 = prices1.loc[commonIdx]
        p2 = prices2.loc[commonIdx]

        log1 = np.log(p1)
        log2 = np.log(p2)

        if hedgeRatio is None:
            X = np.column_stack([np.ones(len(log2)), log2.values])
            beta, _, _, _ = np.linalg.lstsq(X, log1.values, rcond=None)
            hedgeRatio = beta[1]

        spread = log1 - hedgeRatio * log2

        mean = spread.rolling(20).mean()
        std = spread.rolling(20).std()
        zScore = (spread - mean) / std

        return pd.DataFrame({
            'spread': spread,
            'zScore': zScore,
            'mean': mean,
            'upperBand': mean + 2 * std,
            'lowerBand': mean - 2 * std,
        })


class PairTrading:
    """Cointegration-based pair trading strategy with Z-score signals.

    Generates entry, exit, and stop-loss signals based on rolling Z-scores of the
    log-price spread, and provides position sizing and simple backtesting.

    공적분 기반 페어 트레이딩 전략 - 롤링 Z-점수 기반 진입/청산/손절 시그널을
    생성하고, 포지션 계산 및 간단한 백테스트를 지원합니다.

    Attributes:
        hedgeRatio: Hedge ratio between the two assets (헷지 비율).
        entryZ: Z-score threshold for entry (진입 Z-점수).
        exitZ: Z-score threshold for exit (청산 Z-점수).
        stopLossZ: Z-score threshold for stop-loss (손절 Z-점수).
        lookbackPeriod: Rolling window for Z-score computation (Z-점수 계산 기간).

    Example:
        >>> pair = PairTrading(hedgeRatio=0.8, entryZ=2.0)
        >>> signals = pair.generateSignals(prices_a, prices_b)
        >>> positions = pair.calcPositions(signals, capital=10_000_000)
    """

    def __init__(
        self,
        hedgeRatio: float = 1.0,
        entryZ: float = 2.0,
        exitZ: float = 0.5,
        stopLossZ: float = 3.5,
        lookbackPeriod: int = 20,
    ):
        """Initialize the pair trading strategy.

        Args:
            hedgeRatio: Hedge ratio between asset 1 and asset 2 (헷지 비율).
            entryZ: Z-score threshold for entry signals (진입 Z-점수).
            exitZ: Z-score threshold for exit signals (청산 Z-점수).
            stopLossZ: Z-score threshold for stop-loss triggers (손절 Z-점수).
            lookbackPeriod: Rolling window for Z-score calculation (Z-점수 계산 기간).
        """
        self.hedgeRatio = hedgeRatio
        self.entryZ = entryZ
        self.exitZ = exitZ
        self.stopLossZ = stopLossZ
        self.lookbackPeriod = lookbackPeriod

    def generateSignals(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> List[SpreadSignal]:
        """Generate pair trading signals from two price series.

        Args:
            prices1: Price series for asset 1 (자산1 가격 시리즈).
            prices2: Price series for asset 2 (자산2 가격 시리즈).

        Returns:
            List of SpreadSignal instances for actionable signals only
            (실행 가능한 시그널만 포함된 SpreadSignal 리스트).
        """
        commonIdx = prices1.index.intersection(prices2.index)
        p1 = prices1.loc[commonIdx]
        p2 = prices2.loc[commonIdx]

        log1 = np.log(p1)
        log2 = np.log(p2)
        spread = log1 - self.hedgeRatio * log2

        mean = spread.rolling(self.lookbackPeriod).mean()
        std = spread.rolling(self.lookbackPeriod).std()
        zScore = (spread - mean) / std

        signals = []
        currentPosition = SignalType.HOLD

        for i in range(self.lookbackPeriod, len(spread)):
            date = spread.index[i]
            z = zScore.iloc[i]
            s = spread.iloc[i]

            if np.isnan(z):
                continue

            if currentPosition == SignalType.HOLD:
                if z > self.entryZ:
                    signalType = SignalType.SHORT_SPREAD
                    currentPosition = SignalType.SHORT_SPREAD
                elif z < -self.entryZ:
                    signalType = SignalType.LONG_SPREAD
                    currentPosition = SignalType.LONG_SPREAD
                else:
                    signalType = SignalType.HOLD

            elif currentPosition == SignalType.LONG_SPREAD:
                if z > -self.exitZ or z < -self.stopLossZ:
                    signalType = SignalType.CLOSE
                    currentPosition = SignalType.HOLD
                else:
                    signalType = SignalType.HOLD

            elif currentPosition == SignalType.SHORT_SPREAD:
                if z < self.exitZ or z > self.stopLossZ:
                    signalType = SignalType.CLOSE
                    currentPosition = SignalType.HOLD
                else:
                    signalType = SignalType.HOLD

            if signalType != SignalType.HOLD:
                weight1, weight2 = self._calcWeights(signalType)
                confidence = min(abs(z) / self.entryZ, 1.0)

                signals.append(SpreadSignal(
                    date=date,
                    signalType=signalType,
                    spread=s,
                    zScore=z,
                    confidence=confidence,
                    asset1Weight=weight1,
                    asset2Weight=weight2,
                ))

        return signals

    def _calcWeights(self, signalType: SignalType) -> Tuple[float, float]:
        """Compute normalized position weights for each leg of the pair.

        Args:
            signalType: The signal type (long spread, short spread, or close).

        Returns:
            Tuple of (asset1_weight, asset2_weight).
        """
        normalizedHedge = 1 / (1 + self.hedgeRatio)

        if signalType == SignalType.LONG_SPREAD:
            return normalizedHedge, -normalizedHedge * self.hedgeRatio
        elif signalType == SignalType.SHORT_SPREAD:
            return -normalizedHedge, normalizedHedge * self.hedgeRatio
        else:
            return 0.0, 0.0

    def calcPositions(
        self,
        signals: List[SpreadSignal],
        capital: float = 10_000_000,
    ) -> pd.DataFrame:
        """Convert signals into monetary position sizes.

        Args:
            signals: List of SpreadSignal instances (SpreadSignal 리스트).
            capital: Total capital to allocate (투자 자본금).

        Returns:
            DataFrame with columns: date, signal, zScore, asset1Position,
            asset2Position, netExposure (포지션 DataFrame).
        """
        positions = []

        for signal in signals:
            pos1 = capital * signal.asset1Weight * signal.confidence
            pos2 = capital * signal.asset2Weight * signal.confidence

            positions.append({
                'date': signal.date,
                'signal': signal.signalType.value,
                'zScore': signal.zScore,
                'asset1Position': pos1,
                'asset2Position': pos2,
                'netExposure': pos1 + pos2,
            })

        return pd.DataFrame(positions)

    def backtest(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        capital: float = 10_000_000,
    ) -> Dict[str, float]:
        """Run a simple backtest of the pair trading strategy.

        Args:
            prices1: Price series for asset 1 (자산1 가격 시리즈).
            prices2: Price series for asset 2 (자산2 가격 시리즈).
            capital: Initial capital (초기 자본금).

        Returns:
            Dict with totalReturn, avgReturn, winRate, nTrades (백테스트 결과).
        """
        signals = self.generateSignals(prices1, prices2)

        if not signals:
            return {'totalReturn': 0, 'nTrades': 0}

        returns = []
        entryPrices = None
        entryWeights = None

        for signal in signals:
            p1 = prices1.loc[signal.date]
            p2 = prices2.loc[signal.date]

            if signal.signalType in [SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD]:
                entryPrices = (p1, p2)
                entryWeights = (signal.asset1Weight, signal.asset2Weight)

            elif signal.signalType == SignalType.CLOSE and entryPrices:
                ret1 = (p1 - entryPrices[0]) / entryPrices[0]
                ret2 = (p2 - entryPrices[1]) / entryPrices[1]

                tradeReturn = entryWeights[0] * ret1 + entryWeights[1] * ret2
                returns.append(tradeReturn)

                entryPrices = None
                entryWeights = None

        totalReturn = np.sum(returns) if returns else 0
        avgReturn = np.mean(returns) if returns else 0
        winRate = np.mean([r > 0 for r in returns]) if returns else 0

        return {
            'totalReturn': totalReturn,
            'avgReturn': avgReturn,
            'winRate': winRate,
            'nTrades': len(returns),
        }
