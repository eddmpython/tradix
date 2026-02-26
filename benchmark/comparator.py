"""
BenchmarkComparator - 벤치마크 비교

전략 성과를 다양한 벤치마크와 비교 분석

Features:
    - Buy & Hold 비교
    - 시장 지수 비교 (SPY, KOSPI 등)
    - 알파/베타 분석
    - 정보 비율 (Information Ratio)
    - 트래킹 에러
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class BenchmarkMetrics:
    """벤치마크 비교 지표"""
    alpha: float
    beta: float
    correlation: float
    rSquared: float
    informationRatio: float
    trackingError: float
    upCapture: float
    downCapture: float
    activeReturn: float
    treynorRatio: float
    jensenAlpha: float

    def summary(self) -> str:
        lines = [
            "=== 벤치마크 비교 지표 ===",
            f"알파 (연율): {self.alpha:.2%}",
            f"베타: {self.beta:.2f}",
            f"상관계수: {self.correlation:.2f}",
            f"R²: {self.rSquared:.2f}",
            "",
            f"정보 비율: {self.informationRatio:.2f}",
            f"트래킹 에러: {self.trackingError:.2%}",
            f"액티브 수익률: {self.activeReturn:.2%}",
            "",
            f"상승장 포착률: {self.upCapture:.1%}",
            f"하락장 포착률: {self.downCapture:.1%}",
            "",
            f"트레이너 비율: {self.treynorRatio:.2f}",
            f"젠슨 알파: {self.jensenAlpha:.2%}",
        ]
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """비교 결과"""
    strategyName: str
    benchmarkName: str
    strategyMetrics: Dict[str, float]
    benchmarkMetrics: Dict[str, float]
    relativeMetrics: BenchmarkMetrics
    periods: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"성과 비교: {self.strategyName} vs {self.benchmarkName}",
            "=" * 60,
            "",
            f"{'지표':<20} {'전략':>15} {'벤치마크':>15} {'차이':>15}",
            "-" * 60,
        ]

        metrics = ['totalReturn', 'annualReturn', 'volatility', 'sharpeRatio', 'maxDrawdown']
        names = ['총 수익률', '연간 수익률', '변동성', '샤프 비율', '최대 낙폭']

        for metric, name in zip(metrics, names):
            sVal = self.strategyMetrics.get(metric, 0)
            bVal = self.benchmarkMetrics.get(metric, 0)
            diff = sVal - bVal

            if metric in ['totalReturn', 'annualReturn', 'volatility', 'maxDrawdown']:
                lines.append(f"{name:<20} {sVal:>14.2%} {bVal:>14.2%} {diff:>+14.2%}")
            else:
                lines.append(f"{name:<20} {sVal:>15.2f} {bVal:>15.2f} {diff:>+15.2f}")

        lines.extend([
            "",
            self.relativeMetrics.summary(),
            "=" * 60,
        ])

        return "\n".join(lines)


class BenchmarkComparator:
    """
    벤치마크 비교기

    전략의 성과를 다양한 벤치마크와 비교하여
    알파, 베타, 정보비율 등 상대 성과 지표를 계산

    Usage:
        comparator = BenchmarkComparator()

        # 데이터 설정
        comparator.setStrategy(strategyReturns)
        comparator.setBenchmark(benchmarkReturns, 'SPY')

        # 비교 분석
        result = comparator.compare()
        print(result.summary())

        # 기간별 분석
        periodAnalysis = comparator.periodAnalysis(['1Y', '3Y', '5Y'])
    """

    def __init__(self, riskFreeRate: float = 0.02):
        """
        Args:
            riskFreeRate: 무위험 이자율 (연율)
        """
        self.riskFreeRate = riskFreeRate
        self.strategyReturns: pd.Series = None
        self.benchmarkReturns: pd.Series = None
        self.strategyName: str = "Strategy"
        self.benchmarkName: str = "Benchmark"

    def setStrategy(
        self,
        returns: pd.Series,
        name: str = "Strategy"
    ) -> 'BenchmarkComparator':
        """전략 수익률 설정"""
        self.strategyReturns = returns.dropna()
        self.strategyName = name
        return self

    def setBenchmark(
        self,
        returns: pd.Series,
        name: str = "Benchmark"
    ) -> 'BenchmarkComparator':
        """벤치마크 수익률 설정"""
        self.benchmarkReturns = returns.dropna()
        self.benchmarkName = name
        return self

    def compare(self) -> ComparisonResult:
        """
        전략과 벤치마크 비교

        Returns:
            ComparisonResult
        """
        if self.strategyReturns is None or self.benchmarkReturns is None:
            raise ValueError("전략과 벤치마크 수익률을 설정하세요")

        commonIdx = self.strategyReturns.index.intersection(self.benchmarkReturns.index)
        stratRet = self.strategyReturns.loc[commonIdx]
        benchRet = self.benchmarkReturns.loc[commonIdx]

        strategyMetrics = self._calcMetrics(stratRet)
        benchmarkMetrics = self._calcMetrics(benchRet)

        relativeMetrics = self._calcRelativeMetrics(stratRet, benchRet)

        return ComparisonResult(
            strategyName=self.strategyName,
            benchmarkName=self.benchmarkName,
            strategyMetrics=strategyMetrics,
            benchmarkMetrics=benchmarkMetrics,
            relativeMetrics=relativeMetrics,
        )

    def _calcMetrics(self, returns: pd.Series) -> Dict[str, float]:
        """기본 성과 지표 계산"""
        totalReturn = (1 + returns).prod() - 1
        nYears = len(returns) / 252
        annualReturn = (1 + totalReturn) ** (1 / nYears) - 1 if nYears > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annualReturn - self.riskFreeRate) / volatility if volatility > 0 else 0

        cumReturns = (1 + returns).cumprod()
        peak = cumReturns.cummax()
        drawdown = (cumReturns - peak) / peak
        maxDrawdown = drawdown.min()

        return {
            'totalReturn': totalReturn,
            'annualReturn': annualReturn,
            'volatility': volatility,
            'sharpeRatio': sharpe,
            'maxDrawdown': maxDrawdown,
        }

    def _calcRelativeMetrics(
        self,
        strategyReturns: pd.Series,
        benchmarkReturns: pd.Series
    ) -> BenchmarkMetrics:
        """상대 성과 지표 계산"""
        correlation = strategyReturns.corr(benchmarkReturns)

        slope, intercept, rValue, pValue, stdErr = stats.linregress(
            benchmarkReturns.values,
            strategyReturns.values
        )
        beta = slope
        dailyAlpha = intercept
        alpha = dailyAlpha * 252
        rSquared = rValue ** 2

        activeReturns = strategyReturns - benchmarkReturns
        trackingError = activeReturns.std() * np.sqrt(252)
        activeReturn = activeReturns.mean() * 252
        informationRatio = activeReturn / trackingError if trackingError > 0 else 0

        upDays = benchmarkReturns > 0
        downDays = benchmarkReturns < 0

        if upDays.sum() > 0:
            upCapture = strategyReturns[upDays].mean() / benchmarkReturns[upDays].mean()
        else:
            upCapture = 1.0

        if downDays.sum() > 0:
            downCapture = strategyReturns[downDays].mean() / benchmarkReturns[downDays].mean()
        else:
            downCapture = 1.0

        strategyAnnualReturn = strategyReturns.mean() * 252
        benchmarkAnnualReturn = benchmarkReturns.mean() * 252

        treynorRatio = (strategyAnnualReturn - self.riskFreeRate) / beta if beta != 0 else 0

        expectedReturn = self.riskFreeRate + beta * (benchmarkAnnualReturn - self.riskFreeRate)
        jensenAlpha = strategyAnnualReturn - expectedReturn

        return BenchmarkMetrics(
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            rSquared=rSquared,
            informationRatio=informationRatio,
            trackingError=trackingError,
            upCapture=upCapture,
            downCapture=downCapture,
            activeReturn=activeReturn,
            treynorRatio=treynorRatio,
            jensenAlpha=jensenAlpha,
        )

    def periodAnalysis(
        self,
        periods: List[str] = None
    ) -> Dict[str, ComparisonResult]:
        """
        기간별 분석

        Args:
            periods: 분석 기간 리스트 ('1Y', '3Y', '5Y', 'YTD', 'MTD')

        Returns:
            기간별 ComparisonResult
        """
        if periods is None:
            periods = ['1M', '3M', '6M', '1Y', '3Y', 'ALL']

        results = {}
        endDate = self.strategyReturns.index[-1]

        periodDays = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '1Y': 252,
            '2Y': 504,
            '3Y': 756,
            '5Y': 1260,
            'ALL': len(self.strategyReturns),
        }

        for period in periods:
            days = periodDays.get(period, len(self.strategyReturns))
            days = min(days, len(self.strategyReturns))

            stratRet = self.strategyReturns.iloc[-days:]
            benchRet = self.benchmarkReturns.iloc[-days:]

            commonIdx = stratRet.index.intersection(benchRet.index)
            stratRet = stratRet.loc[commonIdx]
            benchRet = benchRet.loc[commonIdx]

            if len(stratRet) < 5:
                continue

            strategyMetrics = self._calcMetrics(stratRet)
            benchmarkMetrics = self._calcMetrics(benchRet)
            relativeMetrics = self._calcRelativeMetrics(stratRet, benchRet)

            results[period] = ComparisonResult(
                strategyName=self.strategyName,
                benchmarkName=self.benchmarkName,
                strategyMetrics=strategyMetrics,
                benchmarkMetrics=benchmarkMetrics,
                relativeMetrics=relativeMetrics,
            )

        return results

    def rollingAnalysis(
        self,
        window: int = 252,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        롤링 분석

        Args:
            window: 롤링 윈도우 크기 (거래일)
            metrics: 계산할 지표 리스트

        Returns:
            롤링 지표 DataFrame
        """
        if metrics is None:
            metrics = ['alpha', 'beta', 'sharpe', 'tracking_error']

        results = []

        for i in range(window, len(self.strategyReturns)):
            stratRet = self.strategyReturns.iloc[i - window:i]
            benchRet = self.benchmarkReturns.iloc[i - window:i]

            commonIdx = stratRet.index.intersection(benchRet.index)
            stratRet = stratRet.loc[commonIdx]
            benchRet = benchRet.loc[commonIdx]

            if len(stratRet) < window * 0.8:
                continue

            row = {'date': self.strategyReturns.index[i]}

            if 'alpha' in metrics or 'beta' in metrics:
                slope, intercept, _, _, _ = stats.linregress(
                    benchRet.values, stratRet.values
                )
                row['beta'] = slope
                row['alpha'] = intercept * 252

            if 'sharpe' in metrics:
                annualRet = stratRet.mean() * 252
                vol = stratRet.std() * np.sqrt(252)
                row['sharpe'] = (annualRet - self.riskFreeRate) / vol if vol > 0 else 0

            if 'tracking_error' in metrics:
                activeRet = stratRet - benchRet
                row['tracking_error'] = activeRet.std() * np.sqrt(252)

            if 'correlation' in metrics:
                row['correlation'] = stratRet.corr(benchRet)

            results.append(row)

        return pd.DataFrame(results)
