# -*- coding: utf-8 -*-
"""
Tradix Monte Carlo Stress Testing Module.

Bootstrap resampling of trade returns to generate thousands of simulated
equity paths, then compute confidence bands, ruin probability, and
distribution of key metrics such as Sharpe ratio and maximum drawdown.

거래 수익률의 부트스트랩 리샘플링을 통해 수천 개의 시뮬레이션 자산 경로를 생성하고,
신뢰 구간, 파산 확률, 샤프 비율 및 최대 낙폭 등 주요 지표의 분포를 계산하는
몬테카를로 스트레스 테스트 모듈입니다.

Features:
    - Bootstrap resampling of trade-level returns with replacement
    - Simulated equity path generation from initial capital
    - Confidence band computation at 50/75/90/95/99% levels
    - Ruin probability estimation (equity below threshold)
    - Per-path Sharpe ratio and maximum drawdown distribution
    - Bilingual summary report (Korean / English)

Usage:
    from tradix.analytics.monteCarloStress import MonteCarloStressAnalyzer

    analyzer = MonteCarloStressAnalyzer()
    stressResult = analyzer.analyze(result, paths=10000, ruinThreshold=0.5)
    print(stressResult.summary())
    print(stressResult.summary(ko=True))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade


TRADING_DAYS_PER_YEAR = 252

MIN_TRADES_REQUIRED = 5

CONFIDENCE_LEVELS = [50, 75, 90, 95, 99]


@dataclass
class MonteCarloStressResult:
    """
    Result container for Monte Carlo stress testing analysis.

    몬테카를로 스트레스 테스트 분석 결과를 담는 데이터 클래스입니다.

    Attributes:
        paths (int): Number of simulation paths generated.
            생성된 시뮬레이션 경로 수.
        confidenceBands (Dict): Confidence band mapping. Keys are level strings
            ('50%', '75%', '90%', '95%', '99%'), values are dicts with
            'upper' and 'lower' final equity values.
            신뢰 구간 매핑. 키는 수준 문자열, 값은 상한/하한 최종 자산 딕셔너리.
        ruinProbability (float): Fraction of paths where equity dropped below
            the ruin threshold at any point. 자산이 파산 임계값 이하로 하락한 경로 비율.
        medianReturn (float): Median total return (%) across all paths.
            전체 경로의 중앙값 총 수익률 (%).
        worstCase (float): Worst path total return (%). 최악 경로 총 수익률 (%).
        bestCase (float): Best path total return (%). 최선 경로 총 수익률 (%).
        sharpeDistribution (List[float]): Sharpe ratio for each simulated path.
            각 시뮬레이션 경로의 샤프 비율.
        mddDistribution (List[float]): Maximum drawdown (%) for each path.
            각 경로의 최대 낙폭 (%).
        details (Dict): Additional statistics and metadata.
            추가 통계 및 메타데이터.

    Example:
        >>> stressResult = analyzer.analyze(result, paths=5000)
        >>> print(stressResult.summary())
    """
    paths: int = 0
    confidenceBands: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ruinProbability: float = 0.0
    medianReturn: float = 0.0
    worstCase: float = 0.0
    bestCase: float = 0.0
    sharpeDistribution: List[float] = field(default_factory=list)
    mddDistribution: List[float] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self, ko: bool = False) -> str:
        """
        Generate a formatted summary report of Monte Carlo stress test results.

        몬테카를로 스트레스 테스트 결과의 포맷된 요약 보고서를 생성합니다.

        Args:
            ko (bool): If True, generate Korean-language summary. If False,
                generate English summary. 한국어 요약 생성 여부.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 구성된 포맷 요약 문자열.
        """
        if ko:
            return self._summaryKo()
        return self._summaryEn()

    def _summaryEn(self) -> str:
        """
        Generate English-language summary report.

        영문 요약 보고서를 생성합니다.

        Returns:
            str: English formatted summary.
        """
        sharpeArr = np.array(self.sharpeDistribution) if self.sharpeDistribution else np.array([0.0])
        mddArr = np.array(self.mddDistribution) if self.mddDistribution else np.array([0.0])

        lines = [
            f"{'='*60}",
            f"  Monte Carlo Stress Test ({self.paths:,} paths)",
            f"{'='*60}",
            f"  Median Return:      {self.medianReturn:+.2f}%",
            f"  Best Case:          {self.bestCase:+.2f}%",
            f"  Worst Case:         {self.worstCase:+.2f}%",
            f"  Ruin Probability:   {self.ruinProbability:.4f} ({self.ruinProbability * 100:.2f}%)",
            f"{'─'*60}",
            f"  Confidence Bands (Final Equity):",
        ]

        for level in CONFIDENCE_LEVELS:
            key = f"{level}%"
            if key in self.confidenceBands:
                band = self.confidenceBands[key]
                lines.append(
                    f"    {key:>4s}:  {band['lower']:>14,.0f} ~ {band['upper']:>14,.0f}"
                )

        lines.extend([
            f"{'─'*60}",
            f"  Sharpe Distribution:",
            f"    Median:  {float(np.median(sharpeArr)):.4f}",
            f"    Mean:    {float(np.mean(sharpeArr)):.4f}",
            f"    Std:     {float(np.std(sharpeArr)):.4f}",
            f"    5th %%:   {float(np.percentile(sharpeArr, 5)):.4f}",
            f"    95th %%:  {float(np.percentile(sharpeArr, 95)):.4f}",
            f"{'─'*60}",
            f"  Max Drawdown Distribution:",
            f"    Median:  {float(np.median(mddArr)):.2f}%",
            f"    Mean:    {float(np.mean(mddArr)):.2f}%",
            f"    Worst:   {float(np.min(mddArr)):.2f}%",
            f"    5th %%:   {float(np.percentile(mddArr, 5)):.2f}%",
            f"    95th %%:  {float(np.percentile(mddArr, 95)):.2f}%",
            f"{'='*60}",
        ])

        return "\n".join(lines)

    def _summaryKo(self) -> str:
        """
        Generate Korean-language summary report.

        한국어 요약 보고서를 생성합니다.

        Returns:
            str: Korean formatted summary.
        """
        sharpeArr = np.array(self.sharpeDistribution) if self.sharpeDistribution else np.array([0.0])
        mddArr = np.array(self.mddDistribution) if self.mddDistribution else np.array([0.0])

        ruinPct = self.ruinProbability * 100

        lines = [
            f"{'='*60}",
            f"  몬테카를로 스트레스 테스트 ({self.paths:,}개 경로)",
            f"{'='*60}",
            f"  중앙값 수익률:      {self.medianReturn:+.2f}%",
            f"  최선 시나리오:      {self.bestCase:+.2f}%",
            f"  최악 시나리오:      {self.worstCase:+.2f}%",
            f"  파산 확률:          {self.ruinProbability:.4f} ({ruinPct:.2f}%)",
            f"{'─'*60}",
            f"  신뢰 구간 (최종 자산):",
        ]

        for level in CONFIDENCE_LEVELS:
            key = f"{level}%"
            if key in self.confidenceBands:
                band = self.confidenceBands[key]
                lines.append(
                    f"    {key:>4s}:  {band['lower']:>14,.0f} ~ {band['upper']:>14,.0f}"
                )

        medianSharpe = float(np.median(sharpeArr))
        meanSharpe = float(np.mean(sharpeArr))
        stdSharpe = float(np.std(sharpeArr))

        medianMdd = float(np.median(mddArr))
        meanMdd = float(np.mean(mddArr))
        worstMdd = float(np.min(mddArr))

        lines.extend([
            f"{'─'*60}",
            f"  샤프 비율 분포:",
            f"    중앙값:  {medianSharpe:.4f}",
            f"    평균:    {meanSharpe:.4f}",
            f"    표준편차: {stdSharpe:.4f}",
            f"{'─'*60}",
            f"  최대 낙폭 분포:",
            f"    중앙값:  {medianMdd:.2f}%",
            f"    평균:    {meanMdd:.2f}%",
            f"    최악:    {worstMdd:.2f}%",
            f"{'='*60}",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MonteCarloStressResult(paths={self.paths}, "
            f"medianReturn={self.medianReturn:+.2f}%, "
            f"ruin={self.ruinProbability:.4f}, "
            f"worstCase={self.worstCase:+.2f}%, "
            f"bestCase={self.bestCase:+.2f}%)"
        )


class MonteCarloStressAnalyzer:
    """
    Monte Carlo stress testing analyzer for backtest results.

    Performs bootstrap resampling of trade-level returns to generate thousands
    of simulated equity paths, computing confidence bands, ruin probability,
    and distributions of Sharpe ratio and maximum drawdown.

    백테스트 결과에 대한 몬테카를로 스트레스 테스트 분석기입니다.
    거래 수준 수익률의 부트스트랩 리샘플링을 통해 수천 개의 시뮬레이션 자산 경로를
    생성하고, 신뢰 구간, 파산 확률, 샤프 비율 및 최대 낙폭의 분포를 계산합니다.

    Example:
        >>> analyzer = MonteCarloStressAnalyzer()
        >>> stressResult = analyzer.analyze(backtestResult, paths=10000)
        >>> print(stressResult.summary())
        >>> print(stressResult.summary(ko=True))
    """

    def analyze(
        self,
        result: BacktestResult,
        paths: int = 10000,
        ruinThreshold: float = 0.5,
    ) -> MonteCarloStressResult:
        """
        Run Monte Carlo stress test on a backtest result.

        백테스트 결과에 대한 몬테카를로 스트레스 테스트를 실행합니다.

        Extracts trade-level profit/loss percentages, performs bootstrap
        resampling to generate simulated equity paths, and computes
        comprehensive risk statistics.

        거래별 손익률을 추출하고, 부트스트랩 리샘플링으로 시뮬레이션 자산 경로를
        생성한 후, 종합 위험 통계를 계산합니다.

        Args:
            result (BacktestResult): Completed backtest result containing trades
                and initial cash. 거래 내역과 초기 자본이 포함된 백테스트 결과.
            paths (int): Number of simulation paths to generate (default: 10000).
                생성할 시뮬레이션 경로 수 (기본값: 10000).
            ruinThreshold (float): Ruin threshold as fraction of initial cash
                (default: 0.5, meaning ruin if equity drops below 50% of initial).
                초기 자본 대비 파산 임계 비율 (기본값: 0.5).

        Returns:
            MonteCarloStressResult: Complete stress test result with confidence
                bands, ruin probability, and metric distributions.
                신뢰 구간, 파산 확률, 지표 분포가 포함된 스트레스 테스트 결과.

        Example:
            >>> analyzer = MonteCarloStressAnalyzer()
            >>> stressResult = analyzer.analyze(result, paths=5000, ruinThreshold=0.3)
            >>> print(f"Ruin probability: {stressResult.ruinProbability:.4f}")
        """
        closedTrades = [t for t in result.trades if t.isClosed]
        initialCash = result.initialCash

        tradeReturns = self._extractTradeReturns(closedTrades)

        if len(tradeReturns) < MIN_TRADES_REQUIRED:
            return MonteCarloStressResult(
                paths=paths,
                confidenceBands={},
                ruinProbability=0.0,
                medianReturn=0.0,
                worstCase=0.0,
                bestCase=0.0,
                sharpeDistribution=[],
                mddDistribution=[],
                details={'error': 'insufficient_trades', 'tradeCount': len(tradeReturns)},
            )

        nTrades = len(tradeReturns)

        bootstrapped = self._bootstrapReturns(tradeReturns, nTrades, paths)

        equityPaths = self._equityPaths(bootstrapped, initialCash)

        finalEquities = equityPaths[:, -1]

        finalReturns = (finalEquities / initialCash - 1.0) * 100.0

        confidenceBands = self._confidenceBands(finalEquities, initialCash, CONFIDENCE_LEVELS)

        ruinLevel = initialCash * ruinThreshold
        ruinProb = self._ruinProbability(equityPaths, ruinLevel)

        medianReturn = float(np.median(finalReturns))
        worstCase = float(np.min(finalReturns))
        bestCase = float(np.max(finalReturns))

        sharpeDistribution = []
        mddDistribution = []

        for i in range(paths):
            pathEquity = equityPaths[i, :]
            sharpeDistribution.append(self._pathSharpe(pathEquity))
            mddDistribution.append(self._pathMaxDrawdown(pathEquity))

        meanReturn = float(np.mean(finalReturns))
        stdReturn = float(np.std(finalReturns))
        sharpeArr = np.array(sharpeDistribution)
        mddArr = np.array(mddDistribution)

        details = {
            'tradeCount': nTrades,
            'initialCash': initialCash,
            'ruinThreshold': ruinThreshold,
            'ruinLevel': ruinLevel,
            'meanReturn': meanReturn,
            'stdReturn': stdReturn,
            'returnSkewness': float(pd.Series(finalReturns).skew()),
            'returnKurtosis': float(pd.Series(finalReturns).kurtosis()),
            'medianSharpe': float(np.median(sharpeArr)),
            'meanSharpe': float(np.mean(sharpeArr)),
            'medianMdd': float(np.median(mddArr)),
            'meanMdd': float(np.mean(mddArr)),
            'worstMdd': float(np.min(mddArr)),
            'var95': float(np.percentile(finalReturns, 5)),
            'var99': float(np.percentile(finalReturns, 1)),
            'positivePathRatio': float(np.mean(finalReturns > 0)),
        }

        return MonteCarloStressResult(
            paths=paths,
            confidenceBands=confidenceBands,
            ruinProbability=ruinProb,
            medianReturn=medianReturn,
            worstCase=worstCase,
            bestCase=bestCase,
            sharpeDistribution=sharpeDistribution,
            mddDistribution=mddDistribution,
            details=details,
        )

    def _extractTradeReturns(self, closedTrades: List[Trade]) -> np.ndarray:
        """
        Extract percentage returns from closed trades.

        청산된 거래에서 수익률(%)을 추출합니다.

        Args:
            closedTrades (List[Trade]): List of closed Trade objects.
                청산된 거래 객체 목록.

        Returns:
            np.ndarray: Array of trade return percentages (e.g., 0.05 = 5%).
                거래 수익률 배열 (예: 0.05 = 5%).
        """
        returns = []
        for trade in closedTrades:
            entryValue = trade.entryPrice * trade.quantity
            if entryValue > 0:
                returns.append(trade.pnl / entryValue)
        return np.array(returns, dtype=np.float64)

    def _bootstrapReturns(
        self,
        tradeReturns: np.ndarray,
        nTrades: int,
        paths: int,
    ) -> np.ndarray:
        """
        Generate bootstrap resampled trade return sequences.

        부트스트랩 리샘플링된 거래 수익률 시퀀스를 생성합니다.

        Randomly samples trade returns with replacement to create multiple
        simulated trade sequences of the same length as the original.

        원본과 같은 길이의 여러 시뮬레이션 거래 시퀀스를 생성하기 위해
        거래 수익률을 복원 추출로 랜덤 샘플링합니다.

        Args:
            tradeReturns (np.ndarray): Original trade return array.
                원본 거래 수익률 배열.
            nTrades (int): Number of trades per simulated path.
                시뮬레이션 경로당 거래 수.
            paths (int): Number of simulation paths to generate.
                생성할 시뮬레이션 경로 수.

        Returns:
            np.ndarray: Shape (paths, nTrades) array of resampled returns.
                (paths, nTrades) 형태의 리샘플링된 수익률 배열.
        """
        indices = np.random.randint(0, len(tradeReturns), size=(paths, nTrades))
        return tradeReturns[indices]

    def _equityPaths(
        self,
        bootstrapped: np.ndarray,
        initialCash: float,
    ) -> np.ndarray:
        """
        Compute cumulative equity paths from bootstrapped trade returns.

        부트스트랩된 거래 수익률로부터 누적 자산 경로를 계산합니다.

        For each simulated path, applies trade returns sequentially to the
        initial cash to build a cumulative equity curve.

        각 시뮬레이션 경로에 대해 거래 수익률을 초기 자본에 순차적으로 적용하여
        누적 자산 곡선을 생성합니다.

        Args:
            bootstrapped (np.ndarray): Shape (paths, nTrades) resampled returns.
                (paths, nTrades) 형태의 리샘플링된 수익률.
            initialCash (float): Starting equity value. 시작 자산 가치.

        Returns:
            np.ndarray: Shape (paths, nTrades+1) equity paths starting at
                initialCash. (paths, nTrades+1) 형태의 자산 경로.
        """
        nPaths, nTrades = bootstrapped.shape

        growthFactors = 1.0 + bootstrapped

        cumulativeGrowth = np.cumprod(growthFactors, axis=1)

        equityPaths = np.empty((nPaths, nTrades + 1), dtype=np.float64)
        equityPaths[:, 0] = initialCash
        equityPaths[:, 1:] = initialCash * cumulativeGrowth

        return equityPaths

    def _confidenceBands(
        self,
        finalEquities: np.ndarray,
        initialCash: float,
        levels: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute confidence bands from final equity distribution.

        최종 자산 분포로부터 신뢰 구간을 계산합니다.

        For each confidence level, computes the symmetric upper and lower
        bounds of the final equity distribution.

        각 신뢰 수준에 대해 최종 자산 분포의 대칭 상한/하한 경계를 계산합니다.

        Args:
            finalEquities (np.ndarray): Final equity values across all paths.
                전체 경로의 최종 자산 값.
            initialCash (float): Initial cash amount. 초기 자본금.
            levels (List[int]): Confidence levels to compute (e.g., [50, 75, 90]).
                계산할 신뢰 수준 (예: [50, 75, 90]).

        Returns:
            Dict[str, Dict[str, float]]: Mapping from level string to dict with
                'upper' and 'lower' keys. 수준 문자열에서 상한/하한 딕셔너리 매핑.
        """
        bands = {}

        for level in levels:
            lowerPct = (100 - level) / 2.0
            upperPct = 100 - lowerPct

            lowerVal = float(np.percentile(finalEquities, lowerPct))
            upperVal = float(np.percentile(finalEquities, upperPct))

            key = f"{level}%"
            bands[key] = {
                'upper': upperVal,
                'lower': lowerVal,
            }

        return bands

    def _ruinProbability(
        self,
        equityPaths: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Calculate the probability of ruin across simulated equity paths.

        시뮬레이션된 자산 경로에서 파산 확률을 계산합니다.

        A path is considered "ruined" if the equity drops below the specified
        threshold at any point during the simulation.

        시뮬레이션 중 어느 시점에서든 자산이 지정된 임계값 이하로 하락하면
        해당 경로는 "파산"으로 간주됩니다.

        Args:
            equityPaths (np.ndarray): Shape (paths, nTrades+1) equity paths.
                (paths, nTrades+1) 형태의 자산 경로.
            threshold (float): Absolute equity level defining ruin.
                파산을 정의하는 절대 자산 수준.

        Returns:
            float: Fraction of paths that experienced ruin (0.0 to 1.0).
                파산을 경험한 경로의 비율 (0.0~1.0).
        """
        minEquities = np.min(equityPaths, axis=1)
        ruinCount = np.sum(minEquities < threshold)
        return float(ruinCount / len(equityPaths))

    def _pathSharpe(self, equityPath: np.ndarray) -> float:
        """
        Compute annualized Sharpe ratio for a single equity path.

        단일 자산 경로의 연환산 샤프 비율을 계산합니다.

        Calculates step-by-step returns from the equity path, then computes
        the annualized Sharpe ratio assuming zero risk-free rate.

        자산 경로에서 단계별 수익률을 계산한 후, 무위험 수익률 0을 가정하여
        연환산 샤프 비율을 계산합니다.

        Args:
            equityPath (np.ndarray): Single equity path array.
                단일 자산 경로 배열.

        Returns:
            float: Annualized Sharpe ratio. 연환산 샤프 비율.
        """
        if len(equityPath) < 2:
            return 0.0

        stepReturns = np.diff(equityPath) / equityPath[:-1]

        meanReturn = np.mean(stepReturns)
        stdReturn = np.std(stepReturns, ddof=1)

        if stdReturn == 0.0 or np.isnan(stdReturn):
            return 0.0

        tradesPerYear = min(len(stepReturns), TRADING_DAYS_PER_YEAR)
        annualizedSharpe = (meanReturn / stdReturn) * np.sqrt(tradesPerYear)

        return float(annualizedSharpe)

    def _pathMaxDrawdown(self, equityPath: np.ndarray) -> float:
        """
        Compute maximum drawdown percentage for a single equity path.

        단일 자산 경로의 최대 낙폭(%)을 계산합니다.

        Finds the largest peak-to-trough decline as a percentage of the
        peak value across the entire equity path.

        전체 자산 경로에서 고점 대비 최대 하락폭을 백분율로 계산합니다.

        Args:
            equityPath (np.ndarray): Single equity path array.
                단일 자산 경로 배열.

        Returns:
            float: Maximum drawdown as a negative percentage (e.g., -15.3).
                최대 낙폭 (음수 백분율, 예: -15.3).
        """
        if len(equityPath) < 2:
            return 0.0

        cumMax = np.maximum.accumulate(equityPath)

        mask = cumMax > 0
        drawdowns = np.where(mask, (equityPath - cumMax) / cumMax, 0.0)

        maxDrawdown = float(np.min(drawdowns)) * 100.0

        return maxDrawdown

    def __repr__(self) -> str:
        return "MonteCarloStressAnalyzer()"
