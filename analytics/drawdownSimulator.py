# -*- coding: utf-8 -*-
"""
Tradix Drawdown Simulator Module.

Simulates and analyzes historical drawdown scenarios from backtest results,
providing comprehensive drawdown metrics including magnitude, duration,
recovery statistics, and risk-adjusted indicators.

백테스트 결과로부터 역사적 낙폭 시나리오를 시뮬레이션하고 분석하는 모듈입니다.
낙폭 크기, 지속 기간, 회복 통계, 위험 조정 지표 등 종합적인 낙폭 분석을 제공합니다.

Features:
    - Complete drawdown period identification and analysis
    - Top-N worst drawdown scenario extraction
    - Underwater (drawdown) equity curve generation
    - Recovery time statistics and distribution analysis
    - Rolling maximum drawdown computation
    - Pain Index, Ulcer Index, and Calmar Ratio calculation

Usage:
    from tradix.analytics.drawdownSimulator import DrawdownSimulator

    simulator = DrawdownSimulator(result)
    analysis = simulator.analyze()
    top5 = simulator.topDrawdowns(5)
    print(simulator.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade

TRADING_DAYS_PER_YEAR = 252


@dataclass
class DrawdownScenario:
    """
    Single drawdown scenario record with magnitude, timing, and pain metrics.

    개별 낙폭 시나리오의 크기, 시점, 고통 지수를 기록하는 데이터 클래스입니다.

    Attributes:
        name (str): Descriptive name for the drawdown scenario.
            낙폭 시나리오 이름.
        drawdownPercent (float): Maximum drawdown magnitude in percent (negative).
            최대 낙폭 크기 (%, 음수).
        duration (int): Number of trading days from peak to trough.
            고점에서 저점까지 거래일 수.
        recoveryDays (int): Number of trading days from trough to full recovery.
            저점에서 완전 회복까지 거래일 수 (미회복 시 -1).
        equityAtBottom (float): Portfolio equity value at the drawdown trough.
            낙폭 저점에서의 포트폴리오 자산 가치.
        maxPain (float): Cumulative pain index during the drawdown period.
            낙폭 기간 중 누적 고통 지수.

    Example:
        >>> scenario = simulator.topDrawdowns(1)[0]
        >>> print(scenario.summary())
    """
    name: str
    drawdownPercent: float
    duration: int
    recoveryDays: int
    equityAtBottom: float
    maxPain: float

    def summary(self) -> str:
        """
        Generate a human-readable summary of the drawdown scenario.

        낙폭 시나리오의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line formatted summary with drawdown details.
        """
        recoveryStr = (
            f"{self.recoveryDays}일" if self.recoveryDays >= 0 else "미회복"
        )
        return (
            f"{'─'*40}\n"
            f"{self.name}\n"
            f"{'─'*40}\n"
            f"최대 낙폭: {self.drawdownPercent:.2f}%\n"
            f"하락 기간: {self.duration}일\n"
            f"회복 기간: {recoveryStr}\n"
            f"저점 자산: {self.equityAtBottom:,.0f}\n"
            f"고통 지수: {self.maxPain:.4f}\n"
            f"{'─'*40}"
        )


class DrawdownSimulator:
    """
    Drawdown scenario simulator and analyzer for backtest equity curves.

    Identifies all drawdown periods from peak-to-trough-to-recovery, computes
    comprehensive statistics, and provides risk indicators such as Pain Index,
    Ulcer Index, and Calmar Ratio.

    백테스트 자산 곡선에 대한 낙폭 시나리오 시뮬레이터 및 분석기입니다.
    고점-저점-회복의 모든 낙폭 기간을 식별하고, 종합 통계를 계산하며,
    고통 지수, 궤양 지수, 칼마 비율 등의 위험 지표를 제공합니다.

    Attributes:
        result (BacktestResult): Original backtest result to analyze.
            분석 대상 원본 백테스트 결과.

    Example:
        >>> simulator = DrawdownSimulator(result)
        >>> analysis = simulator.analyze()
        >>> print(f"Max DD: {analysis['maxDrawdown']:.2f}%")
        >>> print(simulator.summary())
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize the Drawdown Simulator with a backtest result.

        백테스트 결과로 낙폭 시뮬레이터를 초기화합니다.

        Args:
            result (BacktestResult): Completed backtest result containing an
                equity curve with DatetimeIndex. 자산 곡선이 포함된 백테스트 결과.
        """
        self._result = result
        self._equityCurve = result.equityCurve.copy()
        self._drawdownSeries = self._computeDrawdownSeries()
        self._drawdownPeriods = self._identifyDrawdownPeriods()

    def _computeDrawdownSeries(self) -> pd.Series:
        """
        Compute the drawdown series from the equity curve.

        자산 곡선으로부터 낙폭 시리즈를 계산합니다.

        Returns:
            pd.Series: Drawdown values as negative percentages (0 = at peak).
        """
        if len(self._equityCurve) < 2:
            return pd.Series(dtype=float)

        cumMax = self._equityCurve.cummax()
        drawdown = (self._equityCurve - cumMax) / cumMax * 100
        return drawdown

    def _identifyDrawdownPeriods(self) -> List[Dict[str, Any]]:
        """
        Identify all distinct drawdown periods from the equity curve.

        자산 곡선에서 모든 개별 낙폭 기간을 식별합니다.

        A drawdown period starts when equity drops below the running maximum
        and ends when equity recovers to a new high (or at the end of data).

        Returns:
            List[dict]: Each dict contains 'peakIdx', 'troughIdx', 'recoveryIdx',
                'peakDate', 'troughDate', 'recoveryDate', 'peakEquity',
                'troughEquity', 'drawdownPercent', 'duration', 'recoveryDays'.
        """
        if len(self._equityCurve) < 2:
            return []

        equity = self._equityCurve.values
        cumMax = np.maximum.accumulate(equity)
        drawdownValues = (equity - cumMax) / cumMax

        periods = []
        inDrawdown = False
        peakIdx = 0

        for i in range(len(equity)):
            if drawdownValues[i] < 0 and not inDrawdown:
                inDrawdown = True
                peakIdx = i - 1 if i > 0 else 0
                troughIdx = i
                troughValue = drawdownValues[i]

            elif inDrawdown and drawdownValues[i] < troughValue:
                troughIdx = i
                troughValue = drawdownValues[i]

            elif inDrawdown and drawdownValues[i] >= 0:
                inDrawdown = False
                periods.append(self._buildPeriod(
                    peakIdx, troughIdx, recoveryIdx=i
                ))

        if inDrawdown:
            periods.append(self._buildPeriod(
                peakIdx, troughIdx, recoveryIdx=-1
            ))

        return periods

    def _buildPeriod(
        self, peakIdx: int, troughIdx: int, recoveryIdx: int
    ) -> Dict[str, Any]:
        """
        Build a drawdown period dictionary from index positions.

        인덱스 위치로부터 낙폭 기간 딕셔너리를 생성합니다.

        Args:
            peakIdx (int): Index of the pre-drawdown peak. 고점 인덱스.
            troughIdx (int): Index of the drawdown trough. 저점 인덱스.
            recoveryIdx (int): Index of full recovery, or -1 if not recovered.
                회복 인덱스 (미회복 시 -1).

        Returns:
            dict: Drawdown period details.
        """
        dates = self._equityCurve.index
        equityValues = self._equityCurve.values

        peakEquity = float(equityValues[peakIdx])
        troughEquity = float(equityValues[troughIdx])
        drawdownPercent = ((troughEquity - peakEquity) / peakEquity) * 100

        duration = troughIdx - peakIdx
        recoveryDays = (
            (recoveryIdx - troughIdx) if recoveryIdx >= 0
            else -1
        )

        startIdx = peakIdx
        endIdx = recoveryIdx if recoveryIdx >= 0 else len(self._equityCurve) - 1
        ddSlice = self._drawdownSeries.iloc[startIdx:endIdx + 1]
        painDuringPeriod = float(np.mean(np.abs(ddSlice.values))) if len(ddSlice) > 0 else 0.0

        return {
            'peakIdx': peakIdx,
            'troughIdx': troughIdx,
            'recoveryIdx': recoveryIdx,
            'peakDate': dates[peakIdx],
            'troughDate': dates[troughIdx],
            'recoveryDate': dates[recoveryIdx] if recoveryIdx >= 0 else None,
            'peakEquity': peakEquity,
            'troughEquity': troughEquity,
            'drawdownPercent': drawdownPercent,
            'duration': duration,
            'recoveryDays': recoveryDays,
            'painDuringPeriod': painDuringPeriod,
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete drawdown analysis on the equity curve.

        자산 곡선에 대한 종합 낙폭 분석을 수행합니다.

        Computes maximum drawdown, average drawdown, drawdown frequency,
        and recovery statistics across all identified drawdown periods.

        Returns:
            dict: Comprehensive drawdown analysis containing:
                - maxDrawdown (float): Maximum drawdown percentage
                - avgDrawdown (float): Average drawdown across all periods
                - drawdownFrequency (int): Total number of drawdown periods
                - avgDuration (float): Average drawdown duration in days
                - avgRecoveryDays (float): Average recovery time in days
                - longestDrawdown (int): Longest single drawdown duration
                - longestRecovery (int): Longest single recovery period
                - currentDrawdown (float): Current drawdown from peak
                - painIndex (float): Martin's Pain Index
                - ulcerIndex (float): Ulcer Index
                - calmarRatio (float): Calmar Ratio

        Example:
            >>> analysis = simulator.analyze()
            >>> print(f"Max Drawdown: {analysis['maxDrawdown']:.2f}%")
            >>> print(f"Avg Recovery: {analysis['avgRecoveryDays']:.0f} days")
        """
        if not self._drawdownPeriods:
            return {
                'maxDrawdown': 0.0,
                'avgDrawdown': 0.0,
                'drawdownFrequency': 0,
                'avgDuration': 0.0,
                'avgRecoveryDays': 0.0,
                'longestDrawdown': 0,
                'longestRecovery': 0,
                'currentDrawdown': 0.0,
                'painIndex': 0.0,
                'ulcerIndex': 0.0,
                'calmarRatio': 0.0,
            }

        drawdowns = [p['drawdownPercent'] for p in self._drawdownPeriods]
        durations = [p['duration'] for p in self._drawdownPeriods]
        recoveries = [
            p['recoveryDays'] for p in self._drawdownPeriods
            if p['recoveryDays'] >= 0
        ]

        maxDrawdown = float(np.min(drawdowns))
        avgDrawdown = float(np.mean(drawdowns))

        longestDrawdown = int(np.max(durations)) if durations else 0
        longestRecovery = int(np.max(recoveries)) if recoveries else 0
        avgDuration = float(np.mean(durations)) if durations else 0.0
        avgRecoveryDays = float(np.mean(recoveries)) if recoveries else 0.0

        currentDrawdown = float(self._drawdownSeries.iloc[-1]) if len(self._drawdownSeries) > 0 else 0.0

        return {
            'maxDrawdown': maxDrawdown,
            'avgDrawdown': avgDrawdown,
            'drawdownFrequency': len(self._drawdownPeriods),
            'avgDuration': avgDuration,
            'avgRecoveryDays': avgRecoveryDays,
            'longestDrawdown': longestDrawdown,
            'longestRecovery': longestRecovery,
            'currentDrawdown': currentDrawdown,
            'painIndex': self.painIndex(),
            'ulcerIndex': self.ulcerIndex(),
            'calmarRatio': self.calmarRatio(),
        }

    def topDrawdowns(self, n: int = 5) -> List[DrawdownScenario]:
        """
        Return the top N worst drawdown scenarios sorted by magnitude.

        크기 기준 상위 N개 최악의 낙폭 시나리오를 반환합니다.

        Args:
            n (int): Number of worst drawdowns to return (default: 5).
                반환할 최악 낙폭 수 (기본값: 5).

        Returns:
            List[DrawdownScenario]: Top N drawdown scenarios sorted from worst
                to least severe.

        Example:
            >>> top3 = simulator.topDrawdowns(3)
            >>> for dd in top3:
            ...     print(dd.summary())
        """
        sortedPeriods = sorted(
            self._drawdownPeriods, key=lambda p: p['drawdownPercent']
        )

        scenarios = []
        for i, period in enumerate(sortedPeriods[:n]):
            peakDateStr = period['peakDate'].strftime('%Y-%m-%d') if hasattr(period['peakDate'], 'strftime') else str(period['peakDate'])
            troughDateStr = period['troughDate'].strftime('%Y-%m-%d') if hasattr(period['troughDate'], 'strftime') else str(period['troughDate'])

            scenario = DrawdownScenario(
                name=f"#{i+1} 낙폭 ({peakDateStr} ~ {troughDateStr})",
                drawdownPercent=period['drawdownPercent'],
                duration=period['duration'],
                recoveryDays=period['recoveryDays'],
                equityAtBottom=period['troughEquity'],
                maxPain=period['painDuringPeriod'],
            )
            scenarios.append(scenario)

        return scenarios

    def underwaterChart(self) -> pd.Series:
        """
        Generate the underwater (drawdown) equity curve series.

        수중(낙폭) 자산 곡선 시리즈를 생성합니다.

        The underwater chart shows the percentage decline from the running
        maximum at each point in time. Values are zero at peaks and negative
        during drawdowns.

        Returns:
            pd.Series: Drawdown percentages with DatetimeIndex. Zero at peaks,
                negative during drawdowns.

        Example:
            >>> underwater = simulator.underwaterChart()
            >>> underwater.plot(title="Underwater Chart")
        """
        return self._drawdownSeries.copy()

    def recoveryAnalysis(self) -> Dict[str, Any]:
        """
        Compute detailed recovery time statistics across all drawdown periods.

        모든 낙폭 기간에 대한 상세 회복 시간 통계를 계산합니다.

        Returns:
            dict: Recovery statistics containing:
                - avgRecovery (float): Mean recovery time in days
                - medianRecovery (float): Median recovery time in days
                - minRecovery (int): Fastest recovery in days
                - maxRecovery (int): Slowest recovery in days
                - stdRecovery (float): Standard deviation of recovery times
                - totalPeriods (int): Total drawdown periods analyzed
                - recoveredPeriods (int): Periods that fully recovered
                - unrecoveredPeriods (int): Periods still in drawdown
                - recoveryRate (float): Percentage of periods that recovered

        Example:
            >>> recovery = simulator.recoveryAnalysis()
            >>> print(f"Avg recovery: {recovery['avgRecovery']:.0f} days")
        """
        allRecoveries = [
            p['recoveryDays'] for p in self._drawdownPeriods
            if p['recoveryDays'] >= 0
        ]
        unrecovered = [
            p for p in self._drawdownPeriods if p['recoveryDays'] < 0
        ]

        totalPeriods = len(self._drawdownPeriods)
        recoveredPeriods = len(allRecoveries)

        if not allRecoveries:
            return {
                'avgRecovery': 0.0,
                'medianRecovery': 0.0,
                'minRecovery': 0,
                'maxRecovery': 0,
                'stdRecovery': 0.0,
                'totalPeriods': totalPeriods,
                'recoveredPeriods': 0,
                'unrecoveredPeriods': len(unrecovered),
                'recoveryRate': 0.0,
            }

        recoveryArr = np.array(allRecoveries)

        return {
            'avgRecovery': float(np.mean(recoveryArr)),
            'medianRecovery': float(np.median(recoveryArr)),
            'minRecovery': int(np.min(recoveryArr)),
            'maxRecovery': int(np.max(recoveryArr)),
            'stdRecovery': float(np.std(recoveryArr, ddof=1)) if len(recoveryArr) > 1 else 0.0,
            'totalPeriods': totalPeriods,
            'recoveredPeriods': recoveredPeriods,
            'unrecoveredPeriods': len(unrecovered),
            'recoveryRate': (recoveredPeriods / totalPeriods * 100) if totalPeriods > 0 else 0.0,
        }

    def rollingMaxDrawdown(self, window: int = 252) -> pd.Series:
        """
        Compute the rolling maximum drawdown over a sliding window.

        슬라이딩 윈도우에 대한 롤링 최대 낙폭을 계산합니다.

        For each point in time, computes the maximum drawdown observed within
        the trailing window of specified length.

        Args:
            window (int): Rolling window size in trading days (default: 252,
                approximately one year). 롤링 윈도우 크기 (거래일, 기본값: 252).

        Returns:
            pd.Series: Rolling maximum drawdown percentages with DatetimeIndex.
                Values are negative during drawdowns, zero at peaks.

        Example:
            >>> rolling = simulator.rollingMaxDrawdown(window=126)
            >>> rolling.plot(title="Rolling 6-Month Max Drawdown")
        """
        if len(self._equityCurve) < 2:
            return pd.Series(dtype=float)

        equity = self._equityCurve

        def _rollingMdd(windowSeries):
            if len(windowSeries) < 2:
                return 0.0
            cumMax = np.maximum.accumulate(windowSeries.values)
            dd = (windowSeries.values - cumMax) / cumMax
            return float(np.min(dd)) * 100

        rollingMdd = equity.rolling(window=window, min_periods=2).apply(
            _rollingMdd, raw=False
        )

        return rollingMdd

    def drawdownDistribution(self) -> Dict[str, Any]:
        """
        Compute the statistical distribution of drawdown magnitudes.

        낙폭 크기의 통계적 분포를 계산합니다.

        Returns:
            dict: Distribution statistics containing:
                - mean (float): Mean drawdown percentage
                - median (float): Median drawdown percentage
                - std (float): Standard deviation of drawdowns
                - min (float): Deepest drawdown (most negative)
                - max (float): Shallowest drawdown (least negative)
                - percentile25 (float): 25th percentile drawdown
                - percentile75 (float): 75th percentile drawdown
                - count (int): Total number of drawdown periods
                - bins (dict): Histogram bins mapping range labels to counts

        Example:
            >>> dist = simulator.drawdownDistribution()
            >>> print(f"Median drawdown: {dist['median']:.2f}%")
        """
        if not self._drawdownPeriods:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentile25': 0.0,
                'percentile75': 0.0,
                'count': 0,
                'bins': {},
            }

        drawdowns = np.array([p['drawdownPercent'] for p in self._drawdownPeriods])

        binEdges = [0, -2, -5, -10, -20, -50, -100]
        binEdges.sort()
        binLabels = []
        binCounts = {}

        for i in range(len(binEdges) - 1):
            label = f"{binEdges[i]}% ~ {binEdges[i+1]}%"
            count = int(np.sum(
                (drawdowns >= binEdges[i]) & (drawdowns < binEdges[i+1])
            ))
            binLabels.append(label)
            binCounts[label] = count

        return {
            'mean': float(np.mean(drawdowns)),
            'median': float(np.median(drawdowns)),
            'std': float(np.std(drawdowns, ddof=1)) if len(drawdowns) > 1 else 0.0,
            'min': float(np.min(drawdowns)),
            'max': float(np.max(drawdowns)),
            'percentile25': float(np.percentile(drawdowns, 25)),
            'percentile75': float(np.percentile(drawdowns, 75)),
            'count': len(drawdowns),
            'bins': binCounts,
        }

    def painIndex(self) -> float:
        """
        Calculate Martin's Pain Index.

        마틴의 고통 지수(Pain Index)를 계산합니다.

        The Pain Index is the mean of absolute drawdown values over the entire
        observation period. Lower values indicate less painful drawdown
        experiences.

        Pain Index = mean(|drawdown_t|) for all t

        Returns:
            float: Pain Index value (positive, in percent). Higher values
                indicate more painful drawdown experiences.

        Example:
            >>> pain = simulator.painIndex()
            >>> print(f"Pain Index: {pain:.4f}%")
        """
        if len(self._drawdownSeries) == 0:
            return 0.0

        return float(np.mean(np.abs(self._drawdownSeries.values)))

    def ulcerIndex(self) -> float:
        """
        Calculate the Ulcer Index.

        궤양 지수(Ulcer Index)를 계산합니다.

        The Ulcer Index is the root-mean-square of drawdown values, giving
        more weight to larger drawdowns than the Pain Index. It measures the
        depth and duration of drawdowns.

        Ulcer Index = sqrt(mean(drawdown_t^2)) for all t

        Returns:
            float: Ulcer Index value (positive, in percent). Higher values
                indicate deeper and longer drawdowns.

        Example:
            >>> ulcer = simulator.ulcerIndex()
            >>> print(f"Ulcer Index: {ulcer:.4f}%")
        """
        if len(self._drawdownSeries) == 0:
            return 0.0

        ddValues = self._drawdownSeries.values
        return float(np.sqrt(np.mean(ddValues ** 2)))

    def calmarRatio(self) -> float:
        """
        Calculate the Calmar Ratio.

        칼마 비율(Calmar Ratio)을 계산합니다.

        Calmar Ratio = Annualized Return / |Maximum Drawdown|

        A higher Calmar Ratio indicates better risk-adjusted performance
        relative to the worst-case drawdown.

        Returns:
            float: Calmar Ratio. Returns 0.0 if maximum drawdown is zero.

        Example:
            >>> calmar = simulator.calmarRatio()
            >>> print(f"Calmar Ratio: {calmar:.2f}")
        """
        if len(self._equityCurve) < 2:
            return 0.0

        totalReturn = (
            self._equityCurve.iloc[-1] / self._equityCurve.iloc[0]
        ) - 1
        tradingDays = len(self._equityCurve)
        years = tradingDays / TRADING_DAYS_PER_YEAR

        if years <= 0:
            return 0.0

        annualReturn = (1 + totalReturn) ** (1 / years) - 1

        maxDd = float(np.min(self._drawdownSeries.values)) if len(self._drawdownSeries) > 0 else 0.0

        if maxDd == 0.0:
            return 0.0

        return (annualReturn * 100) / abs(maxDd)

    def summary(self) -> str:
        """
        Generate a comprehensive summary of drawdown analysis.

        낙폭 분석의 종합 요약을 생성합니다.

        Returns:
            str: Multi-line formatted summary including max drawdown,
                frequency, recovery stats, and risk indicators.
        """
        analysis = self.analyze()
        recovery = self.recoveryAnalysis()

        lines = [
            f"{'='*55}",
            f"낙폭 분석 요약 (Drawdown Analysis Summary)",
            f"{'='*55}",
            f"최대 낙폭 (Max Drawdown): {analysis['maxDrawdown']:.2f}%",
            f"평균 낙폭 (Avg Drawdown): {analysis['avgDrawdown']:.2f}%",
            f"현재 낙폭 (Current DD): {analysis['currentDrawdown']:.2f}%",
            f"{'─'*55}",
            f"낙폭 횟수: {analysis['drawdownFrequency']}회",
            f"최장 하락 기간: {analysis['longestDrawdown']}일",
            f"최장 회복 기간: {analysis['longestRecovery']}일",
            f"평균 하락 기간: {analysis['avgDuration']:.0f}일",
            f"평균 회복 기간: {analysis['avgRecoveryDays']:.0f}일",
            f"{'─'*55}",
            f"회복률: {recovery['recoveryRate']:.1f}% "
            f"({recovery['recoveredPeriods']}/{recovery['totalPeriods']})",
            f"{'─'*55}",
            f"Pain Index: {analysis['painIndex']:.4f}%",
            f"Ulcer Index: {analysis['ulcerIndex']:.4f}%",
            f"Calmar Ratio: {analysis['calmarRatio']:.2f}",
            f"{'='*55}",
        ]

        topDd = self.topDrawdowns(3)
        if topDd:
            lines.append(f"\n상위 {len(topDd)}개 최악 낙폭:")
            for dd in topDd:
                lines.append(
                    f"  {dd.name}: {dd.drawdownPercent:.2f}% "
                    f"(하락 {dd.duration}일, "
                    f"회복 {'미회복' if dd.recoveryDays < 0 else f'{dd.recoveryDays}일'})"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        maxDd = float(np.min(self._drawdownSeries.values)) if len(self._drawdownSeries) > 0 else 0.0
        return (
            f"DrawdownSimulator(strategy={self._result.strategy}, "
            f"maxDD={maxDd:.2f}%, "
            f"periods={len(self._drawdownPeriods)})"
        )
