"""
Tradex Walk-Forward Analyzer Module.

Implements walk-forward analysis for robust strategy validation. Splits
historical data into sequential in-sample (optimization) and out-of-sample
(validation) periods, optimizing parameters on each IS window and evaluating
on the subsequent OOS window.

전략의 견고성 검증을 위한 워크포워드 분석 모듈입니다. 과거 데이터를 순차적인
인샘플(최적화) 및 아웃오브샘플(검증) 구간으로 분할하고, 각 IS 윈도우에서
파라미터를 최적화한 후 후속 OOS 윈도우에서 평가합니다.

Features:
    - Rolling and anchored window walk-forward analysis
    - Automatic in-sample optimization via grid/random search
    - Out-of-sample validation with robustness ratio computation
    - Parameter stability analysis across folds
    - Combined OOS equity curve reconstruction
    - Overfitting detection and interpretation

Usage:
    from tradex.walkforward import WalkForwardAnalyzer

    wfa = WalkForwardAnalyzer(
        data=data_feed,
        strategyFactory=create_strategy,
        parameterSpace=space,
        metric='sharpeRatio',
        inSampleMonths=12,
        outOfSampleMonths=3,
    )
    result = wfa.run()
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional
import time
import numpy as np
import pandas as pd

from tradex.walkforward.splitter import PeriodSplitter
from tradex.optimize.space import ParameterSpace
from tradex.optimize.optimizer import Optimizer
from tradex.datafeed.feed import DataFeed
from tradex.datafeed.fdr import FinanceDataReaderFeed
from tradex.strategy.base import Strategy
from tradex.engine import BacktestEngine, BacktestResult


@dataclass
class FoldResult:
    """
    Result container for a single walk-forward fold.

    단일 워크포워드 폴드의 결과를 저장하는 데이터 클래스입니다.

    Attributes:
        foldNumber (int): Sequential fold index starting from 1 (폴드 번호).
        isRange (Tuple[str, str]): In-sample date range (start, end)
            (인샘플 기간).
        oosRange (Tuple[str, str]): Out-of-sample date range (start, end)
            (아웃오브샘플 기간).
        bestParams (Dict[str, Any]): Best parameters found during IS
            optimization (IS 최적 파라미터).
        isMetric (float): Best metric value achieved in-sample (IS 지표 값).
        oosMetric (float): Metric value achieved out-of-sample (OOS 지표 값).
        oosResult (Optional[BacktestResult]): Full OOS backtest result
            (OOS 백테스트 결과).
        oosEquity (pd.Series): OOS equity curve (OOS 자산 곡선).
    """
    foldNumber: int
    isRange: Tuple[str, str]
    oosRange: Tuple[str, str]
    bestParams: Dict[str, Any]
    isMetric: float
    oosMetric: float
    oosResult: Optional[BacktestResult] = None
    oosEquity: pd.Series = field(default_factory=pd.Series)

    def __repr__(self) -> str:
        return (
            f"FoldResult({self.foldNumber}, "
            f"IS={self.isMetric:.4f}, OOS={self.oosMetric:.4f})"
        )


@dataclass
class WalkForwardResult:
    """
    Aggregate result container for the entire walk-forward analysis.

    Computes summary statistics across all folds including robustness ratio,
    OOS positive rate, and parameter stability metrics.

    전체 워크포워드 분석의 집계 결과 컨테이너입니다. 모든 폴드에 대한 견고성 비율,
    OOS 양수 비율, 파라미터 안정성 지표를 계산합니다.

    Attributes:
        folds (List[FoldResult]): Per-fold results (폴드별 결과).
        metric (str): Optimization metric name (최적화 지표).
        elapsedTime (float): Total analysis time in seconds (소요 시간, 초).
        method (str): Split method used - 'rolling' or 'anchored' (분할 방법).

    Example:
        >>> result = wfa.run()
        >>> print(f"Robustness: {result.robustnessRatio:.2%}")
        >>> print(result.summary())
    """
    folds: List[FoldResult]
    metric: str
    elapsedTime: float
    method: str = 'rolling'

    @property
    def avgIsMetric(self) -> float:
        """
        Calculate average in-sample metric value across all folds.

        전체 폴드의 인샘플 평균 지표 값을 계산합니다.
        """
        values = [f.isMetric for f in self.folds if not np.isnan(f.isMetric)]
        return np.mean(values) if values else 0

    @property
    def avgOosMetric(self) -> float:
        """
        Calculate average out-of-sample metric value across all folds.

        전체 폴드의 아웃오브샘플 평균 지표 값을 계산합니다.
        """
        values = [f.oosMetric for f in self.folds if not np.isnan(f.oosMetric)]
        return np.mean(values) if values else 0

    @property
    def robustnessRatio(self) -> float:
        """
        Calculate the robustness ratio (OOS average / IS average).

        A value close to 1.0 indicates low overfitting risk; below 0.5
        suggests significant overfitting.

        견고성 비율(OOS 평균 / IS 평균)을 계산합니다.
        1.0에 가까울수록 과적합 위험이 낮으며, 0.5 미만이면 심각한 과적합을 의미합니다.
        """
        if self.avgIsMetric == 0:
            return 0
        return self.avgOosMetric / self.avgIsMetric

    @property
    def oosPositiveRate(self) -> float:
        """
        Calculate the percentage of folds with positive OOS metric values.

        OOS 지표가 양수인 폴드의 비율(%)을 계산합니다.
        """
        positive = sum(1 for f in self.folds if f.oosMetric > 0)
        return positive / len(self.folds) * 100 if self.folds else 0

    @property
    def oosWinRate(self) -> float:
        """
        Calculate the percentage of folds where OOS metric exceeds IS metric.

        OOS 지표가 IS 지표를 초과하는 폴드의 비율(%)을 계산합니다.
        """
        wins = sum(1 for f in self.folds if f.oosMetric >= f.isMetric)
        return wins / len(self.folds) * 100 if self.folds else 0

    @property
    def parameterStability(self) -> Dict[str, float]:
        """
        Measure parameter stability across folds as standard deviation.

        Lower values indicate more stable (consistent) parameter selections
        across different time windows.

        폴드 간 파라미터 안정성을 표준편차로 측정합니다. 값이 낮을수록 안정적입니다.

        Returns:
            Dict[str, float]: Parameter name to stability score mapping.
        """
        if not self.folds:
            return {}

        paramNames = list(self.folds[0].bestParams.keys())
        stability = {}

        for name in paramNames:
            values = [f.bestParams.get(name, 0) for f in self.folds]
            if all(isinstance(v, (int, float)) for v in values):
                stability[name] = np.std(values)
            else:
                uniqueCount = len(set(str(v) for v in values))
                stability[name] = uniqueCount / len(values)

        return stability

    def summary(self) -> str:
        """
        Generate a formatted summary of the walk-forward analysis results.

        워크포워드 분석 결과의 포맷된 요약을 생성합니다.

        Returns:
            str: Multi-line summary with IS/OOS metrics, robustness ratio,
                overfitting interpretation, and elapsed time.
        """
        interpretation = self._interpretation()

        lines = [
            f"\n{'='*60}",
            f"Walk-Forward 분석 결과 ({self.method})",
            f"{'='*60}",
            f"총 폴드: {len(self.folds)}개",
            f"최적화 지표: {self.metric}",
            f"{'─'*60}",
            f"IS 평균 {self.metric}: {self.avgIsMetric:.4f}",
            f"OOS 평균 {self.metric}: {self.avgOosMetric:.4f}",
            f"{'─'*60}",
            f"견고성 비율 (OOS/IS): {self.robustnessRatio:.2%}",
            f"OOS 양수 비율: {self.oosPositiveRate:.1f}%",
            f"{'─'*60}",
            f"해석:",
            f"  {interpretation}",
            f"{'─'*60}",
            f"소요 시간: {self.elapsedTime:.1f}초",
            f"{'='*60}",
        ]

        return "\n".join(lines)

    def _interpretation(self) -> str:
        """
        Interpret the robustness ratio as an overfitting risk assessment.

        견고성 비율을 과적합 위험도로 해석합니다.

        Returns:
            str: Human-readable interpretation of overfitting risk level.
        """
        ratio = self.robustnessRatio

        if ratio >= 0.8:
            return "✓ 우수: 과적합 위험 낮음, 전략 안정적"
        elif ratio >= 0.5:
            return "△ 보통: 약간의 과적합 가능성, 주의 필요"
        elif ratio >= 0.3:
            return "▲ 주의: 과적합 의심, 전략 재검토 권장"
        else:
            return "✗ 위험: 심각한 과적합, 전략 변경 필요"

    def toDataFrame(self) -> pd.DataFrame:
        """
        Convert per-fold results to a DataFrame for analysis.

        폴드별 결과를 분석용 DataFrame으로 변환합니다.

        Returns:
            pd.DataFrame: DataFrame with fold number, IS/OOS periods, metrics,
                and best parameters per fold.
        """
        rows = []
        for f in self.folds:
            row = {
                'Fold': f.foldNumber,
                'IS Period': f"{f.isRange[0]} ~ {f.isRange[1]}",
                'OOS Period': f"{f.oosRange[0]} ~ {f.oosRange[1]}",
                f'IS {self.metric}': f.isMetric,
                f'OOS {self.metric}': f.oosMetric,
            }
            row.update(f.bestParams)
            rows.append(row)

        return pd.DataFrame(rows)

    def getOosCombinedEquity(self) -> pd.Series:
        """
        Concatenate all OOS equity curves into a single continuous series.

        모든 OOS 구간의 자산 곡선을 하나의 연속 시리즈로 연결합니다.

        Returns:
            pd.Series: Combined OOS equity curve sorted by date, or empty
                Series if no OOS equity data exists.
        """
        equities = []

        for f in self.folds:
            if f.oosEquity is not None and len(f.oosEquity) > 0:
                equities.append(f.oosEquity)

        if not equities:
            return pd.Series()

        combined = pd.concat(equities)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()

    def getBestParams(self) -> Dict[str, Any]:
        """
        Retrieve the most frequently selected parameter combination.

        가장 많이 선택된 파라미터 조합을 반환합니다.

        Returns:
            Dict[str, Any]: The most common best parameter combination across
                folds, or empty dict if no folds exist.
        """
        if not self.folds:
            return {}

        paramCounts = {}
        for f in self.folds:
            key = str(sorted(f.bestParams.items()))
            paramCounts[key] = paramCounts.get(key, 0) + 1

        mostCommon = max(paramCounts.items(), key=lambda x: x[1])

        return self.folds[0].bestParams

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult({len(self.folds)} folds, "
            f"robustness={self.robustnessRatio:.2%})"
        )


class WalkForwardAnalyzer:
    """
    Walk-forward analysis engine for overfitting prevention.

    Performs rolling or anchored window walk-forward optimization by
    splitting data into sequential in-sample and out-of-sample periods,
    optimizing parameters on IS, and validating on OOS. Produces
    robustness metrics to assess strategy generalizability.

    과적합 방지를 위한 워크포워드 분석 엔진입니다. 데이터를 순차적인 인샘플/
    아웃오브샘플 구간으로 분할하고, IS에서 파라미터를 최적화한 후 OOS에서
    검증합니다. 전략의 일반화 가능성을 평가하는 견고성 지표를 생성합니다.

    Attributes:
        data (DataFeed): Source data feed (데이터 피드).
        strategyFactory (Callable): Strategy creation function (전략 생성 함수).
        space (ParameterSpace): Parameter search space (파라미터 공간).
        metric (str): Optimization target metric (최적화 지표).
        inSampleMonths (int): In-sample window length in months (IS 기간).
        outOfSampleMonths (int): Out-of-sample window length in months (OOS 기간).
        stepMonths (int): Window step size in months (윈도우 이동 스텝).
        splitMethod (str): Split method - 'rolling' or 'anchored' (분할 방법).
        optimizeMethod (str): Optimization method - 'grid' or 'random'
            (최적화 방법).

    Example:
        >>> wfa = WalkForwardAnalyzer(
        ...     data=data_feed,
        ...     strategyFactory=create_strategy,
        ...     parameterSpace=space,
        ...     metric='sharpeRatio',
        ...     inSampleMonths=12,
        ...     outOfSampleMonths=3,
        ... )
        >>> result = wfa.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        data: DataFeed,
        strategyFactory: Callable[[Dict[str, Any]], Strategy],
        parameterSpace: ParameterSpace,
        metric: str = 'sharpeRatio',
        inSampleMonths: int = 12,
        outOfSampleMonths: int = 3,
        stepMonths: int = 3,
        splitMethod: str = 'rolling',
        optimizeMethod: str = 'grid',
        nRandomTrials: int = 50,
        initialCash: float = 10_000_000,
        nJobs: int = 1,
    ):
        """
        Initialize the WalkForwardAnalyzer.

        WalkForwardAnalyzer를 초기화합니다.

        Args:
            data: Data feed providing OHLCV data (데이터 피드).
            strategyFactory: Callable that accepts a parameter dict and
                returns a Strategy instance (전략 생성 함수).
            parameterSpace: Defined parameter search space (파라미터 공간).
            metric: Target performance metric for optimization (최적화 지표).
            inSampleMonths: In-sample window length in months (IS 기간, 개월).
            outOfSampleMonths: Out-of-sample window length in months
                (OOS 기간, 개월).
            stepMonths: Window advancement step in months (윈도우 이동 스텝, 개월).
            splitMethod: Period splitting strategy - 'rolling' or 'anchored'
                (분할 방법).
            optimizeMethod: Optimization algorithm - 'grid' or 'random'
                (최적화 방법).
            nRandomTrials: Number of trials for random search (랜덤 서치 시도 횟수).
            initialCash: Starting portfolio capital (초기 자본).
            nJobs: Number of parallel worker processes (병렬 프로세스 수).
        """
        self.data = data
        self.strategyFactory = strategyFactory
        self.space = parameterSpace
        self.metric = metric
        self.inSampleMonths = inSampleMonths
        self.outOfSampleMonths = outOfSampleMonths
        self.stepMonths = stepMonths
        self.splitMethod = splitMethod
        self.optimizeMethod = optimizeMethod
        self.nRandomTrials = nRandomTrials
        self.initialCash = initialCash
        self.nJobs = nJobs

        self._dataConfig = {
            'symbol': data.symbol,
            'market': getattr(data, 'market', 'KRX'),
            'timeframe': getattr(data, 'timeframe', 'D'),
        }

    def run(self, verbose: bool = True) -> WalkForwardResult:
        """
        Execute the walk-forward analysis pipeline.

        워크포워드 분석 파이프라인을 실행합니다.

        Args:
            verbose: If True, print progress updates for each fold
                (진행 상황 출력 여부).

        Returns:
            WalkForwardResult: Complete analysis results with per-fold metrics,
                robustness ratio, and parameter stability.

        Raises:
            ValueError: If the data period is too short for the configured
                IS + OOS window sizes.
        """
        if self.splitMethod == 'rolling':
            folds = PeriodSplitter.rolling(
                self.data.startDate,
                self.data.endDate,
                self.inSampleMonths,
                self.outOfSampleMonths,
                self.stepMonths,
            )
        else:
            folds = PeriodSplitter.anchored(
                self.data.startDate,
                self.data.endDate,
                self.outOfSampleMonths,
                self.inSampleMonths,
            )

        if not folds:
            raise ValueError(
                f"기간이 너무 짧습니다. "
                f"IS {self.inSampleMonths}개월 + OOS {self.outOfSampleMonths}개월 필요"
            )

        if verbose:
            print(f"[WalkForward] {len(folds)} 폴드 분석 시작")
            print(f"[WalkForward] 방법: {self.splitMethod}, 최적화: {self.optimizeMethod}")
            print(f"[WalkForward] 지표: {self.metric}")
            print(PeriodSplitter.visualize(folds))
            print()

        startTime = time.time()
        foldResults = []

        for i, ((isStart, isEnd), (oosStart, oosEnd)) in enumerate(folds):
            if verbose:
                print(f"\n[Fold {i+1}/{len(folds)}]")
                print(f"  IS: {isStart} ~ {isEnd}")
                print(f"  OOS: {oosStart} ~ {oosEnd}")

            isData = self._createDataFeed(isStart, isEnd)

            optimizer = Optimizer(
                data=isData,
                strategyFactory=self.strategyFactory,
                parameterSpace=self.space,
                metric=self.metric,
                initialCash=self.initialCash,
                nJobs=self.nJobs,
            )

            if self.optimizeMethod == 'grid':
                optResult = optimizer.gridSearch(verbose=False)
            else:
                optResult = optimizer.randomSearch(
                    nTrials=self.nRandomTrials,
                    verbose=False
                )

            bestParams = optResult.bestParams
            isMetric = optResult.bestMetric

            if verbose:
                print(f"  IS 최적 {self.metric}: {isMetric:.4f}")
                print(f"  최적 파라미터: {bestParams}")

            oosData = self._createDataFeed(oosStart, oosEnd)
            strategy = self.strategyFactory(bestParams)

            engine = BacktestEngine(
                data=oosData,
                strategy=strategy,
                initialCash=self.initialCash,
            )

            oosResult = engine.run(verbose=False)
            oosMetric = oosResult.metrics.get(self.metric, 0)

            if np.isnan(oosMetric) or np.isinf(oosMetric):
                oosMetric = 0

            if verbose:
                print(f"  OOS {self.metric}: {oosMetric:.4f}")
                ratio = oosMetric / isMetric if isMetric != 0 else 0
                print(f"  Robustness: {ratio:.2%}")

            foldResults.append(FoldResult(
                foldNumber=i + 1,
                isRange=(isStart, isEnd),
                oosRange=(oosStart, oosEnd),
                bestParams=bestParams,
                isMetric=isMetric,
                oosMetric=oosMetric,
                oosResult=oosResult,
                oosEquity=oosResult.equityCurve if oosResult else pd.Series(),
            ))

        elapsedTime = time.time() - startTime

        result = WalkForwardResult(
            folds=foldResults,
            metric=self.metric,
            elapsedTime=elapsedTime,
            method=self.splitMethod,
        )

        if verbose:
            print(result.summary())

        return result

    def _createDataFeed(self, startDate: str, endDate: str) -> DataFeed:
        """
        Create a data feed for a specific date range.

        특정 기간의 데이터 피드를 생성합니다.

        Args:
            startDate: Start date string in 'YYYY-MM-DD' format.
            endDate: End date string in 'YYYY-MM-DD' format.

        Returns:
            DataFeed: New data feed instance for the specified period.
        """
        return FinanceDataReaderFeed(
            symbol=self._dataConfig['symbol'],
            startDate=startDate,
            endDate=endDate,
            market=self._dataConfig['market'],
            timeframe=self._dataConfig['timeframe'],
        )

    def previewFolds(self) -> str:
        """
        Generate an ASCII visualization of the planned fold structure.

        계획된 폴드 구조의 ASCII 시각화를 생성합니다.

        Returns:
            str: ASCII art representation of IS and OOS periods per fold.
        """
        if self.splitMethod == 'rolling':
            folds = PeriodSplitter.rolling(
                self.data.startDate,
                self.data.endDate,
                self.inSampleMonths,
                self.outOfSampleMonths,
                self.stepMonths,
            )
        else:
            folds = PeriodSplitter.anchored(
                self.data.startDate,
                self.data.endDate,
                self.outOfSampleMonths,
                self.inSampleMonths,
            )

        return PeriodSplitter.visualize(folds)

    def __repr__(self) -> str:
        return (
            f"WalkForwardAnalyzer("
            f"{self.splitMethod}, "
            f"IS={self.inSampleMonths}m, "
            f"OOS={self.outOfSampleMonths}m)"
        )
