"""
Tradix Strategy Parameter Optimizer Module.

Provides grid search and random search optimization over a defined parameter
space. Runs backtests for each parameter combination and identifies the
optimal configuration based on a target metric. Supports both sequential and
parallel execution via ProcessPoolExecutor.

전략 파라미터에 대한 그리드 서치 및 랜덤 서치 최적화 모듈입니다.
정의된 파라미터 공간에서 각 조합에 대해 백테스트를 실행하고, 목표 지표에 따라
최적 설정을 식별합니다. ProcessPoolExecutor를 통한 순차 및 병렬 실행을 지원합니다.

Features:
    - Grid search over all parameter combinations
    - Random search with configurable trial count
    - Parallel execution support via multiprocessing
    - Automatic best-result backtest re-run
    - Top-N / Bottom-N result analysis
    - Result export to DataFrame

Usage:
    from tradix.optimize import Optimizer, ParameterSpace

    space = ParameterSpace()
    space.addInt('fastPeriod', 5, 20, step=1)
    space.addInt('slowPeriod', 20, 60, step=5)

    optimizer = Optimizer(
        data=data_feed,
        strategyFactory=create_strategy,
        parameterSpace=space,
        metric='sharpeRatio',
    )

    result = optimizer.gridSearch()
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import copy
import pandas as pd
import numpy as np

from tradix.optimize.space import ParameterSpace
from tradix.datafeed.feed import DataFeed
from tradix.strategy.base import Strategy
from tradix.engine import BacktestEngine, BacktestResult


@dataclass
class OptimizeResult:
    """
    Container for strategy parameter optimization results.

    Stores the best parameters, metric values, all tested combinations, and
    provides analysis methods for inspecting optimization outcomes.

    전략 파라미터 최적화 결과를 담는 데이터 클래스입니다. 최적 파라미터, 지표 값,
    전체 테스트 조합을 저장하고 결과 분석 메서드를 제공합니다.

    Attributes:
        bestParams (Dict[str, Any]): Optimal parameter combination (최적 파라미터).
        bestMetric (float): Best metric value achieved (최적 지표 값).
        bestResult (Optional[BacktestResult]): Full backtest result for best
            params (최적 파라미터의 백테스트 결과).
        allResults (List[Tuple[Dict, float]]): All (params, metric) pairs
            tested (모든 테스트 결과).
        elapsedTime (float): Total optimization time in seconds (소요 시간, 초).
        metric (str): Name of the optimization metric (최적화 지표 이름).
        method (str): Optimization method used - 'grid' or 'random'
            (최적화 방법).
        totalTests (int): Total number of parameter combinations tested
            (총 테스트 수).

    Example:
        >>> result = optimizer.gridSearch()
        >>> print(result.summary())
        >>> top_df = result.topN(10)
    """
    bestParams: Dict[str, Any]
    bestMetric: float
    bestResult: Optional[BacktestResult]
    allResults: List[Tuple[Dict[str, Any], float]]
    elapsedTime: float
    metric: str
    method: str
    totalTests: int = 0

    def summary(self) -> str:
        """
        Generate a formatted summary string of optimization results.

        최적화 결과의 포맷된 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line summary with best parameters, metric value,
                test count, and elapsed time.
        """
        return (
            f"\n{'='*50}\n"
            f"최적화 결과 ({self.method})\n"
            f"{'='*50}\n"
            f"최적화 지표: {self.metric}\n"
            f"최적 값: {self.bestMetric:.4f}\n"
            f"{'─'*50}\n"
            f"최적 파라미터:\n"
            + "\n".join(f"  {k}: {v}" for k, v in self.bestParams.items())
            + f"\n{'─'*50}\n"
            f"총 테스트: {self.totalTests}개\n"
            f"소요 시간: {self.elapsedTime:.1f}초\n"
            f"{'='*50}\n"
        )

    def topN(self, n: int = 10) -> pd.DataFrame:
        """
        Retrieve the top N results sorted by metric value descending.

        지표 값 기준 상위 N개 결과를 조회합니다.

        Args:
            n: Number of top results to return (상위 개수).

        Returns:
            pd.DataFrame: DataFrame with parameter columns and metric column,
                sorted by metric descending.
        """
        sortedResults = sorted(
            self.allResults,
            key=lambda x: x[1],
            reverse=True
        )[:n]

        rows = []
        for params, metricVal in sortedResults:
            row = params.copy()
            row[self.metric] = metricVal
            rows.append(row)

        return pd.DataFrame(rows)

    def bottomN(self, n: int = 10) -> pd.DataFrame:
        """
        Retrieve the bottom N results sorted by metric value ascending.

        지표 값 기준 하위 N개 결과를 조회합니다.

        Args:
            n: Number of bottom results to return (하위 개수).

        Returns:
            pd.DataFrame: DataFrame with parameter columns and metric column,
                sorted by metric ascending.
        """
        sortedResults = sorted(
            self.allResults,
            key=lambda x: x[1],
        )[:n]

        rows = []
        for params, metricVal in sortedResults:
            row = params.copy()
            row[self.metric] = metricVal
            rows.append(row)

        return pd.DataFrame(rows)

    def toDataFrame(self) -> pd.DataFrame:
        """
        Convert all optimization results to a sorted DataFrame.

        전체 최적화 결과를 정렬된 DataFrame으로 변환합니다.

        Returns:
            pd.DataFrame: All results with parameter and metric columns,
                sorted by metric descending.
        """
        rows = []
        for params, metricVal in self.allResults:
            row = params.copy()
            row[self.metric] = metricVal
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values(self.metric, ascending=False).reset_index(drop=True)

    def getResultByParams(self, **kwargs) -> Optional[float]:
        """
        Look up the metric value for a specific parameter combination.

        특정 파라미터 조합의 지표 값을 조회합니다.

        Args:
            **kwargs: Parameter name-value pairs to match.

        Returns:
            Optional[float]: Metric value if found, None otherwise.
        """
        for params, metricVal in self.allResults:
            if all(params.get(k) == v for k, v in kwargs.items()):
                return metricVal
        return None

    def statistics(self) -> Dict[str, float]:
        """
        Compute descriptive statistics across all optimization results.

        전체 최적화 결과에 대한 기술 통계를 계산합니다.

        Returns:
            Dict[str, float]: Statistics including mean, std, min, max,
                median, and positiveRate (percentage of positive metrics).
        """
        metrics = [m for _, m in self.allResults]
        return {
            'mean': np.mean(metrics),
            'std': np.std(metrics),
            'min': np.min(metrics),
            'max': np.max(metrics),
            'median': np.median(metrics),
            'positiveRate': sum(1 for m in metrics if m > 0) / len(metrics) * 100,
        }

    def __repr__(self) -> str:
        return (
            f"OptimizeResult({self.method}, "
            f"best_{self.metric}={self.bestMetric:.4f}, "
            f"tests={self.totalTests})"
        )


def _runSingleBacktest(args: Tuple) -> Tuple[Dict[str, Any], float, Optional[Dict]]:
    """
    Execute a single backtest in a worker process for parallel optimization.

    병렬 최적화를 위해 워커 프로세스에서 단일 백테스트를 실행합니다.

    Args:
        args: Tuple of (params, dataConfig, strategyFactory, metric,
            initialCash, brokerConfig).

    Returns:
        Tuple[Dict[str, Any], float, Optional[Dict]]: A tuple of
            (parameters, metric_value, result_dict or None on failure).
    """
    params, dataConfig, strategyFactory, metric, initialCash, brokerConfig = args

    try:
        from tradix.datafeed.fdr import FinanceDataReaderFeed
        from tradix.engine import BacktestEngine, SimpleBroker

        data = FinanceDataReaderFeed(
            symbol=dataConfig['symbol'],
            startDate=dataConfig['startDate'],
            endDate=dataConfig['endDate'],
            market=dataConfig.get('market', 'KRX'),
            timeframe=dataConfig.get('timeframe', 'D'),
        )

        strategy = strategyFactory(params)

        broker = None
        if brokerConfig:
            broker = SimpleBroker(
                commissionRate=brokerConfig.get('commissionRate', 0.00015),
                taxRate=brokerConfig.get('taxRate', 0.0018),
                slippageRate=brokerConfig.get('slippageRate', 0.001),
            )

        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initialCash=initialCash,
            broker=broker,
        )

        result = engine.run(verbose=False)
        metricValue = result.metrics.get(metric, 0)

        return (params, metricValue, result.toDict())

    except Exception as e:
        print(f"[Optimizer] 백테스트 실패: {params} - {e}")
        return (params, float('-inf'), None)


class Optimizer:
    """
    Strategy parameter optimizer with grid search and random search.

    Iterates over a ParameterSpace, running backtests for each parameter
    combination to find the configuration that maximizes (or minimizes) a
    target performance metric. Supports sequential and parallel execution.

    파라미터 공간을 탐색하며 각 조합에 대해 백테스트를 실행하여 목표 성과 지표를
    최대화(또는 최소화)하는 설정을 찾는 최적화기입니다.

    Attributes:
        data (DataFeed): Data feed for backtesting (데이터 피드).
        strategyFactory (Callable): Function that creates a Strategy from
            a parameter dict (전략 생성 팩토리 함수).
        space (ParameterSpace): Parameter search space (파라미터 공간).
        metric (str): Target optimization metric name (최적화 지표).
        minimize (bool): If True, minimize the metric; otherwise maximize
            (True: 최소화, False: 최대화).
        initialCash (float): Initial portfolio capital (초기 자본).
        broker: Broker configuration for commission/tax/slippage (브로커 설정).
        nJobs (int): Number of parallel processes (병렬 프로세스 수).

    Example:
        >>> space = ParameterSpace()
        >>> space.addInt('fastPeriod', 5, 20, step=1)
        >>> space.addInt('slowPeriod', 20, 60, step=5)
        >>> optimizer = Optimizer(
        ...     data=data_feed,
        ...     strategyFactory=create_strategy,
        ...     parameterSpace=space,
        ...     metric='sharpeRatio',
        ... )
        >>> result = optimizer.gridSearch()
        >>> print(result.summary())
    """

    METRICS = [
        'totalReturn',
        'annualReturn',
        'sharpeRatio',
        'sortinoRatio',
        'calmarRatio',
        'maxDrawdown',
        'winRate',
        'profitFactor',
    ]

    def __init__(
        self,
        data: DataFeed,
        strategyFactory: Callable[[Dict[str, Any]], Strategy],
        parameterSpace: ParameterSpace,
        metric: str = 'sharpeRatio',
        minimize: bool = False,
        initialCash: float = 10_000_000,
        broker: 'SimpleBroker' = None,
        nJobs: int = 1,
    ):
        """
        Initialize the Optimizer.

        Optimizer를 초기화합니다.

        Args:
            data: Data feed providing OHLCV data (데이터 피드).
            strategyFactory: Callable that accepts a parameter dict and
                returns a Strategy instance (파라미터를 받아 전략을 생성하는 함수).
            parameterSpace: Defined parameter search space (파라미터 공간).
            metric: Performance metric to optimize, default 'sharpeRatio'
                (최적화할 지표).
            minimize: If True, minimize the metric; if False (default),
                maximize it (True: 최소화, False: 최대화).
            initialCash: Starting portfolio capital (초기 자본).
            broker: Broker with commission/tax/slippage settings (브로커 설정).
            nJobs: Number of parallel processes. 1 for sequential, -1 for
                all CPUs (병렬 프로세스 수, 1=순차, -1=모든 CPU).
        """
        self.data = data
        self.strategyFactory = strategyFactory
        self.space = parameterSpace
        self.metric = metric
        self.minimize = minimize
        self.initialCash = initialCash
        self.broker = broker
        self.nJobs = nJobs if nJobs > 0 else multiprocessing.cpu_count()

        self._dataConfig = {
            'symbol': data.symbol,
            'startDate': data.startDate,
            'endDate': data.endDate,
            'market': getattr(data, 'market', 'KRX'),
            'timeframe': getattr(data, 'timeframe', 'D'),
        }

        self._brokerConfig = None
        if broker:
            self._brokerConfig = {
                'commissionRate': broker.commissionRate,
                'taxRate': broker.taxRate,
                'slippageRate': broker.slippageRate,
            }

    def gridSearch(self, verbose: bool = True) -> OptimizeResult:
        """
        Run exhaustive grid search over all parameter combinations.

        파라미터 공간의 모든 조합을 탐색하는 그리드 서치를 실행합니다.

        Args:
            verbose: If True, print progress updates (진행 상황 출력 여부).

        Returns:
            OptimizeResult: Optimization results with best parameters and
                all tested combinations.
        """
        combinations = self.space.gridCombinations()
        totalTests = len(combinations)

        if verbose:
            print(f"[Optimizer] Grid Search: {totalTests:,} 조합")
            print(f"[Optimizer] 파라미터: {self.space.names}")
            print(f"[Optimizer] 최적화 지표: {self.metric}")

        startTime = time.time()

        if self.nJobs == 1:
            results = self._runSequential(combinations, verbose)
        else:
            results = self._runParallel(combinations, verbose)

        elapsedTime = time.time() - startTime

        return self._buildResult(results, elapsedTime, 'grid', totalTests)

    def randomSearch(
        self,
        nTrials: int = 100,
        seed: int = None,
        verbose: bool = True
    ) -> OptimizeResult:
        """
        Run random search by sampling parameter combinations.

        파라미터 공간에서 랜덤으로 샘플링하여 탐색합니다.

        Args:
            nTrials: Number of random parameter combinations to test
                (시도 횟수).
            seed: Random seed for reproducibility (랜덤 시드).
            verbose: If True, print progress updates (진행 상황 출력 여부).

        Returns:
            OptimizeResult: Optimization results with best parameters and
                all tested combinations.
        """
        samples = self.space.randomSample(nTrials, seed=seed)

        if verbose:
            print(f"[Optimizer] Random Search: {nTrials} 시도")
            print(f"[Optimizer] 파라미터: {self.space.names}")
            print(f"[Optimizer] 최적화 지표: {self.metric}")
            if self.space.totalCombinations > nTrials:
                coverage = nTrials / self.space.totalCombinations * 100
                print(f"[Optimizer] 탐색 범위: {coverage:.1f}% (전체 {self.space.totalCombinations:,} 조합)")

        startTime = time.time()

        if self.nJobs == 1:
            results = self._runSequential(samples, verbose)
        else:
            results = self._runParallel(samples, verbose)

        elapsedTime = time.time() - startTime

        return self._buildResult(results, elapsedTime, 'random', nTrials)

    def _runSequential(
        self,
        paramsList: List[Dict[str, Any]],
        verbose: bool
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute backtests sequentially, reusing loaded data for efficiency.

        데이터를 재사용하여 순차적으로 백테스트를 실행합니다.

        Args:
            paramsList: List of parameter dicts to test.
            verbose: If True, print progress updates.

        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (params, metric) pairs.
        """
        results = []
        total = len(paramsList)

        if not self.data._loaded:
            self.data.load()

        fullDataDf = self.data.toDataFrame()

        for i, params in enumerate(paramsList):
            strategy = self.strategyFactory(params)

            self.data._index = 0

            engine = BacktestEngine(
                data=self.data,
                strategy=strategy,
                initialCash=self.initialCash,
                broker=self.broker,
            )

            try:
                result = engine.run(verbose=False)
                metricValue = result.metrics.get(self.metric, 0)

                if pd.isna(metricValue) or np.isinf(metricValue):
                    metricValue = float('-inf') if not self.minimize else float('inf')

            except Exception as e:
                if verbose:
                    print(f"[Optimizer] 오류: {params} - {e}")
                metricValue = float('-inf') if not self.minimize else float('inf')

            results.append((params, metricValue))

            if verbose and (i + 1) % max(1, total // 10) == 0:
                progress = (i + 1) / total * 100
                print(f"[Optimizer] 진행: {progress:.0f}% ({i + 1}/{total})")

        return results

    def _runParallel(
        self,
        paramsList: List[Dict[str, Any]],
        verbose: bool
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute backtests in parallel using ProcessPoolExecutor.

        ProcessPoolExecutor를 사용하여 백테스트를 병렬로 실행합니다.

        Args:
            paramsList: List of parameter dicts to test.
            verbose: If True, print progress updates.

        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (params, metric) pairs.
        """
        results = []
        total = len(paramsList)

        args = [
            (params, self._dataConfig, self.strategyFactory, self.metric,
             self.initialCash, self._brokerConfig)
            for params in paramsList
        ]

        completed = 0

        with ProcessPoolExecutor(max_workers=self.nJobs) as executor:
            futures = {executor.submit(_runSingleBacktest, arg): arg for arg in args}

            for future in as_completed(futures):
                try:
                    params, metricValue, _ = future.result()
                    results.append((params, metricValue))
                except Exception as e:
                    if verbose:
                        print(f"[Optimizer] 병렬 실행 오류: {e}")

                completed += 1
                if verbose and completed % max(1, total // 10) == 0:
                    progress = completed / total * 100
                    print(f"[Optimizer] 진행: {progress:.0f}% ({completed}/{total})")

        return results

    def _buildResult(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        elapsedTime: float,
        method: str,
        totalTests: int
    ) -> OptimizeResult:
        """
        Build the OptimizeResult from raw results by selecting the best params.

        원시 결과로부터 최적 파라미터를 선정하여 OptimizeResult를 생성합니다.

        Args:
            results: List of (params, metric) tuples from optimization runs.
            elapsedTime: Total wall-clock time in seconds.
            method: Optimization method name ('grid' or 'random').
            totalTests: Total number of combinations tested.

        Returns:
            OptimizeResult: Complete optimization result with re-run best backtest.
        """
        validResults = [
            (p, m) for p, m in results
            if not (np.isinf(m) or np.isnan(m))
        ]

        if not validResults:
            return OptimizeResult(
                bestParams={},
                bestMetric=0.0,
                bestResult=None,
                allResults=results,
                elapsedTime=elapsedTime,
                metric=self.metric,
                method=method,
                totalTests=totalTests,
            )

        if self.minimize:
            bestParams, bestMetric = min(validResults, key=lambda x: x[1])
        else:
            bestParams, bestMetric = max(validResults, key=lambda x: x[1])

        bestStrategy = self.strategyFactory(bestParams)
        self.data._index = 0

        engine = BacktestEngine(
            data=self.data,
            strategy=bestStrategy,
            initialCash=self.initialCash,
            broker=self.broker,
        )

        bestResult = engine.run(verbose=False)

        return OptimizeResult(
            bestParams=bestParams,
            bestMetric=bestMetric,
            bestResult=bestResult,
            allResults=results,
            elapsedTime=elapsedTime,
            metric=self.metric,
            method=method,
            totalTests=totalTests,
        )

    def __repr__(self) -> str:
        return (
            f"Optimizer(metric={self.metric}, "
            f"params={self.space.names}, "
            f"combinations={self.space.totalCombinations})"
        )
