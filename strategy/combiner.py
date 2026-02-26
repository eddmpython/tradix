"""
Tradex Strategy Combiner Module - Ensemble strategy composition.

Combines multiple trading strategies into an ensemble for more robust
performance. Supports various weighting schemes and dynamic rebalancing
to achieve diversification benefits across different market conditions.

전략 앙상블 모듈 - 여러 전략의 신호를 결합하여 더 안정적인 성과를 추구합니다.

Features:
    - Equal Weight: Uniform allocation across strategies
    - Performance Based: Weight by historical return
    - Sharpe Weighted: Weight by risk-adjusted return (Sharpe ratio)
    - Regime Based: Dynamic allocation based on market regime classification
    - Voting: Majority-vote signal aggregation
    - Dynamic Rebalancing: Periodic weight recalculation

Usage:
    >>> from tradex.strategy.combiner import StrategyCombiner, CombineMethod
    >>>
    >>> combiner = StrategyCombiner()
    >>> combiner.addStrategy('SMA', SmaCrossStrategy())
    >>> combiner.addStrategy('MACD', MacdStrategy())
    >>>
    >>> result = combiner.combine(data, method=CombineMethod.SHARPE_WEIGHTED)
    >>> print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Type
from enum import Enum
import pandas as pd
import numpy as np

from tradex.strategy.base import Strategy
from tradex.engine import BacktestEngine, BacktestResult
from tradex.datafeed.feed import DataFeed
from tradex.advisor.marketClassifier import MarketClassifier, MarketRegime


class CombineMethod(Enum):
    """Enumeration of strategy combination methods. / 전략 결합 방법 열거형."""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_BASED = "performance_based"
    SHARPE_WEIGHTED = "sharpe_weighted"
    REGIME_BASED = "regime_based"
    VOTING = "voting"


@dataclass
class StrategySignal:
    """
    Represents a trading signal emitted by a strategy.

    전략에서 발생한 트레이딩 신호를 나타냅니다.

    Attributes:
        strategyName (str): Name of the originating strategy.
        signal (int): Signal value (-1=sell, 0=hold, 1=buy).
        confidence (float): Confidence level of the signal (0.0-1.0).
        timestamp (pd.Timestamp): Time when the signal was generated.
    """
    strategyName: str
    signal: int
    confidence: float
    timestamp: pd.Timestamp


@dataclass
class EnsembleResult:
    """
    Result container for strategy ensemble analysis.

    앙상블 분석 결과를 담는 데이터 클래스입니다.

    Attributes:
        method (CombineMethod): The combination method used.
        weights (Dict[str, float]): Final strategy weight allocation.
        combinedMetrics (Dict[str, float]): Weighted ensemble performance metrics.
        individualResults (Dict[str, BacktestResult]): Per-strategy backtest results.
        weightHistory (List[Dict[str, float]]): Historical weight changes (for dynamic rebalancing).
    """
    method: CombineMethod
    weights: Dict[str, float]
    combinedMetrics: Dict[str, float]
    individualResults: Dict[str, BacktestResult]
    weightHistory: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"전략 앙상블 결과 ({self.method.value})",
            "=" * 60,
            "",
            "전략별 비중:",
        ]

        for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            result = self.individualResults.get(name)
            sharpe = result.metrics.get('sharpeRatio', 0) if result else 0
            lines.append(f"  {name}: {weight:.1%} (샤프: {sharpe:.2f})")

        lines.extend([
            "",
            "앙상블 성과:",
            f"  기대 수익률: {self.combinedMetrics.get('expectedReturn', 0):.2%}",
            f"  기대 샤프: {self.combinedMetrics.get('expectedSharpe', 0):.2f}",
            f"  분산 효과: {self.combinedMetrics.get('diversificationBenefit', 0):.2%}",
            "=" * 60,
        ])

        return "\n".join(lines)


class StrategyCombiner:
    """
    Multi-strategy ensemble combiner with dynamic weighting.

    Runs individual strategies on the same dataset, then combines results
    using configurable weighting methods to achieve diversification benefits.

    다양한 가중 방법으로 여러 전략을 결합하여 앙상블 효과를 달성합니다.

    Attributes:
        classifier (MarketClassifier): Market regime classifier for regime-based weighting.
        lookbackPeriod (int): Lookback period for performance-based weight calculation.
        strategies (Dict[str, Strategy]): Registered strategy instances.
        strategyFactories (Dict[str, Callable]): Strategy factory functions for fresh instantiation.

    Example:
        >>> combiner = StrategyCombiner()
        >>> combiner.addStrategy('SMA', SmaCrossStrategy())
        >>> combiner.addStrategy('MACD', MacdStrategy())
        >>> result = combiner.combine(data, method=CombineMethod.SHARPE_WEIGHTED)
        >>> print(result.summary())
    """

    def __init__(
        self,
        classifier: MarketClassifier = None,
        lookbackPeriod: int = 60,
    ):
        """
        Args:
            classifier: Market regime classifier for regime-based weighting. / 시장 분류기.
            lookbackPeriod: Lookback window for performance-based weighting. / 성과 기반 비중 계산 기간.
        """
        self.classifier = classifier or MarketClassifier()
        self.lookbackPeriod = lookbackPeriod
        self.strategies: Dict[str, Strategy] = {}
        self.strategyFactories: Dict[str, Callable[[], Strategy]] = {}

    def addStrategy(
        self,
        name: str,
        strategy: Strategy = None,
        factory: Callable[[], Strategy] = None
    ):
        """
        Register a strategy for ensemble combination. / 앙상블 결합을 위한 전략 등록.

        Args:
            name: Unique strategy name. / 전략 이름.
            strategy: Strategy instance. / 전략 인스턴스.
            factory: Strategy factory function for fresh instantiation. / 전략 생성 함수.
        """
        if strategy:
            self.strategies[name] = strategy
        if factory:
            self.strategyFactories[name] = factory

    def combine(
        self,
        data: DataFeed,
        method: CombineMethod = CombineMethod.EQUAL_WEIGHT,
        initialCash: float = 10_000_000,
        regimeWeights: Dict[MarketRegime, Dict[str, float]] = None,
    ) -> EnsembleResult:
        """
        Run all strategies and combine results with the specified method. / 모든 전략을 실행하고 결합.

        Args:
            data: DataFeed with OHLCV data. / 데이터 피드.
            method: Combination method (default EQUAL_WEIGHT). / 결합 방법.
            initialCash: Starting capital for each strategy. / 초기 자본.
            regimeWeights: Per-regime weight overrides (REGIME_BASED only). / 레짐별 비중.

        Returns:
            EnsembleResult: Combined metrics and individual results.
        """
        if not data._loaded:
            data.load()

        individualResults = {}
        for name in self.strategies:
            data._index = 0

            if name in self.strategyFactories:
                strategy = self.strategyFactories[name]()
            else:
                strategy = self.strategies[name]

            engine = BacktestEngine(data, strategy, initialCash=initialCash)
            result = engine.run(verbose=False)
            individualResults[name] = result

        if method == CombineMethod.EQUAL_WEIGHT:
            weights = self._equalWeight()
        elif method == CombineMethod.PERFORMANCE_BASED:
            weights = self._performanceBased(individualResults)
        elif method == CombineMethod.SHARPE_WEIGHTED:
            weights = self._sharpeWeighted(individualResults)
        elif method == CombineMethod.REGIME_BASED:
            weights = self._regimeBased(data, individualResults, regimeWeights)
        elif method == CombineMethod.VOTING:
            weights = self._equalWeight()
        else:
            raise ValueError(f"Unknown method: {method}")

        combinedMetrics = self._calcCombinedMetrics(individualResults, weights)

        return EnsembleResult(
            method=method,
            weights=weights,
            combinedMetrics=combinedMetrics,
            individualResults=individualResults,
        )

    def _equalWeight(self) -> Dict[str, float]:
        """Compute equal weights across all strategies. / 동일 비중 계산."""
        n = len(self.strategies)
        return {name: 1.0 / n for name in self.strategies}

    def _performanceBased(
        self,
        results: Dict[str, BacktestResult]
    ) -> Dict[str, float]:
        """Compute weights based on historical return performance. / 과거 수익률 기반 비중 계산."""
        returns = {}
        for name, result in results.items():
            ret = result.metrics.get('totalReturn', 0)
            returns[name] = max(ret, 0.001)

        total = sum(returns.values())
        return {name: ret / total for name, ret in returns.items()}

    def _sharpeWeighted(
        self,
        results: Dict[str, BacktestResult]
    ) -> Dict[str, float]:
        """Compute weights based on Sharpe ratio. / 샤프 비율 기반 비중 계산."""
        sharpes = {}
        for name, result in results.items():
            sharpe = result.metrics.get('sharpeRatio', 0)
            sharpes[name] = max(sharpe, 0.01)

        total = sum(sharpes.values())
        if total <= 0:
            return self._equalWeight()

        return {name: s / total for name, s in sharpes.items()}

    def _regimeBased(
        self,
        data: DataFeed,
        results: Dict[str, BacktestResult],
        regimeWeights: Dict[MarketRegime, Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Compute weights based on current market regime. / 시장 레짐 기반 비중 계산."""
        df = data.toDataFrame()
        analysis = self.classifier.analyze(df)
        currentRegime = analysis.regime

        if regimeWeights and currentRegime in regimeWeights:
            baseWeights = regimeWeights[currentRegime]
            totalWeight = sum(baseWeights.get(name, 0) for name in self.strategies)

            if totalWeight > 0:
                return {
                    name: baseWeights.get(name, 0) / totalWeight
                    for name in self.strategies
                }

        return self._sharpeWeighted(results)

    def _calcCombinedMetrics(
        self,
        results: Dict[str, BacktestResult],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute weighted ensemble performance metrics. / 가중 앙상블 성과 지표 계산."""
        weightedReturn = 0
        weightedSharpe = 0
        avgSharpe = 0

        for name, result in results.items():
            w = weights.get(name, 0)
            ret = result.metrics.get('totalReturn', 0)
            sharpe = result.metrics.get('sharpeRatio', 0)

            weightedReturn += w * ret
            weightedSharpe += w * sharpe
            avgSharpe += sharpe / len(results)

        diversificationBenefit = weightedSharpe - avgSharpe

        returns = []
        for name, result in results.items():
            if hasattr(result, 'equityCurve') and result.equityCurve:
                equity = pd.Series(result.equityCurve)
                dailyReturns = equity.pct_change().dropna()
                returns.append(dailyReturns.values)

        if returns and len(returns) > 1:
            minLen = min(len(r) for r in returns)
            returns = [r[:minLen] for r in returns]
            corrMatrix = np.corrcoef(returns)
            avgCorr = (np.sum(corrMatrix) - len(returns)) / (len(returns) * (len(returns) - 1))
        else:
            avgCorr = 1.0

        return {
            'expectedReturn': weightedReturn,
            'expectedSharpe': weightedSharpe,
            'diversificationBenefit': diversificationBenefit,
            'avgCorrelation': avgCorr,
            'nStrategies': len(results),
        }

    def votingSignal(
        self,
        signals: Dict[str, int],
        threshold: float = 0.5,
    ) -> Tuple[int, float]:
        """
        Combine signals using majority voting. / 다수결 투표로 신호 결합.

        Args:
            signals: Dict mapping strategy name to signal (-1, 0, 1). / {전략명: 신호}.
            threshold: Decision threshold (default 0.5). / 결정 임계값.

        Returns:
            tuple: (combined signal, confidence ratio). / (결합 신호, 신뢰도).
        """
        if not signals:
            return 0, 0.0

        buyVotes = sum(1 for s in signals.values() if s > 0)
        sellVotes = sum(1 for s in signals.values() if s < 0)
        total = len(signals)

        buyRatio = buyVotes / total
        sellRatio = sellVotes / total

        if buyRatio >= threshold:
            return 1, buyRatio
        elif sellRatio >= threshold:
            return -1, sellRatio
        else:
            return 0, 1 - buyRatio - sellRatio

    def dynamicRebalance(
        self,
        data: DataFeed,
        rebalancePeriod: int = 20,
        method: CombineMethod = CombineMethod.SHARPE_WEIGHTED,
        initialCash: float = 10_000_000,
    ) -> pd.DataFrame:
        """
        Simulate dynamic weight rebalancing over time. / 시간에 따른 동적 비중 리밸런싱 시뮬레이션.

        Args:
            data: DataFeed with OHLCV data. / 데이터 피드.
            rebalancePeriod: Rebalancing frequency in trading days. / 리밸런싱 주기 (거래일).
            method: Weight calculation method. / 비중 결정 방법.
            initialCash: Starting capital. / 초기 자본.

        Returns:
            pd.DataFrame: Weight history with date and per-strategy allocations.
        """
        if not data._loaded:
            data.load()

        df = data.toDataFrame()
        nBars = len(df)

        weightHistory = []
        currentWeights = self._equalWeight()

        for i in range(0, nBars, rebalancePeriod):
            if i + self.lookbackPeriod > nBars:
                break

            subData = data.subset(0, i + self.lookbackPeriod)

            try:
                result = self.combine(subData, method=method, initialCash=initialCash)
                currentWeights = result.weights
            except Exception:
                pass

            weightHistory.append({
                'date': df.index[min(i + self.lookbackPeriod - 1, nBars - 1)],
                **currentWeights
            })

        return pd.DataFrame(weightHistory)
