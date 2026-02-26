"""
Tradex Portfolio Optimizer Module.

Computes optimal asset allocation weights across multiple assets or strategies
using classical portfolio optimization methods. Provides efficient frontier
computation and strategy comparison utilities.

여러 자산 또는 전략의 최적 배분 비중을 클래식 포트폴리오 최적화 방법론으로
계산하는 모듈입니다. 효율적 투자선 계산 및 전략 비교 유틸리티를 제공합니다.

Features:
    - Mean-Variance (Markowitz) optimization
    - Maximum Sharpe ratio portfolio
    - Minimum volatility portfolio
    - Risk parity (equal risk contribution)
    - Equal weight benchmark
    - Efficient frontier computation
    - Strategy comparison from BacktestResult objects
    - Weight constraints (min/max per asset)

Usage:
    from tradex.portfolio.optimizer import PortfolioOptimizer, OptimizationMethod

    optimizer = PortfolioOptimizer(riskFreeRate=0.02)
    optimizer.fit(daily_returns_df)

    weights = optimizer.optimize(OptimizationMethod.MAX_SHARPE)
    print(weights.summary())

    frontier = optimizer.efficientFrontier(nPoints=50)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class OptimizationMethod(Enum):
    """
    Enumeration of available portfolio optimization methods.

    사용 가능한 포트폴리오 최적화 방법 열거형입니다.
    """
    EQUAL_WEIGHT = "equal_weight"
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    RISK_PARITY = "risk_parity"


@dataclass
class PortfolioWeights:
    """
    Container for portfolio optimization results with asset weights and metrics.

    자산 비중 및 성과 지표를 담는 포트폴리오 최적화 결과 컨테이너입니다.

    Attributes:
        weights (Dict[str, float]): Asset name to weight mapping (자산별 비중).
        method (OptimizationMethod): Method used for optimization (최적화 방법).
        expectedReturn (float): Annualized expected return (기대 수익률).
        expectedVolatility (float): Annualized expected volatility (기대 변동성).
        sharpeRatio (float): Expected Sharpe ratio (샤프 비율).
        metrics (Dict[str, float]): Additional metrics such as
            diversificationRatio and effectiveN (추가 지표).
    """
    weights: Dict[str, float]
    method: OptimizationMethod
    expectedReturn: float
    expectedVolatility: float
    sharpeRatio: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Generate a formatted summary of portfolio weights and expected metrics.

        포트폴리오 비중 및 기대 성과 지표의 포맷된 요약을 생성합니다.

        Returns:
            str: Multi-line summary with asset weights, expected return,
                volatility, and Sharpe ratio.
        """
        lines = [
            "=" * 50,
            f"포트폴리오 최적화 결과 ({self.method.value})",
            "=" * 50,
            "",
            "비중:",
        ]

        for asset, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            if weight > 0.001:
                lines.append(f"  {asset}: {weight:.1%}")

        lines.extend([
            "",
            f"기대 수익률: {self.expectedReturn:.2%}",
            f"기대 변동성: {self.expectedVolatility:.2%}",
            f"샤프 비율: {self.sharpeRatio:.2f}",
            "=" * 50,
        ])

        return "\n".join(lines)


@dataclass
class EfficientFrontier:
    """
    Efficient frontier data container for mean-variance analysis.

    평균-분산 분석을 위한 효율적 투자선 데이터 컨테이너입니다.

    Attributes:
        returns (List[float]): Expected returns at each frontier point
            (각 포인트의 기대 수익률).
        volatilities (List[float]): Expected volatilities at each point
            (각 포인트의 기대 변동성).
        sharpeRatios (List[float]): Sharpe ratios at each point
            (각 포인트의 샤프 비율).
        weights (List[Dict[str, float]]): Asset weights at each point
            (각 포인트의 자산 비중).
        optimalPoint (int): Index of the maximum Sharpe ratio point
            (최대 샤프 비율 포인트의 인덱스).
    """
    returns: List[float]
    volatilities: List[float]
    sharpeRatios: List[float]
    weights: List[Dict[str, float]]
    optimalPoint: int


class PortfolioOptimizer:
    """
    Multi-asset portfolio optimizer with classical allocation methods.

    Fits on a daily returns DataFrame and computes optimal weights using
    mean-variance, maximum Sharpe, minimum volatility, risk parity, or
    equal weight methods. Also generates the efficient frontier.

    일별 수익률 DataFrame을 학습하고 평균-분산, 최대 샤프, 최소 변동성,
    리스크 패리티, 동일 비중 방법으로 최적 비중을 계산합니다.
    효율적 투자선도 생성합니다.

    Attributes:
        riskFreeRate (float): Annualized risk-free rate (무위험 이자율).
        returns (pd.DataFrame): Fitted daily returns data (수익률 데이터).
        assets (List[str]): Asset names (자산 이름 목록).
        nAssets (int): Number of assets (자산 수).
        meanReturns (np.ndarray): Annualized mean returns (연율화 평균 수익률).
        covMatrix (np.ndarray): Annualized covariance matrix (연율화 공분산 행렬).

    Example:
        >>> optimizer = PortfolioOptimizer(riskFreeRate=0.02)
        >>> optimizer.fit(returns_df)
        >>> weights = optimizer.optimize(OptimizationMethod.MAX_SHARPE)
        >>> print(weights.summary())
    """

    def __init__(self, riskFreeRate: float = 0.02):
        """
        Initialize the PortfolioOptimizer.

        PortfolioOptimizer를 초기화합니다.

        Args:
            riskFreeRate: Annualized risk-free interest rate, default 2%
                (무위험 이자율, 연율, 기본 2%).
        """
        self.riskFreeRate = riskFreeRate
        self.returns: pd.DataFrame = None
        self.assets: List[str] = []
        self.nAssets: int = 0
        self.meanReturns: np.ndarray = None
        self.covMatrix: np.ndarray = None
        self._fitted = False

    def fit(self, returns: pd.DataFrame) -> 'PortfolioOptimizer':
        """
        Fit the optimizer on daily return data.

        일별 수익률 데이터로 최적화기를 학습합니다.

        Args:
            returns: Daily returns DataFrame where columns are asset names
                and rows are dates (일별 수익률 DataFrame, columns=자산, rows=날짜).

        Returns:
            PortfolioOptimizer: Self reference for method chaining.
        """
        self.returns = returns.copy()
        self.assets = list(returns.columns)
        self.nAssets = len(self.assets)

        self.meanReturns = returns.mean().values * 252
        self.covMatrix = returns.cov().values * 252

        self._fitted = True
        return self

    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Dict = None,
    ) -> PortfolioWeights:
        """
        Compute optimal portfolio weights using the specified method.

        지정된 방법으로 최적 포트폴리오 비중을 계산합니다.

        Args:
            method: Optimization method from OptimizationMethod enum
                (최적화 방법).
            constraints: Optional dict with 'min_weight', 'max_weight',
                and/or 'target_return' keys (제약 조건).

        Returns:
            PortfolioWeights: Optimal weights with expected performance metrics.

        Raises:
            ValueError: If fit() has not been called or method is unknown.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        constraints = constraints or {}
        minWeight = constraints.get('min_weight', 0.0)
        maxWeight = constraints.get('max_weight', 1.0)

        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equalWeight()
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._maxSharpe(minWeight, maxWeight)
        elif method == OptimizationMethod.MIN_VOLATILITY:
            weights = self._minVolatility(minWeight, maxWeight)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._riskParity(minWeight, maxWeight)
        elif method == OptimizationMethod.MEAN_VARIANCE:
            targetReturn = constraints.get('target_return', None)
            weights = self._meanVariance(minWeight, maxWeight, targetReturn)
        else:
            raise ValueError(f"Unknown method: {method}")

        expReturn, expVol = self._calcPortfolioStats(weights)
        sharpe = (expReturn - self.riskFreeRate) / expVol if expVol > 0 else 0

        return PortfolioWeights(
            weights={asset: w for asset, w in zip(self.assets, weights)},
            method=method,
            expectedReturn=expReturn,
            expectedVolatility=expVol,
            sharpeRatio=sharpe,
            metrics={
                'diversificationRatio': self._diversificationRatio(weights),
                'effectiveN': self._effectiveN(weights),
            }
        )

    def _equalWeight(self) -> np.ndarray:
        """
        Compute equal weight allocation (1/N).

        동일 비중(1/N) 배분을 계산합니다.

        Returns:
            np.ndarray: Equal weight array.
        """
        return np.ones(self.nAssets) / self.nAssets

    def _maxSharpe(self, minWeight: float, maxWeight: float) -> np.ndarray:
        """
        Optimize for maximum Sharpe ratio.

        샤프 비율을 최대화하는 비중을 계산합니다.

        Args:
            minWeight: Minimum weight per asset.
            maxWeight: Maximum weight per asset.

        Returns:
            np.ndarray: Optimal weight array.
        """
        def negSharpe(w):
            ret, vol = self._calcPortfolioStats(w)
            return -(ret - self.riskFreeRate) / vol if vol > 0 else 0

        return self._optimize(negSharpe, minWeight, maxWeight)

    def _minVolatility(self, minWeight: float, maxWeight: float) -> np.ndarray:
        """
        Optimize for minimum portfolio volatility.

        포트폴리오 변동성을 최소화하는 비중을 계산합니다.

        Args:
            minWeight: Minimum weight per asset.
            maxWeight: Maximum weight per asset.

        Returns:
            np.ndarray: Optimal weight array.
        """
        def volatility(w):
            return np.sqrt(np.dot(w.T, np.dot(self.covMatrix, w)))

        return self._optimize(volatility, minWeight, maxWeight)

    def _riskParity(self, minWeight: float, maxWeight: float) -> np.ndarray:
        """
        Optimize for risk parity (equal risk contribution per asset).

        리스크 패리티(자산별 동일 리스크 기여도) 비중을 계산합니다.

        Args:
            minWeight: Minimum weight per asset.
            maxWeight: Maximum weight per asset.

        Returns:
            np.ndarray: Optimal weight array.
        """
        def riskParityObjective(w):
            portVol = np.sqrt(np.dot(w.T, np.dot(self.covMatrix, w)))
            marginalContrib = np.dot(self.covMatrix, w)
            riskContrib = w * marginalContrib / portVol
            targetRisk = portVol / self.nAssets
            return np.sum((riskContrib - targetRisk) ** 2)

        return self._optimize(riskParityObjective, minWeight, maxWeight)

    def _meanVariance(
        self,
        minWeight: float,
        maxWeight: float,
        targetReturn: float = None
    ) -> np.ndarray:
        """
        Perform Markowitz mean-variance optimization for a target return.

        목표 수익률에 대한 마코위츠 평균-분산 최적화를 수행합니다.

        Args:
            minWeight: Minimum weight per asset.
            maxWeight: Maximum weight per asset.
            targetReturn: Target annualized return; defaults to mean of asset
                returns if None.

        Returns:
            np.ndarray: Optimal weight array.
        """
        if targetReturn is None:
            targetReturn = np.mean(self.meanReturns)

        def volatility(w):
            return np.sqrt(np.dot(w.T, np.dot(self.covMatrix, w)))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.meanReturns) - targetReturn}
        ]

        bounds = [(minWeight, maxWeight) for _ in range(self.nAssets)]
        x0 = np.ones(self.nAssets) / self.nAssets

        result = minimize(
            volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-10}
        )

        return result.x if result.success else x0

    def _optimize(
        self,
        objective,
        minWeight: float,
        maxWeight: float
    ) -> np.ndarray:
        """
        Execute SLSQP optimization with sum-to-one constraint.

        합계 1 제약 조건으로 SLSQP 최적화를 실행합니다.

        Args:
            objective: Objective function to minimize.
            minWeight: Minimum weight per asset.
            maxWeight: Maximum weight per asset.

        Returns:
            np.ndarray: Optimized weight array, or equal weights on failure.
        """
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(minWeight, maxWeight) for _ in range(self.nAssets)]
        x0 = np.ones(self.nAssets) / self.nAssets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-10}
        )

        return result.x if result.success else x0

    def _calcPortfolioStats(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate expected portfolio return and volatility for given weights.

        주어진 비중에 대한 포트폴리오 기대 수익률과 변동성을 계산합니다.

        Args:
            weights: Asset weight array.

        Returns:
            Tuple[float, float]: (expected_return, expected_volatility).
        """
        expReturn = np.dot(weights, self.meanReturns)
        expVol = np.sqrt(np.dot(weights.T, np.dot(self.covMatrix, weights)))
        return expReturn, expVol

    def _diversificationRatio(self, weights: np.ndarray) -> float:
        """
        Calculate diversification ratio (higher means greater diversification).

        분산 비율을 계산합니다 (높을수록 분산 효과가 큼).

        Args:
            weights: Asset weight array.

        Returns:
            float: Diversification ratio.
        """
        assetVols = np.sqrt(np.diag(self.covMatrix))
        weightedVol = np.dot(weights, assetVols)
        portVol = np.sqrt(np.dot(weights.T, np.dot(self.covMatrix, weights)))
        return weightedVol / portVol if portVol > 0 else 1.0

    def _effectiveN(self, weights: np.ndarray) -> float:
        """
        Calculate effective number of assets (inverse Herfindahl Index).

        유효 자산 수를 계산합니다 (허핀달 지수의 역수).

        Args:
            weights: Asset weight array.

        Returns:
            float: Effective number of assets.
        """
        return 1.0 / np.sum(weights ** 2)

    def efficientFrontier(self, nPoints: int = 50) -> EfficientFrontier:
        """
        Compute the efficient frontier as a series of optimal portfolios.

        일련의 최적 포트폴리오로 효율적 투자선을 계산합니다.

        Args:
            nPoints: Number of points along the frontier (포인트 수).

        Returns:
            EfficientFrontier: Frontier data with returns, volatilities,
                Sharpe ratios, weights, and optimal point index.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        minRet = self.meanReturns.min()
        maxRet = self.meanReturns.max()
        targetReturns = np.linspace(minRet, maxRet, nPoints)

        returns = []
        volatilities = []
        sharpeRatios = []
        allWeights = []

        for targetRet in targetReturns:
            try:
                weights = self._meanVariance(0, 1, targetRet)
                ret, vol = self._calcPortfolioStats(weights)
                sharpe = (ret - self.riskFreeRate) / vol if vol > 0 else 0

                returns.append(ret)
                volatilities.append(vol)
                sharpeRatios.append(sharpe)
                allWeights.append({asset: w for asset, w in zip(self.assets, weights)})
            except Exception:
                continue

        optimalIdx = np.argmax(sharpeRatios) if sharpeRatios else 0

        return EfficientFrontier(
            returns=returns,
            volatilities=volatilities,
            sharpeRatios=sharpeRatios,
            weights=allWeights,
            optimalPoint=optimalIdx,
        )

    def compareStrategies(
        self,
        backtestResults: Dict[str, 'BacktestResult']
    ) -> pd.DataFrame:
        """
        Compare multiple strategies using their backtest results.

        여러 전략의 백테스트 결과를 비교합니다.

        Args:
            backtestResults: Dict mapping strategy names to BacktestResult
                objects ({전략명: BacktestResult}).

        Returns:
            pd.DataFrame: Comparison table with return, volatility, Sharpe,
                MDD, and Calmar metrics sorted by Sharpe ratio descending.
        """
        data = []
        for name, result in backtestResults.items():
            m = result.metrics
            data.append({
                'strategy': name,
                'totalReturn': m.get('totalReturn', 0),
                'annualReturn': m.get('annualReturn', 0),
                'volatility': m.get('annualVolatility', 0),
                'sharpeRatio': m.get('sharpeRatio', 0),
                'maxDrawdown': m.get('maxDrawdown', 0),
                'calmarRatio': m.get('calmarRatio', 0),
            })

        df = pd.DataFrame(data)
        df = df.sort_values('sharpeRatio', ascending=False).reset_index(drop=True)
        return df
