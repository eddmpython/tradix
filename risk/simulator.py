"""
Tradix Risk Simulator Module.

Provides comprehensive risk analysis tools including Value at Risk,
Conditional VaR, Monte Carlo simulation, stress testing, tail risk
analysis, and drawdown analysis for portfolio risk assessment.

리스크 시뮬레이터 모듈 - VaR, CVaR, 몬테카를로 시뮬레이션, 스트레스
테스트 등 포트폴리오 리스크 분석 도구를 제공합니다.

Features:
    - Value at Risk: Historical, Parametric (Normal), Monte Carlo methods
    - Conditional VaR (Expected Shortfall) for tail risk measurement
    - Monte Carlo portfolio simulation with bootstrap option
    - Historical stress testing with pre-configured crisis scenarios
    - Tail risk analysis (skewness, kurtosis, tail ratios)
    - Drawdown analysis (max drawdown, duration, Ulcer Index)

Usage:
    >>> from tradix.risk.simulator import RiskSimulator, VaRMethod
    >>> simulator = RiskSimulator()
    >>> simulator.fit(daily_returns)
    >>> var_result = simulator.calcVaR(confidence=0.95, horizon=1)
    >>> mc_result = simulator.monteCarloSimulation(nSim=10000, horizon=252)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats


class VaRMethod(Enum):
    """
    Enumeration of available Value at Risk calculation methods.

    VaR 계산 방법 열거형.
    """
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """
    Data class holding the result of a VaR calculation.

    VaR 계산 결과를 담는 데이터 클래스.

    Attributes:
        var (float): Value at Risk as a positive fraction. VaR 값.
        cvar (float): Conditional VaR (Expected Shortfall). CVaR 값.
        method (VaRMethod): Calculation method used. 계산 방법.
        confidenceLevel (float): Confidence level (e.g., 0.95). 신뢰 수준.
        horizon (int): Holding period in days. 보유 기간 (일).
        details (dict): Additional calculation metadata. 추가 상세 정보.
    """
    var: float
    cvar: float
    method: VaRMethod
    confidenceLevel: float
    horizon: int
    details: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"VaR ({self.method.value}, {self.confidenceLevel:.0%}, {self.horizon}일)\n"
            f"  VaR: {self.var:.2%}\n"
            f"  CVaR: {self.cvar:.2%}"
        )


@dataclass
class MonteCarloResult:
    """
    Data class holding Monte Carlo simulation results.

    몬테카를로 시뮬레이션 결과를 담는 데이터 클래스.

    Attributes:
        paths (np.ndarray): Simulated price paths, shape (nSim, horizon+1).
            시뮬레이션 경로.
        finalValues (np.ndarray): Terminal values for each simulation.
            최종 가치.
        percentiles (dict[int, float]): Percentile distribution of final values.
            백분위 분포.
        expectedValue (float): Mean final value. 기대 가치.
        volatility (float): Standard deviation of final values. 변동성.
        probabilityOfLoss (float): Probability of ending below initial value.
            손실 확률.
        maxDrawdowns (np.ndarray): Maximum drawdown for each simulation path.
            최대 낙폭.
        metrics (dict): Additional performance metrics. 추가 성과 지표.
    """
    paths: np.ndarray
    finalValues: np.ndarray
    percentiles: Dict[int, float]
    expectedValue: float
    volatility: float
    probabilityOfLoss: float
    maxDrawdowns: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "몬테카를로 시뮬레이션 결과",
            "=" * 50,
            f"시뮬레이션 수: {len(self.finalValues):,}",
            f"기대 수익률: {self.expectedValue - 1:.2%}",
            f"변동성: {self.volatility:.2%}",
            f"손실 확률: {self.probabilityOfLoss:.1%}",
            "",
            "수익률 분포:",
        ]

        for p, v in sorted(self.percentiles.items()):
            lines.append(f"  {p}%ile: {v - 1:.2%}")

        lines.append("")
        lines.append(f"최대 MDD 평균: {np.mean(self.maxDrawdowns):.2%}")
        lines.append(f"최대 MDD 최악: {np.min(self.maxDrawdowns):.2%}")
        lines.append("=" * 50)

        return "\n".join(lines)


@dataclass
class StressTestResult:
    """
    Data class holding a single stress test scenario result.

    스트레스 테스트 시나리오 결과를 담는 데이터 클래스.

    Attributes:
        scenario (str): Scenario name. 시나리오명.
        portfolioReturn (float): Expected portfolio return. 포트폴리오 수익률.
        portfolioValue (float): Portfolio value after scenario. 시나리오 후 가치.
        individualReturns (dict): Per-asset returns. 개별 자산 수익률.
        riskMetrics (dict): Risk metrics for the scenario. 리스크 지표.
    """
    scenario: str
    portfolioReturn: float
    portfolioValue: float
    individualReturns: Dict[str, float]
    riskMetrics: Dict[str, float]


class RiskSimulator:
    """
    Comprehensive risk simulator for portfolio risk analysis.

    Provides Value at Risk, Monte Carlo simulation, stress testing,
    tail risk analysis, and drawdown analysis. Must be fitted with
    historical return data before use.

    리스크 시뮬레이터 - VaR, 몬테카를로 시뮬레이션, 스트레스 테스트 등
    다양한 방법으로 포트폴리오 리스크를 분석합니다.

    Example:
        >>> simulator = RiskSimulator(riskFreeRate=0.02)
        >>> simulator.fit(daily_returns)
        >>> var = simulator.calcVaR(confidence=0.95, horizon=1)
        >>> mc = simulator.monteCarloSimulation(nSim=10000, horizon=252)
        >>> stress = simulator.stressTest(scenarios={'crash': -0.20})
    """

    HISTORICAL_SCENARIOS = {
        'covid_crash_2020': {'return': -0.34, 'volatility': 0.80, 'description': '2020 코로나 폭락'},
        'financial_crisis_2008': {'return': -0.50, 'volatility': 0.60, 'description': '2008 금융위기'},
        'dot_com_crash_2000': {'return': -0.45, 'volatility': 0.40, 'description': '2000 닷컴버블'},
        'flash_crash_2010': {'return': -0.10, 'volatility': 1.50, 'description': '2010 플래시 크래시'},
        'mild_correction': {'return': -0.10, 'volatility': 0.25, 'description': '일반적 조정'},
        'severe_bear': {'return': -0.30, 'volatility': 0.50, 'description': '심각한 하락장'},
    }

    def __init__(self, riskFreeRate: float = 0.02):
        self.riskFreeRate = riskFreeRate
        self.returns: pd.DataFrame = None
        self.meanReturn: float = None
        self.volatility: float = None
        self._fitted = False

    def fit(self, returns: pd.DataFrame) -> 'RiskSimulator':
        """
        Fit the simulator with historical return data.

        Computes annualized mean return and volatility from daily returns.

        수익률 데이터로 시뮬레이터를 학습시킵니다.

        Args:
            returns (pd.DataFrame or pd.Series): Daily returns. If a
                multi-column DataFrame, uses the row-wise mean.
                일별 수익률 데이터.

        Returns:
            RiskSimulator: Self, for method chaining.
        """
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] == 1:
                self.returns = returns.iloc[:, 0]
            else:
                self.returns = returns.mean(axis=1)
        else:
            self.returns = returns

        self.returns = self.returns.dropna()
        self.meanReturn = self.returns.mean() * 252
        self.volatility = self.returns.std() * np.sqrt(252)
        self._fitted = True

        return self

    def calcVaR(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        method: VaRMethod = VaRMethod.HISTORICAL,
        nSim: int = 10000,
    ) -> VaRResult:
        """
        Calculate Value at Risk and Conditional VaR.

        VaR과 CVaR(Expected Shortfall)을 계산합니다.

        Args:
            confidence (float): Confidence level (0.95 = 95%).
                Default: 0.95. 신뢰 수준.
            horizon (int): Holding period in trading days.
                Default: 1. 보유 기간.
            method (VaRMethod): Calculation method. Default: HISTORICAL.
                계산 방법.
            nSim (int): Number of simulations (Monte Carlo only).
                Default: 10000. 시뮬레이션 수.

        Returns:
            VaRResult: VaR and CVaR values with metadata.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        alpha = 1 - confidence

        if method == VaRMethod.HISTORICAL:
            var, cvar = self._historicalVaR(alpha, horizon)
        elif method == VaRMethod.PARAMETRIC:
            var, cvar = self._parametricVaR(alpha, horizon)
        elif method == VaRMethod.MONTE_CARLO:
            var, cvar = self._monteCarloVaR(alpha, horizon, nSim)
        else:
            raise ValueError(f"Unknown method: {method}")

        return VaRResult(
            var=var,
            cvar=cvar,
            method=method,
            confidenceLevel=confidence,
            horizon=horizon,
            details={
                'meanReturn': self.meanReturn,
                'volatility': self.volatility,
            }
        )

    def _historicalVaR(self, alpha: float, horizon: int) -> Tuple[float, float]:
        """Calculate VaR using historical simulation. Historical VaR 계산."""
        if horizon > 1:
            multiDayReturns = self.returns.rolling(horizon).sum().dropna()
        else:
            multiDayReturns = self.returns

        var = np.percentile(multiDayReturns, alpha * 100)
        cvar = multiDayReturns[multiDayReturns <= var].mean()

        return -var, -cvar

    def _parametricVaR(self, alpha: float, horizon: int) -> Tuple[float, float]:
        """Calculate VaR using parametric (normal distribution) method. Parametric VaR 계산."""
        dailyMean = self.returns.mean()
        dailyStd = self.returns.std()

        periodMean = dailyMean * horizon
        periodStd = dailyStd * np.sqrt(horizon)

        zScore = stats.norm.ppf(alpha)
        var = -(periodMean + zScore * periodStd)

        cvarZScore = stats.norm.pdf(zScore) / alpha
        cvar = -(periodMean - cvarZScore * periodStd)

        return var, cvar

    def _monteCarloVaR(
        self,
        alpha: float,
        horizon: int,
        nSim: int
    ) -> Tuple[float, float]:
        """Calculate VaR using Monte Carlo simulation. 몬테카를로 VaR 계산."""
        dailyMean = self.returns.mean()
        dailyStd = self.returns.std()

        simReturns = np.random.normal(dailyMean, dailyStd, (nSim, horizon))
        periodReturns = simReturns.sum(axis=1)

        var = np.percentile(periodReturns, alpha * 100)
        cvar = periodReturns[periodReturns <= var].mean()

        return -var, -cvar

    def monteCarloSimulation(
        self,
        initialValue: float = 1.0,
        horizon: int = 252,
        nSim: int = 10000,
        useBootstrap: bool = False,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo portfolio simulation.

        몬테카를로 포트폴리오 시뮬레이션을 실행합니다.

        Args:
            initialValue (float): Starting portfolio value.
                Default: 1.0. 초기 포트폴리오 가치.
            horizon (int): Simulation period in trading days.
                Default: 252. 시뮬레이션 기간.
            nSim (int): Number of simulation paths.
                Default: 10000. 시뮬레이션 횟수.
            useBootstrap (bool): If True, sample from historical returns;
                if False, use parametric normal distribution.
                Default: False. 부트스트랩 사용 여부.

        Returns:
            MonteCarloResult: Simulation paths, final values,
                percentile distribution, and drawdown statistics.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        paths = np.zeros((nSim, horizon + 1))
        paths[:, 0] = initialValue

        if useBootstrap:
            for t in range(1, horizon + 1):
                sampledReturns = np.random.choice(self.returns.values, size=nSim)
                paths[:, t] = paths[:, t - 1] * (1 + sampledReturns)
        else:
            dailyMean = self.returns.mean()
            dailyStd = self.returns.std()

            for t in range(1, horizon + 1):
                randomReturns = np.random.normal(dailyMean, dailyStd, nSim)
                paths[:, t] = paths[:, t - 1] * (1 + randomReturns)

        finalValues = paths[:, -1]

        maxDrawdowns = np.zeros(nSim)
        for i in range(nSim):
            peak = np.maximum.accumulate(paths[i])
            drawdown = (paths[i] - peak) / peak
            maxDrawdowns[i] = drawdown.min()

        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(finalValues, p)

        return MonteCarloResult(
            paths=paths,
            finalValues=finalValues,
            percentiles=percentiles,
            expectedValue=finalValues.mean(),
            volatility=finalValues.std(),
            probabilityOfLoss=(finalValues < initialValue).mean(),
            maxDrawdowns=maxDrawdowns,
            metrics={
                'meanReturn': (finalValues.mean() / initialValue - 1),
                'medianReturn': (np.median(finalValues) / initialValue - 1),
                'worstCase': (finalValues.min() / initialValue - 1),
                'bestCase': (finalValues.max() / initialValue - 1),
            }
        )

    def stressTest(
        self,
        scenarios: Dict[str, float] = None,
        useHistorical: bool = True,
    ) -> List[StressTestResult]:
        """
        Run stress tests against predefined or custom scenarios.

        사전 정의된 또는 사용자 정의 시나리오에 대해 스트레스 테스트를 실행합니다.

        Args:
            scenarios (dict[str, float], optional): Custom scenarios mapping
                scenario name to expected return. If None, uses default
                scenarios. 시나리오 딕셔너리.
            useHistorical (bool): If True and scenarios is None, use
                built-in historical crisis scenarios. Default: True.
                역사적 시나리오 사용 여부.

        Returns:
            list[StressTestResult]: Results for each scenario including
                portfolio return, value, and recovery metrics.
        """
        if scenarios is None and useHistorical:
            scenarios = {
                name: info['return']
                for name, info in self.HISTORICAL_SCENARIOS.items()
            }
        elif scenarios is None:
            scenarios = {
                'mild_drop': -0.10,
                'moderate_drop': -0.20,
                'severe_drop': -0.30,
                'crash': -0.40,
            }

        results = []
        for scenarioName, expectedReturn in scenarios.items():
            description = ""
            if scenarioName in self.HISTORICAL_SCENARIOS:
                description = self.HISTORICAL_SCENARIOS[scenarioName]['description']

            results.append(StressTestResult(
                scenario=scenarioName,
                portfolioReturn=expectedReturn,
                portfolioValue=1 + expectedReturn,
                individualReturns={},
                riskMetrics={
                    'description': description,
                    'recoveryNeeded': -expectedReturn / (1 + expectedReturn),
                }
            ))

        return results

    def tailRiskAnalysis(self) -> Dict[str, float]:
        """
        Analyze tail risk characteristics of the return distribution.

        수익률 분포의 꼬리 리스크 특성을 분석합니다.

        Returns:
            dict: Tail risk metrics including 'skewness', 'kurtosis',
                'leftTailMean', 'rightTailMean', 'tailRatio',
                'maxLoss', and 'maxGain'.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        returns = self.returns.values

        leftTail = returns[returns < np.percentile(returns, 5)]
        rightTail = returns[returns > np.percentile(returns, 95)]

        return {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'leftTailMean': leftTail.mean() if len(leftTail) > 0 else 0,
            'rightTailMean': rightTail.mean() if len(rightTail) > 0 else 0,
            'leftTailStd': leftTail.std() if len(leftTail) > 0 else 0,
            'rightTailStd': rightTail.std() if len(rightTail) > 0 else 0,
            'tailRatio': abs(leftTail.mean() / rightTail.mean()) if rightTail.mean() != 0 else 0,
            'maxLoss': returns.min(),
            'maxGain': returns.max(),
        }

    def drawdownAnalysis(self) -> Dict[str, float]:
        """
        Analyze drawdown characteristics of the return series.

        수익률 시리즈의 드로우다운 특성을 분석합니다.

        Returns:
            dict: Drawdown statistics including 'maxDrawdown',
                'avgDrawdown', 'drawdownDuration', 'maxDrawdownDuration',
                'recoveryTime', and 'ulcerIndex'.

        Raises:
            ValueError: If fit() has not been called.
        """
        if not self._fitted:
            raise ValueError("fit()을 먼저 호출하세요")

        cumReturns = (1 + self.returns).cumprod()
        peak = cumReturns.cummax()
        drawdown = (cumReturns - peak) / peak

        ddPeriods = []
        inDd = False
        ddStart = None

        for i, dd in enumerate(drawdown):
            if dd < 0 and not inDd:
                inDd = True
                ddStart = i
            elif dd >= 0 and inDd:
                inDd = False
                ddPeriods.append(i - ddStart)

        return {
            'maxDrawdown': drawdown.min(),
            'avgDrawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdownDuration': np.mean(ddPeriods) if ddPeriods else 0,
            'maxDrawdownDuration': max(ddPeriods) if ddPeriods else 0,
            'recoveryTime': len(ddPeriods),
            'ulcerIndex': np.sqrt(np.mean(drawdown ** 2)),
        }
