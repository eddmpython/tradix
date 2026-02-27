"""
Tradix Correlation Matrix Module.

Analyzes correlation between multiple strategy backtest results, providing
correlation matrices, rolling correlations, diversification ratios, optimal
combination selection, and strategy clustering.

다수 전략 백테스트 결과 간의 상관관계를 분석하는 모듈입니다.
상관행렬, 롤링 상관관계, 분산 비율, 최적 조합 선택, 전략 군집화를
제공합니다.

Features:
    - Pearson / Spearman / Kendall correlation matrix computation
    - Statistical significance (p-value) testing
    - Rolling (time-varying) correlation analysis
    - Portfolio diversification ratio calculation
    - Optimal low-correlation strategy subset selection
    - K-means strategy clustering (numpy-only, no sklearn)

Usage:
    from tradix.analytics.correlation import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    analyzer.addResult("SMA Cross", result1)
    analyzer.addResult("RSI Mean Reversion", result2)
    analyzer.addResult("Breakout", result3)

    corr = analyzer.correlationMatrix()
    print(corr.summary())
    rolling = analyzer.rollingCorrelation(window=60)
    clusters = analyzer.clusterStrategies(nClusters=2)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats as scipyStats

from tradix.engine import BacktestResult


@dataclass
class CorrelationResult:
    """
    Container for correlation analysis results.

    상관관계 분석 결과를 저장하는 컨테이너.

    Holds the full correlation matrix, p-value matrix for statistical
    significance, cluster assignments from K-means analysis, and the
    portfolio diversification ratio.

    상관행렬, 통계적 유의성을 위한 p-값 행렬, K-means 분석의 군집 할당,
    포트폴리오 분산 비율을 보관합니다.

    Attributes:
        matrix (pd.DataFrame): Strategy-by-strategy correlation matrix.
            전략 간 상관행렬.
        pValues (pd.DataFrame): Statistical significance p-values.
            통계적 유의성 p-값 행렬.
        clustered (dict): Cluster label assignments per strategy.
            전략별 군집 레이블 할당.
        diversificationRatio (float): Portfolio diversification ratio.
            포트폴리오 분산 비율.
    """
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    pValues: pd.DataFrame = field(default_factory=pd.DataFrame)
    clustered: dict = field(default_factory=dict)
    diversificationRatio: float = 0.0

    def summary(self) -> str:
        """
        Generate a Korean-language summary of correlation analysis results.

        상관관계 분석 결과의 한국어 요약을 생성합니다.

        Returns:
            str: Formatted multi-line summary string.
        """
        nStrategies = len(self.matrix)
        if nStrategies == 0:
            return "분석된 전략이 없습니다."

        upperTriangle = self.matrix.values[np.triu_indices(nStrategies, k=1)]
        avgCorr = np.mean(upperTriangle) if len(upperTriangle) > 0 else 0.0
        minCorr = np.min(upperTriangle) if len(upperTriangle) > 0 else 0.0
        maxCorr = np.max(upperTriangle) if len(upperTriangle) > 0 else 0.0

        significantCount = 0
        if not self.pValues.empty:
            upperPValues = self.pValues.values[np.triu_indices(nStrategies, k=1)]
            significantCount = int(np.sum(upperPValues < 0.05))

        totalPairs = len(upperTriangle)

        lines = [
            f"{'='*50}",
            f"상관관계 분석 요약",
            f"{'='*50}",
            f"분석 전략 수: {nStrategies}개",
            f"전략 쌍 수: {totalPairs}개",
            f"{'─'*50}",
            f"평균 상관계수: {avgCorr:.4f}",
            f"최소 상관계수: {minCorr:.4f}",
            f"최대 상관계수: {maxCorr:.4f}",
            f"유의미한 쌍 (p<0.05): {significantCount}/{totalPairs}",
            f"분산 비율: {self.diversificationRatio:.4f}",
        ]

        if self.clustered:
            lines.append(f"{'─'*50}")
            lines.append("군집 할당:")
            clusterGroups: Dict[int, List[str]] = {}
            for name, label in self.clustered.items():
                clusterGroups.setdefault(label, []).append(name)
            for label in sorted(clusterGroups.keys()):
                members = ", ".join(clusterGroups[label])
                lines.append(f"  군집 {label}: {members}")

        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        nStrategies = len(self.matrix)
        return (
            f"CorrelationResult(strategies={nStrategies}, "
            f"diversificationRatio={self.diversificationRatio:.4f})"
        )


class CorrelationAnalyzer:
    """
    Multi-strategy correlation analyzer.

    다수 전략 간 상관관계를 분석하는 클래스.

    Collects equity curves from multiple backtest results and provides
    correlation matrix computation, rolling correlation analysis,
    diversification scoring, optimal subset selection, and clustering.

    다수 백테스트 결과의 자산 곡선을 수집하고, 상관행렬 계산, 롤링 상관관계
    분석, 분산 점수, 최적 부분집합 선택, 군집화를 제공합니다.

    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> analyzer.addResult("SMA", smaResult)
        >>> analyzer.addResult("RSI", rsiResult)
        >>> corr = analyzer.correlationMatrix(method='spearman')
        >>> print(corr.summary())
    """

    def __init__(self):
        """
        Initialize the analyzer with an empty results collection.

        빈 결과 컬렉션으로 분석기를 초기화합니다.
        """
        self._curves: Dict[str, pd.Series] = {}

    def addResult(self, name: str, result: BacktestResult) -> None:
        """
        Add a strategy backtest result for correlation analysis.

        상관관계 분석을 위해 전략 백테스트 결과를 추가합니다.

        Args:
            name (str): Strategy identifier name. 전략 식별 이름.
            result (BacktestResult): Backtest result containing an equity curve.
                자산 곡선이 포함된 백테스트 결과.
        """
        self._curves[name] = result.equityCurve.copy()

    def addEquityCurve(self, name: str, curve: pd.Series) -> None:
        """
        Add a raw equity curve for correlation analysis.

        상관관계 분석을 위해 원시 자산 곡선을 추가합니다.

        Args:
            name (str): Strategy identifier name. 전략 식별 이름.
            curve (pd.Series): Equity curve with DatetimeIndex.
                DatetimeIndex를 가진 자산 곡선.
        """
        self._curves[name] = curve.copy()

    def _alignedReturns(self) -> pd.DataFrame:
        """
        Align all equity curves by date and compute daily returns.

        모든 자산 곡선을 날짜 기준으로 정렬하고 일별 수익률을 계산합니다.

        Returns:
            pd.DataFrame: Date-aligned daily returns with strategy names as columns.
                전략 이름을 열로 가진 날짜 정렬 일별 수익률.
        """
        if len(self._curves) < 2:
            return pd.DataFrame()

        combined = pd.DataFrame(self._curves)
        combined = combined.dropna(how="all")
        combined = combined.ffill()
        combined = combined.dropna()
        returns = combined.pct_change().dropna()
        return returns

    def correlationMatrix(self, method: str = "pearson") -> CorrelationResult:
        """
        Compute the correlation matrix across all added strategies.

        추가된 모든 전략에 대해 상관행렬을 계산합니다.

        Aligns equity curves by date, computes daily returns, and calculates
        the correlation matrix using the specified method. Also computes
        p-values for statistical significance of each pair.

        자산 곡선을 날짜별로 정렬하고, 일별 수익률을 계산한 뒤, 지정된 방법으로
        상관행렬을 계산합니다. 각 쌍에 대한 통계적 유의성 p-값도 계산합니다.

        Args:
            method (str): Correlation method - 'pearson', 'spearman', or 'kendall'.
                상관계수 방법 - 'pearson', 'spearman', 또는 'kendall'.

        Returns:
            CorrelationResult: Complete correlation analysis result.
                상관관계 분석 전체 결과.
        """
        returns = self._alignedReturns()
        if returns.empty:
            return CorrelationResult()

        corrMatrix = returns.corr(method=method)

        names = list(returns.columns)
        nStrategies = len(names)
        pValueMatrix = pd.DataFrame(
            np.ones((nStrategies, nStrategies)),
            index=names,
            columns=names
        )

        corrFuncMap = {
            "pearson": scipyStats.pearsonr,
            "spearman": scipyStats.spearmanr,
            "kendall": scipyStats.kendalltau,
        }
        corrFunc = corrFuncMap.get(method, scipyStats.pearsonr)

        for i in range(nStrategies):
            for j in range(i + 1, nStrategies):
                seriesA = returns.iloc[:, i].values
                seriesB = returns.iloc[:, j].values
                _, pValue = corrFunc(seriesA, seriesB)
                pValueMatrix.iloc[i, j] = pValue
                pValueMatrix.iloc[j, i] = pValue
            pValueMatrix.iloc[i, i] = 0.0

        divRatio = self.diversificationRatio()

        return CorrelationResult(
            matrix=corrMatrix,
            pValues=pValueMatrix,
            clustered={},
            diversificationRatio=divRatio,
        )

    def rollingCorrelation(self, window: int = 60) -> pd.DataFrame:
        """
        Compute time-varying rolling correlation between all strategy pairs.

        모든 전략 쌍 간의 시간 변동 롤링 상관관계를 계산합니다.

        For each pair of strategies, calculates the rolling Pearson correlation
        over the specified window. Returns a DataFrame with one column per pair.

        각 전략 쌍에 대해 지정된 윈도우 크기로 롤링 피어슨 상관계수를
        계산합니다. 쌍당 하나의 열을 가진 DataFrame을 반환합니다.

        Args:
            window (int): Rolling window size in trading days. Default 60.
                롤링 윈도우 크기 (거래일). 기본값 60.

        Returns:
            pd.DataFrame: Rolling correlations with pair labels as columns.
                쌍 레이블을 열로 가진 롤링 상관계수 DataFrame.
        """
        returns = self._alignedReturns()
        if returns.empty:
            return pd.DataFrame()

        names = list(returns.columns)
        rollingResults = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairLabel = f"{names[i]} vs {names[j]}"
                rollingCorr = returns.iloc[:, i].rolling(window).corr(returns.iloc[:, j])
                rollingResults[pairLabel] = rollingCorr

        return pd.DataFrame(rollingResults)

    def diversificationRatio(self, weights: np.ndarray = None) -> float:
        """
        Calculate the portfolio diversification ratio.

        포트폴리오 분산 비율을 계산합니다.

        The diversification ratio is defined as the weighted average of
        individual volatilities divided by the portfolio volatility.
        A higher ratio indicates better diversification.

        분산 비율은 개별 변동성의 가중 평균을 포트폴리오 변동성으로
        나눈 값으로 정의됩니다. 높을수록 분산이 잘 되어 있음을 의미합니다.

        Args:
            weights (np.ndarray, optional): Strategy weight vector. If None,
                equal weights are used. 전략 가중치 벡터. None이면 동일 가중치 사용.

        Returns:
            float: Diversification ratio (>= 1.0 for diversified portfolios).
                분산 비율 (분산된 포트폴리오의 경우 1.0 이상).
        """
        returns = self._alignedReturns()
        if returns.empty or returns.shape[1] < 2:
            return 1.0

        nStrategies = returns.shape[1]
        if weights is None:
            weights = np.ones(nStrategies) / nStrategies
        else:
            weights = np.array(weights, dtype=float)
            weightSum = np.sum(weights)
            if weightSum > 0:
                weights = weights / weightSum

        individualVols = returns.std().values * np.sqrt(252)
        weightedAvgVol = np.dot(weights, individualVols)

        covMatrix = returns.cov().values * 252
        portfolioVariance = np.dot(weights, np.dot(covMatrix, weights))
        portfolioVol = np.sqrt(max(portfolioVariance, 0.0))

        if portfolioVol == 0:
            return 1.0

        return float(weightedAvgVol / portfolioVol)

    def optimalCombination(self, targetCorrelation: float = 0.3) -> List[str]:
        """
        Find the least-correlated subset of strategies.

        가장 상관관계가 낮은 전략 부분집합을 찾습니다.

        Uses a greedy algorithm: starts with the strategy that has the lowest
        average correlation, then iteratively adds the strategy with the lowest
        maximum correlation to any already-selected strategy, as long as it
        stays below the target correlation threshold.

        그리디 알고리즘을 사용합니다: 평균 상관계수가 가장 낮은 전략부터
        시작하여, 이미 선택된 전략과의 최대 상관계수가 목표 임계값 이하인
        전략을 반복적으로 추가합니다.

        Args:
            targetCorrelation (float): Maximum allowed pairwise correlation.
                Default 0.3. 허용 최대 쌍별 상관계수. 기본값 0.3.

        Returns:
            List[str]: Ordered list of selected strategy names.
                선택된 전략 이름의 정렬된 목록.
        """
        returns = self._alignedReturns()
        if returns.empty:
            return []

        corrMatrix = returns.corr().values
        names = list(returns.columns)
        nStrategies = len(names)

        if nStrategies <= 1:
            return names

        avgCorrs = np.zeros(nStrategies)
        for i in range(nStrategies):
            otherCorrs = [abs(corrMatrix[i, j]) for j in range(nStrategies) if i != j]
            avgCorrs[i] = np.mean(otherCorrs)

        selected = [int(np.argmin(avgCorrs))]
        remaining = set(range(nStrategies)) - set(selected)

        while remaining:
            bestCandidate = None
            bestMaxCorr = float("inf")

            for candidate in remaining:
                maxCorrToSelected = max(abs(corrMatrix[candidate, s]) for s in selected)
                if maxCorrToSelected < bestMaxCorr:
                    bestMaxCorr = maxCorrToSelected
                    bestCandidate = candidate

            if bestCandidate is None or bestMaxCorr > targetCorrelation:
                break

            selected.append(bestCandidate)
            remaining.remove(bestCandidate)

        return [names[i] for i in selected]

    def clusterStrategies(self, nClusters: int = 3) -> dict:
        """
        Cluster strategies using K-means on return correlations.

        수익률 상관관계를 기반으로 K-means 군집화를 수행합니다.

        Implements iterative K-means using numpy only (no sklearn dependency).
        Uses the correlation matrix as the feature space for clustering.
        Runs multiple initializations and selects the result with the lowest
        total within-cluster sum-of-squares.

        numpy만 사용한 반복 K-means를 구현합니다 (sklearn 의존성 없음).
        상관행렬을 군집화의 특징 공간으로 사용합니다. 다중 초기화를 수행하고
        클러스터 내 제곱합이 가장 낮은 결과를 선택합니다.

        Args:
            nClusters (int): Number of clusters to form. Default 3.
                형성할 군집 수. 기본값 3.

        Returns:
            dict: Mapping of strategy name to cluster label (int).
                전략 이름에서 군집 레이블(int)로의 매핑.
        """
        returns = self._alignedReturns()
        if returns.empty:
            return {}

        names = list(returns.columns)
        nStrategies = len(names)
        nClusters = min(nClusters, nStrategies)

        if nClusters <= 1:
            return {name: 0 for name in names}

        corrMatrix = returns.corr().values
        features = corrMatrix.copy()

        bestLabels = None
        bestInertia = float("inf")
        maxIterations = 100
        nInitializations = 10

        rng = np.random.RandomState(42)

        for _ in range(nInitializations):
            indices = rng.choice(nStrategies, size=nClusters, replace=False)
            centroids = features[indices].copy()

            labels = np.zeros(nStrategies, dtype=int)

            for _ in range(maxIterations):
                distances = np.zeros((nStrategies, nClusters))
                for k in range(nClusters):
                    diff = features - centroids[k]
                    distances[:, k] = np.sum(diff ** 2, axis=1)

                newLabels = np.argmin(distances, axis=1)

                if np.array_equal(newLabels, labels):
                    break
                labels = newLabels

                for k in range(nClusters):
                    members = features[labels == k]
                    if len(members) > 0:
                        centroids[k] = members.mean(axis=0)

            inertia = 0.0
            for k in range(nClusters):
                members = features[labels == k]
                if len(members) > 0:
                    diff = members - centroids[k]
                    inertia += np.sum(diff ** 2)

            if inertia < bestInertia:
                bestInertia = inertia
                bestLabels = labels.copy()

        return {names[i]: int(bestLabels[i]) for i in range(nStrategies)}

    def summary(self) -> str:
        """
        Generate a Korean-language summary of the current analyzer state.

        현재 분석기 상태의 한국어 요약을 생성합니다.

        Returns:
            str: Formatted summary including strategy count and names.
                전략 수와 이름이 포함된 포맷된 요약.
        """
        names = list(self._curves.keys())
        nStrategies = len(names)

        if nStrategies == 0:
            return "등록된 전략이 없습니다."

        lines = [
            f"{'='*50}",
            f"상관관계 분석기 상태",
            f"{'='*50}",
            f"등록 전략 수: {nStrategies}개",
            f"전략 목록: {', '.join(names)}",
        ]

        if nStrategies >= 2:
            returns = self._alignedReturns()
            if not returns.empty:
                corrMatrix = returns.corr()
                upperIdx = np.triu_indices(nStrategies, k=1)
                upperCorrs = corrMatrix.values[upperIdx]
                lines.append(f"{'─'*50}")
                lines.append(f"평균 상관계수: {np.mean(upperCorrs):.4f}")
                lines.append(f"상관계수 범위: [{np.min(upperCorrs):.4f}, {np.max(upperCorrs):.4f}]")

        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CorrelationAnalyzer(strategies={len(self._curves)})"
