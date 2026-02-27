"""
Market Regime Detector - Gaussian Mixture Model based market regime detection.
시장 레짐 감지기 - 가우시안 혼합 모델 기반 시장 레짐 감지.

Detects market regimes (Bull, Bear, Sideways) from equity curve returns using
a from-scratch Expectation-Maximization (EM) algorithm for Gaussian Mixture
Models. Decomposes strategy performance by regime and computes regime
transition probabilities.

자산 곡선 수익률로부터 가우시안 혼합 모델의 EM 알고리즘을 직접 구현하여
시장 레짐(상승장, 하락장, 횡보장)을 감지합니다. 레짐별 전략 성과를 분해하고
레짐 전환 확률을 계산합니다.

Features:
    - Gaussian Mixture Model fitted via EM algorithm (no sklearn dependency)
    - Automatic regime labeling based on mean return ranking
    - Regime-conditional performance decomposition (return, Sharpe, win rate)
    - Transition probability matrix between regimes
    - Average regime duration analysis
    - Bilingual summary (Korean / English)

Usage:
    from tradix.analytics.regimeDetector import RegimeDetector
    from tradix.engine import BacktestResult

    detector = RegimeDetector()
    analysis = detector.analyze(result, nRegimes=3)
    print(analysis.summary())
    print(analysis.summary(ko=True))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade


REGIME_LABELS = {
    "bull": "Bull",
    "bear": "Bear",
    "sideways": "Sideways",
}

REGIME_LABELS_KO = {
    "bull": "상승장",
    "bear": "하락장",
    "sideways": "횡보장",
}

MIN_DATA_POINTS = 30
DEFAULT_N_REGIMES = 3
MAX_EM_ITERATIONS = 100
EM_TOLERANCE = 1e-6
COV_REGULARIZATION = 1e-6
ROLLING_VOL_WINDOW = 20
ANNUALIZATION_FACTOR = 252


@dataclass
class RegimeAnalysisResult:
    """
    Result container for market regime detection analysis.
    시장 레짐 감지 분석 결과 컨테이너.

    Encapsulates detected regime labels, regime-conditional performance metrics,
    transition probabilities, and duration statistics derived from a Gaussian
    Mixture Model fit on equity curve features.

    가우시안 혼합 모델을 자산 곡선 특징에 적합시켜 도출한 레짐 레이블,
    레짐별 조건부 성과 지표, 전환 확률, 지속 기간 통계를 캡슐화합니다.

    Attributes:
        regimes (List[str]): Detected regime label per day.
            일별 감지된 레짐 레이블.
        regimeReturns (Dict[str, float]): Average strategy return per regime.
            레짐별 평균 전략 수익률.
        transitionMatrix (Dict[str, Dict[str, float]]): Regime transition probabilities.
            레짐 전환 확률 행렬.
        currentRegime (str): Last detected regime.
            마지막으로 감지된 레짐.
        regimeDurations (Dict[str, float]): Average duration in days per regime.
            레짐별 평균 지속 기간 (일).
        regimeSharpe (Dict[str, float]): Sharpe ratio per regime.
            레짐별 샤프 비율.
        regimeDistribution (Dict[str, float]): Fraction of time in each regime.
            각 레짐의 시간 비율.
        details (Dict): Detailed analysis metadata.
            상세 분석 메타데이터.

    Example:
        >>> detector = RegimeDetector()
        >>> analysis = detector.analyze(backtestResult)
        >>> print(analysis.currentRegime)
        'bull'
        >>> print(analysis.transitionMatrix)
    """
    regimes: List[str] = field(default_factory=list)
    regimeReturns: Dict[str, float] = field(default_factory=dict)
    transitionMatrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    currentRegime: str = "sideways"
    regimeDurations: Dict[str, float] = field(default_factory=dict)
    regimeSharpe: Dict[str, float] = field(default_factory=dict)
    regimeDistribution: Dict[str, float] = field(default_factory=dict)
    details: Dict = field(default_factory=dict)

    def summary(self, ko: bool = False) -> str:
        """
        Generate a formatted summary of regime analysis results.
        레짐 분석 결과의 포맷된 요약을 생성합니다.

        Args:
            ko (bool): If True, produce Korean-language summary.
                True이면 한국어 요약을 생성합니다.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 포맷된 요약 문자열.
        """
        labels = REGIME_LABELS_KO if ko else REGIME_LABELS

        if ko:
            title = "시장 레짐 분석 (Market Regime Analysis)"
            currentLabel = "현재 레짐"
            distributionLabel = "레짐 분포"
            returnsLabel = "레짐별 평균 수익률"
            sharpeLabel = "레짐별 샤프 비율"
            durationLabel = "레짐별 평균 지속 기간"
            transitionLabel = "레짐 전환 확률 행렬"
            daysUnit = "일"
        else:
            title = "Market Regime Analysis"
            currentLabel = "Current Regime"
            distributionLabel = "Regime Distribution"
            returnsLabel = "Average Return by Regime"
            sharpeLabel = "Sharpe Ratio by Regime"
            durationLabel = "Average Duration by Regime"
            transitionLabel = "Regime Transition Matrix"
            daysUnit = "days"

        currentRegimeDisplay = labels.get(self.currentRegime, self.currentRegime)

        lines = [
            "",
            "=" * 55,
            "  " + title,
            "=" * 55,
            "  {0}: {1}".format(currentLabel, currentRegimeDisplay),
            "\u2500" * 55,
        ]

        lines.append("  " + distributionLabel + ":")
        for regime, fraction in sorted(self.regimeDistribution.items()):
            regimeDisplay = labels.get(regime, regime)
            barLen = int(fraction * 30)
            bar = "\u2588" * barLen + "\u2591" * (30 - barLen)
            lines.append("    {0:<10s} {1} {2:.1f}%".format(
                regimeDisplay, bar, fraction * 100
            ))

        lines.append("\u2500" * 55)
        lines.append("  " + returnsLabel + ":")
        for regime, ret in sorted(self.regimeReturns.items()):
            regimeDisplay = labels.get(regime, regime)
            lines.append("    {0:<10s} {1:+.4f}".format(regimeDisplay, ret))

        lines.append("\u2500" * 55)
        lines.append("  " + sharpeLabel + ":")
        for regime, sharpe in sorted(self.regimeSharpe.items()):
            regimeDisplay = labels.get(regime, regime)
            lines.append("    {0:<10s} {1:+.2f}".format(regimeDisplay, sharpe))

        lines.append("\u2500" * 55)
        lines.append("  " + durationLabel + ":")
        for regime, dur in sorted(self.regimeDurations.items()):
            regimeDisplay = labels.get(regime, regime)
            lines.append("    {0:<10s} {1:.1f} {2}".format(
                regimeDisplay, dur, daysUnit
            ))

        lines.append("\u2500" * 55)
        lines.append("  " + transitionLabel + ":")
        regimeOrder = sorted(self.transitionMatrix.keys())
        headerCells = [labels.get(r, r) for r in regimeOrder]
        lines.append("    {0:<10s} {1}".format(
            "", "  ".join("{0:>8s}".format(h) for h in headerCells)
        ))
        for fromRegime in regimeOrder:
            fromDisplay = labels.get(fromRegime, fromRegime)
            row = self.transitionMatrix.get(fromRegime, {})
            cells = []
            for toRegime in regimeOrder:
                prob = row.get(toRegime, 0.0)
                cells.append("{0:>8.1f}%".format(prob * 100))
            lines.append("    {0:<10s} {1}".format(fromDisplay, "  ".join(cells)))

        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        regimeCounts = {}
        for r in self.regimes:
            regimeCounts[r] = regimeCounts.get(r, 0) + 1
        return (
            "RegimeAnalysisResult(current={0}, days={1}, distribution={2})".format(
                self.currentRegime, len(self.regimes), regimeCounts
            )
        )


class RegimeDetector:
    """
    Market regime detector using Gaussian Mixture Models fitted via EM algorithm.
    EM 알고리즘으로 적합된 가우시안 혼합 모델을 사용하는 시장 레짐 감지기.

    Extracts features (daily returns, rolling volatility) from the equity curve,
    fits a Gaussian Mixture Model using the Expectation-Maximization algorithm
    implemented from scratch (no sklearn dependency), and classifies each
    trading day into a market regime (bull, bear, or sideways).

    자산 곡선에서 특징(일별 수익률, 롤링 변동성)을 추출하고, 직접 구현한
    EM 알고리즘으로 가우시안 혼합 모델을 적합시켜 각 거래일을 시장
    레짐(상승장, 하락장, 횡보장)으로 분류합니다.

    Features:
        - From-scratch GMM with EM algorithm (E-step, M-step)
        - K-means++ style initialization for robust convergence
        - Covariance regularization for numerical stability
        - Regime-conditional Sharpe ratio and win rate analysis
        - Markov transition probability matrix
        - Average regime duration computation

    기능:
        - 직접 구현한 GMM + EM 알고리즘 (E-step, M-step)
        - 안정적 수렴을 위한 K-means++ 스타일 초기화
        - 수치 안정성을 위한 공분산 정규화
        - 레짐별 조건부 샤프 비율 및 승률 분석
        - 마르코프 전환 확률 행렬
        - 평균 레짐 지속 기간 계산

    Example:
        >>> detector = RegimeDetector()
        >>> analysis = detector.analyze(backtestResult, nRegimes=3)
        >>> print(analysis.summary(ko=True))
        >>> print(analysis.currentRegime)
    """

    def analyze(
        self,
        result: BacktestResult,
        nRegimes: int = DEFAULT_N_REGIMES,
    ) -> RegimeAnalysisResult:
        """
        Perform market regime detection on a backtest result.
        백테스트 결과에 대해 시장 레짐 감지를 수행합니다.

        Extracts daily returns and rolling volatility from the equity curve,
        fits a Gaussian Mixture Model via EM, labels regimes by mean return
        ranking, and computes regime-conditional performance metrics.

        자산 곡선에서 일별 수익률과 롤링 변동성을 추출하고, EM으로 가우시안
        혼합 모델을 적합시키고, 평균 수익률 순위로 레짐에 레이블을 부여한 뒤,
        레짐별 조건부 성과 지표를 계산합니다.

        Args:
            result (BacktestResult): Completed backtest result containing
                equity curve and trade history.
                자산 곡선과 거래 내역이 포함된 백테스트 결과.
            nRegimes (int): Number of regimes to detect.
                감지할 레짐 수. Defaults to 3.

        Returns:
            RegimeAnalysisResult: Complete regime analysis results.
                완전한 레짐 분석 결과.
        """
        equityCurve = result.equityCurve
        if equityCurve is None or len(equityCurve) < MIN_DATA_POINTS:
            return self._defaultResult(nRegimes)

        returns = equityCurve.pct_change().dropna()
        if len(returns) < MIN_DATA_POINTS:
            return self._defaultResult(nRegimes)

        rollingVol = returns.rolling(
            window=ROLLING_VOL_WINDOW, min_periods=max(5, ROLLING_VOL_WINDOW // 2)
        ).std()

        validMask = rollingVol.notna()
        validReturns = returns[validMask]
        validVol = rollingVol[validMask]

        if len(validReturns) < MIN_DATA_POINTS:
            return self._defaultResult(nRegimes)

        features = np.column_stack([
            validReturns.values,
            validVol.values,
        ])

        means, covs, weights, componentLabels = self._fitGMM(
            features, nRegimes
        )

        regimeMapping = self._classifyRegimes(means)

        regimeLabels = [regimeMapping[c] for c in componentLabels]

        transMatrix = self._buildTransitionMatrix(regimeLabels, regimeMapping)

        regimeReturnsSeries = pd.Series(
            validReturns.values, index=validReturns.index
        )
        performance = self._regimePerformance(
            regimeReturnsSeries, regimeLabels, regimeMapping
        )

        durations = self._regimeDurations(regimeLabels, regimeMapping)

        regimeDistribution = {}
        totalDays = len(regimeLabels)
        for regimeName in set(regimeMapping.values()):
            count = sum(1 for r in regimeLabels if r == regimeName)
            regimeDistribution[regimeName] = count / totalDays if totalDays > 0 else 0.0

        currentRegime = regimeLabels[-1] if regimeLabels else "sideways"

        componentMeans = {}
        componentCovs = {}
        componentWeights = {}
        for compIdx, regimeName in regimeMapping.items():
            componentMeans[regimeName] = means[compIdx].tolist()
            componentCovs[regimeName] = covs[compIdx].tolist()
            componentWeights[regimeName] = float(weights[compIdx])

        details = {
            "nRegimes": nRegimes,
            "nDataPoints": len(features),
            "componentMeans": componentMeans,
            "componentCovariances": componentCovs,
            "componentWeights": componentWeights,
            "featureNames": ["dailyReturn", "rollingVolatility"],
        }

        return RegimeAnalysisResult(
            regimes=regimeLabels,
            regimeReturns=performance["regimeReturns"],
            transitionMatrix=transMatrix,
            currentRegime=currentRegime,
            regimeDurations=durations,
            regimeSharpe=performance["regimeSharpe"],
            regimeDistribution=regimeDistribution,
            details=details,
        )

    def _defaultResult(self, nRegimes: int) -> RegimeAnalysisResult:
        """
        Return a default result when insufficient data is available.
        데이터가 부족할 때 기본 결과를 반환합니다.

        Args:
            nRegimes (int): Number of requested regimes.
                요청된 레짐 수.

        Returns:
            RegimeAnalysisResult: Default result with empty or neutral values.
                빈 값 또는 중립값을 가진 기본 결과.
        """
        defaultRegimes = ["bull", "bear", "sideways"][:nRegimes]
        emptyReturns = {r: 0.0 for r in defaultRegimes}
        emptySharpe = {r: 0.0 for r in defaultRegimes}
        emptyDurations = {r: 0.0 for r in defaultRegimes}
        emptyDistribution = {r: 1.0 / nRegimes for r in defaultRegimes}
        emptyTransition = {
            r: {r2: 1.0 / nRegimes for r2 in defaultRegimes}
            for r in defaultRegimes
        }

        return RegimeAnalysisResult(
            regimes=[],
            regimeReturns=emptyReturns,
            transitionMatrix=emptyTransition,
            currentRegime="sideways",
            regimeDurations=emptyDurations,
            regimeSharpe=emptySharpe,
            regimeDistribution=emptyDistribution,
            details={"error": "insufficient_data", "nRegimes": nRegimes},
        )

    def _fitGMM(
        self,
        data: np.ndarray,
        nComponents: int,
        maxIter: int = MAX_EM_ITERATIONS,
        tol: float = EM_TOLERANCE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a Gaussian Mixture Model to data using the EM algorithm.
        EM 알고리즘을 사용하여 데이터에 가우시안 혼합 모델을 적합시킵니다.

        Implements the full Expectation-Maximization procedure from scratch:
        initialization via k-means++ style seeding, iterative E-step (compute
        responsibilities) and M-step (update parameters), with convergence
        checking on log-likelihood.

        K-means++ 스타일 시딩을 통한 초기화, 반복적 E-step(책임 계산) 및
        M-step(파라미터 갱신), 로그 가능도 기반 수렴 확인을 포함한
        완전한 EM 절차를 직접 구현합니다.

        Args:
            data (np.ndarray): Feature matrix of shape (n_samples, n_features).
                (샘플 수, 특징 수) 형태의 특징 행렬.
            nComponents (int): Number of Gaussian components (regimes).
                가우시안 구성 요소(레짐) 수.
            maxIter (int): Maximum EM iterations. 최대 EM 반복 수.
            tol (float): Convergence tolerance for log-likelihood change.
                로그 가능도 변화 수렴 허용오차.

        Returns:
            Tuple of (means, covariances, weights, labels):
                - means (np.ndarray): Shape (nComponents, nFeatures) component means.
                - covariances (np.ndarray): Shape (nComponents, nFeatures, nFeatures).
                - weights (np.ndarray): Shape (nComponents,) mixing weights.
                - labels (np.ndarray): Shape (nSamples,) hard assignments.
        """
        nSamples, nFeatures = data.shape

        nComponents = min(nComponents, nSamples)

        means = self._initializeMeansKMeansPP(data, nComponents)

        covs = np.array([
            np.cov(data, rowvar=False) + np.eye(nFeatures) * COV_REGULARIZATION
            for _ in range(nComponents)
        ])

        weights = np.ones(nComponents) / nComponents

        prevLogLikelihood = -np.inf

        for iteration in range(maxIter):
            responsibilities = self._emStep(data, means, covs, weights)

            effectiveN = responsibilities.sum(axis=0)

            for k in range(nComponents):
                nk = effectiveN[k]
                if nk < 1e-10:
                    means[k] = data[np.random.randint(0, nSamples)]
                    covs[k] = np.cov(data, rowvar=False) + np.eye(nFeatures) * COV_REGULARIZATION
                    weights[k] = 1.0 / nComponents
                    continue

                means[k] = np.dot(responsibilities[:, k], data) / nk

                diff = data - means[k]
                weightedDiff = diff * responsibilities[:, k:k + 1]
                covs[k] = np.dot(weightedDiff.T, diff) / nk
                covs[k] += np.eye(nFeatures) * COV_REGULARIZATION

                weights[k] = nk / nSamples

            weightSum = weights.sum()
            if weightSum > 0:
                weights = weights / weightSum

            logLikelihood = self._computeLogLikelihood(data, means, covs, weights)

            if abs(logLikelihood - prevLogLikelihood) < tol:
                break

            prevLogLikelihood = logLikelihood

        finalResponsibilities = self._emStep(data, means, covs, weights)
        labels = np.argmax(finalResponsibilities, axis=1)

        return means, covs, weights, labels

    def _initializeMeansKMeansPP(
        self,
        data: np.ndarray,
        nComponents: int,
    ) -> np.ndarray:
        """
        Initialize GMM means using k-means++ style initialization.
        K-means++ 스타일 초기화로 GMM 평균을 초기화합니다.

        Selects the first center uniformly at random, then chooses subsequent
        centers with probability proportional to squared distance from the
        nearest existing center, promoting well-spread initial means.

        첫 번째 중심을 균등 랜덤으로 선택한 후, 기존 가장 가까운 중심과의
        제곱 거리에 비례하는 확률로 후속 중심을 선택하여
        잘 분산된 초기 평균을 촉진합니다.

        Args:
            data (np.ndarray): Feature matrix of shape (n_samples, n_features).
                (샘플 수, 특징 수) 형태의 특징 행렬.
            nComponents (int): Number of centers to initialize.
                초기화할 중심 수.

        Returns:
            np.ndarray: Shape (nComponents, nFeatures) initialized means.
                (구성 요소 수, 특징 수) 형태의 초기화된 평균.
        """
        nSamples, nFeatures = data.shape
        rng = np.random.RandomState(42)

        centers = np.zeros((nComponents, nFeatures))
        firstIdx = rng.randint(0, nSamples)
        centers[0] = data[firstIdx]

        for k in range(1, nComponents):
            distances = np.full(nSamples, np.inf)
            for j in range(k):
                dist = np.sum((data - centers[j]) ** 2, axis=1)
                distances = np.minimum(distances, dist)

            distSum = distances.sum()
            if distSum <= 0:
                centers[k] = data[rng.randint(0, nSamples)]
            else:
                probabilities = distances / distSum
                cumulativeProbs = np.cumsum(probabilities)
                r = rng.random()
                selectedIdx = np.searchsorted(cumulativeProbs, r)
                selectedIdx = min(selectedIdx, nSamples - 1)
                centers[k] = data[selectedIdx]

        return centers

    def _emStep(
        self,
        data: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Perform the E-step of the EM algorithm (compute responsibilities).
        EM 알고리즘의 E-step(책임 계산)을 수행합니다.

        For each data point, computes the posterior probability that it belongs
        to each Gaussian component using Bayes' theorem.

        각 데이터 포인트에 대해 베이즈 정리를 사용하여 각 가우시안 구성 요소에
        속할 사후 확률을 계산합니다.

        Args:
            data (np.ndarray): Feature matrix of shape (n_samples, n_features).
                (샘플 수, 특징 수) 형태의 특징 행렬.
            means (np.ndarray): Component means, shape (nComponents, nFeatures).
                구성 요소 평균.
            covs (np.ndarray): Component covariances, shape (nComponents, nFeatures, nFeatures).
                구성 요소 공분산.
            weights (np.ndarray): Mixing weights, shape (nComponents,).
                혼합 가중치.

        Returns:
            np.ndarray: Responsibility matrix of shape (n_samples, nComponents).
                (샘플 수, 구성 요소 수) 형태의 책임 행렬.
        """
        nSamples = data.shape[0]
        nComponents = len(weights)
        nFeatures = data.shape[1]

        responsibilities = np.zeros((nSamples, nComponents))

        for k in range(nComponents):
            cov = covs[k]
            covRegularized = cov + np.eye(nFeatures) * COV_REGULARIZATION

            pdf = multivariate_normal.pdf(
                data, mean=means[k], cov=covRegularized, allow_singular=True
            )
            responsibilities[:, k] = weights[k] * pdf

        rowSums = responsibilities.sum(axis=1, keepdims=True)
        rowSums = np.maximum(rowSums, 1e-300)
        responsibilities = responsibilities / rowSums

        return responsibilities

    def _computeLogLikelihood(
        self,
        data: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute the log-likelihood of the data under the current GMM parameters.
        현재 GMM 파라미터 하에서 데이터의 로그 가능도를 계산합니다.

        Args:
            data (np.ndarray): Feature matrix of shape (n_samples, n_features).
                (샘플 수, 특징 수) 형태의 특징 행렬.
            means (np.ndarray): Component means. 구성 요소 평균.
            covs (np.ndarray): Component covariances. 구성 요소 공분산.
            weights (np.ndarray): Mixing weights. 혼합 가중치.

        Returns:
            float: Total log-likelihood. 총 로그 가능도.
        """
        nSamples = data.shape[0]
        nComponents = len(weights)
        nFeatures = data.shape[1]

        likelihood = np.zeros(nSamples)

        for k in range(nComponents):
            covRegularized = covs[k] + np.eye(nFeatures) * COV_REGULARIZATION
            pdf = multivariate_normal.pdf(
                data, mean=means[k], cov=covRegularized, allow_singular=True
            )
            likelihood += weights[k] * pdf

        likelihood = np.maximum(likelihood, 1e-300)
        return float(np.sum(np.log(likelihood)))

    def _classifyRegimes(
        self,
        means: np.ndarray,
    ) -> Dict[int, str]:
        """
        Classify GMM components into regime labels based on mean return ranking.
        평균 수익률 순위에 기반하여 GMM 구성 요소를 레짐 레이블로 분류합니다.

        The component with the highest mean return (first feature dimension)
        is labeled 'bull', the lowest 'bear', and any remaining are 'sideways'.

        첫 번째 특징 차원(수익률)의 평균이 가장 높은 구성 요소는 'bull',
        가장 낮은 것은 'bear', 나머지는 'sideways'로 레이블됩니다.

        Args:
            means (np.ndarray): Component means, shape (nComponents, nFeatures).
                구성 요소 평균, (구성 요소 수, 특징 수) 형태.

        Returns:
            Dict[int, str]: Mapping from component index to regime name.
                구성 요소 인덱스에서 레짐 이름으로의 매핑.
        """
        nComponents = means.shape[0]
        returnMeans = means[:, 0]
        sortedIndices = np.argsort(returnMeans)

        mapping = {}

        if nComponents == 1:
            mapping[0] = "sideways"
        elif nComponents == 2:
            mapping[int(sortedIndices[0])] = "bear"
            mapping[int(sortedIndices[1])] = "bull"
        else:
            mapping[int(sortedIndices[0])] = "bear"
            mapping[int(sortedIndices[-1])] = "bull"
            for idx in sortedIndices[1:-1]:
                mapping[int(idx)] = "sideways"

        return mapping

    def _buildTransitionMatrix(
        self,
        regimeLabels: List[str],
        regimeMapping: Dict[int, str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a Markov transition probability matrix from regime label sequence.
        레짐 레이블 시퀀스로부터 마르코프 전환 확률 행렬을 구축합니다.

        Computes P(regime_{t+1} = j | regime_t = i) for all regime pairs by
        counting consecutive-day transitions and normalizing row-wise.

        연속일 전환을 세고 행별로 정규화하여 모든 레짐 쌍에 대한
        P(regime_{t+1} = j | regime_t = i)를 계산합니다.

        Args:
            regimeLabels (List[str]): Sequence of regime labels per day.
                일별 레짐 레이블 시퀀스.
            regimeMapping (Dict[int, str]): Component-to-regime name mapping.
                구성 요소-레짐 이름 매핑.

        Returns:
            Dict[str, Dict[str, float]]: Nested dict of transition probabilities.
                전환 확률의 중첩 딕셔너리.
        """
        regimeNames = sorted(set(regimeMapping.values()))

        transitionCounts = {
            fromR: {toR: 0 for toR in regimeNames}
            for fromR in regimeNames
        }

        for i in range(len(regimeLabels) - 1):
            fromRegime = regimeLabels[i]
            toRegime = regimeLabels[i + 1]
            if fromRegime in transitionCounts and toRegime in transitionCounts[fromRegime]:
                transitionCounts[fromRegime][toRegime] += 1

        transitionProbs = {}
        for fromR in regimeNames:
            rowTotal = sum(transitionCounts[fromR].values())
            transitionProbs[fromR] = {}
            for toR in regimeNames:
                if rowTotal > 0:
                    transitionProbs[fromR][toR] = transitionCounts[fromR][toR] / rowTotal
                else:
                    transitionProbs[fromR][toR] = 1.0 / len(regimeNames)

        return transitionProbs

    def _regimePerformance(
        self,
        returns: pd.Series,
        regimeLabels: List[str],
        regimeMapping: Dict[int, str],
    ) -> Dict:
        """
        Compute regime-conditional performance metrics.
        레짐별 조건부 성과 지표를 계산합니다.

        For each regime, filters the daily returns that occurred during that
        regime and calculates the average return and annualized Sharpe ratio.

        각 레짐에 대해 해당 레짐 동안 발생한 일별 수익률을 필터링하고
        평균 수익률과 연환산 샤프 비율을 계산합니다.

        Args:
            returns (pd.Series): Daily return series aligned with regime labels.
                레짐 레이블과 정렬된 일별 수익률 시리즈.
            regimeLabels (List[str]): Regime label per day.
                일별 레짐 레이블.
            regimeMapping (Dict[int, str]): Component-to-regime name mapping.
                구성 요소-레짐 이름 매핑.

        Returns:
            Dict: Dictionary with keys 'regimeReturns' and 'regimeSharpe'.
                'regimeReturns'와 'regimeSharpe' 키를 가진 딕셔너리.
        """
        regimeNames = sorted(set(regimeMapping.values()))
        returnsArr = returns.values

        regimeReturns = {}
        regimeSharpe = {}

        labelsArr = np.array(regimeLabels)

        for regimeName in regimeNames:
            mask = labelsArr == regimeName
            regimeRets = returnsArr[mask]

            if len(regimeRets) > 0:
                meanRet = float(np.mean(regimeRets))
                stdRet = float(np.std(regimeRets, ddof=1)) if len(regimeRets) > 1 else 0.0

                regimeReturns[regimeName] = meanRet

                if stdRet > 1e-10:
                    regimeSharpe[regimeName] = float(
                        (meanRet / stdRet) * np.sqrt(ANNUALIZATION_FACTOR)
                    )
                else:
                    regimeSharpe[regimeName] = 0.0
            else:
                regimeReturns[regimeName] = 0.0
                regimeSharpe[regimeName] = 0.0

        return {
            "regimeReturns": regimeReturns,
            "regimeSharpe": regimeSharpe,
        }

    def _regimeDurations(
        self,
        regimeLabels: List[str],
        regimeMapping: Dict[int, str],
    ) -> Dict[str, float]:
        """
        Compute average duration of consecutive days in each regime.
        각 레짐의 연속 일수 평균 지속 기간을 계산합니다.

        Groups the regime label sequence into contiguous runs and computes
        the average run length (in trading days) for each regime.

        레짐 레이블 시퀀스를 연속 구간으로 그룹화하고 각 레짐의
        평균 구간 길이(거래일)를 계산합니다.

        Args:
            regimeLabels (List[str]): Sequence of regime labels per day.
                일별 레짐 레이블 시퀀스.
            regimeMapping (Dict[int, str]): Component-to-regime name mapping.
                구성 요소-레짐 이름 매핑.

        Returns:
            Dict[str, float]: Average duration in days for each regime.
                각 레짐의 평균 지속 기간 (일).
        """
        regimeNames = sorted(set(regimeMapping.values()))

        runLengths = {r: [] for r in regimeNames}

        if not regimeLabels:
            return {r: 0.0 for r in regimeNames}

        currentRegime = regimeLabels[0]
        currentLength = 1

        for i in range(1, len(regimeLabels)):
            if regimeLabels[i] == currentRegime:
                currentLength += 1
            else:
                runLengths[currentRegime].append(currentLength)
                currentRegime = regimeLabels[i]
                currentLength = 1

        runLengths[currentRegime].append(currentLength)

        durations = {}
        for regimeName in regimeNames:
            runs = runLengths[regimeName]
            if runs:
                durations[regimeName] = float(np.mean(runs))
            else:
                durations[regimeName] = 0.0

        return durations
