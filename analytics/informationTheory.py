# -*- coding: utf-8 -*-
"""
Tradix Information Theory Signal Analysis Module.
Tradix 정보이론 시그널 분석 모듈.

Measures signal quality using Shannon Entropy, Mutual Information, and
Transfer Entropy. Quantifies the non-linear information content between
trading signals and future returns. This is NOT available in ANY
open-source backtesting library.

Shannon 엔트로피, 상호 정보량, 전이 엔트로피를 사용하여 시그널 품질을 측정합니다.
트레이딩 시그널과 미래 수익률 사이의 비선형 정보 함량을 정량화합니다.
이 기능은 어떤 오픈소스 백테스팅 라이브러리에서도 제공되지 않습니다.

Features:
    - Shannon Entropy of trade signals and market returns (bits)
    - Mutual Information between signals and returns (non-linear dependency)
    - Transfer Entropy for directional causality (signal -> returns)
    - Information ratio and redundancy metrics
    - Signal quality grading (excellent / good / moderate / poor / noise)
    - Freedman-Diaconis rule for optimal histogram binning

Usage:
    from tradix.analytics.informationTheory import InformationTheoryAnalyzer

    analyzer = InformationTheoryAnalyzer()
    result = analyzer.analyze(backtestResult)
    print(result.summary())
    print(result.summary(ko=True))
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from tradix.engine import BacktestResult


QUALITY_THRESHOLDS = {
    "excellent": {"mi": 0.3, "te": 0.1},
    "good": {"mi": 0.15, "te": 0.05},
    "moderate": {"mi": 0.05, "te": 0.0},
    "poor": {"mi": 0.01, "te": 0.0},
}

QUALITY_LABELS_KO = {
    "excellent": "우수",
    "good": "양호",
    "moderate": "보통",
    "poor": "미흡",
    "noise": "노이즈",
}

MIN_DATA_POINTS = 30


@dataclass
class InformationTheoryResult:
    """
    Result container for information theory signal analysis.
    정보이론 시그널 분석 결과 컨테이너.

    Encapsulates entropy, mutual information, transfer entropy, and derived
    quality metrics that quantify the information content of trading signals
    relative to market returns.

    트레이딩 시그널의 시장 수익률 대비 정보 함량을 정량화하는 엔트로피,
    상호 정보량, 전이 엔트로피 및 파생 품질 지표를 캡슐화합니다.

    Attributes:
        signalEntropy (float): Shannon entropy of trade signals in bits.
            트레이딩 시그널의 Shannon 엔트로피 (비트).
        marketEntropy (float): Shannon entropy of market returns in bits.
            시장 수익률의 Shannon 엔트로피 (비트).
        mutualInformation (float): Mutual information between signals and returns (bits).
            시그널과 수익률 간 상호 정보량 (비트).
        transferEntropy (float): Directional information flow from signal to returns.
            시그널에서 수익률로의 방향성 정보 흐름.
        informationRatio (float): MI / max(signalEntropy, marketEntropy), 0-1 efficiency.
            정보 효율성 비율 (0~1).
        redundancy (float): Fraction of signal info already in returns, 0-1.
            수익률에 이미 포함된 시그널 정보의 비율 (0~1).
        signalQuality (str): Quality grade - excellent/good/moderate/poor/noise.
            시그널 품질 등급.
        details (Dict): Additional breakdowns and intermediate metrics.
            추가 세부 분석 및 중간 지표.

    Example:
        >>> analyzer = InformationTheoryAnalyzer()
        >>> result = analyzer.analyze(backtestResult)
        >>> print(result.signalQuality)
        'good'
        >>> print(result.summary())
    """
    signalEntropy: float = 0.0
    marketEntropy: float = 0.0
    mutualInformation: float = 0.0
    transferEntropy: float = 0.0
    informationRatio: float = 0.0
    redundancy: float = 0.0
    signalQuality: str = "noise"
    details: Dict = field(default_factory=dict)

    def summary(self, ko: bool = False) -> str:
        """
        Generate a formatted summary report of the information theory analysis.
        정보이론 분석의 포맷된 요약 보고서를 생성합니다.

        Includes entropy values, mutual information, transfer entropy,
        information ratio, redundancy, and the overall signal quality grade.

        엔트로피 값, 상호 정보량, 전이 엔트로피, 정보 비율,
        중복도, 전체 시그널 품질 등급을 포함합니다.

        Args:
            ko (bool): If True, generate Korean-language report.
                True이면 한국어 보고서를 생성합니다. Defaults to False.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 포맷된 요약 문자열.
        """
        qualityLabel = QUALITY_LABELS_KO.get(self.signalQuality, self.signalQuality)

        if ko:
            return (
                f"\n{'='*55}\n"
                f"  정보이론 시그널 분석 (Information Theory Analysis)\n"
                f"{'='*55}\n"
                f"  시그널 엔트로피:   {self.signalEntropy:.4f} bits\n"
                f"  시장 엔트로피:     {self.marketEntropy:.4f} bits\n"
                f"{'─'*55}\n"
                f"  상호 정보량 (MI):  {self.mutualInformation:.4f} bits\n"
                f"  전이 엔트로피 (TE): {self.transferEntropy:.4f} bits\n"
                f"  정보 효율 비율:    {self.informationRatio:.4f}\n"
                f"  정보 중복도:       {self.redundancy:.4f}\n"
                f"{'─'*55}\n"
                f"  시그널 품질:       {qualityLabel}\n"
                f"{'='*55}\n"
            )

        return (
            f"\n{'='*55}\n"
            f"  Information Theory Signal Analysis\n"
            f"{'='*55}\n"
            f"  Signal Entropy:       {self.signalEntropy:.4f} bits\n"
            f"  Market Entropy:       {self.marketEntropy:.4f} bits\n"
            f"{'─'*55}\n"
            f"  Mutual Information:   {self.mutualInformation:.4f} bits\n"
            f"  Transfer Entropy:     {self.transferEntropy:.4f} bits\n"
            f"  Information Ratio:    {self.informationRatio:.4f}\n"
            f"  Redundancy:           {self.redundancy:.4f}\n"
            f"{'─'*55}\n"
            f"  Signal Quality:       {self.signalQuality}\n"
            f"{'='*55}\n"
        )

    def __repr__(self) -> str:
        return (
            f"InformationTheoryResult("
            f"MI={self.mutualInformation:.4f}, "
            f"TE={self.transferEntropy:.4f}, "
            f"quality={self.signalQuality})"
        )


class InformationTheoryAnalyzer:
    """
    Analyzer for measuring signal quality using information theory metrics.
    정보이론 지표를 사용하여 시그널 품질을 측정하는 분석기.

    Extracts binary trading signals from backtest trade history, computes
    Shannon entropy of both signals and market returns, then measures the
    non-linear information coupling via Mutual Information and directional
    causality via Transfer Entropy.

    백테스트 거래 내역에서 이진 트레이딩 시그널을 추출하고, 시그널과
    시장 수익률 모두의 Shannon 엔트로피를 계산한 뒤, 상호 정보량을 통한
    비선형 정보 결합과 전이 엔트로피를 통한 방향성 인과관계를 측정합니다.

    Features:
        - Shannon Entropy via histogram-based estimation
        - Mutual Information from 2D joint histogram
        - Transfer Entropy for directional signal-to-return causality
        - Freedman-Diaconis optimal binning
        - Automatic signal extraction from trade entry/exit times

    Example:
        >>> analyzer = InformationTheoryAnalyzer()
        >>> result = analyzer.analyze(backtestResult)
        >>> print(result.summary(ko=True))
    """

    def __init__(self):
        """
        Initialize the InformationTheoryAnalyzer.
        InformationTheoryAnalyzer를 초기화합니다.
        """
        pass

    def analyze(self, result: BacktestResult) -> InformationTheoryResult:
        """
        Perform information theory analysis on a backtest result.
        백테스트 결과에 대한 정보이론 분석을 수행합니다.

        Extracts a binary signal series (1=in position, 0=not) from trade
        history, computes daily returns from the equity curve, then measures
        Shannon entropy, mutual information, transfer entropy, information
        ratio, and redundancy to grade overall signal quality.

        거래 내역에서 이진 시그널 시리즈(1=포지션 보유, 0=미보유)를 추출하고,
        자산 곡선에서 일별 수익률을 계산한 뒤, Shannon 엔트로피, 상호 정보량,
        전이 엔트로피, 정보 비율, 중복도를 측정하여 전체 시그널 품질을 등급화합니다.

        Args:
            result (BacktestResult): Completed backtest result containing trades
                and equity curve. 거래 내역과 자산 곡선이 포함된 백테스트 결과.

        Returns:
            InformationTheoryResult: Analysis result with entropy metrics and
                signal quality grade. 엔트로피 지표와 시그널 품질 등급이 포함된 분석 결과.
        """
        equityCurve = result.equityCurve
        closedTrades = [t for t in result.trades if t.isClosed]

        if equityCurve is None or len(equityCurve) < MIN_DATA_POINTS:
            return InformationTheoryResult(
                details={"error": "insufficient_data", "dataPoints": len(equityCurve) if equityCurve is not None else 0}
            )

        returns = equityCurve.pct_change().dropna().values.astype(np.float64)

        if len(returns) < MIN_DATA_POINTS:
            return InformationTheoryResult(
                details={"error": "insufficient_returns", "returnPoints": len(returns)}
            )

        signalSeries = self._extractSignal(closedTrades, equityCurve)

        minLen = min(len(signalSeries), len(returns))
        signalSeries = signalSeries[:minLen]
        returns = returns[:minLen]

        if len(signalSeries) < MIN_DATA_POINTS:
            return InformationTheoryResult(
                details={"error": "insufficient_aligned_data", "alignedPoints": len(signalSeries)}
            )

        signalFloat = signalSeries.astype(np.float64)

        signalEntropy = self._shannonEntropy(signalFloat)
        marketEntropy = self._shannonEntropy(returns)

        mi = self._mutualInformation(signalFloat, returns)

        te = self._transferEntropy(signalFloat, returns, lag=1)

        maxEntropy = max(signalEntropy, marketEntropy)
        informationRatio = mi / maxEntropy if maxEntropy > 0 else 0.0
        informationRatio = float(np.clip(informationRatio, 0.0, 1.0))

        jointEnt = self._jointEntropy(signalFloat, returns)
        sumEntropies = signalEntropy + marketEntropy
        redundancy = 1.0 - (jointEnt / sumEntropies) if sumEntropies > 0 else 0.0
        redundancy = float(np.clip(redundancy, 0.0, 1.0))

        signalQuality = self._gradeQuality(mi, te)

        nmi = 0.0
        if signalEntropy > 0 and marketEntropy > 0:
            nmi = mi / np.sqrt(signalEntropy * marketEntropy)
            nmi = float(np.clip(nmi, 0.0, 1.0))

        details = {
            "dataPoints": minLen,
            "signalMean": float(np.mean(signalFloat)),
            "signalStd": float(np.std(signalFloat)),
            "returnMean": float(np.mean(returns)),
            "returnStd": float(np.std(returns)),
            "normalizedMI": nmi,
            "jointEntropy": jointEnt,
            "signalBins": self._optimalBins(signalFloat),
            "returnBins": self._optimalBins(returns),
            "totalTrades": len(closedTrades),
            "positionRatio": float(np.mean(signalFloat)),
        }

        return InformationTheoryResult(
            signalEntropy=signalEntropy,
            marketEntropy=marketEntropy,
            mutualInformation=mi,
            transferEntropy=te,
            informationRatio=informationRatio,
            redundancy=redundancy,
            signalQuality=signalQuality,
            details=details,
        )

    def _shannonEntropy(self, series: np.ndarray, bins: Optional[int] = None) -> float:
        """
        Compute Shannon entropy of a series using histogram-based estimation.
        히스토그램 기반 추정을 사용하여 시리즈의 Shannon 엔트로피를 계산합니다.

        Discretizes the continuous series into bins using the Freedman-Diaconis
        rule (if bins is not specified), then computes H(X) = -sum(p(x) * log2(p(x)))
        over the empirical probability distribution.

        (bins 미지정 시) Freedman-Diaconis 규칙으로 연속 시리즈를 이산화한 뒤,
        경험적 확률 분포에 대해 H(X) = -sum(p(x) * log2(p(x)))을 계산합니다.

        Args:
            series (np.ndarray): 1D numeric array. 1차원 수치 배열.
            bins (Optional[int]): Number of histogram bins. If None, uses
                Freedman-Diaconis rule. 히스토그램 빈 수. None이면 Freedman-Diaconis 규칙 사용.

        Returns:
            float: Shannon entropy in bits (base-2 logarithm). Shannon 엔트로피 (비트, 밑 2 로그).
        """
        if len(series) == 0:
            return 0.0

        if bins is None:
            bins = self._optimalBins(series)

        counts, _ = np.histogram(series, bins=bins)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]

        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    def _jointEntropy(self, x: np.ndarray, y: np.ndarray, bins: Optional[int] = None) -> float:
        """
        Compute joint Shannon entropy H(X, Y) using a 2D histogram.
        2D 히스토그램을 사용하여 결합 Shannon 엔트로피 H(X, Y)를 계산합니다.

        Estimates the joint probability distribution via 2D histogram binning,
        then computes H(X, Y) = -sum(p(x,y) * log2(p(x,y))).

        2D 히스토그램 빈닝으로 결합 확률 분포를 추정한 뒤,
        H(X, Y) = -sum(p(x,y) * log2(p(x,y)))을 계산합니다.

        Args:
            x (np.ndarray): First variable array. 첫 번째 변수 배열.
            y (np.ndarray): Second variable array. 두 번째 변수 배열.
            bins (Optional[int]): Number of bins per dimension. If None, uses
                Freedman-Diaconis rule for each. 차원별 빈 수. None이면 각각 Freedman-Diaconis 사용.

        Returns:
            float: Joint Shannon entropy in bits. 결합 Shannon 엔트로피 (비트).
        """
        if len(x) == 0 or len(y) == 0:
            return 0.0

        binsX = bins if bins is not None else self._optimalBins(x)
        binsY = bins if bins is not None else self._optimalBins(y)

        counts, _, _ = np.histogram2d(x, y, bins=[binsX, binsY])
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]

        jointEntropy = -np.sum(probabilities * np.log2(probabilities))
        return float(jointEntropy)

    def _mutualInformation(self, x: np.ndarray, y: np.ndarray, bins: Optional[int] = None) -> float:
        """
        Compute Mutual Information MI(X; Y) = H(X) + H(Y) - H(X, Y).
        상호 정보량 MI(X; Y) = H(X) + H(Y) - H(X, Y)를 계산합니다.

        Mutual Information measures the amount of information shared between
        two variables. It captures both linear and non-linear dependencies,
        unlike Pearson correlation which only measures linear relationships.

        상호 정보량은 두 변수 간 공유되는 정보의 양을 측정합니다.
        선형 관계만 측정하는 피어슨 상관과 달리 선형 및 비선형
        의존성을 모두 포착합니다.

        Args:
            x (np.ndarray): First variable array. 첫 번째 변수 배열.
            y (np.ndarray): Second variable array. 두 번째 변수 배열.
            bins (Optional[int]): Number of bins per dimension. 차원별 빈 수.

        Returns:
            float: Mutual information in bits (non-negative). 상호 정보량 (비트, 비음수).
        """
        hX = self._shannonEntropy(x, bins=bins)
        hY = self._shannonEntropy(y, bins=bins)
        hXY = self._jointEntropy(x, y, bins=bins)

        mi = hX + hY - hXY
        return float(max(0.0, mi))

    def _conditionalEntropy(self, x: np.ndarray, y: np.ndarray, bins: Optional[int] = None) -> float:
        """
        Compute conditional entropy H(X|Y) = H(X, Y) - H(Y).
        조건부 엔트로피 H(X|Y) = H(X, Y) - H(Y)를 계산합니다.

        Measures the remaining uncertainty in X after observing Y.
        By the chain rule of entropy: H(X|Y) = H(X, Y) - H(Y).

        Y를 관측한 후 X에 남아있는 불확실성을 측정합니다.
        엔트로피의 연쇄 법칙: H(X|Y) = H(X, Y) - H(Y).

        Args:
            x (np.ndarray): Target variable array. 대상 변수 배열.
            y (np.ndarray): Conditioning variable array. 조건 변수 배열.
            bins (Optional[int]): Number of bins per dimension. 차원별 빈 수.

        Returns:
            float: Conditional entropy in bits (non-negative). 조건부 엔트로피 (비트, 비음수).
        """
        hXY = self._jointEntropy(x, y, bins=bins)
        hY = self._shannonEntropy(y, bins=bins)

        condEntropy = hXY - hY
        return float(max(0.0, condEntropy))

    def _transferEntropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        bins: Optional[int] = None,
    ) -> float:
        """
        Compute Transfer Entropy from source to target.
        소스에서 타겟으로의 전이 엔트로피를 계산합니다.

        Transfer Entropy measures the directed information flow from source
        to target: TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}).
        It quantifies how much knowing the past of X reduces uncertainty
        about the future of Y, beyond what Y's own past already provides.

        전이 엔트로피는 소스에서 타겟으로의 방향성 정보 흐름을 측정합니다:
        TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}).
        X의 과거를 아는 것이 Y 자체의 과거가 이미 제공하는 것 이상으로
        Y의 미래에 대한 불확실성을 얼마나 줄이는지를 정량화합니다.

        Implementation uses the equivalent formulation:
        TE(X->Y) = H(Y_t, Y_{t-1}) + H(Y_{t-1}, X_{t-1}) - H(Y_{t-1}) - H(Y_t, Y_{t-1}, X_{t-1})

        Args:
            source (np.ndarray): Source (causal) variable. 소스(원인) 변수.
            target (np.ndarray): Target (effect) variable. 타겟(결과) 변수.
            lag (int): Time lag for the transfer. 전이 시간 지연. Defaults to 1.
            bins (Optional[int]): Number of bins for discretization. 이산화 빈 수.

        Returns:
            float: Transfer entropy in bits (non-negative). 전이 엔트로피 (비트, 비음수).
        """
        n = len(target)
        if n <= lag + 1 or len(source) <= lag + 1:
            return 0.0

        minLen = min(len(source), len(target))
        source = source[:minLen]
        target = target[:minLen]

        yT = target[lag:]
        yPast = target[:-lag]
        xPast = source[:-lag]

        binsVal = bins if bins is not None else max(
            self._optimalBins(yT),
            self._optimalBins(yPast),
            self._optimalBins(xPast),
        )

        hYtYpast = self._jointEntropy(yT, yPast, bins=binsVal)

        hYpastXpast = self._jointEntropy(yPast, xPast, bins=binsVal)

        hYpast = self._shannonEntropy(yPast, bins=binsVal)

        hTriple = self._tripleJointEntropy(yT, yPast, xPast, bins=binsVal)

        te = hYtYpast + hYpastXpast - hYpast - hTriple
        return float(max(0.0, te))

    def _tripleJointEntropy(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        bins: Optional[int] = None,
    ) -> float:
        """
        Compute joint entropy H(A, B, C) of three variables using 3D histogram.
        3D 히스토그램을 사용하여 세 변수의 결합 엔트로피 H(A, B, C)를 계산합니다.

        Args:
            a (np.ndarray): First variable. 첫 번째 변수.
            b (np.ndarray): Second variable. 두 번째 변수.
            c (np.ndarray): Third variable. 세 번째 변수.
            bins (Optional[int]): Number of bins per dimension. 차원별 빈 수.

        Returns:
            float: Triple joint entropy in bits. 3변수 결합 엔트로피 (비트).
        """
        if len(a) == 0:
            return 0.0

        binsA = bins if bins is not None else self._optimalBins(a)
        binsB = bins if bins is not None else self._optimalBins(b)
        binsC = bins if bins is not None else self._optimalBins(c)

        sample = np.column_stack([a, b, c])

        counts, _ = np.histogramdd(sample, bins=[binsA, binsB, binsC])
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]

        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    def _optimalBins(self, series: np.ndarray) -> int:
        """
        Compute optimal histogram bin count using the Freedman-Diaconis rule.
        Freedman-Diaconis 규칙을 사용하여 최적 히스토그램 빈 수를 계산합니다.

        bin_width = 2 * IQR * n^(-1/3)
        num_bins = ceil((max - min) / bin_width)

        Falls back to Sturges' rule (ceil(log2(n)) + 1) when IQR is zero
        (e.g., for binary signals or constant series).

        IQR이 0인 경우(이진 시그널 또는 상수 시리즈 등) Sturges' 규칙
        (ceil(log2(n)) + 1)으로 대체합니다.

        Args:
            series (np.ndarray): 1D numeric array. 1차원 수치 배열.

        Returns:
            int: Optimal number of bins, minimum 2. 최적 빈 수, 최소 2.
        """
        n = len(series)
        if n < 2:
            return 2

        q75 = np.percentile(series, 75)
        q25 = np.percentile(series, 25)
        iqr = q75 - q25

        dataRange = float(np.max(series) - np.min(series))

        if iqr <= 0 or dataRange <= 0:
            sturges = int(np.ceil(np.log2(n))) + 1
            return max(2, sturges)

        binWidth = 2.0 * iqr * (n ** (-1.0 / 3.0))

        if binWidth <= 0:
            return max(2, int(np.ceil(np.log2(n))) + 1)

        numBins = int(np.ceil(dataRange / binWidth))
        return max(2, min(numBins, 256))

    def _extractSignal(self, trades: List, equityCurve: pd.Series) -> np.ndarray:
        """
        Extract a binary signal series from trade entry/exit times.
        거래 진입/청산 시점으로부터 이진 시그널 시리즈를 추출합니다.

        Creates a binary array aligned with the equity curve index where
        1 indicates the strategy held a position and 0 indicates cash.
        For overlapping trades, the signal remains 1 throughout.

        자산 곡선 인덱스에 맞춰 전략이 포지션을 보유하면 1, 현금 상태면 0인
        이진 배열을 생성합니다. 겹치는 거래의 경우 전 기간 동안 1을 유지합니다.

        Args:
            trades (List): List of closed Trade objects with entryDate and exitDate.
                entryDate와 exitDate를 가진 청산된 거래 목록.
            equityCurve (pd.Series): Time-indexed equity values for alignment.
                정렬을 위한 시간 인덱스 자산 곡선.

        Returns:
            np.ndarray: Binary signal array (0 or 1) aligned with equity curve.
                자산 곡선에 정렬된 이진 시그널 배열 (0 또는 1).
        """
        dates = equityCurve.index
        signal = np.zeros(len(dates), dtype=np.float64)

        for trade in trades:
            entryDate = pd.Timestamp(trade.entryDate)
            exitDate = pd.Timestamp(trade.exitDate) if trade.exitDate is not None else dates[-1]

            mask = (dates >= entryDate) & (dates <= exitDate)
            signal[mask] = 1.0

        returnSignal = signal[1:]
        return returnSignal

    def _gradeQuality(self, mi: float, te: float) -> str:
        """
        Grade signal quality based on mutual information and transfer entropy.
        상호 정보량과 전이 엔트로피를 기반으로 시그널 품질을 등급화합니다.

        Grading criteria:
            - excellent: MI > 0.3 and TE > 0.1
            - good: MI > 0.15 and TE > 0.05
            - moderate: MI > 0.05
            - poor: MI > 0.01
            - noise: otherwise

        Args:
            mi (float): Mutual information value in bits. 상호 정보량 (비트).
            te (float): Transfer entropy value in bits. 전이 엔트로피 (비트).

        Returns:
            str: Quality grade string. 품질 등급 문자열.
        """
        if mi > QUALITY_THRESHOLDS["excellent"]["mi"] and te > QUALITY_THRESHOLDS["excellent"]["te"]:
            return "excellent"
        if mi > QUALITY_THRESHOLDS["good"]["mi"] and te > QUALITY_THRESHOLDS["good"]["te"]:
            return "good"
        if mi > QUALITY_THRESHOLDS["moderate"]["mi"]:
            return "moderate"
        if mi > QUALITY_THRESHOLDS["poor"]["mi"]:
            return "poor"
        return "noise"
