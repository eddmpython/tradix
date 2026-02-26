"""
Tradex Strategy DNA Module - 12-dimensional strategy fingerprinting.
Tradex 전략 DNA 모듈 - 12차원 전략 지문(fingerprint) 분석.

Extracts a unique 12-dimensional vector fingerprint from any backtest result,
enabling quantitative strategy comparison, classification, and similarity
search. Each dimension captures a distinct behavioral characteristic of the
trading strategy, normalized to the [0, 1] range.

백테스트 결과로부터 고유한 12차원 벡터 지문을 추출하여, 전략 간 정량적 비교,
분류, 유사도 검색을 가능하게 합니다. 각 차원은 트레이딩 전략의 고유한
행동적 특성을 [0, 1] 범위로 정규화하여 나타냅니다.

Dimensions:
    1. trendSensitivity: Trend-following tendency (추세 추종 민감도)
    2. meanReversionAffinity: Mean reversion tendency (평균회귀 성향)
    3. volatilityPreference: Preference for volatile markets (변동성 선호도)
    4. holdingPeriodProfile: Short-term to long-term holding spectrum (보유 기간 프로파일)
    5. drawdownTolerance: Maximum drawdown tolerance (낙폭 허용도)
    6. winRateProfile: Win rate vs risk/reward tradeoff (승률 프로파일)
    7. marketRegimeDependence: Regime sensitivity (시장 레짐 의존도)
    8. concentrationLevel: Position concentration (포지션 집중도)
    9. tradingFrequency: Trading frequency spectrum (거래 빈도)
    10. riskRewardRatio: Normalized risk/reward ratio (리스크-리워드 비율)
    11. momentumExposure: Momentum factor exposure (모멘텀 익스포져)
    12. defensiveScore: Defensive posture composite (방어적 점수)

Features:
    - Cosine similarity and Euclidean distance between strategy DNAs
    - Radar chart data export for visualization
    - Dominant trait extraction
    - Strategy type classification into 8 archetypes
    - Nearest-neighbor search across strategy populations

Usage:
    from tradex.analytics.strategyDna import StrategyDnaAnalyzer

    analyzer = StrategyDnaAnalyzer()
    dna = analyzer.analyze(backtestResult)
    print(dna.summary())
    print(analyzer.classify(dna))
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from tradex.engine import BacktestResult
from tradex.entities.trade import Trade


DIMENSION_NAMES = [
    "trendSensitivity",
    "meanReversionAffinity",
    "volatilityPreference",
    "holdingPeriodProfile",
    "drawdownTolerance",
    "winRateProfile",
    "marketRegimeDependence",
    "concentrationLevel",
    "tradingFrequency",
    "riskRewardRatio",
    "momentumExposure",
    "defensiveScore",
]

DIMENSION_LABELS_KO = {
    "trendSensitivity": "추세 민감도",
    "meanReversionAffinity": "평균회귀 성향",
    "volatilityPreference": "변동성 선호도",
    "holdingPeriodProfile": "보유 기간",
    "drawdownTolerance": "낙폭 허용도",
    "winRateProfile": "승률 프로파일",
    "marketRegimeDependence": "레짐 의존도",
    "concentrationLevel": "집중도",
    "tradingFrequency": "거래 빈도",
    "riskRewardRatio": "리스크-리워드",
    "momentumExposure": "모멘텀 익스포져",
    "defensiveScore": "방어적 점수",
}

NUM_DIMENSIONS = 12


@dataclass
class StrategyDNA:
    """
    12-dimensional vector fingerprint that uniquely characterizes a trading strategy.
    트레이딩 전략을 고유하게 특성화하는 12차원 벡터 지문.

    Each dimension is a float in [0, 1] representing a normalized behavioral
    trait of the strategy. Together they form a compact, comparable
    representation enabling similarity search, classification, and
    visual analysis via radar charts.

    각 차원은 [0, 1] 범위의 float로 전략의 정규화된 행동적 특성을 나타냅니다.
    12개 차원이 합쳐져 유사도 검색, 분류, 레이더 차트 시각화를 위한
    간결하고 비교 가능한 표현을 형성합니다.

    Attributes:
        trendSensitivity (float): How strongly the strategy follows trends.
            전략이 추세를 얼마나 강하게 따르는지. 0=반추세, 1=순수 추세추종.
        meanReversionAffinity (float): Mean reversion tendency.
            평균회귀 성향. 0=추세추종, 1=순수 평균회귀.
        volatilityPreference (float): Preference for volatile markets.
            변동성 높은 시장 선호도. 0=저변동성 선호, 1=고변동성 선호.
        holdingPeriodProfile (float): Holding period spectrum.
            보유 기간 스펙트럼. 0=초단기, 1=장기 보유.
        drawdownTolerance (float): Maximum drawdown tolerance.
            최대 낙폭 허용도. 0=낙폭 비허용, 1=높은 낙폭 허용.
        winRateProfile (float): Win rate vs risk/reward tradeoff.
            승률 프로파일. 0=저승률 고RR, 1=고승률 저RR.
        marketRegimeDependence (float): How regime-dependent the strategy is.
            시장 레짐 의존도. 0=레짐 독립, 1=높은 레짐 의존.
        concentrationLevel (float): Position concentration level.
            포지션 집중도. 0=분산, 1=집중.
        tradingFrequency (float): Trading frequency spectrum.
            거래 빈도. 0=저빈도, 1=고빈도.
        riskRewardRatio (float): Normalized risk/reward ratio.
            정규화된 리스크-리워드 비율. 0=낮은 RR, 1=높은 RR.
        momentumExposure (float): Momentum factor exposure.
            모멘텀 팩터 익스포져. 0=음의 모멘텀, 1=강한 양의 모멘텀.
        defensiveScore (float): How defensive the strategy is.
            방어적 점수. 0=공격적, 1=매우 방어적.

    Example:
        >>> dna = StrategyDNA(
        ...     trendSensitivity=0.8, meanReversionAffinity=0.2,
        ...     volatilityPreference=0.5, holdingPeriodProfile=0.6,
        ...     drawdownTolerance=0.3, winRateProfile=0.4,
        ...     marketRegimeDependence=0.5, concentrationLevel=0.7,
        ...     tradingFrequency=0.3, riskRewardRatio=0.6,
        ...     momentumExposure=0.7, defensiveScore=0.4,
        ... )
        >>> print(dna.dominantTraits())
        [('trendSensitivity', 0.8), ('concentrationLevel', 0.7), ('momentumExposure', 0.7)]
    """
    trendSensitivity: float = 0.0
    meanReversionAffinity: float = 0.0
    volatilityPreference: float = 0.0
    holdingPeriodProfile: float = 0.0
    drawdownTolerance: float = 0.0
    winRateProfile: float = 0.0
    marketRegimeDependence: float = 0.0
    concentrationLevel: float = 0.0
    tradingFrequency: float = 0.0
    riskRewardRatio: float = 0.0
    momentumExposure: float = 0.0
    defensiveScore: float = 0.0

    def __post_init__(self):
        for name in DIMENSION_NAMES:
            value = getattr(self, name)
            setattr(self, name, float(np.clip(value, 0.0, 1.0)))

    def toVector(self) -> np.ndarray:
        """
        Convert the DNA to a 12-element numpy array.
        DNA를 12차원 numpy 배열로 변환합니다.

        Returns:
            np.ndarray: Shape (12,) array of dimension values in canonical order.
                정규 순서의 차원 값을 담은 (12,) 배열.
        """
        return np.array([getattr(self, name) for name in DIMENSION_NAMES], dtype=np.float64)

    def toDict(self) -> dict:
        """
        Convert the DNA to a dictionary mapping dimension names to values.
        DNA를 차원 이름-값 매핑 딕셔너리로 변환합니다.

        Returns:
            dict: Dimension names as keys, float values in [0, 1] as values.
                차원 이름이 키, [0, 1] 범위의 float 값이 값인 딕셔너리.
        """
        return {name: getattr(self, name) for name in DIMENSION_NAMES}

    def toRadarData(self) -> dict:
        """
        Prepare data for radar (spider) chart visualization.
        레이더(스파이더) 차트 시각화용 데이터를 생성합니다.

        Returns a dict with 'labels' (Korean display names), 'values'
        (dimension values), and 'labelKeys' (English dimension names)
        suitable for direct use with plotting libraries.

        'labels'(한국어 표시명), 'values'(차원 값), 'labelKeys'(영어 차원명)를
        담은 딕셔너리를 반환하며, 차트 라이브러리에서 직접 사용 가능합니다.

        Returns:
            dict: Keys 'labels', 'values', 'labelKeys' for chart rendering.
                차트 렌더링용 'labels', 'values', 'labelKeys' 키를 가진 딕셔너리.
        """
        return {
            "labels": [DIMENSION_LABELS_KO[name] for name in DIMENSION_NAMES],
            "values": [getattr(self, name) for name in DIMENSION_NAMES],
            "labelKeys": list(DIMENSION_NAMES),
        }

    def dominantTraits(self, topN: int = 3) -> List[Tuple[str, float]]:
        """
        Return the top N most dominant (highest-valued) dimensions.
        가장 지배적인(값이 높은) 상위 N개 차원을 반환합니다.

        Args:
            topN: Number of top traits to return (반환할 상위 특성 수). Defaults to 3.

        Returns:
            List[Tuple[str, float]]: Sorted list of (dimension_name, value) pairs,
                descending by value.
                값 기준 내림차순 정렬된 (차원명, 값) 튜플 리스트.
        """
        pairs = [(name, getattr(self, name)) for name in DIMENSION_NAMES]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:topN]

    def similarity(self, other: "StrategyDNA") -> float:
        """
        Compute cosine similarity to another StrategyDNA.
        다른 StrategyDNA와의 코사인 유사도를 계산합니다.

        Cosine similarity measures the angular similarity between two DNA
        vectors, returning a value in [0, 1] where 1 means identical
        direction in the 12-dimensional trait space.

        코사인 유사도는 두 DNA 벡터 간의 각도적 유사성을 측정하며,
        12차원 특성 공간에서 동일 방향이면 1, 직교하면 0을 반환합니다.

        Args:
            other: Another StrategyDNA instance to compare against.
                비교 대상 StrategyDNA 인스턴스.

        Returns:
            float: Cosine similarity in [0, 1]. 0=orthogonal, 1=identical direction.
                [0, 1] 범위의 코사인 유사도. 0=직교, 1=동일 방향.
        """
        vecA = self.toVector()
        vecB = other.toVector()
        normA = np.linalg.norm(vecA)
        normB = np.linalg.norm(vecB)
        if normA == 0.0 or normB == 0.0:
            return 0.0
        cosine = np.dot(vecA, vecB) / (normA * normB)
        return float(np.clip(cosine, 0.0, 1.0))

    def distance(self, other: "StrategyDNA") -> float:
        """
        Compute Euclidean distance to another StrategyDNA.
        다른 StrategyDNA와의 유클리드 거리를 계산합니다.

        Euclidean distance in the 12-dimensional trait space. Since all
        dimensions are in [0, 1], the theoretical maximum distance is
        sqrt(12) ~ 3.464.

        12차원 특성 공간에서의 유클리드 거리입니다. 모든 차원이 [0, 1] 범위이므로
        이론적 최대 거리는 sqrt(12) ~ 3.464입니다.

        Args:
            other: Another StrategyDNA instance to compare against.
                비교 대상 StrategyDNA 인스턴스.

        Returns:
            float: Euclidean distance >= 0. Lower means more similar.
                유클리드 거리 >= 0. 낮을수록 더 유사함.
        """
        vecA = self.toVector()
        vecB = other.toVector()
        return float(np.linalg.norm(vecA - vecB))

    def summary(self) -> str:
        """
        Generate a human-readable Korean summary of the strategy DNA.
        전략 DNA의 한국어 요약 문자열을 생성합니다.

        Includes all 12 dimension values with Korean labels and
        highlights the top 3 dominant traits.

        12개 차원 값을 한국어 레이블과 함께 표시하고
        상위 3개 지배적 특성을 강조합니다.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 포맷된 요약 문자열.
        """
        lines = ["전략 DNA 프로파일"]
        lines.append("=" * 40)
        for name in DIMENSION_NAMES:
            value = getattr(self, name)
            label = DIMENSION_LABELS_KO[name]
            barLength = int(value * 20)
            bar = "\u2588" * barLength + "\u2591" * (20 - barLength)
            lines.append(f"  {label:<12s} {bar} {value:.3f}")

        lines.append("-" * 40)
        lines.append("주요 특성:")
        for name, value in self.dominantTraits(3):
            label = DIMENSION_LABELS_KO[name]
            lines.append(f"  {label}: {value:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        top3 = self.dominantTraits(3)
        traitStr = ", ".join(f"{n}={v:.2f}" for n, v in top3)
        return f"StrategyDNA({traitStr})"


class StrategyDnaAnalyzer:
    """
    Analyzer that extracts StrategyDNA from backtest results.
    백테스트 결과로부터 StrategyDNA를 추출하는 분석기.

    Computes each of the 12 DNA dimensions from trade history, equity curve,
    and performance metrics using statistical methods including autocorrelation,
    rolling-window variance, and distribution analysis.

    자기상관, 롤링 윈도우 분산, 분포 분석 등의 통계적 방법을 사용하여
    거래 내역, 자산 곡선, 성과 지표로부터 12개 DNA 차원을 각각 계산합니다.

    Features:
        - Extract 12-dimensional DNA fingerprint from any BacktestResult
        - Compare two DNA instances with detailed dimension-by-dimension analysis
        - Search for similar strategies in a candidate pool
        - Classify strategy type into one of 8 archetypes

    기능:
        - 모든 BacktestResult로부터 12차원 DNA 지문 추출
        - 차원별 상세 분석으로 두 DNA 인스턴스 비교
        - 후보 풀에서 유사 전략 검색
        - 8가지 전략 원형 중 하나로 전략 유형 분류

    Example:
        >>> analyzer = StrategyDnaAnalyzer()
        >>> dna = analyzer.analyze(backtestResult)
        >>> print(analyzer.classify(dna))
        '트렌드 팔로워'
    """

    ROLLING_WINDOW = 20
    MAX_HOLDING_DAYS = 252
    MAX_TRADES_PER_YEAR = 252

    def __init__(self):
        """
        Initialize the StrategyDnaAnalyzer.
        StrategyDnaAnalyzer를 초기화합니다.
        """
        pass

    def analyze(self, result: BacktestResult) -> StrategyDNA:
        """
        Extract a StrategyDNA fingerprint from a backtest result.
        백테스트 결과로부터 StrategyDNA 지문을 추출합니다.

        Examines the equity curve, trade list, and metrics to compute each
        of the 12 DNA dimensions. All dimensions are normalized to [0, 1].

        자산 곡선, 거래 목록, 지표를 분석하여 12개 DNA 차원을 각각 계산합니다.
        모든 차원은 [0, 1]로 정규화됩니다.

        Args:
            result: A BacktestResult containing trades, equityCurve, and metrics.
                trades, equityCurve, metrics를 포함하는 BacktestResult.

        Returns:
            StrategyDNA: The extracted 12-dimensional fingerprint.
                추출된 12차원 지문.
        """
        closedTrades = [t for t in result.trades if t.isClosed]
        equityCurve = result.equityCurve
        metrics = result.metrics if isinstance(result.metrics, dict) else {}

        returns = self._computeReturns(equityCurve)
        pnlSeries = np.array([t.pnl for t in closedTrades], dtype=np.float64)

        trendSens = self._computeTrendSensitivity(returns)
        meanRevAff = 1.0 - trendSens
        volPref = self._computeVolatilityPreference(returns)
        holdPeriod = self._computeHoldingPeriodProfile(closedTrades)
        ddTolerance = self._computeDrawdownTolerance(equityCurve, result.totalReturn)
        wrProfile = self._computeWinRateProfile(result.winRate)
        regimeDep = self._computeMarketRegimeDependence(returns)
        concLevel = self._computeConcentrationLevel(closedTrades)
        tradFreq = self._computeTradingFrequency(closedTrades, equityCurve)
        rrRatio = self._computeRiskRewardRatio(closedTrades)
        momExp = self._computeMomentumExposure(pnlSeries)
        defScore = self._computeDefensiveScore(equityCurve, closedTrades)

        return StrategyDNA(
            trendSensitivity=trendSens,
            meanReversionAffinity=meanRevAff,
            volatilityPreference=volPref,
            holdingPeriodProfile=holdPeriod,
            drawdownTolerance=ddTolerance,
            winRateProfile=wrProfile,
            marketRegimeDependence=regimeDep,
            concentrationLevel=concLevel,
            tradingFrequency=tradFreq,
            riskRewardRatio=rrRatio,
            momentumExposure=momExp,
            defensiveScore=defScore,
        )

    def compare(self, dna1: StrategyDNA, dna2: StrategyDNA) -> dict:
        """
        Perform a detailed dimension-by-dimension comparison of two StrategyDNAs.
        두 StrategyDNA 간의 차원별 상세 비교를 수행합니다.

        Returns overall similarity and distance metrics plus per-dimension
        deltas showing where the two strategies diverge most.

        전체 유사도 및 거리 지표와 함께, 두 전략이 가장 크게 차이 나는
        차원별 델타를 반환합니다.

        Args:
            dna1: First StrategyDNA instance. 첫 번째 StrategyDNA.
            dna2: Second StrategyDNA instance. 두 번째 StrategyDNA.

        Returns:
            dict: Comparison results with keys:
                - 'similarity' (float): Cosine similarity [0, 1]
                - 'distance' (float): Euclidean distance
                - 'dimensions' (dict): Per-dimension dict with 'dna1', 'dna2', 'delta' values
                - 'largestDifferences' (List[Tuple[str, float]]): Top 3 most divergent dims
                비교 결과 딕셔너리.
        """
        vec1 = dna1.toVector()
        vec2 = dna2.toVector()
        deltas = np.abs(vec1 - vec2)

        dimensions = {}
        for i, name in enumerate(DIMENSION_NAMES):
            dimensions[name] = {
                "dna1": float(vec1[i]),
                "dna2": float(vec2[i]),
                "delta": float(deltas[i]),
            }

        sortedDiffs = sorted(
            [(name, float(deltas[i])) for i, name in enumerate(DIMENSION_NAMES)],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "similarity": dna1.similarity(dna2),
            "distance": dna1.distance(dna2),
            "dimensions": dimensions,
            "largestDifferences": sortedDiffs[:3],
        }

    def findSimilar(
        self,
        target: StrategyDNA,
        candidates: List[StrategyDNA],
        topN: int = 5,
    ) -> List[Tuple[StrategyDNA, float]]:
        """
        Find the most similar strategies to a target DNA from a candidate list.
        후보 리스트에서 대상 DNA와 가장 유사한 전략들을 찾습니다.

        Uses cosine similarity as the ranking metric. Returns up to topN
        results sorted by descending similarity.

        코사인 유사도를 순위 지표로 사용합니다. 유사도 내림차순으로
        최대 topN개의 결과를 반환합니다.

        Args:
            target: The reference StrategyDNA to search against.
                검색 기준이 되는 StrategyDNA.
            candidates: List of StrategyDNA instances to rank.
                순위를 매길 StrategyDNA 인스턴스 리스트.
            topN: Maximum number of results to return (최대 반환 수). Defaults to 5.

        Returns:
            List[Tuple[StrategyDNA, float]]: List of (DNA, similarity) pairs sorted
                by similarity descending.
                유사도 내림차순으로 정렬된 (DNA, 유사도) 튜플 리스트.
        """
        if not candidates:
            return []

        scored = [(cand, target.similarity(cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topN]

    def classify(self, dna: StrategyDNA) -> str:
        """
        Classify a strategy into one of 8 archetypes based on its DNA.
        DNA를 기반으로 전략을 8가지 원형 중 하나로 분류합니다.

        Classification uses a weighted rule-based system that evaluates
        the dominant traits and their interactions to determine the
        best-fitting strategy archetype.

        지배적 특성과 그 상호작용을 평가하는 가중 규칙 기반 시스템으로
        가장 적합한 전략 원형을 결정합니다.

        Archetypes:
            - "트렌드 팔로워": Trend following
            - "평균회귀": Mean reversion
            - "모멘텀": Momentum
            - "스윙 트레이더": Swing trader
            - "스캘퍼": Scalper
            - "방어적 투자자": Defensive investor
            - "공격적 트레이더": Aggressive trader
            - "균형형": Balanced

        Args:
            dna: StrategyDNA instance to classify. 분류할 StrategyDNA 인스턴스.

        Returns:
            str: Korean strategy archetype label. 한국어 전략 원형 레이블.
        """
        scores: Dict[str, float] = {
            "트렌드 팔로워": 0.0,
            "평균회귀": 0.0,
            "모멘텀": 0.0,
            "스윙 트레이더": 0.0,
            "스캘퍼": 0.0,
            "방어적 투자자": 0.0,
            "공격적 트레이더": 0.0,
            "균형형": 0.0,
        }

        scores["트렌드 팔로워"] = (
            dna.trendSensitivity * 2.5
            + dna.holdingPeriodProfile * 1.0
            + (1.0 - dna.meanReversionAffinity) * 1.5
            + dna.momentumExposure * 1.0
        )

        scores["평균회귀"] = (
            dna.meanReversionAffinity * 2.5
            + dna.winRateProfile * 1.5
            + (1.0 - dna.trendSensitivity) * 1.5
            + (1.0 - dna.momentumExposure) * 0.5
        )

        scores["모멘텀"] = (
            dna.momentumExposure * 2.5
            + dna.trendSensitivity * 1.5
            + dna.riskRewardRatio * 1.0
            + (1.0 - dna.meanReversionAffinity) * 1.0
        )

        scores["스윙 트레이더"] = (
            self._bellCurveScore(dna.holdingPeriodProfile, 0.4, 0.2) * 2.5
            + self._bellCurveScore(dna.tradingFrequency, 0.4, 0.2) * 1.5
            + dna.riskRewardRatio * 1.0
            + self._bellCurveScore(dna.trendSensitivity, 0.5, 0.25) * 1.0
        )

        scores["스캘퍼"] = (
            dna.tradingFrequency * 2.5
            + (1.0 - dna.holdingPeriodProfile) * 2.0
            + dna.winRateProfile * 1.0
            + (1.0 - dna.riskRewardRatio) * 0.5
        )

        scores["방어적 투자자"] = (
            dna.defensiveScore * 2.5
            + (1.0 - dna.drawdownTolerance) * 2.0
            + dna.winRateProfile * 1.0
            + (1.0 - dna.volatilityPreference) * 0.5
        )

        scores["공격적 트레이더"] = (
            dna.drawdownTolerance * 2.0
            + dna.volatilityPreference * 2.0
            + dna.concentrationLevel * 1.5
            + (1.0 - dna.defensiveScore) * 0.5
        )

        vec = dna.toVector()
        uniformity = 1.0 - float(np.std(vec)) * 2.0
        scores["균형형"] = max(0.0, uniformity) * 5.0 + 1.0

        return max(scores, key=scores.get)

    def _computeReturns(self, equityCurve: pd.Series) -> np.ndarray:
        """
        Compute daily return series from the equity curve.
        자산 곡선으로부터 일별 수익률 시리즈를 계산합니다.

        Args:
            equityCurve: Time-indexed equity values. 시간 인덱스 자산 곡선.

        Returns:
            np.ndarray: Array of daily returns (1-period percentage changes).
                일별 수익률 (1기간 퍼센트 변화) 배열.
        """
        if equityCurve is None or len(equityCurve) < 2:
            return np.array([], dtype=np.float64)
        returns = equityCurve.pct_change().dropna().values.astype(np.float64)
        return returns

    def _computeTrendSensitivity(self, returns: np.ndarray) -> float:
        """
        Compute trend sensitivity from return autocorrelation.
        수익률 자기상관으로부터 추세 민감도를 계산합니다.

        Uses lag-1 autocorrelation of returns as a proxy for trend-following
        behavior. Positive autocorrelation indicates trend following;
        negative indicates mean reversion. The raw autocorrelation in [-1, 1]
        is mapped to [0, 1] where 0.5 represents zero autocorrelation.

        수익률의 래그-1 자기상관을 추세추종 행동의 대리 변수로 사용합니다.
        양의 자기상관은 추세추종, 음의 자기상관은 평균회귀를 나타냅니다.
        [-1, 1] 범위의 자기상관을 [0, 1]로 매핑하며, 0.5가 자기상관 0에 해당합니다.

        Args:
            returns: Daily return series. 일별 수익률 시리즈.

        Returns:
            float: Trend sensitivity in [0, 1]. 추세 민감도 [0, 1].
        """
        if len(returns) < 10:
            return 0.5
        autocorr = self._autocorrelation(returns, lag=1)
        return float(np.clip((autocorr + 1.0) / 2.0, 0.0, 1.0))

    def _computeVolatilityPreference(self, returns: np.ndarray) -> float:
        """
        Compute volatility preference from the correlation between
        rolling volatility and rolling returns.
        롤링 변동성과 롤링 수익률 간의 상관관계로부터 변동성 선호도를 계산합니다.

        A positive correlation means the strategy earns more in volatile
        periods; negative means it suffers. Mapped from [-1, 1] to [0, 1].

        양의 상관관계는 변동성이 높은 기간에 더 많이 벌었음을 의미하고,
        음의 상관관계는 손해를 보았음을 의미합니다. [-1, 1]에서 [0, 1]로 매핑됩니다.

        Args:
            returns: Daily return series. 일별 수익률 시리즈.

        Returns:
            float: Volatility preference in [0, 1]. 변동성 선호도 [0, 1].
        """
        if len(returns) < self.ROLLING_WINDOW * 2:
            return 0.5
        returnsSeries = pd.Series(returns)
        rollingVol = returnsSeries.rolling(window=self.ROLLING_WINDOW).std()
        rollingRet = returnsSeries.rolling(window=self.ROLLING_WINDOW).mean()
        validMask = rollingVol.notna() & rollingRet.notna()
        if validMask.sum() < 5:
            return 0.5
        corr = rollingVol[validMask].corr(rollingRet[validMask])
        if np.isnan(corr):
            return 0.5
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _computeHoldingPeriodProfile(self, trades: List[Trade]) -> float:
        """
        Compute holding period profile from the average holding days.
        평균 보유일수로부터 보유 기간 프로파일을 계산합니다.

        Normalizes the average holding period against MAX_HOLDING_DAYS (252
        trading days = 1 year) using a logarithmic scale for better
        discrimination at shorter holding periods.

        MAX_HOLDING_DAYS(252 거래일 = 1년) 대비 평균 보유 기간을
        로그 스케일로 정규화하여 짧은 보유 기간에서의 변별력을 높입니다.

        Args:
            trades: List of closed Trade objects. 청산된 거래 목록.

        Returns:
            float: Holding period profile in [0, 1]. 0=short-term, 1=long-term.
                보유 기간 프로파일 [0, 1]. 0=단기, 1=장기.
        """
        if not trades:
            return 0.5
        holdingDays = [t.holdingDays for t in trades if t.holdingDays > 0]
        if not holdingDays:
            return 0.0
        avgDays = float(np.mean(holdingDays))
        normalized = np.log1p(avgDays) / np.log1p(self.MAX_HOLDING_DAYS)
        return float(np.clip(normalized, 0.0, 1.0))

    def _computeDrawdownTolerance(
        self,
        equityCurve: pd.Series,
        totalReturn: float,
    ) -> float:
        """
        Compute drawdown tolerance from max drawdown relative to total return.
        최대 낙폭 대비 총 수익률로부터 낙폭 허용도를 계산합니다.

        Strategies with large drawdowns relative to their returns are considered
        more drawdown-tolerant. The metric is based on the absolute max drawdown
        percentage, normalized via a sigmoid-like mapping.

        수익률 대비 큰 낙폭을 가진 전략은 더 높은 낙폭 허용도를 가진 것으로
        간주됩니다. 절대 최대 낙폭 백분율을 시그모이드형 매핑으로 정규화합니다.

        Args:
            equityCurve: Time-indexed equity values. 시간 인덱스 자산 곡선.
            totalReturn: Total return percentage from the backtest result.
                백테스트 결과의 총 수익률 (%).

        Returns:
            float: Drawdown tolerance in [0, 1]. 낙폭 허용도 [0, 1].
        """
        if equityCurve is None or len(equityCurve) < 2:
            return 0.5
        cumMax = equityCurve.cummax()
        drawdown = (equityCurve - cumMax) / cumMax
        maxDd = abs(float(drawdown.min()))
        normalized = 2.0 / (1.0 + np.exp(-5.0 * maxDd)) - 1.0
        return float(np.clip(normalized, 0.0, 1.0))

    def _computeWinRateProfile(self, winRate: float) -> float:
        """
        Compute win rate profile from the strategy's win rate.
        전략의 승률로부터 승률 프로파일을 계산합니다.

        Directly normalizes the win rate percentage to [0, 1].
        A win rate of 0% maps to 0.0 and 100% maps to 1.0.

        승률 백분율을 [0, 1]로 직접 정규화합니다.
        승률 0%는 0.0, 100%는 1.0에 매핑됩니다.

        Args:
            winRate: Win rate as a percentage (0-100). 승률 (0-100, %).

        Returns:
            float: Win rate profile in [0, 1]. 승률 프로파일 [0, 1].
        """
        return float(np.clip(winRate / 100.0, 0.0, 1.0))

    def _computeMarketRegimeDependence(self, returns: np.ndarray) -> float:
        """
        Compute market regime dependence from rolling performance variance.
        롤링 성과 분산으로부터 시장 레짐 의존도를 계산합니다.

        Measures the coefficient of variation of rolling-window mean returns.
        High variance in rolling performance suggests the strategy is
        sensitive to changing market regimes.

        롤링 윈도우 평균 수익률의 변동 계수를 측정합니다.
        롤링 성과의 높은 분산은 전략이 시장 레짐 변화에 민감함을 시사합니다.

        Args:
            returns: Daily return series. 일별 수익률 시리즈.

        Returns:
            float: Market regime dependence in [0, 1]. 시장 레짐 의존도 [0, 1].
        """
        if len(returns) < self.ROLLING_WINDOW * 3:
            return 0.5
        returnsSeries = pd.Series(returns)
        rollingMean = returnsSeries.rolling(window=self.ROLLING_WINDOW).mean().dropna()
        if len(rollingMean) < 5 or rollingMean.std() == 0:
            return 0.0
        cv = float(rollingMean.std() / (abs(rollingMean.mean()) + 1e-10))
        normalized = 2.0 / (1.0 + np.exp(-3.0 * cv)) - 1.0
        return float(np.clip(normalized, 0.0, 1.0))

    def _computeConcentrationLevel(self, trades: List[Trade]) -> float:
        """
        Compute position concentration level from trade size distribution.
        거래 크기 분포로부터 포지션 집중도를 계산합니다.

        Uses the Gini coefficient of trade entry values as a measure of
        concentration. A Gini of 0 indicates perfectly equal trade sizes
        (diversified), while 1 indicates maximum inequality (concentrated).

        거래 진입 금액의 지니 계수를 집중도 척도로 사용합니다.
        지니 계수 0은 완전히 균등한 거래 크기(분산), 1은 최대 불균등(집중)을 나타냅니다.

        Args:
            trades: List of closed Trade objects. 청산된 거래 목록.

        Returns:
            float: Concentration level in [0, 1]. 포지션 집중도 [0, 1].
        """
        if len(trades) < 2:
            return 0.5
        values = np.array([t.entryPrice * t.quantity for t in trades], dtype=np.float64)
        if np.all(values == 0):
            return 0.0
        return float(np.clip(self._giniCoefficient(values), 0.0, 1.0))

    def _computeTradingFrequency(
        self,
        trades: List[Trade],
        equityCurve: pd.Series,
    ) -> float:
        """
        Compute normalized trading frequency (trades per year).
        정규화된 거래 빈도(연간 거래 수)를 계산합니다.

        Normalizes the annual trade count against MAX_TRADES_PER_YEAR (252)
        using a logarithmic scale for better discrimination across the
        frequency spectrum.

        연간 거래 수를 MAX_TRADES_PER_YEAR(252) 대비 로그 스케일로 정규화하여
        빈도 스펙트럼 전체에서의 변별력을 높입니다.

        Args:
            trades: List of closed Trade objects. 청산된 거래 목록.
            equityCurve: Time-indexed equity values for period calculation.
                기간 계산을 위한 시간 인덱스 자산 곡선.

        Returns:
            float: Trading frequency in [0, 1]. 0=low, 1=high.
                거래 빈도 [0, 1]. 0=저빈도, 1=고빈도.
        """
        if not trades or equityCurve is None or len(equityCurve) < 2:
            return 0.0
        tradingDays = len(equityCurve)
        years = tradingDays / 252.0
        if years <= 0:
            return 0.0
        tradesPerYear = len(trades) / years
        normalized = np.log1p(tradesPerYear) / np.log1p(self.MAX_TRADES_PER_YEAR)
        return float(np.clip(normalized, 0.0, 1.0))

    def _computeRiskRewardRatio(self, trades: List[Trade]) -> float:
        """
        Compute normalized risk/reward ratio from average win vs average loss.
        평균 수익 대비 평균 손실로부터 정규화된 리스크-리워드 비율을 계산합니다.

        Computes the ratio of average winning trade PnL to the absolute value
        of average losing trade PnL, then normalizes using a logarithmic
        mapping so that a ratio of 1:1 maps to ~0.5.

        평균 수익 거래 PnL과 평균 손실 거래 PnL 절대값의 비율을 계산한 뒤,
        1:1 비율이 ~0.5에 매핑되도록 로그 매핑으로 정규화합니다.

        Args:
            trades: List of closed Trade objects. 청산된 거래 목록.

        Returns:
            float: Normalized risk/reward ratio in [0, 1]. 정규화된 RR 비율 [0, 1].
        """
        if not trades:
            return 0.5
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        if not wins or not losses:
            return 0.5 if not wins and not losses else (1.0 if wins else 0.0)
        avgWin = float(np.mean(wins))
        avgLoss = float(np.mean(np.abs(losses)))
        if avgLoss == 0:
            return 1.0
        ratio = avgWin / avgLoss
        normalized = np.log1p(ratio) / np.log1p(5.0)
        return float(np.clip(normalized, 0.0, 1.0))

    def _computeMomentumExposure(self, pnlSeries: np.ndarray) -> float:
        """
        Compute momentum exposure from lag-1 autocorrelation of trade PnL.
        거래 PnL의 래그-1 자기상관으로부터 모멘텀 익스포져를 계산합니다.

        Positive autocorrelation in PnL means winning trades tend to be
        followed by more winners (momentum persistence). Mapped from [-1, 1]
        to [0, 1] where 0.5 is no autocorrelation.

        PnL의 양의 자기상관은 수익 거래 후 또 다른 수익 거래가 이어지는
        경향(모멘텀 지속성)을 의미합니다. [-1, 1]에서 [0, 1]로 매핑되며,
        0.5가 자기상관 없음에 해당합니다.

        Args:
            pnlSeries: Array of trade PnL values. 거래 PnL 값 배열.

        Returns:
            float: Momentum exposure in [0, 1]. 모멘텀 익스포져 [0, 1].
        """
        if len(pnlSeries) < 5:
            return 0.5
        autocorr = self._autocorrelation(pnlSeries, lag=1)
        return float(np.clip((autocorr + 1.0) / 2.0, 0.0, 1.0))

    def _computeDefensiveScore(
        self,
        equityCurve: pd.Series,
        trades: List[Trade],
    ) -> float:
        """
        Compute defensive score as a composite of drawdown management
        and win consistency.
        낙폭 관리와 승률 일관성의 복합 지표로 방어적 점수를 계산합니다.

        Combines three sub-scores:
        1. Drawdown control: Lower max drawdown = more defensive
        2. Win consistency: Higher win rate = more defensive
        3. Recovery efficiency: Shorter drawdown durations = more defensive

        세 가지 하위 점수를 결합합니다:
        1. 낙폭 통제: 낮은 최대 낙폭 = 더 방어적
        2. 승률 일관성: 높은 승률 = 더 방어적
        3. 회복 효율성: 짧은 낙폭 지속 기간 = 더 방어적

        Args:
            equityCurve: Time-indexed equity values. 시간 인덱스 자산 곡선.
            trades: List of closed Trade objects. 청산된 거래 목록.

        Returns:
            float: Defensive score in [0, 1]. 방어적 점수 [0, 1].
        """
        ddControl = 0.5
        if equityCurve is not None and len(equityCurve) >= 2:
            cumMax = equityCurve.cummax()
            drawdown = (equityCurve - cumMax) / cumMax
            maxDd = abs(float(drawdown.min()))
            ddControl = float(np.clip(1.0 - maxDd * 2.0, 0.0, 1.0))

            ddDurations = []
            inDd = False
            currentDuration = 0
            for val in drawdown.values:
                if val < 0:
                    inDd = True
                    currentDuration += 1
                else:
                    if inDd:
                        ddDurations.append(currentDuration)
                    inDd = False
                    currentDuration = 0
            if inDd:
                ddDurations.append(currentDuration)

            recoveryScore = 0.5
            if ddDurations:
                avgDdDuration = float(np.mean(ddDurations))
                recoveryScore = float(np.clip(1.0 - avgDdDuration / 60.0, 0.0, 1.0))
        else:
            recoveryScore = 0.5

        winConsistency = 0.5
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            winRate = wins / len(trades)
            winConsistency = float(np.clip(winRate, 0.0, 1.0))

        return float(np.clip(
            0.4 * ddControl + 0.3 * winConsistency + 0.3 * recoveryScore,
            0.0,
            1.0,
        ))

    @staticmethod
    def _autocorrelation(series: np.ndarray, lag: int = 1) -> float:
        """
        Compute the autocorrelation of a series at a given lag.
        주어진 래그에서 시리즈의 자기상관을 계산합니다.

        Uses the Pearson correlation between the series and its lagged version.
        Returns 0 for degenerate cases (constant series, insufficient data).

        시리즈와 래그된 버전 간의 피어슨 상관을 사용합니다.
        퇴화 케이스(상수 시리즈, 데이터 부족)에서는 0을 반환합니다.

        Args:
            series: 1D numeric array. 1차원 수치 배열.
            lag: Number of periods to lag. 래그 기간 수. Defaults to 1.

        Returns:
            float: Autocorrelation coefficient in [-1, 1]. 자기상관 계수 [-1, 1].
        """
        if len(series) <= lag + 1:
            return 0.0
        x = series[:-lag]
        y = series[lag:]
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        corrMatrix = np.corrcoef(x, y)
        corr = corrMatrix[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def _giniCoefficient(values: np.ndarray) -> float:
        """
        Compute the Gini coefficient of a distribution.
        분포의 지니 계수를 계산합니다.

        The Gini coefficient measures statistical dispersion, commonly used
        as a measure of inequality. Returns 0 for perfect equality and
        approaches 1 for maximal inequality.

        지니 계수는 통계적 분산을 측정하며, 불균등 척도로 널리 사용됩니다.
        완전 균등 시 0, 최대 불균등 시 1에 접근합니다.

        Args:
            values: 1D array of non-negative values. 비음수 1차원 배열.

        Returns:
            float: Gini coefficient in [0, 1]. 지니 계수 [0, 1].
        """
        values = np.abs(values).astype(np.float64)
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sortedVals = np.sort(values)
        n = len(sortedVals)
        index = np.arange(1, n + 1, dtype=np.float64)
        return float((2.0 * np.sum(index * sortedVals) / (n * np.sum(sortedVals))) - (n + 1.0) / n)

    @staticmethod
    def _bellCurveScore(value: float, center: float, width: float) -> float:
        """
        Compute a Gaussian bell curve score centered at a specific value.
        특정 값을 중심으로 한 가우시안 종형 곡선 점수를 계산합니다.

        Returns 1.0 when value equals center and decays as a Gaussian
        with the given width (standard deviation).

        값이 중심과 같을 때 1.0을 반환하고, 주어진 폭(표준편차)의
        가우시안으로 감쇠합니다.

        Args:
            value: The input value to score. 점수를 매길 입력 값.
            center: The center of the bell curve (peak at 1.0). 종형 곡선의 중심.
            width: Standard deviation controlling the curve width. 곡선 폭을 제어하는 표준편차.

        Returns:
            float: Score in [0, 1]. 점수 [0, 1].
        """
        return float(np.exp(-0.5 * ((value - center) / max(width, 1e-10)) ** 2))
