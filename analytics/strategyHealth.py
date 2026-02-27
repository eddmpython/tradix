"""
Strategy Health Score - Composite strategy fitness and overfitting risk analysis.
전략 건강 점수 - 종합 전략 적합성 및 과적합 위험 분석.

Computes a composite 0-100 health score measuring overfitting risk, parameter
stability, performance consistency, trade execution quality, and market
adaptability. Provides Korean-language warnings and actionable recommendations
for strategy improvement.

과적합 위험, 파라미터 안정성, 성과 일관성, 거래 실행 품질, 시장 적응력을
측정하는 0-100 종합 건강 점수를 산출합니다. 한국어 경고 및 전략 개선을 위한
실행 가능한 권고 사항을 제공합니다.

Features:
    - Overfitting risk detection via rolling Sharpe variance and return stability
    - Parameter stability proxy through trade signal consistency over time
    - Rolling 60-day window performance consistency analysis
    - Trade quality scoring from win/loss ratio, win rate stability, profit factor
    - Market adaptability measurement across volatility regimes
    - Automated Korean-language warnings and recommendations generation
    - Full diagnostic report with detailed metric breakdowns

Usage:
    from tradix.analytics.strategyHealth import StrategyHealthAnalyzer
    from tradix.engine import BacktestResult

    analyzer = StrategyHealthAnalyzer()
    score = analyzer.analyze(result)
    print(score.summary())
    diagnostics = analyzer.diagnose(result)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipyStats

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade


OVERFITTING_WEIGHT = 0.25
STABILITY_WEIGHT = 0.20
CONSISTENCY_WEIGHT = 0.25
TRADE_QUALITY_WEIGHT = 0.15
ADAPTABILITY_WEIGHT = 0.15

ROLLING_WINDOW = 60
SHARPE_ROLLING_WINDOW = 120
MIN_TRADES_FOR_ANALYSIS = 5

GRADE_THRESHOLDS = [
    (90, 'A+'),
    (80, 'A'),
    (70, 'B+'),
    (60, 'B'),
    (50, 'C+'),
    (40, 'C'),
    (30, 'D'),
]


def _assignGrade(score: float) -> str:
    """
    Assign a letter grade based on numeric score.
    점수에 기반한 등급을 부여합니다.

    Args:
        score (float): Numeric score from 0 to 100. 0~100 범위의 수치 점수.

    Returns:
        str: Letter grade from A+ to F. A+부터 F까지의 등급 문자열.
    """
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return 'F'


@dataclass
class StrategyHealthScore:
    """
    Composite strategy health score with sub-component breakdown.
    전략 건강 종합 점수 및 세부 구성 요소.

    Encapsulates the overall health assessment along with five sub-scores,
    Korean-language warnings for critical issues, and actionable
    recommendations for improvement.

    다섯 가지 하위 점수와 함께 전체 건강 평가를 캡슐화하며,
    심각한 문제에 대한 한국어 경고와 개선을 위한 실행 가능한
    권고 사항을 포함합니다.

    Attributes:
        overallHealth (float): Weighted composite health score (0-100). 가중 종합 건강 점수.
        overfittingRisk (float): Overfitting resistance score (0-100, 100=no overfitting).
            과적합 저항 점수 (100=과적합 없음).
        parameterStability (float): Parameter sensitivity score (0-100). 파라미터 민감도 점수.
        performanceConsistency (float): Rolling window consistency score (0-100).
            롤링 윈도우 일관성 점수.
        tradeQuality (float): Trade execution quality score (0-100). 거래 실행 품질 점수.
        marketAdaptability (float): Cross-regime adaptability score (0-100).
            체제 간 적응력 점수.
        grade (str): Letter grade from A+ to F. 등급 (A+~F).
        warnings (List[str]): Korean warning messages for critical issues.
            심각한 문제에 대한 한국어 경고 메시지.
        recommendations (List[str]): Korean improvement suggestions.
            한국어 개선 권고 사항.
        details (dict): Detailed metric breakdown. 상세 지표 분해.
    """
    overallHealth: float = 0.0
    overfittingRisk: float = 0.0
    parameterStability: float = 0.0
    performanceConsistency: float = 0.0
    tradeQuality: float = 0.0
    marketAdaptability: float = 0.0
    grade: str = 'F'
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """
        Generate a Korean-language summary of the strategy health score.
        전략 건강 점수의 한국어 요약을 생성합니다.

        Returns:
            str: Multi-line formatted summary string with scores, grade,
                warnings and recommendations.
                점수, 등급, 경고, 권고 사항이 포함된 여러 줄의 포맷 요약 문자열.
        """
        lines = [
            f"\n{'='*55}",
            f"  전략 건강 점수 (Strategy Health Score)",
            f"{'='*55}",
            f"  종합 건강도: {self.overallHealth:.1f}/100 [{self.grade}]",
            f"{'─'*55}",
            f"  과적합 저항:       {self.overfittingRisk:.1f}/100",
            f"  파라미터 안정성:   {self.parameterStability:.1f}/100",
            f"  성과 일관성:       {self.performanceConsistency:.1f}/100",
            f"  거래 품질:         {self.tradeQuality:.1f}/100",
            f"  시장 적응력:       {self.marketAdaptability:.1f}/100",
            f"{'─'*55}",
        ]

        if self.warnings:
            lines.append(f"  경고 ({len(self.warnings)}건):")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append(f"{'─'*55}")

        if self.recommendations:
            lines.append(f"  권고 사항 ({len(self.recommendations)}건):")
            for r in self.recommendations:
                lines.append(f"    - {r}")

        lines.append(f"{'='*55}")
        return '\n'.join(lines) + '\n'

    def __repr__(self) -> str:
        return (
            f"StrategyHealthScore(health={self.overallHealth:.1f} [{self.grade}], "
            f"overfit={self.overfittingRisk:.1f}, stability={self.parameterStability:.1f}, "
            f"consistency={self.performanceConsistency:.1f}, quality={self.tradeQuality:.1f}, "
            f"adapt={self.marketAdaptability:.1f}, "
            f"warnings={len(self.warnings)}, recs={len(self.recommendations)})"
        )


class StrategyHealthAnalyzer:
    """
    Analyzer for computing strategy health scores from backtest results.
    백테스트 결과로부터 전략 건강 점수를 계산하는 분석기.

    Evaluates strategy fitness across five dimensions: overfitting risk,
    parameter stability, performance consistency, trade execution quality,
    and market adaptability. Generates automated warnings and recommendations
    in Korean.

    다섯 가지 차원에서 전략 적합성을 평가합니다: 과적합 위험,
    파라미터 안정성, 성과 일관성, 거래 실행 품질, 시장 적응력.
    한국어로 자동 경고 및 권고 사항을 생성합니다.

    Example:
        >>> analyzer = StrategyHealthAnalyzer()
        >>> score = analyzer.analyze(backtestResult)
        >>> print(score.summary())
        >>> report = analyzer.diagnose(backtestResult)
    """

    def analyze(self, result: BacktestResult) -> StrategyHealthScore:
        """
        Perform comprehensive strategy health analysis on a backtest result.
        백테스트 결과에 대한 종합 전략 건강 분석을 수행합니다.

        Computes five sub-scores and combines them with predefined weights
        into a single composite health score with grade, warnings, and
        recommendations.

        다섯 가지 하위 점수를 계산하고, 사전 정의된 가중치로 결합하여
        등급, 경고, 권고 사항이 포함된 단일 종합 건강 점수를 산출합니다.

        Args:
            result (BacktestResult): Completed backtest result containing equity
                curve, trades, and metrics. 자산 곡선, 거래, 지표가 포함된 백테스트 결과.

        Returns:
            StrategyHealthScore: Composite health score with full breakdown.
                전체 분해가 포함된 종합 건강 점수.
        """
        equityCurve = result.equityCurve
        trades = [t for t in result.trades if t.isClosed]

        if len(equityCurve) < 20:
            return StrategyHealthScore(
                grade='F',
                warnings=['데이터가 부족하여 건강 점수를 산출할 수 없습니다.'],
                details={'error': 'insufficient_data'},
            )

        returns = equityCurve.pct_change().dropna()
        if len(returns) < 10:
            return StrategyHealthScore(
                grade='F',
                warnings=['수익률 데이터가 부족합니다.'],
                details={'error': 'insufficient_returns'},
            )

        overfitDetails = {}
        overfitScore = self._scoreOverfittingRisk(returns, trades, overfitDetails)

        stabilityDetails = {}
        stabilityScore = self._scoreParameterStability(trades, equityCurve, stabilityDetails)

        consistencyDetails = {}
        consistencyScore = self._scorePerformanceConsistency(returns, consistencyDetails)

        qualityDetails = {}
        qualityScore = self._scoreTradeQuality(trades, qualityDetails)

        adaptDetails = {}
        adaptScore = self._scoreMarketAdaptability(returns, adaptDetails)

        overallHealth = (
            overfitScore * OVERFITTING_WEIGHT
            + stabilityScore * STABILITY_WEIGHT
            + consistencyScore * CONSISTENCY_WEIGHT
            + qualityScore * TRADE_QUALITY_WEIGHT
            + adaptScore * ADAPTABILITY_WEIGHT
        )
        overallHealth = float(np.clip(overallHealth, 0.0, 100.0))

        details = {
            'overfitting': overfitDetails,
            'parameterStability': stabilityDetails,
            'performanceConsistency': consistencyDetails,
            'tradeQuality': qualityDetails,
            'marketAdaptability': adaptDetails,
            'weights': {
                'overfitting': OVERFITTING_WEIGHT,
                'stability': STABILITY_WEIGHT,
                'consistency': CONSISTENCY_WEIGHT,
                'tradeQuality': TRADE_QUALITY_WEIGHT,
                'adaptability': ADAPTABILITY_WEIGHT,
            },
        }

        score = StrategyHealthScore(
            overallHealth=overallHealth,
            overfittingRisk=overfitScore,
            parameterStability=stabilityScore,
            performanceConsistency=consistencyScore,
            tradeQuality=qualityScore,
            marketAdaptability=adaptScore,
            grade=_assignGrade(overallHealth),
            details=details,
        )

        score.warnings = self.getWarnings(score)
        score.recommendations = self._generateRecommendations(score)

        return score

    def diagnose(self, result: BacktestResult) -> dict:
        """
        Generate a full diagnostic report for a backtest result.
        백테스트 결과에 대한 전체 진단 보고서를 생성합니다.

        Produces a comprehensive diagnostic dictionary including the health
        score, all sub-scores, detailed metrics, statistical tests for
        overfitting, and a summary assessment.

        건강 점수, 모든 하위 점수, 상세 지표, 과적합에 대한 통계 검정,
        요약 평가를 포함하는 종합 진단 딕셔너리를 생성합니다.

        Args:
            result (BacktestResult): Completed backtest result. 완료된 백테스트 결과.

        Returns:
            dict: Comprehensive diagnostic report with sections for
                healthScore, equityCurveStats, returnsDistribution,
                tradeAnalysis, and assessmentSummary.
                healthScore, equityCurveStats, returnsDistribution,
                tradeAnalysis, assessmentSummary 섹션이 포함된 종합 진단 보고서.
        """
        healthScore = self.analyze(result)

        equityCurve = result.equityCurve
        returns = equityCurve.pct_change().dropna() if len(equityCurve) > 1 else pd.Series(dtype=float)
        trades = [t for t in result.trades if t.isClosed]

        equityCurveStats = {}
        if len(returns) > 5:
            equityCurveStats = {
                'totalDays': len(equityCurve),
                'meanDailyReturn': round(float(returns.mean()) * 100, 6),
                'dailyVolatility': round(float(returns.std()) * 100, 6),
                'annualizedReturn': round(float(returns.mean() * 252) * 100, 4),
                'annualizedVolatility': round(float(returns.std() * np.sqrt(252)) * 100, 4),
                'skewness': round(float(scipyStats.skew(returns.values)), 4),
                'kurtosis': round(float(scipyStats.kurtosis(returns.values, fisher=True)), 4),
            }

            if len(returns) >= 20:
                jbStat, jbPvalue = scipyStats.jarque_bera(returns.values)
                equityCurveStats['jarqueBeraStatistic'] = round(float(jbStat), 4)
                equityCurveStats['jarqueBeaPvalue'] = round(float(jbPvalue), 6)
                equityCurveStats['returnsNormallyDistributed'] = jbPvalue > 0.05

        returnsDistribution = {}
        if len(returns) > 5:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            returnsDistribution = {
                f'p{p}': round(float(np.percentile(returns.values, p)) * 100, 6)
                for p in percentiles
            }
            returnsDistribution['positiveRatio'] = round(
                float((returns > 0).sum() / len(returns)) * 100, 2
            )

        tradeAnalysis = {}
        if len(trades) >= MIN_TRADES_FOR_ANALYSIS:
            pnlValues = [t.pnl for t in trades]
            pnlPctValues = [t.pnlPercent for t in trades]
            holdingDaysValues = [t.holdingDays for t in trades if t.holdingDays > 0]

            tradeAnalysis = {
                'totalTrades': len(trades),
                'meanPnl': round(float(np.mean(pnlValues)), 2),
                'medianPnl': round(float(np.median(pnlValues)), 2),
                'stdPnl': round(float(np.std(pnlValues)), 2),
                'meanPnlPct': round(float(np.mean(pnlPctValues)), 4),
                'stdPnlPct': round(float(np.std(pnlPctValues)), 4),
                'meanHoldingDays': round(float(np.mean(holdingDaysValues)), 1) if holdingDaysValues else 0,
                'medianHoldingDays': round(float(np.median(holdingDaysValues)), 1) if holdingDaysValues else 0,
            }

        assessmentSummary = self._buildAssessmentSummary(healthScore)

        return {
            'healthScore': {
                'overall': healthScore.overallHealth,
                'grade': healthScore.grade,
                'overfittingRisk': healthScore.overfittingRisk,
                'parameterStability': healthScore.parameterStability,
                'performanceConsistency': healthScore.performanceConsistency,
                'tradeQuality': healthScore.tradeQuality,
                'marketAdaptability': healthScore.marketAdaptability,
            },
            'equityCurveStats': equityCurveStats,
            'returnsDistribution': returnsDistribution,
            'tradeAnalysis': tradeAnalysis,
            'warnings': healthScore.warnings,
            'recommendations': healthScore.recommendations,
            'assessmentSummary': assessmentSummary,
            'details': healthScore.details,
        }

    def getWarnings(self, score: StrategyHealthScore) -> List[str]:
        """
        Generate Korean warning messages based on strategy health scores.
        전략 건강 점수에 기반한 한국어 경고 메시지를 생성합니다.

        Produces warnings for each sub-score that falls below critical
        thresholds, alerting users to potential problems.

        임계값 이하로 떨어지는 각 하위 점수에 대한 경고를 생성하여
        사용자에게 잠재적 문제를 알립니다.

        Args:
            score (StrategyHealthScore): Computed health score. 계산된 건강 점수.

        Returns:
            List[str]: Ordered list of Korean warning messages.
                한국어 경고 메시지의 정렬된 목록.
        """
        warnings = []

        if score.overfittingRisk < 30:
            warnings.append(
                "과적합 위험이 매우 높습니다. 전략이 과거 데이터에 과도하게 최적화되어 "
                "실전에서 성과가 크게 저하될 수 있습니다."
            )
        elif score.overfittingRisk < 50:
            warnings.append(
                "과적합 징후가 감지됩니다. 워크포워드 분석으로 검증이 필요합니다."
            )

        if score.parameterStability < 30:
            warnings.append(
                "파라미터 안정성이 매우 낮습니다. 전략 신호가 시간에 따라 "
                "불규칙하게 변동하고 있습니다."
            )
        elif score.parameterStability < 50:
            warnings.append(
                "파라미터 안정성이 보통 이하입니다. 거래 빈도 변동을 확인하세요."
            )

        if score.performanceConsistency < 30:
            warnings.append(
                "성과 일관성이 매우 낮습니다. 특정 기간의 수익에 크게 의존하는 "
                "전략일 수 있습니다."
            )
        elif score.performanceConsistency < 50:
            warnings.append(
                "성과 변동이 큽니다. 롤링 기간별 수익률 분산을 점검하세요."
            )

        if score.tradeQuality < 30:
            warnings.append(
                "거래 품질이 매우 낮습니다. 승률과 손익비를 근본적으로 재검토하세요."
            )
        elif score.tradeQuality < 50:
            warnings.append(
                "거래 품질 개선이 필요합니다. 진입/청산 시점 최적화를 검토하세요."
            )

        if score.marketAdaptability < 30:
            warnings.append(
                "시장 적응력이 매우 부족합니다. 특정 시장 상황에서만 작동하는 "
                "전략일 수 있습니다."
            )
        elif score.marketAdaptability < 50:
            warnings.append(
                "시장 환경 변화에 대한 적응력이 부족합니다."
            )

        if score.overallHealth < 30:
            warnings.append(
                "전략 전반의 건강도가 심각하게 낮습니다. 실전 투입 전 "
                "근본적인 재설계가 필요합니다."
            )

        return warnings

    def _generateRecommendations(self, score: StrategyHealthScore) -> List[str]:
        """
        Generate Korean improvement recommendations based on health scores.
        건강 점수에 기반한 한국어 개선 권고 사항을 생성합니다.

        Analyzes each sub-score and produces specific, actionable
        recommendations prioritized by severity.

        각 하위 점수를 분석하고, 심각도에 따라 우선순위가 매겨진
        구체적이고 실행 가능한 권고 사항을 생성합니다.

        Args:
            score (StrategyHealthScore): Computed health score. 계산된 건강 점수.

        Returns:
            List[str]: Prioritized list of Korean recommendations.
                우선순위가 매겨진 한국어 권고 사항 목록.
        """
        recommendations = []

        if score.overfittingRisk < 60:
            recommendations.append(
                "워크포워드(Walk-Forward) 분석을 실행하여 과적합 여부를 확인하세요. "
                "Out-of-sample 성과가 In-sample 대비 50% 이상 유지되어야 합니다."
            )

        if score.parameterStability < 60:
            recommendations.append(
                "전략 파라미터를 단순화하세요. 파라미터 수를 줄이고, "
                "넓은 범위에서 안정적인 값을 선택하세요."
            )

        if score.performanceConsistency < 60:
            recommendations.append(
                "롤링 윈도우별 성과를 분석하여 특정 기간에 의존하는지 확인하세요. "
                "월별/분기별 수익률 분포가 균일해야 합니다."
            )

        if score.tradeQuality < 60:
            recommendations.append(
                "진입 시점과 청산 시점의 정밀도를 개선하세요. "
                "승률과 평균 손익비(RR ratio) 중 하나라도 개선하면 "
                "전체 수익성이 향상됩니다."
            )

        if score.marketAdaptability < 60:
            recommendations.append(
                "다양한 시장 레짐(상승/횡보/하락, 고변동/저변동)에서 "
                "전략 성과를 개별적으로 검증하세요."
            )

        if score.overallHealth >= 80:
            recommendations.append(
                "전략 건강도가 우수합니다. 현재 체계를 유지하면서 "
                "세부 파라미터 미세 조정을 진행하세요."
            )

        if not recommendations:
            recommendations.append(
                "전략이 양호한 상태입니다. 정기적인 성과 모니터링을 유지하세요."
            )

        return recommendations

    def _scoreOverfittingRisk(
        self,
        returns: pd.Series,
        trades: List[Trade],
        details: dict
    ) -> float:
        """
        Score overfitting risk using rolling Sharpe variance and return patterns.
        롤링 샤프 분산과 수익률 패턴을 사용하여 과적합 위험을 점수화합니다.

        High variance in rolling Sharpe ratios suggests the strategy's edge
        is unstable across time periods, indicating potential overfitting.
        Also checks for suspiciously concentrated returns.

        롤링 샤프 비율의 높은 분산은 전략의 우위가 시간대에 따라 불안정하여
        과적합 가능성을 나타냅니다. 의심스러울 정도로 집중된 수익률도 확인합니다.

        Args:
            returns (pd.Series): Daily return series. 일별 수익률 시리즈.
            trades (List[Trade]): List of closed trades. 청산된 거래 목록.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Overfitting risk score (0-100, 100=no overfitting).
                과적합 위험 점수 (0-100, 100=과적합 없음).
        """
        if len(returns) < SHARPE_ROLLING_WINDOW + 10:
            rollWindow = max(30, len(returns) // 3)
        else:
            rollWindow = SHARPE_ROLLING_WINDOW

        if len(returns) < rollWindow + 5:
            details['error'] = 'insufficient_data_for_rolling'
            return 50.0

        rollingMean = returns.rolling(window=rollWindow, min_periods=rollWindow // 2).mean()
        rollingStd = returns.rolling(window=rollWindow, min_periods=rollWindow // 2).std()
        rollingSharpe = (rollingMean / rollingStd).dropna()
        rollingSharpe = rollingSharpe.replace([np.inf, -np.inf], np.nan).dropna()

        if len(rollingSharpe) < 3:
            details['error'] = 'insufficient_rolling_sharpe_data'
            return 50.0

        sharpeStd = float(rollingSharpe.std())
        sharpeMean = float(rollingSharpe.mean())
        sharpeCoV = sharpeStd / abs(sharpeMean) if abs(sharpeMean) > 1e-8 else sharpeStd * 100

        sharpeConsistencyScore = max(0.0, 100.0 - sharpeCoV * 30.0)

        nSegments = min(4, max(2, len(returns) // 60))
        segmentLength = len(returns) // nSegments
        segmentReturns = []

        for i in range(nSegments):
            segStart = i * segmentLength
            segEnd = (i + 1) * segmentLength if i < nSegments - 1 else len(returns)
            segReturn = float(returns.iloc[segStart:segEnd].sum())
            segmentReturns.append(segReturn)

        segmentArr = np.array(segmentReturns)
        totalReturn = float(returns.sum())

        if abs(totalReturn) > 1e-8:
            maxSegmentContribution = float(np.max(np.abs(segmentArr))) / abs(totalReturn)
        else:
            maxSegmentContribution = 1.0

        concentrationScore = max(0.0, 100.0 - (maxSegmentContribution - 1.0 / nSegments) * 200.0)
        concentrationScore = min(100.0, concentrationScore)

        positiveSegments = np.sum(segmentArr > 0)
        segmentPositiveRatio = positiveSegments / nSegments
        segmentDirectionScore = segmentPositiveRatio * 100.0

        compositeScore = (
            sharpeConsistencyScore * 0.45
            + concentrationScore * 0.30
            + segmentDirectionScore * 0.25
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['rollingSharpeStd'] = round(sharpeStd, 4)
        details['rollingSharpeCoV'] = round(sharpeCoV, 4)
        details['nSegments'] = nSegments
        details['segmentReturns'] = [round(r * 100, 4) for r in segmentReturns]
        details['maxSegmentContribution'] = round(maxSegmentContribution, 4)
        details['positiveSegmentRatio'] = round(segmentPositiveRatio, 4)
        details['sharpeConsistencyScore'] = round(sharpeConsistencyScore, 2)
        details['concentrationScore'] = round(concentrationScore, 2)
        details['segmentDirectionScore'] = round(segmentDirectionScore, 2)

        return compositeScore

    def _scoreParameterStability(
        self,
        trades: List[Trade],
        equityCurve: pd.Series,
        details: dict
    ) -> float:
        """
        Score parameter stability through trade signal consistency over time.
        시간에 따른 거래 신호 일관성을 통해 파라미터 안정성을 점수화합니다.

        Measures how consistently the strategy generates trade signals across
        different time periods by analyzing trade frequency stability and
        holding period consistency.

        거래 빈도 안정성과 보유 기간 일관성을 분석하여 전략이 서로 다른
        시간대에 걸쳐 얼마나 일관되게 거래 신호를 생성하는지 측정합니다.

        Args:
            trades (List[Trade]): List of closed trades. 청산된 거래 목록.
            equityCurve (pd.Series): Time-indexed equity series. 시간 인덱스 자산 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Parameter stability score from 0 to 100. 0~100의 파라미터 안정성 점수.
        """
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            details['error'] = 'insufficient_trades'
            return 50.0

        curveStart = equityCurve.index.min()
        curveEnd = equityCurve.index.max()
        totalDays = (curveEnd - curveStart).days
        if totalDays <= 0:
            details['error'] = 'zero_duration'
            return 50.0

        nQuarters = max(2, totalDays // 90)
        quarterLength = totalDays / nQuarters
        quarterTradeCounts = np.zeros(nQuarters)

        for trade in trades:
            if trade.entryDate is not None:
                daysSinceStart = (pd.Timestamp(trade.entryDate) - curveStart).days
                quarterIdx = min(int(daysSinceStart / quarterLength), nQuarters - 1)
                quarterIdx = max(0, quarterIdx)
                quarterTradeCounts[quarterIdx] += 1

        meanTradesPerQuarter = float(np.mean(quarterTradeCounts))
        if meanTradesPerQuarter > 0:
            tradeFreqCoV = float(np.std(quarterTradeCounts)) / meanTradesPerQuarter
        else:
            tradeFreqCoV = 1.0

        freqStabilityScore = max(0.0, 100.0 - tradeFreqCoV * 50.0)

        holdingDays = np.array([t.holdingDays for t in trades if t.holdingDays > 0])
        if len(holdingDays) >= 3:
            meanHolding = float(np.mean(holdingDays))
            holdingCoV = float(np.std(holdingDays)) / meanHolding if meanHolding > 0 else 1.0
            holdingStabilityScore = max(0.0, 100.0 - holdingCoV * 40.0)
        else:
            holdingCoV = 0.0
            holdingStabilityScore = 50.0

        pnlPctValues = np.array([t.pnlPercent for t in trades])
        if len(pnlPctValues) >= 5:
            halfPoint = len(pnlPctValues) // 2
            firstHalfMean = float(np.mean(pnlPctValues[:halfPoint]))
            secondHalfMean = float(np.mean(pnlPctValues[halfPoint:]))

            overallMean = float(np.mean(pnlPctValues))
            if abs(overallMean) > 1e-6:
                pnlDrift = abs(secondHalfMean - firstHalfMean) / abs(overallMean)
            else:
                pnlDrift = abs(secondHalfMean - firstHalfMean) * 10
            driftScore = max(0.0, 100.0 - pnlDrift * 30.0)
        else:
            pnlDrift = 0.0
            driftScore = 50.0

        compositeScore = (
            freqStabilityScore * 0.40
            + holdingStabilityScore * 0.30
            + driftScore * 0.30
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['nQuarters'] = nQuarters
        details['quarterTradeCounts'] = quarterTradeCounts.tolist()
        details['tradeFreqCoV'] = round(tradeFreqCoV, 4)
        details['holdingDaysCoV'] = round(holdingCoV, 4)
        details['pnlDrift'] = round(pnlDrift, 4)
        details['freqStabilityScore'] = round(freqStabilityScore, 2)
        details['holdingStabilityScore'] = round(holdingStabilityScore, 2)
        details['driftScore'] = round(driftScore, 2)

        return compositeScore

    def _scorePerformanceConsistency(self, returns: pd.Series, details: dict) -> float:
        """
        Score rolling 60-day window return consistency.
        60일 롤링 윈도우 수익률 일관성을 점수화합니다.

        Measures how stable the strategy's performance is across consecutive
        rolling windows using coefficient of variation, positive window ratio,
        and comparison of best/worst windows.

        변동 계수, 양의 윈도우 비율, 최상/최악 윈도우 비교를 사용하여
        연속 롤링 윈도우에 걸친 전략 성과의 안정성을 측정합니다.

        Args:
            returns (pd.Series): Daily return series. 일별 수익률 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Performance consistency score from 0 to 100.
                0~100의 성과 일관성 점수.
        """
        window = min(ROLLING_WINDOW, max(20, len(returns) // 4))

        if len(returns) < window + 5:
            details['error'] = 'insufficient_data'
            return 50.0

        rollingReturn = returns.rolling(window=window, min_periods=window // 2).sum()
        rollingReturn = rollingReturn.dropna()

        if len(rollingReturn) < 3:
            details['error'] = 'insufficient_rolling_data'
            return 50.0

        rollingMean = float(rollingReturn.mean())
        rollingStd = float(rollingReturn.std())

        if abs(rollingMean) > 1e-8:
            rollingCoV = rollingStd / abs(rollingMean)
        else:
            rollingCoV = rollingStd * 100

        consistencyScore = max(0.0, 100.0 - rollingCoV * 20.0)

        positiveWindows = (rollingReturn > 0).sum()
        positiveRatio = float(positiveWindows / len(rollingReturn))
        positiveScore = positiveRatio * 100.0

        bestWindow = float(rollingReturn.max())
        worstWindow = float(rollingReturn.min())

        if abs(bestWindow) > 1e-8:
            asymmetryRatio = abs(worstWindow / bestWindow)
        else:
            asymmetryRatio = abs(worstWindow) * 100

        asymmetryScore = max(0.0, 100.0 - asymmetryRatio * 40.0)

        compositeScore = (
            consistencyScore * 0.40
            + positiveScore * 0.35
            + asymmetryScore * 0.25
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['windowSize'] = window
        details['rollingReturnMean'] = round(rollingMean * 100, 4)
        details['rollingReturnStd'] = round(rollingStd * 100, 4)
        details['rollingCoV'] = round(rollingCoV, 4)
        details['positiveWindowRatio'] = round(positiveRatio, 4)
        details['bestWindowReturn'] = round(bestWindow * 100, 4)
        details['worstWindowReturn'] = round(worstWindow * 100, 4)
        details['consistencyScore'] = round(consistencyScore, 2)
        details['positiveScore'] = round(positiveScore, 2)
        details['asymmetryScore'] = round(asymmetryScore, 2)

        return compositeScore

    def _scoreTradeQuality(self, trades: List[Trade], details: dict) -> float:
        """
        Score trade execution quality from win/loss statistics.
        승/패 통계로부터 거래 실행 품질을 점수화합니다.

        Evaluates average win-to-loss ratio, win rate stability over time,
        and profit factor to produce a composite trade quality metric.

        평균 승/패 비율, 시간에 따른 승률 안정성, 손익비를 평가하여
        종합 거래 품질 지표를 산출합니다.

        Args:
            trades (List[Trade]): List of closed trades. 청산된 거래 목록.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Trade quality score from 0 to 100. 0~100의 거래 품질 점수.
        """
        if len(trades) < MIN_TRADES_FOR_ANALYSIS:
            details['error'] = 'insufficient_trades'
            return 50.0

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        winRate = len(wins) / len(trades)

        avgWinPnl = float(np.mean([abs(t.pnl) for t in wins])) if wins else 0.0
        avgLossPnl = float(np.mean([abs(t.pnl) for t in losses])) if losses else 0.001

        winLossRatio = avgWinPnl / avgLossPnl if avgLossPnl > 0 else avgWinPnl * 100

        rrScore = min(100.0, max(0.0, winLossRatio * 30.0))

        winRateScore = 0.0
        if winRate >= 0.6:
            winRateScore = min(100.0, 60.0 + (winRate - 0.6) * 200.0)
        elif winRate >= 0.4:
            winRateScore = 40.0 + (winRate - 0.4) * 100.0
        else:
            winRateScore = winRate * 100.0

        totalProfit = sum(t.pnl for t in wins) if wins else 0.0
        totalLoss = abs(sum(t.pnl for t in losses)) if losses else 0.001
        profitFactor = totalProfit / totalLoss if totalLoss > 0 else totalProfit * 100

        profitFactorScore = min(100.0, max(0.0, profitFactor * 25.0))

        winRateStabilityScore = 50.0
        if len(trades) >= 10:
            halfPoint = len(trades) // 2
            firstHalfWinRate = sum(1 for t in trades[:halfPoint] if t.pnl > 0) / halfPoint
            secondHalfWinRate = sum(1 for t in trades[halfPoint:] if t.pnl > 0) / (len(trades) - halfPoint)
            winRateShift = abs(secondHalfWinRate - firstHalfWinRate)
            winRateStabilityScore = max(0.0, 100.0 - winRateShift * 200.0)

        compositeScore = (
            rrScore * 0.25
            + winRateScore * 0.25
            + profitFactorScore * 0.25
            + winRateStabilityScore * 0.25
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['winRate'] = round(winRate * 100, 2)
        details['winLossRatio'] = round(winLossRatio, 4)
        details['profitFactor'] = round(profitFactor, 4)
        details['avgWinPnl'] = round(avgWinPnl, 2)
        details['avgLossPnl'] = round(avgLossPnl, 2)
        details['rrScore'] = round(rrScore, 2)
        details['winRateScore'] = round(winRateScore, 2)
        details['profitFactorScore'] = round(profitFactorScore, 2)
        details['winRateStabilityScore'] = round(winRateStabilityScore, 2)

        return compositeScore

    def _scoreMarketAdaptability(self, returns: pd.Series, details: dict) -> float:
        """
        Score strategy performance variance across different volatility regimes.
        서로 다른 변동성 체제에서의 전략 성과 분산을 점수화합니다.

        Divides the backtest period into three volatility regimes (low, medium,
        high) using rolling 20-day volatility terciles, then measures performance
        consistency across regimes.

        20일 롤링 변동성 삼분위를 사용하여 백테스트 기간을 세 가지 변동성
        체제(저, 중, 고)로 분할하고, 체제 간 성과 일관성을 측정합니다.

        Args:
            returns (pd.Series): Daily return series. 일별 수익률 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Market adaptability score from 0 to 100. 0~100의 시장 적응력 점수.
        """
        if len(returns) < 60:
            details['error'] = 'insufficient_data'
            return 50.0

        rollingVol = returns.rolling(window=20, min_periods=10).std().dropna()

        if len(rollingVol) < 30:
            details['error'] = 'insufficient_vol_data'
            return 50.0

        tercile33 = float(np.percentile(rollingVol.values, 33))
        tercile66 = float(np.percentile(rollingVol.values, 66))

        lowVolMask = rollingVol <= tercile33
        midVolMask = (rollingVol > tercile33) & (rollingVol <= tercile66)
        highVolMask = rollingVol > tercile66

        alignedReturns = returns.reindex(rollingVol.index)

        lowVolReturns = alignedReturns[lowVolMask].dropna()
        midVolReturns = alignedReturns[midVolMask].dropna()
        highVolReturns = alignedReturns[highVolMask].dropna()

        regimeMeans = []
        regimeLabels = ['low', 'mid', 'high']
        regimeData = [lowVolReturns, midVolReturns, highVolReturns]

        for label, data in zip(regimeLabels, regimeData):
            if len(data) >= 5:
                regimeMeans.append(float(data.mean()))
            else:
                regimeMeans.append(None)

        validMeans = [m for m in regimeMeans if m is not None]
        if len(validMeans) < 2:
            details['error'] = 'insufficient_regime_data'
            return 50.0

        overallMean = float(np.mean(validMeans))
        meanVariance = float(np.std(validMeans))

        if abs(overallMean) > 1e-8:
            regimeCoV = meanVariance / abs(overallMean)
        else:
            regimeCoV = meanVariance * 100

        varianceScore = max(0.0, 100.0 - regimeCoV * 25.0)

        positiveRegimes = sum(1 for m in validMeans if m > 0)
        positiveRatio = positiveRegimes / len(validMeans)
        positiveScore = positiveRatio * 100.0

        highVolMean = regimeMeans[2] if regimeMeans[2] is not None else 0.0
        highVolScore = 50.0
        if highVolMean > 0:
            highVolScore = min(100.0, 60.0 + highVolMean * 2000)
        elif highVolMean < 0:
            highVolScore = max(0.0, 50.0 + highVolMean * 1000)

        compositeScore = (
            varianceScore * 0.35
            + positiveScore * 0.35
            + highVolScore * 0.30
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['volTercile33'] = round(tercile33 * 100, 4)
        details['volTercile66'] = round(tercile66 * 100, 4)
        details['lowVolDays'] = int(lowVolMask.sum())
        details['midVolDays'] = int(midVolMask.sum())
        details['highVolDays'] = int(highVolMask.sum())
        details['regimeMeans'] = {
            label: round(m * 100, 6) if m is not None else None
            for label, m in zip(regimeLabels, regimeMeans)
        }
        details['regimeCoV'] = round(regimeCoV, 4)
        details['positiveRegimeRatio'] = round(positiveRatio, 4)
        details['varianceScore'] = round(varianceScore, 2)
        details['positiveScore'] = round(positiveScore, 2)
        details['highVolScore'] = round(highVolScore, 2)

        return compositeScore

    def _buildAssessmentSummary(self, score: StrategyHealthScore) -> dict:
        """
        Build a structured assessment summary for the diagnostic report.
        진단 보고서를 위한 구조화된 평가 요약을 생성합니다.

        Args:
            score (StrategyHealthScore): Computed health score. 계산된 건강 점수.

        Returns:
            dict: Assessment summary with status and Korean description for each dimension.
                각 차원에 대한 상태와 한국어 설명이 포함된 평가 요약.
        """
        def _status(val: float) -> str:
            if val >= 80:
                return 'excellent'
            if val >= 60:
                return 'good'
            if val >= 40:
                return 'fair'
            return 'poor'

        def _statusKr(val: float) -> str:
            if val >= 80:
                return '우수'
            if val >= 60:
                return '양호'
            if val >= 40:
                return '보통'
            return '미흡'

        return {
            'overall': {
                'score': round(score.overallHealth, 1),
                'grade': score.grade,
                'status': _status(score.overallHealth),
                'statusKr': _statusKr(score.overallHealth),
            },
            'dimensions': {
                'overfittingRisk': {
                    'score': round(score.overfittingRisk, 1),
                    'status': _status(score.overfittingRisk),
                    'statusKr': _statusKr(score.overfittingRisk),
                    'description': '과적합 저항력',
                },
                'parameterStability': {
                    'score': round(score.parameterStability, 1),
                    'status': _status(score.parameterStability),
                    'statusKr': _statusKr(score.parameterStability),
                    'description': '파라미터 안정성',
                },
                'performanceConsistency': {
                    'score': round(score.performanceConsistency, 1),
                    'status': _status(score.performanceConsistency),
                    'statusKr': _statusKr(score.performanceConsistency),
                    'description': '성과 일관성',
                },
                'tradeQuality': {
                    'score': round(score.tradeQuality, 1),
                    'status': _status(score.tradeQuality),
                    'statusKr': _statusKr(score.tradeQuality),
                    'description': '거래 품질',
                },
                'marketAdaptability': {
                    'score': round(score.marketAdaptability, 1),
                    'status': _status(score.marketAdaptability),
                    'statusKr': _statusKr(score.marketAdaptability),
                    'description': '시장 적응력',
                },
            },
        }
