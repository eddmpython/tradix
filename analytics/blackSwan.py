"""
Black Swan Defense Score - Strategy resilience analysis against extreme market events.
블랙스완 방어 점수 - 극단적 시장 이벤트에 대한 전략 복원력 분석.

Computes a composite 0-100 score measuring how well a trading strategy
withstands tail risk events, recovers from severe drawdowns, adapts to
volatility regime changes, and performs during historical crisis periods.

극단적 꼬리 위험, 심각한 낙폭으로부터의 회복, 변동성 체제 변화 적응력,
역사적 위기 기간 동안의 성과를 종합적으로 평가하여 0-100 점수를 산출합니다.

Features:
    - Tail risk scoring via kurtosis, skewness, and VaR tail ratio analysis
    - Recovery speed measurement from significant drawdown events
    - Drawdown resilience relative to total strategy return
    - Volatility regime adaptation scoring
    - Historical crisis period overlap detection and performance measurement
    - Weighted composite scoring with letter grade assignment
    - Korean-language recommendations for score improvement

Usage:
    from tradix.analytics.blackSwan import BlackSwanAnalyzer
    from tradix.engine import BacktestResult

    analyzer = BlackSwanAnalyzer()
    score = analyzer.analyze(result)
    print(score.summary())
    recommendations = analyzer.getRecommendations(score)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipyStats

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade


TAIL_RISK_WEIGHT = 0.25
RECOVERY_WEIGHT = 0.20
DRAWDOWN_WEIGHT = 0.25
VOL_ADAPT_WEIGHT = 0.15
CRISIS_WEIGHT = 0.15

DRAWDOWN_THRESHOLD = 0.05

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
class BlackSwanScore:
    """
    Composite Black Swan defense score with sub-component breakdown.
    블랙스완 방어 종합 점수 및 세부 구성 요소.

    Encapsulates the overall resilience score along with five sub-scores
    measuring distinct aspects of a strategy's ability to withstand
    extreme market conditions.

    극단적 시장 상황에 대한 전략의 내성을 측정하는 다섯 가지 하위 점수와
    함께 전체 복원력 점수를 캡슐화합니다.

    Attributes:
        overallScore (float): Weighted composite score (0-100). 가중 종합 점수.
        tailRiskScore (float): Tail risk handling score (0-100). 꼬리 위험 대응 점수.
        recoverySpeed (float): Drawdown recovery speed score (0-100). 낙폭 회복 속도 점수.
        drawdownResilience (float): Drawdown resistance score (0-100). 낙폭 저항 점수.
        volatilityAdaptation (float): Volatility adaptation score (0-100). 변동성 적응 점수.
        crisisPerformance (float): Historical crisis performance score (0-100). 위기 기간 성과 점수.
        grade (str): Letter grade from A+ to F. 등급 (A+~F).
        details (dict): Detailed metric breakdown. 상세 지표 분해.
    """
    overallScore: float = 0.0
    tailRiskScore: float = 0.0
    recoverySpeed: float = 0.0
    drawdownResilience: float = 0.0
    volatilityAdaptation: float = 0.0
    crisisPerformance: float = 0.0
    grade: str = 'F'
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """
        Generate a Korean-language summary of the Black Swan defense score.
        블랙스완 방어 점수의 한국어 요약을 생성합니다.

        Returns:
            str: Multi-line formatted summary string.
                여러 줄로 구성된 포맷 요약 문자열.
        """
        return (
            f"\n{'='*55}\n"
            f"  블랙스완 방어 점수 (Black Swan Defense Score)\n"
            f"{'='*55}\n"
            f"  종합 점수: {self.overallScore:.1f}/100 [{self.grade}]\n"
            f"{'─'*55}\n"
            f"  꼬리 위험 대응:    {self.tailRiskScore:.1f}/100\n"
            f"  회복 속도:         {self.recoverySpeed:.1f}/100\n"
            f"  낙폭 저항력:       {self.drawdownResilience:.1f}/100\n"
            f"  변동성 적응력:     {self.volatilityAdaptation:.1f}/100\n"
            f"  위기 기간 성과:    {self.crisisPerformance:.1f}/100\n"
            f"{'='*55}\n"
        )

    def __repr__(self) -> str:
        return (
            f"BlackSwanScore(overall={self.overallScore:.1f} [{self.grade}], "
            f"tail={self.tailRiskScore:.1f}, recovery={self.recoverySpeed:.1f}, "
            f"drawdown={self.drawdownResilience:.1f}, volAdapt={self.volatilityAdaptation:.1f}, "
            f"crisis={self.crisisPerformance:.1f})"
        )


class BlackSwanAnalyzer:
    """
    Analyzer for computing Black Swan defense scores from backtest results.
    백테스트 결과로부터 블랙스완 방어 점수를 계산하는 분석기.

    Evaluates strategy resilience across five dimensions: tail risk handling,
    recovery speed from drawdowns, drawdown severity resistance, volatility
    regime adaptation, and performance during known historical crisis periods.

    다섯 가지 차원에서 전략 복원력을 평가합니다: 꼬리 위험 대응,
    낙폭으로부터의 회복 속도, 낙폭 심각도 저항, 변동성 체제 적응,
    알려진 역사적 위기 기간 동안의 성과.

    Attributes:
        CRISIS_PERIODS (dict): Historical crisis date ranges for overlap detection.
            겹침 감지를 위한 역사적 위기 기간 날짜 범위.

    Example:
        >>> analyzer = BlackSwanAnalyzer()
        >>> score = analyzer.analyze(backtestResult)
        >>> print(score.summary())
        >>> recs = analyzer.getRecommendations(score)
    """

    CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
        'covid_2020': ('2020-02-19', '2020-03-23'),
        'financial_2008': ('2008-09-15', '2008-11-20'),
        'flash_crash_2010': ('2010-05-06', '2010-05-07'),
        'china_shock_2015': ('2015-08-11', '2015-08-25'),
        'covid_recovery': ('2020-03-23', '2020-06-08'),
        'rate_hike_2022': ('2022-01-03', '2022-06-16'),
    }

    CRISIS_LABELS: Dict[str, str] = {
        'covid_2020': '코로나19 폭락 (2020)',
        'financial_2008': '글로벌 금융위기 (2008)',
        'flash_crash_2010': '플래시 크래시 (2010)',
        'china_shock_2015': '중국 충격 (2015)',
        'covid_recovery': '코로나 회복기 (2020)',
        'rate_hike_2022': '금리 인상기 (2022)',
    }

    def analyze(self, result: BacktestResult) -> BlackSwanScore:
        """
        Perform comprehensive Black Swan defense analysis on a backtest result.
        백테스트 결과에 대한 종합 블랙스완 방어 분석을 수행합니다.

        Computes five sub-scores and combines them with predefined weights
        into a single composite score with a letter grade.

        다섯 가지 하위 점수를 계산하고, 사전 정의된 가중치로 결합하여
        등급이 포함된 단일 종합 점수를 산출합니다.

        Args:
            result (BacktestResult): Completed backtest result containing equity
                curve and trade history. 자산 곡선과 거래 내역이 포함된 백테스트 결과.

        Returns:
            BlackSwanScore: Composite defense score with sub-component breakdown.
                세부 구성 요소가 포함된 종합 방어 점수.
        """
        equityCurve = result.equityCurve
        if len(equityCurve) < 10:
            return BlackSwanScore(grade='F', details={'error': 'insufficient_data'})

        returns = equityCurve.pct_change().dropna()
        if len(returns) < 5:
            return BlackSwanScore(grade='F', details={'error': 'insufficient_returns'})

        tailDetails = {}
        tailRiskScore = self._scoreTailRisk(returns, tailDetails)

        recoveryDetails = {}
        recoveryScore = self._scoreRecoverySpeed(equityCurve, recoveryDetails)

        drawdownDetails = {}
        drawdownScore = self._scoreDrawdownResilience(equityCurve, result.totalReturn, drawdownDetails)

        volDetails = {}
        volScore = self._scoreVolatilityAdaptation(returns, volDetails)

        crisisDetails = {}
        crisisScore = self._scoreCrisisPerformance(equityCurve, crisisDetails)

        overallScore = (
            tailRiskScore * TAIL_RISK_WEIGHT
            + recoveryScore * RECOVERY_WEIGHT
            + drawdownScore * DRAWDOWN_WEIGHT
            + volScore * VOL_ADAPT_WEIGHT
            + crisisScore * CRISIS_WEIGHT
        )

        overallScore = float(np.clip(overallScore, 0.0, 100.0))

        details = {
            'tailRisk': tailDetails,
            'recovery': recoveryDetails,
            'drawdown': drawdownDetails,
            'volatilityAdaptation': volDetails,
            'crisisPerformance': crisisDetails,
            'weights': {
                'tailRisk': TAIL_RISK_WEIGHT,
                'recovery': RECOVERY_WEIGHT,
                'drawdown': DRAWDOWN_WEIGHT,
                'volatilityAdaptation': VOL_ADAPT_WEIGHT,
                'crisis': CRISIS_WEIGHT,
            },
        }

        return BlackSwanScore(
            overallScore=overallScore,
            tailRiskScore=tailRiskScore,
            recoverySpeed=recoveryScore,
            drawdownResilience=drawdownScore,
            volatilityAdaptation=volScore,
            crisisPerformance=crisisScore,
            grade=_assignGrade(overallScore),
            details=details,
        )

    def stressTestAgainstCrisis(self, result: BacktestResult) -> dict:
        """
        Test backtest performance against each known historical crisis period.
        각 역사적 위기 기간에 대해 백테스트 성과를 검증합니다.

        For each crisis period that overlaps with the backtest date range,
        computes the strategy's return, maximum drawdown, and volatility
        during that specific period.

        백테스트 기간과 겹치는 각 위기 기간에 대해 해당 기간 동안의
        전략 수익률, 최대 낙폭, 변동성을 계산합니다.

        Args:
            result (BacktestResult): Completed backtest result. 완료된 백테스트 결과.

        Returns:
            dict: Per-crisis performance breakdown with return, maxDrawdown,
                volatility, and overlap status for each period.
                각 기간별 수익률, 최대 낙폭, 변동성, 겹침 여부가 포함된
                위기별 성과 분해 딕셔너리.
        """
        equityCurve = result.equityCurve
        if len(equityCurve) < 2:
            return {}

        curveStart = equityCurve.index.min()
        curveEnd = equityCurve.index.max()
        report = {}

        for crisisName, (startStr, endStr) in self.CRISIS_PERIODS.items():
            crisisStart = pd.Timestamp(startStr)
            crisisEnd = pd.Timestamp(endStr)

            hasOverlap = crisisStart <= curveEnd and crisisEnd >= curveStart

            if not hasOverlap:
                report[crisisName] = {
                    'label': self.CRISIS_LABELS.get(crisisName, crisisName),
                    'overlap': False,
                    'periodReturn': None,
                    'maxDrawdown': None,
                    'volatility': None,
                }
                continue

            overlapStart = max(crisisStart, curveStart)
            overlapEnd = min(crisisEnd, curveEnd)

            crisisSlice = equityCurve.loc[overlapStart:overlapEnd]

            if len(crisisSlice) < 2:
                report[crisisName] = {
                    'label': self.CRISIS_LABELS.get(crisisName, crisisName),
                    'overlap': True,
                    'periodReturn': 0.0,
                    'maxDrawdown': 0.0,
                    'volatility': 0.0,
                    'tradingDays': len(crisisSlice),
                }
                continue

            periodReturn = (crisisSlice.iloc[-1] / crisisSlice.iloc[0] - 1) * 100

            cumMax = crisisSlice.cummax()
            drawdownSeries = (crisisSlice - cumMax) / cumMax
            maxDrawdown = float(drawdownSeries.min()) * 100

            crisisReturns = crisisSlice.pct_change().dropna()
            annualizedVol = float(crisisReturns.std() * np.sqrt(252)) * 100 if len(crisisReturns) > 1 else 0.0

            report[crisisName] = {
                'label': self.CRISIS_LABELS.get(crisisName, crisisName),
                'overlap': True,
                'periodReturn': float(periodReturn),
                'maxDrawdown': float(maxDrawdown),
                'volatility': float(annualizedVol),
                'tradingDays': len(crisisSlice),
            }

        return report

    def getRecommendations(self, score: BlackSwanScore) -> List[str]:
        """
        Generate Korean-language improvement recommendations based on scores.
        점수에 기반한 한국어 개선 권고 사항을 생성합니다.

        Analyzes each sub-score and produces actionable recommendations
        for areas where the strategy shows weakness.

        각 하위 점수를 분석하고, 전략이 약점을 보이는 영역에 대해
        실행 가능한 권고 사항을 생성합니다.

        Args:
            score (BlackSwanScore): Computed Black Swan defense score.
                계산된 블랙스완 방어 점수.

        Returns:
            List[str]: Ordered list of Korean recommendations.
                한국어 권고 사항의 정렬된 목록.
        """
        recommendations = []

        if score.tailRiskScore < 50:
            recommendations.append(
                "꼬리 위험 대응이 취약합니다. 손절매(stop-loss) 규칙을 강화하고, "
                "극단적 손실을 제한하는 포지션 사이징 전략을 도입하세요."
            )
        elif score.tailRiskScore < 70:
            recommendations.append(
                "꼬리 위험 관리가 보통 수준입니다. VaR 기반 동적 포지션 조절을 고려하세요."
            )

        if score.recoverySpeed < 50:
            recommendations.append(
                "낙폭 회복 속도가 매우 느립니다. 추세 반전 시 빠른 재진입 로직을 "
                "추가하거나, 분할 매수 전략을 활용하세요."
            )
        elif score.recoverySpeed < 70:
            recommendations.append(
                "회복 속도가 보통입니다. 낙폭 후 공격적 자산 배분 조정을 검토하세요."
            )

        if score.drawdownResilience < 50:
            recommendations.append(
                "낙폭 저항력이 심각하게 부족합니다. 최대 낙폭 제한 규칙(예: -15% 시 "
                "전체 청산)을 도입하고, 분산 투자를 강화하세요."
            )
        elif score.drawdownResilience < 70:
            recommendations.append(
                "낙폭 저항력을 개선할 수 있습니다. 트레일링 스톱이나 "
                "변동성 기반 포지션 축소 규칙을 적용하세요."
            )

        if score.volatilityAdaptation < 50:
            recommendations.append(
                "변동성 변화에 대한 적응력이 부족합니다. ATR 기반 동적 손절매나 "
                "변동성 스케일링 포지션 사이징을 적용하세요."
            )
        elif score.volatilityAdaptation < 70:
            recommendations.append(
                "변동성 적응력이 보통입니다. 변동성 레짐별 파라미터 조정을 검토하세요."
            )

        if score.crisisPerformance < 50:
            recommendations.append(
                "위기 기간 성과가 취약합니다. 시장 레짐 감지 로직을 추가하여 "
                "위기 시 자동으로 방어 모드로 전환하는 기능을 구현하세요."
            )
        elif score.crisisPerformance < 70:
            recommendations.append(
                "위기 기간 방어가 보통 수준입니다. 헤지 전략이나 "
                "역상관 자산 편입을 고려하세요."
            )

        if score.overallScore >= 80:
            recommendations.append(
                "전반적으로 블랙스완 방어력이 우수합니다. "
                "현재 위험 관리 체계를 유지하면서 세부 최적화를 진행하세요."
            )
        elif score.overallScore < 40:
            recommendations.append(
                "블랙스완 방어력이 매우 취약합니다. 전략의 위험 관리 체계를 "
                "근본적으로 재설계하는 것을 강력히 권고합니다."
            )

        if not recommendations:
            recommendations.append(
                "블랙스완 방어력이 양호합니다. 지속적인 모니터링을 권장합니다."
            )

        return recommendations

    def _scoreTailRisk(self, returns: pd.Series, details: dict) -> float:
        """
        Score tail risk handling from returns distribution analysis.
        수익률 분포 분석을 통해 꼬리 위험 대응 점수를 산출합니다.

        Evaluates excess kurtosis, negative skewness, and the ratio of
        CVaR(99%) to VaR(99%) to measure how extreme the left tail is.

        초과 첨도, 음의 왜도, CVaR(99%)/VaR(99%) 비율을 평가하여
        좌측 꼬리의 극단 정도를 측정합니다.

        Args:
            returns (pd.Series): Daily return series. 일별 수익률 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Tail risk score from 0 to 100. 0~100의 꼬리 위험 점수.
        """
        returnsArr = returns.dropna().values
        if len(returnsArr) < 20:
            details['error'] = 'insufficient_data'
            return 50.0

        excessKurtosis = float(scipyStats.kurtosis(returnsArr, fisher=True))
        skewness = float(scipyStats.skew(returnsArr))

        kurtosisScore = 100.0
        if excessKurtosis > 0:
            kurtosisScore = max(0.0, 100.0 - excessKurtosis * 8.0)

        skewnessScore = 100.0
        if skewness < 0:
            skewnessScore = max(0.0, 100.0 + skewness * 30.0)
        else:
            skewnessScore = min(100.0, 70.0 + skewness * 15.0)

        var99 = float(np.percentile(returnsArr, 1))
        tailReturns = returnsArr[returnsArr <= var99]
        cvar99 = float(np.mean(tailReturns)) if len(tailReturns) > 0 else var99

        tailRatio = abs(cvar99 / var99) if var99 != 0 else 1.0
        tailRatioScore = max(0.0, 100.0 - (tailRatio - 1.0) * 80.0)

        var95 = float(np.percentile(returnsArr, 5))

        compositeScore = (
            kurtosisScore * 0.35
            + skewnessScore * 0.30
            + tailRatioScore * 0.35
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['excessKurtosis'] = round(excessKurtosis, 4)
        details['skewness'] = round(skewness, 4)
        details['var95'] = round(var95 * 100, 4)
        details['var99'] = round(var99 * 100, 4)
        details['cvar99'] = round(cvar99 * 100, 4)
        details['tailRatio'] = round(tailRatio, 4)
        details['kurtosisScore'] = round(kurtosisScore, 2)
        details['skewnessScore'] = round(skewnessScore, 2)
        details['tailRatioScore'] = round(tailRatioScore, 2)

        return compositeScore

    def _scoreRecoverySpeed(self, equityCurve: pd.Series, details: dict) -> float:
        """
        Score the average speed of recovery from significant drawdowns.
        주요 낙폭으로부터의 평균 회복 속도를 점수화합니다.

        Identifies all drawdown events exceeding 5% depth, measures the
        number of trading days required to recover to previous highs,
        and converts recovery speed into a 0-100 score.

        5% 이상의 모든 낙폭 이벤트를 식별하고, 이전 고점으로 회복하는 데
        소요된 거래일 수를 측정하여 0-100 점수로 변환합니다.

        Args:
            equityCurve (pd.Series): Time-indexed equity series. 시간 인덱스 자산 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Recovery speed score from 0 to 100. 0~100의 회복 속도 점수.
        """
        if len(equityCurve) < 10:
            details['error'] = 'insufficient_data'
            return 50.0

        cumMax = equityCurve.cummax()
        drawdownPct = (equityCurve - cumMax) / cumMax

        recoveryPeriods = []
        inDrawdown = False
        drawdownStartIdx = 0
        peakValue = equityCurve.iloc[0]

        for i in range(len(equityCurve)):
            currentDrawdown = drawdownPct.iloc[i]

            if not inDrawdown and currentDrawdown < -DRAWDOWN_THRESHOLD:
                inDrawdown = True
                drawdownStartIdx = i

            if inDrawdown and currentDrawdown >= 0:
                recoveryDays = i - drawdownStartIdx
                recoveryPeriods.append(recoveryDays)
                inDrawdown = False

        if inDrawdown:
            unrecoveredDays = len(equityCurve) - drawdownStartIdx
            recoveryPeriods.append(unrecoveredDays * 1.5)

        if not recoveryPeriods:
            details['drawdownEvents'] = 0
            details['avgRecoveryDays'] = 0
            details['maxRecoveryDays'] = 0
            details['allRecovered'] = True
            return 95.0

        avgRecoveryDays = float(np.mean(recoveryPeriods))
        maxRecoveryDays = float(np.max(recoveryPeriods))
        medianRecoveryDays = float(np.median(recoveryPeriods))

        score = max(0.0, 100.0 - avgRecoveryDays * 1.2)

        if maxRecoveryDays > 120:
            penalty = min(20.0, (maxRecoveryDays - 120) * 0.1)
            score -= penalty

        if inDrawdown:
            score *= 0.8

        score = float(np.clip(score, 0.0, 100.0))

        details['drawdownEvents'] = len(recoveryPeriods)
        details['avgRecoveryDays'] = round(avgRecoveryDays, 1)
        details['medianRecoveryDays'] = round(medianRecoveryDays, 1)
        details['maxRecoveryDays'] = round(maxRecoveryDays, 1)
        details['allRecovered'] = not inDrawdown

        return score

    def _scoreDrawdownResilience(
        self,
        equityCurve: pd.Series,
        totalReturnPct: float,
        details: dict
    ) -> float:
        """
        Score drawdown severity relative to total strategy return.
        총 전략 수익률 대비 낙폭 심각도를 점수화합니다.

        Measures maximum drawdown, average drawdown depth, and the ratio
        of max drawdown to total return (pain-to-gain ratio).

        최대 낙폭, 평균 낙폭 깊이, 최대 낙폭 대 총수익률 비율
        (고통 대비 이익 비율)을 측정합니다.

        Args:
            equityCurve (pd.Series): Time-indexed equity series. 시간 인덱스 자산 시리즈.
            totalReturnPct (float): Total strategy return in percent. 총 전략 수익률(%).
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Drawdown resilience score from 0 to 100. 0~100의 낙폭 저항 점수.
        """
        if len(equityCurve) < 10:
            details['error'] = 'insufficient_data'
            return 50.0

        cumMax = equityCurve.cummax()
        drawdownSeries = (equityCurve - cumMax) / cumMax
        maxDrawdown = abs(float(drawdownSeries.min()))

        drawdownOnly = drawdownSeries[drawdownSeries < 0]
        avgDrawdown = abs(float(drawdownOnly.mean())) if len(drawdownOnly) > 0 else 0.0

        timeInDrawdown = len(drawdownOnly) / len(drawdownSeries) * 100

        maxDdScore = max(0.0, 100.0 - maxDrawdown * 100 * 2.5)

        totalReturnDecimal = totalReturnPct / 100.0 if totalReturnPct != 0 else 0.001
        painToGain = maxDrawdown / abs(totalReturnDecimal) if abs(totalReturnDecimal) > 0.001 else maxDrawdown * 100
        painToGainScore = max(0.0, 100.0 - painToGain * 25.0)

        timeScore = max(0.0, 100.0 - timeInDrawdown * 1.2)

        compositeScore = (
            maxDdScore * 0.45
            + painToGainScore * 0.30
            + timeScore * 0.25
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['maxDrawdownPct'] = round(maxDrawdown * 100, 4)
        details['avgDrawdownPct'] = round(avgDrawdown * 100, 4)
        details['timeInDrawdownPct'] = round(timeInDrawdown, 2)
        details['painToGainRatio'] = round(painToGain, 4)
        details['maxDdScore'] = round(maxDdScore, 2)
        details['painToGainScore'] = round(painToGainScore, 2)
        details['timeScore'] = round(timeScore, 2)

        return compositeScore

    def _scoreVolatilityAdaptation(self, returns: pd.Series, details: dict) -> float:
        """
        Score strategy performance across different volatility regimes.
        서로 다른 변동성 체제에서의 전략 성과를 점수화합니다.

        Splits the backtest period into high-volatility and low-volatility
        regimes using rolling 20-day volatility, then compares risk-adjusted
        performance (Sharpe-like ratio) across regimes.

        20일 롤링 변동성을 사용하여 백테스트 기간을 고변동성/저변동성
        체제로 분할하고, 체제 간 위험 조정 성과(샤프 유사 비율)를 비교합니다.

        Args:
            returns (pd.Series): Daily return series. 일별 수익률 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Volatility adaptation score from 0 to 100. 0~100의 변동성 적응 점수.
        """
        if len(returns) < 40:
            details['error'] = 'insufficient_data'
            return 50.0

        rollingVol = returns.rolling(window=20, min_periods=10).std()
        medianVol = rollingVol.median()

        highVolMask = rollingVol > medianVol
        lowVolMask = rollingVol <= medianVol

        highVolReturns = returns[highVolMask].dropna()
        lowVolReturns = returns[lowVolMask].dropna()

        if len(highVolReturns) < 5 or len(lowVolReturns) < 5:
            details['error'] = 'insufficient_regime_data'
            return 50.0

        highVolMean = float(highVolReturns.mean())
        highVolStd = float(highVolReturns.std())
        highVolSharpe = highVolMean / highVolStd if highVolStd > 0 else 0.0

        lowVolMean = float(lowVolReturns.mean())
        lowVolStd = float(lowVolReturns.std())
        lowVolSharpe = lowVolMean / lowVolStd if lowVolStd > 0 else 0.0

        if lowVolSharpe != 0:
            consistencyRatio = highVolSharpe / lowVolSharpe if lowVolSharpe > 0 else 0.0
        else:
            consistencyRatio = 1.0 if highVolSharpe >= 0 else 0.0

        consistencyScore = min(100.0, max(0.0, consistencyRatio * 50.0))

        highVolPositive = 70.0 if highVolMean > 0 else max(0.0, 30.0 + highVolMean * 1000)

        performanceDrop = lowVolMean - highVolMean if lowVolMean > 0 else 0
        stabilityScore = max(0.0, 80.0 - performanceDrop * 500)
        stabilityScore = min(100.0, stabilityScore)

        compositeScore = (
            consistencyScore * 0.40
            + highVolPositive * 0.35
            + stabilityScore * 0.25
        )
        compositeScore = float(np.clip(compositeScore, 0.0, 100.0))

        details['medianDailyVol'] = round(float(medianVol) * 100, 4)
        details['highVolDays'] = int(highVolMask.sum())
        details['lowVolDays'] = int(lowVolMask.sum())
        details['highVolMeanReturn'] = round(highVolMean * 100, 4)
        details['lowVolMeanReturn'] = round(lowVolMean * 100, 4)
        details['highVolSharpe'] = round(highVolSharpe, 4)
        details['lowVolSharpe'] = round(lowVolSharpe, 4)
        details['consistencyRatio'] = round(consistencyRatio, 4)

        return compositeScore

    def _scoreCrisisPerformance(self, equityCurve: pd.Series, details: dict) -> float:
        """
        Score actual strategy performance during historical crisis periods.
        역사적 위기 기간 동안의 실제 전략 성과를 점수화합니다.

        For each crisis period that overlaps with the equity curve, measures
        the strategy return and converts it to a score. Averages scores
        across all overlapping crises. Returns a neutral score if no
        overlap is detected.

        자산 곡선과 겹치는 각 위기 기간에 대해 전략 수익률을 측정하고
        점수로 변환합니다. 모든 겹치는 위기의 점수를 평균합니다.
        겹침이 감지되지 않으면 중립 점수를 반환합니다.

        Args:
            equityCurve (pd.Series): Time-indexed equity series. 시간 인덱스 자산 시리즈.
            details (dict): Mutable dict to populate with detail metrics.
                상세 지표를 채울 가변 딕셔너리.

        Returns:
            float: Crisis performance score from 0 to 100. 0~100의 위기 성과 점수.
        """
        if len(equityCurve) < 2:
            details['error'] = 'insufficient_data'
            return 50.0

        curveStart = equityCurve.index.min()
        curveEnd = equityCurve.index.max()

        crisisScores = []
        crisisResults = {}

        for crisisName, (startStr, endStr) in self.CRISIS_PERIODS.items():
            crisisStart = pd.Timestamp(startStr)
            crisisEnd = pd.Timestamp(endStr)

            if crisisStart > curveEnd or crisisEnd < curveStart:
                continue

            overlapStart = max(crisisStart, curveStart)
            overlapEnd = min(crisisEnd, curveEnd)

            crisisSlice = equityCurve.loc[overlapStart:overlapEnd]

            if len(crisisSlice) < 2:
                continue

            periodReturn = float(crisisSlice.iloc[-1] / crisisSlice.iloc[0] - 1)

            crisisScore = 50.0 + periodReturn * 500.0
            crisisScore = float(np.clip(crisisScore, 0.0, 100.0))

            crisisScores.append(crisisScore)
            crisisResults[crisisName] = {
                'label': self.CRISIS_LABELS.get(crisisName, crisisName),
                'returnPct': round(periodReturn * 100, 4),
                'score': round(crisisScore, 2),
                'tradingDays': len(crisisSlice),
            }

        details['overlappingCrises'] = len(crisisScores)
        details['crisisBreakdown'] = crisisResults

        if not crisisScores:
            details['note'] = 'no_crisis_overlap'
            return 60.0

        return float(np.mean(crisisScores))
