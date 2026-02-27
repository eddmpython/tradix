# -*- coding: utf-8 -*-
"""
Tradix Portfolio Stress Testing Module.

Applies hypothetical crisis scenarios to strategy equity curves and returns,
simulating future market stress events to evaluate portfolio resilience.
Unlike BlackSwanAnalyzer which examines past crisis performance, this module
simulates hypothetical future crises.

포트폴리오 스트레스 테스팅 모듈입니다.
가상의 위기 시나리오를 전략의 자산 곡선과 수익률에 적용하여
포트폴리오의 미래 위기 대응력을 평가합니다.
과거 위기 성과를 분석하는 BlackSwanAnalyzer와 달리,
이 모듈은 미래의 가상 위기 상황을 시뮬레이션합니다.

Features:
    - Market Crash simulation (30% drop over 20 days)
    - Volatility Spike simulation (3x normal vol for 60 days)
    - Rate Shock simulation (15% gradual decline over 120 days)
    - Liquidity Crisis simulation (10% gap down + spread widening)
    - Flash Crash simulation (8% sudden drop with partial recovery)
    - Correlation Breakdown simulation (diversification loss)
    - Survival analysis and resilience grading (A+ to F)

Usage:
    from tradix.analytics.portfolioStress import PortfolioStressAnalyzer

    analyzer = PortfolioStressAnalyzer()
    stressResult = analyzer.analyze(result)
    print(stressResult.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tradix.engine import BacktestResult


TRADING_DAYS_PER_YEAR = 252
MARKET_ANNUAL_VOL = 0.20
MARKET_DAILY_VOL = MARKET_ANNUAL_VOL / np.sqrt(TRADING_DAYS_PER_YEAR)
SURVIVAL_THRESHOLD = 0.20


@dataclass
class StressScenario:
    """
    Result of a single stress test scenario applied to the strategy.

    단일 스트레스 테스트 시나리오의 결과를 담는 데이터 클래스입니다.

    Attributes:
        name (str): Scenario name in English. 시나리오 영문 이름.
        nameKo (str): Scenario name in Korean. 시나리오 한국어 이름.
        description (str): Detailed scenario description. 시나리오 상세 설명.
        impactReturn (float): Return impact during the scenario (decimal).
            시나리오 중 수익률 영향 (소수).
        peakLoss (float): Worst point loss during the scenario (decimal, negative).
            시나리오 중 최악의 손실 (소수, 음수).
        recoveryDays (int): Estimated trading days to recover from the scenario.
            시나리오로부터의 예상 회복 거래일 수.
        survived (bool): Whether the strategy survives (equity stays above 20%).
            전략 생존 여부 (자산이 초기 자본의 20% 이상 유지).

    Example:
        >>> scenario = stressResult.scenarios['marketCrash']
        >>> print(f"Peak loss: {scenario.peakLoss:.2%}")
    """
    name: str = ""
    nameKo: str = ""
    description: str = ""
    impactReturn: float = 0.0
    peakLoss: float = 0.0
    recoveryDays: int = 0
    survived: bool = True


@dataclass
class PortfolioStressResult:
    """
    Aggregate result of all portfolio stress test scenarios.

    모든 포트폴리오 스트레스 테스트 시나리오의 종합 결과입니다.

    Attributes:
        scenarios (Dict[str, StressScenario]): Scenario name -> result mapping.
            시나리오 이름 -> 결과 매핑.
        worstScenario (str): Name of the worst-performing scenario.
            최악 시나리오 이름.
        worstLoss (float): Worst loss percentage across all scenarios (decimal).
            전 시나리오 중 최악 손실률 (소수).
        survivalRate (float): Fraction of scenarios survived (0-1).
            생존 시나리오 비율 (0-1).
        capitalAtRisk (float): Maximum capital at risk across all scenarios (decimal).
            전 시나리오 중 최대 위험 자본 비율 (소수).
        recoveryTimes (Dict[str, int]): Scenario name -> recovery days mapping.
            시나리오 이름 -> 회복 거래일 매핑.
        overallGrade (str): Resilience grade from A+ to F.
            복원력 등급 (A+ ~ F).
        details (Dict): Additional analysis details.
            추가 분석 상세 정보.

    Example:
        >>> stressResult = analyzer.analyze(result)
        >>> print(stressResult.summary())
        >>> print(f"Grade: {stressResult.overallGrade}")
    """
    scenarios: Dict[str, StressScenario] = field(default_factory=dict)
    worstScenario: str = ""
    worstLoss: float = 0.0
    survivalRate: float = 0.0
    capitalAtRisk: float = 0.0
    recoveryTimes: Dict[str, int] = field(default_factory=dict)
    overallGrade: str = "F"
    details: Dict = field(default_factory=dict)

    def summary(self, ko: bool = False) -> str:
        """
        Generate a formatted stress test report with scenario table.

        시나리오 테이블이 포함된 스트레스 테스트 보고서를 생성합니다.

        Args:
            ko (bool): If True, use Korean labels. True이면 한국어 레이블 사용.

        Returns:
            str: Multi-line formatted report string.
                여러 줄로 구성된 포맷 보고서 문자열.
        """
        if ko:
            title = "포트폴리오 스트레스 테스트"
            gradeLabel = "복원력 등급"
            survivalLabel = "생존률"
            worstLabel = "최악 시나리오"
            worstLossLabel = "최대 손실"
            capitalLabel = "최대 위험 자본"
            scenarioHeader = "시나리오"
            impactHeader = "영향"
            peakLossHeader = "최대 손실"
            recoveryHeader = "회복(일)"
            survivedHeader = "생존"
        else:
            title = "Portfolio Stress Test"
            gradeLabel = "Resilience Grade"
            survivalLabel = "Survival Rate"
            worstLabel = "Worst Scenario"
            worstLossLabel = "Worst Loss"
            capitalLabel = "Capital at Risk"
            scenarioHeader = "Scenario"
            impactHeader = "Impact"
            peakLossHeader = "Peak Loss"
            recoveryHeader = "Recovery"
            survivedHeader = "Survived"

        lines = [
            f"\n{'='*65}",
            f"  {title} [{self.overallGrade}]",
            f"{'='*65}",
            f"  {gradeLabel}: {self.overallGrade}",
            f"  {survivalLabel}: {self.survivalRate:.0%} ({sum(1 for s in self.scenarios.values() if s.survived)}/{len(self.scenarios)})",
            f"  {worstLabel}: {self.worstScenario}",
            f"  {worstLossLabel}: {self.worstLoss:.2%}",
            f"  {capitalLabel}: {self.capitalAtRisk:.2%}",
            f"{'─'*65}",
        ]

        nameWidth = 22
        lines.append(
            f"  {scenarioHeader:<{nameWidth}} {impactHeader:>8} {peakLossHeader:>10} {recoveryHeader:>8} {survivedHeader:>8}"
        )
        lines.append(f"  {'─'*nameWidth} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

        for key, scenario in self.scenarios.items():
            displayName = scenario.nameKo if ko else scenario.name
            if len(displayName) > nameWidth:
                displayName = displayName[:nameWidth - 1] + "."
            survivedMark = "O" if scenario.survived else "X"
            lines.append(
                f"  {displayName:<{nameWidth}} {scenario.impactReturn:>+7.2%} {scenario.peakLoss:>+9.2%} {scenario.recoveryDays:>7d} {survivedMark:>8}"
            )

        lines.append(f"{'='*65}\n")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PortfolioStressResult(grade={self.overallGrade}, "
            f"survival={self.survivalRate:.0%}, "
            f"worstLoss={self.worstLoss:.2%}, "
            f"scenarios={len(self.scenarios)})"
        )


class PortfolioStressAnalyzer:
    """
    Portfolio stress tester that simulates hypothetical crisis scenarios.

    Applies six distinct crisis scenarios to the strategy's equity curve
    and daily returns to evaluate resilience under extreme conditions.

    가상의 위기 시나리오를 전략의 자산 곡선과 일별 수익률에 적용하여
    극한 조건에서의 복원력을 평가하는 포트폴리오 스트레스 테스터입니다.

    Scenarios:
        1. Market Crash (시장 급락): 30% drop over 20 days
        2. Volatility Spike (변동성 폭증): 3x vol for 60 days
        3. Rate Shock (금리 충격): 15% gradual decline over 120 days
        4. Liquidity Crisis (유동성 위기): 10% gap down + spread widening
        5. Flash Crash (플래시 크래시): 8% sudden drop, 6% recovery
        6. Correlation Breakdown (상관관계 붕괴): all correlations -> 0.95

    Example:
        >>> analyzer = PortfolioStressAnalyzer()
        >>> stressResult = analyzer.analyze(backtestResult)
        >>> print(stressResult.summary(ko=True))
        >>> print(f"Grade: {stressResult.overallGrade}")
    """

    def analyze(self, result: BacktestResult) -> PortfolioStressResult:
        """
        Perform comprehensive portfolio stress testing on a backtest result.

        백테스트 결과에 대한 종합 포트폴리오 스트레스 테스트를 수행합니다.

        Extracts strategy characteristics from the equity curve, applies six
        hypothetical crisis scenarios, evaluates survival, and grades overall
        resilience from A+ to F.

        자산 곡선에서 전략 특성을 추출하고, 6가지 가상 위기 시나리오를 적용하여
        생존 여부를 평가하고, A+부터 F까지 전체 복원력 등급을 부여합니다.

        Args:
            result (BacktestResult): Completed backtest result containing equity
                curve and trade history. 자산 곡선과 거래 내역이 포함된 백테스트 결과.

        Returns:
            PortfolioStressResult: Comprehensive stress test result with all
                scenario outcomes and overall grade.
                모든 시나리오 결과와 전체 등급이 포함된 종합 스트레스 테스트 결과.
        """
        equityCurve = result.equityCurve
        if len(equityCurve) < 10:
            return PortfolioStressResult(
                overallGrade="F",
                details={"error": "insufficient_data"},
            )

        characteristics = self._strategyCharacteristics(equityCurve)

        scenarios = {}
        scenarios["marketCrash"] = self._applyMarketCrash(characteristics)
        scenarios["volatilitySpike"] = self._applyVolatilitySpike(characteristics)
        scenarios["rateShock"] = self._applyRateShock(characteristics)
        scenarios["liquidityCrisis"] = self._applyLiquidityCrisis(characteristics)
        scenarios["flashCrash"] = self._applyFlashCrash(characteristics)
        scenarios["correlationBreakdown"] = self._applyCorrelationBreakdown(characteristics)

        worstScenario = ""
        worstLoss = 0.0
        maxCapitalAtRisk = 0.0
        recoveryTimes = {}
        survivedCount = 0

        for key, scenario in scenarios.items():
            recoveryTimes[key] = scenario.recoveryDays
            if scenario.peakLoss < worstLoss:
                worstLoss = scenario.peakLoss
                worstScenario = key
            riskAmount = abs(scenario.peakLoss)
            if riskAmount > maxCapitalAtRisk:
                maxCapitalAtRisk = riskAmount
            if scenario.survived:
                survivedCount += 1

        totalScenarios = len(scenarios)
        survivalRate = survivedCount / totalScenarios if totalScenarios > 0 else 0.0

        overallGrade = self._gradeResilience(scenarios)

        return PortfolioStressResult(
            scenarios=scenarios,
            worstScenario=worstScenario,
            worstLoss=worstLoss,
            survivalRate=survivalRate,
            capitalAtRisk=maxCapitalAtRisk,
            recoveryTimes=recoveryTimes,
            overallGrade=overallGrade,
            details={
                "characteristics": characteristics,
                "totalScenarios": totalScenarios,
                "survivedCount": survivedCount,
            },
        )

    def _strategyCharacteristics(self, equityCurve: pd.Series) -> Dict:
        """
        Extract key strategy characteristics from the equity curve.

        자산 곡선에서 핵심 전략 특성을 추출합니다.

        Computes average daily return, annualized volatility, beta estimate
        relative to the market, and maximum historical drawdown.

        일평균 수익률, 연간 변동성, 시장 대비 베타 추정치, 최대 역사적 낙폭을 계산합니다.

        Args:
            equityCurve (pd.Series): Time-indexed equity series.
                시간 인덱스 자산 시리즈.

        Returns:
            Dict: Strategy characteristics containing avgDailyReturn, dailyVol,
                annualVol, beta, maxDrawdown, totalReturn, equityFinal,
                equityInitial, tradingDays.
        """
        returns = equityCurve.pct_change().dropna()

        if len(returns) < 5:
            return {
                "avgDailyReturn": 0.0,
                "dailyVol": 0.01,
                "annualVol": 0.01 * np.sqrt(TRADING_DAYS_PER_YEAR),
                "beta": 1.0,
                "maxDrawdown": 0.0,
                "totalReturn": 0.0,
                "equityFinal": float(equityCurve.iloc[-1]),
                "equityInitial": float(equityCurve.iloc[0]),
                "tradingDays": len(equityCurve),
            }

        avgDailyReturn = float(np.mean(returns.values))
        dailyVol = float(np.std(returns.values, ddof=1))
        annualVol = dailyVol * np.sqrt(TRADING_DAYS_PER_YEAR)

        if dailyVol > 0 and MARKET_DAILY_VOL > 0:
            beta = dailyVol / MARKET_DAILY_VOL
        else:
            beta = 1.0

        beta = float(np.clip(beta, 0.1, 5.0))

        cumMax = equityCurve.cummax()
        drawdownSeries = (equityCurve - cumMax) / cumMax
        maxDrawdown = float(drawdownSeries.min())

        totalReturn = float(equityCurve.iloc[-1] / equityCurve.iloc[0] - 1)

        return {
            "avgDailyReturn": avgDailyReturn,
            "dailyVol": dailyVol,
            "annualVol": annualVol,
            "beta": beta,
            "maxDrawdown": maxDrawdown,
            "totalReturn": totalReturn,
            "equityFinal": float(equityCurve.iloc[-1]),
            "equityInitial": float(equityCurve.iloc[0]),
            "tradingDays": len(equityCurve),
        }

    def _applyMarketCrash(
        self,
        characteristics: Dict,
        magnitude: float = -0.30,
        duration: int = 20,
    ) -> StressScenario:
        """
        Simulate a market crash scenario.

        시장 급락 시나리오를 시뮬레이션합니다.

        Models a sudden market decline with strategy impact scaled by beta
        and relative volatility. Recovery is estimated from historical
        average daily return.

        베타와 상대 변동성으로 조정된 급격한 시장 하락을 모델링합니다.
        회복은 과거 평균 일별 수익률로부터 추정됩니다.

        Args:
            characteristics (Dict): Strategy characteristics from _strategyCharacteristics.
                _strategyCharacteristics에서 추출한 전략 특성.
            magnitude (float): Market drop magnitude (e.g., -0.30 for 30% drop).
                시장 하락 크기 (예: -0.30 = 30% 하락).
            duration (int): Number of trading days for the crash.
                급락 기간 (거래일 수).

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        beta = characteristics["beta"]
        strategyVol = characteristics["annualVol"]
        marketVol = MARKET_ANNUAL_VOL

        volRatio = strategyVol / marketVol if marketVol > 0 else 1.0
        strategyImpact = magnitude * beta * (1 + volRatio)
        strategyImpact = float(np.clip(strategyImpact, -0.99, 0.0))

        peakLoss = strategyImpact * 1.15
        peakLoss = float(np.clip(peakLoss, -0.99, strategyImpact))

        avgDailyReturn = characteristics["avgDailyReturn"]
        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(peakLoss))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Market Crash",
            nameKo="시장 급락",
            description=f"{abs(magnitude):.0%} market drop over {duration} trading days",
            impactReturn=strategyImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _applyVolatilitySpike(
        self,
        characteristics: Dict,
        multiplier: float = 3.0,
        duration: int = 60,
    ) -> StressScenario:
        """
        Simulate a volatility spike scenario.

        변동성 폭증 시나리오를 시뮬레이션합니다.

        Models a period of extreme volatility with increased slippage and
        wider spreads, degrading risk-adjusted returns proportionally.

        슬리피지 증가와 스프레드 확대를 수반하는 극단적 변동성 기간을
        모델링하여, 위험 조정 수익률을 비례적으로 하락시킵니다.

        Args:
            characteristics (Dict): Strategy characteristics.
                전략 특성.
            multiplier (float): Volatility multiplier (e.g., 3.0 for 3x normal vol).
                변동성 배수 (예: 3.0 = 정상의 3배).
            duration (int): Duration of the volatility spike in trading days.
                변동성 폭증 기간 (거래일 수).

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        dailyVol = characteristics["dailyVol"]
        avgDailyReturn = characteristics["avgDailyReturn"]

        spikedVol = dailyVol * multiplier

        slippageCost = spikedVol * 0.3 * duration
        spreadCost = dailyVol * (multiplier - 1) * 0.15 * duration

        returnDegradation = -(spikedVol ** 2) / 2 * duration

        strategyImpact = returnDegradation - slippageCost - spreadCost
        strategyImpact = float(np.clip(strategyImpact, -0.80, -0.001))

        peakLoss = strategyImpact * 1.3
        peakLoss = float(np.clip(peakLoss, -0.95, strategyImpact))

        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(peakLoss))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Volatility Spike",
            nameKo="변동성 폭증",
            description=f"{multiplier:.0f}x normal volatility for {duration} trading days",
            impactReturn=strategyImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _applyRateShock(
        self,
        characteristics: Dict,
        impact: float = -0.15,
        duration: int = 120,
    ) -> StressScenario:
        """
        Simulate a rate shock scenario (like 2022 rate hike cycle).

        금리 충격 시나리오를 시뮬레이션합니다 (2022년 금리 인상기와 유사).

        Models a gradual grinding decline with duration-dependent impact
        on strategy returns, reflecting a sustained bear market.

        지속적인 약세장을 반영하여, 기간에 따른 점진적 하락이 전략 수익률에
        미치는 영향을 모델링합니다.

        Args:
            characteristics (Dict): Strategy characteristics.
                전략 특성.
            impact (float): Total valuation drop (e.g., -0.15 for 15% decline).
                총 가치 하락 (예: -0.15 = 15% 하락).
            duration (int): Duration of the decline in trading days.
                하락 기간 (거래일 수).

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        beta = characteristics["beta"]
        annualVol = characteristics["annualVol"]

        durationFactor = duration / TRADING_DAYS_PER_YEAR
        volDrag = -(annualVol ** 2) / 2 * durationFactor

        strategyImpact = impact * beta * (1 + durationFactor * 0.5) + volDrag
        strategyImpact = float(np.clip(strategyImpact, -0.85, -0.001))

        peakLoss = strategyImpact * 1.1
        peakLoss = float(np.clip(peakLoss, -0.95, strategyImpact))

        avgDailyReturn = characteristics["avgDailyReturn"]
        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(peakLoss))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Rate Shock",
            nameKo="금리 충격",
            description=f"{abs(impact):.0%} valuation drop over {duration} trading days",
            impactReturn=strategyImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _applyLiquidityCrisis(
        self,
        characteristics: Dict,
        gapDown: float = -0.10,
    ) -> StressScenario:
        """
        Simulate a liquidity crisis scenario.

        유동성 위기 시나리오를 시뮬레이션합니다.

        Models a sudden gap down with 5x spread widening and partial fill
        simulation, where position exits suffer significant slippage.

        5배 스프레드 확대와 부분 체결 시뮬레이션을 수반하는 급격한 갭 하락을
        모델링하며, 포지션 청산 시 상당한 슬리피지가 발생합니다.

        Args:
            characteristics (Dict): Strategy characteristics.
                전략 특성.
            gapDown (float): Initial gap down magnitude (e.g., -0.10 for 10%).
                초기 갭 하락 크기 (예: -0.10 = 10%).

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        beta = characteristics["beta"]
        dailyVol = characteristics["dailyVol"]

        gapImpact = gapDown * beta

        spreadWidening = 5.0
        spreadCost = dailyVol * spreadWidening * 0.5

        partialFillPenalty = abs(gapDown) * 0.25

        strategyImpact = gapImpact - spreadCost - partialFillPenalty
        strategyImpact = float(np.clip(strategyImpact, -0.90, -0.001))

        peakLoss = strategyImpact * 1.2
        peakLoss = float(np.clip(peakLoss, -0.95, strategyImpact))

        avgDailyReturn = characteristics["avgDailyReturn"]
        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(peakLoss))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Liquidity Crisis",
            nameKo="유동성 위기",
            description=f"{abs(gapDown):.0%} gap down with 5x spread widening",
            impactReturn=strategyImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _applyFlashCrash(
        self,
        characteristics: Dict,
        drop: float = -0.08,
        recovery: float = 0.06,
    ) -> StressScenario:
        """
        Simulate a flash crash scenario.

        플래시 크래시 시나리오를 시뮬레이션합니다.

        Models a sudden sharp drop followed by a partial recovery, with
        stop-loss cascade effects amplifying the initial impact.

        급격한 하락 후 부분 회복을 모델링하며, 손절매 연쇄 효과가
        초기 충격을 증폭시킵니다.

        Args:
            characteristics (Dict): Strategy characteristics.
                전략 특성.
            drop (float): Initial sudden drop magnitude (e.g., -0.08 for 8%).
                초기 급락 크기 (예: -0.08 = 8%).
            recovery (float): Subsequent recovery magnitude (e.g., 0.06 for 6%).
                이후 반등 크기 (예: 0.06 = 6%).

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        beta = characteristics["beta"]

        crashImpact = drop * beta

        stopLossCascade = abs(drop) * 0.4
        amplifiedDrop = crashImpact - stopLossCascade

        partialRecovery = recovery * max(0.3, 1.0 - beta * 0.3)

        netImpact = amplifiedDrop + partialRecovery
        netImpact = float(np.clip(netImpact, -0.80, 0.0))

        peakLoss = amplifiedDrop
        peakLoss = float(np.clip(peakLoss, -0.90, -0.001))

        avgDailyReturn = characteristics["avgDailyReturn"]
        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(netImpact))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Flash Crash",
            nameKo="플래시 크래시",
            description=f"{abs(drop):.0%} sudden drop in 5 days, {recovery:.0%} recovery",
            impactReturn=netImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _applyCorrelationBreakdown(
        self,
        characteristics: Dict,
    ) -> StressScenario:
        """
        Simulate a correlation breakdown scenario.

        상관관계 붕괴 시나리오를 시뮬레이션합니다.

        Models a scenario where all asset correlations converge to 0.95,
        eliminating diversification benefits and amplifying portfolio risk.

        모든 자산 상관관계가 0.95로 수렴하여 분산 투자 효과가 사라지고
        포트폴리오 위험이 증폭되는 시나리오를 모델링합니다.

        Args:
            characteristics (Dict): Strategy characteristics.
                전략 특성.

        Returns:
            StressScenario: Scenario result with impact, peak loss, and survival status.
        """
        annualVol = characteristics["annualVol"]
        maxDrawdown = characteristics["maxDrawdown"]
        beta = characteristics["beta"]

        normalCorrelation = 0.3
        crisisCorrelation = 0.95

        diversificationLoss = (crisisCorrelation - normalCorrelation) * annualVol
        riskAmplification = np.sqrt(crisisCorrelation / max(normalCorrelation, 0.01))

        amplifiedDrawdown = maxDrawdown * riskAmplification

        strategyImpact = amplifiedDrawdown - diversificationLoss
        strategyImpact = float(np.clip(strategyImpact, -0.85, -0.001))

        peakLoss = strategyImpact * 1.15
        peakLoss = float(np.clip(peakLoss, -0.95, strategyImpact))

        avgDailyReturn = characteristics["avgDailyReturn"]
        recoveryDays = self._estimateRecovery(avgDailyReturn, abs(peakLoss))

        survived = (1.0 + peakLoss) >= SURVIVAL_THRESHOLD

        return StressScenario(
            name="Correlation Breakdown",
            nameKo="상관관계 붕괴",
            description="All correlations converge to 0.95, diversification eliminated",
            impactReturn=strategyImpact,
            peakLoss=peakLoss,
            recoveryDays=recoveryDays,
            survived=survived,
        )

    def _estimateRecovery(self, avgDailyReturn: float, loss: float) -> int:
        """
        Estimate the number of trading days to recover from a given loss.

        주어진 손실로부터의 회복에 필요한 거래일 수를 추정합니다.

        Uses the strategy's average daily return to project how long recovery
        would take, with a minimum of 1 day and maximum of 9999 days.

        전략의 평균 일별 수익률을 사용하여 회복에 걸리는 기간을 추정합니다.
        최소 1일, 최대 9999일로 제한됩니다.

        Args:
            avgDailyReturn (float): Strategy's average daily return (decimal).
                전략의 평균 일별 수익률 (소수).
            loss (float): Magnitude of loss to recover from (positive decimal).
                회복해야 할 손실 크기 (양수 소수).

        Returns:
            int: Estimated trading days to recover.
                추정 회복 거래일 수.
        """
        if avgDailyReturn <= 0:
            return 9999

        if loss <= 0:
            return 0

        recoveryReturn = loss / (1.0 - loss)
        daysNeeded = recoveryReturn / avgDailyReturn

        return int(np.clip(daysNeeded, 1, 9999))

    def _gradeResilience(self, scenarios: Dict[str, StressScenario]) -> str:
        """
        Assign a resilience grade based on scenario survival and loss severity.

        시나리오 생존율과 손실 심각도에 따라 복원력 등급을 부여합니다.

        Grading criteria:
            A+: Survives all, worst loss < 15%
            A:  Survives all, worst loss < 25%
            B:  Survives 5/6, worst loss < 35%
            C:  Survives 4/6, worst loss < 50%
            D:  Survives 2-3/6
            F:  Survives 0-1/6

        Args:
            scenarios (Dict[str, StressScenario]): All scenario results.
                모든 시나리오 결과.

        Returns:
            str: Letter grade from A+ to F.
                A+부터 F까지의 등급 문자열.
        """
        totalScenarios = len(scenarios)
        if totalScenarios == 0:
            return "F"

        survivedCount = sum(1 for s in scenarios.values() if s.survived)
        worstLoss = min(s.peakLoss for s in scenarios.values())
        absWorstLoss = abs(worstLoss)

        if survivedCount == totalScenarios and absWorstLoss < 0.15:
            return "A+"

        if survivedCount == totalScenarios and absWorstLoss < 0.25:
            return "A"

        if survivedCount >= totalScenarios - 1 and absWorstLoss < 0.35:
            return "B"

        if survivedCount >= totalScenarios - 2 and absWorstLoss < 0.50:
            return "C"

        if survivedCount >= 2:
            return "D"

        return "F"
