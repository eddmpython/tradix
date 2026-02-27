# -*- coding: utf-8 -*-
"""
Tradix What-If Simulator Module.

Analyzes how changing trading conditions (commission, slippage, tax, timing,
capital) affects backtest performance by recalculating trade-level P&L under
alternative scenarios.

트레이딩 조건 변경이 백테스트 성과에 미치는 영향을 분석하는 What-If 시뮬레이터 모듈입니다.
수수료, 슬리피지, 세금, 시작일, 초기 자본 등의 조건을 변경하여 거래별 손익을
재계산합니다.

Features:
    - Single-parameter scenario analysis (commission, slippage, tax, timing, capital)
    - Multi-scenario batch analysis
    - 2D sensitivity matrix for commission x slippage combinations
    - Impact quantification in both absolute and relative terms

Usage:
    from tradix.analytics.whatIf import WhatIfSimulator

    simulator = WhatIfSimulator(result)
    impact = simulator.adjustCommission(0.001)
    print(impact.summary())

    matrix = simulator.sensitivityMatrix(
        commissions=[0.0001, 0.0005, 0.001],
        slippages=[0.0005, 0.001, 0.002],
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from tradix.engine import BacktestResult
from tradix.entities.trade import Trade


DEFAULT_COMMISSION_RATE = 0.00015
DEFAULT_TAX_RATE = 0.0018
DEFAULT_SLIPPAGE_RATE = 0.001


@dataclass
class WhatIfResult:
    """
    Result container for a single what-if scenario analysis.

    단일 What-If 시나리오 분석 결과를 담는 데이터 클래스입니다.

    Attributes:
        scenario (str): Descriptive name of the scenario. 시나리오 이름.
        originalReturn (float): Original total return in percent. 원래 총 수익률 (%).
        adjustedReturn (float): Adjusted total return in percent. 조정된 총 수익률 (%).
        impact (float): Absolute difference in return percentage points.
            수익률 차이 (퍼센트포인트).
        impactPercent (float): Relative impact as percentage of original return.
            원래 수익률 대비 상대적 영향 (%).
        metrics (dict): Full set of adjusted performance metrics.
            조정된 전체 성과 지표 딕셔너리.

    Example:
        >>> result = simulator.adjustCommission(0.001)
        >>> print(result.summary())
    """
    scenario: str
    originalReturn: float
    adjustedReturn: float
    impact: float
    impactPercent: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Generate a human-readable summary of the scenario result.

        시나리오 결과의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line formatted summary with scenario name, returns,
                and impact analysis.
        """
        return (
            f"{'='*50}\n"
            f"시나리오: {self.scenario}\n"
            f"{'─'*50}\n"
            f"원래 수익률: {self.originalReturn:+.2f}%\n"
            f"조정 수익률: {self.adjustedReturn:+.2f}%\n"
            f"영향: {self.impact:+.2f}%p ({self.impactPercent:+.2f}%)\n"
            f"{'='*50}"
        )


class WhatIfSimulator:
    """
    What-If scenario simulator for backtest results.

    Recalculates trade-level P&L under alternative trading conditions to
    quantify the impact of commission rates, slippage, taxes, timing, and
    capital on strategy performance.

    백테스트 결과에 대한 What-If 시나리오 시뮬레이터입니다.
    수수료, 슬리피지, 세금, 시작 시점, 초기 자본 등의 조건 변경이 전략 성과에
    미치는 영향을 거래별 손익 재계산을 통해 정량화합니다.

    Attributes:
        result (BacktestResult): Original backtest result to analyze.
            분석 대상 원본 백테스트 결과.

    Example:
        >>> simulator = WhatIfSimulator(result)
        >>> commission_impact = simulator.adjustCommission(0.001)
        >>> print(commission_impact.summary())
        >>>
        >>> matrix = simulator.sensitivityMatrix(
        ...     commissions=[0.0001, 0.0005, 0.001],
        ...     slippages=[0.0005, 0.001, 0.002],
        ... )
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize the What-If simulator with a backtest result.

        백테스트 결과로 What-If 시뮬레이터를 초기화합니다.

        Args:
            result (BacktestResult): Completed backtest result containing trades,
                equity curve, and metrics. 거래 내역, 자산 곡선, 성과 지표가 포함된
                백테스트 결과.
        """
        self._result = result
        self._closedTrades = [t for t in result.trades if t.isClosed]
        self._originalReturn = result.totalReturn
        self._scenarios: List[WhatIfResult] = []

        self._origCommission = result.metrics.get(
            'commissionRate', DEFAULT_COMMISSION_RATE
        )
        self._origTax = result.metrics.get('taxRate', DEFAULT_TAX_RATE)
        self._origSlippage = result.metrics.get(
            'slippageRate', DEFAULT_SLIPPAGE_RATE
        )

    def _estimateOriginalRates(self) -> Dict[str, float]:
        """
        Estimate original commission, tax, and slippage rates from trade data.

        거래 데이터로부터 원래 수수료율, 세금율, 슬리피지율을 추정합니다.

        Returns:
            dict: Estimated rates for commission, tax, and slippage.
        """
        if not self._closedTrades:
            return {
                'commission': DEFAULT_COMMISSION_RATE,
                'tax': DEFAULT_TAX_RATE,
                'slippage': DEFAULT_SLIPPAGE_RATE,
            }

        totalCommission = sum(t.commission for t in self._closedTrades)
        totalSlippage = sum(t.slippage for t in self._closedTrades)
        totalValue = sum(
            t.entryPrice * t.quantity * 2 for t in self._closedTrades
        )

        if totalValue > 0:
            estCommission = totalCommission / totalValue
            estSlippage = totalSlippage / totalValue
        else:
            estCommission = DEFAULT_COMMISSION_RATE
            estSlippage = DEFAULT_SLIPPAGE_RATE

        return {
            'commission': estCommission if estCommission > 0 else DEFAULT_COMMISSION_RATE,
            'tax': DEFAULT_TAX_RATE,
            'slippage': estSlippage if estSlippage > 0 else DEFAULT_SLIPPAGE_RATE,
        }

    def _recalculatePnl(
        self,
        commissionDelta: float = 0.0,
        slippageDelta: float = 0.0,
        taxDelta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Recalculate aggregate P&L by applying cost deltas to each closed trade.

        각 청산된 거래에 비용 변화량을 적용하여 총 손익을 재계산합니다.

        Args:
            commissionDelta (float): Change in commission rate (new - old).
                수수료율 변화량.
            slippageDelta (float): Change in slippage rate (new - old).
                슬리피지율 변화량.
            taxDelta (float): Change in tax rate (new - old).
                세금율 변화량.

        Returns:
            dict: Adjusted metrics including totalReturn, winRate, profitFactor,
                totalPnl, and individual adjustedPnls.
        """
        adjustedPnls = []
        initialCash = self._result.initialCash

        for trade in self._closedTrades:
            originalPnl = trade.pnl
            tradeValue = trade.quantity * trade.entryPrice

            commissionImpact = trade.quantity * trade.entryPrice * commissionDelta * 2
            slippageImpact = trade.quantity * trade.entryPrice * slippageDelta * 2

            exitValue = trade.quantity * (trade.exitPrice if trade.exitPrice else trade.entryPrice)
            taxImpact = exitValue * taxDelta

            adjustedPnl = originalPnl - commissionImpact - slippageImpact - taxImpact
            adjustedPnls.append(adjustedPnl)

        totalPnl = float(np.sum(adjustedPnls)) if adjustedPnls else 0.0
        adjustedEquity = initialCash + totalPnl
        adjustedReturn = ((adjustedEquity / initialCash) - 1) * 100 if initialCash > 0 else 0.0

        wins = [p for p in adjustedPnls if p > 0]
        losses = [p for p in adjustedPnls if p <= 0]
        totalTrades = len(adjustedPnls)

        winRate = (len(wins) / totalTrades * 100) if totalTrades > 0 else 0.0
        avgWin = float(np.mean(wins)) if wins else 0.0
        avgLoss = float(np.mean(losses)) if losses else 0.0
        totalProfit = float(np.sum(wins)) if wins else 0.0
        totalLoss = abs(float(np.sum(losses))) if losses else 0.0
        profitFactor = totalProfit / totalLoss if totalLoss > 0 else float('inf')

        return {
            'totalReturn': adjustedReturn,
            'totalPnl': totalPnl,
            'adjustedEquity': adjustedEquity,
            'winRate': winRate,
            'avgWin': avgWin,
            'avgLoss': avgLoss,
            'profitFactor': profitFactor,
            'totalTrades': totalTrades,
            'winningTrades': len(wins),
            'losingTrades': len(losses),
            'adjustedPnls': adjustedPnls,
        }

    def _buildWhatIfResult(
        self, scenarioName: str, adjustedMetrics: Dict[str, Any]
    ) -> WhatIfResult:
        """
        Build a WhatIfResult from adjusted metrics.

        조정된 지표로부터 WhatIfResult 객체를 생성합니다.

        Args:
            scenarioName (str): Descriptive scenario name. 시나리오 이름.
            adjustedMetrics (dict): Metrics from _recalculatePnl. 재계산된 지표.

        Returns:
            WhatIfResult: Populated result object.
        """
        adjustedReturn = adjustedMetrics['totalReturn']
        impact = adjustedReturn - self._originalReturn
        impactPercent = (
            (impact / abs(self._originalReturn)) * 100
            if self._originalReturn != 0 else 0.0
        )

        whatIfResult = WhatIfResult(
            scenario=scenarioName,
            originalReturn=self._originalReturn,
            adjustedReturn=adjustedReturn,
            impact=impact,
            impactPercent=impactPercent,
            metrics=adjustedMetrics,
        )
        self._scenarios.append(whatIfResult)
        return whatIfResult

    def adjustCommission(self, newRate: float) -> WhatIfResult:
        """
        Recalculate performance with a different commission rate.

        수수료율을 변경하여 성과를 재계산합니다.

        Commission impact per trade = quantity * entryPrice * (newRate - oldRate) * 2,
        accounting for both buy and sell legs of the round trip.

        Args:
            newRate (float): New commission rate to apply (e.g., 0.001 for 0.1%).
                적용할 새 수수료율 (예: 0.001 = 0.1%).

        Returns:
            WhatIfResult: Scenario result with adjusted metrics.
                조정된 지표가 담긴 시나리오 결과.

        Example:
            >>> result = simulator.adjustCommission(0.001)
            >>> print(f"Impact: {result.impact:+.2f}%p")
        """
        rates = self._estimateOriginalRates()
        commissionDelta = newRate - rates['commission']
        scenarioName = f"수수료 변경: {rates['commission']:.5f} -> {newRate:.5f}"

        adjustedMetrics = self._recalculatePnl(commissionDelta=commissionDelta)
        return self._buildWhatIfResult(scenarioName, adjustedMetrics)

    def adjustSlippage(self, newRate: float) -> WhatIfResult:
        """
        Recalculate performance with a different slippage rate.

        슬리피지율을 변경하여 성과를 재계산합니다.

        Slippage impact per trade = quantity * entryPrice * (newRate - oldRate) * 2,
        accounting for both entry and exit execution slippage.

        Args:
            newRate (float): New slippage rate to apply (e.g., 0.002 for 0.2%).
                적용할 새 슬리피지율 (예: 0.002 = 0.2%).

        Returns:
            WhatIfResult: Scenario result with adjusted metrics.
                조정된 지표가 담긴 시나리오 결과.

        Example:
            >>> result = simulator.adjustSlippage(0.002)
            >>> print(f"Impact: {result.impact:+.2f}%p")
        """
        rates = self._estimateOriginalRates()
        slippageDelta = newRate - rates['slippage']
        scenarioName = f"슬리피지 변경: {rates['slippage']:.5f} -> {newRate:.5f}"

        adjustedMetrics = self._recalculatePnl(slippageDelta=slippageDelta)
        return self._buildWhatIfResult(scenarioName, adjustedMetrics)

    def adjustStartDate(self, newStart: str) -> WhatIfResult:
        """
        Recalculate performance as if the strategy started on a later date.

        전략 시작일을 변경했을 때의 성과를 재계산합니다.

        Filters trades to include only those with entry dates on or after
        the new start date and recalculates aggregate metrics.

        Args:
            newStart (str): New start date in 'YYYY-MM-DD' format.
                새 시작일 ('YYYY-MM-DD' 형식).

        Returns:
            WhatIfResult: Scenario result with adjusted metrics.
                조정된 지표가 담긴 시나리오 결과.

        Example:
            >>> result = simulator.adjustStartDate("2022-01-01")
            >>> print(result.summary())
        """
        newStartDate = pd.Timestamp(newStart)
        scenarioName = f"시작일 변경: {self._result.startDate} -> {newStart}"

        filteredTrades = [
            t for t in self._closedTrades
            if pd.Timestamp(t.entryDate) >= newStartDate
        ]

        initialCash = self._result.initialCash
        adjustedPnls = [t.pnl for t in filteredTrades]
        totalPnl = float(np.sum(adjustedPnls)) if adjustedPnls else 0.0
        adjustedEquity = initialCash + totalPnl
        adjustedReturn = ((adjustedEquity / initialCash) - 1) * 100 if initialCash > 0 else 0.0

        wins = [p for p in adjustedPnls if p > 0]
        losses = [p for p in adjustedPnls if p <= 0]
        totalTrades = len(adjustedPnls)

        adjustedMetrics = {
            'totalReturn': adjustedReturn,
            'totalPnl': totalPnl,
            'adjustedEquity': adjustedEquity,
            'winRate': (len(wins) / totalTrades * 100) if totalTrades > 0 else 0.0,
            'avgWin': float(np.mean(wins)) if wins else 0.0,
            'avgLoss': float(np.mean(losses)) if losses else 0.0,
            'profitFactor': (
                float(np.sum(wins)) / abs(float(np.sum(losses)))
                if losses and float(np.sum(losses)) != 0
                else float('inf')
            ),
            'totalTrades': totalTrades,
            'winningTrades': len(wins),
            'losingTrades': len(losses),
            'filteredStartDate': newStart,
        }

        return self._buildWhatIfResult(scenarioName, adjustedMetrics)

    def adjustInitialCash(self, newCash: float) -> WhatIfResult:
        """
        Recalculate performance with a different initial capital amount.

        초기 자본금을 변경했을 때의 성과를 재계산합니다.

        Scales trade P&L proportionally based on the ratio of new capital to
        original capital, reflecting how position sizes would change.

        Args:
            newCash (float): New initial capital amount.
                새 초기 자본금.

        Returns:
            WhatIfResult: Scenario result with adjusted metrics.
                조정된 지표가 담긴 시나리오 결과.

        Example:
            >>> result = simulator.adjustInitialCash(50_000_000)
            >>> print(f"Adjusted return: {result.adjustedReturn:.2f}%")
        """
        originalCash = self._result.initialCash
        scenarioName = f"초기자본 변경: {originalCash:,.0f} -> {newCash:,.0f}"

        if originalCash <= 0:
            return self._buildWhatIfResult(scenarioName, {
                'totalReturn': 0.0,
                'totalPnl': 0.0,
                'adjustedEquity': newCash,
                'winRate': 0.0,
                'avgWin': 0.0,
                'avgLoss': 0.0,
                'profitFactor': 0.0,
                'totalTrades': 0,
                'winningTrades': 0,
                'losingTrades': 0,
            })

        scaleFactor = newCash / originalCash
        adjustedPnls = [t.pnl * scaleFactor for t in self._closedTrades]
        totalPnl = float(np.sum(adjustedPnls)) if adjustedPnls else 0.0
        adjustedEquity = newCash + totalPnl
        adjustedReturn = ((adjustedEquity / newCash) - 1) * 100 if newCash > 0 else 0.0

        wins = [p for p in adjustedPnls if p > 0]
        losses = [p for p in adjustedPnls if p <= 0]
        totalTrades = len(adjustedPnls)

        adjustedMetrics = {
            'totalReturn': adjustedReturn,
            'totalPnl': totalPnl,
            'adjustedEquity': adjustedEquity,
            'winRate': (len(wins) / totalTrades * 100) if totalTrades > 0 else 0.0,
            'avgWin': float(np.mean(wins)) if wins else 0.0,
            'avgLoss': float(np.mean(losses)) if losses else 0.0,
            'profitFactor': (
                float(np.sum(wins)) / abs(float(np.sum(losses)))
                if losses and float(np.sum(losses)) != 0
                else float('inf')
            ),
            'totalTrades': totalTrades,
            'winningTrades': len(wins),
            'losingTrades': len(losses),
            'scaleFactor': scaleFactor,
            'originalCash': originalCash,
            'newCash': newCash,
        }

        return self._buildWhatIfResult(scenarioName, adjustedMetrics)

    def adjustTax(self, newRate: float) -> WhatIfResult:
        """
        Recalculate performance with a different transaction tax rate.

        거래세율을 변경하여 성과를 재계산합니다.

        Tax impact per trade = quantity * exitPrice * (newRate - oldRate),
        applied only to the sell leg of each round-trip trade.

        Args:
            newRate (float): New tax rate to apply (e.g., 0.0025 for 0.25%).
                적용할 새 세금율 (예: 0.0025 = 0.25%).

        Returns:
            WhatIfResult: Scenario result with adjusted metrics.
                조정된 지표가 담긴 시나리오 결과.

        Example:
            >>> result = simulator.adjustTax(0.0025)
            >>> print(f"Tax impact: {result.impact:+.2f}%p")
        """
        rates = self._estimateOriginalRates()
        taxDelta = newRate - rates['tax']
        scenarioName = f"세금 변경: {rates['tax']:.5f} -> {newRate:.5f}"

        adjustedMetrics = self._recalculatePnl(taxDelta=taxDelta)
        return self._buildWhatIfResult(scenarioName, adjustedMetrics)

    def scenarioAnalysis(self, scenarios: Dict[str, Dict[str, float]]) -> List[WhatIfResult]:
        """
        Run multiple what-if scenarios in batch.

        여러 What-If 시나리오를 일괄 실행합니다.

        Each scenario is defined by a name and a dictionary of parameter
        adjustments. Supported keys: 'commission', 'slippage', 'tax'.

        Args:
            scenarios (dict): Mapping of scenario names to parameter dictionaries.
                Each parameter dict can contain 'commission', 'slippage', and/or 'tax'.
                시나리오 이름 -> 파라미터 딕셔너리 매핑.
                각 파라미터 딕셔너리에 'commission', 'slippage', 'tax' 키 사용 가능.

        Returns:
            List[WhatIfResult]: Results for each scenario.
                각 시나리오의 결과 목록.

        Example:
            >>> scenarios = {
            ...     "Low Cost": {"commission": 0.0001, "slippage": 0.0005},
            ...     "High Cost": {"commission": 0.002, "slippage": 0.003},
            ... }
            >>> results = simulator.scenarioAnalysis(scenarios)
        """
        results = []
        rates = self._estimateOriginalRates()

        for scenarioName, params in scenarios.items():
            commissionDelta = params.get('commission', rates['commission']) - rates['commission']
            slippageDelta = params.get('slippage', rates['slippage']) - rates['slippage']
            taxDelta = params.get('tax', rates['tax']) - rates['tax']

            adjustedMetrics = self._recalculatePnl(
                commissionDelta=commissionDelta,
                slippageDelta=slippageDelta,
                taxDelta=taxDelta,
            )

            whatIfResult = self._buildWhatIfResult(scenarioName, adjustedMetrics)
            results.append(whatIfResult)

        return results

    def sensitivityMatrix(
        self,
        commissions: List[float],
        slippages: List[float],
    ) -> pd.DataFrame:
        """
        Generate a 2D sensitivity matrix of returns across commission and slippage rates.

        수수료율과 슬리피지율 조합에 대한 수익률 2D 민감도 행렬을 생성합니다.

        Rows correspond to commission rates, columns to slippage rates, and
        cell values are adjusted total returns in percent.

        Args:
            commissions (List[float]): List of commission rates to test.
                테스트할 수수료율 목록.
            slippages (List[float]): List of slippage rates to test.
                테스트할 슬리피지율 목록.

        Returns:
            pd.DataFrame: Matrix with commission rates as index, slippage rates
                as columns, and adjusted returns (%) as values.
                수수료율(인덱스) x 슬리피지율(컬럼) 조정 수익률(%) 행렬.

        Example:
            >>> matrix = simulator.sensitivityMatrix(
            ...     commissions=[0.0001, 0.0005, 0.001],
            ...     slippages=[0.0005, 0.001, 0.002],
            ... )
            >>> print(matrix)
        """
        rates = self._estimateOriginalRates()
        matrixData = np.zeros((len(commissions), len(slippages)))

        for i, commRate in enumerate(commissions):
            for j, slipRate in enumerate(slippages):
                commDelta = commRate - rates['commission']
                slipDelta = slipRate - rates['slippage']

                adjustedMetrics = self._recalculatePnl(
                    commissionDelta=commDelta,
                    slippageDelta=slipDelta,
                )
                matrixData[i, j] = adjustedMetrics['totalReturn']

        commLabels = [f"{r:.5f}" for r in commissions]
        slipLabels = [f"{r:.5f}" for r in slippages]

        return pd.DataFrame(
            matrixData,
            index=pd.Index(commLabels, name='commission'),
            columns=pd.Index(slipLabels, name='slippage'),
        )

    def summary(self) -> str:
        """
        Generate a summary of all scenarios that have been run.

        실행된 모든 시나리오의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line formatted summary of all scenario results, or a
                message indicating no scenarios have been run.
        """
        if not self._scenarios:
            return "실행된 시나리오가 없습니다."

        lines = [
            f"{'='*60}",
            f"What-If 시뮬레이션 요약 ({len(self._scenarios)}개 시나리오)",
            f"{'='*60}",
            f"원래 수익률: {self._originalReturn:+.2f}%",
            f"{'─'*60}",
        ]

        for scenario in self._scenarios:
            lines.append(
                f"  {scenario.scenario}: "
                f"{scenario.adjustedReturn:+.2f}% "
                f"(영향: {scenario.impact:+.2f}%p, {scenario.impactPercent:+.1f}%)"
            )

        bestScenario = max(self._scenarios, key=lambda s: s.adjustedReturn)
        worstScenario = min(self._scenarios, key=lambda s: s.adjustedReturn)

        lines.extend([
            f"{'─'*60}",
            f"최선 시나리오: {bestScenario.scenario} ({bestScenario.adjustedReturn:+.2f}%)",
            f"최악 시나리오: {worstScenario.scenario} ({worstScenario.adjustedReturn:+.2f}%)",
            f"{'='*60}",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WhatIfSimulator(strategy={self._result.strategy}, "
            f"scenarios={len(self._scenarios)})"
        )
