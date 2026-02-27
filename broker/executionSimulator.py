"""
Tradix Execution Simulator Module.

Simulates realistic trade execution environments including slippage,
market impact, fill probability, and transaction cost analysis. Supports
multiple slippage and market impact models for comprehensive cost modeling.

실행 시뮬레이터 모듈 - 슬리피지, 시장 충격, 체결 확률을 포함한
실제 거래 환경을 시뮬레이션합니다.

Features:
    - Multiple slippage models (fixed, percentage, volatility-based, volume-based)
    - Market impact models (linear, square-root, Almgren-Chriss)
    - Fill probability estimation based on participation rate
    - Execution history tracking and analysis
    - Batch trade cost estimation

Usage:
    >>> from tradix.broker.executionSimulator import ExecutionSimulator, SlippageModel
    >>> simulator = ExecutionSimulator(
    ...     slippageModel=SlippageModel.VOLATILITY_BASED,
    ...     commissionRate=0.00015,
    ... )
    >>> result = simulator.simulate(
    ...     orderPrice=50000, orderQty=100, side='buy',
    ...     volatility=0.02, volume=1000000,
    ... )
    >>> print(result.slippagePct)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np


class SlippageModel(Enum):
    """
    Enumeration of available slippage calculation methods.

    슬리피지 계산 방법 열거형.
    """
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY_BASED = "volatility_based"
    VOLUME_BASED = "volume_based"


class MarketImpactModel(Enum):
    """
    Enumeration of available market impact calculation models.

    시장 충격 계산 모델 열거형.
    """
    NONE = "none"
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    ALMGREN_CHRISS = "almgren_chriss"


@dataclass
class ExecutionResult:
    """
    Data class holding the result of a single order execution simulation.

    개별 주문 실행 시뮬레이션 결과를 담는 데이터 클래스.

    Attributes:
        orderedPrice (float): Original order price. 주문 가격.
        executedPrice (float): Actual executed price after impacts. 체결 가격.
        slippage (float): Slippage amount in price units. 슬리피지 금액.
        slippagePct (float): Slippage as a fraction of order price. 슬리피지 비율.
        marketImpact (float): Market impact as a fraction. 시장 충격 비율.
        commission (float): Commission amount. 수수료 금액.
        totalCost (float): Total execution cost. 총 거래 비용.
        fillRatio (float): Estimated fill probability (0.0-1.0). 체결 비율.
        details (dict): Additional execution metadata. 추가 상세 정보.
    """
    orderedPrice: float
    executedPrice: float
    slippage: float
    slippagePct: float
    marketImpact: float
    commission: float
    totalCost: float
    fillRatio: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionAnalysis:
    """
    Aggregate analysis of multiple execution simulation results.

    복수 실행 시뮬레이션의 집계 분석 결과.

    Attributes:
        totalSlippage (float): Sum of slippage across all trades. 총 슬리피지.
        totalSlippagePct (float): Sum of slippage percentages. 총 슬리피지 비율.
        totalMarketImpact (float): Sum of market impact fractions. 총 시장 충격.
        totalCommission (float): Sum of commissions. 총 수수료.
        totalCost (float): Sum of total costs. 총 비용.
        avgSlippagePct (float): Average slippage percentage. 평균 슬리피지 비율.
        avgFillRatio (float): Average fill ratio. 평균 체결 비율.
        nTrades (int): Number of trades analyzed. 거래 수.
        costBreakdown (dict): Cost breakdown by category. 비용 내역.
    """
    totalSlippage: float
    totalSlippagePct: float
    totalMarketImpact: float
    totalCommission: float
    totalCost: float
    avgSlippagePct: float
    avgFillRatio: float
    nTrades: int
    costBreakdown: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"=== 실행 비용 분석 ===\n"
            f"총 거래 수: {self.nTrades}\n"
            f"총 슬리피지: {self.totalSlippagePct:.4%}\n"
            f"총 시장 충격: {self.totalMarketImpact:.4%}\n"
            f"총 수수료: {self.totalCommission:,.0f}\n"
            f"총 비용: {self.totalCost:,.0f}\n"
            f"평균 체결률: {self.avgFillRatio:.1%}"
        )


class ExecutionSimulator:
    """
    Execution simulator for modeling real-world trading friction costs.

    Combines slippage, market impact, and commission models to produce
    realistic execution cost estimates. Maintains an execution history
    for aggregate analysis.

    실행 시뮬레이터 - 슬리피지, 시장 충격, 수수료를 결합하여 실제 거래
    환경의 마찰 비용을 시뮬레이션합니다.

    Example:
        >>> simulator = ExecutionSimulator(
        ...     slippageModel=SlippageModel.VOLATILITY_BASED,
        ...     commissionRate=0.00015,
        ... )
        >>> result = simulator.simulate(
        ...     orderPrice=50000, orderQty=100, side='buy',
        ...     volatility=0.02, volume=1000000,
        ... )
        >>> analysis = simulator.analyzeHistory()
    """

    def __init__(
        self,
        slippageModel: SlippageModel = SlippageModel.PERCENTAGE,
        marketImpactModel: MarketImpactModel = MarketImpactModel.NONE,
        fixedSlippage: float = 0.0,
        slippagePct: float = 0.001,
        volatilityMultiplier: float = 0.5,
        volumeImpactCoef: float = 0.1,
        commissionRate: float = 0.00015,
        minCommission: float = 0,
    ):
        """
        Initialize the execution simulator.

        실행 시뮬레이터를 초기화합니다.

        Args:
            slippageModel (SlippageModel): Slippage calculation method.
                Default: PERCENTAGE. 슬리피지 모델.
            marketImpactModel (MarketImpactModel): Market impact model.
                Default: NONE. 시장 충격 모델.
            fixedSlippage (float): Fixed slippage in tick units (for
                FIXED model). Default: 0.0. 고정 슬리피지.
            slippagePct (float): Percentage slippage rate (for PERCENTAGE
                model). Default: 0.001. 비율 슬리피지.
            volatilityMultiplier (float): Volatility-based slippage
                multiplier. Default: 0.5. 변동성 계수.
            volumeImpactCoef (float): Volume-based impact coefficient.
                Default: 0.1. 거래량 충격 계수.
            commissionRate (float): Commission rate as fraction of trade
                value. Default: 0.00015. 수수료율.
            minCommission (float): Minimum commission per trade.
                Default: 0. 최소 수수료.
        """
        self.slippageModel = slippageModel
        self.marketImpactModel = marketImpactModel
        self.fixedSlippage = fixedSlippage
        self.slippagePct = slippagePct
        self.volatilityMultiplier = volatilityMultiplier
        self.volumeImpactCoef = volumeImpactCoef
        self.commissionRate = commissionRate
        self.minCommission = minCommission

        self.executionHistory: List[ExecutionResult] = []

    def simulate(
        self,
        orderPrice: float,
        orderQty: int,
        side: str,
        volatility: float = 0.02,
        volume: float = 1000000,
        spread: float = 0.001,
    ) -> ExecutionResult:
        """
        Simulate order execution with slippage and market impact.

        주문 실행을 시뮬레이션하고 슬리피지, 시장 충격을 적용합니다.

        Args:
            orderPrice (float): Order price. 주문 가격.
            orderQty (int): Order quantity in shares. 주문 수량.
            side (str): Order direction, 'buy' or 'sell'. 매수/매도.
            volatility (float): Daily volatility as a decimal.
                Default: 0.02. 일별 변동성.
            volume (float): Average daily volume. Default: 1000000.
                일평균 거래량.
            spread (float): Bid-ask spread. Default: 0.001. 스프레드.

        Returns:
            ExecutionResult: Detailed execution outcome including
                slippage, market impact, and total cost.
        """
        direction = 1 if side.lower() == 'buy' else -1

        slippage = self._calcSlippage(orderPrice, volatility, volume, orderQty)

        marketImpact = self._calcMarketImpact(
            orderPrice, orderQty, volume, volatility
        )

        totalPriceImpact = (slippage + marketImpact) * direction
        executedPrice = orderPrice + totalPriceImpact

        orderValue = orderPrice * orderQty
        commission = max(orderValue * self.commissionRate, self.minCommission)

        totalCost = abs(totalPriceImpact) * orderQty + commission

        fillRatio = self._calcFillRatio(orderQty, volume)

        result = ExecutionResult(
            orderedPrice=orderPrice,
            executedPrice=executedPrice,
            slippage=slippage,
            slippagePct=slippage / orderPrice,
            marketImpact=marketImpact / orderPrice,
            commission=commission,
            totalCost=totalCost,
            fillRatio=fillRatio,
            details={
                'direction': direction,
                'orderQty': orderQty,
                'volume': volume,
                'volatility': volatility,
            }
        )

        self.executionHistory.append(result)
        return result

    def _calcSlippage(
        self,
        price: float,
        volatility: float,
        volume: float,
        qty: int
    ) -> float:
        """Calculate slippage amount based on the configured model. 슬리피지 계산."""
        if self.slippageModel == SlippageModel.FIXED:
            return self.fixedSlippage

        elif self.slippageModel == SlippageModel.PERCENTAGE:
            return price * self.slippagePct

        elif self.slippageModel == SlippageModel.VOLATILITY_BASED:
            return price * volatility * self.volatilityMultiplier * np.random.uniform(0.5, 1.5)

        elif self.slippageModel == SlippageModel.VOLUME_BASED:
            participationRate = qty / volume if volume > 0 else 0.01
            return price * self.slippagePct * (1 + participationRate * 10)

        return 0

    def _calcMarketImpact(
        self,
        price: float,
        qty: int,
        volume: float,
        volatility: float
    ) -> float:
        """Calculate market impact based on the configured model. 시장 충격 계산."""
        if self.marketImpactModel == MarketImpactModel.NONE:
            return 0

        participationRate = qty / volume if volume > 0 else 0.01

        if self.marketImpactModel == MarketImpactModel.LINEAR:
            impact = price * self.volumeImpactCoef * participationRate

        elif self.marketImpactModel == MarketImpactModel.SQUARE_ROOT:
            impact = price * self.volumeImpactCoef * np.sqrt(participationRate)

        elif self.marketImpactModel == MarketImpactModel.ALMGREN_CHRISS:
            eta = 0.1
            gamma = 0.5
            impact = price * eta * volatility * (participationRate ** gamma)

        else:
            impact = 0

        return impact

    def _calcFillRatio(self, orderQty: int, volume: float) -> float:
        """Estimate fill ratio based on order participation rate. 체결 비율 계산."""
        if volume <= 0:
            return 0.5

        participationRate = orderQty / volume

        if participationRate < 0.01:
            return 1.0
        elif participationRate < 0.05:
            return 0.95
        elif participationRate < 0.10:
            return 0.85
        elif participationRate < 0.20:
            return 0.70
        else:
            return 0.50

    def analyzeHistory(self) -> ExecutionAnalysis:
        """
        Analyze accumulated execution history and return aggregate metrics.

        축적된 실행 히스토리를 분석하여 집계 메트릭을 반환합니다.

        Returns:
            ExecutionAnalysis: Aggregate statistics across all recorded
                executions.
        """
        if not self.executionHistory:
            return ExecutionAnalysis(
                totalSlippage=0, totalSlippagePct=0, totalMarketImpact=0,
                totalCommission=0, totalCost=0, avgSlippagePct=0,
                avgFillRatio=0, nTrades=0
            )

        totalSlippage = sum(r.slippage for r in self.executionHistory)
        totalSlippagePct = sum(r.slippagePct for r in self.executionHistory)
        totalMarketImpact = sum(r.marketImpact for r in self.executionHistory)
        totalCommission = sum(r.commission for r in self.executionHistory)
        totalCost = sum(r.totalCost for r in self.executionHistory)
        avgSlippagePct = totalSlippagePct / len(self.executionHistory)
        avgFillRatio = np.mean([r.fillRatio for r in self.executionHistory])

        return ExecutionAnalysis(
            totalSlippage=totalSlippage,
            totalSlippagePct=totalSlippagePct,
            totalMarketImpact=totalMarketImpact,
            totalCommission=totalCommission,
            totalCost=totalCost,
            avgSlippagePct=avgSlippagePct,
            avgFillRatio=avgFillRatio,
            nTrades=len(self.executionHistory),
            costBreakdown={
                'slippage': totalSlippage,
                'marketImpact': totalMarketImpact,
                'commission': totalCommission,
            }
        )

    def clearHistory(self):
        """Clear the execution history. 실행 히스토리를 초기화합니다."""
        self.executionHistory = []

    def estimateCost(
        self,
        trades: List[Dict],
        avgVolatility: float = 0.02,
        avgVolume: float = 1000000,
    ) -> Dict[str, float]:
        """
        Estimate total transaction costs for a batch of trades.

        다수 거래의 총 예상 거래 비용을 추정합니다.

        Args:
            trades (list[dict]): List of trade dicts, each with keys
                'price' (float), 'qty' (int), and 'side' (str).
                거래 목록.
            avgVolatility (float): Average daily volatility. Default: 0.02.
                평균 변동성.
            avgVolume (float): Average daily volume. Default: 1000000.
                평균 거래량.

        Returns:
            dict: Cost estimation with keys 'totalCost', 'costPct',
                'avgSlippage', 'commission', and 'nTrades'.
        """
        self.clearHistory()

        for trade in trades:
            self.simulate(
                orderPrice=trade['price'],
                orderQty=trade['qty'],
                side=trade['side'],
                volatility=avgVolatility,
                volume=avgVolume,
            )

        analysis = self.analyzeHistory()

        return {
            'totalCost': analysis.totalCost,
            'costPct': analysis.totalSlippagePct + analysis.totalMarketImpact,
            'avgSlippage': analysis.avgSlippagePct,
            'commission': analysis.totalCommission,
            'nTrades': analysis.nTrades,
        }
