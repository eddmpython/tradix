"""
Tradix Broker Simulator Module.

Simulates order execution with realistic commission, slippage, and fill
price modeling for backtesting trading strategies.

브로커 시뮬레이터 모듈 - 백테스트 시 주문 체결을 시뮬레이션합니다.

Features:
    - Pluggable commission models (Korea, US, fixed, percent)
    - Configurable slippage simulation (percent, volume-based)
    - Multiple fill price strategies (close, open, VWAP, worst-case)
    - Factory methods for common broker presets (Korea, US, ideal, conservative)
    - Transaction cost estimation before order execution

Usage:
    >>> from tradix.broker.simulator import BrokerSimulator
    >>> broker = BrokerSimulator.korea(mobileApp=True)
    >>> fill = broker.processOrder(order, bar)
"""

from dataclasses import dataclass
from typing import Optional

from tradix.entities.order import Order
from tradix.entities.bar import Bar
from tradix.events.fill import FillEvent
from tradix.broker.commission import CommissionModel, KoreaStockCommission
from tradix.broker.slippage import SlippageModel, PercentSlippage
from tradix.broker.fill import FillModel, CloseFill


@dataclass
class BrokerSimulator:
    """
    Broker simulator for realistic order execution modeling.

    Simulates the full order execution pipeline including fill price
    determination, slippage application, and commission calculation.
    Supports pluggable models for each component, enabling flexible
    cost modeling across different markets and scenarios.

    브로커 시뮬레이터 - 체결 가격, 슬리피지, 수수료를 포함한 주문 체결을 시뮬레이션합니다.

    Attributes:
        commissionModel (CommissionModel): Commission calculation model.
            Default: KoreaStockCommission (한국 주식 수수료).
        slippageModel (SlippageModel): Slippage simulation model.
            Default: PercentSlippage(0.001) (0.1% slippage).
        fillModel (FillModel): Fill price determination model.
            Default: CloseFill (종가 체결).

    Example:
        >>> broker = BrokerSimulator()
        >>> broker = BrokerSimulator(
        ...     commissionModel=KoreaStockCommission.mobileApp(),
        ...     slippageModel=PercentSlippage(0.0005),
        ...     fillModel=VwapFill()
        ... )
        >>> fill = broker.processOrder(order, bar)
    """
    commissionModel: CommissionModel = None
    slippageModel: SlippageModel = None
    fillModel: FillModel = None

    def __post_init__(self):
        if self.commissionModel is None:
            self.commissionModel = KoreaStockCommission()
        if self.slippageModel is None:
            self.slippageModel = PercentSlippage(0.001)
        if self.fillModel is None:
            self.fillModel = CloseFill()

    def processOrder(self, order: Order, bar: Bar) -> FillEvent:
        """
        Process an order and generate a fill event.

        Determines the fill price, applies slippage, and calculates
        commission to produce a complete FillEvent.

        주문을 처리하고 체결 이벤트를 생성합니다.

        Args:
            order (Order): The order to execute. 실행할 주문.
            bar (Bar): Current bar data for price reference. 현재 바 데이터.

        Returns:
            FillEvent: The resulting fill event with price, quantity,
                commission, and slippage details. Returns zero-filled
                event if fill price is non-positive.
        """
        basePrice = self.fillModel.getFillPrice(order, bar)

        if basePrice <= 0:
            return FillEvent(
                timestamp=bar.datetime,
                order=order,
                fillPrice=0.0,
                fillQuantity=0.0,
                commission=0.0,
                slippage=0.0
            )

        fillPrice = self.slippageModel.apply(basePrice, order, bar)

        commission = self.commissionModel.calculate(order, fillPrice)

        slippageAmount = abs(fillPrice - basePrice) * order.quantity

        return FillEvent(
            timestamp=bar.datetime,
            order=order,
            fillPrice=fillPrice,
            fillQuantity=order.quantity,
            commission=commission,
            slippage=slippageAmount
        )

    def estimateCost(self, order: Order, price: float) -> dict:
        """
        Estimate transaction costs for a hypothetical order.

        Calculates expected commission, slippage, and total cost
        without actually executing the order.

        주문 실행 전 예상 거래 비용을 추정합니다.

        Args:
            order (Order): The order to estimate costs for. 비용을 추정할 주문.
            price (float): Expected fill price. 예상 체결 가격.

        Returns:
            dict: Cost breakdown with keys 'value', 'commission',
                'slippage', 'totalCost', and 'costPercent'.
        """
        value = order.quantity * price
        commission = self.commissionModel.calculate(order, price)

        mockBar = Bar(
            symbol=order.symbol,
            datetime=None,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000000
        )
        slippedPrice = self.slippageModel.apply(price, order, mockBar)
        slippage = abs(slippedPrice - price) * order.quantity

        return {
            'value': value,
            'commission': commission,
            'slippage': slippage,
            'totalCost': commission + slippage,
            'costPercent': (commission + slippage) / value * 100 if value > 0 else 0
        }

    @classmethod
    def korea(cls, mobileApp: bool = True) -> 'BrokerSimulator':
        """
        Create a broker simulator configured for Korean stock market.

        한국 주식 시장에 맞게 설정된 브로커 시뮬레이터를 생성합니다.

        Args:
            mobileApp (bool): If True, use mobile app commission rates
                (typically zero broker fee). Default: True.

        Returns:
            BrokerSimulator: Configured for Korean market.
        """
        if mobileApp:
            commission = KoreaStockCommission.mobileApp()
        else:
            commission = KoreaStockCommission.hts()

        return cls(
            commissionModel=commission,
            slippageModel=PercentSlippage(0.001),
            fillModel=CloseFill()
        )

    @classmethod
    def us(cls) -> 'BrokerSimulator':
        """
        Create a broker simulator configured for US stock market.

        미국 주식 시장에 맞게 설정된 브로커 시뮬레이터를 생성합니다.

        Returns:
            BrokerSimulator: Configured with Interactive Brokers commission.
        """
        from tradix.broker.commission import USStockCommission
        return cls(
            commissionModel=USStockCommission.interactiveBrokers(),
            slippageModel=PercentSlippage(0.001),
            fillModel=CloseFill()
        )

    @classmethod
    def ideal(cls) -> 'BrokerSimulator':
        """
        Create an ideal broker with zero commission and slippage.

        Useful for testing strategy logic in isolation without
        transaction cost interference.

        수수료/슬리피지 없는 이상적 브로커 (테스트용).

        Returns:
            BrokerSimulator: Zero-cost broker simulator.
        """
        from tradix.broker.commission import NoCommission
        from tradix.broker.slippage import NoSlippage
        return cls(
            commissionModel=NoCommission(),
            slippageModel=NoSlippage(),
            fillModel=CloseFill()
        )

    @classmethod
    def conservative(cls) -> 'BrokerSimulator':
        """
        Create a conservative broker with high transaction costs.

        Uses worst-case fill prices, volume-based slippage, and
        higher commission rates for pessimistic backtesting.

        높은 거래 비용의 보수적 브로커 (보수적 백테스트용).

        Returns:
            BrokerSimulator: High-cost broker simulator.
        """
        from tradix.broker.slippage import VolumeSlippage
        from tradix.broker.fill import WorstFill
        return cls(
            commissionModel=KoreaStockCommission(brokerFee=0.0003, taxRate=0.0023),
            slippageModel=VolumeSlippage(impactFactor=0.2),
            fillModel=WorstFill()
        )

    def __repr__(self) -> str:
        return (
            f"BrokerSimulator("
            f"commission={self.commissionModel.__class__.__name__}, "
            f"slippage={self.slippageModel.__class__.__name__}, "
            f"fill={self.fillModel.__class__.__name__})"
        )
