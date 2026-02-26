"""
Tradex Commission Models Module.

Provides commission calculation models for simulating brokerage fees
during backtesting. Includes presets for Korean and US stock markets.

수수료 모델 모듈 - 백테스트 시 증권사 수수료를 계산합니다.

Features:
    - Abstract CommissionModel base for custom implementations
    - Korean stock commission with broker fee and transaction tax
    - US stock per-share commission with min/max caps
    - Fixed and percentage-based commission models
    - Zero-commission model for frictionless testing

Usage:
    >>> from tradex.broker.commission import KoreaStockCommission
    >>> commission_model = KoreaStockCommission.mobileApp()
    >>> fee = commission_model.calculate(order, fill_price=50000.0)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tradex.entities.order import Order, OrderSide


class CommissionModel(ABC):
    """
    Abstract base class for commission calculation models.

    Subclass this to implement custom commission logic for different
    brokers or markets.

    수수료 계산 모델의 추상 기반 클래스입니다.
    """

    @abstractmethod
    def calculate(self, order: Order, fillPrice: float) -> float:
        """
        Calculate commission for a given order and fill price.

        주문과 체결 가격에 대한 수수료를 계산합니다.

        Args:
            order (Order): The executed order. 실행된 주문.
            fillPrice (float): The price at which the order was filled.
                체결 가격.

        Returns:
            float: Commission amount in currency units. 수수료 금액.
        """
        pass


class NoCommission(CommissionModel):
    """
    Zero commission model for frictionless testing.

    Always returns 0.0 commission regardless of order details.
    Useful for isolating strategy logic from transaction costs.

    수수료 없음 모델 (테스트용) - 항상 0을 반환합니다.
    """

    def calculate(self, order: Order, fillPrice: float) -> float:
        return 0.0


@dataclass
class FixedCommission(CommissionModel):
    """
    Fixed commission model charging a constant fee per trade.

    Charges the same amount regardless of order size or price.

    거래당 고정 금액을 부과하는 수수료 모델입니다.

    Attributes:
        amount (float): Fixed commission per trade in currency units.
            Default: 5000.0. 거래당 고정 수수료.
    """
    amount: float = 5000.0

    def calculate(self, order: Order, fillPrice: float) -> float:
        return self.amount


@dataclass
class PercentCommission(CommissionModel):
    """
    Percentage-based commission model.

    Charges a percentage of the total trade value with an optional
    minimum commission floor.

    거래 금액의 일정 비율을 수수료로 부과하는 모델입니다.

    Attributes:
        rate (float): Commission rate as a decimal (0.001 = 0.1%).
            Default: 0.001. 수수료율.
        minimum (float): Minimum commission amount. Default: 0.0.
            최소 수수료.
    """
    rate: float = 0.001
    minimum: float = 0.0

    def calculate(self, order: Order, fillPrice: float) -> float:
        value = order.quantity * fillPrice
        commission = value * self.rate
        return max(commission, self.minimum)


@dataclass
class KoreaStockCommission(CommissionModel):
    """
    Korean stock market commission model.

    Applies broker commission on both buy and sell, plus transaction
    tax (securities transaction tax) on sell orders only.

    한국 주식 수수료 모델 - 매수 시 증권사 수수료, 매도 시 수수료 + 거래세.

    Attributes:
        brokerFee (float): Broker commission rate (0.00015 = 0.015%).
            Default: 0.00015. 증권사 수수료율.
        taxRate (float): Securities transaction tax rate (0.0018 = 0.18%).
            Applied only on sell orders. Default: 0.0018. 거래세율.
        minimum (float): Minimum commission amount. Default: 0.0.
            최소 수수료.

    Example:
        >>> model = KoreaStockCommission.mobileApp()
        >>> fee = model.calculate(sell_order, fillPrice=50000.0)
    """
    brokerFee: float = 0.00015
    taxRate: float = 0.0018
    minimum: float = 0.0

    def calculate(self, order: Order, fillPrice: float) -> float:
        value = order.quantity * fillPrice

        commission = value * self.brokerFee

        if order.side == OrderSide.SELL:
            commission += value * self.taxRate

        return max(commission, self.minimum)

    @classmethod
    def mobileApp(cls) -> 'KoreaStockCommission':
        """
        Create commission model for Korean mobile app trading.

        한국 모바일 앱 거래 수수료 (증권사 수수료 무료, 거래세만 적용).

        Returns:
            KoreaStockCommission: Model with zero broker fee.
        """
        return cls(brokerFee=0.0, taxRate=0.0018)

    @classmethod
    def hts(cls) -> 'KoreaStockCommission':
        """
        Create commission model for Korean HTS (Home Trading System).

        한국 HTS 거래 수수료 (증권사 수수료 0.015% + 거래세).

        Returns:
            KoreaStockCommission: Model with standard HTS broker fee.
        """
        return cls(brokerFee=0.00015, taxRate=0.0018)

    @classmethod
    def premium(cls) -> 'KoreaStockCommission':
        """
        Create commission model for premium/institutional trading.

        프리미엄/기관 투자자 수수료 (증권사 수수료 0.01% + 거래세).

        Returns:
            KoreaStockCommission: Model with reduced institutional fee.
        """
        return cls(brokerFee=0.0001, taxRate=0.0018)


@dataclass
class USStockCommission(CommissionModel):
    """
    US stock market commission model.

    Charges a per-share fee with minimum and maximum caps.
    The maximum is expressed as a percentage of trade value.

    미국 주식 수수료 모델 - 주당 수수료에 최소/최대 한도를 적용합니다.

    Attributes:
        perShare (float): Commission per share in USD. Default: 0.005.
            주당 수수료.
        minimum (float): Minimum commission per trade in USD. Default: 1.0.
            최소 수수료.
        maximum (float): Maximum commission as fraction of trade value
            (0.01 = 1%). Default: 0.01. 최대 수수료율.

    Example:
        >>> model = USStockCommission.interactiveBrokers()
        >>> fee = model.calculate(order, fillPrice=150.0)
    """
    perShare: float = 0.005
    minimum: float = 1.0
    maximum: float = 0.01

    def calculate(self, order: Order, fillPrice: float) -> float:
        commission = order.quantity * self.perShare
        commission = max(commission, self.minimum)

        value = order.quantity * fillPrice
        maxCommission = value * self.maximum
        commission = min(commission, maxCommission)

        return commission

    @classmethod
    def interactiveBrokers(cls) -> 'USStockCommission':
        """
        Create commission model for Interactive Brokers.

        인터랙티브 브로커스 수수료 설정.

        Returns:
            USStockCommission: Model with IB's standard tiered pricing.
        """
        return cls(perShare=0.005, minimum=1.0, maximum=0.01)

    @classmethod
    def free(cls) -> 'USStockCommission':
        """
        Create zero-commission model for commission-free brokers.

        무료 수수료 (Robinhood 등 무수수료 브로커).

        Returns:
            USStockCommission: Model with zero commission.
        """
        return cls(perShare=0.0, minimum=0.0, maximum=0.0)
