"""
Tradix Slippage Models Module.

Provides slippage simulation models to account for the difference between
expected and actual fill prices during backtesting.

슬리피지 모델 모듈 - 백테스트 시 체결 가격 차이를 시뮬레이션합니다.

Features:
    - Abstract SlippageModel base for custom implementations
    - Fixed amount and percentage-based slippage
    - Volume-dependent market impact slippage
    - Bid-ask spread simulation
    - Random slippage for stress testing
    - Zero-slippage model for frictionless testing

Usage:
    >>> from tradix.broker.slippage import PercentSlippage
    >>> slippage_model = PercentSlippage(rate=0.001)
    >>> adjusted_price = slippage_model.apply(price=50000.0, order=order, bar=bar)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from tradix.entities.order import Order, OrderSide
from tradix.entities.bar import Bar


class SlippageModel(ABC):
    """
    Abstract base class for slippage simulation models.

    Subclass this to implement custom slippage logic representing
    the cost of market friction during order execution.

    슬리피지 시뮬레이션 모델의 추상 기반 클래스입니다.
    """

    @abstractmethod
    def apply(self, price: float, order: Order, bar: Bar) -> float:
        """
        Apply slippage to a base price and return the adjusted price.

        기준 가격에 슬리피지를 적용하여 조정된 가격을 반환합니다.

        Args:
            price (float): Base fill price before slippage. 기준 가격.
            order (Order): The order being executed. 실행 중인 주문.
            bar (Bar): Current bar data for market context. 현재 바 데이터.

        Returns:
            float: Adjusted price after slippage. Buy orders get a
                higher price, sell orders get a lower price.
        """
        pass


class NoSlippage(SlippageModel):
    """
    Zero slippage model for frictionless testing.

    Returns the input price unchanged. Useful for isolating strategy
    logic from market friction effects.

    슬리피지 없음 모델 (테스트용) - 입력 가격을 그대로 반환합니다.
    """

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        return price


@dataclass
class FixedSlippage(SlippageModel):
    """
    Fixed amount slippage model.

    Adds or subtracts a constant amount from the fill price
    depending on order direction.

    고정 금액 슬리피지 모델 - 방향에 따라 일정 금액을 가감합니다.

    Attributes:
        amount (float): Fixed slippage amount in price units.
            Default: 10.0. 고정 슬리피지 금액.
    """
    amount: float = 10.0

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        if order.side == OrderSide.BUY:
            return price + self.amount
        return price - self.amount


@dataclass
class PercentSlippage(SlippageModel):
    """
    Percentage-based slippage model.

    Applies slippage as a fixed percentage of the base price.

    비율 기반 슬리피지 모델 - 기준 가격의 일정 비율을 슬리피지로 적용합니다.

    Attributes:
        rate (float): Slippage rate as a decimal (0.001 = 0.1%).
            Default: 0.001. 슬리피지율.
    """
    rate: float = 0.001

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        slippage = price * self.rate
        if order.side == OrderSide.BUY:
            return price + slippage
        return price - slippage


@dataclass
class VolumeSlippage(SlippageModel):
    """
    Volume-based slippage model simulating market impact.

    Slippage increases proportionally to the order's participation
    rate (order quantity / bar volume), modeling the price impact
    of large orders on the market.

    거래량 기반 슬리피지 (시장 충격 모델) - 주문 수량이 거래량 대비
    클수록 슬리피지가 증가합니다.

    Attributes:
        impactFactor (float): Market impact coefficient. Default: 0.1.
            충격 계수.
        maxSlippage (float): Maximum slippage rate cap (0.02 = 2%).
            Default: 0.02. 최대 슬리피지율.

    Example:
        >>> model = VolumeSlippage(impactFactor=0.1, maxSlippage=0.02)
        >>> adjusted = model.apply(price=50000.0, order=order, bar=bar)
    """
    impactFactor: float = 0.1
    maxSlippage: float = 0.02

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        if bar.volume <= 0:
            volumeRatio = 0.01
        else:
            volumeRatio = order.quantity / bar.volume

        slippageRate = min(volumeRatio * self.impactFactor, self.maxSlippage)
        slippage = price * slippageRate

        if order.side == OrderSide.BUY:
            return price + slippage
        return price - slippage


@dataclass
class SpreadSlippage(SlippageModel):
    """
    Bid-ask spread based slippage model.

    Simulates the cost of crossing the bid-ask spread by applying
    half the spread to the fill price in the adverse direction.

    호가 스프레드 기반 슬리피지 모델 - 매수/매도 호가 차이를 시뮬레이션합니다.

    Attributes:
        spreadRate (float): Total bid-ask spread as a fraction of
            price (0.001 = 0.1%). Default: 0.001. 스프레드율.
    """
    spreadRate: float = 0.001

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        halfSpread = price * self.spreadRate / 2

        if order.side == OrderSide.BUY:
            return price + halfSpread
        return price - halfSpread


@dataclass
class RandomSlippage(SlippageModel):
    """
    Random slippage model for stress testing.

    Applies a uniformly random slippage rate between configurable
    minimum and maximum bounds on each execution.

    랜덤 슬리피지 모델 (스트레스 테스트용) - 최소/최대 범위 내에서
    무작위 슬리피지를 적용합니다.

    Attributes:
        minRate (float): Minimum slippage rate. Default: 0.0.
            최소 슬리피지율.
        maxRate (float): Maximum slippage rate. Default: 0.002.
            최대 슬리피지율.
    """
    minRate: float = 0.0
    maxRate: float = 0.002

    def apply(self, price: float, order: Order, bar: Bar) -> float:
        rate = random.uniform(self.minRate, self.maxRate)
        slippage = price * rate

        if order.side == OrderSide.BUY:
            return price + slippage
        return price - slippage
