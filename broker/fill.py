"""
Tradex Fill Price Models Module.

Provides fill price determination strategies for order execution simulation
during backtesting. Each model defines how the execution price is derived
from OHLCV bar data.

체결 가격 모델 모듈 - 백테스트 시 주문 체결 가격을 결정합니다.

Features:
    - Abstract FillModel base for custom implementations
    - Close price fill (simplest, default)
    - Open price fill (more realistic for next-bar execution)
    - VWAP approximation fill using (H+L+C)/3
    - Worst-case fill for conservative backtesting (buy at high, sell at low)
    - Best-case fill for optimistic scenarios (buy at low, sell at high)
    - Random fill within bar range for stress testing
    - Limit order support across all fill models

Usage:
    >>> from tradex.broker.fill import VwapFill
    >>> fill_model = VwapFill()
    >>> price = fill_model.getFillPrice(order, bar)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from tradex.entities.order import Order, OrderType
from tradex.entities.bar import Bar


class FillModel(ABC):
    """
    Abstract base class for fill price determination models.

    Subclass this to implement custom fill price logic defining how
    the execution price is derived from bar data and order parameters.

    체결 가격 결정 모델의 추상 기반 클래스입니다.
    """

    @abstractmethod
    def getFillPrice(self, order: Order, bar: Bar) -> float:
        """
        Determine the fill price for an order given bar data.

        주어진 바 데이터를 기반으로 주문의 체결 가격을 결정합니다.

        Args:
            order (Order): The order to fill. 체결할 주문.
            bar (Bar): Current OHLCV bar data. 현재 바 데이터.

        Returns:
            float: Fill price. Returns 0.0 if a limit order cannot
                be filled within the bar's price range.
        """
        pass


class CloseFill(FillModel):
    """
    Close price fill model.

    Fills orders at the bar's closing price. Simplest model but
    least realistic since close price is not known at order time.
    Supports limit orders by checking price feasibility within
    the bar's range.

    종가 체결 모델 - 당일 종가로 체결합니다 (가장 간단하지만 비현실적).
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        if order.orderType == OrderType.LIMIT and order.price:
            if order.isBuy and bar.low <= order.price:
                return min(order.price, bar.close)
            elif order.isSell and bar.high >= order.price:
                return max(order.price, bar.close)
            return 0.0

        return bar.close


class OpenFill(FillModel):
    """
    Open price fill model.

    Fills orders at the bar's opening price. More realistic than
    close fill when simulating next-bar execution. Note that in
    practice, the bar passed should be the next bar after the
    signal bar.

    시가 체결 모델 - 시가로 체결합니다 (종가 체결보다 현실적).
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        if order.orderType == OrderType.LIMIT and order.price:
            if order.isBuy and bar.low <= order.price:
                return min(order.price, bar.open)
            elif order.isSell and bar.high >= order.price:
                return max(order.price, bar.open)
            return 0.0

        return bar.open


class VwapFill(FillModel):
    """
    VWAP approximation fill model.

    Fills orders at an approximate VWAP calculated as
    (High + Low + Close) / 3. Provides a more representative
    average execution price than close or open fills.

    VWAP 근사값 체결 모델 - (고가 + 저가 + 종가) / 3으로 체결합니다.
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        vwap = (bar.high + bar.low + bar.close) / 3

        if order.orderType == OrderType.LIMIT and order.price:
            if order.isBuy and order.price >= bar.low:
                return min(order.price, vwap)
            elif order.isSell and order.price <= bar.high:
                return max(order.price, vwap)
            return 0.0

        return vwap


class RandomFill(FillModel):
    """
    Random fill model for stress testing.

    Fills orders at a uniformly random price within the bar's
    high-low range. Useful for Monte Carlo-style testing of
    strategy robustness to fill price uncertainty.

    랜덤 체결 모델 (테스트용) - 고가와 저가 사이의 무작위 가격으로 체결합니다.
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        if bar.high == bar.low:
            return bar.close

        price = random.uniform(bar.low, bar.high)

        if order.orderType == OrderType.LIMIT and order.price:
            if order.isBuy:
                return min(order.price, price) if order.price >= bar.low else 0.0
            else:
                return max(order.price, price) if order.price <= bar.high else 0.0

        return price


@dataclass
class WorstFill(FillModel):
    """
    Worst-case fill model for conservative backtesting.

    Fills buy orders at the bar's high and sell orders at the
    bar's low, representing the worst possible execution price.
    Useful for stress testing strategy robustness.

    최악 체결 모델 (보수적 테스트용) - 매수: 고가, 매도: 저가로 체결합니다.
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        if order.isBuy:
            return bar.high
        return bar.low


@dataclass
class BestFill(FillModel):
    """
    Best-case fill model for optimistic backtesting.

    Fills buy orders at the bar's low and sell orders at the
    bar's high, representing the best possible execution price.
    Useful for estimating the upper bound of strategy performance.

    최선 체결 모델 (낙관적 테스트용) - 매수: 저가, 매도: 고가로 체결합니다.
    """

    def getFillPrice(self, order: Order, bar: Bar) -> float:
        if order.isBuy:
            return bar.low
        return bar.high
