"""
Tradix Position Sizing Models Module.

Provides position sizing strategies that determine the number of shares
to trade based on equity, price, risk parameters, and historical
performance metrics.

포지션 사이징 모델 모듈 - 자산, 가격, 리스크 파라미터를 기반으로
매매 수량을 결정합니다.

Features:
    - Fixed quantity and fixed amount sizers
    - Percent of equity sizer with configurable caps
    - Fixed risk sizer based on stop-loss distance
    - Kelly criterion sizer with fractional Kelly support
    - Volatility-based sizer using ATR (Average True Range)

Usage:
    >>> from tradix.risk.sizing import PercentEquitySizer
    >>> sizer = PercentEquitySizer(percent=0.1)
    >>> qty = sizer.calculate(equity=10_000_000, price=50000)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from tradix.entities.trade import Trade


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.

    Subclass this to implement custom position sizing logic that
    determines how many shares to buy based on equity, price, and
    additional parameters.

    포지션 사이징 전략의 추상 기반 클래스입니다.
    """

    @abstractmethod
    def calculate(self, equity: float, price: float, **kwargs) -> int:
        """
        Calculate the number of shares to trade.

        자산과 가격을 기반으로 매매 수량을 계산합니다.

        Args:
            equity (float): Current portfolio equity. 현재 자산.
            price (float): Current price per share. 현재 가격.
            **kwargs: Additional parameters (e.g., stopLoss, atr, trades).
                추가 파라미터.

        Returns:
            int: Number of shares to trade. 매매 수량.
        """
        pass


@dataclass
class FixedQuantitySizer(PositionSizer):
    """
    Fixed quantity position sizer.

    Always returns the same number of shares regardless of equity
    or price. Simplest sizing model.

    고정 수량 사이저 - 항상 동일한 수량을 반환합니다.

    Attributes:
        quantity (int): Fixed number of shares per trade. Default: 10.
            고정 매수 수량.
    """
    quantity: int = 10

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        return self.quantity


@dataclass
class FixedAmountSizer(PositionSizer):
    """
    Fixed amount position sizer.

    Invests a fixed currency amount per trade, calculating the
    number of shares based on the current price.

    고정 금액 사이저 - 거래당 고정 금액을 투자합니다.

    Attributes:
        amount (float): Fixed investment amount per trade.
            Default: 1,000,000. 고정 투자 금액.
    """
    amount: float = 1_000_000

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        if price <= 0:
            return 0
        return int(self.amount / price)


@dataclass
class PercentEquitySizer(PositionSizer):
    """
    Percent of equity position sizer.

    Invests a configurable percentage of current equity per trade,
    with an optional maximum percentage cap.

    자산 비율 사이저 - 현재 자산의 일정 비율을 투자합니다.

    Attributes:
        percent (float): Target investment as fraction of equity
            (0.1 = 10%). Default: 0.1. 자산 대비 투자 비율.
        maxPercent (float): Maximum allowed percentage (0-1.0).
            Default: 1.0. 최대 비율.
    """
    percent: float = 0.1
    maxPercent: float = 1.0

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        if price <= 0:
            return 0

        percent = min(self.percent, self.maxPercent)
        amount = equity * percent
        return int(amount / price)


@dataclass
class FixedRiskSizer(PositionSizer):
    """
    Fixed risk position sizer.

    Limits maximum loss per trade to a fixed percentage of equity.
    Calculates position size based on the distance between entry
    price and stop-loss level.

    고정 리스크 사이저 - 거래당 최대 손실을 자산의 일정 비율로 제한합니다.

    Attributes:
        riskPercent (float): Maximum risk per trade as fraction of
            equity (0.02 = 2%). Default: 0.02. 거래당 리스크 비율.
        defaultStopPercent (float): Default stop-loss distance as
            fraction of price, used when no stop-loss is specified.
            Default: 0.05. 기본 손절 비율.

    Example:
        >>> sizer = FixedRiskSizer(riskPercent=0.02)
        >>> qty = sizer.calculate(equity=10_000_000, price=50000, stopLoss=47500)
    """
    riskPercent: float = 0.02
    defaultStopPercent: float = 0.05

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        if price <= 0:
            return 0

        stopLoss = kwargs.get('stopLoss')

        if stopLoss and stopLoss > 0:
            riskPerShare = abs(price - stopLoss)
        else:
            riskPerShare = price * self.defaultStopPercent

        if riskPerShare <= 0:
            return 0

        maxRiskAmount = equity * self.riskPercent

        quantity = int(maxRiskAmount / riskPerShare)
        return max(1, quantity)


@dataclass
class KellySizer(PositionSizer):
    """
    Kelly criterion position sizer.

    Calculates optimal position size using the Kelly formula:
    f = (b*p - q) / b, where b is the win/loss ratio, p is the
    win rate, and q is the loss rate (1 - p). Supports fractional
    Kelly for more conservative sizing.

    켈리 공식 기반 사이저 - 승률과 손익비를 기반으로 최적 투자 비율을
    계산합니다.

    Attributes:
        fraction (float): Kelly fraction (0.5 = Half Kelly for
            conservative sizing). Default: 0.5. 켈리 비율.
        minTrades (int): Minimum closed trades required for
            statistical significance. Default: 20. 최소 거래 수.
        defaultPercent (float): Fallback percent when insufficient
            trade history. Default: 0.1. 기본 투자 비율.
        maxPercent (float): Maximum position size cap as fraction
            of equity. Default: 0.25. 최대 투자 비율.

    Example:
        >>> sizer = KellySizer(fraction=0.5, minTrades=20)
        >>> qty = sizer.calculate(equity=10_000_000, price=50000, trades=trade_list)
    """
    fraction: float = 0.5
    minTrades: int = 20
    defaultPercent: float = 0.1
    maxPercent: float = 0.25
    _trades: List[Trade] = field(default_factory=list)

    def updateTrades(self, trades: List[Trade]):
        """
        Update internal trade history for Kelly ratio calculation.

        켈리 비율 계산을 위해 내부 거래 기록을 업데이트합니다.

        Args:
            trades (list[Trade]): List of trades (only closed trades
                are retained). 거래 목록 (종료된 거래만 유지).
        """
        self._trades = [t for t in trades if t.isClosed]

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        if price <= 0:
            return 0

        trades = kwargs.get('trades', self._trades)

        closedTrades = [t for t in trades if t.isClosed]

        if len(closedTrades) < self.minTrades:
            amount = equity * self.defaultPercent
            return int(amount / price)

        wins = [t for t in closedTrades if t.pnl > 0]
        losses = [t for t in closedTrades if t.pnl <= 0]

        if not wins or not losses:
            amount = equity * self.defaultPercent
            return int(amount / price)

        winRate = len(wins) / len(closedTrades)
        avgWinPct = sum(t.pnlPercent for t in wins) / len(wins)
        avgLossPct = abs(sum(t.pnlPercent for t in losses) / len(losses))

        if avgLossPct == 0:
            amount = equity * self.defaultPercent
            return int(amount / price)

        b = avgWinPct / avgLossPct
        kelly = (b * winRate - (1 - winRate)) / b

        kelly = max(0, min(kelly, 1))

        targetPercent = kelly * self.fraction
        targetPercent = min(targetPercent, self.maxPercent)

        amount = equity * targetPercent
        return int(amount / price)

    def getKellyRatio(self, trades: List[Trade] = None) -> Optional[float]:
        """
        Calculate and return the current Kelly ratio.

        현재 켈리 비율을 계산하여 반환합니다.

        Args:
            trades (list[Trade], optional): Trade list to use. Falls
                back to internal trades if None. 사용할 거래 목록.

        Returns:
            float or None: Kelly ratio (0.0-1.0), or None if
                insufficient trade data. 켈리 비율 또는 None.
        """
        trades = trades or self._trades
        closedTrades = [t for t in trades if t.isClosed]

        if len(closedTrades) < self.minTrades:
            return None

        wins = [t for t in closedTrades if t.pnl > 0]
        losses = [t for t in closedTrades if t.pnl <= 0]

        if not wins or not losses:
            return None

        winRate = len(wins) / len(closedTrades)
        avgWinPct = sum(t.pnlPercent for t in wins) / len(wins)
        avgLossPct = abs(sum(t.pnlPercent for t in losses) / len(losses))

        if avgLossPct == 0:
            return None

        b = avgWinPct / avgLossPct
        kelly = (b * winRate - (1 - winRate)) / b

        return max(0, min(kelly, 1))


@dataclass
class VolatilitySizer(PositionSizer):
    """
    Volatility-based position sizer using ATR (Average True Range).

    Calculates position size based on ATR to normalize risk across
    instruments with different volatilities. The stop distance is
    defined as a multiple of ATR.

    변동성 기반 사이저 (ATR 사용) - ATR을 이용하여 다양한 변동성의
    종목에 대해 리스크를 정규화합니다.

    Attributes:
        riskPercent (float): Maximum risk per trade as fraction of
            equity. Default: 0.02. 거래당 리스크 비율.
        atrMultiple (float): ATR multiplier for stop-loss distance.
            Default: 2.0. ATR 배수 (손절 거리).

    Example:
        >>> sizer = VolatilitySizer(riskPercent=0.02, atrMultiple=2.0)
        >>> qty = sizer.calculate(equity=10_000_000, price=50000, atr=1500)
    """
    riskPercent: float = 0.02
    atrMultiple: float = 2.0

    def calculate(self, equity: float, price: float, **kwargs) -> int:
        if price <= 0:
            return 0

        atr = kwargs.get('atr')

        if atr is None or atr <= 0:
            return int(equity * 0.1 / price)

        stopDistance = atr * self.atrMultiple

        maxRiskAmount = equity * self.riskPercent

        quantity = int(maxRiskAmount / stopDistance)
        return max(1, quantity)
