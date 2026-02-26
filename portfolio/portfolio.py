"""
Tradex Portfolio Management Module.

Manages portfolio state including cash, positions, orders, trades, and equity
tracking during backtesting. Handles buy/sell fill processing, position
updates, and provides comprehensive portfolio statistics.

백테스트 중 현금, 포지션, 주문, 거래, 자산 추적을 포함한 포트폴리오 상태를
관리하는 모듈입니다. 매수/매도 체결 처리, 포지션 업데이트, 종합 포트폴리오
통계를 제공합니다.

Features:
    - Cash and position management
    - Order placement and cancellation
    - Fill event processing with commission and slippage
    - Trade lifecycle tracking (open to close)
    - Real-time equity calculation and history recording
    - Trade statistics computation

Usage:
    from tradex.portfolio import Portfolio

    portfolio = Portfolio(initialCash=10_000_000)
    portfolio.processFill(fill_event)
    print(f"Equity: {portfolio.equity:,.0f}")
    print(f"Return: {portfolio.totalReturn:+.2f}%")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from tradex.entities.order import Order, OrderSide, OrderStatus
from tradex.entities.position import Position
from tradex.entities.trade import Trade
from tradex.events.fill import FillEvent


class Portfolio:
    """
    Core portfolio management class for backtest execution.

    Tracks cash balance, open positions, pending/filled orders, and trade
    history. Processes fill events to update positions and records equity
    snapshots for performance analysis.

    백테스트 실행을 위한 핵심 포트폴리오 관리 클래스입니다. 현금 잔액, 보유 포지션,
    대기/체결 주문, 거래 이력을 추적하며, 체결 이벤트를 처리하여 포지션을 업데이트하고
    성과 분석을 위한 자산 스냅샷을 기록합니다.

    Attributes:
        initialCash (float): Starting capital amount (초기 자본금).

    Example:
        >>> portfolio = Portfolio(initialCash=10_000_000)
        >>> portfolio.processFill(fill_event)
        >>> print(f"Equity: {portfolio.equity:,.0f}")
        >>> print(f"Positions: {portfolio.positionCount}")
    """

    def __init__(self, initialCash: float = 10_000_000):
        """
        Initialize the Portfolio with starting capital.

        포트폴리오를 초기 자본금으로 초기화합니다.

        Args:
            initialCash: Starting cash amount, default 10,000,000 KRW
                (초기 현금, 기본 1천만원).
        """
        self.initialCash = initialCash
        self._cash = initialCash
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._openTrades: Dict[str, Trade] = {}
        self._pendingOrders: List[Order] = []
        self._filledOrders: List[Order] = []
        self._equityHistory: List[tuple] = []

    @property
    def cash(self) -> float:
        """Return the available cash balance (사용 가능 현금)."""
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        """Return a copy of current positions keyed by symbol (보유 포지션)."""
        return self._positions.copy()

    @property
    def positionCount(self) -> int:
        """Return the number of open positions (보유 종목 수)."""
        return len(self._positions)

    @property
    def positionValue(self) -> float:
        """Return total market value of all positions (포지션 총 가치)."""
        return sum(p.marketValue for p in self._positions.values())

    @property
    def equity(self) -> float:
        """Return total equity as cash plus position value (총 자산 = 현금 + 포지션)."""
        return self._cash + self.positionValue

    @property
    def totalExposure(self) -> float:
        """Return total exposure as sum of absolute position values (총 노출)."""
        return sum(abs(p.marketValue) for p in self._positions.values())

    @property
    def unrealizedPnl(self) -> float:
        """Return total unrealized PnL across all open positions (미실현 손익)."""
        return sum(p.unrealizedPnl for p in self._positions.values())

    @property
    def realizedPnl(self) -> float:
        """Return total realized PnL from closed trades (실현 손익)."""
        return sum(t.pnl for t in self._trades if t.isClosed)

    @property
    def totalPnl(self) -> float:
        """Return total PnL as equity minus initial cash (총 손익)."""
        return self.equity - self.initialCash

    @property
    def totalReturn(self) -> float:
        """Return total return as percentage (총 수익률, %)."""
        if self.initialCash == 0:
            return 0.0
        return (self.equity / self.initialCash - 1) * 100

    @property
    def trades(self) -> List[Trade]:
        """Return a list of closed (completed) trades (완료된 거래 목록)."""
        return [t for t in self._trades if t.isClosed]

    @property
    def openTrades(self) -> List[Trade]:
        """Return a list of currently open trades (진행 중인 거래 목록)."""
        return list(self._openTrades.values())

    @property
    def pendingOrders(self) -> List[Order]:
        """Return a copy of pending (unfilled) orders (대기 중인 주문)."""
        return self._pendingOrders.copy()

    def getPosition(self, symbol: str) -> Optional[Position]:
        """
        Retrieve the position for a specific symbol.

        특정 종목의 포지션을 조회합니다.

        Args:
            symbol: Ticker symbol to look up.

        Returns:
            Optional[Position]: Position object if found, None otherwise.
        """
        return self._positions.get(symbol)

    def hasPosition(self, symbol: str) -> bool:
        """
        Check whether a position exists for a symbol.

        특정 종목의 포지션 보유 여부를 확인합니다.

        Args:
            symbol: Ticker symbol to check.

        Returns:
            bool: True if a position exists for the symbol.
        """
        return symbol in self._positions

    def getPositionQuantity(self, symbol: str) -> float:
        """
        Return the quantity held for a symbol, or 0.0 if no position.

        특정 종목의 보유 수량을 반환합니다. 포지션이 없으면 0.0을 반환합니다.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            float: Position quantity.
        """
        pos = self._positions.get(symbol)
        return pos.quantity if pos else 0.0

    def getPositionValue(self, symbol: str) -> float:
        """
        Return the market value of a position, or 0.0 if no position.

        특정 종목 포지션의 시장 가치를 반환합니다.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            float: Position market value.
        """
        pos = self._positions.get(symbol)
        return pos.marketValue if pos else 0.0

    def updatePrice(self, symbol: str, price: float):
        """
        Update the current market price for a position.

        포지션의 현재 시장가를 업데이트합니다.

        Args:
            symbol: Ticker symbol to update.
            price: New market price.
        """
        if symbol in self._positions:
            self._positions[symbol].updatePrice(price)

    def updatePrices(self, prices: Dict[str, float]):
        """
        Update market prices for multiple symbols at once.

        여러 종목의 현재가를 일괄 업데이트합니다.

        Args:
            prices: Dict mapping symbol to new market price.
        """
        for symbol, price in prices.items():
            self.updatePrice(symbol, price)

    def addOrder(self, order: Order):
        """
        Add an order to the pending orders queue.

        대기 주문 큐에 주문을 추가합니다.

        Args:
            order: Order object to enqueue.
        """
        self._pendingOrders.append(order)

    def cancelOrder(self, orderId: str) -> bool:
        """
        Cancel a pending order by its ID.

        주문 ID로 대기 주문을 취소합니다.

        Args:
            orderId: Unique identifier of the order to cancel.

        Returns:
            bool: True if the order was found and cancelled, False otherwise.
        """
        for i, order in enumerate(self._pendingOrders):
            if order.id == orderId:
                order.cancel()
                self._pendingOrders.pop(i)
                return True
        return False

    def cancelAllOrders(self, symbol: str = None):
        """
        Cancel all pending orders, optionally filtered by symbol.

        대기 주문을 전체 취소합니다. symbol 지정 시 해당 종목만 취소합니다.

        Args:
            symbol: If provided, cancel only orders for this symbol;
                if None, cancel all pending orders.
        """
        if symbol:
            self._pendingOrders = [o for o in self._pendingOrders if o.symbol != symbol]
        else:
            for order in self._pendingOrders:
                order.cancel()
            self._pendingOrders.clear()

    def processFill(self, fill: FillEvent):
        """
        Process a fill event, updating cash, positions, and trades.

        체결 이벤트를 처리하여 현금, 포지션, 거래를 업데이트합니다.

        Args:
            fill: FillEvent containing execution details including price,
                quantity, commission, and slippage (체결 이벤트).
        """
        order = fill.order
        symbol = fill.symbol

        if order.side == OrderSide.BUY:
            self._processBuyFill(fill)
        else:
            self._processSellFill(fill)

        order.fill(
            price=fill.fillPrice,
            quantity=fill.fillQuantity,
            commission=fill.commission,
            slippage=fill.slippage
        )

        if order.isFilled:
            if order in self._pendingOrders:
                self._pendingOrders.remove(order)
            self._filledOrders.append(order)

    def _processBuyFill(self, fill: FillEvent):
        """
        Process a buy fill: deduct cash, create/update position, open trade.

        매수 체결 처리: 현금 차감, 포지션 생성/업데이트, 거래 개시.

        Args:
            fill: FillEvent for a buy order.
        """
        symbol = fill.symbol
        cost = fill.fillPrice * fill.fillQuantity + fill.commission + fill.slippage

        self._cash -= cost

        if symbol in self._positions:
            self._positions[symbol].addQuantity(fill.fillQuantity, fill.fillPrice)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=fill.fillQuantity,
                avgPrice=fill.fillPrice,
                currentPrice=fill.fillPrice,
                openedAt=fill.timestamp
            )

            trade = Trade(
                id=str(uuid.uuid4())[:8],
                symbol=symbol,
                side=OrderSide.BUY,
                entryDate=fill.timestamp,
                entryPrice=fill.fillPrice,
                quantity=fill.fillQuantity,
                commission=fill.commission,
                slippage=fill.slippage,
                entryOrderId=fill.orderId
            )
            self._trades.append(trade)
            self._openTrades[symbol] = trade

    def _processSellFill(self, fill: FillEvent):
        """
        Process a sell fill: add proceeds, reduce/close position, close trade.

        매도 체결 처리: 현금 추가, 포지션 축소/청산, 거래 종료.

        Args:
            fill: FillEvent for a sell order.
        """
        symbol = fill.symbol
        proceeds = fill.fillPrice * fill.fillQuantity - fill.commission - fill.slippage

        self._cash += proceeds

        if symbol in self._positions:
            position = self._positions[symbol]
            position.reduceQuantity(fill.fillQuantity, fill.fillPrice)

            if position.quantity <= 0:
                del self._positions[symbol]

                if symbol in self._openTrades:
                    trade = self._openTrades[symbol]
                    trade.close(
                        exitDate=fill.timestamp,
                        exitPrice=fill.fillPrice,
                        commission=fill.commission,
                        slippage=fill.slippage,
                        orderId=fill.orderId
                    )
                    del self._openTrades[symbol]

    def recordEquity(self, timestamp: datetime):
        """
        Record current equity value with a timestamp for history tracking.

        현재 자산 가치를 타임스탬프와 함께 기록합니다.

        Args:
            timestamp: Datetime of the equity snapshot.
        """
        self._equityHistory.append((timestamp, self.equity))

    def getEquityHistory(self) -> List[tuple]:
        """
        Return a copy of the equity history as (timestamp, equity) tuples.

        자산 기록을 (타임스탬프, 자산가치) 튜플 리스트로 반환합니다.

        Returns:
            List[tuple]: List of (datetime, float) equity snapshots.
        """
        return self._equityHistory.copy()

    def getTradeStats(self) -> dict:
        """
        Compute summary trade statistics for all closed trades.

        모든 완료된 거래에 대한 요약 통계를 계산합니다.

        Returns:
            dict: Statistics including totalTrades, wins, losses, winRate,
                avgPnl, avgWin, avgLoss, and totalPnl.
        """
        closedTrades = self.trades
        if not closedTrades:
            return {
                'totalTrades': 0,
                'wins': 0,
                'losses': 0,
                'winRate': 0.0,
                'avgPnl': 0.0,
                'avgWin': 0.0,
                'avgLoss': 0.0,
                'totalPnl': 0.0,
            }

        wins = [t for t in closedTrades if t.pnl > 0]
        losses = [t for t in closedTrades if t.pnl <= 0]

        return {
            'totalTrades': len(closedTrades),
            'wins': len(wins),
            'losses': len(losses),
            'winRate': len(wins) / len(closedTrades) * 100 if closedTrades else 0,
            'avgPnl': sum(t.pnl for t in closedTrades) / len(closedTrades),
            'avgWin': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avgLoss': sum(t.pnl for t in losses) / len(losses) if losses else 0,
            'totalPnl': sum(t.pnl for t in closedTrades),
        }

    def reset(self):
        """
        Reset the portfolio to its initial state.

        포트폴리오를 초기 상태로 재설정합니다. 현금, 포지션, 거래, 주문,
        자산 기록이 모두 초기화됩니다.
        """
        self._cash = self.initialCash
        self._positions.clear()
        self._trades.clear()
        self._openTrades.clear()
        self._pendingOrders.clear()
        self._filledOrders.clear()
        self._equityHistory.clear()

    def __repr__(self) -> str:
        return (
            f"Portfolio(cash={self._cash:,.0f}, "
            f"positions={self.positionCount}, "
            f"equity={self.equity:,.0f}, "
            f"return={self.totalReturn:+.2f}%)"
        )
