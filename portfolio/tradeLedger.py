"""
Tradix Trade Ledger Module.

Maintains lot-based open and closed trades so portfolio-level PnL reconciles
with trade-level PnL even when a strategy scales in or out of positions.
"""

from typing import Dict, List
import uuid

from tradix.entities.trade import Trade
from tradix.events.fill import FillEvent


class TradeLedger:
    """
    Lot-based trade ledger for portfolio accounting.

    Each entry fill opens a new trade lot. Exit fills close those lots in FIFO
    order so scale-in/scale-out behaviour produces auditable round-trip trades.
    """

    def __init__(self):
        self._closedTrades: List[Trade] = []
        self._openTradeLots: Dict[str, List[Trade]] = {}

    @property
    def closedTrades(self) -> List[Trade]:
        """Return closed round-trip trades."""
        return self._closedTrades.copy()

    @property
    def openTrades(self) -> List[Trade]:
        """Return all currently open trade lots."""
        openTrades: List[Trade] = []
        for tradeLots in self._openTradeLots.values():
            openTrades.extend(tradeLots)
        return openTrades

    def getOpenTrades(self, symbol: str) -> List[Trade]:
        """Return open trade lots for a symbol."""
        return self._openTradeLots.get(symbol, []).copy()

    def recordEntry(self, fill: FillEvent) -> Trade:
        """Create a new open trade lot from an entry fill."""
        trade = Trade(
            id=str(uuid.uuid4())[:8],
            symbol=fill.symbol,
            side=fill.side,
            entryDate=fill.timestamp,
            entryPrice=fill.fillPrice,
            quantity=fill.fillQuantity,
            commission=fill.commission,
            slippage=fill.slippage,
            entryOrderId=fill.orderId,
        )
        self._openTradeLots.setdefault(fill.symbol, []).append(trade)
        return trade

    def recordExit(self, fill: FillEvent) -> List[Trade]:
        """Close open trade lots using FIFO accounting."""
        symbol = fill.symbol
        openTradeLots = self._openTradeLots.get(symbol, [])
        if not openTradeLots:
            return []

        remainingQuantity = fill.fillQuantity
        totalExitQuantity = fill.fillQuantity if fill.fillQuantity > 0 else 1.0
        closedTrades: List[Trade] = []

        while remainingQuantity > 0 and openTradeLots:
            openTrade = openTradeLots[0]
            closeQuantity = min(openTrade.quantity, remainingQuantity)
            exitFraction = closeQuantity / totalExitQuantity
            exitCommission = fill.commission * exitFraction
            exitSlippage = fill.slippage * exitFraction

            if closeQuantity >= openTrade.quantity:
                openTrade.close(
                    exitDate=fill.timestamp,
                    exitPrice=fill.fillPrice,
                    commission=exitCommission,
                    slippage=exitSlippage,
                    orderId=fill.orderId,
                )
                closedTrade = openTrade
                openTradeLots.pop(0)
            else:
                closedTrade = self._splitOpenTrade(
                    openTrade=openTrade,
                    closeQuantity=closeQuantity,
                    fill=fill,
                    exitCommission=exitCommission,
                    exitSlippage=exitSlippage,
                )

            self._closedTrades.append(closedTrade)
            closedTrades.append(closedTrade)
            remainingQuantity -= closeQuantity

        if not openTradeLots and symbol in self._openTradeLots:
            del self._openTradeLots[symbol]

        return closedTrades

    def reset(self):
        """Clear all ledger state."""
        self._closedTrades.clear()
        self._openTradeLots.clear()

    def _splitOpenTrade(
        self,
        openTrade: Trade,
        closeQuantity: float,
        fill: FillEvent,
        exitCommission: float,
        exitSlippage: float,
    ) -> Trade:
        """Split an open trade lot when only part of it is closed."""
        originalQuantity = openTrade.quantity
        entryFraction = closeQuantity / originalQuantity
        entryCommission = openTrade.commission * entryFraction
        entrySlippage = openTrade.slippage * entryFraction

        openTrade.quantity = originalQuantity - closeQuantity
        openTrade.commission -= entryCommission
        openTrade.slippage -= entrySlippage

        closedTrade = Trade(
            id=str(uuid.uuid4())[:8],
            symbol=openTrade.symbol,
            side=openTrade.side,
            entryDate=openTrade.entryDate,
            entryPrice=openTrade.entryPrice,
            quantity=closeQuantity,
            commission=entryCommission,
            slippage=entrySlippage,
            entryOrderId=openTrade.entryOrderId,
        )
        closedTrade.close(
            exitDate=fill.timestamp,
            exitPrice=fill.fillPrice,
            commission=exitCommission,
            slippage=exitSlippage,
            orderId=fill.orderId,
        )
        return closedTrade
