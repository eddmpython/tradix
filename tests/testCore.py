"""
Core Engine Tests

Tests for the core backtest engine and strategies.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tradex.entities import Order, OrderSide, OrderType, Position, Trade, Bar
from tradex.strategy import Strategy, Indicators


class TestBar:
    """Test Bar dataclass."""

    def testBarCreation(self):
        """Test Bar creation."""
        bar = Bar(
            symbol="005930",
            datetime=datetime(2024, 1, 15),
            open=70000,
            high=71000,
            low=69500,
            close=70500,
            volume=1000000,
        )

        assert bar.symbol == "005930"
        assert bar.close == 70500
        assert bar.high == 71000
        assert bar.low == 69500

    def testBarOhlc(self):
        """Test OHLC values."""
        from datetime import datetime as dt
        bar = Bar(
            symbol="TEST",
            datetime=dt.now(),
            open=100,
            high=110,
            low=95,
            close=105,
            volume=50000,
        )

        assert bar.high >= bar.low
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close


class TestOrder:
    """Test Order class."""

    def testMarketOrder(self):
        """Test market order creation."""
        order = Order(
            symbol="005930",
            side=OrderSide.BUY,
            orderType=OrderType.MARKET,
            quantity=100,
        )

        assert order.symbol == "005930"
        assert order.side == OrderSide.BUY
        assert order.orderType == OrderType.MARKET
        assert order.quantity == 100

    def testLimitOrder(self):
        """Test limit order creation."""
        order = Order(
            symbol="005930",
            side=OrderSide.SELL,
            orderType=OrderType.LIMIT,
            quantity=50,
            price=75000,
        )

        assert order.orderType == OrderType.LIMIT
        assert order.price == 75000


class TestPosition:
    """Test Position class."""

    def testPositionCreation(self):
        """Test position creation."""
        position = Position(
            symbol="005930",
            quantity=100,
            avgPrice=70000,
        )

        assert position.symbol == "005930"
        assert position.quantity == 100
        assert position.avgPrice == 70000

    def testPositionValue(self):
        """Test position value calculation."""
        position = Position(
            symbol="005930",
            quantity=100,
            avgPrice=70000,
        )

        currentPrice = 75000
        value = position.quantity * currentPrice
        pnl = (currentPrice - position.avgPrice) * position.quantity

        assert value == 7500000
        assert pnl == 500000


class TestTrade:
    """Test Trade class."""

    def testTradeCreation(self):
        """Test trade creation."""
        trade = Trade(
            id="t001",
            symbol="005930",
            side=OrderSide.BUY,
            entryDate=datetime.now(),
            entryPrice=70000,
            quantity=100,
            commission=1050,
        )

        assert trade.symbol == "005930"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 100
        assert trade.entryPrice == 70000
        assert trade.commission == 1050


class TestIndicators:
    """Test Indicators class."""

    @pytest.fixture
    def priceData(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0.001, 0.02, n)
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        return {"high": high, "low": low, "close": close}

    def testSmaCaching(self, priceData):
        """Test SMA calculation with caching."""
        indicators = Indicators()

        n = len(priceData["close"])
        df = pd.DataFrame({
            "close": priceData["close"],
            "high": priceData["high"],
            "low": priceData["low"],
            "open": priceData["close"],
            "volume": [100000] * n,
        })
        indicators.setFullData(df)
        indicators.setIndex(n)

        sma1 = indicators.sma(20)
        sma2 = indicators.sma(20)

        assert sma1 == sma2

    def testEmaCaching(self, priceData):
        """Test EMA calculation with caching."""
        indicators = Indicators()

        n = len(priceData["close"])
        df = pd.DataFrame({
            "close": priceData["close"],
            "high": priceData["high"],
            "low": priceData["low"],
            "open": priceData["close"],
            "volume": [100000] * n,
        })
        indicators.setFullData(df)
        indicators.setIndex(n)

        ema1 = indicators.ema(12)
        ema2 = indicators.ema(12)

        assert ema1 == ema2

    def testRsiRange(self, priceData):
        """Test RSI is within 0-100 range."""
        indicators = Indicators()

        n = len(priceData["close"])
        df = pd.DataFrame({
            "close": priceData["close"],
            "high": priceData["high"],
            "low": priceData["low"],
            "open": priceData["close"],
            "volume": [100000] * n,
        })
        indicators.setFullData(df)
        indicators.setIndex(n)

        rsi = indicators.rsi(14)
        if rsi is not None:
            assert 0 <= rsi <= 100


class TestStrategy:
    """Test Strategy base class."""

    def testCustomStrategy(self):
        """Test custom strategy creation."""

        class TestStrategy(Strategy):
            def initialize(self):
                self.fastPeriod = 10
                self.slowPeriod = 30

            def onBar(self, bar: Bar):
                pass

        strategy = TestStrategy()
        strategy.initialize()

        assert strategy.fastPeriod == 10
        assert strategy.slowPeriod == 30

    def testStrategyWithIndicators(self):
        """Test strategy using indicators."""

        class SmaStrategy(Strategy):
            def initialize(self):
                self.signals = []

            def onBar(self, bar: Bar):
                fastSma = self.sma(10)
                slowSma = self.sma(30)

                if fastSma is not None and slowSma is not None:
                    if fastSma > slowSma:
                        self.signals.append("BUY")
                    else:
                        self.signals.append("SELL")

        strategy = SmaStrategy()
        strategy.initialize()

        assert isinstance(strategy, Strategy)
        assert strategy.signals == []


class TestOrderSide:
    """Test OrderSide enum."""

    def testOrderSideValues(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def testOrderSideComparison(self):
        """Test OrderSide comparison."""
        assert OrderSide.BUY == OrderSide.BUY
        assert OrderSide.BUY != OrderSide.SELL


class TestOrderType:
    """Test OrderType enum."""

    def testOrderTypeValues(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
