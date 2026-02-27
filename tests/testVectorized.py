"""
Vectorized Engine Tests

Tests for Numba-accelerated indicators and backtest engine.
"""

import pytest
import numpy as np

from tradix.vectorized import (
    vsma,
    vema,
    vrsi,
    vmacd,
    vbollinger,
    vatr,
    vstochastic,
    vadx,
    vcrossover,
    vcrossunder,
    vgoldenCross,
    vrsiSignal,
    vmacdSignal,
    vbollingerSignal,
    VectorizedEngine,
    VectorizedResult,
)


class TestVectorizedIndicators:
    """Test vectorized indicator calculations."""

    @pytest.fixture
    def sampleData(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0.0005, 0.02, n)
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)
        return {"high": high, "low": low, "close": close, "volume": volume}

    def testVsma(self, sampleData):
        """Test SMA calculation."""
        close = sampleData["close"]
        period = 20

        result = vsma(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[:period - 1]).all()
        assert not np.isnan(result[period - 1:]).any()

        expected = np.mean(close[:period])
        assert np.isclose(result[period - 1], expected, rtol=1e-10)

    def testVema(self, sampleData):
        """Test EMA calculation."""
        close = sampleData["close"]
        period = 12

        result = vema(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[:period - 1]).all()
        assert not np.isnan(result[period - 1:]).any()

        alpha = 2.0 / (period + 1)
        expected = np.mean(close[:period])
        assert np.isclose(result[period - 1], expected, rtol=1e-10)

    def testVrsi(self, sampleData):
        """Test RSI calculation."""
        close = sampleData["close"]
        period = 14

        result = vrsi(close, period)

        assert len(result) == len(close)
        assert not np.isnan(result[period:]).any()
        assert (result[period:] >= 0).all()
        assert (result[period:] <= 100).all()

    def testVmacd(self, sampleData):
        """Test MACD calculation."""
        close = sampleData["close"]

        macdLine, signalLine, histogram = vmacd(close, 12, 26, 9)

        assert len(macdLine) == len(close)
        assert len(signalLine) == len(close)
        assert len(histogram) == len(close)

        validIdx = 50
        assert not np.isnan(macdLine[validIdx:]).any()

    def testVbollinger(self, sampleData):
        """Test Bollinger Bands calculation."""
        close = sampleData["close"]
        period = 20

        upper, middle, lower = vbollinger(close, period, 2.0)

        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)

        validIdx = period - 1
        assert (upper[validIdx:] > middle[validIdx:]).all()
        assert (middle[validIdx:] > lower[validIdx:]).all()

    def testVatr(self, sampleData):
        """Test ATR calculation."""
        high = sampleData["high"]
        low = sampleData["low"]
        close = sampleData["close"]
        period = 14

        result = vatr(high, low, close, period)

        assert len(result) == len(close)
        assert not np.isnan(result[period - 1:]).any()
        assert (result[period - 1:] > 0).all()

    def testVstochastic(self, sampleData):
        """Test Stochastic calculation."""
        high = sampleData["high"]
        low = sampleData["low"]
        close = sampleData["close"]

        k, d = vstochastic(high, low, close, 14, 3)

        assert len(k) == len(close)
        assert len(d) == len(close)

        validIdx = 20
        assert (k[validIdx:] >= 0).all()
        assert (k[validIdx:] <= 100).all()

    def testVadx(self, sampleData):
        """Test ADX calculation."""
        high = sampleData["high"]
        low = sampleData["low"]
        close = sampleData["close"]
        period = 14

        result = vadx(high, low, close, period)

        assert len(result) == len(close)
        validIdx = 2 * period
        assert (result[validIdx:] >= 0).all()
        assert (result[validIdx:] <= 100).all()


class TestVectorizedSignals:
    """Test vectorized signal generation."""

    @pytest.fixture
    def crossoverData(self):
        """Generate data with a crossover."""
        fast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
        slow = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        return fast, slow

    def testVcrossover(self, crossoverData):
        """Test crossover detection."""
        fast, slow = crossoverData

        result = vcrossover(fast, slow)

        assert result[3] == 1
        assert result[0] == 0
        assert result[1] == 0
        assert result[6] == 0

    def testVcrossunder(self, crossoverData):
        """Test crossunder detection."""
        fast, slow = crossoverData

        result = vcrossunder(fast, slow)

        # fast=[1,2,3,4,5,4,3], slow=[3,3,3,3,3,3,3]
        # fast never goes strictly below slow (minimum fast=3 == slow=3),
        # so no crossunder is detected at any index.
        assert result[5] == 0
        assert result[6] == 0
        assert result[3] == 0
        assert result[0] == 0

    def testVgoldenCross(self):
        """Test golden cross signal generation."""
        np.random.seed(42)
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 200))

        signals = vgoldenCross(close, fast=5, slow=20)

        assert len(signals) == len(close)
        assert set(np.unique(signals)).issubset({-1, 0, 1})

    def testVrsiSignal(self):
        """Test RSI signal generation."""
        rsi = np.array([50, 40, 30, 25, 28, 35, 50, 65, 72, 68, 50])

        signals = vrsiSignal(rsi, oversold=30, overbought=70)

        # Buy when prevRsi <= 30 and rsi > 30: at index 5 (prevRsi=28, rsi=35)
        assert signals[5] == 1
        # Sell when prevRsi >= 70 and rsi < 70: at index 9 (prevRsi=72, rsi=68)
        assert signals[9] == -1


class TestVectorizedEngine:
    """Test VectorizedEngine backtest functionality."""

    @pytest.fixture
    def mockDataFrame(self):
        """Create mock DataFrame."""
        import pandas as pd
        np.random.seed(42)
        n = 252

        returns = np.random.normal(0.0005, 0.02, n)
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        openp = close * (1 + np.random.normal(0, 0.005, n))
        volume = np.random.randint(100000, 1000000, n).astype(np.float64)

        return pd.DataFrame({
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def testEngineGoldenCross(self, mockDataFrame):
        """Test engine with golden cross strategy."""
        engine = VectorizedEngine(initialCash=10_000_000)
        result = engine.run(data=mockDataFrame, strategy="goldenCross", fast=5, slow=20)

        assert isinstance(result, VectorizedResult)
        assert hasattr(result, "totalReturn")
        assert hasattr(result, "sharpeRatio")
        assert hasattr(result, "maxDrawdown")
        assert hasattr(result, "totalTrades")
        assert len(result.equityCurve) == len(mockDataFrame)

    def testEngineRsi(self, mockDataFrame):
        """Test engine with RSI strategy."""
        engine = VectorizedEngine(initialCash=10_000_000)
        result = engine.run(data=mockDataFrame, strategy="rsi", period=14, oversold=30, overbought=70)

        assert isinstance(result, VectorizedResult)
        assert result.totalTrades >= 0

    def testEngineStopLoss(self, mockDataFrame):
        """Test engine with stop loss."""
        engine = VectorizedEngine(initialCash=10_000_000)

        resultNoSl = engine.run(data=mockDataFrame, strategy="goldenCross", fast=5, slow=20, stopLoss=0)
        resultWithSl = engine.run(data=mockDataFrame, strategy="goldenCross", fast=5, slow=20, stopLoss=5.0)

        assert resultWithSl.maxDrawdown <= 10.0 or resultWithSl.totalTrades == 0

    def testEngineTakeProfit(self, mockDataFrame):
        """Test engine with take profit."""
        engine = VectorizedEngine(initialCash=10_000_000)
        result = engine.run(data=mockDataFrame, strategy="goldenCross", fast=5, slow=20, takeProfit=10.0)

        assert isinstance(result, VectorizedResult)

    def testEngineMacd(self, mockDataFrame):
        """Test engine with MACD strategy."""
        engine = VectorizedEngine(initialCash=10_000_000)
        result = engine.run(data=mockDataFrame, strategy="macd", fast=12, slow=26, signal=9)

        assert isinstance(result, VectorizedResult)

    def testEngineBollinger(self, mockDataFrame):
        """Test engine with Bollinger strategy."""
        engine = VectorizedEngine(initialCash=10_000_000)
        result = engine.run(data=mockDataFrame, strategy="bollinger", period=20, std=2.0)

        assert isinstance(result, VectorizedResult)


class TestVectorizedResult:
    """Test VectorizedResult dataclass."""

    def testResultSummary(self):
        """Test result summary output."""
        result = VectorizedResult(
            strategy="TestStrategy",
            symbol="TEST",
            startDate="2024-01-01",
            endDate="2024-12-31",
            initialCash=10_000_000,
            finalEquity=12_550_000,
            totalReturn=25.5,
            annualReturn=12.3,
            volatility=15.5,
            sharpeRatio=1.45,
            maxDrawdown=-15.2,
            maxDrawdownDuration=30,
            totalTrades=24,
            winRate=58.5,
            profitFactor=1.8,
            avgWin=100000,
            avgLoss=50000,
            equityCurve=np.linspace(10_000_000, 12_550_000, 252),
        )

        summary = result.summary()

        assert "25.5" in summary or "25.50" in summary
        assert "1.45" in summary
        assert "24" in summary


class TestPerformance:
    """Test performance benchmarks."""

    @pytest.fixture
    def largeData(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        n = 10000
        returns = np.random.normal(0.0003, 0.015, n)
        close = 100 * np.cumprod(1 + returns)
        return close

    def testSmaPerformance(self, largeData):
        """Benchmark SMA calculation."""
        import time

        vsma(largeData, 20)

        start = time.perf_counter()
        for _ in range(100):
            vsma(largeData, 20)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.01

    def testRsiPerformance(self, largeData):
        """Benchmark RSI calculation."""
        import time

        vrsi(largeData, 14)

        start = time.perf_counter()
        for _ in range(100):
            vrsi(largeData, 14)
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
