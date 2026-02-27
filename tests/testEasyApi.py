"""
Easy API Tests

Tests for the simplified backtest API.
"""

import pytest
import numpy as np

from tradix.easy import (
    QuickStrategy,
    goldenCross,
    rsiOversold,
    bollingerBreakout,
    macdCross,
    breakout,
    meanReversion,
    trendFollowing,
    sma,
    ema,
    rsi,
    macd,
    bollinger,
    atr,
    price,
    crossover,
    crossunder,
)
from tradix.easy.conditions import Indicator, CrossCondition, CompareCondition, macdHist, bollingerUpper, bollingerLower


class TestPresetStrategies:
    """Test preset strategy creation."""

    def testGoldenCross(self):
        """Test golden cross preset."""
        strategy = goldenCross()

        assert strategy is not None
        assert hasattr(strategy, "_buyConditions")
        assert hasattr(strategy, "_sellConditions")

    def testGoldenCrossParams(self):
        """Test golden cross with custom parameters."""
        strategy = goldenCross(fast=5, slow=20, stopLoss=5, takeProfit=15)

        assert strategy._stopLossPct == 0.05
        assert strategy._takeProfitPct == 0.15

    def testRsiOversold(self):
        """Test RSI oversold preset."""
        strategy = rsiOversold(period=14, oversold=30, overbought=70)

        assert strategy is not None

    def testBollingerBreakout(self):
        """Test Bollinger breakout preset."""
        strategy = bollingerBreakout(period=20, std=2.0)

        assert strategy is not None

    def testMacdCross(self):
        """Test MACD cross preset."""
        strategy = macdCross(fast=12, slow=26, signal=9)

        assert strategy is not None

    def testBreakout(self):
        """Test breakout preset."""
        strategy = breakout(period=20)

        assert strategy is not None

    def testMeanReversion(self):
        """Test mean reversion preset."""
        strategy = meanReversion(period=20, threshold=2.0)

        assert strategy is not None

    def testTrendFollowing(self):
        """Test trend following preset."""
        strategy = trendFollowing(fastPeriod=10, slowPeriod=30, adxThreshold=25)

        assert strategy is not None


class TestConditionBuilders:
    """Test condition builder functions."""

    def testSmaIndicator(self):
        """Test SMA indicator creation."""
        indicator = sma(20)

        assert isinstance(indicator, Indicator)
        assert indicator.type.value == "sma"
        assert indicator.params["period"] == 20

    def testEmaIndicator(self):
        """Test EMA indicator creation."""
        indicator = ema(12)

        assert isinstance(indicator, Indicator)
        assert indicator.type.value == "ema"

    def testRsiIndicator(self):
        """Test RSI indicator creation."""
        indicator = rsi(14)

        assert isinstance(indicator, Indicator)
        assert indicator.type.value == "rsi"

    def testMacdIndicator(self):
        """Test MACD indicator creation."""
        indicator = macd(12, 26, 9)

        assert isinstance(indicator, Indicator)
        assert indicator.type.value == "macd"

    def testBollingerIndicator(self):
        """Test Bollinger indicator creation."""
        upper, middle, lower = bollinger(20, 2.0)

        assert isinstance(upper, Indicator)
        assert isinstance(middle, Indicator)
        assert isinstance(lower, Indicator)
        assert upper.type.value == "bollingerUpper"
        assert middle.type.value == "bollingerMiddle"
        assert lower.type.value == "bollingerLower"

    def testAtrIndicator(self):
        """Test ATR indicator creation."""
        indicator = atr(14)

        assert isinstance(indicator, Indicator)
        assert indicator.type.value == "atr"

    def testPriceIndicator(self):
        """Test price indicator creation."""
        assert isinstance(price, Indicator)
        assert price.type.value == "close"

    def testCrossover(self):
        """Test crossover condition creation."""
        condition = crossover(sma(10), sma(30))

        assert isinstance(condition, CrossCondition)
        assert condition.crossType == "over"

    def testCrossunder(self):
        """Test crossunder condition creation."""
        condition = crossunder(sma(10), sma(30))

        assert isinstance(condition, CrossCondition)
        assert condition.crossType == "under"


class TestIndicatorComparison:
    """Test indicator comparison operators."""

    def testIndicatorGreaterThan(self):
        """Test indicator > comparison."""
        condition = sma(10) > sma(30)

        assert isinstance(condition, CompareCondition)
        assert condition.operator == ">"

    def testIndicatorLessThan(self):
        """Test indicator < comparison."""
        condition = rsi(14) < 30

        assert isinstance(condition, CompareCondition)
        assert condition.operator == "<"

    def testIndicatorGreaterEqual(self):
        """Test indicator >= comparison."""
        condition = sma(10) >= sma(30)

        assert isinstance(condition, CompareCondition)
        assert condition.operator == ">="

    def testIndicatorLessEqual(self):
        """Test indicator <= comparison."""
        condition = rsi(14) <= 30

        assert isinstance(condition, CompareCondition)
        assert condition.operator == "<="


class TestQuickStrategy:
    """Test QuickStrategy builder."""

    def testBasicCreation(self):
        """Test basic strategy creation."""
        strategy = QuickStrategy("TestStrategy")

        assert strategy.name == "TestStrategy"
        assert strategy._buyConditions == []
        assert strategy._sellConditions == []

    def testBuyWhen(self):
        """Test buyWhen method."""
        strategy = QuickStrategy("Test").buyWhen(crossover(sma(10), sma(30)))

        assert len(strategy._buyConditions) == 1

    def testSellWhen(self):
        """Test sellWhen method."""
        strategy = QuickStrategy("Test").sellWhen(rsi(14) > 70)

        assert len(strategy._sellConditions) == 1

    def testChaining(self):
        """Test method chaining."""
        strategy = (
            QuickStrategy("Test")
            .buyWhen(crossover(sma(10), sma(30)))
            .buyWhen(rsi(14) < 30)
            .sellWhen(crossunder(sma(10), sma(30)))
            .sellWhen(rsi(14) > 70)
            .stopLoss(5)
            .takeProfit(15)
        )

        assert len(strategy._buyConditions) == 2
        assert len(strategy._sellConditions) == 2
        assert strategy._stopLossPct == 0.05
        assert strategy._takeProfitPct == 0.15

    def testStopLoss(self):
        """Test stopLoss method."""
        strategy = QuickStrategy("Test").stopLoss(5)

        assert strategy._stopLossPct == 0.05

    def testTakeProfit(self):
        """Test takeProfit method."""
        strategy = QuickStrategy("Test").takeProfit(15)

        assert strategy._takeProfitPct == 0.15

    def testTrailingStop(self):
        """Test trailingStop method."""
        strategy = QuickStrategy("Test").trailingStop(10)

        assert strategy._trailingStopPct == 0.10


class TestComplexStrategies:
    """Test complex strategy combinations."""

    def testMultiConditionStrategy(self):
        """Test strategy with multiple conditions."""
        strategy = (
            QuickStrategy("MultiCondition")
            .buyWhen(crossover(ema(10), ema(30)))
            .buyWhen(rsi(14) < 35)
            .buyWhen(macdHist() > 0)
            .sellWhen(crossunder(ema(10), ema(30)))
            .sellWhen(rsi(14) > 65)
            .stopLoss(7)
            .takeProfit(20)
        )

        assert len(strategy._buyConditions) == 3
        assert len(strategy._sellConditions) == 2
        assert strategy._stopLossPct == 0.07
        assert strategy._takeProfitPct == 0.20

    def testBollingerRsiStrategy(self):
        """Test Bollinger + RSI combination."""
        strategy = (
            QuickStrategy("BollingerRsi")
            .buyWhen(price < bollingerLower(20, 2.0))
            .buyWhen(rsi(14) < 25)
            .sellWhen(price > bollingerUpper(20, 2.0))
            .sellWhen(rsi(14) > 75)
        )

        assert len(strategy._buyConditions) == 2
        assert len(strategy._sellConditions) == 2


class TestKoreanApi:
    """Test Korean language API."""

    def testKoreanPresets(self):
        """Test Korean preset functions."""
        from tradix.easy.korean import (
            골든크로스,
            RSI과매도,
            볼린저돌파,
            MACD크로스,
            돌파전략,
            평균회귀,
            추세추종,
            전략,
        )

        assert 골든크로스() is not None
        assert RSI과매도() is not None
        assert 볼린저돌파() is not None
        assert MACD크로스() is not None
        assert 돌파전략() is not None
        assert 평균회귀() is not None
        assert 추세추종() is not None

    def testKoreanStrategyBuilder(self):
        """Test Korean strategy builder."""
        from tradix.easy.korean import 전략
        from tradix.easy.conditions import sma, crossover

        내전략 = 전략("테스트전략").buyWhen(crossover(sma(10), sma(30))).stopLoss(5)

        assert 내전략.name == "테스트전략"
        assert len(내전략._buyConditions) == 1
        assert 내전략._stopLossPct == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
