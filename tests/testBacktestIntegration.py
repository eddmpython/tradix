"""
Backtest integration and regression tests.

Focuses on execution-path correctness instead of surface-level object creation.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from tradix.broker import BrokerSimulator
from tradix.datafeed.feed import DataFeed
from tradix.easy import api as easy_api
from tradix.easy.quick import QuickStrategy
from tradix.engine import BacktestEngine, SimpleBroker
from tradix.strategy.base import Strategy
from tradix.walkforward.analyzer import FoldResult, WalkForwardResult


class DummyFeed(DataFeed):
    """Small in-memory feed for deterministic execution tests."""

    def __init__(self, symbol: str, rows: list[tuple[str, float, float, float, float]]):
        super().__init__(symbol, rows[0][0], rows[-1][0])
        self._rows = rows

    def load(self) -> pd.DataFrame:
        self._data = pd.DataFrame(
            {
                "open": [row[1] for row in self._rows],
                "high": [row[2] for row in self._rows],
                "low": [row[3] for row in self._rows],
                "close": [row[4] for row in self._rows],
                "volume": [1.0] * len(self._rows),
            },
            index=pd.to_datetime([row[0] for row in self._rows]),
        )
        self._buildNumpyArrays()
        self._loaded = True
        return self._data


def makeDummyFeedFactory(rows: list[tuple[str, float, float, float, float]]):
    """Return a FinanceDataReaderFeed-compatible factory backed by DummyFeed."""

    def factory(symbol: str, startDate: str, endDate: str):
        return DummyFeed(symbol, rows)

    return factory


class BuyThenSell(Strategy):
    """Buy on the first bar, exit on the second bar."""

    def onBar(self, bar):
        if bar.datetime.day == 1:
            self.buy(bar.symbol, quantity=1)
        elif bar.datetime.day == 2 and self.hasPosition(bar.symbol):
            self.sell(bar.symbol, quantity=1)


class ScaleInThenExit(Strategy):
    """Scale into a position and exit all lots together."""

    def onBar(self, bar):
        if bar.datetime.day == 1:
            self.buy(bar.symbol, quantity=1)
        elif bar.datetime.day == 2:
            self.buy(bar.symbol, quantity=1)
        elif bar.datetime.day == 3 and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)


class PartialExit(Strategy):
    """Exit one lot at a time to verify partial close accounting."""

    def onBar(self, bar):
        if bar.datetime.day == 1:
            self.buy(bar.symbol, quantity=2)
        elif bar.datetime.day == 2 and self.hasPosition(bar.symbol):
            self.sell(bar.symbol, quantity=1)
        elif bar.datetime.day == 3 and self.hasPosition(bar.symbol):
            self.sell(bar.symbol, quantity=1)


def testBrokerSimulatorSupportsNextBarOpenExecution():
    feed = DummyFeed(
        "TEST",
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 110, 115, 109, 112),
            ("2024-01-03", 120, 121, 118, 120),
        ],
    )

    engine = BacktestEngine(
        data=feed,
        strategy=BuyThenSell(),
        initialCash=1_000,
        broker=BrokerSimulator.ideal(),
        fillOnNextBar=True,
    )

    result = engine.run()

    assert result.totalTrades == 1
    assert result.finalEquity == 1_010
    assert result.trades[0].entryPrice == 110
    assert result.trades[0].exitPrice == 120


def testQuickStrategyTakeProfitUsesActualFillPrice():
    feed = DummyFeed(
        "TEST",
        [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 120, 121, 118, 119),
            ("2024-01-03", 118, 119, 117, 118),
        ],
    )

    strategy = (
        QuickStrategy("GapTakeProfit")
        .buyWhen(lambda s, bar: bar.datetime.day == 1)
        .takeProfit(10)
    )

    engine = BacktestEngine(
        data=feed,
        strategy=strategy,
        initialCash=1_000,
        broker=SimpleBroker(commissionRate=0.0, taxRate=0.0, slippageRate=0.0),
        fillOnNextBar=True,
    )

    result = engine.run()

    assert result.totalTrades == 0
    assert result.finalEquity < result.initialCash


def testOptimizeReturnsActualMaxDrawdownMetric(monkeypatch):
    class DummyStrategy:
        def __init__(self, depth):
            self.depth = depth

    def strategyFactory(depth):
        return DummyStrategy(depth)

    def fakeBacktest(symbol, strategy, period, verbose=False):
        return SimpleNamespace(metrics={"maxDrawdown": -strategy.depth})

    monkeypatch.setattr(easy_api, "backtest", fakeBacktest)

    best = easy_api.optimize(
        "TEST",
        strategyFactory,
        metric="maxDrawdown",
        depth=[3, 1, 2],
    )

    assert best["best"]["params"] == {"depth": 1}
    assert best["best"]["metric"] == -1


def testWalkForwardGetBestParamsReturnsMostCommonFold():
    folds = [
        FoldResult(1, ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-09-30"), {"fast": 10, "slow": 30}, 1.0, 0.8),
        FoldResult(2, ("2024-04-01", "2024-09-30"), ("2024-10-01", "2024-12-31"), {"fast": 10, "slow": 30}, 1.1, 0.7),
        FoldResult(3, ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-03-31"), {"fast": 20, "slow": 40}, 0.9, 0.4),
    ]

    result = WalkForwardResult(folds=folds, metric="sharpeRatio", elapsedTime=0.0)

    assert result.getBestParams() == {"fast": 10, "slow": 30}


def testScaleInTradesReconcileWithPortfolioPnl():
    feed = DummyFeed(
        "TEST",
        [
            ("2024-01-01", 100, 100, 100, 100),
            ("2024-01-02", 200, 200, 200, 200),
            ("2024-01-03", 150, 150, 150, 150),
        ],
    )

    engine = BacktestEngine(
        data=feed,
        strategy=ScaleInThenExit(),
        initialCash=1_000,
        broker=SimpleBroker(commissionRate=0.0, taxRate=0.0, slippageRate=0.0),
        fillOnNextBar=False,
    )

    result = engine.run()

    assert result.finalEquity == 1_000
    assert sum(trade.pnl for trade in result.trades) == 0
    assert [trade.entryPrice for trade in result.trades] == [100, 200]
    assert [trade.exitPrice for trade in result.trades] == [150, 150]
    assert [trade.pnl for trade in result.trades] == [50, -50]


def testPartialExitCreatesSeparateClosedLots():
    feed = DummyFeed(
        "TEST",
        [
            ("2024-01-01", 100, 100, 100, 100),
            ("2024-01-02", 110, 110, 110, 110),
            ("2024-01-03", 120, 120, 120, 120),
        ],
    )

    engine = BacktestEngine(
        data=feed,
        strategy=PartialExit(),
        initialCash=1_000,
        broker=SimpleBroker(commissionRate=0.0, taxRate=0.0, slippageRate=0.0),
        fillOnNextBar=False,
    )

    result = engine.run()

    assert len(result.trades) == 2
    assert result.finalEquity == 1_030
    assert [trade.quantity for trade in result.trades] == [1, 1]
    assert [trade.pnl for trade in result.trades] == [10, 20]


def testEasyBacktestResolvesStringPresetInEventMode(monkeypatch):
    rows = [
        ("2024-01-01", 100, 101, 99, 100),
        ("2024-01-02", 101, 102, 100, 101),
        ("2024-01-03", 102, 103, 101, 102),
        ("2024-01-04", 103, 104, 102, 103),
        ("2024-01-05", 102, 103, 101, 102),
        ("2024-01-06", 101, 102, 100, 101),
    ]
    monkeypatch.setattr(easy_api, "FinanceDataReaderFeed", makeDummyFeedFactory(rows))

    result = easy_api.backtest("TEST", "goldenCross", mode="event", fast=2, slow=3)

    assert result._result.strategy == "골든크로스(2/3)"
    assert result._result.symbol == "TEST"


def testEasyBacktestDefaultsToPresetInEventMode(monkeypatch):
    rows = [
        ("2024-01-01", 100, 101, 99, 100),
        ("2024-01-02", 101, 102, 100, 101),
        ("2024-01-03", 102, 103, 101, 102),
    ]
    monkeypatch.setattr(easy_api, "FinanceDataReaderFeed", makeDummyFeedFactory(rows))

    result = easy_api.backtest("TEST", mode="event", fast=2, slow=3)

    assert result._result.strategy == "골든크로스(2/3)"
    assert result.totalTrades >= 0


def testEasyBacktestRejectsUnknownPresetName():
    with pytest.raises(ValueError, match="Unknown preset strategy"):
        easy_api.backtest("TEST", "notARealStrategy", mode="event")


def testVbacktestFallbackKeepsPresetParams(monkeypatch):
    rows = [
        ("2024-01-01", 100, 101, 99, 100),
        ("2024-01-02", 102, 103, 101, 102),
        ("2024-01-03", 104, 105, 103, 104),
        ("2024-01-04", 103, 104, 102, 103),
        ("2024-01-05", 101, 102, 100, 101),
    ]
    feedFactory = makeDummyFeedFactory(rows)
    monkeypatch.setattr(easy_api, "FinanceDataReaderFeed", feedFactory)

    import tradix.datafeed.fdr as fdr_module
    import tradix.easy.presets as presets
    from tradix.vectorized.engine import vbacktest

    monkeypatch.setattr(fdr_module, "FinanceDataReaderFeed", feedFactory)

    captured = {}

    def spyFactory(**kwargs):
        captured.update(kwargs)
        return presets.superTrend(**kwargs)

    monkeypatch.setattr(
        easy_api,
        "_getPresetFactories",
        lambda: {
            "goldenCross": presets.goldenCross,
            "superTrend": spyFactory,
        },
    )

    with pytest.warns(UserWarning, match="Falling back to event-driven mode"):
        result = vbacktest(
            "TEST",
            "superTrend",
            atrPeriod=5,
            multiplier=2.5,
            stopLoss=4,
            takeProfit=8,
        )

    assert result._result.strategy == "슈퍼트렌드(5x2.5)"
    assert captured == {
        "atrPeriod": 5,
        "multiplier": 2.5,
        "stopLoss": 4,
        "takeProfit": 8,
    }
