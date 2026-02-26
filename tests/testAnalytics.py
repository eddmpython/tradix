"""
Advanced Analytics Module Tests

Tests for Monte Carlo Stress, Fractal Analysis, Regime Detection,
Information Theory, and Portfolio Stress modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tradex.engine import BacktestResult
from tradex.entities.trade import Trade
from tradex.entities.order import OrderSide


def _makeTrade(entryPrice, exitPrice, quantity=100, daysHeld=5, idx=0):
    entryDate = datetime(2024, 1, 15) + timedelta(days=idx * daysHeld)
    exitDate = entryDate + timedelta(days=daysHeld)
    return Trade(
        id=f"t{idx:04d}",
        symbol="TEST",
        side=OrderSide.BUY,
        entryDate=entryDate,
        entryPrice=entryPrice,
        quantity=quantity,
        exitDate=exitDate,
        exitPrice=exitPrice,
        commission=0.0,
    )


def _makeResult(nTrades=20, winRate=0.6, initialCash=10_000_000):
    """Helper to create a BacktestResult with synthetic trades and equity curve."""
    np.random.seed(42)
    trades = []
    for i in range(nTrades):
        entryPrice = 50000 + np.random.randint(-5000, 5000)
        if np.random.random() < winRate:
            exitPrice = entryPrice * (1 + np.random.uniform(0.01, 0.08))
        else:
            exitPrice = entryPrice * (1 - np.random.uniform(0.01, 0.05))
        trades.append(_makeTrade(entryPrice, exitPrice, idx=i))

    nDays = 252
    dates = pd.date_range("2024-01-01", periods=nDays, freq="B")
    returns = np.random.normal(0.0003, 0.015, nDays)
    equity = initialCash * np.cumprod(1 + returns)
    equityCurve = pd.Series(equity, index=dates)

    totalReturn = (equity[-1] - initialCash) / initialCash * 100
    winCount = sum(1 for t in trades if t.pnl > 0)

    return BacktestResult(
        strategy="TestStrategy",
        symbol="TEST",
        startDate="2024-01-01",
        endDate="2024-12-31",
        initialCash=initialCash,
        finalEquity=equity[-1],
        totalReturn=totalReturn,
        totalTrades=nTrades,
        winRate=winCount / nTrades * 100 if nTrades > 0 else 0,
        trades=trades,
        equityCurve=equityCurve,
        metrics={
            "sharpeRatio": 1.2,
            "maxDrawdown": -15.0,
            "sortinoRatio": 1.8,
        },
    )


class TestMonteCarloStress:
    """Test MonteCarloStressAnalyzer."""

    def testBasicAnalysis(self):
        from tradex.analytics.monteCarloStress import MonteCarloStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = MonteCarloStressAnalyzer()
        mc = analyzer.analyze(result, paths=100)

        assert mc.paths == 100
        assert 0.0 <= mc.ruinProbability <= 1.0
        assert isinstance(mc.confidenceBands, dict)
        assert len(mc.sharpeDistribution) == 100
        assert len(mc.mddDistribution) == 100
        assert mc.worstCase <= mc.medianReturn <= mc.bestCase

    def testConfidenceBands(self):
        from tradex.analytics.monteCarloStress import MonteCarloStressAnalyzer
        result = _makeResult(nTrades=30)
        analyzer = MonteCarloStressAnalyzer()
        mc = analyzer.analyze(result, paths=500)

        for level in ["50%", "75%", "90%", "95%", "99%"]:
            assert level in mc.confidenceBands

    def testFewTrades(self):
        from tradex.analytics.monteCarloStress import MonteCarloStressAnalyzer
        result = _makeResult(nTrades=3)
        analyzer = MonteCarloStressAnalyzer()
        mc = analyzer.analyze(result, paths=100)

        assert mc.paths == 100

    def testSummary(self):
        from tradex.analytics.monteCarloStress import MonteCarloStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = MonteCarloStressAnalyzer()
        mc = analyzer.analyze(result, paths=100)

        summaryEn = mc.summary(ko=False)
        summaryKo = mc.summary(ko=True)
        assert len(summaryEn) > 0
        assert len(summaryKo) > 0


class TestFractalAnalysis:
    """Test FractalAnalyzer."""

    def testBasicAnalysis(self):
        from tradex.analytics.fractalAnalysis import FractalAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = FractalAnalyzer()
        fractal = analyzer.analyze(result)

        assert 0.0 <= fractal.hurstExponent <= 1.0
        assert 1.0 <= fractal.fractalDimension <= 2.0
        assert fractal.marketCharacter in ("trending", "random", "meanReverting")
        assert 0.0 <= fractal.confidence <= 1.0

    def testStrategyFit(self):
        from tradex.analytics.fractalAnalysis import FractalAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = FractalAnalyzer()
        fractal = analyzer.analyze(result)

        assert isinstance(fractal.strategyFit, dict)
        assert len(fractal.strategyFit) > 0
        for score in fractal.strategyFit.values():
            assert 0 <= score <= 100

    def testHurstByPeriod(self):
        from tradex.analytics.fractalAnalysis import FractalAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = FractalAnalyzer()
        fractal = analyzer.analyze(result)

        assert isinstance(fractal.hurstByPeriod, dict)

    def testSummary(self):
        from tradex.analytics.fractalAnalysis import FractalAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = FractalAnalyzer()
        fractal = analyzer.analyze(result)

        summary = fractal.summary()
        assert len(summary) > 0


class TestRegimeDetector:
    """Test RegimeDetector."""

    def testBasicAnalysis(self):
        from tradex.analytics.regimeDetector import RegimeDetector
        result = _makeResult(nTrades=20)
        analyzer = RegimeDetector()
        regime = analyzer.analyze(result, nRegimes=3)

        assert len(regime.regimes) > 0
        assert regime.currentRegime in ("bull", "bear", "sideways")
        assert isinstance(regime.transitionMatrix, dict)
        assert isinstance(regime.regimeDistribution, dict)

    def testRegimeReturns(self):
        from tradex.analytics.regimeDetector import RegimeDetector
        result = _makeResult(nTrades=20)
        analyzer = RegimeDetector()
        regime = analyzer.analyze(result)

        assert isinstance(regime.regimeReturns, dict)
        assert isinstance(regime.regimeSharpe, dict)

    def testTransitionMatrix(self):
        from tradex.analytics.regimeDetector import RegimeDetector
        result = _makeResult(nTrades=20)
        analyzer = RegimeDetector()
        regime = analyzer.analyze(result)

        for fromRegime, transitions in regime.transitionMatrix.items():
            total = sum(transitions.values())
            assert abs(total - 1.0) < 0.01 or total == 0

    def testSummary(self):
        from tradex.analytics.regimeDetector import RegimeDetector
        result = _makeResult(nTrades=20)
        analyzer = RegimeDetector()
        regime = analyzer.analyze(result)

        summary = regime.summary()
        assert len(summary) > 0


class TestInformationTheory:
    """Test InformationTheoryAnalyzer."""

    def testBasicAnalysis(self):
        from tradex.analytics.informationTheory import InformationTheoryAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = InformationTheoryAnalyzer()
        info = analyzer.analyze(result)

        assert info.signalEntropy >= 0
        assert info.marketEntropy >= 0
        assert info.mutualInformation >= 0
        assert 0.0 <= info.informationRatio <= 1.0
        assert 0.0 <= info.redundancy <= 1.0

    def testSignalQuality(self):
        from tradex.analytics.informationTheory import InformationTheoryAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = InformationTheoryAnalyzer()
        info = analyzer.analyze(result)

        assert info.signalQuality in ("excellent", "good", "moderate", "poor", "noise")

    def testSummary(self):
        from tradex.analytics.informationTheory import InformationTheoryAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = InformationTheoryAnalyzer()
        info = analyzer.analyze(result)

        summary = info.summary()
        assert len(summary) > 0


class TestPortfolioStress:
    """Test PortfolioStressAnalyzer."""

    def testBasicAnalysis(self):
        from tradex.analytics.portfolioStress import PortfolioStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = PortfolioStressAnalyzer()
        stress = analyzer.analyze(result)

        assert len(stress.scenarios) == 6
        assert 0.0 <= stress.survivalRate <= 1.0
        assert stress.worstLoss <= 0
        assert stress.overallGrade in ("A+", "A", "B", "C", "D", "F")

    def testScenarioNames(self):
        from tradex.analytics.portfolioStress import PortfolioStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = PortfolioStressAnalyzer()
        stress = analyzer.analyze(result)

        expectedScenarios = [
            "marketCrash", "volatilitySpike", "rateShock",
            "liquidityCrisis", "flashCrash", "correlationBreakdown",
        ]
        for name in expectedScenarios:
            assert name in stress.scenarios

    def testRecoveryTimes(self):
        from tradex.analytics.portfolioStress import PortfolioStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = PortfolioStressAnalyzer()
        stress = analyzer.analyze(result)

        assert isinstance(stress.recoveryTimes, dict)
        for days in stress.recoveryTimes.values():
            assert days >= 0

    def testSummary(self):
        from tradex.analytics.portfolioStress import PortfolioStressAnalyzer
        result = _makeResult(nTrades=20)
        analyzer = PortfolioStressAnalyzer()
        stress = analyzer.analyze(result)

        summary = stress.summary()
        assert len(summary) > 0
