"""Tradex Signal Backtest Verification Module.

Validates signal quality by backtesting SignalPredictor signals and comparing
performance against Buy-and-Hold and Random benchmarks.

신호 품질 백테스트 검증 모듈 - SignalPredictor가 생성한 신호를 백테스트하여
Buy&Hold, Random 전략 대비 성과를 비교합니다.

Features:
    - Signal-based backtest with realistic commission and slippage modeling
    - Buy-and-Hold benchmark backtest
    - Random trading benchmark backtest (seeded for reproducibility)
    - Comprehensive metric comparison (return, Sharpe, MDD, win rate, profit factor)
    - Alpha calculation vs. Buy-and-Hold
    - Quick one-call evaluation via quickEvaluate()

Usage:
    from tradex.signals.backtest import SignalBacktester, quickEvaluate

    backtester = SignalBacktester(df)
    result = backtester.evaluate(predictor)
    print(result.summary)
    print(result.comparison.betterThanBuyHold)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest run.

    백테스트 성과 지표.

    Attributes:
        totalReturn: Total return over the period (전체 수익률).
        annualizedReturn: Annualized return (연율 수익률).
        sharpeRatio: Annualized Sharpe ratio (연율 샤프 비율).
        maxDrawdown: Maximum drawdown (최대 낙폭).
        winRate: Win rate (fraction of profitable trades) (승률).
        profitFactor: Gross profit / gross loss (손익비).
        totalTrades: Total number of completed trades (총 거래 횟수).
        avgHoldingDays: Average holding period in days (평균 보유 기간, 일).
    """
    totalReturn: float
    annualizedReturn: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float
    profitFactor: float
    totalTrades: int
    avgHoldingDays: float


@dataclass
class BenchmarkComparison:
    """Comparison of signal strategy performance against benchmarks.

    벤치마크 대비 신호 전략 성과 비교.

    Attributes:
        vsBuyHold: Return difference vs. Buy-and-Hold (Buy&Hold 대비 수익률 차이).
        vsRandom: Return difference vs. Random strategy (Random 대비 수익률 차이).
        alpha: Annualized alpha vs. Buy-and-Hold (연율 알파).
        betterThanBuyHold: Whether signal outperforms Buy-and-Hold (Buy&Hold 초과 여부).
        betterThanRandom: Whether signal outperforms Random (Random 초과 여부).
    """
    vsBuyHold: float
    vsRandom: float
    alpha: float
    betterThanBuyHold: bool
    betterThanRandom: bool


@dataclass
class SignalBacktestResult:
    """Complete signal backtest result with metrics and benchmark comparison.

    신호 백테스트 전체 결과 (지표 및 벤치마크 비교 포함).

    Attributes:
        signal: BacktestMetrics for the signal strategy (신호 전략 성과 지표).
        buyHold: BacktestMetrics for the Buy-and-Hold benchmark (Buy&Hold 성과 지표).
        random: BacktestMetrics for the Random benchmark (Random 성과 지표).
        comparison: BenchmarkComparison with relative performance (벤치마크 비교).
        summary: Formatted summary string (결과 요약 문자열).
        details: Additional details including trade list (거래 내역 등 추가 상세).
    """
    signal: BacktestMetrics
    buyHold: BacktestMetrics
    random: BacktestMetrics
    comparison: BenchmarkComparison
    summary: str
    details: Dict


class SignalBacktester:
    """Signal quality backtester comparing signal performance against benchmarks.

    Backtests SignalPredictor signals with realistic commission and slippage,
    then compares performance against Buy-and-Hold and Random strategies.

    신호 품질 백테스트 검증기 - SignalPredictor 신호를 수수료와 슬리피지를 반영하여
    백테스트하고, Buy&Hold 및 Random 전략과 성과를 비교합니다.

    Attributes:
        df: OHLCV DataFrame (OHLCV 데이터).
        initialCapital: Starting capital (초기 자본금).
        commission: Commission rate per trade (거래당 수수료율).
        slippage: Slippage rate per trade (거래당 슬리피지율).

    Example:
        >>> backtester = SignalBacktester(df, initialCapital=10_000_000)
        >>> result = backtester.evaluate(predictor)
        >>> if result.comparison.betterThanBuyHold:
        ...     print("Signal outperforms Buy & Hold!")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initialCapital: float = 10000000,
        commission: float = 0.00015,
        slippage: float = 0.0005,
    ):
        """Initialize the signal backtester.

        Args:
            df: OHLCV DataFrame (OHLCV 데이터).
            initialCapital: Starting capital in KRW (default 10,000,000)
                            (초기 자본금, 기본 1000만원).
            commission: Commission rate per trade (default 0.015%)
                        (거래 수수료율).
            slippage: Slippage rate per trade (default 0.05%)
                      (슬리피지율).
        """
        self.df = df
        self.initialCapital = initialCapital
        self.commission = commission
        self.slippage = slippage

    def evaluate(
        self,
        predictor,
        strategies: Optional[List[str]] = None,
        consensusThreshold: float = 0.5,
        holdingDays: int = 10,
    ) -> SignalBacktestResult:
        """Evaluate signal quality by backtesting against benchmarks.

        Args:
            predictor: SignalPredictor instance to evaluate (평가할 SignalPredictor).
            strategies: Strategy names to use. If None, all are used (사용할 전략).
            consensusThreshold: Consensus threshold for signal generation
                                (컨센서스 임계값).
            holdingDays: Maximum holding period per trade in days (거래당 최대 보유 기간).

        Returns:
            SignalBacktestResult with signal, Buy&Hold, and Random metrics
            plus comparison.
        """
        signalMetrics, signalTrades = self._backtestSignal(
            predictor, strategies, consensusThreshold, holdingDays
        )

        buyHoldMetrics = self._backtestBuyHold()

        randomMetrics = self._backtestRandom(
            numTrades=max(10, signalMetrics.totalTrades),
            holdingDays=holdingDays,
            seed=42
        )

        comparison = self._compare(signalMetrics, buyHoldMetrics, randomMetrics)

        summary = self._buildSummary(signalMetrics, buyHoldMetrics, randomMetrics, comparison)

        return SignalBacktestResult(
            signal=signalMetrics,
            buyHold=buyHoldMetrics,
            random=randomMetrics,
            comparison=comparison,
            summary=summary,
            details={
                'trades': signalTrades,
                'strategies': strategies,
                'threshold': consensusThreshold,
            }
        )

    def _backtestSignal(
        self,
        predictor,
        strategies: Optional[List[str]],
        threshold: float,
        holdingDays: int,
    ) -> Tuple[BacktestMetrics, List[Dict]]:
        """Run signal-based backtest with commission and slippage.

        Args:
            predictor: SignalPredictor instance.
            strategies: Strategy names to use.
            threshold: Consensus threshold.
            holdingDays: Maximum holding period.

        Returns:
            Tuple of (BacktestMetrics, list of trade dicts).
        """
        trades = []
        capital = self.initialCapital
        position = 0
        entryPrice = 0
        entryIdx = 0

        equityCurve = [capital]
        peakCapital = capital

        minIdx = 60

        for i in range(minIdx, len(self.df) - holdingDays):
            subDf = self.df.iloc[:i+1].copy()

            from tradex.signals import SignalPredictor, SignalConfig
            tempPredictor = SignalPredictor(subDf, predictor.config)
            result = tempPredictor.predict(strategies=strategies, consensusThreshold=threshold)

            if position == 0 and result.signal == 1:
                entryPrice = self.df['close'].iloc[i] * (1 + self.slippage)
                shares = int(capital * 0.95 / entryPrice)
                if shares > 0:
                    cost = shares * entryPrice * (1 + self.commission)
                    capital -= cost
                    position = shares
                    entryIdx = i

            elif position > 0 and (result.signal == -1 or i - entryIdx >= holdingDays):
                exitPrice = self.df['close'].iloc[i] * (1 - self.slippage)
                proceeds = position * exitPrice * (1 - self.commission)
                capital += proceeds

                pnl = (exitPrice - entryPrice) / entryPrice
                trades.append({
                    'entryIdx': entryIdx,
                    'exitIdx': i,
                    'entryPrice': entryPrice,
                    'exitPrice': exitPrice,
                    'pnl': pnl,
                    'holdingDays': i - entryIdx,
                })

                position = 0
                entryPrice = 0

            currentValue = capital + position * self.df['close'].iloc[i]
            equityCurve.append(currentValue)
            peakCapital = max(peakCapital, currentValue)

        if position > 0:
            exitPrice = self.df['close'].iloc[-1]
            capital += position * exitPrice * (1 - self.commission)

        finalCapital = capital
        totalReturn = (finalCapital - self.initialCapital) / self.initialCapital

        nDays = len(self.df)
        annualizedReturn = (1 + totalReturn) ** (252 / nDays) - 1 if nDays > 0 else 0

        returns = np.diff(equityCurve) / equityCurve[:-1] if len(equityCurve) > 1 else [0]
        sharpeRatio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        maxDrawdown = 0
        peak = self.initialCapital
        for val in equityCurve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            maxDrawdown = max(maxDrawdown, dd)

        winTrades = [t for t in trades if t['pnl'] > 0]
        lossTrades = [t for t in trades if t['pnl'] <= 0]
        winRate = len(winTrades) / len(trades) if trades else 0

        totalProfit = sum(t['pnl'] for t in winTrades) if winTrades else 0
        totalLoss = abs(sum(t['pnl'] for t in lossTrades)) if lossTrades else 0.001
        profitFactor = totalProfit / totalLoss if totalLoss > 0 else 0

        avgHoldingDays = np.mean([t['holdingDays'] for t in trades]) if trades else 0

        return BacktestMetrics(
            totalReturn=totalReturn,
            annualizedReturn=annualizedReturn,
            sharpeRatio=sharpeRatio,
            maxDrawdown=maxDrawdown,
            winRate=winRate,
            profitFactor=profitFactor,
            totalTrades=len(trades),
            avgHoldingDays=avgHoldingDays,
        ), trades

    def _backtestBuyHold(self) -> BacktestMetrics:
        """Run Buy-and-Hold benchmark backtest.

        Returns:
            BacktestMetrics for a buy-at-open, sell-at-close strategy.
        """
        entryPrice = self.df['close'].iloc[0]
        exitPrice = self.df['close'].iloc[-1]

        totalReturn = (exitPrice - entryPrice) / entryPrice

        nDays = len(self.df)
        annualizedReturn = (1 + totalReturn) ** (252 / nDays) - 1 if nDays > 0 else 0

        prices = self.df['close'].values
        returns = np.diff(prices) / prices[:-1]
        sharpeRatio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        peak = prices[0]
        maxDrawdown = 0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            maxDrawdown = max(maxDrawdown, dd)

        return BacktestMetrics(
            totalReturn=totalReturn,
            annualizedReturn=annualizedReturn,
            sharpeRatio=sharpeRatio,
            maxDrawdown=maxDrawdown,
            winRate=1.0 if totalReturn > 0 else 0.0,
            profitFactor=abs(totalReturn) if totalReturn > 0 else 0,
            totalTrades=1,
            avgHoldingDays=nDays,
        )

    def _backtestRandom(
        self,
        numTrades: int,
        holdingDays: int,
        seed: int = 42,
    ) -> BacktestMetrics:
        """Run random trading benchmark backtest with seeded reproducibility.

        Args:
            numTrades: Number of random trades to simulate.
            holdingDays: Holding period per trade.
            seed: Random seed for reproducibility.

        Returns:
            BacktestMetrics for the random strategy.
        """
        np.random.seed(seed)

        trades = []
        n = len(self.df)

        for _ in range(numTrades):
            entryIdx = np.random.randint(60, n - holdingDays - 1)
            exitIdx = min(entryIdx + holdingDays, n - 1)

            entryPrice = self.df['close'].iloc[entryIdx]
            exitPrice = self.df['close'].iloc[exitIdx]

            pnl = (exitPrice - entryPrice) / entryPrice
            trades.append({
                'entryIdx': entryIdx,
                'exitIdx': exitIdx,
                'pnl': pnl,
            })

        totalReturn = np.mean([t['pnl'] for t in trades]) if trades else 0

        nDays = len(self.df)
        annualizedReturn = (1 + totalReturn) ** (252 / nDays) - 1 if nDays > 0 else 0

        returns = [t['pnl'] for t in trades]
        sharpeRatio = np.mean(returns) / np.std(returns) * np.sqrt(252 / holdingDays) if np.std(returns) > 0 else 0

        winTrades = [t for t in trades if t['pnl'] > 0]
        winRate = len(winTrades) / len(trades) if trades else 0

        totalProfit = sum(t['pnl'] for t in winTrades) if winTrades else 0
        totalLoss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0)) or 0.001
        profitFactor = totalProfit / totalLoss

        return BacktestMetrics(
            totalReturn=totalReturn,
            annualizedReturn=annualizedReturn,
            sharpeRatio=sharpeRatio,
            maxDrawdown=0,
            winRate=winRate,
            profitFactor=profitFactor,
            totalTrades=numTrades,
            avgHoldingDays=holdingDays,
        )

    def _compare(
        self,
        signal: BacktestMetrics,
        buyHold: BacktestMetrics,
        random: BacktestMetrics,
    ) -> BenchmarkComparison:
        """Compare signal performance against benchmark strategies.

        Args:
            signal: Signal strategy metrics.
            buyHold: Buy-and-Hold benchmark metrics.
            random: Random benchmark metrics.

        Returns:
            BenchmarkComparison with relative performance.
        """
        vsBuyHold = signal.totalReturn - buyHold.totalReturn
        vsRandom = signal.totalReturn - random.totalReturn

        alpha = signal.annualizedReturn - buyHold.annualizedReturn

        return BenchmarkComparison(
            vsBuyHold=vsBuyHold,
            vsRandom=vsRandom,
            alpha=alpha,
            betterThanBuyHold=signal.totalReturn > buyHold.totalReturn,
            betterThanRandom=signal.totalReturn > random.totalReturn,
        )

    def _buildSummary(
        self,
        signal: BacktestMetrics,
        buyHold: BacktestMetrics,
        random: BacktestMetrics,
        comparison: BenchmarkComparison,
    ) -> str:
        """Build a formatted summary string of backtest results.

        Args:
            signal: Signal strategy metrics.
            buyHold: Buy-and-Hold metrics.
            random: Random strategy metrics.
            comparison: Benchmark comparison.

        Returns:
            Multi-line formatted summary string.
        """
        lines = [
            "=" * 60,
            "신호 백테스트 결과",
            "=" * 60,
            "",
            f"{'전략':<20} {'수익률':>10} {'샤프':>8} {'MDD':>8} {'승률':>8}",
            "-" * 60,
            f"{'Signal':<20} {signal.totalReturn:>9.1%} {signal.sharpeRatio:>8.2f} {signal.maxDrawdown:>7.1%} {signal.winRate:>7.1%}",
            f"{'Buy & Hold':<20} {buyHold.totalReturn:>9.1%} {buyHold.sharpeRatio:>8.2f} {buyHold.maxDrawdown:>7.1%} {'-':>8}",
            f"{'Random':<20} {random.totalReturn:>9.1%} {random.sharpeRatio:>8.2f} {'-':>8} {random.winRate:>7.1%}",
            "",
            "-" * 60,
            "벤치마크 대비",
            "-" * 60,
            f"vs Buy&Hold: {comparison.vsBuyHold:+.1%} {'✓' if comparison.betterThanBuyHold else '✗'}",
            f"vs Random:   {comparison.vsRandom:+.1%} {'✓' if comparison.betterThanRandom else '✗'}",
            f"Alpha:       {comparison.alpha:+.1%}",
            "",
            f"거래 횟수: {signal.totalTrades}회, 평균 보유: {signal.avgHoldingDays:.1f}일",
            "=" * 60,
        ]

        return "\n".join(lines)


def quickEvaluate(
    df: pd.DataFrame,
    strategies: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> SignalBacktestResult:
    """One-call convenience function for signal quality evaluation.

    Creates a SignalPredictor and SignalBacktester with default settings and
    runs the full evaluation pipeline.

    간편 신호 평가 함수 - 기본 설정으로 신호 품질을 즉시 평가합니다.

    Args:
        df: OHLCV DataFrame (OHLCV 데이터).
        strategies: Strategy names to use. If None, all are used (사용할 전략).
        threshold: Consensus threshold (컨센서스 임계값).

    Returns:
        SignalBacktestResult with full benchmark comparison.
    """
    from tradex.signals import SignalPredictor

    predictor = SignalPredictor(df)
    backtester = SignalBacktester(df)

    return backtester.evaluate(
        predictor,
        strategies=strategies,
        consensusThreshold=threshold,
    )
