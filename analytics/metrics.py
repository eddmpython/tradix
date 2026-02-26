"""
Tradex Performance Metrics Module.

Calculates comprehensive performance metrics from backtest results, including
return metrics, risk-adjusted ratios, drawdown analysis, and trade statistics.

백테스트 결과로부터 종합적인 성과 지표를 계산하는 모듈입니다.
수익률 지표, 위험조정 비율, 낙폭 분석, 거래 통계를 포함합니다.

Features:
    - Total and annualized return computation
    - Sharpe, Sortino, and Calmar ratio calculation
    - Maximum drawdown and drawdown duration analysis
    - Win rate, profit factor, and expectancy metrics
    - Monthly return decomposition

Usage:
    from tradex.analytics.metrics import PerformanceMetrics

    metrics = PerformanceMetrics.calculate(
        trades=trades,
        equityCurve=equity_series,
        riskFreeRate=0.035,
    )
    print(metrics.summary())
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from tradex.entities.trade import Trade


@dataclass
class PerformanceMetrics:
    """
    Comprehensive backtest performance metrics dataclass.

    Computes and stores a full suite of performance statistics from backtest
    trade results and an equity curve. Supports return analysis, risk-adjusted
    ratios, drawdown metrics, and trade-level statistics.

    백테스트 거래 결과와 자산 곡선으로부터 종합 성과 지표를 계산하고 저장하는
    데이터 클래스입니다.

    Attributes:
        totalReturn (float): Total cumulative return in percent (총 수익률, %).
        annualReturn (float): Annualized return in percent (연율화 수익률, %).
        volatility (float): Annualized volatility in percent (연율화 변동성, %).
        sharpeRatio (float): Sharpe ratio (샤프 비율).
        sortinoRatio (float): Sortino ratio (소르티노 비율).
        calmarRatio (float): Calmar ratio (칼마 비율).
        maxDrawdown (float): Maximum drawdown in percent (최대 낙폭, %).
        maxDrawdownDuration (int): Longest drawdown period in days (최대 낙폭 지속 기간, 일).
        totalTrades (int): Total number of closed trades (총 거래 횟수).
        winningTrades (int): Number of profitable trades (수익 거래 수).
        losingTrades (int): Number of losing trades (손실 거래 수).
        winRate (float): Win rate in percent (승률, %).
        avgWin (float): Average profit per winning trade (평균 수익).
        avgLoss (float): Average loss per losing trade (평균 손실).
        avgWinPercent (float): Average return of winning trades in percent (평균 수익률, %).
        avgLossPercent (float): Average return of losing trades in percent (평균 손실률, %).
        profitFactor (float): Ratio of gross profit to gross loss (손익비).
        expectancy (float): Expected return per trade (기대 수익).
        avgHoldingDays (float): Average holding period in days (평균 보유 기간).
        maxConsecutiveWins (int): Longest winning streak (최대 연속 수익).
        maxConsecutiveLosses (int): Longest losing streak (최대 연속 손실).
        monthlyReturns (List[float]): Monthly returns in percent (월별 수익률, %).
        bestMonth (float): Best monthly return in percent (최고 월 수익률, %).
        worstMonth (float): Worst monthly return in percent (최저 월 수익률, %).

    Example:
        >>> metrics = PerformanceMetrics.calculate(trades, equityCurve)
        >>> print(f"Sharpe: {metrics.sharpeRatio:.2f}")
        >>> print(metrics.summary())
    """
    totalReturn: float = 0.0
    annualReturn: float = 0.0
    volatility: float = 0.0
    sharpeRatio: float = 0.0
    sortinoRatio: float = 0.0
    calmarRatio: float = 0.0
    maxDrawdown: float = 0.0
    maxDrawdownDuration: int = 0
    totalTrades: int = 0
    winningTrades: int = 0
    losingTrades: int = 0
    winRate: float = 0.0
    avgWin: float = 0.0
    avgLoss: float = 0.0
    avgWinPercent: float = 0.0
    avgLossPercent: float = 0.0
    profitFactor: float = 0.0
    expectancy: float = 0.0
    avgHoldingDays: float = 0.0
    maxConsecutiveWins: int = 0
    maxConsecutiveLosses: int = 0
    monthlyReturns: List[float] = None
    bestMonth: float = 0.0
    worstMonth: float = 0.0

    def __post_init__(self):
        if self.monthlyReturns is None:
            self.monthlyReturns = []

    @classmethod
    def calculate(
        cls,
        trades: List[Trade],
        equityCurve: pd.Series,
        riskFreeRate: float = 0.035
    ) -> 'PerformanceMetrics':
        """
        Calculate performance metrics from trades and equity curve.

        거래 목록과 자산 곡선으로부터 성과 지표를 계산합니다.

        Args:
            trades: List of Trade objects from the backtest (거래 목록).
            equityCurve: Equity curve as a pandas Series with datetime index
                and portfolio values (자산 곡선, index: datetime, values: equity).
            riskFreeRate: Annualized risk-free rate, default 3.5%
                (무위험 수익률, 연율, 기본 3.5%).

        Returns:
            PerformanceMetrics: Populated metrics instance, or an empty instance
                if the equity curve has fewer than 2 data points.

        Example:
            >>> metrics = PerformanceMetrics.calculate(
            ...     trades=engine_result.trades,
            ...     equityCurve=engine_result.equityCurve,
            ...     riskFreeRate=0.035,
            ... )
            >>> print(f"Total Return: {metrics.totalReturn:+.2f}%")
        """
        if len(equityCurve) < 2:
            return cls()

        closedTrades = [t for t in trades if t.isClosed]

        returns = equityCurve.pct_change().dropna()
        totalReturn = (equityCurve.iloc[-1] / equityCurve.iloc[0]) - 1

        tradingDays = len(equityCurve)
        years = tradingDays / 252
        annualReturn = (1 + totalReturn) ** (1 / years) - 1 if years > 0 else 0

        volatility = returns.std() * np.sqrt(252)

        excessReturn = annualReturn - riskFreeRate
        sharpeRatio = excessReturn / volatility if volatility > 0 else 0

        negativeReturns = returns[returns < 0]
        downsideVol = negativeReturns.std() * np.sqrt(252) if len(negativeReturns) > 0 else 0
        sortinoRatio = excessReturn / downsideVol if downsideVol > 0 else 0

        cumMax = equityCurve.cummax()
        drawdown = (equityCurve - cumMax) / cumMax
        maxDrawdown = drawdown.min()

        calmarRatio = annualReturn / abs(maxDrawdown) if maxDrawdown != 0 else 0

        maxDrawdownDuration = cls._calculateMaxDrawdownDuration(drawdown)

        wins = [t for t in closedTrades if t.pnl > 0]
        losses = [t for t in closedTrades if t.pnl <= 0]

        totalTrades = len(closedTrades)
        winningTrades = len(wins)
        losingTrades = len(losses)
        winRate = winningTrades / totalTrades * 100 if totalTrades > 0 else 0

        avgWin = np.mean([t.pnl for t in wins]) if wins else 0
        avgLoss = np.mean([t.pnl for t in losses]) if losses else 0
        avgWinPercent = np.mean([t.pnlPercent for t in wins]) if wins else 0
        avgLossPercent = np.mean([t.pnlPercent for t in losses]) if losses else 0

        totalProfit = sum(t.pnl for t in wins)
        totalLosses = abs(sum(t.pnl for t in losses))
        profitFactor = totalProfit / totalLosses if totalLosses > 0 else float('inf')

        expectancy = (winRate / 100 * avgWinPercent) + ((1 - winRate / 100) * avgLossPercent)

        holdingDays = [t.holdingDays for t in closedTrades if t.holdingDays > 0]
        avgHoldingDays = np.mean(holdingDays) if holdingDays else 0

        maxConsecutiveWins, maxConsecutiveLosses = cls._calculateConsecutive(closedTrades)

        monthlyReturns = cls._calculateMonthlyReturns(equityCurve)

        return cls(
            totalReturn=totalReturn * 100,
            annualReturn=annualReturn * 100,
            volatility=volatility * 100,
            sharpeRatio=sharpeRatio,
            sortinoRatio=sortinoRatio,
            calmarRatio=calmarRatio,
            maxDrawdown=maxDrawdown * 100,
            maxDrawdownDuration=maxDrawdownDuration,
            totalTrades=totalTrades,
            winningTrades=winningTrades,
            losingTrades=losingTrades,
            winRate=winRate,
            avgWin=avgWin,
            avgLoss=avgLoss,
            avgWinPercent=avgWinPercent,
            avgLossPercent=avgLossPercent,
            profitFactor=profitFactor,
            expectancy=expectancy,
            avgHoldingDays=avgHoldingDays,
            maxConsecutiveWins=maxConsecutiveWins,
            maxConsecutiveLosses=maxConsecutiveLosses,
            monthlyReturns=monthlyReturns,
            bestMonth=max(monthlyReturns) if monthlyReturns else 0,
            worstMonth=min(monthlyReturns) if monthlyReturns else 0,
        )

    @staticmethod
    def _calculateMaxDrawdownDuration(drawdown: pd.Series) -> int:
        """
        Calculate the maximum drawdown duration in trading days.

        최대 낙폭 지속 기간을 거래일 기준으로 계산합니다.

        Args:
            drawdown: Drawdown series (negative values indicate drawdown).

        Returns:
            int: Maximum number of consecutive days in drawdown.
        """
        isDrawdown = drawdown < 0
        if not isDrawdown.any():
            return 0

        groups = (isDrawdown != isDrawdown.shift()).cumsum()
        drawdownGroups = isDrawdown.groupby(groups)

        maxDuration = 0
        for name, group in drawdownGroups:
            if group.iloc[0]:
                maxDuration = max(maxDuration, len(group))

        return maxDuration

    @staticmethod
    def _calculateConsecutive(trades: List[Trade]) -> tuple:
        """
        Calculate maximum consecutive wins and losses.

        최대 연속 수익/손실 횟수를 계산합니다.

        Args:
            trades: List of closed Trade objects.

        Returns:
            tuple: (max_consecutive_wins, max_consecutive_losses).
        """
        if not trades:
            return 0, 0

        maxWins = 0
        maxLosses = 0
        currentWins = 0
        currentLosses = 0

        for trade in trades:
            if trade.pnl > 0:
                currentWins += 1
                currentLosses = 0
                maxWins = max(maxWins, currentWins)
            else:
                currentLosses += 1
                currentWins = 0
                maxLosses = max(maxLosses, currentLosses)

        return maxWins, maxLosses

    @staticmethod
    def _calculateMonthlyReturns(equityCurve: pd.Series) -> List[float]:
        """
        Calculate monthly returns from the equity curve.

        자산 곡선으로부터 월별 수익률을 계산합니다.

        Args:
            equityCurve: Equity curve as a pandas Series with datetime index.

        Returns:
            List[float]: Monthly returns in percent.
        """
        if len(equityCurve) < 2:
            return []

        monthly = equityCurve.resample('ME').last()
        returns = monthly.pct_change().dropna()
        return (returns * 100).tolist()

    def toDict(self) -> dict:
        """
        Convert metrics to a plain dictionary.

        성과 지표를 딕셔너리로 변환합니다.

        Returns:
            dict: All metric fields as key-value pairs.
        """
        return {
            'totalReturn': self.totalReturn,
            'annualReturn': self.annualReturn,
            'volatility': self.volatility,
            'sharpeRatio': self.sharpeRatio,
            'sortinoRatio': self.sortinoRatio,
            'calmarRatio': self.calmarRatio,
            'maxDrawdown': self.maxDrawdown,
            'maxDrawdownDuration': self.maxDrawdownDuration,
            'totalTrades': self.totalTrades,
            'winningTrades': self.winningTrades,
            'losingTrades': self.losingTrades,
            'winRate': self.winRate,
            'avgWin': self.avgWin,
            'avgLoss': self.avgLoss,
            'avgWinPercent': self.avgWinPercent,
            'avgLossPercent': self.avgLossPercent,
            'profitFactor': self.profitFactor,
            'expectancy': self.expectancy,
            'avgHoldingDays': self.avgHoldingDays,
            'maxConsecutiveWins': self.maxConsecutiveWins,
            'maxConsecutiveLosses': self.maxConsecutiveLosses,
            'bestMonth': self.bestMonth,
            'worstMonth': self.worstMonth,
        }

    def summary(self) -> str:
        """
        Generate a human-readable summary string of key metrics.

        주요 성과 지표의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line formatted summary including return, ratios, MDD,
                trade count, win rate, profit factor, and expectancy.
        """
        return (
            f"수익률: {self.totalReturn:+.2f}% (연 {self.annualReturn:+.2f}%)\n"
            f"샤프: {self.sharpeRatio:.2f} | 소르티노: {self.sortinoRatio:.2f}\n"
            f"MDD: {self.maxDrawdown:.2f}% ({self.maxDrawdownDuration}일)\n"
            f"거래: {self.totalTrades}회 | 승률: {self.winRate:.1f}%\n"
            f"손익비: {self.profitFactor:.2f} | 기대수익: {self.expectancy:.2f}%"
        )

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(return={self.totalReturn:+.2f}%, "
            f"sharpe={self.sharpeRatio:.2f}, "
            f"mdd={self.maxDrawdown:.2f}%, "
            f"winRate={self.winRate:.1f}%)"
        )
