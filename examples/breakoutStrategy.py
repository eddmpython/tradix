"""
BreakoutStrategy - 돌파 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class BreakoutStrategy(Strategy):
    """
    채널 돌파 전략 (Donchian Channel 기반)

    - 가격이 N일 최고가를 돌파 → 매수
    - 가격이 M일 최저가를 하향 돌파 → 매도

    Parameters:
        entryPeriod: 진입 채널 기간 (기본 20, N일 최고가 돌파시 진입)
        exitPeriod: 청산 채널 기간 (기본 10, N일 최저가 이탈시 청산)

    Usage:
        strategy = BreakoutStrategy()
        strategy.entryPeriod = 20
        strategy.exitPeriod = 10

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.entryPeriod = 20
        self.exitPeriod = 10

    def onBar(self, bar: Bar):
        entryChannel = self.donchian(self.entryPeriod, offset=1)
        exitChannel = self.donchian(self.exitPeriod, offset=1)

        if entryChannel[0] is None or exitChannel[2] is None:
            return

        entryHigh = entryChannel[0]
        exitLow = exitChannel[2]

        hasPos = self.hasPosition(bar.symbol)

        if not hasPos:
            if bar.close > entryHigh:
                self.buy(bar.symbol)

        else:
            if bar.close < exitLow:
                self.closePosition(bar.symbol)
