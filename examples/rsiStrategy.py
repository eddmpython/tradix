"""
RsiStrategy - RSI 역추세 전략
"""

from tradix.strategy.base import Strategy
from tradix.entities.bar import Bar


class RsiStrategy(Strategy):
    """
    RSI 역추세 전략

    - RSI가 과매도 구간(30 이하)에 진입 → 매수
    - RSI가 과매수 구간(70 이상)에 진입 → 매도

    Parameters:
        rsiPeriod: RSI 기간 (기본 14)
        oversold: 과매도 기준 (기본 30)
        overbought: 과매수 기준 (기본 70)

    Usage:
        strategy = RsiStrategy()
        strategy.oversold = 25
        strategy.overbought = 75

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.rsiPeriod = 14
        self.oversold = 30
        self.overbought = 70

    def onBar(self, bar: Bar):
        if len(self.history) < self.rsiPeriod + 1:
            return

        rsi = self.rsi(self.rsiPeriod)

        if rsi is None:
            return

        if rsi < self.oversold and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)

        elif rsi > self.overbought and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)
