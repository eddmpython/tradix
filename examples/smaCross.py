"""
SmaCrossStrategy - 이동평균 교차 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class SmaCrossStrategy(Strategy):
    """
    이동평균 교차 전략 (골든크로스/데드크로스)

    - 단기 SMA가 장기 SMA를 상향 돌파 → 매수
    - 단기 SMA가 장기 SMA를 하향 돌파 → 매도

    Parameters:
        fastPeriod: 단기 이동평균 기간 (기본 10)
        slowPeriod: 장기 이동평균 기간 (기본 30)

    Usage:
        strategy = SmaCrossStrategy()
        strategy.fastPeriod = 5
        strategy.slowPeriod = 20

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30

    def onBar(self, bar: Bar):
        fastSma = self.sma(self.fastPeriod)
        slowSma = self.sma(self.slowPeriod)

        if fastSma is None or slowSma is None:
            return

        prevFastSma = self.sma(self.fastPeriod, offset=1)
        prevSlowSma = self.sma(self.slowPeriod, offset=1)

        if prevFastSma is None or prevSlowSma is None:
            return

        goldenCross = prevFastSma <= prevSlowSma and fastSma > slowSma
        deadCross = prevFastSma >= prevSlowSma and fastSma < slowSma

        if goldenCross and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)

        elif deadCross and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)
