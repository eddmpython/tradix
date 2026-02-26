"""
BollingerStrategy - 볼린저 밴드 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class BollingerStrategy(Strategy):
    """
    볼린저 밴드 전략

    - 가격이 하단 밴드 터치 → 매수 (과매도)
    - 가격이 상단 밴드 터치 → 매도 (과매수)

    Parameters:
        period: 볼린저 밴드 기간 (기본 20)
        std: 표준편차 배수 (기본 2.0)

    Usage:
        strategy = BollingerStrategy()
        strategy.period = 20
        strategy.std = 2.5

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.period = 20
        self.std = 2.0

    def onBar(self, bar: Bar):
        if len(self.history) < self.period:
            return

        upper, middle, lower = self.bollinger(self.period, self.std)

        if upper is None or lower is None:
            return

        if bar.close <= lower and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)

        elif bar.close >= upper and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)
