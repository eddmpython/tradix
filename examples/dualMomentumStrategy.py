"""
DualMomentumStrategy - 듀얼 모멘텀 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class DualMomentumStrategy(Strategy):
    """
    듀얼 모멘텀 전략

    1. 절대 모멘텀: 현재가 > N일 전 가격 → 상승 추세
    2. 상대 모멘텀: 수익률 > 기준 수익률 → 강한 종목

    - 두 조건 모두 충족 → 매수
    - 하나라도 불충족 → 매도

    Parameters:
        lookbackPeriod: 모멘텀 측정 기간 (기본 20)
        absoluteThreshold: 절대 모멘텀 임계값 (기본 -0.05, -5% 이상)
        useTrailingStop: 트레일링 스탑 사용 여부
        trailingStopPct: 트레일링 스탑 비율 (기본 10%)

    Usage:
        strategy = DualMomentumStrategy()
        strategy.lookbackPeriod = 20
        strategy.absoluteThreshold = 0.0

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.lookbackPeriod = 20
        self.absoluteThreshold = -0.05
        self.useTrailingStop = False
        self.trailingStopPct = 0.10
        self._highestPrice = 0.0

    def onBar(self, bar: Bar):
        currentPrice = bar.close

        pastPrice = self.sma(1, offset=self.lookbackPeriod)
        if pastPrice is None or pastPrice <= 0:
            return

        momentum = (currentPrice - pastPrice) / pastPrice

        absoluteMomentumOk = momentum > self.absoluteThreshold

        hasPos = self.hasPosition(bar.symbol)

        if hasPos:
            if currentPrice > self._highestPrice:
                self._highestPrice = currentPrice

            if self.useTrailingStop:
                stopPrice = self._highestPrice * (1 - self.trailingStopPct)
                if currentPrice < stopPrice:
                    self.closePosition(bar.symbol)
                    self._highestPrice = 0.0
                    return

            if not absoluteMomentumOk:
                self.closePosition(bar.symbol)
                self._highestPrice = 0.0

        else:
            if absoluteMomentumOk:
                self.buy(bar.symbol)
                self._highestPrice = currentPrice
