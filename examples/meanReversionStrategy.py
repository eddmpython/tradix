"""
MeanReversionStrategy - 평균회귀 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class MeanReversionStrategy(Strategy):
    """
    평균회귀 전략 (볼린저 밴드 + RSI)

    - 가격이 하단 밴드 아래 + RSI 과매도 → 매수
    - 가격이 상단 밴드 위 + RSI 과매수 → 매도
    - 가격이 중심선 복귀 → 청산

    Parameters:
        bbPeriod: 볼린저 밴드 기간 (기본 20)
        bbStd: 볼린저 밴드 표준편차 (기본 2.0)
        rsiPeriod: RSI 기간 (기본 14)
        rsiOversold: RSI 과매도 기준 (기본 30)
        rsiOverbought: RSI 과매수 기준 (기본 70)
        exitAtMiddle: 중심선에서 청산 여부 (기본 True)

    Usage:
        strategy = MeanReversionStrategy()
        strategy.bbPeriod = 20
        strategy.rsiPeriod = 14

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.bbPeriod = 20
        self.bbStd = 2.0
        self.rsiPeriod = 14
        self.rsiOversold = 30
        self.rsiOverbought = 70
        self.exitAtMiddle = True
        self._entryType = None

    def onBar(self, bar: Bar):
        bbResult = self.bollinger(period=self.bbPeriod, std=self.bbStd)
        if bbResult is None:
            return

        upper, middle, lower = bbResult

        if upper is None or middle is None or lower is None:
            return

        rsi = self.rsi(self.rsiPeriod)

        if rsi is None:
            return

        hasPos = self.hasPosition(bar.symbol)
        currentPrice = bar.close

        if not hasPos:
            oversoldCondition = currentPrice < lower and rsi < self.rsiOversold
            overboughtCondition = currentPrice > upper and rsi > self.rsiOverbought

            if oversoldCondition:
                self.buy(bar.symbol)
                self._entryType = 'long'

        else:
            if self._entryType == 'long':
                if self.exitAtMiddle and currentPrice >= middle:
                    self.closePosition(bar.symbol)
                    self._entryType = None
                elif currentPrice > upper:
                    self.closePosition(bar.symbol)
                    self._entryType = None
