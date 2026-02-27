"""
MacdStrategy - MACD 기반 전략
"""

from tradix.strategy.base import Strategy
from tradix.entities.bar import Bar


class MacdStrategy(Strategy):
    """
    MACD 교차 전략

    - MACD 라인이 시그널 라인을 상향 돌파 → 매수
    - MACD 라인이 시그널 라인을 하향 돌파 → 매도
    - 히스토그램 기반 추가 필터 옵션

    Parameters:
        fastPeriod: 단기 EMA 기간 (기본 12)
        slowPeriod: 장기 EMA 기간 (기본 26)
        signalPeriod: 시그널 EMA 기간 (기본 9)
        useHistogramFilter: 히스토그램 필터 사용 여부

    Usage:
        strategy = MacdStrategy()
        strategy.fastPeriod = 12
        strategy.slowPeriod = 26

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.fastPeriod = 12
        self.slowPeriod = 26
        self.signalPeriod = 9
        self.useHistogramFilter = False

    def onBar(self, bar: Bar):
        result = self.macd(
            fast=self.fastPeriod,
            slow=self.slowPeriod,
            signal=self.signalPeriod
        )

        macdLine, signalLine, histogram = result

        if macdLine is None or signalLine is None:
            return

        prevResult = self.macd(
            fast=self.fastPeriod,
            slow=self.slowPeriod,
            signal=self.signalPeriod,
            offset=1
        )

        prevMacd, prevSignal, prevHist = prevResult

        if prevMacd is None or prevSignal is None:
            return

        bullishCross = prevMacd <= prevSignal and macdLine > signalLine
        bearishCross = prevMacd >= prevSignal and macdLine < signalLine

        if self.useHistogramFilter:
            bullishCross = bullishCross and histogram > 0
            bearishCross = bearishCross and histogram < 0

        if bullishCross and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)

        elif bearishCross and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)
