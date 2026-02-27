"""
TrendFilterStrategy - 트렌드 필터가 적용된 복합 전략
"""

from tradix.strategy.base import Strategy
from tradix.entities.bar import Bar


class TrendFilterStrategy(Strategy):
    """
    트렌드 필터 복합 전략

    1. 트렌드 필터: 200일 이평선 위 = 상승장, 아래 = 하락장
    2. 상승장: SMA 골든크로스에서 매수
    3. 하락장: 매수 금지, 기존 포지션 청산

    + ADX 필터로 추세 강도 확인
    + RSI 필터로 과매수 구간 매수 회피

    Parameters:
        trendPeriod: 트렌드 판단 이평선 (기본 200)
        fastPeriod: 단기 이평선 (기본 10)
        slowPeriod: 장기 이평선 (기본 30)
        useAdxFilter: ADX 필터 사용 (기본 True)
        adxThreshold: ADX 임계값 (기본 20)
        useRsiFilter: RSI 필터 사용 (기본 True)
        rsiOverbought: RSI 과매수 기준 (기본 70)
    """

    def initialize(self):
        self.trendPeriod = 200
        self.fastPeriod = 10
        self.slowPeriod = 30
        self.useAdxFilter = True
        self.adxThreshold = 20
        self.useRsiFilter = True
        self.rsiOverbought = 70

    def onBar(self, bar: Bar):
        trendSma = self.sma(self.trendPeriod)
        if trendSma is None:
            return

        isUptrend = bar.close > trendSma

        hasPos = self.hasPosition(bar.symbol)

        if not isUptrend and hasPos:
            self.closePosition(bar.symbol)
            return

        if not isUptrend:
            return

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

        if hasPos and deadCross:
            self.closePosition(bar.symbol)
            return

        if hasPos:
            return

        if not goldenCross:
            return

        if self.useAdxFilter:
            adx = self.adx(14)
            if adx is None or adx < self.adxThreshold:
                return

        if self.useRsiFilter:
            rsi = self.rsi(14)
            if rsi is not None and rsi > self.rsiOverbought:
                return

        self.buy(bar.symbol)
