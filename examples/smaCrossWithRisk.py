"""
SmaCrossWithRiskStrategy - 손절/익절 기능이 포함된 SMA 교차 전략
"""

from tradex.strategy.base import Strategy
from tradex.entities.bar import Bar


class SmaCrossWithRiskStrategy(Strategy):
    """
    SMA 교차 + 리스크 관리 전략

    - 골든크로스 → 매수
    - 데드크로스 → 매도
    - 손절 (Stop Loss): 진입가 대비 N% 하락 시 청산
    - 익절 (Take Profit): 진입가 대비 N% 상승 시 청산
    - 트레일링 스탑: 최고점 대비 N% 하락 시 청산

    Parameters:
        fastPeriod: 단기 이동평균 기간 (기본 10)
        slowPeriod: 장기 이동평균 기간 (기본 30)
        stopLossPct: 손절 비율 (기본 0.05 = 5%)
        takeProfitPct: 익절 비율 (기본 0.15 = 15%)
        useTrailingStop: 트레일링 스탑 사용 여부 (기본 False)
        trailingStopPct: 트레일링 스탑 비율 (기본 0.08 = 8%)

    Usage:
        strategy = SmaCrossWithRiskStrategy()
        strategy.stopLossPct = 0.05
        strategy.takeProfitPct = 0.10

        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
    """

    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30
        self.stopLossPct = 0.05
        self.takeProfitPct = 0.15
        self.useTrailingStop = False
        self.trailingStopPct = 0.08

        self._entryPrice = 0.0
        self._highestPrice = 0.0

    def onBar(self, bar: Bar):
        hasPos = self.hasPosition(bar.symbol)

        if hasPos:
            self._checkRiskManagement(bar)
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

        if goldenCross:
            self.buy(bar.symbol)
            self._entryPrice = bar.close
            self._highestPrice = bar.close

    def _checkRiskManagement(self, bar: Bar):
        """리스크 관리 체크"""
        currentPrice = bar.close

        if currentPrice > self._highestPrice:
            self._highestPrice = currentPrice

        pnlPct = (currentPrice - self._entryPrice) / self._entryPrice

        if pnlPct <= -self.stopLossPct:
            self.closePosition(bar.symbol)
            self._resetState()
            return

        if pnlPct >= self.takeProfitPct:
            self.closePosition(bar.symbol)
            self._resetState()
            return

        if self.useTrailingStop:
            drawdownFromHigh = (self._highestPrice - currentPrice) / self._highestPrice
            if drawdownFromHigh >= self.trailingStopPct:
                self.closePosition(bar.symbol)
                self._resetState()
                return

        fastSma = self.sma(self.fastPeriod)
        slowSma = self.sma(self.slowPeriod)

        if fastSma is not None and slowSma is not None:
            prevFastSma = self.sma(self.fastPeriod, offset=1)
            prevSlowSma = self.sma(self.slowPeriod, offset=1)

            if prevFastSma is not None and prevSlowSma is not None:
                deadCross = prevFastSma >= prevSlowSma and fastSma < slowSma
                if deadCross:
                    self.closePosition(bar.symbol)
                    self._resetState()

    def _resetState(self):
        """상태 초기화"""
        self._entryPrice = 0.0
        self._highestPrice = 0.0
