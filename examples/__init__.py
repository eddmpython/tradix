"""
examples - 예제 전략 모음

트렌드 추종:
    - SmaCrossStrategy: 이동평균 교차 (골든/데드크로스)
    - SmaCrossWithRiskStrategy: SMA 교차 + 손절/익절
    - MacdStrategy: MACD 시그널 교차
    - BreakoutStrategy: 채널 돌파 (터틀 스타일)
    - DualMomentumStrategy: 절대/상대 모멘텀

평균회귀:
    - RsiStrategy: RSI 과매수/과매도
    - BollingerStrategy: 볼린저 밴드 돌파
    - MeanReversionStrategy: 볼린저 + RSI 복합
"""

from tradex.examples.smaCross import SmaCrossStrategy
from tradex.examples.smaCrossWithRisk import SmaCrossWithRiskStrategy
from tradex.examples.rsiStrategy import RsiStrategy
from tradex.examples.bollingerStrategy import BollingerStrategy
from tradex.examples.macdStrategy import MacdStrategy
from tradex.examples.dualMomentumStrategy import DualMomentumStrategy
from tradex.examples.breakoutStrategy import BreakoutStrategy
from tradex.examples.meanReversionStrategy import MeanReversionStrategy
from tradex.examples.trendFilterStrategy import TrendFilterStrategy

__all__ = [
    "SmaCrossStrategy",
    "SmaCrossWithRiskStrategy",
    "RsiStrategy",
    "BollingerStrategy",
    "MacdStrategy",
    "DualMomentumStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "TrendFilterStrategy",
]
