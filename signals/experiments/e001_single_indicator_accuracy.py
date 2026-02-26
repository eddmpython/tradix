"""
==============================================================================
실험 ID: E001
실험명: 단일 지표 신호 정확도
==============================================================================

목적:
- 각 기술 지표가 생성하는 신호의 정확도 측정
- 어떤 지표가 가장 신뢰할 수 있는지 파악

방법:
1. RSI, MACD, 볼린저, 이평선 등 개별 지표 신호 생성
2. 신호 발생 후 N일 수익률 측정 (N = 5, 10, 20일)
3. 정확도 = (올바른 방향 예측 / 전체 신호) × 100

==============================================================================
결과
==============================================================================

결과 요약 (합성 데이터 500일):
| 지표        | 신호수 | 정확도 | 10일수익 | 승률  | 손익비 |
|-------------|--------|--------|----------|-------|--------|
| SMA(10/30)  | 18     | 61.1%  | -0.15%   | 61.1% | 0.88   |
| MACD        | 27     | 55.6%  | 0.03%    | 55.6% | 1.03   |
| ADX         | 219    | 54.3%  | **0.76%**| 54.3% | **2.01**|
| RSI         | 24     | 54.2%  | 0.21%    | 54.2% | 1.24   |
| Bollinger   | 47     | 48.9%  | -0.40%   | 48.9% | 0.69   |
| Supertrend  | 19     | 47.4%  | -0.48%   | 47.4% | 0.67   |
| SMA(20/50)  | 8      | 37.5%  | 0.25%    | 37.5% | 1.19   |

핵심 발견:
1. **ADX가 최고 손익비(2.01)**: 추세 강도 필터링이 효과적
2. **SMA 크로스가 최고 정확도(61.1%)**: 단순하지만 안정적
3. **MACD**: 균형 잡힌 성능 (정확도 55.6%, 손익비 1.03)
4. **평균회귀 지표 부진**: 볼린저, 스토캐스틱 성능 낮음

시사점:
- 추세 추종 지표(ADX, SMA)가 모멘텀 반전 지표보다 안정적
- 신호 수가 적을수록 정확도가 높은 경향
- 손익비 > 1.0인 지표: ADX, RSI, MACD, SMA(20/50)

실험일: 2026-02-05
==============================================================================
"""

import numpy as np
import pandas as pd
import sys
import io
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradex.strategy.indicators import Indicators


@dataclass
class SignalAccuracy:
    """신호 정확도 결과"""
    indicatorName: str
    totalSignals: int
    correctSignals: int
    accuracy: float
    avgReturn5d: float
    avgReturn10d: float
    avgReturn20d: float
    winRate: float
    profitFactor: float


def generateTestData(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    다양한 시장 상황을 포함한 테스트 데이터 생성

    - 상승 추세 구간
    - 하락 추세 구간
    - 횡보 구간
    - 급등/급락 구간
    """
    np.random.seed(seed)

    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    close = np.zeros(n)
    close[0] = 50000

    for i in range(1, n):
        if i < 100:
            drift = 0.001
        elif i < 200:
            drift = -0.0008
        elif i < 300:
            drift = 0.0002
        elif i < 400:
            drift = 0.0015
        else:
            drift = -0.0005

        volatility = 0.015 if 150 < i < 180 or 350 < i < 380 else 0.008
        close[i] = close[i-1] * (1 + drift + np.random.normal(0, volatility))

    high = close * (1 + np.abs(np.random.normal(0.003, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0.003, 0.002, n)))
    openPrice = close * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.randint(1000000, 5000000, n)

    return pd.DataFrame({
        'date': dates,
        'open': openPrice,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }).set_index('date')


def calcFutureReturns(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """미래 수익률 계산"""
    result = df.copy()
    for p in periods:
        result[f'return_{p}d'] = df['close'].shift(-p) / df['close'] - 1
    return result


def evaluateRsiSignals(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> SignalAccuracy:
    """RSI 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(30, len(df) - 20):
        indicators.setIndex(i + 1)
        rsi = indicators.rsi(14)
        prevRsi = indicators.rsi(14, offset=1)

        if rsi is None or prevRsi is None:
            continue

        if prevRsi <= oversold and rsi > oversold:
            buySignals.append(i)
        elif prevRsi >= overbought and rsi < overbought:
            sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'RSI')


def evaluateMacdSignals(df: pd.DataFrame) -> SignalAccuracy:
    """MACD 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(40, len(df) - 20):
        indicators.setIndex(i + 1)
        macd, signal, hist = indicators.macd()
        indicators.setIndex(i)
        prevMacd, prevSignal, prevHist = indicators.macd()

        if macd is None or prevMacd is None:
            continue

        if prevMacd <= prevSignal and macd > signal:
            buySignals.append(i)
        elif prevMacd >= prevSignal and macd < signal:
            sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'MACD')


def evaluateBollingerSignals(df: pd.DataFrame) -> SignalAccuracy:
    """볼린저 밴드 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(25, len(df) - 20):
        indicators.setIndex(i + 1)
        upper, middle, lower = indicators.bollinger(20, 2.0)
        close = df['close'].iloc[i]
        prevClose = df['close'].iloc[i - 1]

        if upper is None or lower is None:
            continue

        indicators.setIndex(i)
        prevUpper, prevMiddle, prevLower = indicators.bollinger(20, 2.0)

        if prevLower is not None:
            if prevClose <= prevLower and close > lower:
                buySignals.append(i)
            elif close >= upper:
                sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'Bollinger')


def evaluateSmaCrossSignals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> SignalAccuracy:
    """SMA 크로스 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(slow + 5, len(df) - 20):
        indicators.setIndex(i + 1)
        fastSma = indicators.sma(fast)
        slowSma = indicators.sma(slow)

        indicators.setIndex(i)
        prevFastSma = indicators.sma(fast)
        prevSlowSma = indicators.sma(slow)

        if fastSma is None or slowSma is None or prevFastSma is None or prevSlowSma is None:
            continue

        if prevFastSma <= prevSlowSma and fastSma > slowSma:
            buySignals.append(i)
        elif prevFastSma >= prevSlowSma and fastSma < slowSma:
            sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, f'SMA({fast}/{slow})')


def evaluateStochasticSignals(df: pd.DataFrame) -> SignalAccuracy:
    """스토캐스틱 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(20, len(df) - 20):
        indicators.setIndex(i + 1)
        stochK, stochD = indicators.stochastic(14, 3)

        if stochK is None or stochD is None:
            continue

        indicators.setIndex(i)
        prevK, prevD = indicators.stochastic(14, 3)

        if prevK is None:
            continue

        if prevK < 20 and stochK > 20 and stochK > stochD:
            buySignals.append(i)
        elif prevK > 80 and stochK < 80 and stochK < stochD:
            sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'Stochastic')


def evaluateAdxSignals(df: pd.DataFrame) -> SignalAccuracy:
    """ADX 추세 강도 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(30, len(df) - 20):
        indicators.setIndex(i + 1)
        adx, plusDi, minusDi = indicators.adxWithDi(14)

        if adx is None or plusDi is None or minusDi is None:
            continue

        if adx > 25:
            if plusDi > minusDi:
                buySignals.append(i)
            else:
                sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'ADX')


def evaluateSupertrendSignals(df: pd.DataFrame) -> SignalAccuracy:
    """슈퍼트렌드 신호 평가"""
    indicators = Indicators()
    indicators.setData(df)
    indicators.setFullData(df)

    buySignals = []
    sellSignals = []

    for i in range(15, len(df) - 20):
        indicators.setIndex(i + 1)
        st, direction = indicators.supertrend(10, 3.0)

        indicators.setIndex(i)
        prevSt, prevDir = indicators.supertrend(10, 3.0)

        if direction is None or prevDir is None:
            continue

        if prevDir == -1 and direction == 1:
            buySignals.append(i)
        elif prevDir == 1 and direction == -1:
            sellSignals.append(i)

    return _evaluateSignals(df, buySignals, sellSignals, 'Supertrend')


def _evaluateSignals(
    df: pd.DataFrame,
    buySignals: List[int],
    sellSignals: List[int],
    indicatorName: str
) -> SignalAccuracy:
    """신호 평가 공통 함수"""
    dfWithReturns = calcFutureReturns(df)

    correct5d = 0
    correct10d = 0
    correct20d = 0
    totalSignals = 0

    returns5d = []
    returns10d = []
    returns20d = []

    for idx in buySignals:
        if idx >= len(dfWithReturns) - 20:
            continue
        r5 = dfWithReturns.iloc[idx]['return_5d']
        r10 = dfWithReturns.iloc[idx]['return_10d']
        r20 = dfWithReturns.iloc[idx]['return_20d']

        if pd.notna(r5):
            returns5d.append(r5)
            if r5 > 0:
                correct5d += 1
        if pd.notna(r10):
            returns10d.append(r10)
            if r10 > 0:
                correct10d += 1
        if pd.notna(r20):
            returns20d.append(r20)
            if r20 > 0:
                correct20d += 1
        totalSignals += 1

    for idx in sellSignals:
        if idx >= len(dfWithReturns) - 20:
            continue
        r5 = dfWithReturns.iloc[idx]['return_5d']
        r10 = dfWithReturns.iloc[idx]['return_10d']
        r20 = dfWithReturns.iloc[idx]['return_20d']

        if pd.notna(r5):
            returns5d.append(-r5)
            if r5 < 0:
                correct5d += 1
        if pd.notna(r10):
            returns10d.append(-r10)
            if r10 < 0:
                correct10d += 1
        if pd.notna(r20):
            returns20d.append(-r20)
            if r20 < 0:
                correct20d += 1
        totalSignals += 1

    if totalSignals == 0:
        return SignalAccuracy(
            indicatorName=indicatorName,
            totalSignals=0,
            correctSignals=0,
            accuracy=0.0,
            avgReturn5d=0.0,
            avgReturn10d=0.0,
            avgReturn20d=0.0,
            winRate=0.0,
            profitFactor=0.0,
        )

    accuracy = correct10d / totalSignals if totalSignals > 0 else 0
    avgReturn5d = np.mean(returns5d) if returns5d else 0
    avgReturn10d = np.mean(returns10d) if returns10d else 0
    avgReturn20d = np.mean(returns20d) if returns20d else 0

    winRate = len([r for r in returns10d if r > 0]) / len(returns10d) if returns10d else 0

    profits = [r for r in returns10d if r > 0]
    losses = [abs(r) for r in returns10d if r < 0]
    profitFactor = (sum(profits) / sum(losses)) if losses and sum(losses) > 0 else 0

    return SignalAccuracy(
        indicatorName=indicatorName,
        totalSignals=totalSignals,
        correctSignals=correct10d,
        accuracy=accuracy,
        avgReturn5d=avgReturn5d,
        avgReturn10d=avgReturn10d,
        avgReturn20d=avgReturn20d,
        winRate=winRate,
        profitFactor=profitFactor,
    )


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E001: 단일 지표 신호 정확도 실험")
    print("=" * 70)

    print("\n테스트 데이터 생성 중...")
    df = generateTestData(500, seed=42)
    print(f"데이터 크기: {len(df)}일")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"가격 범위: {df['close'].min():.0f} ~ {df['close'].max():.0f}")

    print("\n" + "-" * 70)
    print("각 지표별 신호 평가 중...")
    print("-" * 70)

    results = []

    print("\n1. RSI 평가...")
    results.append(evaluateRsiSignals(df))

    print("2. MACD 평가...")
    results.append(evaluateMacdSignals(df))

    print("3. 볼린저 밴드 평가...")
    results.append(evaluateBollingerSignals(df))

    print("4. SMA 크로스 (10/30) 평가...")
    results.append(evaluateSmaCrossSignals(df, 10, 30))

    print("5. SMA 크로스 (20/50) 평가...")
    results.append(evaluateSmaCrossSignals(df, 20, 50))

    print("6. 스토캐스틱 평가...")
    results.append(evaluateStochasticSignals(df))

    print("7. ADX 평가...")
    results.append(evaluateAdxSignals(df))

    print("8. 슈퍼트렌드 평가...")
    results.append(evaluateSupertrendSignals(df))

    print("\n" + "=" * 70)
    print("실험 결과")
    print("=" * 70)

    print("\n{:<15} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "지표", "신호수", "정확도", "5일수익", "10일수익", "승률", "손익비"
    ))
    print("-" * 75)

    results.sort(key=lambda x: x.accuracy, reverse=True)

    for r in results:
        print("{:<15} {:>8} {:>9.1%} {:>9.2%} {:>9.2%} {:>9.1%} {:>10.2f}".format(
            r.indicatorName,
            r.totalSignals,
            r.accuracy,
            r.avgReturn5d,
            r.avgReturn10d,
            r.winRate,
            r.profitFactor,
        ))

    print("\n" + "-" * 70)
    print("분석")
    print("-" * 70)

    bestByAccuracy = max(results, key=lambda x: x.accuracy)
    bestByReturn = max(results, key=lambda x: x.avgReturn10d)
    bestByWinRate = max(results, key=lambda x: x.winRate)
    bestByPF = max(results, key=lambda x: x.profitFactor)

    print(f"최고 정확도: {bestByAccuracy.indicatorName} ({bestByAccuracy.accuracy:.1%})")
    print(f"최고 수익률: {bestByReturn.indicatorName} ({bestByReturn.avgReturn10d:.2%})")
    print(f"최고 승률: {bestByWinRate.indicatorName} ({bestByWinRate.winRate:.1%})")
    print(f"최고 손익비: {bestByPF.indicatorName} ({bestByPF.profitFactor:.2f})")

    return results


if __name__ == '__main__':
    results = runExperiment()
