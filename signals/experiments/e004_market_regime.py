"""
==============================================================================
실험 ID: E004
실험명: 시장 레짐별 신호 성능
==============================================================================

목적:
- MarketClassifier와 연동하여 시장 상황별 신호 성능 분석
- 어떤 시장에서 어떤 전략이 유효한지 파악
- 레짐별 최적 전략 조합 도출

방법:
1. 데이터를 시장 레짐으로 분류 (상승/하락/횡보/고변동)
2. 각 레짐에서 전략별 신호 정확도 측정
3. 레짐별 최적 전략 조합 도출

가설:
- 상승장: 추세 추종 신호가 유효
- 하락장: 모멘텀 반전 신호가 유효
- 횡보장: 평균회귀 신호가 유효

==============================================================================
결과
==============================================================================

[실험 실행 후 기록]

실험일: 2026-02-05
==============================================================================
"""

import numpy as np
import pandas as pd
import sys
import io
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradex.signals import SignalPredictor
from tradex.signals.backtest import SignalBacktester


class MarketRegime(Enum):
    """시장 레짐"""
    UPTREND = "상승장"
    DOWNTREND = "하락장"
    SIDEWAYS = "횡보장"
    HIGH_VOLATILITY = "고변동장"


@dataclass
class RegimeResult:
    """레짐별 결과"""
    regime: MarketRegime
    strategy: str
    totalSignals: int
    accuracy: float
    avgReturn: float
    winRate: float
    profitFactor: float


def classifyRegime(df: pd.DataFrame, window: int = 20) -> MarketRegime:
    """시장 레짐 분류"""
    if len(df) < window:
        return MarketRegime.SIDEWAYS

    closes = df['close'].values
    returns = np.diff(closes) / closes[:-1]

    recentReturns = returns[-window:]
    totalReturn = (closes[-1] / closes[-window]) - 1
    volatility = np.std(recentReturns) * np.sqrt(252)

    if volatility > 0.4:
        return MarketRegime.HIGH_VOLATILITY
    elif totalReturn > 0.05:
        return MarketRegime.UPTREND
    elif totalReturn < -0.05:
        return MarketRegime.DOWNTREND
    else:
        return MarketRegime.SIDEWAYS


def generateRegimeData(regime: MarketRegime, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """레짐별 데이터 생성"""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    close = np.zeros(n)
    close[0] = 50000

    for i in range(1, n):
        if regime == MarketRegime.UPTREND:
            drift = 0.0012 + np.sin(i / 30) * 0.0003
            vol = 0.01
        elif regime == MarketRegime.DOWNTREND:
            drift = -0.001 - np.sin(i / 30) * 0.0002
            vol = 0.012
        elif regime == MarketRegime.SIDEWAYS:
            drift = np.sin(i / 20) * 0.0004
            vol = 0.006
        else:  # HIGH_VOLATILITY
            drift = np.sin(i / 15) * 0.002
            vol = 0.025

        close[i] = close[i-1] * (1 + drift + np.random.normal(0, vol))

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


def evaluateStrategyInRegime(
    df: pd.DataFrame,
    regime: MarketRegime,
    strategies: List[str],
    strategyName: str,
) -> RegimeResult:
    """레짐에서 전략 평가"""
    signals = []
    returns10d = []

    for i in range(60, len(df) - 15):
        subDf = df.iloc[:i+1]

        predictor = SignalPredictor(subDf)
        result = predictor.predict(strategies=strategies, consensusThreshold=0.5)

        if result.signal != 0:
            futureReturn = (df['close'].iloc[i + 10] / df['close'].iloc[i]) - 1

            if result.signal == 1:
                signals.append((i, 1, futureReturn))
                returns10d.append(futureReturn)
            else:
                signals.append((i, -1, -futureReturn))
                returns10d.append(-futureReturn)

    if not signals:
        return RegimeResult(
            regime=regime,
            strategy=strategyName,
            totalSignals=0,
            accuracy=0.0,
            avgReturn=0.0,
            winRate=0.0,
            profitFactor=0.0,
        )

    totalSignals = len(signals)
    correctSignals = len([r for r in returns10d if r > 0])
    accuracy = correctSignals / totalSignals

    avgReturn = np.mean(returns10d)
    winRate = len([r for r in returns10d if r > 0]) / len(returns10d)

    profits = [r for r in returns10d if r > 0]
    losses = [abs(r) for r in returns10d if r < 0]
    profitFactor = (sum(profits) / sum(losses)) if losses and sum(losses) > 0 else 0

    return RegimeResult(
        regime=regime,
        strategy=strategyName,
        totalSignals=totalSignals,
        accuracy=accuracy,
        avgReturn=avgReturn,
        winRate=winRate,
        profitFactor=profitFactor,
    )


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E004: 시장 레짐별 신호 성능")
    print("=" * 70)
    print()
    print("가설:")
    print("  - 상승장: 추세 추종(trend) 유효")
    print("  - 하락장: 모멘텀 반전(momentum) 유효")
    print("  - 횡보장: 평균회귀(meanReversion) 유효")
    print("  - 고변동장: 변동성(volatility) 전략 유효")
    print()

    regimes = [
        MarketRegime.UPTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.SIDEWAYS,
        MarketRegime.HIGH_VOLATILITY,
    ]

    strategies = {
        'trend': ['trend'],
        'momentum': ['momentum'],
        'meanReversion': ['meanReversion'],
        'breakout': ['breakout'],
        'trend+momentum': ['trend', 'momentum'],
        'all': None,
    }

    results = []

    print("-" * 70)
    print("레짐별 전략 평가")
    print("-" * 70)

    for regime in regimes:
        print(f"\n{regime.value} 데이터 생성 및 평가...")
        df = generateRegimeData(regime, n=350, seed=42)

        detectedRegime = classifyRegime(df)
        print(f"  감지된 레짐: {detectedRegime.value}")

        for strategyName, strategyList in strategies.items():
            result = evaluateStrategyInRegime(df, regime, strategyList, strategyName)
            results.append(result)

    print("\n" + "=" * 70)
    print("실험 결과")
    print("=" * 70)

    for regime in regimes:
        print(f"\n### {regime.value} ###")
        print("{:<20} {:>8} {:>8} {:>10} {:>10}".format(
            "전략", "신호수", "정확도", "평균수익", "손익비"
        ))
        print("-" * 60)

        regimeResults = [r for r in results if r.regime == regime]
        regimeResults.sort(key=lambda x: x.profitFactor, reverse=True)

        for r in regimeResults:
            if r.totalSignals > 0:
                print("{:<20} {:>8} {:>7.1%} {:>9.2%} {:>10.2f}".format(
                    r.strategy,
                    r.totalSignals,
                    r.accuracy,
                    r.avgReturn,
                    r.profitFactor,
                ))

    print("\n" + "-" * 70)
    print("레짐별 최적 전략")
    print("-" * 70)

    for regime in regimes:
        regimeResults = [r for r in results if r.regime == regime and r.totalSignals >= 5]
        if regimeResults:
            best = max(regimeResults, key=lambda x: x.profitFactor)
            print(f"{regime.value:<12}: {best.strategy} (PF={best.profitFactor:.2f}, 정확도={best.accuracy:.1%})")

    print("\n" + "-" * 70)
    print("가설 검증")
    print("-" * 70)

    hypotheses = {
        MarketRegime.UPTREND: 'trend',
        MarketRegime.DOWNTREND: 'momentum',
        MarketRegime.SIDEWAYS: 'meanReversion',
        MarketRegime.HIGH_VOLATILITY: 'breakout',
    }

    for regime, expectedStrategy in hypotheses.items():
        regimeResults = [r for r in results if r.regime == regime and r.totalSignals >= 5]
        if regimeResults:
            best = max(regimeResults, key=lambda x: x.profitFactor)
            expectedResult = next((r for r in regimeResults if r.strategy == expectedStrategy), None)

            if expectedResult and expectedResult.strategy == best.strategy:
                print(f"{regime.value}: 가설 확인 ✓ ({expectedStrategy}가 최적)")
            elif expectedResult:
                print(f"{regime.value}: 가설 기각 ✗ (예상: {expectedStrategy}, 실제: {best.strategy})")
            else:
                print(f"{regime.value}: 데이터 부족")

    return results


if __name__ == '__main__':
    results = runExperiment()
