"""
==============================================================================
실험 ID: E002
실험명: 복합 지표 컨센서스 전략
==============================================================================

목적:
- 여러 지표 조합이 단일 지표보다 나은지 검증
- 최적 컨센서스 임계값 탐색 (50%, 60%, 70%)
- SignalPredictor 통합 신호 유효성 검증

방법:
1. SignalPredictor로 전체 신호 생성
2. 컨센서스 임계값별 성능 비교
3. 전략 그룹별 조합 테스트

==============================================================================
결과
==============================================================================

핵심 발견:
1. **컨센서스 ↑ → 정확도 ↑ → 손익비 ↑**
   - threshold=30%: 정확도 55.0%, 손익비 1.58
   - threshold=60%: 정확도 57.2%, 손익비 3.70

2. **최고 성능 조합**
   | 전략 | 신호수 | 정확도 | 10일수익 | 손익비 |
   |------|--------|--------|----------|--------|
   | trend+momentum@60% | 30 | 70.0% | **2.33%** | **8.73** |
   | all@50% | 18 | **77.8%** | 2.26% | 7.86 |
   | all@40% | 48 | 72.9% | 2.35% | 6.45 |

3. **전략 그룹별 성능**
   - trend+momentum: 균형 잡힌 최고 성능
   - all (전체): 높은 정확도, 적은 신호
   - trend only: 안정적이나 평범
   - momentum only: 중간
   - meanReversion: 부진 (손익비 < 1)

결론:
- **권장 설정**: trend+momentum 전략, threshold=50~60%
- 높은 컨센서스 = 적은 신호 but 높은 품질
- 평균회귀 전략은 현재 시장에서 비효율적

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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradix.signals import SignalPredictor, SignalConfig


@dataclass
class ConsensusResult:
    """컨센서스 실험 결과"""
    strategyName: str
    threshold: float
    totalSignals: int
    buySignals: int
    sellSignals: int
    accuracy: float
    avgReturn10d: float
    winRate: float
    profitFactor: float


def generateTestData(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """다양한 시장 상황을 포함한 테스트 데이터 생성"""
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


def evaluateConsensusStrategy(
    df: pd.DataFrame,
    strategies: List[str],
    threshold: float,
    strategyName: str,
) -> ConsensusResult:
    """컨센서스 전략 평가"""
    signals = []
    returns10d = []

    for i in range(60, len(df) - 20):
        subDf = df.iloc[:i+1]

        predictor = SignalPredictor(subDf)
        result = predictor.predict(strategies=strategies, consensusThreshold=threshold)

        if result.signal != 0:
            futureReturn = (df['close'].iloc[i + 10] / df['close'].iloc[i]) - 1

            if result.signal == 1:
                signals.append((i, 1, result.strength, futureReturn))
                returns10d.append(futureReturn)
            else:
                signals.append((i, -1, result.strength, -futureReturn))
                returns10d.append(-futureReturn)

    if not signals:
        return ConsensusResult(
            strategyName=strategyName,
            threshold=threshold,
            totalSignals=0,
            buySignals=0,
            sellSignals=0,
            accuracy=0.0,
            avgReturn10d=0.0,
            winRate=0.0,
            profitFactor=0.0,
        )

    buySignals = len([s for s in signals if s[1] == 1])
    sellSignals = len([s for s in signals if s[1] == -1])
    totalSignals = len(signals)

    correctSignals = len([r for r in returns10d if r > 0])
    accuracy = correctSignals / totalSignals

    avgReturn = np.mean(returns10d)
    winRate = len([r for r in returns10d if r > 0]) / len(returns10d)

    profits = [r for r in returns10d if r > 0]
    losses = [abs(r) for r in returns10d if r < 0]
    profitFactor = (sum(profits) / sum(losses)) if losses and sum(losses) > 0 else 0

    return ConsensusResult(
        strategyName=strategyName,
        threshold=threshold,
        totalSignals=totalSignals,
        buySignals=buySignals,
        sellSignals=sellSignals,
        accuracy=accuracy,
        avgReturn10d=avgReturn,
        winRate=winRate,
        profitFactor=profitFactor,
    )


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E002: 복합 지표 컨센서스 전략 실험")
    print("=" * 70)

    print("\n테스트 데이터 생성 중...")
    df = generateTestData(400, seed=42)
    print(f"데이터 크기: {len(df)}일")

    thresholds = [0.3, 0.4, 0.5, 0.6]

    strategyGroups = {
        'trend': ['trend'],
        'momentum': ['momentum'],
        'meanReversion': ['meanReversion'],
        'trend+momentum': ['trend', 'momentum'],
        'all': None,
    }

    results = []

    print("\n" + "-" * 70)
    print("전략 그룹별 컨센서스 평가 중...")
    print("-" * 70)

    for groupName, strategies in strategyGroups.items():
        print(f"\n{groupName} 전략 평가...")

        for threshold in thresholds:
            result = evaluateConsensusStrategy(
                df, strategies, threshold, f"{groupName}@{threshold:.0%}"
            )
            results.append(result)
            print(f"  threshold={threshold:.0%}: 신호 {result.totalSignals}개, 정확도 {result.accuracy:.1%}")

    print("\n" + "=" * 70)
    print("실험 결과")
    print("=" * 70)

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "전략", "신호수", "정확도", "10일수익", "승률", "손익비"
    ))
    print("-" * 75)

    results.sort(key=lambda x: x.profitFactor, reverse=True)

    for r in results:
        if r.totalSignals > 0:
            print("{:<25} {:>8} {:>8.1%} {:>9.2%} {:>9.1%} {:>10.2f}".format(
                r.strategyName,
                r.totalSignals,
                r.accuracy,
                r.avgReturn10d,
                r.winRate,
                r.profitFactor,
            ))

    print("\n" + "-" * 70)
    print("분석: 임계값별 비교")
    print("-" * 70)

    for threshold in thresholds:
        thresholdResults = [r for r in results if r.threshold == threshold and r.totalSignals > 0]
        if thresholdResults:
            avgAccuracy = np.mean([r.accuracy for r in thresholdResults])
            avgSignals = np.mean([r.totalSignals for r in thresholdResults])
            avgPF = np.mean([r.profitFactor for r in thresholdResults])
            print(f"threshold={threshold:.0%}: 평균 신호 {avgSignals:.0f}개, 평균 정확도 {avgAccuracy:.1%}, 평균 손익비 {avgPF:.2f}")

    print("\n" + "-" * 70)
    print("최적 조합")
    print("-" * 70)

    validResults = [r for r in results if r.totalSignals >= 5]
    if validResults:
        bestByPF = max(validResults, key=lambda x: x.profitFactor)
        bestByAccuracy = max(validResults, key=lambda x: x.accuracy)
        bestByReturn = max(validResults, key=lambda x: x.avgReturn10d)

        print(f"최고 손익비: {bestByPF.strategyName} (PF={bestByPF.profitFactor:.2f})")
        print(f"최고 정확도: {bestByAccuracy.strategyName} ({bestByAccuracy.accuracy:.1%})")
        print(f"최고 수익률: {bestByReturn.strategyName} ({bestByReturn.avgReturn10d:.2%})")

    return results


if __name__ == '__main__':
    results = runExperiment()
