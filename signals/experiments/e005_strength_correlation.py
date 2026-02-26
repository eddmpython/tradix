"""
==============================================================================
실험 ID: E005
실험명: 신호 강도와 수익률 상관관계
==============================================================================

목적:
- SignalResult.strength 값이 실제 수익률과 상관있는지 검증
- 강한 신호만 사용하는 전략의 유효성 검증
- strength 기반 필터링의 효과 측정

방법:
1. strength 구간별 그룹화 (0.3-0.5, 0.5-0.7, 0.7-0.9, 0.9+)
2. 각 구간별 평균 수익률 및 승률 측정
3. strength와 수익률 간 상관계수 계산

기대 결과:
- strength ↑ → 수익률 ↑ (양의 상관관계)

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
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradex.signals import SignalPredictor


@dataclass
class StrengthBucket:
    """강도 구간별 결과"""
    rangeMin: float
    rangeMax: float
    count: int
    avgReturn: float
    winRate: float
    avgStrength: float


def generateTestData(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """테스트 데이터 생성"""
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
            drift = 0.0003
        elif i < 400:
            drift = 0.0012
        else:
            drift = -0.0005

        vol = 0.012 if 150 < i < 180 or 350 < i < 380 else 0.008
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


def collectStrengthData(df: pd.DataFrame) -> List[Tuple[float, float, int]]:
    """strength와 수익률 데이터 수집"""
    data = []

    for i in range(60, len(df) - 15):
        subDf = df.iloc[:i+1]

        predictor = SignalPredictor(subDf)
        result = predictor.predict(strategies=['trend', 'momentum'], consensusThreshold=0.3)

        if result.signal != 0:
            futureReturn = (df['close'].iloc[i + 10] / df['close'].iloc[i]) - 1

            if result.signal == 1:
                data.append((result.strength, futureReturn, result.signal))
            else:
                data.append((result.strength, -futureReturn, result.signal))

    return data


def analyzeByStrengthBucket(data: List[Tuple[float, float, int]]) -> List[StrengthBucket]:
    """강도 구간별 분석"""
    buckets = [
        (0.0, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 1.0),
    ]

    results = []

    for minVal, maxVal in buckets:
        bucketData = [(s, r) for s, r, _ in data if minVal <= s < maxVal]

        if not bucketData:
            results.append(StrengthBucket(
                rangeMin=minVal,
                rangeMax=maxVal,
                count=0,
                avgReturn=0.0,
                winRate=0.0,
                avgStrength=0.0,
            ))
            continue

        strengths = [s for s, _ in bucketData]
        returns = [r for _, r in bucketData]

        results.append(StrengthBucket(
            rangeMin=minVal,
            rangeMax=maxVal,
            count=len(bucketData),
            avgReturn=np.mean(returns),
            winRate=len([r for r in returns if r > 0]) / len(returns),
            avgStrength=np.mean(strengths),
        ))

    return results


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E005: 신호 강도와 수익률 상관관계")
    print("=" * 70)
    print()
    print("기대: strength ↑ → 수익률 ↑ (양의 상관관계)")
    print()

    print("테스트 데이터 생성 중...")
    df = generateTestData(500, seed=42)
    print(f"데이터 크기: {len(df)}일")

    print("\nstrength 데이터 수집 중...")
    data = collectStrengthData(df)
    print(f"수집된 신호: {len(data)}개")

    if len(data) < 10:
        print("신호 데이터 부족!")
        return None

    print("\n" + "-" * 70)
    print("강도 구간별 분석")
    print("-" * 70)

    buckets = analyzeByStrengthBucket(data)

    print("\n{:<15} {:>8} {:>10} {:>10} {:>12}".format(
        "Strength 구간", "신호수", "평균수익", "승률", "평균강도"
    ))
    print("-" * 60)

    for b in buckets:
        if b.count > 0:
            print("{:.1f} - {:.1f}       {:>8} {:>9.2%} {:>9.1%} {:>12.3f}".format(
                b.rangeMin,
                b.rangeMax,
                b.count,
                b.avgReturn,
                b.winRate,
                b.avgStrength,
            ))

    print("\n" + "-" * 70)
    print("상관관계 분석")
    print("-" * 70)

    strengths = [s for s, _, _ in data]
    returns = [r for _, r, _ in data]

    correlation, pValue = stats.pearsonr(strengths, returns)
    spearmanCorr, spearmanP = stats.spearmanr(strengths, returns)

    print(f"\nPearson 상관계수:  {correlation:+.4f} (p-value: {pValue:.4f})")
    print(f"Spearman 상관계수: {spearmanCorr:+.4f} (p-value: {spearmanP:.4f})")

    if pValue < 0.05:
        if correlation > 0:
            print("\n결론: strength와 수익률 간 유의미한 양의 상관관계 ✓")
        else:
            print("\n결론: strength와 수익률 간 유의미한 음의 상관관계 ✗")
    else:
        print("\n결론: strength와 수익률 간 통계적으로 유의미한 상관관계 없음")

    print("\n" + "-" * 70)
    print("강도 필터링 효과")
    print("-" * 70)

    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7]

    print("\n{:<20} {:>8} {:>10} {:>10}".format(
        "필터 조건", "신호수", "평균수익", "승률"
    ))
    print("-" * 50)

    for threshold in thresholds:
        filtered = [(s, r) for s, r, _ in data if s >= threshold]
        if filtered:
            avgReturn = np.mean([r for _, r in filtered])
            winRate = len([r for _, r in filtered if r > 0]) / len(filtered)
            print("strength >= {:.1f}    {:>8} {:>9.2%} {:>9.1%}".format(
                threshold,
                len(filtered),
                avgReturn,
                winRate,
            ))

    print("\n" + "-" * 70)
    print("권장 사항")
    print("-" * 70)

    highStrength = [(s, r) for s, r, _ in data if s >= 0.6]
    lowStrength = [(s, r) for s, r, _ in data if s < 0.6]

    if highStrength and lowStrength:
        highAvg = np.mean([r for _, r in highStrength])
        lowAvg = np.mean([r for _, r in lowStrength])

        if highAvg > lowAvg:
            print(f"\n고강도(≥0.6) 평균수익: {highAvg:.2%}")
            print(f"저강도(<0.6) 평균수익: {lowAvg:.2%}")
            print(f"차이: {highAvg - lowAvg:+.2%}")
            print("\n→ strength 0.6 이상 필터링 권장")
        else:
            print(f"\n고강도(≥0.6) 평균수익: {highAvg:.2%}")
            print(f"저강도(<0.6) 평균수익: {lowAvg:.2%}")
            print("\n→ strength 필터링 효과 미미")

    return {
        'data': data,
        'buckets': buckets,
        'correlation': correlation,
        'pValue': pValue,
    }


if __name__ == '__main__':
    results = runExperiment()
