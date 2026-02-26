"""
==============================================================================
실험 ID: E003
실험명: 예측 통합 효과 검증
==============================================================================

목적:
- forecast 전략(시계열 예측) 추가 시 신호 품질 변화 측정
- 예측은 "참고치"로만 활용, 신호 보조 역할 검증
- SignalBacktester로 벤치마크 대비 성과 평가

방법:
1. forecast 제외 vs 포함 비교
2. 백테스트로 Buy&Hold, Random 대비 성과 측정
3. 다양한 시장 상황(상승/하락/횡보)에서 검증

핵심 원칙:
- 예측은 절대 주 신호가 아님
- 기술 지표 신호를 보조하는 참고치

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
from typing import Dict, List
from dataclasses import dataclass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradex.signals import SignalPredictor
from tradex.signals.backtest import SignalBacktester, quickEvaluate


@dataclass
class ForecastIntegrationResult:
    """예측 통합 실험 결과"""
    scenarioName: str
    withForecast: Dict
    withoutForecast: Dict
    improvement: float
    forecastHelped: bool


def generateMarketScenario(scenario: str, n: int = 400, seed: int = 42) -> pd.DataFrame:
    """시장 시나리오별 데이터 생성"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    close = np.zeros(n)
    close[0] = 50000

    for i in range(1, n):
        if scenario == 'uptrend':
            drift = 0.001 + np.sin(i / 50) * 0.0003
            vol = 0.008
        elif scenario == 'downtrend':
            drift = -0.0008 - np.sin(i / 50) * 0.0002
            vol = 0.01
        elif scenario == 'sideways':
            drift = np.sin(i / 30) * 0.0005
            vol = 0.006
        elif scenario == 'volatile':
            drift = np.sin(i / 20) * 0.001
            vol = 0.02
        else:  # mixed
            if i < 100:
                drift = 0.001
            elif i < 200:
                drift = -0.0008
            elif i < 300:
                drift = 0.0002
            else:
                drift = 0.0012
            vol = 0.01

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


def evaluateForecastEffect(df: pd.DataFrame, scenarioName: str) -> ForecastIntegrationResult:
    """forecast 전략 효과 평가"""

    # forecast 제외
    resultWithout = quickEvaluate(
        df,
        strategies=['trend', 'momentum'],
        threshold=0.5
    )

    # forecast 포함
    resultWith = quickEvaluate(
        df,
        strategies=['trend', 'momentum', 'forecast'],
        threshold=0.5
    )

    withoutMetrics = {
        'totalReturn': resultWithout.signal.totalReturn,
        'sharpeRatio': resultWithout.signal.sharpeRatio,
        'winRate': resultWithout.signal.winRate,
        'trades': resultWithout.signal.totalTrades,
        'vsBuyHold': resultWithout.comparison.vsBuyHold,
        'betterThanBH': resultWithout.comparison.betterThanBuyHold,
    }

    withMetrics = {
        'totalReturn': resultWith.signal.totalReturn,
        'sharpeRatio': resultWith.signal.sharpeRatio,
        'winRate': resultWith.signal.winRate,
        'trades': resultWith.signal.totalTrades,
        'vsBuyHold': resultWith.comparison.vsBuyHold,
        'betterThanBH': resultWith.comparison.betterThanBuyHold,
    }

    improvement = withMetrics['totalReturn'] - withoutMetrics['totalReturn']
    forecastHelped = improvement > 0

    return ForecastIntegrationResult(
        scenarioName=scenarioName,
        withForecast=withMetrics,
        withoutForecast=withoutMetrics,
        improvement=improvement,
        forecastHelped=forecastHelped,
    )


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E003: 예측 통합 효과 검증")
    print("=" * 70)
    print()
    print("핵심 원칙: 예측은 참고치, 신호가 주 기능")
    print()

    scenarios = ['uptrend', 'downtrend', 'sideways', 'volatile', 'mixed']
    results = []

    print("-" * 70)
    print("시나리오별 forecast 전략 효과 평가")
    print("-" * 70)

    for scenario in scenarios:
        print(f"\n{scenario} 시나리오 평가 중...")
        df = generateMarketScenario(scenario, n=400, seed=42)
        result = evaluateForecastEffect(df, scenario)
        results.append(result)

        marker = "✓" if result.forecastHelped else "✗"
        print(f"  forecast 효과: {result.improvement:+.2%} {marker}")

    print("\n" + "=" * 70)
    print("실험 결과")
    print("=" * 70)

    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "시나리오", "제외수익", "포함수익", "개선", "포함승률", "효과"
    ))
    print("-" * 65)

    for r in results:
        marker = "✓" if r.forecastHelped else "✗"
        print("{:<12} {:>9.1%} {:>9.1%} {:>9.1%} {:>9.1%} {:>8}".format(
            r.scenarioName,
            r.withoutForecast['totalReturn'],
            r.withForecast['totalReturn'],
            r.improvement,
            r.withForecast['winRate'],
            marker,
        ))

    print("\n" + "-" * 70)
    print("분석")
    print("-" * 70)

    helpedCount = sum(1 for r in results if r.forecastHelped)
    avgImprovement = np.mean([r.improvement for r in results])

    print(f"forecast 도움된 시나리오: {helpedCount}/{len(results)}")
    print(f"평균 개선: {avgImprovement:+.2%}")

    if avgImprovement > 0:
        print("\n결론: forecast 전략이 평균적으로 도움됨 (참고치 역할 유효)")
    else:
        print("\n결론: forecast 전략이 현재 설정에서 큰 효과 없음")
        print("       → 예측은 참고치일 뿐, 신호 자체가 중요")

    print("\n" + "-" * 70)
    print("Buy&Hold 대비 성과")
    print("-" * 70)

    for r in results:
        withBH = "✓" if r.withForecast['betterThanBH'] else "✗"
        withoutBH = "✓" if r.withoutForecast['betterThanBH'] else "✗"
        print(f"{r.scenarioName:<12}: 제외 vs B&H {r.withoutForecast['vsBuyHold']:+.1%} {withoutBH}  |  포함 vs B&H {r.withForecast['vsBuyHold']:+.1%} {withBH}")

    return results


if __name__ == '__main__':
    results = runExperiment()
