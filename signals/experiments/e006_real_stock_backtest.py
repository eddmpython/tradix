"""
==============================================================================
실험 ID: E006
실험명: 실제 주가 데이터 백테스트
==============================================================================

목적:
- 한국 주식 실데이터로 전체 시스템 검증
- 실제 트레이딩 가능성 평가
- SignalBacktester로 벤치마크 대비 성과 측정

대상 종목:
- KOSPI: 삼성전자(005930), SK하이닉스(000660)
- ETF: KODEX 200(069500)

테스트 기간:
- 2023-01-01 ~ 2024-12-31 (2년)

측정 지표:
- 연간 수익률
- 최대 낙폭 (MDD)
- 샤프 비율
- 거래 횟수
- Buy&Hold 대비 성과

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
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradix.signals import SignalPredictor
from tradix.signals.backtest import SignalBacktester, quickEvaluate


@dataclass
class StockResult:
    """종목별 결과"""
    symbol: str
    name: str
    signalReturn: float
    buyHoldReturn: float
    vsBuyHold: float
    sharpeRatio: float
    maxDrawdown: float
    winRate: float
    totalTrades: int
    better: bool


def loadStockData(symbol: str, startDate: str, endDate: str) -> Optional[pd.DataFrame]:
    """주식 데이터 로드"""
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(symbol, startDate, endDate)

        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })

        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                print(f"  경고: {col} 컬럼 없음")
                return None

        df = df[required]
        df = df.dropna()

        return df

    except Exception as e:
        print(f"  데이터 로드 실패: {e}")
        return None


def evaluateStock(
    symbol: str,
    name: str,
    df: pd.DataFrame,
    strategies: List[str],
    threshold: float,
) -> StockResult:
    """종목 평가"""
    result = quickEvaluate(df, strategies=strategies, threshold=threshold)

    return StockResult(
        symbol=symbol,
        name=name,
        signalReturn=result.signal.totalReturn,
        buyHoldReturn=result.buyHold.totalReturn,
        vsBuyHold=result.comparison.vsBuyHold,
        sharpeRatio=result.signal.sharpeRatio,
        maxDrawdown=result.signal.maxDrawdown,
        winRate=result.signal.winRate,
        totalTrades=result.signal.totalTrades,
        better=result.comparison.betterThanBuyHold,
    )


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E006: 실제 주가 데이터 백테스트")
    print("=" * 70)
    print()

    stocks = [
        ('005930', '삼성전자'),
        ('000660', 'SK하이닉스'),
        ('069500', 'KODEX 200'),
        ('035720', '카카오'),
        ('035420', 'NAVER'),
    ]

    startDate = '2023-01-01'
    endDate = '2024-12-31'

    print(f"테스트 기간: {startDate} ~ {endDate}")
    print()

    print("-" * 70)
    print("데이터 로드")
    print("-" * 70)

    stockData = {}
    for symbol, name in stocks:
        print(f"\n{name}({symbol}) 로드 중...")
        df = loadStockData(symbol, startDate, endDate)
        if df is not None:
            stockData[symbol] = (name, df)
            print(f"  로드 완료: {len(df)}일")
        else:
            print(f"  로드 실패")

    if not stockData:
        print("\n로드된 데이터가 없습니다!")
        print("FinanceDataReader가 설치되어 있는지 확인하세요.")
        print("  pip install finance-datareader")
        return None

    print("\n" + "-" * 70)
    print("전략별 백테스트")
    print("-" * 70)

    strategyConfigs = [
        (['trend', 'momentum'], 0.5, 'trend+momentum@50%'),
        (['trend'], 0.5, 'trend@50%'),
        (None, 0.4, 'all@40%'),
    ]

    allResults = []

    for strategies, threshold, strategyName in strategyConfigs:
        print(f"\n### {strategyName} ###")
        print("{:<12} {:>10} {:>10} {:>10} {:>8} {:>8}".format(
            "종목", "신호수익", "B&H수익", "vs B&H", "승률", "거래수"
        ))
        print("-" * 65)

        for symbol, (name, df) in stockData.items():
            result = evaluateStock(symbol, name, df, strategies, threshold)
            allResults.append((strategyName, result))

            marker = "✓" if result.better else "✗"
            print("{:<12} {:>9.1%} {:>9.1%} {:>9.1%} {:>7.1%} {:>8} {}".format(
                name,
                result.signalReturn,
                result.buyHoldReturn,
                result.vsBuyHold,
                result.winRate,
                result.totalTrades,
                marker,
            ))

    print("\n" + "=" * 70)
    print("종합 결과")
    print("=" * 70)

    for strategyName in [s[2] for s in strategyConfigs]:
        strategyResults = [r for sn, r in allResults if sn == strategyName]
        if strategyResults:
            avgVsBH = np.mean([r.vsBuyHold for r in strategyResults])
            winCount = sum(1 for r in strategyResults if r.better)
            avgWinRate = np.mean([r.winRate for r in strategyResults])

            print(f"\n{strategyName}:")
            print(f"  평균 vs B&H: {avgVsBH:+.1%}")
            print(f"  B&H 초과 종목: {winCount}/{len(strategyResults)}")
            print(f"  평균 승률: {avgWinRate:.1%}")

    print("\n" + "-" * 70)
    print("결론")
    print("-" * 70)

    bestStrategy = None
    bestAvgVsBH = float('-inf')

    for strategyName in [s[2] for s in strategyConfigs]:
        strategyResults = [r for sn, r in allResults if sn == strategyName]
        if strategyResults:
            avgVsBH = np.mean([r.vsBuyHold for r in strategyResults])
            if avgVsBH > bestAvgVsBH:
                bestAvgVsBH = avgVsBH
                bestStrategy = strategyName

    if bestStrategy:
        print(f"\n최적 전략: {bestStrategy} (평균 vs B&H: {bestAvgVsBH:+.1%})")

        if bestAvgVsBH > 0:
            print("\n→ 신호 전략이 Buy&Hold를 평균적으로 초과!")
        else:
            print("\n→ 신호 전략이 Buy&Hold를 초과하지 못함")
            print("   주식은 예측 불가능 - 시그널은 참고용")

    return allResults


if __name__ == '__main__':
    results = runExperiment()
