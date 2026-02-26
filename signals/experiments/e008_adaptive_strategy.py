"""
==============================================================================
실험 ID: E008
실험명: 적응형 전략 선택기 효과 검증
==============================================================================

목적:
- AdaptiveSignalPredictor가 고정 전략보다 나은지 검증
- 레짐 감지 정확도 확인
- 종목별 최적 전략 vs 적응형 전략 비교

방법:
1. 고정 전략 (trend+momentum) vs 적응형 전략 비교
2. 다양한 종목에서 B&H 대비 성과 측정
3. 레짐 감지 정확도 분석

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

from tradex.signals import SignalPredictor
from tradex.signals.adaptive import AdaptiveSignalPredictor, MarketRegime
from tradex.signals.backtest import SignalBacktester


@dataclass
class AdaptiveTestResult:
    """적응형 테스트 결과"""
    symbol: str
    name: str
    adaptiveReturn: float
    fixedReturn: float
    buyHoldReturn: float
    adaptiveVsBH: float
    fixedVsBH: float
    adaptiveBetter: bool
    regimeCounts: Dict


def loadStockData(symbol: str, startDate: str, endDate: str) -> Optional[pd.DataFrame]:
    """주식 데이터 로드"""
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(symbol, startDate, endDate)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        return df if len(df) > 100 else None
    except Exception:
        return None


def backtestAdaptive(df: pd.DataFrame, holdingDays: int = 10) -> tuple:
    """적응형 전략 백테스트"""
    trades = []
    regimeCounts = {r.value: 0 for r in MarketRegime}

    capital = 10000000
    position = 0
    entryPrice = 0
    entryIdx = 0

    for i in range(60, len(df) - holdingDays):
        subDf = df.iloc[:i+1]

        predictor = AdaptiveSignalPredictor(subDf)
        result = predictor.predict(strengthFilter=0.6)

        regimeCounts[result.regime.value] += 1

        if position == 0 and result.signal == 1:
            entryPrice = df['close'].iloc[i] * 1.0005
            shares = int(capital * 0.95 / entryPrice)
            if shares > 0:
                capital -= shares * entryPrice * 1.00015
                position = shares
                entryIdx = i

        elif position > 0 and (result.signal == -1 or i - entryIdx >= holdingDays):
            exitPrice = df['close'].iloc[i] * 0.9995
            capital += position * exitPrice * 0.99985

            pnl = (exitPrice - entryPrice) / entryPrice
            trades.append(pnl)

            position = 0
            entryPrice = 0

    if position > 0:
        capital += position * df['close'].iloc[-1] * 0.99985

    totalReturn = (capital - 10000000) / 10000000
    return totalReturn, trades, regimeCounts


def backtestFixed(df: pd.DataFrame, strategies: List[str], threshold: float, holdingDays: int = 10) -> tuple:
    """고정 전략 백테스트"""
    trades = []

    capital = 10000000
    position = 0
    entryPrice = 0
    entryIdx = 0

    for i in range(60, len(df) - holdingDays):
        subDf = df.iloc[:i+1]

        predictor = SignalPredictor(subDf)
        result = predictor.predict(strategies=strategies, consensusThreshold=threshold)

        if result.strength < 0.6:
            signal = 0
        else:
            signal = result.signal

        if position == 0 and signal == 1:
            entryPrice = df['close'].iloc[i] * 1.0005
            shares = int(capital * 0.95 / entryPrice)
            if shares > 0:
                capital -= shares * entryPrice * 1.00015
                position = shares
                entryIdx = i

        elif position > 0 and (signal == -1 or i - entryIdx >= holdingDays):
            exitPrice = df['close'].iloc[i] * 0.9995
            capital += position * exitPrice * 0.99985

            pnl = (exitPrice - entryPrice) / entryPrice
            trades.append(pnl)

            position = 0
            entryPrice = 0

    if position > 0:
        capital += position * df['close'].iloc[-1] * 0.99985

    totalReturn = (capital - 10000000) / 10000000
    return totalReturn, trades


def runExperiment():
    """실험 실행"""
    print("=" * 70)
    print("E008: 적응형 전략 선택기 효과 검증")
    print("=" * 70)
    print()

    stocks = [
        ('005930', '삼성전자'),
        ('000660', 'SK하이닉스'),
        ('035420', 'NAVER'),
        ('035720', '카카오'),
        ('051910', 'LG화학'),
        ('006400', '삼성SDI'),
        ('247540', '에코프로비엠'),
        ('069500', 'KODEX 200'),
        ('305720', 'KODEX 2차전지'),
    ]

    startDate = '2022-01-01'
    endDate = '2024-12-31'

    print(f"테스트 기간: {startDate} ~ {endDate}")
    print()

    print("-" * 70)
    print("데이터 로드")
    print("-" * 70)

    stockData = {}
    for symbol, name in stocks:
        df = loadStockData(symbol, startDate, endDate)
        if df is not None:
            stockData[symbol] = (name, df)
            print(f"  ✓ {name}({symbol}): {len(df)}일")
        else:
            print(f"  ✗ {name}({symbol}): 로드 실패")

    if not stockData:
        print("로드된 데이터 없음!")
        return None

    print("\n" + "-" * 70)
    print("적응형 vs 고정 전략 비교")
    print("-" * 70)

    results = []

    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "종목", "적응형", "고정", "B&H", "적응vsBH", "고정vsBH"
    ))
    print("-" * 65)

    for symbol, (name, df) in stockData.items():
        adaptiveReturn, adaptiveTrades, regimeCounts = backtestAdaptive(df)
        fixedReturn, fixedTrades = backtestFixed(df, ['trend', 'momentum'], 0.5)

        buyHoldReturn = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1

        adaptiveVsBH = adaptiveReturn - buyHoldReturn
        fixedVsBH = fixedReturn - buyHoldReturn

        adaptiveBetter = adaptiveVsBH > fixedVsBH

        results.append(AdaptiveTestResult(
            symbol=symbol,
            name=name,
            adaptiveReturn=adaptiveReturn,
            fixedReturn=fixedReturn,
            buyHoldReturn=buyHoldReturn,
            adaptiveVsBH=adaptiveVsBH,
            fixedVsBH=fixedVsBH,
            adaptiveBetter=adaptiveBetter,
            regimeCounts=regimeCounts,
        ))

        marker = "✓" if adaptiveBetter else "✗"
        print("{:<12} {:>9.1%} {:>9.1%} {:>9.1%} {:>9.1%} {:>9.1%} {}".format(
            name[:10],
            adaptiveReturn,
            fixedReturn,
            buyHoldReturn,
            adaptiveVsBH,
            fixedVsBH,
            marker,
        ))

    print("\n" + "=" * 70)
    print("결과 분석")
    print("=" * 70)

    adaptiveWins = sum(1 for r in results if r.adaptiveBetter)
    adaptiveBeatsBH = sum(1 for r in results if r.adaptiveVsBH > 0)
    fixedBeatsBH = sum(1 for r in results if r.fixedVsBH > 0)

    avgAdaptiveVsBH = np.mean([r.adaptiveVsBH for r in results])
    avgFixedVsBH = np.mean([r.fixedVsBH for r in results])

    print(f"\n적응형 > 고정: {adaptiveWins}/{len(results)} ({adaptiveWins/len(results)*100:.1f}%)")
    print(f"적응형 > B&H: {adaptiveBeatsBH}/{len(results)} ({adaptiveBeatsBH/len(results)*100:.1f}%)")
    print(f"고정 > B&H: {fixedBeatsBH}/{len(results)} ({fixedBeatsBH/len(results)*100:.1f}%)")

    print(f"\n평균 적응형 vs B&H: {avgAdaptiveVsBH:+.1%}")
    print(f"평균 고정 vs B&H: {avgFixedVsBH:+.1%}")
    print(f"적응형 개선: {avgAdaptiveVsBH - avgFixedVsBH:+.1%}")

    print("\n" + "-" * 70)
    print("레짐 감지 분포")
    print("-" * 70)

    totalRegimes = {}
    for r in results:
        for regime, count in r.regimeCounts.items():
            totalRegimes[regime] = totalRegimes.get(regime, 0) + count

    totalCount = sum(totalRegimes.values())
    for regime, count in sorted(totalRegimes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {count}회 ({count/totalCount*100:.1f}%)")

    print("\n" + "-" * 70)
    print("결론")
    print("-" * 70)

    if avgAdaptiveVsBH > avgFixedVsBH:
        print(f"\n✓ 적응형 전략이 고정 전략보다 {avgAdaptiveVsBH - avgFixedVsBH:+.1%} 우수")
    else:
        print(f"\n✗ 고정 전략이 적응형 전략보다 {avgFixedVsBH - avgAdaptiveVsBH:+.1%} 우수")

    if adaptiveBeatsBH / len(results) > 0.5:
        print(f"✓ 적응형 전략이 {adaptiveBeatsBH}/{len(results)} 종목에서 B&H 초과")
    else:
        print(f"✗ 적응형 전략이 대부분 B&H 미달")

    return results


if __name__ == '__main__':
    results = runExperiment()
