"""
==============================================================================
실험 ID: E007
실험명: 확장 종목 백테스트 - Buy&Hold 초과 가능성 탐색
==============================================================================

목적:
- 다양한 종목/섹터에서 Buy&Hold를 초과할 수 있는지 검증
- 어떤 종목 유형에서 신호 전략이 유효한지 파악
- 최적 전략 조합 탐색

대상:
- KOSPI 대형주 (10개)
- KOSDAQ (5개)
- ETF (5개)
- 미국 주식 (5개) - yfinance

테스트 기간: 2022-01-01 ~ 2024-12-31 (3년)

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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from tradex.signals import SignalPredictor
from tradex.signals.backtest import SignalBacktester, quickEvaluate


@dataclass
class ExtendedResult:
    """확장 결과"""
    symbol: str
    name: str
    category: str
    signalReturn: float
    buyHoldReturn: float
    vsBuyHold: float
    sharpeRatio: float
    winRate: float
    totalTrades: int
    better: bool
    strategy: str


def loadKoreanStock(symbol: str, startDate: str, endDate: str) -> Optional[pd.DataFrame]:
    """한국 주식 데이터 로드"""
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(symbol, startDate, endDate)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        return df if len(df) > 100 else None
    except Exception as e:
        return None


def loadUSStock(symbol: str, startDate: str, endDate: str) -> Optional[pd.DataFrame]:
    """미국 주식 데이터 로드 (yfinance)"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=startDate, end=endDate)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        return df if len(df) > 100 else None
    except Exception as e:
        return None


def evaluateWithBestStrategy(
    symbol: str,
    name: str,
    category: str,
    df: pd.DataFrame,
) -> List[ExtendedResult]:
    """여러 전략으로 평가하고 최적 전략 찾기"""
    strategies = [
        (['trend'], 0.5, 'trend'),
        (['momentum'], 0.5, 'momentum'),
        (['meanReversion'], 0.5, 'meanReversion'),
        (['trend', 'momentum'], 0.5, 'trend+momentum'),
        (['trend', 'momentum'], 0.6, 'trend+momentum@60%'),
        (None, 0.4, 'all@40%'),
        (None, 0.5, 'all@50%'),
    ]

    results = []

    for strategyList, threshold, strategyName in strategies:
        try:
            result = quickEvaluate(df, strategies=strategyList, threshold=threshold)

            results.append(ExtendedResult(
                symbol=symbol,
                name=name,
                category=category,
                signalReturn=result.signal.totalReturn,
                buyHoldReturn=result.buyHold.totalReturn,
                vsBuyHold=result.comparison.vsBuyHold,
                sharpeRatio=result.signal.sharpeRatio,
                winRate=result.signal.winRate,
                totalTrades=result.signal.totalTrades,
                better=result.comparison.betterThanBuyHold,
                strategy=strategyName,
            ))
        except Exception:
            pass

    return results


def runExperiment():
    """실험 실행"""
    print("=" * 80)
    print("E007: 확장 종목 백테스트 - Buy&Hold 초과 가능성 탐색")
    print("=" * 80)
    print()

    # 종목 리스트
    koreanStocks = [
        # KOSPI 대형주
        ('005930', '삼성전자', 'KOSPI'),
        ('000660', 'SK하이닉스', 'KOSPI'),
        ('005380', '현대차', 'KOSPI'),
        ('035420', 'NAVER', 'KOSPI'),
        ('035720', '카카오', 'KOSPI'),
        ('051910', 'LG화학', 'KOSPI'),
        ('006400', '삼성SDI', 'KOSPI'),
        ('003670', '포스코퓨처엠', 'KOSPI'),
        ('105560', 'KB금융', 'KOSPI'),
        ('055550', '신한지주', 'KOSPI'),
        # KOSDAQ
        ('247540', '에코프로비엠', 'KOSDAQ'),
        ('086520', '에코프로', 'KOSDAQ'),
        ('028300', 'HLB', 'KOSDAQ'),
        ('293490', '카카오게임즈', 'KOSDAQ'),
        ('383220', 'F&F', 'KOSDAQ'),
        # ETF
        ('069500', 'KODEX 200', 'ETF'),
        ('229200', 'KODEX 코스닥150', 'ETF'),
        ('305720', 'KODEX 2차전지', 'ETF'),
        ('091160', 'KODEX 반도체', 'ETF'),
        ('251340', 'KODEX 미국S&P500', 'ETF'),
    ]

    usStocks = [
        ('AAPL', 'Apple', 'US'),
        ('MSFT', 'Microsoft', 'US'),
        ('GOOGL', 'Google', 'US'),
        ('NVDA', 'NVIDIA', 'US'),
        ('TSLA', 'Tesla', 'US'),
    ]

    startDate = '2022-01-01'
    endDate = '2024-12-31'

    print(f"테스트 기간: {startDate} ~ {endDate}")
    print()

    # 데이터 로드
    print("-" * 80)
    print("데이터 로드")
    print("-" * 80)

    allData = {}

    print("\n한국 주식 로드 중...")
    for symbol, name, category in koreanStocks:
        df = loadKoreanStock(symbol, startDate, endDate)
        if df is not None:
            allData[symbol] = (name, category, df)
            print(f"  ✓ {name}({symbol}): {len(df)}일")
        else:
            print(f"  ✗ {name}({symbol}): 로드 실패")

    print("\n미국 주식 로드 중...")
    for symbol, name, category in usStocks:
        df = loadUSStock(symbol, startDate, endDate)
        if df is not None:
            allData[symbol] = (name, category, df)
            print(f"  ✓ {name}({symbol}): {len(df)}일")
        else:
            print(f"  ✗ {name}({symbol}): 로드 실패")

    if not allData:
        print("\n로드된 데이터가 없습니다!")
        return None

    print(f"\n총 {len(allData)}개 종목 로드 완료")

    # 전략별 평가
    print("\n" + "-" * 80)
    print("전략별 평가 진행")
    print("-" * 80)

    allResults = []

    for symbol, (name, category, df) in allData.items():
        print(f"\n{name}({symbol}) 평가 중...")
        results = evaluateWithBestStrategy(symbol, name, category, df)
        allResults.extend(results)

        # 최적 전략 출력
        if results:
            bestResult = max(results, key=lambda x: x.vsBuyHold)
            marker = "✓" if bestResult.better else "✗"
            print(f"  최적: {bestResult.strategy} (vs B&H: {bestResult.vsBuyHold:+.1%}) {marker}")

    # 결과 분석
    print("\n" + "=" * 80)
    print("결과 분석")
    print("=" * 80)

    # 1. Buy&Hold를 이긴 종목/전략 조합
    print("\n### Buy&Hold 초과 성공 케이스 ###")
    print("{:<12} {:<15} {:<20} {:>10} {:>10}".format(
        "종목", "카테고리", "전략", "vs B&H", "승률"
    ))
    print("-" * 70)

    winners = [r for r in allResults if r.better]
    winners.sort(key=lambda x: x.vsBuyHold, reverse=True)

    for r in winners[:20]:  # 상위 20개
        print("{:<12} {:<15} {:<20} {:>9.1%} {:>9.1%}".format(
            r.name[:10],
            r.category,
            r.strategy,
            r.vsBuyHold,
            r.winRate,
        ))

    print(f"\n총 {len(winners)}/{len(allResults)} 케이스에서 B&H 초과 ({len(winners)/len(allResults)*100:.1f}%)")

    # 2. 종목별 최적 전략
    print("\n### 종목별 최적 전략 ###")
    print("{:<15} {:<10} {:<20} {:>10} {:>10} {:>10}".format(
        "종목", "카테고리", "최적전략", "신호수익", "B&H수익", "vs B&H"
    ))
    print("-" * 80)

    symbols = set(r.symbol for r in allResults)
    stockBests = []

    for symbol in symbols:
        symbolResults = [r for r in allResults if r.symbol == symbol]
        if symbolResults:
            best = max(symbolResults, key=lambda x: x.vsBuyHold)
            stockBests.append(best)

    stockBests.sort(key=lambda x: x.vsBuyHold, reverse=True)

    for r in stockBests:
        marker = "✓" if r.better else "✗"
        print("{:<15} {:<10} {:<20} {:>9.1%} {:>9.1%} {:>9.1%} {}".format(
            r.name[:13],
            r.category,
            r.strategy,
            r.signalReturn,
            r.buyHoldReturn,
            r.vsBuyHold,
            marker,
        ))

    # 3. 전략별 성공률
    print("\n### 전략별 B&H 초과 성공률 ###")
    print("{:<25} {:>10} {:>10} {:>12}".format(
        "전략", "성공", "실패", "성공률"
    ))
    print("-" * 60)

    strategies = set(r.strategy for r in allResults)
    strategyStats = []

    for strategy in strategies:
        strategyResults = [r for r in allResults if r.strategy == strategy]
        wins = sum(1 for r in strategyResults if r.better)
        total = len(strategyResults)
        avgVsBH = np.mean([r.vsBuyHold for r in strategyResults])
        strategyStats.append((strategy, wins, total - wins, wins/total if total > 0 else 0, avgVsBH))

    strategyStats.sort(key=lambda x: x[3], reverse=True)

    for strategy, wins, losses, rate, avgVsBH in strategyStats:
        print("{:<25} {:>10} {:>10} {:>11.1%}".format(
            strategy, wins, losses, rate
        ))

    # 4. 카테고리별 분석
    print("\n### 카테고리별 B&H 초과 성공률 ###")
    print("{:<15} {:>10} {:>10} {:>12} {:>12}".format(
        "카테고리", "성공", "실패", "성공률", "평균vsBH"
    ))
    print("-" * 65)

    categories = set(r.category for r in allResults)

    for category in categories:
        catResults = [r for r in allResults if r.category == category]
        wins = sum(1 for r in catResults if r.better)
        total = len(catResults)
        avgVsBH = np.mean([r.vsBuyHold for r in catResults])
        print("{:<15} {:>10} {:>10} {:>11.1%} {:>11.1%}".format(
            category, wins, total - wins, wins/total if total > 0 else 0, avgVsBH
        ))

    # 5. 핵심 결론
    print("\n" + "=" * 80)
    print("핵심 결론")
    print("=" * 80)

    totalWins = sum(1 for r in stockBests if r.better)
    totalStocks = len(stockBests)
    overallRate = totalWins / totalStocks if totalStocks > 0 else 0

    print(f"\n종목별 최적 전략 사용 시:")
    print(f"  - B&H 초과 종목: {totalWins}/{totalStocks} ({overallRate:.1%})")

    if overallRate > 0.5:
        print("\n→ 절반 이상의 종목에서 B&H 초과 가능!")
        print("  단, 종목별로 최적 전략이 다름 - 레짐 감지 필요")
    else:
        print("\n→ 대부분의 종목에서 B&H 초과 실패")
        print("  핵심 원칙 재확인: 주식은 예측 불가능, 시그널은 참고용")

    # B&H를 크게 이긴 케이스 분석
    bigWinners = [r for r in stockBests if r.vsBuyHold > 0.1]  # 10% 이상 초과
    if bigWinners:
        print(f"\n10% 이상 B&H 초과 종목: {len(bigWinners)}개")
        for r in bigWinners:
            print(f"  - {r.name}: {r.strategy} ({r.vsBuyHold:+.1%})")

    return allResults


if __name__ == '__main__':
    results = runExperiment()
