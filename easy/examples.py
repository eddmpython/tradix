"""
Tradix Easy API Examples Module - Demonstration of all Easy API usage levels.

Provides runnable example functions showcasing the Easy API at different
skill levels: Level 1 (preset one-liners for beginners), Level 2
(declarative strategy builder for intermediate users), and advanced
features like optimization and strategy comparison.

Easy API 예제 모듈 - 초급부터 고급까지 Easy API의 다양한 사용법을 보여줍니다.
실행: python -m tradix.easy.examples

Features:
    - example_level1_preset: One-liner backtesting with presets
    - example_level1_korean: Full Korean-language backtesting
    - example_level2_builder: Declarative strategy builder with chaining
    - example_level2_conditions: Combining multiple conditions with AND/OR
    - example_level2_korean_builder: Korean declarative builder
    - example_level2_lambda: Lambda-based condition definitions
    - example_presets_comparison: Side-by-side preset strategy comparison
    - example_optimization: Parameter optimization with grid search
    - example_comparison_old_vs_new: Traditional vs Easy API comparison
"""


def example_level1_preset():
    """
    Level 1 example: One-liner backtesting with presets (beginner). / 프리셋으로 1줄 백테스트 (초보자용).

    Demonstrates the simplest usage pattern -- running a backtest
    with a single function call using pre-built strategy presets.
    No coding experience required beyond basic Python.
    """
    print("\n" + "="*60)
    print("Level 1: 프리셋으로 1줄 백테스트")
    print("="*60)

    from tradix.easy import backtest, goldenCross, rsiOversold

    result = backtest("005930", goldenCross())
    print(result.summary())

    result2 = backtest("005930", rsiOversold())
    print(f"RSI 전략 수익률: {result2.totalReturn:+.2f}%")


def example_level1_korean():
    """
    Level 1 example: Full Korean-language backtesting. / 한글 API로 백테스트.

    Demonstrates backtesting entirely in Korean -- function names,
    parameters, and result properties are all in Korean. No English
    required for complete usage.
    """
    print("\n" + "="*60)
    print("Level 1: 한글 API")
    print("="*60)

    from tradix.easy import 백테스트, 골든크로스, RSI과매도

    결과 = 백테스트("삼성전자", 골든크로스())
    print(결과.요약())

    print(f"수익률: {결과.수익률:+.2f}%")
    print(f"최대낙폭: {결과.최대낙폭:.2f}%")
    print(f"샤프비율: {결과.샤프비율:.2f}")
    print(f"승률: {결과.승률:.1f}%")


def example_level2_builder():
    """
    Level 2 example: Declarative strategy builder with method chaining (intermediate). / 선언형 전략 빌더 (중급자용).

    Demonstrates building a custom strategy using QuickStrategy's
    fluent API with .buyWhen(), .sellWhen(), .stopLoss(), and
    .takeProfit() chained methods.
    """
    print("\n" + "="*60)
    print("Level 2: 선언형 전략 빌더")
    print("="*60)

    from tradix.easy import (
        backtest, QuickStrategy,
        sma, rsi, crossover, crossunder
    )

    strategy = (
        QuickStrategy("나만의 골든크로스")
        .buyWhen(crossover(sma(10), sma(30)))
        .sellWhen(crossunder(sma(10), sma(30)))
        .stopLoss(5)
        .takeProfit(15)
    )

    result = backtest("005930", strategy, period="3년")
    print(result.summary())
    print(f"전략: {strategy}")


def example_level2_conditions():
    """
    Level 2 example: Combining multiple conditions with AND/OR operators. / 복합 조건 조합.

    Demonstrates combining indicator conditions using the ``&`` (AND)
    and ``|`` (OR) operators to create multi-factor entry/exit rules.
    """
    print("\n" + "="*60)
    print("Level 2: 복합 조건")
    print("="*60)

    from tradix.easy import (
        backtest, QuickStrategy,
        sma, rsi, price, crossover
    )

    strategy = (
        QuickStrategy("RSI+SMA 복합전략")
        .buyWhen((rsi(14) < 30) & (price > sma(50)))
        .sellWhen(rsi(14) > 70)
        .stopLoss(5)
    )

    result = backtest("005930", strategy)
    print(result.summary())


def example_level2_korean_builder():
    """
    Level 2 example: Korean declarative strategy builder. / 한글 선언형 빌더.

    Demonstrates building a custom strategy entirely in Korean using
    Korean method aliases (매수조건, 매도조건, 손절, 익절) and Korean
    parameter names for the backtest function.
    """
    print("\n" + "="*60)
    print("Level 2: 한글 선언형 빌더")
    print("="*60)

    from tradix.easy import 백테스트, 전략, sma, crossover, crossunder

    내전략 = (
        전략("한글전략")
        .매수조건(crossover(sma(10), sma(30)))
        .매도조건(crossunder(sma(10), sma(30)))
        .손절(5)
        .익절(15)
    )

    결과 = 백테스트("삼성전자", 내전략, 기간="3년", 초기자금="1000만원")
    print(결과.요약())


def example_level2_lambda():
    """
    Level 2 example: Lambda-based condition definitions. / 람다 함수로 조건 정의.

    Demonstrates using the ``quickTest()`` function with lambda
    expressions for buy/sell conditions, enabling complex logic
    without creating a full strategy object.
    """
    print("\n" + "="*60)
    print("Level 2: 람다 함수 조건")
    print("="*60)

    from tradix.easy import backtest, quickTest

    result = quickTest(
        "삼성전자",
        buyCondition=lambda s, bar: (
            s.sma(20) is not None and
            bar.close > s.sma(20) and
            s.rsi(14) is not None and
            s.rsi(14) < 40
        ),
        sellCondition=lambda s, bar: (
            s.rsi(14) is not None and
            s.rsi(14) > 70
        ),
        stopLoss=5,
        takeProfit=10
    )

    print(result.summary())


def example_presets_comparison():
    """
    Compare multiple preset strategies side-by-side. / 여러 프리셋 전략을 비교합니다.

    Runs all major preset strategies on the same stock and period,
    then displays a formatted comparison table with return, max
    drawdown, Sharpe ratio, and win rate for each strategy.
    """
    print("\n" + "="*60)
    print("프리셋 전략 비교")
    print("="*60)

    from tradix.easy import (
        backtest,
        goldenCross,
        rsiOversold,
        bollingerBreakout,
        macdCross,
        meanReversion,
    )

    strategies = [
        ("골든크로스", goldenCross()),
        ("RSI과매도", rsiOversold()),
        ("볼린저돌파", bollingerBreakout()),
        ("MACD크로스", macdCross()),
        ("평균회귀", meanReversion()),
    ]

    print(f"{'전략':<15} {'수익률':>10} {'MDD':>10} {'샤프':>8} {'승률':>8}")
    print("-" * 55)

    for name, strategy in strategies:
        try:
            result = backtest("005930", strategy, period="3년")
            print(
                f"{name:<15} "
                f"{result.totalReturn:>+9.2f}% "
                f"{result.maxDrawdown:>9.2f}% "
                f"{result.sharpeRatio:>8.2f} "
                f"{result.winRate:>7.1f}%"
            )
        except Exception as e:
            print(f"{name:<15} Error: {e}")


def example_optimization():
    """
    Parameter optimization with grid search. / 파라미터 그리드 서치 최적화.

    Demonstrates using the ``optimize()`` function to find the best
    parameter combination for the goldenCross strategy by searching
    over fast/slow period ranges and maximizing the Sharpe ratio.
    """
    print("\n" + "="*60)
    print("파라미터 최적화")
    print("="*60)

    from tradix.easy import optimize, goldenCross

    result = optimize(
        "005930",
        goldenCross,
        period="3년",
        metric="sharpeRatio",
        fast=(5, 15, 5),
        slow=(20, 40, 10),
    )

    print(f"최적 파라미터: {result['best']['params']}")
    print(f"최적 샤프비율: {result['best']['metric']:.2f}")
    if result['best']['result']:
        print(f"최적 수익률: {result['best']['result'].totalReturn:+.2f}%")

    print("\nTop 5 결과:")
    for i, r in enumerate(result['all'][:5], 1):
        print(f"  {i}. {r['params']} → 샤프={r['metric']:.2f}")


def example_comparison_old_vs_new():
    """
    Compare traditional Strategy class approach vs Easy API. / 기존 방식 vs Easy API 비교.

    Shows the dramatic code reduction from the traditional 54-line
    Strategy subclass approach to the Easy API 2-line equivalent,
    demonstrating the same backtest with both approaches.
    """
    print("\n" + "="*60)
    print("기존 방식 vs Easy API 비교")
    print("="*60)

    print("\n[기존 방식 - 54줄]")
    print("""
from tradix import BacktestEngine, Strategy, Bar
from tradix.datafeed import FinanceDataReaderFeed

class SmaCrossStrategy(Strategy):
    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30

    def onBar(self, bar: Bar):
        fastSma = self.sma(self.fastPeriod)
        slowSma = self.sma(self.slowPeriod)

        if fastSma is None or slowSma is None:
            return

        prevFastSma = self.sma(self.fastPeriod, offset=1)
        prevSlowSma = self.sma(self.slowPeriod, offset=1)

        if prevFastSma is None or prevSlowSma is None:
            return

        goldenCross = prevFastSma <= prevSlowSma and fastSma > slowSma
        deadCross = prevFastSma >= prevSlowSma and fastSma < slowSma

        if goldenCross and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)
        elif deadCross and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)

data = FinanceDataReaderFeed('005930', '2021-01-01', '2024-01-01')
engine = BacktestEngine(data=data, strategy=SmaCrossStrategy())
result = engine.run()
print(result.summary())
""")

    print("\n[Easy API - 2줄]")
    print("""
from tradix.easy import 백테스트, 골든크로스
결과 = 백테스트("삼성전자", 골든크로스())
print(결과.요약())
""")

    from tradix.easy import 백테스트, 골든크로스
    결과 = 백테스트("삼성전자", 골든크로스(), 기간="3년")
    print(결과.요약())


def main():
    """Run all example functions sequentially. / 모든 예제를 순차적으로 실행합니다."""
    print("=" * 60)
    print("Tradix Easy API 예제")
    print("=" * 60)

    examples = [
        ("Level 1: 프리셋", example_level1_preset),
        ("Level 1: 한글 API", example_level1_korean),
        ("Level 2: 선언형 빌더", example_level2_builder),
        ("Level 2: 한글 빌더", example_level2_korean_builder),
        ("Level 2: 람다 조건", example_level2_lambda),
        ("프리셋 비교", example_presets_comparison),
        ("기존 vs Easy 비교", example_comparison_old_vs_new),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n{name} 예제 오류: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
