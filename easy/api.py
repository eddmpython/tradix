"""
Tradex Easy API Module - One-liner backtest and optimization functions.

Provides simplified entry points for backtesting and parameter optimization.
Supports Korean stock names, Korean currency notation, Korean period strings,
and both event-driven and vectorized execution modes.

Easy API 모듈 - 원라이너 백테스트 및 최적화 함수를 제공합니다.
한글 종목명, 한글 금액 표기, 한글 기간 문자열, 이벤트/벡터화 실행 모드를 지원합니다.

Features:
    - One-liner backtesting with ``backtest()``
    - Quick idea testing with ``quickTest()``
    - Grid-search parameter optimization with ``optimize()``
    - Korean stock name resolution (e.g., "삼성전자" -> "005930")
    - Korean currency parsing (e.g., "1억", "1000만원")
    - Auto mode selection (vectorized vs. event-driven)

Usage:
    >>> from tradex.easy import backtest, optimize, goldenCross
    >>>
    >>> result = backtest("005930", goldenCross())
    >>> print(result.totalReturn)
    >>>
    >>> result = backtest("삼성전자", goldenCross(), mode="vectorized")
    >>>
    >>> best = optimize("005930", goldenCross, fast=(5, 20), slow=(20, 60))
"""

from typing import Union, Optional, Dict, Any, Callable, List, Tuple
from datetime import datetime, timedelta
import re

from tradex.engine import BacktestEngine, BacktestResult, SimpleBroker, SimpleSizer
from tradex.datafeed.fdr import FinanceDataReaderFeed
from tradex.strategy.base import Strategy
from tradex.easy.quick import QuickStrategy


KOREAN_TICKERS = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940",
    "현대차": "005380",
    "기아": "000270",
    "셀트리온": "068270",
    "KB금융": "105560",
    "신한지주": "055550",
    "POSCO홀딩스": "005490",
    "네이버": "035420",
    "카카오": "035720",
    "삼성SDI": "006400",
    "LG화학": "051910",
    "현대모비스": "012330",
    "포스코퓨처엠": "003670",
    "삼성물산": "028260",
    "SK이노베이션": "096770",
    "한국전력": "015760",
    "삼성생명": "032830",
    "하나금융지주": "086790",
    "LG전자": "066570",
    "KT&G": "033780",
    "SK텔레콤": "017670",
    "LG": "003550",
    "코스피": "KS11",
    "코스닥": "KQ11",
}


class EasyResult:
    """
    User-friendly backtest result wrapper with bilingual property access.

    Wraps BacktestResult to provide both English (totalReturn, sharpeRatio)
    and Korean (수익률, 샤프비율) property accessors, plus a Korean summary method.

    백테스트 결과를 영문/한글 속성으로 모두 접근할 수 있는 래퍼 클래스입니다.

    Attributes:
        totalReturn (float): Total return in percent.
        annualReturn (float): Annualized return in percent.
        maxDrawdown (float): Maximum drawdown in percent.
        sharpeRatio (float): Sharpe ratio.
        winRate (float): Win rate in percent.
        totalTrades (int): Total number of trades.
        profitFactor (float): Profit factor.
        trades: List of individual trade records.
        equityCurve: Equity curve data.
        metrics: Full metrics dictionary.

    Example:
        >>> result = backtest("005930", goldenCross())
        >>> print(result.totalReturn)
        >>> print(result.수익률)
        >>> print(result.요약())
    """

    def __init__(self, result: BacktestResult):
        self._result = result

    @property
    def 수익률(self) -> float:
        """Total return in percent. / 총 수익률(%)."""
        return self._result.totalReturn

    @property
    def 연수익률(self) -> float:
        """Annualized return in percent. / 연환산 수익률(%)."""
        return self._result.metrics.get('annualReturn', 0)

    @property
    def 최대낙폭(self) -> float:
        """Maximum drawdown in percent. / 최대 낙폭(%)."""
        return self._result.metrics.get('maxDrawdown', 0)

    @property
    def 샤프비율(self) -> float:
        """Sharpe ratio. / 샤프 비율."""
        return self._result.metrics.get('sharpeRatio', 0)

    @property
    def 승률(self) -> float:
        """Win rate in percent. / 승률(%)."""
        return self._result.winRate

    @property
    def 거래횟수(self) -> int:
        """총 거래 횟수"""
        return self._result.totalTrades

    @property
    def 초기자금(self) -> float:
        """Initial capital amount. / 초기 자금."""
        return self._result.initialCash

    @property
    def 최종자산(self) -> float:
        """Final equity value. / 최종 자산."""
        return self._result.finalEquity

    @property
    def 손익비(self) -> float:
        """Profit factor. / 손익비."""
        return self._result.metrics.get('profitFactor', 0)

    @property
    def totalReturn(self) -> float:
        return self._result.totalReturn

    @property
    def annualReturn(self) -> float:
        return self._result.metrics.get('annualReturn', 0)

    @property
    def maxDrawdown(self) -> float:
        return self._result.metrics.get('maxDrawdown', 0)

    @property
    def sharpeRatio(self) -> float:
        return self._result.metrics.get('sharpeRatio', 0)

    @property
    def winRate(self) -> float:
        return self._result.winRate

    @property
    def totalTrades(self) -> int:
        return self._result.totalTrades

    @property
    def profitFactor(self) -> float:
        return self._result.metrics.get('profitFactor', 0)

    @property
    def trades(self):
        return self._result.trades

    @property
    def equityCurve(self):
        return self._result.equityCurve

    @property
    def metrics(self):
        return self._result.metrics

    def summary(self) -> str:
        return self._result.summary()

    def show(self, lang: str = "en", style: str = "modern") -> None:
        """Print Rich-styled result table. 터미널에 Rich 스타일 결과 출력.

        Args:
            lang: Language ('en' or 'ko'). / 언어.
            style: Display style ('modern', 'bloomberg', 'minimal'). / 출력 스타일.
        """
        from tradex.tui.console import printResult
        printResult(self, lang=lang, style=style)

    def chart(self, lang: str = "en") -> None:
        """Print full dashboard with charts. 차트 포함 대시보드 출력."""
        from tradex.tui.charts import plotDashboard
        plotDashboard(self, lang=lang)

    def 보기(self) -> None:
        """Rich 스타일 결과를 터미널에 출력."""
        self.show(lang="ko")

    def 차트(self) -> None:
        """차트 포함 대시보드를 터미널에 출력."""
        self.chart(lang="ko")

    def 요약(self) -> str:
        """Return a formatted Korean-language result summary. / 한글 결과 요약 반환."""
        return (
            f"\n{'='*50}\n"
            f"백테스트 결과\n"
            f"{'='*50}\n"
            f"종목: {self._result.symbol}\n"
            f"기간: {self._result.startDate} ~ {self._result.endDate}\n"
            f"{'─'*50}\n"
            f"초기 자금: {self._result.initialCash:,.0f}원\n"
            f"최종 자산: {self._result.finalEquity:,.0f}원\n"
            f"총 수익률: {self._result.totalReturn:+.2f}%\n"
            f"연 수익률: {self.연수익률:+.2f}%\n"
            f"{'─'*50}\n"
            f"최대 낙폭: {self.최대낙폭:.2f}%\n"
            f"샤프 비율: {self.샤프비율:.2f}\n"
            f"손익비: {self.손익비:.2f}\n"
            f"{'─'*50}\n"
            f"총 거래: {self._result.totalTrades}회\n"
            f"승률: {self._result.winRate:.1f}%\n"
            f"{'='*50}\n"
        )

    def __repr__(self) -> str:
        return f"EasyResult(수익률={self.수익률:+.2f}%, 승률={self.승률:.1f}%)"


def _resolveSymbol(symbol: str) -> str:
    """Resolve Korean stock name to ticker code. / 한글 종목명을 종목 코드로 변환."""
    if symbol in KOREAN_TICKERS:
        return KOREAN_TICKERS[symbol]
    return symbol


def _resolvePeriod(period: str) -> Tuple[str, str]:
    """Parse period string into (start_date, end_date) tuple. / 기간 문자열을 (시작일, 종료일)로 파싱."""
    if "~" in period:
        parts = period.replace(" ", "").split("~")
        return parts[0], parts[1]

    today = datetime.now()

    if period.endswith("년"):
        years = int(period[:-1])
        start = today - timedelta(days=years * 365)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    if period.endswith("개월"):
        months = int(period[:-2])
        start = today - timedelta(days=months * 30)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    if period.endswith("일"):
        days = int(period[:-1])
        start = today - timedelta(days=days)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    return "2020-01-01", today.strftime("%Y-%m-%d")


def backtest(
    symbol: str,
    strategy: Union[Strategy, QuickStrategy, str] = None,
    period: str = "3년",
    initialCash: Union[int, float, str] = 10_000_000,
    commission: float = 0.00015,
    verbose: bool = False,
    mode: str = "auto",
    stopLoss: float = 0,
    takeProfit: float = 0,
    **strategyParams
) -> EasyResult:
    """
    Run a one-liner backtest with automatic configuration.

    원라이너 백테스트를 자동 설정으로 실행합니다.

    Args:
        symbol: Ticker code or Korean name (e.g., "005930", "삼성전자").
        strategy: Strategy object or name (QuickStrategy, Strategy, or "goldenCross").
        period: Time period (e.g., "3년", "6개월", "2020-01-01~2024-12-31").
        initialCash: Starting capital (e.g., 10_000_000, "1000만원", "1억").
        commission: Commission rate (default 0.015%).
        verbose: Enable detailed logging output.
        mode: Execution mode ("auto", "vectorized", "event").
            - auto: Vectorized for simple strategies, event-driven for complex ones.
            - vectorized: Numba-accelerated (up to 100x faster).
            - event: Traditional event-driven engine.
        stopLoss: Stop-loss percent (vectorized mode only).
        takeProfit: Take-profit percent (vectorized mode only).
        **strategyParams: Strategy parameters (vectorized mode).

    Returns:
        EasyResult or VectorizedResult: Backtest results with bilingual access.

    Example:
        >>> result = backtest("삼성전자", goldenCross())
        >>> result = backtest("005930", "goldenCross", mode="vectorized", fast=10, slow=30)
        >>> result = backtest("005930", myComplexStrategy, mode="event")
    """
    ticker = _resolveSymbol(symbol)
    startDate, endDate = _resolvePeriod(period)
    cash = _parseCash(initialCash)

    useVectorized = False

    if mode == "vectorized":
        useVectorized = True
    elif mode == "auto":
        if isinstance(strategy, str):
            useVectorized = True
        elif strategy is None:
            useVectorized = True

    if useVectorized:
        from tradex.vectorized import vbacktest, VectorizedResult

        strategyName = strategy if isinstance(strategy, str) else "goldenCross"

        vresult = vbacktest(
            symbol=ticker,
            strategy=strategyName,
            period=period,
            initialCash=cash,
            stopLoss=stopLoss,
            takeProfit=takeProfit,
            **strategyParams
        )

        return vresult

    data = FinanceDataReaderFeed(ticker, startDate, endDate)
    broker = SimpleBroker(commissionRate=commission)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initialCash=cash,
        broker=broker,
    )

    result = engine.run(verbose=verbose)
    return EasyResult(result)


def _parseCash(value: Union[int, float, str]) -> float:
    """Parse capital amount with Korean currency notation support. / 한글 금액 표기를 파싱."""
    if isinstance(value, (int, float)):
        return float(value)

    value = value.replace(",", "").replace(" ", "")

    if "억" in value:
        num = float(re.sub(r"[^\d.]", "", value.split("억")[0]) or 1)
        return num * 100_000_000
    if "천만" in value:
        num = float(re.sub(r"[^\d.]", "", value.split("천만")[0]) or 1)
        return num * 10_000_000
    if "만" in value:
        num = float(re.sub(r"[^\d.]", "", value.split("만")[0]) or 1)
        return num * 10_000

    return float(re.sub(r"[^\d.]", "", value) or 10_000_000)


def quickTest(
    symbol: str,
    buyCondition: Callable,
    sellCondition: Callable = None,
    period: str = "3년",
    stopLoss: float = None,
    takeProfit: float = None,
) -> EasyResult:
    """
    Quick idea test using lambda conditions.

    람다 함수로 간단히 매수/매도 조건을 정의하여 빠르게 아이디어를 테스트합니다.

    Args:
        symbol: Ticker code or Korean name. / 종목 코드/한글명.
        buyCondition: Buy condition lambda (strategy, bar) -> bool. / 매수 조건 함수.
        sellCondition: Sell condition lambda (optional). / 매도 조건 함수 (선택).
        period: Time period string. / 기간.
        stopLoss: Stop-loss percent. / 손절%.
        takeProfit: Take-profit percent. / 익절%.

    Returns:
        EasyResult: Backtest results.

    Example:
        >>> result = quickTest(
        ...     "삼성전자",
        ...     buyCondition=lambda s, bar: bar.close > s.sma(20),
        ...     sellCondition=lambda s, bar: bar.close < s.sma(20)
        ... )
    """
    strategy = QuickStrategy("QuickTest")
    strategy.buyWhen(buyCondition)

    if sellCondition:
        strategy.sellWhen(sellCondition)
    if stopLoss:
        strategy.stopLoss(stopLoss)
    if takeProfit:
        strategy.takeProfit(takeProfit)

    return backtest(symbol, strategy, period)


def optimize(
    symbol: str,
    strategyFactory: Callable[..., Strategy],
    period: str = "3년",
    metric: str = "sharpeRatio",
    **paramRanges
) -> Dict[str, Any]:
    """
    Grid-search parameter optimization for a strategy.

    전략 파라미터를 그리드 서치로 최적화합니다.

    Args:
        symbol: Ticker code or Korean name. / 종목 코드/한글명.
        strategyFactory: Callable that creates a Strategy from keyword args. / 전략 생성 함수.
        period: Time period string. / 기간.
        metric: Optimization metric ("sharpeRatio", "totalReturn", "maxDrawdown"). / 최적화 지표.
        **paramRanges: Parameter ranges as tuples (start, end[, step]) or lists. / 파라미터 범위.

    Returns:
        dict: {"best": {"params", "metric", "result"}, "all": top-10 results}.

    Example:
        >>> best = optimize(
        ...     "삼성전자", goldenCross,
        ...     fast=(5, 20, 5), slow=(20, 60, 10)
        ... )
        >>> print(best['best']['params'])
    """
    ticker = _resolveSymbol(symbol)
    startDate, endDate = _resolvePeriod(period)

    def generateParams():
        """Generate all parameter combinations from ranges. / 파라미터 범위에서 모든 조합 생성."""
        import itertools

        paramNames = list(paramRanges.keys())
        paramValues = []

        for name, rangeVal in paramRanges.items():
            if isinstance(rangeVal, tuple):
                if len(rangeVal) == 2:
                    start, end = rangeVal
                    step = 1
                else:
                    start, end, step = rangeVal
                values = list(range(start, end + 1, step))
            elif isinstance(rangeVal, list):
                values = rangeVal
            else:
                values = [rangeVal]
            paramValues.append(values)

        for combo in itertools.product(*paramValues):
            yield dict(zip(paramNames, combo))

    bestResult = None
    bestParams = None
    bestMetric = float('-inf')

    results = []

    for params in generateParams():
        try:
            strategy = strategyFactory(**params)
            result = backtest(ticker, strategy, period, verbose=False)

            metricValue = result.metrics.get(metric, 0)

            compareValue = metricValue if metric != 'maxDrawdown' else -abs(metricValue)

            if compareValue > bestMetric:
                bestMetric = compareValue
                bestResult = result
                bestParams = params

            results.append({
                'params': params,
                'metric': metricValue,
                'result': result
            })

        except Exception as e:
            import warnings
            warnings.warn(f"Optimization failed for params {params}: {e}", stacklevel=2)
            continue

    if metric == 'maxDrawdown':
        results.sort(key=lambda x: abs(x['metric']))
    else:
        results.sort(key=lambda x: x['metric'], reverse=True)

    return {
        'best': {
            'params': bestParams,
            'metric': bestMetric,
            'result': bestResult
        },
        'all': results[:10]
    }
