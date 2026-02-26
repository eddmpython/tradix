"""
Tradex Core Backtest Engine.

Event-driven backtesting engine that simulates order execution, portfolio
management, and performance measurement for single-asset trading strategies.

트레이덱스 핵심 백테스트 엔진.
이벤트 기반 백테스팅 엔진으로, 단일 자산 전략에 대한 주문 체결 시뮬레이션,
포트폴리오 관리, 성과 측정을 수행합니다.

Features:
    - Event-driven architecture for realistic simulation
    - Configurable broker with commission, tax, and slippage modeling
    - Next-bar execution to prevent look-ahead bias
    - Automatic position sizing with configurable equity allocation
    - Comprehensive performance metrics (Sharpe, drawdown, profit factor, etc.)

Usage:
    >>> from tradex import BacktestEngine
    >>> from tradex.datafeed import FinanceDataReaderFeed
    >>> from tradex.strategy import Strategy
    >>>
    >>> class MyStrategy(Strategy):
    ...     def onBar(self, bar):
    ...         if not self.hasPosition(bar.symbol):
    ...             self.buy(bar.symbol)
    >>>
    >>> engine = BacktestEngine(
    ...     data=FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
    ...     strategy=MyStrategy(),
    ...     initialCash=10_000_000,
    ... )
    >>> result = engine.run()
    >>> print(result.summary())
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd

from tradex.entities.bar import Bar
from tradex.entities.order import Order, OrderSide, OrderStatus
from tradex.entities.trade import Trade
from tradex.events.market import MarketEvent
from tradex.events.fill import FillEvent
from tradex.datafeed.feed import DataFeed
from tradex.strategy.base import Strategy
from tradex.portfolio.portfolio import Portfolio


class SimpleBroker:
    """
    Simple broker simulator for order execution with realistic cost modeling.

    Simulates market order fills with configurable commission, tax, and slippage
    rates. Supports both close-price and open-price execution modes.

    간단한 브로커 시뮬레이터.
    수수료, 세금, 슬리피지를 반영한 시장가 주문 체결을 시뮬레이션합니다.
    Phase 3에서 broker/ 모듈로 분리 예정입니다.

    Attributes:
        commissionRate (float): Trading commission rate applied to fill value.
            수수료율 (체결 금액 대비).
        taxRate (float): Transaction tax rate applied to sell orders only.
            거래세율 (매도 주문에만 적용).
        slippageRate (float): Slippage rate applied to execution price.
            슬리피지율 (체결 가격에 적용).

    Example:
        >>> broker = SimpleBroker(commissionRate=0.00015, taxRate=0.0018, slippageRate=0.001)
        >>> fill = broker.processOrder(order, bar)
    """

    def __init__(
        self,
        commissionRate: float = 0.00015,
        taxRate: float = 0.0018,
        slippageRate: float = 0.001,
    ):
        self.commissionRate = commissionRate
        self.taxRate = taxRate
        self.slippageRate = slippageRate

    def processOrder(self, order: Order, bar: Bar) -> FillEvent:
        """Execute an order at the bar's close price. 종가 기준 주문 체결.

        Args:
            order: The order to execute. 체결할 주문.
            bar: The current price bar. 현재 가격 바.

        Returns:
            FillEvent containing execution details. 체결 상세 정보가 담긴 FillEvent.
        """
        return self._processAtPrice(order, bar, bar.close)

    def processOrderAtOpen(self, order: Order, bar: Bar) -> FillEvent:
        """Execute an order at the bar's open price. 시가 기준 주문 체결.

        Args:
            order: The order to execute. 체결할 주문.
            bar: The current price bar. 현재 가격 바.

        Returns:
            FillEvent containing execution details. 체결 상세 정보가 담긴 FillEvent.
        """
        return self._processAtPrice(order, bar, bar.open)

    def _processAtPrice(self, order: Order, bar: Bar, basePrice: float) -> FillEvent:
        """Execute an order at the specified base price with cost modeling. 지정 가격으로 주문 체결.

        Applies slippage to the base price (upward for buys, downward for sells),
        calculates commission on fill value, and adds transaction tax for sell orders.

        Args:
            order: The order to execute. 체결할 주문.
            bar: The current price bar (used for timestamp). 현재 가격 바 (타임스탬프 참조).
            basePrice: The reference price before slippage. 슬리피지 적용 전 기준 가격.

        Returns:
            FillEvent with computed fill price, commission, and slippage.
            체결 가격, 수수료, 슬리피지가 계산된 FillEvent.
        """
        if order.side == OrderSide.BUY:
            fillPrice = basePrice * (1 + self.slippageRate)
        else:
            fillPrice = basePrice * (1 - self.slippageRate)

        fillValue = order.quantity * fillPrice
        commission = fillValue * self.commissionRate

        if order.side == OrderSide.SELL:
            commission += fillValue * self.taxRate

        slippage = abs(fillPrice - basePrice) * order.quantity

        return FillEvent(
            timestamp=bar.datetime,
            order=order,
            fillPrice=fillPrice,
            fillQuantity=order.quantity,
            commission=commission,
            slippage=slippage
        )


class SimpleSizer:
    """
    Simple position sizer using fixed percentage of equity.

    Calculates the number of shares to purchase based on a configurable
    percentage of total portfolio equity. Returns at least 1 share.

    간단한 포지션 사이저.
    총 포트폴리오 자산 대비 고정 비율로 매수 수량을 계산합니다.
    Phase 4에서 risk/ 모듈로 분리 예정입니다.

    Attributes:
        percentOfEquity (float): Fraction of equity to allocate per trade (0.0-1.0).
            거래당 배분할 자산 비율 (0.0~1.0).

    Example:
        >>> sizer = SimpleSizer(percentOfEquity=0.95)
        >>> quantity = sizer.calculate(equity=10_000_000, price=50_000)
    """

    def __init__(self, percentOfEquity: float = 0.95):
        self.percentOfEquity = percentOfEquity

    def calculate(self, equity: float, price: float) -> int:
        """Calculate the number of shares to buy. 매수 수량 계산.

        Args:
            equity: Current total portfolio equity. 현재 포트폴리오 총 자산.
            price: Current price per share. 현재 주당 가격.

        Returns:
            Number of shares to purchase (minimum 1). 매수할 주식 수 (최소 1).
        """
        amount = equity * self.percentOfEquity
        quantity = int(amount / price)
        return max(1, quantity)


@dataclass
class BacktestResult:
    """
    Container for single-asset backtest results and performance metrics.

    Stores the complete output of a backtest run including trade history,
    equity curve, and calculated performance metrics.

    단일 자산 백테스트 결과 및 성과 지표 컨테이너.
    거래 내역, 자산 곡선, 성과 지표를 포함한 백테스트 실행 결과를 저장합니다.

    Attributes:
        strategy (str): Name of the strategy used. 사용된 전략 이름.
        symbol (str): Ticker symbol of the backtested asset. 백테스트 대상 종목 코드.
        startDate (str): Backtest start date string. 백테스트 시작일.
        endDate (str): Backtest end date string. 백테스트 종료일.
        initialCash (float): Starting cash amount. 초기 자본금.
        finalEquity (float): Final portfolio equity value. 최종 포트폴리오 자산 가치.
        totalReturn (float): Total return percentage. 총 수익률 (%).
        totalTrades (int): Total number of completed round-trip trades. 총 완결 거래 수.
        winRate (float): Percentage of profitable trades. 승률 (%).
        trades (List[Trade]): List of all completed trades. 완결된 거래 목록.
        equityCurve (pd.Series): Time-indexed equity values. 시간 인덱스 자산 곡선.
        metrics (Dict[str, Any]): Extended performance metrics dictionary.
            확장 성과 지표 딕셔너리.

    Example:
        >>> result = engine.run()
        >>> print(result.summary())
        >>> print(f"Sharpe: {result.metrics['sharpeRatio']:.2f}")
    """
    strategy: str
    symbol: str
    startDate: str
    endDate: str
    initialCash: float
    finalEquity: float
    totalReturn: float
    totalTrades: int
    winRate: float
    trades: List[Trade] = field(default_factory=list)
    equityCurve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the backtest results. 백테스트 결과 요약 생성.

        Returns:
            Formatted string with key performance statistics. 주요 성과 통계가 담긴 포맷 문자열.
        """
        return (
            f"\n{'='*50}\n"
            f"백테스트 결과: {self.strategy}\n"
            f"{'='*50}\n"
            f"종목: {self.symbol}\n"
            f"기간: {self.startDate} ~ {self.endDate}\n"
            f"{'─'*50}\n"
            f"초기 자본: {self.initialCash:,.0f}원\n"
            f"최종 자산: {self.finalEquity:,.0f}원\n"
            f"총 수익률: {self.totalReturn:+.2f}%\n"
            f"{'─'*50}\n"
            f"총 거래: {self.totalTrades}회\n"
            f"승률: {self.winRate:.1f}%\n"
            f"{'='*50}\n"
        )

    def show(self, lang: str = "en") -> None:
        """Print Rich-styled result table to terminal. 터미널에 Rich 스타일 결과 출력."""
        from tradex.tui.console import printResult
        printResult(self, lang=lang)

    def chart(self, lang: str = "en") -> None:
        """Print full dashboard with charts. 차트 포함 대시보드 출력."""
        from tradex.tui.charts import plotDashboard
        plotDashboard(self, lang=lang)

    def toDict(self) -> dict:
        """Convert the result to a serializable dictionary. 결과를 직렬화 가능한 딕셔너리로 변환.

        Returns:
            Dictionary containing all result fields except trades and equityCurve.
            trades와 equityCurve를 제외한 모든 결과 필드가 담긴 딕셔너리.
        """
        return {
            'strategy': self.strategy,
            'symbol': self.symbol,
            'startDate': self.startDate,
            'endDate': self.endDate,
            'initialCash': self.initialCash,
            'finalEquity': self.finalEquity,
            'totalReturn': self.totalReturn,
            'totalTrades': self.totalTrades,
            'winRate': self.winRate,
            'metrics': self.metrics,
        }


class BacktestEngine:
    """
    Core event-driven backtest engine for single-asset strategy evaluation.

    Orchestrates the full backtesting pipeline: iterates through price bars,
    dispatches events to the strategy, processes orders through the broker,
    and collects performance metrics. Supports both same-bar close-price
    execution and next-bar open-price execution to prevent look-ahead bias.

    핵심 이벤트 기반 백테스트 엔진 (단일 자산 전략 평가).
    가격 바 순회, 전략 이벤트 전달, 브로커 주문 처리, 성과 지표 수집으로
    구성된 전체 백테스팅 파이프라인을 관리합니다. Look-ahead bias 방지를 위한
    다음 바 시가 체결 모드를 지원합니다.

    Attributes:
        data (DataFeed): Price data feed for the backtest. 백테스트용 가격 데이터 피드.
        strategy (Strategy): Trading strategy instance. 트레이딩 전략 인스턴스.
        initialCash (float): Starting capital. 초기 자본금.
        broker (SimpleBroker): Broker simulator for order execution. 주문 체결 브로커 시뮬레이터.
        sizer (SimpleSizer): Position sizer for quantity calculation. 수량 계산 포지션 사이저.
        fillOnNextBar (bool): If True, orders fill at next bar's open price.
            True이면 주문이 다음 바 시가에 체결됩니다.

    Example:
        >>> from tradex import BacktestEngine
        >>> from tradex.datafeed import FinanceDataReaderFeed
        >>> from tradex.strategy import Strategy
        >>>
        >>> class MyStrategy(Strategy):
        ...     def onBar(self, bar):
        ...         if not self.hasPosition(bar.symbol):
        ...             self.buy(bar.symbol)
        >>>
        >>> engine = BacktestEngine(
        ...     data=FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
        ...     strategy=MyStrategy(),
        ...     initialCash=10_000_000,
        ... )
        >>> result = engine.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        data: DataFeed,
        strategy: Strategy,
        initialCash: float = 10_000_000,
        broker: SimpleBroker = None,
        sizer: SimpleSizer = None,
        fillOnNextBar: bool = True,
    ):
        """Initialize the backtest engine. 백테스트 엔진 초기화.

        Args:
            data: Price data feed providing Bar objects.
                Bar 객체를 제공하는 가격 데이터 피드.
            strategy: Trading strategy to evaluate.
                평가할 트레이딩 전략.
            initialCash: Starting capital in currency units (default: 10,000,000).
                초기 자본금 (기본값: 1천만원).
            broker: Broker simulator for order execution (default: SimpleBroker).
                주문 체결 브로커 시뮬레이터 (기본값: SimpleBroker).
            sizer: Position sizer for quantity calculation (default: SimpleSizer).
                수량 계산 포지션 사이저 (기본값: SimpleSizer).
            fillOnNextBar: If True, execute orders at next bar's open to prevent
                look-ahead bias (default: True).
                True이면 look-ahead bias 방지를 위해 다음 바 시가에 체결 (기본값: True).
        """
        self.data = data
        self.strategy = strategy
        self.initialCash = initialCash
        self.broker = broker or SimpleBroker()
        self.sizer = sizer or SimpleSizer()
        self.fillOnNextBar = fillOnNextBar

        self._portfolio = Portfolio(initialCash)
        self._equityHistory: List[tuple] = []
        self._pendingOrdersForNextBar: List[Order] = []

    def run(self, verbose: bool = False) -> BacktestResult:
        """Run the backtest simulation from start to end. 백테스트 시뮬레이션 실행.

        Iterates through all bars in the data feed, processes strategy signals,
        executes orders, and builds the final result with performance metrics.

        Args:
            verbose: If True, print detailed execution logs to stdout.
                True이면 상세 체결 로그를 출력합니다.

        Returns:
            BacktestResult containing trades, equity curve, and performance metrics.
            거래 내역, 자산 곡선, 성과 지표가 담긴 BacktestResult.
        """
        self.data.load()
        self._portfolio.reset()
        self._equityHistory.clear()
        self._pendingOrdersForNextBar.clear()

        self.strategy._setup(self._portfolio, self.broker)
        self.strategy._setFullData(self.data.toDataFrame())
        self.strategy.initialize()

        if verbose:
            print(f"[BacktestEngine] 시작: {self.data.symbol}")
            print(f"[BacktestEngine] 기간: {self.data.startDate} ~ {self.data.endDate}")
            print(f"[BacktestEngine] 총 {self.data.totalBars} 바")
            print(f"[BacktestEngine] 체결방식: {'다음바 시가' if self.fillOnNextBar else '당일 종가'}")

        for bar in self.data:
            self._portfolio.updatePrice(bar.symbol, bar.close)

            if self.fillOnNextBar and self._pendingOrdersForNextBar:
                self._processNextBarOrders(bar, verbose)

            self.strategy._updateIndex(self.data.currentIndex)
            self.strategy._updateCurrentBar(bar)

            self.strategy.onBar(bar)

            if self.fillOnNextBar:
                self._queueOrdersForNextBar()
            else:
                self._processOrders(bar, verbose)

            self._equityHistory.append((bar.datetime, self._portfolio.equity))

        self.strategy.onEnd()

        return self._buildResult()

    def _queueOrdersForNextBar(self):
        """Move current bar's pending orders to the next-bar execution queue. 현재 바 주문을 다음 바 체결 대기열로 이동."""
        pendingOrders = self._portfolio.pendingOrders.copy()
        for order in pendingOrders:
            self._pendingOrdersForNextBar.append(order)
            self._portfolio._pendingOrders.remove(order)

    def _processNextBarOrders(self, bar: Bar, verbose: bool = False):
        """Execute queued orders at the current bar's open price. 대기 주문을 현재 바 시가에 체결.

        Handles position sizing for unspecified quantities, validates cash
        sufficiency for buy orders, and delegates to the broker for execution.

        Args:
            bar: The current price bar whose open price is used for fills.
                시가를 체결 가격으로 사용할 현재 가격 바.
            verbose: If True, print fill details to stdout.
                True이면 체결 상세 내역을 출력합니다.
        """
        for order in self._pendingOrdersForNextBar:
            if order.quantity <= 0:
                order.quantity = self.sizer.calculate(
                    self._portfolio.equity,
                    bar.open
                )

            if order.side == OrderSide.BUY:
                cost = order.quantity * bar.open * 1.01
                if cost > self._portfolio.cash:
                    maxQty = int(self._portfolio.cash * 0.99 / bar.open)
                    if maxQty <= 0:
                        order.reject("잔고 부족")
                        continue
                    order.quantity = maxQty

            fill = self.broker.processOrderAtOpen(order, bar)

            self._portfolio.processFill(fill)

            self.strategy.onOrderFill(fill)

            if verbose:
                print(
                    f"[{bar.datetime}] {order.side.value.upper()} "
                    f"{fill.fillQuantity} @ {fill.fillPrice:,.0f} (시가체결) "
                    f"(수수료: {fill.commission:,.0f})"
                )

        self._pendingOrdersForNextBar.clear()

    def _processOrders(self, bar: Bar, verbose: bool = False):
        """Process pending orders at the current bar's close price. 대기 주문을 당일 종가에 체결.

        Args:
            bar: The current price bar whose close price is used for fills.
                종가를 체결 가격으로 사용할 현재 가격 바.
            verbose: If True, print fill details to stdout.
                True이면 체결 상세 내역을 출력합니다.
        """
        pendingOrders = self._portfolio.pendingOrders.copy()

        for order in pendingOrders:
            if order.quantity <= 0:
                order.quantity = self.sizer.calculate(
                    self._portfolio.equity,
                    bar.close
                )

            if order.side == OrderSide.BUY:
                cost = order.quantity * bar.close * 1.01
                if cost > self._portfolio.cash:
                    maxQty = int(self._portfolio.cash * 0.99 / bar.close)
                    if maxQty <= 0:
                        order.reject("잔고 부족")
                        self._portfolio._pendingOrders.remove(order)
                        continue
                    order.quantity = maxQty

            fill = self.broker.processOrder(order, bar)

            self._portfolio.processFill(fill)

            self.strategy.onOrderFill(fill)

            if verbose:
                print(
                    f"[{bar.datetime}] {order.side.value.upper()} "
                    f"{fill.fillQuantity} @ {fill.fillPrice:,.0f} (종가체결) "
                    f"(수수료: {fill.commission:,.0f})"
                )

    def _buildResult(self) -> BacktestResult:
        """Build the final BacktestResult from accumulated data. 누적된 데이터로 최종 BacktestResult 생성.

        Returns:
            BacktestResult populated with trades, equity curve, and metrics.
            거래 내역, 자산 곡선, 성과 지표가 포함된 BacktestResult.
        """
        trades = self._portfolio.trades
        stats = self._portfolio.getTradeStats()

        equitySeries = pd.Series(
            [e[1] for e in self._equityHistory],
            index=pd.DatetimeIndex([e[0] for e in self._equityHistory])
        )

        metrics = self._calculateMetrics(equitySeries, trades)

        return BacktestResult(
            strategy=self.strategy.name,
            symbol=self.data.symbol,
            startDate=self.data.startDate,
            endDate=self.data.endDate,
            initialCash=self.initialCash,
            finalEquity=self._portfolio.equity,
            totalReturn=self._portfolio.totalReturn,
            totalTrades=stats['totalTrades'],
            winRate=stats['winRate'],
            trades=trades,
            equityCurve=equitySeries,
            metrics=metrics
        )

    def _calculateMetrics(
        self,
        equityCurve: pd.Series,
        trades: List[Trade]
    ) -> dict:
        """Calculate comprehensive performance metrics from equity curve and trades. 자산 곡선과 거래 내역으로 종합 성과 지표 계산.

        Computes total return, annualized return, volatility, Sharpe ratio,
        maximum drawdown, win rate, profit factor, and average win/loss statistics.

        Args:
            equityCurve: Time-indexed series of portfolio equity values.
                시간 인덱스 포트폴리오 자산 가치 시리즈.
            trades: List of completed round-trip trades.
                완결된 왕복 거래 목록.

        Returns:
            Dictionary of performance metric names to values. Empty if insufficient data.
            성과 지표 이름-값 딕셔너리. 데이터 부족 시 빈 딕셔너리 반환.
        """
        if len(equityCurve) < 2:
            return {}

        returns = equityCurve.pct_change().dropna()

        totalReturn = (equityCurve.iloc[-1] / equityCurve.iloc[0]) - 1

        tradingDays = len(equityCurve)
        years = tradingDays / 252
        annualReturn = (1 + totalReturn) ** (1 / years) - 1 if years > 0 else 0

        volatility = returns.std() * (252 ** 0.5)

        riskFreeRate = 0.035
        sharpeRatio = (annualReturn - riskFreeRate) / volatility if volatility > 0 else 0

        cumMax = equityCurve.cummax()
        drawdown = (equityCurve - cumMax) / cumMax
        maxDrawdown = drawdown.min()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        avgWin = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avgLoss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
        avgWinPercent = sum(t.pnlPercent for t in wins) / len(wins) if wins else 0
        avgLossPercent = abs(sum(t.pnlPercent for t in losses) / len(losses)) if losses else 0
        profitFactor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

        return {
            'totalReturn': totalReturn * 100,
            'annualReturn': annualReturn * 100,
            'volatility': volatility * 100,
            'sharpeRatio': sharpeRatio,
            'maxDrawdown': maxDrawdown * 100,
            'winRate': len(wins) / len(trades) * 100 if trades else 0,
            'profitFactor': profitFactor,
            'avgWin': avgWin,
            'avgLoss': avgLoss,
            'avgWinPercent': avgWinPercent,
            'avgLossPercent': avgLossPercent,
            'totalTrades': len(trades),
        }

    def __repr__(self) -> str:
        return (
            f"BacktestEngine({self.strategy.name}, "
            f"{self.data.symbol}, "
            f"cash={self.initialCash:,.0f})"
        )
