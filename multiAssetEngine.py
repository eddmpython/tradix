"""
Tradex Multi-Asset Backtest Engine.

Event-driven backtesting engine for multi-asset portfolio strategies. Supports
simultaneous trading across multiple symbols with synchronized bar data,
portfolio rebalancing, and cross-asset indicator computation.

트레이덱스 멀티에셋 백테스트 엔진.
다중 자산 포트폴리오 전략을 위한 이벤트 기반 백테스팅 엔진입니다.
동기화된 바 데이터를 통한 다중 종목 동시 거래, 포트폴리오 리밸런싱,
자산 간 지표 계산을 지원합니다.

Features:
    - Multi-asset portfolio management with weight-based rebalancing
    - Synchronized bar iteration across all symbols
    - Per-symbol technical indicator access (SMA, EMA, RSI, MACD, Bollinger, etc.)
    - Monthly and weekly rebalancing schedule helpers
    - Position history tracking per symbol
    - Next-bar execution to prevent look-ahead bias

Usage:
    >>> from tradex.datafeed import FinanceDataReaderFeed, MultiDataFeed
    >>> from tradex.multiAssetEngine import MultiAssetEngine, MultiAssetStrategy
    >>>
    >>> feeds = {
    ...     '005930': FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
    ...     '000660': FinanceDataReaderFeed('000660', '2020-01-01', '2024-12-31'),
    ... }
    >>>
    >>> class RotationStrategy(MultiAssetStrategy):
    ...     def onBars(self, bars):
    ...         pass  # rotation logic
    >>>
    >>> engine = MultiAssetEngine(
    ...     data=MultiDataFeed(feeds),
    ...     strategy=RotationStrategy(),
    ...     initialCash=100_000_000,
    ... )
    >>> result = engine.run()
    >>> print(result.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from tradex.entities.bar import Bar
from tradex.entities.order import Order, OrderSide, OrderStatus
from tradex.entities.trade import Trade
from tradex.datafeed.multiDataFeed import MultiDataFeed
from tradex.strategy.base import Strategy
from tradex.strategy.indicators import Indicators
from tradex.portfolio.portfolio import Portfolio
from tradex.engine import SimpleBroker, SimpleSizer


@dataclass
class MultiAssetResult:
    """
    Container for multi-asset backtest results and performance metrics.

    Stores the complete output of a multi-asset backtest run including trade
    history, equity curve, per-symbol position history, and performance metrics.

    멀티에셋 백테스트 결과 및 성과 지표 컨테이너.
    거래 내역, 자산 곡선, 종목별 포지션 이력, 성과 지표를 포함한
    멀티에셋 백테스트 실행 결과를 저장합니다.

    Attributes:
        strategy (str): Name of the strategy used. 사용된 전략 이름.
        symbols (List[str]): List of ticker symbols included. 포함된 종목 코드 목록.
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
        positionHistory (Dict[str, pd.Series]): Per-symbol position quantity over time.
            종목별 시간에 따른 포지션 수량 이력.

    Example:
        >>> result = engine.run()
        >>> print(result.summary())
        >>> print(result.positionHistory['005930'].tail())
    """
    strategy: str
    symbols: List[str]
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
    positionHistory: Dict[str, pd.Series] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the multi-asset backtest results. 멀티에셋 백테스트 결과 요약 생성.

        Returns:
            Formatted string with key performance statistics including symbols traded.
            거래 종목을 포함한 주요 성과 통계가 담긴 포맷 문자열.
        """
        symbolStr = ', '.join(self.symbols[:3])
        if len(self.symbols) > 3:
            symbolStr += f' 외 {len(self.symbols) - 3}개'

        return (
            f"\n{'='*50}\n"
            f"멀티에셋 백테스트 결과: {self.strategy}\n"
            f"{'='*50}\n"
            f"종목: {symbolStr}\n"
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

    def toDict(self) -> dict:
        """Convert the result to a serializable dictionary. 결과를 직렬화 가능한 딕셔너리로 변환.

        Returns:
            Dictionary containing all result fields except trades, equityCurve,
            and positionHistory.
            trades, equityCurve, positionHistory를 제외한 모든 결과 필드가 담긴 딕셔너리.
        """
        return {
            'strategy': self.strategy,
            'symbols': self.symbols,
            'startDate': self.startDate,
            'endDate': self.endDate,
            'initialCash': self.initialCash,
            'finalEquity': self.finalEquity,
            'totalReturn': self.totalReturn,
            'totalTrades': self.totalTrades,
            'winRate': self.winRate,
            'metrics': self.metrics,
        }


class MultiAssetStrategy(Strategy):
    """
    Base class for multi-asset trading strategies.

    Extends Strategy to support simultaneous trading across multiple symbols.
    Override ``onBars()`` instead of ``onBar()`` to receive a dictionary of
    current bars for all symbols on each time step. Provides per-symbol
    technical indicators, portfolio weight management, and rebalancing utilities.

    멀티에셋 트레이딩 전략 베이스 클래스.
    Strategy를 확장하여 다중 종목 동시 거래를 지원합니다.
    ``onBar()`` 대신 ``onBars()``를 오버라이드하여 각 타임스텝에서
    모든 종목의 현재 바 딕셔너리를 수신합니다.

    Attributes:
        _multiIndicators (Dict[str, Indicators]): Per-symbol indicator instances.
            종목별 지표 인스턴스.
        _symbols (List[str]): List of all tradeable symbols.
            거래 가능 종목 목록.
        _currentBars (Dict[str, Bar]): Current bar for each symbol.
            종목별 현재 바.
        _targetWeights (Dict[str, float]): Target portfolio weights per symbol.
            종목별 목표 포트폴리오 비중.
        _lastRebalanceDate (datetime): Timestamp of the most recent rebalance.
            가장 최근 리밸런싱 시각.

    Example:
        >>> class MomentumRotation(MultiAssetStrategy):
        ...     def initialize(self):
        ...         self.topN = 3
        ...         self.lookback = 60
        ...
        ...     def onBars(self, bars: Dict[str, Bar]):
        ...         if not self._isMonthStart():
        ...             return
        ...         momentums = {}
        ...         for symbol in bars.keys():
        ...             roc = self.roc(symbol, self.lookback)
        ...             if roc is not None:
        ...                 momentums[symbol] = roc
        ...         sorted_symbols = sorted(
        ...             momentums.items(), key=lambda x: x[1], reverse=True
        ...         )[:self.topN]
        ...         weights = {s: 1.0 / self.topN for s, _ in sorted_symbols}
        ...         self.rebalance(weights)
    """

    def __init__(self):
        super().__init__()
        self._multiIndicators: Dict[str, Indicators] = {}
        self._symbols: List[str] = []
        self._currentBars: Dict[str, Bar] = {}
        self._targetWeights: Dict[str, float] = {}
        self._lastRebalanceDate: datetime = None

    def _setupMulti(self, portfolio: Portfolio, broker: SimpleBroker, symbols: List[str]):
        """Initialize multi-asset strategy with portfolio, broker, and symbol list. 포트폴리오, 브로커, 종목 목록으로 멀티에셋 전략 초기화.

        Args:
            portfolio: Portfolio instance for position tracking. 포지션 추적용 포트폴리오 인스턴스.
            broker: Broker simulator for order execution. 주문 체결 브로커 시뮬레이터.
            symbols: List of tradeable ticker symbols. 거래 가능 종목 코드 목록.
        """
        self._portfolio = portfolio
        self._broker = broker
        self._symbols = symbols

        for symbol in symbols:
            self._multiIndicators[symbol] = Indicators()

    def _setMultiFullData(self, dataFrames: Dict[str, pd.DataFrame]):
        """Set full historical data for all symbols to enable indicator caching. 지표 캐싱을 위한 전체 종목 히스토리 데이터 설정.

        Args:
            dataFrames: Mapping of symbol to its full OHLCV DataFrame.
                종목 코드와 해당 전체 OHLCV DataFrame의 매핑.
        """
        for symbol, df in dataFrames.items():
            if symbol in self._multiIndicators:
                self._multiIndicators[symbol].setFullData(df)

    def _updateMultiIndex(self, index: int):
        """Update the current bar index for all per-symbol indicators. 모든 종목별 지표의 현재 바 인덱스 업데이트.

        Args:
            index: The current bar index in the data feed. 데이터 피드에서의 현재 바 인덱스.
        """
        for indicator in self._multiIndicators.values():
            indicator.setIndex(index)

    def _updateCurrentBars(self, bars: Dict[str, Bar]):
        """Update current bars for all symbols. 모든 종목의 현재 바 업데이트.

        Args:
            bars: Mapping of symbol to its current Bar. 종목 코드와 현재 Bar의 매핑.
        """
        self._currentBars = bars
        if bars:
            firstBar = list(bars.values())[0]
            self._currentBar = firstBar

    def onBar(self, bar: Bar):
        """Handle a single bar event (no-op for compatibility with base Strategy). 단일 바 이벤트 처리 (기본 Strategy 호환용 no-op)."""
        pass

    def onBars(self, bars: Dict[str, Bar]):
        """Receive current bars for all symbols (must be overridden by subclass). 모든 종목의 현재 바 수신 (서브클래스에서 구현 필요).

        Called once per time step with a dictionary mapping each symbol to its
        current Bar. Implement your multi-asset trading logic here.

        Args:
            bars: Mapping of symbol to its current Bar object.
                종목 코드와 현재 Bar 객체의 매핑.
        """
        pass

    def sma(self, symbol: str, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """Compute Simple Moving Average for a specific symbol. 특정 종목의 단순 이동 평균(SMA) 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: Number of bars for the moving average window. 이동 평균 윈도우 바 수.
            column: Data column to use (default: 'close'). 사용할 데이터 컬럼 (기본값: 'close').
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            SMA value, or None if the symbol is not found. SMA 값. 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].sma(period, column, offset)
        return None

    def ema(self, symbol: str, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """Compute Exponential Moving Average for a specific symbol. 특정 종목의 지수 이동 평균(EMA) 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: Number of bars for the EMA span. EMA 스팬 바 수.
            column: Data column to use (default: 'close'). 사용할 데이터 컬럼 (기본값: 'close').
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            EMA value, or None if the symbol is not found. EMA 값. 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].ema(period, column, offset)
        return None

    def rsi(self, symbol: str, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Relative Strength Index for a specific symbol. 특정 종목의 상대강도지수(RSI) 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: RSI lookback period (default: 14). RSI 산출 기간 (기본값: 14).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            RSI value (0-100), or None if the symbol is not found. RSI 값 (0~100). 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].rsi(period, offset)
        return None

    def roc(self, symbol: str, period: int = 12, offset: int = 0) -> Optional[float]:
        """Compute Rate of Change for a specific symbol. 특정 종목의 변화율(ROC) 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: ROC lookback period (default: 12). ROC 산출 기간 (기본값: 12).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            ROC value, or None if the symbol is not found. ROC 값. 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].roc(period, offset)
        return None

    def momentum(self, symbol: str, period: int = 10, offset: int = 0) -> Optional[float]:
        """Compute momentum for a specific symbol. 특정 종목의 모멘텀 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: Momentum lookback period (default: 10). 모멘텀 산출 기간 (기본값: 10).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            Momentum value, or None if the symbol is not found. 모멘텀 값. 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].momentum(period, offset)
        return None

    def atr(self, symbol: str, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Average True Range for a specific symbol. 특정 종목의 평균 진폭(ATR) 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: ATR lookback period (default: 14). ATR 산출 기간 (기본값: 14).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            ATR value, or None if the symbol is not found. ATR 값. 종목 미발견 시 None.
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].atr(period, offset)
        return None

    def bollinger(self, symbol: str, period: int = 20, std: float = 2.0, offset: int = 0) -> tuple:
        """Compute Bollinger Bands for a specific symbol. 특정 종목의 볼린저 밴드 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            period: Moving average period (default: 20). 이동 평균 기간 (기본값: 20).
            std: Number of standard deviations for bands (default: 2.0). 밴드 표준편차 배수 (기본값: 2.0).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            Tuple of (upper, middle, lower) band values, or (None, None, None) if not found.
            (상단, 중간, 하단) 밴드 값 튜플. 종목 미발견 시 (None, None, None).
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].bollinger(period, std, offset)
        return (None, None, None)

    def macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9, offset: int = 0) -> tuple:
        """Compute MACD for a specific symbol. 특정 종목의 MACD 계산.

        Args:
            symbol: Ticker symbol. 종목 코드.
            fast: Fast EMA period (default: 12). 단기 EMA 기간 (기본값: 12).
            slow: Slow EMA period (default: 26). 장기 EMA 기간 (기본값: 26).
            signal: Signal line EMA period (default: 9). 시그널 라인 EMA 기간 (기본값: 9).
            offset: Number of bars to look back from current (default: 0). 현재로부터의 오프셋 (기본값: 0).

        Returns:
            Tuple of (macd_line, signal_line, histogram), or (None, None, None) if not found.
            (MACD 라인, 시그널 라인, 히스토그램) 튜플. 종목 미발견 시 (None, None, None).
        """
        if symbol in self._multiIndicators:
            return self._multiIndicators[symbol].macd(fast, slow, signal, offset)
        return (None, None, None)

    def getWeight(self, symbol: str) -> float:
        """Get the current portfolio weight of a symbol based on market value. 시가 기준 현재 종목 포트폴리오 비중 조회.

        Args:
            symbol: Ticker symbol. 종목 코드.

        Returns:
            Current weight as a fraction of total equity (0.0-1.0).
            총 자산 대비 현재 비중 (0.0~1.0).
        """
        if not self._portfolio or self._portfolio.equity <= 0:
            return 0.0

        position = self._portfolio.getPosition(symbol)
        if not position:
            return 0.0

        bar = self._currentBars.get(symbol)
        if not bar:
            return 0.0

        positionValue = position.quantity * bar.close
        return positionValue / self._portfolio.equity

    def setTargetWeight(self, symbol: str, weight: float):
        """Set the target portfolio weight for a symbol. 종목의 목표 포트폴리오 비중 설정.

        Args:
            symbol: Ticker symbol. 종목 코드.
            weight: Target weight clamped to [0.0, 1.0]. 목표 비중 (0.0~1.0으로 클램핑).
        """
        self._targetWeights[symbol] = max(0.0, min(1.0, weight))

    def rebalance(self, weights: Dict[str, float]):
        """Rebalance the portfolio to match target weights. 포트폴리오를 목표 비중에 맞게 리밸런싱.

        Calculates the difference between current and target weights for each
        symbol and generates buy/sell orders to converge. Weights exceeding
        a total of 1.0 are normalized proportionally. Weight differences below
        1% are ignored to avoid excessive trading.

        Args:
            weights: Mapping of symbol to target weight (0.0-1.0). Symbols not
                included default to a target weight of 0.0.
                종목 코드와 목표 비중(0.0~1.0)의 매핑. 미포함 종목은 0.0으로 처리.

        Example:
            >>> self.rebalance({'005930': 0.4, '000660': 0.3, '035720': 0.3})
        """
        if not self._portfolio:
            return

        totalWeight = sum(weights.values())
        if totalWeight > 1.0:
            weights = {s: w / totalWeight for s, w in weights.items()}

        for symbol in self._symbols:
            currentWeight = self.getWeight(symbol)
            targetWeight = weights.get(symbol, 0.0)

            bar = self._currentBars.get(symbol)
            if not bar:
                continue

            weightDiff = targetWeight - currentWeight

            if abs(weightDiff) < 0.01:
                continue

            targetValue = self._portfolio.equity * targetWeight
            currentPosition = self._portfolio.getPosition(symbol)
            currentQuantity = currentPosition.quantity if currentPosition else 0
            currentValue = currentQuantity * bar.close

            valueDiff = targetValue - currentValue

            if valueDiff > 0:
                quantity = int(valueDiff / bar.close)
                if quantity > 0:
                    self.buy(symbol, quantity)

            elif valueDiff < 0:
                quantity = int(abs(valueDiff) / bar.close)
                if quantity > 0 and currentQuantity >= quantity:
                    self.sell(symbol, quantity)

        self._lastRebalanceDate = self._currentBar.datetime if self._currentBar else None

    def closeAllPositions(self):
        """Close all open positions across all symbols. 모든 종목의 보유 포지션 전량 청산."""
        for symbol in self._symbols:
            if self.hasPosition(symbol):
                self.closePosition(symbol)

    def _isMonthStart(self) -> bool:
        """Check if the current bar is at the start of a new month. 현재 바가 새로운 월의 시작인지 확인.

        Returns:
            True if this is the first bar of a new month or no rebalance has occurred yet.
            새로운 월의 첫 바이거나 아직 리밸런싱이 없었으면 True.
        """
        if not self._currentBar:
            return False

        currentDate = self._currentBar.datetime
        if self._lastRebalanceDate is None:
            return True

        return currentDate.month != self._lastRebalanceDate.month

    def _isWeekStart(self) -> bool:
        """Check if the current bar falls on a Monday (start of trading week). 현재 바가 월요일(주초)인지 확인.

        Returns:
            True if the current bar's date is a Monday. 현재 바의 날짜가 월요일이면 True.
        """
        if not self._currentBar:
            return False

        currentDate = self._currentBar.datetime
        return currentDate.weekday() == 0

    @property
    def symbols(self) -> List[str]:
        """List of all tradeable symbols (copy). 거래 가능 종목 코드 목록 (복사본)."""
        return self._symbols.copy()

    @property
    def currentBars(self) -> Dict[str, Bar]:
        """Current bar for each symbol (copy). 종목별 현재 바 (복사본)."""
        return self._currentBars.copy()


class MultiAssetEngine:
    """
    Multi-asset event-driven backtest engine for portfolio strategy evaluation.

    Orchestrates the full multi-asset backtesting pipeline: iterates through
    synchronized bars across all symbols, dispatches events to the strategy,
    processes orders through the broker, tracks per-symbol positions, and
    collects performance metrics.

    멀티에셋 이벤트 기반 백테스트 엔진 (포트폴리오 전략 평가).
    모든 종목에 대해 동기화된 바를 순회하며, 전략 이벤트 전달, 브로커 주문 처리,
    종목별 포지션 추적, 성과 지표 수집으로 구성된 전체 백테스팅 파이프라인을 관리합니다.

    Attributes:
        data (MultiDataFeed): Multi-symbol price data feed. 멀티 종목 가격 데이터 피드.
        strategy (MultiAssetStrategy): Multi-asset trading strategy instance.
            멀티에셋 트레이딩 전략 인스턴스.
        initialCash (float): Starting capital. 초기 자본금.
        broker (SimpleBroker): Broker simulator for order execution. 주문 체결 브로커 시뮬레이터.
        sizer (SimpleSizer): Position sizer for quantity calculation. 수량 계산 포지션 사이저.
        fillOnNextBar (bool): If True, orders fill at next bar's open price.
            True이면 주문이 다음 바 시가에 체결됩니다.

    Example:
        >>> from tradex.datafeed import FinanceDataReaderFeed, MultiDataFeed
        >>> from tradex.multiAssetEngine import MultiAssetEngine, MultiAssetStrategy
        >>>
        >>> feeds = {
        ...     '005930': FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31'),
        ...     '000660': FinanceDataReaderFeed('000660', '2020-01-01', '2024-12-31'),
        ... }
        >>> engine = MultiAssetEngine(
        ...     data=MultiDataFeed(feeds),
        ...     strategy=RotationStrategy(),
        ...     initialCash=100_000_000,
        ... )
        >>> result = engine.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        data: MultiDataFeed,
        strategy: MultiAssetStrategy,
        initialCash: float = 100_000_000,
        broker: SimpleBroker = None,
        sizer: SimpleSizer = None,
        fillOnNextBar: bool = True,
    ):
        """Initialize the multi-asset backtest engine. 멀티에셋 백테스트 엔진 초기화.

        Args:
            data: Multi-symbol data feed providing synchronized bars.
                동기화된 바를 제공하는 멀티 종목 데이터 피드.
            strategy: Multi-asset trading strategy to evaluate.
                평가할 멀티에셋 트레이딩 전략.
            initialCash: Starting capital in currency units (default: 100,000,000).
                초기 자본금 (기본값: 1억원).
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
        self._positionHistory: Dict[str, List[tuple]] = {}
        self._pendingOrders: List[Order] = []

    def run(self, verbose: bool = False) -> MultiAssetResult:
        """Run the multi-asset backtest simulation from start to end. 멀티에셋 백테스트 시뮬레이션 실행.

        Iterates through all synchronized bars, processes strategy signals,
        executes orders, tracks per-symbol positions, and builds the final result.

        Args:
            verbose: If True, print detailed execution logs to stdout.
                True이면 상세 체결 로그를 출력합니다.

        Returns:
            MultiAssetResult containing trades, equity curve, position history,
            and performance metrics.
            거래 내역, 자산 곡선, 포지션 이력, 성과 지표가 담긴 MultiAssetResult.
        """
        self.data.load()
        self._portfolio.reset()
        self._equityHistory.clear()
        self._positionHistory = {s: [] for s in self.data.symbols}
        self._pendingOrders.clear()

        self.strategy._setupMulti(self._portfolio, self.broker, self.data.symbols)
        self.strategy._setMultiFullData(self.data.getAllDataFrames())
        self.strategy.initialize()

        if verbose:
            print(f"[MultiAssetEngine] 시작: {self.data.symbols}")
            print(f"[MultiAssetEngine] 기간: {self.data.startDate} ~ {self.data.endDate}")
            print(f"[MultiAssetEngine] 총 {self.data.totalBars} 바")

        for bars in self.data:
            for symbol, bar in bars.items():
                self._portfolio.updatePrice(symbol, bar.close)

            if self.fillOnNextBar and self._pendingOrders:
                self._processPendingOrders(bars, verbose)

            self.strategy._updateMultiIndex(self.data.currentIndex)
            self.strategy._updateCurrentBars(bars)

            self.strategy.onBars(bars)

            if self.fillOnNextBar:
                self._queueOrders()
            else:
                self._processOrders(bars, verbose)

            firstBar = list(bars.values())[0] if bars else None
            if firstBar:
                self._equityHistory.append((firstBar.datetime, self._portfolio.equity))

            for symbol in self.data.symbols:
                position = self._portfolio.getPosition(symbol)
                qty = position.quantity if position else 0
                bar = bars.get(symbol)
                dt = bar.datetime if bar else None
                self._positionHistory[symbol].append((dt, qty))

        self.strategy.onEnd()

        return self._buildResult()

    def _queueOrders(self):
        """Move current bar's pending orders to the next-bar execution queue. 현재 바 주문을 다음 바 체결 대기열로 이동."""
        pendingOrders = self._portfolio.pendingOrders.copy()
        for order in pendingOrders:
            self._pendingOrders.append(order)
            self._portfolio._pendingOrders.remove(order)

    def _processPendingOrders(self, bars: Dict[str, Bar], verbose: bool = False):
        """Execute queued orders at the current bars' open prices. 대기 주문을 현재 바 시가에 체결.

        Args:
            bars: Mapping of symbol to its current Bar. 종목 코드와 현재 Bar의 매핑.
            verbose: If True, print fill details to stdout. True이면 체결 상세 내역을 출력.
        """
        for order in self._pendingOrders:
            bar = bars.get(order.symbol)
            if not bar:
                continue

            if order.quantity <= 0:
                order.quantity = self.sizer.calculate(
                    self._portfolio.equity / len(self.data.symbols),
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
                    f"[{bar.datetime}] {order.symbol} {order.side.value.upper()} "
                    f"{fill.fillQuantity} @ {fill.fillPrice:,.0f}"
                )

        self._pendingOrders.clear()

    def _processOrders(self, bars: Dict[str, Bar], verbose: bool = False):
        """Process pending orders at the current bars' close prices. 대기 주문을 현재 바 종가에 체결.

        Args:
            bars: Mapping of symbol to its current Bar. 종목 코드와 현재 Bar의 매핑.
            verbose: If True, print fill details to stdout. True이면 체결 상세 내역을 출력.
        """
        pendingOrders = self._portfolio.pendingOrders.copy()

        for order in pendingOrders:
            bar = bars.get(order.symbol)
            if not bar:
                continue

            if order.quantity <= 0:
                order.quantity = self.sizer.calculate(
                    self._portfolio.equity / len(self.data.symbols),
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
                    f"[{bar.datetime}] {order.symbol} {order.side.value.upper()} "
                    f"{fill.fillQuantity} @ {fill.fillPrice:,.0f}"
                )

    def _buildResult(self) -> MultiAssetResult:
        """Build the final MultiAssetResult from accumulated data. 누적된 데이터로 최종 MultiAssetResult 생성.

        Returns:
            MultiAssetResult populated with trades, equity curve, position history,
            and performance metrics.
            거래 내역, 자산 곡선, 포지션 이력, 성과 지표가 포함된 MultiAssetResult.
        """
        trades = self._portfolio.trades
        stats = self._portfolio.getTradeStats()

        equitySeries = pd.Series(
            [e[1] for e in self._equityHistory],
            index=pd.DatetimeIndex([e[0] for e in self._equityHistory])
        )

        positionSeries = {}
        for symbol, history in self._positionHistory.items():
            if history:
                positionSeries[symbol] = pd.Series(
                    [h[1] for h in history],
                    index=pd.DatetimeIndex([h[0] for h in history if h[0]])
                )

        metrics = self._calculateMetrics(equitySeries, trades)

        return MultiAssetResult(
            strategy=self.strategy.name,
            symbols=self.data.symbols,
            startDate=self.data.startDate,
            endDate=self.data.endDate,
            initialCash=self.initialCash,
            finalEquity=self._portfolio.equity,
            totalReturn=self._portfolio.totalReturn,
            totalTrades=stats['totalTrades'],
            winRate=stats['winRate'],
            trades=trades,
            equityCurve=equitySeries,
            metrics=metrics,
            positionHistory=positionSeries,
        )

    def _calculateMetrics(
        self,
        equityCurve: pd.Series,
        trades: List[Trade]
    ) -> dict:
        """Calculate comprehensive performance metrics from equity curve and trades. 자산 곡선과 거래 내역으로 종합 성과 지표 계산.

        Computes total return, annualized return, volatility, Sharpe ratio,
        maximum drawdown, win rate, and total trade count.

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

        return {
            'totalReturn': totalReturn * 100,
            'annualReturn': annualReturn * 100,
            'volatility': volatility * 100,
            'sharpeRatio': sharpeRatio,
            'maxDrawdown': maxDrawdown * 100,
            'winRate': len(wins) / len(trades) * 100 if trades else 0,
            'totalTrades': len(trades),
        }

    def __repr__(self) -> str:
        return (
            f"MultiAssetEngine({self.strategy.name}, "
            f"{len(self.data.symbols)} symbols, "
            f"cash={self.initialCash:,.0f})"
        )
