"""
Tradex Strategy Base Module - Abstract base class for all trading strategies.

Provides the foundational Strategy class that all user-defined and preset
strategies must inherit from. Implements lifecycle methods (initialize, onBar,
onOrderFill, onEnd), portfolio access, order management, and a comprehensive
suite of 40+ built-in technical indicator methods.

전략 베이스 모듈 - 모든 트레이딩 전략의 추상 기반 클래스.
라이프사이클 메서드, 포트폴리오 접근, 주문 관리, 40개 이상의
내장 기술 지표 메서드를 제공합니다.

Features:
    - Lifecycle methods: initialize, onBar, onOrderFill, onEnd
    - Portfolio access: positions, cash, equity
    - Order management: buy, sell, closePosition, cancelOrder
    - 40+ built-in indicators: SMA, EMA, RSI, MACD, Bollinger, ATR, etc.
    - Cross detection: crossover, crossunder

Usage:
    >>> from tradex.strategy.base import Strategy
    >>> from tradex.entities.bar import Bar
    >>>
    >>> class MyStrategy(Strategy):
    ...     def initialize(self):
    ...         self.fastPeriod = 10
    ...         self.slowPeriod = 30
    ...
    ...     def onBar(self, bar: Bar):
    ...         fast = self.sma(self.fastPeriod)
    ...         slow = self.sma(self.slowPeriod)
    ...         if fast and slow and fast > slow and not self.hasPosition(bar.symbol):
    ...             self.buy(bar.symbol)
    ...         elif fast and slow and fast < slow and self.hasPosition(bar.symbol):
    ...             self.sell(bar.symbol)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd

from tradex.entities.bar import Bar
from tradex.entities.order import Order, OrderSide, OrderType, TimeInForce
from tradex.entities.position import Position
from tradex.events.fill import FillEvent
from tradex.events.signal import SignalEvent, SignalType
from tradex.strategy.indicators import Indicators

if TYPE_CHECKING:
    from tradex.portfolio.portfolio import Portfolio
    from tradex.broker.simulator import BrokerSimulator


class Strategy(ABC):
    """
    Abstract base class for all Tradex trading strategies.

    Provides a React/Lumibot-style lifecycle with initialize(), onBar(),
    onOrderFill(), and onEnd() hooks. Subclass this to implement custom
    trading logic. Built-in indicator methods (sma, ema, rsi, macd, etc.)
    are available directly on the strategy instance.

    모든 Tradex 트레이딩 전략의 추상 기반 클래스.
    라이프사이클 훅과 내장 지표 메서드를 제공합니다.

    Attributes:
        name (str): Strategy name, defaults to class name.
        positions (Dict[str, Position]): Currently held positions.
        cash (float): Available cash balance.
        equity (float): Total equity (cash + positions).
        history (pd.DataFrame): Historical OHLCV data up to current bar.

    Example:
        >>> class GoldenCross(Strategy):
        ...     def initialize(self):
        ...         self.fast = 10
        ...         self.slow = 30
        ...
        ...     def onBar(self, bar: Bar):
        ...         if self.sma(self.fast) > self.sma(self.slow):
        ...             if not self.hasPosition(bar.symbol):
        ...                 self.buy(bar.symbol)
        ...         else:
        ...             self.closePosition(bar.symbol)
    """

    def __init__(self):
        self.name: str = self.__class__.__name__

        self._portfolio: Optional['Portfolio'] = None
        self._broker: Optional['BrokerSimulator'] = None
        self._history: Optional[pd.DataFrame] = None
        self._currentBar: Optional[Bar] = None
        self._indicators: Optional[Indicators] = None

    def _setup(self, portfolio: 'Portfolio', broker: 'BrokerSimulator'):
        """Initialize internal references to portfolio and broker (called by engine). / 엔진에서 호출하는 내부 설정."""
        self._portfolio = portfolio
        self._broker = broker
        self._indicators = Indicators()

    def _setFullData(self, fullData: pd.DataFrame):
        """Set full dataset for indicator pre-computation and caching. / 지표 캐싱을 위한 전체 데이터 설정."""
        self._fullData = fullData
        if self._indicators:
            self._indicators.setFullData(fullData)

    def _updateIndex(self, index: int):
        """Update current bar index for optimized indicator lookup. / 최적화된 지표 조회를 위한 현재 인덱스 업데이트."""
        if self._indicators:
            self._indicators.setIndex(index)

    def _updateHistory(self, history: pd.DataFrame):
        """Update rolling history DataFrame (legacy compatibility). / 히스토리 업데이트 (레거시 호환)."""
        self._history = history
        if self._indicators:
            self._indicators.setData(history)

    def _updateCurrentBar(self, bar: Bar):
        """Update the current bar reference. / 현재 바 업데이트."""
        self._currentBar = bar

    def initialize(self):
        """
        Initialize strategy state before backtest begins.

        Called once at the start of a backtest. Override to set parameters,
        configure indicators, or initialize internal state.

        백테스트 시작 시 한 번 호출됩니다.
        오버라이드하여 파라미터 설정, 지표 초기화 등을 수행합니다.
        """
        pass

    @abstractmethod
    def onBar(self, bar: Bar):
        """
        Core trading logic called on each new bar.

        Must be implemented by all subclasses. This is where buy/sell
        decisions are made based on indicator values and bar data.

        새로운 바마다 호출되는 핵심 트레이딩 로직입니다. 모든 서브클래스에서 구현해야 합니다.

        Args:
            bar: Current OHLCV bar data. / 현재 OHLCV 바 데이터.
        """
        pass

    def onOrderFill(self, fill: FillEvent):
        """
        Called when an order is filled.

        Override to implement custom logic on trade execution (e.g., logging,
        position tracking, or adjusting stop-loss levels).

        주문이 체결되었을 때 호출됩니다.

        Args:
            fill: Fill event containing execution details. / 체결 상세 정보를 담은 이벤트.
        """
        pass

    def onEnd(self):
        """Called when the backtest completes. / 백테스트 종료 시 호출."""
        pass

    @property
    def positions(self) -> Dict[str, Position]:
        """Return currently held positions. / 현재 보유 포지션 반환."""
        if self._portfolio:
            return self._portfolio.positions
        return {}

    @property
    def cash(self) -> float:
        """Return available cash balance. / 사용 가능 현금 반환."""
        if self._portfolio:
            return self._portfolio.cash
        return 0.0

    @property
    def equity(self) -> float:
        """Return total equity (cash + position value). / 총 자산 (현금 + 포지션 가치) 반환."""
        if self._portfolio:
            return self._portfolio.equity
        return 0.0

    @property
    def history(self) -> pd.DataFrame:
        """Return historical OHLCV data up to the current bar. / 현재까지의 과거 데이터 반환."""
        if self._history is not None:
            return self._history.copy()
        return pd.DataFrame()

    def getPosition(self, symbol: str) -> Optional[Position]:
        """Return the position for a specific symbol, or None. / 특정 종목 포지션 조회."""
        if self._portfolio:
            return self._portfolio.getPosition(symbol)
        return None

    def hasPosition(self, symbol: str) -> bool:
        """Check whether a position is held for the given symbol. / 해당 종목 포지션 보유 여부 확인."""
        if self._portfolio:
            return self._portfolio.hasPosition(symbol)
        return False

    def getPositionQuantity(self, symbol: str) -> float:
        """Return the quantity held for a specific symbol. / 특정 종목 포지션 수량 반환."""
        if self._portfolio:
            return self._portfolio.getPositionQuantity(symbol)
        return 0.0

    def buy(
        self,
        symbol: str,
        quantity: float = None,
        price: float = None,
        orderType: OrderType = OrderType.MARKET,
        stopLoss: float = None,
        takeProfit: float = None,
    ) -> Order:
        """
        Submit a buy order.

        매수 주문을 제출합니다.

        Args:
            symbol: Ticker symbol. / 종목 코드.
            quantity: Order quantity (None lets the sizer decide). / 수량 (None이면 사이저가 계산).
            price: Limit price (for limit orders). / 지정가 (limit 주문용).
            orderType: Order type (MARKET, LIMIT, etc.). / 주문 유형.
            stopLoss: Stop-loss price. / 손절가.
            takeProfit: Take-profit price. / 익절가.

        Returns:
            Order: The created order object. / 생성된 주문 객체.
        """
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            orderType=orderType,
            quantity=quantity or 0,
            price=price,
            createdAt=self._currentBar.datetime if self._currentBar else None
        )

        if self._portfolio:
            self._portfolio.addOrder(order)

        return order

    def sell(
        self,
        symbol: str,
        quantity: float = None,
        price: float = None,
        orderType: OrderType = OrderType.MARKET,
    ) -> Order:
        """
        Submit a sell order.

        매도 주문을 제출합니다.

        Args:
            symbol: Ticker symbol. / 종목 코드.
            quantity: Order quantity (None sells entire position). / 수량 (None이면 전량 매도).
            price: Limit price (for limit orders). / 지정가 (limit 주문용).
            orderType: Order type (MARKET, LIMIT, etc.). / 주문 유형.

        Returns:
            Order: The created order object. / 생성된 주문 객체.
        """
        if quantity is None:
            quantity = self.getPositionQuantity(symbol)

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            orderType=orderType,
            quantity=quantity,
            price=price,
            createdAt=self._currentBar.datetime if self._currentBar else None
        )

        if self._portfolio:
            self._portfolio.addOrder(order)

        return order

    def closePosition(self, symbol: str) -> Optional[Order]:
        """Close the entire position for a symbol. / 해당 종목 포지션 전량 청산."""
        quantity = self.getPositionQuantity(symbol)
        if quantity > 0:
            return self.sell(symbol, quantity)
        return None

    def closeAllPositions(self) -> List[Order]:
        """Close all open positions. / 모든 보유 포지션 청산."""
        orders = []
        for symbol in list(self.positions.keys()):
            order = self.closePosition(symbol)
            if order:
                orders.append(order)
        return orders

    def cancelOrder(self, orderId: str) -> bool:
        """Cancel a pending order by ID. / 주문 ID로 미체결 주문 취소."""
        if self._portfolio:
            return self._portfolio.cancelOrder(orderId)
        return False

    def cancelAllOrders(self, symbol: str = None):
        """Cancel all pending orders, optionally filtered by symbol. / 모든 미체결 주문 취소 (종목 필터 선택 가능)."""
        if self._portfolio:
            self._portfolio.cancelAllOrders(symbol)

    def sma(self, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """Compute Simple Moving Average (SMA). / 단순 이동평균 계산."""
        if self._indicators:
            return self._indicators.sma(period, column, offset)
        return None

    def ema(self, period: int, column: str = 'close', offset: int = 0) -> Optional[float]:
        """Compute Exponential Moving Average (EMA). / 지수 이동평균 계산."""
        if self._indicators:
            return self._indicators.ema(period, column, offset)
        return None

    def rsi(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Relative Strength Index (RSI, 0-100). / 상대 강도 지수 계산."""
        if self._indicators:
            return self._indicators.rsi(period, offset)
        return None

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9, offset: int = 0) -> tuple:
        """Compute MACD returning (macd_line, signal_line, histogram). / MACD 계산 (MACD선, 시그널선, 히스토그램)."""
        if self._indicators:
            return self._indicators.macd(fast, slow, signal, offset)
        return (None, None, None)

    def bollinger(self, period: int = 20, std: float = 2.0, offset: int = 0) -> tuple:
        """Compute Bollinger Bands returning (upper, middle, lower). / 볼린저 밴드 계산 (상단, 중간, 하단)."""
        if self._indicators:
            return self._indicators.bollinger(period, std, offset)
        return (None, None, None)

    def atr(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Average True Range (ATR). / 평균 실제 범위 계산."""
        if self._indicators:
            return self._indicators.atr(period, offset)
        return None

    def stochastic(self, kPeriod: int = 14, dPeriod: int = 3) -> tuple:
        """Compute Stochastic Oscillator returning (%K, %D). / 스토캐스틱 오실레이터 계산."""
        if self._indicators:
            return self._indicators.stochastic(kPeriod, dPeriod)
        return (None, None)

    def adx(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Average Directional Index (ADX) for trend strength. / ADX (추세 강도 지표) 계산."""
        if self._indicators:
            return self._indicators.adx(period, offset)
        return None

    def adxWithDi(self, period: int = 14, offset: int = 0) -> tuple:
        """Compute ADX with directional indicators returning (ADX, +DI, -DI). / ADX와 방향 지표 계산."""
        if self._indicators:
            return self._indicators.adxWithDi(period, offset)
        return (None, None, None)

    def obv(self, offset: int = 0) -> Optional[float]:
        """Compute On Balance Volume (OBV). / OBV (거래량 기반 추세 지표) 계산."""
        if self._indicators:
            return self._indicators.obv(offset)
        return None

    def williamsR(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Williams %R oscillator (-100 to 0). / Williams %R 오실레이터 계산."""
        if self._indicators:
            return self._indicators.williamsR(period, offset)
        return None

    def cci(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Commodity Channel Index (CCI). / CCI (상품 채널 지수) 계산."""
        if self._indicators:
            return self._indicators.cci(period, offset)
        return None

    def vwap(self, offset: int = 0) -> Optional[float]:
        """Compute Volume Weighted Average Price (VWAP). / 거래량 가중 평균가 계산."""
        if self._indicators:
            return self._indicators.vwap(offset)
        return None

    def mfi(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Money Flow Index (MFI), a volume-weighted RSI. / MFI (거래량 기반 RSI) 계산."""
        if self._indicators:
            return self._indicators.mfi(period, offset)
        return None

    def roc(self, period: int = 12, offset: int = 0) -> Optional[float]:
        """Compute Rate of Change (ROC) as percentage. / 변화율(%) 계산."""
        if self._indicators:
            return self._indicators.roc(period, offset)
        return None

    def momentum(self, period: int = 10, offset: int = 0) -> Optional[float]:
        """Compute price momentum (current close minus N-period-ago close). / 모멘텀 계산."""
        if self._indicators:
            return self._indicators.momentum(period, offset)
        return None

    def psar(self, afStart: float = 0.02, afStep: float = 0.02, afMax: float = 0.2, offset: int = 0) -> Optional[float]:
        """Compute Parabolic SAR for trend following. / 파라볼릭 SAR (추세 추종 지표) 계산."""
        if self._indicators:
            return self._indicators.psar(afStart, afStep, afMax, offset)
        return None

    def supertrend(self, period: int = 10, multiplier: float = 3.0, offset: int = 0) -> tuple:
        """Compute Supertrend returning (value, direction: 1=bullish, -1=bearish). / 슈퍼트렌드 계산."""
        if self._indicators:
            return self._indicators.supertrend(period, multiplier, offset)
        return (None, None)

    def ichimoku(self, tenkanPeriod: int = 9, kijunPeriod: int = 26, senkouBPeriod: int = 52, offset: int = 0) -> tuple:
        """Compute Ichimoku Cloud returning (tenkan, kijun, senkouA, senkouB, chikou). / 일목균형표 계산."""
        if self._indicators:
            return self._indicators.ichimoku(tenkanPeriod, kijunPeriod, senkouBPeriod, offset)
        return (None, None, None, None, None)

    def keltner(self, period: int = 20, atrPeriod: int = 10, multiplier: float = 2.0, offset: int = 0) -> tuple:
        """Compute Keltner Channel returning (upper, middle, lower). / 켈트너 채널 계산."""
        if self._indicators:
            return self._indicators.keltner(period, atrPeriod, multiplier, offset)
        return (None, None, None)

    def donchian(self, period: int = 20, offset: int = 0) -> tuple:
        """Compute Donchian Channel returning (upper, middle, lower). / 돈치안 채널 계산."""
        if self._indicators:
            return self._indicators.donchian(period, offset)
        return (None, None, None)

    def trix(self, period: int = 15, signalPeriod: int = 9, offset: int = 0) -> tuple:
        """Compute TRIX triple-smoothed EMA returning (trix, signal). / TRIX 계산."""
        if self._indicators:
            return self._indicators.trix(period, signalPeriod, offset)
        return (None, None)

    def dpo(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Detrended Price Oscillator (DPO). / 추세 제거 가격 오실레이터 계산."""
        if self._indicators:
            return self._indicators.dpo(period, offset)
        return None

    def cmo(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Chande Momentum Oscillator (CMO, -100 to 100). / CMO 계산."""
        if self._indicators:
            return self._indicators.cmo(period, offset)
        return None

    def ulcer(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Ulcer Index for downside risk measurement. / 울서 인덱스 (하락 리스크 측정) 계산."""
        if self._indicators:
            return self._indicators.ulcer(period, offset)
        return None

    def elderRay(self, period: int = 13, offset: int = 0) -> tuple:
        """Compute Elder Ray Index returning (bull_power, bear_power). / 엘더 레이 지수 계산."""
        if self._indicators:
            return self._indicators.elderRay(period, offset)
        return (None, None)

    def chaikin(self, fastPeriod: int = 3, slowPeriod: int = 10, offset: int = 0) -> Optional[float]:
        """Compute Chaikin Oscillator (A/D Line based). / 차이킨 오실레이터 계산."""
        if self._indicators:
            return self._indicators.chaikin(fastPeriod, slowPeriod, offset)
        return None

    def adl(self, offset: int = 0) -> Optional[float]:
        """Compute Accumulation/Distribution Line (A/D Line). / 누적/분배 라인 계산."""
        if self._indicators:
            return self._indicators.adl(offset)
        return None

    def emv(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Ease of Movement (EMV). / EMV (가격 변화 용이성) 계산."""
        if self._indicators:
            return self._indicators.emv(period, offset)
        return None

    def forceIndex(self, period: int = 13, offset: int = 0) -> Optional[float]:
        """Compute Force Index (price change * volume). / 포스 인덱스 계산."""
        if self._indicators:
            return self._indicators.forceIndex(period, offset)
        return None

    def nvi(self, offset: int = 0) -> Optional[float]:
        """Compute Negative Volume Index (NVI). / NVI (음의 거래량 지수) 계산."""
        if self._indicators:
            return self._indicators.nvi(offset)
        return None

    def pvi(self, offset: int = 0) -> Optional[float]:
        """Compute Positive Volume Index (PVI). / PVI (양의 거래량 지수) 계산."""
        if self._indicators:
            return self._indicators.pvi(offset)
        return None

    def vroc(self, period: int = 14, offset: int = 0) -> Optional[float]:
        """Compute Volume Rate of Change (VROC). / 거래량 변화율 계산."""
        if self._indicators:
            return self._indicators.vroc(period, offset)
        return None

    def pvt(self, offset: int = 0) -> Optional[float]:
        """Compute Price Volume Trend (PVT). / 가격 거래량 추세 계산."""
        if self._indicators:
            return self._indicators.pvt(offset)
        return None

    def zigzag(self, threshold: float = 5.0, offset: int = 0) -> Optional[float]:
        """Compute ZigZag pivot points for trend reversal detection. / 지그재그 추세 전환점 계산."""
        if self._indicators:
            return self._indicators.zigzag(threshold, offset)
        return None

    def hma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Hull Moving Average (HMA) for reduced lag. / HMA (헐 이동평균) 계산."""
        if self._indicators:
            return self._indicators.hma(period, offset)
        return None

    def tema(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Triple Exponential Moving Average (TEMA). / TEMA (삼중 지수 이동평균) 계산."""
        if self._indicators:
            return self._indicators.tema(period, offset)
        return None

    def dema(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Double Exponential Moving Average (DEMA). / DEMA (이중 지수 이동평균) 계산."""
        if self._indicators:
            return self._indicators.dema(period, offset)
        return None

    def wma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Weighted Moving Average (WMA). / 가중 이동평균 계산."""
        if self._indicators:
            return self._indicators.wma(period, offset)
        return None

    def vwma(self, period: int = 20, offset: int = 0) -> Optional[float]:
        """Compute Volume Weighted Moving Average (VWMA). / 거래량 가중 이동평균 계산."""
        if self._indicators:
            return self._indicators.vwma(period, offset)
        return None

    def alma(self, period: int = 20, sigma: float = 6.0, offsetFactor: float = 0.85, offset: int = 0) -> Optional[float]:
        """Compute Arnaud Legoux Moving Average (ALMA). / ALMA (아르노 르구 이동평균) 계산."""
        if self._indicators:
            return self._indicators.alma(period, sigma, offsetFactor, offset)
        return None

    def pivotPoints(self, offset: int = 0) -> tuple:
        """Compute Pivot Points returning (Pivot, R1, R2, R3, S1, S2, S3). / 피봇 포인트 계산."""
        if self._indicators:
            return self._indicators.pivotPoints(offset)
        return (None, None, None, None, None, None, None)

    def fibonacciRetracement(self, period: int = 50, offset: int = 0) -> tuple:
        """Compute Fibonacci Retracement levels (0%, 23.6%, 38.2%, 50%, 61.8%). / 피보나치 되돌림 계산."""
        if self._indicators:
            return self._indicators.fibonacciRetracement(period, offset)
        return (None, None, None, None, None)

    def crossover(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Detect if series1 crossed above series2. / series1이 series2를 상향 돌파했는지 감지."""
        if self._indicators:
            return self._indicators.crossover(series1, series2)
        return False

    def crossunder(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Detect if series1 crossed below series2. / series1이 series2를 하향 돌파했는지 감지."""
        if self._indicators:
            return self._indicators.crossunder(series1, series2)
        return False

    def log(self, message: str):
        """Print a timestamped log message. / 타임스탬프가 포함된 로그 메시지 출력."""
        timestamp = self._currentBar.datetime if self._currentBar else ""
        print(f"[{timestamp}] [{self.name}] {message}")

    def __repr__(self) -> str:
        return f"Strategy({self.name})"
