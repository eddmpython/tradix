"""Tradix Signal Predictor Module.

Technical-indicator-based trading signal predictor that leverages Tradix's 70+
built-in indicators to generate, score, and synthesize buy/sell/hold signals
through a consensus-based approach.

기술 지표 기반 매매 신호 예측기 - Tradix의 70개 이상의 기술 지표를 활용하여
매매 신호를 생성하고 컨센서스 방식으로 통합 분석합니다.

Features:
    - Single and composite signal generation across 7 strategy categories
    - Signal strength scoring (0.0 to 1.0) for each individual indicator
    - Consensus-based signal integration with configurable threshold
    - Signal history generation for lookback analysis
    - Seamless integration with Tradix backtesting engine

Usage:
    from tradix.signals import SignalPredictor

    predictor = SignalPredictor(df)
    result = predictor.predict()

    print(result.signal)     # 1: buy, -1: sell, 0: hold
    print(result.strength)   # 0.0 ~ 1.0
    print(result.reasons)    # List of signal reasons
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np

from tradix.strategy.indicators import Indicators


class SignalType(Enum):
    """Trading signal type enumeration.

    매매 신호 유형 열거형.

    Attributes:
        BUY: Buy signal (+1) (매수 신호).
        SELL: Sell signal (-1) (매도 신호).
        HOLD: Hold / no-action signal (0) (관망 신호).
    """
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class SignalResult:
    """Result of a signal prediction combining multiple indicator signals.

    여러 지표 신호를 통합한 신호 예측 결과.

    Attributes:
        signal: Final signal value (1=buy, -1=sell, 0=hold) (최종 신호 값).
        signalType: SignalType enum value (신호 유형 열거값).
        strength: Signal strength from 0.0 to 1.0 (신호 강도).
        confidence: Confidence level based on consensus ratio (컨센서스 기반 신뢰도).
        reasons: List of human-readable reasons for the signal (신호 발생 사유 목록).
        details: Detailed breakdown of individual signals and scores (상세 내역).
        timestamp: Optional timestamp of the signal (신호 시점).
    """
    signal: int
    signalType: SignalType
    strength: float
    confidence: float
    reasons: List[str]
    details: Dict[str, Any]
    timestamp: Optional[pd.Timestamp] = None

    @property
    def isBuy(self) -> bool:
        return self.signal == 1

    @property
    def isSell(self) -> bool:
        return self.signal == -1

    @property
    def isHold(self) -> bool:
        return self.signal == 0

    def __repr__(self) -> str:
        signalStr = {1: "BUY", -1: "SELL", 0: "HOLD"}[self.signal]
        return f"SignalResult({signalStr}, strength={self.strength:.2f}, confidence={self.confidence:.2f})"


@dataclass
class SignalConfig:
    """Configuration parameters for signal generation thresholds.

    신호 생성에 사용되는 임계값 설정.

    Attributes:
        rsiOversold: RSI oversold threshold (RSI 과매도 기준, 기본 30).
        rsiOverbought: RSI overbought threshold (RSI 과매수 기준, 기본 70).
        macdSignalThreshold: MACD signal line threshold (MACD 시그널 임계값).
        bollingerStd: Bollinger Band standard deviation multiplier (볼린저 밴드 표준편차 배수).
        adxTrendThreshold: ADX threshold for trend strength (ADX 추세 강도 기준).
        volumeSurgeRatio: Volume surge detection ratio (거래량 급증 판단 비율).
        momentumPeriod: Momentum calculation period (모멘텀 계산 기간).
        trendMaPeriod: Trend moving average period (추세 이동평균 기간).
    """
    rsiOversold: float = 30.0
    rsiOverbought: float = 70.0
    macdSignalThreshold: float = 0.0
    bollingerStd: float = 2.0
    adxTrendThreshold: float = 25.0
    volumeSurgeRatio: float = 2.0
    momentumPeriod: int = 10
    trendMaPeriod: int = 50


class SignalPredictor:
    """Technical-indicator-based trading signal predictor.

    Leverages Tradix's Indicators class to generate signals from 7 strategy
    categories (trend, momentum, mean-reversion, breakout, volume, volatility,
    forecast) and synthesizes them via a configurable consensus mechanism.

    Tradix의 Indicators 클래스를 활용하여 7개 전략 범주의 기술 지표 기반
    매매 신호를 생성하고, 컨센서스 메커니즘으로 통합합니다.

    Attributes:
        df: OHLCV DataFrame (OHLCV 데이터).
        config: SignalConfig instance (신호 설정).
        indicators: Tradix Indicators engine instance (지표 엔진).
        AVAILABLE_STRATEGIES: List of supported strategy names (지원 전략 목록).

    Example:
        >>> predictor = SignalPredictor(df)
        >>> result = predictor.predict()
        >>> result = predictor.predict(strategies=['trend', 'momentum'])
        >>> all_signals = predictor.getAllSignals()
    """

    AVAILABLE_STRATEGIES = [
        'trend',
        'momentum',
        'meanReversion',
        'breakout',
        'volume',
        'volatility',
        'forecast',
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[SignalConfig] = None,
    ):
        """Initialize the signal predictor.

        Args:
            df: OHLCV DataFrame with required columns: open, high, low, close, volume.
                OHLCV DataFrame (필수 컬럼: open, high, low, close, volume).
            config: Signal generation configuration. Defaults to SignalConfig()
                    (신호 생성 설정).
        """
        self.df = df
        self.config = config or SignalConfig()
        self.indicators = Indicators()
        self.indicators.setData(df)
        self.indicators.setFullData(df)
        self.indicators.setIndex(len(df))

        self._signals: Dict[str, int] = {}
        self._strengths: Dict[str, float] = {}
        self._reasons: Dict[str, str] = {}

    def predict(
        self,
        strategies: Optional[List[str]] = None,
        consensusThreshold: float = 0.5,
    ) -> SignalResult:
        """Generate a consensus trading signal from multiple indicator strategies.

        Args:
            strategies: List of strategy names to use. If None, all available
                        strategies are used (사용할 전략 리스트, None이면 전체).
            consensusThreshold: Fraction of signals required for consensus
                                (0.5 = majority) (컨센서스 임계값).

        Returns:
            SignalResult with integrated signal, strength, confidence, and reasons.
        """
        if strategies is None:
            strategies = self.AVAILABLE_STRATEGIES

        self._signals = {}
        self._strengths = {}
        self._reasons = {}

        for strategy in strategies:
            if strategy == 'trend':
                self._analyzeTrend()
            elif strategy == 'momentum':
                self._analyzeMomentum()
            elif strategy == 'meanReversion':
                self._analyzeMeanReversion()
            elif strategy == 'breakout':
                self._analyzeBreakout()
            elif strategy == 'volume':
                self._analyzeVolume()
            elif strategy == 'volatility':
                self._analyzeVolatility()
            elif strategy == 'forecast':
                self._analyzeForecast()

        return self._buildConsensus(consensusThreshold)

    def getAllSignals(self) -> Dict[str, Dict[str, Any]]:
        """Return all individual indicator signals with their details.

        Returns:
            Dict mapping signal name to dict with 'signal', 'strength', 'reason'.
        """
        if not self._signals:
            self.predict()

        result = {}
        for name in self._signals:
            result[name] = {
                'signal': self._signals[name],
                'strength': self._strengths.get(name, 0.5),
                'reason': self._reasons.get(name, ''),
            }
        return result

    def _analyzeTrend(self):
        """Generate trend-following signals from SMA/EMA crossovers and ADX."""
        sma20 = self.indicators.sma(20)
        sma50 = self.indicators.sma(50)
        sma200 = self.indicators.sma(200)
        ema12 = self.indicators.ema(12)
        ema26 = self.indicators.ema(26)
        adx = self.indicators.adx(14)
        close = self.df['close'].iloc[-1]

        if sma20 is None or sma50 is None:
            return

        if sma20 > sma50 and close > sma20:
            self._signals['sma_cross'] = 1
            self._strengths['sma_cross'] = 0.7
            self._reasons['sma_cross'] = f"SMA20({sma20:.0f}) > SMA50({sma50:.0f}), 상승 추세"
        elif sma20 < sma50 and close < sma20:
            self._signals['sma_cross'] = -1
            self._strengths['sma_cross'] = 0.7
            self._reasons['sma_cross'] = f"SMA20({sma20:.0f}) < SMA50({sma50:.0f}), 하락 추세"
        else:
            self._signals['sma_cross'] = 0
            self._strengths['sma_cross'] = 0.3
            self._reasons['sma_cross'] = "이평선 혼조"

        if ema12 is not None and ema26 is not None:
            if ema12 > ema26:
                self._signals['ema_cross'] = 1
                self._strengths['ema_cross'] = 0.6
                self._reasons['ema_cross'] = f"EMA12 > EMA26, 단기 상승"
            elif ema12 < ema26:
                self._signals['ema_cross'] = -1
                self._strengths['ema_cross'] = 0.6
                self._reasons['ema_cross'] = f"EMA12 < EMA26, 단기 하락"
            else:
                self._signals['ema_cross'] = 0
                self._strengths['ema_cross'] = 0.3
                self._reasons['ema_cross'] = "EMA 중립"

        if adx is not None:
            if adx > self.config.adxTrendThreshold:
                adxStrength = min(1.0, adx / 50)
                self._strengths['sma_cross'] *= adxStrength
                self._strengths['ema_cross'] = self._strengths.get('ema_cross', 0.5) * adxStrength

        if sma200 is not None:
            if close > sma200:
                self._signals['sma200_trend'] = 1
                self._strengths['sma200_trend'] = 0.8
                self._reasons['sma200_trend'] = f"가격({close:.0f}) > SMA200({sma200:.0f}), 장기 상승장"
            else:
                self._signals['sma200_trend'] = -1
                self._strengths['sma200_trend'] = 0.8
                self._reasons['sma200_trend'] = f"가격({close:.0f}) < SMA200({sma200:.0f}), 장기 하락장"

    def _analyzeMomentum(self):
        """Generate momentum signals from RSI, MACD, Stochastic, and ROC."""
        rsi = self.indicators.rsi(14)
        macdLine, signalLine, histogram = self.indicators.macd()
        stochK, stochD = self.indicators.stochastic()
        roc = self.indicators.roc(12)
        momentum = self.indicators.momentum(self.config.momentumPeriod)

        if rsi is not None:
            if rsi < self.config.rsiOversold:
                self._signals['rsi'] = 1
                self._strengths['rsi'] = 0.8
                self._reasons['rsi'] = f"RSI({rsi:.1f}) 과매도 구간 - 반등 기대"
            elif rsi > self.config.rsiOverbought:
                self._signals['rsi'] = -1
                self._strengths['rsi'] = 0.8
                self._reasons['rsi'] = f"RSI({rsi:.1f}) 과매수 구간 - 조정 기대"
            else:
                self._signals['rsi'] = 0
                self._strengths['rsi'] = 0.4
                self._reasons['rsi'] = f"RSI({rsi:.1f}) 중립 구간"

        if macdLine is not None and signalLine is not None:
            if macdLine > signalLine and histogram > 0:
                self._signals['macd'] = 1
                self._strengths['macd'] = 0.7
                self._reasons['macd'] = f"MACD 골든크로스, 히스토그램 양수"
            elif macdLine < signalLine and histogram < 0:
                self._signals['macd'] = -1
                self._strengths['macd'] = 0.7
                self._reasons['macd'] = f"MACD 데드크로스, 히스토그램 음수"
            else:
                self._signals['macd'] = 0
                self._strengths['macd'] = 0.4
                self._reasons['macd'] = "MACD 중립"

        if stochK is not None and stochD is not None:
            if stochK < 20 and stochK > stochD:
                self._signals['stoch'] = 1
                self._strengths['stoch'] = 0.6
                self._reasons['stoch'] = f"스토캐스틱(%K={stochK:.1f}) 과매도 탈출"
            elif stochK > 80 and stochK < stochD:
                self._signals['stoch'] = -1
                self._strengths['stoch'] = 0.6
                self._reasons['stoch'] = f"스토캐스틱(%K={stochK:.1f}) 과매수 이탈"
            else:
                self._signals['stoch'] = 0
                self._strengths['stoch'] = 0.3
                self._reasons['stoch'] = "스토캐스틱 중립"

        if roc is not None:
            if roc > 5:
                self._signals['roc'] = 1
                self._strengths['roc'] = 0.5
                self._reasons['roc'] = f"ROC({roc:.1f}%) 강한 상승 모멘텀"
            elif roc < -5:
                self._signals['roc'] = -1
                self._strengths['roc'] = 0.5
                self._reasons['roc'] = f"ROC({roc:.1f}%) 강한 하락 모멘텀"
            else:
                self._signals['roc'] = 0
                self._strengths['roc'] = 0.3
                self._reasons['roc'] = f"ROC({roc:.1f}%) 중립"

    def _analyzeMeanReversion(self):
        """Generate mean-reversion signals from Bollinger Bands, Williams %R, and CCI."""
        upper, middle, lower = self.indicators.bollinger(20, self.config.bollingerStd)
        close = self.df['close'].iloc[-1]
        williamsR = self.indicators.williamsR(14)
        cci = self.indicators.cci(20)

        if upper is not None and lower is not None:
            bbPosition = (close - lower) / (upper - lower) if upper != lower else 0.5

            if close <= lower:
                self._signals['bollinger'] = 1
                self._strengths['bollinger'] = 0.8
                self._reasons['bollinger'] = f"볼린저 하단({lower:.0f}) 터치 - 반등 기대"
            elif close >= upper:
                self._signals['bollinger'] = -1
                self._strengths['bollinger'] = 0.8
                self._reasons['bollinger'] = f"볼린저 상단({upper:.0f}) 터치 - 조정 기대"
            elif bbPosition < 0.2:
                self._signals['bollinger'] = 1
                self._strengths['bollinger'] = 0.5
                self._reasons['bollinger'] = f"볼린저 밴드 하단 근접 ({bbPosition:.1%})"
            elif bbPosition > 0.8:
                self._signals['bollinger'] = -1
                self._strengths['bollinger'] = 0.5
                self._reasons['bollinger'] = f"볼린저 밴드 상단 근접 ({bbPosition:.1%})"
            else:
                self._signals['bollinger'] = 0
                self._strengths['bollinger'] = 0.3
                self._reasons['bollinger'] = f"볼린저 밴드 중립 ({bbPosition:.1%})"

        if williamsR is not None:
            if williamsR < -80:
                self._signals['williams_r'] = 1
                self._strengths['williams_r'] = 0.6
                self._reasons['williams_r'] = f"Williams %R({williamsR:.1f}) 과매도"
            elif williamsR > -20:
                self._signals['williams_r'] = -1
                self._strengths['williams_r'] = 0.6
                self._reasons['williams_r'] = f"Williams %R({williamsR:.1f}) 과매수"
            else:
                self._signals['williams_r'] = 0
                self._strengths['williams_r'] = 0.3
                self._reasons['williams_r'] = f"Williams %R({williamsR:.1f}) 중립"

        if cci is not None:
            if cci < -100:
                self._signals['cci'] = 1
                self._strengths['cci'] = 0.6
                self._reasons['cci'] = f"CCI({cci:.1f}) 과매도"
            elif cci > 100:
                self._signals['cci'] = -1
                self._strengths['cci'] = 0.6
                self._reasons['cci'] = f"CCI({cci:.1f}) 과매수"
            else:
                self._signals['cci'] = 0
                self._strengths['cci'] = 0.3
                self._reasons['cci'] = f"CCI({cci:.1f}) 중립"

    def _analyzeBreakout(self):
        """Generate breakout signals from Donchian, Keltner channels, and Supertrend."""
        donchianUpper, donchianMid, donchianLower = self.indicators.donchian(20)
        close = self.df['close'].iloc[-1]
        high = self.df['high'].iloc[-1]
        low = self.df['low'].iloc[-1]
        atr = self.indicators.atr(14)

        if donchianUpper is not None and donchianLower is not None:
            if high >= donchianUpper:
                self._signals['donchian'] = 1
                self._strengths['donchian'] = 0.7
                self._reasons['donchian'] = f"도니치안 채널 상단({donchianUpper:.0f}) 돌파"
            elif low <= donchianLower:
                self._signals['donchian'] = -1
                self._strengths['donchian'] = 0.7
                self._reasons['donchian'] = f"도니치안 채널 하단({donchianLower:.0f}) 돌파"
            else:
                self._signals['donchian'] = 0
                self._strengths['donchian'] = 0.3
                self._reasons['donchian'] = "도니치안 채널 내부"

        keltnerUpper, keltnerMid, keltnerLower = self.indicators.keltner(20, 10, 2.0)
        if keltnerUpper is not None and keltnerLower is not None:
            if close > keltnerUpper:
                self._signals['keltner'] = 1
                self._strengths['keltner'] = 0.6
                self._reasons['keltner'] = f"켈트너 채널 상단({keltnerUpper:.0f}) 돌파"
            elif close < keltnerLower:
                self._signals['keltner'] = -1
                self._strengths['keltner'] = 0.6
                self._reasons['keltner'] = f"켈트너 채널 하단({keltnerLower:.0f}) 돌파"
            else:
                self._signals['keltner'] = 0
                self._strengths['keltner'] = 0.3
                self._reasons['keltner'] = "켈트너 채널 내부"

        supertrendVal, supertrendDir = self.indicators.supertrend(10, 3.0)
        if supertrendDir is not None:
            if supertrendDir == 1:
                self._signals['supertrend'] = 1
                self._strengths['supertrend'] = 0.7
                self._reasons['supertrend'] = f"슈퍼트렌드 상승 ({supertrendVal:.0f})"
            else:
                self._signals['supertrend'] = -1
                self._strengths['supertrend'] = 0.7
                self._reasons['supertrend'] = f"슈퍼트렌드 하락 ({supertrendVal:.0f})"

    def _analyzeVolume(self):
        """Generate volume-based signals from volume surge, MFI, and Chaikin oscillator."""
        obv = self.indicators.obv()
        mfi = self.indicators.mfi(14)
        chaikin = self.indicators.chaikin()
        volume = self.df['volume'].iloc[-1]
        avgVolume = self.df['volume'].rolling(20).mean().iloc[-1]

        if volume is not None and avgVolume is not None and avgVolume > 0:
            volumeRatio = volume / avgVolume

            if volumeRatio > self.config.volumeSurgeRatio:
                close = self.df['close'].iloc[-1]
                prevClose = self.df['close'].iloc[-2]

                if close > prevClose:
                    self._signals['volume_surge'] = 1
                    self._strengths['volume_surge'] = 0.7
                    self._reasons['volume_surge'] = f"거래량 급증({volumeRatio:.1f}x) + 상승"
                else:
                    self._signals['volume_surge'] = -1
                    self._strengths['volume_surge'] = 0.7
                    self._reasons['volume_surge'] = f"거래량 급증({volumeRatio:.1f}x) + 하락"
            else:
                self._signals['volume_surge'] = 0
                self._strengths['volume_surge'] = 0.3
                self._reasons['volume_surge'] = f"거래량 정상({volumeRatio:.1f}x)"

        if mfi is not None:
            if mfi < 20:
                self._signals['mfi'] = 1
                self._strengths['mfi'] = 0.6
                self._reasons['mfi'] = f"MFI({mfi:.1f}) 과매도 - 자금 유입 기대"
            elif mfi > 80:
                self._signals['mfi'] = -1
                self._strengths['mfi'] = 0.6
                self._reasons['mfi'] = f"MFI({mfi:.1f}) 과매수 - 자금 유출 기대"
            else:
                self._signals['mfi'] = 0
                self._strengths['mfi'] = 0.3
                self._reasons['mfi'] = f"MFI({mfi:.1f}) 중립"

        if chaikin is not None:
            if chaikin > 0:
                self._signals['chaikin'] = 1
                self._strengths['chaikin'] = 0.5
                self._reasons['chaikin'] = f"차이킨 오실레이터 양수 - 매집"
            else:
                self._signals['chaikin'] = -1
                self._strengths['chaikin'] = 0.5
                self._reasons['chaikin'] = f"차이킨 오실레이터 음수 - 분산"

    def _analyzeVolatility(self):
        """Generate volatility-based signals from Bollinger squeeze and ATR."""
        atr = self.indicators.atr(14)
        close = self.df['close'].iloc[-1]
        volatility = self.indicators.volatility(20)

        upper, middle, lower = self.indicators.bollinger(20, 2.0)

        if upper is not None and lower is not None:
            bandWidth = (upper - lower) / middle if middle > 0 else 0

            if bandWidth < 0.05:
                self._signals['bb_squeeze'] = 0
                self._strengths['bb_squeeze'] = 0.8
                self._reasons['bb_squeeze'] = f"볼린저 밴드 스퀴즈 - 큰 움직임 예상"
            else:
                self._signals['bb_squeeze'] = 0
                self._strengths['bb_squeeze'] = 0.3
                self._reasons['bb_squeeze'] = f"볼린저 밴드 폭 정상 ({bandWidth:.1%})"

        if atr is not None and close > 0:
            atrPercent = (atr / close) * 100

            if atrPercent > 3:
                self._signals['atr_high'] = 0
                self._strengths['atr_high'] = 0.7
                self._reasons['atr_high'] = f"ATR({atrPercent:.1f}%) 높음 - 고변동성"
            else:
                self._signals['atr_high'] = 0
                self._strengths['atr_high'] = 0.3
                self._reasons['atr_high'] = f"ATR({atrPercent:.1f}%) 정상"

    def _analyzeForecast(self):
        """Generate time-series forecast signals as a supplementary reference.

        These signals are intentionally weighted low (0.15-0.25) based on E003
        experiment results showing an average -6.54% degradation when forecast
        signals are weighted heavily. They serve as auxiliary indicators only.

        시계열 예측 신호 (참고용) - E003 실험 결과 반영하여 가중치를 낮게 유지.
        """
        try:
            from tradix.signals.forecast import PriceForecast, TrendAnalyzer

            closeValues = self.df['close'].values

            if len(closeValues) < 30:
                return

            forecast = PriceForecast(closeValues)
            result = forecast.predict(steps=5)

            # E003 결과 반영: 가중치 대폭 하향 (0.4 → 0.2)
            if result.direction == 1 and result.confidence > 0.6:
                self._signals['forecast_5d'] = 1
                self._strengths['forecast_5d'] = 0.2 * result.confidence
                self._reasons['forecast_5d'] = f"[참고] 5일 예측: 상승 ({result.expectedReturn:.1%})"
            elif result.direction == -1 and result.confidence > 0.6:
                self._signals['forecast_5d'] = -1
                self._strengths['forecast_5d'] = 0.2 * result.confidence
                self._reasons['forecast_5d'] = f"[참고] 5일 예측: 하락 ({result.expectedReturn:.1%})"
            else:
                self._signals['forecast_5d'] = 0
                self._strengths['forecast_5d'] = 0.1
                self._reasons['forecast_5d'] = f"[참고] 5일 예측: 불확실"

            trendAnalyzer = TrendAnalyzer(closeValues)
            trendResult = trendAnalyzer.analyze()

            # E003 결과 반영: 가중치 대폭 하향 (0.5 → 0.25)
            if trendResult['direction'] == 1 and trendResult['strength'] > 0.6:
                self._signals['trend_analysis'] = 1
                self._strengths['trend_analysis'] = 0.25 * trendResult['strength']
                self._reasons['trend_analysis'] = f"[참고] 추세: 상승"
            elif trendResult['direction'] == -1 and trendResult['strength'] > 0.6:
                self._signals['trend_analysis'] = -1
                self._strengths['trend_analysis'] = 0.25 * trendResult['strength']
                self._reasons['trend_analysis'] = f"[참고] 추세: 하락"
            else:
                self._signals['trend_analysis'] = 0
                self._strengths['trend_analysis'] = 0.1
                self._reasons['trend_analysis'] = "[참고] 추세: 불명확"

        except Exception:
            pass

    def _buildConsensus(self, threshold: float) -> SignalResult:
        """Build a consensus signal from all collected individual signals.

        Args:
            threshold: Minimum ratio of buy or sell signals for consensus.

        Returns:
            Integrated SignalResult based on majority voting and strength weighting.
        """
        if not self._signals:
            return SignalResult(
                signal=0,
                signalType=SignalType.HOLD,
                strength=0.0,
                confidence=0.0,
                reasons=["분석 가능한 신호 없음"],
                details={},
            )

        buySignals = []
        sellSignals = []
        holdSignals = []

        for name, signal in self._signals.items():
            strength = self._strengths.get(name, 0.5)
            reason = self._reasons.get(name, '')

            if signal == 1:
                buySignals.append((name, strength, reason))
            elif signal == -1:
                sellSignals.append((name, strength, reason))
            else:
                holdSignals.append((name, strength, reason))

        totalSignals = len(self._signals)
        buyScore = sum(s[1] for s in buySignals)
        sellScore = sum(s[1] for s in sellSignals)

        buyRatio = len(buySignals) / totalSignals if totalSignals > 0 else 0
        sellRatio = len(sellSignals) / totalSignals if totalSignals > 0 else 0

        if buyRatio >= threshold and buyScore > sellScore:
            finalSignal = 1
            signalType = SignalType.BUY
            strength = buyScore / (buyScore + sellScore + 0.001)
            confidence = buyRatio
            reasons = [r for _, _, r in buySignals if r]
        elif sellRatio >= threshold and sellScore > buyScore:
            finalSignal = -1
            signalType = SignalType.SELL
            strength = sellScore / (buyScore + sellScore + 0.001)
            confidence = sellRatio
            reasons = [r for _, _, r in sellSignals if r]
        else:
            finalSignal = 0
            signalType = SignalType.HOLD
            strength = 0.5
            confidence = 1 - (buyRatio + sellRatio)
            reasons = ["매수/매도 신호 혼조 - 관망 권장"]

        return SignalResult(
            signal=finalSignal,
            signalType=signalType,
            strength=min(1.0, strength),
            confidence=min(1.0, confidence),
            reasons=reasons[:5],
            details={
                'buySignals': len(buySignals),
                'sellSignals': len(sellSignals),
                'holdSignals': len(holdSignals),
                'buyScore': buyScore,
                'sellScore': sellScore,
                'allSignals': self.getAllSignals(),
            },
            timestamp=self.df.index[-1] if isinstance(self.df.index[-1], pd.Timestamp) else None,
        )

    def getSignalHistory(
        self,
        lookback: int = 60,
        strategies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate signal history over the past N days.

        Re-runs signal prediction on incrementally expanding sub-DataFrames
        to produce a time series of signals.

        Args:
            lookback: Number of days to look back (조회 기간, 일).
            strategies: Strategy names to use. If None, all are used (사용할 전략).

        Returns:
            DataFrame with columns: date, signal, strength, confidence, close
            (신호 히스토리 DataFrame).
        """
        results = []

        for i in range(max(0, len(self.df) - lookback), len(self.df)):
            subDf = self.df.iloc[:i+1]
            if len(subDf) < 50:
                continue

            predictor = SignalPredictor(subDf, self.config)
            result = predictor.predict(strategies)

            results.append({
                'date': subDf.index[-1] if hasattr(subDf.index, '__getitem__') else i,
                'signal': result.signal,
                'strength': result.strength,
                'confidence': result.confidence,
                'close': subDf['close'].iloc[-1],
            })

        return pd.DataFrame(results)
