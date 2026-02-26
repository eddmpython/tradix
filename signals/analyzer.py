"""Tradex Multi-Symbol Signal Analyzer Module.

Provides parallel signal analysis across multiple symbols for portfolio-level
trading opportunity discovery, including market consensus, relative ranking,
and divergence detection.

다중 종목 시그널 분석기 - 여러 종목의 신호를 동시에 분석하고 포트폴리오
차원의 매매 기회를 발굴합니다.

Features:
    - Multi-symbol parallel analysis using thread pool
    - Signal strength-based ranking across symbols
    - Market-level consensus (bullish / bearish / mixed)
    - Signal divergence detection between related symbols
    - Export to DataFrame for further analysis

Usage:
    from tradex.signals.analyzer import MultiSignalAnalyzer, SignalScanner

    analyzer = MultiSignalAnalyzer()
    analyzer.addSymbol("AAPL", aapl_df)
    analyzer.addSymbol("MSFT", msft_df)
    market = analyzer.analyze()
    print(market.summary)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from tradex.signals.predictor import SignalPredictor, SignalResult, SignalConfig


@dataclass
class SymbolSignal:
    """Signal result for a single symbol with ranking information.

    개별 종목의 신호 결과와 순위 정보.

    Attributes:
        symbol: Ticker symbol (종목 코드).
        result: SignalResult from the predictor (예측 결과).
        rank: Relative rank among analyzed symbols (상대 순위).
        score: Composite score (signal * strength * confidence) (종합 점수).
    """
    symbol: str
    result: SignalResult
    rank: int = 0
    score: float = 0.0


@dataclass
class MarketSignal:
    """Aggregated market-level signal across all analyzed symbols.

    분석된 전체 종목의 시장 수준 통합 신호.

    Attributes:
        consensus: Market consensus (1=bullish, -1=bearish, 0=mixed) (시장 컨센서스).
        bullishRatio: Fraction of symbols with buy signals (매수 신호 비율).
        bearishRatio: Fraction of symbols with sell signals (매도 신호 비율).
        avgStrength: Average signal strength across all symbols (평균 신호 강도).
        topBuy: Top-ranked buy signal symbols (상위 매수 종목).
        topSell: Top-ranked sell signal symbols (상위 매도 종목).
        summary: Human-readable market summary (시장 요약 문자열).
    """
    consensus: int
    bullishRatio: float
    bearishRatio: float
    avgStrength: float
    topBuy: List[SymbolSignal]
    topSell: List[SymbolSignal]
    summary: str


class MultiSignalAnalyzer:
    """Multi-symbol signal analyzer with parallel processing and market consensus.

    Analyzes multiple symbols concurrently using SignalPredictor, ranks them
    by signal strength, and produces a market-level consensus.

    다중 종목 시그널 분석기 - 여러 종목을 병렬로 분석하고 신호 강도 기반
    순위 및 시장 컨센서스를 산출합니다.

    Attributes:
        config: Shared SignalConfig for all symbols (공유 신호 설정).
        maxWorkers: Thread pool worker count (병렬 처리 워커 수).

    Example:
        >>> analyzer = MultiSignalAnalyzer()
        >>> analyzer.addSymbol("005930", samsung_df)
        >>> analyzer.addSymbol("000660", skhynix_df)
        >>> market = analyzer.analyze()
        >>> print(market.topBuy)
    """

    def __init__(
        self,
        config: Optional[SignalConfig] = None,
        maxWorkers: int = 4,
    ):
        """Initialize the multi-symbol signal analyzer.

        Args:
            config: SignalConfig instance shared across all symbols (신호 설정).
            maxWorkers: Number of threads for parallel analysis (병렬 처리 워커 수).
        """
        self.config = config or SignalConfig()
        self.maxWorkers = maxWorkers
        self._symbols: Dict[str, pd.DataFrame] = {}
        self._results: Dict[str, SignalResult] = {}

    def addSymbol(self, symbol: str, df: pd.DataFrame):
        """Add a symbol's OHLCV data for analysis.

        Args:
            symbol: Ticker symbol identifier (종목 코드).
            df: OHLCV DataFrame for the symbol (종목 OHLCV 데이터).
        """
        self._symbols[symbol] = df

    def addSymbols(self, dataDict: Dict[str, pd.DataFrame]):
        """Add multiple symbols' data at once.

        Args:
            dataDict: Mapping of symbol to OHLCV DataFrame (종목별 OHLCV 데이터 딕셔너리).
        """
        self._symbols.update(dataDict)

    def removeSymbol(self, symbol: str):
        """Remove a symbol and its cached results.

        Args:
            symbol: Ticker symbol to remove (제거할 종목 코드).
        """
        if symbol in self._symbols:
            del self._symbols[symbol]
        if symbol in self._results:
            del self._results[symbol]

    def clearSymbols(self):
        """Remove all symbols and clear cached results."""
        self._symbols.clear()
        self._results.clear()

    def analyzeSymbol(self, symbol: str) -> Optional[SignalResult]:
        """Analyze a single symbol and return its signal result.

        Args:
            symbol: Ticker symbol to analyze (분석할 종목 코드).

        Returns:
            SignalResult or None if insufficient data.
        """
        df = self._symbols.get(symbol)
        if df is None or len(df) < 50:
            return None

        predictor = SignalPredictor(df, self.config)
        return predictor.predict()

    def analyze(
        self,
        strategies: Optional[List[str]] = None,
        topN: int = 5,
    ) -> MarketSignal:
        """Analyze all registered symbols in parallel and produce market consensus.

        Args:
            strategies: Strategy names to use. If None, all are used (사용할 전략).
            topN: Number of top buy/sell symbols to include (상위 N개 종목 수).

        Returns:
            MarketSignal with consensus, rankings, and summary.
        """
        self._results = {}

        with ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
            futures = {
                executor.submit(self._analyzeOne, symbol, strategies): symbol
                for symbol in self._symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        self._results[symbol] = result
                except Exception:
                    pass

        return self._buildMarketSignal(topN)

    def _analyzeOne(
        self,
        symbol: str,
        strategies: Optional[List[str]] = None,
    ) -> Optional[SignalResult]:
        """Analyze a single symbol (internal worker for thread pool).

        Args:
            symbol: Ticker symbol.
            strategies: Strategy names to use.

        Returns:
            SignalResult or None if insufficient data.
        """
        df = self._symbols.get(symbol)
        if df is None or len(df) < 50:
            return None

        predictor = SignalPredictor(df, self.config)
        return predictor.predict(strategies)

    def _buildMarketSignal(self, topN: int) -> MarketSignal:
        """Aggregate individual symbol results into a market-level signal.

        Args:
            topN: Number of top buy/sell symbols to include.

        Returns:
            MarketSignal with consensus and rankings.
        """
        if not self._results:
            return MarketSignal(
                consensus=0,
                bullishRatio=0.0,
                bearishRatio=0.0,
                avgStrength=0.0,
                topBuy=[],
                topSell=[],
                summary="분석 데이터 없음",
            )

        symbolSignals = []
        for symbol, result in self._results.items():
            score = result.signal * result.strength * result.confidence
            symbolSignals.append(SymbolSignal(
                symbol=symbol,
                result=result,
                score=score,
            ))

        symbolSignals.sort(key=lambda x: x.score, reverse=True)
        for i, ss in enumerate(symbolSignals):
            ss.rank = i + 1

        buySignals = [ss for ss in symbolSignals if ss.result.signal == 1]
        sellSignals = [ss for ss in symbolSignals if ss.result.signal == -1]

        total = len(self._results)
        bullishRatio = len(buySignals) / total if total > 0 else 0
        bearishRatio = len(sellSignals) / total if total > 0 else 0
        avgStrength = np.mean([r.strength for r in self._results.values()])

        if bullishRatio > 0.6:
            consensus = 1
            summaryText = f"시장 컨센서스: 강세 ({bullishRatio:.0%} 매수 신호)"
        elif bearishRatio > 0.6:
            consensus = -1
            summaryText = f"시장 컨센서스: 약세 ({bearishRatio:.0%} 매도 신호)"
        else:
            consensus = 0
            summaryText = f"시장 컨센서스: 혼조 (매수 {bullishRatio:.0%}, 매도 {bearishRatio:.0%})"

        return MarketSignal(
            consensus=consensus,
            bullishRatio=bullishRatio,
            bearishRatio=bearishRatio,
            avgStrength=avgStrength,
            topBuy=buySignals[:topN],
            topSell=list(reversed(sellSignals))[:topN],
            summary=summaryText,
        )

    def getSymbolResult(self, symbol: str) -> Optional[SignalResult]:
        """Retrieve the cached signal result for a specific symbol.

        Args:
            symbol: Ticker symbol (종목 코드).

        Returns:
            SignalResult or None if not analyzed.
        """
        return self._results.get(symbol)

    def getRanking(self, ascending: bool = False) -> List[SymbolSignal]:
        """Get symbols ranked by composite signal score.

        Args:
            ascending: If True, rank from weakest to strongest (오름차순 정렬 여부).

        Returns:
            List of SymbolSignal sorted by score (점수순 종목 시그널 리스트).
        """
        symbolSignals = []
        for symbol, result in self._results.items():
            score = result.signal * result.strength * result.confidence
            symbolSignals.append(SymbolSignal(
                symbol=symbol,
                result=result,
                score=score,
            ))

        symbolSignals.sort(key=lambda x: x.score, reverse=not ascending)
        for i, ss in enumerate(symbolSignals):
            ss.rank = i + 1

        return symbolSignals

    def getDivergences(self) -> List[Tuple[str, str, str]]:
        """Detect signal divergences between symbol pairs with strong opposing signals.

        Identifies cases where two symbols both have strong (>0.7) but opposing
        signals, which may indicate sector rotation or pair trading opportunities.

        Returns:
            List of (symbol1, symbol2, description) tuples for each divergence found
            (다이버전스 감지 결과 리스트).
        """
        divergences = []

        symbols = list(self._results.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                r1, r2 = self._results[s1], self._results[s2]

                if r1.signal != 0 and r2.signal != 0 and r1.signal != r2.signal:
                    if r1.strength > 0.7 and r2.strength > 0.7:
                        signalStr1 = "매수" if r1.signal == 1 else "매도"
                        signalStr2 = "매수" if r2.signal == 1 else "매도"
                        desc = f"{s1}({signalStr1}) vs {s2}({signalStr2}) - 강한 신호 충돌"
                        divergences.append((s1, s2, desc))

        return divergences

    def toDataFrame(self) -> pd.DataFrame:
        """Export all analysis results as a DataFrame sorted by score.

        Returns:
            DataFrame with columns: symbol, signal, signalType, strength,
            confidence, score, reasons (분석 결과 DataFrame).
        """
        data = []
        for symbol, result in self._results.items():
            data.append({
                'symbol': symbol,
                'signal': result.signal,
                'signalType': result.signalType.name,
                'strength': result.strength,
                'confidence': result.confidence,
                'score': result.signal * result.strength * result.confidence,
                'reasons': '; '.join(result.reasons[:3]),
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
        return df


class SignalScanner:
    """Signal scanner for filtering symbols by specific criteria.

    Wraps a MultiSignalAnalyzer to provide convenient scanning methods for
    finding symbols that meet buy/sell strength thresholds or signal count criteria.

    신호 스캐너 - MultiSignalAnalyzer의 결과에서 특정 조건을 만족하는 종목을 검색합니다.

    Attributes:
        analyzer: MultiSignalAnalyzer instance to scan (분석기 인스턴스).

    Example:
        >>> scanner = SignalScanner(analyzer)
        >>> buy_list = scanner.scanBuy(minStrength=0.7)
        >>> sell_list = scanner.scanSell(minStrength=0.7)
    """

    def __init__(self, analyzer: MultiSignalAnalyzer):
        self.analyzer = analyzer

    def scanBuy(
        self,
        minStrength: float = 0.6,
        minConfidence: float = 0.5,
    ) -> List[SymbolSignal]:
        """Scan for symbols with buy signals meeting minimum thresholds.

        Args:
            minStrength: Minimum signal strength (최소 신호 강도).
            minConfidence: Minimum signal confidence (최소 신뢰도).

        Returns:
            List of SymbolSignal sorted by score descending (매수 종목 리스트).
        """
        results = []
        for symbol, result in self.analyzer._results.items():
            if result.signal == 1:
                if result.strength >= minStrength and result.confidence >= minConfidence:
                    results.append(SymbolSignal(
                        symbol=symbol,
                        result=result,
                        score=result.strength * result.confidence,
                    ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def scanSell(
        self,
        minStrength: float = 0.6,
        minConfidence: float = 0.5,
    ) -> List[SymbolSignal]:
        """Scan for symbols with sell signals meeting minimum thresholds.

        Args:
            minStrength: Minimum signal strength (최소 신호 강도).
            minConfidence: Minimum signal confidence (최소 신뢰도).

        Returns:
            List of SymbolSignal sorted by score descending (매도 종목 리스트).
        """
        results = []
        for symbol, result in self.analyzer._results.items():
            if result.signal == -1:
                if result.strength >= minStrength and result.confidence >= minConfidence:
                    results.append(SymbolSignal(
                        symbol=symbol,
                        result=result,
                        score=result.strength * result.confidence,
                    ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def scanBySignalCount(
        self,
        minBuySignals: int = 3,
        minSellSignals: int = 0,
    ) -> List[SymbolSignal]:
        """Scan for symbols by minimum buy signal count and maximum sell signal count.

        Args:
            minBuySignals: Minimum number of individual buy signals required (최소 매수 신호 수).
            minSellSignals: Maximum allowed sell signals (허용 최대 매도 신호 수).

        Returns:
            List of SymbolSignal sorted by net signal count descending.
        """
        results = []
        for symbol, result in self.analyzer._results.items():
            details = result.details
            buyCount = details.get('buySignals', 0)
            sellCount = details.get('sellSignals', 0)

            if buyCount >= minBuySignals and sellCount <= minSellSignals:
                results.append(SymbolSignal(
                    symbol=symbol,
                    result=result,
                    score=buyCount - sellCount,
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results
