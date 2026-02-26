"""
Textual-based interactive TUI dashboard for backtest results.

5-screen interactive dashboard with Overview, Metrics, Trades, Charts, Compare.
Requires `tradex[tui]` extras: textual, textual-plotext.

Textual 기반 대화형 TUI 대시보드 (5개 화면).
개요, 지표, 거래, 차트, 비교 뷰를 제공합니다.
`tradex[tui]` 설치 시 사용 가능.
"""

from typing import Any, Dict, List, Optional


LABELS_EN = {
    "header": "TRADEX",
    "overview": "Overview",
    "metrics": "Metrics",
    "trades": "Trades",
    "charts": "Charts",
    "compare": "Compare",
    "strategy": "Strategy",
    "symbol": "Symbol",
    "period": "Period",
    "return": "Return",
    "sharpe": "Sharpe",
    "mdd": "MDD",
    "winRate": "Win Rate",
    "totalTrades": "Trades",
    "annual": "Annual",
    "volatility": "Volatility",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "profitFactor": "P/F",
    "avgWin": "Avg Win",
    "avgLoss": "Avg Loss",
    "expectancy": "Expectancy",
    "initial": "Initial",
    "final": "Final",
    "totalReturn": "Total Return",
    "annualReturn": "Annual Return",
    "bestMonth": "Best Month",
    "worstMonth": "Worst Month",
    "monthlyAvg": "Monthly Avg",
    "maxDD": "Max Drawdown",
    "ddDuration": "DD Duration",
    "var": "VaR (95%)",
    "winningTrades": "Winning",
    "losingTrades": "Losing",
    "avgHold": "Avg Hold",
    "maxWinStreak": "Max Win Streak",
    "maxLossStreak": "Max Loss Streak",
    "returnPanel": "Returns",
    "riskPanel": "Risk",
    "ratioPanel": "Ratios",
    "tradeStats": "Trade Statistics",
    "filter": "Filter",
    "sort": "Sort",
    "search": "Search",
    "page": "Page",
    "all": "All",
    "winners": "Winners",
    "losers": "Losers",
    "side": "Side",
    "entry": "Entry",
    "exit": "Exit",
    "entryPrice": "Entry $",
    "exitPrice": "Exit $",
    "pnl": "P&L",
    "returnPct": "Return %",
    "hold": "Hold",
    "tradeDetail": "Trade Detail",
    "equityCurve": "Equity Curve",
    "drawdown": "Drawdown",
    "priceChart": "Equity Curve",
    "volumeChart": "Trade Volume",
    "equityMarkers": "Equity + Trade Markers",
    "ranking": "Strategy Ranking",
    "equityOverlay": "Equity Overlay (Normalized)",
    "compositeScore": "Score",
    "days": "days",
    "noCompare": "No comparison data. Pass compareResults to launchDashboard().",
    "installMsg": "Textual not installed. Install with:",
    "installCmd": "pip install tradex-backtest[tui]",
    "quit": "Quit",
    "theme": "Theme",
    "help": "Help",
}

LABELS_KO = {
    "header": "TRADEX",
    "overview": "개요",
    "metrics": "지표",
    "trades": "거래",
    "charts": "차트",
    "compare": "비교",
    "strategy": "전략",
    "symbol": "종목",
    "period": "기간",
    "return": "수익률",
    "sharpe": "샤프",
    "mdd": "최대낙폭",
    "winRate": "승률",
    "totalTrades": "거래수",
    "annual": "연수익률",
    "volatility": "변동성",
    "sortino": "소르티노",
    "calmar": "칼마",
    "profitFactor": "손익비",
    "avgWin": "평균수익",
    "avgLoss": "평균손실",
    "expectancy": "기대수익",
    "initial": "초기자금",
    "final": "최종자산",
    "totalReturn": "총수익률",
    "annualReturn": "연수익률",
    "bestMonth": "최고월수익",
    "worstMonth": "최저월수익",
    "monthlyAvg": "월평균수익",
    "maxDD": "최대낙폭",
    "ddDuration": "낙폭기간",
    "var": "VaR (95%)",
    "winningTrades": "수익거래",
    "losingTrades": "손실거래",
    "avgHold": "평균보유",
    "maxWinStreak": "최대연승",
    "maxLossStreak": "최대연패",
    "returnPanel": "수익률",
    "riskPanel": "위험",
    "ratioPanel": "비율",
    "tradeStats": "거래 통계",
    "filter": "필터",
    "sort": "정렬",
    "search": "검색",
    "page": "페이지",
    "all": "전체",
    "winners": "수익",
    "losers": "손실",
    "side": "방향",
    "entry": "진입일",
    "exit": "청산일",
    "entryPrice": "진입가",
    "exitPrice": "청산가",
    "pnl": "손익",
    "returnPct": "수익률%",
    "hold": "보유",
    "tradeDetail": "거래 상세",
    "equityCurve": "자산곡선",
    "drawdown": "낙폭",
    "priceChart": "자산곡선",
    "volumeChart": "거래량",
    "equityMarkers": "자산곡선 + 매매마커",
    "ranking": "전략 순위",
    "equityOverlay": "자산곡선 오버레이 (정규화)",
    "compositeScore": "점수",
    "days": "일",
    "noCompare": "비교 데이터 없음. launchDashboard()에 compareResults를 전달하세요.",
    "installMsg": "Textual 미설치. 설치 명령:",
    "installCmd": "pip install tradex-backtest[tui]",
    "quit": "종료",
    "theme": "테마",
    "help": "도움말",
}

TRADES_PER_PAGE = 25


def _unwrap(result: Any) -> Any:
    """Unwrap EasyResult to underlying BacktestResult.
    EasyResult를 내부 BacktestResult로 언래핑합니다."""
    if hasattr(result, '_result'):
        return result._result
    return result


def _getEquityList(r: Any) -> list:
    """Extract equity curve as a plain list.
    자산곡선을 일반 리스트로 추출합니다."""
    ec = getattr(r, 'equityCurve', None)
    if ec is None:
        return []
    try:
        import numpy as np
        if isinstance(ec, np.ndarray):
            return ec.tolist()
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(ec, pd.Series):
            return ec.tolist()
    except ImportError:
        pass
    if isinstance(ec, list):
        return ec
    return list(ec) if hasattr(ec, '__iter__') else []


def _getMetric(r: Any, name: str, default: float = 0.0) -> float:
    """Get metric by name from result object or its metrics dict.
    결과 객체 또는 metrics 딕셔너리에서 이름으로 지표를 가져옵니다."""
    val = getattr(r, name, None)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    metrics = getattr(r, 'metrics', {})
    if isinstance(metrics, dict):
        v = metrics.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return default


def _getDateLabels(r: Any) -> list:
    """Extract date labels from equity curve index.
    자산곡선 인덱스에서 날짜 라벨을 추출합니다."""
    ec = getattr(r, 'equityCurve', None)
    try:
        import pandas as pd
        if isinstance(ec, pd.Series) and hasattr(ec.index, 'strftime'):
            return ec.index.strftime('%Y-%m-%d').tolist()
    except (ImportError, AttributeError):
        pass
    return []


def _sma(data: list, period: int) -> list:
    """Simple moving average.
    단순이동평균을 계산합니다."""
    result = [None] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result


def _getTradeField(trade: Any, name: str, default: Any = None) -> Any:
    """Get field from trade (dict or object style).
    거래 객체(dict 또는 object)에서 필드를 가져옵니다."""
    if isinstance(trade, dict):
        return trade.get(name, default)
    return getattr(trade, name, default)


def _computeDrawdown(equity: list) -> list:
    """Compute drawdown percentage series from equity list.
    자산 리스트로부터 낙폭(%) 시리즈를 계산합니다."""
    if not equity:
        return []
    peak = equity[0]
    dd = []
    for v in equity:
        if v > peak:
            peak = v
        dd.append((v / peak - 1) * 100 if peak else 0)
    return dd


def _compositeScore(r: Any) -> float:
    """Calculate composite ranking score for strategy comparison.
    전략 비교를 위한 복합 순위 점수를 계산합니다."""
    sharpe = _getMetric(r, 'sharpeRatio', 0)
    sortino = _getMetric(r, 'sortinoRatio', 0)
    calmar = _getMetric(r, 'calmarRatio', 0)
    totalReturn = _getMetric(r, 'totalReturn', 0)
    mdd = abs(_getMetric(r, 'maxDrawdown', 0))
    winRate = _getMetric(r, 'winRate', 0)
    pf = _getMetric(r, 'profitFactor', 0)
    score = (
        sharpe * 25
        + sortino * 15
        + calmar * 10
        + totalReturn * 0.5
        + winRate * 0.3
        + pf * 5
        - mdd * 0.5
    )
    return round(score, 2)


def launchDashboard(
    result: Any,
    lang: str = "en",
    compareResults: Optional[List[Any]] = None,
) -> None:
    """Launch interactive Textual dashboard.

    Args:
        result: Primary backtest result.
        lang: "en" or "ko".
        compareResults: Optional list of results for comparison view.

    대화형 Textual 대시보드를 실행합니다.

    Args:
        result: 기본 백테스트 결과.
        lang: "en" 또는 "ko".
        compareResults: 비교 뷰를 위한 결과 목록 (선택사항).
    """
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.widgets import (
            Header,
            Footer,
            Static,
            DataTable,
            Select,
            Input,
            ContentSwitcher,
            Label,
        )
        from textual.containers import (
            Horizontal,
            Vertical,
            Container,
            HorizontalGroup,
            VerticalGroup,
        )
        from textual.reactive import reactive
        from textual.message import Message
        from textual import on
    except ImportError:
        from rich.console import Console
        c = Console()
        lb = LABELS_KO if lang == "ko" else LABELS_EN
        c.print(f"[yellow]{lb['installMsg']}[/yellow]")
        c.print(f"[cyan]  {lb['installCmd']}[/cyan]")
        return

    try:
        from textual_plotext import PlotextPlot
        HAS_PLOTEXT = True
    except ImportError:
        HAS_PLOTEXT = False

    lb = LABELS_KO if lang == "ko" else LABELS_EN
    r = _unwrap(result)
    compareList = compareResults or []

    class MetricCard(Static):
        """Single metric display card with label, value, and color.
        라벨, 값, 색상이 있는 단일 지표 표시 카드."""

        def __init__(
            self,
            label: str,
            value: str,
            colorClass: str = "",
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self._label = label
            self._value = value
            self._colorClass = colorClass

        def compose(self) -> ComposeResult:
            yield Label(self._label, classes="metric-label")
            yield Label(self._value, classes=f"metric-value {self._colorClass}")

        def on_mount(self) -> None:
            self.add_class("metric-card")

    class StatRow(Static):
        """A single stat row: label + value for sidebar.
        사이드바용 단일 통계 행: 라벨 + 값."""

        def __init__(self, label: str, value: str, colorClass: str = "", **kwargs) -> None:
            super().__init__(**kwargs)
            self._label = label
            self._value = value
            self._colorClass = colorClass

        def compose(self) -> ComposeResult:
            yield Label(f"{self._label}: ", classes="stat-label")
            yield Label(self._value, classes=f"stat-value {self._colorClass}")

    class MetricPanel(Static):
        """Panel containing a title and list of stat rows.
        제목과 통계 행 목록이 있는 패널."""

        def __init__(self, title: str, rows: List[tuple], **kwargs) -> None:
            super().__init__(**kwargs)
            self._title = title
            self._rows = rows

        def compose(self) -> ComposeResult:
            yield Label(self._title, classes="panel-title")
            for labelText, valueText, colorCls in self._rows:
                yield StatRow(labelText, valueText, colorCls)

    if HAS_PLOTEXT:
        class EquityChartWidget(PlotextPlot):
            """Equity curve chart with SMA overlays.
            SMA 오버레이가 있는 자산곡선 차트."""

            def __init__(self, resultData: Any, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = resultData

            def on_mount(self) -> None:
                equity = _getEquityList(self._data)
                if not equity:
                    return
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['equityCurve'])
                p.plot(equity, label="Equity", color="cyan")
                initial = getattr(self._data, 'initialCash', equity[0] if equity else 0)
                if initial:
                    p.hline(initial, color="gray")
                sma20 = _sma(equity, 20)
                valid20 = [(j, v) for j, v in enumerate(sma20) if v is not None]
                if valid20:
                    p.plot(
                        [j for j, _ in valid20],
                        [v for _, v in valid20],
                        label="SMA20",
                        color="yellow",
                    )
                sma60 = _sma(equity, 60)
                valid60 = [(j, v) for j, v in enumerate(sma60) if v is not None]
                if valid60:
                    p.plot(
                        [j for j, _ in valid60],
                        [v for _, v in valid60],
                        label="SMA60",
                        color="magenta",
                    )

        class DrawdownChartWidget(PlotextPlot):
            """Drawdown percentage chart.
            낙폭(%) 차트."""

            def __init__(self, resultData: Any, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = resultData

            def on_mount(self) -> None:
                equity = _getEquityList(self._data)
                dd = _computeDrawdown(equity)
                if not dd:
                    return
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['drawdown'])
                p.plot(dd, color="red", label="Drawdown %")
                p.hline(0, color="gray")

        class PriceChartWidget(PlotextPlot):
            """Equity curve line chart for Charts view.
            차트 뷰용 자산곡선 라인 차트."""

            def __init__(self, resultData: Any, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = resultData

            def on_mount(self) -> None:
                equity = _getEquityList(self._data)
                if not equity:
                    return
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['priceChart'])
                p.plot(equity, label="Equity", color="cyan")
                initial = getattr(self._data, 'initialCash', equity[0] if equity else 0)
                if initial:
                    p.hline(initial, color="gray")

        class VolumeChartWidget(PlotextPlot):
            """Trade volume per period chart.
            기간별 거래량 차트."""

            def __init__(self, resultData: Any, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = resultData

            def on_mount(self) -> None:
                trades = getattr(self._data, 'trades', [])
                equity = _getEquityList(self._data)
                if not equity:
                    return
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['volumeChart'])
                if trades:
                    bucketCount = min(50, max(10, len(equity) // 5))
                    bucketSize = max(1, len(equity) // bucketCount)
                    volumes = [0] * bucketCount
                    dates = _getDateLabels(self._data)
                    dateToIdx = {}
                    if dates:
                        dateToIdx = {d: i for i, d in enumerate(dates)}
                    for t in trades:
                        entryDateStr = str(_getTradeField(t, 'entryDate', ''))[:10]
                        idx = dateToIdx.get(entryDateStr, -1)
                        if idx < 0:
                            continue
                        bucket = min(idx // bucketSize, bucketCount - 1)
                        qty = _getTradeField(t, 'quantity', 1)
                        try:
                            volumes[bucket] += float(qty)
                        except (TypeError, ValueError):
                            volumes[bucket] += 1
                    p.bar(list(range(bucketCount)), volumes, color="gray")
                else:
                    p.text("No trade data", x=0, y=0)

        class EquityMarkersChartWidget(PlotextPlot):
            """Equity curve with buy/sell trade markers.
            매수/매도 마커가 있는 자산곡선 차트."""

            def __init__(self, resultData: Any, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = resultData

            def on_mount(self) -> None:
                equity = _getEquityList(self._data)
                if not equity:
                    return
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['equityMarkers'])
                p.plot(equity, label="Equity", color="cyan")
                initial = getattr(self._data, 'initialCash', equity[0] if equity else 0)
                if initial:
                    p.hline(initial, color="gray")
                trades = getattr(self._data, 'trades', [])
                dates = _getDateLabels(self._data)
                if trades and dates:
                    dateToIdx = {d: i for i, d in enumerate(dates)}
                    buyX, buyY = [], []
                    sellX, sellY = [], []
                    for t in trades:
                        entryD = str(_getTradeField(t, 'entryDate', ''))[:10]
                        exitD = str(_getTradeField(t, 'exitDate', ''))[:10]
                        if entryD in dateToIdx:
                            idx = dateToIdx[entryD]
                            if idx < len(equity):
                                buyX.append(idx)
                                buyY.append(equity[idx])
                        if exitD in dateToIdx:
                            idx = dateToIdx[exitD]
                            if idx < len(equity):
                                sellX.append(idx)
                                sellY.append(equity[idx])
                    if buyX:
                        p.scatter(buyX, buyY, color="green", label="Buy", marker="dot")
                    if sellX:
                        p.scatter(sellX, sellY, color="red", label="Sell", marker="dot")

        class EquityOverlayWidget(PlotextPlot):
            """Multiple equity curves overlaid, normalized to 100.
            100으로 정규화된 여러 자산곡선을 오버레이합니다."""

            def __init__(self, resultList: List[Any], **kwargs) -> None:
                super().__init__(**kwargs)
                self._resultList = resultList

            def on_mount(self) -> None:
                p = self.plt
                p.clear_figure()
                p.theme("dark")
                p.title(lb['equityOverlay'])
                colors = ["cyan", "yellow", "magenta", "green", "red", "white", "blue"]
                for i, res in enumerate(self._resultList):
                    unwrapped = _unwrap(res)
                    equity = _getEquityList(unwrapped)
                    if not equity or equity[0] == 0:
                        continue
                    normalized = [v / equity[0] * 100 for v in equity]
                    stratName = getattr(unwrapped, 'strategy', f"Strategy {i + 1}")
                    symbolName = getattr(unwrapped, 'symbol', '')
                    labelText = f"{stratName}"
                    if symbolName:
                        labelText = f"{stratName} ({symbolName})"
                    p.plot(
                        normalized,
                        label=labelText,
                        color=colors[i % len(colors)],
                    )
                p.hline(100, color="gray")

    class TradexDashboard(App):
        """Tradex interactive backtest dashboard.
        Tradex 대화형 백테스트 대시보드."""

        CSS = """
        Screen {
            background: #0C0C0C;
        }
        Header {
            background: #1E3A5F;
        }
        Footer {
            background: #1A1A2E;
        }
        .metric-card {
            width: 1fr;
            border: solid #1E3A5F;
            padding: 0 1;
            margin: 0 1;
            height: auto;
            min-height: 3;
        }
        .metric-label {
            color: #888888;
            text-style: bold;
        }
        .metric-value {
            text-style: bold;
        }
        .positive {
            color: #00FF88;
        }
        .negative {
            color: #FF4444;
        }
        .neutral {
            color: #CCCCCC;
        }
        .panel-title {
            text-style: bold;
            color: #00D4FF;
            padding: 1 0 0 1;
        }
        .stat-label {
            color: #888888;
            padding: 0 0 0 1;
        }
        .stat-value {
            text-style: bold;
            padding: 0 1 0 0;
        }
        #nav-bar {
            height: 3;
            dock: top;
            background: #1A1A2E;
            padding: 0 1;
        }
        .nav-btn {
            width: auto;
            min-width: 12;
            margin: 0 1;
            padding: 0 2;
            text-style: bold;
            color: #888888;
            background: #1A1A2E;
            border: none;
        }
        .nav-btn:hover {
            color: #00D4FF;
            background: #2A2A3E;
        }
        .nav-btn.-active {
            background: #1E3A5F;
            color: #00D4FF;
            text-style: bold;
        }
        #view-overview {
            height: 1fr;
        }
        #view-metrics {
            height: 1fr;
        }
        #view-trades {
            height: 1fr;
        }
        #view-charts {
            height: 1fr;
        }
        #view-compare {
            height: 1fr;
        }
        #metric-cards-row {
            height: auto;
            padding: 1 0;
        }
        #overview-main {
            height: 4fr;
        }
        #equity-chart {
            width: 4fr;
        }
        #stats-sidebar {
            width: 1fr;
            border: solid #1E3A5F;
            padding: 1;
            min-width: 28;
        }
        #drawdown-chart-container {
            height: 1fr;
        }
        #metrics-top-row {
            height: 2fr;
        }
        .metrics-panel {
            width: 1fr;
            border: solid #1E3A5F;
            padding: 1;
            margin: 0 1;
        }
        #trade-stats-panel {
            height: 1fr;
            border: solid #1E3A5F;
            padding: 1;
            margin: 1;
        }
        #filter-bar {
            height: 3;
            padding: 0 1;
        }
        #filter-bar Select {
            width: 20;
            margin: 0 1;
        }
        #filter-bar Input {
            width: 30;
            margin: 0 1;
        }
        #page-indicator {
            width: auto;
            min-width: 16;
            padding: 0 2;
            text-align: center;
            color: #888888;
        }
        #trade-table {
            height: 1fr;
        }
        #trade-detail {
            display: none;
            border: tall #FFD700;
            background: #1A1A2E;
            height: auto;
            max-height: 12;
            padding: 1;
            margin: 1;
        }
        #trade-detail.visible {
            display: block;
        }
        #price-chart {
            height: 18;
        }
        #volume-chart {
            height: 5;
        }
        #equity-markers-chart {
            height: 12;
        }
        #ranking-table {
            height: 12;
        }
        #equity-overlay-chart {
            height: 14;
        }
        #no-compare-msg {
            padding: 2;
            color: #888888;
            text-align: center;
        }
        .no-plotext-msg {
            padding: 2;
            color: #FFD700;
        }
        #header-info {
            height: auto;
            padding: 0 1;
            background: #1A1A2E;
            color: #00D4FF;
        }
        """

        TITLE = "Tradex Dashboard"

        BINDINGS = [
            Binding("1", "switch_view('overview')", "Overview", show=True),
            Binding("2", "switch_view('metrics')", "Metrics", show=True),
            Binding("3", "switch_view('trades')", "Trades", show=True),
            Binding("4", "switch_view('charts')", "Charts", show=True),
            Binding("5", "switch_view('compare')", "Compare", show=True),
            Binding("q", "quit", "Quit", show=True),
            Binding("d", "toggle_dark", "Theme", show=True),
            Binding("question_mark", "show_help", "Help", show=True),
            Binding("pagedown", "next_page", "PgDn", show=False),
            Binding("pageup", "prev_page", "PgUp", show=False),
            Binding("enter", "toggle_detail", "Detail", show=False),
        ]

        currentView = reactive("overview")
        currentPage = reactive(0)
        tradeFilter = reactive("all")
        tradeSort = reactive("default")
        tradeSearch = reactive("")

        def __init__(self) -> None:
            super().__init__()
            self._allTrades = self._loadTrades()
            self._filteredTrades = list(self._allTrades)
            self._totalPages = max(1, (len(self._filteredTrades) + TRADES_PER_PAGE - 1) // TRADES_PER_PAGE)
            self._detailVisible = False

        def _loadTrades(self) -> List[dict]:
            rawTrades = getattr(r, 'trades', [])
            tradeList = []
            for i, t in enumerate(rawTrades):
                rec = {
                    'index': i + 1,
                    'side': str(_getTradeField(t, 'side', 'BUY')).upper().replace('ORDESIDE.', '').replace('ORDERSIDE.', ''),
                    'entryDate': str(_getTradeField(t, 'entryDate', ''))[:10],
                    'exitDate': str(_getTradeField(t, 'exitDate', ''))[:10],
                    'entryPrice': float(_getTradeField(t, 'entryPrice', 0) or 0),
                    'exitPrice': float(_getTradeField(t, 'exitPrice', 0) or 0),
                    'pnl': float(_getTradeField(t, 'pnl', 0) or 0),
                    'returnPct': float(_getTradeField(t, 'returnPct', 0) or _getTradeField(t, 'pnlPercent', 0) or 0),
                    'holdingDays': int(_getTradeField(t, 'holdingDays', 0) or 0),
                    'quantity': float(_getTradeField(t, 'quantity', 0) or 0),
                    'commission': float(_getTradeField(t, 'commission', 0) or 0),
                    'slippage': float(_getTradeField(t, 'slippage', 0) or 0),
                    'symbol': str(_getTradeField(t, 'symbol', getattr(r, 'symbol', '')) or ''),
                }
                sideRaw = rec['side']
                if 'BUY' in sideRaw:
                    rec['side'] = 'BUY'
                elif 'SELL' in sideRaw:
                    rec['side'] = 'SELL'
                tradeList.append(rec)
            return tradeList

        def _colorClass(self, val: float) -> str:
            if val > 0:
                return "positive"
            elif val < 0:
                return "negative"
            return "neutral"

        def _fmtPct(self, val: float) -> str:
            return f"{val:+.2f}%"

        def _fmtNum(self, val: float, decimals: int = 0) -> str:
            if decimals == 0:
                return f"{val:+,.0f}"
            return f"{val:+,.{decimals}f}"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(self._buildHeaderInfo(), id="header-info")
            with Horizontal(id="nav-bar"):
                yield Static(f" [1] {lb['overview']} ", classes="nav-btn -active", id="nav-overview")
                yield Static(f" [2] {lb['metrics']} ", classes="nav-btn", id="nav-metrics")
                yield Static(f" [3] {lb['trades']} ", classes="nav-btn", id="nav-trades")
                yield Static(f" [4] {lb['charts']} ", classes="nav-btn", id="nav-charts")
                yield Static(f" [5] {lb['compare']} ", classes="nav-btn", id="nav-compare")

            with ContentSwitcher(initial="view-overview"):
                with Vertical(id="view-overview"):
                    yield from self._composeOverview()
                with Vertical(id="view-metrics"):
                    yield from self._composeMetrics()
                with Vertical(id="view-trades"):
                    yield from self._composeTrades()
                with Vertical(id="view-charts"):
                    yield from self._composeCharts()
                with Vertical(id="view-compare"):
                    yield from self._composeCompare()

            yield Footer()

        def _buildHeaderInfo(self) -> str:
            strategy = getattr(r, 'strategy', '?')
            symbol = getattr(r, 'symbol', '?')
            startDate = getattr(r, 'startDate', '')
            endDate = getattr(r, 'endDate', '')
            return f" {lb['header']} | {lb['strategy']}: {strategy} | {lb['symbol']}: {symbol} | {lb['period']}: {startDate} ~ {endDate}"

        def _composeOverview(self) -> ComposeResult:
            totalRet = _getMetric(r, 'totalReturn', 0)
            sharpe = _getMetric(r, 'sharpeRatio', 0)
            mdd = _getMetric(r, 'maxDrawdown', 0)
            winRate = _getMetric(r, 'winRate', 0)
            totalTrades = int(_getMetric(r, 'totalTrades', 0))

            with Horizontal(id="metric-cards-row"):
                yield MetricCard(
                    lb['return'],
                    self._fmtPct(totalRet),
                    self._colorClass(totalRet),
                    id="mc-return",
                )
                yield MetricCard(
                    lb['sharpe'],
                    f"{sharpe:.2f}",
                    self._colorClass(sharpe),
                    id="mc-sharpe",
                )
                yield MetricCard(
                    lb['mdd'],
                    f"{mdd:.2f}%",
                    "negative" if mdd < 0 else "neutral",
                    id="mc-mdd",
                )
                yield MetricCard(
                    lb['winRate'],
                    f"{winRate:.1f}%",
                    "positive" if winRate >= 50 else "negative",
                    id="mc-winrate",
                )
                yield MetricCard(
                    lb['totalTrades'],
                    str(totalTrades),
                    "neutral",
                    id="mc-trades",
                )

            if HAS_PLOTEXT:
                with Horizontal(id="overview-main"):
                    yield EquityChartWidget(r, id="equity-chart")
                    yield self._buildStatsSidebar()
                yield DrawdownChartWidget(r, id="drawdown-chart-container")
            else:
                with Horizontal(id="overview-main"):
                    yield Static(
                        "textual-plotext not installed.\npip install tradex-backtest[tui]",
                        classes="no-plotext-msg",
                    )
                    yield self._buildStatsSidebar()

        def _buildStatsSidebar(self) -> Static:
            annual = _getMetric(r, 'annualReturn', 0)
            volatility = _getMetric(r, 'volatility', 0)
            sortino = _getMetric(r, 'sortinoRatio', 0)
            calmar = _getMetric(r, 'calmarRatio', 0)
            pf = _getMetric(r, 'profitFactor', 0)
            avgWin = _getMetric(r, 'avgWin', 0)
            avgLoss = _getMetric(r, 'avgLoss', 0)
            initial = getattr(r, 'initialCash', 0)
            final = getattr(r, 'finalEquity', 0)

            lines = []
            lines.append(f"[b]{lb['initial']}:[/b] {initial:,.0f}")
            lines.append(f"[b]{lb['final']}:[/b] {final:,.0f}")
            lines.append("")
            lines.append(f"[b]{lb['annual']}:[/b] {annual:+.2f}%")
            lines.append(f"[b]{lb['volatility']}:[/b] {volatility:.2f}%")
            lines.append(f"[b]{lb['sortino']}:[/b] {sortino:.2f}")
            lines.append(f"[b]{lb['calmar']}:[/b] {calmar:.2f}")
            lines.append(f"[b]{lb['profitFactor']}:[/b] {pf:.2f}")
            lines.append(f"[b]{lb['avgWin']}:[/b] {avgWin:+,.0f}")
            lines.append(f"[b]{lb['avgLoss']}:[/b] {avgLoss:+,.0f}")

            return Static("\n".join(lines), id="stats-sidebar", markup=True)

        def _composeMetrics(self) -> ComposeResult:
            totalRet = _getMetric(r, 'totalReturn', 0)
            annual = _getMetric(r, 'annualReturn', 0)
            bestMonth = _getMetric(r, 'bestMonth', 0)
            worstMonth = _getMetric(r, 'worstMonth', 0)
            monthlyReturns = getattr(r, 'monthlyReturns', None)
            if monthlyReturns is None:
                metricsDict = getattr(r, 'metrics', {})
                if isinstance(metricsDict, dict):
                    monthlyReturns = metricsDict.get('monthlyReturns', [])
                else:
                    monthlyReturns = []
            monthlyAvg = 0
            if monthlyReturns and len(monthlyReturns) > 0:
                try:
                    monthlyAvg = sum(monthlyReturns) / len(monthlyReturns)
                except (TypeError, ZeroDivisionError):
                    monthlyAvg = 0

            volatility = _getMetric(r, 'volatility', 0)
            mdd = _getMetric(r, 'maxDrawdown', 0)
            ddDuration = int(_getMetric(r, 'maxDrawdownDuration', 0))

            equity = _getEquityList(r)
            var95 = 0.0
            if equity and len(equity) > 10:
                dailyReturns = [(equity[i] / equity[i - 1] - 1) * 100 for i in range(1, len(equity)) if equity[i - 1] != 0]
                if dailyReturns:
                    sortedReturns = sorted(dailyReturns)
                    idx5 = max(0, int(len(sortedReturns) * 0.05))
                    var95 = sortedReturns[idx5]

            sharpe = _getMetric(r, 'sharpeRatio', 0)
            sortino = _getMetric(r, 'sortinoRatio', 0)
            calmar = _getMetric(r, 'calmarRatio', 0)
            pf = _getMetric(r, 'profitFactor', 0)
            expectancy = _getMetric(r, 'expectancy', 0)

            totalTrades = int(_getMetric(r, 'totalTrades', 0))
            winningTrades = int(_getMetric(r, 'winningTrades', 0))
            losingTrades = int(_getMetric(r, 'losingTrades', 0))
            winRate = _getMetric(r, 'winRate', 0)
            avgHold = _getMetric(r, 'avgHoldingDays', 0)
            maxWinStreak = int(_getMetric(r, 'maxConsecutiveWins', 0))
            maxLossStreak = int(_getMetric(r, 'maxConsecutiveLosses', 0))

            with Horizontal(id="metrics-top-row"):
                yield MetricPanel(
                    lb['returnPanel'],
                    [
                        (lb['totalReturn'], self._fmtPct(totalRet), self._colorClass(totalRet)),
                        (lb['annualReturn'], self._fmtPct(annual), self._colorClass(annual)),
                        (lb['bestMonth'], self._fmtPct(bestMonth), self._colorClass(bestMonth)),
                        (lb['worstMonth'], self._fmtPct(worstMonth), self._colorClass(worstMonth)),
                        (lb['monthlyAvg'], self._fmtPct(monthlyAvg), self._colorClass(monthlyAvg)),
                    ],
                    classes="metrics-panel",
                )
                yield MetricPanel(
                    lb['riskPanel'],
                    [
                        (lb['volatility'], f"{volatility:.2f}%", "neutral"),
                        (lb['maxDD'], f"{mdd:.2f}%", "negative" if mdd < 0 else "neutral"),
                        (lb['ddDuration'], f"{ddDuration} {lb['days']}", "neutral"),
                        (lb['var'], f"{var95:.2f}%", self._colorClass(var95)),
                    ],
                    classes="metrics-panel",
                )
                yield MetricPanel(
                    lb['ratioPanel'],
                    [
                        (lb['sharpe'], f"{sharpe:.2f}", self._colorClass(sharpe)),
                        (lb['sortino'], f"{sortino:.2f}", self._colorClass(sortino)),
                        (lb['calmar'], f"{calmar:.2f}", self._colorClass(calmar)),
                        (lb['profitFactor'], f"{pf:.2f}", self._colorClass(pf - 1)),
                        (lb['expectancy'], f"{expectancy:.2f}%", self._colorClass(expectancy)),
                    ],
                    classes="metrics-panel",
                )

            yield MetricPanel(
                lb['tradeStats'],
                [
                    (lb['totalTrades'], str(totalTrades), "neutral"),
                    (lb['winningTrades'], str(winningTrades), "positive" if winningTrades > 0 else "neutral"),
                    (lb['losingTrades'], str(losingTrades), "negative" if losingTrades > 0 else "neutral"),
                    (lb['winRate'], f"{winRate:.1f}%", "positive" if winRate >= 50 else "negative"),
                    (lb['avgHold'], f"{avgHold:.1f} {lb['days']}", "neutral"),
                    (lb['maxWinStreak'], str(maxWinStreak), "positive" if maxWinStreak > 0 else "neutral"),
                    (lb['maxLossStreak'], str(maxLossStreak), "negative" if maxLossStreak > 0 else "neutral"),
                ],
                id="trade-stats-panel",
            )

        def _composeTrades(self) -> ComposeResult:
            with Horizontal(id="filter-bar"):
                yield Select(
                    [(lb['all'], "all"), (lb['winners'], "winners"), (lb['losers'], "losers")],
                    value="all",
                    id="trade-filter-select",
                    allow_blank=False,
                )
                yield Select(
                    [
                        ("# Asc", "index_asc"),
                        ("# Desc", "index_desc"),
                        ("P&L Asc", "pnl_asc"),
                        ("P&L Desc", "pnl_desc"),
                        ("Return Asc", "ret_asc"),
                        ("Return Desc", "ret_desc"),
                        ("Hold Asc", "hold_asc"),
                        ("Hold Desc", "hold_desc"),
                    ],
                    value="index_asc",
                    id="trade-sort-select",
                    allow_blank=False,
                )
                yield Input(placeholder=lb['search'], id="trade-search-input")
                yield Static(
                    f" {lb['page']} 1/{self._totalPages} ",
                    id="page-indicator",
                )

            yield DataTable(id="trade-table", cursor_type="row")
            yield Static("", id="trade-detail")

        def _composeCharts(self) -> ComposeResult:
            if HAS_PLOTEXT:
                yield PriceChartWidget(r, id="price-chart")
                yield VolumeChartWidget(r, id="volume-chart")
                yield EquityMarkersChartWidget(r, id="equity-markers-chart")
            else:
                yield Static(
                    "textual-plotext not installed.\npip install tradex-backtest[tui]",
                    classes="no-plotext-msg",
                )

        def _composeCompare(self) -> ComposeResult:
            if not compareList:
                yield Static(lb['noCompare'], id="no-compare-msg")
                return

            yield DataTable(id="ranking-table", cursor_type="row")
            if HAS_PLOTEXT:
                allResults = [result] + compareList
                yield EquityOverlayWidget(allResults, id="equity-overlay-chart")
            else:
                yield Static(
                    "textual-plotext not installed.\npip install tradex-backtest[tui]",
                    classes="no-plotext-msg",
                )

        def on_mount(self) -> None:
            self._populateTradeTable()
            if compareList:
                self._populateRankingTable()

        def _populateTradeTable(self) -> None:
            try:
                table = self.query_one("#trade-table", DataTable)
            except Exception:
                return
            table.clear(columns=True)
            table.add_columns(
                "#",
                lb['side'],
                lb['entry'],
                lb['exit'],
                lb['entryPrice'],
                lb['exitPrice'],
                lb['pnl'],
                lb['returnPct'],
                lb['hold'],
            )
            start = self.currentPage * TRADES_PER_PAGE
            end = start + TRADES_PER_PAGE
            pageData = self._filteredTrades[start:end]
            for rec in pageData:
                pnlColor = "green" if rec['pnl'] > 0 else ("red" if rec['pnl'] < 0 else "white")
                table.add_row(
                    str(rec['index']),
                    rec['side'],
                    rec['entryDate'],
                    rec['exitDate'],
                    f"{rec['entryPrice']:,.0f}",
                    f"{rec['exitPrice']:,.0f}",
                    f"{rec['pnl']:+,.0f}",
                    f"{rec['returnPct']:+.2f}%",
                    f"{rec['holdingDays']}d",
                )

        def _populateRankingTable(self) -> None:
            try:
                table = self.query_one("#ranking-table", DataTable)
            except Exception:
                return
            table.clear(columns=True)
            table.add_columns(
                "#",
                lb['strategy'],
                lb['symbol'],
                lb['return'],
                lb['sharpe'],
                lb['mdd'],
                lb['winRate'],
                lb['profitFactor'],
                lb['compositeScore'],
            )
            allResults = [result] + compareList
            ranked = []
            for res in allResults:
                unwrapped = _unwrap(res)
                score = _compositeScore(unwrapped)
                ranked.append((unwrapped, score))
            ranked.sort(key=lambda x: x[1], reverse=True)
            for rank, (unwrapped, score) in enumerate(ranked, 1):
                stratName = getattr(unwrapped, 'strategy', '?')
                symbolName = getattr(unwrapped, 'symbol', '?')
                totalRet = _getMetric(unwrapped, 'totalReturn', 0)
                sharpe = _getMetric(unwrapped, 'sharpeRatio', 0)
                mdd = _getMetric(unwrapped, 'maxDrawdown', 0)
                winRate = _getMetric(unwrapped, 'winRate', 0)
                pf = _getMetric(unwrapped, 'profitFactor', 0)
                table.add_row(
                    str(rank),
                    stratName,
                    symbolName,
                    f"{totalRet:+.2f}%",
                    f"{sharpe:.2f}",
                    f"{mdd:.2f}%",
                    f"{winRate:.1f}%",
                    f"{pf:.2f}",
                    f"{score:.1f}",
                )

        def _applyTradeFilters(self) -> None:
            filtered = list(self._allTrades)
            if self.tradeFilter == "winners":
                filtered = [t for t in filtered if t['pnl'] > 0]
            elif self.tradeFilter == "losers":
                filtered = [t for t in filtered if t['pnl'] < 0]
            searchTerm = self.tradeSearch.strip().lower()
            if searchTerm:
                filtered = [
                    t for t in filtered
                    if searchTerm in t['entryDate'].lower()
                    or searchTerm in t['exitDate'].lower()
                    or searchTerm in t['side'].lower()
                    or searchTerm in t['symbol'].lower()
                    or searchTerm in str(t['index'])
                ]
            sortKey = self.tradeSort
            if sortKey == "index_asc":
                filtered.sort(key=lambda t: t['index'])
            elif sortKey == "index_desc":
                filtered.sort(key=lambda t: t['index'], reverse=True)
            elif sortKey == "pnl_asc":
                filtered.sort(key=lambda t: t['pnl'])
            elif sortKey == "pnl_desc":
                filtered.sort(key=lambda t: t['pnl'], reverse=True)
            elif sortKey == "ret_asc":
                filtered.sort(key=lambda t: t['returnPct'])
            elif sortKey == "ret_desc":
                filtered.sort(key=lambda t: t['returnPct'], reverse=True)
            elif sortKey == "hold_asc":
                filtered.sort(key=lambda t: t['holdingDays'])
            elif sortKey == "hold_desc":
                filtered.sort(key=lambda t: t['holdingDays'], reverse=True)
            self._filteredTrades = filtered
            self._totalPages = max(1, (len(self._filteredTrades) + TRADES_PER_PAGE - 1) // TRADES_PER_PAGE)
            self.currentPage = 0
            self._populateTradeTable()
            self._updatePageIndicator()

        def _updatePageIndicator(self) -> None:
            try:
                indicator = self.query_one("#page-indicator", Static)
                indicator.update(f" {lb['page']} {self.currentPage + 1}/{self._totalPages} ")
            except Exception:
                pass

        def _updateNavHighlight(self, viewName: str) -> None:
            navIds = ["nav-overview", "nav-metrics", "nav-trades", "nav-charts", "nav-compare"]
            targetId = f"nav-{viewName}"
            for navId in navIds:
                try:
                    btn = self.query_one(f"#{navId}", Static)
                    if navId == targetId:
                        btn.add_class("-active")
                    else:
                        btn.remove_class("-active")
                except Exception:
                    pass

        @on(Select.Changed, "#trade-filter-select")
        def _onFilterChange(self, event: Select.Changed) -> None:
            self.tradeFilter = str(event.value) if event.value is not None else "all"
            self._applyTradeFilters()

        @on(Select.Changed, "#trade-sort-select")
        def _onSortChange(self, event: Select.Changed) -> None:
            self.tradeSort = str(event.value) if event.value is not None else "default"
            self._applyTradeFilters()

        @on(Input.Changed, "#trade-search-input")
        def _onSearchChange(self, event: Input.Changed) -> None:
            self.tradeSearch = event.value
            self._applyTradeFilters()

        def action_switch_view(self, viewName: str) -> None:
            """Switch to a named view.
            지정된 뷰로 전환합니다."""
            viewId = f"view-{viewName}"
            try:
                switcher = self.query_one(ContentSwitcher)
                switcher.current = viewId
                self.currentView = viewName
                self._updateNavHighlight(viewName)
            except Exception:
                pass

        def action_toggle_dark(self) -> None:
            """Toggle dark/light theme.
            다크/라이트 테마를 전환합니다."""
            self.dark = not self.dark

        def action_show_help(self) -> None:
            """Show keyboard shortcuts help.
            키보드 단축키 도움말을 표시합니다."""
            helpLines = [
                "[b]Keyboard Shortcuts / 키보드 단축키[/b]",
                "",
                "[cyan]1[/cyan] - Overview / 개요",
                "[cyan]2[/cyan] - Metrics / 지표",
                "[cyan]3[/cyan] - Trades / 거래",
                "[cyan]4[/cyan] - Charts / 차트",
                "[cyan]5[/cyan] - Compare / 비교",
                "[cyan]q[/cyan] - Quit / 종료",
                "[cyan]d[/cyan] - Toggle Theme / 테마 전환",
                "[cyan]PgUp/PgDn[/cyan] - Trade Page / 거래 페이지",
                "[cyan]Enter[/cyan] - Trade Detail / 거래 상세",
            ]
            self.notify("\n".join(helpLines), title="Help / 도움말", timeout=8)

        def action_next_page(self) -> None:
            """Go to next trade page.
            다음 거래 페이지로 이동합니다."""
            if self.currentView != "trades":
                return
            if self.currentPage < self._totalPages - 1:
                self.currentPage += 1
                self._populateTradeTable()
                self._updatePageIndicator()

        def action_prev_page(self) -> None:
            """Go to previous trade page.
            이전 거래 페이지로 이동합니다."""
            if self.currentView != "trades":
                return
            if self.currentPage > 0:
                self.currentPage -= 1
                self._populateTradeTable()
                self._updatePageIndicator()

        def action_toggle_detail(self) -> None:
            """Toggle trade detail panel.
            거래 상세 패널을 토글합니다."""
            if self.currentView != "trades":
                return
            try:
                table = self.query_one("#trade-table", DataTable)
                detailPanel = self.query_one("#trade-detail", Static)
            except Exception:
                return

            if self._detailVisible:
                self._detailVisible = False
                detailPanel.remove_class("visible")
                detailPanel.update("")
                return

            rowKey = table.cursor_row
            if rowKey is None:
                return

            start = self.currentPage * TRADES_PER_PAGE
            tradeIdx = start + rowKey
            if tradeIdx < 0 or tradeIdx >= len(self._filteredTrades):
                return

            rec = self._filteredTrades[tradeIdx]
            lines = [
                f"[b]{lb['tradeDetail']}[/b]",
                "",
                f"  # {rec['index']}  |  {lb['side']}: {rec['side']}  |  {lb['symbol']}: {rec['symbol']}",
                f"  {lb['entry']}: {rec['entryDate']}  {lb['entryPrice']}: {rec['entryPrice']:,.0f}",
                f"  {lb['exit']}: {rec['exitDate']}  {lb['exitPrice']}: {rec['exitPrice']:,.0f}",
                f"  {lb['pnl']}: {rec['pnl']:+,.0f}  |  {lb['returnPct']}: {rec['returnPct']:+.2f}%",
                f"  {lb['hold']}: {rec['holdingDays']} {lb['days']}",
                f"  Qty: {rec['quantity']:,.2f}  |  Comm: {rec['commission']:,.0f}  |  Slip: {rec['slippage']:,.0f}",
            ]
            detailPanel.update("\n".join(lines))
            detailPanel.add_class("visible")
            self._detailVisible = True

    dashboard = TradexDashboard()
    dashboard.run()
