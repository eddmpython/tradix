"""
Textual-based interactive TUI dashboard for backtest results.

Requires `tradex[tui]` extras: textual, textual-plotext.
Optional full-screen interactive dashboard with tabs, charts, and trade browser.

Textual 기반 대화형 TUI 대시보드. `tradex[tui]` 설치 시 사용 가능.
"""

from typing import Any, Optional


def launchDashboard(result: Any, lang: str = "en") -> None:
    """Launch interactive Textual dashboard for a backtest result.

    Args:
        result: BacktestResult, VectorizedResult, or EasyResult.
        lang: Language ("en" or "ko").
    """
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Footer, Static, DataTable, TabbedContent, TabPane
        from textual.containers import Horizontal, Vertical
    except ImportError:
        from rich.console import Console
        c = Console()
        c.print("[yellow]Textual not installed. Install with:[/yellow]")
        c.print("[cyan]  pip install tradex-backtest\\[tui][/cyan]")
        return

    try:
        from textual_plotext import PlotextPlot
        HAS_PLOTEXT = True
    except ImportError:
        HAS_PLOTEXT = False

    r = _unwrap(result)

    class TradexDashboard(App):
        CSS = """
        Screen {
            background: $surface;
        }
        #summary-panel {
            height: auto;
            padding: 1 2;
        }
        .metric-box {
            width: 1fr;
            height: auto;
            padding: 1 2;
            border: solid $primary;
            margin: 0 1;
        }
        .positive { color: $success; }
        .negative { color: $error; }
        #equity-chart { height: 20; }
        #drawdown-chart { height: 15; }
        """

        TITLE = "Tradex Dashboard"
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("d", "toggle_dark", "Theme"),
        ]

        def compose(self) -> ComposeResult:
            yield Header()
            with TabbedContent():
                with TabPane("Summary" if lang != "ko" else "요약", id="tab-summary"):
                    yield self._buildSummary()
                with TabPane("Charts" if lang != "ko" else "차트", id="tab-charts"):
                    yield self._buildCharts()
                with TabPane("Trades" if lang != "ko" else "거래", id="tab-trades"):
                    yield self._buildTrades()
            yield Footer()

        def _buildSummary(self) -> Static:
            ko = lang == "ko"
            total_ret = getattr(r, 'totalReturn', 0)
            sharpe = _getMetric(r, 'sharpeRatio')
            mdd = _getMetric(r, 'maxDrawdown')
            win_rate = getattr(r, 'winRate', 0)
            total_trades = getattr(r, 'totalTrades', 0)
            initial = getattr(r, 'initialCash', 0)
            final = getattr(r, 'finalEquity', 0)
            annual = _getMetric(r, 'annualReturn')
            pf = _getMetric(r, 'profitFactor')

            lines = []
            lines.append(f"{'Strategy' if not ko else '전략'}: {getattr(r, 'strategy', '?')}")
            lines.append(f"{'Symbol' if not ko else '종목'}: {getattr(r, 'symbol', '?')}")
            lines.append(f"{'Period' if not ko else '기간'}: {getattr(r, 'startDate', '')} ~ {getattr(r, 'endDate', '')}")
            lines.append("")
            lines.append(f"{'Initial' if not ko else '초기자금'}: {initial:,.0f}")
            lines.append(f"{'Final' if not ko else '최종자산'}: {final:,.0f}")
            lines.append(f"{'Total Return' if not ko else '총수익률'}: {total_ret:+.2f}%")
            lines.append(f"{'Annual Return' if not ko else '연수익률'}: {annual:+.2f}%")
            lines.append(f"{'Sharpe' if not ko else '샤프'}: {sharpe:.2f}")
            lines.append(f"{'Max DD' if not ko else '최대낙폭'}: {mdd:.2f}%")
            lines.append(f"{'Win Rate' if not ko else '승률'}: {win_rate:.1f}%")
            lines.append(f"{'Trades' if not ko else '거래수'}: {total_trades}")
            lines.append(f"{'P/F' if not ko else '손익비'}: {pf:.2f}")

            return Static("\n".join(lines), id="summary-panel")

        def _buildCharts(self) -> Static:
            if not HAS_PLOTEXT:
                return Static("textual-plotext not installed.\npip install tradex-backtest[tui]")
            return _EquityChartWidget(r)

        def _buildTrades(self) -> DataTable:
            table = DataTable()
            table.add_columns("#", "Entry", "Exit", "Entry$", "Exit$", "P&L", "Return%")
            trades = getattr(r, 'trades', [])
            for i, t in enumerate(trades[-50:], 1):
                if isinstance(t, dict):
                    ed = str(t.get('entryDate', ''))[:10]
                    xd = str(t.get('exitDate', ''))[:10]
                    ep = f"{t.get('entryPrice', 0):,.0f}"
                    xp = f"{t.get('exitPrice', 0):,.0f}"
                    pnl = t.get('pnl', 0)
                    ret = t.get('returnPct', 0)
                else:
                    ed = str(getattr(t, 'entryDate', ''))[:10]
                    xd = str(getattr(t, 'exitDate', ''))[:10]
                    ep = f"{getattr(t, 'entryPrice', 0):,.0f}"
                    xp = f"{getattr(t, 'exitPrice', 0):,.0f}"
                    pnl = getattr(t, 'pnl', 0)
                    ret = getattr(t, 'returnPct', 0)
                table.add_row(str(i), ed, xd, ep, xp, f"{pnl:+,.0f}", f"{ret:+.2f}%")
            return table

        def action_toggle_dark(self):
            self.dark = not self.dark

    if HAS_PLOTEXT:
        class _EquityChartWidget(PlotextPlot):
            def __init__(self, result_data):
                super().__init__()
                self._data = result_data

            def on_mount(self):
                equity = _getEquityList(self._data)
                if equity:
                    p = self.plt
                    p.title("Equity Curve")
                    p.plot(equity)
                    initial = getattr(self._data, 'initialCash', equity[0])
                    if initial:
                        p.hline(initial, color="gray")
    else:
        _EquityChartWidget = None

    dashboard = TradexDashboard()
    dashboard.run()


def _unwrap(result):
    if hasattr(result, '_result'):
        return result._result
    return result


def _getEquityList(r):
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


def _getMetric(r, name, default=0.0):
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    metrics = getattr(r, 'metrics', {})
    if isinstance(metrics, dict):
        return float(metrics.get(name, default))
    return default
