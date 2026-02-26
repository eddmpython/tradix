"""
Tradex Report Generator Module.

Generates interactive Plotly-based HTML backtest reports with configurable
sections, themes, and chart components. Supports both BacktestResult objects
and raw data inputs for flexible report creation.

Plotly 기반의 인터랙티브 HTML 백테스트 보고서를 생성하는 모듈입니다.
섹션, 테마, 차트 구성 요소를 설정할 수 있으며, BacktestResult 객체 또는
원시 데이터를 직접 입력하여 유연하게 보고서를 생성할 수 있습니다.

Features:
    - Configurable report sections (summary, equity, drawdown, monthly, trades)
    - Light and dark theme support
    - Summary metrics dashboard with color-coded indicators
    - Combined equity curve and drawdown chart
    - Monthly returns heatmap
    - Regime-based performance analysis charts
    - Trade distribution and recent trade table
    - Direct file export

Usage:
    from tradex.analytics.generator import ReportGenerator, ReportConfig

    generator = ReportGenerator(ReportConfig(title="My Report", theme="dark"))
    html = generator.generate(backtest_result)
    generator.save(backtest_result, "report.html")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from tradex.report.charts import (
    ChartBuilder,
    equityCurveChart,
    drawdownChart,
    monthlyReturnsHeatmap,
    regimePerformanceChart,
    tradeDistributionChart,
    combinedEquityDrawdownChart,
)


class ReportSection(Enum):
    """
    Enumeration of available report sections.

    보고서에 포함 가능한 섹션 열거형입니다.
    """
    SUMMARY = "summary"
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    MONTHLY_RETURNS = "monthly_returns"
    REGIME_ANALYSIS = "regime_analysis"
    TRADES = "trades"
    RISK_METRICS = "risk_metrics"


@dataclass
class ReportConfig:
    """
    Configuration for report generation.

    보고서 생성 설정 데이터 클래스입니다.

    Attributes:
        title (str): Report title displayed in the header (보고서 제목).
        theme (str): Visual theme, 'light' or 'dark' (테마).
        includePlotlyJs (str): Plotly JS inclusion mode - 'cdn', True, or False
            (Plotly JS 포함 방식).
        sections (List[ReportSection]): Sections to include in the report
            (포함할 섹션 목록).
        showBenchmark (bool): Whether to display benchmark comparison
            (벤치마크 비교 표시 여부).
        language (str): Report language code (보고서 언어).
    """
    title: str = "백테스트 보고서"
    theme: str = "light"
    includePlotlyJs: str = "cdn"
    sections: List[ReportSection] = field(default_factory=lambda: [
        ReportSection.SUMMARY,
        ReportSection.EQUITY_CURVE,
        ReportSection.DRAWDOWN,
        ReportSection.MONTHLY_RETURNS,
        ReportSection.TRADES,
    ])
    showBenchmark: bool = True
    language: str = "ko"


class ReportGenerator:
    """
    Backtest report generator producing interactive HTML documents.

    Accepts BacktestResult objects or raw data and assembles configurable
    HTML reports with embedded Plotly charts, metrics dashboards, and
    trade analysis sections. Supports light and dark themes.

    BacktestResult 객체 또는 원시 데이터를 받아 Plotly 차트, 지표 대시보드,
    거래 분석 섹션이 포함된 인터랙티브 HTML 보고서를 생성합니다.

    Attributes:
        config (ReportConfig): Report configuration (보고서 설정).
        chartBuilder (ChartBuilder): Chart-to-HTML converter (차트 빌더).

    Example:
        >>> generator = ReportGenerator()
        >>> html = generator.generate(result)
        >>> generator.save(result, 'report.html')
    """

    CSS_LIGHT = """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --accent-color: #2E86AB;
            --positive-color: #28a745;
            --negative-color: #dc3545;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header .meta { color: var(--text-secondary); font-size: 14px; }
        .card {
            background: var(--bg-primary);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--accent-color);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .metric-box {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-box .label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-box .value {
            font-size: 24px;
            font-weight: 600;
            margin-top: 4px;
        }
        .metric-box .value.positive { color: var(--positive-color); }
        .metric-box .value.negative { color: var(--negative-color); }
        .chart-container { margin: 20px 0; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: var(--bg-secondary);
            font-weight: 600;
        }
        tr:hover { background: var(--bg-secondary); }
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 12px;
        }
    """

    CSS_DARK = """
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --border-color: #2a2a4a;
            --accent-color: #4fc3f7;
            --positive-color: #4caf50;
            --negative-color: #f44336;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header .meta { color: var(--text-secondary); font-size: 14px; }
        .card {
            background: var(--bg-primary);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--accent-color);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .metric-box {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-box .label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-box .value {
            font-size: 24px;
            font-weight: 600;
            margin-top: 4px;
        }
        .metric-box .value.positive { color: var(--positive-color); }
        .metric-box .value.negative { color: var(--negative-color); }
        .chart-container { margin: 20px 0; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: var(--bg-secondary);
            font-weight: 600;
        }
        tr:hover { background: var(--bg-secondary); }
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 12px;
        }
    """

    def __init__(self, config: ReportConfig = None):
        """
        Initialize the ReportGenerator.

        ReportGenerator를 초기화합니다.

        Args:
            config: Report configuration; uses defaults if None (보고서 설정).
        """
        self.config = config or ReportConfig()
        self.chartBuilder = ChartBuilder(
            includePlotlyJs=self.config.includePlotlyJs,
            theme='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
        )

    def generate(
        self,
        result: Any,
        benchmark: pd.Series = None,
        regimeStats: Dict[str, Dict[str, float]] = None,
    ) -> str:
        """
        Generate an HTML report from a backtest result.

        백테스트 결과로부터 HTML 보고서를 생성합니다.

        Args:
            result: BacktestResult object with equity, returns, trades, and
                metrics attributes (백테스트 결과 객체).
            benchmark: Optional benchmark return series for comparison
                (벤치마크 수익률 시리즈, 선택).
            regimeStats: Optional dict of regime performance stats, keyed by
                regime name (레짐별 성과 딕셔너리, 선택).

        Returns:
            str: Complete HTML document string.
        """
        sections = []

        if ReportSection.SUMMARY in self.config.sections:
            sections.append(self._buildSummarySection(result))

        if ReportSection.EQUITY_CURVE in self.config.sections:
            sections.append(self._buildEquitySection(result, benchmark))

        if ReportSection.DRAWDOWN in self.config.sections:
            sections.append(self._buildDrawdownSection(result))

        if ReportSection.MONTHLY_RETURNS in self.config.sections:
            sections.append(self._buildMonthlySection(result))

        if ReportSection.REGIME_ANALYSIS in self.config.sections and regimeStats:
            sections.append(self._buildRegimeSection(regimeStats))

        if ReportSection.TRADES in self.config.sections:
            sections.append(self._buildTradesSection(result))

        if ReportSection.RISK_METRICS in self.config.sections:
            sections.append(self._buildRiskSection(result))

        return self._buildHtml(sections)

    def generateFromData(
        self,
        equity: pd.Series,
        returns: pd.Series = None,
        trades: pd.DataFrame = None,
        metrics: Dict[str, float] = None,
        benchmark: pd.Series = None,
        regimeStats: Dict[str, Dict[str, float]] = None,
        strategyName: str = "전략",
    ) -> str:
        """
        Generate an HTML report directly from raw data inputs.

        원시 데이터를 직접 입력하여 HTML 보고서를 생성합니다.

        Args:
            equity: Portfolio equity value series with datetime index
                (자산 가치 시리즈).
            returns: Daily returns series; computed from equity if None
                (일별 수익률 시리즈, 미입력 시 equity에서 계산).
            trades: Trade history DataFrame with columns such as 'pnlPct',
                'entryPrice', 'exitPrice', 'side' (거래 DataFrame).
            metrics: Performance metrics dict; computed from data if None
                (성과 지표 딕셔너리).
            benchmark: Benchmark equity series for comparison
                (벤치마크 시리즈).
            regimeStats: Regime performance statistics (레짐별 성과).
            strategyName: Strategy display name (전략 이름).

        Returns:
            str: Complete HTML document string.
        """
        if returns is None and equity is not None:
            returns = equity.pct_change().dropna()

        if metrics is None:
            metrics = self._calcMetrics(equity, returns)

        class SimpleResult:
            pass

        result = SimpleResult()
        result.equity = equity
        result.returns = returns
        result.trades = trades if trades is not None else pd.DataFrame()
        result.metrics = metrics
        result.strategyName = strategyName
        result.startDate = equity.index[0] if equity is not None else None
        result.endDate = equity.index[-1] if equity is not None else None

        return self.generate(result, benchmark, regimeStats)

    def _calcMetrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate basic performance metrics from equity and returns.

        자산 곡선과 수익률로부터 기본 성과 지표를 계산합니다.

        Args:
            equity: Portfolio equity value series.
            returns: Daily returns series.

        Returns:
            Dict[str, float]: Computed metrics including totalReturn,
                annualReturn, volatility, sharpeRatio, maxDrawdown.
        """
        if equity is None or len(equity) < 2:
            return {}

        totalReturn = (equity.iloc[-1] / equity.iloc[0] - 1)
        nDays = len(equity)
        nYears = nDays / 252

        annualReturn = (1 + totalReturn) ** (1 / nYears) - 1 if nYears > 0 else 0
        volatility = returns.std() * np.sqrt(252) if returns is not None else 0
        sharpe = annualReturn / volatility if volatility > 0 else 0

        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        maxDrawdown = drawdown.min()

        return {
            'totalReturn': totalReturn,
            'annualReturn': annualReturn,
            'volatility': volatility,
            'sharpeRatio': sharpe,
            'maxDrawdown': maxDrawdown,
        }

    def _buildSummarySection(self, result: Any) -> str:
        """
        Build the performance summary HTML section.

        성과 요약 HTML 섹션을 생성합니다.

        Args:
            result: BacktestResult or compatible object with metrics attribute.

        Returns:
            str: HTML string for the summary section.
        """
        metrics = getattr(result, 'metrics', {})
        if isinstance(metrics, dict):
            m = metrics
        else:
            m = {
                'totalReturn': getattr(metrics, 'totalReturn', 0),
                'annualReturn': getattr(metrics, 'annualReturn', 0),
                'volatility': getattr(metrics, 'volatility', 0),
                'sharpeRatio': getattr(metrics, 'sharpeRatio', 0),
                'maxDrawdown': getattr(metrics, 'maxDrawdown', 0),
                'winRate': getattr(metrics, 'winRate', 0),
                'totalTrades': getattr(metrics, 'totalTrades', 0),
            }

        def formatValue(key, value):
            if key in ['totalReturn', 'annualReturn', 'volatility', 'maxDrawdown', 'winRate']:
                pct = value * 100
                cls = 'positive' if pct >= 0 else 'negative'
                if key == 'maxDrawdown':
                    cls = 'negative'
                return f'<span class="value {cls}">{pct:.1f}%</span>'
            elif key == 'sharpeRatio':
                cls = 'positive' if value >= 0 else 'negative'
                return f'<span class="value {cls}">{value:.2f}</span>'
            else:
                return f'<span class="value">{value:,.0f}</span>'

        labels = {
            'totalReturn': '총 수익률',
            'annualReturn': '연간 수익률',
            'volatility': '변동성',
            'sharpeRatio': '샤프 비율',
            'maxDrawdown': '최대 낙폭',
            'winRate': '승률',
            'totalTrades': '총 거래',
        }

        metricsHtml = ""
        for key in ['totalReturn', 'annualReturn', 'sharpeRatio', 'maxDrawdown', 'volatility', 'winRate']:
            if key in m:
                metricsHtml += f'''
                    <div class="metric-box">
                        <div class="label">{labels.get(key, key)}</div>
                        {formatValue(key, m[key])}
                    </div>
                '''

        return f'''
            <div class="card">
                <h2>성과 요약</h2>
                <div class="metrics-grid">
                    {metricsHtml}
                </div>
            </div>
        '''

    def _buildEquitySection(self, result: Any, benchmark: pd.Series = None) -> str:
        """
        Build the equity curve and drawdown chart HTML section.

        수익 곡선 및 드로다운 차트 HTML 섹션을 생성합니다.

        Args:
            result: BacktestResult or compatible object with equity attribute.
            benchmark: Optional benchmark series.

        Returns:
            str: HTML string for the equity section.
        """
        equity = getattr(result, 'equity', None)
        if equity is None:
            return ""

        fig = combinedEquityDrawdownChart(equity)
        chartHtml = self.chartBuilder.toHtml(fig)

        return f'''
            <div class="card">
                <h2>수익 곡선 & 드로다운</h2>
                <div class="chart-container">
                    {chartHtml}
                </div>
            </div>
        '''

    def _buildDrawdownSection(self, result: Any) -> str:
        """
        Build a standalone drawdown HTML section (currently unused placeholder).

        별도 드로다운 HTML 섹션을 생성합니다 (현재 미사용 플레이스홀더).

        Args:
            result: BacktestResult or compatible object.

        Returns:
            str: Empty string (reserved for future use).
        """
        return ""

    def _buildMonthlySection(self, result: Any) -> str:
        """
        Build the monthly returns heatmap HTML section.

        월별 수익률 히트맵 HTML 섹션을 생성합니다.

        Args:
            result: BacktestResult or compatible object with returns or equity.

        Returns:
            str: HTML string for the monthly returns section, or empty string
                if insufficient data (fewer than 30 data points).
        """
        returns = getattr(result, 'returns', None)
        if returns is None:
            equity = getattr(result, 'equity', None)
            if equity is not None:
                returns = equity.pct_change().dropna()

        if returns is None or len(returns) < 30:
            return ""

        fig = monthlyReturnsHeatmap(returns)
        chartHtml = self.chartBuilder.toHtml(fig)

        return f'''
            <div class="card">
                <h2>월별 수익률</h2>
                <div class="chart-container">
                    {chartHtml}
                </div>
            </div>
        '''

    def _buildRegimeSection(self, regimeStats: Dict[str, Dict[str, float]]) -> str:
        """
        Build the regime performance analysis HTML section.

        레짐별 성과 분석 HTML 섹션을 생성합니다.

        Args:
            regimeStats: Dict mapping regime names to performance dicts
                containing 'return', 'sharpe', 'trades' keys.

        Returns:
            str: HTML string for the regime analysis section.
        """
        fig = regimePerformanceChart(regimeStats)
        chartHtml = self.chartBuilder.toHtml(fig)

        return f'''
            <div class="card">
                <h2>레짐별 성과 분석</h2>
                <div class="chart-container">
                    {chartHtml}
                </div>
            </div>
        '''

    def _buildTradesSection(self, result: Any) -> str:
        """
        Build the trade analysis HTML section with distribution chart and table.

        거래 분포 차트와 최근 거래 테이블이 포함된 거래 분석 HTML 섹션을 생성합니다.

        Args:
            result: BacktestResult or compatible object with trades DataFrame.

        Returns:
            str: HTML string for the trades section, or empty string if no trades.
        """
        trades = getattr(result, 'trades', None)
        if trades is None or len(trades) == 0:
            return ""

        fig = tradeDistributionChart(trades)
        chartHtml = self.chartBuilder.toHtml(fig)

        recentTrades = trades.tail(10) if len(trades) > 10 else trades

        tableRows = ""
        for _, trade in recentTrades.iterrows():
            pnl = trade.get('pnlPct', trade.get('return', 0)) * 100
            cls = 'positive' if pnl >= 0 else 'negative'
            date = trade.get('exitDate', trade.get('date', ''))
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')

            tableRows += f'''
                <tr>
                    <td>{date}</td>
                    <td>{trade.get('side', '-')}</td>
                    <td>{trade.get('entryPrice', 0):,.0f}</td>
                    <td>{trade.get('exitPrice', 0):,.0f}</td>
                    <td class="{cls}">{pnl:.1f}%</td>
                </tr>
            '''

        return f'''
            <div class="card">
                <h2>거래 분석</h2>
                <div class="chart-container">
                    {chartHtml}
                </div>
                <h3 style="margin-top: 20px; margin-bottom: 10px;">최근 거래</h3>
                <table>
                    <thead>
                        <tr>
                            <th>날짜</th>
                            <th>방향</th>
                            <th>진입가</th>
                            <th>청산가</th>
                            <th>수익률</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tableRows}
                    </tbody>
                </table>
            </div>
        '''

    def _buildRiskSection(self, result: Any) -> str:
        """
        Build the risk metrics HTML section (currently unused placeholder).

        리스크 지표 HTML 섹션을 생성합니다 (현재 미사용 플레이스홀더).

        Args:
            result: BacktestResult or compatible object.

        Returns:
            str: Empty string (reserved for future use).
        """
        metrics = getattr(result, 'metrics', {})
        if isinstance(metrics, dict):
            m = metrics
        else:
            m = {}

        return ""

    def _buildHtml(self, sections: List[str]) -> str:
        """
        Assemble the complete HTML document from individual sections.

        개별 섹션들을 결합하여 완전한 HTML 문서를 조립합니다.

        Args:
            sections: List of HTML section strings to embed in the body.

        Returns:
            str: Complete HTML document string with head, styles, and footer.
        """
        css = self.CSS_DARK if self.config.theme == 'dark' else self.CSS_LIGHT
        plotlyJs = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>' \
            if self.config.includePlotlyJs == 'cdn' else ''

        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        return f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    {plotlyJs}
    <style>
        {css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.config.title}</h1>
            <div class="meta">생성일: {now} | Tradex v2.0</div>
        </div>

        {''.join(sections)}

        <div class="footer">
            Generated by Tradex Report Generator
        </div>
    </div>
</body>
</html>
        '''

    def save(
        self,
        result: Any,
        filepath: str,
        benchmark: pd.Series = None,
        regimeStats: Dict[str, Dict[str, float]] = None,
    ) -> str:
        """
        Generate and save an HTML report to a file.

        HTML 보고서를 생성하여 파일로 저장합니다.

        Args:
            result: BacktestResult object (백테스트 결과 객체).
            filepath: Destination file path (저장 경로).
            benchmark: Optional benchmark series (벤치마크).
            regimeStats: Optional regime performance dict (레짐별 성과).

        Returns:
            str: The file path where the report was saved (저장된 파일 경로).
        """
        html = self.generate(result, benchmark, regimeStats)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath
