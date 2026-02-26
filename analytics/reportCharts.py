"""
Tradex Report Chart Components Module.

Provides reusable Plotly chart-building functions for backtest report
generation. Includes equity curve, drawdown, monthly returns heatmap,
regime performance, trade distribution, and combined equity-drawdown charts.

백테스트 보고서 생성을 위한 재사용 가능한 Plotly 차트 빌딩 함수들을
제공하는 모듈입니다. 수익 곡선, 드로다운, 월별 수익률 히트맵, 레짐별 성과,
거래 분포, 수익-드로다운 결합 차트를 포함합니다.

Features:
    - ChartBuilder: Converts Plotly figures to embeddable HTML fragments
    - equityCurveChart: Strategy equity curve with optional benchmark
    - drawdownChart: Drawdown area chart with MDD annotation
    - monthlyReturnsHeatmap: Year-by-month return heatmap
    - regimePerformanceChart: Bar charts for regime-based analysis
    - tradeDistributionChart: Histogram and cumulative PnL plots
    - combinedEquityDrawdownChart: Stacked equity + drawdown subplot

Usage:
    from tradex.analytics.reportCharts import ChartBuilder, equityCurveChart

    fig = equityCurveChart(equity_series, benchmark_series)
    builder = ChartBuilder()
    html_fragment = builder.toHtml(fig)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def checkPlotly():
    """
    Verify that Plotly is installed and available.

    Plotly 설치 여부를 확인합니다.

    Raises:
        ImportError: If Plotly is not installed.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly가 필요합니다. 설치: pip install plotly"
        )


class ChartBuilder:
    """
    Plotly figure to HTML converter for embedding in reports.

    Converts Plotly Figure objects into HTML div fragments or full HTML
    documents, applying theme settings and JS inclusion preferences.

    Plotly Figure 객체를 보고서에 삽입 가능한 HTML div 조각 또는 전체 HTML
    문서로 변환합니다.

    Attributes:
        includePlotlyJs (str): JS inclusion mode - 'cdn', True, or False
            (Plotly JS 포함 방식).
        theme (str): Plotly template name (Plotly 테마).

    Example:
        >>> builder = ChartBuilder(theme='plotly_dark')
        >>> html = builder.toHtml(fig)
    """

    def __init__(
        self,
        includePlotlyJs: str = 'cdn',
        theme: str = 'plotly_white',
    ):
        """
        Initialize the ChartBuilder.

        ChartBuilder를 초기화합니다.

        Args:
            includePlotlyJs: JS inclusion strategy - 'cdn' for CDN link,
                True for inline JS, False for external loading
                ('cdn': CDN, True: 인라인, False: 별도 로드).
            theme: Plotly template name. One of 'plotly', 'plotly_white',
                'plotly_dark' (테마).
        """
        checkPlotly()
        self.includePlotlyJs = includePlotlyJs
        self.theme = theme

    def toHtml(
        self,
        fig: 'go.Figure',
        fullHtml: bool = False,
        divId: str = None,
    ) -> str:
        """
        Convert a Plotly Figure to an HTML string.

        Plotly Figure를 HTML 문자열로 변환합니다.

        Args:
            fig: Plotly Figure object to convert.
            fullHtml: If True, generates a complete HTML document;
                if False, generates only a div fragment (기본: div 조각).
            divId: Optional HTML div ID for the chart container.

        Returns:
            str: HTML string representation of the figure.
        """
        fig.update_layout(template=self.theme)
        return fig.to_html(
            full_html=fullHtml,
            include_plotlyjs=self.includePlotlyJs if fullHtml else False,
            div_id=divId,
        )


def equityCurveChart(
    equity: pd.Series,
    benchmark: pd.Series = None,
    title: str = "수익 곡선",
) -> 'go.Figure':
    """
    Generate an equity curve chart with optional benchmark overlay.

    전략의 수익 곡선 차트를 생성합니다. 벤치마크 오버레이를 선택적으로 포함합니다.

    Args:
        equity: Portfolio equity value series with datetime index
            (자산 가치 시리즈).
        benchmark: Optional benchmark equity series; scaled to match
            initial equity value (벤치마크 시리즈, 선택).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with equity curve line chart.
    """
    checkPlotly()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='전략',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='%{x}<br>자산: %{y:,.0f}<extra></extra>',
    ))

    if benchmark is not None:
        scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=scaled.index,
            y=scaled.values,
            mode='lines',
            name='벤치마크',
            line=dict(color='#A23B72', width=1.5, dash='dash'),
            hovertemplate='%{x}<br>벤치마크: %{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="자산 가치",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    return fig


def drawdownChart(
    equity: pd.Series,
    title: str = "드로다운",
) -> 'go.Figure':
    """
    Generate a drawdown area chart with maximum drawdown annotation.

    최대 낙폭(MDD) 주석이 포함된 드로다운 영역 차트를 생성합니다.

    Args:
        equity: Portfolio equity value series with datetime index
            (자산 가치 시리즈).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with filled drawdown area chart.
    """
    checkPlotly()

    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        fill='tozeroy',
        name='드로다운',
        line=dict(color='#E94F37', width=1),
        fillcolor='rgba(233, 79, 55, 0.3)',
        hovertemplate='%{x}<br>드로다운: %{y:.1f}%<extra></extra>',
    ))

    maxDd = drawdown.min()
    maxDdDate = drawdown.idxmin()

    fig.add_annotation(
        x=maxDdDate,
        y=maxDd,
        text=f"최대 낙폭: {maxDd:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#E94F37",
        font=dict(color="#E94F37"),
    )

    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="드로다운 (%)",
        hovermode='x unified',
        margin=dict(l=60, r=30, t=50, b=50),
    )

    fig.update_yaxes(range=[min(maxDd * 1.1, -50), 5])

    return fig


def monthlyReturnsHeatmap(
    returns: pd.Series,
    title: str = "월별 수익률",
) -> 'go.Figure':
    """
    Generate a monthly returns heatmap organized by year and month.

    일별 수익률로부터 연도/월 기준 수익률 히트맵을 생성합니다.

    Args:
        returns: Daily returns series with datetime index (일별 수익률 시리즈).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with color-coded monthly returns heatmap.
    """
    checkPlotly()

    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

    pivot = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values,
    }).pivot(index='year', columns='month', values='return')

    monthNames = ['1월', '2월', '3월', '4월', '5월', '6월',
                  '7월', '8월', '9월', '10월', '11월', '12월']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=monthNames[:pivot.shape[1]],
        y=pivot.index.astype(str),
        colorscale=[
            [0, '#E94F37'],
            [0.5, '#FFFFFF'],
            [1, '#2E86AB'],
        ],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text:.1f}%',
        textfont=dict(size=10),
        hovertemplate='%{y}년 %{x}<br>수익률: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='수익률 (%)'),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="월",
        yaxis_title="연도",
        margin=dict(l=60, r=30, t=50, b=50),
    )

    fig.update_yaxes(autorange='reversed')

    return fig


def regimePerformanceChart(
    regimeStats: Dict[str, Dict[str, float]],
    title: str = "레짐별 성과",
) -> 'go.Figure':
    """
    Generate a regime performance comparison chart.

    시장 레짐별 수익률 및 샤프 비율을 비교하는 차트를 생성합니다.

    Args:
        regimeStats: Dict mapping regime names to performance dicts with
            'return', 'sharpe', and 'trades' keys
            ({레짐명: {return, sharpe, trades, ...}}).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with side-by-side return and Sharpe bar charts.
    """
    checkPlotly()

    regimes = list(regimeStats.keys())
    returns = [regimeStats[r].get('return', 0) * 100 for r in regimes]
    sharpes = [regimeStats[r].get('sharpe', 0) for r in regimes]
    trades = [regimeStats[r].get('trades', 0) for r in regimes]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('수익률 (%)', '샤프 비율'),
        horizontal_spacing=0.15,
    )

    colors = ['#2E86AB' if r >= 0 else '#E94F37' for r in returns]

    fig.add_trace(
        go.Bar(
            x=regimes,
            y=returns,
            marker_color=colors,
            text=[f'{r:.1f}%' for r in returns],
            textposition='outside',
            name='수익률',
            hovertemplate='%{x}<br>수익률: %{y:.1f}%<extra></extra>',
        ),
        row=1, col=1
    )

    sharpeColors = ['#2E86AB' if s >= 0 else '#E94F37' for s in sharpes]

    fig.add_trace(
        go.Bar(
            x=regimes,
            y=sharpes,
            marker_color=sharpeColors,
            text=[f'{s:.2f}' for s in sharpes],
            textposition='outside',
            name='샤프',
            hovertemplate='%{x}<br>샤프: %{y:.2f}<extra></extra>',
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=50),
    )

    return fig


def tradeDistributionChart(
    trades: pd.DataFrame,
    title: str = "거래 분포",
) -> 'go.Figure':
    """
    Generate a trade return distribution histogram and cumulative PnL chart.

    거래 수익률 분포 히스토그램과 누적 손익 차트를 생성합니다.

    Args:
        trades: Trade DataFrame requiring one of 'pnlPct', 'return', or 'pnl'
            columns (거래 DataFrame, pnlPct/return/pnl 컬럼 필요).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with histogram and cumulative PnL subplots.
    """
    checkPlotly()

    if 'pnlPct' in trades.columns:
        pnl = trades['pnlPct'].values * 100
    elif 'return' in trades.columns:
        pnl = trades['return'].values * 100
    elif 'pnl' in trades.columns:
        pnl = trades['pnl'].values
    else:
        pnl = np.array([])

    if len(pnl) == 0:
        fig = go.Figure()
        fig.add_annotation(text="거래 데이터 없음", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('수익률 분포', '누적 수익'),
        horizontal_spacing=0.15,
    )

    colors = ['#2E86AB' if p >= 0 else '#E94F37' for p in pnl]

    fig.add_trace(
        go.Histogram(
            x=pnl,
            nbinsx=30,
            marker_color='#2E86AB',
            opacity=0.7,
            name='분포',
            hovertemplate='수익률: %{x:.1f}%<br>빈도: %{y}<extra></extra>',
        ),
        row=1, col=1
    )

    cumPnl = np.cumsum(pnl)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumPnl) + 1)),
            y=cumPnl,
            mode='lines+markers',
            marker=dict(size=4, color=colors),
            line=dict(color='#2E86AB', width=1),
            name='누적',
            hovertemplate='거래 #%{x}<br>누적: %{y:.1f}%<extra></extra>',
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=50),
    )

    fig.update_xaxes(title_text="수익률 (%)", row=1, col=1)
    fig.update_xaxes(title_text="거래 번호", row=1, col=2)
    fig.update_yaxes(title_text="빈도", row=1, col=1)
    fig.update_yaxes(title_text="누적 수익률 (%)", row=1, col=2)

    return fig


def combinedEquityDrawdownChart(
    equity: pd.Series,
    title: str = "수익 곡선 & 드로다운",
) -> 'go.Figure':
    """
    Generate a combined equity curve and drawdown subplot chart.

    수익 곡선과 드로다운을 상하 서브플롯으로 결합한 차트를 생성합니다.

    Args:
        equity: Portfolio equity value series with datetime index
            (자산 가치 시리즈).
        title: Chart title (차트 제목).

    Returns:
        go.Figure: Plotly Figure with equity curve (top) and drawdown (bottom)
            sharing the x-axis.
    """
    checkPlotly()

    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('자산 가치', '드로다운'),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='자산',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='%{x}<br>자산: %{y:,.0f}<extra></extra>',
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='드로다운',
            line=dict(color='#E94F37', width=1),
            fillcolor='rgba(233, 79, 55, 0.3)',
            hovertemplate='%{x}<br>드로다운: %{y:.1f}%<extra></extra>',
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=50),
    )

    fig.update_yaxes(title_text="자산 가치", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_xaxes(title_text="날짜", row=2, col=1)

    return fig
