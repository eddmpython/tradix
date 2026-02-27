"""
Tradix Backtest Chart Module.

Provides interactive Plotly-based visualizations for backtest results,
including equity curves, drawdown charts, monthly return heatmaps,
trade analysis, and a combined dashboard view.

Plotly 기반의 인터랙티브 백테스트 결과 시각화 모듈입니다.
자산 곡선, 드로다운 차트, 월별 수익률 히트맵, 거래 분석,
종합 대시보드를 제공합니다.

Features:
    - Equity curve with optional benchmark overlay
    - Drawdown visualization with MDD annotation
    - Monthly returns heatmap by year/month
    - Trade PnL bar chart and win/loss pie chart
    - All-in-one dashboard combining all views
    - HTML export for sharing and archiving

Usage:
    from tradix.analytics.charts import BacktestChart

    chart = BacktestChart(result)
    chart.show('dashboard')
    chart.saveHtml('backtest_result.html')
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from tradix.engine import BacktestResult


class BacktestChart:
    """
    Interactive backtest result visualizer built on Plotly.

    Generates equity curves, drawdown charts, monthly return heatmaps,
    trade analysis charts, and a combined dashboard from a BacktestResult.

    Plotly 기반의 인터랙티브 백테스트 결과 시각화 클래스입니다.
    자산 곡선, 드로다운, 월별 수익률 히트맵, 거래 분석, 종합 대시보드를 생성합니다.

    Attributes:
        result (BacktestResult): The backtest result to visualize (백테스트 결과).
        title (str): Chart title displayed on figures (차트 제목).

    Example:
        >>> from tradix.analytics.charts import BacktestChart
        >>> result = engine.run()
        >>> chart = BacktestChart(result)
        >>> chart.equityCurve()
        >>> chart.dashboard()
        >>> chart.saveHtml('backtest_result.html')
    """

    def __init__(self, result: BacktestResult, title: str = None):
        """
        Initialize the BacktestChart.

        BacktestChart를 초기화합니다.

        Args:
            result: BacktestResult object containing equity curve, trades,
                and metrics (백테스트 결과 객체).
            title: Optional chart title; defaults to strategy name from result
                (차트 제목, 미입력 시 전략 이름 사용).
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly가 설치되어 있지 않습니다: pip install plotly")

        self.result = result
        self.title = title or f"백테스트: {result.strategy}"

        self._colorUp = '#26A69A'
        self._colorDown = '#EF5350'
        self._colorEquity = '#2196F3'
        self._colorDrawdown = '#FF5722'
        self._colorBenchmark = '#9E9E9E'

    def equityCurve(self, showBenchmark: bool = True, height: int = 500) -> go.Figure:
        """
        Generate an equity curve chart with optional benchmark and trade markers.

        자산 곡선 차트를 생성합니다. 벤치마크 오버레이 및 매수/매도 마커를 포함합니다.

        Args:
            showBenchmark: Whether to overlay benchmark equity if available
                (벤치마크 표시 여부).
            height: Chart height in pixels (차트 높이, 픽셀).

        Returns:
            go.Figure: Plotly Figure object with the equity curve.
        """
        fig = go.Figure()

        equity = self.result.equityCurve
        if equity is None or len(equity) == 0:
            return fig

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            name='포트폴리오',
            line=dict(color=self._colorEquity, width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)',
        ))

        if showBenchmark and hasattr(self.result, 'benchmarkEquity'):
            benchmark = self.result.benchmarkEquity
            if benchmark is not None and len(benchmark) > 0:
                fig.add_trace(go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    name='벤치마크 (B&H)',
                    line=dict(color=self._colorBenchmark, width=1, dash='dash'),
                ))

        trades = self.result.trades
        if trades:
            buyDates = []
            buyValues = []
            sellDates = []
            sellValues = []

            for trade in trades:
                if hasattr(trade, 'entryTime') and trade.entryTime in equity.index:
                    buyDates.append(trade.entryTime)
                    buyValues.append(equity.loc[trade.entryTime])

                if hasattr(trade, 'exitTime') and trade.exitTime and trade.exitTime in equity.index:
                    sellDates.append(trade.exitTime)
                    sellValues.append(equity.loc[trade.exitTime])

            if buyDates:
                fig.add_trace(go.Scatter(
                    x=buyDates,
                    y=buyValues,
                    mode='markers',
                    name='매수',
                    marker=dict(color=self._colorUp, size=8, symbol='triangle-up'),
                ))

            if sellDates:
                fig.add_trace(go.Scatter(
                    x=sellDates,
                    y=sellValues,
                    mode='markers',
                    name='매도',
                    marker=dict(color=self._colorDown, size=8, symbol='triangle-down'),
                ))

        fig.update_layout(
            title=f'{self.title} - 자산 곡선',
            xaxis_title='날짜',
            yaxis_title='자산 (원)',
            height=height,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            yaxis=dict(tickformat=','),
        )

        return fig

    def drawdown(self, height: int = 300) -> go.Figure:
        """
        Generate a drawdown chart with maximum drawdown annotation.

        최대 낙폭(MDD) 주석이 포함된 드로다운 차트를 생성합니다.

        Args:
            height: Chart height in pixels (차트 높이, 픽셀).

        Returns:
            go.Figure: Plotly Figure object with the drawdown area chart.
        """
        fig = go.Figure()

        equity = self.result.equityCurve
        if equity is None or len(equity) == 0:
            return fig

        runningMax = equity.cummax()
        drawdown = (equity - runningMax) / runningMax * 100

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='드로다운',
            line=dict(color=self._colorDrawdown, width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 87, 34, 0.3)',
        ))

        maxDd = drawdown.min()
        maxDdDate = drawdown.idxmin()

        fig.add_annotation(
            x=maxDdDate,
            y=maxDd,
            text=f'MDD: {maxDd:.1f}%',
            showarrow=True,
            arrowhead=2,
            arrowcolor=self._colorDrawdown,
            font=dict(color=self._colorDrawdown),
        )

        fig.update_layout(
            title='드로다운',
            xaxis_title='날짜',
            yaxis_title='드로다운 (%)',
            height=height,
            hovermode='x unified',
            yaxis=dict(ticksuffix='%'),
        )

        return fig

    def monthlyReturns(self, height: int = 400) -> go.Figure:
        """
        Generate a monthly returns heatmap organized by year and month.

        연도별/월별 수익률 히트맵을 생성합니다.

        Args:
            height: Chart height in pixels (차트 높이, 픽셀).

        Returns:
            go.Figure: Plotly Figure object with the heatmap.
        """
        fig = go.Figure()

        equity = self.result.equityCurve
        if equity is None or len(equity) == 0:
            return fig

        returns = equity.pct_change().dropna()
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

        if len(monthly) == 0:
            return fig

        df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })

        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['1월', '2월', '3월', '4월', '5월', '6월',
                         '7월', '8월', '9월', '10월', '11월', '12월'][:len(pivot.columns)]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, self._colorDown],
                [0.5, '#FFFFFF'],
                [1, self._colorUp]
            ],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont=dict(size=10),
            hovertemplate='%{y}년 %{x}: %{z:.1f}%<extra></extra>',
        ))

        fig.update_layout(
            title='월별 수익률 (%)',
            height=height,
            xaxis_title='월',
            yaxis_title='연도',
            yaxis=dict(autorange='reversed'),
        )

        return fig

    def tradeAnalysis(self, height: int = 400) -> go.Figure:
        """
        Generate a trade analysis chart with PnL bar chart and win/loss pie.

        거래별 손익 막대 차트와 승/패 비율 파이 차트를 생성합니다.

        Args:
            height: Chart height in pixels (차트 높이, 픽셀).

        Returns:
            go.Figure: Plotly Figure with side-by-side trade analysis subplots.
        """
        trades = self.result.trades
        if not trades:
            return go.Figure()

        pnls = [t.pnl for t in trades if hasattr(t, 'pnl')]
        if not pnls:
            return go.Figure()

        colors = [self._colorUp if p >= 0 else self._colorDown for p in pnls]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'bar'}, {'type': 'pie'}]],
            subplot_titles=('거래별 손익', '승/패 비율'),
        )

        fig.add_trace(
            go.Bar(
                x=list(range(1, len(pnls) + 1)),
                y=pnls,
                marker_color=colors,
                name='손익',
                hovertemplate='거래 %{x}: %{y:,.0f}원<extra></extra>',
            ),
            row=1, col=1
        )

        wins = sum(1 for p in pnls if p >= 0)
        losses = len(pnls) - wins

        fig.add_trace(
            go.Pie(
                labels=['승', '패'],
                values=[wins, losses],
                marker_colors=[self._colorUp, self._colorDown],
                textinfo='label+percent',
                hole=0.4,
            ),
            row=1, col=2
        )

        fig.update_layout(
            title='거래 분석',
            height=height,
            showlegend=False,
        )

        return fig

    def dashboard(self, height: int = 1200) -> go.Figure:
        """
        Generate a comprehensive dashboard combining all chart types.

        자산 곡선, 드로다운, 거래 분석, 월별 수익률을 하나의 대시보드로 결합합니다.

        Args:
            height: Chart height in pixels (차트 높이, 픽셀).

        Returns:
            go.Figure: Plotly Figure with multi-row dashboard layout.
        """
        fig = make_subplots(
            rows=4, cols=2,
            specs=[
                [{'colspan': 2}, None],
                [{'colspan': 2}, None],
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'colspan': 2}, None],
            ],
            subplot_titles=(
                '자산 곡선',
                '드로다운',
                '거래별 손익', '승/패 비율',
                '월별 수익률',
            ),
            row_heights=[0.35, 0.2, 0.25, 0.2],
            vertical_spacing=0.08,
        )

        equity = self.result.equityCurve
        if equity is not None and len(equity) > 0:
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    name='포트폴리오',
                    line=dict(color=self._colorEquity, width=2),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.1)',
                ),
                row=1, col=1
            )

            runningMax = equity.cummax()
            drawdown = (equity - runningMax) / runningMax * 100

            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name='드로다운',
                    line=dict(color=self._colorDrawdown, width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 87, 34, 0.3)',
                ),
                row=2, col=1
            )

        trades = self.result.trades
        if trades:
            pnls = [t.pnl for t in trades if hasattr(t, 'pnl')]
            if pnls:
                colors = [self._colorUp if p >= 0 else self._colorDown for p in pnls]

                fig.add_trace(
                    go.Bar(
                        x=list(range(1, len(pnls) + 1)),
                        y=pnls,
                        marker_color=colors,
                        name='손익',
                    ),
                    row=3, col=1
                )

                wins = sum(1 for p in pnls if p >= 0)
                losses = len(pnls) - wins

                fig.add_trace(
                    go.Pie(
                        labels=['승', '패'],
                        values=[wins, losses],
                        marker_colors=[self._colorUp, self._colorDown],
                        textinfo='label+percent',
                        hole=0.4,
                    ),
                    row=3, col=2
                )

        if equity is not None and len(equity) > 0:
            returns = equity.pct_change().dropna()
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

            if len(monthly) > 0:
                colors = [self._colorUp if r >= 0 else self._colorDown for r in monthly.values]

                fig.add_trace(
                    go.Bar(
                        x=monthly.index,
                        y=monthly.values,
                        marker_color=colors,
                        name='월별 수익률',
                    ),
                    row=4, col=1
                )

        metrics = self.result.metrics
        metricsText = (
            f"총 수익률: {metrics.get('totalReturn', 0):.2f}% | "
            f"연간 수익률: {metrics.get('annualReturn', 0):.2f}% | "
            f"샤프: {metrics.get('sharpeRatio', 0):.2f} | "
            f"MDD: {metrics.get('maxDrawdown', 0):.2f}% | "
            f"승률: {metrics.get('winRate', 0):.1f}%"
        )

        fig.update_layout(
            title=dict(
                text=f'{self.title}<br><sub>{metricsText}</sub>',
                x=0.5,
            ),
            height=height,
            showlegend=False,
            hovermode='x unified',
        )

        fig.update_yaxes(tickformat=',', row=1, col=1)
        fig.update_yaxes(ticksuffix='%', row=2, col=1)
        fig.update_yaxes(tickformat=',', row=3, col=1)
        fig.update_yaxes(ticksuffix='%', row=4, col=1)

        return fig

    def saveHtml(self, filepath: str, chart: str = 'dashboard'):
        """
        Save a chart as a standalone HTML file.

        차트를 독립 실행 가능한 HTML 파일로 저장합니다.

        Args:
            filepath: Destination file path (저장 경로).
            chart: Chart type to render. One of 'dashboard', 'equity',
                'drawdown', 'monthly', 'trades' (차트 종류).

        Raises:
            ValueError: If the specified chart type is not supported.
        """
        chartMap = {
            'dashboard': self.dashboard,
            'equity': self.equityCurve,
            'drawdown': self.drawdown,
            'monthly': self.monthlyReturns,
            'trades': self.tradeAnalysis,
        }

        if chart not in chartMap:
            raise ValueError(f"지원하지 않는 차트: {chart}")

        fig = chartMap[chart]()
        fig.write_html(filepath)
        print(f"저장 완료: {filepath}")

    def show(self, chart: str = 'dashboard'):
        """
        Display a chart in the default browser or notebook.

        차트를 기본 브라우저 또는 노트북에 표시합니다.

        Args:
            chart: Chart type to display. One of 'dashboard', 'equity',
                'drawdown', 'monthly', 'trades' (차트 종류).

        Raises:
            ValueError: If the specified chart type is not supported.
        """
        chartMap = {
            'dashboard': self.dashboard,
            'equity': self.equityCurve,
            'drawdown': self.drawdown,
            'monthly': self.monthlyReturns,
            'trades': self.tradeAnalysis,
        }

        if chart not in chartMap:
            raise ValueError(f"지원하지 않는 차트: {chart}")

        fig = chartMap[chart]()
        fig.show()
