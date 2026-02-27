"""
Tradix Tearsheet Module.

Generates QuantStats-style HTML performance tearsheets from backtest results,
presenting key metrics, trade history, and summary statistics in a clean,
responsive layout.

백테스트 결과를 기반으로 QuantStats 스타일의 HTML 성과 리포트(Tearsheet)를
생성하는 모듈입니다.

Features:
    - Responsive HTML report with CSS styling
    - Performance metrics dashboard (return, Sharpe, MDD, win rate)
    - Return, risk, and trade statistics sections
    - Tabular trade history (up to 50 most recent trades)
    - One-click file export via save()

Usage:
    from tradix.analytics.tearsheet import Tearsheet

    tearsheet = Tearsheet(
        strategyName="SMA Cross",
        symbol="005930",
        metrics=metrics,
        trades=trades,
        equityCurve=equityCurve,
    )
    tearsheet.save("report.html")
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import pandas as pd

from tradix.entities.trade import Trade
from tradix.analytics.metrics import PerformanceMetrics


@dataclass
class Tearsheet:
    """
    HTML performance tearsheet generator for backtest results.

    Produces a self-contained, QuantStats-style HTML report that includes
    headline metrics cards, detailed return/risk/trade statistics sections,
    and a recent trade history table.

    백테스트 결과를 위한 HTML 성과 리포트 생성기입니다. 주요 지표 카드, 수익률/리스크/
    거래 통계 섹션, 최근 거래 내역 테이블을 포함하는 독립 실행 가능한 HTML 리포트를
    생성합니다.

    Attributes:
        strategyName (str): Name of the strategy (전략 이름).
        symbol (str): Ticker symbol for the instrument (종목 코드).
        metrics (PerformanceMetrics): Calculated performance metrics (성과 지표).
        trades (List[Trade]): List of executed trades (거래 목록).
        equityCurve (pd.Series): Portfolio equity over time (자산 곡선).
        startDate (str): Backtest start date string (백테스트 시작일).
        endDate (str): Backtest end date string (백테스트 종료일).
        initialCash (float): Initial capital amount (초기 자본금).

    Example:
        >>> tearsheet = Tearsheet(
        ...     strategyName="SMA Cross",
        ...     symbol="005930",
        ...     metrics=metrics,
        ...     trades=trades,
        ...     equityCurve=equityCurve,
        ... )
        >>> tearsheet.save("report.html")
    """
    strategyName: str
    symbol: str
    metrics: PerformanceMetrics
    trades: List[Trade]
    equityCurve: pd.Series
    startDate: str = ""
    endDate: str = ""
    initialCash: float = 0.0

    def __post_init__(self):
        if self.equityCurve is not None and len(self.equityCurve) > 0:
            if not self.startDate:
                self.startDate = str(self.equityCurve.index[0].date())
            if not self.endDate:
                self.endDate = str(self.equityCurve.index[-1].date())
            if not self.initialCash:
                self.initialCash = self.equityCurve.iloc[0]

    def generateHtml(self) -> str:
        """
        Generate the full HTML tearsheet as a string.

        전체 HTML 리포트를 문자열로 생성합니다.

        Returns:
            str: Complete HTML document string with embedded CSS.
        """
        m = self.metrics

        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>백테스트 리포트 - {self.strategyName}</title>
    <style>
        :root {{
            --ct-text: #09090b;
            --ct-text-secondary: #52525b;
            --ct-text-muted: #71717a;
            --ct-border: #e4e4e7;
            --ct-bg: #ffffff;
            --ct-bg-alt: #f4f4f5;
            --ct-card: #ffffff;
            --ct-success: #22c55e;
            --ct-danger: #ef4444;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--ct-bg-alt);
            color: var(--ct-text);
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 32px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }}

        .header .meta {{
            color: var(--ct-text-muted);
            font-size: 14px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .card {{
            background: var(--ct-card);
            border: 1px solid var(--ct-border);
            border-radius: 12px;
            padding: 20px;
        }}

        .card-title {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--ct-text-muted);
            margin-bottom: 12px;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--ct-border);
        }}

        .metric:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            color: var(--ct-text-secondary);
            font-size: 14px;
        }}

        .metric-value {{
            font-weight: 600;
            font-size: 14px;
        }}

        .metric-value.positive {{
            color: var(--ct-success);
        }}

        .metric-value.negative {{
            color: var(--ct-danger);
        }}

        .big-number {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .big-label {{
            font-size: 13px;
            color: var(--ct-text-muted);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--ct-border);
        }}

        th {{
            font-weight: 600;
            color: var(--ct-text-muted);
            font-size: 11px;
            text-transform: uppercase;
        }}

        tr:hover {{
            background: var(--ct-bg-alt);
        }}

        .text-right {{
            text-align: right;
        }}

        .footer {{
            text-align: center;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--ct-border);
            color: var(--ct-text-muted);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.strategyName}</h1>
            <div class="meta">
                {self.symbol} | {self.startDate} ~ {self.endDate} |
                초기자본: {self.initialCash:,.0f}원
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="big-number {'positive' if m.totalReturn >= 0 else 'negative'}">
                    {m.totalReturn:+.2f}%
                </div>
                <div class="big-label">총 수익률</div>
            </div>
            <div class="card">
                <div class="big-number">{m.sharpeRatio:.2f}</div>
                <div class="big-label">샤프 비율</div>
            </div>
            <div class="card">
                <div class="big-number negative">{m.maxDrawdown:.2f}%</div>
                <div class="big-label">최대 낙폭</div>
            </div>
            <div class="card">
                <div class="big-number">{m.winRate:.1f}%</div>
                <div class="big-label">승률</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">수익률 지표</div>
                <div class="metric">
                    <span class="metric-label">연율화 수익률</span>
                    <span class="metric-value {'positive' if m.annualReturn >= 0 else 'negative'}">
                        {m.annualReturn:+.2f}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">변동성 (연율)</span>
                    <span class="metric-value">{m.volatility:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">소르티노 비율</span>
                    <span class="metric-value">{m.sortinoRatio:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">칼마 비율</span>
                    <span class="metric-value">{m.calmarRatio:.2f}</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">리스크 지표</div>
                <div class="metric">
                    <span class="metric-label">최대 낙폭</span>
                    <span class="metric-value negative">{m.maxDrawdown:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">낙폭 지속기간</span>
                    <span class="metric-value">{m.maxDrawdownDuration}일</span>
                </div>
                <div class="metric">
                    <span class="metric-label">최악의 달</span>
                    <span class="metric-value negative">{m.worstMonth:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">최고의 달</span>
                    <span class="metric-value positive">{m.bestMonth:.2f}%</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">거래 통계</div>
                <div class="metric">
                    <span class="metric-label">총 거래</span>
                    <span class="metric-value">{m.totalTrades}회</span>
                </div>
                <div class="metric">
                    <span class="metric-label">수익 거래</span>
                    <span class="metric-value positive">{m.winningTrades}회</span>
                </div>
                <div class="metric">
                    <span class="metric-label">손실 거래</span>
                    <span class="metric-value negative">{m.losingTrades}회</span>
                </div>
                <div class="metric">
                    <span class="metric-label">평균 보유기간</span>
                    <span class="metric-value">{m.avgHoldingDays:.1f}일</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">손익 분석</div>
                <div class="metric">
                    <span class="metric-label">평균 수익</span>
                    <span class="metric-value positive">{m.avgWin:,.0f}원 ({m.avgWinPercent:+.2f}%)</span>
                </div>
                <div class="metric">
                    <span class="metric-label">평균 손실</span>
                    <span class="metric-value negative">{m.avgLoss:,.0f}원 ({m.avgLossPercent:.2f}%)</span>
                </div>
                <div class="metric">
                    <span class="metric-label">손익비</span>
                    <span class="metric-value">{m.profitFactor:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">기대수익</span>
                    <span class="metric-value">{m.expectancy:.2f}%</span>
                </div>
            </div>
        </div>

        {self._generateTradeTable()}

        <div class="footer">
            Generated by Tradix | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """

        return html

    def _generateTradeTable(self) -> str:
        """
        Generate the HTML trade history table section.

        최근 50건의 거래 내역을 HTML 테이블로 생성합니다.

        Returns:
            str: HTML string for the trade table, or empty string if no trades.
        """
        if not self.trades:
            return ""

        rows = []
        for i, t in enumerate(self.trades[:50], 1):
            pnlClass = 'positive' if t.pnl > 0 else 'negative'
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{t.entryDate.strftime('%Y-%m-%d') if t.entryDate else '-'}</td>
                    <td>{t.exitDate.strftime('%Y-%m-%d') if t.exitDate else '-'}</td>
                    <td>{t.side.value.upper()}</td>
                    <td class="text-right">{t.entryPrice:,.0f}</td>
                    <td class="text-right">{t.exitPrice:,.0f if t.exitPrice else '-'}</td>
                    <td class="text-right">{t.quantity:,.0f}</td>
                    <td class="text-right {pnlClass}">{t.pnl:+,.0f}</td>
                    <td class="text-right {pnlClass}">{t.pnlPercent:+.2f}%</td>
                    <td class="text-right">{t.holdingDays}일</td>
                </tr>
            """)

        return f"""
        <div class="card" style="margin-top: 24px;">
            <div class="card-title">거래 내역 (최근 50건)</div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>진입일</th>
                        <th>청산일</th>
                        <th>방향</th>
                        <th class="text-right">진입가</th>
                        <th class="text-right">청산가</th>
                        <th class="text-right">수량</th>
                        <th class="text-right">손익</th>
                        <th class="text-right">수익률</th>
                        <th class="text-right">보유</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def save(self, filepath: str):
        """
        Save the HTML tearsheet to a file.

        HTML 리포트를 파일로 저장합니다.

        Args:
            filepath: Destination file path for the HTML report (저장 경로).
        """
        html = self.generateHtml()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    def __repr__(self) -> str:
        return (
            f"Tearsheet({self.strategyName}, "
            f"{self.symbol}, "
            f"{len(self.trades)} trades)"
        )
