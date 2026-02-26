"""
TradingView-inspired terminal charts using Plotext.

Renders candlestick + volume + indicator overlays + trade markers + drawdown
+ monthly heatmap + rolling metrics + trade scatter + correlation bars
+ strategy DNA + seasonality directly in the terminal.

TradingView 스타일 터미널 차트. 캔들스틱+거래량+지표 오버레이+매매 마커+낙폭
+월별 히트맵+롤링 지표+거래 산점도+상관 바+전략 DNA+계절성을
터미널에 직접 렌더링합니다.
"""

from typing import Any, Dict, List, Optional
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import plotext as plt


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _unwrap(result: Any) -> Any:
    """Unwrap proxy result objects. 프록시 결과 객체를 언래핑합니다."""
    if hasattr(result, '_result'):
        return result._result
    return result


def _getEquityList(r: Any) -> list:
    """Extract equity curve as a plain list. 자산 곡선을 리스트로 추출합니다."""
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


def _getDateLabels(r: Any) -> list:
    """Extract date labels from equity curve index. 자산 곡선 인덱스에서 날짜 레이블을 추출합니다."""
    ec = getattr(r, 'equityCurve', None)
    try:
        import pandas as pd
        if isinstance(ec, pd.Series) and hasattr(ec.index, 'strftime'):
            return ec.index.strftime('%Y-%m-%d').tolist()
    except (ImportError, AttributeError):
        pass
    return []


def _sma(data: list, period: int) -> list:
    """Simple moving average. 단순 이동평균."""
    result = [None] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result


def _dateTicks(dates: list, nTicks: int = 6) -> tuple:
    """Generate evenly spaced date tick positions and labels. 균등 간격 날짜 틱 위치와 레이블을 생성합니다."""
    if not dates:
        return [], []
    n = len(dates)
    step = max(1, n // nTicks)
    indices = list(range(0, n, step))
    labels = [dates[i] for i in indices]
    return indices, labels


def _getMetric(r: Any, name: str, default: float = 0.0) -> float:
    """Get a metric value from result object by name. 결과 객체에서 이름으로 지표 값을 가져옵니다."""
    val = getattr(r, name, None)
    if val is not None:
        return float(val)
    metrics = getattr(r, 'metrics', {})
    if isinstance(metrics, dict):
        return float(metrics.get(name, default))
    return default


def _getReturns(r: Any) -> list:
    """Compute daily returns from equity curve. 자산 곡선으로부터 일별 수익률을 계산합니다."""
    data = _getEquityList(r)
    if not data or len(data) < 2:
        return []
    returns = []
    for i in range(1, len(data)):
        if data[i - 1] != 0:
            returns.append((data[i] / data[i - 1] - 1) * 100)
        else:
            returns.append(0.0)
    return returns


def _getMonthlyReturns(r: Any) -> Dict[int, Dict[int, float]]:
    """Compute monthly returns grouped by year and month.
    연도와 월별로 그룹화된 월간 수익률을 계산합니다.

    Returns:
        Dict mapping year -> {month -> return%}.
        연도 -> {월 -> 수익률%} 매핑 딕셔너리.
    """
    ec = getattr(r, 'equityCurve', None)
    if ec is None:
        return {}

    try:
        import pandas as pd
        if not isinstance(ec, pd.Series):
            return {}
        if not hasattr(ec.index, 'year'):
            return {}
    except ImportError:
        return {}

    monthly = {}
    years = sorted(set(ec.index.year))
    for year in years:
        monthly[year] = {}
        yearData = ec[ec.index.year == year]
        months = sorted(set(yearData.index.month))
        for month in months:
            monthData = yearData[yearData.index.month == month]
            if len(monthData) >= 2:
                ret = (monthData.iloc[-1] / monthData.iloc[0] - 1) * 100
                monthly[year][month] = ret
            elif len(monthData) == 1 and month > 1:
                prevMonths = yearData[yearData.index.month < month]
                if len(prevMonths) > 0:
                    ret = (monthData.iloc[0] / prevMonths.iloc[-1] - 1) * 100
                    monthly[year][month] = ret

    return monthly


def _getTradeData(r: Any) -> Dict[str, list]:
    """Extract trade arrays from result object.
    결과 객체에서 거래 배열을 추출합니다.

    Returns:
        Dict with keys: entryDates, exitDates, pnls, pnlPercents, holdingDays, isWins.
        키가 entryDates, exitDates, pnls, pnlPercents, holdingDays, isWins인 딕셔너리.
    """
    trades = getattr(r, 'trades', [])
    entryDates = []
    exitDates = []
    pnls = []
    pnlPercents = []
    holdingDays = []
    isWins = []

    for t in trades:
        if isinstance(t, dict):
            isClosed = t.get('exitDate') is not None or t.get('exitPrice') is not None
            if not isClosed:
                continue
            entryDates.append(t.get('entryDate'))
            exitDates.append(t.get('exitDate'))
            pnl = t.get('pnl', 0.0)
            pnls.append(float(pnl))
            pnlPercents.append(float(t.get('pnlPercent', 0.0)))
            holdingDays.append(int(t.get('holdingDays', 0)))
            isWins.append(pnl > 0)
        else:
            if hasattr(t, 'isClosed') and not t.isClosed:
                continue
            entryDates.append(getattr(t, 'entryDate', None))
            exitDates.append(getattr(t, 'exitDate', None))
            pnl = getattr(t, 'pnl', 0.0)
            pnls.append(float(pnl))
            pnlPct = getattr(t, 'pnlPercent', 0.0)
            pnlPercents.append(float(pnlPct))
            hd = getattr(t, 'holdingDays', 0)
            holdingDays.append(int(hd))
            isWins.append(pnl > 0)

    return {
        "entryDates": entryDates,
        "exitDates": exitDates,
        "pnls": pnls,
        "pnlPercents": pnlPercents,
        "holdingDays": holdingDays,
        "isWins": isWins,
    }


def plotEquityCurve(
    result: Any,
    title: str = "Equity Curve",
    height: int = 18,
    width: Optional[int] = None,
    smaPeriods: List[int] = None,
    benchmark: Any = None,
) -> None:
    """Plot equity curve with optional SMA overlays, benchmark, and trade markers.

    자산 곡선을 SMA 오버레이, 벤치마크, 매매 마커와 함께 플롯합니다.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
        smaPeriods: SMA periods to overlay (e.g., [20, 60]). SMA 기간 오버레이.
        benchmark: Optional second result for benchmark overlay. 벤치마크 오버레이용 두 번째 결과.
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data:
        print("[No equity data]")
        return

    plt.clear_figure()
    w = width or plt.terminal_width()
    plt.plotsize(w, height)
    plt.title(title)
    plt.theme("dark")

    plt.plot(data, label="Equity", color="cyan")

    initial = getattr(r, 'initialCash', data[0] if data else 0)
    if initial:
        plt.hline(initial, color="gray")

    if benchmark is not None:
        bR = _unwrap(benchmark)
        bData = _getEquityList(bR)
        if bData:
            if len(bData) != len(data) and len(bData) > 0 and len(data) > 0:
                scale = data[0] / bData[0] if bData[0] != 0 else 1.0
                bDataScaled = [v * scale for v in bData]
            else:
                bDataScaled = bData
            plt.plot(bDataScaled, label="Benchmark", color="orange")

    if smaPeriods:
        colors = ["yellow", "magenta", "white"]
        for i, period in enumerate(smaPeriods):
            ma = _sma(data, period)
            valid = [(j, v) for j, v in enumerate(ma) if v is not None]
            if valid:
                plt.plot(
                    [j for j, _ in valid],
                    [v for _, v in valid],
                    label=f"SMA{period}",
                    color=colors[i % len(colors)],
                )

    trades = getattr(r, 'trades', [])
    dates = _getDateLabels(r)
    if trades and dates:
        dateToIdx = {d: i for i, d in enumerate(dates)}
        buyX, buyY = [], []
        sellX, sellY = [], []

        for t in trades:
            if isinstance(t, dict):
                entryD = str(t.get('entryDate', ''))[:10]
                exitD = str(t.get('exitDate', ''))[:10]
            else:
                entryD = str(getattr(t, 'entryDate', ''))[:10]
                exitD = str(getattr(t, 'exitDate', ''))[:10]

            if entryD in dateToIdx:
                idx = dateToIdx[entryD]
                buyX.append(idx)
                buyY.append(data[idx])
            if exitD in dateToIdx:
                idx = dateToIdx[exitD]
                sellX.append(idx)
                sellY.append(data[idx])

        if buyX:
            plt.scatter(buyX, buyY, color="green", label="Buy", marker="dot")
        if sellX:
            plt.scatter(sellX, sellY, color="red", label="Sell", marker="dot")

    if dates and len(dates) == len(data):
        idx, lbl = _dateTicks(dates, nTicks=8)
        plt.xticks(idx, lbl)

    plt.ylabel("Value")
    plt.show()
    plt.clear_figure()


def plotDrawdown(
    result: Any,
    title: str = "Drawdown",
    height: int = 10,
    width: Optional[int] = None,
) -> None:
    """Plot drawdown curve with area fill and MDD annotation.

    낙폭 곡선을 영역 채우기 및 MDD 주석과 함께 플롯합니다.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data or len(data) < 2:
        print("[No equity data for drawdown]")
        return

    peak = data[0]
    dd = []
    for v in data:
        if v > peak:
            peak = v
        dd.append((v / peak - 1) * 100 if peak else 0)

    mdd = min(dd)

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(f"{title}  (MDD: {mdd:.2f}%)")
    plt.theme("dark")

    zeroLine = [0.0] * len(dd)
    plt.plot(zeroLine, color="gray")
    plt.plot(dd, color="red", label="Drawdown %", fillx=True)

    plt.hline(mdd, color="yellow")

    plt.ylabel("%")

    dates = _getDateLabels(r)
    if dates and len(dates) == len(dd):
        idx, lbl = _dateTicks(dates, nTicks=8)
        plt.xticks(idx, lbl)

    plt.show()
    plt.clear_figure()


def plotCandlestick(
    df: Any,
    title: str = "Price",
    height: int = 18,
    width: Optional[int] = None,
    lastN: int = 100,
    smaPeriods: List[int] = None,
    volume: bool = True,
) -> None:
    """Plot candlestick chart with SMA overlay and colored volume bars.

    캔들스틱 차트를 SMA 오버레이 및 색상별 거래량 바와 함께 플롯합니다.
    TradingView style: candlestick on top, volume bars below.

    Args:
        df: DataFrame with OHLCV columns. OHLCV 컬럼이 있는 DataFrame.
        title: Chart title. 차트 제목.
        height: Chart height. 차트 높이.
        width: Chart width. 차트 너비.
        lastN: Number of recent bars to show. 표시할 최근 바 수.
        smaPeriods: SMA periods to overlay (e.g., [5, 20, 60]). SMA 기간 오버레이.
        volume: Show volume subplot below. 하단 거래량 서브플롯 표시 여부.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        print("[Candlestick requires DataFrame with OHLC columns]")
        return

    colMap = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ('open', 'o'):
            colMap['open'] = col
        elif cl in ('high', 'h'):
            colMap['high'] = col
        elif cl in ('low', 'l'):
            colMap['low'] = col
        elif cl in ('close', 'c'):
            colMap['close'] = col
        elif cl in ('volume', 'v', 'vol'):
            colMap['volume'] = col

    if len(colMap) < 4:
        print(f"[Need OHLC columns, found: {list(df.columns)}]")
        return

    df = df.tail(lastN).copy().reset_index(drop=False)
    opens = df[colMap['open']].tolist()
    highs = df[colMap['high']].tolist()
    lows = df[colMap['low']].tolist()
    closes = df[colMap['close']].tolist()

    dates = None
    idxCol = df.columns[0]
    try:
        if hasattr(df[idxCol], 'dt'):
            dates = df[idxCol].dt.strftime('%Y-%m-%d').tolist()
        elif hasattr(df[idxCol].iloc[0], 'strftime'):
            dates = [d.strftime('%Y-%m-%d') for d in df[idxCol]]
    except Exception:
        pass

    hasVol = 'volume' in colMap and volume
    w = width or plt.terminal_width()

    if hasVol:
        volData = df[colMap['volume']].tolist()
        plt.clear_figure()
        plt.subplots(2, 1)

        plt.subplot(1, 1)
        plt.plotsize(w, height)
        plt.title(title)
        plt.theme("dark")
        plt.candlestick(opens, closes, highs, lows)

        if smaPeriods:
            colors = ["yellow", "magenta", "white"]
            for i, period in enumerate(smaPeriods):
                ma = _sma(closes, period)
                valid = [(j, v) for j, v in enumerate(ma) if v is not None]
                if valid:
                    plt.plot(
                        [j for j, _ in valid],
                        [v for _, v in valid],
                        label=f"SMA{period}",
                        color=colors[i % len(colors)],
                    )

        if dates:
            idx, lbl = _dateTicks(dates, nTicks=8)
            plt.xticks(idx, lbl)

        plt.subplot(2, 1)
        plt.plotsize(w, 6)
        plt.theme("dark")
        plt.title("Volume")

        upIdx, upVol = [], []
        dnIdx, dnVol = [], []
        for i in range(len(volData)):
            if closes[i] >= opens[i]:
                upIdx.append(i)
                upVol.append(volData[i])
            else:
                dnIdx.append(i)
                dnVol.append(volData[i])

        if upIdx:
            plt.bar(upIdx, upVol, color="green", width=0.8)
        if dnIdx:
            plt.bar(dnIdx, dnVol, color="red", width=0.8)

        if dates:
            idx, lbl = _dateTicks(dates, nTicks=8)
            plt.xticks(idx, lbl)

        plt.show()
        plt.clear_figure()
    else:
        plt.clear_figure()
        plt.plotsize(w, height)
        plt.title(title)
        plt.theme("dark")
        plt.candlestick(opens, closes, highs, lows)

        if smaPeriods:
            colors = ["yellow", "magenta", "white"]
            for i, period in enumerate(smaPeriods):
                ma = _sma(closes, period)
                valid = [(j, v) for j, v in enumerate(ma) if v is not None]
                if valid:
                    plt.plot(
                        [j for j, _ in valid],
                        [v for _, v in valid],
                        label=f"SMA{period}",
                        color=colors[i % len(colors)],
                    )

        if dates:
            idx, lbl = _dateTicks(dates, nTicks=8)
            plt.xticks(idx, lbl)

        plt.show()
        plt.clear_figure()


def plotReturns(
    result: Any,
    title: str = "Daily Returns",
    height: int = 10,
    width: Optional[int] = None,
) -> None:
    """Plot return distribution histogram with mean, +/- 1 std, skewness/kurtosis.

    일별 수익률 분포 히스토그램을 평균선, +/- 1 표준편차선, 왜도/첨도와 함께 플롯합니다.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data or len(data) < 3:
        print("[Not enough equity data]")
        return

    returns = [(data[i] / data[i - 1] - 1) * 100 for i in range(1, len(data)) if data[i - 1] != 0]
    if not returns:
        return

    n = len(returns)
    meanRet = sum(returns) / n
    variance = sum((x - meanRet) ** 2 for x in returns) / n
    stdRet = variance ** 0.5

    if stdRet > 0 and n >= 3:
        skew = sum(((x - meanRet) / stdRet) ** 3 for x in returns) * n / ((n - 1) * (n - 2)) if n > 2 else 0.0
    else:
        skew = 0.0

    if stdRet > 0 and n >= 4:
        m4 = sum((x - meanRet) ** 4 for x in returns) / n
        kurt = (m4 / (variance ** 2)) - 3.0
    else:
        kurt = 0.0

    enhancedTitle = f"{title}  (skew={skew:.2f}, kurt={kurt:.2f})"

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(enhancedTitle)
    plt.theme("dark")
    plt.hist(returns, bins=40, color="cyan")
    plt.vline(0, color="gray")
    plt.vline(meanRet, color="yellow")
    if stdRet > 0:
        plt.vline(meanRet + stdRet, color="magenta")
        plt.vline(meanRet - stdRet, color="magenta")
    plt.xlabel("Return %")
    plt.ylabel("Count")
    plt.show()
    plt.clear_figure()


def plotTradeMarkers(
    result: Any,
    title: str = "Equity + Trades",
    height: int = 18,
    width: Optional[int] = None,
) -> None:
    """Plot equity curve with buy/sell trade markers.

    자산 곡선에 매수/매도 마커를 표시합니다.
    Green dots = entry, Red dots = exit.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    trades = getattr(r, 'trades', [])
    if not data:
        print("[No equity data]")
        return

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    plt.plot(data, label="Equity", color="cyan")

    initial = getattr(r, 'initialCash', data[0])
    if initial:
        plt.hline(initial, color="gray")

    dates = _getDateLabels(r)
    if trades and dates:
        dateToIdx = {d: i for i, d in enumerate(dates)}
        buyX, buyY = [], []
        sellX, sellY = [], []

        for t in trades:
            if isinstance(t, dict):
                entryD = str(t.get('entryDate', ''))[:10]
                exitD = str(t.get('exitDate', ''))[:10]
            else:
                entryD = str(getattr(t, 'entryDate', ''))[:10]
                exitD = str(getattr(t, 'exitDate', ''))[:10]

            if entryD in dateToIdx:
                idx = dateToIdx[entryD]
                buyX.append(idx)
                buyY.append(data[idx])
            if exitD in dateToIdx:
                idx = dateToIdx[exitD]
                sellX.append(idx)
                sellY.append(data[idx])

        if buyX:
            plt.scatter(buyX, buyY, color="green", label="Buy", marker="dot")
        if sellX:
            plt.scatter(sellX, sellY, color="red", label="Sell", marker="dot")

    if dates:
        idx, lbl = _dateTicks(dates, nTicks=8)
        plt.xticks(idx, lbl)

    plt.ylabel("Value")
    plt.show()
    plt.clear_figure()


def plotMonthlyHeatmap(
    result: Any,
    title: str = "Monthly Returns",
    height: int = 14,
    width: Optional[int] = None,
) -> None:
    """Plot monthly returns heatmap as a grouped bar chart.

    월별 수익률 히트맵을 그룹화된 바 차트로 플롯합니다.
    X-axis: months (Jan-Dec), grouped by year.
    Color: green for positive, red for negative returns.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    monthly = _getMonthlyReturns(r)
    if not monthly:
        print("[No monthly return data]")
        return

    years = sorted(monthly.keys())
    yearColors = ["cyan", "yellow", "magenta", "white", "orange", "blue+"]

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    for yIdx, year in enumerate(years):
        monthData = monthly[year]
        positions = []
        values = []
        for month in range(1, 13):
            if month in monthData:
                positions.append(month)
                values.append(monthData[month])

        if positions and values:
            posColors = []
            for v in values:
                if v >= 0:
                    posColors.append("green")
                else:
                    posColors.append("red")

            clr = yearColors[yIdx % len(yearColors)]
            plt.bar(positions, values, label=str(year), color=clr, width=0.8 / max(len(years), 1))

    plt.hline(0, color="gray")
    plt.xticks(list(range(1, 13)), MONTH_LABELS)
    plt.xlabel("Month")
    plt.ylabel("Return %")
    plt.show()
    plt.clear_figure()


def plotRollingMetrics(
    result: Any,
    title: str = "Rolling Metrics",
    height: int = 16,
    width: Optional[int] = None,
    window: int = 252,
) -> None:
    """Plot rolling Sharpe ratio and rolling volatility in a 2-subplot layout.

    롤링 샤프 비율과 롤링 변동성을 2개 서브플롯 레이아웃으로 플롯합니다.
    Reference lines: Sharpe=1.0, Volatility=average.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Total chart height in rows. 전체 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
        window: Rolling window size in trading days. 롤링 윈도우 크기(거래일). Default 252.
    """
    r = _unwrap(result)
    data = _getEquityList(r)
    if not data or len(data) < window + 2:
        print(f"[Not enough data for rolling metrics (need > {window} points)]")
        return

    returns = []
    for i in range(1, len(data)):
        if data[i - 1] != 0:
            returns.append(data[i] / data[i - 1] - 1)
        else:
            returns.append(0.0)

    rollingSharpe = []
    rollingVol = []
    riskFreeDaily = 0.035 / 252

    for i in range(window - 1, len(returns)):
        windowReturns = returns[i - window + 1:i + 1]
        meanRet = sum(windowReturns) / window
        variance = sum((x - meanRet) ** 2 for x in windowReturns) / window
        stdRet = variance ** 0.5
        annualVol = stdRet * (252 ** 0.5) * 100
        rollingVol.append(annualVol)

        if stdRet > 0:
            sharpe = (meanRet - riskFreeDaily) / stdRet * (252 ** 0.5)
        else:
            sharpe = 0.0
        rollingSharpe.append(sharpe)

    if not rollingSharpe:
        print("[Insufficient data for rolling calculation]")
        return

    avgVol = sum(rollingVol) / len(rollingVol) if rollingVol else 0.0
    xAxis = list(range(len(rollingSharpe)))

    dates = _getDateLabels(r)
    offsetDates = []
    if dates and len(dates) == len(data):
        offsetDates = dates[window:]

    w = width or plt.terminal_width()
    topH = max(height // 2, 6)
    botH = max(height - topH, 6)

    plt.clear_figure()
    plt.subplots(2, 1)

    plt.subplot(1, 1)
    plt.plotsize(w, topH)
    plt.title(f"{title} - Rolling Sharpe ({window}d)")
    plt.theme("dark")
    plt.plot(xAxis, rollingSharpe, label="Sharpe", color="cyan")
    plt.hline(1.0, color="yellow")
    plt.hline(0.0, color="gray")
    plt.ylabel("Sharpe")
    if offsetDates:
        idx, lbl = _dateTicks(offsetDates, nTicks=6)
        plt.xticks(idx, lbl)

    plt.subplot(2, 1)
    plt.plotsize(w, botH)
    plt.title(f"Rolling Volatility ({window}d)")
    plt.theme("dark")
    plt.plot(xAxis, rollingVol, label="Volatility %", color="magenta")
    plt.hline(avgVol, color="yellow")
    plt.ylabel("Vol %")
    if offsetDates:
        idx, lbl = _dateTicks(offsetDates, nTicks=6)
        plt.xticks(idx, lbl)

    plt.show()
    plt.clear_figure()


def plotTradeScatter(
    result: Any,
    title: str = "Trade Scatter",
    height: int = 14,
    width: Optional[int] = None,
) -> None:
    """Plot trade scatter: X=holding days, Y=return %.

    거래 산점도: X=보유일수, Y=수익률(%).
    Green dots for winning trades, red for losing.
    Horizontal line at 0% return.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    td = _getTradeData(r)
    if not td["pnlPercents"]:
        print("[No trade data for scatter plot]")
        return

    winX, winY = [], []
    loseX, loseY = [], []

    for i in range(len(td["pnlPercents"])):
        hd = td["holdingDays"][i]
        pnlPct = td["pnlPercents"][i]
        if td["isWins"][i]:
            winX.append(hd)
            winY.append(pnlPct)
        else:
            loseX.append(hd)
            loseY.append(pnlPct)

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(title)
    plt.theme("dark")

    if winX:
        plt.scatter(winX, winY, color="green", label="Win", marker="dot")
    if loseX:
        plt.scatter(loseX, loseY, color="red", label="Loss", marker="dot")

    plt.hline(0, color="gray")
    plt.xlabel("Holding Days")
    plt.ylabel("Return %")
    plt.show()
    plt.clear_figure()


def plotCorrelationBars(
    names: List[str],
    matrix: Any,
    title: str = "Strategy Correlations",
    height: int = 12,
    width: Optional[int] = None,
) -> None:
    """Plot correlation pairs as horizontal bar chart.

    상관관계 쌍을 수평 바 차트로 플롯합니다.
    High correlation = red, Low correlation = green.

    Args:
        names: List of strategy names. 전략 이름 리스트.
        matrix: Correlation matrix (2D list, numpy array, or DataFrame).
            상관 행렬 (2D 리스트, numpy 배열, 또는 DataFrame).
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    try:
        import numpy as np
        if hasattr(matrix, 'values'):
            mat = matrix.values
        elif isinstance(matrix, list):
            mat = np.array(matrix)
        else:
            mat = np.array(matrix)
    except ImportError:
        mat = matrix

    pairLabels = []
    pairValues = []
    n = len(names)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                val = float(mat[i][j])
            except (IndexError, TypeError):
                continue
            pairLabels.append(f"{names[i][:8]}-{names[j][:8]}")
            pairValues.append(val)

    if not pairValues:
        print("[No correlation pairs to display]")
        return

    sortedPairs = sorted(zip(pairLabels, pairValues), key=lambda x: abs(x[1]), reverse=True)
    pairLabels = [p[0] for p in sortedPairs]
    pairValues = [p[1] for p in sortedPairs]

    barColors = []
    for v in pairValues:
        absV = abs(v)
        if absV >= 0.7:
            barColors.append("red")
        elif absV >= 0.4:
            barColors.append("yellow")
        else:
            barColors.append("green")

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), max(height, len(pairLabels) + 4))
    plt.title(title)
    plt.theme("dark")

    positions = list(range(1, len(pairValues) + 1))
    plt.bar(positions, pairValues, orientation="h", color=barColors[0] if len(set(barColors)) == 1 else "cyan")
    plt.yticks(positions, pairLabels)
    plt.xlabel("Correlation")
    plt.vline(0, color="gray")
    plt.show()
    plt.clear_figure()


def plotStrategyDna(
    dna: Any,
    title: str = "Strategy DNA",
    height: int = 16,
    width: Optional[int] = None,
) -> None:
    """Plot Strategy DNA as horizontal bar chart for 12 dimensions.

    12차원 전략 DNA를 수평 바 차트로 플롯합니다.
    Each bar colored by value level: green (<0.4), yellow (0.4-0.7), red (>0.7).

    Args:
        dna: StrategyDNA object or dict with dimension values.
            StrategyDNA 객체 또는 차원 값을 가진 딕셔너리.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    dimensionNames = [
        "trendSensitivity",
        "meanReversionAffinity",
        "volatilityPreference",
        "holdingPeriodProfile",
        "drawdownTolerance",
        "winRateProfile",
        "marketRegimeDependence",
        "concentrationLevel",
        "tradingFrequency",
        "riskRewardRatio",
        "momentumExposure",
        "defensiveScore",
    ]

    dimensionLabels = [
        "Trend Sens.",
        "MeanRev Aff.",
        "Vol. Pref.",
        "Hold Period",
        "DD Tolerance",
        "WinRate Prof.",
        "Regime Dep.",
        "Concentration",
        "Trade Freq.",
        "Risk/Reward",
        "Momentum Exp.",
        "Defensive Sc.",
    ]

    values = []
    if isinstance(dna, dict):
        for name in dimensionNames:
            values.append(float(dna.get(name, 0.0)))
    elif hasattr(dna, 'toDict'):
        dnaDict = dna.toDict()
        for name in dimensionNames:
            values.append(float(dnaDict.get(name, 0.0)))
    else:
        for name in dimensionNames:
            values.append(float(getattr(dna, name, 0.0)))

    barColors = []
    for v in values:
        if v >= 0.7:
            barColors.append("red")
        elif v >= 0.4:
            barColors.append("yellow")
        else:
            barColors.append("green")

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), max(height, 18))
    plt.title(title)
    plt.theme("dark")

    positions = list(range(1, len(values) + 1))

    greenPos, greenVal = [], []
    yellowPos, yellowVal = [], []
    redPos, redVal = [], []
    for i, v in enumerate(values):
        if barColors[i] == "green":
            greenPos.append(positions[i])
            greenVal.append(v)
        elif barColors[i] == "yellow":
            yellowPos.append(positions[i])
            yellowVal.append(v)
        else:
            redPos.append(positions[i])
            redVal.append(v)

    if greenPos:
        plt.bar(greenPos, greenVal, orientation="h", color="green", label="Low", width=0.7)
    if yellowPos:
        plt.bar(yellowPos, yellowVal, orientation="h", color="yellow", label="Mid", width=0.7)
    if redPos:
        plt.bar(redPos, redVal, orientation="h", color="red", label="High", width=0.7)

    plt.yticks(positions, dimensionLabels)
    plt.xlabel("Value (0-1)")
    plt.xlim(0, 1.05)
    plt.show()
    plt.clear_figure()


def plotSeasonality(
    result: Any,
    title: str = "Seasonality",
    height: int = 12,
    width: Optional[int] = None,
) -> None:
    """Plot average monthly returns bar chart (seasonality analysis).

    평균 월별 수익률 바 차트를 플롯합니다 (계절성 분석).
    12 bars (Jan-Dec), green for positive, red for negative.
    Annotates best and worst months.

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        title: Chart title. 차트 제목.
        height: Chart height in rows. 차트 높이(행).
        width: Chart width in columns. 차트 너비(열).
    """
    r = _unwrap(result)
    monthly = _getMonthlyReturns(r)
    if not monthly:
        print("[No monthly return data for seasonality]")
        return

    monthlyAccum = {m: [] for m in range(1, 13)}
    for year in monthly:
        for month, ret in monthly[year].items():
            monthlyAccum[month].append(ret)

    avgReturns = []
    for m in range(1, 13):
        if monthlyAccum[m]:
            avgReturns.append(sum(monthlyAccum[m]) / len(monthlyAccum[m]))
        else:
            avgReturns.append(0.0)

    bestMonth = -1
    worstMonth = -1
    bestVal = -float('inf')
    worstVal = float('inf')
    for i, v in enumerate(avgReturns):
        if v > bestVal:
            bestVal = v
            bestMonth = i
        if v < worstVal:
            worstVal = v
            worstMonth = i

    bestLabel = MONTH_LABELS[bestMonth] if bestMonth >= 0 else "?"
    worstLabel = MONTH_LABELS[worstMonth] if worstMonth >= 0 else "?"

    enhancedTitle = f"{title}  (Best: {bestLabel} {bestVal:+.2f}%, Worst: {worstLabel} {worstVal:+.2f}%)"

    posMonths, posVals = [], []
    negMonths, negVals = [], []
    for i, v in enumerate(avgReturns):
        if v >= 0:
            posMonths.append(i + 1)
            posVals.append(v)
        else:
            negMonths.append(i + 1)
            negVals.append(v)

    plt.clear_figure()
    plt.plotsize(width or plt.terminal_width(), height)
    plt.title(enhancedTitle)
    plt.theme("dark")

    if posMonths:
        plt.bar(posMonths, posVals, color="green", label="Positive", width=0.7)
    if negMonths:
        plt.bar(negMonths, negVals, color="red", label="Negative", width=0.7)

    plt.hline(0, color="gray")
    plt.xticks(list(range(1, 13)), MONTH_LABELS)
    plt.xlabel("Month")
    plt.ylabel("Avg Return %")
    plt.show()
    plt.clear_figure()


def plotDashboard(result: Any, lang: str = "en") -> None:
    """Print TradingView-inspired dashboard with all enhanced charts.

    TradingView 스타일 대시보드를 모든 개선된 차트와 함께 출력합니다.

    Layout:
        [Header] Strategy / Symbol / Period / Key Metrics
        [Chart 1] Equity Curve + Trade Markers + SMA overlays
        [Chart 2] Drawdown with MDD annotation
        [Chart 3] Return Distribution with skewness/kurtosis
        [Chart 4] Seasonality

    Args:
        result: Backtest result object. 백테스트 결과 객체.
        lang: Language ("en" or "ko"). 언어.
    """
    from tradex.tui.console import printResult

    ko = lang == "ko"
    r = _unwrap(result)
    data = _getEquityList(r)

    printResult(result, lang=lang)

    if not data:
        return

    plotEquityCurve(
        result,
        title="Equity Curve" if not ko else "자산곡선",
        height=16,
        smaPeriods=[20, 60],
    )

    plotDrawdown(
        result,
        title="Drawdown" if not ko else "낙폭",
        height=10,
    )

    plotReturns(
        result,
        title="Daily Returns" if not ko else "일간 수익률 분포",
        height=10,
    )

    plotSeasonality(
        result,
        title="Seasonality" if not ko else "계절성",
        height=10,
    )
