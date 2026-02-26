# -*- coding: utf-8 -*-
"""
Tradex Seasonality Analyzer Module.

Discovers and quantifies seasonal patterns in strategy performance, including
monthly, weekday, and quarterly return patterns, with statistical significance
testing via one-way ANOVA.

전략 성과의 계절성 패턴을 발견하고 정량화하는 모듈입니다.
월별, 요일별, 분기별 수익률 패턴을 분석하며, 일원분산분석(ANOVA)을 통해
통계적 유의성을 검정합니다.

Features:
    - Monthly, weekday, and quarterly return pattern detection
    - Year x Month heatmap generation
    - January Effect and "Sell in May" seasonal anomaly testing
    - Korean holiday effect analysis (Seollal, Chuseok, Christmas)
    - ANOVA-based statistical significance testing
    - Best/worst period identification

Usage:
    from tradex.analytics.seasonality import SeasonalityAnalyzer

    analyzer = SeasonalityAnalyzer(result)
    monthly = analyzer.monthlyPattern()
    print(monthly.summary())

    heatmap = analyzer.monthlyHeatmap()
    full = analyzer.fullAnalysis()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from tradex.engine import BacktestResult
from tradex.entities.trade import Trade

MONTH_NAMES = {
    1: '1월', 2: '2월', 3: '3월', 4: '4월',
    5: '5월', 6: '6월', 7: '7월', 8: '8월',
    9: '9월', 10: '10월', 11: '11월', 12: '12월',
}

WEEKDAY_NAMES = {
    0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일',
}

WEEKDAY_NAMES_EN = {
    0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri',
}

QUARTER_NAMES = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}

DEFAULT_KOREAN_HOLIDAYS = {
    '설날': [(1, 21), (1, 22), (1, 29), (1, 30), (2, 1), (2, 10), (2, 11), (2, 12)],
    '추석': [(9, 10), (9, 11), (9, 12), (9, 17), (9, 18), (9, 19), (9, 28), (9, 29), (9, 30)],
    '크리스마스': [(12, 25)],
}


@dataclass
class SeasonalPattern:
    """
    Container for a single seasonal pattern analysis result.

    단일 계절성 패턴 분석 결과를 담는 데이터 클래스입니다.

    Attributes:
        period (str): Pattern period type - "monthly", "weekday", or "quarterly".
            패턴 기간 유형 ("monthly", "weekday", "quarterly").
        data (dict): Mapping of period labels to average returns in percent.
            기간 레이블 -> 평균 수익률(%) 매핑.
        bestPeriod (str): Label of the best performing period.
            최고 수익률 기간 레이블.
        worstPeriod (str): Label of the worst performing period.
            최저 수익률 기간 레이블.
        statisticalSignificance (float): p-value from one-way ANOVA test.
            일원분산분석(ANOVA) p-value. Lower values indicate stronger evidence
            that returns differ significantly across periods (유의 수준 < 0.05이면
            기간별 수익률 차이가 통계적으로 유의).

    Example:
        >>> pattern = analyzer.monthlyPattern()
        >>> print(f"Best month: {pattern.bestPeriod}")
        >>> print(f"p-value: {pattern.statisticalSignificance:.4f}")
    """
    period: str
    data: Dict[str, float] = field(default_factory=dict)
    bestPeriod: str = ""
    worstPeriod: str = ""
    statisticalSignificance: float = 1.0

    def summary(self) -> str:
        """
        Generate a human-readable summary of the seasonal pattern.

        계절성 패턴의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line formatted summary with period returns and significance.
        """
        periodLabel = {
            'monthly': '월별 패턴',
            'weekday': '요일별 패턴',
            'quarterly': '분기별 패턴',
        }.get(self.period, self.period)

        significanceStr = (
            "유의" if self.statisticalSignificance < 0.05 else "비유의"
        )

        lines = [
            f"{'='*50}",
            f"{periodLabel} 분석 (Seasonal Pattern Analysis)",
            f"{'='*50}",
        ]

        for label, returnVal in sorted(self.data.items()):
            bar = "+" * max(0, int(returnVal * 10)) if returnVal > 0 else "-" * max(0, int(abs(returnVal) * 10))
            lines.append(f"  {label:>8}: {returnVal:+.4f}%  {bar}")

        lines.extend([
            f"{'─'*50}",
            f"최고: {self.bestPeriod} ({self.data.get(self.bestPeriod, 0):+.4f}%)",
            f"최저: {self.worstPeriod} ({self.data.get(self.worstPeriod, 0):+.4f}%)",
            f"통계적 유의성: p={self.statisticalSignificance:.4f} ({significanceStr})",
            f"{'='*50}",
        ])

        return "\n".join(lines)


class SeasonalityAnalyzer:
    """
    Seasonality pattern analyzer for backtest equity curves.

    Discovers and quantifies seasonal return patterns by decomposing the
    equity curve's daily returns into monthly, weekday, and quarterly groups.
    Uses one-way ANOVA for statistical significance testing.

    백테스트 자산 곡선에 대한 계절성 패턴 분석기입니다.
    일별 수익률을 월별, 요일별, 분기별 그룹으로 분해하여 계절적 수익률 패턴을
    발견하고 정량화합니다. 통계적 유의성 검정에 일원분산분석(ANOVA)을 사용합니다.

    Attributes:
        result (BacktestResult): Original backtest result to analyze.
            분석 대상 원본 백테스트 결과.

    Example:
        >>> analyzer = SeasonalityAnalyzer(result)
        >>> monthly = analyzer.monthlyPattern()
        >>> print(monthly.summary())
        >>>
        >>> heatmap = analyzer.monthlyHeatmap()
        >>> full = analyzer.fullAnalysis()
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize the Seasonality Analyzer with a backtest result.

        백테스트 결과로 계절성 분석기를 초기화합니다.

        Args:
            result (BacktestResult): Completed backtest result containing an
                equity curve with DatetimeIndex. 자산 곡선이 포함된 백테스트 결과.
        """
        self._result = result
        self._equityCurve = result.equityCurve.copy()
        self._dailyReturns = self._computeDailyReturns()

    def _computeDailyReturns(self) -> pd.Series:
        """
        Compute daily percentage returns from the equity curve.

        자산 곡선으로부터 일별 수익률(%)을 계산합니다.

        Returns:
            pd.Series: Daily returns in percent with DatetimeIndex.
        """
        if len(self._equityCurve) < 2:
            return pd.Series(dtype=float)

        returns = self._equityCurve.pct_change().dropna() * 100
        return returns

    def _performAnova(self, groups: List[np.ndarray]) -> float:
        """
        Perform one-way ANOVA test on grouped return data.

        그룹화된 수익률 데이터에 일원분산분석(ANOVA) 검정을 수행합니다.

        Args:
            groups (List[np.ndarray]): List of return arrays for each group.
                각 그룹의 수익률 배열 목록.

        Returns:
            float: p-value from the F-test. Lower values indicate stronger
                evidence of significant differences between groups.
        """
        validGroups = [g for g in groups if len(g) >= 2]

        if len(validGroups) < 2:
            return 1.0

        try:
            fStat, pValue = stats.f_oneway(*validGroups)
            if np.isnan(pValue):
                return 1.0
            return float(pValue)
        except Exception:
            return 1.0

    def monthlyPattern(self) -> SeasonalPattern:
        """
        Analyze average daily return by calendar month (1-12).

        월별(1~12월) 평균 일별 수익률을 분석합니다.

        Groups daily returns by calendar month and computes the mean return
        for each month. Tests statistical significance using one-way ANOVA.

        Returns:
            SeasonalPattern: Monthly pattern with average returns per month,
                best/worst months, and ANOVA p-value.

        Example:
            >>> pattern = analyzer.monthlyPattern()
            >>> print(f"Best month: {pattern.bestPeriod}")
        """
        if len(self._dailyReturns) == 0:
            return SeasonalPattern(period='monthly')

        months = self._dailyReturns.index.month
        monthlyData = {}
        monthlyGroups = []

        for m in range(1, 13):
            mask = months == m
            monthReturns = self._dailyReturns[mask].values
            if len(monthReturns) > 0:
                monthlyData[MONTH_NAMES[m]] = float(np.mean(monthReturns))
                monthlyGroups.append(monthReturns)
            else:
                monthlyData[MONTH_NAMES[m]] = 0.0

        pValue = self._performAnova(monthlyGroups)

        bestMonth = max(monthlyData, key=monthlyData.get)
        worstMonth = min(monthlyData, key=monthlyData.get)

        return SeasonalPattern(
            period='monthly',
            data=monthlyData,
            bestPeriod=bestMonth,
            worstPeriod=worstMonth,
            statisticalSignificance=pValue,
        )

    def weekdayPattern(self) -> SeasonalPattern:
        """
        Analyze average daily return by weekday (Monday-Friday).

        요일별(월~금) 평균 일별 수익률을 분석합니다.

        Groups daily returns by weekday and computes the mean return for each
        trading day. Tests statistical significance using one-way ANOVA.

        Returns:
            SeasonalPattern: Weekday pattern with average returns per day,
                best/worst days, and ANOVA p-value.

        Example:
            >>> pattern = analyzer.weekdayPattern()
            >>> print(f"Best day: {pattern.bestPeriod}")
        """
        if len(self._dailyReturns) == 0:
            return SeasonalPattern(period='weekday')

        weekdays = self._dailyReturns.index.dayofweek
        weekdayData = {}
        weekdayGroups = []

        for d in range(5):
            mask = weekdays == d
            dayReturns = self._dailyReturns[mask].values
            label = f"{WEEKDAY_NAMES[d]}({WEEKDAY_NAMES_EN[d]})"
            if len(dayReturns) > 0:
                weekdayData[label] = float(np.mean(dayReturns))
                weekdayGroups.append(dayReturns)
            else:
                weekdayData[label] = 0.0

        pValue = self._performAnova(weekdayGroups)

        bestDay = max(weekdayData, key=weekdayData.get)
        worstDay = min(weekdayData, key=weekdayData.get)

        return SeasonalPattern(
            period='weekday',
            data=weekdayData,
            bestPeriod=bestDay,
            worstPeriod=worstDay,
            statisticalSignificance=pValue,
        )

    def quarterlyPattern(self) -> SeasonalPattern:
        """
        Analyze average daily return by quarter (Q1-Q4).

        분기별(Q1~Q4) 평균 일별 수익률을 분석합니다.

        Groups daily returns by fiscal quarter and computes the mean return
        for each quarter. Tests statistical significance using one-way ANOVA.

        Returns:
            SeasonalPattern: Quarterly pattern with average returns per quarter,
                best/worst quarters, and ANOVA p-value.

        Example:
            >>> pattern = analyzer.quarterlyPattern()
            >>> print(f"Best quarter: {pattern.bestPeriod}")
        """
        if len(self._dailyReturns) == 0:
            return SeasonalPattern(period='quarterly')

        quarters = self._dailyReturns.index.quarter
        quarterlyData = {}
        quarterlyGroups = []

        for q in range(1, 5):
            mask = quarters == q
            qReturns = self._dailyReturns[mask].values
            label = QUARTER_NAMES[q]
            if len(qReturns) > 0:
                quarterlyData[label] = float(np.mean(qReturns))
                quarterlyGroups.append(qReturns)
            else:
                quarterlyData[label] = 0.0

        pValue = self._performAnova(quarterlyGroups)

        bestQ = max(quarterlyData, key=quarterlyData.get)
        worstQ = min(quarterlyData, key=quarterlyData.get)

        return SeasonalPattern(
            period='quarterly',
            data=quarterlyData,
            bestPeriod=bestQ,
            worstPeriod=worstQ,
            statisticalSignificance=pValue,
        )

    def monthlyHeatmap(self) -> pd.DataFrame:
        """
        Generate a Year x Month heatmap of monthly returns.

        연도 x 월 월간 수익률 히트맵을 생성합니다.

        Computes monthly returns from the equity curve and arranges them
        in a matrix with years as rows and months (1-12) as columns.

        Returns:
            pd.DataFrame: Heatmap with years as index, month names as columns,
                and monthly returns (%) as values. Missing months are NaN.

        Example:
            >>> heatmap = analyzer.monthlyHeatmap()
            >>> print(heatmap.to_string())
        """
        if len(self._equityCurve) < 2:
            return pd.DataFrame()

        monthly = self._equityCurve.resample('ME').last()
        monthlyReturns = monthly.pct_change().dropna() * 100

        if len(monthlyReturns) == 0:
            return pd.DataFrame()

        years = monthlyReturns.index.year
        months = monthlyReturns.index.month

        heatmapData = {}
        for year in sorted(years.unique()):
            yearMask = years == year
            yearReturns = monthlyReturns[yearMask]
            yearMonths = months[yearMask]

            rowData = {}
            for m, r in zip(yearMonths, yearReturns.values):
                rowData[MONTH_NAMES[m]] = float(r)
            heatmapData[year] = rowData

        columnOrder = [MONTH_NAMES[m] for m in range(1, 13)]
        heatmap = pd.DataFrame.from_dict(heatmapData, orient='index')
        heatmap = heatmap.reindex(columns=columnOrder)
        heatmap.index.name = '연도'

        return heatmap

    def bestWorstMonths(self, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find the top and bottom N individual months by return.

        수익률 기준 상위/하위 N개 개별 월을 찾습니다.

        Args:
            n (int): Number of best and worst months to return (default: 5).
                반환할 최고/최저 월 수 (기본값: 5).

        Returns:
            dict: Contains 'best' and 'worst' keys, each mapping to a list of
                dicts with 'date' (str), 'return' (float) keys.

        Example:
            >>> bw = analyzer.bestWorstMonths(3)
            >>> for m in bw['best']:
            ...     print(f"{m['date']}: {m['return']:+.2f}%")
        """
        if len(self._equityCurve) < 2:
            return {'best': [], 'worst': []}

        monthly = self._equityCurve.resample('ME').last()
        monthlyReturns = monthly.pct_change().dropna() * 100

        if len(monthlyReturns) == 0:
            return {'best': [], 'worst': []}

        sortedReturns = monthlyReturns.sort_values(ascending=False)

        bestMonths = []
        for date, ret in sortedReturns.head(n).items():
            bestMonths.append({
                'date': date.strftime('%Y-%m'),
                'return': float(ret),
            })

        worstMonths = []
        for date, ret in sortedReturns.tail(n).items():
            worstMonths.append({
                'date': date.strftime('%Y-%m'),
                'return': float(ret),
            })

        return {'best': bestMonths, 'worst': worstMonths}

    def januaryEffect(self) -> Dict[str, Any]:
        """
        Analyze the January Effect anomaly.

        1월 효과(January Effect) 이상현상을 분석합니다.

        Compares average January returns against all other months to test
        whether January exhibits statistically higher returns, as predicted
        by the January Effect hypothesis.

        Returns:
            dict: Analysis results containing:
                - januaryAvgReturn (float): Average daily return in January (%)
                - otherMonthsAvgReturn (float): Average daily return in other months (%)
                - januaryTotalMonths (int): Number of January observations
                - difference (float): January minus other months return spread
                - pValue (float): t-test p-value for the difference
                - isSignificant (bool): Whether the effect is statistically significant

        Example:
            >>> effect = analyzer.januaryEffect()
            >>> if effect['isSignificant']:
            ...     print("January Effect detected!")
        """
        if len(self._dailyReturns) == 0:
            return {
                'januaryAvgReturn': 0.0,
                'otherMonthsAvgReturn': 0.0,
                'januaryTotalMonths': 0,
                'difference': 0.0,
                'pValue': 1.0,
                'isSignificant': False,
            }

        months = self._dailyReturns.index.month
        janReturns = self._dailyReturns[months == 1].values
        otherReturns = self._dailyReturns[months != 1].values

        janAvg = float(np.mean(janReturns)) if len(janReturns) > 0 else 0.0
        otherAvg = float(np.mean(otherReturns)) if len(otherReturns) > 0 else 0.0

        if len(janReturns) >= 2 and len(otherReturns) >= 2:
            try:
                tStat, pValue = stats.ttest_ind(janReturns, otherReturns, equal_var=False)
                pValue = float(pValue) if not np.isnan(pValue) else 1.0
            except Exception:
                pValue = 1.0
        else:
            pValue = 1.0

        janYears = self._dailyReturns[months == 1].index.year.nunique()

        return {
            'januaryAvgReturn': janAvg,
            'otherMonthsAvgReturn': otherAvg,
            'januaryTotalMonths': janYears,
            'difference': janAvg - otherAvg,
            'pValue': pValue,
            'isSignificant': pValue < 0.05,
        }

    def sellInMay(self) -> Dict[str, Any]:
        """
        Analyze the "Sell in May and Go Away" seasonal anomaly.

        "5월에 팔고 떠나라(Sell in May)" 계절적 이상현상을 분석합니다.

        Compares average daily returns during the "summer" period (May-October)
        versus the "winter" period (November-April) to test whether the market
        underperforms in summer months.

        Returns:
            dict: Analysis results containing:
                - summerAvgReturn (float): Avg daily return May-Oct (%)
                - winterAvgReturn (float): Avg daily return Nov-Apr (%)
                - summerTotalDays (int): Trading days in summer periods
                - winterTotalDays (int): Trading days in winter periods
                - difference (float): Winter minus summer return spread
                - pValue (float): t-test p-value for the difference
                - isSignificant (bool): Whether the effect is statistically significant
                - recommendation (str): Korean language recommendation

        Example:
            >>> sim = analyzer.sellInMay()
            >>> print(sim['recommendation'])
        """
        if len(self._dailyReturns) == 0:
            return {
                'summerAvgReturn': 0.0,
                'winterAvgReturn': 0.0,
                'summerTotalDays': 0,
                'winterTotalDays': 0,
                'difference': 0.0,
                'pValue': 1.0,
                'isSignificant': False,
                'recommendation': '데이터 부족',
            }

        months = self._dailyReturns.index.month
        summerMask = (months >= 5) & (months <= 10)
        winterMask = ~summerMask

        summerReturns = self._dailyReturns[summerMask].values
        winterReturns = self._dailyReturns[winterMask].values

        summerAvg = float(np.mean(summerReturns)) if len(summerReturns) > 0 else 0.0
        winterAvg = float(np.mean(winterReturns)) if len(winterReturns) > 0 else 0.0

        if len(summerReturns) >= 2 and len(winterReturns) >= 2:
            try:
                tStat, pValue = stats.ttest_ind(winterReturns, summerReturns, equal_var=False)
                pValue = float(pValue) if not np.isnan(pValue) else 1.0
            except Exception:
                pValue = 1.0
        else:
            pValue = 1.0

        difference = winterAvg - summerAvg
        isSignificant = pValue < 0.05

        if isSignificant and difference > 0:
            recommendation = "겨울(11-4월) 수익률이 유의하게 높음: Sell in May 전략 고려"
        elif difference > 0:
            recommendation = "겨울 수익률이 높으나 통계적으로 유의하지 않음"
        else:
            recommendation = "여름 수익률이 오히려 높음: Sell in May 전략 부적합"

        return {
            'summerAvgReturn': summerAvg,
            'winterAvgReturn': winterAvg,
            'summerTotalDays': len(summerReturns),
            'winterTotalDays': len(winterReturns),
            'difference': difference,
            'pValue': pValue,
            'isSignificant': isSignificant,
            'recommendation': recommendation,
        }

    def holidayEffect(self, holidays: Optional[Dict[str, List[tuple]]] = None) -> Dict[str, Any]:
        """
        Analyze pre-holiday and post-holiday return effects.

        공휴일 전후 수익률 효과를 분석합니다.

        Examines whether returns in the days immediately before and after
        major holidays differ significantly from normal trading days.
        Defaults to Korean holidays: Seollal, Chuseok, Christmas.

        Args:
            holidays (dict, optional): Mapping of holiday names to lists of
                (month, day) tuples representing approximate holiday dates.
                Defaults to Korean holiday calendar.
                공휴일 이름 -> (월, 일) 튜플 리스트 매핑.
                기본값: 한국 공휴일 (설날, 추석, 크리스마스).

        Returns:
            dict: Per-holiday analysis containing pre/post holiday returns,
                normal day returns, and statistical test results.

        Example:
            >>> effect = analyzer.holidayEffect()
            >>> for holiday, data in effect.items():
            ...     print(f"{holiday}: pre={data['preHolidayReturn']:+.4f}%")
        """
        if holidays is None:
            holidays = DEFAULT_KOREAN_HOLIDAYS

        if len(self._dailyReturns) == 0:
            return {name: {
                'preHolidayReturn': 0.0,
                'postHolidayReturn': 0.0,
                'normalReturn': 0.0,
                'preHolidayCount': 0,
                'postHolidayCount': 0,
                'pValue': 1.0,
            } for name in holidays}

        results = {}
        returnsDf = self._dailyReturns.to_frame(name='return')
        returnsDf['month'] = returnsDf.index.month
        returnsDf['day'] = returnsDf.index.day

        allPreIndices = set()
        allPostIndices = set()

        for holidayName, dates in holidays.items():
            preHolidayReturns = []
            postHolidayReturns = []

            for monthVal, dayVal in dates:
                for year in returnsDf.index.year.unique():
                    holidayDate = pd.Timestamp(year=year, month=monthVal, day=dayVal)

                    tradingDates = returnsDf.index
                    preDates = tradingDates[tradingDates < holidayDate]
                    postDates = tradingDates[tradingDates > holidayDate]

                    if len(preDates) > 0:
                        preDate = preDates[-1]
                        preDatePos = tradingDates.get_loc(preDate)
                        if isinstance(preDatePos, int):
                            preHolidayReturns.append(float(returnsDf.iloc[preDatePos]['return']))
                            allPreIndices.add(preDatePos)

                    if len(postDates) > 0:
                        postDate = postDates[0]
                        postDatePos = tradingDates.get_loc(postDate)
                        if isinstance(postDatePos, int):
                            postHolidayReturns.append(float(returnsDf.iloc[postDatePos]['return']))
                            allPostIndices.add(postDatePos)

            preArr = np.array(preHolidayReturns) if preHolidayReturns else np.array([])
            postArr = np.array(postHolidayReturns) if postHolidayReturns else np.array([])

            normalMask = np.ones(len(returnsDf), dtype=bool)
            for idx in allPreIndices | allPostIndices:
                if idx < len(normalMask):
                    normalMask[idx] = False
            normalReturns = returnsDf.iloc[normalMask]['return'].values

            preAvg = float(np.mean(preArr)) if len(preArr) > 0 else 0.0
            postAvg = float(np.mean(postArr)) if len(postArr) > 0 else 0.0
            normalAvg = float(np.mean(normalReturns)) if len(normalReturns) > 0 else 0.0

            holidayReturnsAll = np.concatenate([preArr, postArr]) if (len(preArr) + len(postArr)) > 0 else np.array([])

            if len(holidayReturnsAll) >= 2 and len(normalReturns) >= 2:
                try:
                    tStat, pValue = stats.ttest_ind(holidayReturnsAll, normalReturns, equal_var=False)
                    pValue = float(pValue) if not np.isnan(pValue) else 1.0
                except Exception:
                    pValue = 1.0
            else:
                pValue = 1.0

            results[holidayName] = {
                'preHolidayReturn': preAvg,
                'postHolidayReturn': postAvg,
                'normalReturn': normalAvg,
                'preHolidayCount': len(preArr),
                'postHolidayCount': len(postArr),
                'pValue': pValue,
            }

        return results

    def fullAnalysis(self) -> Dict[str, Any]:
        """
        Perform complete seasonal analysis combining all pattern types.

        모든 패턴 유형을 결합한 종합 계절성 분석을 수행합니다.

        Returns:
            dict: Comprehensive analysis containing:
                - monthly (SeasonalPattern): Monthly return pattern
                - weekday (SeasonalPattern): Weekday return pattern
                - quarterly (SeasonalPattern): Quarterly return pattern
                - heatmap (pd.DataFrame): Year x Month heatmap
                - bestWorstMonths (dict): Top/bottom months
                - januaryEffect (dict): January Effect analysis
                - sellInMay (dict): Sell in May analysis
                - holidayEffect (dict): Holiday effect analysis

        Example:
            >>> full = analyzer.fullAnalysis()
            >>> print(full['monthly'].summary())
            >>> print(full['sellInMay']['recommendation'])
        """
        return {
            'monthly': self.monthlyPattern(),
            'weekday': self.weekdayPattern(),
            'quarterly': self.quarterlyPattern(),
            'heatmap': self.monthlyHeatmap(),
            'bestWorstMonths': self.bestWorstMonths(),
            'januaryEffect': self.januaryEffect(),
            'sellInMay': self.sellInMay(),
            'holidayEffect': self.holidayEffect(),
        }

    def summary(self) -> str:
        """
        Generate a comprehensive summary of all seasonal analyses.

        모든 계절성 분석의 종합 요약을 생성합니다.

        Returns:
            str: Multi-line formatted summary covering monthly, weekday,
                quarterly patterns, seasonal anomalies, and holiday effects.
        """
        monthly = self.monthlyPattern()
        weekday = self.weekdayPattern()
        quarterly = self.quarterlyPattern()
        janEffect = self.januaryEffect()
        simEffect = self.sellInMay()

        lines = [
            f"{'='*60}",
            f"계절성 분석 종합 요약 (Seasonality Analysis Summary)",
            f"{'='*60}",
            f"",
            f"[월별 패턴]",
            f"  최고 월: {monthly.bestPeriod} ({monthly.data.get(monthly.bestPeriod, 0):+.4f}%)",
            f"  최저 월: {monthly.worstPeriod} ({monthly.data.get(monthly.worstPeriod, 0):+.4f}%)",
            f"  ANOVA p-value: {monthly.statisticalSignificance:.4f}",
            f"",
            f"[요일별 패턴]",
            f"  최고 요일: {weekday.bestPeriod} ({weekday.data.get(weekday.bestPeriod, 0):+.4f}%)",
            f"  최저 요일: {weekday.worstPeriod} ({weekday.data.get(weekday.worstPeriod, 0):+.4f}%)",
            f"  ANOVA p-value: {weekday.statisticalSignificance:.4f}",
            f"",
            f"[분기별 패턴]",
            f"  최고 분기: {quarterly.bestPeriod} ({quarterly.data.get(quarterly.bestPeriod, 0):+.4f}%)",
            f"  최저 분기: {quarterly.worstPeriod} ({quarterly.data.get(quarterly.worstPeriod, 0):+.4f}%)",
            f"  ANOVA p-value: {quarterly.statisticalSignificance:.4f}",
            f"",
            f"{'─'*60}",
            f"[1월 효과 (January Effect)]",
            f"  1월 평균: {janEffect['januaryAvgReturn']:+.4f}%",
            f"  타 월 평균: {janEffect['otherMonthsAvgReturn']:+.4f}%",
            f"  유의성: {'유의 (p={:.4f})'.format(janEffect['pValue']) if janEffect['isSignificant'] else '비유의'}",
            f"",
            f"[Sell in May]",
            f"  여름(5-10월): {simEffect['summerAvgReturn']:+.4f}%",
            f"  겨울(11-4월): {simEffect['winterAvgReturn']:+.4f}%",
            f"  {simEffect['recommendation']}",
            f"{'='*60}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SeasonalityAnalyzer(strategy={self._result.strategy}, "
            f"days={len(self._dailyReturns)})"
        )
