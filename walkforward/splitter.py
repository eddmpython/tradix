"""
Tradex Period Splitter Module.

Provides date-range splitting utilities for walk-forward analysis, supporting
rolling window, anchored (expanding) window, K-fold, and custom period
definitions. Includes ASCII visualization of fold structures.

워크포워드 분석을 위한 기간 분할 유틸리티 모듈입니다. 롤링 윈도우, 앵커드(확장)
윈도우, K-Fold, 커스텀 기간 정의를 지원하며 폴드 구조의 ASCII 시각화를 포함합니다.

Features:
    - Rolling window: Fixed-size IS window sliding forward
    - Anchored window: Fixed start with expanding IS window
    - K-fold: Sequential equal-partition splitting
    - Custom: User-defined period pairs
    - ASCII fold structure visualization

Usage:
    from tradex.walkforward.splitter import PeriodSplitter

    folds = PeriodSplitter.rolling(
        '2020-01-01', '2024-12-31',
        inSampleMonths=12, outOfSampleMonths=3, stepMonths=3,
    )
    print(PeriodSplitter.visualize(folds))
"""

from typing import List, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


class PeriodSplitter:
    """
    Static utility class for splitting date ranges into IS/OOS period pairs.

    Provides multiple splitting strategies for walk-forward analysis:
    rolling window, anchored (expanding) window, K-fold, and custom.
    All methods are static and return lists of ((IS_start, IS_end),
    (OOS_start, OOS_end)) tuples.

    워크포워드 분석을 위한 IS/OOS 기간 쌍 분할 정적 유틸리티 클래스입니다.
    롤링 윈도우, 앵커드(확장) 윈도우, K-Fold, 커스텀 분할 전략을 제공합니다.

    Example:
        >>> folds = PeriodSplitter.rolling(
        ...     '2020-01-01', '2024-12-31',
        ...     inSampleMonths=12,
        ...     outOfSampleMonths=3,
        ...     stepMonths=3,
        ... )
        >>> for (isStart, isEnd), (oosStart, oosEnd) in folds:
        ...     print(f"IS: {isStart} ~ {isEnd}, OOS: {oosStart} ~ {oosEnd}")
    """

    @staticmethod
    def rolling(
        startDate: str,
        endDate: str,
        inSampleMonths: int = 12,
        outOfSampleMonths: int = 3,
        stepMonths: int = 3,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Split dates using a rolling (fixed-size) window approach.

        The IS window has a fixed length and slides forward by stepMonths.

        고정 크기 IS 윈도우를 stepMonths씩 이동시키는 롤링 윈도우 분할입니다.

        Args:
            startDate: Overall start date in 'YYYY-MM-DD' format
                (전체 시작일).
            endDate: Overall end date in 'YYYY-MM-DD' format (전체 종료일).
            inSampleMonths: Length of in-sample period in months (IS 기간, 개월).
            outOfSampleMonths: Length of out-of-sample period in months
                (OOS 기간, 개월).
            stepMonths: Number of months to advance the window each step
                (윈도우 이동 스텝, 개월).

        Returns:
            List[Tuple[Tuple[str, str], Tuple[str, str]]]: List of
                ((IS_start, IS_end), (OOS_start, OOS_end)) date string pairs.
        """
        start = datetime.strptime(startDate, '%Y-%m-%d')
        end = datetime.strptime(endDate, '%Y-%m-%d')

        folds = []
        currentStart = start

        while True:
            isEnd = currentStart + relativedelta(months=inSampleMonths) - relativedelta(days=1)
            oosStart = isEnd + relativedelta(days=1)
            oosEnd = oosStart + relativedelta(months=outOfSampleMonths) - relativedelta(days=1)

            if oosEnd > end:
                oosEnd = end

            if oosStart >= end:
                break

            if isEnd < oosStart:
                folds.append((
                    (currentStart.strftime('%Y-%m-%d'), isEnd.strftime('%Y-%m-%d')),
                    (oosStart.strftime('%Y-%m-%d'), oosEnd.strftime('%Y-%m-%d'))
                ))

            if oosEnd >= end:
                break

            currentStart = currentStart + relativedelta(months=stepMonths)

        return folds

    @staticmethod
    def anchored(
        startDate: str,
        endDate: str,
        outOfSampleMonths: int = 3,
        minInSampleMonths: int = 12,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Split dates using an anchored (expanding window) approach.

        The IS start date is fixed while the IS end expands forward.

        IS 시작점이 고정되고 IS 종료점이 점진적으로 확장되는 앵커드 분할입니다.

        Args:
            startDate: Overall start date in 'YYYY-MM-DD' format
                (전체 시작일).
            endDate: Overall end date in 'YYYY-MM-DD' format (전체 종료일).
            outOfSampleMonths: Length of OOS period in months (OOS 기간, 개월).
            minInSampleMonths: Minimum initial IS period length in months
                (최소 IS 기간, 개월).

        Returns:
            List[Tuple[Tuple[str, str], Tuple[str, str]]]: List of
                ((IS_start, IS_end), (OOS_start, OOS_end)) date string pairs.
        """
        start = datetime.strptime(startDate, '%Y-%m-%d')
        end = datetime.strptime(endDate, '%Y-%m-%d')

        folds = []

        isStart = start
        isEnd = start + relativedelta(months=minInSampleMonths) - relativedelta(days=1)

        while True:
            oosStart = isEnd + relativedelta(days=1)
            oosEnd = oosStart + relativedelta(months=outOfSampleMonths) - relativedelta(days=1)

            if oosEnd > end:
                oosEnd = end

            if oosStart >= end:
                break

            folds.append((
                (isStart.strftime('%Y-%m-%d'), isEnd.strftime('%Y-%m-%d')),
                (oosStart.strftime('%Y-%m-%d'), oosEnd.strftime('%Y-%m-%d'))
            ))

            if oosEnd >= end:
                break

            isEnd = oosEnd

        return folds

    @staticmethod
    def kfold(
        startDate: str,
        endDate: str,
        nFolds: int = 5,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Split dates using a K-fold sequential approach.

        Divides the total period into K equal segments and creates folds
        where each segment (except the first) serves as OOS with all
        preceding data as IS.

        전체 기간을 K등분하여 각 구간을 순차적으로 OOS로 사용하는 K-Fold 분할입니다.

        Args:
            startDate: Overall start date in 'YYYY-MM-DD' format
                (전체 시작일).
            endDate: Overall end date in 'YYYY-MM-DD' format (전체 종료일).
            nFolds: Number of folds to create (폴드 수).

        Returns:
            List[Tuple[Tuple[str, str], Tuple[str, str]]]: List of
                ((IS_start, IS_end), (OOS_start, OOS_end)) date string pairs.
        """
        start = datetime.strptime(startDate, '%Y-%m-%d')
        end = datetime.strptime(endDate, '%Y-%m-%d')

        totalDays = (end - start).days
        foldDays = totalDays // nFolds

        folds = []

        for i in range(nFolds - 1):
            oosStart = start + relativedelta(days=foldDays * (i + 1))
            oosEnd = start + relativedelta(days=foldDays * (i + 2) - 1)

            if oosEnd > end:
                oosEnd = end

            isEnd = oosStart - relativedelta(days=1)

            folds.append((
                (start.strftime('%Y-%m-%d'), isEnd.strftime('%Y-%m-%d')),
                (oosStart.strftime('%Y-%m-%d'), oosEnd.strftime('%Y-%m-%d'))
            ))

        return folds

    @staticmethod
    def custom(
        periods: List[Tuple[Tuple[str, str], Tuple[str, str]]]
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Accept user-defined custom period splits (pass-through).

        사용자 정의 기간 분할을 그대로 반환합니다 (패스스루).

        Args:
            periods: User-defined list of ((IS_start, IS_end),
                (OOS_start, OOS_end)) tuples (사용자 정의 기간 리스트).

        Returns:
            List[Tuple[Tuple[str, str], Tuple[str, str]]]: Same as input.
        """
        return periods

    @staticmethod
    def visualize(
        folds: List[Tuple[Tuple[str, str], Tuple[str, str]]],
        width: int = 60
    ) -> str:
        """
        Generate an ASCII visualization of the fold structure.

        폴드 구조의 ASCII 시각화 문자열을 생성합니다.

        Args:
            folds: List of ((IS_start, IS_end), (OOS_start, OOS_end)) tuples
                (폴드 리스트).
            width: Character width of the visualization output (출력 폭).

        Returns:
            str: Multi-line ASCII art showing IS (===) and OOS (###)
                periods for each fold.
        """
        if not folds:
            return "No folds"

        allDates = []
        for (isStart, isEnd), (oosStart, oosEnd) in folds:
            allDates.extend([isStart, isEnd, oosStart, oosEnd])

        minDate = datetime.strptime(min(allDates), '%Y-%m-%d')
        maxDate = datetime.strptime(max(allDates), '%Y-%m-%d')
        totalDays = (maxDate - minDate).days or 1

        lines = []
        lines.append(f"Walk-Forward Folds ({len(folds)} folds)")
        lines.append("=" * width)
        lines.append(f"Period: {min(allDates)} ~ {max(allDates)}")
        lines.append("-" * width)

        for i, ((isStart, isEnd), (oosStart, oosEnd)) in enumerate(folds):
            isStartDays = (datetime.strptime(isStart, '%Y-%m-%d') - minDate).days
            isEndDays = (datetime.strptime(isEnd, '%Y-%m-%d') - minDate).days
            oosStartDays = (datetime.strptime(oosStart, '%Y-%m-%d') - minDate).days
            oosEndDays = (datetime.strptime(oosEnd, '%Y-%m-%d') - minDate).days

            isStartPos = int(isStartDays / totalDays * (width - 10))
            isEndPos = int(isEndDays / totalDays * (width - 10))
            oosStartPos = int(oosStartDays / totalDays * (width - 10))
            oosEndPos = int(oosEndDays / totalDays * (width - 10))

            line = [' '] * (width - 10)
            for j in range(isStartPos, min(isEndPos + 1, width - 10)):
                line[j] = '='
            for j in range(oosStartPos, min(oosEndPos + 1, width - 10)):
                line[j] = '#'

            lines.append(f"Fold {i+1:2d}: {''.join(line)}")

        lines.append("-" * width)
        lines.append("Legend: [===] In-Sample, [###] Out-of-Sample")

        return "\n".join(lines)
