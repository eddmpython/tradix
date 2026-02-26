"""
Tradex Trading Journal Module.

Automatic trading diary with per-trade analytics, including MFE/MAE
(Maximum Favorable/Adverse Excursion), R-multiple grading, auto-generated
Korean tags, and comprehensive streak/time-based analysis.

자동 트레이딩 일지 모듈입니다.
MFE/MAE (최대 유리/불리 이탈), R-배수 등급, 자동 생성 한국어 태그,
연승/연패 분석, 시간 기반 분석을 포함한 거래별 분석을 제공합니다.

Features:
    - Per-trade MFE/MAE computation from equity curve
    - R-multiple calculation and A-F grade assignment
    - Automatic Korean tag categorization per trade
    - Win/loss streak analysis
    - Monthly and weekday trade grouping
    - DataFrame export for further analysis

Usage:
    from tradex.analytics.tradingJournal import TradingJournal

    journal = TradingJournal(result)
    entries = journal.generate()
    print(journal.summary())
    df = journal.toDataFrame()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

from tradex.engine import BacktestResult
from tradex.entities.trade import Trade


@dataclass
class JournalEntry:
    """
    Single trade journal entry with detailed analytics.

    상세 분석이 포함된 단일 거래 일지 항목.

    Each entry captures the complete lifecycle of a trade along with
    computed analytics such as MFE, MAE, R-multiple, grade, and
    auto-generated descriptive tags.

    각 항목은 거래의 전체 수명주기와 함께 MFE, MAE, R-배수, 등급,
    자동 생성된 설명 태그 등 계산된 분석을 담고 있습니다.

    Attributes:
        tradeNumber (int): Sequential trade number. 순차 거래 번호.
        entryDate (str): Entry date string. 진입일.
        exitDate (str): Exit date string. 청산일.
        side (str): Trade direction - "매수" or "매도". 거래 방향.
        symbol (str): Ticker symbol. 종목 코드.
        entryPrice (float): Entry execution price. 진입 가격.
        exitPrice (float): Exit execution price. 청산 가격.
        quantity (int): Number of shares traded. 거래 수량.
        pnl (float): Net profit/loss amount. 순손익.
        pnlPercent (float): Net P&L as percentage. 손익률(%).
        holdingDays (int): Holding period in calendar days. 보유 기간(일).
        maxFavorableExcursion (float): Maximum favorable price movement (%)
            during the trade. 거래 중 최대 유리 가격 이동(%).
        maxAdverseExcursion (float): Maximum adverse price movement (%)
            during the trade. 거래 중 최대 불리 가격 이동(%).
        rMultiple (float): R-multiple (PnL / initial risk unit).
            R-배수 (손익 / 초기 위험 단위).
        grade (str): Trade quality grade A through F. 거래 품질 등급 A~F.
        tags (List[str]): Auto-generated Korean categorization tags.
            자동 생성된 한국어 분류 태그.
        notes (str): Auto-generated analysis note. 자동 생성된 분석 메모.
    """
    tradeNumber: int = 0
    entryDate: str = ""
    exitDate: str = ""
    side: str = ""
    symbol: str = ""
    entryPrice: float = 0.0
    exitPrice: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnlPercent: float = 0.0
    holdingDays: int = 0
    maxFavorableExcursion: float = 0.0
    maxAdverseExcursion: float = 0.0
    rMultiple: float = 0.0
    grade: str = "F"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        """
        Generate a Korean-language summary of this journal entry.

        이 일지 항목의 한국어 요약을 생성합니다.

        Returns:
            str: Formatted single-trade summary string.
        """
        tagStr = ", ".join(self.tags) if self.tags else "없음"
        return (
            f"[거래 #{self.tradeNumber}] {self.symbol} {self.side}\n"
            f"  기간: {self.entryDate} ~ {self.exitDate} ({self.holdingDays}일)\n"
            f"  가격: {self.entryPrice:,.0f} -> {self.exitPrice:,.0f}\n"
            f"  손익: {self.pnl:+,.0f} ({self.pnlPercent:+.2f}%)\n"
            f"  MFE: {self.maxFavorableExcursion:+.2f}% | MAE: {self.maxAdverseExcursion:+.2f}%\n"
            f"  R-배수: {self.rMultiple:.2f} | 등급: {self.grade}\n"
            f"  태그: {tagStr}\n"
            f"  메모: {self.notes}"
        )

    def __repr__(self) -> str:
        return (
            f"JournalEntry(#{self.tradeNumber} {self.symbol} {self.side} "
            f"pnl={self.pnl:+,.0f} grade={self.grade})"
        )


class TradingJournal:
    """
    Automatic trading journal generator with per-trade analytics.

    거래별 분석이 포함된 자동 트레이딩 일지 생성기.

    Processes all closed trades from a BacktestResult to produce journal
    entries with MFE/MAE, R-multiples, grades, and tags. Provides filtering,
    grouping, streak analysis, and DataFrame export.

    BacktestResult의 모든 청산 거래를 처리하여 MFE/MAE, R-배수, 등급,
    태그가 포함된 일지 항목을 생성합니다. 필터링, 그룹화, 연승/연패 분석,
    DataFrame 내보내기를 제공합니다.

    Example:
        >>> journal = TradingJournal(result)
        >>> entries = journal.generate()
        >>> print(journal.summary())
        >>> best = journal.bestTrade()
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize the journal from a backtest result.

        백테스트 결과로 일지를 초기화합니다.

        Args:
            result (BacktestResult): Completed backtest result with trades
                and equity curve. 거래 내역과 자산 곡선이 포함된 백테스트 결과.
        """
        self._result = result
        self._closedTrades = [t for t in result.trades if t.isClosed]
        self._equityCurve = result.equityCurve
        self._entries: List[JournalEntry] = []
        self._generated = False

    def generate(self) -> List[JournalEntry]:
        """
        Generate the full trading journal with analytics for each trade.

        각 거래에 대한 분석이 포함된 전체 트레이딩 일지를 생성합니다.

        For each closed trade, computes MFE/MAE from the equity curve during
        the trade period, calculates R-multiple, assigns a grade, generates
        Korean tags, and creates an analysis note.

        각 청산 거래에 대해, 거래 기간 동안의 자산 곡선에서 MFE/MAE를 계산하고,
        R-배수를 산출하고, 등급을 부여하고, 한국어 태그를 생성하고, 분석 메모를
        작성합니다.

        Returns:
            List[JournalEntry]: Complete list of journal entries.
                전체 일지 항목 목록.
        """
        self._entries = []
        avgLossPercent = self._calculateAverageLossPercent()

        for idx, trade in enumerate(self._closedTrades):
            mfe, mae = self._computeMfeMae(trade)
            rMultiple = self._calculateRMultiple(trade, avgLossPercent)
            grade = self._assignGrade(rMultiple)
            tags = self._generateTags(trade, mfe, mae, rMultiple)
            notes = self._generateNotes(trade, mfe, mae, rMultiple, grade)

            sideLabel = "매수" if trade.side.value == "buy" else "매도"

            entry = JournalEntry(
                tradeNumber=idx + 1,
                entryDate=trade.entryDate.strftime("%Y-%m-%d"),
                exitDate=trade.exitDate.strftime("%Y-%m-%d"),
                side=sideLabel,
                symbol=trade.symbol,
                entryPrice=trade.entryPrice,
                exitPrice=trade.exitPrice,
                quantity=int(trade.quantity),
                pnl=trade.pnl,
                pnlPercent=trade.pnlPercent,
                holdingDays=trade.holdingDays,
                maxFavorableExcursion=mfe,
                maxAdverseExcursion=mae,
                rMultiple=rMultiple,
                grade=grade,
                tags=tags,
                notes=notes,
            )
            self._entries.append(entry)

        self._generated = True
        return self._entries

    def _ensureGenerated(self) -> None:
        """
        Ensure journal entries have been generated.

        일지 항목이 생성되었는지 확인합니다.
        """
        if not self._generated:
            self.generate()

    def _calculateAverageLossPercent(self) -> float:
        """
        Calculate the average loss percentage across all losing trades.

        모든 손실 거래의 평균 손실률을 계산합니다.

        Returns:
            float: Average loss percentage (positive value), or 1.0 if no losses.
                평균 손실률(양수 값), 손실 거래가 없으면 1.0.
        """
        losses = [abs(t.pnlPercent) for t in self._closedTrades if t.pnl < 0]
        if not losses:
            return 1.0
        return float(np.mean(losses))

    def _computeMfeMae(self, trade: Trade) -> tuple:
        """
        Compute Maximum Favorable and Adverse Excursion for a trade.

        거래의 최대 유리 이탈(MFE)과 최대 불리 이탈(MAE)을 계산합니다.

        Uses the equity curve during the trade's holding period to determine
        the best and worst unrealized P&L relative to the entry equity.

        거래 보유 기간 동안의 자산 곡선을 사용하여 진입 시점 자산 대비
        최고 및 최저 미실현 손익을 결정합니다.

        Args:
            trade (Trade): The closed trade to analyze. 분석할 청산 거래.

        Returns:
            tuple: (MFE percentage, MAE percentage). (MFE 퍼센트, MAE 퍼센트).
        """
        if self._equityCurve.empty:
            return 0.0, 0.0

        mask = (self._equityCurve.index >= trade.entryDate) & (
            self._equityCurve.index <= trade.exitDate
        )
        tradeEquity = self._equityCurve[mask]

        if len(tradeEquity) < 2:
            return 0.0, 0.0

        entryEquity = tradeEquity.iloc[0]
        if entryEquity == 0:
            return 0.0, 0.0

        equityReturns = (tradeEquity - entryEquity) / entryEquity * 100
        mfe = float(equityReturns.max())
        mae = float(equityReturns.min())

        return mfe, mae

    def _calculateRMultiple(self, trade: Trade, avgLossPercent: float) -> float:
        """
        Calculate the R-multiple for a trade.

        거래의 R-배수를 계산합니다.

        R-multiple measures how many units of initial risk the trade captured.
        Defined as pnlPercent / avgLossPercent.

        R-배수는 거래가 초기 위험의 몇 배를 포착했는지를 측정합니다.
        pnlPercent / avgLossPercent로 정의됩니다.

        Args:
            trade (Trade): The trade to evaluate. 평가할 거래.
            avgLossPercent (float): Average loss percentage as risk unit.
                위험 단위로서의 평균 손실률.

        Returns:
            float: R-multiple value. R-배수 값.
        """
        if avgLossPercent == 0:
            return 0.0
        return trade.pnlPercent / avgLossPercent

    @staticmethod
    def _assignGrade(rMultiple: float) -> str:
        """
        Assign a trade quality grade based on R-multiple.

        R-배수 기반으로 거래 품질 등급을 부여합니다.

        Grading scale:
            A: R > 3 (exceptional trade)
            B: R > 2 (good trade)
            C: R > 1 (acceptable trade)
            D: R > 0 (marginally profitable)
            F: R <= 0 (losing trade)

        Args:
            rMultiple (float): The computed R-multiple. 계산된 R-배수.

        Returns:
            str: Grade letter A through F. 등급 문자 A~F.
        """
        if rMultiple > 3:
            return "A"
        if rMultiple > 2:
            return "B"
        if rMultiple > 1:
            return "C"
        if rMultiple > 0:
            return "D"
        return "F"

    @staticmethod
    def _generateTags(trade: Trade, mfe: float, mae: float, rMultiple: float) -> List[str]:
        """
        Auto-generate Korean categorization tags for a trade.

        거래에 대한 한국어 분류 태그를 자동 생성합니다.

        Tag categories:
            - "큰수익": PnL% > 5%
            - "손절": Losing trade
            - "장기보유": Holding > 20 days
            - "단기거래": Holding <= 5 days
            - "추세추종": MFE > 3% and profitable
            - "역추세": MAE < -3% and still profitable

        Args:
            trade (Trade): The trade to tag. 태그할 거래.
            mfe (float): Maximum Favorable Excursion %. MFE(%).
            mae (float): Maximum Adverse Excursion %. MAE(%).
            rMultiple (float): R-multiple value. R-배수 값.

        Returns:
            List[str]: Generated Korean tag list. 생성된 한국어 태그 목록.
        """
        tags = []

        if trade.pnlPercent > 5.0:
            tags.append("큰수익")
        if trade.pnl < 0:
            tags.append("손절")
        if trade.holdingDays > 20:
            tags.append("장기보유")
        if trade.holdingDays <= 5:
            tags.append("단기거래")
        if mfe > 3.0 and trade.pnl > 0:
            tags.append("추세추종")
        if mae < -3.0 and trade.pnl > 0:
            tags.append("역추세")

        return tags

    @staticmethod
    def _generateNotes(
        trade: Trade, mfe: float, mae: float, rMultiple: float, grade: str
    ) -> str:
        """
        Auto-generate an analysis note for the trade.

        거래에 대한 분석 메모를 자동 생성합니다.

        Args:
            trade (Trade): The trade to annotate. 메모할 거래.
            mfe (float): Maximum Favorable Excursion %. MFE(%).
            mae (float): Maximum Adverse Excursion %. MAE(%).
            rMultiple (float): R-multiple. R-배수.
            grade (str): Trade grade. 거래 등급.

        Returns:
            str: Korean analysis note. 한국어 분석 메모.
        """
        parts = []

        if grade in ("A", "B"):
            parts.append("우수한 거래")
        elif grade == "C":
            parts.append("양호한 거래")
        elif grade == "D":
            parts.append("소폭 수익 거래")
        else:
            parts.append("손실 거래")

        captureRatio = 0.0
        if mfe != 0:
            captureRatio = trade.pnlPercent / mfe * 100

        if mfe > 0 and captureRatio < 50:
            parts.append(f"MFE 대비 수익 포착률 낮음({captureRatio:.0f}%)")
        elif mfe > 0 and captureRatio >= 80:
            parts.append(f"MFE 대비 수익 포착률 우수({captureRatio:.0f}%)")

        if mae < -5:
            parts.append(f"큰 역행 발생(MAE {mae:.1f}%)")

        if trade.holdingDays > 60:
            parts.append("장기 보유 전략")
        elif trade.holdingDays <= 3:
            parts.append("초단기 거래")

        return ". ".join(parts) + "."

    def winningTrades(self) -> List[JournalEntry]:
        """
        Filter and return only winning trade entries.

        수익 거래 항목만 필터링하여 반환합니다.

        Returns:
            List[JournalEntry]: Entries with positive P&L. 양의 손익을 가진 항목.
        """
        self._ensureGenerated()
        return [e for e in self._entries if e.pnl > 0]

    def losingTrades(self) -> List[JournalEntry]:
        """
        Filter and return only losing trade entries.

        손실 거래 항목만 필터링하여 반환합니다.

        Returns:
            List[JournalEntry]: Entries with non-positive P&L. 0 이하 손익을 가진 항목.
        """
        self._ensureGenerated()
        return [e for e in self._entries if e.pnl <= 0]

    def bestTrade(self) -> Optional[JournalEntry]:
        """
        Return the single best trade by P&L percentage.

        손익률 기준 최고 거래를 반환합니다.

        Returns:
            JournalEntry: Highest PnL% entry, or None if no entries.
                최고 손익률 항목, 항목 없으면 None.
        """
        self._ensureGenerated()
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.pnlPercent)

    def worstTrade(self) -> Optional[JournalEntry]:
        """
        Return the single worst trade by P&L percentage.

        손익률 기준 최악 거래를 반환합니다.

        Returns:
            JournalEntry: Lowest PnL% entry, or None if no entries.
                최저 손익률 항목, 항목 없으면 None.
        """
        self._ensureGenerated()
        if not self._entries:
            return None
        return min(self._entries, key=lambda e: e.pnlPercent)

    def tradesByMonth(self) -> Dict[str, List[JournalEntry]]:
        """
        Group journal entries by year-month.

        일지 항목을 연-월별로 그룹화합니다.

        Returns:
            dict: Mapping of "YYYY-MM" strings to lists of JournalEntry.
                "YYYY-MM" 문자열에서 JournalEntry 목록으로의 매핑.
        """
        self._ensureGenerated()
        grouped: Dict[str, List[JournalEntry]] = defaultdict(list)
        for entry in self._entries:
            monthKey = entry.entryDate[:7]
            grouped[monthKey].append(entry)
        return dict(grouped)

    def tradesByWeekday(self) -> Dict[str, List[JournalEntry]]:
        """
        Group journal entries by weekday of the entry date.

        진입일의 요일별로 일지 항목을 그룹화합니다.

        Returns:
            dict: Mapping of Korean weekday names to lists of JournalEntry.
                한국어 요일 이름에서 JournalEntry 목록으로의 매핑.
        """
        self._ensureGenerated()
        weekdayNames = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        grouped: Dict[str, List[JournalEntry]] = defaultdict(list)

        for entry in self._entries:
            entryDt = pd.Timestamp(entry.entryDate)
            dayName = weekdayNames[entryDt.weekday()]
            grouped[dayName].append(entry)

        return dict(grouped)

    def streakAnalysis(self) -> dict:
        """
        Analyze winning and losing streaks across all trades.

        모든 거래에 대한 연승/연패 분석을 수행합니다.

        Returns:
            dict: Streak statistics including current, max, and average
                win/loss streaks. 현재, 최대, 평균 연승/연패 통계.
        """
        self._ensureGenerated()
        if not self._entries:
            return {
                "maxWinStreak": 0,
                "maxLossStreak": 0,
                "currentStreak": 0,
                "currentStreakType": "",
                "avgWinStreak": 0.0,
                "avgLossStreak": 0.0,
            }

        winStreaks = []
        lossStreaks = []
        currentWin = 0
        currentLoss = 0
        maxWin = 0
        maxLoss = 0

        for entry in self._entries:
            if entry.pnl > 0:
                currentWin += 1
                if currentLoss > 0:
                    lossStreaks.append(currentLoss)
                currentLoss = 0
                maxWin = max(maxWin, currentWin)
            else:
                currentLoss += 1
                if currentWin > 0:
                    winStreaks.append(currentWin)
                currentWin = 0
                maxLoss = max(maxLoss, currentLoss)

        if currentWin > 0:
            winStreaks.append(currentWin)
        if currentLoss > 0:
            lossStreaks.append(currentLoss)

        lastEntry = self._entries[-1]
        currentStreakType = "승" if lastEntry.pnl > 0 else "패"
        currentStreak = currentWin if lastEntry.pnl > 0 else currentLoss

        return {
            "maxWinStreak": maxWin,
            "maxLossStreak": maxLoss,
            "currentStreak": currentStreak,
            "currentStreakType": currentStreakType,
            "avgWinStreak": float(np.mean(winStreaks)) if winStreaks else 0.0,
            "avgLossStreak": float(np.mean(lossStreaks)) if lossStreaks else 0.0,
        }

    def mfeAnalysis(self) -> dict:
        """
        Analyze Maximum Favorable Excursion statistics across all trades.

        모든 거래에 대한 최대 유리 이탈(MFE) 통계를 분석합니다.

        Returns:
            dict: MFE statistics including mean, median, max, and
                capture ratio for winners. MFE 평균, 중앙값, 최대, 수익 포착률.
        """
        self._ensureGenerated()
        if not self._entries:
            return {
                "meanMfe": 0.0,
                "medianMfe": 0.0,
                "maxMfe": 0.0,
                "avgCaptureRatio": 0.0,
            }

        mfeValues = np.array([e.maxFavorableExcursion for e in self._entries])

        captureRatios = []
        for entry in self._entries:
            if entry.maxFavorableExcursion > 0:
                ratio = entry.pnlPercent / entry.maxFavorableExcursion * 100
                captureRatios.append(ratio)

        return {
            "meanMfe": float(np.mean(mfeValues)),
            "medianMfe": float(np.median(mfeValues)),
            "maxMfe": float(np.max(mfeValues)),
            "avgCaptureRatio": float(np.mean(captureRatios)) if captureRatios else 0.0,
        }

    def maeAnalysis(self) -> dict:
        """
        Analyze Maximum Adverse Excursion statistics across all trades.

        모든 거래에 대한 최대 불리 이탈(MAE) 통계를 분석합니다.

        Returns:
            dict: MAE statistics including mean, median, worst, and
                recovery ratio. MAE 평균, 중앙값, 최악, 회복률.
        """
        self._ensureGenerated()
        if not self._entries:
            return {
                "meanMae": 0.0,
                "medianMae": 0.0,
                "worstMae": 0.0,
                "recoveryRate": 0.0,
            }

        maeValues = np.array([e.maxAdverseExcursion for e in self._entries])

        recoveredCount = sum(
            1 for e in self._entries
            if e.maxAdverseExcursion < -1.0 and e.pnl > 0
        )
        adverseCount = sum(1 for e in self._entries if e.maxAdverseExcursion < -1.0)
        recoveryRate = (recoveredCount / adverseCount * 100) if adverseCount > 0 else 0.0

        return {
            "meanMae": float(np.mean(maeValues)),
            "medianMae": float(np.median(maeValues)),
            "worstMae": float(np.min(maeValues)),
            "recoveryRate": recoveryRate,
        }

    def toDataFrame(self) -> pd.DataFrame:
        """
        Export all journal entries as a pandas DataFrame.

        모든 일지 항목을 pandas DataFrame으로 내보냅니다.

        Returns:
            pd.DataFrame: DataFrame with one row per trade entry.
                거래 항목당 한 행을 가진 DataFrame.
        """
        self._ensureGenerated()
        if not self._entries:
            return pd.DataFrame()

        records = []
        for entry in self._entries:
            records.append({
                "tradeNumber": entry.tradeNumber,
                "entryDate": entry.entryDate,
                "exitDate": entry.exitDate,
                "side": entry.side,
                "symbol": entry.symbol,
                "entryPrice": entry.entryPrice,
                "exitPrice": entry.exitPrice,
                "quantity": entry.quantity,
                "pnl": entry.pnl,
                "pnlPercent": entry.pnlPercent,
                "holdingDays": entry.holdingDays,
                "mfe": entry.maxFavorableExcursion,
                "mae": entry.maxAdverseExcursion,
                "rMultiple": entry.rMultiple,
                "grade": entry.grade,
                "tags": ", ".join(entry.tags),
                "notes": entry.notes,
            })

        return pd.DataFrame(records)

    def summary(self) -> str:
        """
        Generate a Korean-language summary of the trading journal.

        트레이딩 일지의 한국어 요약을 생성합니다.

        Returns:
            str: Formatted multi-line journal summary.
                포맷된 여러 줄 일지 요약.
        """
        self._ensureGenerated()

        if not self._entries:
            return "기록된 거래가 없습니다."

        totalTrades = len(self._entries)
        winners = self.winningTrades()
        losers = self.losingTrades()

        avgPnlPercent = float(np.mean([e.pnlPercent for e in self._entries]))
        avgMfe = float(np.mean([e.maxFavorableExcursion for e in self._entries]))
        avgMae = float(np.mean([e.maxAdverseExcursion for e in self._entries]))
        avgRMultiple = float(np.mean([e.rMultiple for e in self._entries]))

        gradeDistribution: Dict[str, int] = defaultdict(int)
        for entry in self._entries:
            gradeDistribution[entry.grade] += 1

        streaks = self.streakAnalysis()

        best = self.bestTrade()
        worst = self.worstTrade()

        lines = [
            f"{'='*50}",
            f"트레이딩 일지 요약",
            f"{'='*50}",
            f"총 거래: {totalTrades}회",
            f"수익 거래: {len(winners)}회 | 손실 거래: {len(losers)}회",
            f"{'─'*50}",
            f"평균 손익률: {avgPnlPercent:+.2f}%",
            f"평균 MFE: {avgMfe:+.2f}% | 평균 MAE: {avgMae:+.2f}%",
            f"평균 R-배수: {avgRMultiple:.2f}",
            f"{'─'*50}",
            f"등급 분포:",
        ]

        for grade in ["A", "B", "C", "D", "F"]:
            count = gradeDistribution.get(grade, 0)
            pct = count / totalTrades * 100
            lines.append(f"  {grade}: {count}회 ({pct:.1f}%)")

        lines.extend([
            f"{'─'*50}",
            f"최대 연승: {streaks['maxWinStreak']}회 | 최대 연패: {streaks['maxLossStreak']}회",
            f"현재 연속: {streaks['currentStreak']}연{streaks['currentStreakType']}",
        ])

        if best:
            lines.append(f"{'─'*50}")
            lines.append(
                f"최고 거래: #{best.tradeNumber} "
                f"{best.symbol} ({best.pnlPercent:+.2f}%, 등급 {best.grade})"
            )
        if worst:
            lines.append(
                f"최악 거래: #{worst.tradeNumber} "
                f"{worst.symbol} ({worst.pnlPercent:+.2f}%, 등급 {worst.grade})"
            )

        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TradingJournal(trades={len(self._closedTrades)}, "
            f"generated={self._generated})"
        )
