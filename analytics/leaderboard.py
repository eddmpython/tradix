"""
Tradex Strategy Leaderboard Module.

Ranking system for comparing multiple backtest strategies using a weighted
composite score. Awards Korean achievement badges and provides head-to-head
comparison, top/bottom filtering, and DataFrame export.

다수 백테스트 전략을 가중 복합 점수로 비교하는 랭킹 시스템입니다.
한국어 업적 배지를 수여하고, 1:1 비교, 상위/하위 필터링,
DataFrame 내보내기를 제공합니다.

Features:
    - Weighted composite scoring (Sharpe 30%, Return 20%, MDD 20%, Win Rate 15%, PF 15%)
    - Min-max normalization to 0-100 scale
    - Korean achievement badge system
    - Head-to-head strategy comparison
    - Top N / Bottom N filtering
    - Full leaderboard DataFrame export

Usage:
    from tradex.analytics.leaderboard import StrategyLeaderboard

    board = StrategyLeaderboard()
    board.addResult(result1)
    board.addResult(result2)
    board.addResult(result3)

    ranking = board.ranking()
    print(board.summary())
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from tradex.engine import BacktestResult


@dataclass
class LeaderboardEntry:
    """
    Single entry in the strategy leaderboard.

    전략 리더보드의 단일 항목.

    Contains the strategy's key metrics, computed composite score,
    overall rank, and earned achievement badges.

    전략의 주요 지표, 계산된 복합 점수, 전체 순위, 획득한 업적 배지를
    포함합니다.

    Attributes:
        rank (int): Overall leaderboard rank. 전체 순위.
        strategyName (str): Strategy identifier name. 전략 이름.
        totalReturn (float): Total return percentage. 총 수익률(%).
        sharpeRatio (float): Sharpe ratio. 샤프 비율.
        maxDrawdown (float): Maximum drawdown percentage. 최대 낙폭(%).
        winRate (float): Win rate percentage. 승률(%).
        profitFactor (float): Profit factor. 손익비.
        totalTrades (int): Total number of closed trades. 총 거래 수.
        compositeScore (float): Weighted composite score (0-100). 가중 복합 점수(0-100).
        badges (List[str]): Earned Korean achievement badges. 획득한 한국어 업적 배지.
    """
    rank: int = 0
    strategyName: str = ""
    totalReturn: float = 0.0
    sharpeRatio: float = 0.0
    maxDrawdown: float = 0.0
    winRate: float = 0.0
    profitFactor: float = 0.0
    totalTrades: int = 0
    compositeScore: float = 0.0
    badges: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate a Korean-language summary of this leaderboard entry.

        이 리더보드 항목의 한국어 요약을 생성합니다.

        Returns:
            str: Formatted entry summary string.
        """
        badgeStr = " ".join(self.badges) if self.badges else "없음"
        return (
            f"[#{self.rank}] {self.strategyName}\n"
            f"  복합 점수: {self.compositeScore:.1f}/100\n"
            f"  수익률: {self.totalReturn:+.2f}% | 샤프: {self.sharpeRatio:.2f}\n"
            f"  MDD: {self.maxDrawdown:.2f}% | 승률: {self.winRate:.1f}%\n"
            f"  손익비: {self.profitFactor:.2f} | 거래: {self.totalTrades}회\n"
            f"  배지: {badgeStr}"
        )

    def __repr__(self) -> str:
        return (
            f"LeaderboardEntry(#{self.rank} {self.strategyName} "
            f"score={self.compositeScore:.1f})"
        )


class StrategyLeaderboard:
    """
    Multi-strategy ranking and comparison system.

    다수 전략 랭킹 및 비교 시스템.

    Collects backtest results from multiple strategies and ranks them using
    a weighted composite score. Provides badge awards, head-to-head comparison,
    filtering, and export capabilities.

    다수 전략의 백테스트 결과를 수집하고 가중 복합 점수로 순위를 매깁니다.
    배지 수여, 1:1 비교, 필터링, 내보내기 기능을 제공합니다.

    Example:
        >>> board = StrategyLeaderboard()
        >>> board.addResult(smaResult)
        >>> board.addResult(rsiResult)
        >>> top3 = board.topN(3)
        >>> print(board.summary())
    """

    COMPOSITE_WEIGHTS = {
        "sharpeRatio": 0.30,
        "totalReturn": 0.20,
        "maxDrawdownInverted": 0.20,
        "winRate": 0.15,
        "profitFactor": 0.15,
    }

    def __init__(self):
        """
        Initialize the leaderboard with an empty results collection.

        빈 결과 컬렉션으로 리더보드를 초기화합니다.
        """
        self._results: List[BacktestResult] = []
        self._cachedRanking: Optional[List[LeaderboardEntry]] = None

    def addResult(self, result: BacktestResult) -> None:
        """
        Add a single backtest result to the leaderboard.

        리더보드에 단일 백테스트 결과를 추가합니다.

        Args:
            result (BacktestResult): Completed backtest result.
                완료된 백테스트 결과.
        """
        self._results.append(result)
        self._cachedRanking = None

    def addResults(self, results: List[BacktestResult]) -> None:
        """
        Add multiple backtest results to the leaderboard.

        리더보드에 다수 백테스트 결과를 추가합니다.

        Args:
            results (List[BacktestResult]): List of completed backtest results.
                완료된 백테스트 결과 목록.
        """
        self._results.extend(results)
        self._cachedRanking = None

    def _extractMetrics(self, result: BacktestResult) -> dict:
        """
        Extract standardized metrics from a BacktestResult.

        BacktestResult에서 표준화된 지표를 추출합니다.

        Args:
            result (BacktestResult): Source result. 원본 결과.

        Returns:
            dict: Extracted metric values. 추출된 지표 값.
        """
        metrics = result.metrics if result.metrics else {}
        return {
            "strategyName": result.strategy,
            "totalReturn": metrics.get("totalReturn", result.totalReturn),
            "sharpeRatio": metrics.get("sharpeRatio", 0.0),
            "maxDrawdown": metrics.get("maxDrawdown", 0.0),
            "winRate": metrics.get("winRate", result.winRate),
            "profitFactor": metrics.get("profitFactor", 0.0),
            "totalTrades": metrics.get("totalTrades", result.totalTrades),
        }

    def _normalizeMinMax(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize an array to the 0-100 scale using min-max normalization.

        min-max 정규화를 사용하여 배열을 0-100 스케일로 정규화합니다.

        Args:
            values (np.ndarray): Raw metric values. 원시 지표 값.

        Returns:
            np.ndarray: Normalized values in [0, 100]. [0, 100] 범위의 정규화 값.
        """
        minVal = np.min(values)
        maxVal = np.max(values)
        if maxVal == minVal:
            return np.full_like(values, 50.0, dtype=float)
        return (values - minVal) / (maxVal - minVal) * 100

    def ranking(self, metric: str = "compositeScore", ascending: bool = False) -> List[LeaderboardEntry]:
        """
        Compute and return the ranked leaderboard entries.

        순위가 매겨진 리더보드 항목을 계산하고 반환합니다.

        Composite score formula (all normalized to 0-100):
            - Sharpe ratio: 30%
            - Total return: 20%
            - Max drawdown inverted: 20% (lower MDD = higher score)
            - Win rate: 15%
            - Profit factor: 15%

        복합 점수 공식 (모두 0-100으로 정규화):
            - 샤프 비율: 30%
            - 총 수익률: 20%
            - 최대 낙폭 반전: 20% (낮은 MDD = 높은 점수)
            - 승률: 15%
            - 손익비: 15%

        Args:
            metric (str): Metric to sort by. Default 'compositeScore'.
                정렬 기준 지표. 기본값 'compositeScore'.
            ascending (bool): Sort in ascending order. Default False.
                오름차순 정렬. 기본값 False.

        Returns:
            List[LeaderboardEntry]: Ranked list of leaderboard entries.
                순위가 매겨진 리더보드 항목 목록.
        """
        if not self._results:
            return []

        allMetrics = [self._extractMetrics(r) for r in self._results]

        totalReturns = np.array([m["totalReturn"] for m in allMetrics], dtype=float)
        sharpeRatios = np.array([m["sharpeRatio"] for m in allMetrics], dtype=float)
        maxDrawdowns = np.array([m["maxDrawdown"] for m in allMetrics], dtype=float)
        winRates = np.array([m["winRate"] for m in allMetrics], dtype=float)
        profitFactors = np.array([m["profitFactor"] for m in allMetrics], dtype=float)

        profitFactors = np.clip(profitFactors, 0.0, 10.0)

        normReturn = self._normalizeMinMax(totalReturns)
        normSharpe = self._normalizeMinMax(sharpeRatios)
        normMddInverted = self._normalizeMinMax(-maxDrawdowns)
        normWinRate = self._normalizeMinMax(winRates)
        normProfitFactor = self._normalizeMinMax(profitFactors)

        compositeScores = (
            self.COMPOSITE_WEIGHTS["sharpeRatio"] * normSharpe
            + self.COMPOSITE_WEIGHTS["totalReturn"] * normReturn
            + self.COMPOSITE_WEIGHTS["maxDrawdownInverted"] * normMddInverted
            + self.COMPOSITE_WEIGHTS["winRate"] * normWinRate
            + self.COMPOSITE_WEIGHTS["profitFactor"] * normProfitFactor
        )

        entries = []
        for i, m in enumerate(allMetrics):
            entry = LeaderboardEntry(
                rank=0,
                strategyName=m["strategyName"],
                totalReturn=m["totalReturn"],
                sharpeRatio=m["sharpeRatio"],
                maxDrawdown=m["maxDrawdown"],
                winRate=m["winRate"],
                profitFactor=m["profitFactor"],
                totalTrades=m["totalTrades"],
                compositeScore=float(compositeScores[i]),
                badges=[],
            )
            entries.append(entry)

        sortKeyMap = {
            "compositeScore": lambda e: e.compositeScore,
            "totalReturn": lambda e: e.totalReturn,
            "sharpeRatio": lambda e: e.sharpeRatio,
            "maxDrawdown": lambda e: e.maxDrawdown,
            "winRate": lambda e: e.winRate,
            "profitFactor": lambda e: e.profitFactor,
            "totalTrades": lambda e: e.totalTrades,
        }
        sortKey = sortKeyMap.get(metric, lambda e: e.compositeScore)
        entries.sort(key=sortKey, reverse=(not ascending))

        for idx, entry in enumerate(entries):
            entry.rank = idx + 1
            entry.badges = self.badges(entry, entries)

        self._cachedRanking = entries
        return entries

    def badges(self, entry: LeaderboardEntry, allEntries: List[LeaderboardEntry] = None) -> List[str]:
        """
        Award Korean achievement badges to a leaderboard entry.

        리더보드 항목에 한국어 업적 배지를 수여합니다.

        Badge criteria:
            - "수익왕": Highest total return
            - "안정왕": Lowest (least negative) max drawdown
            - "샤프왕": Highest Sharpe ratio
            - "승률왕": Highest win rate
            - "거래왕": Most trades
            - "수비왕": Best defensive metrics (MDD > -10%)
            - "손익비왕": Best profit factor
            - "올라운더": All metrics above average

        Args:
            entry (LeaderboardEntry): Entry to evaluate for badges.
                배지를 평가할 항목.
            allEntries (List[LeaderboardEntry], optional): Full list for
                comparison. If None, uses cached ranking.
                비교를 위한 전체 목록. None이면 캐시된 순위 사용.

        Returns:
            List[str]: List of earned Korean badge strings. 획득한 한국어 배지 목록.
        """
        if allEntries is None:
            if self._cachedRanking:
                allEntries = self._cachedRanking
            else:
                return []

        if not allEntries:
            return []

        earned = []

        maxReturn = max(e.totalReturn for e in allEntries)
        if entry.totalReturn == maxReturn:
            earned.append("수익왕")

        bestMdd = max(e.maxDrawdown for e in allEntries)
        if entry.maxDrawdown == bestMdd:
            earned.append("안정왕")

        maxSharpe = max(e.sharpeRatio for e in allEntries)
        if entry.sharpeRatio == maxSharpe:
            earned.append("샤프왕")

        maxWinRate = max(e.winRate for e in allEntries)
        if entry.winRate == maxWinRate:
            earned.append("승률왕")

        maxTrades = max(e.totalTrades for e in allEntries)
        if entry.totalTrades == maxTrades:
            earned.append("거래왕")

        if entry.maxDrawdown > -10.0:
            earned.append("수비왕")

        maxPf = max(e.profitFactor for e in allEntries)
        if entry.profitFactor == maxPf:
            earned.append("손익비왕")

        avgReturn = np.mean([e.totalReturn for e in allEntries])
        avgSharpe = np.mean([e.sharpeRatio for e in allEntries])
        avgMdd = np.mean([e.maxDrawdown for e in allEntries])
        avgWinRate = np.mean([e.winRate for e in allEntries])
        avgPf = np.mean([e.profitFactor for e in allEntries])

        if (
            entry.totalReturn > avgReturn
            and entry.sharpeRatio > avgSharpe
            and entry.maxDrawdown > avgMdd
            and entry.winRate > avgWinRate
            and entry.profitFactor > avgPf
        ):
            earned.append("올라운더")

        return earned

    def topN(self, n: int = 10) -> List[LeaderboardEntry]:
        """
        Return the top N strategies by composite score.

        복합 점수 기준 상위 N개 전략을 반환합니다.

        Args:
            n (int): Number of top entries to return. Default 10.
                반환할 상위 항목 수. 기본값 10.

        Returns:
            List[LeaderboardEntry]: Top N ranked entries. 상위 N개 순위 항목.
        """
        ranked = self.ranking()
        return ranked[:n]

    def bottomN(self, n: int = 5) -> List[LeaderboardEntry]:
        """
        Return the bottom N strategies by composite score.

        복합 점수 기준 하위 N개 전략을 반환합니다.

        Args:
            n (int): Number of bottom entries to return. Default 5.
                반환할 하위 항목 수. 기본값 5.

        Returns:
            List[LeaderboardEntry]: Bottom N ranked entries. 하위 N개 순위 항목.
        """
        ranked = self.ranking()
        return ranked[-n:] if len(ranked) >= n else ranked

    def compareTwo(self, name1: str, name2: str) -> dict:
        """
        Perform a head-to-head comparison between two strategies.

        두 전략 간 1:1 비교를 수행합니다.

        Args:
            name1 (str): First strategy name. 첫 번째 전략 이름.
            name2 (str): Second strategy name. 두 번째 전략 이름.

        Returns:
            dict: Comparison results with metric-by-metric winners and
                overall winner. 지표별 승자와 전체 승자가 포함된 비교 결과.

        Raises:
            ValueError: If either strategy name is not found.
                전략 이름을 찾을 수 없으면 ValueError 발생.
        """
        ranked = self.ranking()
        entry1 = None
        entry2 = None

        for entry in ranked:
            if entry.strategyName == name1:
                entry1 = entry
            if entry.strategyName == name2:
                entry2 = entry

        if entry1 is None:
            raise ValueError(f"전략 '{name1}'을(를) 찾을 수 없습니다.")
        if entry2 is None:
            raise ValueError(f"전략 '{name2}'을(를) 찾을 수 없습니다.")

        metrics = {
            "totalReturn": ("총 수익률", entry1.totalReturn, entry2.totalReturn, True),
            "sharpeRatio": ("샤프 비율", entry1.sharpeRatio, entry2.sharpeRatio, True),
            "maxDrawdown": ("최대 낙폭", entry1.maxDrawdown, entry2.maxDrawdown, False),
            "winRate": ("승률", entry1.winRate, entry2.winRate, True),
            "profitFactor": ("손익비", entry1.profitFactor, entry2.profitFactor, True),
            "compositeScore": ("복합 점수", entry1.compositeScore, entry2.compositeScore, True),
        }

        comparison = {}
        winsForName1 = 0
        winsForName2 = 0

        for key, (label, val1, val2, higherIsBetter) in metrics.items():
            if higherIsBetter:
                winner = name1 if val1 >= val2 else name2
            else:
                winner = name1 if val1 >= val2 else name2

            if winner == name1:
                winsForName1 += 1
            else:
                winsForName2 += 1

            comparison[key] = {
                "label": label,
                name1: val1,
                name2: val2,
                "winner": winner,
            }

        overallWinner = name1 if entry1.compositeScore >= entry2.compositeScore else name2

        return {
            "metrics": comparison,
            "winsForName1": winsForName1,
            "winsForName2": winsForName2,
            "overallWinner": overallWinner,
            "entry1": entry1,
            "entry2": entry2,
        }

    def toDataFrame(self) -> pd.DataFrame:
        """
        Export the full leaderboard as a pandas DataFrame.

        전체 리더보드를 pandas DataFrame으로 내보냅니다.

        Returns:
            pd.DataFrame: DataFrame with one row per ranked strategy.
                순위별 전략당 한 행을 가진 DataFrame.
        """
        ranked = self.ranking()
        if not ranked:
            return pd.DataFrame()

        records = []
        for entry in ranked:
            records.append({
                "rank": entry.rank,
                "strategyName": entry.strategyName,
                "totalReturn": entry.totalReturn,
                "sharpeRatio": entry.sharpeRatio,
                "maxDrawdown": entry.maxDrawdown,
                "winRate": entry.winRate,
                "profitFactor": entry.profitFactor,
                "totalTrades": entry.totalTrades,
                "compositeScore": entry.compositeScore,
                "badges": " ".join(entry.badges),
            })

        return pd.DataFrame(records)

    def summary(self) -> str:
        """
        Generate a Korean-language formatted leaderboard table.

        한국어 포맷의 리더보드 테이블을 생성합니다.

        Returns:
            str: Formatted multi-line leaderboard summary.
                포맷된 여러 줄 리더보드 요약.
        """
        ranked = self.ranking()
        if not ranked:
            return "등록된 전략이 없습니다."

        lines = [
            f"{'='*70}",
            f"전략 리더보드",
            f"{'='*70}",
            f"{'순위':>4} | {'전략명':<20} | {'점수':>6} | {'수익률':>8} | "
            f"{'샤프':>6} | {'MDD':>7} | {'승률':>6} | 배지",
            f"{'─'*70}",
        ]

        for entry in ranked:
            badgeStr = " ".join(entry.badges) if entry.badges else ""
            lines.append(
                f"{entry.rank:>4} | {entry.strategyName:<20} | "
                f"{entry.compositeScore:>5.1f} | "
                f"{entry.totalReturn:>+7.2f}% | "
                f"{entry.sharpeRatio:>5.2f} | "
                f"{entry.maxDrawdown:>6.2f}% | "
                f"{entry.winRate:>5.1f}% | {badgeStr}"
            )

        lines.append(f"{'='*70}")

        if len(ranked) >= 2:
            best = ranked[0]
            lines.append(f"\n최우수 전략: {best.strategyName} (점수: {best.compositeScore:.1f}/100)")
            if best.badges:
                lines.append(f"획득 배지: {' '.join(best.badges)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"StrategyLeaderboard(strategies={len(self._results)})"
