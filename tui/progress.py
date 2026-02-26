"""
Rich-based progress bars for optimization and backtesting.

Provides styled progress indicators for long-running operations.
Rich 기반 프로그레스 바. 최적화, 데이터 로딩 등 장시간 작업 진행률 표시.
"""

from typing import Any, Iterable, Callable, Optional
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.console import Console

console = Console()


def optimizeProgress(
    total: int,
    description: str = "Optimizing",
) -> Progress:
    """Create a Rich progress bar for optimization loops.

    Usage:
        with optimizeProgress(100) as progress:
            task = progress.add_task("Optimizing", total=100)
            for params in param_grid:
                run_backtest(params)
                progress.advance(task)

    Args:
        total: Total number of iterations.
        description: Progress bar description.

    Returns:
        Rich Progress context manager.
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def backtestProgress(description: str = "Backtesting") -> Progress:
    """Create a Rich progress bar for backtest execution.

    Args:
        description: Progress bar description.

    Returns:
        Rich Progress context manager.
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="blue", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def wrapIterable(iterable: Iterable, total: int = None, description: str = "Processing") -> Iterable:
    """Wrap an iterable with a Rich progress bar.

    Args:
        iterable: Any iterable.
        total: Total count (auto-detected if iterable has __len__).
        description: Progress description.

    Yields:
        Items from the iterable.
    """
    if total is None:
        total = len(iterable) if hasattr(iterable, '__len__') else None

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)
