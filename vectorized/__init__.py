"""
Tradix Vectorized Engine - Pure NumPy high-performance backtesting.

Ultra-fast backtesting engine and technical indicators implemented with
pure NumPy vectorized operations. No Numba or C extensions required.

Performance (10-year daily data, 2,458 bars):
    - SMA calculation: 0.006ms
    - RSI calculation: 0.009ms
    - MACD calculation: 0.040ms
    - Full backtest: 0.132ms
    - 1,000 parameter optimization: 0.02s

Usage:
    >>> from tradix.vectorized import vsma, vgoldenCross, vbacktest
    >>> from tradix import vsma, vgoldenCross  # also available at top level

    >>> result = vbacktest("005930", "goldenCross", fast=10, slow=30)
    >>> print(f"Return: {result.totalReturn:+.2f}%")

Submodules:
    - indicators: 10 vectorized technical indicators (vsma, vema, vrsi, etc.)
    - signals: 9 vectorized signal generators (vcrossover, vgoldenCross, etc.)
    - engine: VectorizedEngine, vbacktest(), voptimize()
"""

from tradix.vectorized.indicators import (
    vsma,
    vema,
    vrsi,
    vmacd,
    vbollinger,
    vatr,
    vstochastic,
    vadx,
    vroc,
    vmomentum,
)
from tradix.vectorized.signals import (
    vcrossover,
    vcrossunder,
    vcross,
    vgoldenCross,
    vrsiSignal,
    vmacdSignal,
    vbollingerSignal,
    vbreakoutSignal,
    vTrendFilter,
)
from tradix.vectorized.engine import (
    VectorizedEngine,
    VectorizedResult,
    vbacktest,
    voptimize,
)

__all__ = [
    "vsma",
    "vema",
    "vrsi",
    "vmacd",
    "vbollinger",
    "vatr",
    "vstochastic",
    "vadx",
    "vroc",
    "vmomentum",
    "vcrossover",
    "vcrossunder",
    "vcross",
    "vgoldenCross",
    "vrsiSignal",
    "vmacdSignal",
    "vbollingerSignal",
    "vbreakoutSignal",
    "vTrendFilter",
    "VectorizedEngine",
    "VectorizedResult",
    "vbacktest",
    "voptimize",
]
