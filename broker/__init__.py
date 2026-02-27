"""
Tradix Broker Package.

Provides order execution simulation with pluggable commission, slippage,
and fill price models for realistic backtesting across different markets.

브로커 패키지 - 다양한 시장에 대한 현실적 주문 체결 시뮬레이션을 제공합니다.

Features:
    - BrokerSimulator with configurable execution pipeline
    - Commission models: Korea stock, US stock, fixed, percentage, zero
    - Slippage models: percent, volume-based, spread, fixed, random, zero
    - Fill price models: close, open, VWAP, worst-case, best-case, random

Usage:
    >>> from tradix.broker import BrokerSimulator
    >>> broker = BrokerSimulator.korea(mobileApp=True)
    >>> fill_event = broker.processOrder(order, bar)
"""

from tradix.broker.commission import (
    CommissionModel,
    NoCommission,
    FixedCommission,
    PercentCommission,
    KoreaStockCommission,
    USStockCommission,
)
from tradix.broker.slippage import (
    SlippageModel,
    NoSlippage,
    FixedSlippage,
    PercentSlippage,
    VolumeSlippage,
)
from tradix.broker.fill import (
    FillModel,
    CloseFill,
    OpenFill,
    VwapFill,
    RandomFill,
)
from tradix.broker.simulator import BrokerSimulator

__all__ = [
    "CommissionModel",
    "NoCommission",
    "FixedCommission",
    "PercentCommission",
    "KoreaStockCommission",
    "USStockCommission",
    "SlippageModel",
    "NoSlippage",
    "FixedSlippage",
    "PercentSlippage",
    "VolumeSlippage",
    "FillModel",
    "CloseFill",
    "OpenFill",
    "VwapFill",
    "RandomFill",
    "BrokerSimulator",
]
