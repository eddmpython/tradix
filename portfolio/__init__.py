"""
Tradex Portfolio Management Package.

Provides portfolio state management for backtesting, including cash and
position tracking, order processing, trade lifecycle management, and
equity history recording.

백테스트를 위한 포트폴리오 상태 관리 패키지입니다. 현금/포지션 추적,
주문 처리, 거래 수명주기 관리, 자산 기록을 제공합니다.

Features:
    - Portfolio: Core cash, position, order, and trade management

Usage:
    from tradex.portfolio import Portfolio

    portfolio = Portfolio(initialCash=10_000_000)
    portfolio.processFill(fill_event)
    print(portfolio.equity)
"""

from tradex.portfolio.portfolio import Portfolio

__all__ = [
    "Portfolio",
]
