"""
Tradix Event Base - Abstract base class for all backtesting events.
Tradix 이벤트 베이스 - 모든 백테스팅 이벤트의 추상 기본 클래스.

This module defines the Event abstract base class and EventType enumeration
that form the foundation of the Tradix event-driven backtesting architecture.
All concrete events (market, signal, order, fill) inherit from Event and
support timestamp-based ordering for priority queue processing.

Features:
    - Abstract base class enforcing consistent event interface
    - UUID-based unique event identification
    - Timestamp-based comparison operators for event ordering
    - EventType enumeration for runtime event classification

Usage:
    >>> from tradix.events.base import Event, EventType
    >>> class CustomEvent(Event):
    ...     eventType: EventType = EventType.MARKET
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class EventType(Enum):
    """Event type enumeration for classifying events in the pipeline. 이벤트 유형 열거형."""
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


@dataclass
class Event(ABC):
    """
    Abstract base class for all events in the Tradix event-driven architecture.
    Tradix 이벤트 기반 아키텍처의 모든 이벤트에 대한 추상 기본 클래스.

    Provides a common interface and timestamp-based ordering for all event
    types in the backtesting pipeline. Inspired by LEAN-style event-driven
    architecture, events are comparable by timestamp for priority queue
    processing.

    LEAN 방식의 이벤트 기반 아키텍처에서 영감을 받아, 모든 이벤트 유형에 대한
    공통 인터페이스와 타임스탬프 기반 정렬을 제공합니다.

    Attributes:
        timestamp (datetime): Event occurrence timestamp. 이벤트 발생 시간.
        eventType (EventType): Event type classifier (set by subclasses, not init).
            이벤트 유형 (하위 클래스에서 설정, init 불가).
        id (str): Unique event identifier (auto-generated 8-char UUID).
            이벤트 고유 ID (자동 생성 8자 UUID).

    Example:
        >>> class MyEvent(Event):
        ...     eventType: EventType = field(default=EventType.MARKET, init=False)
    """
    timestamp: datetime
    eventType: EventType = field(init=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __lt__(self, other: "Event") -> bool:
        """Compare events by timestamp for priority ordering. 타임스탬프 기준 정렬."""
        return self.timestamp < other.timestamp

    def __le__(self, other: "Event") -> bool:
        return self.timestamp <= other.timestamp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, timestamp={self.timestamp})"
