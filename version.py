"""
Tradex Version Management.

Provides semantic versioning, compatibility checking, feature discovery, and
dependency management for the Tradex backtesting library. Tracks the complete
version history and maps each release to its introduced features.

트레이덱스 버전 관리 모듈.
시맨틱 버저닝, 호환성 체크, 기능 탐색, 의존성 관리를 제공합니다.
전체 버전 히스토리를 추적하고 각 릴리스에 도입된 기능을 매핑합니다.

Features:
    - Semantic versioning with major.minor.patch comparison
    - Forward and backward compatibility checking
    - Per-version feature mapping and discovery
    - Dependency requirement tracking per version
    - Migration support with new feature detection

Version History:
    v1.0.0: Core backtest engine, Strategy, Indicators (50+)
    v1.1.0: Optimizer (Grid/Random Search), Walk-Forward Analysis
    v1.2.0: Multi-asset support (MultiAssetEngine, MultiDataFeed)
    v1.3.0: Strategy Advisor system (MarketClassifier, LearnedPatterns)
    v1.4.0: Regime forecasting, Portfolio optimization, Risk simulation
    v2.0.0: Quant extensions (Factor analysis, Statistical arbitrage)

Usage:
    >>> from tradex.version import getVersion, checkVersion, VersionManager
    >>>
    >>> print(getVersion())
    '2.0.0'
    >>> assert checkVersion("1.4.0")
    >>>
    >>> manager = VersionManager()
    >>> features = manager.getAvailableFeatures()
    >>> new = manager.getNewFeatures("1.2.0")
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class VersionLevel(Enum):
    """
    Enumeration of semantic version level identifiers.

    Used to indicate which part of a version number is being referenced
    (major, minor, or patch).

    시맨틱 버전 수준 열거형.
    버전 번호의 어느 부분(메이저, 마이너, 패치)을 참조하는지 나타냅니다.

    Attributes:
        MAJOR: Major version level for breaking changes. 주요 변경을 나타내는 메이저 버전 수준.
        MINOR: Minor version level for new features. 신규 기능을 나타내는 마이너 버전 수준.
        PATCH: Patch version level for bug fixes. 버그 수정을 나타내는 패치 버전 수준.
    """
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class Version:
    """
    Semantic version representation with comparison support.

    Represents a version as three integer components (major, minor, patch)
    and supports equality, less-than, and less-than-or-equal comparisons
    for compatibility checking.

    시맨틱 버전 표현 (비교 연산 지원).
    버전을 세 개의 정수 구성 요소(메이저, 마이너, 패치)로 표현하며,
    호환성 체크를 위한 동등, 미만, 이하 비교를 지원합니다.

    Attributes:
        major (int): Major version number (breaking changes). 메이저 버전 번호 (주요 변경).
        minor (int): Minor version number (new features). 마이너 버전 번호 (신규 기능).
        patch (int): Patch version number (bug fixes). 패치 버전 번호 (버그 수정).

    Example:
        >>> v = Version(2, 0, 0)
        >>> print(v)
        '2.0.0'
        >>> v == Version.fromString("2.0.0")
        True
        >>> Version(1, 4, 0) < Version(2, 0, 0)
        True
    """
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Return the version as a 'major.minor.patch' string. 'major.minor.patch' 형식 문자열 반환."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: 'Version') -> bool:
        """Check equality by comparing all three version components. 세 버전 구성 요소를 비교하여 동등성 확인."""
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: 'Version') -> bool:
        """Check if this version is strictly less than another. 이 버전이 다른 버전보다 엄격히 낮은지 확인."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: 'Version') -> bool:
        """Check if this version is less than or equal to another. 이 버전이 다른 버전 이하인지 확인."""
        return self == other or self < other

    @classmethod
    def fromString(cls, versionStr: str) -> 'Version':
        """Parse a version string into a Version instance. 버전 문자열을 Version 인스턴스로 파싱.

        Accepts formats like '2.0.0', 'v2.0.0', '2.0', or '2'. Missing
        minor or patch components default to 0.

        Args:
            versionStr: Version string to parse (e.g., "2.0.0" or "v1.4.0").
                파싱할 버전 문자열 (예: "2.0.0" 또는 "v1.4.0").

        Returns:
            Parsed Version instance. 파싱된 Version 인스턴스.

        Example:
            >>> Version.fromString("v1.4.0")
            Version(major=1, minor=4, patch=0)
        """
        parts = versionStr.replace('v', '').split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


# 현재 버전
CURRENT_VERSION = Version(1, 2, 0)

VERSION_FEATURES: Dict[str, List[str]] = {
    "1.0.0": [
        "BacktestEngine",
        "Strategy",
        "Indicators (50+)",
        "Portfolio",
        "Order/Position/Trade",
    ],
    "1.1.0": [
        "Optimizer (Grid/Random Search)",
        "ParameterSpace",
        "WalkForwardAnalyzer",
        "MultiAssetEngine",
        "MultiDataFeed",
        "StrategyAdvisor",
        "MarketClassifier",
        "FactorAnalyzer",
        "StatArbStrategy",
        "StrategyDNA (12-dimensional fingerprinting)",
        "BlackSwanAnalyzer (extreme event resilience)",
        "StrategyHealthAnalyzer (overfitting diagnostics)",
        "WhatIfSimulator (sensitivity analysis)",
        "DrawdownSimulator (worst-case scenarios)",
        "SeasonalityAnalyzer (monthly/weekday patterns)",
        "CorrelationAnalyzer (multi-strategy correlation)",
        "TradingJournal (automatic trade diary)",
        "StrategyLeaderboard (ranking system)",
        "33 preset strategies",
        "TradingView-style TUI (3 styles)",
        "12 terminal chart types (Plotext)",
        "Typer CLI (backtest/optimize/chart/compare)",
    ],
    "1.2.0": [
        "MonteCarloStressAnalyzer (10K path stress test)",
        "FractalAnalyzer (Hurst exponent / fractal dimension)",
        "RegimeDetector (GMM-based regime detection)",
        "InformationTheoryAnalyzer (entropy / mutual information)",
        "PortfolioStressAnalyzer (6 crisis scenarios)",
    ],
}

# 버전별 최소 요구 의존성
VERSION_DEPENDENCIES: Dict[str, Dict[str, str]] = {
    "1.0.0": {
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
    },
    "1.1.0": {
        "scipy": ">=1.7.0",
        "rich": ">=13.0",
        "plotext": ">=5.3",
        "typer": ">=0.12",
    },
}


@dataclass
class CompatibilityCheck:
    """
    Result of a version compatibility check.

    Contains the compatibility verdict along with details about the current
    and required versions, any missing features, and relevant warnings.

    버전 호환성 체크 결과.
    현재 버전과 요구 버전 간의 호환 여부, 누락된 기능, 관련 경고를 포함합니다.

    Attributes:
        compatible (bool): True if current version meets the requirement.
            현재 버전이 요구 조건을 충족하면 True.
        currentVersion (Version): The installed Tradex version.
            설치된 Tradex 버전.
        requiredVersion (Version): The version required by the caller.
            호출자가 요구하는 버전.
        missingFeatures (List[str]): Features unavailable in current version.
            현재 버전에서 사용 불가능한 기능 목록.
        warnings (List[str]): Non-critical compatibility warnings.
            비치명적 호환성 경고 목록.

    Example:
        >>> manager = VersionManager()
        >>> check = manager.checkCompatibility("1.4.0")
        >>> if check.compatible:
        ...     print("All good!")
        >>> else:
        ...     print(check.summary())
    """
    compatible: bool
    currentVersion: Version
    requiredVersion: Version
    missingFeatures: List[str]
    warnings: List[str]

    def summary(self) -> str:
        """Generate a human-readable compatibility check report. 사람이 읽을 수 있는 호환성 체크 보고서 생성.

        Returns:
            Formatted string with compatibility status, versions, missing features,
            and warnings.
            호환성 상태, 버전, 누락 기능, 경고가 포함된 포맷 문자열.
        """
        status = "✓ 호환" if self.compatible else "✗ 비호환"
        lines = [
            f"=== 호환성 체크 ===",
            f"상태: {status}",
            f"현재 버전: v{self.currentVersion}",
            f"요구 버전: v{self.requiredVersion}",
        ]

        if self.missingFeatures:
            lines.append(f"\n누락된 기능:")
            for feat in self.missingFeatures:
                lines.append(f"  - {feat}")

        if self.warnings:
            lines.append(f"\n경고:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        return "\n".join(lines)


class VersionManager:
    """
    Central version manager for compatibility checking, feature discovery, and migration support.

    Provides methods to verify version compatibility, list available and new
    features, retrieve version history, and determine required dependencies
    for the current Tradex installation.

    버전 호환성 체크, 기능 탐색, 마이그레이션 지원을 위한 중앙 버전 관리자.
    현재 Tradex 설치의 버전 호환성 확인, 사용 가능한 기능 및 신규 기능 조회,
    버전 히스토리 검색, 필수 의존성 확인 메서드를 제공합니다.

    Attributes:
        currentVersion (Version): The current installed Tradex version.
            현재 설치된 Tradex 버전.

    Example:
        >>> manager = VersionManager()
        >>> check = manager.checkCompatibility("1.3.0")
        >>> print(check.summary())
        >>> features = manager.getAvailableFeatures()
        >>> history = manager.getVersionHistory()
    """

    def __init__(self):
        self.currentVersion = CURRENT_VERSION

    def checkCompatibility(
        self,
        requiredVersionStr: str
    ) -> CompatibilityCheck:
        """Check compatibility between the current version and a required version. 현재 버전과 요구 버전 간 호환성 체크.

        Determines whether the current Tradex version satisfies the given
        requirement and identifies any missing features or potential issues.

        Args:
            requiredVersionStr: Required version string (e.g., "1.4.0").
                요구 버전 문자열 (예: "1.4.0").

        Returns:
            CompatibilityCheck with compatibility verdict, missing features, and warnings.
            호환 여부, 누락 기능, 경고가 담긴 CompatibilityCheck.
        """
        requiredVersion = Version.fromString(requiredVersionStr)

        compatible = self.currentVersion >= requiredVersion
        missingFeatures = []
        warnings = []

        if not compatible:
            for verStr, features in VERSION_FEATURES.items():
                ver = Version.fromString(verStr)
                if ver > self.currentVersion and ver <= requiredVersion:
                    missingFeatures.extend(features)

        if self.currentVersion.major > requiredVersion.major:
            warnings.append(
                f"메이저 버전 차이 - 일부 API가 변경되었을 수 있음"
            )

        return CompatibilityCheck(
            compatible=compatible,
            currentVersion=self.currentVersion,
            requiredVersion=requiredVersion,
            missingFeatures=missingFeatures,
            warnings=warnings,
        )

    def getAvailableFeatures(self) -> List[str]:
        """Get all features available in the current version. 현재 버전에서 사용 가능한 모든 기능 조회.

        Returns:
            List of feature names from all versions up to and including the current.
            현재 버전까지의 모든 버전에서 제공하는 기능 이름 목록.
        """
        features = []
        for verStr, feats in VERSION_FEATURES.items():
            ver = Version.fromString(verStr)
            if ver <= self.currentVersion:
                features.extend(feats)
        return features

    def getVersionHistory(self) -> List[Tuple[str, List[str]]]:
        """Get the complete version history with features per release. 릴리스별 기능이 포함된 전체 버전 히스토리 조회.

        Returns:
            List of (version_string, feature_list) tuples in chronological order.
            (버전 문자열, 기능 목록) 튜플의 시간순 리스트.
        """
        return [
            (ver, features)
            for ver, features in VERSION_FEATURES.items()
        ]

    def getNewFeatures(self, fromVersionStr: str) -> List[str]:
        """Get features added after a specific version. 특정 버전 이후 추가된 기능 조회.

        Args:
            fromVersionStr: Base version string to compare from (e.g., "1.2.0").
                비교 기준 버전 문자열 (예: "1.2.0").

        Returns:
            List of feature names introduced between the given version and the
            current version (exclusive of the given, inclusive of the current).
            주어진 버전과 현재 버전 사이에 도입된 기능 이름 목록.
        """
        fromVersion = Version.fromString(fromVersionStr)
        newFeatures = []

        for verStr, features in VERSION_FEATURES.items():
            ver = Version.fromString(verStr)
            if ver > fromVersion and ver <= self.currentVersion:
                newFeatures.extend(features)

        return newFeatures

    def getRequiredDependencies(self) -> Dict[str, str]:
        """Get all required dependencies for the current version. 현재 버전에 필요한 모든 의존성 조회.

        Returns:
            Dictionary mapping package names to version requirement strings
            (e.g., {'pandas': '>=1.3.0', 'scipy': '>=1.7.0'}).
            패키지 이름과 버전 요구 문자열의 딕셔너리.
        """
        deps = {}
        for verStr, verDeps in VERSION_DEPENDENCIES.items():
            ver = Version.fromString(verStr)
            if ver <= self.currentVersion:
                deps.update(verDeps)
        return deps


def getVersion() -> str:
    """Return the current Tradex version as a string. 현재 Tradex 버전을 문자열로 반환.

    Returns:
        Version string in 'major.minor.patch' format (e.g., '2.0.0').
        'major.minor.patch' 형식의 버전 문자열 (예: '2.0.0').
    """
    return str(CURRENT_VERSION)


def checkVersion(required: str) -> bool:
    """Check if the current Tradex version meets a version requirement. 현재 Tradex 버전이 요구 버전을 충족하는지 확인.

    Args:
        required: Required version string (e.g., "1.4.0").
            요구 버전 문자열 (예: "1.4.0").

    Returns:
        True if the current version is greater than or equal to the required version.
        현재 버전이 요구 버전 이상이면 True.
    """
    manager = VersionManager()
    check = manager.checkCompatibility(required)
    return check.compatible
