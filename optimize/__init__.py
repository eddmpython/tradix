"""
Tradex Parameter Optimization Package.

Provides strategy parameter optimization tools including grid search,
random search, and flexible parameter space definitions with integer, float,
and categorical support.

전략 파라미터 최적화 도구를 제공하는 패키지입니다. 그리드 서치, 랜덤 서치,
정수/실수/범주형 파라미터 공간 정의를 지원합니다.

Features:
    - ParameterSpace: Define multi-dimensional search spaces
    - Parameter: Individual parameter type definitions
    - Optimizer: Run grid or random search optimizations
    - OptimizeResult: Analyze and export optimization outcomes

Usage:
    from tradex.optimize import Optimizer, ParameterSpace

    space = ParameterSpace()
    space.addInt('fastPeriod', 5, 20, step=1)
    space.addInt('slowPeriod', 20, 60, step=5)

    optimizer = Optimizer(
        data=data,
        strategyFactory=createStrategy,
        parameterSpace=space,
        metric='sharpeRatio',
    )

    result = optimizer.gridSearch()
    print(result.summary())
"""

from tradex.optimize.space import Parameter, ParameterSpace
from tradex.optimize.optimizer import Optimizer, OptimizeResult

__all__ = [
    'Parameter',
    'ParameterSpace',
    'Optimizer',
    'OptimizeResult',
]
