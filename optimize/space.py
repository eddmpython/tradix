"""
Tradix Parameter Space Definition Module.

Defines the search space for strategy parameter optimization, supporting
integer, float, and categorical parameter types. Provides grid enumeration
and random sampling capabilities.

전략 파라미터 최적화를 위한 탐색 공간 정의 모듈입니다.
정수, 실수, 범주형 파라미터 타입을 지원하며, 그리드 열거와 랜덤 샘플링 기능을
제공합니다.

Features:
    - Integer, float, and categorical parameter definitions
    - Grid combination generation via Cartesian product
    - Random sampling with optional seed
    - Fluent API for chaining parameter additions
    - Summary statistics and combination counting

Usage:
    from tradix.optimize.space import ParameterSpace

    space = ParameterSpace()
    space.addInt('fastPeriod', 5, 20, step=1)
    space.addFloat('threshold', 0.01, 0.10, step=0.01)
    space.addChoice('maType', ['sma', 'ema', 'hma'])

    combinations = space.gridCombinations()
    samples = space.randomSample(100, seed=42)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import itertools
import random
import numpy as np


@dataclass
class Parameter:
    """
    Single parameter definition for the optimization search space.

    Represents one dimension of the parameter space with type-specific
    configuration for grid generation and random sampling.

    최적화 탐색 공간의 단일 파라미터 정의입니다. 그리드 생성과 랜덤 샘플링을 위한
    타입별 설정을 포함합니다.

    Attributes:
        name (str): Parameter name (파라미터 이름).
        paramType (str): Type - 'int', 'float', or 'choice'
            (타입: 정수, 실수, 선택).
        low (Optional[float]): Minimum value for int/float types (최소값).
        high (Optional[float]): Maximum value for int/float types (최대값).
        step (Optional[float]): Step size for grid generation (스텝 크기).
        choices (List[Any]): List of choices for 'choice' type (선택지 목록).

    Example:
        >>> param = Parameter(name='period', paramType='int', low=5, high=20, step=1)
        >>> param.grid()
        [5, 6, 7, ..., 20]
    """
    name: str
    paramType: str
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.paramType == 'choice' and not self.choices:
            raise ValueError(f"Parameter '{self.name}': choice 타입은 choices가 필요합니다")
        if self.paramType in ('int', 'float') and (self.low is None or self.high is None):
            raise ValueError(f"Parameter '{self.name}': {self.paramType} 타입은 low, high가 필요합니다")

    def grid(self) -> List[Any]:
        """
        Generate all grid values for this parameter.

        이 파라미터의 모든 그리드 값을 생성합니다.

        Returns:
            List[Any]: Enumerated values based on parameter type and step.
        """
        if self.paramType == 'choice':
            return self.choices.copy()

        elif self.paramType == 'int':
            step = int(self.step) if self.step else 1
            return list(range(int(self.low), int(self.high) + 1, step))

        else:
            step = self.step if self.step else 0.1
            values = np.arange(self.low, self.high + step * 0.5, step)
            return [round(v, 6) for v in values]

    def random(self) -> Any:
        """
        Generate a single random value for this parameter.

        이 파라미터의 랜덤 값을 하나 생성합니다.

        Returns:
            Any: Random value within the parameter's range or choices.
        """
        if self.paramType == 'choice':
            return random.choice(self.choices)

        elif self.paramType == 'int':
            return random.randint(int(self.low), int(self.high))

        else:
            return random.uniform(self.low, self.high)

    @property
    def gridSize(self) -> int:
        """
        Return the number of discrete grid values for this parameter.

        이 파라미터의 그리드 값 개수를 반환합니다.
        """
        return len(self.grid())

    def __repr__(self) -> str:
        if self.paramType == 'choice':
            return f"Parameter({self.name}, choices={self.choices})"
        return f"Parameter({self.name}, {self.paramType}, {self.low}~{self.high}, step={self.step})"


class ParameterSpace:
    """
    Multi-dimensional parameter search space for strategy optimization.

    Manages a collection of Parameter definitions and provides methods for
    generating all grid combinations (Cartesian product) or random samples
    from the defined space. Supports a fluent API for chaining additions.

    전략 최적화를 위한 다차원 파라미터 탐색 공간입니다. 파라미터 정의 모음을 관리하며,
    전체 그리드 조합(카테시안 곱) 생성 또는 랜덤 샘플링 메서드를 제공합니다.

    Example:
        >>> space = ParameterSpace()
        >>> space.addInt('fastPeriod', 5, 20, step=1)
        >>> space.addFloat('threshold', 20.0, 40.0, step=5.0)
        >>> space.addChoice('maType', ['sma', 'ema', 'hma'])
        >>> print(f"Total combinations: {space.totalCombinations}")
        >>> for params in space.gridCombinations():
        ...     print(params)
    """

    def __init__(self):
        self._params: Dict[str, Parameter] = {}

    def add(
        self,
        name: str,
        paramType: str,
        low: float = None,
        high: float = None,
        step: float = None,
        choices: List[Any] = None,
    ) -> 'ParameterSpace':
        """
        Add a parameter to the search space.

        파라미터를 탐색 공간에 추가합니다.

        Args:
            name: Parameter name, used as key in result dicts (파라미터 이름).
            paramType: Type string - 'int', 'float', or 'choice' (타입).
            low: Minimum value for int/float types (최소값).
            high: Maximum value for int/float types (최대값).
            step: Step size for grid generation on int/float types (스텝).
            choices: List of valid values for 'choice' type (선택지).

        Returns:
            ParameterSpace: Self reference for method chaining (체이닝용 self).
        """
        self._params[name] = Parameter(
            name=name,
            paramType=paramType,
            low=low,
            high=high,
            step=step,
            choices=choices or [],
        )
        return self

    def addInt(self, name: str, low: int, high: int, step: int = 1) -> 'ParameterSpace':
        """
        Add an integer parameter (convenience shortcut).

        정수 파라미터를 추가합니다 (단축 메서드).

        Args:
            name: Parameter name (파라미터 이름).
            low: Minimum integer value (최소값).
            high: Maximum integer value (최대값).
            step: Integer step size, default 1 (스텝).

        Returns:
            ParameterSpace: Self reference for method chaining.
        """
        return self.add(name, 'int', low, high, step)

    def addFloat(self, name: str, low: float, high: float, step: float = 0.1) -> 'ParameterSpace':
        """
        Add a float parameter (convenience shortcut).

        실수 파라미터를 추가합니다 (단축 메서드).

        Args:
            name: Parameter name (파라미터 이름).
            low: Minimum float value (최소값).
            high: Maximum float value (최대값).
            step: Float step size, default 0.1 (스텝).

        Returns:
            ParameterSpace: Self reference for method chaining.
        """
        return self.add(name, 'float', low, high, step)

    def addChoice(self, name: str, choices: List[Any]) -> 'ParameterSpace':
        """
        Add a categorical choice parameter (convenience shortcut).

        범주형 선택 파라미터를 추가합니다 (단축 메서드).

        Args:
            name: Parameter name (파라미터 이름).
            choices: List of valid categorical values (선택지 목록).

        Returns:
            ParameterSpace: Self reference for method chaining.
        """
        return self.add(name, 'choice', choices=choices)

    def get(self, name: str) -> Optional[Parameter]:
        """
        Retrieve a parameter definition by name.

        이름으로 파라미터 정의를 조회합니다.

        Args:
            name: Parameter name to look up.

        Returns:
            Optional[Parameter]: Parameter object if found, None otherwise.
        """
        return self._params.get(name)

    def remove(self, name: str) -> bool:
        """
        Remove a parameter from the search space.

        탐색 공간에서 파라미터를 제거합니다.

        Args:
            name: Parameter name to remove.

        Returns:
            bool: True if removed, False if not found.
        """
        if name in self._params:
            del self._params[name]
            return True
        return False

    def gridCombinations(self) -> List[Dict[str, Any]]:
        """
        Generate all grid combinations via Cartesian product.

        카테시안 곱을 통해 모든 그리드 조합을 생성합니다.

        Returns:
            List[Dict[str, Any]]: List of parameter dicts, one per combination.
                e.g., [{'param1': val1, 'param2': val2}, ...].
        """
        if not self._params:
            return [{}]

        grids = {name: param.grid() for name, param in self._params.items()}
        keys = list(grids.keys())
        values = [grids[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def randomSample(self, n: int, seed: int = None) -> List[Dict[str, Any]]:
        """
        Generate random parameter samples from the search space.

        탐색 공간에서 랜덤 파라미터 샘플을 생성합니다.

        Args:
            n: Number of random samples to generate (샘플 수).
            seed: Optional random seed for reproducibility (랜덤 시드).

        Returns:
            List[Dict[str, Any]]: List of randomly sampled parameter dicts.
        """
        if seed is not None:
            random.seed(seed)

        samples = []
        for _ in range(n):
            sample = {name: param.random() for name, param in self._params.items()}
            samples.append(sample)

        return samples

    @property
    def params(self) -> Dict[str, Parameter]:
        """
        Return a copy of the parameter definitions dictionary.

        파라미터 정의 딕셔너리의 복사본을 반환합니다.
        """
        return self._params.copy()

    @property
    def names(self) -> List[str]:
        """
        Return a list of all parameter names in the space.

        탐색 공간의 모든 파라미터 이름 목록을 반환합니다.
        """
        return list(self._params.keys())

    @property
    def totalCombinations(self) -> int:
        """
        Calculate the total number of grid combinations.

        전체 그리드 조합 수를 계산합니다.
        """
        if not self._params:
            return 0

        total = 1
        for param in self._params.values():
            total *= param.gridSize
        return total

    def summary(self) -> str:
        """
        Generate a human-readable summary of the parameter space.

        파라미터 공간의 요약 문자열을 생성합니다.

        Returns:
            str: Multi-line summary with parameter details and total combinations.
        """
        lines = [
            f"ParameterSpace ({len(self._params)} parameters)",
            f"Total combinations: {self.totalCombinations:,}",
            "-" * 40,
        ]

        for name, param in self._params.items():
            if param.paramType == 'choice':
                lines.append(f"  {name}: {param.choices} ({len(param.choices)} choices)")
            else:
                lines.append(
                    f"  {name}: {param.low} ~ {param.high} "
                    f"(step={param.step}, {param.gridSize} values)"
                )

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __repr__(self) -> str:
        return f"ParameterSpace({list(self._params.keys())})"
