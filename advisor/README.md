# Strategy Advisor - 전략 추천 시스템

대규모 백테스트 결과를 기반으로 학습된 패턴을 활용하여 시장 상황에 최적화된 전략을 추천하는 시스템입니다.

## 구성 요소

### MarketClassifier
시장 상황을 6가지 레짐으로 분류합니다.

```python
from core.backtest.advisor import MarketClassifier, MarketRegime

classifier = MarketClassifier()
analysis = classifier.analyze(df)

print(analysis.regime)      # MarketRegime.UPTREND
print(analysis.confidence)  # 0.85
print(analysis.summary())
```

**시장 레짐 종류:**
| 레짐 | 설명 | 주요 특징 |
|------|------|----------|
| STRONG_UPTREND | 강한 상승장 | trend > 15%, ADX > 25 |
| UPTREND | 상승장 | trend > 5% |
| SIDEWAYS | 횡보장 | -5% < trend < 5% |
| DOWNTREND | 하락장 | trend < -5% |
| STRONG_DOWNTREND | 강한 하락장 | trend < -15%, ADX > 25 |
| HIGH_VOLATILITY | 고변동성 | volatility > 40% |

### StrategyAdvisor
시장 상황에 맞는 전략을 추천합니다.

```python
from core.backtest.advisor import StrategyAdvisor
from core.backtest.data import FinanceDataReaderFeed

data = FinanceDataReaderFeed('005930', '2020-01-01', '2024-12-31')

advisor = StrategyAdvisor()
result = advisor.recommend(data, runBacktest=True)

print(result.summary())

# 최상위 추천 전략 사용
topStrategy = result.topStrategy().profile.createStrategy()
```

### learnedPatterns
대규모 백테스트(10종목 × 3기간 = 30개 테스트)에서 학습된 패턴입니다.

```python
from core.backtest.advisor import (
    getRecommendedStrategies,
    getAdjustedParams,
    getBenchmark,
    KEY_INSIGHTS,
)

# 시장 상황별 추천 전략
strategies = getRecommendedStrategies(MarketRegime.UPTREND)
# [("SMA Cross", 0.90, "상승장에서 가장 안정적인 성과 (43%)"), ...]

# 시장 상황별 최적 파라미터
params = getAdjustedParams("SMA Cross", MarketRegime.STRONG_UPTREND)
# {"fastPeriod": 5, "slowPeriod": 15}

# 기대 성과 벤치마크
benchmark = getBenchmark(MarketRegime.UPTREND)
# {"expected_sharpe": 0.8, "expected_return": 20.0, "max_acceptable_mdd": -25.0}

# 학습된 인사이트
print(KEY_INSIGHTS)
```

## 학습된 핵심 인사이트

### 시장 상황별 최적 전략

| 시장 상황 | 1순위 전략 | 2순위 전략 | 핵심 포인트 |
|----------|-----------|-----------|------------|
| 강한 상승장 | MACD | SMA Cross | 모멘텀 추종이 핵심 |
| 상승장 | SMA Cross (43%) | Trend Filter | 단순한 골든크로스가 강력 |
| 횡보장 | SMA Cross (60%) | MACD | 작은 추세도 있음 |
| 하락장 | Trend Filter (50%) | Mean Reversion | "안 잃는 것"이 최고 |
| 강한 하락장 | Mean Reversion | Trend Filter | 유일한 양수 샤프 (0.43) |
| 고변동성 | MACD (33%) | SMA Cross | 모멘텀 지표가 강점 |

### 학습 데이터

**테스트 종목 (10개):**
- 한국: 삼성전자, SK하이닉스, 현대차, NAVER, 카카오
- 미국: AAPL, MSFT, GOOGL, AMZN, TSLA

**테스트 기간 (3개):**
1. 2020-2021: COVID 회복기 (강한 상승)
2. 2022: 금리인상 하락장
3. 2023-2024: 회복기 (변동성)

## 실전 적용 가이드

```python
from core.backtest.advisor import StrategyAdvisor, MarketClassifier
from core.backtest.data import FinanceDataReaderFeed
from core.backtest import BacktestEngine

# 1. 데이터 로드
data = FinanceDataReaderFeed('AAPL', '2020-01-01', '2024-12-31')

# 2. 전략 추천 받기
advisor = StrategyAdvisor()
result = advisor.recommend(data, runBacktest=True)

# 3. 추천 결과 확인
print(result.summary())

# 4. 벤치마크 대비 성과 확인
top = result.topStrategy()
print(top.compareWithBenchmark())

# 5. 최적 파라미터로 전략 생성
strategy = top.profile.createStrategy(top.suggestedParams)

# 6. Walk-Forward로 검증 후 실전 적용
from core.backtest import WalkForwardAnalyzer, ParameterSpace

space = ParameterSpace()
space.addInt('fastPeriod', 5, 20)
space.addInt('slowPeriod', 20, 60)

wfa = WalkForwardAnalyzer(data, lambda p: strategy, space)
wfResult = wfa.run()
print(wfResult.summary())
```

## 기대 성과 벤치마크

| 시장 상황 | 기대 샤프 | 기대 수익률 | 최대 허용 MDD |
|----------|----------|------------|--------------|
| 강한 상승장 | 1.5 | +50% | -30% |
| 상승장 | 0.8 | +20% | -25% |
| 횡보장 | 0.4 | +5% | -15% |
| 하락장 | 0.0 | 0% | -20% |
| 강한 하락장 | -0.2 | -10% | -25% |
| 고변동성 | 0.5 | +15% | -35% |

## API Reference

### MarketClassifier

```python
class MarketClassifier:
    def __init__(
        self,
        trendPeriod: int = 50,
        volatilityPeriod: int = 20,
        adxPeriod: int = 14,
        rsiPeriod: int = 14,
    )

    def analyze(self, df: pd.DataFrame) -> MarketAnalysis
    def analyzeHistory(self, df, windowSize=60, stepSize=20) -> pd.DataFrame
```

### StrategyAdvisor

```python
class StrategyAdvisor:
    def __init__(self)
    def registerStrategy(self, profile: StrategyProfile)
    def recommend(self, data, runBacktest=True, initialCash=10_000_000) -> AdvisorResult
    def analyzeAndCompare(self, data, initialCash=10_000_000) -> pd.DataFrame
```

### Helper Functions

```python
def getRecommendedStrategies(regime: MarketRegime) -> List[Tuple[str, float, str]]
def getAdjustedParams(strategyName: str, regime: MarketRegime) -> Dict[str, any]
def getBenchmark(regime: MarketRegime) -> Dict[str, float]
```
