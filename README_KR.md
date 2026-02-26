<p align="center">
  <h1 align="center">Tradex</h1>
  <p align="center">
    <strong>퀀트 트레이딩을 위한 초고속 백테스팅 엔진</strong>
  </p>
  <p align="center">
    <a href="#설치">설치</a> &middot;
    <a href="#빠른-시작">빠른 시작</a> &middot;
    <a href="#주요-기능">주요 기능</a> &middot;
    <a href="#api-레퍼런스">API 레퍼런스</a> &middot;
    <a href="README.md">English</a>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/tradex-backtest/"><img src="https://img.shields.io/pypi/v/tradex-backtest?style=flat-square&color=blue" alt="PyPI"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"></a>
    <a href="https://www.buymeacoffee.com/chani"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-후원하기-orange?style=flat-square&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee"></a>
  </p>
</p>

---

Tradex는 속도와 간결함을 위해 밑바닥부터 설계된 고성능 백테스팅 라이브러리입니다. NumPy 벡터화 연산 기반으로, **1,000개 파라미터 최적화를 0.02초**에 처리합니다. Numba도, C 확장도 없이 순수 Python으로.

```python
from tradex import 백테스트, 골든크로스

결과 = 백테스트("삼성전자", 골든크로스())
print(결과.요약())
```

2줄이면 충분합니다. 수수료, 슬리피지, 성과 지표까지 전부 포함.

## 왜 Tradex인가?

| 라이브러리 | 문제점 |
|-----------|--------|
| **VectorBT** | 무료 버전 개발 중단, Pro는 월 $29 유료 |
| **Backtesting.py** | AGPL 라이선스, 포트폴리오 지원 부족 |
| **Lumibot** | 무겁고 커스터마이징이 어려움 |
| **LEAN** | C# 기반, Docker 필요, 학습 곡선 높음 |
| **bt** | 포지션 사이징 없음, 거래비용 미반영 |

Tradex는 이 모든 기능을 **무료, MIT 라이선스**로 제공합니다. 한국 주식 시장에 특화된 설계는 덤입니다.

## 주요 기능

- **벡터화 엔진** — NumPy 기반, 이벤트 루프 대비 100배 빠름
- **50+ 기술적 지표** — SMA, EMA, RSI, MACD, 볼린저, ATR, 일목균형표, 슈퍼트렌드 등
- **선언형 전략 빌더** — 메서드 체이닝으로 전략 구성, 클래스 상속 불필요
- **Walk-Forward 분석** — 시계열 교차 검증으로 과적합 방지
- **파라미터 최적화** — Grid/Random 탐색, 모든 지표 기준 최적화
- **멀티 자산 포트폴리오** — 다중 종목 리밸런싱 백테스트
- **현실적 시뮬레이션** — 수수료, 슬리피지, 체결 로직, 포지션 사이징
- **리스크 분석** — VaR, CVaR, 몬테카를로, Sharpe, Sortino, Calmar
- **팩터 분석** — 멀티팩터 모델, 통계적 차익거래
- **한국 시장 네이티브** — 거래세(0.18%), 증권사 수수료, KRX 종목 매핑 기본 내장
- **한글 API** — `백테스트("삼성전자", 골든크로스())` 완전한 한글 함수명 지원

## 설치

### uv 사용 (권장)

[uv](https://docs.astral.sh/uv/)는 가장 빠른 Python 패키지 매니저입니다. 아직 없다면:

```bash
# uv 설치
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Tradex 설치:

```bash
# 새 프로젝트 생성 후 tradex 추가
uv init my-backtest && cd my-backtest
uv add tradex-backtest

# 전체 기능 포함 (scipy, plotly, statsmodels, scikit-learn)
uv add "tradex-backtest[full]"
```

### pip 사용

```bash
pip install tradex-backtest

# 전체 기능 포함
pip install "tradex-backtest[full]"
```

### 소스에서 설치 (개발용)

```bash
git clone https://github.com/chani/tradex.git
cd tradex

# uv 사용
uv sync --dev

# pip 사용
pip install -e ".[dev]"
```

**요구사항:** Python 3.9+, NumPy, Pandas

## 빠른 시작

### 1. 2줄 백테스트

프리셋 전략으로 가장 간단하게 백테스트 실행:

```python
from tradex import 백테스트, 골든크로스

결과 = 백테스트("삼성전자", 골든크로스())
print(결과.요약())
```

출력:
```
=== 백테스트 결과 ===
전략: GoldenCross
기간: 2020-01-02 ~ 2024-12-30
초기 자금: 10,000,000원
최종 자금: 14,230,000원
수익률: +42.30%
샤프비율: 1.23
최대낙폭: -12.45%
거래 수: 18
승률: 61.1%
```

영어 API도 동일하게 동작합니다:

```python
from tradex import backtest, goldenCross

result = backtest("005930", goldenCross())
print(result.summary())
```

### 2. 벡터화 모드 (100배 빠름)

전체 가격 히스토리를 한 번에 계산:

```python
from tradex import vbacktest

result = vbacktest("005930", "goldenCross", fast=10, slow=30)
print(f"수익률: {result.totalReturn:+.2f}%")
print(f"샤프: {result.sharpeRatio:.2f}")
print(f"최대낙폭: {result.maxDrawdown:.2f}%")
```

### 3. 선언형 전략 빌더

보일러플레이트 없이 메서드 체이닝으로 전략 구성:

```python
from tradex import QuickStrategy, backtest, sma, rsi, crossover, crossunder

strategy = (
    QuickStrategy("모멘텀RSI")
    .buyWhen(crossover(sma(10), sma(30)))
    .buyWhen(rsi(14) < 30)
    .sellWhen(crossunder(sma(10), sma(30)))
    .sellWhen(rsi(14) > 70)
    .stopLoss(5)
    .takeProfit(15)
)

result = backtest("005930", strategy)
print(result.summary())
```

### 4. 클래스 기반 전략 (전체 제어)

최대 유연성을 위한 전략 클래스 직접 작성:

```python
from tradex import Strategy, Bar, BacktestEngine
from tradex.datafeed import FinanceDataReaderFeed

class 듀얼모멘텀(Strategy):
    def initialize(self):
        self.fastPeriod = 10
        self.slowPeriod = 30

    def onBar(self, bar: Bar):
        fast = self.sma(self.fastPeriod)
        slow = self.sma(self.slowPeriod)

        if fast is None or slow is None:
            return

        if fast > slow and not self.hasPosition(bar.symbol):
            self.buy(bar.symbol)
        elif fast < slow and self.hasPosition(bar.symbol):
            self.closePosition(bar.symbol)

data = FinanceDataReaderFeed("005930", "2020-01-01", "2024-12-31")
engine = BacktestEngine(data, 듀얼모멘텀(), initialCash=10_000_000)
result = engine.run()
print(result.summary())
```

### 5. 파라미터 최적화

몇 초 만에 최적 파라미터 탐색:

```python
from tradex import voptimize

best = voptimize(
    "005930",
    "goldenCross",
    fast=(5, 20, 5),     # (시작, 끝, 스텝)
    slow=(20, 60, 10),
    metric="sharpeRatio"
)

print(f"최적: fast={best['best']['params']['fast']}, slow={best['best']['params']['slow']}")
print(f"샤프: {best['best']['metric']:.2f}")
```

### 6. Walk-Forward 분석

표본 외 검증으로 과적합 방지:

```python
from tradex import WalkForwardAnalyzer, ParameterSpace

space = ParameterSpace()
space.addInt("fast", 5, 20, step=5)
space.addInt("slow", 20, 60, step=10)

wfa = WalkForwardAnalyzer(
    data=data,
    strategyFactory=createStrategy,
    parameterSpace=space,
    inSampleMonths=12,
    outOfSampleMonths=3,
)

result = wfa.run()
print(f"견고성: {result.robustnessRatio:.1%}")
```

### 7. 멀티 자산 포트폴리오

```python
from tradex import MultiAssetEngine, MultiAssetStrategy

class 균등배분(MultiAssetStrategy):
    def onBars(self, bars):
        self.rebalance({
            "005930": 0.4,  # 삼성전자
            "000660": 0.3,  # SK하이닉스
            "035420": 0.3,  # 네이버
        })

engine = MultiAssetEngine(strategy=균등배분())
result = engine.run()
print(result.summary())
```

### 8. 리스크 시뮬레이션

```python
from tradex.risk import RiskSimulator, VaRMethod

simulator = RiskSimulator()
simulator.fit(returns)

var = simulator.calcVaR(confidence=0.95, method=VaRMethod.HISTORICAL)
print(f"95% VaR: {var.var:.2%}")
print(f"CVaR: {var.cvar:.2%}")

mc = simulator.monteCarloSimulation(horizon=252, nSim=10000)
```

## 프리셋 전략

| 전략 (영어 / 한글) | 설명 | 주요 파라미터 |
|-------------------|------|-------------|
| `goldenCross()` / `골든크로스()` | SMA 교차 | `fast`, `slow`, `stopLoss`, `takeProfit` |
| `rsiOversold()` / `RSI과매도()` | RSI 반전 | `period`, `oversold`, `overbought` |
| `bollingerBreakout()` / `볼린저돌파()` | 볼린저 밴드 돌파 | `period`, `std` |
| `macdCross()` / `MACD크로스()` | MACD 히스토그램 교차 | `fast`, `slow`, `signal` |
| `breakout()` / `돌파전략()` | 채널 돌파 (터틀) | `period` |
| `meanReversion()` / `평균회귀()` | 평균 회귀 | `period`, `threshold` |
| `trendFollowing()` / `추세추종()` | 추세 추종 + ADX 필터 | `fastPeriod`, `slowPeriod`, `adxThreshold` |

## 기술적 지표 (50+)

| 카테고리 | 지표 |
|---------|------|
| **이동평균** | `sma` `ema` `wma` `hma` `tema` `dema` `vwma` `alma` |
| **모멘텀** | `rsi` `macd` `stochastic` `roc` `momentum` `cci` `williamsR` `cmo` |
| **변동성** | `atr` `bollinger` `keltner` `donchian` |
| **거래량** | `obv` `vwap` `mfi` `adl` `chaikin` `emv` `forceIndex` `nvi` `pvi` |
| **추세** | `adx` `supertrend` `psar` `ichimoku` `trix` `dpo` |
| **기타** | `pivotPoints` `fibonacciRetracement` `zigzag` `elderRay` `ulcer` |

## 성능 벤치마크

10년 일봉 데이터(2,458개 바) 기준:

| 연산 | 소요 시간 |
|------|----------|
| SMA 계산 | **0.006ms** |
| RSI 계산 | **0.009ms** |
| MACD 계산 | **0.040ms** |
| 단일 백테스트 | **0.132ms** |
| 1,000개 파라미터 최적화 | **0.02초** |

## API 레퍼런스

### 핵심 함수

| 클래스/함수 | 설명 |
|------------|------|
| `backtest(symbol, strategy)` / `백테스트()` | 프리셋/커스텀 전략으로 백테스트 실행 |
| `vbacktest(symbol, strategy, **params)` | 벡터화 백테스트 실행 |
| `voptimize(symbol, strategy, **ranges)` | Grid Search 파라미터 최적화 |
| `BacktestEngine(data, strategy)` | 이벤트 드리븐 백테스트 엔진 |
| `VectorizedEngine(initialCash)` | 벡터화 백테스트 엔진 |
| `QuickStrategy(name)` / `전략()` | 선언형 전략 빌더 |

### 전략 빌더 메서드

| 메서드 | 설명 |
|--------|------|
| `.buyWhen(condition)` | 매수 조건 추가 |
| `.sellWhen(condition)` | 매도 조건 추가 |
| `.stopLoss(pct)` | 손절 퍼센트 설정 |
| `.takeProfit(pct)` | 익절 퍼센트 설정 |
| `.trailingStop(pct)` | 트레일링 스탑 퍼센트 설정 |

### 조건 빌더

| 함수 | 반환 |
|------|------|
| `sma(period)` | SMA 인디케이터 |
| `ema(period)` | EMA 인디케이터 |
| `rsi(period)` | RSI 인디케이터 |
| `macd(fast, slow, signal)` | MACD 인디케이터 |
| `atr(period)` | ATR 인디케이터 |
| `crossover(fast, slow)` | 상향 교차 조건 |
| `crossunder(fast, slow)` | 하향 교차 조건 |

비교 연산자 지원: `sma(10) > sma(30)`, `rsi(14) < 30`

## 아키텍처

```
tradex/
├── engine.py              # 핵심 백테스트 엔진
├── multiAssetEngine.py    # 멀티 자산 포트폴리오 엔진
├── strategy/              # 전략 베이스 + 50+ 지표 + 앙상블 결합기
├── easy/                  # 2줄 API, 프리셋, 한글 API
├── vectorized/            # 벡터화 엔진 + 지표 (순수 NumPy)
├── datafeed/              # 데이터 피드 (FinanceDataReader + Parquet 캐시)
├── broker/                # 수수료, 슬리피지, 체결, 실행 시뮬레이션
├── risk/                  # 포지션 사이징, VaR, 몬테카를로
├── optimize/              # Grid / Random 탐색 최적화
├── walkforward/           # Walk-Forward 분석
├── analytics/             # 성과 지표, 차트, 보고서 생성
├── portfolio/             # 포트폴리오 추적 + 최적화
├── quant/                 # 팩터 분석, 통계적 차익거래
├── signals/               # 시그널 예측 + 적응형 시그널
├── advisor/               # 시장 레짐 + 전략 추천 시스템
├── entities/              # Bar, Order, Position, Trade
├── events/                # 이벤트 시스템 (Market, Signal, Order, Fill)
└── tests/                 # 유닛 + 통합 테스트 (68개)
```

## 테스트 실행

```bash
# uv 사용
uv run pytest

# pip 사용
pytest
```

## 기여하기

기여는 언제나 환영합니다! Issue와 Pull Request를 자유롭게 올려주세요.

```bash
git clone https://github.com/chani/tradex.git
cd tradex

# uv 사용 (권장)
uv sync --dev
uv run pytest

# pip 사용
pip install -e ".[dev]"
pytest
```

## 후원하기

Tradex가 트레이딩 리서치에 도움이 되셨다면, 프로젝트를 후원해 주세요:

<a href="https://www.buymeacoffee.com/chani">
  <img src="https://img.buymeacoffee.com/button-api/?text=커피 한 잔 후원하기&emoji=&slug=chani&button_colour=FF5F5F&font_colour=ffffff&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" />
</a>

여러분의 후원이 이 프로젝트를 무료로 유지하고 새로운 기능을 개발하는 원동력이 됩니다. 커피 한 잔이 새로운 기능이 됩니다.

## 라이선스

MIT License. 개인 및 상업 프로젝트에서 자유롭게 사용하세요.
