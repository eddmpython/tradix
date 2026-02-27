# Tradix Easy API

**겁나 쉬운 백테스팅** - TradingView보다 쉽게

## 난이도별 사용법

### Level 1: 프리셋 (코딩 0줄)

```python
from tradix.easy import 백테스트, 골든크로스

결과 = 백테스트("삼성전자", 골든크로스())
print(결과.수익률)
```

### Level 2: 선언형 빌더 (3-5줄)

```python
from tradix.easy import 백테스트, 전략, sma, crossover

내전략 = 전략().매수조건(crossover(sma(10), sma(30))).손절(5)
결과 = 백테스트("삼성전자", 내전략)
```

### Level 3: 기존 방식 (고급)

```python
from tradix import BacktestEngine, Strategy
# ... 기존 클래스 상속 방식
```

---

## 프리셋 전략

| 전략 | 영어 | 설명 |
|------|------|------|
| `골든크로스()` | `goldenCross()` | SMA 골든/데드크로스 |
| `RSI과매도()` | `rsiOversold()` | RSI 30/70 반등 |
| `볼린저돌파()` | `bollingerBreakout()` | 볼린저 밴드 돌파 |
| `MACD크로스()` | `macdCross()` | MACD 시그널 크로스 |
| `돌파전략()` | `breakout()` | N일 고/저가 돌파 |
| `평균회귀()` | `meanReversion()` | 평균 회귀 전략 |
| `추세추종()` | `trendFollowing()` | ADX + 추적손절 |

---

## 조건 빌더

```python
from tradix.easy import sma, ema, rsi, macd, price, crossover, crossunder

# 지표
sma(20)           # 20일 단순이동평균
ema(12)           # 12일 지수이동평균
rsi(14)           # RSI
macd()            # MACD
price             # 현재가

# 비교
rsi(14) < 30      # RSI가 30 미만
price > sma(20)   # 가격이 SMA 위

# 크로스
crossover(sma(10), sma(30))   # 골든크로스
crossunder(sma(10), sma(30))  # 데드크로스

# 조합
(rsi(14) < 30) & (price > sma(50))  # AND
(rsi(14) < 30) | (price < sma(20))  # OR
```

---

## 한글 API

```python
from tradix.easy import 백테스트, 최적화, 전략, 골든크로스

# 백테스트
결과 = 백테스트(
    종목="삼성전자",
    전략=골든크로스(),
    기간="3년",
    초기자금="1000만원"
)

# 결과 조회
print(결과.수익률)      # 총 수익률
print(결과.연수익률)    # 연환산 수익률
print(결과.최대낙폭)    # MDD
print(결과.샤프비율)    # 샤프 비율
print(결과.승률)        # 승률
print(결과.거래횟수)    # 거래 횟수
print(결과.요약())      # 전체 요약

# 커스텀 전략
내전략 = (
    전략("내전략")
    .매수조건(crossover(sma(10), sma(30)))
    .매도조건(crossunder(sma(10), sma(30)))
    .손절(5)
    .익절(15)
)
```

---

## 기간 표현

```python
# 상대 기간
백테스트("삼성전자", 골든크로스(), 기간="3년")
백테스트("삼성전자", 골든크로스(), 기간="6개월")
백테스트("삼성전자", 골든크로스(), 기간="100일")

# 절대 기간
백테스트("삼성전자", 골든크로스(), 기간="2020-01-01~2024-12-31")
```

---

## 자금 표현

```python
# 숫자
초기자금=10_000_000

# 한글
초기자금="1000만원"
초기자금="1억"
초기자금="5천만원"
```

---

## 종목 표현

```python
# 코드
백테스트("005930", 골든크로스())

# 한글명
백테스트("삼성전자", 골든크로스())
백테스트("SK하이닉스", 골든크로스())
백테스트("네이버", 골든크로스())
```

**지원 종목**: 삼성전자, SK하이닉스, 현대차, 기아, 네이버, 카카오, LG화학, 삼성SDI 등 30개+

---

## 파라미터 최적화

```python
from tradix.easy import optimize, goldenCross

best = optimize(
    "삼성전자",
    goldenCross,
    fast=(5, 20, 5),    # 시작, 끝, 스텝
    slow=(20, 60, 10)
)

print(f"최적 파라미터: {best['best']['params']}")
print(f"수익률: {best['best']['result'].수익률}%")
```

---

## 빠른 아이디어 테스트

```python
from tradix.easy import quickTest

result = quickTest(
    "삼성전자",
    buyCondition=lambda s, bar: bar.close > s.sma(20),
    sellCondition=lambda s, bar: bar.close < s.sma(20),
    stopLoss=5
)
```

---

## 비교: 기존 vs Easy

| 항목 | 기존 | Easy |
|------|------|------|
| 코드 라인 | 54줄 | **2줄** |
| 클래스 상속 | 필요 | **불필요** |
| None 체크 | 필수 | **자동** |
| 크로스오버 | 수동 구현 | **crossover()** |
| 한글 지원 | 없음 | **완전 지원** |

---

## 파일 구조

```
easy/
├── __init__.py      # 모듈 진입점
├── api.py           # backtest(), optimize() 등
├── quick.py         # QuickStrategy 클래스
├── presets.py       # 프리셋 전략들
├── conditions.py    # sma(), rsi(), crossover() 등
├── korean.py        # 한글 API
├── examples.py      # 예제 코드
└── README.md        # 이 문서
```
