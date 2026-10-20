# Tradex TUI 세계적 수준 리뉴얼 기획서

## 전문가 토론 결과 종합

3명의 전문 에이전트(보고서, 차트, 대시보드)가 세계적 금융 터미널 UI를 분석한 결과를 종합.

---

## 1. 보고서 (console.py) - 3종 스타일

### Style A: Bloomberg Terminal (정보 밀도 극대화)
- 더블 보더 프레임, 4분할 메트릭 레이아웃
- amber/cyan 색상, 인라인 스파크라인
- 월간 수익률 히트맵, 전체 메트릭 테이블

### Style B: TradingView Modern (기본값, 현재 확장)
- 라운드 보더, 4개 메트릭 카드 + 스파크라인
- 에쿼티 차트 + 드로다운 결합, PnL 분포 히스토그램
- 넉넉한 패딩, 시각적 계층구조

### Style C: Hedge Fund Minimal (복사/붙여넣기 최적화)
- 보더 없음, Rule() 라인 구분
- 2열 메트릭 그리드, 품질 지표 (^/v)
- 인라인 스파크라인, 최소 장식

### 공통 기능
- 반응형 터미널 너비 (80/100/120+ 대응)
- 조건부 컬러링 (수익=green, 손실=red, 중립=dim)
- 한국어/영어 이중 언어
- 월간 수익률 히트맵 (RGB 배경색)

---

## 2. 차트 (charts.py) - 11종

1. Equity Curve (SMA 오버레이, 벤치마크 비교)
2. Candlestick/OHLCV (컬러 볼륨 서브플롯)
3. Drawdown (underwater 영역 차트)
4. Returns Distribution (히스토그램 + 정규분포 오버레이)
5. Monthly Returns Heatmap (RGB 셀 배경, GitHub 스타일)
6. Rolling Metrics (롤링 Sharpe + 변동성 서브플롯)
7. Trade Scatter (PnL vs 보유기간)
8. Correlation Matrix (blue-white-red 분기 색상)
9. Strategy DNA (수평 바 게이지, 레이더 대체)
10. Volume Profile (가격대별 거래량 수평 바)
11. Seasonality Chart (월별 평균 수익률 컬러 바)

---

## 3. 대시보드 (dashboard.py) - 5화면 Textual

### Screen 1: Overview (개요)
- 5개 메트릭 카드 (수익률, 샤프, MDD, 승률, 거래수)
- 에쿼티 차트 + 사이드바 통계
- 드로다운 차트 + 수익률 히스토그램 + 월간 히트맵

### Screen 2: Metrics (상세 지표)
- 3열 패널: 수익 / 리스크 / 위험조정
- 거래 통계 패널
- 월간 수익률 테이블
- 롤링 메트릭 스파크라인

### Screen 3: Trades (거래 브라우저)
- 필터/정렬/검색 바
- 행 선택 가능 DataTable
- 거래 상세 확장 패널 (MFE/MAE, R-multiple)
- 페이지네이션

### Screen 4: Charts (차트)
- 가격 차트 (캔들스틱/라인 전환)
- 볼륨 바
- 에쿼티 + 매매 마커
- 줌/스크롤 키보드 제어

### Screen 5: Compare (전략 비교)
- 멀티전략 랭킹 테이블
- 정규화 에쿼티 오버레이
- 드로다운 오버레이

### 키보드
- 1-5: 화면 전환
- j/k: 스크롤
- q: 종료
- d: 테마 전환
- /: 검색
- f: 필터
- z/x: 줌 인/아웃
- ?: 도움말

### 테마
- TradingView Midnight: #0C0C0C 배경, #00D4FF 시안, #00FF88 수익, #FF4444 손실
- Bloomberg Terminal: #000000 배경, #F39F41 앰버, #4AF6C3 틸

---

## 4. 구현 우선순위

### Phase A: console.py 리뉴얼 (3종 스타일)
- printResult() → 3종 스타일 지원 (modern/bloomberg/minimal)
- 월간 히트맵 함수 추가
- 반응형 너비 대응

### Phase B: charts.py 리뉴얼 (11종 차트)
- 기존 5종 개선 + 신규 6종 추가
- 히트맵, 롤링메트릭, 트레이드스캐터, 상관행렬, DNA, 시즌

### Phase C: dashboard.py 리뉴얼 (5화면)
- 기존 스켈레톤 → 완전한 5화면 구현
- ContentSwitcher 패턴, 커스텀 위젯
- CSS 테마 시스템

---

## 5. 파일 구조

```
tui/
├── __init__.py          (exports 업데이트)
├── console.py           (3종 스타일 보고서)
├── charts.py            (11종 차트)
├── progress.py          (기존 유지)
├── dashboard.py         (5화면 Textual 대시보드)
├── themes.py            (NEW: CSS 테마 + 컬러 시스템)
└── widgets.py           (NEW: MetricCard, HeatmapTable 커스텀 위젯)
```
