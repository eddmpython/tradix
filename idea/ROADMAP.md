# Tradex v2.0 로드맵 & 기획 문서

## 현재 완료된 작업 (v2.0.0)

### Phase 1: 독창적 분석 모듈 9개
| 모듈 | 파일 | 핵심 기능 |
|------|------|-----------|
| Strategy DNA | analytics/strategyDna.py | 12차원 벡터 지문, 코사인 유사도, 8가지 아키타입 분류 |
| Black Swan Defense | analytics/blackSwan.py | 0-100 방어력 점수, 6개 역사적 위기 분석, A+~F 등급 |
| Strategy Health | analytics/strategyHealth.py | 과적합 진단, 파라미터 안정성, Jarque-Bera 검정 |
| What-If Simulator | analytics/whatIf.py | 수수료/슬리피지/세금 민감도, 감도 매트릭스 |
| Drawdown Simulator | analytics/drawdownSimulator.py | Pain Index, Ulcer Index, Top-N 드로다운 |
| Seasonality Analyzer | analytics/seasonality.py | 월/요일/분기 패턴, ANOVA, 설날/추석 효과 |
| Correlation Analyzer | analytics/correlation.py | Pearson/Spearman/Kendall, K-means 클러스터링 |
| Trading Journal | analytics/tradingJournal.py | MFE/MAE, R-multiple, 한국어 자동 태그 |
| Strategy Leaderboard | analytics/leaderboard.py | 복합점수 랭킹, 한국어 뱃지 시스템 |

### Phase 2: 프리셋 전략 33개
- 기존 9개 + 신규 24개 (모멘텀 3, 오실레이터 4, 변동성 3, 추세 5, 멀티지표 3, 고급 6)

### Phase 3: 통합 + 테스트
- v2.0.0 버전, 150개 exports, 68/68 테스트 통과

### Phase 4: WASM/Pyodide 포팅 검토
- 판정: GO (70% 수정 없이 호환)
- 핵심 엔진 100%, 분석 모듈 100%, 프리셋 100% 호환
- 차단 요소: FinanceDataReader(HTTP), Rich/Plotext(TUI), multiprocessing

---

## 기획 중인 작업

### [1] TUI 세계적 수준 리뉴얼 (다음 단계)
- 상세 기획: idea/TUI_PLAN.md 참조
- 목표: 터미널에서도 TradingView급 경험 제공
- 보고서 + 차트 + 인터랙티브 대시보드 전면 재설계

### [2] WASM 브릿지 구현 (미래)
- wasm-bridge/ 모듈 (~1,400 LOC)
- JS Fetch API 데이터 로더, IndexedDB 캐시
- 예상 4주 소요

### [3] 추가 검토 항목
- 글로벌 데이터 피드 확장 (Yahoo Finance, CCXT)
- 라이브 트레이딩 브릿지 (Alpaca, IBKR)
- 유전 알고리즘 최적화
- Bayesian 최적화
- AI 전략 어드바이저 (LLM 통합)

---

## 프로젝트 통계

| 항목 | 수치 |
|------|------|
| 총 모듈 | 12개 패키지 |
| 내장 지표 | 50+ |
| 프리셋 전략 | 33개 |
| 분석 도구 | 9개 독창적 모듈 |
| Exports | 150개 |
| 테스트 | 68/68 통과 |
| 버전 | 2.0.0 |
