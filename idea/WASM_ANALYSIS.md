# WASM/Pyodide 포팅 분석 보고서

## 판정: GO - 70% 수정 없이 호환

## 호환성 매트릭스

| 구성요소 | Pyodide 호환성 | 비고 |
|----------|---------------|------|
| 핵심 엔진 (NumPy 벡터화) | 100% | 수정 불필요 |
| 이벤트 드리븐 엔진 | 100% | 순수 Python |
| 분석 모듈 9개 | 100% | numpy/pandas/scipy 모두 지원 |
| 33개 프리셋 전략 | 100% | 순수 Python 로직 |
| 리스크/퀀트 모듈 | 100% | scipy/statsmodels 지원 |
| 버전/엔티티/이벤트 | 100% | 순수 Python |
| FinanceDataReader | **차단** | HTTP 요청 → JS fetch API 브릿지 필요 |
| Rich/Plotext/Typer (TUI) | **차단** | 터미널 없음 → JS 콘솔/DOM 렌더링 |
| multiprocessing | **차단** | Web Worker 대체 필요 |
| Textual (대시보드) | **차단** | 터미널 TUI → 웹 UI 전환 필요 |

## 필요한 브릿지 모듈 (~1,400 LOC)

### 1. data_loader.py
- JS Fetch API를 통한 데이터 로딩
- Pyodide의 `pyodide.http.pyfetch()` 활용
- CSV/JSON 파싱 → pandas DataFrame 변환

### 2. http_bridge.py
- Pyodide ↔ JavaScript 통신 레이어
- `JsProxy` 객체 핸들링
- Promise → async/await 변환

### 3. storage.py
- IndexedDB 기반 캐시 시스템
- 주가 데이터 로컬 저장
- 전략 결과 영속화

## 예상 작업량
- 개발 기간: 4주
- 코드 변경: ~1,400 LOC
- 브라우저 성능: 백테스트 1회당 200-500ms

## 아키텍처 제안

```
[Browser]
  ├── index.html (UI)
  ├── main.js (Pyodide 로더)
  ├── worker.js (Web Worker - 백테스트 실행)
  └── [Pyodide WASM]
       ├── tradex (핵심 엔진)
       ├── wasm-bridge/ (브릿지 모듈)
       ├── numpy, pandas, scipy (Pyodide 내장)
       └── data_loader (JS fetch 브릿지)
```
