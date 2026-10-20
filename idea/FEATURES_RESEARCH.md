# Tradex 기능 연구 보고서

## 세계 15개 백테스팅 라이브러리 비교 분석

### 비교 대상
1. Backtrader (Python) - 130+ 지표, 이벤트 드리븐, 2020년 이후 유지보수 중단
2. Zipline/Zipline-Reloaded - Pipeline API, 팩터 투자 특화, Quantopian 폐쇄
3. VectorBT/Pro - 벡터화 백테스팅 100-1000x 속도, 유료 Pro
4. QuantConnect/LEAN - 기관급, 전 자산 클래스, C# 코어
5. PyAlgoTrade - 경량, Twitter 이벤트, 2018년 이후 중단
6. bt (pmorissette) - 포트폴리오 Algo 트리, 리밸런싱 특화
7. Backtesting.py - 초경량 API, Bokeh 시각화
8. QSTrader - 기관급 포트폴리오, 스케줄 드리븐
9. FinLab - 대만 시장 특화
10. Jesse - 크립토 전용, 300+ 지표
11. FreqTrade - 크립토, FreqAI ML, Telegram 봇
12. Catalyst - Zipline 크립토 포크, 2019년 중단
13. Blankly - 멀티에셋, 한 줄 백테스트→라이브
14. Moonshot/QuantRocket - 벡터화, ML/DL 통합
15. QuantStats - 성과 분석 전용, 50+ 메트릭

### Tradex 차별화 포인트
1. **레짐 인식 어드바이저** - MarketClassifier + RegimeForecaster (경쟁사 미보유)
2. **한국 시장 First-Class** - KRX, 거래세, 수수료, 한글 API
3. **전략 DNA 12차원 지문** - 세계 최초 구현
4. **블랙스완 방어력 점수** - 독자적 0-100 점수 체계
5. **벡터화 + 이벤트 드리븐 하이브리드** - VectorBT급 속도 + Backtrader급 유연성

---

## 업계 표준 투자전략 분류 (160+ 전략)

### A. Trend Following (15-20가지)
- MA Crossover, Dual/Triple MA, Donchian Channel, MACD, Parabolic SAR
- Supertrend, Ichimoku Cloud, ADX, Keltner Channel, HMA, Aroon, TRIX

### B. Mean Reversion (10-15가지)
- Bollinger Band, RSI, Z-Score, Stochastic, CCI, Williams %R
- VWAP Reversion, Ornstein-Uhlenbeck, Johansen Cointegration

### C. Momentum (10-12가지)
- Absolute/Relative/Dual Momentum, ROC, 52-Week High
- Sector Rotation, Earnings/Time-Series Momentum

### D. Volatility (8-10가지)
- Bollinger Squeeze, ATR Trailing, VIX Allocation
- Larry Williams Volatility Breakout, GARCH, Volatility Targeting

### E. Statistical Arbitrage (8-10가지)
- Pairs Trading (Distance/Cointegration/Kalman), PCA Stat Arb, Copula

### F. Factor Investing (10-15가지)
- Value, Momentum, Quality, Size, Low Vol, Dividend, Growth, Multi-Factor

### G. Options/Derivatives (12-15가지)
- Covered Call, Iron Condor, Straddle/Strangle, Calendar Spread

### H. Crypto/DeFi (10-12가지)
- Grid Trading, DCA, Funding Rate Arb, DEX/CEX Arb

### I. Machine Learning (10-15가지)
- Random Forest, XGBoost, LSTM, RL(DQN/PPO), NLP Sentiment

---

## 독창적 기능 연구 결과

### 구현 완료 (Phase 1)
1. Strategy DNA (12차원 벡터 지문)
2. Black Swan Defense Score (0-100)
3. Strategy Health Score (과적합 진단)
4. What-If Simulator (민감도 분석)
5. Drawdown Simulator (Pain/Ulcer Index)
6. Seasonality Analyzer (ANOVA 기반)
7. Correlation Analyzer (K-means 클러스터링)
8. Trading Journal (MFE/MAE, R-multiple)
9. Strategy Leaderboard (복합 뱃지)

### 미래 구현 후보
1. 유전 알고리즘 전략 진화
2. Bayesian 최적화
3. HMM 레짐 감지 (확장)
4. 섹터 로테이션
5. GARCH 변동성 모델링
6. Block Bootstrap MC 시뮬레이션
7. 포트폴리오 가상 리밸런싱 시뮬레이터
8. 실시간 대시보드
9. AI 전략 어드바이저 (LLM)
10. 시각적 전략 빌더 (No-Code)
