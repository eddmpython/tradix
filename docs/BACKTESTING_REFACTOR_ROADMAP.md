# Tradix Backtesting Refactor Roadmap

## Goal

Tradix should become a world-class backtesting library by making the execution
engine trustworthy first and then expanding features on top of a single,
consistent simulation contract.

The target is not "more modules". The target is:

- one execution truth model
- one portfolio accounting truth model
- consistent event/vectorized semantics
- realistic market simulation
- deterministic, regression-tested research workflows

## Current Patch Status

This repository has already started the refactor with the first corrective
patch.

Completed in this patch:

- `BrokerSimulator` now supports next-bar open execution used by the core engine
- `QuickStrategy` stop-loss/take-profit state is now synchronized to actual fills
- trade accounting has been split into `portfolio/tradeLedger.py` for lot-based reconciliation
- `easy.optimize()` and `voptimize()` now return the real best metric value
- `WalkForwardResult.getBestParams()` now returns the actual most common fold params
- runtime package version is aligned with `pyproject.toml`
- regression tests were added for execution-path correctness

## Non-Negotiable Design Principles

1. Execution must be fill-driven, not signal-driven.
2. Every reported metric must be derivable from an auditable fill ledger.
3. Event-driven and vectorized engines must share the same economic semantics.
4. Feature claims in README must only describe features wired into the main path.
5. Analytics are downstream consumers of core truth, not independent truth sources.

## Target Architecture

### 1. Execution Core

- `ExecutionBroker` contract shared by all engines
- fill modes: next-open, same-close, VWAP, worst, best, randomized stress
- native support for market, limit, stop, stop-limit, trailing stop
- order grouping: OCO, bracket, linked exits, parent/child orders
- explicit rejection reasons and order lifecycle audit trail

### 2. Portfolio Accounting Core

- event-sourced cash ledger
- lot-based position accounting
- FIFO/LIFO/average-cost modes
- partial fill and partial exit support
- margin, leverage, borrow fee, short locate, interest accrual hooks
- multi-currency cash buckets and FX conversion hooks

### 3. Data Platform

- unified symbol resolver across FDR, Yahoo, CCXT
- exchange calendars and timezone normalization
- split/dividend/corporate-action handling
- deterministic local bundle/cache layer
- intraday and daily compatibility with explicit bar semantics

### 4. Strategy Runtime

- strategy lifecycle aligned to execution timing
- fill callbacks as the source of position-state truth
- reusable state containers for trailing stops, bracket state, risk budgets
- strategy API separated from internal engine state

### 5. Vectorized Engine

- parity suite against the event engine
- same fee/slippage/stop/take-profit semantics
- parameter sweep infrastructure decoupled from data fetching
- portfolio-level vectorized research path for basket strategies

### 6. Research Layer

- optimizer on top of a pure backtest runner contract
- walk-forward on immutable datasets, not refetched side effects
- Monte Carlo, robustness, and regime analysis consuming the same result schema
- benchmark and factor pipelines with explicit alignment rules

## Target Package Layout

The repository should move toward this shape while preserving import
compatibility during the transition:

```text
tradix/
├── core/
│   ├── engine/
│   ├── execution/
│   ├── portfolio/
│   └── results/
├── data/
│   ├── feeds/
│   ├── cache/
│   ├── calendars/
│   └── symbols/
├── research/
│   ├── optimize/
│   ├── walkforward/
│   ├── analytics/
│   └── benchmark/
├── strategy/
├── easy/
├── broker/
├── risk/
└── tui/
```

The current repository has already started this decomposition by splitting
lot-based trade accounting into `portfolio/tradeLedger.py`. The next folder
refactor should be alias-first: add new package paths, migrate internals, and
keep legacy imports working until the public API is intentionally revised.

## Refactor Phases

## Phase 1. Execution Truth

Priority: highest

Deliverables:

- replace `SimpleBroker`/`BrokerSimulator` split with one broker contract
- move order timing rules out of ad-hoc engine branches into broker/fill policy
- make `QuickStrategy` and preset strategies fill-aware
- add integration tests for gap entries, next-open fills, stop/limit behavior

Acceptance gates:

- broker swap cannot break `BacktestEngine`
- no stop-loss/take-profit decision may depend on a pre-fill synthetic entry price
- one-bar gap regression tests must pass

## Phase 2. Trade Ledger Rewrite

Priority: highest

Deliverables:

- replace single-trade-per-symbol shortcut with lot-aware ledger
- support scale-in, scale-out, partial exits, and aggregated trade views
- derive win rate, profit factor, MFE/MAE, journal entries from fills/lots

Acceptance gates:

- portfolio PnL and trade PnL must reconcile exactly
- pyramid and partial exit tests must pass
- analytics must stop reading inconsistent shortcut fields

## Phase 3. Advanced Orders Become Real

Priority: high

Deliverables:

- integrate OCO, bracket, and trailing orders into the execution loop
- make parent-child order activation explicit
- support cancellation propagation and conflict resolution

Acceptance gates:

- advanced orders work through `backtest()` and the main engine path
- README claims are backed by engine integration tests

## Phase 4. Data and Calendar Layer

Priority: high

Deliverables:

- feed router for equity, ETF, index, futures-ready, and crypto symbols
- calendar-aware sessions and timezone-safe bars
- corporate action adjustment policy
- local cache manifest with deterministic dataset identity

Acceptance gates:

- the same symbol and date range resolves to the same canonical dataset id
- feed selection is explicit and testable

## Phase 5. Event/Vectorized Parity

Priority: high

Deliverables:

- unified result schema
- parity tests for fees, slippage, stops, and trade counts
- vectorized optimizer detached from direct network fetches

Acceptance gates:

- supported strategies produce materially equivalent event/vectorized outcomes
- trade count semantics are identical across engines

## Phase 6. Portfolio and Institutional Features

Priority: medium

Deliverables:

- portfolio constraints, rebalancing policies, turnover caps, exposure limits
- benchmark-relative analytics
- margin and leverage simulation
- cross-asset portfolio accounting

Acceptance gates:

- multi-asset and single-asset engines share accounting primitives
- turnover and exposure metrics are first-class outputs

## Phase 7. Developer Experience and Release Discipline

Priority: medium

Deliverables:

- stable public API tiers: `easy`, `research`, `core`
- contract tests for public examples
- benchmark harness with reproducible methodology
- release checklist that blocks doc/runtime/version drift

Acceptance gates:

- examples in README are executable tests
- package version, runtime version, and docs version always match

## World-Class Feature Backlog

After the core refactor, these are the features worth adding because they
increase library quality instead of just increasing surface area.

- realistic intrabar stop/limit trigger policies
- borrow fee and funding rate simulation
- market impact models by volume participation
- exchange calendar bundles and holiday-aware walk-forward windows
- transaction-cost what-if sweeps built on the real execution engine
- portfolio cash sharing and rebalancing groups
- deterministic dataset snapshots for research reproducibility
- strategy parity test harness across engines
- benchmark bundles and factor-pipeline integration

## Testing Strategy

The project needs a deeper test pyramid.

### Unit

- indicators
- fill models
- commission/slippage models
- parameter-space combinatorics

### Integration

- next-open vs same-close execution
- gap entry and exit behavior
- limit/stop/OCO/bracket lifecycle
- scale-in/scale-out accounting
- multi-feed synchronization

### Parity

- event vs vectorized results
- same strategy, same data, same fees, same stop rules

### Release

- README example execution
- version consistency
- benchmark smoke tests

## Immediate Next Coding Queue

1. Introduce a real broker interface and remove `SimpleBroker` as the default truth path.
2. Rewrite portfolio trade tracking around fill lots.
3. Wire advanced orders into `BacktestEngine` and `MultiAssetEngine`.
4. Add parity tests between event and vectorized engines.
5. Build a feed router used by `easy.backtest()` instead of hardcoding FDR.
