# Tradix TUI Dashboard - Design Specification

> Implementation-ready design for the Textual-based interactive TUI dashboard.
> Target: 120 columns x 40 rows minimum. Responsive down to 80 columns.

---

## Table of Contents

1. [View Modes](#1-view-modes)
2. [Main Dashboard (Overview)](#2-main-dashboard-overview)
3. [Trade Browser View](#3-trade-browser-view)
4. [Comparison View](#4-comparison-view)
5. [Detailed Metrics View](#5-detailed-metrics-view)
6. [Charts View](#6-charts-view)
7. [Keyboard Shortcut Map](#7-keyboard-shortcut-map)
8. [Color Palette Specification](#8-color-palette-specification)
9. [Widget-to-Section Mapping](#9-widget-to-section-mapping)
10. [Navigation Flow](#10-navigation-flow)
11. [Data Source Mapping](#11-data-source-mapping)

---

## 1. View Modes

| Key | Mode             | Description                              |
|-----|------------------|------------------------------------------|
| `1` | Overview         | Main dashboard: metrics + equity + DD    |
| `2` | Detailed Metrics | Full performance statistics table         |
| `3` | Trade Browser    | Scrollable trade history with filtering  |
| `4` | Charts           | Full-height equity + price + indicators  |
| `5` | Comparison       | Side-by-side strategy comparison         |

---

## 2. Main Dashboard (Overview) - ASCII Mockup

```
====[ 120 chars ]=====================================================================[ row 01-40 ]=====
                                                                                                col:001
 TRADIX  SMA Crossover  AAPL  2020-01-01 ~ 2024-12-31    [1]Overview [2]Metrics [3]Trades [4]Charts [5]Compare
+------------------------------------------------------------------------------------------------------+  row 02
|                                                                                                      |
| +----RETURN--------+ +----ANNUAL--------+ +----SHARPE--------+ +----MDD-----------+ +----WIN RATE---+ |  row 03
| |  +42.57%         | |  +9.23%          | |  1.34            | |  -18.42%         | |  58.3%        | |
| |  ^^^^ green      | |  ^^^^ green      | |  ^^^^ cyan       | |  ^^^^ red        | |  ^^^^ green   | |  row 05
| +------------------+ +------------------+ +------------------+ +------------------+ +--------------+ |
|                                                                                                      |  row 07
| +--EQUITY CURVE------[SMA20]--[SMA60]-------------------------------------------------------+--------+
| |                                                                       .-''```-.            | STATS  |  row 09
| |                                                              .--''``           \           |--------|
| |                                                      .--'``                     `.         | Trades |  row 11
| |                                             ___..--''                             \        |   47   |
| |                                    ___..--''                                      |        |--------|
| |                           ___..--''                                                `.      | P/F    |  row 13
| |                  ___..--''                                                          \      |  1.82  |
| |         ___..--''                                                                   |     |--------|
| |  __.--''                                                                             \    | Expect |  row 15
| |''                                                                                    `.   |  0.47% |
| |--- initial cash ($10,000,000) --------------------------------------------------------|   |--------|
| |                                                                                       |   | AvgHld |  row 17
| | 2020-01     2020-07     2021-01     2021-07     2022-01     2022-07     2023-01  2024 |   |  12.3d |
| +------------------------------------------------------------------------------------+---+--------+  row 19
|                                                                                                      |
| +--DRAWDOWN----------------------------------------------------------------------------------+       |  row 21
| | 0% -----.                        .-------.              .---.          .--.                 |       |
| |          \                      /         \            /     \        /    \     .-.        |       |
| |           \                    /           \          /       \      /      \   / | \       |  row 24
| |            \                  /             \        /         \    /        \ /  |  \      |
| |             \       .-.     /               \------/           \--/          V    |   \     |
| |-18.4% ------\-----/ | \---/                                                      |    `--  |  row 26
| |              `---'   `  `-'                                                                |
| | 2020-01     2020-07     2021-01     2021-07     2022-01     2022-07     2023-01  2024      |
| +--------------------------------------------------------------------------------------------+  row 28
|                                                                                                      |
| +--DAILY RETURNS DISTRIBUTION-----------+ +--MONTHLY RETURNS HEATMAP---------------------------+     |  row 30
| |          ..::||::..                    | |      J    F    M    A    M    J    J    A    S    O |     |
| |       .::||||||||||||::.               | | 2020 [+2] [+1] [-3] [+4] [+1] [-1] [+2] [+3] [+1]|     |
| |     ::||||||||||||||||||::             | | 2021 [+3] [+2] [+1] [-2] [+1] [-1] [+4] [+1] [-1]|     |  row 33
| |   ::||||||||||||||||||||||||::         | | 2022 [-2] [-3] [+1] [-1] [-2] [+1] [-1] [+2] [+1]|     |
| |  :||||||||||||||||||||||||||||||:      | | 2023 [+4] [+2] [+3] [+1] [+2] [-1] [+1] [+3] [+2]|     |
| |  ^^^^^^^^^^^^0%^^^^^^^^^^^^^^^^^^^^   | | 2024 [+1] [+2] [+1] [-1] [+3] [+2] [+1] [+1] [+2]|     |  row 36
| |  -4%  -2%   0%   +2%  +4%  +6%       | +----------------------------------------------------+     |
| +----------------------------------------+                                                           |
|                                                                                                      |  row 38
| [q]Quit  [1-5]Views  [d]Theme  [/]Search  [?]Help                            Tradix v1.0.0          |
+------------------------------------------------------------------------------------------------------+  row 40
```

### Layout Grid (120 x 40)

| Section                  | Row Start | Row End | Col Start | Col End | Height | Width |
|--------------------------|-----------|---------|-----------|---------|--------|-------|
| Header Bar               | 1         | 1       | 1         | 120     | 1      | 120   |
| Metric Cards (5x)        | 3         | 6       | 3         | 118     | 4      | ~22ea |
| Equity Chart             | 8         | 19      | 3         | 96      | 12     | 94    |
| Stats Sidebar            | 8         | 19      | 97        | 118     | 12     | 22    |
| Drawdown Chart           | 21        | 28      | 3         | 118     | 8      | 116   |
| Returns Distribution     | 30        | 37      | 3         | 52      | 8      | 50    |
| Monthly Returns Heatmap  | 30        | 37      | 54        | 118     | 8      | 65    |
| Footer / Status Bar      | 39        | 40      | 1         | 120     | 2      | 120   |

---

## 3. Trade Browser View - ASCII Mockup

```
====[ 120 chars ]=====================================================================[ row 01-40 ]=====

 TRADIX  SMA Crossover  AAPL  47 Trades    [1]Overview [2]Metrics [3]Trades [4]Charts [5]Compare     row 01
+------------------------------------------------------------------------------------------------------+
| [Filter: ALL v]  [Sort: Date v]  [Search: ____________]      Showing 1-20 of 47    Page 1/3         |  row 03
|------------------------------------------------------------------------------------------------------|
|  #  | Side | Entry Date | Exit Date  | Entry $   | Exit $    |    P&L    | Return% | Hold | Cumul% |  row 05
|------|------|------------|------------|-----------|-----------|-----------|---------|------|--------|
|   1  | BUY  | 2020-02-03 | 2020-03-15 |   310.00  |   285.50  |  -2,450   |  -7.90% |  41d | -7.90% |  row 07
|   2  | BUY  | 2020-04-01 | 2020-05-20 |   254.30  |   316.70  |  +6,240   | +24.54% |  49d |+14.70% |
|   3  | BUY  | 2020-06-08 | 2020-07-22 |   322.10  |   370.50  |  +4,840   | +15.02% |  44d |+31.93% |
| > 4  | BUY  | 2020-08-03 | 2020-08-28 |   435.10  |   499.60  |  +6,450   | +14.83% |  25d |+51.50% |  row 10
|   5  | BUY  | 2020-09-14 | 2020-10-12 |   115.00  |   121.10  |    +610   |  +5.30% |  28d |+59.53% |
|   6  | BUY  | 2020-11-02 | 2020-12-28 |   108.80  |   136.70  |  +2,790   | +25.64% |  56d |+100.4% |
|   7  | BUY  | 2021-01-11 | 2021-02-01 |   128.90  |   131.30  |    +240   |  +1.86% |  21d |+104.1% |  row 13
|   8  | BUY  | 2021-02-22 | 2021-03-30 |   126.00  |   119.90  |    -610   |  -4.84% |  36d | +94.3% |
|   9  | BUY  | 2021-04-12 | 2021-05-14 |   134.10  |   127.40  |    -670   |  -4.99% |  32d | +84.6% |
|  10  | BUY  | 2021-06-01 | 2021-07-19 |   124.30  |   146.40  |  +2,210   | +17.78% |  48d |+117.4% |  row 16
|  11  | BUY  | 2021-08-02 | 2021-09-10 |   145.60  |   155.00  |    +940   |  +6.46% |  39d |+131.5% |
|  12  | BUY  | 2021-10-04 | 2021-11-19 |   139.10  |   160.50  |  +2,140   | +15.38% |  46d |+166.1% |
|  13  | BUY  | 2021-12-06 | 2022-01-20 |   165.30  |   162.40  |    -290   |  -1.75% |  45d |+161.4% |  row 19
|  14  | BUY  | 2022-02-07 | 2022-03-11 |   172.40  |   158.50  |  -1,390   |  -8.06% |  32d |+140.3% |
|  15  | BUY  | 2022-04-01 | 2022-05-10 |   174.60  |   154.50  |  -2,010   | -11.51% |  39d |+112.7% |
|  16  | BUY  | 2022-06-06 | 2022-07-15 |   146.80  |   150.20  |    +340   |  +2.32% |  39d |+117.6% |  row 22
|  17  | BUY  | 2022-08-01 | 2022-09-12 |   162.50  |   153.70  |    -880   |  -5.42% |  42d |+105.8% |
|  18  | BUY  | 2022-10-10 | 2022-11-28 |   140.10  |   148.00  |    +790   |  +5.64% |  49d |+117.3% |
|  19  | BUY  | 2023-01-03 | 2023-02-17 |   125.10  |   153.70  |  +2,860   | +22.86% |  45d |+167.0% |  row 25
|  20  | BUY  | 2023-03-06 | 2023-04-14 |   151.00  |   165.20  |  +1,420   |  +9.40% |  39d |+192.1% |
|------------------------------------------------------------------------------------------------------|
|                                                                                                      |  row 27
| +--TRADE DETAIL (Trade #4)-------------------------------------------------------------------+       |
| |  Symbol: AAPL              Side: BUY              Quantity: 100 shares                     |       |  row 29
| |  Entry:  2020-08-03  @ $435.10    Exit:  2020-08-28  @ $499.60                             |       |
| |  Gross P&L: +$6,502.00           Commission: -$13.04         Slippage: -$38.96             |       |  row 31
| |  Net P&L:   +$6,450.00 (+14.83%)                                                          |       |
| |  Holding:   25 days              Entry Value: $43,510         Exit Value: $49,960           |       |  row 33
| |                                                                                            |       |
| |  Equity at Entry: $14,312,500    Equity at Exit: $14,962,500   Position Size: 3.04%        |       |  row 35
| +--------------------------------------------------------------------------------------------+       |
|                                                                                                      |
| [j/k]Scroll [Enter]Detail [f]Filter [s]Sort [/]Search [PgDn]Page  [Esc]Back        47 trades total  |  row 38
+------------------------------------------------------------------------------------------------------+  row 40
```

### Trade Browser Layout Grid

| Section              | Row Start | Row End | Col Start | Col End | Height | Width |
|----------------------|-----------|---------|-----------|---------|--------|-------|
| Header Bar           | 1         | 1       | 1         | 120     | 1      | 120   |
| Filter/Search Bar    | 3         | 3       | 3         | 118     | 1      | 116   |
| Table Header         | 5         | 5       | 3         | 118     | 1      | 116   |
| Trade Rows (scroll)  | 7         | 25      | 3         | 118     | 19     | 116   |
| Trade Detail Panel   | 28        | 36      | 3         | 118     | 9      | 116   |
| Footer / Keys        | 38        | 40      | 1         | 120     | 3      | 120   |

### Trade Table Columns

| Column     | Width | Align  | Format           | Color Logic                           |
|------------|-------|--------|------------------|---------------------------------------|
| `#`        | 5     | Right  | Integer          | dim white                             |
| `Side`     | 6     | Center | BUY/SELL         | green=BUY, red=SELL                   |
| `Entry`    | 12    | Left   | YYYY-MM-DD       | white                                 |
| `Exit`     | 12    | Left   | YYYY-MM-DD       | white                                 |
| `Entry $`  | 11    | Right  | Comma-separated  | white                                 |
| `Exit $`   | 11    | Right  | Comma-separated  | white                                 |
| `P&L`      | 11    | Right  | +/-,###          | green=positive, red=negative          |
| `Return%`  | 9     | Right  | +/-#.##%         | green=positive, red=negative          |
| `Hold`     | 6     | Right  | ##d              | dim white                             |
| `Cumul%`   | 8     | Right  | +/-#.#%          | green=positive, red=negative          |

### Filter Options

| Filter    | Values                                  |
|-----------|-----------------------------------------|
| Side      | ALL, BUY only, SELL only                |
| Result    | ALL, Winners only, Losers only          |
| Period    | ALL, This Year, Last Year, Custom       |
| Hold      | ALL, <7d, 7-30d, 30-90d, >90d          |

### Sort Options

| Sort By     | Default Direction |
|-------------|-------------------|
| Date        | Ascending (old first) |
| P&L         | Descending        |
| Return %    | Descending        |
| Hold Days   | Descending        |

---

## 4. Comparison View - ASCII Mockup

```
====[ 120 chars ]=====================================================================[ row 01-40 ]=====

 TRADIX  Strategy Comparison  AAPL  2020-01-01 ~ 2024-12-31  [1]Overview [2]Metrics [3]Trades [4]Charts [5]Compare
+------------------------------------------------------------------------------------------------------+  row 02
|                                                                                                      |
| STRATEGY RANKING (sorted by Sharpe Ratio)                                         [s]Sort metric     |  row 04
|------------------------------------------------------------------------------------------------------|
|  #  | Strategy           | Return  | Annual  | Sharpe | Sortino | MDD     | WinR  | Trades | P/F    |  row 06
|------|--------------------|---------|---------|---------|---------|---------| ------|--------|--------|
| > 1  | SMA Crossover      | +42.57% | +9.23%  |   1.34  |   1.87  | -18.42% | 58.3% |    47  |  1.82  |  row 08
|   2  | MACD Cross         | +38.12% | +8.41%  |   1.21  |   1.65  | -22.10% | 52.1% |    63  |  1.54  |
|   3  | RSI Oversold       | +31.44% | +7.08%  |   1.08  |   1.42  | -19.85% | 61.5% |    26  |  1.91  |  row 10
|   4  | Trend Following    | +28.91% | +6.56%  |   0.95  |   1.28  | -24.33% | 45.2% |    31  |  1.63  |
|   5  | Bollinger Breakout | +15.23% | +3.62%  |   0.52  |   0.71  | -28.60% | 50.0% |    42  |  1.12  |
|   6  | Mean Reversion     |  -4.21% | -0.86%  |  -0.15  |  -0.21  | -31.50% | 43.8% |    48  |  0.87  |  row 13
|------------------------------------------------------------------------------------------------------|
|                                                                                                      |  row 15
| +--EQUITY CURVES (overlaid)----------------------------------------------------------------------+   |
| |                                                                                                |   |  row 17
| |                                         ___..----SMA Crossover (cyan)                          |   |
| |                                   __.--''    .---MACD Cross (yellow)                           |   |
| |                              _.--'    _.--'''  .--RSI Oversold (magenta)                        |   |  row 20
| |                         _.--'   _.--''    __--' .--Trend Following (green)                      |   |
| |                    _.--'  _.--''    __--''   _.-'                                               |   |
| |               __.-' __.-''   __--''   ___.--'                                                  |   |  row 23
| |          __.-' _.-''  __--'' ___.---'''                                                        |   |
| |     __.-'_.-'' __--''___---'''                                                                 |   |
| |  .-'.-''_--''--'''                                                                             |   |  row 26
| | ''                                                                                             |   |
| | 2020-01     2020-07     2021-01     2021-07     2022-01     2022-07     2023-01     2024        |   |
| +------------------------------------------------------------------------------------------------+   |  row 28
|                                                                                                      |
| +--DRAWDOWNS (overlaid)--------------------------------------------------------------------------+   |  row 30
| | 0% ----                                                                                        |   |
| |         \    /\      /\                 /\.           /\                                        |   |
| |          \  /  \    /  \               /   \         /  \                                       |   |  row 33
| |           \/    \  /    \       /\    /     \       /    \            /\                        |   |
| |                  \/      \     /  \  /       \     /      \    /\   /  \                        |   |
| |                           \   /    \/         \---/        \  /  \/    \                        |   |  row 36
| | -31.5%                     `-'                              \/                                  |   |
| +------------------------------------------------------------------------------------------------+   |
|                                                                                                      |  row 38
| [j/k]Select [Enter]View  [s]Sort metric  [c]Cycle colors  [e]Export       Best: SMA Crossover        |
+------------------------------------------------------------------------------------------------------+  row 40
```

### Comparison Layout Grid

| Section              | Row Start | Row End | Col Start | Col End | Height | Width |
|----------------------|-----------|---------|-----------|---------|--------|-------|
| Header Bar           | 1         | 1       | 1         | 120     | 1      | 120   |
| Section Title        | 4         | 4       | 3         | 118     | 1      | 116   |
| Ranking Table        | 6         | 13      | 3         | 118     | 8      | 116   |
| Equity Overlay Chart | 16        | 28      | 3         | 118     | 13     | 116   |
| Drawdown Overlay     | 30        | 37      | 3         | 118     | 8      | 116   |
| Footer / Keys        | 39        | 40      | 1         | 120     | 2      | 120   |

### Comparison Table Columns

| Column     | Width | Align  | Color Logic                                  |
|------------|-------|--------|----------------------------------------------|
| `#`        | 5     | Right  | dim; #1 = bold gold                          |
| `Strategy` | 20    | Left   | white; selected = bold cyan                  |
| `Return`   | 9     | Right  | green=positive, red=negative                 |
| `Annual`   | 9     | Right  | green=positive, red=negative                 |
| `Sharpe`   | 9     | Right  | >1.0 green, 0-1 yellow, <0 red              |
| `Sortino`  | 9     | Right  | >1.5 green, 0-1.5 yellow, <0 red            |
| `MDD`      | 9     | Right  | >-10% green, -10~-25% yellow, <-25% red     |
| `WinR`     | 7     | Right  | >55% green, 45-55% yellow, <45% red         |
| `Trades`   | 8     | Right  | dim white                                    |
| `P/F`      | 8     | Right  | >1.5 green, 1.0-1.5 yellow, <1.0 red        |

### Strategy Color Assignment (for overlaid charts)

| Slot | Color   | Rich/Textual Style |
|------|---------|--------------------|
| 1    | Cyan    | `cyan`             |
| 2    | Yellow  | `yellow`           |
| 3    | Magenta | `magenta`          |
| 4    | Green   | `green`            |
| 5    | Red     | `red`              |
| 6    | White   | `white`            |
| 7    | Blue    | `blue`             |
| 8    | Orange  | `dark_orange`      |

---

## 5. Detailed Metrics View - ASCII Mockup

```
====[ 120 chars ]=====================================================================[ row 01-40 ]=====

 TRADIX  SMA Crossover  AAPL  2020-01-01 ~ 2024-12-31    [1]Overview [2]Metrics [3]Trades [4]Charts [5]Compare
+------------------------------------------------------------------------------------------------------+  row 02
|                                                                                                      |
| PERFORMANCE METRICS                                                                                  |  row 04
|                                                                                                      |
| +--RETURN METRICS------------------+ +--RISK METRICS---------------------+ +--RISK-ADJUSTED--------+ |  row 06
| |                                  | |                                   | |                        | |
| |  Total Return     +42.57%       | |  Volatility        16.82%        | |  Sharpe        1.34    | |  row 08
| |  Annual Return     +9.23%       | |  Max Drawdown     -18.42%        | |  Sortino       1.87    | |
| |  Best Month       +8.42%        | |  MDD Duration       87 days      | |  Calmar        0.50    | |  row 10
| |  Worst Month      -6.31%        | |  Downside Vol      11.20%        | |                        | |
| |  Positive Months      68%       | |  Avg Drawdown      -4.21%        | |                        | |  row 12
| |                                  | |                                   | |                        | |
| +----------------------------------+ +-----------------------------------+ +------------------------+ |  row 14
|                                                                                                      |
| +--TRADE METRICS----------------------------------------------------------------------------------+  |  row 16
| |                                                                                                  |  |
| |  Total Trades          47        Winning Trades        27        Losing Trades        20         |  |  row 18
| |  Win Rate           58.3%        Avg Win           +2,847        Avg Loss          -1,421        |  |
| |  Avg Win %         +12.4%        Avg Loss %         -5.8%        Expectancy        +0.47%        |  |  row 20
| |  Profit Factor       1.82        Max Consec Wins        6        Max Consec Loss        4        |  |
| |  Avg Holding        12.3d        Best Trade       +25.6%        Worst Trade       -11.5%         |  |  row 22
| |                                                                                                  |  |
| +--------------------------------------------------------------------------------------------------+  |  row 24
|                                                                                                      |
| +--MONTHLY RETURN TABLE---------------------------------------------------------------------------+  |  row 26
| |       Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec   | Year  |  |
| |-----------------------------------------------------------------------------------------------|  |  row 28
| | 2020        +1.2   -3.1   +4.2   +1.0   -1.2   +2.3   +3.4   +1.1   -0.8   +2.1   +1.5  +11.7|  |
| | 2021  +3.1  +2.0   +1.2   -2.3   +1.1   -1.0   +4.2   +1.3   -1.4   +0.8   +3.2   -0.6  +11.6|  |  row 30
| | 2022  -2.1  -3.4   +1.0   -1.2   -2.3   +1.1   -1.0   +2.1   +1.3   -0.5   +1.4   +0.8   -2.8|  |
| | 2023  +4.2  +2.3   +3.1   +1.0   +2.1   -1.2   +1.3   +3.2   +2.1   -0.4   +1.5   +2.6  +21.8|  |  row 32
| | 2024  +1.0  +2.1   +1.3   -1.0   +3.2   +2.0   +1.1   +1.3   +2.0   -0.3   +0.8          +13.5|  |
| +--------------------------------------------------------------------------------------------------+  |  row 34
|                                                                                                      |
| +--ROLLING METRICS (12-month)---------------------------------------------------------------------+  |  row 36
| |  Rolling Sharpe: 0.85 -> 1.12 -> 0.94 -> 1.45 -> 1.34    Trend: UP                             |  |
| |  Rolling Return: +8.2% -> +12.1% -> +3.4% -> +14.2% -> +9.2%                                   |  |  row 38
| +--------------------------------------------------------------------------------------------------+  |
|                                                                                                      |
| [j/k]Scroll  [Tab]Section  [e]Export CSV                                                             |  row 40
+------------------------------------------------------------------------------------------------------+
```

---

## 6. Charts View - ASCII Mockup

```
====[ 120 chars ]=====================================================================[ row 01-40 ]=====

 TRADIX  SMA Crossover  AAPL    [1]Overview [2]Metrics [3]Trades [4]Charts [5]Compare   [c]Candle [e]Equity
+------------------------------------------------------------------------------------------------------+  row 02
|                                                                                                      |
| +--PRICE CHART--[SMA 20 yellow]--[SMA 60 magenta]--[Trades: green=buy red=sell]------------------+   |  row 04
| |  $195 +                                                                   ___.---```---.        |   |
| |       |                                                             _.---''              \      |   |  row 06
| |  $180 +                                                      ___.-''                      |     |   |
| |       |                                               ____.--'                            |     |   |  row 08
| |  $165 +                                         ___.-''                                    \    |   |
| |       |                                   __.-''                                           |    |   |  row 10
| |  $150 +                             ___.-'                                                  |   |   |
| |       |                      ____.--'      ........                                         |   |   |  row 12
| |  $135 +               ___.-''         ....''       ````....                                 |   |   |
| |       |         ____.-'          ...'''                     ````....                         |   |   |  row 14
| |  $120 +   ___.-'           ...'''                                   ```....                  |   |   |
| |       | .--'          ...''                                                 ```..            |   |   |  row 16
| |  $105 +         ...'''                                                          `..         |   |   |
| |       |    ..'''                                                                   `-.      |   |   |  row 18
| |   $90 +..''                                                                          |     |   |   |
| |       +--------+--------+--------+--------+--------+--------+--------+--------+-----+      |   |   |
| |       2020-01  2020-07  2021-01  2021-07  2022-01  2022-07  2023-01  2023-07  2024         |   |   |  row 21
| +--------------------------------------------------------------------------------------------+   |
|                                                                                                      |  row 23
| +--VOLUME BARS------------------------------------------------------------------------------------+  |
| |  |||| ||| ||||| || |||| ||| || ||||| |||| ||| || |||| ||||| ||| |||| || ||||| |||| ||| |||| || |  |  row 25
| |  ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  |
| +------------------------------------------------------------------------------------------------+  |  row 27
|                                                                                                      |
| +--EQUITY CURVE + TRADE MARKERS-------------------------------------------------------------------+  |  row 29
| |  $14.9M                                                          .---```---.  green=Buy          |  |
| |  $14.5M                                                   .---'''           |  red=Sell           |  |  row 31
| |  $14.0M                                             .---''                   |                    |  |
| |  $13.0M                                      .---''                           \                   |  |  row 33
| |  $12.0M                               .---''                                  |                   |  |
| |  $11.0M                        .---''                                           \                 |  |  row 35
| |  $10.0M  -----.          .---''                                                  |                |  |
| |               `--.  .---''                                                        |               |  |  row 37
| +--------------------------------------------------------------------------------------------------+  |
|                                                                                                      |
| [z]Zoom+ [x]Zoom- [</>]Scroll [n]Last N bars  [i]Indicators [o]Overlays          Bars: 100/1,200    |  row 40
+------------------------------------------------------------------------------------------------------+
```

---

## 7. Keyboard Shortcut Map

### Global Shortcuts (available in all views)

| Key         | Action                     | Textual Binding          |
|-------------|----------------------------|--------------------------|
| `1`         | Switch to Overview         | `action_view_overview`   |
| `2`         | Switch to Detailed Metrics | `action_view_metrics`    |
| `3`         | Switch to Trade Browser    | `action_view_trades`     |
| `4`         | Switch to Charts           | `action_view_charts`     |
| `5`         | Switch to Comparison       | `action_view_compare`    |
| `q`         | Quit application           | `action_quit`            |
| `d`         | Toggle dark/light theme    | `action_toggle_dark`     |
| `?`         | Show help overlay          | `action_show_help`       |
| `Escape`    | Close overlay / go back    | `action_dismiss`         |
| `Tab`       | Next focus section         | `action_focus_next`      |
| `Shift+Tab` | Previous focus section     | `action_focus_previous`  |
| `l`         | Toggle language (en/ko)    | `action_toggle_lang`     |

### Overview View

| Key    | Action                    |
|--------|---------------------------|
| `j/k`  | Scroll equity chart       |
| `+/-`  | Zoom equity chart in/out  |
| `m`    | Toggle SMA overlay lines  |

### Trade Browser View

| Key         | Action                          |
|-------------|----------------------------------|
| `j` / `Down`  | Select next trade              |
| `k` / `Up`    | Select previous trade          |
| `Enter`     | Show trade detail panel          |
| `Escape`    | Close detail panel               |
| `f`         | Open filter dropdown             |
| `s`         | Open sort dropdown               |
| `/`         | Focus search input               |
| `PgDn`      | Next page (20 trades)            |
| `PgUp`      | Previous page                    |
| `g`         | Jump to first trade              |
| `G`         | Jump to last trade               |
| `w`         | Filter: winners only             |
| `e`         | Filter: losers only              |
| `a`         | Filter: show all (reset)         |

### Charts View

| Key    | Action                           |
|--------|----------------------------------|
| `c`    | Toggle candlestick/line chart    |
| `e`    | Toggle equity curve              |
| `z`    | Zoom in (show fewer bars)        |
| `x`    | Zoom out (show more bars)        |
| `<`    | Scroll chart left                |
| `>`    | Scroll chart right               |
| `n`    | Set number of bars (input)       |
| `i`    | Toggle indicator panel           |
| `o`    | Cycle overlay (SMA, EMA, BB)     |
| `v`    | Toggle volume subplot            |

### Comparison View

| Key     | Action                          |
|---------|----------------------------------|
| `j/k`   | Move selection in ranking table |
| `Enter` | View selected strategy detail   |
| `s`     | Cycle sort metric               |
| `c`     | Cycle color assignment          |
| `r`     | Reverse sort order              |

---

## 8. Color Palette Specification

### Base Theme (Dark Mode - Default)

| Element               | Foreground     | Background     | Hex FG    | Hex BG    | Textual/Rich Style         |
|-----------------------|----------------|----------------|-----------|-----------|----------------------------|
| Screen Background     | -              | #0C0C0C        | -         | #0C0C0C   | `$surface`                 |
| Panel Background      | -              | #1A1A2E        | -         | #1A1A2E   | `$panel`                   |
| Primary Text          | #E0E0E0        | -              | #E0E0E0   | -         | `white`                    |
| Secondary Text        | #888888        | -              | #888888   | -         | `dim`                      |
| Accent / Brand        | #00D4FF        | -              | #00D4FF   | -         | `cyan`                     |
| Header Bar BG         | #E0E0E0        | #1E3A5F        | #E0E0E0   | #1E3A5F   | `bold white on dark_blue`  |
| Footer Bar BG         | #888888        | #1A1A1A        | #888888   | #1A1A1A   | `dim on $surface-darken-1` |

### Semantic Colors

| Meaning             | Color Name  | Hex       | Rich Style       | Usage                           |
|---------------------|-------------|-----------|------------------|---------------------------------|
| Profit / Positive   | Green       | #00FF88   | `bold green`     | P&L > 0, Return > 0, Win       |
| Loss / Negative     | Red         | #FF4444   | `bold red`       | P&L < 0, Return < 0, Loss      |
| Neutral / Zero      | Gray        | #666666   | `dim`            | Zero values, unchanged          |
| Accent / Highlight  | Cyan        | #00D4FF   | `bold cyan`      | Selected row, brand, sparklines |
| Warning / Caution   | Yellow      | #FFD700   | `yellow`         | Sharpe 0-1, moderate values     |
| Indicator SMA20     | Yellow      | #FFD700   | `yellow`         | Short-term moving average       |
| Indicator SMA60     | Magenta     | #FF66FF   | `magenta`        | Long-term moving average        |
| Indicator EMA       | White       | #FFFFFF   | `white`          | Exponential moving average      |
| Buy Marker          | Green       | #00FF88   | `green`          | Entry point on chart            |
| Sell Marker         | Red         | #FF4444   | `red`            | Exit point on chart             |
| Initial Cash Line   | Gray        | #555555   | `gray`           | Horizontal reference line       |
| Drawdown Fill       | Dark Red    | #8B0000   | `dark_red`       | Drawdown area fill              |
| Grid Lines          | Dark Gray   | #333333   | `$surface-lighten-1` | Chart grid                 |

### Metric Card Color Rules

| Metric        | Green Threshold | Yellow Range | Red Threshold  |
|---------------|-----------------|--------------|----------------|
| Total Return  | > 0%            | -            | < 0%           |
| Annual Return | > 0%            | -            | < 0%           |
| Sharpe Ratio  | > 1.0           | 0.0 - 1.0   | < 0.0          |
| Sortino Ratio | > 1.5           | 0.0 - 1.5   | < 0.0          |
| Calmar Ratio  | > 0.5           | 0.0 - 0.5   | < 0.0          |
| Max Drawdown  | > -10%          | -10% ~ -25%  | < -25%         |
| Win Rate      | > 55%           | 45% - 55%    | < 45%          |
| Profit Factor | > 1.5           | 1.0 - 1.5   | < 1.0          |

### Light Theme Overrides

| Element               | Foreground     | Background     |
|-----------------------|----------------|----------------|
| Screen Background     | -              | #FAFAFA        |
| Panel Background      | -              | #FFFFFF        |
| Primary Text          | #1A1A1A        | -              |
| Secondary Text        | #666666        | -              |
| Accent                | #0088CC        | -              |
| Profit Green          | #008844        | -              |
| Loss Red              | #CC0000        | -              |

---

## 9. Widget-to-Section Mapping

### Textual Widget Assignments

| UI Section                | Textual Widget              | ID                    | Notes                                |
|---------------------------|-----------------------------|-----------------------|--------------------------------------|
| **App Shell**             | `App`                       | -                     | TradixDashboard(App)                 |
| **Header Bar**            | `Header`                    | `#header`             | Built-in Textual Header              |
| **Footer / Status**       | `Footer`                    | `#footer`             | Built-in Textual Footer w/ bindings  |
| **View Switcher**         | `ContentSwitcher`           | `#view-switcher`      | Switches between 5 view screens      |
| **Tab Bar**               | `Tabs` or custom            | `#tab-bar`            | [1]Overview [2]Metrics ... buttons   |
|                           |                             |                       |                                      |
| **--- Overview View ---** |                             |                       |                                      |
| Metric Card (x5)         | `Static` in `Horizontal`    | `#card-return` etc.   | Custom styled Static panels          |
| Equity Chart              | `PlotextPlot`               | `#equity-chart`       | textual-plotext widget               |
| Stats Sidebar             | `Static` in `Vertical`      | `#stats-sidebar`      | Key-value pairs stacked vertically   |
| Drawdown Chart            | `PlotextPlot`               | `#drawdown-chart`     | textual-plotext widget               |
| Returns Histogram         | `PlotextPlot`               | `#returns-hist`       | textual-plotext widget               |
| Monthly Heatmap           | `DataTable` or `Static`     | `#monthly-heatmap`    | Colored cells with monthly returns   |
|                           |                             |                       |                                      |
| **--- Metrics View ---**  |                             |                       |                                      |
| Return Metrics Panel      | `Static` in `Vertical`      | `#return-metrics`     | Bordered panel with key-value pairs  |
| Risk Metrics Panel        | `Static` in `Vertical`      | `#risk-metrics`       | Bordered panel with key-value pairs  |
| Risk-Adjusted Panel       | `Static` in `Vertical`      | `#riskadjusted`       | Bordered panel with key-value pairs  |
| Trade Metrics Panel       | `Static`                    | `#trade-metrics`      | Wide panel, 3-column layout          |
| Monthly Return Table      | `DataTable`                 | `#monthly-table`      | Scrollable DataTable with colors     |
| Rolling Metrics           | `Static`                    | `#rolling-metrics`    | Sparkline-style text display         |
|                           |                             |                       |                                      |
| **--- Trades View ---**   |                             |                       |                                      |
| Filter Bar                | `Horizontal` w/ `Select`    | `#filter-bar`         | Select + Input widgets               |
| Filter Dropdown           | `Select`                    | `#filter-select`      | Side/Result/Period filter            |
| Sort Dropdown             | `Select`                    | `#sort-select`        | Sort column selector                 |
| Search Input              | `Input`                     | `#search-input`       | Text search on trade rows            |
| Trade Table               | `DataTable`                 | `#trade-table`        | Scrollable, sortable DataTable       |
| Trade Detail Panel        | `Static`                    | `#trade-detail`       | Appears below on Enter; hidden else  |
| Page Indicator            | `Static`                    | `#page-indicator`     | "Page 1/3" text                      |
|                           |                             |                       |                                      |
| **--- Charts View ---**   |                             |                       |                                      |
| Price Chart               | `PlotextPlot`               | `#price-chart`        | Candlestick or line chart            |
| Volume Bars               | `PlotextPlot`               | `#volume-chart`       | Volume subplot below price           |
| Equity + Markers          | `PlotextPlot`               | `#equity-markers`     | Equity with buy/sell dots            |
| Chart Controls            | `Horizontal` w/ buttons     | `#chart-controls`     | Zoom, scroll, indicator toggles      |
|                           |                             |                       |                                      |
| **--- Compare View ---**  |                             |                       |                                      |
| Ranking Table             | `DataTable`                 | `#ranking-table`      | Sortable, selectable rows            |
| Equity Overlay Chart      | `PlotextPlot`               | `#equity-overlay`     | Multiple equity curves overlaid      |
| Drawdown Overlay          | `PlotextPlot`               | `#drawdown-overlay`   | Multiple drawdown curves overlaid    |
|                           |                             |                       |                                      |
| **--- Shared ---**        |                             |                       |                                      |
| Help Overlay              | `Screen` (modal)            | `#help-screen`        | Full shortcut reference              |
| Export Dialog              | `Screen` (modal)            | `#export-dialog`      | Export to CSV/JSON                   |

### Container Layout Hierarchy

```
TradixDashboard (App)
 +-- Header
 +-- Tabs (#tab-bar)
 |    +-- Tab("Overview", id="tab-1")
 |    +-- Tab("Metrics", id="tab-2")
 |    +-- Tab("Trades", id="tab-3")
 |    +-- Tab("Charts", id="tab-4")
 |    +-- Tab("Compare", id="tab-5")
 +-- ContentSwitcher (#view-switcher)
 |    +-- OverviewView (Vertical, id="view-overview")
 |    |    +-- Horizontal (#metric-cards)
 |    |    |    +-- MetricCard x5 (Static)
 |    |    +-- Horizontal (#main-row)
 |    |    |    +-- PlotextPlot (#equity-chart)   [weight=4]
 |    |    |    +-- Vertical (#stats-sidebar)     [weight=1]
 |    |    +-- PlotextPlot (#drawdown-chart)
 |    |    +-- Horizontal (#bottom-row)
 |    |         +-- PlotextPlot (#returns-hist)   [weight=1]
 |    |         +-- DataTable (#monthly-heatmap)  [weight=1]
 |    +-- MetricsView (Vertical, id="view-metrics")
 |    |    +-- Horizontal (#ratio-panels)
 |    |    |    +-- Static (#return-metrics)
 |    |    |    +-- Static (#risk-metrics)
 |    |    |    +-- Static (#riskadjusted)
 |    |    +-- Static (#trade-metrics)
 |    |    +-- DataTable (#monthly-table)
 |    |    +-- Static (#rolling-metrics)
 |    +-- TradesView (Vertical, id="view-trades")
 |    |    +-- Horizontal (#filter-bar)
 |    |    +-- DataTable (#trade-table)
 |    |    +-- Static (#trade-detail)  [display: none by default]
 |    +-- ChartsView (Vertical, id="view-charts")
 |    |    +-- PlotextPlot (#price-chart)
 |    |    +-- PlotextPlot (#volume-chart)
 |    |    +-- PlotextPlot (#equity-markers)
 |    +-- CompareView (Vertical, id="view-compare")
 |         +-- DataTable (#ranking-table)
 |         +-- PlotextPlot (#equity-overlay)
 |         +-- PlotextPlot (#drawdown-overlay)
 +-- Footer
```

---

## 10. Navigation Flow

### State Machine

```
                     +-----[?]------+
                     |   Help       |
                     |  Overlay     |
                     +---[Esc]------+
                          |
  [App Launch] ----> [OVERVIEW] <---[1]--- All views
                       |    |
             [2]-------+    +-------[3]
             v                      v
        [METRICS]              [TRADES]
             |                   |    |
             +---[4]--+   [Enter]+    |
                      v         v     |
                  [CHARTS]  [Trade    |
                   |    |   Detail]   |
                   |    |    [Esc]----+
                   +----+
                        |
              [5]-------+
              v
          [COMPARE]
              |
        [Enter]----> [OVERVIEW of selected strategy]
```

### View Transitions

| From        | Key     | To              | Animation       |
|-------------|---------|-----------------|-----------------|
| Any View    | `1`     | Overview        | Instant switch  |
| Any View    | `2`     | Detailed Metrics| Instant switch  |
| Any View    | `3`     | Trade Browser   | Instant switch  |
| Any View    | `4`     | Charts          | Instant switch  |
| Any View    | `5`     | Comparison      | Instant switch  |
| Any View    | `?`     | Help Overlay    | Slide from top  |
| Help        | `Esc`   | Previous View   | Slide to top    |
| Trades      | `Enter` | Trade Detail    | Expand panel    |
| Trade Detail| `Esc`   | Trades (table)  | Collapse panel  |
| Compare     | `Enter` | Overview (strat)| Instant switch  |

### URL-style Routing (for Textual)

| View              | Route / Screen ID    |
|-------------------|----------------------|
| Overview          | `view-overview`      |
| Detailed Metrics  | `view-metrics`       |
| Trade Browser     | `view-trades`        |
| Charts            | `view-charts`        |
| Comparison        | `view-compare`       |
| Help Overlay      | `help-screen`        |

---

## 11. Data Source Mapping

### BacktestResult fields -> Widget data

| Widget / Section        | Data Source                                           | Transform                    |
|-------------------------|-------------------------------------------------------|------------------------------|
| Card: Return            | `result.totalReturn`                                  | `f"{val:+.2f}%"`            |
| Card: Annual            | `result.metrics['annualReturn']`                      | `f"{val:+.2f}%"`            |
| Card: Sharpe            | `result.metrics['sharpeRatio']`                       | `f"{val:.2f}"`              |
| Card: MDD               | `result.metrics['maxDrawdown']`                       | `f"{val:.2f}%"`             |
| Card: Win Rate          | `result.winRate`                                      | `f"{val:.1f}%"`             |
| Equity Chart            | `result.equityCurve` (pd.Series)                      | `.tolist()` for plotext     |
| Equity Chart dates      | `result.equityCurve.index`                            | `.strftime('%Y-%m-%d')`     |
| Stats: Trades           | `result.totalTrades`                                  | `str(val)`                  |
| Stats: P/F              | `result.metrics['profitFactor']`                      | `f"{val:.2f}"`              |
| Stats: Expectancy       | `result.metrics['expectancy']`                        | `f"{val:.2f}%"`             |
| Stats: Avg Holding      | `result.metrics['avgHoldingDays']`                    | `f"{val:.1f}d"`             |
| Drawdown Chart          | Derived: `(equity - cummax) / cummax * 100`           | List of floats              |
| Returns Histogram       | Derived: `equityCurve.pct_change().dropna() * 100`    | List of floats for hist     |
| Monthly Heatmap         | `result.metrics['monthlyReturns']` or derive          | Resample to ME, pct_change  |
| Trade Table rows        | `result.trades` (List[Trade])                         | Access .entryDate, .pnl etc |
| Trade Detail            | Single `Trade` object from `result.trades[i]`         | All properties              |
| Comparison Ranking      | List of `BacktestResult` objects                      | Extract metrics from each   |
| Equity Overlay          | Multiple `result.equityCurve` series                  | Normalize to % or overlay   |

### PerformanceMetrics -> Detailed Metrics View

| Section          | Fields                                                                                  |
|------------------|-----------------------------------------------------------------------------------------|
| Return Metrics   | `totalReturn`, `annualReturn`, `bestMonth`, `worstMonth`, count positive months         |
| Risk Metrics     | `volatility`, `maxDrawdown`, `maxDrawdownDuration`, downside vol, avg drawdown          |
| Risk-Adjusted    | `sharpeRatio`, `sortinoRatio`, `calmarRatio`                                            |
| Trade Metrics    | `totalTrades`, `winningTrades`, `losingTrades`, `winRate`, `avgWin`, `avgLoss`,         |
|                  | `avgWinPercent`, `avgLossPercent`, `expectancy`, `profitFactor`,                        |
|                  | `maxConsecutiveWins`, `maxConsecutiveLosses`, `avgHoldingDays`                          |
| Monthly Table    | `monthlyReturns` list -> reshape to 12-column year x month grid                         |

---

## Appendix A: CSS Skeleton for Textual

```css
Screen {
    background: $surface;
}

#tab-bar {
    dock: top;
    height: 3;
    background: $primary-darken-3;
}

#view-switcher {
    height: 1fr;
}

/* ── Overview ── */
#metric-cards {
    height: 5;
    padding: 0 1;
}

.metric-card {
    width: 1fr;
    height: 5;
    border: solid $primary;
    padding: 0 1;
    margin: 0 1;
    content-align: center middle;
}

.metric-card .label {
    color: $text-muted;
    text-style: dim;
}

.metric-card .value {
    text-style: bold;
}

.metric-card.positive .value {
    color: $success;
}

.metric-card.negative .value {
    color: $error;
}

#main-row {
    height: 14;
}

#equity-chart {
    width: 4fr;
    border: solid $primary-darken-2;
}

#stats-sidebar {
    width: 1fr;
    border: solid $primary-darken-2;
    padding: 1 2;
}

#drawdown-chart {
    height: 10;
    border: solid $primary-darken-2;
}

#bottom-row {
    height: 10;
}

#returns-hist {
    width: 1fr;
    border: solid $primary-darken-2;
}

#monthly-heatmap {
    width: 1fr;
    border: solid $primary-darken-2;
}

/* ── Trade Browser ── */
#filter-bar {
    height: 3;
    padding: 0 1;
}

#trade-table {
    height: 1fr;
}

#trade-detail {
    height: auto;
    max-height: 12;
    display: none;
    border: solid $accent;
    padding: 1 2;
}

#trade-detail.visible {
    display: block;
}

/* ── Comparison ── */
#ranking-table {
    height: 12;
    border: solid $primary-darken-2;
}

#equity-overlay {
    height: 14;
    border: solid $primary-darken-2;
}

#drawdown-overlay {
    height: 10;
    border: solid $primary-darken-2;
}

/* ── Charts ── */
#price-chart {
    height: 20;
    border: solid $primary-darken-2;
}

#volume-chart {
    height: 5;
    border: solid $primary-darken-2;
}

#equity-markers {
    height: 12;
    border: solid $primary-darken-2;
}
```

## Appendix B: Responsive Breakpoints

| Terminal Width | Layout Adaptation                                        |
|----------------|----------------------------------------------------------|
| >= 120 cols    | Full layout as designed above                            |
| 100-119 cols   | Stats sidebar moves below equity chart                   |
| 80-99 cols     | Metric cards wrap to 2 rows of 3; Monthly heatmap hidden |
| < 80 cols      | Single column stack; cards stack vertically               |

## Appendix C: Textual BINDINGS Definition

```python
BINDINGS = [
    Binding("q", "quit", "Quit"),
    Binding("1", "view('overview')", "Overview", show=True),
    Binding("2", "view('metrics')", "Metrics", show=True),
    Binding("3", "view('trades')", "Trades", show=True),
    Binding("4", "view('charts')", "Charts", show=True),
    Binding("5", "view('compare')", "Compare", show=True),
    Binding("d", "toggle_dark", "Theme", show=True),
    Binding("question_mark", "help", "Help", show=True),
    Binding("l", "toggle_lang", "Lang"),
    Binding("escape", "dismiss", "Back", show=False),
]
```
