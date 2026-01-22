# Copilot Instructions: Trend Validator Protocol v2.2

## Project Overview

**Trend Validator Backtester** is a production-grade backtesting engine for Taiwan stocks with realistic market execution modeling. It implements a multi-stage momentum filtering protocol supporting both v1.1 and v2 variants with different risk controls.

### Core Architecture

- **Single-class design**: `TrendValidatorBacktest` handles all logic (indicators, staging, execution, reporting)
- **Taiwan-only universe**: 4-digit numeric tickers (regex `^\d{4}$`) validated via `validate_ticker()`
- **Exchange coverage**: TWSE (`.TW`) and TPEx/OTC (`.TWO`) downloaded sequentially per ticker
- **Execution model**: End-of-day signal generation → Next trading session fill at Open + slippage
- **Mark-to-liquidation daily**: Equity recorded every benchmark day including days with no trading activity

## Critical Patterns & Conventions

### 1. Four-Stage Filtering Pipeline
All entry signals follow this sequence (see lines 572-639):

1. **Stage 0 (Market Regime)**: `stage_0_market_regime()` — Bulls-only gate using SMA alignments
   - v1.1: Close > SMA_200
   - v2: Adds SMA_50 > SMA_200 and neutral regime
2. **Stage 1 (Trend Efficiency)**: `stage_1_trend_efficiency()` — Kaufman ER ≥ threshold + MA slope check
   - v2 uses dynamic cross-sectional ER percentile (70th) computed in `calculate_dynamic_er_threshold()`
3. **Stage 2 (Momentum Persistence)**: `stage_2_momentum_persistence()` — 60-day statistical metrics
   - v1.1: 4-check scoring system (positive ratio, skewness, t-stat, q05)
   - v2: Stricter thresholds with CVaR and max drawdown checks
4. **Stage 3 (Confirmation)**: `stage_3_confirmation()` — ROC, volume, displacement, recent drawdown
   - v1.1: Simpler ROC/volume/displacement logic
   - v2: Breakout-based (close > Highest_20) with volatility-scaled metrics

**Scoring**: Linear weighted combination (Stage1: 25%, Stage2: 45%, Stage3: 30%) — candidates sorted by score, allocated sequentially.

### 2. Order Lifecycle (Option C: Next-Available + 5-Day Expiry)

**BUY orders** (created line 653–667):
- Reserved cash allocated immediately on signal date
- Executed on next available ticker date at Open price + slippage
- **Expiry**: `days_pending >= 5` → auto-cancel, reserved cash released once
- **Stop recalculation** (line 381): After fill, recompute using `min(signal_close, entry_price) - atr_mult * ATR`

**SELL orders** (created lines 530–540):
- **No expiry** — persists across missing sessions until next available ticker date
- Exit reasons: Stop Loss, Trend Break, Time Stop, Donchian Break (v2 only)
- **Execution**: `execute_pending_orders()` is the single authoritative entry point (line 352)

**Cash integrity**: `reserved_cash` is deducted on signal, released on execution/cancellation/expiry (never negative).

### 3. Indicator Calculations (lines 108–140)

**Custom implementations**:
- **Wilder's ATR**: RMA smoothing (`ewm(alpha=1/period)`) not SMA
- **Rolling VWAP**: Sum-based with zero-volume protection (`vol.replace(0, np.nan)`)
- **Kaufman ER**: Directional change / volatility ratio for trend-following filter
- **SMA 20/50/200**: Standard rolling means
- **Donchian Channels**: **CRITICAL** — use PRIOR window (`shift(1)`) to avoid look-ahead bias
  - Line 128: `df['Lowest_20'] = df['Close'].rolling(window=20).min().shift(1)`
  - Line 129: `df['Highest_20'] = df['High'].rolling(window=20).max().shift(1)`
- **Vol_60 & EWMA_Vol_60**: Annualized returns volatility (×√252)
- **Displacement_over_ATR**: `(Close - Lowest_20) / (ATR_20 × 2.0)` — tracks distance from recent low

### 4. Position Management & Stops

**Daily trailing stop** (lines 299–315):
- Called every day: `update_trailing_stops(date, stock_data)`
- Only raises stop: `new_stop = max(prev_stop, close - atr_mult * atr)`
- Never lowers stop even if ATR decreases

**Initial stop after fill** (lines 381–385):
- `stop_base = min(signal_close, execution_price)` — use lower of two prices
- `initial_stop = stop_base - atr_mult * entry_atr`
- If entry_atr = 0, fallback to `entry_atr` from signal date

**Exit checks** (lines 318–342):
- Stop Loss: `close <= stop_loss`
- Trend Break: Two consecutive closes below `SMA_20 - 0.5×ATR`
- Time Stop: 20+ days held AND recovery ≤ 0.5×ATR
- Donchian Break (v2): `close < Lowest_20`

### 5. Taiwan-Specific Cost Model (lines 46–48)

```python
buy_fee_pct = 0.001425      # 0.1425%
sell_fee_pct = 0.001425     # 0.1425%
sell_tax_pct = 0.003        # 0.3% stock tax
slippage_pct = 0.001        # 0.1% (configurable)
```

**Applied as**:
- BUY fill cost: `price × (1 + slippage) × (1 + buy_fee)`
- SELL proceeds: `price × (1 - slippage) × (1 - sell_fee - sell_tax)`
- P&L reported: **Net %** (includes all costs); gross % also stored in trade record

### 6. Equity Recording & End-of-Test (Lines 664–675, 642–657)

**EVERY benchmark day** is recorded:
- Even if no trading activity
- Uses `record_equity(date, stock_data)` which mark-to-liquidates all open positions
- **Exit cost discount applied**: `market_value × (1 - slippage - sell_fee - sell_tax)`

**End-of-Test Option 1** (lines 642–657):
- Cancel all pending BUY orders (release reserved cash)
- **Keep all positions open** — do not force-liquidate
- Final equity = cash + sum of mark-to-liquidated positions
- Trades DataFrame includes only actual fills (no forced exits)

## Implementation Notes

### Commonly Modified Parameters
- `atr_period = 20` — lookback for volatility calculation
- `atr_multiplier = 2.5` — stop distance multiplier
- `er_percentile = 0.70` — v2 only, cross-sectional ER threshold
- `min_turnover_twd = 50M` — liquidity gate
- `order_expiry_days = 5` — BUY order lifespan

### Data Download & Normalization (lines 763–793)
- **Per-ticker download**: Avoid multi-symbol calls (prevents MultiIndex columns)
- **Exchange fallback**: Try `.TW` first, then `.TWO`
- **Column selection**: Keep only `[Open, High, Low, Close, Volume]`
- Validation: 4-digit check logs acceptance/rejection

### Version Differences (v1.1 vs v2)
- **ER threshold**: Fixed 0.40 (v1.1) vs dynamic percentile (v2)
- **Stage 2 weights**: Less strict positive ratio, CVaR-based tail risk (v2)
- **Regime filter**: Binary Bull/Bear (v1.1) vs Bull/Neutral/Bear (v2)
- **Stage 3 breakout**: ROC-based (v1.1) vs price-based with Donchian (v2)
- **Position sizing**: Vol_60 (v1.1) vs EWMA_Vol_60 (v2) with regime multipliers

## Key Dependencies
- **yfinance**: Data download via `.TW`/`.TWO` suffixes
- **pandas**: OHLCV manipulation, rolling calculations, indicator storage
- **numpy**: Statistical metrics (skewness, percentiles, CVaR)
- **scipy.stats**: Used indirectly for quantile calculations
- **matplotlib**: Plotting equity curve, drawdown, cash management charts

## When Modifying This Codebase

1. **Add/remove indicators**: Update `calculate_indicators()` and corresponding stage methods
2. **Change entry logic**: Modify stage methods or create new candidacy gates before ranking
3. **Adjust stops**: Modify `calculate_position_size()`, `update_trailing_stops()`, or `check_exit_conditions()`
4. **New version variant**: Duplicate class with new version tag, branch logic on `self.version`
5. **Order execution changes**: Keep exactly **ONE** `execute_pending_orders()` method; all fills flow through it

## Testing & Validation

- **Universe filter**: Confirm `validate_ticker()` rejects non-4-digit codes
- **Order flow**: Check `reserved_cash` is released and never negative
- **Equity completeness**: Assert `len(equity_curve)` = number of benchmark trading days (no gaps)
- **No NaN equity**: All equity values must be numeric (no missing dates)
- **Position cleanup**: Verify open positions at end-of-test are retained (not force-sold)
