# Project README — Systematic Bond Strategy (team submission)

## Objective
Beat the ALBI benchmark by building and testing systematic bond allocation strategies using bond-level data and macro signals. We evaluate performance using walk-forward testing over 2023–2024 and enforce risk constraints (notably active modified duration).

---

## What we know / rationale
- **US rates** are a strong global signal and commonly lead other markets.  
- **JSE Top40** reflects local equity performance and is used as a proxy for local risk sentiment / GDP dynamics.  
- **Stocks and bonds** tend to be negatively correlated in return; equity weakness often implies falling yields and higher bond prices — useful for directional signals.  
- **Stable bonds** (low volatility, low duration shocks) are useful as a core sleeve to reduce turnover and stabilise returns.

---

## Team split
- **Inhle**: stability/core sleeve, data exploration, portfolio construction for the stable allocation.  
- **Devin**: macro-driven active sleeve and optimisation, model development for dynamic allocation.

---

## Strategy 1 — Hybrid: 50% Stable + 50% Market-based (original idea)
**Summary**
- Analyse all available bonds.
- Select top 4 bonds by a stability/signal metric.
- Use the signal `md_per_conv*100 - top40_return/10 + comdty_fut/100` for ranking.
- Allocate **50%** to the selected bonds (equal split among them).
- Allocate remaining **50%** across all bonds using the dynamic optimiser.
- Selection constraints: only the selected bonds should receive the target 50% stable allocation.

**Steps (Inhle)**
1. Inspect available bonds and their latest characteristics (duration, yield, convexity).  
2. Select 3–4 best bonds (by chosen stability metric or signal).  
3. Fix stable allocation (e.g., 12.5% per bond for four bonds = 50%).  
4. Optimise the remaining 50% dynamically across all bonds (or across the remaining bonds).  
5. Update constraints: ensure the stable sleeve is fixed and that the dynamic sleeve sums to its budget.  
6. Test and validate.

**Pros**
- Concentrated exposure to your best picks.
- Potentially higher returns if top picks are correct.
- Lower transaction costs for the stable sleeve.

**Cons**
- Higher concentration risk.
- Less diversification and potentially higher volatility.

---

## Strategy 2 — Stability-first (Faheem Kolia’s recommendation)
**Summary**
- Build a stability metric using:
  - Sharpe ratio (30%),
  - Inverse volatility (20%),
  - Drawdown protection (20%),
  - Consistency of positive returns (30%).
- Rank bonds by the composite stability score.
- Allocate **100%** to the top stable bonds (example weights: `[20%, 20%, 15%, 15%, 10%, 10%, 5%, 5%]` over top 8).
- Backtest and compare to ALBI.

**Pros**
- Very low turnover (stable core).
- Clear, defensible allocation based on risk-adjusted stability.

**Cons**
- May miss opportunistic macro-driven moves.
- Might underperform during strong directional market moves.

---

## Strategy 3 — Final working strategy (Devin’s macro-driven model)
**Summary**
- Uses macro predictors (JSE, US 2y/10y, commodities) to forecast yield moves.
- Uses bond features (yield momentum, convexity, volatility, duration shocks) to create per-bond signals.
- Runs constrained optimisation to maximise signal subject to:
  - weights ∈ [0, 0.2],
  - weights sum to 1 (or to the active sleeve budget),
  - active modified duration within ±1.2y (or within chosen cap).
- Walk-forward testing over 2023–2024 with turnover penalties and transaction cost assumptions.

**Enhancement**
- Add a **stability sleeve** (40% fixed) that holds the top-4 stable bonds (unchanging within a chosen update frequency), while the remaining 60% is optimised dynamically.

---

## Implementing the 40% Stability Sleeve (recommended approach)
**Design**
- **Core sleeve** = 40% of portfolio. This is equally split across a top-4 stable basket (10% each). The core basket is either:
  - Fixed for the entire backtest, or
  - Rebalanced/reselected infrequently (quarterly or every N trading days).
- **Active sleeve** = 60% of portfolio. Run the existing optimiser on the remaining universe (all bonds not in the core basket, or all bonds but with the core weights fixed), subject to the same constraints scaled to the 60% budget.

**Selection filters for top-4 stable bonds (examples)**
- Lowest 90-day return volatility.  
- Lowest 90-day duration spike metric.  
- High liquidity proxy (if available).  
- Or composite stability score (Sharpe, inverse vol, drawdown protection).

**Practical implementation details**
1. Compute stability score or volatility on `df_train_bonds`.
2. Pick `stable_bonds = top4_by_stability`.
3. Fix `w_stable = {bond: 0.10 for bond in stable_bonds}`.
4. Build optimiser for active sleeve:
   - Let `w_active` be optimizer variables for the active bond set.
   - Constrain `sum(w_active) == 0.60`.
   - Duration constraint should consider combined portfolio: `port_md_total = sum(w_stable*md_stable) + sum(w_active*md_active)`. Enforce `|port_md_total - bench_md| <= p_active_md`.
   - Bounds for `w_active` remain [0, 0.2] but you may want to rescale or cap per-instrument active weight to avoid exceeding total limits.
5. After optimisation, final weights = union of `w_stable` and `w_active`.
6. Calculate turnover relative to previous *full* portfolio (core + active) to correctly charge turnover costs.

**Example pseudo-code snippet**
```python
# choose stable bonds (from training data)
stable_bonds = df_train_bonds.groupby('bond_code')['return'].std().nsmallest(4).index.tolist()
w_stable = {b: 0.10 for b in stable_bonds}  # 40% core

# define active universe
active_bonds = [b for b in bond_codes if b not in stable_bonds]

# objective uses signals for the active universe; optimise for sum(weights) == 0.60
# duration constraint must include the fixed sleeve contribution:
fixed_md = np.dot([w_stable[b] for b in stable_bonds], df_train_bonds_current.set_index('bond_code').loc[stable_bonds]['modified_duration'])
# then enforce: abs(fixed_md + dot(w_active, md_active) - bench_md) <= p_active_md

# run minimise(...) for w_active with bounds and constraints
# combine: final_weights = {**w_stable, **dict(zip(active_bonds, w_active_opt))}
