# Alpha Pattern Cards

Use this file when you need fresh theses or a way to diversify a batch.
These are scaffolds, not guarantees.

## 1. Price-volume disagreement

Use when recent winners depend on flow or liquidity imbalance.

Common ingredients:

- `volume`
- `ts_mean(volume, N)`
- `close`
- `vwap`
- `ts_corr(...)`
- `ts_rank(...)`

Typical scaffolds:

- divergence between ranked price move and ranked volume move
- negative correlation between price strength and liquidity strength
- distance from VWAP conditioned on abnormal volume

Diversify by changing:

- correlation vs delta vs zscore framing
- raw price vs VWAP spread
- short horizon vs medium horizon

Avoid:

- making 5 formulas that are the same correlation with different windows

## 2. Short-horizon reversal with conditioning

Use when raw reversal works but fails turnover or concentration checks.

Common ingredients:

- `returns`
- `close`
- `ts_delay(close, N)`
- `ts_std_dev(returns, N)`
- `rank(...)`
- `zscore(...)`

Typical scaffolds:

- reversal signal scaled by recent volatility
- reversal only when volatility or liquidity surprise is high
- reversal on open-close or close-VWAP dislocation

Diversify by changing:

- conditioning variable
- cross-sectional transform
- horizon split between trigger and conditioner

Avoid:

- plain negative short return clones with only small window tweaks

## 3. Residual or de-beta structure

Use when many candidates fail self-correlation or behave like market beta.

Common ingredients:

- `beta_last_60_days_spy`
- `ts_corr(close,spy,63)`
- `systematic_risk_last_60_days`
- `unsystematic_risk_last_60_days`
- `ts_regression(...)`

Typical scaffolds:

- residualize a price or volume feature against market proxy
- rank idiosyncratic risk relative to systematic risk
- combine residual signal with reversal or divergence

Diversify by changing:

- regression target
- whether you use coefficient or residual
- whether the residual is ranked, zscored, or mixed with another feature

Avoid:

- turning every idea into the same beta hedge wrapper

## 4. VWAP and intraday dislocation

Use when open, close, VWAP, high, and low are available and the search space overuses returns.

Common ingredients:

- `open`
- `close`
- `high`
- `low`
- `vwap`

Typical scaffolds:

- close relative to mid-range
- absolute or signed distance from VWAP
- open-close spread normalized by recent volatility

Diversify by changing:

- signed vs absolute dislocation
- whether you smooth first or normalize later
- blend with volume or volatility terms

Avoid:

- stacking many price-only transforms that all chase the same effect

## 5. Surprise and liquidity shock

Use when you want broader, lower-turnover structure.

Common ingredients:

- `volume / ts_mean(volume, N)`
- `ts_zscore(close, N)`
- `ts_std_dev(returns, N)`
- `winsorize(...)`

Typical scaffolds:

- price anomaly weighted by liquidity surprise
- volatility shock followed by mean reversion
- ranked signal only when volume is abnormally high

Diversify by changing:

- the surprise anchor
- whether surprise gates or scales the core signal
- short vs medium lookbacks

Avoid:

- unstable denominators without winsorization or ranking

## 6. Regime switch

Use when one family works only in part of the universe or part of the time.

Common ingredients:

- momentum proxy
- reversal proxy
- volatility proxy
- correlation proxy

Typical scaffolds:

- momentum when volatility is calm, reversal when volatility is high
- divergence signal that activates only when correlation to market is elevated
- two normalized sub-signals blended by a regime proxy

Diversify by changing:

- gating variable
- blend form
- slow vs fast regime detector

Avoid:

- hard-coded complexity that hides a weak core thesis

## Batch Checklist

Before finalizing a batch, confirm:

- at least 3 distinct theses appear
- at least 1 candidate attacks concentration risk explicitly
- at least 1 candidate attacks self-correlation explicitly
- no more than 2 candidates share the same skeleton after number replacement
- every formula has a one-sentence thesis you can defend
