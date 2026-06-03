# Why 28 Pairs

> **Purpose:** Document why Arc 10 trades these specific 28 currency pairs.

## The 28 pairs

All crosses among 8 base currencies: **AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD**.

```
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD
CADCHF, CADJPY
CHFJPY
EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD
GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD
NZDCAD, NZDCHF, NZDJPY, NZDUSD
USDCAD, USDCHF, USDJPY
```

Count: 8 × 7 / 2 = 28. Every unique pair of base currencies.

## Why 8 currencies (not more, not fewer)

**Major + safe-haven + key crosses:** USD, EUR, GBP, JPY are the four most-traded currencies globally. CHF and JPY are the two main safe-haven currencies. AUD, CAD, NZD are the major commodity currencies.

These 8 are:
- Universally offered by every major prop firm broker
- Highly liquid 24/5 during forex hours
- Have decade+ of clean historical data for backtesting
- Diverse enough to capture different macro regimes (commodity flows, safe-haven flows, central bank divergence)

## Why not more (e.g. include SEK, NOK, ZAR, MXN)

**Liquidity concerns:**
- SEK, NOK, ZAR have lower liquidity than the core 8
- Spreads are wider, especially during off-hours
- Slippage on entries/exits is meaningfully worse

**Broker support:**
- Not all prop brokers offer the full minor/emerging universe
- A pair list that includes ZAR might work on FTMO but not FundedNext
- Reducing portability across brokers

**Backtest cleanliness:**
- Minor pairs have more data quality issues (missing bars, spread spikes)
- Older history (pre-2015) is less reliable for minors

**Signal contribution:**
- Tested adding minor pairs in early arcs; marginal effect on aggregate worst-fold ratio
- Not worth the complexity tax

## Why not fewer (e.g. just majors)

**Diversification:**
- ~225 trades/year × 28 pairs vs ~80 trades/year × 7 majors
- Higher trade count = lower variance per year
- Multi-pair allows different macro regimes to net out

**Signal independence:**
- Crosses (e.g. AUDJPY, EURGBP) are not perfectly correlated with their constituent majors
- Each pair has its own signal profile and contributes independent edge

**WFO statistical power:**
- WFO with 7 pairs would have 5-10x lower trade count → much higher noise in worst-fold metric
- 28 pairs provides enough statistical mass to make worst-fold gate meaningful

## What changes if we add or remove pairs

**Adding pairs:**
- Re-run WFO with new pair set
- Cost sweep changes (different spread/slippage profiles)
- Anchor verification per pair on each broker
- ~1 month of revalidation work

**Removing pairs:**
- Re-run WFO without that pair (different signal pool)
- Worst-fold metric may shift
- ~2 weeks revalidation

In both cases: the deployed config_hash changes, EA `Expected_Config_Hash` must update, etc. Not trivial.

## Cross-broker availability check

Both 5ers and FundedNext offer all 28 pairs as plain symbols (no suffix like "EURUSDi" or "EURUSDm"). This is true of most major prop brokers, but worth verifying for any new broker:

```powershell
# In MT5: View → Symbols → search for each of the 28
# All should appear and be enableable
```

If a broker is missing one or more of the 28, three options:
1. Remove the missing pair(s) from the strategy → requires re-validation (rejected)
2. Use a different broker → simpler
3. Run a reduced-pair sidecar specifically for this broker → operational complexity

Default: pick brokers that offer all 28.

## Symbol naming variations across brokers

Some brokers use suffixes (`EURUSDm`, `EURUSDpro`, `EURUSDi` for ECN/raw spread accounts). The sidecar can handle this via the `mt5_symbol_map` config field, mapping Arc 10's pair names to broker-specific symbols.

Both 5ers and FundedNext use plain naming (no suffix). No mapping needed for current deployment.

## What's not included and why

- **Bitcoin / crypto:** different microstructure, different hours, different volatility regimes. Out of scope.
- **Gold (XAUUSD), Silver (XAGUSD):** metals trade like commodities but with different macro drivers. Not in the core 28. Could be added later as a separate strategy variant if validated.
- **Indices, oil, stocks:** different asset classes entirely.
- **Exotic FX (TRY, BRL, INR):** insufficient broker support and liquidity.

The 28-pair universe is the broadest core-FX universe that:
1. All major prop brokers offer
2. Has clean 14+ years of backtest data
3. Provides enough trade count for meaningful WFO
4. Doesn't introduce liquidity or microstructure heterogeneity

This is the deliberate scope of Arc 10.
