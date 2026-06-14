# Phase 2 synthetic imbalance review

> **STALE PREDICTIVE RESULTS:** June 1-7 processed/features artifacts were built
> with the former future-aware outlier filter. The proxy implementation itself
> remains causal, but all multiday edge numbers in this report are invalid
> until the seven days are rebuilt. These dates are development-only afterward.

## Status

The trade-only imbalance fallback is complete and reproducible.

| Item | Status |
|---|---|
| Signed buy/sell volume formula | PASS |
| Trailing event-time rolling window | PASS |
| Current trade excluded by one-trade lag | PASS |
| Rolling state reset at gap segments | PASS |
| Warm-up rows explicitly marked with `obi_valid` | PASS |
| Configurable source and window | PASS |
| Metadata provenance | PASS |
| Historical artifacts rebuilt | PASS |
| Phase 2 checkpoint | PASS |

## Definition

For trade index `t`, window length `W=100`, and aggressive trade sign
`s_i in {+1,-1}`:

```text
trade_imbalance_t =
    sum(i=t-W..t-1, s_i * quantity_i)
    ----------------------------------
    sum(i=t-W..t-1, quantity_i)
```

The window excludes trade `t`. Calculation restarts after each detected market
data gap. The first 20 observations of each segment are warm-up rows with
`obi_valid=False`.

When LOB data is absent, the model-compatible `obi` column contains this proxy.
When LOB data exists, `obi` remains the real top-level order-book imbalance.

## Historical artifacts

Seven BTCUSDT daily feature files from June 1 through June 7, 2026 were rebuilt:

- Historical feature rows: `47,627,346`
- Proxy source files: `7`
- Minimum valid coverage: above `99.9995%`
- Finite feature matrices: `7/7`
- Feature rows equal processed rows: exact match

The synchronized June 12, 2026 live file remains a real-LOB artifact, giving the
workspace seven proxy files and one real-LOB file.

## Proxy validation

On the 120-second synchronized live sample, correlation between trade imbalance
and real LOB OBI was:

| Window | Correlation | MAE |
|---:|---:|---:|
| 20 trades | 0.8012 | 0.3205 |
| 50 trades | 0.6985 | 0.4285 |
| 100 trades | 0.5925 | 0.5583 |
| 200 trades | 0.4611 | 0.6783 |
| 500 trades | 0.4334 | 0.7551 |

Window 100 remains the default as a configurable compromise. The live comparison
is too short to optimize the window without overfitting.

On June 1, window 100 has standard deviation `0.8821`; `62.16%` of valid values
have absolute magnitude above `0.95`. This reflects highly persistent aggressive
trade flow and means the proxy is often saturated. It must not be interpreted as
literal resting-book imbalance.

## Phase 3 handoff

Calibration on the first 250,000 rows of
`features_BTCUSDT_2026-06-07.parquet` produced:

- `gamma = 0.0710467`
- `alpha_obi = 1.06775`
- `alpha_direction = 1.11395`
- `alpha_obi_change = -0.0685681`
- `alpha_abs_obi = -0.159547`
- `gamma_intensity = -2.0`
- selected regularization = `0.0001`
- validation Brier = `0.073285`
- 1,000 paths x 1,000 steps = `0.0216` seconds

The adaptive model uses log trade intensity to modulate event decoherence and
uses OBI, tick direction, OBI change, and absolute OBI in the coin signal.

The stricter multi-day audit fits structural parameters on June 1-4 and tests
June 5-7 without refitting them. Across `4,587,200` held-out moving events:

- QRW minus nonlinear logistic Brier = `-0.013822`
- Moving-block 95% interval = `[-0.014108, -0.013527]`
- QRW minus pairwise logistic Brier = `+0.003401`

Thus adaptive-decoherence QRW has a reproducible predictive edge over standard
nonlinear logistic regression using the same five raw features. Pairwise
logistic remains stronger, so the result is not a universal model advantage.

Artifacts:

- `results/calibrated_params_synthetic_obi.json`
- `reports/performance_benchmark_synthetic_obi.txt`
- `reports/phase3_multiday_edge_audit.md`
- `reports/phase3_adaptive_decoherence_audit.md`
- `results/calibrated_params_adaptive_multiday.json`

## Conclusion

Synthetic trade imbalance fixes the missing historical input and allows Phase 2
and Phase 3 to run on all seven historical days. It is a causal order-flow proxy,
not a substitute measurement of historical LOB depth. Comparisons between proxy
and real-LOB experiments must remain separate.
