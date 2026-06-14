# Phase 3 Checkpoint

**Updated:** 2026-06-14

## Engineering

| Check | Status | Observed |
|---|---|---:|
| Full automated test suite | PASS | 61/61 |
| Probability normalization | PASS | tested to 1e-12 |
| Density-matrix trace | PASS | tested to 1e-12 |
| Local sampled moves | PASS | increments are in {-1, 0, +1} |
| Performance target | PASS | 1,000 paths x 1,000 steps under 60 seconds |

## Overfitting Controls

| Control | Status |
|---|---|
| Outlier cleaning uses future ticks | FIXED; boundary-prefix and `t+1` mutation tests pass |
| Structural final refit includes validation | FIXED |
| Bias update reuses structural warmup | FIXED |
| Calibration metadata records split integrity | PASS |
| Historical June 1-7 artifacts use causal pipeline | PASS; rebuilt 2026-06-14 |
| Fresh untouched confirmatory holdout | MISSING |

Current calibration records:

- `final_refit_includes_validation=false`
- `bias_update_reuses_warmup=false`
- Structural adaptive fit: `77` moving events for `6` parameters
- Events per structural parameter: `12.8333`
- `low_sample_warning=true`

## Post-Fix Predictive Result

The causal-cleaned June 12 live artifact contains `1,908` rows.

| Comparison | QRW minus baseline Brier | 95% interval | Result |
|---|---:|---:|---|
| Final holdout vs fair affine | -0.046880 | [-0.067598, -0.026675] | QRW better |
| Pooled walk-forward vs fair affine | +0.049889 | [0.020831, 0.078994] | QRW worse |

**Verdict:** engineering pass, but the predictive result is unstable across
evaluation schemes and does not establish a general edge. Historical June 1-7
features were rebuilt for development on June 14, 2026; archived pre-fix audits
may not be cited. Fresh untouched dates are still required for confirmation.

Invalidated artifacts are retained under
`reports/archive/invalidated_2026-06-13/` and
`results/archive/invalidated_2026-06-13/`.
