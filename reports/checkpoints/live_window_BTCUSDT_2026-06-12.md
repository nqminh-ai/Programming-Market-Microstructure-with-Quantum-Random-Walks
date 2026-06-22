# Synchronized Live Window: BTCUSDT 2026-06-12

**Updated after causal feature and benchmark rebuild:** 2026-06-14

## Data

| Metric | Observed |
|---|---:|
| Raw trades | 1,939 |
| Causal-cleaned trades | 1,908 |
| Feature rows | 1,908 |
| Imbalance source | causal lagged trade-volume proxy |
| Outlier alignment | past-only regime confirmation |

## Calibration

| Metric | Observed |
|---|---:|
| Structural rows | 763 |
| Structural moving events | 103 |
| Structural fit events | 77 |
| Selection validation events | 26 |
| Structural parameters | 6 |
| Events per parameter | 12.8333 |
| Bias update rows | 1,145 |
| Bias update moving events | 224 |
| Validation included in final structural refit | No |
| Bias update reuses warmup | No |
| Low-sample warning | Yes |

`low_sample_warning=true`, so this window is suitable for pipeline validation
but not robust structural inference.

## Predictive Audit

- QRW test Brier: `0.197840`
- Fair affine test Brier: `0.244720`
- QRW minus fair affine: `-0.046880`
- 95% interval: `[-0.067598, -0.026675]`
- Pooled walk-forward delta: `+0.049889`
- Walk-forward interval: `[0.020831, 0.078994]`

QRW wins the final holdout but loses the pooled walk-forward comparison.
The result is exploratory and does not establish a stable predictive edge.
