# Phase 3 overfitting audit

Feature source: `data\assets\btcusdt\features\features_BTCUSDT_multiday.parquet`

The split is chronological: 60% train, 20% validation, 20% test.
Directional scores condition on events where the next trade price
changes; the separate movement gate models zero-price-change events.

| Split | Moving events | QRW Brier | Linear-OBI Brier | Linear-market Brier | Neutral Brier | QRW log loss | QRW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3670916 | 0.090324 | 0.101949 | 0.088184 | 0.250000 | 0.317098 | 88.96% |
| validation | 1212609 | 0.091435 | 0.102357 | 0.089494 | 0.250000 | 0.320310 | 88.90% |
| test | 1109338 | 0.125942 | 0.112856 | 0.122674 | 0.250000 | 0.404232 | 82.87% |

## Stability

- Rolling blocks: `8`
- Structural parameters are frozen after the 40% warmup.
- Alpha range: `0.655046` to `0.655046`
- Alpha standard deviation: `0.000000`
- Direction-coupling range: `0.525367` to `0.525367`
- Circular-shift p-value on train: `0.000500`

## Walk-forward

- Expanding-window folds: `3`; mean QRW Brier: `0.134426`
- Mean linear-OBI Brier: `0.114318`
- Mean fair linear-market Brier: `0.123987`
- Pooled QRW minus fair baseline Brier: `0.010037`
- Moving-block 95% interval: `[0.009869, 0.010186]`

## Test uncertainty

- QRW minus linear-OBI Brier: `0.013086`
- 95% bootstrap interval: `[0.012863, 0.013304]`
- QRW minus fair linear-market Brier: `0.003269`
- Fair-baseline 95% interval: `[0.003222, 0.003319]`

## Verdict

The QRW is statistically significantly worse than the fair affine baseline in pooled walk-forward Brier score. No QRW predictive advantage is supported. The final single holdout is reported as a secondary diagnostic and cannot override the pooled walk-forward result.
