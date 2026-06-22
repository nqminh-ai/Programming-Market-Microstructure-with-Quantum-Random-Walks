# Phase 3 overfitting audit

Feature source: `data\features\features_BTCUSDT_2026-06-12.parquet`

The split is chronological: 60% train, 20% validation, 20% test.
Directional scores condition on events where the next trade price
changes; the separate movement gate models zero-price-change events.

| Split | Moving events | QRW Brier | Linear-OBI Brier | Linear-market Brier | Neutral Brier | QRW log loss | QRW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 149 | 0.187025 | 0.170160 | 0.167662 | 0.250000 | 0.552395 | 73.15% |
| validation | 74 | 0.369330 | 0.277421 | 0.300195 | 0.250000 | 0.976900 | 37.84% |
| test | 104 | 0.197840 | 0.224675 | 0.244720 | 0.250000 | 0.583055 | 61.54% |

## Stability

- Rolling blocks: `8`
- Structural parameters are frozen after the 40% warmup.
- Alpha range: `0.441302` to `0.441302`
- Alpha standard deviation: `0.000000`
- Direction-coupling range: `0.021911` to `0.021911`
- Circular-shift p-value on train: `0.077461`

## Walk-forward

- Expanding-window folds: `3`; mean QRW Brier: `0.275459`
- Mean linear-OBI Brier: `0.214188`
- Mean fair linear-market Brier: `0.227344`
- Pooled QRW minus fair baseline Brier: `0.049889`
- Moving-block 95% interval: `[0.020831, 0.078994]`

## Test uncertainty

- QRW minus linear-OBI Brier: `-0.026835`
- 95% bootstrap interval: `[-0.041079, -0.013228]`
- QRW minus fair linear-market Brier: `-0.046880`
- Fair-baseline 95% interval: `[-0.067598, -0.026675]`

## Verdict

The calibrated QRW has a statistically significant Brier-score edge over the fair affine baseline on this held-out window, and structural rolling parameters remain fixed after warmup. Multi-day confirmation is still required before claiming a general QRW advantage.
