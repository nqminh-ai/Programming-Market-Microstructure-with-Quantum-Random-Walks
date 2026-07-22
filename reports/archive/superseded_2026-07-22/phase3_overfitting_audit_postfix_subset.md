# Phase 3 overfitting audit

Feature source: `D:\Project made by me\AI_Quantum\Quantum 1\data\assets\btcusdt\features\features_BTCUSDT_recent_subset.parquet`

The split is chronological: 60% train, 20% validation, 20% test.
Directional scores condition on events where the next trade price
changes; the separate movement gate models zero-price-change events.

| Split | Moving events | QRW Brier | Linear-OBI Brier | Linear-market Brier | Neutral Brier | QRW log loss | QRW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 416381 | 0.112050 | 0.120231 | 0.120206 | 0.250000 | 0.374555 | 81.05% |
| validation | 133609 | 0.117942 | 0.122770 | 0.122995 | 0.250000 | 0.387353 | 78.85% |
| test | 137878 | 0.094503 | 0.102806 | 0.102891 | 0.250000 | 0.334352 | 85.94% |

## Stability

- Rolling blocks: `8`
- Structural parameters are frozen after the 40% warmup.
- Alpha range: `0.000000` to `0.000000`
- Alpha standard deviation: `0.000000`
- Direction-coupling range: `-1.031917` to `-1.031917`
- Circular-shift p-value on train: `0.000500`

## Walk-forward

- Expanding-window folds: `3`; mean QRW Brier: `0.106031`
- Mean linear-OBI Brier: `0.113278`
- Mean fair linear-market Brier: `0.113396`
- Pooled QRW minus fair baseline Brier: `-0.007383`
- Moving-block 95% interval: `[-0.008306, -0.006482]`

## Test uncertainty

- QRW minus linear-OBI Brier: `-0.008304`
- 95% bootstrap interval: `[-0.009081, -0.007501]`
- QRW minus fair linear-market Brier: `-0.008389`
- Fair-baseline 95% interval: `[-0.009187, -0.007600]`

## Verdict

The fixed-structure QRW has a statistically significant pooled walk-forward Brier edge over the fair affine baseline. The final holdout fold remains separately reported. The independent multi-day historical proxy result is archived under reports/archive/invalidated_2026-06-13/.
