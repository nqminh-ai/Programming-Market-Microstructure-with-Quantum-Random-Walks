# Phase 3 overfitting audit

Feature source: `D:\Project made by me\AI_Quantum\Quantum 1\data\assets\bnbusdt\features\features_BNBUSDT_2026-05-13.parquet`

The split is chronological: 60% train, 20% validation, 20% test.
Directional scores condition on events where the next trade price
changes; the separate movement gate models zero-price-change events.

| Split | Moving events | QRW Brier | Linear-OBI Brier | Linear-market Brier | Neutral Brier | QRW log loss | QRW accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 103825 | 0.177378 | 0.194380 | 0.190790 | 0.250000 | 0.538377 | 72.98% |
| validation | 36535 | 0.160469 | 0.196271 | 0.179868 | 0.250000 | 0.502387 | 79.37% |
| test | 37722 | 0.183213 | 0.211101 | 0.205073 | 0.250000 | 0.551301 | 72.17% |

## Stability

- Rolling blocks: `8`
- Structural parameters are frozen after the 40% warmup.
- Alpha range: `0.000000` to `0.000000`
- Alpha standard deviation: `0.000000`
- Direction-coupling range: `-0.582379` to `-0.582379`
- Circular-shift p-value on train: `0.090909`

## Walk-forward

- Expanding-window folds: `3`; mean QRW Brier: `0.173193`
- Mean linear-OBI Brier: `0.201586`
- Mean fair linear-market Brier: `0.192285`
- Pooled QRW minus fair baseline Brier: `0.228808`
- Moving-block 95% interval: `[0.225014, 0.230680]`

## Test uncertainty

- QRW minus linear-OBI Brier: `-0.027888`
- 95% bootstrap interval: `[-0.028891, -0.026774]`
- QRW minus fair linear-market Brier: `-0.021860`
- Fair-baseline 95% interval: `[-0.023521, -0.020691]`

## Verdict

The calibrated QRW has a statistically significant Brier-score edge over the fair affine baseline on this held-out window, and structural rolling parameters remain fixed after warmup. Multi-day confirmation is still required before claiming a general QRW advantage.
