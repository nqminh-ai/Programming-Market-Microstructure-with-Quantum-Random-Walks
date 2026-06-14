# Synchronized live window: BTCUSDT 2026-06-12

> **SUPERSEDED ON 2026-06-13:** The collection facts remain useful, but the
> original cleaning/calibration values below predate the causal-cleaning and
> disjoint-calibration fixes. The rebuilt artifact has 1,908 feature rows,
> 78 structural fit events for 6 parameters, and no edge over the fair affine
> baseline. See `reports/overfitting_critical_fixes_2026-06-13.md`.

## Collection

| Metric | Observed |
|---|---:|
| Window duration | 120 seconds |
| Trades collected | 1,939 |
| Processed trades | 1,937 |
| LOB snapshots | 113 |
| WebSocket reconnects | 0 |
| Feature rows | 1,937 |
| LOB match coverage | 100% |

## OBI quality

| Metric | Observed |
|---|---:|
| Nonzero OBI rows | 100% |
| OBI variance | 0.748599 |
| OBI minimum | -0.998369 |
| OBI maximum | 0.996845 |
| Unique OBI values | 100 |

## Phase 3 calibration

| Parameter | Value |
|---|---:|
| rho_1 | 0.868651049502 |
| gamma | 0.140813788444 |
| alpha_obi | 5.000000000000 |
| alpha_direction | -4.359417636830 |
| alpha_obi_change | -0.034860068333 |
| alpha_abs_obi | 0.039755026446 |
| gamma_intensity | -0.618190655470 |
| obi_bias | -0.832508657804 |
| Structural warmup rows | 774 |
| Selected regularization | 0.0001 |
| Validation Brier | 0.034351519981 |
| Validation log loss | 0.153487275564 |
| Calibration status | fixed_structure_bias_updated |

## Input sensitivity at 50 steps

| Input | Mean position | Variance |
|---|---:|---:|
| Observed OBI | -0.677213 | 175.535133 |
| OBI set to zero | -0.418333 | 170.962235 |
| OBI sign flipped | 0.537015 | 203.571974 |

- L1 distance, observed versus zero OBI: `0.027176`
- L1 distance, observed versus sign-flipped OBI: `0.106523`
- Sampled path mean at 50 steps, observed OBI: `-15.58264`
- Sampled path mean at 50 steps, zero OBI: `37.43898`
- Sampled path mean at 50 steps, sign-flipped OBI: `40.75172`
- Maximum absolute sampled increment: `1`

The trial confirms synchronized OBI coverage and input-sensitive distributions.
The exact density rows and sampled paths use the two documented simulation
protocols. The live window is too short for the six-parameter adaptive model:
`alpha_obi` reaches its optimization bound. It remains a pipeline and
input-sensitivity check; predictive claims use the multi-day historical audit.
