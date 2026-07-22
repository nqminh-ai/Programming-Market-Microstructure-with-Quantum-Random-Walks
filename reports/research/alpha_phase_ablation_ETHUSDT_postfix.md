# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `08e0865002d5fd1794f1119e89d9d2fccb757504` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 1.619e-04 | 0.0100 | 0.094726 | 0.346446 | 0.8979 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.094737 | 0.346480 | 0.8979 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.094726 | 0.346446 | 0.8979 |
| C_affine | False | n/a | n/a | 0.085159 | 0.299274 | 0.8963 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | +0.009568 | [+0.008682, +0.010449] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | +0.009579 | [+0.008695, +0.010474] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | -0.000011 | [-0.000011, -0.000011] | ✔ |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.094783 | 0.085104 | +0.009679 | [+0.008818, +0.010566] | ✘ |
| 3 | 0.094726 | 0.085159 | +0.009568 | [+0.008679, +0.010409] | ✘ |
| 4 | 0.094663 | 0.085095 | +0.009569 | [+0.008711, +0.010437] | ✘ |
| 5 | 0.094633 | 0.085072 | +0.009561 | [+0.008645, +0.010449] | ✘ |
| 6 | 0.094612 | 0.085013 | +0.009599 | [+0.008704, +0.010445] | ✘ |
| 8 | 0.094591 | 0.084933 | +0.009658 | [+0.008780, +0.010589] | ✘ |

## Verdict

The phase (interference) term is **practically zero** (-1.09e-05 Brier, below the 1e-04 threshold): despite large-N significance it is orders of magnitude smaller than the main edge, so quantum interference does not drive the result. The phase-free windowed model shows no significant edge over the affine baseline (B_refit vs C_affine). At the reported protocol's fold count, the full quantum model does not significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

QRW does not significantly beat the affine baseline at any fold count tested.
