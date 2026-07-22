# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `bnb_combined.parquet` (4,000,000 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `08e0865002d5fd1794f1119e89d9d2fccb757504` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 1.104e-05 | 0.0100 | 0.170243 | 0.522438 | 0.7643 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.170242 | 0.522437 | 0.7643 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.170243 | 0.522438 | 0.7643 |
| C_affine | False | n/a | n/a | 0.181382 | 0.546577 | 0.7543 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | -0.011139 | [-0.011875, -0.010354] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | -0.011140 | [-0.011906, -0.010310] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | +0.000000 | [+0.000000, +0.000000] | ✔ |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.170215 | 0.181623 | -0.011408 | [-0.012196, -0.010651] | ✔ |
| 3 | 0.170243 | 0.181382 | -0.011139 | [-0.011962, -0.010327] | ✔ |
| 4 | 0.170201 | 0.181256 | -0.011055 | [-0.011809, -0.010314] | ✔ |
| 5 | 0.170216 | 0.181256 | -0.011040 | [-0.011822, -0.010272] | ✔ |
| 6 | 0.170195 | 0.181150 | -0.010954 | [-0.011754, -0.010142] | ✔ |
| 8 | 0.170175 | 0.181149 | -0.010974 | [-0.011736, -0.010213] | ✔ |

## Verdict

The phase (interference) term is **practically zero** (+3.15e-07 Brier, below the 1e-04 threshold): despite large-N significance it is orders of magnitude smaller than the main edge, so quantum interference does not drive the result. The phase-free windowed density-matrix model (B_refit) *does* beat the independently-fit affine baseline (B_refit vs C_affine), so the edge comes from windowing/decoherence, not from quantum interference. At the reported protocol's fold count, the full quantum model does significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

QRW wins (or ties) the affine baseline across every fold count tested -- the advantage is robust to this evaluation hyperparameter.
