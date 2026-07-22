# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `features_BTCUSDT_recent_subset.parquet` (4,000,000 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `08e0865002d5fd1794f1119e89d9d2fccb757504` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 3.588e-05 | 0.0100 | 0.100424 | 0.331274 | 0.8269 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.100424 | 0.331275 | 0.8269 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.100424 | 0.331274 | 0.8269 |
| C_affine | False | n/a | n/a | 0.113292 | 0.370513 | 0.8446 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | -0.012868 | [-0.013771, -0.011940] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | -0.012868 | [-0.013762, -0.011919] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | -0.000000 | [-0.000000, -0.000000] | ✔ |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.100421 | 0.113134 | -0.012714 | [-0.013633, -0.011788] | ✔ |
| 3 | 0.100424 | 0.113292 | -0.012868 | [-0.013781, -0.011960] | ✔ |
| 4 | 0.100423 | 0.113350 | -0.012927 | [-0.013898, -0.011999] | ✔ |
| 5 | 0.100425 | 0.113378 | -0.012953 | [-0.013866, -0.012061] | ✔ |
| 6 | 0.100428 | 0.113255 | -0.012828 | [-0.013720, -0.011913] | ✔ |
| 8 | 0.100429 | 0.113242 | -0.012813 | [-0.013753, -0.011892] | ✔ |

## Verdict

The phase (interference) term contributes a statistically significant Brier improvement (A_full beats B_refit). The phase-free windowed density-matrix model (B_refit) *does* beat the independently-fit affine baseline (B_refit vs C_affine), so the edge comes from windowing/decoherence, not from quantum interference. At the reported protocol's fold count, the full quantum model does significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

QRW wins (or ties) the affine baseline across every fold count tested -- the advantage is robust to this evaluation hyperparameter.
