# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `3af0ded78e1a65998cb5ba3a7b33f2284b085eee` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 1.619e-04 | 0.0100 | 0.094983 | 0.346976 | 0.8979 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.094994 | 0.347010 | 0.8979 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.094983 | 0.346976 | 0.8979 |
| C_affine | False | n/a | n/a | 0.085159 | 0.299274 | 0.8963 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | +0.009825 | [+0.008932, +0.010728] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | +0.009836 | [+0.008962, +0.010728] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | -0.000011 | [-0.000011, -0.000011] | ✔ |

## Forced phase sweep (A's structural fit, alpha_phase overridden)

| alpha_phase | pooled Brier | log loss | accuracy |
|---:|---:|---:|---:|
| 0.000 | 0.094983 | 0.346976 | 0.8979 |
| 0.050 | 0.094987 | 0.346987 | 0.8979 |
| 0.100 | 0.094998 | 0.347022 | 0.8979 |
| 0.250 | 0.095074 | 0.347267 | 0.8979 |
| 0.500 | 0.095347 | 0.348144 | 0.8979 |
| 1.000 | 0.096459 | 0.351676 | 0.8979 |
| 2.000 | 0.101222 | 0.366148 | 0.8979 |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.094828 | 0.085104 | +0.009724 | [+0.008859, +0.010614] | ✘ |
| 3 | 0.094983 | 0.085159 | +0.009825 | [+0.008917, +0.010665] | ✘ |
| 4 | 0.094843 | 0.085095 | +0.009748 | [+0.008885, +0.010617] | ✘ |
| 5 | 0.094747 | 0.085072 | +0.009675 | [+0.008742, +0.010556] | ✘ |
| 6 | 0.094672 | 0.085013 | +0.009659 | [+0.008762, +0.010507] | ✘ |
| 8 | 0.094619 | 0.084933 | +0.009687 | [+0.008802, +0.010622] | ✘ |

## Verdict

The phase (interference) term contributes a statistically significant Brier improvement (A_full beats B_refit). The phase-free windowed model shows no significant edge over the affine baseline (B_refit vs C_affine). At the reported protocol's fold count, the full quantum model does not significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

QRW does not significantly beat the affine baseline at any fold count tested.
