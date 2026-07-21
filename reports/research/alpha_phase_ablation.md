# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `features_BTCUSDT_recent_subset.parquet` (4,000,000 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `3af0ded78e1a65998cb5ba3a7b33f2284b085eee` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 3.588e-05 | 0.0100 | 0.100424 | 0.331279 | 0.8269 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.100424 | 0.331279 | 0.8269 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.100424 | 0.331279 | 0.8269 |
| C_affine | False | n/a | n/a | 0.113292 | 0.370513 | 0.8446 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | -0.012867 | [-0.013765, -0.011939] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | -0.012867 | [-0.013765, -0.011914] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | +0.000000 | [+0.000000, +0.000000] | ✔ |

## Forced phase sweep (A's structural fit, alpha_phase overridden)

| alpha_phase | pooled Brier | log loss | accuracy |
|---:|---:|---:|---:|
| 0.000 | 0.100424 | 0.331279 | 0.8269 |
| 0.050 | 0.100425 | 0.331286 | 0.8269 |
| 0.100 | 0.100427 | 0.331306 | 0.8269 |
| 0.250 | 0.100440 | 0.331450 | 0.8269 |
| 0.500 | 0.100491 | 0.331971 | 0.8269 |
| 1.000 | 0.100735 | 0.334173 | 0.8269 |
| 2.000 | 0.102337 | 0.344558 | 0.8269 |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.100419 | 0.113134 | -0.012716 | [-0.013635, -0.011789] | ✔ |
| 3 | 0.100424 | 0.113292 | -0.012867 | [-0.013777, -0.011962] | ✔ |
| 4 | 0.102915 | 0.113350 | -0.010435 | [-0.011411, -0.009539] | ✔ |
| 5 | 0.142450 | 0.113378 | +0.029072 | [+0.027778, +0.030356] | ✘ |
| 6 | 0.158408 | 0.113255 | +0.045152 | [+0.043716, +0.046624] | ✘ |
| 8 | 0.177498 | 0.113242 | +0.064256 | [+0.062657, +0.065853] | ✘ |

## Verdict

The phase (interference) term shows **no** statistically significant Brier benefit (A_full vs B_refit CI spans 0): on this data the quantum-interference mechanism does not drive the advantage. The phase-free windowed density-matrix model (B_refit) *does* beat the independently-fit affine baseline (B_refit vs C_affine), so the edge comes from windowing/decoherence, not from quantum interference. At the reported protocol's fold count, the full quantum model does significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

**The QRW-vs-affine verdict is not robust to the fold count.** QRW wins significantly at folds {2, 3, 4} but loses significantly at folds {5, 6, 8}, while the affine baseline stays stable. A verdict that flips sign under an arbitrary evaluation hyperparameter is an artifact of that choice, not evidence of a genuine predictive advantage.
