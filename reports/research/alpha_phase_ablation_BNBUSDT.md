# Alpha-phase ablation — isolating the quantum-interference contribution

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory mechanism diagnostic, not a confirmatory run. Do not relabel as confirmatory evidence.

- Feature file: `bnb_combined.parquet` (4,000,000 rows)
- Walk-forward folds: 3; block bootstrap samples: 2000 (block size 16)
- Git commit: `3af0ded78e1a65998cb5ba3a7b33f2284b085eee` · Python 3.14.5 · seed 2026

## Configurations

| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |
|---|---|---:|---:|---:|---:|---:|
| A_full | True | 1.104e-05 | 0.0100 | 0.170327 | 0.522600 | 0.7643 |
| B_refit | True | 0.000e+00 | 0.0100 | 0.170326 | 0.522599 | 0.7643 |
| B_posthoc | True | 0.000e+00 | 0.0100 | 0.170327 | 0.522600 | 0.7643 |
| C_affine | False | n/a | n/a | 0.181382 | 0.546577 | 0.7543 |

## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)

A negative mean = the first-listed config has the lower (better) Brier.

| Comparison | Meaning | mean (a−b) | 95% CI | significant |
|---|---|---:|---|:--:|
| A_full_vs_C_affine | total quantum vs affine baseline (windowing + phase) | -0.011056 | [-0.011795, -0.010267] | ✔ |
| B_refit_vs_C_affine | phase-free quantum vs affine baseline | -0.011056 | [-0.011806, -0.010245] | ✔ |
| A_full_vs_B_refit | pure phase (interference) contribution | +0.000000 | [+0.000000, +0.000000] | ✔ |

## Forced phase sweep (A's structural fit, alpha_phase overridden)

| alpha_phase | pooled Brier | log loss | accuracy |
|---:|---:|---:|---:|
| 0.000 | 0.170327 | 0.522600 | 0.7643 |
| 0.050 | 0.170328 | 0.522605 | 0.7643 |
| 0.100 | 0.170334 | 0.522619 | 0.7643 |
| 0.250 | 0.170371 | 0.522720 | 0.7643 |
| 0.500 | 0.170505 | 0.523080 | 0.7643 |
| 1.000 | 0.171051 | 0.524542 | 0.7643 |
| 2.000 | 0.173411 | 0.530685 | 0.7643 |

## Fold-count robustness (A_full vs affine baseline)

Negative edge = QRW has the lower (better) Brier. The affine baseline is refit per fold and stays stable; a sign flip in the edge exposes a non-robust verdict.

| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---|:--:|
| 2 | 0.170283 | 0.181623 | -0.011339 | [-0.012128, -0.010575] | ✔ |
| 3 | 0.170327 | 0.181382 | -0.011056 | [-0.011870, -0.010254] | ✔ |
| 4 | 0.170397 | 0.181256 | -0.010860 | [-0.011617, -0.010097] | ✔ |
| 5 | 0.170394 | 0.181256 | -0.010862 | [-0.011665, -0.010097] | ✔ |
| 6 | 0.170429 | 0.181150 | -0.010721 | [-0.011519, -0.009905] | ✔ |
| 8 | 0.170432 | 0.181149 | -0.010718 | [-0.011488, -0.009950] | ✔ |

## Verdict

The phase (interference) term shows **no** statistically significant Brier benefit (A_full vs B_refit CI spans 0): on this data the quantum-interference mechanism does not drive the advantage. The phase-free windowed density-matrix model (B_refit) *does* beat the independently-fit affine baseline (B_refit vs C_affine), so the edge comes from windowing/decoherence, not from quantum interference. At the reported protocol's fold count, the full quantum model does significantly beat the affine baseline (A_full vs C_affine) -- but see the fold-sensitivity table for whether that verdict is robust.

QRW wins (or ties) the affine baseline across every fold count tested -- the advantage is robust to this evaluation hyperparameter.
