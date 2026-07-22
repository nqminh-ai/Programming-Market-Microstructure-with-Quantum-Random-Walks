# Full-dataset directional confirmation — BTCUSDT_full

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory, closes report limitation #6 for the directional endpoint.

- Feature file: `features_BTCUSDT_multiday.parquet` (32,439,057 rows, 1070 MB in memory via column-subset + float32 downcast)
- quantum_improved=True · alpha_phase=-1.219e-05 · gamma=0.0100
- Git commit: `db945c0ea44f41c911ff5213dc2b137e79a93701` · Python 3.14.5

## Post-fix QRW vs affine on the full dataset

Negative edge = QRW has the lower (better) Brier. Compare against the stale pre-fix figure **+0.049889** (reported before the Phase 2 bias fix, never reproducible post-fix).

| folds | events | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---:|---|:--:|
| 3 | 3,540,565 | 0.087771 | 0.100862 | -0.013091 | [-0.013320, -0.012868] | ✔ |
| 5 | 3,540,563 | 0.087770 | 0.100541 | -0.012771 | [-0.013003, -0.012539] | ✔ |

## Note

On the full dataset the post-fix windowed-QRW beats the affine baseline stably across every fold count tested. This only restates the §5b affine comparison at full scale; §5c already shows the windowed-QRW loses to competitive classical baselines (OrderFlow AR(5), Logistic+Pairwise) regardless, so this does not change the overall verdict. The old +0.049889 was an artifact of the pre-Phase-2 bias bug.
