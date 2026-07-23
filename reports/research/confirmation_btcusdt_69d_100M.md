# Full-dataset directional confirmation — BTCUSDT_69d_100M

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory, closes report limitation #6 for the directional endpoint.

- Feature file: `features_BTCUSDT_69d.parquet` (100,000,000 rows, 3300 MB in memory via column-subset + float32 downcast)
- quantum_improved=True · alpha_phase=-1.227e-05 · gamma=0.0100
- Git commit: `5a558c8fe84c2cf2866bbd5ba226c82dfcc1dc82` · Python 3.14.5

## Post-fix QRW vs affine on the full dataset

Negative edge = QRW has the lower (better) Brier. Compare against the stale pre-fix figure **+0.049889** (reported before the Phase 2 bias fix, never reproducible post-fix).

| folds | events | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |
|---:|---:|---:|---:|---:|---|:--:|
| 2 | 11,886,739 | 0.077026 | 0.092305 | -0.015280 | [-0.015455, -0.015101] | ✔ |
| 3 | 11,886,739 | 0.077025 | 0.091791 | -0.014766 | [-0.014954, -0.014570] | ✔ |
| 5 | 11,886,737 | 0.077026 | 0.090783 | -0.013758 | [-0.013934, -0.013587] | ✔ |

## Note

On the full dataset the post-fix windowed-QRW beats the affine baseline stably across every fold count tested. This only restates the §5b affine comparison at full scale; §5c already shows the windowed-QRW loses to competitive classical baselines (OrderFlow AR(5), Logistic+Pairwise) regardless, so this does not change the overall verdict. The old +0.049889 was an artifact of the pre-Phase-2 bias bug.
