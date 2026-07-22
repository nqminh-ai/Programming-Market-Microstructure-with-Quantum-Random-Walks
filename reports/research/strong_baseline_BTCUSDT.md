# Strong-baseline comparison — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory, not confirmatory.

- Feature file: `features_BTCUSDT_recent_subset.parquet` (4,000,000 rows)
- Events train/val/test: 350,727 / 166,670 / 170,471
- Git commit: `0a502fdb24786521c596a21332f4ec5d72a87736` · Python 3.14.5 · seed 2026

## Test-set scores (ranked by Brier, lower is better)

| Rank | Model | Brier | log loss | accuracy |
|---:|---|---:|---:|---:|
| 1 | Logistic L2 + Pairwise | 0.049647 | 0.211074 | 0.9432 |
| 2 | OrderFlow AR(5) | 0.054654 | 0.211252 | 0.9349 |
| 3 | QRW Directional Link | 0.089998 | 0.317223 | 0.8542 |
| 4 | Windowed-QRW (density matrix) **(QRW)** | 0.101923 | 0.353274 | 0.8410 |
| 5 | Nonlinear Calibrated | 0.105741 | 0.346246 | 0.8535 |
| 6 | Marked Hawkes Logit | 0.107750 | 0.357683 | 0.8537 |
| 7 | Logistic L2 (5F) | 0.109697 | 0.364036 | 0.8543 |

## Windowed-QRW vs each baseline (paired Brier diff, 95% block-bootstrap)

Negative mean = QRW has the lower (better) Brier.

| Baseline | mean (QRW−base) | 95% CI | QRW better? |
|---|---:|---|:--:|
| Logistic L2 (5F) | -0.007774 | [-0.009177, -0.006253] | ✔ better |
| Marked Hawkes Logit | -0.005826 | [-0.007354, -0.004355] | ✔ better |
| Nonlinear Calibrated | -0.003817 | [-0.005007, -0.002548] | ✔ better |
| QRW Directional Link | +0.011926 | [+0.011165, +0.012728] | ✘ worse |
| OrderFlow AR(5) | +0.047269 | [+0.045932, +0.048586] | ✘ worse |
| Logistic L2 + Pairwise | +0.052277 | [+0.050577, +0.053980] | ✘ worse |

## Verdict

Windowed-QRW is significantly WORSE than the strongest baseline ('Logistic L2 + Pairwise') on BTCUSDT: the edge over affine does not survive a competitive classical model.
