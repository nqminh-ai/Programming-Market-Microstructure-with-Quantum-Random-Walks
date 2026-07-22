# Strong-baseline comparison — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory, not confirmatory.

- Feature file: `bnb_combined.parquet` (4,000,000 rows)
- Events train/val/test: 495,055 / 227,834 / 203,462
- Git commit: `0a502fdb24786521c596a21332f4ec5d72a87736` · Python 3.14.5 · seed 2026

## Test-set scores (ranked by Brier, lower is better)

| Rank | Model | Brier | log loss | accuracy |
|---:|---|---:|---:|---:|
| 1 | OrderFlow AR(5) | 0.146578 | 0.462689 | 0.8047 |
| 2 | Logistic L2 + Pairwise | 0.162804 | 0.500879 | 0.7299 |
| 3 | QRW Directional Link | 0.163851 | 0.501723 | 0.7194 |
| 4 | Windowed-QRW (density matrix) **(QRW)** | 0.176656 | 0.535877 | 0.7437 |
| 5 | Nonlinear Calibrated | 0.192934 | 0.570032 | 0.7286 |
| 6 | Marked Hawkes Logit | 0.193792 | 0.576950 | 0.7177 |
| 7 | Logistic L2 (5F) | 0.200128 | 0.589521 | 0.7193 |

## Windowed-QRW vs each baseline (paired Brier diff, 95% block-bootstrap)

Negative mean = QRW has the lower (better) Brier.

| Baseline | mean (QRW−base) | 95% CI | QRW better? |
|---|---:|---|:--:|
| Logistic L2 (5F) | -0.023472 | [-0.024893, -0.022154] | ✔ better |
| Marked Hawkes Logit | -0.017136 | [-0.018382, -0.015953] | ✔ better |
| Nonlinear Calibrated | -0.016278 | [-0.017495, -0.015037] | ✔ better |
| QRW Directional Link | +0.012804 | [+0.011980, +0.013583] | ✘ worse |
| Logistic L2 + Pairwise | +0.013852 | [+0.012945, +0.014786] | ✘ worse |
| OrderFlow AR(5) | +0.030078 | [+0.029476, +0.030641] | ✘ worse |

## Verdict

Windowed-QRW is significantly WORSE than the strongest baseline ('OrderFlow AR(5)') on BNBUSDT: the edge over affine does not survive a competitive classical model.
