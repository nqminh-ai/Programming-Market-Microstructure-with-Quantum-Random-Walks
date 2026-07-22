# Strong-baseline comparison — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory, not confirmatory.

- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 rows)
- Events train/val/test: 221,253 / 108,906 / 102,164
- Git commit: `0a502fdb24786521c596a21332f4ec5d72a87736` · Python 3.14.5 · seed 2026

## Test-set scores (ranked by Brier, lower is better)

| Rank | Model | Brier | log loss | accuracy |
|---:|---|---:|---:|---:|
| 1 | OrderFlow AR(5) | 0.065707 | 0.241882 | 0.9198 |
| 2 | Logistic L2 + Pairwise | 0.068671 | 0.253996 | 0.9113 |
| 3 | QRW Directional Link | 0.073910 | 0.269883 | 0.8720 |
| 4 | Marked Hawkes Logit | 0.082403 | 0.290310 | 0.8744 |
| 5 | Nonlinear Calibrated | 0.099484 | 0.339302 | 0.8814 |
| 6 | Logistic L2 (5F) | 0.099629 | 0.337287 | 0.8713 |
| 7 | Windowed-QRW (density matrix) **(QRW)** | 0.100145 | 0.343180 | 0.8713 |

## Windowed-QRW vs each baseline (paired Brier diff, 95% block-bootstrap)

Negative mean = QRW has the lower (better) Brier.

| Baseline | mean (QRW−base) | 95% CI | QRW better? |
|---|---:|---|:--:|
| Logistic L2 (5F) | +0.000516 | [+0.000178, +0.000845] | ✘ worse |
| Nonlinear Calibrated | +0.000661 | [-0.000101, +0.001409] | — tie |
| Marked Hawkes Logit | +0.017742 | [+0.017132, +0.018343] | ✘ worse |
| QRW Directional Link | +0.026234 | [+0.025038, +0.027470] | ✘ worse |
| Logistic L2 + Pairwise | +0.031474 | [+0.030095, +0.032847] | ✘ worse |
| OrderFlow AR(5) | +0.034438 | [+0.032743, +0.036111] | ✘ worse |

## Verdict

Windowed-QRW is significantly WORSE than the strongest baseline ('OrderFlow AR(5)') on ETHUSDT: the edge over affine does not survive a competitive classical model.
