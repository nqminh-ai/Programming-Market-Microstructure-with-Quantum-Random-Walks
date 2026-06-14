# Adaptive-decoherence QRW edge audit

> **STALE / INVALIDATED 2026-06-13:** Inputs were built with the former
> `shift(-1)` outlier filter, and June 5-7 has been repeatedly inspected.
> Do not cite the edge or confidence intervals as out-of-sample evidence.

Model selection uses June 1-3 for fitting and June 4 for validation.
June 5-7 are prequential tests. All models receive the same five causal
raw features: OBI, current tick direction, OBI change, absolute OBI,
and log trade intensity. QRW uses intensity to modulate decoherence.

| Day | Events | Adaptive QRW Brier | Logistic Brier | Pairwise logistic Brier | bias |
|---|---:|---:|---:|---:|---:|
| 2026-06-05 | 2488424 | 0.061337 | 0.072088 | 0.059835 | -0.079611 |
| 2026-06-06 | 1089936 | 0.054163 | 0.070095 | 0.049449 | -0.072969 |
| 2026-06-07 | 1008840 | 0.062516 | 0.081634 | 0.055848 | -0.068723 |

## Block bootstrap

- QRW minus nonlinear logistic Brier: `-0.013822`, 95% CI `[-0.014108, -0.013527]`
- QRW minus pairwise logistic Brier: `0.003401`, 95% CI `[0.003210, 0.003603]`

## Verdict

Adaptive-decoherence QRW has a statistically significant edge over the registered nonlinear logistic baseline using the same raw features. Pairwise logistic remains stronger, so this is not a universal or uniquely quantum advantage.
