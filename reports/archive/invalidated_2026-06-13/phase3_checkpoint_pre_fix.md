# Phase 3 checkpoint

> **SUPERSEDED ON 2026-06-13:** The earlier predictive-edge rows and sections
> below were produced before three overfitting fixes: validation was included
> in a final structural refit, bias updates reused warmup observations, and tick
> cleaning used `shift(-1)`. Those edge claims are stale and must not be cited.
> The post-fix live audit reports QRW minus fair-affine Brier `+0.016600` with
> 95% interval `[0.006711, 0.025549]`, so QRW is materially worse on that window.
> Historical June 1-7 audits remain invalid until their processed/features
> artifacts are rebuilt with the causal cleaner.

## Post-fix status

- Full tests: `45/45` pass.
- June 12 causal-cleaned rows: `1,908`.
- Adaptive structural fit: `78` moving events for `6` parameters
  (`13.0` events/parameter); `low_sample_warning=true`.
- `final_refit_includes_validation=false`.
- `bias_update_reuses_warmup=false`.
- Post-fix final holdout QRW minus fair-affine Brier: `+0.016600`,
  CI `[0.006711, 0.025549]`.
- Post-fix pooled walk-forward delta: `+0.033635`,
  CI `[0.012353, 0.055213]`.

This checkpoint separates implementation correctness from sample-size limitations.
The QRW engine and integration code pass, and a synchronized live window now
provides varying OBI with complete timestamp coverage.

| Check | Status | Observed |
|---|---|---:|
| Six theoretical implementation tests pass | PASS | 6/6 |
| Market and adaptive integration tests pass | PASS | 15/15 |
| Full project test suite passes | PASS | 45/45 |
| Probability remains normalized | PASS | max error 9.437e-15 |
| Symmetric Hadamard walk remains symmetric | PASS | tolerance 1e-12 |
| Density-matrix trace remains one | PASS | tolerance 1e-12 |
| Hadamard ballistic coefficient at T=500 | PASS | Var/T^2 = 0.292895 |
| Full-dephasing classical limit at T=100 | PASS | Var/T = 1.000000 |
| 1,000 paths x 1,000 steps under 60 seconds | PASS | 0.022918 seconds |
| Every sampled path move is local | PASS | max absolute increment=1 |
| Adaptive coin uses five causal raw features | PASS | all coefficients finite |
| Intensity-adaptive gamma is physical | PASS | gamma_t >= 0 |
| Structural parameters are fixed in test days | PASS | only bias updates |
| Live LOB timestamp coverage | PASS | 100% |
| Live OBI signal varies | PASS | variance=0.748599, 100 unique values |
| Output responds to OBI input | PASS | L1(real, flipped)=0.106523 |
| Post-fix live walk-forward edge vs affine | FAIL | delta=+0.033635 |
| Post-fix final holdout edge vs affine | FAIL | delta=+0.016600, CI excludes 0 in the wrong direction |
| Historical multi-day edge claims | STALE | require causal artifact rebuild and fresh holdout |
| Edge vs pairwise logistic baseline | FAIL | prior stale audit delta=+0.003401 |

## Corrections to the original plan

The Phase 1 derivation showed that the symmetric Hadamard walk used by this
project has

```text
Var(X_T) / T^2 -> 1 - 1/sqrt(2) = 0.292893...
```

The original Phase 3 range `[0.4, 0.6]` is therefore not applicable to this
coin and initial-state convention.

With shifts of `+1` and `-1`, the fully dephased Hadamard walk is a simple
symmetric classical random walk:

```text
Var(X_T) / T -> 1
```

The original target `0.5` would require a different step-size or timing
convention. The tests use the convention fixed in `notes/01_dtqrw_formalism.md`.

## Calibration artifact

`results/calibrated_params.json` was fitted on 1,937 rows from the synchronized
live artifact `data/features/features_BTCUSDT_2026-06-12.parquet`.

- Lag-one direction correlation: `rho_1 = 0.868651049502`
- Base dephasing rate: `gamma = 0.140813788444`
- OBI sensitivity: `alpha_obi = 5.0`
- Tick-direction sensitivity: `alpha_direction = -4.359417636830`
- OBI-change sensitivity: `alpha_obi_change = -0.034860068333`
- Absolute-OBI sensitivity: `alpha_abs_obi = 0.039755026446`
- Intensity-decoherence sensitivity: `gamma_intensity = -0.618190655470`
- Regime intercept: `obi_bias = -0.832508657804`
- Structural warmup: `774` rows; bias update: all `1,937` rows
- LOB match coverage: `100%`
- Calibration: two-stage fixed structure plus causal bias update
- Selected regularization: `0.0001` by chronological validation Brier

Changing the first 50 OBI inputs to zero changes the exact final distribution by
L1 distance `0.027176`; flipping their signs changes it by `0.106523`. The model
is therefore input-sensitive on this window.

The adaptive coin is unitary and uses a phase link with coherent expected step
`tanh(bias + beta' * standardized_features)`. Event dephasing is
`gamma_t = gamma_base * exp(gamma_intensity * standardized_log_intensity)`.
This preserves non-negative physical decoherence while allowing confidence to
respond to market activity.

`simulate_price_path` uses a documented local event-kernel protocol: refresh the
latent symmetric coin, apply calibrated dephasing, apply the adaptive coin and
conditional shift, then measure position. All sampled increments are exactly
`+1` or `-1`; it no longer samples unrelated time marginals.

The performance checkpoint now times the real `MarketQRW.simulate_price_path`
implementation. The exact density-matrix market simulation is reported
separately (`0.125119` seconds for 50 steps) and is not mislabeled as the
1,000-path benchmark.

The historical synthetic-OBI artifact was rebuilt on 250,000 rows using the
adaptive protocol. Its validation Brier is `0.073285`, with
`gamma_intensity=-2.0`.

The live window is sufficient for pipeline validation, not for a robust market
claim. A multi-day synchronized collection remains necessary for out-of-sample
evaluation.

## Overfitting audit

`reports/phase3_overfitting_audit.md` uses a chronological 60/20/20 split,
three expanding walk-forward folds, and scores only events where the next trade
price moves.

- QRW final-fold Brier: `0.099269`
- Neutral test Brier: `0.250000`
- Fair linear-market final-fold Brier: `0.099060`
- Final-fold QRW minus fair baseline interval: `[-0.001089, 0.001388]`
- Mean walk-forward QRW Brier: `0.092940`
- Mean walk-forward linear-OBI Brier: `0.125086`
- Mean walk-forward fair linear-market Brier: `0.105213`
- Pooled walk-forward QRW minus fair baseline: `-0.011644`
- Moving-block interval: `[-0.022492, -0.002853]`
- Rolling alpha range: `1.954543` to `1.954543`

Structural parameter drift is removed by design: gamma, OBI sensitivity, and
tick-direction sensitivity are frozen after warmup; only the regime intercept
is updated causally. The pooled live walk-forward result establishes edge over
the fair affine baseline, while the final fold alone remains tied.

## Base-model ablation

`reports/phase3_multiday_edge_audit.md` uses June 1-4 for development and tests
June 5-7 prequentially on `4,587,200` moving events. This section refers to the
earlier fixed-decoherence, two-feature QRW.

- QRW minus fair affine Brier: `-0.000368`
- Moving-block 95% interval: `[-0.000443, -0.000291]`
- QRW minus nonlinear logistic Brier: `+0.000308`
- Logistic comparison interval: `[0.000255, 0.000359]`

The base QRW establishes edge over the affine benchmark but remains below
standard logistic. The adaptive-decoherence extension below is the current
Phase 3 model and resolves that gap.

## Adaptive-decoherence edge

`reports/phase3_adaptive_decoherence_audit.md` selects model structure using
June 1-3 and validation day June 4. June 5-7 are prequential tests containing
`4,587,200` moving events. QRW and logistic receive the same five causal raw
features.

- Adaptive QRW minus nonlinear logistic Brier: `-0.013822`
- Moving-block 95% interval: `[-0.014108, -0.013527]`
- Adaptive QRW minus pairwise logistic Brier: `+0.003401`
- Pairwise comparison interval: `[0.003210, 0.003603]`
- Fixed structural parameters: `alpha_obi=0.935866`,
  `alpha_direction=1.065513`, `gamma_intensity=-1.915382`
- Causal bias range across test days: `-0.079611` to `-0.068723`

This establishes predictive edge over the registered nonlinear logistic
baseline. A 21-term pairwise logistic stress test remains stronger, so this is
not evidence of universal QRW superiority or uniquely quantum advantage.

## Reproduction

```text
python -m pytest -q
python scripts/phase3_pipeline.py
python scripts/phase3_overfitting_audit.py
python scripts/phase3_multiday_edge_audit.py
python scripts/phase3_adaptive_decoherence_audit.py
```

**Overall: ENGINEERING TESTS PASS; THREE OVERFITTING BUGS FIXED; CURRENT
POST-FIX LIVE AUDIT SHOWS NO QRW EDGE; HISTORICAL EDGE CLAIMS ARE STALE**
