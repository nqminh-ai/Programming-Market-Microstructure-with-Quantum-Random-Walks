# ML/DL implementation plan — frozen Phase 0 protocol

**Protocol:** `ml_directional_v7`

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY`

**Frozen:** 2026-07-29

This document freezes the research question and evaluation contract before any
new ML or deep-learning model is fitted. It does not modify the registered
Phase 3–6 QRW protocol or any existing project conclusion.

The machine-readable source of truth is
[`config/ml_experiment.yaml`](../config/ml_experiment.yaml). Later phases must
load it through
[`src/evaluation/ml_protocol.py`](../src/evaluation/ml_protocol.py). If a
frozen choice changes, the protocol version must be bumped and the result must
be reported as a new experiment.

`ml_directional_v2` supersedes `v1` before any official artifact or test-fold
access. The review before Phase 3 found that the registered OrderFlow AR(5)
reference requires tick-direction lags 1 through 5, while `v1` had only
1, 2, 5 and 10. Version 2 adds lags 3 and 4; all other frozen choices remain
unchanged.

`ml_directional_v3` supersedes `v2`, also before official artifacts or test
access, to freeze the Phase 4 sequence representation: 128 causal events,
channels-first float32 UTC-day shards and train-only normalization statistics.

`ml_directional_v4` supersedes `v3` before any TCN fit or test access. It freezes
the Phase 5 causal residual architecture, optimizer, early-stopping rule,
gradient clipping, deterministic seed, device policy and calibration choices.
Because the config hash changes, all Phase 1 and Phase 4 artifacts used by an
official v4 run must be rebuilt.

`ml_directional_v5` supersedes `v4` before robustness runs or holdout access.
It preregisters seeds 2026, 2027 and 2028, limits diagnostics to selection and
calibration, fits volatility/liquidity tertiles on train only and requires
independent per-asset refits with an equal-asset summary. Official artifacts
must again be rebuilt because the config hash changes.

`ml_directional_v6` supersedes `v5` before optional hybrid fitting. It freezes
an eight-step differentiable complex-amplitude QRW, bounded adaptive coin and
decoherence parameters, a fixed 50/50 neural/QRW blend and five mandatory
independently trained ablations. This model is a new directional hybrid; it is
not the registered Windowed-QRW density-matrix model and cannot fill that
model's unresolved multi-horizon benchmark row.

`ml_directional_v7` supersedes `v6` for Phase 8 release packaging. It freezes a
single-asset/single-horizon pretest bundle, six required artifact roles,
SHA-256 inventory, PowerShell reproduction script, read-only dashboard policy
and an official-release gate requiring a clean source tree. Release packaging
does not authorize test access.

## 1. Research question

Can a nonlinear tabular model or a causal sequence model improve calibrated
directional probabilities over the project's registered classical and QRW
baselines on a chronological, non-overlapping, held-out sample?

This is primarily a forecasting benchmark. Profitability after costs is a
secondary endpoint and no live trading is authorized.

## 2. Scope

The canonical assets are BTCUSDT, ETHUSDT and BNBUSDT, using their frozen 69-day
feature stores.

Two horizon groups answer different questions:

| Group | Horizons (ticks) | Question |
|---|---|---|
| Predictive skill | 1,000; 5,000; 10,000; 50,000 | Does the model predict direction better? |
| Economic feasibility | 50,000; 100,000; 200,000 | Is any measured skill large enough to survive costs? |

The label at anchor `t` and horizon `h` is one when `price[t+h] > price[t]` and
zero when it is lower. Zero-return labels are excluded. Anchors are spaced by
exactly `h`, both endpoints must remain in the same segment and UTC day, and
overlapping labels are forbidden.

## 3. Chronological split

Each complete 69-day asset store is divided once:

| Fold | Days | Permitted use |
|---|---:|---|
| Train | 45 | Fit preprocessing and model parameters |
| Selection | 10 | Select architecture and hyperparameters |
| Calibration | 5 | Fit probability calibration only |
| Test | 9 | One evaluation after model and calibrator are frozen |

A 200,000-tick purge covers the largest registered horizon at every split
boundary. No random row split is permitted. Test labels, metrics and plots must
not influence model, feature, calibration or threshold selection.

## 4. Frozen causal feature families

Base features extend the current five-feature directional design with
`price_mid_deviation`. Phase 1 implements trailing tick-direction lags,
signed-volume aggregates, returns, realized volatility and inter-event time at
the exact windows declared in the YAML protocol.

Every feature must be computable at or before its anchor. Scaling and any other
learned preprocessing use the train fold only. A future-mutation test must prove
that changing rows after an anchor cannot change that anchor's features.

### Phase 1 implementation status

Phase 1 is complete. The implementation consists of:

- `src/data/temporal_features.py`: prefix-sum feature evaluation at causal
  anchors, shared across all horizons;
- `src/data/ml_dataset.py`: UTC-day streaming, non-overlapping labels,
  chronological folds, boundary purge, parquet outputs and metadata;
- `scripts/research/build_ml_dataset.py`: development and official build CLI;
- `tests/test_temporal_features.py` and `tests/test_ml_dataset.py`: future
  mutation, segment/day boundary, zero-return, overlap, purge, determinism and
  out-of-core regression tests.

The reader materializes at most one UTC day. It converts the string trade side
to an `int8` sign in Arrow before pandas and evaluates all six horizons from one
shared set of rolling prefixes.

Development smoke run:

```powershell
python -m scripts.research.build_ml_dataset --asset BNBUSDT --max-days 1
```

Full exploratory build:

```powershell
python -m scripts.research.build_ml_dataset --asset BTCUSDT
```

An official build additionally requires exactly 69 consecutive UTC dates, the
registered feature path, a clean source tree and SHA-256 hashes:

```powershell
python -m scripts.research.build_ml_dataset --asset BTCUSDT --official
```

Generated datasets live under `data/assets/<symbol>/ml/` and are intentionally
ignored by Git. The metadata JSON is the entry point for later model phases.

### Phase 2 implementation status

Phase 2 is complete at the training boundary. It pins scikit-learn 1.9.0 and
implements:

- a deterministic 16-candidate
  `HistGradientBoostingClassifier` grid with internal random early stopping
  disabled;
- selection by Brier score and log-loss tie-break on the selection fold;
- identity, Platt and isotonic calibration chosen on the calibration fold;
- versioned model serialization and SHA-256 diagnostics;
- a hard failure if the training API receives a test row.

Train one Phase 1 dataset without opening its test fold:

```powershell
python -m scripts.research.train_ml_baseline `
  --metadata data/assets/bnbusdt/ml/ml_directional_BNBUSDT_metadata.json `
  --horizon 50000
```

The CLI uses a parquet fold predicate to read only `train`, `selection` and
`calibration`. Its diagnostics deliberately store `test_fold_read: false` and
`test_metrics: null`. Test prediction and comparative metrics belong to Phase
3, not Phase 2.

### Phase 3 implementation status

The Phase 3 benchmark engine is implemented, but the real holdout remains
closed in the development worktree. `src/evaluation/ml_benchmark.py` provides:

- common-timestamp Brier, log loss, ECE and accuracy;
- the registered causal directional links plus a raw-tick OrderFlow AR(5);
- calibration on the dedicated calibration fold;
- paired cluster bootstrap that resamples complete UTC days;
- an explicit record of any registered model that cannot be evaluated.

`Windowed-QRW (density matrix)` is currently recorded as not evaluated. Its
available probability is a one-tick forecast, and relabelling it as a `t+h`
forecast would be a horizon mismatch. Completing that row requires a separately
preregistered causal multi-horizon QRW adapter.

Opening the real test fold requires an explicit flag and matching hashes for the
config, dataset, training diagnostics and serialized model:

```powershell
python -m scripts.research.evaluate_ml_baseline `
  --metadata data/assets/bnbusdt/ml/ml_directional_BNBUSDT_metadata.json `
  --training-diagnostics results/ml_models/hist_gradient_boosting_BNBUSDT_h50000.json `
  --model results/ml_models/hist_gradient_boosting_BNBUSDT_h50000.pkl `
  --horizon 50000 `
  --open-test
```

The command writes a ranked summary, common-sample prediction parquet and a
JSON artifact containing the paired HGB-minus-baseline confidence intervals.
It should be run only after an official `ml_directional_v7` dataset and model
have been frozen.

### Phase 4 implementation status

Phase 4 is implemented without opening the holdout. The sequence builder:

- gathers 128 events ending at each Phase 1 `anchor_row`;
- writes channels-first float32 `.npz` shards by UTC day, horizon and fold;
- validates timestamp and anchor alignment against the Phase 1 parquet;
- never lets a sequence cross a segment or day boundary;
- computes per-horizon normalization mean/scale from train shards only.

The eight frozen channels are OBI, OBI validity, tick direction, log trade
intensity, signed log quantity, one-tick log return, price-mid deviation and
inter-event time. `obi_valid` is retained as an explicit channel rather than
silently dropping or inventing order-book observations.

Build the sequence manifest after Phase 1:

```powershell
python -m scripts.research.build_sequence_dataset `
  --asset BNBUSDT `
  --phase1-metadata data/assets/bnbusdt/ml/ml_directional_BNBUSDT_metadata.json
```

The manifest records every shard, its fold and row count, channel order,
train-only normalization statistics and provenance. Phase 5 consumes this
manifest; it must not recompute normalization from calibration or test shards.

### Phase 5 implementation status

Phase 5 is implemented with PyTorch 2.12.0 and keeps the real holdout closed:

- six causal residual blocks use dilations 1, 2, 4, 8, 16 and 32;
- left-only effective padding prevents future sequence values from affecting an
  earlier logit, with a dedicated future-mutation test;
- train shards are streamed one UTC day at a time and normalized using only the
  Phase 4 train statistics;
- selection-fold Brier score, then log loss, chooses the best epoch with
  patience-based early stopping;
- Identity, Platt and isotonic calibration are fitted and compared only on the
  calibration fold;
- versioned checkpoints store a CPU `state_dict`, architecture, channel order,
  normalization and calibrator.

Train one horizon after rebuilding its sequence manifest under v7:

```powershell
python -m scripts.research.train_tcn_baseline `
  --manifest data/assets/bnbusdt/ml/sequences/sequence_BNBUSDT_manifest.json `
  --horizon 50000
```

CPU is the deterministic default. CUDA requires the explicit `--device cuda`
flag and an available CUDA runtime. The CLI records `test_fold_read: false` and
`test_metrics: null`; comparative holdout evaluation remains a later gated
step.

### Phase 6 implementation status

Phase 6 implements pre-holdout diagnostics without authorizing test access:

- TCN training accepts only the preregistered seeds 2026, 2027 and 2028;
- volatility is the sequence RMS of one-tick log return, while liquidity is
  represented by mean inter-event time;
- both regime boundaries are train-fold tertiles and are never refitted on
  selection, calibration or test;
- each seed receives overall, UTC-day and regime Brier/log-loss/accuracy rows;
- the report records cross-seed Brier dispersion and mean prediction
  disagreement, plus selection-to-calibration and worst-day stability;
- cross-asset aggregation requires independently refitted BTCUSDT, ETHUSDT and
  BNBUSDT reports and weights the three assets equally.

Train the registered seeds for an asset:

```powershell
2026, 2027, 2028 | ForEach-Object {
  python -m scripts.research.train_tcn_baseline `
    --manifest data/assets/bnbusdt/ml/sequences/sequence_BNBUSDT_manifest.json `
    --horizon 50000 `
    --seed $_
}
```

Generate the asset-level report:

```powershell
python -m scripts.research.evaluate_tcn_robustness `
  --manifest data/assets/bnbusdt/ml/sequences/sequence_BNBUSDT_manifest.json `
  --horizon 50000 `
  --model 2026=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2026.pt `
  --model 2027=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2027.pt `
  --model 2028=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2028.pt `
  --diagnostics 2026=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2026.json `
  --diagnostics 2027=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2027.json `
  --diagnostics 2028=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2028.json
```

After generating one report for each registered asset:

```powershell
python -m scripts.research.aggregate_tcn_robustness `
  --report reports/research/tcn_robustness_BTCUSDT_h50000.json `
  --report reports/research/tcn_robustness_ETHUSDT_h50000.json `
  --report reports/research/tcn_robustness_BNBUSDT_h50000.json
```

These are model-development diagnostics. Selection chose the epoch and
calibration fitted the probability mapping, so neither fold is an untouched
generalization estimate. The real test fold remains closed.

### Phase 7 implementation status

Phase 7 implements the optional neural-adaptive QRW experiment:

- a causal TCN maps each 128-event sequence to a neural probability, a coin
  angle and a decoherence rate;
- coin angle is bounded to `[pi/8, 3pi/8]` and decoherence to `[0, 0.35]`;
- an eight-step two-state complex-amplitude walk evolves unitarily before a
  bounded symmetric decoherence mixture produces right-side probability;
- the anchor OBI is the only direct initial-state signal and all adaptive
  parameters are computed causally from the same trailing sequence;
- five variants are mandatory: neural-only, fixed-QRW-only,
  adaptive-QRW-only, neural plus fixed QRW and the full neural-adaptive hybrid;
- every learnable variant is fitted independently using the same chronological
  train/selection/calibration policy; the fixed QRW correctly records epoch 0;
- the ablation report refuses incomplete variant sets and never ranks on test.

Run the complete ablation suite:

```powershell
python -m scripts.research.train_neural_qrw_hybrid `
  --manifest data/assets/bnbusdt/ml/sequences/sequence_BNBUSDT_manifest.json `
  --horizon 50000
```

The CLI writes five versioned state-dict checkpoints, five diagnostics files
and one ablation report containing full-minus-neural, full-minus-adaptive-QRW
and full-minus-fixed-hybrid selection Brier differences. These remain
development diagnostics; they do not authorize a hybrid superiority claim or
test-fold access.

### Phase 8 implementation status

Phase 8 implements an auditable pretest release bundle:

- six required JSON roles cover dataset metadata, sequence manifest, HGB
  training, TCN training, TCN robustness and the hybrid ablation report;
- every artifact must share protocol, config hash, asset and horizon;
- recursive safety validation rejects any `test_fold_read` value other than
  false or any non-null `test_metrics`;
- the manifest records Git commit, dirty source paths, file sizes and SHA-256
  for every artifact plus the generated reproduction script;
- development bundles are explicitly labelled and cannot support claims;
- official mode additionally requires clean source plus official Phase 1 and
  Phase 4 inputs;
- the Streamlit dashboard reads only the manifest and optionally verifies
  hashes; it never imports predictions, trains a model or computes metrics.

Build a bundle by supplying the six role/path pairs:

```powershell
python -m scripts.research.build_ml_release `
  --asset BNBUSDT `
  --horizon 50000 `
  --artifact phase1_dataset_metadata=data/assets/bnbusdt/ml/ml_directional_BNBUSDT_metadata.json `
  --artifact phase4_sequence_manifest=data/assets/bnbusdt/ml/sequences/sequence_BNBUSDT_manifest.json `
  --artifact phase2_hgb_training=results/ml_models/hist_gradient_boosting_BNBUSDT_h50000.json `
  --artifact phase5_tcn_training=results/ml_models/temporal_convolutional_BNBUSDT_h50000_s2026.json `
  --artifact phase6_tcn_robustness=reports/research/tcn_robustness_BNBUSDT_h50000.json `
  --artifact phase7_hybrid_ablation=results/ml_models/neural_qrw_BNBUSDT_h50000_ablation.json
```

Open the read-only dashboard:

```powershell
streamlit run src/dashboard/ml_release_dashboard.py
```

The default dashboard path points to the BNBUSDT 50,000-tick bundle and can be
changed in the sidebar.

## 5. Registered comparison

All models must predict the same timestamps:

- Majority Class;
- the six registered causal directional models;
- Windowed-QRW (density matrix);
- Histogram Gradient Boosting in Phase 2;
- a causal Temporal Convolutional Network in Phase 5;
- the separately labelled neural-adaptive QRW ablation suite in Phase 7.

New models initially run in a separate exploratory comparison script. They must
not be inserted into the existing hard-coded `MODEL_NAMES` set until a later
release explicitly bumps the relevant protocol.

Hyperparameters are selected by selection-fold Brier score, with log loss as
the tie-break. Identity, Platt and isotonic calibration are compared on the
dedicated calibration fold; the choice with the lower calibration-fold Brier
score is frozen before test access.

## 6. Endpoints and uncertainty

The primary endpoint is test Brier score. Secondary endpoints are log loss,
expected calibration error, accuracy, return after costs, inference latency and
peak memory.

Pairwise model comparisons use a paired block bootstrap with a 95% confidence
interval. Economic evaluation must include spread, fees and an adverse-selection
scenario. Accuracy is compared with the majority class on the same test sample,
not with an assumed 50% rate.

Success means the experiment is causal, reproducible and fairly scored. It does
not require ML, deep learning or QRW to win.

## 7. Artifacts and change control

Every official run records:

- protocol version, Git commit and random seed;
- feature path, feature SHA-256 and config SHA-256;
- exact split dates and evaluated timestamps;
- model parameters and all registered metrics.

Official artifacts require a clean source tree and remain labelled exploratory.
Changing any asset, label, horizon, split, feature family, selection rule,
endpoint or cost policy requires a new protocol version.

## 8. Runtime compatibility decision

The Phase 0 compatibility spike ran on CPython 3.14.5 and successfully imported
scikit-learn 1.9.0, XGBoost 3.3.0 and PyTorch 2.12.0+cpu. Phase 2 will use
scikit-learn histogram gradient boosting first; XGBoost remains a performance
fallback. Phase 5 pins PyTorch 2.12.0 for the TCN.

These packages are present in the development environment but are not added to
the project's exact dependency pins during Phase 0. Each dependency will be
pinned only in the phase that first imports it, together with a model smoke test,
so the current QRW installation remains unchanged until the feature is real.

## 9. Delivery phases

1. **Phase 0 — protocol:** this document, validated YAML configuration and
   leakage guardrails.
2. **Phase 1 — data:** out-of-core causal feature and horizon dataset builder.
3. **Phase 2 — ML:** calibrated histogram gradient boosting.
4. **Phase 3 — benchmark:** common-sample comparison and uncertainty.
5. **Phase 4 — sequences:** streaming causal sequence dataset.
6. **Phase 5 — DL:** calibrated TCN baseline.
7. **Phase 6 — robustness:** cross-asset, regime, fold and seed checks.
8. **Phase 7 — optional hybrid:** neural adaptive QRW with mandatory ablations.
9. **Phase 8 — release:** dashboard, reproduction commands and manifest.

Phase 2 may begin only when the Phase 1 causality and dataset tests pass.
