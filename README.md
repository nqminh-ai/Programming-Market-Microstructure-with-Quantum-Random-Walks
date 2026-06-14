# QRW Market Microstructure

Research code for quantum-random-walk market microstructure experiments.

## Current Status

- Phase 1 theory and Phase 2/3 engineering prototypes are implemented.
- Phase 4 classical baselines and the common benchmark suite are implemented.
- Phase 5 statistical validation and scorecard are implemented.
- Phase 6 figures, report, slides, and artifact checks are implemented.
- All 61 automated tests pass.
- Causality, forecast leakage, zero-move calibration, bootstrap, timestamp,
  scorecard, and dashboard issues were fixed by June 14, 2026.
- The current live audit is mixed: QRW wins the final holdout but loses the
  pooled walk-forward comparison against the fair affine baseline.
- Historical June 1-7 derived artifacts were rebuilt with the causal pipeline
  and remain development-only.
- Phase 5 is an engineering pass on one June 12, 2026 window; its results
  remain exploratory and do not establish QRW superiority.
- Phase 2 still reports a data-quality threshold miss because cleaning removes
  1.127277% of raw rows versus the roadmap limit of 0.5%.

See:

- `docs/ke_hoach_QRW_market_microstructure.md`
- `reports/phase3_checkpoint.md`
- `reports/phase5_checkpoint.md`
- `reports/phase6_checkpoint.md`

The June 13 audit files remain available as historical remediation snapshots.

## Structure

```text
config/       Runtime configuration
data/         Raw, processed and feature data
docs/         Project plan and compiled documents
notebooks/    Theory verification notebooks
notes/        Theory source notes
reports/      Current checkpoints, audits and diagnostics
results/      Current model parameters/results
figures/      Generated statistical comparison figures
scripts/      Reproducible pipeline entry points
src/          Data and model implementation
tests/        Automated tests
```

Invalidated outputs are retained under `reports/archive/` and
`results/archive/` for audit history, not for active conclusions.

## Installation

```text
pip install -r requirements.txt
```

## Verification

```text
python -m pytest
python scripts/phase2_pipeline.py checkpoint
python scripts/phase3_overfitting_audit.py
python scripts/phase4_pipeline.py
python scripts/phase5_pipeline.py
```

## Run Phase 6

```text
python scripts/phase6_pipeline.py
```

## Rebuild Historical Derived Data

```text
python scripts/phase2_pipeline.py process
python scripts/phase2_pipeline.py features --obi-source trade_imbalance
python scripts/phase2_pipeline.py checkpoint
```
