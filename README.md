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
- `reports/checkpoints/phase3_checkpoint.md`
- `reports/checkpoints/phase5_checkpoint.md`
- `reports/checkpoints/phase6_checkpoint.md`

The June 13 audit files remain available as historical remediation snapshots.

## Structure

```text
config/                  Runtime configuration (data_config.yaml)
data/
├── raw/                 Compressed tick CSVs and LOB snapshots
├── processed/           Cleaned Parquet files
└── features/            Feature matrices for modeling
docs/
├── theory/              QRW formalism, QRW-vs-CRW, market mapping notes
├── ke_hoach_*.md        Master project plan
├── bao_cao_toan_dien.md Full report (Vietnamese)
├── final_report.*       Summary report + PDF
└── presentation_slides  Slides source + PDF
figures/                 Generated statistical comparison figures (DPI=300)
notebooks/               Theory verification notebooks
reports/
├── checkpoints/         Phase 1-6 checkpoint reports and diagnostics
├── data_quality/        Per-day data quality reports
├── feature_metadata/    Per-day feature metadata and statistics
├── audits/              Overfitting audits and remediation logs
└── archive/             Invalidated outputs (audit history only)
results/
├── *.csv / *.json       Current model parameters and comparison tables
└── archive/             Invalidated results (audit history only)
scripts/
├── phase*_pipeline.py   Reproducible pipeline entry points
└── audits/              Standalone overfitting audit scripts
src/
├── models/              QRW core, adaptive QRW, classical RW, GARCH, GBM
├── data/                Tick download, processing, feature engineering
├── evaluation/          Benchmark suite and statistical tests
├── visualization/       Plot suite
├── reporting/           Report builder
└── dashboard/           Streamlit interactive dashboard
tests/                   61 automated tests
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
python scripts/audits/phase3_overfitting_audit.py
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
