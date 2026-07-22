# Phase 2 checkpoint

This report audits real-data artifacts only. Synthetic pytest fixtures do not count
toward the one-million-record checkpoint.

| Check | Status | Observed |
|---|---|---:|
| At least 7 raw daily files | PASS | 8 |
| At least 1,000,000 raw records | PASS | 47,636,914 |
| Every raw date has processed and feature artifacts | PASS | missing processed=none, missing features=none |
| No duplicate trade_id and IDs increase across raw files | PASS | duplicates=0, monotonic=True |
| Cleaning removed less than 0.5% | FAIL | 1.127277% |
| Cleaning uses causal trailing windows | PASS | 8 quality reports |
| Feature matrices are finite | PASS | 8 files / 47,099,914 rows |
| Feature artifacts match processed tick rows by date | PASS | processed=47,099,914, features=47,099,914 |
| Imbalance features have documented provenance and coverage | PASS | sources={'trade_volume_imbalance': 8}, min valid=98.95%, nonzero imbalance=True |
| Synthetic trade imbalance is causal | PASS | 8 files, causal=True |
| Trade intensity is causal | PASS | 8 files, causal=True |
| Tick-direction lag-1 autocorrelation has p < 0.05 | PASS | rho_1=0.957286, p=0 |
| All Phase 2 pytest tests pass | PASS | 23 passed in 2.21s |

Run the logic tests with:

```text
python -m pytest tests/test_data_pipeline.py -v
```
