# Phase 6 Checkpoint

**Final engineering status:** PASS

## Acceptance

| Check | Status | Observed |
|---|---|---:|
| Seven required figures exist and are nonblank | PASS | 7/7 |
| Every required figure exceeds 50 KB | PASS | 7/7 |
| Final report has at least 15 pages | PASS | 20 pages |
| Presentation has 15-20 slides | PASS | 16 slides |
| Report contains a quantitative p-value table | PASS | KS p-value and adjusted p-value |
| Final pytest run passes | PASS | 61 passed in 5.50s |
| README has installation and run instructions | PASS | installation + reproducibility commands |

## Scientific Guardrail

- Top Phase 5 scorecard model: `CRW Correlated`.
- QRW Adaptive overall rank: `3`.
- Scope: one June 12, 2026 window; exploratory only.
- The report explicitly rejects a current QRW superiority claim.
- A frozen protocol and fresh multi-day synchronized LOB holdout are
  still required for confirmation.

## Artifacts

- `src/visualization/plot_suite.py`
- `figures/prob_evolution.gif`
- `figures/variance_scaling.png`
- `figures/return_distributions.png`
- `figures/acf_comparison.png`
- `figures/sample_paths.png`
- `figures/coin_operator_heatmap.png`
- `figures/scorecard.png`
- `docs/final_report.md`
- `docs/final_report.pdf`
- `docs/presentation_slides.md`
- `docs/presentation_slides.pdf`
- `reports/phase6_diagnostics.json`

The optional Streamlit dashboard is available at `src/dashboard/app.py`.
Its static preview is `figures/dashboard_screenshot.png`.
