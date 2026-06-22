# Phase 1 checkpoint

| Kiểm tra | Kết quả |
|---|---|
| formalism_note_at_least_800_words | PASS |
| all_three_notes_exist | PASS |
| summary_exists | PASS |
| bibliography_has_at_least_5_entries | PASS |
| theory_pdf_at_least_10_pages | PASS |
| notebook_executed_without_errors | PASS |
| unitarity_norm_below_1e-10 | PASS |
| eigenvalues_on_unit_circle | PASS |
| probability_normalized | PASS |
| symmetric_initial_state_is_symmetric | PASS |
| variance_ratio_above_1_5 | PASS |

## Số liệu

- Formalism note: 1507 words
- Theory summary: 5067 words
- `docs/theory_notes.pdf`: 20 pages
- `reports/theory_verification.pdf`: 8 pages
- Unitarity Frobenius norm: 1.047e-15
- Max eigenvalue radius error: 1.443e-15
- QRW variance at T=100: 2929.422331
- CRW variance at T=100: 100.000000
- Variance ratio: 29.294223
- Fitted QRW scaling exponent: 1.997996

## Mathematical correction

The original plan's candidate `T^2/2 - T/4 + O(1)` is not the asymptotic variance
for the symmetric Hadamard walk used here. The verified leading coefficient is
`1 - 1/sqrt(2)`.

**Overall: PASS**
