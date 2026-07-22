# Superseded root artifacts (2026-07-22)

Moved here during a repository audit. **Không được trích dẫn làm bằng chứng.**

Cả bốn file từng nằm ở `reports/` gốc và không đạt tiêu chí artifact hợp lệ
trong [`ARTIFACT_STATUS.md`](../../ARTIFACT_STATUS.md).

| File | Lý do |
|---|---|
| `phase3_overfitting_audit.{json,md}` | Bản trùng, **không có `provenance` block**. Bản được tiêu thụ thật là `reports/audits/phase3_overfitting_audit.*` (đọc bởi `freeze_release.py`, `phase6_pipeline.py`) và bản đó có provenance đầy đủ. |
| `phase3_overfitting_audit_postfix_subset.{json,md}` | **Không có tham chiếu nào** trong toàn bộ source tree, cũng không có provenance block. |

Nguyên nhân gốc đã sửa cùng lúc: `scripts/audits/phase3_overfitting_audit.py`
trước đây mặc định ghi ra `reports/` gốc, nên mỗi lần chạy lại sinh thêm một bản
không ai đọc. Default nay trỏ thẳng vào `reports/audits/`.
