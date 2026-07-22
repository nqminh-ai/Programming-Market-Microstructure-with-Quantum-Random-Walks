# Trạng thái artifact nghiên cứu

## Không hợp lệ cho kết luận hiện tại

Mọi artifact Phase 4/5/6 được tạo trước protocol
`fixed_origin_marginal_density_matrix_ar1_obi_v4` đều là **stale**. Các file
đó đã được gom vào `results/archive/invalidated_pre_v4_2026-07-06/` và
`reports/archive/invalidated_pre_v4_2026-07-06/` để truy vết lịch sử, nhưng
không được trích dẫn làm bằng chứng.

Các thư mục `results/` và `figures/` hiện không chứa artifact Phase 4–6 đang
hoạt động. Chúng chỉ được tạo lại sau khi một release vượt đủ các gate bên dưới.

Lý do vô hiệu hóa:

- protocol cũ ghép các marginal QRW độc lập thành path;
- ACF, tail và path-based DM của QRW vì thế không hợp lệ;
- scorecard cũ trung bình hạng của các metric không đồng nhất;
- diagnostics cũ thiếu full Git commit và SHA-256 feature artifact.

## Điều kiện artifact hợp lệ

Artifact chỉ được xem là chính thức khi:

1. `provenance.protocol_version` đúng protocol v4;
2. `provenance.code_commit` trùng commit đang chạy;
3. canonical feature path và SHA-256 trùng khớp;
4. source tree sạch tại thời điểm sinh;
5. `reports/release_manifest.json` chứa SHA-256 cho toàn bộ input/output;
6. full test suite đạt.

Cho tới khi Phase 3–6 được tái tạo sau một commit sạch, verdict duy nhất được
phép dùng là kết quả âm đã ghi trong `docs/final_report.md`.
