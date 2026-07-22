# Báo cáo khắc phục sau kiểm toán

## Tóm tắt

Mã nguồn đã sửa các lỗi ngữ nghĩa QRW, leakage và thống kê chính. Full test
suite đạt **164/164**. Release vẫn chưa được tuyên bố hoàn tất vì artifact
Phase 3–6 chưa được tái tạo từ một commit sạch. File credential cục bộ đã được
xóa và chặn ở cả Git lẫn Docker context.

| Finding | Trạng thái | Bằng chứng | Commit |
|---|---|---|---|
| Blocker 1 — protocol/artifact mismatch | PARTIALLY FIXED | Guard kiểm tra protocol, full commit, canonical path và SHA-256 đã có; artifact v4 và manifest chính thức chưa được sinh | `fa4d14d` + phần nối pipeline chờ commit |
| Blocker 2 — marginal bị coi là path | FIXED | Hoán vị độc lập từng horizon không đổi score; QRW bị loại khỏi ACF/tail; DM dùng rolling one-step | `b56d308` |
| Blocker 3 — claim paper không có bằng chứng | FIXED | `final_report.md` và nguồn sinh PDF dùng tiếng Việt UTF-8; đã rút claim 31 ngày, heavy-tail unitary, tail range và robust DM | `44954d7` + cập nhật generator chờ commit |
| Leakage — movement probability | FIXED | Test xác nhận chỉ dùng warmup | `a9e2be6` |
| Leakage — Phase 3 in-sample benchmark | FIXED | Train/holdout tách thời gian, holdout explicit | `a9e2be6` |
| Leakage — live trade-flow OBI | PARTIALLY FIXED | Code và test xác nhận OBI tính trước khi append trade hiện tại; chưa commit | chờ commit |
| P-hacking — verdict holdout lật walk-forward | PARTIALLY FIXED | Pooled walk-forward là evidence chính trong code; artifact audit hiện tại vẫn stale | chờ commit và rebuild |
| AIC/BIC chéo likelihood | FIXED | Rank chỉ trong `likelihood_type`; có test bảng rỗng và hai họ likelihood | `661a7fc` |
| DM fixed-origin không căn chỉnh | FIXED | Chỉ nhận loss rolling-origin một bước trên timestamp chung tăng nghiêm ngặt, không trùng; test từ chối thiếu/sai alignment | `b56d308` + cập nhật chờ commit |
| Bootstrap scorecard chỉ MAE | PARTIALLY FIXED | Bootstrap 7 metric biên hợp lệ, rerank từng replicate bằng CRPS + log-loss tie-break; 8 test thống kê đạt, chưa commit | chờ commit |
| Equal-rank scorecard | FIXED | Primary CRPS, log-loss tie-break; diagnostics không tham gia rank | `b56d308` |
| Data path phân mảnh | FIXED | Config, helper, tài liệu và test dùng `data/assets/<symbol>`; report dùng `reports/assets/<symbol>`; script đã tách `pipelines/operations/audits/research` | chờ commit |
| Dependency ranges | PARTIALLY FIXED | Exact direct pins, transitive `requirements.lock` và test đồng bộ; chưa commit | chờ commit |
| Cross-asset chỉ ngày cuối | PARTIALLY FIXED | Expanding UTC-day folds đã có 3 test chống overlap; chưa chạy benchmark mới theo ràng buộc phiên | chờ commit và dữ liệu mới |
| Baseline cổ điển yếu | PARTIALLY FIXED | QRW link + logistic 5F, pairwise, nonlinear calibrated, AR(5), marked Hawkes dùng chung validation/test; chưa có artifact chính thức | chờ commit và rebuild |
| Tick phụ thuộc bị coi là độc lập | PARTIALLY FIXED | Per-day output, UTC-day cluster bootstrap, seed/block sensitivity và hiệu chỉnh BH toàn cục đã có trong code; chưa có artifact | chờ commit và rebuild |
| Nguồn gốc số liệu báo cáo | PARTIALLY FIXED | Test dựng context từ CSV + diagnostics có commit/path/SHA-256; manifest chính thức chưa tồn tại | chờ commit và rebuild |
| Heavy-tail recovery test | FIXED | Pareto alpha 2,5 được recovery trong sai số 12% | `44954d7` |
| Dữ liệu confirmatory thiếu | NOT FIXED | Mới có pre-registration ≥20 ngày UTC/tài sản và yêu cầu L2 đồng bộ; chưa thể có dữ liệu tương lai | `docs/data_collection_todo.md`, chờ commit |
| Credential-looking file | FIXED | Đã xóa SDK/file credential cục bộ; `.gitignore` và `.dockerignore` chặn cả thư mục lẫn tên file nhạy cảm | chờ commit |
| Collector tự đọc credential trong repo | FIXED | Không còn default repo path; chỉ đọc file khi đặt `SSI_CREDENTIALS_FILE`; full suite 164/164 đạt | chờ commit |
| Artifact Phase 3–6 v4 | NOT FIXED | Artifact v2 đã đánh dấu stale; chưa có `reports/release_manifest.json` | chưa có |

## Điểm rubric ước lượng lại

Đây là điểm cho **trạng thái release hiện tại**, không phải điểm tiềm năng của
mã nguồn. Artifact stale và thiếu dữ liệu confirmatory bị trừ điểm trực tiếp.

| Nhóm | Điểm | Tối đa | Lý do |
|---|---:|---:|---|
| Tính đúng khoa học và ngữ nghĩa QRW | 20 | 25 | Marginal semantics và claim đã sửa; chưa có artifact v4 xác nhận |
| Dữ liệu và bằng chứng thực nghiệm | 5 | 25 | Mẫu hoạt động quá ngắn; chưa có ≥20 ngày UTC mới hay L2 đồng bộ |
| Phương pháp thống kê và baseline | 15 | 20 | DM, endpoint, bootstrap và baseline đã sửa trong code; chưa có kết quả rebuild |
| Tái lập, provenance và release engineering | 12 | 20 | Guard/hash/lockfile đã có; worktree chưa sạch và manifest chưa tồn tại |
| Tài liệu và bảo mật vận hành | 9 | 10 | Báo cáo trung thực; credential cục bộ đã xóa và bị chặn khỏi Git/Docker |
| **Tổng** | **61** | **100** | Tăng từ 39/100 về tính đúng implementation, nhưng chưa đủ điều kiện release khoa học |

Nếu commit sạch và rebuild Phase 3–6 tạo manifest hợp lệ, điểm ước lượng có thể
tăng lên khoảng **74/100**. Phần còn thiếu sau đó chủ yếu là dữ liệu confirmatory
đa ngày chưa từng chạm tới.

## Vấn đề chưa giải quyết

1. Cần quyết định giữ hay commit xóa ba tài liệu root đang ở trạng thái deleted.
2. Cần quyền ghi Git index cho các commit còn lại.
3. Cần phê duyệt trước khi chạy tái tạo Phase 3–6 nếu tổng thời gian dự kiến
   vượt 5 phút.
4. Cần thu thập dữ liệu mới theo pre-registration để vượt khỏi mức exploratory.
