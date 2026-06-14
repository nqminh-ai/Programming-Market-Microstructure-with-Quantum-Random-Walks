# Kế Hoạch Dự Án: Lập Trình Vi Cấu Trúc Thị Trường bằng Quantum Random Walks (Cơ Học Thống Kê)

> **Phiên bản:** 1.0 | **Cập nhật:** 2025 | **Hình thức:** Solo researcher, part-time (~3–4 giờ/ngày)

---

## Bảng Tóm Tắt Tổng Quan

| Phase | Tên | Thời gian | Deliverable Chính |
|-------|-----|-----------|-------------------|
| 1 | Nền Tảng Lý Thuyết QRW | 2 tuần | `theory_notes.pdf`, `qrw_math_summary.md`, notebook kiểm tra toán học |
| 2 | Thu Thập & Xử Lý Dữ Liệu | 1.5 tuần | `tick_data_pipeline.py`, `orderbook_processor.py`, dataset HDF5 |
| 3 | Triển Khai Mô Hình QRW | 2.5 tuần | `qrw_core.py`, `coin_operators.py`, `qrw_market_sim.py` |
| 4 | Classical Baseline & So Sánh | 1 tuần | `classical_rw.py`, `benchmark_suite.py`, bảng kết quả |
| 5 | Kiểm Định Thống Kê | 1.5 tuần | `statistical_tests.py`, báo cáo kiểm định, p-value tables |
| 6 | Trực Quan Hóa & Báo Cáo | 1 tuần | Dashboard, `final_report.pdf`, slide trình bày |
| **Tổng** | | **~9.5 tuần** | **Project hoàn chỉnh** |

---

## Phase 1: Nền Tảng Lý Thuyết QRW

**Mục tiêu:** Xây dựng nền tảng toán học vững chắc về Quantum Random Walk và hiểu rõ sự ánh xạ sang vi cấu trúc thị trường.

**Thời gian ước lượng:** 2 tuần (10 ngày làm việc)

**Công cụ / thư viện:** Python (NumPy, SciPy, SymPy), Jupyter Notebook, LaTeX / Markdown

---

### 1.1 — Nghiên Cứu QRW Discrete-Time (DTQRW)

**Mô tả:** Nắm vững formalism Hilbert space cho QRW 1D trên đường thẳng.

**Các bước:**

- Đọc và tóm tắt paper gốc: Aharonov et al. (1993) *"Quantum random walks"* và Kempe (2003) *"Quantum random walks: an introductory overview"*
- Định nghĩa formal: không gian trạng thái `H = H_coin ⊗ H_position`, với `H_coin = C^2` (qubit), `H_position = l^2(Z)`
- Viết ra ma trận Hadamard coin `H = (1/√2)[[1,1],[1,-1]]` và shift operator `S`
- Tính evolution operator `U = S · (C ⊗ I)` cho bước đơn
- Ghi chú tóm tắt các điều kiện unitary, decoherence, và classical limit

**Deliverable:** File `notes/01_dtqrw_formalism.md` (≥ 800 từ, có công thức LaTeX đầy đủ)

---

### 1.2 — So Sánh QRW vs Classical Random Walk (Lý Thuyết)

**Mô tả:** Phân tích lý thuyết sự khác biệt phân phối xác suất giữa QRW và CRW.

**Các bước:**

- Chứng minh tính chất ballistic spreading của QRW: `σ(t) ~ O(t)` so với diffusive `σ(t) ~ O(√t)` của CRW
- Phân tích phân phối xác suất sau T bước: QRW có bimodal distribution với peak ở ±T/√2
- Ghi lại ảnh hưởng của lựa chọn coin operator (Hadamard, Grover, Fourier) lên phân phối
- Ghi chú về vai trò của điều kiện ban đầu (initial state) `|ψ_0⟩` — symmetric vs asymmetric

**Deliverable:** File `notes/02_qrw_vs_crw_theory.md` với bảng so sánh và công thức

---

### 1.3 — Ánh Xạ QRW → Vi Cấu Trúc Thị Trường

**Mô tả:** Xây dựng framework lý thuyết nối QRW với các khái niệm market microstructure.

**Các bước:**

- Định nghĩa ánh xạ: position `x ∈ Z` ↔ price level (tick); coin state `{↑, ↓}` ↔ buy/sell pressure
- Ánh xạ coin operator `C` ↔ market maker decision operator (probability of bid/ask flip)
- Xác định decoherence channel `D` ↔ noise từ market impact, latency, information asymmetry
- Ghi chú tương đương: probability distribution `P(x,t)` ↔ order flow density tại mức giá x vào thời điểm t
- Mô tả điều kiện khi mô hình suy biến về CRW (classical limit): khi decoherence hoàn toàn (`D → 1`)

**Deliverable:** File `notes/03_market_mapping.md` với sơ đồ ánh xạ và justification cho từng lựa chọn

---

### 1.4 — Notebook Kiểm Tra Toán Học (Symbolic Verification)

**Mô tả:** Verify toán học các công thức lý thuyết bằng SymPy và NumPy trước khi code simulation.

**Các bước:**

- Mở Jupyter notebook `notebooks/01_theory_verification.ipynb`
- Dùng SymPy verify: `U†U = I` (unitarity của evolution operator) với coin Hadamard
- Tính eigenvalues của `U` bằng NumPy — xác nhận tất cả nằm trên unit circle
- Tính analytical variance `Var(X_T) = t^2/2 - t/4 + O(1)` cho Hadamard coin, so sánh với simulation Monte Carlo nhanh (T ≤ 100 bước, 1000 lần)
- Export notebook thành PDF

**Deliverable:** `notebooks/01_theory_verification.ipynb` + `reports/theory_verification.pdf`

**[→ Phụ thuộc vào Task 1.1, 1.2]**

---

### 1.5 — Tổng Hợp Tài Liệu Lý Thuyết

**Mô tả:** Compile tất cả notes thành document tham khảo chính thức cho project.

**Các bước:**

- Gộp `01_dtqrw_formalism.md`, `02_qrw_vs_crw_theory.md`, `03_market_mapping.md` thành `docs/theory_summary.md`
- Thêm danh sách ký hiệu (notation glossary): `U`, `C`, `S`, `H_coin`, `P(x,t)`, `σ(t)`, v.v.
- Thêm bibliography với ≥ 5 papers tham khảo (BibTeX format)
- Convert sang PDF bằng Pandoc: `pandoc theory_summary.md -o theory_notes.pdf`

**Deliverable:** `docs/theory_notes.pdf` (≥ 10 trang)

**[→ Phụ thuộc vào Task 1.1, 1.2, 1.3, 1.4]**

---

### ✅ CHECKPOINT Phase 1

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| `theory_notes.pdf` tồn tại và ≥ 10 trang | Có đủ 3 sections lý thuyết | Thiếu section |
| Notebook 1.4 chạy không lỗi end-to-end | Tất cả cells execute thành công | Bất kỳ cell nào báo lỗi |
| Kiểm tra unitary: `||U†U - I||_F < 1e-10` | Norm nhỏ hơn ngưỡng | Norm ≥ 1e-10 |
| Variance QRW > Variance CRW tại T=100 | `Var_QRW / Var_CRW > 1.5` | Tỷ lệ ≤ 1.5 |

---

## Phase 2: Thu Thập & Xử Lý Dữ Liệu

**Mục tiêu:** Xây dựng pipeline tự động thu thập và làm sạch dữ liệu tick/order book thực tế, sẵn sàng cho modeling.

**Thời gian ước lượng:** 1.5 tuần (7–8 ngày làm việc)

**Công cụ / thư viện:** Python (pandas, NumPy, h5py, requests, websocket-client), dữ liệu từ Binance public API hoặc Polygon.io (free tier), hoặc dữ liệu lịch sử từ Kaggle

---

### 2.1 — Xác Định Nguồn Dữ Liệu & Thiết Lập Môi Trường

**Mô tả:** Chọn asset, exchange, và thiết lập environment Python đầy đủ.

**Các bước:**

- Tạo virtual environment: `python -m venv .venv && source .venv/bin/activate`
- Cài dependencies: `pip install pandas numpy scipy h5py requests websocket-client matplotlib seaborn tqdm`
- Tạo file `requirements.txt` và `config/data_config.yaml`
- Trong `data_config.yaml`, ghi rõ: symbol (ví dụ `BTC/USDT`), exchange (`binance`), time range, tick granularity
- Chọn nguồn dữ liệu phù hợp:
  - **Option A (Free, real-time):** Binance WebSocket API — `/ws/<symbol>@trade` và `/ws/<symbol>@depth`
  - **Option B (Historical, free):** Kaggle dataset "Crypto Order Book" hoặc Binance historical data từ `data.binance.vision`
  - **Khuyến nghị:** Dùng Option B để reproducibility — download 1 tháng dữ liệu BTC-USDT tick data

**Deliverable:** `requirements.txt`, `config/data_config.yaml`, môi trường hoạt động

---

### 2.2 — Thu Thập Dữ Liệu Tick

**Mô tả:** Download hoặc stream và lưu raw tick data vào disk.

**Các bước:**

- Tạo file `src/data/tick_downloader.py` với class `TickDownloader`
- Implement method `download_historical(symbol, start_date, end_date, save_path)`:
  - Gọi Binance historical data endpoint hoặc đọc Kaggle CSV
  - Lưu raw data dưới dạng compressed CSV: `data/raw/tick_{symbol}_{date}.csv.gz`
- Mỗi record cần có fields: `timestamp` (nanosecond epoch), `price` (float64), `quantity` (float64), `side` (buy/sell), `trade_id`
- Thêm logging với `loguru` — log số lượng records mỗi file và bất kỳ gap nào trong timestamp
- Verify: không có duplicate `trade_id`, timestamp monotonically increasing

**Deliverable:** `src/data/tick_downloader.py` + `data/raw/` chứa ≥ 7 ngày dữ liệu (≥ 1M records)

---

### 2.3 — Thu Thập Dữ Liệu Order Book (LOB Snapshots)

**Mô tả:** Thu thập Level-2 order book snapshots để phân tích bid-ask spread và depth.

**Các bước:**

- Tạo `src/data/orderbook_collector.py` với class `OrderBookCollector`
- Implement `get_snapshot(symbol, depth=20)`: trả về dict `{bids: [(price, qty), ...], asks: [(price, qty), ...]}`
- Lưu snapshots mỗi 1 giây vào HDF5: `data/raw/lob_{symbol}_{date}.h5`
  - HDF5 structure: `/snapshots/{timestamp}/bids`, `/snapshots/{timestamp}/asks`
- Tính và lưu kèm derived fields: `mid_price`, `bid_ask_spread`, `order_book_imbalance (OBI)`
  - `OBI = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)` tại top 5 levels
- Nếu không có real-time access: reconstruct LOB từ tick data + L2 snapshots Kaggle

**Deliverable:** `src/data/orderbook_collector.py` + `data/raw/lob_{symbol}.h5`

---

### 2.4 — Làm Sạch & Tiền Xử Lý Tick Data

**Mô tả:** Phát hiện và xử lý anomalies trong raw tick data trước khi model.

**Các bước:**

- Tạo `src/data/tick_processor.py` với class `TickProcessor`
- Implement các bước cleaning:
  1. **Filter outliers:** Loại bỏ trades có giá chênh lệch > 3σ so với rolling mean (window 100 ticks)
  2. **Handle gaps:** Mark timestamp gaps > 5 phút là "market closed" segment — không interpolate
  3. **Deduplication:** Drop duplicate `trade_id`
  4. **Normalization:** Tính log returns `r_t = ln(P_t / P_{t-1})` và price increments `ΔP_t = P_t - P_{t-1}`
- Lưu processed data: `data/processed/tick_processed_{symbol}_{date}.parquet` (dùng Parquet cho hiệu năng)
- Tạo data quality report: `reports/data_quality_{date}.txt` liệt kê số records removed, gap locations

**Deliverable:** `src/data/tick_processor.py` + `data/processed/*.parquet` + `reports/data_quality_*.txt`

**[→ Phụ thuộc vào Task 2.2]**

---

### 2.5 — Feature Engineering cho Modeling

**Mô tả:** Tạo các features từ tick/LOB data phục vụ trực tiếp cho QRW model.

**Các bước:**

- Tạo `src/data/feature_engineer.py` với class `FeatureEngineer`
- Implement các features:
  1. **Tick direction series** `d_t ∈ {+1, -1}`: dấu của `ΔP_t`, dùng làm "coin state" proxy
  2. **Trade intensity** `λ_t`: số trades mỗi giây (rate) — dùng làm timestep scaling
  3. **OBI series:** từ LOB data — dùng làm bias parameter cho coin operator
  4. **Autocorrelation of tick direction** `ρ_k = Corr(d_t, d_{t-k})` cho k = 1..20
  5. **Volume-weighted mid price** (VWMP) và deviation từ mid price
- Lưu feature matrix: `data/features/features_{symbol}_{date}.parquet`
- Thống kê mô tả: `reports/feature_stats_{date}.csv` (mean, std, min, max, skewness, kurtosis)

**Deliverable:** `src/data/feature_engineer.py` + `data/features/*.parquet`

**[→ Phụ thuộc vào Task 2.3, 2.4]**

---

### 2.6 — Unit Tests cho Data Pipeline

**Mô tả:** Đảm bảo pipeline hoạt động đúng và reproducible.

**Các bước:**

- Tạo `tests/test_data_pipeline.py` dùng `pytest`
- Test cases:
  1. `test_no_duplicate_trade_ids()`: assert `df['trade_id'].nunique() == len(df)`
  2. `test_timestamps_monotonic()`: assert `df['timestamp'].is_monotonic_increasing`
  3. `test_obi_range()`: assert `OBI ∈ [-1, 1]` for all rows
  4. `test_log_returns_finite()`: assert no `NaN` or `Inf` trong log returns sau processing
  5. `test_feature_matrix_shape()`: assert output shape matches expected columns
- Chạy: `pytest tests/test_data_pipeline.py -v` — tất cả phải PASS

**Deliverable:** `tests/test_data_pipeline.py` + output `pytest --tb=short` tất cả PASS

**[→ Phụ thuộc vào Task 2.4, 2.5]**

---

### ✅ CHECKPOINT Phase 2

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| Tick data: ≥ 1,000,000 records, không có duplicate trade_id | Verified bằng test 2.6 | Test fail hoặc < 1M records |
| Data quality: < 0.5% records removed sau cleaning | `(removed/total) < 0.005` | Tỷ lệ ≥ 0.5% (cần re-examine pipeline) |
| Feature matrix không có `NaN` hoặc `Inf` | `df.isnull().sum().sum() == 0` | Có giá trị null |
| Autocorrelation ρ_1 của tick direction có ý nghĩa thống kê (p < 0.05) | p-value < 0.05 | Không có autocorrelation (kiểm tra lại data) |
| Tất cả pytest tests PASS | 5/5 tests PASS | Bất kỳ test nào FAIL |

---

## Phase 3: Triển Khai Mô Hình QRW

**Mục tiêu:** Xây dựng implementation QRW hoàn chỉnh và tích hợp với dữ liệu thị trường thực tế.

**Thời gian ước lượng:** 2.5 tuần (12–13 ngày làm việc)

**Công cụ / thư viện:** NumPy, SciPy, (tùy chọn) PennyLane hoặc Qiskit Aer cho circuit representation — KHÔNG dùng quantum hardware

---

### 3.1 — Core QRW Engine (Pure State)

**Mô tả:** Implement DTQRW thuần túy trên 1D lattice bằng statevector simulation.

**Các bước:**

- Tạo `src/models/qrw_core.py` với class `QuantumRandomWalk`
- Constructor: `__init__(self, n_positions: int, coin: str = 'hadamard', initial_position: int = 0, initial_coin_state: np.ndarray = None)`
  - `n_positions`: kích thước không gian vị trí (ví dụ: 201 cho lattice từ -100 đến +100)
  - `initial_coin_state`: mặc định `(|0⟩ + i|1⟩)/√2` (symmetric initial state)
- Implement statevector: `self.psi` là numpy array shape `(2, n_positions)` — axis 0 là coin, axis 1 là position
  - `psi[0, x]` = amplitude của |↑, x⟩
  - `psi[1, x]` = amplitude của |↓, x⟩
- Method `step(self)`: áp dụng `U = S · (C ⊗ I)`
  - Coin step: `psi_new = coin_matrix @ psi` (matrix mul trên axis 0)
  - Shift step: `psi_new[0, x+1] = psi_coin[0, x]` (spin-up dịch phải), `psi_new[1, x-1] = psi_coin[1, x]` (spin-down dịch trái), dùng `np.roll` với boundary handling
- Method `get_probability(self) -> np.ndarray`: tính `P(x) = |psi[0,x]|^2 + |psi[1,x]|^2`
- Method `run(self, n_steps: int) -> np.ndarray`: chạy T bước, return probability array cuối cùng

**Deliverable:** `src/models/qrw_core.py`

---

### 3.2 — Coin Operators Library

**Mô tả:** Implement các coin operator khác nhau để thử nghiệm trong market context.

**Các bước:**

- Tạo `src/models/coin_operators.py` với các functions:
  1. `hadamard_coin() -> np.ndarray`: `(1/√2) * [[1,1],[1,-1]]`
  2. `grover_coin() -> np.ndarray`: `[[0,1],[1,0]]` (swap coin — đặc biệt cho mean-reversion)
  3. `biased_coin(theta: float) -> np.ndarray`: rotation matrix `[[cos(θ), -sin(θ)],[sin(θ), cos(θ)]]` — `theta` điều khiển drift
  4. `obi_adaptive_coin(obi: float) -> np.ndarray`: coin phụ thuộc order book imbalance
     - `theta = π/4 + α * obi` với α là sensitivity parameter
     - Khi `obi > 0` (buy pressure): coin bias về phía phải (price up)
     - Khi `obi < 0` (sell pressure): coin bias về phía trái (price down)
  5. `dephasing_channel(psi: np.ndarray, gamma: float) -> np.ndarray`: áp dụng decoherence
     - Mix quantum state với classical probability: `psi_new = (1-gamma)*psi + gamma*classical_component`
     - `gamma=0`: pure QRW; `gamma=1`: full decoherence = CRW

**Deliverable:** `src/models/coin_operators.py`

**[→ Phụ thuộc vào Task 3.1 để test]**

---

### 3.3 — Density Matrix Simulation (Mixed State / Decoherence)

**Mô tả:** Mở rộng core sang density matrix formalism để mô hình hóa noise thị trường.

**Các bước:**

- Thêm class `DensityMatrixQRW` vào `src/models/qrw_core.py`
- State: `self.rho` — density matrix shape `(2*n_positions, 2*n_positions)`, complex128
  - Index encoding: state `|c, x⟩` ↔ index `c * n_positions + x`
- Method `step_with_decoherence(self, gamma: float)`:
  1. Unitary evolution: `rho_new = U @ rho @ U†`
  2. Dephasing: `rho_deph[i,j] = rho_new[i,j] * exp(-gamma)` cho `i ≠ j` (off-diagonal decay)
- Method `get_probability(self) -> np.ndarray`: `P(x) = sum over coin states of rho[(c,x),(c,x)]`
- **Lưu ý hiệu năng:** density matrix `(402 x 402)` complex128 = ~2.6 MB, simulation 1000 bước ≈ 30s trên CPU. Nếu cần nhanh hơn: dùng sparse matrix hoặc giới hạn `n_positions ≤ 201`

**Deliverable:** class `DensityMatrixQRW` trong `src/models/qrw_core.py`

**[→ Phụ thuộc vào Task 3.1, 3.2]**

---

### 3.4 — Market-Adapted QRW Model

**Mô tả:** Tích hợp dữ liệu thị trường thực tế (OBI, tick direction) vào QRW evolution.

**Các bước:**

- Tạo `src/models/qrw_market_sim.py` với class `MarketQRW`
- Constructor: `__init__(self, tick_data: pd.DataFrame, config: dict)`
  - `tick_data`: processed DataFrame với columns `['timestamp', 'price', 'tick_direction', 'obi', 'trade_intensity']`
  - `config`: dict với keys `n_positions`, `gamma_base`, `alpha_obi`, `coin_type`
- Method `calibrate(self) -> dict`:
  - Ước lượng `gamma` (decoherence rate) từ autocorrelation decay của tick direction series
  - `gamma_est = -log(|ρ_1|)` với `ρ_1` là lag-1 autocorrelation
  - Ước lượng `alpha` (OBI sensitivity) bằng linear regression: `Δprice ~ OBI`
  - Lưu calibrated parameters vào `results/calibrated_params.json`
- Method `simulate(self, T: int) -> pd.DataFrame`:
  - Tại mỗi timestep t: lấy `obi_t` từ data, tính `coin_t = obi_adaptive_coin(obi_t, alpha)`
  - Chạy 1 bước QRW với coin này, áp dụng decoherence với `gamma`
  - Lưu probability distribution `P(x, t)` tại mỗi bước
  - Return DataFrame với columns `['t', 'position', 'probability']`
- Method `simulate_price_path(self, n_paths: int) -> np.ndarray`:
  - Sample `n_paths` trajectories bằng cách sample vị trí từ probability distribution tại mỗi bước
  - Return array shape `(n_paths, T)`

**Deliverable:** `src/models/qrw_market_sim.py` + `results/calibrated_params.json`

**[→ Phụ thuộc vào Task 3.2, 3.3, 2.5]**

---

### 3.5 — Validation của QRW Implementation

**Mô tả:** Kiểm tra implementation đúng với các tính chất lý thuyết đã biết trước khi dùng với data thực.

**Các bước:**

- Tạo `tests/test_qrw_implementation.py`
- Test cases:
  1. `test_probability_normalization()`: `sum(P(x, t)) == 1.0` tại mọi timestep (tolerance 1e-10)
  2. `test_ballistic_spreading()`: với Hadamard coin, `Var(X_T) / T^2 → 1/2` khi T → ∞. Test tại T=500: assert `0.4 < Var/T^2 < 0.6`
  3. `test_unitarity_preserved()`: `||rho||_trace = 1.0` sau mỗi bước (density matrix)
  4. `test_decoherence_classical_limit()`: khi `gamma → ∞`, `Var(X_T)/T → 0.5` (CRW scaling)
  5. `test_symmetric_initial_state()`: với `|ψ_0⟩ = (|0⟩ + i|1⟩)/√2`, P(x,t) phải symmetric quanh x=0
  6. `test_grover_coin_localization()`: Grover coin trên cycle graph phải show localization (std không tăng tuyến tính)
- Chạy `pytest tests/test_qrw_implementation.py -v --tb=short`

**Deliverable:** `tests/test_qrw_implementation.py` — tất cả 6 tests PASS

**[→ Phụ thuộc vào Task 3.1, 3.2, 3.3]**

---

### 3.6 — Performance Optimization

**Mô tả:** Tối ưu code để simulation đủ nhanh cho backtesting (≥ 1000 paths × 1000 steps < 60s).

**Các bước:**

- Profile bottleneck: `python -m cProfile -s cumulative src/models/qrw_core.py > profile_output.txt`
- Áp dụng tối ưu (theo thứ tự ưu tiên):
  1. Vectorize coin + shift operation bằng NumPy broadcasting thay vì Python loops
  2. Dùng `np.einsum` cho matrix operations trên statevector
  3. Pre-allocate arrays: `np.empty` thay vì tạo array mới mỗi bước
  4. Nếu cần: dùng `numba.jit` cho vòng lặp time evolution
- Benchmark: `timeit` module — đo thời gian `run(1000 steps)` × 1000 paths
- Ghi kết quả benchmark vào `reports/performance_benchmark.txt`

**Deliverable:** `reports/performance_benchmark.txt` + code optimized trong `qrw_core.py`

**[→ Phụ thuộc vào Task 3.1, 3.5]**

---

### ✅ CHECKPOINT Phase 3

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| Tất cả 6 test cases trong Task 3.5 PASS | 6/6 PASS | Bất kỳ test nào FAIL |
| Ballistic spreading: `Var(X_T)/T^2 ∈ [0.4, 0.6]` tại T=500 | Trong range | Ngoài range |
| Performance: 1000 paths × 1000 steps < 60s | Đo được < 60s | ≥ 60s (cần optimize thêm) |
| `calibrated_params.json` tồn tại với giá trị hợp lệ | `gamma > 0`, `alpha` finite | File không tồn tại hoặc giá trị NaN |
| Decoherence limit: khi gamma=10, Var scaling ≈ linear (CRW) | `|Var/T - 0.5| < 0.1` | Ngoài range |

---

## Phase 4: Classical Baseline & Benchmark

**Mục tiêu:** Xây dựng các mô hình classical làm baseline để so sánh công bằng với QRW.

**Thời gian ước lượng:** 1 tuần (5 ngày làm việc)

**Công cụ / thư viện:** NumPy, SciPy, pandas, statsmodels

---

### 4.1 — Classical Random Walk (CRW) Baseline

**Mô tả:** Implement Simple Random Walk và Biased Random Walk làm baseline cơ bản nhất.

**Các bước:**

- Tạo `src/models/classical_rw.py` với class `ClassicalRandomWalk`
- Implement:
  1. `SimpleRW`: `X_{t+1} = X_t + ξ_t` với `ξ_t ~ Bernoulli(0.5)` (±1)
  2. `BiasedRW`: `P(ξ=+1) = p` — calibrate `p` từ empirical tick direction frequency
  3. `CorrelatedRW`: incorporate lag-1 autocorrelation: `P(ξ_t = ξ_{t-1}) = (1+ρ)/2`
- Method `simulate(n_steps, n_paths) -> np.ndarray`: return shape `(n_paths, n_steps+1)`
- Calibrate parameters từ cùng dataset dùng cho QRW để đảm bảo so sánh công bằng

**Deliverable:** `src/models/classical_rw.py`

---

### 4.2 — GARCH Model Baseline

**Mô tả:** Implement GARCH(1,1) làm baseline tiêu chuẩn trong tài chính định lượng.

**Các bước:**

- Dùng `arch` library: `pip install arch`
- Tạo `src/models/garch_model.py` với class `GARCHBaseline`
- Fit GARCH(1,1) trên log returns từ processed tick data: `arch_model(returns, vol='Garch', p=1, q=1).fit()`
- Method `simulate(n_steps, n_paths) -> np.ndarray`: simulate paths từ fitted GARCH
- Lưu model params: `results/garch_params.json` (omega, alpha, beta, log-likelihood)
- In AIC, BIC: `model.aic`, `model.bic` — lưu vào `results/model_comparison_table.csv`

**Deliverable:** `src/models/garch_model.py` + `results/garch_params.json`

**[→ Phụ thuộc vào Task 2.4]**

---

### 4.3 — Geometric Brownian Motion (GBM) Baseline

**Mô tả:** Implement GBM (Black-Scholes diffusion) làm baseline continuous-time.

**Các bước:**

- Tạo `src/models/gbm_model.py` với class `GBMBaseline`
- Estimate `μ` (drift) và `σ` (volatility) từ log returns: MLE estimates
- Simulate: `S_{t+Δt} = S_t * exp((μ - σ²/2)Δt + σ√Δt * Z)` với `Z ~ N(0,1)`
- Method `simulate(n_steps, n_paths, dt=1) -> np.ndarray`
- So sánh empirical volatility vs GBM volatility → lưu vào `results/model_comparison_table.csv`

**Deliverable:** `src/models/gbm_model.py`

**[→ Phụ thuộc vào Task 2.4]**

---

### 4.4 — Benchmark Suite

**Mô tả:** Chạy tất cả models trên cùng dataset, thu thập metrics đầu ra để so sánh.

**Các bước:**

- Tạo `src/evaluation/benchmark_suite.py` với class `BenchmarkSuite`
- Chạy mỗi model với config:
  - `n_paths = 5000`
  - `n_steps = 500`
  - Initial condition: price tại t=0 của test set (không train)
- Thu thập các metrics cho từng model:
  1. **Mean Absolute Error (MAE)** của predicted vs realized price distribution (Earth Mover's Distance)
  2. **Variance ratio** `Var(X_T) / T` — đánh giá scaling
  3. **Tail heaviness** — empirical kurtosis của return distribution
  4. **Hit rate** — tỷ lệ dự báo đúng hướng tick (up/down) tại horizon 1, 5, 10 bước
  5. **Log-likelihood** của realized path dưới mỗi model
- Lưu tất cả kết quả vào `results/benchmark_results.csv` với format: `model, metric, value, std`

**Deliverable:** `src/evaluation/benchmark_suite.py` + `results/benchmark_results.csv`

**[→ Phụ thuộc vào Task 3.4, 4.1, 4.2, 4.3]**

---

### ✅ CHECKPOINT Phase 4

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| `benchmark_results.csv` tồn tại với ≥ 4 models × 5 metrics | File có đủ rows | Thiếu model hoặc metric |
| CRW SimpleRW variance scaling gần 0.5: `|Var/T - 0.5| < 0.05` | Đúng như lý thuyết | Sai — kiểm tra lại implementation |
| GARCH fit converged: `model.convergence_flag == 0` | Converged | Không converge — thử khác starting values |
| QRW variance ratio > CRW variance ratio trong regime không có decoherence | `Var_QRW/T > Var_CRW/T * 1.3` | Không thể hiện sự khác biệt |

---

## Phase 5: Kiểm Định Thống Kê

**Mục tiêu:** Thực hiện đầy đủ các kiểm định thống kê để xác định liệu QRW có mô hình hóa tốt hơn CRW/GBM hay không, với độ tin cậy đo lường cụ thể.

**Thời gian ước lượng:** 1.5 tuần (7–8 ngày làm việc)

**Công cụ / thư viện:** SciPy, statsmodels, pingouin, pandas

---

### 5.1 — Kiểm Định Phân Phối (Distribution Tests)

**Mô tả:** So sánh phân phối xác suất simulated vs empirical bằng các test thống kê.

**Các bước:**

- Tạo `src/evaluation/statistical_tests.py` với class `StatisticalTestSuite`
- Implement các tests:
  1. **Kolmogorov-Smirnov (KS) Test** (`scipy.stats.ks_2samp`):
     - So sánh simulated return distribution vs empirical return distribution
     - Chạy cho từng model: QRW, CRW, GARCH, GBM
     - Lưu: KS statistic và p-value cho từng model
  2. **Anderson-Darling Test** (`scipy.stats.anderson_ksamp`):
     - Nhạy hơn KS với tail behavior — quan trọng cho tài chính
     - Áp dụng tại multiple horizons: t=1, t=10, t=50, t=100
  3. **Wasserstein Distance / Earth Mover's Distance** (`scipy.stats.wasserstein_distance`):
     - Metric liên tục đo "khoảng cách" giữa hai phân phối
     - Lợi thế: không cần binning, nhạy với shape toàn bộ distribution
- Lưu tất cả kết quả vào `results/distribution_tests.csv`

**Deliverable:** Phần distribution tests trong `src/evaluation/statistical_tests.py` + `results/distribution_tests.csv`

**[→ Phụ thuộc vào Task 4.4]**

---

### 5.2 — Kiểm Định Variance Scaling

**Mô tả:** Kiểm tra thống kê QRW có ballistic scaling hay không, và so sánh với data thực.

**Các bước:**

- Implement test `variance_scaling_test(paths, max_T=500) -> dict`:
  1. Tính `Var(X_t)` cho t = 1, 5, 10, 20, 50, 100, 200, 500
  2. Fit log-linear model: `log(Var) = α + β * log(t)` bằng OLS
  3. Kiểm tra `β`: QRW lý thuyết `β ≈ 2`, CRW `β ≈ 1`, thực tế có thể nằm giữa
  4. Confidence interval của `β` (95%): dùng bootstrapping (1000 iterations)
  5. Test H0: `β = 1` (CRW) vs Ha: `β > 1` bằng t-test trên bootstrap distribution
- Áp dụng cho: empirical data, QRW sim, CRW sim, GARCH sim
- Plot `log(Var) vs log(t)` với regression lines → save `figures/variance_scaling.png`

**Deliverable:** `results/variance_scaling_results.csv` + `figures/variance_scaling.png`

**[→ Phụ thuộc vào Task 5.1]**

---

### 5.3 — Kiểm Định Autocorrelation

**Mô tả:** So sánh autocorrelation structure của simulated paths vs data thực.

**Các bước:**

- Implement `autocorrelation_test(returns, max_lag=20) -> dict`:
  1. Tính ACF (Autocorrelation Function) và PACF tại lag 1..20 bằng `statsmodels.tsa.stattools.acf`
  2. Tính empirical ACF từ tick data
  3. **Ljung-Box test** (`statsmodels.stats.diagnostic.acorr_ljungbox`): test H0 "no autocorrelation" tại lag 1, 5, 10
  4. So sánh ACF profile: QRW simulated vs Empirical vs CRW — vẽ ACF plots
  5. Tính `MSE` giữa simulated ACF và empirical ACF: `MSE_ACF = mean((ACF_sim - ACF_emp)^2)` cho lag 1..20
- Lưu: Ljung-Box p-values và `MSE_ACF` cho từng model vào `results/autocorrelation_tests.csv`

**Deliverable:** `results/autocorrelation_tests.csv` + `figures/acf_comparison.png`

**[→ Phụ thuộc vào Task 5.1]**

---

### 5.4 — Tail Risk Analysis (Heavy Tails)

**Mô tả:** Phân tích khả năng model capture extreme events — quan trọng trong risk management.

**Các bước:**

- Implement `tail_analysis(returns) -> dict`:
  1. **Kurtosis test** (`scipy.stats.kurtosistest`): test H0 "normal kurtosis" — empirical data thường reject (fat tails)
  2. **Hill Estimator** cho tail index `α_tail`:
     - Sort `|returns|` giảm dần, lấy top 10% làm tail
     - `1/α_tail = mean(log(X_i / X_k))` với X_k là threshold
  3. **Value at Risk (VaR) 95%, 99%** bằng empirical quantile — so sánh giữa models
  4. **Expected Shortfall (CVaR)** tại 95%, 99% — mean loss beyond VaR
  5. So sánh tail index của QRW sim vs CRW sim vs empirical: `α_QRW`, `α_CRW`, `α_emp`

- Lưu `results/tail_analysis.csv` với columns: `model, kurtosis, tail_index, VaR_95, VaR_99, CVaR_95, CVaR_99`

**Deliverable:** `results/tail_analysis.csv`

**[→ Phụ thuộc vào Task 4.4]**

---

### 5.5 — Tổng Hợp Kết Quả Kiểm Định

**Mô tả:** Compile tất cả test results thành bảng kết quả thống nhất để đánh giá.

**Các bước:**

- Tạo `src/evaluation/results_compiler.py`
- Load tất cả CSV results từ `results/` directory
- Tạo master comparison table: `results/final_comparison_table.csv`
  - Rows: từng model (QRW-Hadamard, QRW-OBI-Adaptive, CRW-Simple, CRW-Correlated, GARCH, GBM)
  - Columns: KS_stat, KS_pval, Wasserstein, Var_scaling_beta, ACF_MSE, kurtosis, VaR_99, CVaR_99
- Tạo "scorecard" — cho mỗi metric, rank models (1=tốt nhất, 6=kém nhất)
- Tổng hợp: `results/scorecard.csv` với `overall_rank` là mean rank

**Deliverable:** `results/final_comparison_table.csv` + `results/scorecard.csv`

**[→ Phụ thuộc vào Task 5.1, 5.2, 5.3, 5.4]**

---

### ✅ CHECKPOINT Phase 5

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| Tất cả 4 test categories có kết quả đầy đủ | 4 CSV files tồn tại | Bất kỳ file nào thiếu |
| KS test p-value của QRW vs empirical được tính | Có p-value cụ thể | Chưa chạy test |
| Variance scaling: β được ước lượng với confidence interval | CI 95% được tính | Chỉ có point estimate |
| QRW tail index `α_QRW` được tính và so sánh với `α_emp` | Hai giá trị tồn tại | Chỉ có một |
| Scorecard tạo ra được | `scorecard.csv` tồn tại với overall_rank | Chưa tổng hợp |

---

## Phase 6: Trực Quan Hóa & Báo Cáo Cuối

**Mục tiêu:** Tạo bộ visualization hoàn chỉnh và báo cáo kỹ thuật trình bày đầy đủ phương pháp, kết quả, và kết luận.

**Thời gian ước lượng:** 1 tuần (5 ngày làm việc)

**Công cụ / thư viện:** Matplotlib, Seaborn, Plotly (interactive), Pandoc/LaTeX

---

### 6.1 — Visualization Suite

**Mô tả:** Tạo bộ đầy đủ các biểu đồ chuẩn mực cho research paper.

**Các bước:**

- Tạo `src/visualization/plot_suite.py` với các functions:
  1. `plot_probability_evolution(qrw, t_snapshots)`: animation của `P(x,t)` theo thời gian — QRW vs CRW side-by-side. Save: `figures/prob_evolution.gif`
  2. `plot_variance_scaling()`: log-log plot `Var(X_t)` vs t cho tất cả models với regression lines. Save: `figures/variance_scaling.png`
  3. `plot_return_distribution_comparison()`: KDE plots của return distributions (empirical vs QRW vs CRW vs GARCH vs GBM). Save: `figures/return_distributions.png`
  4. `plot_acf_comparison()`: ACF bar charts cho 4 models. Save: `figures/acf_comparison.png`
  5. `plot_sample_paths()`: 10 sample price paths từ QRW-OBI-Adaptive và CRW trên cùng axis với empirical. Save: `figures/sample_paths.png`
  6. `plot_heatmap_coin_operator()`: Heatmap của `|U|^2` (magnitude squared) để visualize evolution operator. Save: `figures/coin_operator_heatmap.png`
  7. `plot_benchmark_scorecard()`: Horizontal bar chart của overall rankings. Save: `figures/scorecard.png`
- Tất cả figures: DPI=150, figsize=(10,6), sử dụng scientific color palette (seaborn `colorblind`)

**Deliverable:** `src/visualization/plot_suite.py` + 7 figures trong `figures/`

**[→ Phụ thuộc vào Task 5.5]**

---

### 6.2 — Interactive Dashboard (Optional nhưng Recommended)

**Mô tả:** Tạo dashboard tương tác để explore QRW parameters và xem kết quả real-time.

**Các bước:**

- Dùng Plotly Dash hoặc Streamlit: `pip install streamlit plotly`
- Tạo `src/dashboard/app.py`:
  - Sidebar controls: slider cho `gamma` (decoherence), `alpha` (OBI sensitivity), `coin_type` dropdown, `n_steps` slider
  - Real-time plot: probability distribution `P(x,t)` sau n_steps
  - Panel 2: variance scaling plot cập nhật theo parameters
  - Panel 3: comparison table với current QRW config vs baselines
- Chạy: `streamlit run src/dashboard/app.py`
- Capture screenshot: `figures/dashboard_screenshot.png`

**Deliverable:** `src/dashboard/app.py` + `figures/dashboard_screenshot.png`

**[→ Phụ thuộc vào Task 3.4, 6.1]**

---

### 6.3 — Báo Cáo Kỹ Thuật Cuối

**Mô tả:** Viết báo cáo kỹ thuật đầy đủ theo format research paper.

**Các bước:**

- Tạo `docs/final_report.md` với cấu trúc:
  1. **Abstract** (250 từ): tóm tắt mục tiêu, phương pháp, kết quả chính
  2. **1. Introduction**: động lực nghiên cứu, gap trong literature
  3. **2. Theoretical Framework**: QRW formalism, ánh xạ sang market microstructure (reference `theory_notes.pdf`)
  4. **3. Data & Methodology**: mô tả dataset, pipeline, calibration approach
  5. **4. Model Implementations**: QRW variants, baselines, parameter calibration
  6. **5. Results**: tất cả statistical tests với tables và figures (embed `figures/*.png`)
  7. **6. Discussion**: interpret kết quả, limitations, khi nào QRW tốt hơn/kém hơn CRW
  8. **7. Conclusion & Future Work**: 3–5 hướng mở rộng (QRW với machine learning coin, 2D QRW cho multi-asset, continuous-time CTQW)
  9. **References**: ≥ 10 papers
- Convert sang PDF: `pandoc docs/final_report.md --pdf-engine=xelatex -o docs/final_report.pdf`
- Kiểm tra: PDF ≥ 15 trang, tất cả figures được embed đúng

**Deliverable:** `docs/final_report.pdf` (≥ 15 trang)

**[→ Phụ thuộc vào Task 6.1, 5.5]**

---

### 6.4 — Slide Trình Bày

**Mô tả:** Tạo bộ slide gọn cho presentation (journal club / seminar format).

**Các bước:**

- Tạo 15–20 slides trong Markdown/Marp hoặc PowerPoint:
  - Slide 1: Title + authors
  - Slides 2–4: Motivation + QRW vs CRW intuition (dùng `prob_evolution.gif`)
  - Slides 5–7: Data description + pipeline
  - Slides 8–10: Model architecture + calibration
  - Slides 11–14: Key results (variance scaling, distribution comparison, scorecard)
  - Slides 15–16: Conclusion + Future Work
- Export: `docs/presentation_slides.pdf`

**Deliverable:** `docs/presentation_slides.pdf`

**[→ Phụ thuộc vào Task 6.1, 6.3]**

---

### 6.5 — Repository Cleanup & Documentation

**Mô tả:** Đảm bảo codebase sạch, documented, và reproducible cho người khác.

**Các bước:**

- Tạo `README.md` hoàn chỉnh với: project overview, installation steps, how to run each component, expected outputs
- Đảm bảo tất cả modules có docstrings (Google format)
- Tạo `Makefile` với targets: `make data`, `make simulate`, `make test`, `make report`
- Đảm bảo `requirements.txt` với pinned versions: `pip freeze > requirements.txt`
- Chạy `pytest --tb=short` lần cuối — tất cả tests PASS
- Tạo structure diagram của project trong `README.md`

**Deliverable:** `README.md` hoàn chỉnh + `Makefile` + final `requirements.txt`

---

### ✅ CHECKPOINT Phase 6 (Final)

| Tiêu Chí | PASS | FAIL |
|----------|------|------|
| Tất cả 7 figures tồn tại và không phải blank | `figures/*.png` size > 50KB mỗi file | File trống hoặc thiếu |
| `final_report.pdf` ≥ 15 trang | Đếm trang được | < 15 trang |
| Report có ≥ 1 table kết quả quantitative | Bảng tồn tại với p-values | Chỉ có mô tả định tính |
| `pytest` cuối cùng: tất cả tests PASS | 0 failures, 0 errors | Bất kỳ failure |
| `README.md` có đủ installation + run instructions | Người khác có thể chạy được | Thiếu steps |

---

## Cấu Trúc Thư Mục Dự Án

```
qrw_market_microstructure/
│
├── config/
│   └── data_config.yaml
│
├── data/
│   ├── raw/                    # Tick data, LOB snapshots
│   ├── processed/              # Cleaned Parquet files
│   └── features/               # Feature matrices
│
├── docs/
│   ├── theory_notes.pdf        # Phase 1 deliverable
│   ├── final_report.pdf        # Phase 6 deliverable
│   └── presentation_slides.pdf
│
├── figures/                    # Tất cả plots
│
├── notebooks/
│   └── 01_theory_verification.ipynb
│
├── notes/
│   ├── 01_dtqrw_formalism.md
│   ├── 02_qrw_vs_crw_theory.md
│   └── 03_market_mapping.md
│
├── reports/                    # Intermediate results
│
├── results/                    # CSV kết quả thống kê
│
├── src/
│   ├── data/
│   │   ├── tick_downloader.py
│   │   ├── orderbook_collector.py
│   │   ├── tick_processor.py
│   │   └── feature_engineer.py
│   ├── models/
│   │   ├── qrw_core.py
│   │   ├── coin_operators.py
│   │   ├── qrw_market_sim.py
│   │   ├── classical_rw.py
│   │   ├── garch_model.py
│   │   └── gbm_model.py
│   ├── evaluation/
│   │   ├── benchmark_suite.py
│   │   ├── statistical_tests.py
│   │   └── results_compiler.py
│   └── visualization/
│       └── plot_suite.py
│
├── tests/
│   ├── test_data_pipeline.py
│   └── test_qrw_implementation.py
│
├── Makefile
├── README.md
└── requirements.txt
```

---

## Rủi Ro & Giải Pháp

### Rủi Ro 1: Dữ Liệu Tick Không Đủ Chất Lượng / Khó Truy Cập

**Mô tả:** Free API bị rate limit, data có gaps lớn, hoặc data format không consistent giữa ngày.

**Dấu hiệu nhận biết:** Nhiều hơn 5% records bị loại trong Task 2.4; timestamp gaps > 10 phút trong giờ giao dịch.

**Giải pháp:**
- **Chính:** Dùng Binance historical data từ `data.binance.vision` (free, pre-packaged zip files theo ngày)
- **Dự phòng 1:** Dùng Kaggle dataset "Crypto LOB Dataset" (sẵn có, không cần API)
- **Dự phòng 2:** Simulate synthetic tick data với known properties (AR(1) returns, Poisson trade arrival) — vẫn đủ để kiểm định mô hình
- **Không làm:** Interpolate gaps trong tick data — sẽ tạo artifacts trong autocorrelation analysis

---

### Rủi Ro 2: QRW Không Thể Hiện Sự Khác Biệt Rõ Ràng Với CRW Trên Data Thực

**Mô tả:** Sau khi thêm decoherence và calibrate với market data, QRW converge về CRW behavior và không có superior performance.

**Dấu hiệu nhận biết:** KS test p-value của QRW vs empirical ≈ CRW vs empirical; scorecard rank QRW thấp.

**Giải pháp:**
- **Chính:** Đây là kết quả hợp lệ về mặt khoa học — report honestly với analysis tại sao decoherence mạnh (thị trường hiệu quả)
- **Hướng cứu vãn:** Thử `OBI-Adaptive QRW` với nhiều mức alpha; thử segmentation (QRW fit tốt hơn trong low-liquidity regimes)
- **Extension:** Thêm 2-period QRW với memory (non-Markovian coin) — có thể capture microstructure noise tốt hơn
- **Không làm:** p-hacking hay cherry-pick kết quả favoring QRW

---

### Rủi Ro 3: Hiệu Năng Simulation Quá Chậm

**Mô tả:** Density matrix simulation với n_positions=201 và 1000 paths × 1000 steps mất > 5 phút, không feasible cho calibration grid search.

**Dấu hiệu nhận biết:** Task 3.6 benchmark > 60s.

**Giải pháp:**
- **Chính:** Dùng statevector (pure state) thay vì density matrix khi không cần mixed states — 100× nhanh hơn
- **Tối ưu 1:** Numba JIT compilation cho evolution loop: `@numba.jit(nopython=True, parallel=True)` — speedup 10–50×
- **Tối ưu 2:** Giảm n_positions về 101 (lattice ±50) — đủ cho 1000 bước mà không wrap
- **Tối ưu 3:** Dùng sparse matrix `scipy.sparse` cho shift operator (shift matrix là sparse)
- **Fallback:** Giới hạn benchmark ở 100 paths × 500 steps — vẫn đủ cho statistical tests

---

### Rủi Ro 4: Lỗi Toán Học / Implementation Bug Khó Phát Hiện

**Mô tả:** Bug trong shift operator (wrap-around boundary, off-by-one) hoặc coin application dẫn đến kết quả sai nhưng vẫn "trông hợp lý".

**Dấu hiệu nhận biết:** Variance scaling exponent β > 2.1 (vật lý không hợp lệ) hoặc probability không normalize.

**Giải pháp:**
- **Phòng ngừa:** Task 3.5 kiểm định 6 properties đã biết từ lý thuyết — đây là defense line đầu tiên
- **Debug protocol:**
  1. Verify step-by-step với n_positions=5, T=3 (tractable bằng tay)
  2. Compare vs reference implementation: `pip install quantumwalk` (third-party) cho small test cases
  3. Kiểm tra: `psi` array không có NaN sau mỗi step (thêm `assert`)
- **Boundary condition:** Dùng reflecting boundary (x=-N → không dịch; x=+N → không dịch) thay vì periodic — tránh unphysical wrap-around

---

### Rủi Ro 5: Kiến Thức Nền Tảng QRW Không Đủ Gây Hiểu Lầm Kết Quả

**Mô tả:** Nhầm lẫn giữa quantum coherence vs classical correlation; interpret ballistic spreading sai trong market context.

**Dấu hiệu nhận biết:** Kết luận trong Phase 6 report không consistent với kết quả Phase 5.

**Giải pháp:**
- **Chính:** Đọc kỹ Task 1.1–1.3 trước khi code bất kỳ model nào — không bỏ qua Phase 1
- **Resources bổ sung:**
  - Kempe (2003) "Quantum random walks: an introductory overview" — overview rõ ràng nhất
  - Venegas-Andraca (2012) "Quantum walks: a comprehensive review" — reference đầy đủ
  - Romanelli et al. "Decoherence in the quantum walk on the line" — decoherence specifics
- **Peer check:** Sau Phase 1, viết 1-paragraph explanation của QRW cho người không chuyên — nếu không giải thích được, đọc lại
- **Không làm:** Bắt đầu code Phase 3 khi chưa qua Checkpoint Phase 1

---

## Tổng Kết Timeline

```
Tuần 1–2:   Phase 1 — Lý thuyết QRW (theory_notes.pdf)
Tuần 3–4:   Phase 2 — Data pipeline (tick + LOB processing)
Tuần 4–7:   Phase 3 — QRW implementation + market model
Tuần 7–8:   Phase 4 — Classical baselines + benchmark
Tuần 8–9:   Phase 5 — Statistical validation
Tuần 9–10:  Phase 6 — Visualization + final report
─────────────────────────────────────────────────────
Tổng:       ~9.5 tuần (~285 giờ làm việc thực tế)
```

> **Lưu ý quan trọng:** Ước lượng trên giả định researcher đã quen với Python scientific stack và không gặp blocking issues với data access. Nếu đây là lần đầu làm việc với QRW formalism, Phase 1 có thể kéo dài thêm 3–5 ngày. Luôn hoàn thành Checkpoint của Phase trước trước khi bắt đầu Phase tiếp theo.
