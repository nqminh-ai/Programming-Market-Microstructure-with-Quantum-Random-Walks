# Phase 2 data directories

## Current artifact status

- Workspace có các file theo ngày 13-05 đến 12-06-2026 cho BTCUSDT, ETHUSDT
  và BNBUSDT. Đây là dữ liệu development đã được xem, không phải holdout
  confirmatory untouched.
- Artifact BTCUSDT đang dùng trong báo cáo cũ chỉ có 1.908 tick, khoảng 118,5
  giây; không được mô tả là một ngày giao dịch đầy đủ.
- Không có tập confirmatory hợp lệ. Kế hoạch thu thập mới nằm tại
  `docs/data_collection_todo.md`.

- `assets/<symbol>/raw/`: downloaded trades and LOB HDF5 snapshots.
- `assets/<symbol>/processed/`: cleaned tick Parquet files.
- `assets/<symbol>/features/`: QRW-oriented feature matrices.

Market data is intentionally not committed or generated as fake production data.
Unit tests use deterministic synthetic fixtures under pytest temporary directories.

Feature engineering uses the configured imbalance source:

- `auto`: use same-period LOB data when available, otherwise use causal
  trade-volume imbalance;
- `lob`: require `raw/lob_<symbol>_<date>.h5`;
- `trade_imbalance`: force the trade-only proxy.

The proxy is

```text
(trailing buy volume - trailing sell volume)
------------------------------------------------
(trailing buy volume + trailing sell volume)
```

It is calculated separately inside each gap segment and shifted by one trade, so
the current trade cannot predict itself. The default window is 100 trades with 20
warm-up observations. `obi_valid` identifies rows past that warm-up.

Synthetic trade imbalance is not reconstructed order-book depth. Metadata records
`obi_source`, `obi_is_proxy`, window length, lag, formula, and valid coverage so
LOB OBI and trade-flow imbalance cannot be silently mixed.

To rebuild historical trade-only features:

```text
python -m scripts.pipelines.phase2_pipeline process
python -m scripts.pipelines.phase2_pipeline features --obi-source trade_imbalance
python -m scripts.pipelines.phase2_pipeline checkpoint
```

For a synchronized live trial window:

```text
python -m scripts.pipelines.phase2_pipeline collect-live --duration-seconds 120
python -m scripts.pipelines.phase2_pipeline process --input data/assets/btcusdt/raw/tick_BTCUSDT_<date>_live.csv.gz
python -m scripts.pipelines.phase2_pipeline features --input data/assets/btcusdt/processed/tick_processed_BTCUSDT_<date>.parquet
python -m scripts.pipelines.phase3_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
python -m scripts.pipelines.phase4_pipeline --feature-path data/assets/btcusdt/features/features_BTCUSDT_<date>.parquet
```

The live collector subscribes to Binance trade and partial-depth streams over one
combined WebSocket. Trades are persisted only after the first LOB snapshot so the
feature join begins with contemporaneous order-book information.
