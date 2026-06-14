"""Unit and integration tests for the Phase 2 data pipeline."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer
from src.data.live_market_collector import LiveMarketCollector
from src.data.orderbook_collector import OrderBookCollector
from src.data.tick_downloader import TickDownloader
from src.data.tick_processor import TickProcessor


@pytest.fixture
def raw_ticks() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    count = 500
    timestamps = 1_750_000_000_000_000_000 + np.arange(count) * 100_000_000
    timestamps[250:] += 6 * 60 * 1_000_000_000

    increments = rng.choice([-0.01, 0.0, 0.01], size=count, p=[0.35, 0.30, 0.35])
    price = 100.0 + np.cumsum(increments)
    quantity = rng.uniform(0.01, 2.0, size=count)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": price,
            "quantity": quantity,
            "side": np.where(increments >= 0, "buy", "sell"),
            "trade_id": np.arange(count),
        }
    )
    duplicate = frame.iloc[[100]].copy()
    return pd.concat([frame, duplicate], ignore_index=True)


@pytest.fixture
def processed_ticks(raw_ticks: pd.DataFrame) -> pd.DataFrame:
    processor = TickProcessor(
        rolling_window=50,
        outlier_z_score=8.0,
        gap_threshold_seconds=300,
    )
    processed, _ = processor.process(raw_ticks)
    return processed


@pytest.fixture
def lob_frame(processed_ticks: pd.DataFrame) -> pd.DataFrame:
    timestamps = processed_ticks["timestamp"].iloc[::10].to_numpy()
    mid = processed_ticks["price"].iloc[::10].to_numpy()
    obi = np.linspace(-0.8, 0.8, len(timestamps))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "obi": obi,
            "mid_price": mid,
            "vwmp": mid + obi * 0.005,
        }
    )


def test_no_duplicate_trade_ids(processed_ticks: pd.DataFrame) -> None:
    assert processed_ticks["trade_id"].nunique() == len(processed_ticks)


def test_timestamps_monotonic(processed_ticks: pd.DataFrame) -> None:
    assert processed_ticks["timestamp"].is_monotonic_increasing


def test_obi_range() -> None:
    collector = OrderBookCollector()
    snapshot = {
        "bids": [(100.0, 2.0), (99.5, 1.0)],
        "asks": [(100.5, 1.5), (101.0, 1.0)],
    }
    metrics = collector.calculate_metrics(snapshot)
    assert -1.0 <= metrics["obi"] <= 1.0
    assert metrics["bid_ask_spread"] == pytest.approx(0.5)


def test_log_returns_finite(processed_ticks: pd.DataFrame) -> None:
    assert np.isfinite(processed_ticks["log_return"]).all()
    assert not processed_ticks["log_return"].isna().any()
    assert processed_ticks.loc[processed_ticks["is_gap_start"], "log_return"].eq(0).all()


def test_feature_matrix_shape(
    processed_ticks: pd.DataFrame,
    lob_frame: pd.DataFrame,
) -> None:
    engineer = FeatureEngineer(autocorrelation_lags=20)
    features, metadata = engineer.engineer(processed_ticks, lob_frame)
    expected = set(FeatureEngineer.BASE_OUTPUT_COLUMNS)
    assert expected.issubset(features.columns)
    assert len(features) == len(processed_ticks)
    assert features["tick_direction"].isin([-1, 1]).all()
    assert np.isfinite(features.select_dtypes(include=[np.number])).all().all()
    assert metadata["rows"] == len(features)
    assert set(metadata["autocorrelation"]) == {str(lag) for lag in range(1, 21)}
    assert metadata["lob_attached"] is True
    assert metadata["lob_match_fraction"] == pytest.approx(1.0)


def test_trade_intensity_is_causal_and_prefix_invariant(
    processed_ticks: pd.DataFrame,
) -> None:
    engineer = FeatureEngineer(
        trade_imbalance_window=10,
        trade_imbalance_min_periods=1,
    )
    full, metadata = engineer.engineer(
        processed_ticks,
        obi_source="trade_imbalance",
    )
    prefix, _ = engineer.engineer(
        processed_ticks.iloc[:300].copy(),
        obi_source="trade_imbalance",
    )
    seconds = full["timestamp"] // 1_000_000_000
    expected = full.groupby(seconds, sort=False).cumcount().add(1)

    assert metadata["trade_intensity_causal"] is True
    assert full["trade_intensity"].to_numpy() == pytest.approx(expected)
    assert prefix["trade_intensity"].to_numpy() == pytest.approx(
        full["trade_intensity"].iloc[: len(prefix)].to_numpy()
    )


def test_gap_creates_new_segment(processed_ticks: pd.DataFrame) -> None:
    assert processed_ticks["segment_id"].nunique() == 2
    assert processed_ticks["is_gap_start"].sum() == 1


def test_trailing_outlier_filter_preserves_trend_and_removes_spike() -> None:
    count = 1_000
    price = np.linspace(100.0, 101.0, count)
    price[500] += 10.0
    frame = pd.DataFrame(
        {
            "timestamp": 1_750_000_000_000_000_000
            + np.arange(count) * 1_000_000,
            "price": price,
            "quantity": np.ones(count),
            "side": np.where(np.arange(count) % 2 == 0, "buy", "sell"),
            "trade_id": np.arange(count),
        }
    )

    processed, report = TickProcessor(rolling_window=100).process(frame)

    assert report["price_outliers_removed"] == 1
    assert (
        report["rolling_window_alignment"]
        == "trailing_with_past_only_regime_confirmation"
    )
    assert len(processed) == count - 1


def test_outlier_filter_preserves_confirmed_price_regime_shift() -> None:
    count = 300
    price = np.full(count, 100.0)
    price[150:] = 110.0
    frame = pd.DataFrame(
        {
            "timestamp": 1_750_000_000_000_000_000
            + np.arange(count) * 1_000_000,
            "price": price,
            "quantity": np.ones(count),
            "side": np.where(np.arange(count) % 2 == 0, "buy", "sell"),
            "trade_id": np.arange(count),
        }
    )

    processed, report = TickProcessor(rolling_window=100).process(frame)

    assert report["price_outlier_candidates"] > 0
    assert report["price_outliers_removed"] == 0
    assert len(processed) == count


def test_outlier_filter_is_prefix_invariant_at_decision_boundary() -> None:
    count = 400
    frame = pd.DataFrame(
        {
            "timestamp": 1_750_000_000_000_000_000
            + np.arange(count) * 1_000_000,
            "price": 100.0 + np.sin(np.arange(count) / 20.0),
            "quantity": np.ones(count),
            "side": np.where(np.arange(count) % 2 == 0, "buy", "sell"),
            "trade_id": np.arange(count),
        }
    )
    frame.loc[250, "price"] += 10.0
    processor = TickProcessor(rolling_window=50)
    spike_index = 250

    prefix, _ = processor.process(frame.iloc[: spike_index + 1].copy())
    full, _ = processor.process(frame)
    comparable = full.loc[full["trade_id"] <= spike_index].reset_index(drop=True)

    pd.testing.assert_frame_equal(prefix, comparable)


def test_outlier_decision_does_not_depend_on_next_tick_value() -> None:
    count = 220
    spike_index = 120
    frame = pd.DataFrame(
        {
            "timestamp": 1_750_000_000_000_000_000
            + np.arange(count) * 1_000_000,
            "price": 100.0 + np.sin(np.arange(count) / 20.0),
            "quantity": np.ones(count),
            "side": np.where(np.arange(count) % 2 == 0, "buy", "sell"),
            "trade_id": np.arange(count),
        }
    )
    frame.loc[spike_index, "price"] += 10.0
    changed_future = frame.copy()
    changed_future.loc[spike_index + 1, "price"] = frame.loc[spike_index, "price"]
    processor = TickProcessor(rolling_window=50)

    normal_future, _ = processor.process(frame)
    elevated_future, _ = processor.process(changed_future)
    normal_prefix = normal_future.loc[
        normal_future["trade_id"] <= spike_index
    ].reset_index(drop=True)
    elevated_prefix = elevated_future.loc[
        elevated_future["trade_id"] <= spike_index
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(normal_prefix, elevated_prefix)


def test_tick_direction_does_not_cross_gap_segments() -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": [
                1_750_000_000_000_000_000,
                1_750_000_001_000_000_000,
                1_750_000_002_000_000_000,
                1_750_000_400_000_000_000,
                1_750_000_401_000_000_000,
                1_750_000_402_000_000_000,
            ],
            "price": [100.0, 101.0, 101.0, 101.0, 100.0, 100.0],
            "quantity": np.ones(6),
            "side": ["buy", "buy", "sell", "sell", "sell", "buy"],
            "trade_id": np.arange(6),
            "log_return": np.zeros(6),
            "price_increment": [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            "segment_id": [0, 0, 0, 1, 1, 1],
        }
    )

    features, _ = FeatureEngineer().engineer(ticks)

    assert features["tick_direction"].tolist() == [1, 1, 1, -1, -1, -1]


def test_production_features_require_lob(processed_ticks: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="LOB data is required"):
        FeatureEngineer().engineer(processed_ticks, require_lob=True)


def test_synthetic_trade_imbalance_is_causal_and_resets_at_gap() -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": np.arange(5, dtype=np.int64) + 1_750_000_000_000_000_000,
            "price": [100.0, 100.0, 100.0, 101.0, 101.0],
            "quantity": [2.0, 1.0, 3.0, 4.0, 1.0],
            "side": ["buy", "sell", "buy", "sell", "buy"],
            "trade_id": np.arange(5),
            "log_return": np.zeros(5),
            "price_increment": [0.0, 0.0, 0.0, 0.0, 0.0],
            "segment_id": [0, 0, 0, 1, 1],
        }
    )
    engineer = FeatureEngineer(
        trade_imbalance_window=2,
        trade_imbalance_min_periods=1,
    )

    features, metadata = engineer.engineer(
        ticks,
        obi_source="trade_imbalance",
    )

    assert features["obi"].to_numpy() == pytest.approx(
        [0.0, 1.0, 1.0 / 3.0, 0.0, -1.0]
    )
    assert features["obi_valid"].tolist() == [False, True, True, False, True]
    assert metadata["obi_source"] == "trade_volume_imbalance"
    assert metadata["obi_is_proxy"] is True
    assert metadata["obi_causal"] is True
    assert metadata["obi_window_trades"] == 2
    assert metadata["obi_min_periods"] == 1
    assert metadata["obi_lag_trades"] == 1


def test_synthetic_obi_excludes_current_trade() -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": np.arange(4, dtype=np.int64) + 1_750_000_000_000_000_000,
            "price": np.full(4, 100.0),
            "quantity": np.ones(4),
            "side": ["buy", "sell", "buy", "sell"],
            "trade_id": np.arange(4),
            "log_return": np.zeros(4),
            "price_increment": np.zeros(4),
            "segment_id": np.zeros(4, dtype=np.int32),
        }
    )
    engineer = FeatureEngineer(
        trade_imbalance_window=3,
        trade_imbalance_min_periods=1,
    )
    original, _ = engineer.engineer(ticks, obi_source="trade_imbalance")
    changed_ticks = ticks.copy()
    changed_ticks.loc[2, "side"] = "sell"
    changed, _ = engineer.engineer(
        changed_ticks,
        obi_source="trade_imbalance",
    )

    assert changed.loc[2, "obi"] == pytest.approx(original.loc[2, "obi"])
    assert changed.loc[3, "obi"] != pytest.approx(original.loc[3, "obi"])


def test_auto_obi_source_prefers_real_lob(
    processed_ticks: pd.DataFrame,
    lob_frame: pd.DataFrame,
) -> None:
    features, metadata = FeatureEngineer().engineer(
        processed_ticks,
        lob_frame,
        obi_source="auto",
    )

    assert metadata["obi_source"] == "lob_order_book"
    assert metadata["obi_is_proxy"] is False
    assert features["obi_valid"].all()


def test_production_features_reject_stale_lob(
    processed_ticks: pd.DataFrame,
    lob_frame: pd.DataFrame,
) -> None:
    stale_lob = lob_frame.copy()
    stale_lob["timestamp"] += 24 * 60 * 60 * 1_000_000_000

    with pytest.raises(ValueError, match="no timestamp overlap"):
        FeatureEngineer().engineer(
            processed_ticks,
            stale_lob,
            require_lob=True,
        )


def test_production_features_enforce_lob_coverage(
    processed_ticks: pd.DataFrame,
    lob_frame: pd.DataFrame,
) -> None:
    sparse_lob = lob_frame.iloc[[0]].copy()

    with pytest.raises(ValueError, match="below the required"):
        FeatureEngineer(
            lob_tolerance_seconds=0.05,
            minimum_lob_match_fraction=0.95,
        ).engineer(
            processed_ticks,
            sparse_lob,
            require_lob=True,
        )


def test_hdf5_snapshot_roundtrip(tmp_path) -> None:
    collector = OrderBookCollector()
    snapshot = {
        "timestamp": 1_750_000_000_000_000_000,
        "last_update_id": 123,
        "bids": [(100.0, 2.0), (99.5, 1.0)],
        "asks": [(100.5, 1.5), (101.0, 1.0)],
    }
    path = tmp_path / "lob.h5"
    collector.save_snapshot(path, snapshot)
    frame = collector.snapshots_to_frame(path)
    assert len(frame) == 1
    assert frame.loc[0, "last_update_id"] == 123
    assert -1.0 <= frame.loc[0, "obi"] <= 1.0


def test_binance_archive_parser_normalizes_microseconds() -> None:
    csv_content = (
        "10,100.5,0.2,20.1,1735689600010866,False,True\n"
        "11,100.4,0.3,30.12,1735689600011866,True,True\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-trades-2025-01-01.csv", csv_content)

    frame = TickDownloader.parse_archive(buffer.getvalue(), dataset="trades")
    assert frame["timestamp"].iloc[0] == 1_735_689_600_010_866_000
    assert frame["side"].tolist() == ["buy", "sell"]


def test_binance_archive_parser_handles_string_boolean_values(monkeypatch) -> None:
    original_read_csv = pd.read_csv

    def read_csv_as_strings(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        frame["is_buyer_maker"] = frame["is_buyer_maker"].astype("string")
        return frame

    csv_content = (
        "10,100.5,0.2,20.1,1735689600010866,False,True\n"
        "11,100.4,0.3,30.12,1735689600011866,True,True\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-trades-2025-01-01.csv", csv_content)
    monkeypatch.setattr(pd, "read_csv", read_csv_as_strings)

    frame = TickDownloader.parse_archive(buffer.getvalue(), dataset="trades")
    assert frame["side"].tolist() == ["buy", "sell"]


def test_live_trade_and_depth_parsers() -> None:
    payload = {
        "t": 123,
        "p": "100.5",
        "q": "0.2",
        "T": 1_735_689_600_010_866,
        "m": False,
    }
    trade = LiveMarketCollector._trade_row(payload)
    assert trade == {
        "timestamp": 1_735_689_600_010_866_000,
        "price": 100.5,
        "quantity": 0.2,
        "side": "buy",
        "trade_id": 123,
    }
    assert LiveMarketCollector._trade_row(
        payload,
        timestamp_ns=1_735_689_600_020_000_000,
    )["timestamp"] == 1_735_689_600_020_000_000

    snapshot = LiveMarketCollector._depth_snapshot(
        {
            "lastUpdateId": 456,
            "bids": [["100.0", "2.0"], ["99.5", "1.0"]],
            "asks": [["100.5", "1.5"], ["101.0", "1.0"]],
        },
        1_735_689_600_020_000_000,
    )
    metrics = OrderBookCollector().calculate_metrics(snapshot)
    assert snapshot["last_update_id"] == 456
    assert -1.0 <= metrics["obi"] <= 1.0


def test_persist_processed_and_features(
    tmp_path,
    raw_ticks: pd.DataFrame,
    lob_frame: pd.DataFrame,
) -> None:
    raw_path = tmp_path / "raw.csv.gz"
    processed_path = tmp_path / "processed.parquet"
    quality_path = tmp_path / "quality.txt"
    feature_path = tmp_path / "features.parquet"
    stats_path = tmp_path / "stats.csv"
    metadata_path = tmp_path / "metadata.json"
    raw_ticks.to_csv(raw_path, index=False, compression="gzip")

    processor = TickProcessor(rolling_window=50, outlier_z_score=8.0)
    processor.process_file(raw_path, processed_path, quality_path)
    ticks = pd.read_parquet(processed_path)

    engineer = FeatureEngineer()
    features, metadata = engineer.engineer(ticks, lob_frame)
    engineer.save(features, metadata, feature_path, stats_path, metadata_path)

    assert processed_path.exists()
    assert quality_path.exists()
    assert feature_path.exists()
    assert stats_path.exists()
    assert metadata_path.exists()
