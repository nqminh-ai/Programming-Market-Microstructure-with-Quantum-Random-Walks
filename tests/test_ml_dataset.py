"""Tests for chronological, non-overlapping Phase 1 ML datasets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ml_dataset import (
    boundary_purge_mask,
    build_horizon_dataset,
    build_horizon_datasets,
    build_ml_dataset_files,
    discover_parquet_utc_days,
    fold_for_day_index,
    iter_parquet_utc_days,
    validate_utc_day_sequence,
)
from src.data.temporal_features import TemporalFeatureSpec
from src.evaluation.ml_protocol import load_ml_protocol


DAY_NS = 86_400_000_000_000


def _spec() -> TemporalFeatureSpec:
    return TemporalFeatureSpec(
        tick_direction_lags=(1, 2),
        signed_volume_windows=(2, 4),
        return_windows=(2, 4),
        realised_volatility_windows=(2, 4),
    )


def _frame(days: int = 1, rows_per_day: int = 100) -> pd.DataFrame:
    blocks = []
    base = 1_783_036_800_000_000_000
    global_row = 0
    for day in range(days):
        row = np.arange(rows_per_day)
        price = 100.0 * np.exp((global_row + row) * 1e-5)
        direction = np.where(row % 2 == 0, 1, -1)
        blocks.append(
            pd.DataFrame(
                {
                    "timestamp": base
                    + day * DAY_NS
                    + row * 1_000_000,
                    "price": price,
                    "quantity": 1.0 + row / 1000.0,
                    "side": np.where(direction > 0, "buy", "sell"),
                    "trade_sign": direction.astype(np.int8),
                    "tick_direction": direction.astype(np.int8),
                    "trade_intensity": 1 + row % 5,
                    "obi": np.clip(direction * 0.4 + row / 1000.0, -1, 1),
                    "obi_valid": np.ones(rows_per_day, dtype=bool),
                    "segment_id": np.full(
                        rows_per_day, day, dtype=np.int32
                    ),
                    "mid_price": price - 0.01,
                    "price_mid_deviation": np.full(rows_per_day, 0.01),
                }
            )
        )
        global_row += rows_per_day
    return pd.concat(blocks, ignore_index=True)


def _write_store(path, frame: pd.DataFrame, row_group_size: int = 17) -> None:
    stored = frame.drop(columns=["trade_sign"])
    pq.write_table(
        pa.Table.from_pandas(stored, preserve_index=False),
        path,
        row_group_size=row_group_size,
    )


def test_horizon_events_are_deterministic_and_never_overlap() -> None:
    frame = _frame()

    first = build_horizon_dataset(frame, 10, _spec())
    second = build_horizon_dataset(frame, 10, _spec())

    pd.testing.assert_frame_equal(first, second)
    assert (np.diff(first["anchor_row"]) >= 10).all()
    assert (first["target_timestamp"] > first["timestamp"]).all()
    assert first["target_up"].eq(1).all()
    assert first["utc_day"].nunique() == 1


def test_multi_horizon_builder_matches_independent_builds() -> None:
    frame = _frame()
    combined = build_horizon_datasets(frame, (10, 20), _spec())

    for horizon in (10, 20):
        independent = build_horizon_dataset(frame, horizon, _spec())
        pd.testing.assert_frame_equal(combined[horizon], independent)


def test_horizon_events_cannot_cross_a_utc_day() -> None:
    frame = _frame(days=2)

    try:
        build_horizon_dataset(frame, 10, _spec())
    except ValueError as error:
        assert "one UTC day" in str(error)
    else:
        raise AssertionError("multi-day frame was accepted")


def test_horizon_labels_never_cross_a_segment_boundary() -> None:
    frame = _frame()
    frame.loc[50:, "segment_id"] = 1

    events = build_horizon_dataset(frame, 10, _spec())

    crossing_timestamp = frame.loc[44, "timestamp"]
    assert crossing_timestamp not in set(events["timestamp"])


def test_zero_return_windows_are_excluded_not_called_down() -> None:
    frame = _frame()
    frame["price"] = 100.0
    frame["mid_price"] = 99.99

    events = build_horizon_dataset(frame, 10, _spec())

    assert events.empty


def test_fold_assignment_uses_the_frozen_day_counts() -> None:
    protocol = load_ml_protocol()

    assert fold_for_day_index(0, protocol) == "train"
    assert fold_for_day_index(44, protocol) == "train"
    assert fold_for_day_index(45, protocol) == "selection"
    assert fold_for_day_index(54, protocol) == "selection"
    assert fold_for_day_index(55, protocol) == "calibration"
    assert fold_for_day_index(60, protocol) == "test"
    assert fold_for_day_index(68, protocol) == "test"


def test_boundary_purge_embargoes_both_sides_of_a_fold_change() -> None:
    protocol = load_ml_protocol()
    anchors = np.array([10_000, 210_000, 1_710_000])
    future = anchors + 100_000

    first_selection = boundary_purge_mask(
        anchors,
        future=future,
        day_rows=2_000_000,
        day_index=45,
        protocol=protocol,
        maximum_lookback=10_000,
    )
    last_train = boundary_purge_mask(
        anchors,
        future=future,
        day_rows=2_000_000,
        day_index=44,
        protocol=protocol,
        maximum_lookback=10_000,
    )

    assert first_selection.tolist() == [False, True, True]
    assert last_train.tolist() == [True, True, False]


def test_parquet_streamer_yields_complete_days_across_batch_boundaries(
    tmp_path,
) -> None:
    path = tmp_path / "features.parquet"
    _write_store(path, _frame(days=3, rows_per_day=12), row_group_size=5)

    discovered = discover_parquet_utc_days(path, batch_size=7)
    streamed = list(iter_parquet_utc_days(path, batch_size=7))

    assert discovered == tuple(day for day, _ in streamed)
    assert [len(frame) for _, frame in streamed] == [12, 12, 12]
    assert all("side" not in frame for _, frame in streamed)
    assert all("trade_sign" in frame for _, frame in streamed)
    np.testing.assert_array_equal(
        streamed[0][1]["trade_sign"].to_numpy(),
        np.where(np.arange(12) % 2 == 0, 1, -1),
    )


def test_registered_day_sequence_must_be_consecutive() -> None:
    validate_utc_day_sequence(
        ("2026-07-01", "2026-07-02", "2026-07-03"),
        expected_days=3,
    )

    try:
        validate_utc_day_sequence(
            ("2026-07-01", "2026-07-03", "2026-07-04"),
            expected_days=3,
        )
    except ValueError as error:
        assert "consecutive" in str(error)
    else:
        raise AssertionError("a missing UTC day was accepted")


def test_streamed_builder_writes_folded_datasets_and_metadata(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _write_store(path, _frame(days=4, rows_per_day=100))
    base = load_ml_protocol()
    raw = deepcopy(base.raw)
    raw["features"]["tick_direction_lags"] = [1, 2]
    raw["features"]["signed_volume_windows_ticks"] = [2, 4]
    raw["features"]["return_windows_ticks"] = [2, 4]
    raw["features"]["realised_volatility_windows_ticks"] = [2, 4]
    protocol = replace(
        base,
        prediction_skill_horizons=(10,),
        economic_horizons=(20,),
        evaluation_horizons=(10, 20),
        train_days=1,
        selection_days=1,
        calibration_days=1,
        test_days=1,
        purge_ticks=4,
        raw=raw,
    )

    build = build_ml_dataset_files(
        path,
        tmp_path / "ml",
        asset="BTCUSDT",
        protocol=protocol,
        batch_size=23,
        repo_root=tmp_path,
    )

    assert build.metadata["days"] == [
        "2026-07-03",
        "2026-07-04",
        "2026-07-05",
        "2026-07-06",
    ]
    assert build.metadata["config_sha256"]
    for horizon, output in build.datasets.items():
        frame = pd.read_parquet(output)
        assert set(frame["fold"]) == {
            "train",
            "selection",
            "calibration",
            "test",
        }
        assert frame["horizon_ticks"].eq(horizon).all()
        assert not frame[list(_spec().feature_names)].isna().any().any()
