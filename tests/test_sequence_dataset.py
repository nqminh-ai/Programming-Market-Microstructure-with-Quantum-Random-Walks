"""Tests for Phase 4 causal sequence shards and train-only statistics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ml_dataset import build_ml_dataset_files
from src.data.sequence_dataset import (
    SequenceFeatureSpec,
    build_sequence_dataset_shards,
    build_sequence_tensor,
    iter_sequence_shards,
)
from src.evaluation.ml_protocol import load_ml_protocol


DAY_NS = 86_400_000_000_000


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
                    "timestamp": base + day * DAY_NS + row * 1_000_000,
                    "price": price,
                    "quantity": 1.0 + row / 1000.0,
                    "side": np.where(direction > 0, "buy", "sell"),
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


def _write_store(path, frame: pd.DataFrame) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        row_group_size=17,
    )


def _small_protocol():
    base = load_ml_protocol()
    raw = deepcopy(base.raw)
    raw["features"]["tick_direction_lags"] = [1, 2]
    raw["features"]["signed_volume_windows_ticks"] = [2, 8]
    raw["features"]["return_windows_ticks"] = [2, 8]
    raw["features"]["realised_volatility_windows_ticks"] = [2, 8]
    raw["models"]["phase_4_sequence_dataset"]["sequence_length"] = 8
    return replace(
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


def test_sequence_tensor_is_causal_channels_first_and_deterministic() -> None:
    protocol = _small_protocol()
    spec = SequenceFeatureSpec.from_protocol(protocol.raw)
    frame = _frame()
    anchors = np.array([15, 25])

    first, first_valid = build_sequence_tensor(frame, anchors, spec)
    mutated = frame.copy()
    mutated.loc[26:, "quantity"] *= 1000.0
    mutated.loc[26:, "price"] *= 2.0
    mutated.loc[26:, "obi"] *= -1.0
    second, second_valid = build_sequence_tensor(mutated, anchors, spec)

    assert first.shape == (2, len(spec.channels), 8)
    assert first.dtype == np.float32
    assert first_valid.all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_valid, second_valid)
    obi_index = spec.channels.index("obi")
    assert first[0, obi_index, -1] == np.float32(frame.loc[15, "obi"])


def test_sequence_cannot_cross_a_segment_boundary() -> None:
    protocol = _small_protocol()
    spec = SequenceFeatureSpec.from_protocol(protocol.raw)
    frame = _frame()
    frame.loc[20:, "segment_id"] = 1

    _, valid = build_sequence_tensor(frame, np.array([22, 30]), spec)

    assert valid.tolist() == [False, True]


def test_shards_align_to_phase1_and_normalization_ignores_test(
    tmp_path,
) -> None:
    protocol = _small_protocol()
    source = tmp_path / "features.parquet"
    original = _frame(days=4)
    _write_store(source, original)
    phase1 = build_ml_dataset_files(
        source,
        tmp_path / "phase1",
        asset="BTCUSDT",
        protocol=protocol,
        batch_size=23,
        repo_root=tmp_path,
    )
    first = build_sequence_dataset_shards(
        source,
        phase1.metadata_path,
        tmp_path / "sequences_a",
        asset="BTCUSDT",
        protocol=protocol,
        batch_size=23,
        repo_root=tmp_path,
    )

    mutated = original.copy()
    test_day = mutated["timestamp"] >= (
        mutated["timestamp"].min() + 3 * DAY_NS
    )
    mutated.loc[test_day, "quantity"] *= 1000.0
    mutated.loc[test_day, "obi"] *= -1.0
    mutated_source = tmp_path / "features_mutated_test.parquet"
    _write_store(mutated_source, mutated)
    second = build_sequence_dataset_shards(
        mutated_source,
        phase1.metadata_path,
        tmp_path / "sequences_b",
        asset="BTCUSDT",
        protocol=protocol,
        batch_size=23,
        repo_root=tmp_path,
    )

    assert len(first.shard_paths) == 8
    assert first.manifest["test_labels_used_for_normalization"] is False
    assert (
        first.manifest["normalization"]
        == second.manifest["normalization"]
    )
    assert {
        entry["fold"] for entry in first.manifest["shards"]
    } == {"train", "selection", "calibration", "test"}
    test_shards = list(
        iter_sequence_shards(
            first.manifest,
            folds=("test",),
            repo_root=tmp_path,
        )
    )
    assert len(test_shards) == 2
    assert all(shard["features"].shape[1:] == (8, 8) for shard in test_shards)
