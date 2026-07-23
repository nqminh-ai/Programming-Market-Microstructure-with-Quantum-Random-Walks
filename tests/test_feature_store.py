"""Tests for the shared feature-store loader.

Three research scripts each had their own copy of this, and each copy read every
column at its stored width, sorted unconditionally, and downcast afterwards --
which on the 227M-row store is 7.7GB before the sort doubles it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.feature_store import load_feature_columns


def _store(path, rows: int = 500, shuffled: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(rows) * 50_000_000 + 1_783_099_484_231_822_000,
            "price": 100.0 + np.arange(rows) * 0.01,
            "obi": np.linspace(-1.0, 1.0, rows),
            "segment_id": np.zeros(rows, dtype=np.int32),
            "obi_valid": np.ones(rows, dtype=bool),
            "unused": np.arange(rows, dtype=np.int64),
        }
    )
    if shuffled:
        frame = frame.iloc[::-1].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return frame


def test_only_the_requested_columns_are_returned(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path)

    frame = load_feature_columns(path, ["timestamp", "price", "obi"])

    assert list(frame.columns) == ["timestamp", "price", "obi"]
    assert len(frame) == 500


def test_a_column_the_store_predates_is_skipped_not_raised(tmp_path) -> None:
    """A store written before a feature existed is a reason to fall back."""
    path = tmp_path / "features.parquet"
    _store(path)

    frame = load_feature_columns(path, ["timestamp", "price", "mid_price"])

    assert list(frame.columns) == ["timestamp", "price"]


def test_requesting_nothing_the_store_has_is_an_error(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path)

    with pytest.raises(ValueError, match="none of the requested columns"):
        load_feature_columns(path, ["nope", "also_nope"])


def test_downcast_narrows_the_named_columns_only(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path)

    frame = load_feature_columns(
        path, ["timestamp", "price", "obi"], downcast={"obi": "float32"}
    )

    assert frame["obi"].dtype == np.float32
    # price must stay float64: float32 on a ~60,000 price loses tick-scale moves.
    assert frame["price"].dtype == np.float64


def test_downcasting_does_not_change_the_values_it_can_represent(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    expected = _store(path)

    frame = load_feature_columns(
        path, ["timestamp", "obi"], downcast={"obi": "float32"}
    )

    np.testing.assert_allclose(frame["obi"], expected["obi"], rtol=1e-6)
    np.testing.assert_array_equal(frame["timestamp"], expected["timestamp"])


def test_max_rows_takes_a_prefix(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    expected = _store(path)

    frame = load_feature_columns(path, ["timestamp", "price"], max_rows=100)

    assert len(frame) == 100
    np.testing.assert_array_equal(frame["price"], expected["price"].to_numpy()[:100])


def test_a_capped_read_does_not_keep_the_whole_column_alive(tmp_path) -> None:
    """Slicing after a full read bounds the length and nothing else.

    A numpy slice keeps its base array alive, so capping a 227M-row store at
    100M rows still cost all 227M -- the frame reported the capped length while
    holding the full column behind it.
    """
    path = tmp_path / "features.parquet"
    frame_in = pd.DataFrame(
        {
            "timestamp": np.arange(10_000, dtype=np.int64),
            "price": np.arange(10_000, dtype=np.float64),
        }
    )
    pq.write_table(
        pa.Table.from_pandas(frame_in, preserve_index=False), path, row_group_size=500
    )

    frame = load_feature_columns(path, ["timestamp", "price"], max_rows=1_000)

    assert len(frame) == 1_000
    for name in ("timestamp", "price"):
        values = frame[name].to_numpy()
        base = values.base if values.base is not None else values
        # At most one row group of overshoot, not the other 9,000 rows.
        assert base.size <= 1_000 + 500, f"{name} still holds {base.size} rows"


def test_a_cap_larger_than_the_store_returns_everything(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path, rows=500)

    frame = load_feature_columns(path, ["timestamp", "price"], max_rows=10_000)

    assert len(frame) == 500


def test_an_unordered_store_is_sorted(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path, shuffled=True)

    frame = load_feature_columns(path, ["timestamp", "price"])

    assert frame["timestamp"].is_monotonic_increasing
    # Sorting must carry the other columns with it, not just reorder timestamps.
    assert frame["price"].iloc[0] == pytest.approx(100.0)
    assert frame["price"].iloc[-1] == pytest.approx(100.0 + 499 * 0.01)


def test_an_ordered_store_is_returned_untouched(tmp_path) -> None:
    """Sorting copies the whole frame; the stores are already in date order."""
    path = tmp_path / "features.parquet"
    expected = _store(path)

    frame = load_feature_columns(path, ["timestamp", "price"])

    np.testing.assert_array_equal(frame["timestamp"], expected["timestamp"])
    assert frame.index[0] == 0


def test_sorting_can_be_switched_off(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    _store(path, shuffled=True)

    frame = load_feature_columns(path, ["timestamp", "price"], sort_by=None)

    assert not frame["timestamp"].is_monotonic_increasing
