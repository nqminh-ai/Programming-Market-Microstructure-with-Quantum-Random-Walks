"""Tests for causal UTC-day folds in the cross-asset benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.cross_asset_benchmark import (
    combine_training_days,
    expanding_day_folds,
)


def day_entry(day_number: int) -> tuple[str, Path, pd.DataFrame]:
    label = f"2026-01-{day_number:02d}"
    timestamp_start = day_number * 1_000
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp_start, timestamp_start + 1],
            "price": [100.0, 101.0],
            "segment_id": [0, 0],
        }
    )
    return label, Path(f"features_{label}.parquet"), frame


def test_expanding_folds_use_every_eligible_later_test_day() -> None:
    days = [day_entry(day) for day in range(7, 0, -1)]

    folds = expanding_day_folds(days, minimum_training_days=3)

    assert [fold.test[0] for fold in folds] == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ]
    assert [fold.validation[0] for fold in folds] == [
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
    ]
    assert [len(fold.training) for fold in folds] == [3, 4, 5]


def test_fold_days_paths_and_timestamps_are_strictly_separated() -> None:
    folds = expanding_day_folds(
        [day_entry(day) for day in range(1, 7)], minimum_training_days=3
    )

    for fold in folds:
        training_labels = {entry[0] for entry in fold.training}
        training_paths = {entry[1] for entry in fold.training}
        training_frame = combine_training_days([entry[2] for entry in fold.training])

        assert fold.validation[0] not in training_labels
        assert fold.test[0] not in training_labels
        assert fold.validation[1] not in training_paths
        assert fold.test[1] not in training_paths
        assert training_frame["timestamp"].max() < fold.validation[2]["timestamp"].min()
        assert fold.validation[2]["timestamp"].max() < fold.test[2]["timestamp"].min()
        assert training_frame["segment_id"].nunique() == len(fold.training)


def test_duplicate_day_labels_are_rejected() -> None:
    days = [day_entry(day) for day in range(1, 5)]
    days.append(day_entry(4))

    with pytest.raises(ValueError, match="unique"):
        expanding_day_folds(days, minimum_training_days=2)
