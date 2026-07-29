"""Tests for causal multi-scale features gathered at ML anchors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.temporal_features import (
    TemporalFeatureSpec,
    build_temporal_feature_matrix,
)


def _spec() -> TemporalFeatureSpec:
    return TemporalFeatureSpec(
        tick_direction_lags=(1, 2),
        signed_volume_windows=(2, 4),
        return_windows=(2, 4),
        realised_volatility_windows=(2, 4),
    )


def _frame(rows: int = 80) -> pd.DataFrame:
    timestamp = 1_783_036_800_000_000_000 + np.arange(rows) * 1_000_000
    price = 100.0 * np.exp(np.arange(rows) * 1e-4)
    direction = np.where(np.arange(rows) % 2 == 0, 1, -1)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "price": price,
            "quantity": 1.0 + np.arange(rows) / 100.0,
            "trade_sign": direction.astype(np.int8),
            "tick_direction": direction.astype(np.int8),
            "trade_intensity": 1 + np.arange(rows) % 7,
            "obi": np.linspace(-0.8, 0.8, rows),
            "obi_valid": np.ones(rows, dtype=bool),
            "segment_id": np.zeros(rows, dtype=np.int32),
            "mid_price": price - 0.01,
            "price_mid_deviation": np.full(rows, 0.01),
        }
    )


def test_feature_names_and_matrix_width_are_frozen() -> None:
    spec = _spec()
    matrix, valid = build_temporal_feature_matrix(
        _frame(), np.array([10, 20]), spec
    )

    assert matrix.shape == (2, len(spec.feature_names))
    assert valid.all()
    assert spec.feature_names == (
        "obi",
        "tick_direction",
        "obi_change",
        "abs_obi",
        "log_trade_intensity",
        "price_mid_deviation",
        "tick_direction_lag_1",
        "tick_direction_lag_2",
        "signed_volume_sum_2",
        "signed_volume_sum_4",
        "log_return_2",
        "log_return_4",
        "realised_volatility_2",
        "realised_volatility_4",
        "time_since_previous_event_seconds",
    )


def test_future_mutation_cannot_change_anchor_features() -> None:
    """This is the central Phase 1 leakage regression test."""
    frame = _frame()
    anchors = np.array([10, 20], dtype=np.int64)
    baseline, baseline_valid = build_temporal_feature_matrix(
        frame, anchors, _spec()
    )

    mutated = frame.copy()
    mutated.loc[21:, "price"] *= 10.0
    mutated.loc[21:, "quantity"] *= 100.0
    mutated.loc[21:, "obi"] *= -1.0
    mutated.loc[21:, "tick_direction"] *= -1
    changed, changed_valid = build_temporal_feature_matrix(
        mutated, anchors, _spec()
    )

    np.testing.assert_array_equal(changed, baseline)
    np.testing.assert_array_equal(changed_valid, baseline_valid)


def test_rolling_features_do_not_cross_a_segment_boundary() -> None:
    frame = _frame()
    frame.loc[18:, "segment_id"] = 1

    _, valid = build_temporal_feature_matrix(
        frame, np.array([20, 25]), _spec()
    )

    assert valid.tolist() == [False, True]


def test_signed_volume_is_trailing_and_includes_the_known_anchor_trade() -> None:
    frame = _frame()
    anchor = np.array([10])
    matrix, valid = build_temporal_feature_matrix(frame, anchor, _spec())
    index = _spec().feature_names.index("signed_volume_sum_4")
    expected = (
        frame["quantity"].to_numpy()[7:11]
        * frame["trade_sign"].to_numpy()[7:11]
    ).sum()

    assert valid[0]
    assert matrix[0, index] == expected
