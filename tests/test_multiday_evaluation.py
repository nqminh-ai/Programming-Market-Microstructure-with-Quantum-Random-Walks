"""Tests for UTC-day clustered bootstrap and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.multiday import (
    day_cluster_bootstrap,
    day_cluster_sensitivity,
    paired_day_comparisons,
)


def daily_metrics(days: int = 20) -> pd.DataFrame:
    rows = []
    for day in range(days):
        for model, offset in (("QRW", 0.02), ("Logistic", 0.0)):
            rows.append(
                {
                    "day": f"2026-07-{day + 1:02d}",
                    "model": model,
                    "mean_marginal_crps": 0.10 + offset + 0.001 * day,
                    "mean_direction_log_loss": 0.60 + offset + 0.002 * day,
                }
            )
    return pd.DataFrame(rows)


def test_day_cluster_bootstrap_uses_days_not_ticks() -> None:
    result = day_cluster_bootstrap(
        daily_metrics(),
        value_columns=("mean_marginal_crps", "mean_direction_log_loss"),
        iterations=200,
        random_seed=7,
    )

    assert set(result["cluster_unit"]) == {"UTC_day"}
    assert set(result["evaluated_days"]) == {20}
    qrw = result.loc[result["model"] == "QRW"].iloc[0]
    assert qrw["mean_marginal_crps"] == pytest.approx(0.1295)
    assert (
        qrw["mean_marginal_crps_ci_low"]
        < qrw["mean_marginal_crps"]
        < qrw["mean_marginal_crps_ci_high"]
    )


def test_sensitivity_covers_registered_seeds_and_block_sizes() -> None:
    result = day_cluster_sensitivity(
        daily_metrics(),
        value_columns=("mean_marginal_crps",),
        seeds=(1, 2),
        block_sizes=(1, 2, 5),
        iterations=100,
    )

    assert set(result["random_seed"]) == {1, 2}
    assert set(result["block_size_days"]) == {1, 2, 5}
    assert len(result) == 2 * 3 * 2


def test_duplicate_model_day_is_rejected() -> None:
    frame = daily_metrics(2)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="at most one row per day"):
        day_cluster_bootstrap(
            duplicated,
            value_columns=("mean_marginal_crps",),
            iterations=100,
        )


def test_bootstrap_is_reproducible_for_fixed_seed() -> None:
    first = day_cluster_bootstrap(
        daily_metrics(),
        value_columns=("mean_marginal_crps",),
        iterations=100,
        random_seed=42,
        block_size=2,
    )
    second = day_cluster_bootstrap(
        daily_metrics(),
        value_columns=("mean_marginal_crps",),
        iterations=100,
        random_seed=42,
        block_size=2,
    )

    pd.testing.assert_frame_equal(first, second)


def test_pairwise_correction_spans_all_models_and_metrics() -> None:
    frame = daily_metrics()
    frame = pd.concat(
        [
            frame,
            frame.loc[frame["model"] == "Logistic"]
            .assign(model="Hawkes", mean_marginal_crps=lambda value: value["mean_marginal_crps"] + 0.01),
        ],
        ignore_index=True,
    )
    result = paired_day_comparisons(
        frame,
        metric_directions={
            "mean_marginal_crps": "lower",
            "mean_direction_log_loss": "lower",
        },
        iterations=200,
        random_seed=9,
        block_size=2,
    )

    assert len(result) == 3 * 2
    assert (result["pvalue_bh"] >= result["pvalue"]).all()
    assert result["pvalue"].between(1 / 201, 1.0).all()
    assert set(result["pvalue_method"]) == {
        "centered_moving_block_bootstrap"
    }
    assert set(result["correction_scope"]) == {
        "all_selected_model_metric_comparisons"
    }
