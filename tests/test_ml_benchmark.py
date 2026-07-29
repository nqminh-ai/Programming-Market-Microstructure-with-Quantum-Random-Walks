"""Tests for Phase 3 common-sample scoring and UTC-day uncertainty."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.temporal_features import TemporalFeatureSpec
from src.evaluation.ml_benchmark import (
    HGB_NAME,
    WINDOWED_QRW_NAME,
    paired_utc_day_bootstrap,
    run_ml_common_sample_benchmark,
)
from src.evaluation.ml_protocol import load_ml_protocol
from src.models.gradient_boosted_direction import (
    train_hist_gradient_boosting,
)


def _protocol():
    base = load_ml_protocol()
    raw = deepcopy(base.raw)
    settings = raw["models"]["phase_2_hist_gradient_boosting"]
    settings["grid"] = {
        "learning_rate": [0.07],
        "max_iter": [40],
        "max_leaf_nodes": [15],
        "min_samples_leaf": [10],
        "l2_regularization": [0.0],
    }
    return replace(base, raw=raw)


def _benchmark_frame(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    protocol = _protocol()
    feature_names = TemporalFeatureSpec.from_protocol(
        protocol.raw
    ).feature_names
    fold_sizes = {
        "train": 360,
        "selection": 140,
        "calibration": 140,
        "test": 180,
    }
    blocks = []
    offset = 0
    day_offset = 0
    for fold, rows in fold_sizes.items():
        matrix = rng.normal(size=(rows, len(feature_names)))
        block = pd.DataFrame(matrix, columns=feature_names)
        block["obi"] = np.tanh(block["obi"])
        direction = np.where(block["tick_direction"] >= 0.0, 1.0, -1.0)
        block["tick_direction"] = direction
        block["abs_obi"] = block["obi"].abs()
        block["log_trade_intensity"] = np.abs(
            block["log_trade_intensity"]
        )
        for lag in range(1, 6):
            column = f"tick_direction_lag_{lag}"
            block[column] = np.where(block[column] >= 0.0, 1.0, -1.0)
        score = (
            1.2 * block["obi"].to_numpy()
            + 0.5 * direction
            + 0.45
            * block["tick_direction_lag_1"].to_numpy()
            * block["tick_direction_lag_2"].to_numpy()
            + 0.25 * block["log_return_100"].to_numpy()
            + rng.normal(0.0, 0.7, rows)
        )
        block["target_up"] = (score > np.median(score)).astype(np.int8)
        block["fold"] = fold
        block["anchor_row"] = np.arange(offset, offset + rows)
        block["timestamp"] = (
            1_783_036_800_000_000_000
            + np.arange(offset, offset + rows) * 1_000_000
        )
        local_days = np.minimum(np.arange(rows) * 3 // rows, 2)
        block["utc_day"] = [
            str(np.datetime64("2026-07-01") + int(day_offset + value))
            for value in local_days
        ]
        blocks.append(block)
        offset += rows
        day_offset += 3
    return pd.concat(blocks, ignore_index=True)


def _model_and_frame():
    protocol = _protocol()
    frame = _benchmark_frame()
    features = TemporalFeatureSpec.from_protocol(protocol.raw).feature_names
    training = frame.loc[frame["fold"].ne("test")].copy()
    model, _ = train_hist_gradient_boosting(
        training, features, protocol
    )
    return protocol, frame, model


def test_common_sample_benchmark_scores_every_prediction_on_test() -> None:
    protocol, frame, model = _model_and_frame()

    result = run_ml_common_sample_benchmark(
        frame, model, protocol, bootstrap_resamples=50
    )

    assert result.diagnostics["test_fold_opened"] is True
    assert result.diagnostics["same_test_sample_for_all_models"] is True
    assert result.summary["sample_size"].nunique() == 1
    assert result.summary["sample_size"].iloc[0] == 180
    assert HGB_NAME in set(result.summary["model"])
    probability_columns = [
        column
        for column in result.predictions
        if column
        not in {"timestamp", "utc_day", "anchor_row", "target_up"}
    ]
    assert (
        (result.predictions[probability_columns] > 0.0)
        & (result.predictions[probability_columns] < 1.0)
    ).all().all()
    assert set(result.comparisons) == set(probability_columns) - {HGB_NAME}


def test_windowed_qrw_is_explicitly_excluded_instead_of_mislabelled() -> None:
    protocol, frame, model = _model_and_frame()

    result = run_ml_common_sample_benchmark(
        frame, model, protocol, bootstrap_resamples=20
    )

    missing = result.diagnostics["registered_model_not_evaluated"]
    assert WINDOWED_QRW_NAME in missing
    assert "t_plus_h" in missing[WINDOWED_QRW_NAME]
    assert result.diagnostics["complete_registered_model_set"] is False


def test_paired_day_bootstrap_is_deterministic_and_paired() -> None:
    target = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    challenger_probability = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    baseline_probability = np.repeat(0.5, len(target))
    days = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    challenger_loss = (challenger_probability - target) ** 2
    baseline_loss = (baseline_probability - target) ** 2

    first = paired_utc_day_bootstrap(
        challenger_loss,
        baseline_loss,
        days,
        resamples=100,
        confidence_level=0.95,
        seed=2026,
    )
    second = paired_utc_day_bootstrap(
        challenger_loss,
        baseline_loss,
        days,
        resamples=100,
        confidence_level=0.95,
        seed=2026,
    )

    assert first == second
    assert first["mean_challenger_minus_baseline"] < 0.0
    assert first["challenger_significantly_better"] is True
