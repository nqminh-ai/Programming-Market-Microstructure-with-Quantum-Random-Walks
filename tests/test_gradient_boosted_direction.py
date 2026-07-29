"""Tests for Phase 2 chronological Histogram Gradient Boosting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ml_protocol import load_ml_protocol
from src.models.gradient_boosted_direction import (
    load_hist_gradient_boosting_model,
    save_hist_gradient_boosting_model,
    train_hist_gradient_boosting,
)


FEATURES = ("feature_a", "feature_b", "feature_c")


def _protocol():
    base = load_ml_protocol()
    raw = deepcopy(base.raw)
    settings = raw["models"]["phase_2_hist_gradient_boosting"]
    settings["grid"] = {
        "learning_rate": [0.07],
        "max_iter": [50],
        "max_leaf_nodes": [15],
        "min_samples_leaf": [10],
        "l2_regularization": [0.0],
    }
    return replace(base, raw=raw)


def _training_frame(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fold_sizes = {"train": 400, "selection": 160, "calibration": 160}
    blocks = []
    for fold, rows in fold_sizes.items():
        matrix = rng.normal(size=(rows, len(FEATURES)))
        score = (
            matrix[:, 0] * matrix[:, 1]
            + 0.4 * matrix[:, 2]
            + rng.normal(0.0, 0.35, rows)
        )
        block = pd.DataFrame(matrix, columns=FEATURES)
        block["target_up"] = (score > 0.0).astype(np.int8)
        block["fold"] = fold
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_train_select_and_calibrate_without_test_access() -> None:
    frame = _training_frame()

    model, diagnostics = train_hist_gradient_boosting(
        frame, FEATURES, _protocol()
    )
    probability = model.predict_proba(frame.loc[:49, FEATURES])

    assert len(probability) == 50
    assert ((probability > 0.0) & (probability < 1.0)).all()
    assert diagnostics["test_fold_read"] is False
    assert diagnostics["fold_rows"] == {
        "train": 400,
        "selection": 160,
        "calibration": 160,
    }
    assert len(diagnostics["candidates"]) == 1
    assert {row["kind"] for row in diagnostics["calibrators"]} == {
        "identity",
        "platt",
        "isotonic",
    }


def test_training_rejects_a_test_row_instead_of_silently_using_it() -> None:
    frame = _training_frame()
    test_row = frame.iloc[[0]].copy()
    test_row["fold"] = "test"
    contaminated = pd.concat([frame, test_row], ignore_index=True)

    with pytest.raises(ValueError, match="cannot access"):
        train_hist_gradient_boosting(
            contaminated, FEATURES, _protocol()
        )


def test_training_is_deterministic_for_the_frozen_seed() -> None:
    frame = _training_frame()
    first, first_diagnostics = train_hist_gradient_boosting(
        frame, FEATURES, _protocol()
    )
    second, second_diagnostics = train_hist_gradient_boosting(
        frame, FEATURES, _protocol()
    )

    probe = frame.loc[:99, FEATURES]
    np.testing.assert_array_equal(
        first.predict_proba(probe), second.predict_proba(probe)
    )
    assert (
        first_diagnostics["selected_parameters"]
        == second_diagnostics["selected_parameters"]
    )
    assert (
        first_diagnostics["selected_calibrator"]
        == second_diagnostics["selected_calibrator"]
    )


def test_model_serialization_round_trip_preserves_probabilities(
    tmp_path,
) -> None:
    frame = _training_frame()
    model, _ = train_hist_gradient_boosting(
        frame, FEATURES, _protocol()
    )
    destination = save_hist_gradient_boosting_model(
        model, tmp_path / "model.pkl"
    )
    restored = load_hist_gradient_boosting_model(destination)
    probe = frame.loc[:49, FEATURES]

    np.testing.assert_array_equal(
        restored.predict_proba(probe), model.predict_proba(probe)
    )
    assert restored.feature_names == FEATURES


def test_prediction_rejects_missing_or_nonfinite_features() -> None:
    frame = _training_frame()
    model, _ = train_hist_gradient_boosting(
        frame, FEATURES, _protocol()
    )

    with pytest.raises(ValueError, match="missing features"):
        model.predict_proba(frame.loc[:5, ["feature_a", "feature_b"]])
    invalid = frame.loc[:5, FEATURES].copy()
    invalid.loc[0, "feature_a"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict_proba(invalid)
