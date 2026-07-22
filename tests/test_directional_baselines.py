"""Tests for strong classical directional baselines and shared folds."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.evaluation.directional_baselines import (
    FEATURE_NAMES,
    MODEL_NAMES,
    _marked_hawkes_state,
    chronological_day_split,
    directional_events,
    evaluate_directional_baselines,
    fit_directional_baselines,
)


def market_frame(rows: int = 1_200) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    obi = np.clip(rng.normal(0.0, 0.55, rows), -1.0, 1.0)
    intensity = rng.lognormal(2.0, 0.6, rows)
    direction = np.ones(rows, dtype=np.float64)
    price = np.empty(rows, dtype=np.float64)
    price[0] = 100.0
    for index in range(rows - 1):
        probability = 1.0 / (
            1.0 + np.exp(-(1.4 * obi[index] + 0.45 * direction[index]))
        )
        move = 1.0 if rng.random() < probability else -1.0
        price[index + 1] = price[index] + 0.01 * move
        direction[index + 1] = move
    return pd.DataFrame(
        {
            "timestamp": np.arange(rows, dtype=np.int64),
            "price": price,
            "tick_direction": direction,
            "obi": obi,
            "trade_intensity": intensity,
            "obi_valid": True,
            "segment_id": 0,
        }
    )


def test_directional_events_use_exactly_five_causal_features() -> None:
    events = directional_events(market_frame(300))

    assert events.features.shape == (299, 5)
    assert len(FEATURE_NAMES) == 5
    assert set(np.unique(events.target)) == {0.0, 1.0}


def test_daily_folds_are_disjoint_and_chronological() -> None:
    train, validation, test = chronological_day_split(market_frame())

    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert min(len(train), len(validation), len(test)) >= 20


def test_all_strong_baselines_share_validation_and_test_samples() -> None:
    summary, prediction, diagnostics = evaluate_directional_baselines(
        market_frame()
    )

    assert tuple(summary["model"]) == MODEL_NAMES
    assert summary["sample_size"].nunique() == 1
    assert summary["sample_size"].iloc[0] == len(prediction)
    assert prediction[list(MODEL_NAMES)].notna().all().all()
    assert (
        (prediction[list(MODEL_NAMES)] > 0.0)
        & (prediction[list(MODEL_NAMES)] < 1.0)
    ).all().all()
    assert diagnostics["same_test_sample_for_all_models"] is True
    assert diagnostics["selection_fold"] == "shared_chronological_validation"
    assert diagnostics["marked_hawkes_semantics"].endswith(
        "not_full_arrival_process_likelihood"
    )


def test_designs_cover_main_interactions_nonlinear_ar_and_marked_hawkes() -> None:
    train, validation, test = chronological_day_split(market_frame())
    models = fit_directional_baselines(
        train,
        validation,
        regularization_grid=(1e-3, 1e-2),
        decay_grid=(0.5, 0.9),
    )

    assert models["Logistic L2 (5F)"].design_feature_count == 5
    assert models["QRW Directional Link"].design_feature_count == 5
    assert models["QRW Directional Link"].gamma is not None
    assert models["Logistic L2 + Pairwise"].design_feature_count == 15
    assert models["Nonlinear Calibrated"].design_feature_count == 10
    assert models["OrderFlow AR(5)"].design_feature_count == 10
    hawkes = models["Marked Hawkes Logit"]
    assert hawkes.design_feature_count == 7
    assert hawkes.decay in {0.5, 0.9}
    assert len(hawkes.history_features) == len(train) + len(validation)
    reset_hawkes = replace(
        hawkes, history_features=None, history_timestamp=None
    )
    assert not np.allclose(hawkes.predict(test), reset_hawkes.predict(test))
    nonlinear = models["Nonlinear Calibrated"]
    assert nonlinear.calibration_threshold is not None
    assert np.all(np.diff(nonlinear.calibration_probability) >= 0.0)


def test_model_selection_does_not_read_test_targets() -> None:
    train, validation, test = chronological_day_split(market_frame())
    changed_test = type(test)(
        test.features.copy(), 1.0 - test.target, test.timestamp.copy()
    )

    first = fit_directional_baselines(train, validation)
    second = fit_directional_baselines(train, validation)

    assert changed_test.target.mean() != test.target.mean()
    for name in MODEL_NAMES:
        assert first[name].regularization == second[name].regularization
        assert first[name].decay == second[name].decay
        np.testing.assert_allclose(first[name].beta, second[name].beta)


def test_marked_hawkes_state_uses_time_and_only_current_or_past_marks() -> None:
    features = np.zeros((4, 5), dtype=np.float64)
    features[:, 1] = [1.0, -1.0, 1.0, -1.0]
    features[:, 4] = 1.0
    regular_time = np.array([0.0, 1.0, 2.0, 3.0])
    long_gap = np.array([0.0, 1.0, 20.0, 21.0])

    regular = _marked_hawkes_state(features, regular_time, decay=0.8)
    separated = _marked_hawkes_state(features, long_gap, decay=0.8)
    changed_future = features.copy()
    changed_future[-1, 1] = 1.0
    causal = _marked_hawkes_state(changed_future, regular_time, decay=0.8)

    assert separated[2, 1] < regular[2, 1]
    np.testing.assert_allclose(causal[:-1], regular[:-1])
