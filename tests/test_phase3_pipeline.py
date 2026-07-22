"""Tests for Phase 3 pipeline integration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pipelines.phase3_pipeline import (
    benchmark_paths,
    chronological_train_holdout_split,
)


def test_phase3_benchmark_accepts_calibrated_zero_moves() -> None:
    count = 20
    market = pd.DataFrame(
        {
            "timestamp": np.arange(count, dtype=np.int64),
            "price": np.full(count, 100.0),
            "tick_direction": np.ones(count),
            "obi": np.linspace(-0.5, 0.5, count),
            "trade_intensity": np.arange(1, count + 1),
        }
    )
    parameters = {
        "gamma": 0.0,
        "alpha_obi": 0.0,
        "alpha_direction": 0.0,
        "alpha_obi_change": 0.0,
        "alpha_abs_obi": 0.0,
        "gamma_intensity": 0.0,
        "obi_bias": 0.0,
        "feature_mean": [0.0] * 5,
        "feature_scale": [1.0] * 5,
        "movement_probability": 0.0,
    }

    result = benchmark_paths(
        market_data=market,
        parameters=parameters,
        n_steps=count,
        n_paths=100,
        random_seed=2026,
    )

    assert result["paths_are_local"] is True
    assert result["max_absolute_increment"] == 0
    assert result["simulated_move_fraction"] == pytest.approx(0.0)
    assert result["sampler_semantics"] == (
        "classical_bernoulli_directional_surrogate"
    )
    assert result["official_quantum_evaluation"] is False


def test_phase3_split_is_strictly_chronological() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(150, dtype=np.int64)[::-1],
            "price": 100.0,
        }
    )

    train, holdout = chronological_train_holdout_split(
        frame,
        train_fraction=0.8,
    )

    assert len(train) == 120
    assert len(holdout) == 30
    assert train["timestamp"].max() < holdout["timestamp"].min()
