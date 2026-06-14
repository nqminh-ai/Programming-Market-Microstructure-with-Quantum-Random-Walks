"""Tests for intensity-adaptive market decoherence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.adaptive_market_qrw import AdaptiveDecoherenceQRW


def _market_data(count: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    obi = rng.uniform(-0.9, 0.9, count)
    direction = np.where(np.arange(count) % 3, 1.0, -1.0)
    intensity = np.linspace(2.0, 100.0, count)
    signal = obi[:-1] + 0.3 * direction[:-1]
    price = 100.0 + np.concatenate(
        [[0.0], np.cumsum(np.where(signal > 0.0, 1.0, -1.0))]
    )
    return pd.DataFrame(
        {
            "timestamp": np.arange(count),
            "price": price,
            "tick_direction": direction,
            "obi": obi,
            "trade_intensity": intensity,
            "obi_valid": True,
            "segment_id": 0,
        }
    )


def test_adaptive_probability_uses_intensity_decoherence() -> None:
    model = AdaptiveDecoherenceQRW(
        _market_data(),
        {
            "n_positions": 41,
            "gamma_base": 0.2,
            "obi_bias": 0.1,
            "alpha_obi": 0.8,
            "gamma_intensity": -1.0,
            "feature_mean": [0.0, 0.0, 0.0, 0.0, 2.0],
            "feature_scale": [1.0, 1.0, 1.0, 1.0, 0.5],
        },
    )
    low = np.array([0.5, 0.0, 0.0, 0.5, 1.5])
    high = np.array([0.5, 0.0, 0.0, 0.5, 2.5])

    low_gamma, _ = model._event_kernel(low)
    high_gamma, _ = model._event_kernel(high)

    assert high_gamma < low_gamma


def test_adaptive_calibration_persists_finite_parameters(tmp_path) -> None:
    output = tmp_path / "adaptive.json"
    model = AdaptiveDecoherenceQRW(
        _market_data(),
        {"n_positions": 41},
    )

    parameters = model.calibrate_two_stage(output)

    assert json.loads(output.read_text(encoding="utf-8")) == parameters
    assert parameters["calibration_method"] == (
        "adaptive_decoherence_disjoint_two_stage_brier_validation"
    )
    assert parameters["final_refit_includes_validation"] is False
    assert parameters["bias_update_reuses_warmup"] is False
    assert np.isfinite(
        [
            parameters["gamma"],
            parameters["alpha_obi"],
            parameters["alpha_direction"],
            parameters["alpha_obi_change"],
            parameters["alpha_abs_obi"],
            parameters["gamma_intensity"],
        ]
    ).all()


def test_adaptive_simulation_is_normalized_and_local() -> None:
    model = AdaptiveDecoherenceQRW(
        _market_data(),
        {
            "n_positions": 41,
            "gamma_base": 0.2,
            "alpha_obi": 0.8,
            "alpha_direction": 0.2,
            "gamma_intensity": -0.5,
            "feature_mean": [0.0, 0.0, 0.0, 0.5, 3.0],
            "feature_scale": [1.0, 1.0, 1.0, 0.5, 1.0],
        },
    )

    simulation = model.simulate(20)
    paths = model.simulate_price_path(500, T=20, random_state=2026)

    totals = simulation.groupby("t")["probability"].sum().to_numpy()
    assert totals == pytest.approx(np.ones(20), abs=1e-12)
    assert np.all(np.abs(paths[:, 0]) == 1)
    assert np.all(np.abs(np.diff(paths, axis=1)) == 1)
