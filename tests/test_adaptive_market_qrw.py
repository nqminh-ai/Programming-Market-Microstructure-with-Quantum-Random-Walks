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


def test_movement_probability_uses_warmup_only() -> None:
    frame = _market_data()
    warmup_rows = int(len(frame) * 0.4)
    increments = np.ones(len(frame) - 1, dtype=np.float64)
    increments[warmup_rows - 1 :] = np.resize(
        np.array([0.0, 1.0]),
        len(increments) - warmup_rows + 1,
    )
    frame["price"] = 100.0 + np.concatenate([[0.0], np.cumsum(increments)])
    model = AdaptiveDecoherenceQRW(frame, {"n_positions": 41})

    parameters = model.calibrate_two_stage(None)

    assert parameters["movement_probability"] == pytest.approx(1.0)
    assert parameters["movement_probability_calibration_rows"] == warmup_rows
    assert parameters["movement_probability_uses_post_warmup"] is False


def test_adaptive_simulation_is_normalized() -> None:
    """Density-matrix simulate() must sum to 1.0 at every time step."""
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

    totals = simulation.groupby("t")["probability"].sum().to_numpy()
    assert totals == pytest.approx(np.ones(20), abs=1e-12)


def test_benchmark_simulate_qrw_evolves_density_matrix() -> None:
    """BenchmarkSuite._simulate_qrw must use density matrix, not Bernoulli.

    If the engine is a density-matrix walk (not a coin-flip surrogate),
    variance must grow with forecast horizon — reflecting the ballistic
    spreading of the underlying quantum state.
    """
    from src.evaluation.benchmark_suite import BenchmarkSuite

    def _market_data_for_benchmark(count: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        obi = rng.uniform(-0.9, 0.9, count)
        direction = np.where(obi + 0.2 * rng.standard_normal(count) >= 0, 1.0, -1.0)
        price = 100.0 + np.concatenate([[0.0], np.cumsum(0.01 * direction[:-1])])
        return pd.DataFrame(
            {
                "timestamp": np.arange(count, dtype=np.int64),
                "price": price,
                "tick_direction": direction,
                "obi": obi,
                "trade_intensity": rng.integers(1, 100, count).astype(float),
                "obi_valid": True,
                "segment_id": 0,
            }
        )

    suite = BenchmarkSuite(_market_data_for_benchmark(), n_steps=20, n_paths=100)
    model, _ = suite._fit_qrw()
    marginals = suite._simulate_qrw(model, seed=42)

    var_h1 = float(np.var(marginals[:, 1]))
    var_h10 = float(np.var(marginals[:, min(10, marginals.shape[1] - 1)]))
    # variance must grow with horizon (density-matrix ballistic property)
    assert var_h10 > var_h1 * 2, (
        f"Expected variance to grow with horizon: var_h1={var_h1:.4f}, "
        f"var_h10={var_h10:.4f}"
    )
    # state_evolution key must confirm density-matrix engine
    diag = suite._qrw_forecast_diagnostics
    assert diag.get("state_evolution", "").startswith("density_matrix"), (
        f"Expected density_matrix engine, got: {diag.get('state_evolution')}"
    )
