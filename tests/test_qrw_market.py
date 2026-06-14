"""Integration tests for market-adapted QRW calibration and simulation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.coin_operators import (
    biased_coin,
    dephasing_channel,
    grover_coin,
    hadamard_coin,
    obi_adaptive_coin,
    obi_expected_step,
)
from src.models.qrw_market_sim import MarketQRW


def _synthetic_market_data(count: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    obi = rng.uniform(-0.9, 0.9, size=count)
    delta_price = 0.2 * obi[:-1] + 0.01
    price = np.concatenate([[100.0], 100.0 + np.cumsum(delta_price)])
    direction = np.where(np.arange(count) % 3 == 0, -1, 1)
    return pd.DataFrame(
        {
            "timestamp": np.arange(count, dtype=np.int64),
            "price": price,
            "tick_direction": direction,
            "obi": obi,
            "trade_intensity": np.full(count, 10),
        }
    )


def test_coin_library_is_unitary_and_dephasing_preserves_populations() -> None:
    coins = [
        hadamard_coin(),
        grover_coin(),
        biased_coin(0.3),
        obi_adaptive_coin(0.5, alpha=0.2),
    ]
    for coin in coins:
        assert coin.conj().T @ coin == pytest.approx(np.eye(2), abs=1e-12)
    assert obi_adaptive_coin(0.0, alpha=2.0) == pytest.approx(hadamard_coin())

    state = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    rho = np.outer(state, state.conj())
    dephased = dephasing_channel(rho, np.inf)
    assert np.diag(dephased) == pytest.approx(np.diag(rho))
    assert dephased[0, 1] == pytest.approx(0.0)
    assert np.trace(dephased) == pytest.approx(1.0)


def test_obi_coin_has_directional_one_step_response() -> None:
    initial = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    positions = np.array([1.0, -1.0])
    for obi in (-1.0, -0.5, 0.0, 0.5, 1.0):
        evolved = obi_adaptive_coin(obi, alpha=0.2) @ initial
        mean_step = float(np.abs(evolved) ** 2 @ positions)
        assert mean_step == pytest.approx(obi_expected_step(obi, 0.2), abs=1e-12)
        if obi != 0.0:
            assert np.sign(mean_step) == np.sign(obi)


def test_market_calibration_persists_finite_parameters(tmp_path) -> None:
    output = tmp_path / "calibrated_params.json"
    model = MarketQRW(
        _synthetic_market_data(),
        {
            "n_positions": 41,
            "gamma_base": 0.01,
            "alpha_obi": 0.0,
            "coin_type": "obi_adaptive",
            "tick_size": 1.0,
        },
    )

    parameters = model.calibrate(output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == parameters
    assert parameters["gamma"] >= 0.01
    assert parameters["alpha_obi"] > 0.0
    assert parameters["calibration_status"].startswith("accepted_")
    assert parameters["calibration_method"] == (
        "regularized_market_coin_chronological_validation"
    )
    assert parameters["final_refit_includes_validation"] is False
    assert parameters["validation_beats_neutral"] is True
    assert parameters["tick_size"] == pytest.approx(1.0)
    assert parameters["obi_direction_supported"] is True
    numeric_keys = [
        "gamma",
        "alpha_obi",
        "alpha_direction",
        "obi_bias",
        "tick_size",
        "validation_log_loss",
        "neutral_validation_log_loss",
        "linear_obi_validation_log_loss",
        "selected_regularization",
    ]
    assert np.isfinite([parameters[key] for key in numeric_keys]).all()

    candidates = parameters["regularization_candidates"]
    best = min(candidates, key=lambda item: item["validation_log_loss"])
    threshold = best["validation_log_loss"] + best["validation_log_loss_se"]
    expected_regularization = max(
        item["regularization"]
        for item in candidates
        if item["validation_log_loss"] <= threshold
    )
    assert parameters["selected_regularization"] == expected_regularization


def test_market_calibration_excludes_invalid_obi_warmup(tmp_path) -> None:
    data = _synthetic_market_data()
    data["obi_valid"] = True
    data.loc[:9, "obi_valid"] = False
    data.loc[:9, "obi"] = 1.0
    model = MarketQRW(
        data,
        {
            "n_positions": 41,
            "gamma_base": 0.0,
            "alpha_obi": 0.0,
            "coin_type": "obi_adaptive",
            "tick_size": 1.0,
        },
    )

    parameters = model.calibrate(tmp_path / "params.json")
    predictor = data["obi"].to_numpy(dtype=np.float64)[:-1]
    valid = data["obi_valid"].to_numpy(dtype=bool)[:-1]
    expected_predictor = predictor[valid]

    assert parameters["obi_variance"] == pytest.approx(
        np.var(expected_predictor)
    )


def test_calibration_rejects_signal_that_reverses_in_validation(tmp_path) -> None:
    count = 100
    obi = np.where(np.arange(count) % 2 == 0, -0.9, 0.9)
    target = np.empty(count - 1, dtype=np.float64)
    target[:79] = obi[:79] > 0.0
    target[79:] = obi[79:-1] < 0.0
    price = 100.0 + np.concatenate(
        [[0.0], np.cumsum(np.where(target > 0.0, 1.0, -1.0))]
    )
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(count),
            "price": price,
            "tick_direction": np.where(np.arange(count) % 2, 1, -1),
            "obi": obi,
            "trade_intensity": np.full(count, 10),
        }
    )
    model = MarketQRW(
        frame,
        {
            "n_positions": 41,
            "gamma_base": 0.0,
            "alpha_obi": 0.0,
            "coin_type": "obi_adaptive",
            "tick_size": 1.0,
        },
    )

    parameters = model.calibrate(tmp_path / "params.json")

    assert parameters["calibration_status"] == "rejected_to_neutral"
    assert parameters["validation_beats_neutral"] is False
    assert model.alpha_obi == 0.0
    assert model.obi_bias == 0.0
    assert model._one_step_right_probability(0.9) == pytest.approx(0.5)


def test_one_step_probability_matches_calibration_link() -> None:
    model = MarketQRW(
        _synthetic_market_data(20),
        {
            "n_positions": 41,
            "gamma_base": 0.7,
            "alpha_obi": 0.8,
            "obi_bias": -0.15,
            "coin_type": "obi_adaptive",
        },
    )
    obi = 0.6
    expected = 0.5 + 0.5 * np.exp(-0.7) * np.tanh(-0.15 + 0.8 * obi)

    assert model._one_step_right_probability(obi) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_tick_direction_changes_market_coin_probability() -> None:
    model = MarketQRW(
        _synthetic_market_data(20),
        {
            "n_positions": 41,
            "gamma_base": 0.2,
            "alpha_obi": 0.5,
            "alpha_direction": -0.3,
            "obi_bias": 0.1,
            "coin_type": "obi_adaptive",
        },
    )
    expected = (
        0.5
        + 0.5
        * np.exp(-0.2)
        * np.tanh(0.1 + 0.5 * 0.4 - 0.3 * -1.0)
    )

    assert model._one_step_right_probability(0.4, -1.0) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_bias_update_keeps_structural_parameters_fixed() -> None:
    model = MarketQRW(
        _synthetic_market_data(),
        {
            "n_positions": 41,
            "gamma_base": 0.2,
            "alpha_obi": 0.7,
            "alpha_direction": -0.25,
            "obi_bias": 0.0,
            "coin_type": "obi_adaptive",
        },
    )

    update = model.calibrate_bias()

    assert update["calibration_method"] == "fixed_structure_bias_update"
    assert model.gamma == pytest.approx(0.2)
    assert model.alpha_obi == pytest.approx(0.7)
    assert model.alpha_direction == pytest.approx(-0.25)
    assert np.isfinite(model.obi_bias)


def test_two_stage_calibration_persists_fixed_structure(tmp_path) -> None:
    output = tmp_path / "two_stage.json"
    model = MarketQRW(
        _synthetic_market_data(100),
        {
            "n_positions": 41,
            "gamma_base": 0.0,
            "alpha_obi": 0.0,
            "alpha_direction": 0.0,
            "coin_type": "obi_adaptive",
            "tick_size": 1.0,
        },
    )

    parameters = model.calibrate_two_stage(output)

    assert output.exists()
    assert parameters["calibration_method"] == (
        "two_stage_disjoint_fixed_structure_bias_update"
    )
    assert parameters["structural_calibration_rows"] == 40
    assert parameters["calibration_rows"] == 100
    assert parameters["bias_update_rows"] == 60
    assert parameters["bias_update_reuses_warmup"] is False
    assert model.alpha_obi == pytest.approx(parameters["alpha_obi"])
    assert model.alpha_direction == pytest.approx(
        parameters["alpha_direction"]
    )


def test_market_simulation_returns_normalized_distributions_and_paths() -> None:
    model = MarketQRW(
        _synthetic_market_data(),
        {
            "n_positions": 41,
            "gamma_base": 0.1,
            "alpha_obi": 0.15,
            "coin_type": "obi_adaptive",
        },
    )
    simulation = model.simulate(20)
    paths = model.simulate_price_path(50, random_state=2026)

    totals = simulation.groupby("t", sort=True)["probability"].sum()
    assert totals.to_numpy() == pytest.approx(np.ones(20), abs=1e-12)
    assert paths.shape == (50, 20)
    assert np.abs(paths).max() <= 20
    assert np.all(np.abs(paths[:, 0]) == 1)
    assert np.all(np.abs(np.diff(paths, axis=1)) == 1)


def test_market_simulation_direction_changes_with_obi() -> None:
    positive = _synthetic_market_data(20)
    positive["obi"] = 1.0
    negative = positive.copy()
    negative["obi"] = -1.0
    config = {
        "n_positions": 41,
        "gamma_base": 0.0,
        "alpha_obi": 0.2,
        "coin_type": "obi_adaptive",
    }

    positive_final = MarketQRW(positive, config).simulate(1)
    negative_final = MarketQRW(negative, config).simulate(1)
    positive_mean = float(
        positive_final["position"] @ positive_final["probability"]
    )
    negative_mean = float(
        negative_final["position"] @ negative_final["probability"]
    )

    assert positive_mean > 0.0
    assert negative_mean < 0.0

    positive_paths = MarketQRW(positive, config).simulate_price_path(
        10_000,
        T=20,
        random_state=2026,
    )
    negative_paths = MarketQRW(negative, config).simulate_price_path(
        10_000,
        T=20,
        random_state=2026,
    )
    assert positive_paths[:, -1].mean() > 3.0
    assert negative_paths[:, -1].mean() < -3.0


def test_path_decoherence_reduces_directional_signal() -> None:
    data = _synthetic_market_data(20)
    data["obi"] = 1.0
    coherent = MarketQRW(
        data,
        {
            "n_positions": 41,
            "gamma_base": 0.0,
            "alpha_obi": 0.2,
            "coin_type": "obi_adaptive",
        },
    )
    dephased = MarketQRW(
        data,
        {
            "n_positions": 41,
            "gamma_base": 10.0,
            "alpha_obi": 0.2,
            "coin_type": "obi_adaptive",
        },
    )

    assert coherent._one_step_right_probability(1.0) > 0.59
    assert dephased._one_step_right_probability(1.0) == pytest.approx(
        0.5,
        abs=2e-5,
    )
