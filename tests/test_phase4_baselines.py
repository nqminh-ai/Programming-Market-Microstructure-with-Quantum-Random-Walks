"""Tests for Phase 4 classical baseline implementations."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.classical_rw import ClassicalRandomWalk
from src.models.garch_model import GARCHBaseline
from src.models.gbm_model import GBMBaseline


def test_simple_random_walk_has_local_steps_and_diffusive_variance() -> None:
    model = ClassicalRandomWalk(kind="simple", random_state=2026)
    paths = model.simulate(200, 20_000)
    increments = np.diff(paths, axis=1)

    assert paths.shape == (20_000, 201)
    assert np.all(np.abs(increments) == 1.0)
    assert np.var(paths[:, -1]) / 200 == pytest.approx(1.0, abs=0.03)


def test_biased_and_correlated_walks_calibrate_empirical_directions() -> None:
    directions = np.array(
        [1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
        dtype=np.float64,
    )
    biased = ClassicalRandomWalk(kind="biased")
    correlated = ClassicalRandomWalk(kind="correlated")

    biased_fit = biased.fit(directions)
    correlated_fit = correlated.fit(directions)

    expected_persistence = np.mean(directions[1:] == directions[:-1])
    assert biased_fit["p_up"] == pytest.approx(0.5)
    assert correlated_fit["rho"] == pytest.approx(
        2.0 * expected_persistence - 1.0
    )
    assert correlated.simulate(20, 50, random_state=7).shape == (50, 21)


def test_classical_walks_reproduce_empirical_zero_move_probability() -> None:
    directions = np.array(
        [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0]
    )
    model = ClassicalRandomWalk(kind="biased")

    parameters = model.fit(directions)
    paths = model.simulate(100, 2_000, random_state=2026)
    increments = np.diff(paths, axis=1)

    assert parameters["p_move"] == pytest.approx(0.5)
    assert set(np.unique(increments)) == {-1.0, 0.0, 1.0}
    assert np.mean(increments != 0.0) == pytest.approx(0.5, abs=0.01)


def _garch_returns(count: int = 800) -> np.ndarray:
    rng = np.random.default_rng(2026)
    omega, alpha, beta = 0.02, 0.08, 0.88
    variance = omega / (1.0 - alpha - beta)
    residual = 0.0
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        variance = omega + alpha * residual**2 + beta * variance
        residual = np.sqrt(variance) * rng.standard_normal()
        values[index] = (0.01 + residual) / 10_000.0
    return values


def test_garch_fits_stationary_process_and_persists_parameters(tmp_path) -> None:
    output = tmp_path / "garch.json"
    model = GARCHBaseline(initial_price=100.0)
    parameters = model.fit(_garch_returns(), output_path=output)
    paths = model.simulate(30, 200, random_state=2026)

    assert parameters["convergence_flag"] == 0
    assert 0.0 <= parameters["alpha"] < 1.0
    assert 0.0 <= parameters["beta"] < 1.0
    assert parameters["persistence"] < 1.0
    assert json.loads(output.read_text(encoding="utf-8")) == parameters
    assert paths.shape == (200, 31)
    assert np.isfinite(paths).all()
    assert np.all(paths > 0.0)


def test_gbm_mle_and_simulation_match_log_return_moments() -> None:
    rng = np.random.default_rng(2026)
    log_returns = 0.0002 + 0.0015 * rng.standard_normal(10_000)
    model = GBMBaseline(initial_price=100.0)
    parameters = model.fit(log_returns)
    paths = model.simulate(20, 500, random_state=7)

    assert parameters["sigma"] == pytest.approx(0.0015, rel=0.03)
    assert (
        parameters["mu"] - 0.5 * parameters["sigma"] ** 2
        == pytest.approx(np.mean(log_returns), abs=1e-12)
    )
    assert paths.shape == (500, 21)
    assert np.all(paths > 0.0)
