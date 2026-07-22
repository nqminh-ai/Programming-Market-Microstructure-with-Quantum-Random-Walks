"""Acceptance tests for Track A Phase 3 (risk backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.risk_simulator import QRWRiskSimulator


def symmetric_state(n: int = 101) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    probability = np.exp(-0.5 * (x / 0.18) ** 2)
    return probability / probability.sum()


def market_frame(size: int = 3000) -> pd.DataFrame:
    returns = np.random.default_rng(7).normal(0.0, 0.001, size)
    return pd.DataFrame({"price": 100 * np.exp(np.cumsum(returns))})


def test_simulate_paths_shape():
    paths = QRWRiskSimulator(n_paths=123).simulate_paths(symmetric_state(), horizon=17)
    assert paths.shape == (123, 17)


def test_var_monotone_in_confidence():
    sim = QRWRiskSimulator(n_paths=8000, seed=1)
    paths = sim.simulate_paths(symmetric_state(), horizon=10)
    assert sim.compute_var(paths, 0.99) <= sim.compute_var(paths, 0.95)


def test_es_less_than_var():
    sim = QRWRiskSimulator(n_paths=8000, seed=2)
    paths = sim.simulate_paths(symmetric_state(), horizon=10)
    assert sim.compute_es(paths, 0.95) <= sim.compute_var(paths, 0.95)


def test_backtest_violation_rate_is_near_target():
    result = QRWRiskSimulator().backtest_var(market_frame(), window=250)
    assert 0.03 <= result["violation_rate"] <= 0.08
    assert result["n_tests"] == 2750


def test_qrw_var_can_capture_fatter_tail_than_gaussian():
    probability = np.zeros(101)
    probability[0] = probability[-1] = 0.06
    probability[50] = 0.88
    sim = QRWRiskSimulator(n_paths=50_000, seed=3)
    paths = sim.simulate_paths(probability, horizon=1, return_scale=0.01)
    final = paths[:, -1]
    gaussian = float(norm_ppf_05() * final.std(ddof=1))
    assert sim.compute_var(paths, 0.95) < gaussian


def norm_ppf_05() -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.05))


def test_seed_reproducible():
    first = QRWRiskSimulator(n_paths=30, seed=99).simulate_paths(symmetric_state(), 5)
    second = QRWRiskSimulator(n_paths=30, seed=99).simulate_paths(symmetric_state(), 5)
    np.testing.assert_allclose(first, second)


def test_normal_scenario_lower_risk_than_stress():
    normal = QRWRiskSimulator(n_paths=20_000, seed=11).scenario_var(symmetric_state(), "normal")
    stress = QRWRiskSimulator(n_paths=20_000, seed=11).scenario_var(symmetric_state(), "stress")
    assert stress["var_95"] < normal["var_95"]


def test_stress_scenario_lower_risk_than_crisis():
    stress = QRWRiskSimulator(n_paths=20_000, seed=12).scenario_var(symmetric_state(), "stress")
    crisis = QRWRiskSimulator(n_paths=20_000, seed=12).scenario_var(symmetric_state(), "crisis")
    assert crisis["var_95"] < stress["var_95"]


def test_n_paths_validation_and_acceptance():
    assert QRWRiskSimulator(n_paths=7).simulate_paths(symmetric_state(), 2).shape[0] == 7
    with pytest.raises(ValueError):
        QRWRiskSimulator(n_paths=0)


def test_var_95_less_extreme_than_99():
    sim = QRWRiskSimulator(n_paths=10_000, seed=13)
    result = sim.scenario_var(symmetric_state(), "crisis")
    assert result["var_95"] >= result["var_99"]
