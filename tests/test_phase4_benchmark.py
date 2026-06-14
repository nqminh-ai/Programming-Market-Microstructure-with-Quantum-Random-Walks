"""Integration tests for the Phase 4 benchmark suite."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark_suite import BenchmarkSuite


def _benchmark_market(count: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    obi = rng.uniform(-0.9, 0.9, count)
    direction = np.where(
        obi + 0.3 * rng.standard_normal(count) >= 0.0,
        1.0,
        -1.0,
    )
    price = 100.0 + np.concatenate(
        [[0.0], np.cumsum(0.01 * direction[:-1])]
    )
    return pd.DataFrame(
        {
            "timestamp": np.arange(count, dtype=np.int64),
            "price": price,
            "tick_direction": direction,
            "obi": obi,
            "trade_intensity": rng.integers(1, 100, count),
            "obi_valid": True,
            "segment_id": 0,
        }
    )


def _zero_inflated_market(count: int = 400) -> pd.DataFrame:
    frame = _benchmark_market(count)
    changes = np.resize(
        np.array([0.01, 0.0, -0.01, 0.0], dtype=np.float64),
        count - 1,
    )
    frame["price"] = 100.0 + np.concatenate([[0.0], np.cumsum(changes)])
    direction = pd.Series(np.sign(np.concatenate([[0.0], changes])))
    frame["tick_direction"] = direction.replace(0.0, np.nan).ffill().fillna(1.0)
    return frame


def test_benchmark_suite_writes_complete_reproducible_outputs(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.csv"
    comparison = tmp_path / "comparison.csv"
    garch = tmp_path / "garch.json"
    diagnostics = tmp_path / "diagnostics.json"
    suite = BenchmarkSuite(
        _benchmark_market(),
        n_steps=50,
        n_paths=300,
        random_seed=2026,
    )

    results = suite.run(
        benchmark_output=benchmark,
        comparison_output=comparison,
        garch_output=garch,
        diagnostics_output=diagnostics,
    )

    assert results["model"].nunique() == 6
    assert set(results["metric"]) == set(BenchmarkSuite.METRICS)
    assert len(results) == 6 * len(BenchmarkSuite.METRICS)
    assert np.isfinite(results[["value", "std"]].to_numpy()).all()
    assert benchmark.exists()
    assert comparison.exists()
    assert garch.exists()
    assert diagnostics.exists()
    assert set(suite.simulated_paths) == set(results["model"])
    assert all(
        paths.shape == (300, 51)
        for paths in suite.simulated_paths.values()
    )
    stored = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert stored["garch_converged"] is True
    assert stored["coherent_no_decoherence"]["passed"] is True
    assert stored["roadmap_simple_crw_target_0_5_corrected"] is True
    assert stored["protocol_version"] == BenchmarkSuite.PROTOCOL_VERSION
    assert stored["uses_holdout_features_for_simulation"] is False


def test_qrw_forecast_does_not_use_holdout_covariates() -> None:
    original = _benchmark_market(260)
    changed = original.copy()
    cut = int(len(changed) * 0.6)
    changed.loc[cut:, "obi"] = np.linspace(-1.0, 1.0, len(changed) - cut)
    changed.loc[cut:, "tick_direction"] *= -1.0
    changed.loc[cut:, "trade_intensity"] = 10_000

    first = BenchmarkSuite(original, n_steps=30, n_paths=100)
    second = BenchmarkSuite(changed, n_steps=30, n_paths=100)
    first_model, _ = first._fit_qrw()
    second_model, _ = second._fit_qrw()

    first_paths = first._simulate_qrw(first_model, seed=7)
    second_paths = second._simulate_qrw(second_model, seed=7)
    assert np.array_equal(first_paths, second_paths)


def test_qrw_forecast_reproduces_zero_move_probability() -> None:
    suite = BenchmarkSuite(
        _zero_inflated_market(),
        n_steps=40,
        n_paths=1_000,
    )
    model, parameters = suite._fit_qrw()
    paths = suite._simulate_qrw(model, seed=2026)
    moving_fraction = np.mean(np.abs(np.diff(paths, axis=1)) > 1e-12)

    assert suite.movement_probability == pytest.approx(0.5, abs=0.01)
    assert parameters["movement_probability"] == pytest.approx(
        suite.movement_probability
    )
    assert moving_fraction == pytest.approx(
        suite.movement_probability,
        abs=0.01,
    )
