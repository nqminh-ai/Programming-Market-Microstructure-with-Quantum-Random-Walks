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
    assert stored["qrw_forecast"]["state_evolution"].startswith("density_matrix")
    assert len(stored["qrw_forecast"]["position_variance_by_horizon"]) == 50
    assert stored["evaluation_semantics"] == "fixed_origin_marginals_only"
    assert stored["sample_semantics"]["QRW Adaptive"] == (
        "fixed_origin_marginals"
    )
    assert stored["path_dependent_metrics_include_qrw"] is False
    assert set(suite.model_comparison["information_criterion_scope"]) == {
        "within_likelihood_type_only"
    }
    assert (
        suite.model_comparison.groupby("likelihood_type")[
            "aic_rank_within_likelihood"
        ].min()
        == 1.0
    ).all()


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
    first_horizon_move_fraction = np.mean(
        np.abs(paths[:, 1] - paths[:, 0]) > 1e-12
    )

    assert suite.movement_probability == pytest.approx(0.5, abs=0.01)
    assert parameters["movement_probability"] == pytest.approx(0.5, abs=0.01)
    assert first_horizon_move_fraction == pytest.approx(
        parameters["movement_probability"],
        abs=0.04,
    )


def test_benchmark_accepts_explicit_non_overlapping_holdout() -> None:
    train = _benchmark_market(200)
    holdout = _benchmark_market(80)
    holdout["timestamp"] += len(train)
    suite = BenchmarkSuite(
        train,
        holdout_data=holdout,
        n_steps=20,
        n_paths=100,
    )

    assert suite.split_strategy == "explicit_chronological_holdout"
    assert len(suite.train) == 200
    assert len(suite.test) == 21
    assert suite.initial_price == pytest.approx(train["price"].iloc[-1])
    assert suite.train["timestamp"].max() < suite.holdout["timestamp"].min()


def test_qrw_score_is_invariant_to_cross_horizon_row_pairing() -> None:
    suite = BenchmarkSuite(
        _benchmark_market(),
        n_steps=30,
        n_paths=200,
        random_seed=11,
    )
    model, _ = suite._fit_qrw()
    marginals = suite._simulate_qrw(model, seed=12)
    independently_permuted = marginals.copy()
    rng = np.random.default_rng(13)
    for horizon in range(1, independently_permuted.shape[1]):
        rng.shuffle(independently_permuted[:, horizon])

    original, _ = suite._score_marginals("QRW", marginals)
    permuted, _ = suite._score_marginals("QRW", independently_permuted)
    assert [row["metric"] for row in original] == [
        row["metric"] for row in permuted
    ]
    assert [row["value"] for row in original] == pytest.approx(
        [row["value"] for row in permuted]
    )
    assert [row["std"] for row in original] == pytest.approx(
        [row["std"] for row in permuted]
    )
