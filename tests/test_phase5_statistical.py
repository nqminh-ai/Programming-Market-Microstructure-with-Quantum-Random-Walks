"""Tests for Phase 5 statistical validation and result compilation."""

from __future__ import annotations

import numpy as np

from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.statistical_tests import StatisticalTestSuite


def _phase5_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(2026)
    empirical_steps = rng.choice([-0.01, 0.01], size=180)
    empirical = 100.0 + np.concatenate([[0.0], np.cumsum(empirical_steps)])

    n_paths = 240
    n_steps = 120
    diffusive_steps = rng.choice(
        [-0.01, 0.01],
        size=(n_paths, n_steps),
    )
    persistent_direction = np.empty((n_paths, n_steps), dtype=np.float64)
    persistent_direction[:, 0] = rng.choice([-1.0, 1.0], size=n_paths)
    for index in range(1, n_steps):
        keep = rng.random(n_paths) < 0.75
        persistent_direction[:, index] = np.where(
            keep,
            persistent_direction[:, index - 1],
            -persistent_direction[:, index - 1],
        )
    paths = {}
    for model, increments in {
        "Diffusive": diffusive_steps,
        "Persistent": 0.01 * persistent_direction,
    }.items():
        values = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        values[:, 0] = 100.0
        values[:, 1:] = 100.0 + np.cumsum(increments, axis=1)
        paths[model] = values
    return empirical, paths


def test_statistical_suite_writes_all_categories_and_figures(tmp_path) -> None:
    empirical, paths = _phase5_inputs()
    suite = StatisticalTestSuite(
        empirical,
        paths,
        bootstrap_iterations=60,
        max_lag=10,
        random_seed=2026,
    )

    results = suite.run_all(
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )

    assert set(results) == {
        "distribution",
        "variance_scaling",
        "autocorrelation",
        "tail",
    }
    distribution = results["distribution"]
    assert len(distribution) == 2 * 4
    assert distribution.loc[
        distribution["horizon"] == 1,
        ["ks_statistic", "ks_pvalue", "ks_pvalue_bh"],
    ].notna().all().all()
    scaling = results["variance_scaling"]
    assert set(scaling["model"]) == {
        "Empirical",
        "Diffusive",
        "Persistent",
    }
    assert (
        scaling["beta_ci_low"] <= scaling["beta_ci_high"]
    ).all()
    assert scaling.set_index("model").loc[
        "Empirical", "bootstrap_method"
    ] == "moving_block_returns"
    assert (
        scaling.loc[
            scaling["model"] != "Empirical",
            "bootstrap_method",
        ]
        == "path_resampling"
    ).all()
    diffusive_beta = scaling.loc[
        scaling["model"] == "Diffusive",
        "beta",
    ].iloc[0]
    assert 0.5 < diffusive_beta < 1.5
    assert set(results["autocorrelation"]["model"]) == {
        "Empirical",
        "Diffusive",
        "Persistent",
    }
    assert results["tail"]["tail_index"].notna().all()

    expected = [
        tmp_path / "results" / "distribution_tests.csv",
        tmp_path / "results" / "variance_scaling_results.csv",
        tmp_path / "results" / "autocorrelation_tests.csv",
        tmp_path / "results" / "tail_analysis.csv",
        tmp_path / "figures" / "variance_scaling.png",
        tmp_path / "figures" / "acf_comparison.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)


def test_results_compiler_ranks_every_simulated_model(tmp_path) -> None:
    empirical, paths = _phase5_inputs()
    suite = StatisticalTestSuite(
        empirical,
        paths,
        bootstrap_iterations=50,
        max_lag=10,
        random_seed=7,
    )
    results = suite.run_all(
        results_dir=tmp_path / "inputs",
        figures_dir=tmp_path / "figures",
    )

    comparison, scorecard = ResultsCompiler().compile(
        results,
        comparison_output=tmp_path / "comparison.csv",
        scorecard_output=tmp_path / "scorecard.csv",
    )

    assert set(comparison["model"]) == {"Diffusive", "Persistent"}
    assert set(scorecard["model"]) == {"Diffusive", "Persistent"}
    assert "ks_pvalue_rank" not in scorecard.columns
    assert comparison.select_dtypes(include=[np.number]).notna().all().all()
    assert scorecard["overall_rank"].min() == 1.0
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "scorecard.csv").exists()


def test_benjamini_hochberg_is_monotone_in_pvalue_order() -> None:
    adjusted = StatisticalTestSuite._benjamini_hochberg(
        [0.04, 0.001, 0.03, np.nan]
    )

    assert np.isnan(adjusted[-1])
    assert adjusted[1] <= adjusted[2] <= adjusted[0]
    assert np.all(adjusted[:3] >= np.array([0.04, 0.001, 0.03]))
