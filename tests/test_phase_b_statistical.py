import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import StatisticalTestSuite
from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.benchmark_suite import BenchmarkSuite

@pytest.fixture
def mock_statistical_suite() -> StatisticalTestSuite:
    rng = np.random.default_rng(42)
    empirical = rng.normal(100, 1, size=101)
    simulated_paths = {
        "ModelA": rng.normal(100, 1.2, size=(20, 101)),
        "ModelB": rng.normal(100, 0.8, size=(20, 101)),
    }
    realized = empirical[1:]
    losses = {
        model: np.abs(paths[:, 1:].mean(axis=0) - realized)
        for model, paths in simulated_paths.items()
    }
    return StatisticalTestSuite(
        empirical,
        simulated_paths,
        rolling_one_step_losses=losses,
        random_seed=42,
        bootstrap_iterations=50,
        max_lag=5,
    )

def test_diebold_mariano(mock_statistical_suite: StatisticalTestSuite) -> None:
    dm = mock_statistical_suite.diebold_mariano_tests()
    assert isinstance(dm, pd.DataFrame)
    # 2 models -> 1 comparison (ModelA vs ModelB) or 2 depending on loop
    # In our implementation, i < j, so 1 comparison
    assert len(dm) == 1
    assert "model1" in dm.columns
    assert "model2" in dm.columns
    assert "dm_statistic" in dm.columns
    assert "dm_pvalue" in dm.columns
    assert "dm_pvalue_bh" in dm.columns

def test_bootstrap_scorecard_ci(mock_statistical_suite: StatisticalTestSuite) -> None:
    ci = mock_statistical_suite.run_bootstrap_scorecard_ci()
    assert isinstance(ci, pd.DataFrame)
    assert len(ci) == 2
    assert "model" in ci.columns
    assert "mean_crps_ci_low" in ci.columns
    assert "mean_crps_ci_high" in ci.columns
    assert "mean_direction_log_loss_ci_low" in ci.columns
    assert "mean_direction_log_loss_ci_high" in ci.columns

def test_results_compiler_aic_bic_mock() -> None:
    comparison = pd.DataFrame(
        {
            "model": ["Bernoulli", "Gaussian"],
            "likelihood_type": ["directional_bernoulli", "continuous_gaussian"],
            "aic": [10.0, 2.0],
            "bic": [11.0, 3.0],
        }
    )
    ranked = BenchmarkSuite.rank_information_criteria(comparison)
    assert ranked["aic_rank_within_likelihood"].tolist() == [1.0, 1.0]
    assert set(ranked["information_criterion_scope"]) == {
        "within_likelihood_type_only"
    }


def test_empty_aic_bic_table_is_handled_without_cross_family_ranking() -> None:
    empty = pd.DataFrame(columns=["likelihood_type", "aic", "bic"])

    ranked = BenchmarkSuite.rank_information_criteria(empty)

    assert ranked.empty
    assert {
        "aic_rank_within_likelihood",
        "bic_rank_within_likelihood",
        "information_criterion_scope",
    } <= set(ranked.columns)


def test_marginal_samples_are_excluded_from_path_statistics() -> None:
    rng = np.random.default_rng(9)
    empirical = 100.0 + np.cumsum(rng.choice([-0.01, 0.01], 101))
    marginal = rng.normal(100.0, 0.1, size=(30, 101))
    marginal[:, 0] = empirical[0]
    suite = StatisticalTestSuite(
        empirical,
        {"QRW": marginal},
        sample_semantics={"QRW": "fixed_origin_marginals"},
        bootstrap_iterations=50,
        max_lag=5,
    )

    assert set(suite.autocorrelation_tests()["model"]) == {"Empirical"}
    assert set(suite.tail_analysis()["model"]) == {"Empirical"}
    with pytest.raises(ValueError, match="rolling_one_step_losses"):
        suite.diebold_mariano_tests()
