import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import StatisticalTestSuite
from src.evaluation.results_compiler import ResultsCompiler

@pytest.fixture
def mock_statistical_suite() -> StatisticalTestSuite:
    rng = np.random.default_rng(42)
    empirical = rng.normal(100, 1, size=101)
    simulated_paths = {
        "ModelA": rng.normal(100, 1.2, size=(20, 101)),
        "ModelB": rng.normal(100, 0.8, size=(20, 101)),
    }
    return StatisticalTestSuite(
        empirical,
        simulated_paths,
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
    assert "rank_mean" in ci.columns
    assert "rank_ci_low" in ci.columns
    assert "rank_ci_high" in ci.columns

@pytest.fixture
def mock_comparison_table() -> pd.DataFrame:
    """Two likelihood families with deliberately different AIC scales.

    Continuous AIC values are large-negative (more observations, different
    units) and must NOT be ranked against the directional Bernoulli family.
    """
    return pd.DataFrame(
        {
            "model": [
                "QRW Adaptive",
                "CRW Simple",
                "CRW Biased",
                "GARCH(1,1)",
                "GBM",
            ],
            "aic": [400.7, 209.3, 178.6, -26599.8, -26091.1],
            "bic": [418.7, 209.3, 181.6, -26579.6, -26081.1],
            "log_likelihood": [-194.4, -104.7, -88.3, 13303.9, 13047.6],
            "parameter_count": [6, 0, 1, 4, 2],
            "observations": [149, 151, 151, 1143, 1143],
            "likelihood_type": [
                "directional_bernoulli",
                "directional_bernoulli",
                "directional_bernoulli",
                "continuous_gaussian",
                "continuous_gaussian",
            ],
        }
    )


def test_model_selection_schema(
    mock_statistical_suite: StatisticalTestSuite,
    mock_comparison_table: pd.DataFrame,
) -> None:
    result = mock_statistical_suite.run_model_selection_tests(mock_comparison_table)
    assert isinstance(result, pd.DataFrame)
    for column in (
        "model",
        "likelihood_type",
        "aic",
        "bic",
        "aic_rank_within_family",
        "bic_rank_within_family",
        "delta_aic_within_family",
        "is_best_in_family",
    ):
        assert column in result.columns
    assert len(result) == len(mock_comparison_table)


def test_model_selection_ranks_within_family_only(
    mock_statistical_suite: StatisticalTestSuite,
    mock_comparison_table: pd.DataFrame,
) -> None:
    result = mock_statistical_suite.run_model_selection_tests(mock_comparison_table)
    result = result.set_index("model")

    # Best of the directional family is the lowest-AIC directional model,
    # NOT the large-negative GARCH from the continuous family.
    assert result.loc["CRW Biased", "aic_rank_within_family"] == 1
    assert bool(result.loc["CRW Biased", "is_best_in_family"]) is True
    assert bool(result.loc["GARCH(1,1)", "is_best_in_family"]) is True

    # Exactly one winner per family (two families -> two winners).
    assert int(result["is_best_in_family"].sum()) == 2

    # delta_aic is measured against the family minimum, so the family best is 0.
    assert result.loc["CRW Biased", "delta_aic_within_family"] == pytest.approx(0.0)
    assert result.loc["QRW Adaptive", "delta_aic_within_family"] == pytest.approx(
        400.7 - 178.6
    )


def test_model_selection_requires_likelihood_type(
    mock_statistical_suite: StatisticalTestSuite,
    mock_comparison_table: pd.DataFrame,
) -> None:
    bad = mock_comparison_table.drop(columns=["likelihood_type"])
    with pytest.raises(ValueError):
        mock_statistical_suite.run_model_selection_tests(bad)
