import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

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
        rolling_one_step_timestamps=np.arange(len(realized)),
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
    assert set(dm["timestamp_alignment"]) == {
        "strictly_increasing_common_timestamp_index"
    }


def test_diebold_mariano_statistic_matches_hand_computed_hac_reference() -> None:
    """Regression test for audit findings H3/M3.

    No prior test asserted a numeric value for dm_statistic/dm_pvalue, and
    the HAC variance estimator used to center each lag's autocovariance on
    its own sub-array means (via np.cov) instead of the shared sample mean.
    This independently re-derives the Newey-West-style HAC formula by hand
    and checks it against diebold_mariano_tests()'s output.
    """
    rng = np.random.default_rng(11)
    n = 60
    lossA = np.abs(rng.normal(0.02, 0.01, n))
    lossB = np.abs(rng.normal(0.018, 0.01, n))
    empirical = rng.normal(100, 1, size=n + 1)
    suite = StatisticalTestSuite(
        empirical,
        {"A": rng.normal(100, 1, size=(20, n + 1)), "B": rng.normal(100, 1, size=(20, n + 1))},
        rolling_one_step_losses={"A": lossA, "B": lossB},
        rolling_one_step_timestamps=np.arange(n),
        random_seed=42,
        bootstrap_iterations=50,
        max_lag=4,
    )

    dm = suite.diebold_mariano_tests()
    row = dm.iloc[0]

    d = lossA - lossB
    mean_d = float(np.mean(d))
    centered = d - mean_d
    variance_d = float(np.var(d, ddof=1))
    for lag in range(1, 5):
        gamma_lag = float(
            np.sum(centered[:-lag] * centered[lag:]) / (len(d) - lag - 1)
        )
        weight = 1.0 - lag / 5.0
        variance_d += 2 * weight * gamma_lag
    expected_stat = mean_d / np.sqrt(variance_d / n)
    expected_pvalue = float(2.0 * scipy_stats.norm.sf(abs(expected_stat)))

    assert row["dm_statistic"] == pytest.approx(expected_stat, rel=1e-10)
    assert row["dm_pvalue"] == pytest.approx(expected_pvalue, rel=1e-10)


def test_ljung_box_matches_statsmodels_reference() -> None:
    """Regression test for audit finding H3: no test pinned a numeric
    Ljung-Box value; cross-check against statsmodels' independent
    implementation on a synthetic AR(1) series with genuine autocorrelation.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rng = np.random.default_rng(7)
    n = 200
    series = np.empty(n)
    series[0] = rng.normal()
    for index in range(1, n):
        series[index] = 0.4 * series[index - 1] + rng.normal()

    lag = 5
    acf = StatisticalTestSuite._acf(series, lag)
    statistic, pvalue = StatisticalTestSuite._ljung_box_pvalue(
        acf, sample_size=n, lag=lag
    )

    reference = acorr_ljungbox(series, lags=[lag], return_df=True)
    assert statistic == pytest.approx(float(reference["lb_stat"].iloc[0]), rel=1e-9)
    assert pvalue == pytest.approx(float(reference["lb_pvalue"].iloc[0]), rel=1e-6)


def test_dm_rejects_missing_or_nonmonotone_timestamp_alignment(
    mock_statistical_suite: StatisticalTestSuite,
) -> None:
    common = {
        "empirical_prices": mock_statistical_suite.empirical_prices,
        "forecast_samples": mock_statistical_suite.forecast_samples,
        "rolling_one_step_losses": (
            mock_statistical_suite.rolling_one_step_losses
        ),
        "bootstrap_iterations": 50,
    }
    with pytest.raises(ValueError, match="timestamps are required"):
        StatisticalTestSuite(**common)

    timestamps = mock_statistical_suite.rolling_one_step_timestamps.copy()
    timestamps[50] = timestamps[49]
    with pytest.raises(ValueError, match="finite and unique"):
        StatisticalTestSuite(
            **common,
            rolling_one_step_timestamps=timestamps,
        )

def test_bootstrap_scorecard_ci(mock_statistical_suite: StatisticalTestSuite) -> None:
    ci = mock_statistical_suite.run_bootstrap_scorecard_ci()
    assert isinstance(ci, pd.DataFrame)
    assert len(ci) == 2
    assert set(ci["resampled_metric_count"]) == {7}
    assert set(ci["rank_selection_rule"]) == {
        StatisticalTestSuite.BOOTSTRAP_SELECTION_RULE
    }
    assert ci["probability_rank_1"].sum() == pytest.approx(1.0)
    assert ci["probability_rank_1"].between(0.0, 1.0).all()
    assert ci["overall_rank_ci_low"].between(1.0, 2.0).all()
    assert ci["overall_rank_ci_high"].between(1.0, 2.0).all()
    for metric in StatisticalTestSuite.BOOTSTRAP_SCORECARD_METRICS:
        assert metric in ci.columns
        assert f"{metric}_ci_low" in ci.columns
        assert f"{metric}_ci_high" in ci.columns
        assert (
            ci[f"{metric}_ci_low"] <= ci[f"{metric}_ci_high"]
        ).all()

def test_aic_bic_matches_reference_values_and_ranks_within_family() -> None:
    reference = BenchmarkSuite._comparison_row(
        "Known",
        log_likelihood=-123.4,
        parameter_count=3,
        observations=100,
        empirical_volatility=0.01,
        model_volatility=0.02,
        likelihood_type="directional_bernoulli",
    )
    assert reference["aic"] == pytest.approx(252.8)
    assert reference["bic"] == pytest.approx(
        246.8 + 3.0 * np.log(100.0)
    )

    comparison = pd.DataFrame(
        {
            "model": ["Bernoulli A", "Bernoulli B", "Gaussian"],
            "likelihood_type": [
                "directional_bernoulli",
                "directional_bernoulli",
                "continuous_gaussian",
            ],
            "aic": [10.0, 12.0, 2.0],
            "bic": [11.0, 13.0, 3.0],
        }
    )
    ranked = BenchmarkSuite.rank_information_criteria(comparison)
    assert ranked["aic_rank_within_likelihood"].tolist() == [1.0, 2.0, 1.0]
    assert ranked["bic_rank_within_likelihood"].tolist() == [1.0, 2.0, 1.0]
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
