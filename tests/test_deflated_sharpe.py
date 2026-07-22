"""Tests for the Probabilistic and Deflated Sharpe Ratios."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_moments,
)


def test_moments_recover_a_normal_series() -> None:
    rng = np.random.default_rng(2026)
    moments = sharpe_moments(rng.normal(0.0, 1.0, 200_000))
    assert moments["skewness"] == pytest.approx(0.0, abs=0.05)
    # Raw, not excess: a normal distribution has kurtosis 3.
    assert moments["kurtosis"] == pytest.approx(3.0, abs=0.1)


def test_flat_series_is_refused_rather_than_scored() -> None:
    with pytest.raises(ValueError, match="dispersion"):
        sharpe_moments(np.full(100, 0.01))


def test_psr_rises_with_sample_length() -> None:
    """The same Sharpe is more convincing when measured over more data."""
    short = probabilistic_sharpe_ratio(0.1, 100, 0.0, 3.0)
    long = probabilistic_sharpe_ratio(0.1, 10_000, 0.0, 3.0)
    assert long > short


def test_psr_penalises_negative_skew_and_fat_tails() -> None:
    """Non-normality is the reason PSR exists; it must actually bite."""
    normal = probabilistic_sharpe_ratio(0.1, 1_000, 0.0, 3.0)
    skewed = probabilistic_sharpe_ratio(0.1, 1_000, -1.5, 3.0)
    fat = probabilistic_sharpe_ratio(0.1, 1_000, 0.0, 12.0)
    assert skewed < normal
    assert fat < normal


def test_expected_maximum_sharpe_grows_with_the_number_of_trials() -> None:
    variance = 0.01
    assert expected_maximum_sharpe(1, variance) == 0.0
    values = [expected_maximum_sharpe(n, variance) for n in (10, 100, 1_000, 10_000)]
    assert values == sorted(values)


def test_a_single_trial_needs_no_deflation() -> None:
    assert expected_maximum_sharpe(1, 0.5) == 0.0
    assert expected_maximum_sharpe(500, 0.0) == 0.0


def test_best_of_many_skill_free_trials_is_rejected() -> None:
    """The case the ad hoc penalty was meant to catch, done properly.

    Cherry-picking the best of 400 pure-noise series produces a Sharpe whose
    plain PSR is above 0.99. Deflating against the expected maximum of those
    400 trials must take it back below significance.
    """
    rng = np.random.default_rng(2026)
    trials = [rng.normal(0.0, 0.01, 1_000) for _ in range(400)]
    sharpes = np.array([sharpe_moments(series)["sharpe"] for series in trials])
    best = trials[int(np.argmax(sharpes))]

    naive = probabilistic_sharpe_ratio(
        *(
            sharpe_moments(best)[key]
            for key in ("sharpe", "n_observations", "skewness", "kurtosis")
        )
    )
    result = deflated_sharpe_ratio(best, n_trials=400, trial_sharpes=sharpes)

    assert naive > 0.99
    assert result["deflated_sharpe_ratio"] < 0.95
    assert result["is_deflated"] is True


def test_genuine_skill_survives_deflation() -> None:
    """Guards the other direction, so the rejection above is not blanket."""
    rng = np.random.default_rng(7)
    sharpes = np.array(
        [sharpe_moments(rng.normal(0.0, 0.01, 1_000))["sharpe"] for _ in range(400)]
    )
    skilled = rng.normal(0.003, 0.01, 1_000)
    result = deflated_sharpe_ratio(skilled, n_trials=400, trial_sharpes=sharpes)
    assert result["deflated_sharpe_ratio"] > 0.95


def test_result_is_labelled_undeflated_when_the_search_is_unknown() -> None:
    """A DSR that ignores the search is not a DSR and must not claim to be."""
    rng = np.random.default_rng(11)
    result = deflated_sharpe_ratio(rng.normal(0.001, 0.01, 1_000), n_trials=1)
    assert result["is_deflated"] is False
    assert result["expected_maximum_sharpe"] == 0.0
