"""Probabilistic and Deflated Sharpe Ratios (Bailey & López de Prado, 2014).

A grid search reports the best of many trials, and the best of many noisy trials
is biased upward even when no configuration has any edge. The optimizer used to
handle this with a penalty its own comment described as "ad hoc" with
"uncalibrated constants". This module replaces that with the published
treatment.

Three pieces:

``probabilistic_sharpe_ratio``
    P(true SR > benchmark) given the observed SR, the sample length, and the
    skewness and kurtosis of the returns. Non-normal returns matter: negative
    skew and fat tails both make a given SR less trustworthy, which is exactly
    the shape a strategy that sells volatility produces.

``expected_maximum_sharpe``
    E[max SR] across ``n_trials`` independent trials whose SRs have variance
    ``sharpe_variance``, from the Gumbel approximation to the maximum of
    Gaussian draws. This is the SR you would expect to see from the *luckiest*
    configuration when none of them has skill.

``deflated_sharpe_ratio``
    The PSR measured against that expected maximum rather than against zero, so
    a strategy must beat what selection bias alone would have produced.

Reference: Bailey, D. and López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Euler-Mascheroni constant, from the Gumbel approximation for the expected
# maximum of independent Gaussian draws.
EULER_MASCHERONI = 0.5772156649015329


def sharpe_moments(returns: np.ndarray) -> dict[str, float]:
    """Per-observation Sharpe plus the skewness and (non-excess) kurtosis.

    Kurtosis is returned in its raw form, 3.0 for a normal distribution, since
    that is the convention the PSR formula expects.
    """
    values = np.asarray(returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        raise ValueError("at least three observations are required")
    mean = float(values.mean())
    deviation = float(values.std(ddof=1))
    # A constant series does not produce a standard deviation of exactly zero:
    # np.full(100, 0.01) gives 1.7e-18, which passed a `<= 0` guard and yielded
    # a Sharpe of 5.7e15. The threshold is therefore relative to the scale of
    # the data, so genuinely small P&L series (~1e-6 here) are unaffected.
    scale = max(abs(mean), float(np.abs(values).max()), np.finfo(np.float64).tiny)
    if deviation <= 1e-12 * scale:
        raise ValueError("returns have no dispersion; Sharpe is undefined")
    centred = (values - mean) / deviation
    return {
        "n_observations": int(n),
        "sharpe": mean / deviation,
        "skewness": float((centred**3).mean()),
        "kurtosis": float((centred**4).mean()),
    }


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """P(true SR > ``benchmark_sharpe``) for a non-normal return series."""
    if n_observations < 3:
        raise ValueError("at least three observations are required")
    variance = (
        1.0
        - skewness * observed_sharpe
        + 0.25 * (kurtosis - 1.0) * observed_sharpe**2
    )
    if variance <= 0.0:
        # The moment estimates are mutually inconsistent; refuse rather than
        # emit a probability from a negative variance.
        return float("nan")
    statistic = (
        (observed_sharpe - benchmark_sharpe)
        * np.sqrt(n_observations - 1)
        / np.sqrt(variance)
    )
    return float(norm.cdf(statistic))


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max SR] over ``n_trials`` skill-free trials with the given SR variance."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0.0:
        raise ValueError("sharpe_variance must be non-negative")
    if n_trials == 1 or sharpe_variance == 0.0:
        return 0.0
    scale = np.sqrt(sharpe_variance)
    quantile_high = norm.ppf(1.0 - 1.0 / n_trials)
    quantile_low = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(
        scale
        * ((1.0 - EULER_MASCHERONI) * quantile_high + EULER_MASCHERONI * quantile_low)
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
    sharpe_variance: float | None = None,
) -> dict[str, float]:
    """Deflate the observed Sharpe by what selection bias alone would produce.

    Either pass the Sharpes of every trial in the search (``trial_sharpes``) so
    their variance can be measured, or supply ``sharpe_variance`` directly. With
    neither, the deflation reduces to a plain PSR against zero and the result is
    labelled accordingly, because a DSR that ignores the search is not a DSR.
    """
    moments = sharpe_moments(returns)

    if sharpe_variance is None and trial_sharpes is not None:
        finite = np.asarray(trial_sharpes, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        sharpe_variance = float(finite.var(ddof=1)) if finite.size > 1 else 0.0

    deflated = sharpe_variance is not None and n_trials > 1
    benchmark = (
        expected_maximum_sharpe(n_trials, sharpe_variance) if deflated else 0.0
    )
    probability = probabilistic_sharpe_ratio(
        moments["sharpe"],
        moments["n_observations"],
        moments["skewness"],
        moments["kurtosis"],
        benchmark_sharpe=benchmark,
    )
    return {
        **moments,
        "n_trials": int(n_trials),
        "sharpe_variance": 0.0 if sharpe_variance is None else float(sharpe_variance),
        "expected_maximum_sharpe": benchmark,
        "deflated_sharpe_ratio": probability,
        "is_deflated": bool(deflated),
    }
