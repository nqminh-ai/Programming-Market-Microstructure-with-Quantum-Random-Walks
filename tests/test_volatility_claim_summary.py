"""Tests for the cross-asset ruling on the report's volatility claim.

One asset failing to reach significance is not the same as the claim being
wrong, and one asset reaching it is not the same as the claim being right.
These cover the pooling that turns three per-asset tests into one answer, and
the recomputation that stops window scale from leaking into it.
"""

from __future__ import annotations

import pytest

from scripts.research.volatility_claim_summary import (
    combine_assets,
    one_sided_p,
    relative_gaps,
)


def _asset(rho: float, p: float, label: str = "X") -> dict:
    return {"label": label, "spearman": rho, "p_value": p}


# ---------------------------------------------------------------------------
# Rebuilding the gap from the stored scores
# ---------------------------------------------------------------------------


def _window(qrw: float, rival: float, volatility: float = 1e-5, index: int = 0) -> dict:
    return {
        "window": index,
        "realised_volatility": volatility,
        "qrw_crps_gap": qrw - rival,
        "scores": {
            "QRW Adaptive": {"mean_marginal_crps": qrw},
            "GARCH(1,1)": {"mean_marginal_crps": rival},
            "GBM": {"mean_marginal_crps": rival * 1.5},
        },
    }


def test_the_gap_is_rebuilt_as_a_fraction_of_the_best_rival() -> None:
    rebuilt = relative_gaps([_window(qrw=0.11, rival=0.10)])

    assert rebuilt[0]["qrw_crps_gap_relative"] == pytest.approx(0.10)


def test_the_same_relative_shortfall_reads_the_same_at_any_window_scale() -> None:
    """A 10% shortfall is a 10% shortfall whether the window scores 0.05 or 5.0.

    The stored absolute gap does not say that -- it differs by 100x here -- and
    correlating it was partly measuring how big the window was.
    """
    small = relative_gaps([_window(qrw=0.055, rival=0.05)])[0]
    large = relative_gaps([_window(qrw=5.5, rival=5.0)])[0]

    assert small["qrw_crps_gap_relative"] == pytest.approx(
        large["qrw_crps_gap_relative"]
    )
    assert small["qrw_crps_gap"] != pytest.approx(large["qrw_crps_gap"])


def test_a_window_the_qrw_wins_reads_negative() -> None:
    assert relative_gaps([_window(qrw=0.09, rival=0.10)])[0][
        "qrw_crps_gap_relative"
    ] < 0


def test_the_qrw_is_never_its_own_rival() -> None:
    """Comparing the QRW against itself would put every gap at zero or above."""
    rebuilt = relative_gaps([_window(qrw=0.05, rival=0.20)])

    assert rebuilt[0]["qrw_crps_gap_relative"] == pytest.approx(0.05 / 0.20 - 1.0)


def test_a_window_missing_the_qrw_score_is_dropped_not_guessed() -> None:
    broken = _window(qrw=0.1, rival=0.1)
    del broken["scores"]["QRW Adaptive"]

    assert relative_gaps([broken]) == []


# ---------------------------------------------------------------------------
# One-sided testing in the asserted direction
# ---------------------------------------------------------------------------


def test_a_correlation_in_the_claimed_direction_gets_half_the_two_sided_p() -> None:
    assert one_sided_p(0.3, 0.20) == pytest.approx(0.10)


def test_a_correlation_the_wrong_way_is_penalised_not_rewarded() -> None:
    """Halving here would turn evidence against the claim into evidence for it."""
    assert one_sided_p(-0.3, 0.20) == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Pooling across assets
# ---------------------------------------------------------------------------


def test_three_strong_agreeing_assets_support_the_claim() -> None:
    result = combine_assets([_asset(0.6, 0.001), _asset(0.5, 0.004), _asset(0.55, 0.002)])

    assert result["all_in_claimed_direction"] is True
    assert result["supports_claim"] is True


def test_three_weak_agreeing_assets_do_not_reach_the_bar() -> None:
    """What the actual runs produced: same direction everywhere, none decisive."""
    result = combine_assets([_asset(0.19, 0.228), _asset(0.06, 0.725), _asset(0.20, 0.215)])

    assert result["all_in_claimed_direction"] is True
    assert result["supports_claim"] is False


def test_one_asset_pointing_the_other_way_blocks_support() -> None:
    """A claim that reverses on an asset is not a claim about the model."""
    result = combine_assets([_asset(0.7, 0.0001), _asset(0.7, 0.0001), _asset(-0.5, 0.01)])

    assert result["fisher_p"] < 0.05  # pooling alone would have said yes
    assert result["all_in_claimed_direction"] is False
    assert result["supports_claim"] is False


def test_the_two_poolings_must_agree_before_the_claim_stands() -> None:
    """Fisher follows the smallest p-value; Stouffer weighs them all.

    These are the measured numbers, and they straddle alpha: Stouffer clears
    it, Fisher does not. Requiring both stops the answer being chosen by which
    test gets quoted.
    """
    result = combine_assets([_asset(0.22, 0.169), _asset(0.05, 0.773), _asset(0.21, 0.187)])

    assert result["stouffer_p"] < 0.05
    assert result["fisher_p"] > 0.05
    assert result["supports_claim"] is False


def test_a_single_asset_is_not_pooled_into_a_verdict() -> None:
    result = combine_assets([_asset(0.6, 0.001)])

    assert result["assets_used"] == 1
    assert "fisher_p" not in result
    assert result.get("supports_claim") is None
