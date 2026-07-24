"""Tests for the pre-trade feasibility screen.

The tool's value is that it cannot flatter a scenario: it reuses the report's
own cost model, so if it ever says "tradable" it is saying so on the same terms
§5e used to say "not". These cover the arithmetic, the three verdicts, and that
it stays consistent with the measured artifacts.
"""

from __future__ import annotations

import json

import pytest

from scripts.tools.tradeability import (
    IMPOSSIBLE,
    IMPLAUSIBLE,
    POSSIBLE,
    assess,
    classify,
    from_artifact,
)


def test_a_perfect_forecast_that_still_loses_is_flagged_impossible() -> None:
    """The §5e headline: BTC 1-tick, taker 5bps -- cost dwarfs the move."""
    a = assess(expected_move=0.62e-4, fee_bps_per_side=5.0, crosses_spread=True)

    assert a.breakeven_accuracy > 1.0
    assert a.verdict == IMPOSSIBLE
    assert a.cost_to_move == pytest.approx(10e-4 / 0.62e-4, rel=1e-6)


def test_the_break_even_formula_is_the_reports() -> None:
    """p* = 0.5 + cost / (2 * move)."""
    a = assess(expected_move=40e-4, fee_bps_per_side=1.0, crosses_spread=True)
    # cost = 2 bps = 2e-4; move = 40e-4; p* = 0.5 + 2e-4/(2*40e-4) = 0.525
    assert a.round_trip_cost == pytest.approx(2e-4)
    assert a.breakeven_accuracy == pytest.approx(0.525)


def test_a_reachable_break_even_below_the_ceiling_reads_possible() -> None:
    a = assess(expected_move=40e-4, fee_bps_per_side=1.0, crosses_spread=True,
               half_spread=0.05e-4, ceiling=0.60)

    assert a.breakeven_accuracy < 0.60
    assert a.verdict == POSSIBLE


def test_break_even_between_the_ceiling_and_one_reads_implausible() -> None:
    a = assess(expected_move=8e-4, fee_bps_per_side=2.0, crosses_spread=False,
               realised=-1.2e-4, ceiling=0.60)

    assert 0.60 < a.breakeven_accuracy < 1.0
    assert a.verdict == IMPLAUSIBLE


def test_a_maker_earning_a_negative_realised_spread_pays_rather_than_earns() -> None:
    """The self-refutation baked into the tool: resting orders pay ~1.2 bps."""
    earning = assess(expected_move=5e-4, fee_bps_per_side=2.0, crosses_spread=False,
                     realised=+1.2e-4)
    paying = assess(expected_move=5e-4, fee_bps_per_side=2.0, crosses_spread=False,
                    realised=-1.2e-4)

    # Paying more than earning means a higher cost and a higher break-even.
    assert paying.round_trip_cost > earning.round_trip_cost
    assert paying.breakeven_accuracy > earning.breakeven_accuracy


def test_classify_boundaries() -> None:
    assert classify(1.0, 0.60) == IMPOSSIBLE
    assert classify(0.999, 0.60) == IMPLAUSIBLE
    assert classify(0.60, 0.60) == POSSIBLE
    assert classify(0.601, 0.60) == IMPLAUSIBLE


def test_zero_or_negative_move_is_never_tradable() -> None:
    a = assess(expected_move=0.0, fee_bps_per_side=1.0, crosses_spread=True)

    assert a.verdict == IMPOSSIBLE
    assert a.breakeven_accuracy == float("inf")


# ---------------------------------------------------------------------------
# Reading the measured artifacts
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(asset: str) -> Path:
    return ROOT / "reports" / "research" / f"horizon_feasibility_{asset}.json"


@pytest.mark.parametrize("asset", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
def test_no_one_tick_scenario_is_reachable_on_any_asset(asset: str) -> None:
    """§5e: at the 1-tick horizon no fee scenario clears break-even on any asset.

    Not the same as every cell being 'impossible' -- BNB's larger tick means its
    zero-fee maker cell has a break-even below 100% (so 'implausible', not
    'impossible'). But 84% directional accuracy is still far past what markets
    show, so no 1-tick cell is ever 'possible'. The tool must not over-promise.
    """
    path = _artifact_path(asset)
    if not path.is_file():
        pytest.skip(f"no artifact for {asset}")

    one_tick = [r for r in from_artifact(path) if r.horizon_ticks == 1]

    assert one_tick, "artifact has no 1-tick horizon"
    assert all(r.assessment.verdict != POSSIBLE for r in one_tick)
    # The taker scenarios cannot even be reached by a perfect forecast.
    takers = [r for r in one_tick if r.scenario.startswith("taker")]
    assert all(r.assessment.verdict == IMPOSSIBLE for r in takers)


def test_the_tool_reproduces_the_artifacts_own_break_even_numbers() -> None:
    """Recomputed, not read off -- so a drift in the cost model would surface."""
    path = _artifact_path("BTCUSDT")
    if not path.is_file():
        pytest.skip("no BTC artifact")

    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = from_artifact(path)

    for horizon in audit["analysis"]["horizons"]:
        for name, scenario in horizon["scenarios"].items():
            row = next(
                r for r in rows
                if r.horizon_ticks == horizon["horizon_ticks"] and r.scenario == name
            )
            assert row.assessment.breakeven_accuracy == pytest.approx(
                scenario["breakeven_accuracy"], rel=1e-6
            )
