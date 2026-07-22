"""Tests for the horizon-label edge study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import horizon_label_baselines as hlb
from src.evaluation.directional_baselines import _lagged_direction


def _events_via_full_length_arrays(frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """The original formulation, kept as the reference for the gathered one.

    It materialises every intermediate at full length, which is exactly why the
    real script no longer does it; here the frames are small and it serves as
    an independent statement of what the features are supposed to be.
    """
    price = frame["price"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
    direction = frame["tick_direction"].to_numpy(dtype=np.float64)
    intensity = frame["trade_intensity"].to_numpy(dtype=np.float64)

    obi_change = np.zeros(len(frame), dtype=np.float64)
    obi_change[1:] = np.diff(obi)
    if "segment_id" in frame.columns:
        segment = frame["segment_id"].to_numpy(copy=False)
        obi_change[1:][segment[:-1] != segment[1:]] = 0.0
    else:
        segment = np.zeros(len(frame), dtype=np.int64)

    features = np.column_stack(
        [obi, direction, obi_change, np.abs(obi), np.log1p(np.maximum(intensity, 0.0))]
    )
    lagged = _lagged_direction(direction, lags=5)

    anchors = np.arange(0, len(frame) - horizon, horizon, dtype=np.int64)
    future = anchors + horizon
    usable = (
        (segment[anchors] == segment[future])
        & np.isfinite(price[anchors])
        & np.isfinite(price[future])
        & (price[anchors] > 0)
        & (price[future] > 0)
        & np.isfinite(features[anchors]).all(axis=1)
    )
    if "obi_valid" in frame.columns:
        usable &= frame["obi_valid"].to_numpy()[anchors].astype(bool)

    anchors, future = anchors[usable], future[usable]
    log_return = np.log(price[future] / price[anchors])
    moved = np.abs(log_return) > 1e-12
    anchors, log_return = anchors[moved], log_return[moved]

    return {
        "features": features[anchors],
        "lagged": lagged[anchors],
        "target": (log_return > 0.0).astype(np.float64),
        "log_return": log_return,
        "timestamp": frame["timestamp"].to_numpy()[anchors],
    }


def _frame(n: int = 150_000, seed: int = 2026, drift_from_flow: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Trade sign with the long memory real order flow shows.
    flow = np.empty(n)
    flow[0] = 1.0
    for i in range(1, n):
        flow[i] = flow[i - 1] if rng.random() < 0.97 else -flow[i - 1]
    step = rng.normal(0.0, 1e-5, n) + drift_from_flow * flow
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 50_000_000 + 1_783_099_484_231_822_000,
            "price": 100.0 * np.exp(np.cumsum(step)),
            "tick_direction": flow,
            "obi": np.clip(flow * 0.5 + rng.normal(0, 0.2, n), -1, 1),
            "obi_valid": np.ones(n, dtype=bool),
            "trade_intensity": np.abs(rng.normal(5.0, 1.0, n)),
            "segment_id": np.zeros(n, dtype=np.int32),
            "mid_price": 100.0 * np.exp(np.cumsum(step)),
        }
    )


def test_anchors_are_spaced_so_labels_never_overlap() -> None:
    """Overlapping labels share future returns and inflate the effective sample."""
    horizon = 1_000
    events = hlb.build_horizon_events(_frame(), horizon)
    gaps = np.diff(events["timestamp"].astype("int64"))
    # 50ms per tick, so consecutive anchors must be at least horizon*50ms apart.
    assert gaps.min() >= horizon * 50_000_000


def test_window_count_is_bounded_by_rows_over_horizon() -> None:
    frame = _frame(60_000)
    events = hlb.build_horizon_events(frame, 5_000)
    assert len(events["target"]) <= len(frame) // 5_000 + 1


def test_labels_never_span_a_segment_boundary() -> None:
    frame = _frame(20_000)
    frame.loc[10_000:, "segment_id"] = 1
    events = hlb.build_horizon_events(frame, 1_000)
    assert len(events["target"]) < len(frame) // 1_000


def test_majority_class_is_reported_as_a_baseline() -> None:
    """Long-horizon labels can be class-imbalanced; 50% is the wrong reference."""
    scored = hlb.evaluate_models(hlb.build_horizon_events(_frame(), 200), 0.70)
    assert "Majority class" in scored["models"]
    assert scored["majority_class_rate"] >= 0.5


def test_no_skill_on_a_driftless_random_walk() -> None:
    """With flow carrying no price impact, nothing should beat the majority rate."""
    events = hlb.build_horizon_events(_frame(drift_from_flow=0.0), 200)
    scored = hlb.evaluate_models(events, 0.70)
    fitted = {
        name: metrics["accuracy"]
        for name, metrics in scored["models"].items()
        if name != "Majority class"
    }
    assert max(fitted.values()) < scored["majority_class_rate"] + 0.10


def test_skill_appears_when_order_flow_actually_moves_the_price() -> None:
    """Sanity check the other direction, so the null result above means something.

    The horizon is 50 rather than 200 because the simulated flow flips every ~33
    ticks, so over a longer window its price impact largely cancels itself out.
    """
    events = hlb.build_horizon_events(_frame(drift_from_flow=3e-6), 50)
    scored = hlb.evaluate_models(events, 0.70)
    assert scored["models"]["Logistic L2 (5F)"]["accuracy"] > 0.60


def test_shuffling_training_labels_destroys_the_edge() -> None:
    """Guards the pipeline itself: a leak here would survive the shuffle."""
    events = hlb.build_horizon_events(_frame(drift_from_flow=3e-6), 50)
    split = int(len(events["target"]) * 0.70)
    rng = np.random.default_rng(7)
    shuffled = dict(events)
    target = events["target"].copy()
    target[:split] = rng.permutation(target[:split])
    shuffled["target"] = target
    scored = hlb.evaluate_models(shuffled, 0.70)
    assert scored["models"]["Logistic L2 (5F)"]["accuracy"] < 0.58


def test_net_edge_subtracts_the_round_trip_cost() -> None:
    analysis = hlb.analyse(_frame(drift_from_flow=3e-6), (200,), 0.70)
    row = analysis["horizons"][0]
    for name, net in row["net_edge_per_trade"].items():
        expected = (2 * row["best_accuracy"] - 1) * row["expected_abs_move"] - analysis[
            "round_trip_costs"
        ][name]
        assert net == pytest.approx(expected)


def test_too_few_windows_is_reported_not_silently_scored() -> None:
    analysis = hlb.analyse(_frame(20_000), (10_000,), 0.70)
    assert "skipped" in analysis["horizons"][0]


@pytest.mark.parametrize("horizon", [7, 50, 200, 1_000])
def test_gathering_only_anchor_rows_matches_the_full_length_formulation(horizon) -> None:
    """The memory rewrite must not move a single number.

    The frame carries the awkward cases on purpose: a segment break, a
    non-finite price, and an invalid OBI row, each of which the two code paths
    have to exclude identically.
    """
    frame = _frame(20_000, drift_from_flow=2e-6)
    frame.loc[9_000:, "segment_id"] = 1
    frame.loc[13_000, "price"] = np.nan
    frame.loc[15_000, "obi_valid"] = False
    frame.loc[17_000, "obi"] = np.nan

    gathered = hlb.build_horizon_events(frame, horizon)
    reference = _events_via_full_length_arrays(frame, horizon)

    assert len(gathered["target"]) == len(reference["target"]) > 0
    for key in ("features", "lagged", "target", "log_return", "timestamp"):
        np.testing.assert_array_equal(gathered[key], reference[key])


def test_obi_change_is_zero_at_the_first_row_and_across_a_segment_break() -> None:
    """Anchor 0 has no predecessor, and a break must not be differenced across."""
    frame = _frame(3_000)
    frame.loc[1_000:, "segment_id"] = 1
    frame["obi"] = np.linspace(-1.0, 1.0, len(frame))

    events = hlb.build_horizon_events(frame, 500)

    # Anchors 0, 500, 1000, 1500, 2000. Anchor 500 is dropped because its label
    # would span the break; anchor 1000 sits exactly on it.
    assert events["timestamp"].size == 4
    obi_change = events["features"][:, 2]
    assert obi_change[0] == 0.0, "row 0 has no predecessor to difference against"
    assert obi_change[1] == 0.0, "differenced across the segment break"
    step = 2.0 / (len(frame) - 1)
    assert obi_change[2:] == pytest.approx(step)


def test_clearing_breakeven_by_eye_is_not_reported_as_clearing_it() -> None:
    """A point estimate above a threshold is not evidence of being above it.

    On the 69-day BNB store the best model at h=50,000 scored 53.6% against a
    52.3% break-even -- but on 323 non-overlapping windows, which puts p at
    0.35 with an interval that still covers a coin flip.
    """
    analysis = hlb.analyse(_frame(drift_from_flow=3e-6), (200,), 0.70)
    row = analysis["horizons"][0]

    for name, significant in row["clears_breakeven_significant"].items():
        if not significant:
            continue
        assert row["clears_breakeven"][name], "significant but not above the threshold"
        assert row["clears_breakeven_p_value"][name] < 0.05

    low, high = row["best_accuracy_ci95"]
    assert low <= row["best_accuracy"] <= high


def test_a_borderline_result_is_named_in_the_verdict_not_dropped() -> None:
    """Silently omitting it invites rediscovery in the table as a finding."""
    analysis = hlb.analyse(_frame(drift_from_flow=3e-6), (200,), 0.70)
    row = analysis["horizons"][0]
    row["clears_breakeven"]["maker_futures_2bps"] = True
    row["clears_breakeven_significant"]["maker_futures_2bps"] = False
    row["clears_breakeven_p_value"]["maker_futures_2bps"] = 0.35

    verdict = hlb.build_verdict(analysis)

    assert "Không horizon nào vượt ngưỡng hoà vốn" in verdict
    assert "không qua được kiểm định" in verdict.replace("**", "")
    assert "p=0.350" in verdict


def test_a_significant_result_is_reported_as_one() -> None:
    analysis = hlb.analyse(_frame(drift_from_flow=3e-6), (200,), 0.70)
    row = analysis["horizons"][0]
    row["clears_breakeven"]["maker_futures_2bps"] = True
    row["clears_breakeven_significant"]["maker_futures_2bps"] = True
    row["clears_breakeven_p_value"]["maker_futures_2bps"] = 0.01

    verdict = hlb.build_verdict(analysis)

    assert "có ý nghĩa" in verdict
    assert "p=0.010" in verdict


def test_half_spread_is_unchanged_by_the_chunk_size() -> None:
    """Chunking exists only to bound memory; it must not move the estimate."""
    frame = _frame(20_000)
    frame["mid_price"] = frame["price"] * (1.0 + 2e-5)

    whole = hlb.measure_half_spread(frame, chunk=len(frame))
    split = hlb.measure_half_spread(frame, chunk=1_000)
    uneven = hlb.measure_half_spread(frame, chunk=7_777)

    # The deviation is normalised by mid, not by price, so it is 2e-5/(1+2e-5).
    assert whole == pytest.approx(2e-5 / (1 + 2e-5), rel=1e-9)
    assert split == pytest.approx(whole, rel=1e-12)
    assert uneven == pytest.approx(whole, rel=1e-12)


def test_half_spread_ignores_rows_a_chunk_boundary_could_hide() -> None:
    frame = _frame(5_000)
    frame["mid_price"] = frame["price"]
    frame.loc[999, "mid_price"] = 0.0  # invalid, and the last row of chunk 0
    frame.loc[1_000, "mid_price"] = np.nan  # invalid, and the first of chunk 1

    assert hlb.measure_half_spread(frame, chunk=1_000) == pytest.approx(0.0)
