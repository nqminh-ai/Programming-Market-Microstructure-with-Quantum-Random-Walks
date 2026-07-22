"""Tests for the horizon-label edge study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import horizon_label_baselines as hlb


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
