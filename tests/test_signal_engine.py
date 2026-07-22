"""Acceptance tests for Track A Phase 5 (signal backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.signal_engine import QRWSignalEngine


def market_frame(size: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    returns = 0.00015 * np.sin(np.arange(size) / 8) + rng.normal(0, 0.0003, size)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-12", periods=size, freq="s"),
            "price": 100 * np.exp(np.cumsum(returns)),
            "obi": np.clip(np.sin(np.arange(size) / 10), -1, 1),
        }
    )


def test_signal_one_of_three_values():
    engine = QRWSignalEngine()
    for state in (np.array([0.1, 0.1, 0.8]), np.array([0.8, 0.1, 0.1]), np.array([0.4, 0.2, 0.4])):
        assert engine.amplitude_to_signal(state)["signal"] in engine.VALID_SIGNALS


def test_confidence_in_range():
    engine = QRWSignalEngine()
    for state in (np.array([0.01, 0.01, 0.98]), np.array([0.4, 0.2, 0.4])):
        assert 0.0 <= engine.amplitude_to_signal(state)["confidence"] <= 0.5


def test_probabilities_sum_to_one():
    result = QRWSignalEngine().amplitude_to_signal(np.array([2.0, 1.0, 7.0]))
    assert result["p_up"] + result["p_down"] + result["p_flat"] == pytest.approx(1.0)


def test_buy_signal_when_p_up_high():
    assert QRWSignalEngine().amplitude_to_signal(np.array([0.1, 0.1, 0.8]))["signal"] == "BUY"


def test_sell_signal_when_p_down_high():
    assert QRWSignalEngine().amplitude_to_signal(np.array([0.8, 0.1, 0.1]))["signal"] == "SELL"


def test_hold_signal_when_uncertain():
    assert QRWSignalEngine().amplitude_to_signal(np.array([0.4, 0.2, 0.4]))["signal"] == "HOLD"


def test_backtest_has_no_lookahead():
    data = market_frame()
    baseline = QRWSignalEngine().backtest_signals(data)
    mutated = data.copy()
    mutated.loc[90:, "price"] *= np.linspace(1.0, 4.0, len(mutated) - 90)
    mutated.loc[90:, "obi"] = -1.0
    changed = QRWSignalEngine().backtest_signals(mutated)
    columns = ["p_up", "p_down", "p_flat", "signal"]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["row"] < 90, columns].reset_index(drop=True),
        changed.loc[changed["row"] < 90, columns].reset_index(drop=True),
    )


def test_metrics_computed_with_expected_keys():
    backtest = QRWSignalEngine(0.52, 0.52).backtest_signals(market_frame())
    metrics = QRWSignalEngine.compute_signal_metrics(backtest)
    assert {
        "hit_rate",
        "profit_factor",
        "net_pnl",
        "max_drawdown",
        "n_trades",
        "n_bars_in_position",
        "avg_confidence",
        "t_stat",
        "sharpe_annualised",
    } <= set(metrics)
    # "sharpe" was mean/std*sqrt(n_bars) -- a t-statistic that grows without
    # bound with sample size. It must not come back under that name.
    assert "sharpe" not in metrics
    assert metrics["n_trades"] > 0


def test_build_probability_frame_bounded_window_matches_full_prefix_reference():
    """Regression/parity test for the M16 performance fix: build_probability_frame
    used to pass the entire growing causal prefix (tick_df.iloc[:index+1]) to
    _causal_directional_probability on every iteration, even though that
    function only ever reads the trailing 31 rows (a 30-return price diff
    plus a 30-row OBI tail) -- an O(n^2) cost across the whole walk-forward
    loop. It was bounded to a trailing window for speed. This independently
    reimplements the old full-prefix loop and checks every output column
    matches exactly, so a future edit that changes the lookback bound
    without updating _CAUSAL_LOOKBACK_ROWS would be caught here.
    """
    data = market_frame()
    engine = QRWSignalEngine()

    bounded = engine.build_probability_frame(data, warmup=30)

    prices = data["price"].to_numpy(dtype=float)
    reference_rows = []
    for index in range(30, len(data) - 1):
        history = data.iloc[: index + 1]
        state = engine._causal_directional_probability(history)
        probability = engine._probability(state)
        center = len(probability) // 2
        reference_rows.append(
            {
                "row": index,
                "timestamp": data.iloc[index].get("timestamp", data.index[index]),
                "price": float(prices[index]),
                "p_up": float(probability[center + 1 :].sum()),
                "p_down": float(probability[:center].sum()),
                "p_flat": float(probability[center]),
                "ret_1step": float(prices[index + 1] / prices[index] - 1.0),
            }
        )
    reference = pd.DataFrame(reference_rows)

    pd.testing.assert_frame_equal(bounded, reference)


def test_latest_signal_uses_current_in_memory_buffer():
    data = market_frame(45)
    result = QRWSignalEngine().latest_signal(data, min_history=30)

    assert result["timestamp"] == data.iloc[-1]["timestamp"]
    assert result["price"] == pytest.approx(data.iloc[-1]["price"])
    assert result["observations"] == 45
    assert result["signal"] in QRWSignalEngine.VALID_SIGNALS
    assert result["p_up"] + result["p_down"] + result["p_flat"] == pytest.approx(1.0)
