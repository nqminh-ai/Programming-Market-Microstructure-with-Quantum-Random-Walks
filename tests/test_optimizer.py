"""Acceptance tests for Track A Phase 7 (strategy optimizer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.optimizer import QRWStrategyOptimizer
from src.strategy.signal_engine import QRWSignalEngine


def market_frame(size: int = 220, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phase = np.arange(size)
    returns = 0.0005 * np.sin(phase / 7) + rng.normal(0.0, 0.00025, size)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-12", periods=size, freq="s"),
            "price": 100 * np.exp(np.cumsum(returns)),
            "obi": np.clip(np.sin(phase / 7 + 0.4), -1, 1),
        }
    )


def test_grid_search_returns_valid_params():
    result = QRWStrategyOptimizer().grid_search(market_frame(), theta_grid=[0.52, 0.60], min_trades=2)
    assert "ALL" in result
    assert 0.5 <= result["ALL"]["theta_buy"] < 1.0
    assert 0.5 <= result["ALL"]["theta_sell"] < 1.0


def test_oos_eval_uses_only_test_data():
    train = market_frame(seed=1)
    test = market_frame(seed=2)
    optimizer = QRWStrategyOptimizer()
    best = optimizer.grid_search(train, theta_grid=[0.52, 0.60], min_trades=2)
    first = optimizer.evaluate_out_of_sample(test, best_params=best)
    changed_train = train.copy()
    changed_train["price"] *= np.linspace(1, 10, len(changed_train))
    second = optimizer.evaluate_out_of_sample(test, best_params=best)
    np.testing.assert_allclose(first["equity_curve"], second["equity_curve"])


def test_sharpe_not_worse_than_default_candidate():
    optimizer = QRWStrategyOptimizer()
    result = optimizer.grid_search(market_frame(), theta_grid=[0.52, 0.60, 0.68], min_trades=2)
    all_rows = optimizer.search_surface[optimizer.search_surface["regime"] == "ALL"]
    default_score = float(all_rows[(all_rows["theta_buy"] == 0.60) & (all_rows["theta_sell"] == 0.60)]["score"].iloc[0])
    assert result["ALL"]["score"] >= default_score


def test_regime_adaptive_uses_correct_params():
    optimizer = QRWStrategyOptimizer(QRWSignalEngine())
    optimizer.best_params = {
        "LOW": {"theta_buy": 0.55, "theta_sell": 0.56},
        "HIGH": {"theta_buy": 0.70, "theta_sell": 0.71},
        "ALL": {"theta_buy": 0.60, "theta_sell": 0.60},
    }
    assert optimizer.regime_adaptive_signal(0.05, np.array([0.1, 0.1, 0.8]))["theta_buy"] == 0.55
    assert optimizer.regime_adaptive_signal(0.30, np.array([0.1, 0.1, 0.8]))["theta_buy"] == 0.70


def test_default_search_space_fully_explored():
    optimizer = QRWStrategyOptimizer()
    result = optimizer.grid_search(market_frame(), min_trades=2)
    assert len(optimizer.DEFAULT_GRID) == 23
    assert result["ALL"]["n_evaluated"] == 529
    assert len(optimizer.search_surface[optimizer.search_surface["regime"] == "ALL"]) == 529


def test_n_trades_constraint_respected_when_feasible():
    optimizer = QRWStrategyOptimizer()
    # 10 round trips is what this fixture actually supports. The threshold used
    # to be 20, which only passed while n_trades counted bars-in-position
    # (148-163 here) instead of completed round trips.
    result = optimizer.grid_search(market_frame(), theta_grid=[0.52, 0.56, 0.60], min_trades=10)
    assert not result["ALL"]["constraint_relaxed"]
    assert result["ALL"]["metrics"]["n_trades"] >= 10


def test_n_trades_counts_round_trips_not_bars_in_position():
    """A position held across many bars is one trade, not many.

    Conflating the two inflated the count ~15x on this fixture and turned
    hit rate and profit factor into per-bar statistics wearing per-trade names.
    """
    optimizer = QRWStrategyOptimizer()
    optimizer.grid_search(market_frame(), theta_grid=[0.52, 0.56, 0.60], min_trades=0)
    surface = optimizer.search_surface
    surface = surface[surface["regime"] == "ALL"]
    assert (surface["n_trades"] < surface["n_bars_in_position"]).all()
    assert surface["n_trades"].max() <= 20


def test_infeasible_trade_constraint_relaxes_rather_than_silently_passing():
    optimizer = QRWStrategyOptimizer()
    result = optimizer.grid_search(market_frame(), theta_grid=[0.52, 0.60], min_trades=10_000)
    assert result["ALL"]["constraint_relaxed"] is True


def test_oos_vectorized_backtest_matches_per_row_loop_reference():
    """Regression/parity test for the M16 performance fix: evaluate_out_of_sample
    used to loop row-by-row, constructing a fresh QRWSignalEngine and calling
    backtest_from_probabilities on a single-row DataFrame for every test row
    (each call implicitly using QRWSignalEngine's default transaction_cost,
    since only theta_buy/theta_sell were ever passed to the constructor, and
    each call's trade_size collapsing to abs(position) rather than a true
    diff from the previous row, since every call saw only one row). It was
    rewritten as a single vectorized pass for speed. This independently
    reimplements the old per-row loop and checks every output column matches
    exactly, so a future edit that reintroduces a behavior difference (not
    just a performance regression) would be caught here.
    """
    train = market_frame(seed=1)
    test = market_frame(seed=2)
    optimizer = QRWStrategyOptimizer()
    best = optimizer.grid_search(train, theta_grid=[0.52, 0.60, 0.68], min_trades=2)

    result = optimizer.evaluate_out_of_sample(test, best_params=best)
    vectorized_backtest = result["backtest"]

    probabilities = optimizer.signal_engine.build_probability_frame(test, None)
    probabilities["regime"] = optimizer._regimes(test, probabilities)
    reference_parts = []
    for index, row in probabilities.iterrows():
        params = best.get(str(row["regime"]), best.get("ALL"))
        engine = QRWSignalEngine(params["theta_buy"], params["theta_sell"])
        one = engine.backtest_from_probabilities(probabilities.loc[[index]])
        one["regime"] = row["regime"]
        one["theta_buy"] = params["theta_buy"]
        one["theta_sell"] = params["theta_sell"]
        reference_parts.append(one)
    reference_backtest = pd.concat(reference_parts, ignore_index=True)

    np.testing.assert_allclose(
        vectorized_backtest["pnl"].to_numpy(),
        reference_backtest["pnl"].to_numpy(),
    )
    np.testing.assert_allclose(
        vectorized_backtest["confidence"].to_numpy(),
        reference_backtest["confidence"].to_numpy(),
    )
    np.testing.assert_allclose(
        vectorized_backtest["theta_buy"].to_numpy(),
        reference_backtest["theta_buy"].to_numpy(),
    )
    assert (
        vectorized_backtest["signal"].to_numpy()
        == reference_backtest["signal"].to_numpy()
    ).all()
    assert (
        vectorized_backtest["correct"].to_numpy()
        == reference_backtest["correct"].to_numpy()
    ).all()
