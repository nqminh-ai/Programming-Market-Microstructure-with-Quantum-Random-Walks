"""Acceptance tests for Track A Phase 1 (volatility backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.qrw_core import QuantumRandomWalk
from src.models.volatility_forecaster import QRWVolatilityForecaster


def tick_frame(size: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    returns = rng.normal(0.0, 0.0008, size)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-12", periods=size, freq="s"),
            "price": 100.0 * np.exp(np.cumsum(returns)),
            "obi": rng.uniform(-1.0, 1.0, size),
        }
    )


def forecaster(positions: int = 63) -> QRWVolatilityForecaster:
    return QRWVolatilityForecaster(QuantumRandomWalk(n_positions=positions))


def test_variance_positive_and_normalized():
    model = forecaster()
    probability = np.array([0.0, 2.0, 1.0, 0.0, 0.0])
    assert model.compute_distribution_variance(probability) >= 0.0
    with pytest.raises(ValueError):
        model.compute_distribution_variance(np.zeros(5))


def test_qrw_vol_shape_is_wider_than_concentrated_state():
    model = forecaster()
    concentrated = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    spread = np.ones(5) / 5
    assert model.compute_distribution_variance(spread) > model.compute_distribution_variance(concentrated)


def test_rolling_window_is_strictly_causal():
    data = tick_frame()
    baseline = forecaster().rolling_vol_forecast(data, window=10)
    mutated = data.copy()
    mutated.loc[50:, "price"] *= np.linspace(1.0, 3.0, len(mutated) - 50)
    changed = forecaster().rolling_vol_forecast(mutated, window=10)
    pd.testing.assert_series_equal(baseline.loc[:50], changed.loc[:50])


def test_output_shape_matches_input_after_warmup():
    data = tick_frame(55)
    result = forecaster().rolling_vol_forecast(data, window=15)
    assert len(result) == 40
    assert result.index.equals(data.index[15:])
    assert np.isfinite(result).all()


def test_annualization_and_regimes_are_consistent():
    model = forecaster()
    assert model.ANNUALIZE == pytest.approx(np.sqrt(252 * 6.5 * 3600))
    assert [model.vol_regime(value) for value in (0.05, 0.15, 0.30)] == ["LOW", "MID", "HIGH"]


def test_garch_comparison_runs_and_is_aligned():
    comparison = forecaster().compare_with_garch(tick_frame(90), window=10)
    assert list(comparison.columns) == ["vol_qrw", "vol_garch", "vol_realized"]
    assert len(comparison) > 20
    assert np.isfinite(comparison.to_numpy()).all()


def test_supported_window_sizes_produce_metrics():
    model = forecaster(positions=63)
    metrics = model.benchmark_metrics_by_window(tick_frame(90), [5, 15, 30])
    assert set(metrics) == {"window_5", "window_15", "window_30"}
    assert all("mae_qrw" in row and "mae_garch" in row for row in metrics.values())


def test_empty_or_invalid_input_is_handled():
    empty = tick_frame(5)
    assert forecaster().rolling_vol_forecast(empty, window=5).empty
    with pytest.raises(ValueError):
        forecaster().rolling_vol_forecast(pd.DataFrame({"x": [1, 2]}), window=2)


def test_vol_forecast_depends_on_obi_not_only_on_window():
    """Regression test for audit finding C3.

    reset()+run(window) with a fixed coin is deterministic given ``window``
    alone, so vol_qrw used to be a pure rescaling of vol_realized for the
    same window. The coin must now be refreshed from trailing OBI/direction
    before each run, so two windows with identical size but different OBI
    signal must not collapse to the same shape multiplier.
    """
    size = 60
    base_returns = np.random.default_rng(2026).normal(0.0, 0.0008, size)
    price = 100.0 * np.exp(np.cumsum(base_returns))
    timestamp = pd.date_range("2026-06-12", periods=size, freq="s")

    calm = pd.DataFrame({"timestamp": timestamp, "price": price, "obi": np.zeros(size)})
    trending = pd.DataFrame(
        {"timestamp": timestamp, "price": price, "obi": np.full(size, 0.95)}
    )

    calm_result = forecaster().latest_vol_forecast(calm, window=15)
    trending_result = forecaster().latest_vol_forecast(trending, window=15)

    assert calm_result["vol_qrw"] != pytest.approx(trending_result["vol_qrw"])


def test_refresh_market_coin_requires_set_coin_support():
    class _NoSetCoinQRW:
        def reset(self) -> None:
            pass

        def run(self, window: int) -> np.ndarray:
            return np.ones(1)

    model = QRWVolatilityForecaster(_NoSetCoinQRW())
    with pytest.raises(RuntimeError, match="set_coin"):
        model.rolling_vol_forecast(tick_frame(40), window=15)


def test_latest_forecast_accepts_in_memory_ring_buffer():
    data = tick_frame(40)
    result = forecaster().latest_vol_forecast(data, window=15)

    assert result["timestamp"] == data.iloc[-1]["timestamp"]
    assert result["observations"] == 40
    assert result["window"] == 15
    assert result["vol_qrw"] >= 0
    assert result["vol_realized"] >= 0
    assert result["regime"] in {"LOW", "MID", "HIGH"}
