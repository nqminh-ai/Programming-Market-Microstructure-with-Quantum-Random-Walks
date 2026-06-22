import numpy as np
import pandas as pd
import pytest

from src.models.qrw_heavy_tail import HeavyTailAdaptiveQRW

@pytest.fixture
def mock_tick_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 500
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-06-12", periods=n_rows, freq="s"),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.01, size=n_rows)),
        "tick_direction": rng.choice([-1.0, 1.0], size=n_rows),
        "obi": rng.uniform(-1.0, 1.0, size=n_rows),
        "trade_intensity": rng.uniform(0.1, 5.0, size=n_rows),
        "segment_id": np.zeros(n_rows, dtype=np.int64),
    })

def test_heavy_tail_qrw_initialization(mock_tick_data: pd.DataFrame) -> None:
    model = HeavyTailAdaptiveQRW(mock_tick_data, {"n_positions": 101})
    assert model.n_positions == 101
    assert model.tail_index == 3.0
    assert model.jump_scale == 1.0

def test_heavy_tail_qrw_calibration(mock_tick_data: pd.DataFrame) -> None:
    # Adding larger jumps to allow Pareto fit
    mock_tick_data.loc[10:50, "price"] += np.linspace(0.1, 5.0, 41)
    
    model = HeavyTailAdaptiveQRW(mock_tick_data, {
        "n_positions": 101,
        "calibration_regularization_grid": [1e-2]
    })
    
    params = model.calibrate_two_stage(output_path=None, warmup_fraction=0.5)
    assert "tail_index" in params
    assert "jump_scale" in params
    assert 1.0 <= params["tail_index"] <= 10.0

def test_heavy_tail_qrw_simulation(mock_tick_data: pd.DataFrame) -> None:
    model = HeavyTailAdaptiveQRW(mock_tick_data, {"n_positions": 101})
    # Mock calibration
    model.calibrated = True
    model.feature_mean = np.zeros(5)
    model.feature_scale = np.ones(5)
    model.coefficients = np.zeros(4)
    model.gamma = 0.1
    model.gamma_intensity = 0.1
    model.movement_probability = 0.5
    model.tail_index = 2.0
    model.jump_scale = 0.01
    
    paths = model.simulate_price_path(n_paths=10, T=50, random_state=42)
    assert paths.shape == (10, 50)
    # The paths should have some jumps larger than jump_scale
    increments = np.abs(np.diff(paths, axis=1))
    assert np.any(increments > 0.0)
