"""Tests for the reproducible Phase 3 overfitting audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.phase3_overfitting_audit import (
    fit_linear_market_probability,
    fit_linear_probability,
    market_events,
    moving_events,
    score,
)


def test_moving_events_excludes_zero_price_changes() -> None:
    frame = pd.DataFrame(
        {
            "price": [100.0, 100.0, 101.0, 105.0, 104.0],
            "obi": [-0.5, 0.2, 0.8, -0.9, 0.0],
            "segment_id": [0, 0, 0, 1, 1],
        }
    )

    obi, target = moving_events(frame)

    assert obi == pytest.approx([0.2, -0.9])
    assert target == pytest.approx([1.0, 0.0])


def test_linear_probability_recovers_directional_obi_signal() -> None:
    obi = np.linspace(-1.0, 1.0, 101)
    target = (obi[:-1] > 0.0).astype(np.float64)
    price = 100.0 + np.concatenate([[0.0], np.cumsum(2.0 * target - 1.0)])
    frame = pd.DataFrame({"price": price, "obi": obi})

    coefficients = fit_linear_probability(frame)
    prediction = np.clip(
        coefficients[0] + coefficients[1] * obi[:-1],
        0.0,
        1.0,
    )

    assert coefficients[1] > 0.0
    assert score(prediction, target)["brier"] < 0.1


def test_fair_linear_market_baseline_uses_tick_direction() -> None:
    count = 101
    obi = np.zeros(count)
    direction = np.where(np.arange(count) % 2, 1.0, -1.0)
    target = (direction[:-1] > 0.0).astype(np.float64)
    price = 100.0 + np.concatenate(
        [[0.0], np.cumsum(np.where(target > 0.0, 1.0, -1.0))]
    )
    frame = pd.DataFrame(
        {
            "price": price,
            "obi": obi,
            "tick_direction": direction,
        }
    )

    coefficients = fit_linear_market_probability(frame)
    event_obi, event_direction, event_target = market_events(frame)
    probability = np.clip(
        coefficients[0]
        + coefficients[1] * event_obi
        + coefficients[2] * event_direction,
        0.0,
        1.0,
    )

    assert coefficients[2] > 0.0
    assert score(probability, event_target)["brier"] < 1e-12
