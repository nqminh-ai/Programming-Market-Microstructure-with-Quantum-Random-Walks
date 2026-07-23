"""Tests for the volatility claim in the CRPS study.

The report has carried "QRW wins quiet windows and loses volatile ones, so it
does not model volatility dynamics" as an interpretation read off five windows
per asset, with one acknowledged counter-example. These cover the machinery
that turns that into a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.marginal_crps_comparison import (
    volatility_relationship,
    window_volatility,
)


def _window(returns: np.ndarray, segments: np.ndarray | None = None) -> pd.DataFrame:
    price = 100.0 * np.exp(np.cumsum(np.r_[0.0, returns]))
    frame = pd.DataFrame({"price": price})
    frame["segment_id"] = (
        np.zeros(len(price), dtype=np.int32) if segments is None else segments
    )
    return frame


def test_volatility_recovers_the_dispersion_it_was_built_from() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 3e-5, 50_000)

    assert window_volatility(_window(returns)) == pytest.approx(3e-5, rel=0.02)


def test_a_gap_jump_is_not_counted_as_a_price_move() -> None:
    """The step across a segment break would otherwise dominate the estimate."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 1e-5, 2_000)
    returns[1_000] = 0.5  # the sort of jump a data gap leaves behind
    segments = np.zeros(2_001, dtype=np.int32)
    segments[1_001:] = 1

    with_break = window_volatility(_window(returns, segments))
    without_jump = window_volatility(_window(np.delete(returns, 1_000)))

    assert with_break == pytest.approx(without_jump, rel=0.05)


def test_too_short_a_window_reports_nan_rather_than_a_number() -> None:
    assert np.isnan(window_volatility(pd.DataFrame({"price": [100.0, 101.0]})))


def _rows(pairs: list[tuple[float, float]]) -> list[dict]:
    return [
        {"realised_volatility": v, "qrw_crps_gap": g, "window": i}
        for i, (v, g) in enumerate(pairs)
    ]


def test_a_clean_rising_relationship_supports_the_claim() -> None:
    """More volatility, further behind -- what the report asserts."""
    result = volatility_relationship(
        _rows([(1e-5, 0.01), (2e-5, 0.02), (3e-5, 0.03), (4e-5, 0.05), (5e-5, 0.09)])
    )

    assert result["spearman"] == pytest.approx(1.0)
    assert result["supports_claim"] is True


def test_no_relationship_does_not_support_the_claim() -> None:
    result = volatility_relationship(
        _rows([(1e-5, 0.05), (2e-5, 0.01), (3e-5, 0.04), (4e-5, 0.02), (5e-5, 0.03)])
    )

    assert result["supports_claim"] is False


def test_a_falling_relationship_does_not_support_it_either() -> None:
    """A strong correlation the wrong way is not evidence for the claim."""
    result = volatility_relationship(
        _rows([(1e-5, 0.09), (2e-5, 0.05), (3e-5, 0.03), (4e-5, 0.02), (5e-5, 0.01)])
    )

    assert result["spearman"] < 0
    assert result["supports_claim"] is False


def test_too_few_windows_reports_nothing_rather_than_a_correlation() -> None:
    """Five windows per asset is what made the original claim unfounded."""
    result = volatility_relationship(_rows([(1e-5, 0.01), (2e-5, 0.02)]))

    assert result["spearman"] is None
    assert result["supports_claim"] is None


def test_windows_with_no_volatility_estimate_are_dropped(_=None) -> None:
    rows = _rows([(1e-5, 0.01), (2e-5, 0.02), (3e-5, 0.03), (4e-5, 0.05)])
    rows.append({"realised_volatility": float("nan"), "qrw_crps_gap": 0.9, "window": 9})

    result = volatility_relationship(rows)

    assert result["windows_used"] == 4
    assert result["spearman"] is None  # 4 usable windows is below the floor


# ---------------------------------------------------------------------------
# Day-cluster windowing (report limitation #4)
# ---------------------------------------------------------------------------

from scripts.research.marginal_crps_comparison import day_cluster_boundaries  # noqa: E402

DAY_NS = 86_400_000_000_000


def _days(rows_per_day: list[int], start_day: int = 20_600) -> pd.DataFrame:
    """A frame whose rows carry real UTC-day structure, days of unequal size."""
    stamps: list[int] = []
    for offset, count in enumerate(rows_per_day):
        base = (start_day + offset) * DAY_NS
        stamps.extend(base + np.linspace(0, DAY_NS - 1, count, dtype=np.int64))
    return pd.DataFrame({"timestamp": np.array(stamps, dtype=np.int64)})


def test_every_boundary_lands_on_a_day_edge() -> None:
    """A window holding part of a day is the thinness this exists to fix."""
    frame = _days([100, 250, 80, 300, 120, 200])
    day = frame["timestamp"].to_numpy() // DAY_NS

    boundaries = day_cluster_boundaries(frame, windows=3)

    for edge in boundaries[1:-1]:
        assert day[edge] != day[edge - 1], "boundary fell inside a UTC day"
    assert boundaries[0] == 0
    assert boundaries[-1] == len(frame)


def test_windows_are_contiguous_and_cover_every_row() -> None:
    frame = _days([100, 250, 80, 300, 120, 200])

    boundaries = day_cluster_boundaries(frame, windows=3)

    assert len(boundaries) == 4
    assert list(boundaries) == sorted(boundaries)
    assert sum(
        boundaries[i + 1] - boundaries[i] for i in range(3)
    ) == len(frame)


def test_unequal_day_sizes_do_not_split_a_day() -> None:
    """Row-count splitting would cut the 5,000-row day in half; this must not."""
    frame = _days([50, 5_000, 50, 50])
    day = frame["timestamp"].to_numpy() // DAY_NS

    boundaries = day_cluster_boundaries(frame, windows=2)

    for edge in boundaries[1:-1]:
        assert day[edge] != day[edge - 1]


def test_asking_for_more_windows_than_days_is_refused() -> None:
    """Returning fewer silently would restate the thinness being fixed."""
    frame = _days([100, 100, 100])

    with pytest.raises(ValueError, match="cannot fill"):
        day_cluster_boundaries(frame, windows=5)


def test_one_window_per_day_is_allowed() -> None:
    frame = _days([100, 120, 90, 110])
    boundaries = day_cluster_boundaries(frame, windows=4)
    assert len(boundaries) == 5
