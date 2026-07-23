"""Tests for the trading-horizon feasibility analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import horizon_feasibility as hf


def _frame(n: int = 5_000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-5, n)))
    return pd.DataFrame(
        {
            # Nanosecond epochs, 50ms apart, as the Binance feeds arrive.
            "timestamp": (np.arange(n) * 50_000_000 + 1_783_099_484_231_822_000),
            "price": price,
            "mid_price": price * (1.0 + rng.normal(0.0, 2e-6, n)),
            "segment_id": np.zeros(n, dtype=np.int32),
        }
    )


def test_seconds_per_tick_reads_nanosecond_epochs() -> None:
    """A 50ms nanosecond feed must not be read as microseconds.

    The span-based inference this replaced accepted the microsecond reading as
    plausible and reported 28 days where the true horizon was 41 minutes.
    """
    assert hf.seconds_per_tick(_frame()) == pytest.approx(0.05, rel=1e-6)


def test_seconds_per_tick_agrees_across_timestamp_units() -> None:
    frame = _frame()
    nanoseconds = frame["timestamp"].to_numpy()
    reference = hf.seconds_per_tick(frame)
    for divisor in (1_000, 1_000_000, 1_000_000_000):
        scaled = frame.assign(timestamp=(nanoseconds // divisor))
        assert hf.seconds_per_tick(scaled) == pytest.approx(reference, rel=1e-3)


def test_expected_move_never_spans_a_segment_boundary() -> None:
    frame = _frame(1_000)
    frame.loc[500:, "segment_id"] = 1
    # A jump across the boundary would dominate the mean if it were included.
    frame.loc[500:, "price"] *= 5.0
    move, pairs = hf.expected_absolute_move(frame, horizon=10)
    assert pairs < len(frame)
    assert move < 0.01


def test_taker_pays_the_spread_and_maker_earns_it() -> None:
    half_spread = 1e-4
    taker = hf.round_trip_cost(hf.FEE_SCENARIOS["taker_repo_5bps"], half_spread)
    maker = hf.round_trip_cost(hf.FEE_SCENARIOS["maker_futures_2bps"], half_spread)
    assert taker == pytest.approx(2 * 5e-4 + 2 * half_spread)
    assert maker == pytest.approx(2 * 2e-4 - 2 * half_spread)
    assert maker < taker


def test_breakeven_accuracy_exceeds_one_when_cost_dwarfs_the_move() -> None:
    """The one-tick case: no model, however good, can pay the toll."""
    accuracy = hf.breakeven_accuracy(cost=1e-3, expected_move=5.4e-7)
    assert accuracy > 1.0


def test_breakeven_accuracy_is_a_half_when_trading_is_free() -> None:
    assert hf.breakeven_accuracy(cost=0.0, expected_move=1e-3) == pytest.approx(0.5)


def test_analysis_marks_the_one_tick_horizon_untradable() -> None:
    analysis = hf.analyse(_frame(20_000), horizons=(1, 10, 1_000))
    one_tick = next(r for r in analysis["horizons"] if r["horizon_ticks"] == 1)
    for scenario in one_tick["scenarios"].values():
        assert scenario["tradable"] is False


def test_verdict_states_that_clearing_costs_is_not_sufficient() -> None:
    analysis = hf.analyse(_frame(20_000), horizons=(1, 1_000))
    verdict = hf.build_verdict(analysis, "TEST")
    assert "cần, không đủ" in verdict


def test_expected_move_is_unchanged_by_the_chunk_size() -> None:
    """Chunking bounds memory on a 200M-row store; it must not move the number."""
    frame = _frame(20_000)
    frame.loc[9_000:, "segment_id"] = 1
    frame.loc[13_000, "price"] = np.nan

    whole, pairs = hf.expected_absolute_move(frame, 100, chunk=len(frame))
    split, split_pairs = hf.expected_absolute_move(frame, 100, chunk=1_000)
    uneven, uneven_pairs = hf.expected_absolute_move(frame, 100, chunk=3_333)

    assert pairs == split_pairs == uneven_pairs > 0
    assert split == pytest.approx(whole, rel=1e-12)
    assert uneven == pytest.approx(whole, rel=1e-12)


def test_expected_move_never_pairs_across_a_segment_break() -> None:
    """A pair spanning a data gap is not a price move that could be traded."""
    frame = _frame(1_000)
    frame.loc[500:, "segment_id"] = 1

    _, pairs = hf.expected_absolute_move(frame, 100, chunk=64)

    # Anchors 400..499 would reach into segment 1 and must be excluded.
    assert pairs == 800


def test_loading_a_capped_frame_reads_only_what_the_cap_needs(tmp_path) -> None:
    """Reading the file and then slicing materialises every row to keep a few."""
    import pyarrow.parquet as pq

    path = tmp_path / "features_BTCUSDT_69d.parquet"
    pq.write_table(
        pa_table := __import__("pyarrow").Table.from_pandas(_frame(50_000)),
        path,
        row_group_size=1_000,
    )
    assert pa_table.num_rows == 50_000

    frame = hf._load(path, max_rows=2_500)

    assert len(frame) == 2_500
    assert frame["timestamp"].is_monotonic_increasing
    assert list(frame.columns) == [c for c in hf.NEEDED_COLUMNS]


def _bounce_frame(n: int = 200_000, spread: float = 0.0004, seed: int = 7) -> pd.DataFrame:
    """A random walk observed through a bid-ask bounce of known width.

    The efficient price drifts; each trade prints at the efficient price plus
    or minus half the spread depending on which side the taker hits. This is
    exactly the process Roll's estimator inverts.
    """
    rng = np.random.default_rng(seed)
    efficient = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 2e-6, n)))
    side = rng.choice([-1.0, 1.0], size=n)
    price = efficient * (1.0 + side * spread / 2.0)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 50_000_000 + 1_783_099_484_231_822_000,
            "price": price,
            "mid_price": efficient,
            "segment_id": np.zeros(n, dtype=np.int32),
        }
    )


def test_roll_recovers_a_known_bid_ask_spread() -> None:
    frame = _bounce_frame(spread=0.0004)

    half = hf.roll_half_spread(frame)

    # The true half-spread is 2bps; recovery within 5% is ample for the use.
    assert half == pytest.approx(0.0002, rel=0.05)


def test_roll_is_unchanged_by_the_chunk_size() -> None:
    frame = _bounce_frame(50_000, spread=0.0004)

    whole = hf.roll_half_spread(frame, chunk=len(frame))
    split = hf.roll_half_spread(frame, chunk=5_000)
    uneven = hf.roll_half_spread(frame, chunk=3_331)

    assert split == pytest.approx(whole, rel=1e-9)
    assert uneven == pytest.approx(whole, rel=1e-9)


def test_roll_returns_none_when_the_covariance_is_not_negative() -> None:
    """A trending series has no bounce to invert; inventing a spread is wrong."""
    n = 10_000
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(n) * 50_000_000 + 1_783_099_484_231_822_000,
            "price": 100.0 * np.exp(np.cumsum(np.full(n, 1e-6))),
            "mid_price": np.full(n, 100.0),
            "segment_id": np.zeros(n, dtype=np.int32),
        }
    )

    assert hf.roll_half_spread(frame) is None


def test_roll_never_pairs_price_changes_across_a_segment_break() -> None:
    frame = _bounce_frame(20_000, spread=0.0004)
    frame.loc[10_000:, "segment_id"] = 1
    frame.loc[10_000:, "price"] *= 1.5  # a jump a gap would produce

    half = hf.roll_half_spread(frame)

    # The 50% jump would swamp the estimate if it were differenced across.
    assert half == pytest.approx(0.0002, rel=0.10)


def test_the_vwap_deviation_is_not_reported_as_a_spread() -> None:
    """mid_price is a trailing VWAP, so |price - mid| is dispersion, not spread.

    In a real market the spread is far narrower than the drift accumulated over
    the 100 trades the VWAP averages, so the deviation is dominated by drift
    and overstates the spread badly -- by a factor of 33 on BTCUSDT. That is
    why the cost model no longer uses it.
    """
    frame = _bounce_frame(200_000, spread=2e-6)
    frame["mid_price"] = frame["price"].rolling(100, min_periods=1).mean().shift(1).bfill()

    deviation = hf.measure_half_spread(frame)
    roll = hf.roll_half_spread(frame)

    assert roll == pytest.approx(1e-6, rel=0.05), "Roll should recover the true half"
    assert deviation > 5 * roll, "the deviation is drift, and is the larger quantity"
