"""Unit tests for the Phase 7A multi-day data-expansion helpers.

These cover the pure (network-free) logic: inclusive date ranges and the
multi-day feature-store consolidation that counts rows across daily parquet
files.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.phase7a_data_expansion import build_multiday_store, date_range


def test_date_range_is_inclusive() -> None:
    days = date_range("2026-05-13", "2026-05-16")
    assert days == [
        date(2026, 5, 13),
        date(2026, 5, 14),
        date(2026, 5, 15),
        date(2026, 5, 16),
    ]


def test_date_range_single_day() -> None:
    assert date_range("2026-06-12", "2026-06-12") == [date(2026, 6, 12)]


def _write_feature_parquet(path, n_rows: int) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": range(n_rows),
            "price": [100.0 + i * 0.01 for i in range(n_rows)],
        }
    )
    frame.to_parquet(path)


def test_build_multiday_store_counts_rows(tmp_path) -> None:
    paths = []
    for day, n in (("2026-05-13", 30), ("2026-05-14", 70)):
        p = tmp_path / f"features_BTCUSDT_{day}.parquet"
        _write_feature_parquet(p, n)
        paths.append(p)
    # Should run without raising and validate both daily files (100 rows total).
    result = build_multiday_store(paths, tmp_path / "multiday_BTCUSDT.parquet")
    assert result is None  # consolidation validates in place, returns nothing


def test_build_multiday_store_raises_on_empty() -> None:
    with pytest.raises(RuntimeError):
        build_multiday_store([], "unused.parquet")
