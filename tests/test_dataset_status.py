"""Tests for the pipeline-coverage report."""

from __future__ import annotations

import pytest

from scripts.operations import dataset_status as ds


@pytest.fixture
def assets(tmp_path, monkeypatch):
    def _asset_data_dir(symbol: str, kind: str, *, create: bool = False):
        path = tmp_path / symbol.lower() / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(ds, "asset_data_dir", _asset_data_dir)
    return tmp_path


def _write(root, symbol: str, kind: str, name: str) -> None:
    directory = root / symbol.lower() / kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"x")


def test_days_present_upstream_but_missing_downstream_are_listed(assets) -> None:
    """A raw day that was never processed produces no error anywhere else."""
    for day in ("2026-07-01", "2026-07-02", "2026-07-03"):
        _write(assets, "BTCUSDT", "raw", f"tick_BTCUSDT_{day}.parquet")
    _write(assets, "BTCUSDT", "processed", "tick_processed_BTCUSDT_2026-07-01.parquet")

    status = ds.asset_status("BTCUSDT")

    assert len(status["raw"].days) == 3
    assert status["unprocessed"] == ["2026-07-02", "2026-07-03"]


def test_processed_days_without_features_are_listed(assets) -> None:
    _write(assets, "ETHUSDT", "processed", "tick_processed_ETHUSDT_2026-07-01.parquet")
    _write(assets, "ETHUSDT", "processed", "tick_processed_ETHUSDT_2026-07-02.parquet")
    _write(assets, "ETHUSDT", "features", "features_ETHUSDT_2026-07-01.parquet")

    assert ds.asset_status("ETHUSDT")["no_features"] == ["2026-07-02"]


def test_two_raw_files_for_one_day_are_counted_as_duplicates(assets) -> None:
    """BTCUSDT carried both .csv.gz and .parquet for seven days."""
    _write(assets, "BTCUSDT", "raw", "tick_BTCUSDT_2026-06-06.parquet")
    _write(assets, "BTCUSDT", "raw", "tick_BTCUSDT_2026-06-06.csv.gz")

    status = ds.asset_status("BTCUSDT")

    assert len(status["raw"].days) == 1
    assert status["duplicate_raw"] == 1


def test_combined_stores_are_never_counted_as_days(assets) -> None:
    """Counting an aggregate as a day is how combine_features doubles its input."""
    _write(assets, "BNBUSDT", "features", "features_BNBUSDT_2026-07-01.parquet")
    _write(assets, "BNBUSDT", "features", "features_BNBUSDT_multiday.parquet")

    status = ds.asset_status("BNBUSDT")

    assert status["features"].days == {"2026-07-01"}
    assert [item["name"] for item in status["aggregates"]] == [
        "features_BNBUSDT_multiday.parquet"
    ]


def test_unreadable_aggregate_is_reported_not_raised(assets) -> None:
    """A half-written store must not stop the whole report."""
    _write(assets, "BNBUSDT", "features", "features_BNBUSDT_multiday.parquet")
    assert ds.asset_status("BNBUSDT")["aggregates"][0]["rows"] == -1


def test_missing_asset_reports_empty_rather_than_failing(assets) -> None:
    status = ds.asset_status("BTCUSDT")
    assert len(status["raw"].days) == 0
    assert status["raw"].span == "-"
