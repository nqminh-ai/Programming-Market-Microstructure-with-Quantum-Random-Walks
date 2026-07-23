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

    def _asset_report_dir(symbol: str, kind: str, *, create: bool = False):
        path = tmp_path / "reports" / symbol.lower() / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(ds, "asset_data_dir", _asset_data_dir)
    monkeypatch.setattr(ds, "asset_report_dir", _asset_report_dir)
    return tmp_path


def _feature_day(root, symbol: str, day: str, columns: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = root / symbol.lower() / "features"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(columns), directory / f"features_{symbol}_{day}.parquet"
    )


def _metadata(root, symbol: str, day: str, record: dict) -> None:
    import json

    directory = root / "reports" / symbol.lower() / "feature_metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"feature_metadata_{symbol}_{day}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


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


def test_metadata_that_no_longer_describes_its_feature_file_is_flagged(assets) -> None:
    """Dropping `day` rewrote 121 feature files but not the metadata beside them.

    Nothing reads the column list, which is exactly why it stayed wrong.
    """
    _feature_day(assets, "BTCUSDT", "2026-07-01", {"timestamp": [1, 2], "price": [1.0, 2.0]})
    _metadata(
        assets,
        "BTCUSDT",
        "2026-07-01",
        {"rows": 2, "columns": ["timestamp", "price", "day"]},
    )

    assert ds.asset_status("BTCUSDT")["metadata_drift"] == ["2026-07-01"]


def test_metadata_matching_its_feature_file_is_not_flagged(assets) -> None:
    _feature_day(assets, "BTCUSDT", "2026-07-01", {"timestamp": [1, 2], "price": [1.0, 2.0]})
    _metadata(
        assets, "BTCUSDT", "2026-07-01", {"rows": 2, "columns": ["timestamp", "price"]}
    )

    assert ds.asset_status("BTCUSDT")["metadata_drift"] == []


def test_a_row_count_disagreement_is_also_drift(assets) -> None:
    """A stale row count means something other than a column drop happened."""
    _feature_day(assets, "BTCUSDT", "2026-07-01", {"timestamp": [1, 2], "price": [1.0, 2.0]})
    _metadata(
        assets, "BTCUSDT", "2026-07-01", {"rows": 99, "columns": ["timestamp", "price"]}
    )

    assert ds.asset_status("BTCUSDT")["metadata_drift"] == ["2026-07-01"]


def test_a_day_without_metadata_is_not_reported_as_drift(assets) -> None:
    _feature_day(assets, "BTCUSDT", "2026-07-01", {"timestamp": [1], "price": [1.0]})
    assert ds.asset_status("BTCUSDT")["metadata_drift"] == []


def test_missing_asset_reports_empty_rather_than_failing(assets) -> None:
    status = ds.asset_status("BTCUSDT")
    assert len(status["raw"].days) == 0
    assert status["raw"].span == "-"
