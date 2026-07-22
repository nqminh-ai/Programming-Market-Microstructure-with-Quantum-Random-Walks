"""Tests for moving downloaded day files into an asset's raw directory."""

from __future__ import annotations

import pytest

from scripts.operations import ingest_downloads as ing


@pytest.fixture
def asset_root(tmp_path, monkeypatch):
    def _asset_data_dir(symbol: str, kind: str, *, create: bool = False):
        path = tmp_path / "assets" / symbol.lower() / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(ing, "asset_data_dir", _asset_data_dir)
    return tmp_path


def _download(root, symbol_lower: str, day: str) -> None:
    directory = root / "assets" / symbol_lower
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{day}.parquet").write_bytes(b"payload")


def test_downloads_are_renamed_into_raw_with_an_uppercase_symbol(asset_root) -> None:
    """Lower-cased names match the pipeline glob on Windows and nothing on Linux."""
    _download(asset_root, "btcusdt", "2026-07-06")
    result = ing.ingest("BTCUSDT", asset_root / "assets")

    assert result["moved"] == ["tick_BTCUSDT_2026-07-06.parquet"]
    raw = asset_root / "assets" / "btcusdt" / "raw"
    assert (raw / "tick_BTCUSDT_2026-07-06.parquet").read_bytes() == b"payload"
    # The download must not be left behind to be ingested twice.
    assert not (asset_root / "assets" / "btcusdt" / "2026-07-06.parquet").exists()


def test_existing_raw_file_is_never_overwritten(asset_root) -> None:
    _download(asset_root, "btcusdt", "2026-07-06")
    raw = asset_root / "assets" / "btcusdt" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "tick_BTCUSDT_2026-07-06.parquet").write_bytes(b"already processed")

    result = ing.ingest("BTCUSDT", asset_root / "assets")

    assert result["moved"] == []
    assert result["skipped"] == ["2026-07-06.parquet"]
    assert (raw / "tick_BTCUSDT_2026-07-06.parquet").read_bytes() == b"already processed"


def test_only_date_named_files_are_ingested(asset_root) -> None:
    """The asset root also holds combined stores that are not day downloads."""
    _download(asset_root, "btcusdt", "2026-07-06")
    (asset_root / "assets" / "btcusdt" / "features_BTCUSDT_31d.parquet").write_bytes(b"x")
    (asset_root / "assets" / "btcusdt" / "notes.parquet").write_bytes(b"x")

    result = ing.ingest("BTCUSDT", asset_root / "assets")

    assert result["moved"] == ["tick_BTCUSDT_2026-07-06.parquet"]
    assert (asset_root / "assets" / "btcusdt" / "features_BTCUSDT_31d.parquet").exists()
    assert (asset_root / "assets" / "btcusdt" / "notes.parquet").exists()


def test_dry_run_reports_without_moving(asset_root) -> None:
    _download(asset_root, "ethusdt", "2026-07-06")
    result = ing.ingest("ETHUSDT", asset_root / "assets", dry_run=True)

    assert result["moved"] == ["tick_ETHUSDT_2026-07-06.parquet"]
    assert (asset_root / "assets" / "ethusdt" / "2026-07-06.parquet").exists()


def test_missing_asset_directory_is_not_an_error(asset_root) -> None:
    assert ing.ingest("BNBUSDT", asset_root / "assets") == {"moved": [], "skipped": []}
