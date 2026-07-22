"""Tests for the multi-day feature combiner.

Both behaviours here were added after the combiner damaged real data: once by
folding its own previous output back into a rebuild, and once by leaving a
1.2GB file with no footer at the destination path after being killed part-way.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _day(directory: Path, symbol: str, day: str, rows: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"features_{symbol}_{day}.parquet"
    pq.write_table(
        pa.table({"timestamp": list(range(rows)), "price": [1.0] * rows}), path
    )
    return path


def _run(input_dir: Path, output: Path, symbol: str = "BTCUSDT", days: int = 30):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.operations.combine_features",
            "--symbol", symbol,
            "--input-dir", str(input_dir),
            "--output", str(output),
            "--days", str(days),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_daily_files_are_combined(tmp_path) -> None:
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    _day(tmp_path, "BTCUSDT", "2026-05-14", 15)
    output = tmp_path / "features_BTCUSDT_combined.parquet"

    result = _run(tmp_path, output)

    assert result.returncode == 0, result.stdout + result.stderr
    assert pq.ParquetFile(output).metadata.num_rows == 25


def test_previous_aggregates_are_not_folded_back_in(tmp_path) -> None:
    """The glob also matches the combiner's own output living alongside the days."""
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    _day(tmp_path, "BTCUSDT", "2026-05-14", 15)
    # An earlier run's aggregate, matching features_BTCUSDT_*.parquet.
    pq.write_table(
        pa.table({"timestamp": list(range(25)), "price": [1.0] * 25}),
        tmp_path / "features_BTCUSDT_multiday.parquet",
    )
    output = tmp_path / "features_BTCUSDT_new.parquet"

    result = _run(tmp_path, output)

    assert pq.ParquetFile(output).metadata.num_rows == 25, "aggregate was counted twice"
    assert "Skipping 1 non-daily file" in result.stdout


def test_output_appears_only_after_a_verified_write(tmp_path) -> None:
    """No partial file may be left at the destination path."""
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    output = tmp_path / "features_BTCUSDT_combined.parquet"

    _run(tmp_path, output)

    assert output.exists()
    assert not (tmp_path / "features_BTCUSDT_combined.parquet.partial").exists()
    # Readable end to end, which a footer-less file is not.
    assert pq.read_table(output).num_rows == 10


def test_a_stale_partial_from_an_earlier_kill_is_discarded(tmp_path) -> None:
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    output = tmp_path / "features_BTCUSDT_combined.parquet"
    stale = tmp_path / "features_BTCUSDT_combined.parquet.partial"
    stale.write_bytes(b"truncated garbage from a killed run")

    result = _run(tmp_path, output)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale.exists()
    assert pq.read_table(output).num_rows == 10


def test_a_differing_schema_is_named_before_any_writing_starts(tmp_path) -> None:
    """Days fetched by different acquisition paths really do differ.

    The bulk parquet collector writes a `day` column and the csv.gz downloader
    does not, so a store spanning both used to die at file 32 of 69 with two
    truncated schema dumps that named neither the file nor the column.
    """
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    pq.write_table(
        pa.table(
            {
                "timestamp": list(range(5)),
                "price": [1.0] * 5,
                "day": ["2026-05-14"] * 5,
            }
        ),
        tmp_path / "features_BTCUSDT_2026-05-14.parquet",
    )
    output = tmp_path / "features_BTCUSDT_combined.parquet"

    result = _run(tmp_path, output)

    assert result.returncode == 1
    assert "features_BTCUSDT_2026-05-14.parquet" in result.stdout
    assert "day" in result.stdout
    assert not output.exists(), "nothing may be written when the inputs disagree"


def test_a_differing_column_type_is_reported(tmp_path) -> None:
    _day(tmp_path, "BTCUSDT", "2026-05-13", 10)
    pq.write_table(
        pa.table({"timestamp": list(range(5)), "price": [1] * 5}),  # int, not float
        tmp_path / "features_BTCUSDT_2026-05-14.parquet",
    )

    result = _run(tmp_path, tmp_path / "out.parquet")

    assert result.returncode == 1
    assert "differing types" in result.stdout
    assert "price" in result.stdout


def test_no_daily_files_is_an_error(tmp_path) -> None:
    pq.write_table(
        pa.table({"timestamp": [1], "price": [1.0]}),
        tmp_path / "features_BTCUSDT_multiday.parquet",
    )
    result = _run(tmp_path, tmp_path / "out.parquet")

    assert result.returncode == 1
    assert "No files found" in result.stdout


def test_row_total_is_reported(tmp_path) -> None:
    _day(tmp_path, "BTCUSDT", "2026-05-13", 7)
    _day(tmp_path, "BTCUSDT", "2026-05-14", 8)
    result = _run(tmp_path, tmp_path / "out.parquet")

    assert "Expecting 15 rows" in result.stdout
    assert "Total rows written: 15" in result.stdout
