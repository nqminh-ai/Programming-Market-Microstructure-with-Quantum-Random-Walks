"""Tests for scripts/collect_multi_day.py's Binance Vision CSV parsing.

Regression coverage for audit finding C4: the sibling copy of this module
(scripts/operations/collect_multi_day.py, since deleted) referenced
DUMP_CSV_7_COLUMNS/DUMP_CSV_6_COLUMNS without ever defining them, so any call
through _parse_dump_csv raised NameError. Neither file had any test.
"""

from __future__ import annotations

import io
import zipfile

from scripts.collect_multi_day import OUTPUT_SCHEMA_COLUMNS, _parse_dump_csv


def _zip_csv(rows: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("trades.csv", "\n".join(rows) + "\n")
    return buffer.getvalue()


def test_parse_dump_csv_handles_seven_column_format_with_ms_timestamp():
    rows = [
        "1,100.5,0.01,1.005,1700000000123,True,True",
        "2,100.6,0.02,2.012,1700000000456,False,True",
    ]
    result = _parse_dump_csv(_zip_csv(rows), "2023-11-14")

    assert result is not None
    assert list(result.columns) == OUTPUT_SCHEMA_COLUMNS
    assert len(result) == 2
    assert set(result["side"]) == {"buy", "sell"}
    assert result["day"].eq("2023-11-14").all()
    assert (result["timestamp"] > 0).all()


def test_parse_dump_csv_handles_six_column_format_with_us_timestamp():
    rows = [
        "10,200.1,0.5,100.05,1700000000123456,False",
        "11,200.2,0.4,80.08,1700000000456789,True",
    ]
    result = _parse_dump_csv(_zip_csv(rows), "2023-11-14")

    assert result is not None
    assert list(result.columns) == OUTPUT_SCHEMA_COLUMNS
    assert len(result) == 2
    assert set(result["side"]) == {"buy", "sell"}


def test_parse_dump_csv_drops_rows_with_missing_values():
    rows = [
        "1,100.5,0.01,1.005,1700000000123,True,True",
        ",100.6,0.02,2.012,1700000000456,False,True",
    ]
    result = _parse_dump_csv(_zip_csv(rows), "2023-11-14")

    assert result is not None
    assert len(result) == 1


def test_parse_dump_csv_returns_none_for_bad_zip():
    assert _parse_dump_csv(b"not a zip file", "2023-11-14") is None
