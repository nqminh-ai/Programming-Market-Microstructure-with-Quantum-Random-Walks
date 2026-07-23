"""Tests for the run-progress reader.

These studies write their report only at the end, so a log line is the only
evidence a run is alive. Everything here is about that log being readable in
the state it is actually written in -- coloured, buffered, and half-finished.
"""

from __future__ import annotations

from scripts.operations.study_status import read_progress

MARKER = "Quantum Val Brier: 0.15, Classical Val Brier: 0.17"
GREEN = "\x1b[32m"
RESET = "\x1b[0m"


def _log(tmp_path, lines: list[str], name: str = "run.log"):
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_windows_are_counted_from_the_per_window_marker(tmp_path) -> None:
    path = _log(
        tmp_path,
        [f"2026-07-23 19:0{i}:00 | DEBUG | {MARKER}" for i in range(4)],
    )

    assert read_progress(path, total=40).done == 4


def test_the_rate_survives_loguru_colour_codes(tmp_path) -> None:
    """The timestamp is not at the start of the line -- an escape code is.

    Without stripping it the rate never resolves and the report is useless.
    """
    path = _log(
        tmp_path,
        [
            f"{GREEN}2026-07-23 19:00:00{RESET} | DEBUG | {MARKER}",
            f"{GREEN}2026-07-23 19:01:00{RESET} | DEBUG | {MARKER}",
            f"{GREEN}2026-07-23 19:02:00{RESET} | DEBUG | {MARKER}",
        ],
    )

    progress = read_progress(path, total=40)

    assert progress.done == 3
    assert progress.per_window_seconds == 60.0


def test_remaining_time_uses_the_observed_rate(tmp_path) -> None:
    path = _log(
        tmp_path,
        [
            "2026-07-23 19:00:00 | DEBUG | " + MARKER,
            "2026-07-23 19:01:00 | DEBUG | " + MARKER,
        ],
    )

    # 2 of 12 done at 60s each, so 10 windows or about 10 minutes left.
    assert "10 phut" in read_progress(path, total=12).remaining_text


def test_a_single_window_gives_no_rate_rather_than_a_wrong_one(tmp_path) -> None:
    path = _log(tmp_path, ["2026-07-23 19:00:00 | DEBUG | " + MARKER])

    progress = read_progress(path, total=40)

    assert progress.done == 1
    assert progress.per_window_seconds is None
    assert progress.remaining_text == "chua uoc luong duoc"


def test_a_finished_run_is_reported_as_finished(tmp_path) -> None:
    path = _log(
        tmp_path,
        [
            "2026-07-23 19:00:00 | DEBUG | " + MARKER,
            "[crps] wrote reports/research/x.json",
        ],
    )

    assert read_progress(path, total=40).finished is True


def test_the_window_total_is_read_from_the_log_when_it_is_there(tmp_path) -> None:
    """Available once stdout flushes; --windows is the fallback before that."""
    path = _log(
        tmp_path,
        [
            "[crps] window 1/40 (500,000 rows)...",
            "2026-07-23 19:00:00 | DEBUG | " + MARKER,
        ],
    )

    assert read_progress(path).total == 40


def test_a_missing_log_is_skipped_rather_than_raising(tmp_path) -> None:
    assert read_progress(tmp_path / "nope.log") is None
