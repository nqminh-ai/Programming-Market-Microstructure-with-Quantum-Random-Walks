"""Tests for the pre-registration-compliant confirmatory collection runner.

A fake collector stands in for the live WebSocket so the protocol rules — UTC
day segmentation, quality gating, immutability and provenance — are testable
without network access.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.operations import collect_confirmatory as cc

NANOS = cc.NANOS_PER_SECOND


class _FakeWorld:
    """Couples a clock to a fake collector.

    Wall-clock time advances only when collection consumes it, which is how the
    real runner behaves; a clock that ticked on every read would leave the last
    chunk of the day uncollected and fabricate a trailing gap.
    """

    def __init__(
        self,
        day: date,
        *,
        trades_per_chunk: int = 500,
        snapshots_per_chunk: int = 500,
        gap_seconds: float = 0.0,
        start_offset_seconds: float = 0.0,
    ) -> None:
        start, _ = cc.utc_day_bounds(day)
        self.now = start + timedelta(seconds=start_offset_seconds)
        self.cursor_ns = int(self.now.timestamp() * NANOS)
        self.trades_per_chunk = trades_per_chunk
        self.snapshots_per_chunk = snapshots_per_chunk
        self.gap_ns = int(gap_seconds * NANOS)
        self.calls = 0

    def clock(self) -> datetime:
        return self.now

    def collect(
        self,
        symbol: str,
        tick_path: str | Path,
        lob_path: str | Path,
        *,
        duration_seconds: float,
        snapshot_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        self.calls += 1
        for path in (Path(tick_path), Path(lob_path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(b"x")
        first = self.cursor_ns + self.gap_ns
        last = first + int(duration_seconds * NANOS)
        self.cursor_ns = last
        self.now += timedelta(seconds=duration_seconds)
        return {
            "symbol": symbol,
            "trades": self.trades_per_chunk,
            "lob_snapshots": self.snapshots_per_chunk,
            "reconnects": 0,
            "dropped_before_first_snapshot": 0,
            "first_trade_ns": first,
            "last_trade_ns": last,
            "first_lob_ns": first,
            "last_lob_ns": last,
        }


def _run_day(symbol: str, day: date, **kwargs) -> tuple[dict[str, Any], "_FakeWorld"]:
    """Collect one day through a coupled fake world."""
    world = _FakeWorld(day, **kwargs)
    manifest = cc.collect_utc_day(
        symbol,
        day,
        collector=world,
        chunk_seconds=3600.0,
        now=world.clock,
    )
    return manifest, world


@pytest.fixture
def isolated_data_root(tmp_path, monkeypatch):
    """Redirect asset raw storage into tmp_path."""

    def _asset_data_dir(symbol: str, kind: str, *, create: bool = False) -> Path:
        path = tmp_path / "assets" / symbol.lower() / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # ROOT deliberately stays the real repository: the manifest records the
    # collecting commit, and it already falls back to an absolute path for
    # files outside the repo.
    monkeypatch.setattr(cc, "asset_data_dir", _asset_data_dir)
    return tmp_path


def test_utc_day_bounds_span_exactly_one_day() -> None:
    start, end = cc.utc_day_bounds(date(2026, 8, 1))
    assert start == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert (end - start) == timedelta(days=1)


def test_day_paths_follow_the_confirmatory_layout(isolated_data_root) -> None:
    paths = cc.day_paths("BTCUSDT", date(2026, 8, 1))
    assert paths.directory.parts[-3:] == ("raw", "confirmatory", "2026-08-01")
    assert paths.ticks.name == "ticks.csv.gz"
    assert paths.lob.name == "lob.h5"


def test_full_coverage_day_is_marked_complete(isolated_data_root) -> None:
    day = date(2026, 8, 1)
    manifest, _ = _run_day("BTCUSDT", day)
    assert manifest["complete"] is True
    assert manifest["rejection_reasons"] == []
    assert manifest["summary"]["coverage_fraction"] == pytest.approx(1.0, abs=1e-6)
    # Protocol guards: confirmatory OBI must come from the real book.
    assert manifest["obi_source"] == "lob"
    assert manifest["venue"] == cc.VENUE
    assert manifest["timezone"] == "UTC"
    assert manifest["protocol_version"] == cc.PROTOCOL_VERSION
    # Provenance: every raw file is hashed.
    assert set(manifest["files"]) == {"ticks", "lob"}
    for entry in manifest["files"].values():
        assert len(entry["sha256"]) == 64
    assert cc.is_day_complete("BTCUSDT", day) is True


def test_a_large_gap_rejects_the_day(isolated_data_root) -> None:
    day = date(2026, 8, 2)
    # Push every chunk 2 hours late: coverage drops and the leading gap is huge.
    manifest, _ = _run_day("ETHUSDT", day, gap_seconds=7200.0)
    assert manifest["complete"] is False
    assert manifest["rejection_reasons"]
    assert cc.is_day_complete("ETHUSDT", day) is False


def test_too_few_trades_rejects_the_day(isolated_data_root) -> None:
    day = date(2026, 8, 3)
    manifest, _ = _run_day("BNBUSDT", day, trades_per_chunk=1, snapshots_per_chunk=1)
    assert manifest["complete"] is False
    assert any("trades" in reason for reason in manifest["rejection_reasons"])


def test_completed_day_is_immutable(isolated_data_root) -> None:
    day = date(2026, 8, 4)
    _run_day("BTCUSDT", day)
    with pytest.raises(RuntimeError, match="immutable"):
        _run_day("BTCUSDT", day)


def test_future_day_is_refused(isolated_data_root) -> None:
    day = date(2026, 8, 5)
    start, _ = cc.utc_day_bounds(day)
    earlier = start - timedelta(days=2)
    with pytest.raises(RuntimeError, match="has not started"):
        cc.collect_utc_day(
            "BTCUSDT",
            day,
            collector=_FakeWorld(day),
            now=lambda: earlier,
        )


def test_coverage_merges_overlaps_and_reports_largest_gap() -> None:
    start_ns, end_ns = 0, 100 * NANOS
    chunks = [
        {"first_trade_ns": 0, "last_trade_ns": 20 * NANOS,
         "first_lob_ns": 0, "last_lob_ns": 20 * NANOS},
        # Overlaps the previous span; must merge rather than double count.
        {"first_trade_ns": 10 * NANOS, "last_trade_ns": 30 * NANOS,
         "first_lob_ns": 10 * NANOS, "last_lob_ns": 30 * NANOS},
        # Leaves a 40s hole before it.
        {"first_trade_ns": 70 * NANOS, "last_trade_ns": 100 * NANOS,
         "first_lob_ns": 70 * NANOS, "last_lob_ns": 100 * NANOS},
    ]
    coverage = cc._coverage_from_chunks(chunks, start_ns, end_ns)
    assert coverage["observed_seconds"] == pytest.approx(60.0)
    assert coverage["coverage_fraction"] == pytest.approx(0.6)
    assert coverage["largest_gap_seconds"] == pytest.approx(40.0)


def test_status_counts_complete_and_remaining(isolated_data_root) -> None:
    for day in (date(2026, 9, 1), date(2026, 9, 2)):
        _run_day("BTCUSDT", day)
    report = cc.confirmatory_status(["BTCUSDT", "ETHUSDT"], required_days=20)
    btc = report["assets"]["BTCUSDT"]
    assert btc["complete_days"] == 2
    assert btc["remaining_days"] == 18
    assert btc["target_met"] is False
    assert report["assets"]["ETHUSDT"]["complete_days"] == 0
    assert report["all_targets_met"] is False


def test_rejected_days_are_listed_separately(isolated_data_root) -> None:
    day = date(2026, 9, 10)
    _run_day("BNBUSDT", day, trades_per_chunk=1, snapshots_per_chunk=1)
    report = cc.confirmatory_status(["BNBUSDT"], required_days=20)
    entry = report["assets"]["BNBUSDT"]
    assert entry["complete_days"] == 0
    assert entry["rejected"] == [day.isoformat()]


def test_manifest_is_valid_json_on_disk(isolated_data_root) -> None:
    day = date(2026, 9, 20)
    _run_day("BTCUSDT", day)
    manifest_path = cc.day_paths("BTCUSDT", day).manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["summary"]["utc_day"] == day.isoformat()
    assert payload["summary"]["chunks"] == 24
