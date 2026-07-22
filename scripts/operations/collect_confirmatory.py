"""Pre-registration-compliant L2 LOB + trade collection, segmented by UTC day.

``docs/data_collection_todo.md`` freezes a confirmatory protocol that the repo
had no tooling for ("Trạng thái: chưa bắt đầu"). This runner implements it:

* collects **synchronized** trade and L2 depth over one combined stream
  (``LiveMarketCollector``), so the confirmatory OBI comes from a real book —
  ``obi_source`` is hard-coded to ``"lob"`` and can never silently fall back to
  the trade-flow proxy the exploratory data used;
* segments by **complete UTC day**, the protocol's split unit, writing each day
  under ``data/assets/<symbol>/raw/confirmatory/<YYYY-MM-DD>/``;
* runs in chunks so a restart resumes the day instead of losing it;
* records venue, timezone, reconnects, dropped messages, observed coverage,
  largest gap and clock drift, then validates the day against explicit
  thresholds before marking it complete;
* writes an **immutable** per-day manifest (SHA-256 of every raw file, protocol
  version, git commit, dependency lock) and refuses to overwrite a day that is
  already complete.

Wall-clock note: the protocol requires >= 20 *future* UTC days per asset, so a
complete confirmatory set takes >= 20 days of real time and live venue access.
This module is the tooling and the bookkeeping; ``--status`` reports progress
toward the target.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from src.data.paths import asset_data_dir
from src.evaluation.provenance import git_commit, sha256_file

ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_VERSION = "confirmatory_l2_lob_v1"
VENUE = "binance"
OBI_SOURCE = "lob"
TIMEZONE = "UTC"
NANOS_PER_SECOND = 1_000_000_000
SECONDS_PER_DAY = 86_400

#: A UTC day is only usable when the stream actually covered it.
DEFAULT_THRESHOLDS = {
    "min_coverage_fraction": 0.95,
    "max_gap_seconds": 300.0,
    "min_trades": 1000,
    "min_lob_snapshots": 1000,
}


class SupportsCollect(Protocol):
    """The slice of ``LiveMarketCollector`` this runner depends on."""

    def collect(
        self,
        symbol: str,
        tick_path: str | Path,
        lob_path: str | Path,
        *,
        duration_seconds: float,
        snapshot_interval_seconds: float = ...,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DayPaths:
    directory: Path
    ticks: Path
    lob: Path
    manifest: Path


def utc_day_bounds(day: date) -> tuple[datetime, datetime]:
    """Return the [start, end) UTC datetimes of ``day``."""
    start = datetime.combine(day, dt_time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def day_paths(symbol: str, day: date, *, create: bool = False) -> DayPaths:
    """Return the canonical confirmatory raw paths for one asset-day."""
    raw_root = asset_data_dir(symbol, "raw", create=create)
    directory = raw_root / "confirmatory" / day.isoformat()
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return DayPaths(
        directory=directory,
        ticks=directory / "ticks.csv.gz",
        lob=directory / "lob.h5",
        manifest=directory / "manifest.json",
    )


def is_day_complete(symbol: str, day: date) -> bool:
    """True when a validated manifest already exists for the asset-day."""
    manifest = day_paths(symbol, day).manifest
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("complete"))


def _coverage_from_chunks(
    chunks: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
) -> dict[str, float]:
    """Derive observed coverage and the largest gap from chunk summaries.

    Each chunk reports the first/last event it saw; the union of those spans is
    what the stream actually covered inside the UTC day.
    """
    spans: list[tuple[int, int]] = []
    for chunk in chunks:
        candidates = [
            value
            for value in (
                chunk.get("first_trade_ns"),
                chunk.get("first_lob_ns"),
            )
            if value is not None
        ]
        ends = [
            value
            for value in (chunk.get("last_trade_ns"), chunk.get("last_lob_ns"))
            if value is not None
        ]
        if not candidates or not ends:
            continue
        first = max(min(candidates), start_ns)
        last = min(max(ends), end_ns)
        if last > first:
            spans.append((first, last))

    if not spans:
        return {
            "coverage_fraction": 0.0,
            "largest_gap_seconds": float(SECONDS_PER_DAY),
            "observed_seconds": 0.0,
        }

    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for first, last in spans[1:]:
        if first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])

    observed_ns = sum(last - first for first, last in merged)
    gaps = [merged[0][0] - start_ns, end_ns - merged[-1][1]]
    for index in range(1, len(merged)):
        gaps.append(merged[index][0] - merged[index - 1][1])
    largest_gap_ns = max(max(gaps), 0)
    return {
        "coverage_fraction": observed_ns / (end_ns - start_ns),
        "largest_gap_seconds": largest_gap_ns / NANOS_PER_SECOND,
        "observed_seconds": observed_ns / NANOS_PER_SECOND,
    }


def evaluate_day_quality(
    summary: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Return ``(valid, reasons)`` for a collected UTC day."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []
    if summary["coverage_fraction"] < limits["min_coverage_fraction"]:
        reasons.append(
            f"coverage {summary['coverage_fraction']:.4f} < "
            f"{limits['min_coverage_fraction']}"
        )
    if summary["largest_gap_seconds"] > limits["max_gap_seconds"]:
        reasons.append(
            f"largest gap {summary['largest_gap_seconds']:.1f}s > "
            f"{limits['max_gap_seconds']}s"
        )
    if summary["trades"] < limits["min_trades"]:
        reasons.append(f"trades {summary['trades']} < {limits['min_trades']}")
    if summary["lob_snapshots"] < limits["min_lob_snapshots"]:
        reasons.append(
            f"lob_snapshots {summary['lob_snapshots']} < "
            f"{limits['min_lob_snapshots']}"
        )
    return (not reasons), reasons


def collect_utc_day(
    symbol: str,
    day: date,
    *,
    collector: SupportsCollect,
    chunk_seconds: float = 900.0,
    snapshot_interval_seconds: float = 1.0,
    thresholds: dict[str, float] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Collect (or resume) one complete UTC day and write its manifest.

    Refuses to touch a day that is already complete: the protocol requires raw
    confirmatory data to be immutable once accepted.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if is_day_complete(symbol, day):
        raise RuntimeError(
            f"{symbol} {day.isoformat()} is already complete; confirmatory raw "
            "data is immutable"
        )

    clock = now or (lambda: datetime.now(timezone.utc))
    start, end = utc_day_bounds(day)
    current = clock()
    if current < start:
        raise RuntimeError(
            f"{day.isoformat()} has not started yet (now {current.isoformat()})"
        )

    paths = day_paths(symbol, day, create=True)
    start_ns = int(start.timestamp() * NANOS_PER_SECOND)
    end_ns = int(end.timestamp() * NANOS_PER_SECOND)

    chunks: list[dict[str, Any]] = []
    while True:
        current = clock()
        if current >= end:
            break
        remaining = (end - current).total_seconds()
        duration = min(chunk_seconds, remaining)
        if duration <= 0:
            break
        chunks.append(
            collector.collect(
                symbol,
                paths.ticks,
                paths.lob,
                duration_seconds=duration,
                snapshot_interval_seconds=snapshot_interval_seconds,
            )
        )

    coverage = _coverage_from_chunks(chunks, start_ns, end_ns)
    summary = {
        "symbol": symbol.upper(),
        "utc_day": day.isoformat(),
        "trades": sum(int(c.get("trades", 0)) for c in chunks),
        "lob_snapshots": sum(int(c.get("lob_snapshots", 0)) for c in chunks),
        "reconnects": sum(int(c.get("reconnects", 0)) for c in chunks),
        "dropped_before_first_snapshot": sum(
            int(c.get("dropped_before_first_snapshot", 0)) for c in chunks
        ),
        "chunks": len(chunks),
        **coverage,
    }
    valid, reasons = evaluate_day_quality(summary, thresholds)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "venue": VENUE,
        "timezone": TIMEZONE,
        "obi_source": OBI_SOURCE,
        "generated_utc": clock().isoformat(),
        "git_commit": git_commit(ROOT),
        "python": platform.python_version(),
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
        "summary": summary,
        "complete": bool(valid),
        "rejection_reasons": reasons,
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in (("ticks", paths.ticks), ("lob", paths.lob))
            if path.exists()
        },
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def confirmatory_status(
    symbols: list[str],
    *,
    required_days: int = 20,
) -> dict[str, Any]:
    """Report how many validated UTC days each asset has collected."""
    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        root = asset_data_dir(symbol, "raw") / "confirmatory"
        complete: list[str] = []
        rejected: list[str] = []
        if root.exists():
            for directory in sorted(p for p in root.iterdir() if p.is_dir()):
                manifest = directory / "manifest.json"
                if not manifest.exists():
                    continue
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                (complete if payload.get("complete") else rejected).append(
                    directory.name
                )
        per_symbol[symbol.upper()] = {
            "complete_days": len(complete),
            "required_days": required_days,
            "remaining_days": max(0, required_days - len(complete)),
            "target_met": len(complete) >= required_days,
            "complete": complete,
            "rejected": rejected,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "required_days": required_days,
        "assets": per_symbol,
        "all_targets_met": all(
            entry["target_met"] for entry in per_symbol.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT",
        help="Comma-separated assets (protocol fixes these three).",
    )
    parser.add_argument(
        "--day",
        default="",
        help="UTC day to collect (YYYY-MM-DD). Defaults to today (UTC).",
    )
    parser.add_argument("--chunk-seconds", type=float, default=900.0)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=1.0)
    parser.add_argument("--required-days", type=int, default=20)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only report progress toward the confirmatory target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.status:
        report = confirmatory_status(symbols, required_days=args.required_days)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    from src.data.live_market_collector import LiveMarketCollector

    day = (
        date.fromisoformat(args.day)
        if args.day
        else datetime.now(timezone.utc).date()
    )
    collector = LiveMarketCollector()
    for symbol in symbols:
        if is_day_complete(symbol, day):
            print(f"[confirmatory] {symbol} {day}: already complete, skipping")
            continue
        print(f"[confirmatory] {symbol} {day}: collecting...")
        manifest = collect_utc_day(
            symbol,
            day,
            collector=collector,
            chunk_seconds=args.chunk_seconds,
            snapshot_interval_seconds=args.snapshot_interval_seconds,
        )
        state = "COMPLETE" if manifest["complete"] else "REJECTED"
        print(
            f"[confirmatory] {symbol} {day}: {state} "
            f"coverage={manifest['summary']['coverage_fraction']:.4f} "
            f"trades={manifest['summary']['trades']} "
            f"snapshots={manifest['summary']['lob_snapshots']}"
        )
        if not manifest["complete"]:
            for reason in manifest["rejection_reasons"]:
                print(f"[confirmatory]   reason: {reason}")


if __name__ == "__main__":
    main()
