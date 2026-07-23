"""How far along is each long-running research study?

These studies take an hour or more per asset and write their report only at the
end, so "is it working or is it stuck?" is not answerable from the output
files. This reads the run logs instead and reports windows completed, rate, and
an estimate of the time left.

    python -m scripts.operations.study_status
    python -m scripts.operations.study_status --logs /tmp/vol_bnb.log
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# One of these is emitted per window, on stderr, which is unbuffered -- so it
# shows up even when a run's stdout is still sitting in a block buffer.
WINDOW_MARKER = "Quantum Val Brier"
TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# loguru colours its output, so the timestamp is not at the start of the raw
# line -- an escape sequence is. Without stripping it the rate never resolves.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Resolved through gettempdir rather than written as "/tmp/...": the shell
# these runs are launched from maps /tmp to the OS temp directory, but Python
# on Windows reads "/tmp" as a path on the current drive and finds nothing.
TEMP = Path(tempfile.gettempdir())
DEFAULT_LOGS = tuple(
    str(TEMP / name)
    for name in ("vol_bnb.log", "vol_eth.log", "vol_btc.log", "conf100b.log")
)


@dataclass(frozen=True)
class Progress:
    name: str
    done: int
    total: int | None
    first_seen: float | None
    last_seen: float | None
    finished: bool

    stamps: tuple[float, ...] = ()

    @property
    def per_window_seconds(self) -> float | None:
        """Rate over the recent windows, not over the whole run.

        Averaging since the first window answers the wrong question once
        conditions change mid-run: starting a second study alongside this one
        pushed the since-start figure to 185s while the job was actually doing
        112s, so the estimate stayed pessimistic long after the contention that
        caused it had gone.
        """
        recent = self.stamps[-6:]
        if len(recent) < 2:
            return None
        elapsed = recent[-1] - recent[0]
        return elapsed / (len(recent) - 1) if elapsed > 0 else None

    @property
    def remaining_text(self) -> str:
        if self.finished:
            return "xong"
        if self.total is None or self.per_window_seconds is None:
            return "chua uoc luong duoc"
        left = (self.total - self.done) * self.per_window_seconds
        if left <= 0:
            return "gan xong"
        return f"~{left / 60:.0f} phut nua"


def _parse_time(line: str) -> float | None:
    match = TIMESTAMP.match(ANSI.sub("", line).strip())
    if not match:
        return None
    try:
        return time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def read_progress(path: Path, total: int | None = None) -> Progress | None:
    """Return progress for one log, or None when it does not exist."""
    if not path.is_file():
        return None
    stamps: list[float] = []
    done = 0
    finished = False
    detected_total = total
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        if WINDOW_MARKER in line:
            done += 1
            moment = _parse_time(line)
            if moment is not None:
                stamps.append(moment)
        elif detected_total is None:
            found = re.search(r"window \d+/(\d+)", line)
            if found:
                detected_total = int(found.group(1))
        if "DONE" in line or "wrote " in line:
            finished = True
    return Progress(
        name=path.name,
        done=done,
        total=detected_total,
        first_seen=stamps[0] if stamps else None,
        last_seen=stamps[-1] if stamps else None,
        finished=finished,
        stamps=tuple(stamps),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="*", default=list(DEFAULT_LOGS))
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Expected window count, when the log does not state it.",
    )
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rows = [
        progress
        for path in args.logs
        if (progress := read_progress(Path(path), args.windows)) is not None
    ]
    if not rows:
        print("Khong tim thay log nao dang chay.")
        return

    print(f"{'log':<22}{'xong':>10}{'giay/window':>14}   con lai")
    print("-" * 66)
    for row in rows:
        total = f"/{row.total}" if row.total else ""
        rate = (
            f"{row.per_window_seconds:.0f}s"
            if row.per_window_seconds is not None
            else "-"
        )
        print(f"{row.name:<22}{str(row.done) + total:>10}{rate:>14}   {row.remaining_text}")


if __name__ == "__main__":
    main()
