"""Move freshly downloaded daily trade files into an asset's raw directory.

``scripts/collect_multi_day.py`` writes ``<output_dir>/<symbol_lower>/YYYY-MM-DD.parquet``,
which is neither where the Phase 2 pipeline looks nor the name it expects. That
pipeline globs ``<raw>/tick_<SYMBOL>_*.{csv.gz,parquet}`` with the symbol
upper-cased, so a downloaded file left as-is is silently invisible: the pipeline
reports no error, it simply processes fewer days than were collected.

Case matters even though Windows hides it. Files named ``tick_btcusdt_*`` match
the upper-cased glob on a case-insensitive filesystem and match nothing at all
on Linux, so this normalises the case on the way in rather than leaving a
platform-dependent trap.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.data.common import normalize_symbol
from src.data.paths import asset_data_dir

DATE_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.parquet$")


def pending_downloads(symbol: str, root: Path) -> list[Path]:
    """Downloaded day files still sitting in the asset directory."""
    asset_root = root / normalize_symbol(symbol).lower()
    if not asset_root.is_dir():
        return []
    return sorted(path for path in asset_root.glob("*.parquet") if DATE_FILE.match(path.name))


def ingest(symbol: str, root: Path, *, dry_run: bool = False) -> dict[str, list[str]]:
    """Rename pending downloads into ``<raw>/tick_<SYMBOL>_<date>.parquet``."""
    upper = normalize_symbol(symbol)
    raw_dir = asset_data_dir(upper, "raw", create=not dry_run)
    moved: list[str] = []
    skipped: list[str] = []
    for source in pending_downloads(symbol, root):
        destination = raw_dir / f"tick_{upper}_{source.stem}.parquet"
        # A same-named file already in raw is authoritative; a download must not
        # overwrite data the pipeline has already processed.
        if destination.exists():
            skipped.append(source.name)
            continue
        if not dry_run:
            source.replace(destination)
        moved.append(destination.name)
    return {"moved": moved, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/assets"),
        help="Directory collect_multi_day.py wrote into.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2] / args.root
    for symbol in (value.strip() for value in args.symbols.split(",") if value.strip()):
        result = ingest(symbol, root, dry_run=args.dry_run)
        prefix = "[dry-run] " if args.dry_run else ""
        print(
            f"{prefix}{normalize_symbol(symbol)}: "
            f"{len(result['moved'])} moved, {len(result['skipped'])} already present"
        )
        for name in result["skipped"]:
            print(f"    skipped (exists): {name}")


if __name__ == "__main__":
    main()
