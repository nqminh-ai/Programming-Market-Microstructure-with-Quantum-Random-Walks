"""Report how far each asset has been carried through the data pipeline.

Raw days, processed days and feature days are counted separately because the
pipeline can succeed while lagging: a downloaded file that never reached the
raw directory, or a raw day that was never processed, produces no error at all,
just fewer days in the next stage. Comparing the three columns makes that
visible.

Combined feature stores are listed apart from per-day files. Those two must
never be mixed -- ``combine_features`` globs the same directory it writes into,
so an aggregate counted as a day would be folded back into the next rebuild.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from src.data.common import normalize_symbol
from src.data.paths import asset_data_dir

DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


@dataclass(frozen=True)
class StageDays:
    """The distinct dates present at one pipeline stage."""

    days: set[str]
    files: int

    @property
    def span(self) -> str:
        if not self.days:
            return "-"
        ordered = sorted(self.days)
        return f"{ordered[0]} -> {ordered[-1]}"


def _stage(directory: Path, pattern: str) -> StageDays:
    if not directory.is_dir():
        return StageDays(set(), 0)
    days: set[str] = set()
    files = 0
    for path in directory.glob(pattern):
        match = DAY.search(path.name)
        if match:
            days.add(match.group(1))
            files += 1
    return StageDays(days, files)


def asset_status(symbol: str) -> dict[str, object]:
    upper = normalize_symbol(symbol)
    raw = _stage(asset_data_dir(upper, "raw"), f"tick_{upper}_*")
    processed = _stage(asset_data_dir(upper, "processed"), f"tick_processed_{upper}_*.parquet")
    features = _stage(asset_data_dir(upper, "features"), f"features_{upper}_*.parquet")

    feature_dir = asset_data_dir(upper, "features")
    aggregates = []
    if feature_dir.is_dir():
        for path in sorted(feature_dir.glob(f"features_{upper}_*.parquet")):
            if DAY.search(path.name):
                continue
            try:
                rows = pq.ParquetFile(path).metadata.num_rows
            except Exception:  # noqa: BLE001 - a partial file must not stop the report
                rows = -1
            aggregates.append(
                {"name": path.name, "rows": rows, "bytes": path.stat().st_size}
            )

    return {
        "symbol": upper,
        "raw": raw,
        "processed": processed,
        "features": features,
        "aggregates": aggregates,
        # Days present upstream but missing downstream: the silent-lag cases.
        "unprocessed": sorted(raw.days - processed.days),
        "no_features": sorted(processed.days - features.days),
        "duplicate_raw": raw.files - len(raw.days),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [value.strip() for value in args.symbols.split(",") if value.strip()]

    print(f"{'asset':<9} {'raw':>5} {'processed':>10} {'features':>9}   khoang ngay")
    print("-" * 72)
    for symbol in symbols:
        status = asset_status(symbol)
        print(
            f"{status['symbol']:<9} {len(status['raw'].days):>5} "
            f"{len(status['processed'].days):>10} {len(status['features'].days):>9}   "
            f"{status['raw'].span}"
        )
        if status["duplicate_raw"]:
            print(f"          ! {status['duplicate_raw']} file raw trung ngay")
        if status["unprocessed"]:
            days = status["unprocessed"]
            shown = ", ".join(days[:4]) + (" ..." if len(days) > 4 else "")
            print(f"          ! {len(days)} ngay raw chua xu ly: {shown}")
        if status["no_features"]:
            days = status["no_features"]
            shown = ", ".join(days[:4]) + (" ..." if len(days) > 4 else "")
            print(f"          ! {len(days)} ngay chua co feature: {shown}")

    print("\nFile gop (khong phai file theo ngay):")
    for symbol in symbols:
        for aggregate in asset_status(symbol)["aggregates"]:
            rows = "loi doc" if aggregate["rows"] < 0 else f"{aggregate['rows']:,}"
            print(
                f"  {normalize_symbol(symbol):<9} {aggregate['name']:<40} "
                f"{rows:>15} dong  {aggregate['bytes'] / 1e9:>5.2f} GB"
            )


if __name__ == "__main__":
    main()
