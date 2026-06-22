"""Phase 7A: Multi-day data expansion for BTCUSDT.

Downloads missing trading days (2026-05-13 to 2026-06-11) and runs the full
Phase 2 pipeline (process + features) on each new day. Produces one feature
parquet per day and a combined multi-day feature store.

Usage
-----
    python scripts/phase7a_data_expansion.py [--skip-existing] [--skip-missing]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import normalize_symbol
from src.data.feature_engineer import FeatureEngineer
from src.data.tick_downloader import TickDownloader
from src.data.tick_processor import TickProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_console() -> None:
    """Use UTF-8 for project paths containing Vietnamese characters on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_range(start: str, end: str) -> list[date]:
    """Return every calendar day in [start, end] inclusive."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    days = []
    current = d0
    while current <= d1:
        days.append(current)
        current += timedelta(days=1)
    return days


def resolve_path(config: dict, key: str) -> Path:
    value = config["paths"][key]
    p = ROOT / value
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------

def download_day(
    symbol: str,
    day: date,
    raw_dir: Path,
    downloader: TickDownloader,
    skip_missing: bool = True,
) -> Path | None:
    """Download a single day. Returns the output path or None if skipped."""
    sym = normalize_symbol(symbol)
    target = raw_dir / f"tick_{sym}_{day.isoformat()}.csv.gz"
    if target.exists():
        print(f"  [SKIP] {target.name} already exists")
        return target
    try:
        outputs = downloader.download_historical(
            symbol,
            day.isoformat(),
            day.isoformat(),
            raw_dir,
            dataset="trades",
            skip_missing=skip_missing,
        )
        if outputs:
            print(f"  [OK]   Downloaded {target.name}")
            return target
        print(f"  [MISS] No data available for {day.isoformat()}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERR]  {day.isoformat()}: {exc}")
        return None


def process_day(
    raw_path: Path,
    symbol: str,
    processed_dir: Path,
    reports_dir: Path,
    processor: TickProcessor,
    skip_existing: bool,
) -> Path | None:
    """Clean raw tick file. Returns processed parquet path."""
    sym = normalize_symbol(symbol)
    # Extract date token from filename
    stem = raw_path.name.replace(".csv.gz", "")
    parts = stem.split("_")
    date_token = next((p for p in parts if "-" in p and len(p) == 10), "unknown")

    output = processed_dir / f"tick_processed_{sym}_{date_token}.parquet"
    if skip_existing and output.exists():
        print(f"  [SKIP] {output.name} already processed")
        return output
    try:
        report = reports_dir / f"data_quality_{sym}_{date_token}.txt"
        processor.process_file(raw_path, output, report)
        print(f"  [OK]   Processed → {output.name}")
        return output
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERR]  Processing {raw_path.name}: {exc}")
        return None


def featurize_day(
    processed_path: Path,
    symbol: str,
    features_dir: Path,
    reports_dir: Path,
    engineer: FeatureEngineer,
    raw_dir: Path,
    obi_source: str,
    skip_existing: bool,
) -> Path | None:
    """Generate features for one processed file. Returns feature parquet path."""
    sym = normalize_symbol(symbol)
    stem = processed_path.name.replace(".parquet", "")
    parts = stem.split("_")
    date_token = next((p for p in parts if "-" in p and len(p) == 10), "unknown")

    output = features_dir / f"features_{sym}_{date_token}.parquet"
    if skip_existing and output.exists():
        print(f"  [SKIP] {output.name} already featurized")
        return output
    try:
        # Check for LOB file
        lob_path = raw_dir / f"lob_{sym}_{date_token}.h5"
        resolved_obi = obi_source
        if resolved_obi == "auto":
            resolved_obi = "lob" if lob_path.exists() else "trade_imbalance"
        use_lob = resolved_obi == "lob" and lob_path.exists()

        if not use_lob:
            print(f"  {date_token}: using causal trade-volume imbalance proxy")

        metadata_path = reports_dir / f"feature_metadata_{sym}_{date_token}.json"
        stats_path = reports_dir / f"feature_stats_{sym}_{date_token}.csv"

        engineer.engineer_files(
            processed_path,
            output,
            stats_path,
            lob_path=lob_path if use_lob else None,
            metadata_path=metadata_path,
            require_lob=use_lob,
            obi_source=resolved_obi,
        )
        df = pd.read_parquet(output)
        print(f"  [OK]   Features → {output.name}  ({len(df):,} rows)")
        return output
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERR]  Featurizing {processed_path.name}: {exc}")
        return None



# ---------------------------------------------------------------------------
# Multi-day consolidation
# ---------------------------------------------------------------------------

def build_multiday_store(
    feature_paths: list[Path],
    output_path: Path,
    holdout_start: str = "2026-06-12",
) -> None:
    """Instead of concatenating into memory, we just validate the daily files.
    The benchmark will load individual days as needed to prevent OOM on 100M+ rows.
    """
    total_rows = 0
    days = 0
    for fp in sorted(feature_paths):
        try:
            # Only read metadata to get row count
            from pyarrow.parquet import read_metadata
            meta = read_metadata(fp)
            total_rows += meta.num_rows
            days += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Could not read metadata for {fp.name}: {exc}")

    if days == 0:
        raise RuntimeError("No feature files could be loaded.")

    print(f"\n[STORE] Multi-day files stored individually in {feature_paths[0].parent}")
    print(f"        Total rows : {total_rows:,}")
    print(f"        Days       : {days}")
    return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "data_config.yaml",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip downloading/processing files that already exist",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Silently skip days not available on Binance (e.g. weekends)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually downloading",
    )
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    config = load_config(args.config)

    symbol = config["symbol"]
    sym = normalize_symbol(symbol)
    start = config["start_date"]
    end = config["end_date"]
    days = date_range(start, end)

    raw_dir = resolve_path(config, "raw")
    processed_dir = resolve_path(config, "processed")
    features_dir = resolve_path(config, "features")
    reports_dir = ROOT / config["paths"].get("reports", "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 7A: Data Expansion ===")
    print(f"Symbol  : {symbol}")
    print(f"Range   : {start} -> {end} ({len(days)} calendar days)")
    print(f"Raw dir : {raw_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Would process these days:")
        for d in days:
            print(f"  {d.isoformat()}")
        return

    # Instantiate pipeline components
    downloader = TickDownloader(base_url=config["historical_base_url"])
    proc_cfg = config.get("processing", {})
    processor = TickProcessor(
        rolling_window=int(proc_cfg.get("rolling_window", 100)),
        outlier_z_score=float(proc_cfg.get("outlier_z_score", 3.0)),
        gap_threshold_seconds=float(proc_cfg.get("gap_threshold_seconds", 300)),
    )
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        autocorrelation_lags=int(feat_cfg.get("autocorrelation_lags", 20)),
        lob_tolerance_seconds=float(feat_cfg.get("lob_tolerance_seconds", 5)),
        minimum_lob_match_fraction=float(feat_cfg.get("minimum_lob_match_fraction", 0.95)),
        trade_imbalance_window=int(feat_cfg.get("trade_imbalance_window", 100)),
        trade_imbalance_min_periods=int(feat_cfg.get("trade_imbalance_min_periods", 20)),
    )
    configured_obi_source = feat_cfg.get("obi_source", "auto")

    downloaded: list[Path] = []
    processed_paths: list[Path] = []
    feature_paths: list[Path] = []

    # Step 1: Download all days
    print(f"\n--- Step 1: Download ({len(days)} days) ---")
    for day in days:
        result = download_day(symbol, day, raw_dir, downloader, args.skip_missing)
        if result is not None:
            downloaded.append(result)

    print(f"\n  Downloaded/found: {len(downloaded)} files")

    # Step 2: Process
    print(f"\n--- Step 2: Process ({len(downloaded)} raw files) ---")
    for raw_path in sorted(downloaded):
        result = process_day(
            raw_path, symbol, processed_dir, reports_dir, processor, args.skip_existing
        )
        if result is not None:
            processed_paths.append(result)

    print(f"\n  Processed: {len(processed_paths)} files")

    # Step 3: Featurize
    print(f"\n--- Step 3: Featurize ({len(processed_paths)} processed files) ---")
    for proc_path in sorted(processed_paths):
        result = featurize_day(
            proc_path,
            symbol,
            features_dir,
            reports_dir,
            engineer,
            raw_dir,
            configured_obi_source,
            args.skip_existing,
        )
        if result is not None:
            feature_paths.append(result)

    print(f"\n  Feature files: {len(feature_paths)}")

    # Step 4: Build multi-day store
    if feature_paths:
        print(f"\n--- Step 4: Build multi-day store ---")
        store_path = features_dir / f"multiday_{sym}.parquet"
        build_multiday_store(feature_paths, store_path)
        print(f"\n=== Phase 7A Complete ===")
    else:
        print("\n[WARN] No feature files produced. Multi-day store not built.")
        sys.exit(1)


if __name__ == "__main__":
    main()
