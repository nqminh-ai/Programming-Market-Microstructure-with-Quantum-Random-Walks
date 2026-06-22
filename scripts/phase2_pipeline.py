"""Command-line entry point for the Phase 2 market-data pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import normalize_symbol  # noqa: E402
from src.data.feature_engineer import FeatureEngineer  # noqa: E402
from src.data.live_market_collector import LiveMarketCollector  # noqa: E402
from src.data.orderbook_collector import OrderBookCollector  # noqa: E402
from src.data.tick_downloader import TickDownloader  # noqa: E402
from src.data.tick_processor import TickProcessor  # noqa: E402


def configure_console() -> None:
    """Use UTF-8 for project paths containing Vietnamese characters on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the Phase 2 YAML configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured project-relative path and create it."""
    value = config.get("paths", {}).get(key)
    if not value:
        raise ValueError(f"configuration is missing paths.{key}")
    path = ROOT / value
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_download(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Download the configured historical date range."""
    downloader = TickDownloader(base_url=config["historical_base_url"])
    outputs = downloader.download_historical(
        config["symbol"],
        args.start_date or config["start_date"],
        args.end_date or config["end_date"],
        resolve_config_path(config, "raw"),
        dataset=config.get("historical_dataset", "trades"),
        skip_missing=args.skip_missing,
    )
    print(f"Downloaded {len(outputs)} daily files.")


def command_collect_lob(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Collect a finite series of REST order-book snapshots."""
    symbol = normalize_symbol(config["symbol"])
    settings = config.get("order_book", {})
    output = (
        Path(args.output)
        if args.output
        else resolve_config_path(config, "raw")
        / f"lob_{symbol}_{date.today().isoformat()}.h5"
    )
    if not output.is_absolute():
        output = ROOT / output

    collector = OrderBookCollector(
        rest_base_url=config["rest_base_url"],
        top_levels_for_obi=int(settings.get("top_levels_for_obi", 5)),
    )
    collector.collect(
        config["symbol"],
        output,
        depth=int(settings.get("depth", 20)),
        interval_seconds=float(settings.get("snapshot_interval_seconds", 1.0)),
        n_snapshots=args.snapshots,
    )
    print(f"Saved {args.snapshots} LOB snapshots to {output}")


def command_collect_live(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Collect synchronized trade and partial-depth streams."""
    symbol = normalize_symbol(config["symbol"])
    settings = config.get("order_book", {})
    token = date.today().isoformat()
    raw_dir = resolve_config_path(config, "raw")
    tick_output = (
        Path(args.tick_output)
        if args.tick_output
        else raw_dir / f"tick_{symbol}_{token}_live.csv.gz"
    )
    lob_output = (
        Path(args.lob_output)
        if args.lob_output
        else raw_dir / f"lob_{symbol}_{token}.h5"
    )
    if not tick_output.is_absolute():
        tick_output = ROOT / tick_output
    if not lob_output.is_absolute():
        lob_output = ROOT / lob_output

    collector = LiveMarketCollector(
        top_levels_for_obi=int(settings.get("top_levels_for_obi", 5)),
    )
    summary = collector.collect(
        config["symbol"],
        tick_output,
        lob_output,
        duration_seconds=args.duration_seconds,
        snapshot_interval_seconds=float(
            settings.get("snapshot_interval_seconds", 1.0)
        ),
    )
    print(json.dumps(summary, indent=2))


def raw_tick_files(config: dict[str, Any], explicit: list[str] | None) -> list[Path]:
    """Return explicit or configured raw tick files."""
    if explicit:
        paths = [Path(value) for value in explicit]
        return [path if path.is_absolute() else ROOT / path for path in paths]
    symbol = normalize_symbol(config["symbol"])
    return sorted(resolve_config_path(config, "raw").glob(f"tick_{symbol}_*.csv.gz"))


def processed_tick_files(
    config: dict[str, Any],
    explicit: list[str] | None,
) -> list[Path]:
    """Return explicit or configured processed tick files."""
    if explicit:
        paths = [Path(value) for value in explicit]
        return [path if path.is_absolute() else ROOT / path for path in paths]
    symbol = normalize_symbol(config["symbol"])
    return sorted(
        resolve_config_path(config, "processed").glob(
            f"tick_processed_{symbol}_*.parquet"
        )
    )


def date_token(path: Path) -> str:
    """Extract an ISO date from an artifact filename."""
    for part in path.name.replace(".csv.gz", "").replace(".parquet", "").split("_"):
        try:
            date.fromisoformat(part)
            return part
        except ValueError:
            continue
    return "unknown-date"


def command_process(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Clean every selected raw tick file."""
    files = raw_tick_files(config, args.input)
    if not files:
        raise FileNotFoundError("no raw tick files found; run the download command first")

    settings = config.get("processing", {})
    processor = TickProcessor(
        rolling_window=int(settings.get("rolling_window", 100)),
        outlier_z_score=float(settings.get("outlier_z_score", 3.0)),
        gap_threshold_seconds=float(settings.get("gap_threshold_seconds", 300)),
    )
    symbol = normalize_symbol(config["symbol"])
    processed_dir = resolve_config_path(config, "processed")
    reports_dir = resolve_config_path(config, "reports")

    for source in files:
        token = date_token(source)
        output = processed_dir / f"tick_processed_{symbol}_{token}.parquet"
        report = reports_dir / f"data_quality_{symbol}_{token}.txt"
        processor.process_file(source, output, report)
    print(f"Processed {len(files)} daily files.")


def command_features(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Create one feature matrix per selected processed tick file."""
    files = processed_tick_files(config, args.input)
    if not files:
        raise FileNotFoundError("no processed tick files found; run process first")

    settings = config.get("features", {})
    engineer = FeatureEngineer(
        autocorrelation_lags=int(settings.get("autocorrelation_lags", 20)),
        lob_tolerance_seconds=float(settings.get("lob_tolerance_seconds", 5)),
        minimum_lob_match_fraction=float(
            settings.get("minimum_lob_match_fraction", 0.95)
        ),
        trade_imbalance_window=int(
            args.trade_imbalance_window
            or settings.get("trade_imbalance_window", 100)
        ),
        trade_imbalance_min_periods=int(
            settings.get("trade_imbalance_min_periods", 20)
        ),
    )
    configured_obi_source = args.obi_source or settings.get("obi_source", "auto")
    lob_source = None
    if args.lob:
        lob_source = Path(args.lob)
        if not lob_source.is_absolute():
            lob_source = ROOT / lob_source

    symbol = normalize_symbol(config["symbol"])
    raw_dir = resolve_config_path(config, "raw")
    feature_dir = resolve_config_path(config, "features")
    reports_dir = resolve_config_path(config, "reports")
    real_lob_count = 0
    synthetic_count = 0
    for source in files:
        token = date_token(source)
        if lob_source is None:
            lob_path = raw_dir / f"lob_{symbol}_{token}.h5"
        elif lob_source.is_dir():
            lob_path = lob_source / f"lob_{symbol}_{token}.h5"
        else:
            lob_path = lob_source
        has_lob = lob_path.exists()
        selected_source = configured_obi_source
        if selected_source == "auto":
            selected_source = "lob" if has_lob else "trade_imbalance"
        if selected_source == "lob" and not has_lob:
            raise FileNotFoundError(
                f"missing same-period LOB data for {token}: {lob_path}"
            )
        use_lob = selected_source == "lob"
        if not use_lob:
            print(
                f"  {token}: using causal trade-volume imbalance proxy"
            )
            synthetic_count += 1
        else:
            real_lob_count += 1
        engineer.engineer_files(
            source,
            feature_dir / f"features_{symbol}_{token}.parquet",
            reports_dir / f"feature_stats_{symbol}_{token}.csv",
            lob_path=lob_path if use_lob else None,
            metadata_path=reports_dir / f"feature_metadata_{symbol}_{token}.json",
            require_lob=use_lob,
            obi_source=selected_source,
        )
    print(
        f"Created {len(files)} feature matrices"
        f" ({real_lob_count} with real LOB, {synthetic_count} with synthetic OBI)."
    )


def command_test(_: dict[str, Any], __: argparse.Namespace) -> None:
    """Run the Phase 2 pytest suite."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_data_pipeline.py",
            "-v",
            f"--basetemp={ROOT / 'reports' / '.pytest-tmp-phase2'}",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        check=False,
    )
    raise SystemExit(result.returncode)


def command_checkpoint(config: dict[str, Any], _: argparse.Namespace) -> None:
    """Audit available real-data artifacts without claiming missing data passed."""
    symbol = normalize_symbol(config["symbol"])
    raw_files = sorted(
        resolve_config_path(config, "raw").glob(f"tick_{symbol}_*.csv.gz")
    )
    processed_files = sorted(
        resolve_config_path(config, "processed").glob(
            f"tick_processed_{symbol}_*.parquet"
        )
    )
    feature_files = sorted(
        resolve_config_path(config, "features").glob(f"features_{symbol}_*.parquet")
    )
    raw_dates = {date_token(path) for path in raw_files}
    processed_dates = {date_token(path) for path in processed_files}
    feature_dates = {date_token(path) for path in feature_files}
    missing_processed_dates = sorted(raw_dates - processed_dates)
    missing_feature_dates = sorted(raw_dates - feature_dates)

    raw_records = 0
    duplicate_ids = 0
    trade_ids_monotonic = True
    previous_trade_id: int | None = None
    for path in raw_files:
        for frame in pd.read_csv(path, usecols=["trade_id"], chunksize=250_000):
            raw_records += len(frame)
            ids = pd.to_numeric(frame["trade_id"], errors="raise").to_numpy(
                dtype=np.int64,
                copy=False,
            )
            if ids.size == 0:
                continue
            if previous_trade_id is not None:
                duplicate_ids += int(ids[0] == previous_trade_id)
                trade_ids_monotonic &= bool(ids[0] > previous_trade_id)
            differences = np.diff(ids)
            duplicate_ids += int(np.count_nonzero(differences == 0))
            trade_ids_monotonic &= bool(np.all(differences > 0))
            previous_trade_id = int(ids[-1])

    removed_records = 0
    input_records = 0
    quality_report_count = 0
    processing_is_causal = True
    reports_dir = resolve_config_path(config, "reports")
    for path in reports_dir.glob(f"data_quality_{symbol}_*.txt"):
        report = json.loads(path.read_text(encoding="utf-8"))
        quality_report_count += 1
        input_records += int(report["input_records"])
        removed_records += int(report["removed_records"])
        processing_is_causal &= (
            report.get("rolling_window_alignment")
            == "trailing_with_past_only_regime_confirmation"
        )
    removed_fraction = removed_records / input_records if input_records else None

    feature_rows = 0
    feature_values_finite = True
    obi_signal_present = False
    processed_rows_by_date = {
        date_token(path): pq.ParquetFile(path).metadata.num_rows
        for path in processed_files
    }
    feature_rows_by_date: dict[str, int] = {}
    pair_count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    minimum_lob_match_fraction = float(
        config.get("features", {}).get("minimum_lob_match_fraction", 0.95)
    )
    imbalance_metadata_complete = bool(feature_files)
    synthetic_is_causal = True
    trade_intensity_is_causal = bool(feature_files)
    lob_match_fractions: list[float] = []
    obi_valid_fractions: list[float] = []
    source_counts: dict[str, int] = {}
    for path in feature_files:
        token = date_token(path)
        metadata_path = reports_dir / f"feature_metadata_{symbol}_{token}.json"
        if not metadata_path.exists():
            imbalance_metadata_complete = False
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trade_intensity_is_causal &= bool(
            metadata.get("trade_intensity_causal", False)
        )
        source = metadata.get("obi_source")
        if source is None:
            source = (
                "lob_order_book"
                if metadata.get("lob_source") == "real"
                else "trade_volume_imbalance"
                if metadata.get("lob_source") == "synthetic_trade_imbalance"
                else "unknown"
            )
        source_counts[source] = source_counts.get(source, 0) + 1
        valid_fraction = float(metadata.get("obi_valid_fraction", 0.0))
        obi_valid_fractions.append(valid_fraction)
        imbalance_metadata_complete &= valid_fraction >= 0.95
        if source == "lob_order_book":
            match_fraction = float(metadata.get("lob_match_fraction", 0.0))
            lob_match_fractions.append(match_fraction)
            imbalance_metadata_complete &= bool(metadata.get("lob_attached", False))
            imbalance_metadata_complete &= (
                match_fraction >= minimum_lob_match_fraction
            )
        elif source == "trade_volume_imbalance":
            synthetic_is_causal &= bool(metadata.get("obi_causal", False))
            synthetic_is_causal &= int(metadata.get("obi_lag_trades", 0)) >= 1
            synthetic_is_causal &= int(metadata.get("obi_window_trades", 0)) >= 2
        else:
            imbalance_metadata_complete = False

    def accumulate_pairs(x: np.ndarray, y: np.ndarray) -> None:
        nonlocal pair_count, sum_x, sum_y, sum_xx, sum_yy, sum_xy
        pair_count += len(x)
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xx += float(np.dot(x, x))
        sum_yy += float(np.dot(y, y))
        sum_xy += float(np.dot(x, y))

    for path in feature_files:
        parquet = pq.ParquetFile(path)
        feature_rows += parquet.metadata.num_rows
        feature_rows_by_date[date_token(path)] = parquet.metadata.num_rows
        previous_direction: float | None = None
        previous_segment: int | None = None
        for batch in parquet.iter_batches(batch_size=250_000):
            frame = batch.to_pandas()
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            feature_values_finite &= bool(np.isfinite(numeric).all())
            feature_values_finite &= not frame.isna().any().any()
            if "obi" in frame:
                obi_signal_present |= bool(
                    frame["obi"].abs().gt(np.finfo(float).eps).any()
                )
            if (
                "tick_direction" not in frame
                or "segment_id" not in frame
                or frame.empty
            ):
                continue
            direction = frame["tick_direction"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
            segments = frame["segment_id"].to_numpy(dtype=np.int64, copy=False)
            if previous_direction is not None and segments[0] == previous_segment:
                accumulate_pairs(
                    direction[:1],
                    np.asarray([previous_direction], dtype=np.float64),
                )
            if len(direction) > 1:
                same_segment = segments[1:] == segments[:-1]
                accumulate_pairs(
                    direction[1:][same_segment],
                    direction[:-1][same_segment],
                )
            previous_direction = float(direction[-1])
            previous_segment = int(segments[-1])

    rho_1 = None
    rho_1_p_value = None
    if pair_count >= 3:
        covariance = pair_count * sum_xy - sum_x * sum_y
        variance_x = pair_count * sum_xx - sum_x**2
        variance_y = pair_count * sum_yy - sum_y**2
        denominator = np.sqrt(variance_x * variance_y)
        rho_1 = float(covariance / denominator) if denominator > 0 else 0.0
        if abs(rho_1) < 1.0 and denominator > 0:
            statistic = rho_1 * np.sqrt(
                (pair_count - 2) / max(1.0 - rho_1**2, np.finfo(float).eps)
            )
            rho_1_p_value = float(
                2.0 * student_t.sf(abs(statistic), df=pair_count - 2)
            )
        else:
            rho_1_p_value = 0.0 if denominator > 0 else 1.0

    test_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_data_pipeline.py",
            "-q",
            f"--basetemp={ROOT / 'reports' / '.pytest-tmp-phase2'}",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    checks: list[tuple[str, str, str]] = [
        (
            "At least 7 raw daily files",
            "PASS" if len(raw_files) >= 7 else "PENDING",
            str(len(raw_files)),
        ),
        (
            "At least 1,000,000 raw records",
            "PASS" if raw_records >= 1_000_000 else "PENDING",
            f"{raw_records:,}",
        ),
        (
            "Every raw date has processed and feature artifacts",
            (
                "PASS"
                if raw_dates
                and not missing_processed_dates
                and not missing_feature_dates
                else "FAIL"
                if raw_dates
                else "PENDING"
            ),
            (
                f"missing processed={missing_processed_dates or 'none'}, "
                f"missing features={missing_feature_dates or 'none'}"
            ),
        ),
        (
            "No duplicate trade_id and IDs increase across raw files",
            (
                "PASS"
                if raw_files and duplicate_ids == 0 and trade_ids_monotonic
                else "FAIL"
                if raw_files
                else "PENDING"
            ),
            f"duplicates={duplicate_ids}, monotonic={trade_ids_monotonic}",
        ),
        (
            "Cleaning removed less than 2%",
            (
                "PASS"
                if removed_fraction is not None and removed_fraction < 0.02
                else "FAIL"
                if removed_fraction is not None
                else "PENDING"
            ),
            "n/a" if removed_fraction is None else f"{removed_fraction:.6%}",
        ),
        (
            "Cleaning uses causal trailing windows",
            (
                "PASS"
                if quality_report_count and processing_is_causal
                else "FAIL"
                if quality_report_count
                else "PENDING"
            ),
            f"{quality_report_count} quality reports",
        ),
        (
            "Feature matrices are finite",
            (
                "PASS"
                if feature_files and feature_values_finite
                else "FAIL"
                if feature_files
                else "PENDING"
            ),
            f"{len(feature_files)} files / {feature_rows:,} rows",
        ),
        (
            "Feature artifacts match processed tick rows by date",
            (
                "PASS"
                if feature_rows_by_date
                and feature_rows_by_date == processed_rows_by_date
                else "FAIL"
                if feature_files
                else "PENDING"
            ),
            (
                f"processed={sum(processed_rows_by_date.values()):,}, "
                f"features={feature_rows:,}"
            ),
        ),
        (
            "Imbalance features have documented provenance and coverage",
            (
                "PASS"
                if imbalance_metadata_complete and obi_signal_present
                else "FAIL"
                if feature_files
                else "PENDING"
            ),
            (
                "n/a"
                if not source_counts
                else (
                    f"sources={source_counts}, "
                    f"min valid={min(obi_valid_fractions):.2%}, "
                    f"nonzero imbalance={obi_signal_present}"
                )
            ),
        ),
        (
            "Synthetic trade imbalance is causal",
            (
                "PASS"
                if source_counts.get("trade_volume_imbalance", 0) == 0
                or synthetic_is_causal
                else "FAIL"
            ),
            (
                f"{source_counts.get('trade_volume_imbalance', 0)} files, "
                f"causal={synthetic_is_causal}"
            ),
        ),
        (
            "Trade intensity is causal",
            (
                "PASS"
                if trade_intensity_is_causal
                else "FAIL"
                if feature_files
                else "PENDING"
            ),
            (
                f"{len(feature_files)} files, "
                f"causal={trade_intensity_is_causal}"
            ),
        ),
        (
            "Tick-direction lag-1 autocorrelation has p < 0.05",
            (
                "PASS"
                if rho_1_p_value is not None and rho_1_p_value < 0.05
                else "FAIL"
                if rho_1_p_value is not None
                else "PENDING"
            ),
            (
                "n/a"
                if rho_1_p_value is None
                else f"rho_1={rho_1:.6f}, p={rho_1_p_value:.6g}"
            ),
        ),
        (
            "All Phase 2 pytest tests pass",
            "PASS" if test_result.returncode == 0 else "FAIL",
            test_result.stdout.strip().splitlines()[-1]
            if test_result.stdout.strip()
            else f"exit code {test_result.returncode}",
        ),
    ]

    rows = "\n".join(
        f"| {name} | {status} | {value} |" for name, status, value in checks
    )
    output = reports_dir / "phase2_checkpoint.md"
    output.write_text(
        f"""# Phase 2 checkpoint

This report audits real-data artifacts only. Synthetic pytest fixtures do not count
toward the one-million-record checkpoint.

| Check | Status | Observed |
|---|---|---:|
{rows}

Run the logic tests with:

```text
python -m pytest tests/test_data_pipeline.py -v
```
""",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))
    if any(status != "PASS" for _, status, _ in checks):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="Project-relative YAML configuration path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download daily Binance trades.")
    download.add_argument("--start-date")
    download.add_argument("--end-date")
    download.add_argument("--skip-missing", action="store_true")
    download.set_defaults(handler=command_download)

    lob = subparsers.add_parser("collect-lob", help="Collect REST LOB snapshots.")
    lob.add_argument("--snapshots", type=int, default=60)
    lob.add_argument("--output")
    lob.set_defaults(handler=command_collect_lob)

    live = subparsers.add_parser(
        "collect-live",
        help="Collect synchronized trade and LOB WebSocket streams.",
    )
    live.add_argument("--duration-seconds", type=float, default=120.0)
    live.add_argument("--tick-output")
    live.add_argument("--lob-output")
    live.set_defaults(handler=command_collect_live)

    process = subparsers.add_parser("process", help="Clean raw daily tick files.")
    process.add_argument("--input", action="append")
    process.set_defaults(handler=command_process)

    features = subparsers.add_parser("features", help="Engineer QRW features.")
    features.add_argument("--input", action="append")
    features.add_argument(
        "--lob",
        help=(
            "Same-period LOB HDF5 file or directory. In auto mode, missing LOB "
            "files fall back to causal trade-volume imbalance."
        ),
    )
    features.add_argument(
        "--obi-source",
        choices=["auto", "lob", "trade_imbalance"],
        help="Override config.features.obi_source.",
    )
    features.add_argument(
        "--trade-imbalance-window",
        type=int,
        help="Override the trailing number of trades used by the proxy.",
    )
    features.set_defaults(handler=command_features)

    test = subparsers.add_parser("test", help="Run the Phase 2 tests.")
    test.set_defaults(handler=command_test)

    checkpoint = subparsers.add_parser(
        "checkpoint", help="Audit real-data Phase 2 artifacts."
    )
    checkpoint.set_defaults(handler=command_checkpoint)
    return parser


def main() -> None:
    configure_console()
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    args.handler(config, args)


if __name__ == "__main__":
    main()
