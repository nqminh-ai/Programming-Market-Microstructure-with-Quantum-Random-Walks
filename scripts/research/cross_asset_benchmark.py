"""Phase 7B: multi-day robustness check for BTC, ETH, and BNB.

Runs the full Phase 2 + Phase 5 pipeline for ETH/USDT and BNB/USDT and
produces a combined cross-asset scorecard for robustness validation.

Usage
-----
    python scripts/research/cross_asset_benchmark.py [--skip-existing] [--n-paths 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import normalize_symbol
from src.data.feature_engineer import FeatureEngineer
from src.data.tick_downloader import TickDownloader
from src.data.tick_processor import TickProcessor
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.directional_baselines import (
    directional_events,
    fit_directional_baselines,
    score_directional_baselines,
)
from src.evaluation.multiday import (
    day_cluster_bootstrap,
    day_cluster_sensitivity,
    paired_day_comparisons,
)
from src.evaluation.provenance import (
    canonical_repo_path,
    git_commit,
    sha256_file,
)
from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.statistical_tests import StatisticalTestSuite


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


def resolve_path(config: dict, key: str) -> Path:
    value = config["paths"][key]
    p = ROOT / value
    p.mkdir(parents=True, exist_ok=True)
    return p


def combine_training_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine earlier UTC days without allowing features across boundaries."""
    combined: list[pd.DataFrame] = []
    for day_index, frame in enumerate(frames):
        value = frame.copy()
        if "segment_id" in value:
            value["segment_id"] = [
                f"day{day_index}:{segment}" for segment in value["segment_id"]
            ]
        else:
            value["segment_id"] = f"day{day_index}:0"
        combined.append(value)
    return (
        pd.concat(combined, ignore_index=True)
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )


DayEntry = tuple[str, Path, pd.DataFrame]


@dataclass(frozen=True)
class ExpandingDayFold:
    """One causal UTC-day fold with a validation day before the test day."""

    training: tuple[DayEntry, ...]
    validation: DayEntry
    test: DayEntry


def expanding_day_folds(
    days: list[DayEntry], *, minimum_training_days: int = 3
) -> list[ExpandingDayFold]:
    """Build all expanding folds without reusing validation/test days in train."""
    if minimum_training_days < 1:
        raise ValueError("minimum_training_days must be positive")

    ordered = sorted(days, key=lambda entry: entry[0])
    labels = [entry[0] for entry in ordered]
    if len(labels) != len(set(labels)):
        raise ValueError("UTC-day labels must be unique")

    return [
        ExpandingDayFold(
            training=tuple(ordered[: test_index - 1]),
            validation=ordered[test_index - 1],
            test=ordered[test_index],
        )
        for test_index in range(minimum_training_days + 1, len(ordered))
    ]


# ---------------------------------------------------------------------------
# Single-asset pipeline wrapper
# ---------------------------------------------------------------------------

def run_asset_pipeline(
    config_path: Path,
    *,
    n_paths: int = 2_000,
    n_steps: int = 500,
    random_seed: int = 2026,
    skip_existing: bool = True,
) -> dict:
    """Download, process, featurize, and benchmark one asset. Returns summary."""
    config = load_config(config_path)
    symbol = config["symbol"]
    sym = normalize_symbol(symbol)

    raw_dir = resolve_path(config, "raw")
    processed_dir = resolve_path(config, "processed")
    features_dir = resolve_path(config, "features")
    reports_dir = ROOT / config["paths"].get("reports", f"reports/{sym.lower()}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Asset: {symbol}")
    print(f"  Config: {config_path.name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    from scripts.research.data_expansion import (
        date_range,
        download_day,
        process_day,
        featurize_day,
        build_multiday_store,
    )
    from datetime import date

    downloader = TickDownloader(base_url=config["historical_base_url"])
    days = date_range(config["start_date"], config["end_date"])

    print(f"\n--- Download ({len(days)} days) ---")
    downloaded = []
    for day in days:
        result = download_day(symbol, day, raw_dir, downloader, skip_missing=True)
        if result is not None:
            downloaded.append(result)
    print(f"  Found: {len(downloaded)} raw files")

    # ------------------------------------------------------------------
    # Step 2: Process
    # ------------------------------------------------------------------
    proc_cfg = config.get("processing", {})
    processor = TickProcessor(
        rolling_window=int(proc_cfg.get("rolling_window", 100)),
        outlier_z_score=float(proc_cfg.get("outlier_z_score", 3.0)),
        gap_threshold_seconds=float(proc_cfg.get("gap_threshold_seconds", 300)),
    )
    print(f"\n--- Process ({len(downloaded)} files) ---")
    processed_paths = []
    for raw_path in sorted(downloaded):
        result = process_day(
            raw_path, symbol, processed_dir, reports_dir, processor, skip_existing
        )
        if result is not None:
            processed_paths.append(result)
    print(f"  Processed: {len(processed_paths)} files")

    # ------------------------------------------------------------------
    # Step 3: Featurize
    # ------------------------------------------------------------------
    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(
        autocorrelation_lags=int(feat_cfg.get("autocorrelation_lags", 20)),
        lob_tolerance_seconds=float(feat_cfg.get("lob_tolerance_seconds", 5)),
        minimum_lob_match_fraction=float(feat_cfg.get("minimum_lob_match_fraction", 0.95)),
        trade_imbalance_window=int(feat_cfg.get("trade_imbalance_window", 100)),
        trade_imbalance_min_periods=int(feat_cfg.get("trade_imbalance_min_periods", 20)),
    )
    obi_source = feat_cfg.get("obi_source", "auto")

    print(f"\n--- Featurize ({len(processed_paths)} files) ---")
    feature_paths = []
    for proc_path in sorted(processed_paths):
        result = featurize_day(
            proc_path, symbol, features_dir, reports_dir,
            engineer, raw_dir, obi_source, skip_existing,
        )
        if result is not None:
            feature_paths.append(result)
    print(f"  Feature files: {len(feature_paths)}")

    if not feature_paths:
        print(f"  [WARN] No features for {symbol}. Skipping benchmark.")
        return {"asset": symbol, "status": "no_data"}

    # ------------------------------------------------------------------
    # Step 4: Evaluate every available day; never select one favorable day
    # ------------------------------------------------------------------
    store_path = features_dir / f"multiday_{sym}.parquet"
    build_multiday_store(feature_paths, store_path)
    daily_results: list[dict] = []
    failures: list[dict[str, str]] = []
    eligible_days: list[tuple[str, Path, pd.DataFrame]] = []
    for feature_file in sorted(feature_paths):
        parts = feature_file.stem.split("_")
        benchmark_day = next(
            (part for part in parts if "-" in part and len(part) == 10),
            "unknown",
        )
        frame = pd.read_parquet(feature_file)
        if len(frame) < 200:
            failures.append(
                {
                    "day": benchmark_day,
                    "reason": f"insufficient_rows:{len(frame)}",
                }
            )
            continue
        eligible_days.append((benchmark_day, feature_file, frame))

    minimum_training_days = 3
    folds = expanding_day_folds(
        eligible_days, minimum_training_days=minimum_training_days
    )
    for fold_index, fold in enumerate(folds):
        training_days = fold.training
        validation_day, validation_file, validation_frame = fold.validation
        benchmark_day, feature_file, benchmark_frame = fold.test
        training_frame = combine_training_days(
            [entry[2] for entry in training_days]
        )

        print(
            f"\n--- Benchmark: {symbol}; train={len(training_days)} days, "
            f"validation={validation_day}, test={benchmark_day} ---"
        )
        results_dir = ROOT / "results" / "cross_asset" / sym.lower() / benchmark_day
        figures_dir = ROOT / "figures" / "cross_asset" / sym.lower() / benchmark_day
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            day_seed = random_seed + fold_index
            train_events = directional_events(training_frame)
            validation_events = directional_events(validation_frame)
            test_events = directional_events(benchmark_frame)
            directional_models = fit_directional_baselines(
                train_events, validation_events
            )
            directional_summary, directional_predictions, directional_diagnostics = (
                score_directional_baselines(
                    directional_models,
                    test_events,
                    train_events=len(train_events),
                    validation_events=len(validation_events),
                )
            )
            directional_summary.to_csv(
                results_dir / "strong_directional_baselines.csv", index=False
            )
            directional_predictions.to_csv(
                results_dir / "strong_directional_predictions.csv", index=False
            )
            directional_diagnostics.update(
                {
                    "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
                    "code_commit": git_commit(ROOT),
                    "feature_path": canonical_repo_path(feature_file, ROOT),
                    "feature_sha256": sha256_file(feature_file),
                    "split_unit": "UTC_day",
                    "training_days": [entry[0] for entry in training_days],
                    "validation_day": validation_day,
                    "test_day": benchmark_day,
                    "input_feature_sha256": {
                        canonical_repo_path(path, ROOT): sha256_file(path)
                        for _, path, _ in (
                            *training_days,
                            (validation_day, validation_file, validation_frame),
                            (benchmark_day, feature_file, benchmark_frame),
                        )
                    },
                }
            )
            (results_dir / "strong_directional_diagnostics.json").write_text(
                json.dumps(directional_diagnostics, indent=2),
                encoding="utf-8",
            )
            suite = BenchmarkSuite(
                training_frame,
                holdout_data=benchmark_frame,
                n_steps=n_steps,
                n_paths=n_paths,
                random_seed=day_seed,
            )
            suite.run()

            tests = StatisticalTestSuite(
                suite.test["price"].to_numpy(dtype=np.float64),
                suite.forecast_samples,
                sample_semantics=suite.sample_semantics,
                rolling_one_step_losses=suite.rolling_one_step_losses,
                rolling_one_step_timestamps=suite.rolling_one_step_timestamps,
                random_seed=day_seed,
                bootstrap_iterations=500,
            )
            stat_results = tests.run_all(
                results_dir=results_dir,
                figures_dir=figures_dir,
            )
            _, scorecard = ResultsCompiler().compile(
                stat_results,
                comparison_output=results_dir / "final_comparison_table.csv",
                scorecard_output=results_dir / "scorecard.csv",
            )
            daily_results.append(
                {
                    "day": benchmark_day,
                    "training_days": [entry[0] for entry in training_days],
                    "validation_day": validation_day,
                    "train_rows": len(suite.train),
                    "holdout_rows": len(suite.holdout),
                    "scorecard": scorecard.to_dict(orient="records"),
                    "directional_baselines": directional_summary.to_dict(
                        orient="records"
                    ),
                    "directional_diagnostics": directional_diagnostics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"day": benchmark_day, "reason": str(exc)})

    if not daily_results:
        return {
            "asset": symbol,
            "status": "benchmark_error",
            "evaluated_days": 0,
            "failures": failures,
        }

    if len(daily_results) < 2:
        return {
            "asset": symbol,
            "status": "insufficient_multiday_data",
            "feature_days_available": len(feature_paths),
            "evaluated_days": len(daily_results),
            "failed_days": failures,
            "daily_results": daily_results,
        }

    model_rows = [
        {"day": result["day"], **entry}
        for result in daily_results
        for entry in result["scorecard"]
    ]
    daily_metric_frame = pd.DataFrame(model_rows)
    daily_metric_frame.to_csv(
        ROOT / "results" / "cross_asset" / sym.lower() / "daily_scorecards.csv",
        index=False,
    )
    bootstrap = day_cluster_bootstrap(
        daily_metric_frame,
        value_columns=("mean_marginal_crps", "mean_direction_log_loss"),
        iterations=2_000,
        random_seed=random_seed,
    )
    sensitivity = day_cluster_sensitivity(
        daily_metric_frame,
        value_columns=("mean_marginal_crps", "mean_direction_log_loss"),
        seeds=(random_seed, random_seed + 1, random_seed + 2),
        block_sizes=(1, 2, 5),
        iterations=500,
    )
    asset_result_dir = ROOT / "results" / "cross_asset" / sym.lower()
    bootstrap.to_csv(asset_result_dir / "day_cluster_bootstrap.csv", index=False)
    sensitivity.to_csv(
        asset_result_dir / "day_cluster_sensitivity.csv", index=False
    )
    directional_daily = pd.DataFrame(
        [
            {"day": result["day"], **entry}
            for result in daily_results
            for entry in result["directional_baselines"]
        ]
    )
    directional_daily.to_csv(
        asset_result_dir / "directional_daily_scores.csv", index=False
    )
    directional_bootstrap = day_cluster_bootstrap(
        directional_daily,
        value_columns=("brier", "log_loss", "accuracy"),
        iterations=2_000,
        random_seed=random_seed,
    )
    directional_sensitivity = day_cluster_sensitivity(
        directional_daily,
        value_columns=("brier", "log_loss"),
        seeds=(random_seed, random_seed + 1, random_seed + 2),
        block_sizes=(1, 2, 5),
        iterations=500,
    )
    directional_comparisons = paired_day_comparisons(
        directional_daily,
        metric_directions={
            "brier": "lower",
            "log_loss": "lower",
            "accuracy": "higher",
        },
        iterations=2_000,
        random_seed=random_seed,
    )
    directional_bootstrap.to_csv(
        asset_result_dir / "directional_day_cluster_bootstrap.csv", index=False
    )
    directional_sensitivity.to_csv(
        asset_result_dir / "directional_day_cluster_sensitivity.csv", index=False
    )
    directional_comparisons.to_csv(
        asset_result_dir / "directional_pairwise_tests.csv", index=False
    )
    rank_diagnostics = (
        daily_metric_frame
        .groupby("model", sort=False)
        .agg(
            mean_daily_rank=("overall_rank", "mean"),
        )
        .reset_index()
    )
    aggregate = (
        bootstrap.merge(rank_diagnostics, on="model", validate="one_to_one")
        .sort_values(
            ["mean_marginal_crps", "mean_direction_log_loss", "model"],
            kind="stable",
        )
    )
    aggregate["overall_rank"] = aggregate["mean_marginal_crps"].rank(
        method="min", ascending=True
    )
    qrw = aggregate.loc[aggregate["model"] == "QRW Adaptive"].iloc[0]
    return {
        "asset": symbol,
        "status": "ok",
        "feature_days_available": len(feature_paths),
        "evaluated_days": len(daily_results),
        "failed_days": failures,
        "qrw_overall_rank": int(qrw["overall_rank"]),
        "qrw_mean_marginal_crps": float(qrw["mean_marginal_crps"]),
        "top_model": str(aggregate.iloc[0]["model"]),
        "scorecard": aggregate.to_dict(orient="records"),
        "daily_results": daily_results,
        "day_cluster_sensitivity": sensitivity.to_dict(orient="records"),
        "directional_day_cluster_bootstrap": directional_bootstrap.to_dict(
            orient="records"
        ),
        "directional_pairwise_tests": directional_comparisons.to_dict(
            orient="records"
        ),
    }

# ---------------------------------------------------------------------------
# Cross-asset scorecard builder
# ---------------------------------------------------------------------------

def build_cross_asset_scorecard(summaries: list[dict], output_path: Path) -> pd.DataFrame:
    """Combine per-asset scorecards into one cross-asset comparison table."""
    rows = []
    for summary in summaries:
        if summary.get("status") != "ok":
            continue
        asset = summary["asset"]
        for entry in summary.get("scorecard", []):
            rows.append({
                "asset": asset,
                "model": entry["model"],
                "mean_marginal_crps": entry["mean_marginal_crps"],
                "overall_rank": entry["overall_rank"],
                "evaluated_days": entry["evaluated_days"],
            })
    if not rows:
        print("[WARN] No valid scorecard rows to aggregate")
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # Average rank per model across assets
    avg_rank = (
        frame.groupby("model")
        .agg(
            avg_mean_marginal_crps=("mean_marginal_crps", "mean"),
            avg_overall_rank=("overall_rank", "mean"),
            n_assets=("asset", "count"),
            total_evaluated_days=("evaluated_days", "sum"),
        )
        .reset_index()
        .sort_values("avg_overall_rank")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_rank.to_csv(output_path, index=False)
    print(f"\n[CROSS-ASSET] Saved: {output_path}")
    print(avg_rank.to_string(index=False))
    return avg_rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=2_000)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages whose output already exists "
        "(use --no-skip-existing to force a rebuild)",
    )
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()

    configs = {
        "BTC/USDT": ROOT / "config" / "data_config.yaml",
        "ETH/USDT": ROOT / "config" / "data_config_eth.yaml",
        "BNB/USDT": ROOT / "config" / "data_config_bnb.yaml",
    }

    print("=== Phase 7B: Cross-Asset Robustness ===")
    print(f"Assets: {list(configs.keys())}")
    print(f"n_paths: {args.n_paths}, n_steps: {args.n_steps}")

    summaries = []
    for asset_name, config_path in configs.items():
        if not config_path.exists():
            print(f"\n[SKIP] Config not found: {config_path}")
            continue
        summary = run_asset_pipeline(
            config_path,
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            random_seed=args.random_seed,
            skip_existing=args.skip_existing,
        )
        summaries.append(summary)

    # Save per-asset summaries
    cross_asset_dir = ROOT / "results" / "cross_asset"
    cross_asset_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cross_asset_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n[SUMMARY] Saved: {summary_path}")

    # Build cross-asset scorecard
    scorecard_path = cross_asset_dir / "scorecard.csv"
    build_cross_asset_scorecard(summaries, scorecard_path)

    print("\n=== Phase 7B Complete ===")
    for s in summaries:
        status = s.get("status", "unknown")
        qrw_rank = s.get("qrw_overall_rank", "N/A")
        print(f"  {s['asset']}: status={status}, QRW rank={qrw_rank}")

    failed = [s["asset"] for s in summaries if s.get("status") != "ok"]
    if not summaries or failed:
        print(f"\n[FAIL] Asset(s) did not complete successfully: {failed or 'no assets ran'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
