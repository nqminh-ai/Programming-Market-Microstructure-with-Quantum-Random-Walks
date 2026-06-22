"""Phase 7B: Cross-asset robustness check (BTC/USDT, ETH/USDT and BNB/USDT).

Runs the full Phase 2 + Phase 5 pipeline for BTC/USDT, ETH/USDT and BNB/USDT
and produces a combined cross-asset scorecard for robustness validation.

Usage
-----
    python scripts/phase7b_cross_asset.py [--skip-existing] [--n-paths 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
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
from src.evaluation.benchmark_suite import BenchmarkSuite
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
    from scripts.phase7a_data_expansion import (
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
    # Step 4: Pick holdout day (last available day)
    # ------------------------------------------------------------------
    store_path = features_dir / f"multiday_{sym}.parquet"
    build_multiday_store(feature_paths, store_path)
    
    # Sort files chronologically and pick the last one
    last_feature_file = sorted(feature_paths)[-1]
    # Extract date
    parts = last_feature_file.stem.split("_")
    benchmark_day = next((p for p in parts if "-" in p and len(p) == 10), "unknown")
    benchmark_frame = pd.read_parquet(last_feature_file)
    
    if len(benchmark_frame) < 200:
        print(f"  [WARN] Benchmark day {benchmark_day} has < 200 rows ({len(benchmark_frame)})")
        return {"asset": symbol, "status": "insufficient_data"}

    print(f"\n--- Benchmark: {symbol} on day {benchmark_day} ({len(benchmark_frame):,} rows) ---")

    # ------------------------------------------------------------------
    # Step 5: Run benchmark + statistical tests
    # ------------------------------------------------------------------
    results_dir = ROOT / "results" / sym.lower()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "figures" / sym.lower()
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        suite = BenchmarkSuite(
            benchmark_frame,
            n_steps=n_steps,
            n_paths=n_paths,
            random_seed=random_seed,
        )
        suite.run()

        tests = StatisticalTestSuite(
            suite.holdout["price"].to_numpy(dtype=np.float64),
            suite.simulated_paths,
            random_seed=random_seed,
            bootstrap_iterations=500,
        )
        stat_results = tests.run_all(
            results_dir=results_dir,
            figures_dir=figures_dir,
        )
        tests.run_model_selection_tests(
            suite.model_comparison,
            output=results_dir / "model_selection_corrected.csv",
        )

        _, scorecard = ResultsCompiler().compile(
            stat_results,
            comparison_output=results_dir / "final_comparison_table.csv",
            scorecard_output=results_dir / "scorecard.csv",
        )
        qrw_rank = int(scorecard.loc[scorecard["model"] == "QRW Adaptive", "overall_rank"].iloc[0])
        top_model = str(scorecard.iloc[0]["model"])

        print(f"  Top model: {top_model}")
        print(f"  QRW rank: {qrw_rank}")

        return {
            "asset": symbol,
            "status": "ok",
            "benchmark_day": benchmark_day,
            "train_rows": len(suite.train),
            "holdout_rows": len(suite.holdout),
            "qrw_overall_rank": qrw_rank,
            "qrw_mean_rank": float(
                scorecard.loc[scorecard["model"] == "QRW Adaptive", "mean_rank"].iloc[0]
            ),
            "top_model": top_model,
            "scorecard": scorecard.to_dict(orient="records"),
        }

    except Exception as exc:  # noqa: BLE001
        print(f"  [ERR] Benchmark failed for {symbol}: {exc}")
        return {"asset": symbol, "status": "benchmark_error", "error": str(exc)}


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
                "mean_rank": entry["mean_rank"],
                "overall_rank": entry["overall_rank"],
            })
    if not rows:
        print("[WARN] No valid scorecard rows to aggregate")
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # Average rank per model across assets
    avg_rank = (
        frame.groupby("model")
        .agg(
            avg_mean_rank=("mean_rank", "mean"),
            avg_overall_rank=("overall_rank", "mean"),
            n_assets=("asset", "count"),
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
    parser.add_argument("--skip-existing", action="store_true", default=True)
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
    summary_path = ROOT / "results" / "cross_asset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n[SUMMARY] Saved: {summary_path}")

    # Build cross-asset scorecard
    scorecard_path = ROOT / "results" / "cross_asset_scorecard.csv"
    build_cross_asset_scorecard(summaries, scorecard_path)

    print("\n=== Phase 7B Complete ===")
    for s in summaries:
        status = s.get("status", "unknown")
        qrw_rank = s.get("qrw_overall_rank", "N/A")
        print(f"  {s['asset']}: status={status}, QRW rank={qrw_rank}")


if __name__ == "__main__":
    main()
