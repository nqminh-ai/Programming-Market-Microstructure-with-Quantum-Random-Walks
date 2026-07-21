"""Phase 5 Statistical Audit: Confirming the Quantum Edge."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.statistical_tests import StatisticalTestSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports/phase5_results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/phase5_figures"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/phase5_comparison.csv"),
    )
    parser.add_argument(
        "--scorecard-output",
        type=Path,
        default=Path("reports/phase5_scorecard.csv"),
    )
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--tail", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    # Load dataset
    frame = pd.read_parquet(feature_path).sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)
    
    if args.tail is not None:
        frame = frame.tail(args.tail).reset_index(drop=True)

    print(f"Loaded {len(frame)} rows from {feature_path.name}")
    print("Running BenchmarkSuite to generate forecast samples...")
    
    bench_suite = BenchmarkSuite(
        frame,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )
    # We must run it to populate internal states
    bench_suite.run()
    
    print("Benchmark complete. Initiating StatisticalTestSuite...")
    
    # We need the empirical realized path for the same timeframe.
    # The benchmark test starts from the initial price of the test set, but 
    # test set has length n_steps + 1. The first is origin.
    empirical = bench_suite.test["price"].to_numpy(dtype=np.float64)

    stat_suite = StatisticalTestSuite(
        empirical_prices=empirical,
        forecast_samples=bench_suite.forecast_samples,
        rolling_one_step_losses=bench_suite.rolling_one_step_losses,
        rolling_one_step_timestamps=bench_suite.rolling_one_step_timestamps,
        bootstrap_iterations=args.bootstrap_iterations,
        max_lag=10,
        random_seed=args.random_seed,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = stat_suite.run_all(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    
    print("Compiling scorecard and comparison results...")
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    
    comparison, scorecard = ResultsCompiler().compile(
        results,
        comparison_output=args.comparison_output,
        scorecard_output=args.scorecard_output,
    )
    
    print("Phase 5 Statistical Audit complete.")
    print(f"Results written to {args.results_dir}")


if __name__ == "__main__":
    main()
