"""Audit script to run the Phase 4 Quantum Benchmark Suite."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.benchmark_suite import BenchmarkSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument(
        "--output-benchmark",
        type=Path,
        default=Path("reports/phase4_benchmark.csv"),
    )
    parser.add_argument(
        "--output-comparison",
        type=Path,
        default=Path("reports/phase4_comparison.csv"),
    )
    parser.add_argument(
        "--output-garch",
        type=Path,
        default=Path("reports/phase4_garch.json"),
    )
    parser.add_argument(
        "--output-diagnostics",
        type=Path,
        default=Path("reports/phase4_diagnostics.json"),
    )
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-paths", type=int, default=1_000)
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
    print(f"Running Phase 4 BenchmarkSuite with {args.n_steps} steps and {args.n_paths} paths...")

    # For the benchmark suite, we can use the whole frame if it manages splits, 
    # or pass a subset. Wait, BenchmarkSuite splits internally if no holdout is provided.
    suite = BenchmarkSuite(
        frame,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )

    args.output_benchmark.parent.mkdir(parents=True, exist_ok=True)
    
    suite.run(
        benchmark_output=args.output_benchmark,
        comparison_output=args.output_comparison,
        garch_output=args.output_garch,
        diagnostics_output=args.output_diagnostics,
    )

    print("Phase 4 Benchmark generation complete.")
    print(f"Diagnostics written to {args.output_diagnostics}")


if __name__ == "__main__":
    main()
