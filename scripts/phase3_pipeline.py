"""Calibrate, validate, and benchmark the Phase 3 QRW implementation."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adaptive_market_qrw import AdaptiveDecoherenceQRW


MARKET_COLUMNS = [
    "timestamp",
    "price",
    "tick_direction",
    "obi",
    "trade_intensity",
]


def load_calibration_rows(path: Path, max_rows: int) -> pd.DataFrame:
    """Read only the columns and batches required for calibration."""
    parquet = pq.ParquetFile(path)
    columns = MARKET_COLUMNS.copy()
    if "obi_valid" in parquet.schema_arrow.names:
        columns.append("obi_valid")
    if "segment_id" in parquet.schema_arrow.names:
        columns.append("segment_id")
    frames: list[pd.DataFrame] = []
    rows_read = 0
    for batch in parquet.iter_batches(
        batch_size=min(max_rows, 100_000),
        columns=columns,
    ):
        remaining = max_rows - rows_read
        if remaining <= 0:
            break
        frame = batch.slice(0, remaining).to_pandas()
        frames.append(frame)
        rows_read += len(frame)
        if rows_read >= max_rows:
            break
    if not frames:
        raise ValueError(f"no calibration rows found in {path}")
    return pd.concat(frames, ignore_index=True)


def resolve_feature_path(explicit: Path | None) -> Path:
    """Return an explicit feature file or the newest file with varying OBI."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit

    candidates = sorted(
        Path("data/features").glob("features_*.parquet"),
        reverse=True,
    )
    for candidate in candidates:
        parquet = pq.ParquetFile(candidate)
        minimum = np.inf
        maximum = -np.inf
        for batch in parquet.iter_batches(batch_size=100_000, columns=["obi"]):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            if len(values):
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))
            if maximum > minimum:
                return candidate
    raise FileNotFoundError(
        "no feature file with varying OBI was found; pass --feature-path"
    )


def benchmark_paths(
    *,
    market_data: pd.DataFrame,
    parameters: dict[str, object],
    n_steps: int,
    n_paths: int,
    random_seed: int,
) -> dict[str, float | int | bool]:
    """Benchmark measured adaptive-decoherence QRW trajectories."""
    if market_data.empty:
        raise ValueError("market_data cannot be empty")
    indices = np.arange(n_steps) % len(market_data)
    benchmark_data = market_data.iloc[indices].copy().reset_index(drop=True)
    benchmark_data["timestamp"] = np.arange(n_steps, dtype=np.int64)
    model = AdaptiveDecoherenceQRW(
        benchmark_data,
        {
            "n_positions": 2 * n_steps + 1,
            "gamma_base": parameters["gamma"],
            "alpha_obi": parameters["alpha_obi"],
            "alpha_direction": parameters["alpha_direction"],
            "alpha_obi_change": parameters["alpha_obi_change"],
            "alpha_abs_obi": parameters["alpha_abs_obi"],
            "gamma_intensity": parameters["gamma_intensity"],
            "obi_bias": parameters["obi_bias"],
            "feature_mean": parameters["feature_mean"],
            "feature_scale": parameters["feature_scale"],
            "movement_probability": parameters["movement_probability"],
        },
    )

    started = time.perf_counter()
    paths = model.simulate_price_path(
        n_paths,
        T=n_steps,
        random_state=random_seed,
    )
    elapsed = time.perf_counter() - started
    increments = np.diff(paths, axis=1)
    paths_are_local = bool(
        np.all(np.abs(paths[:, 0]) <= 1)
        and np.all(np.abs(increments) <= 1)
    )
    all_increments = np.column_stack([paths[:, 0], increments])

    return {
        "n_steps": n_steps,
        "n_paths": n_paths,
        "elapsed_seconds": elapsed,
        "target_seconds": 60.0,
        "target_passed": elapsed < 60.0,
        "paths_are_local": paths_are_local,
        "max_absolute_increment": int(np.max(np.abs(increments), initial=0)),
        "movement_probability": float(parameters["movement_probability"]),
        "simulated_move_fraction": float(np.mean(all_increments != 0)),
        "sample_mean_final": float(np.mean(paths[:, -1])),
        "sample_variance_final": float(np.var(paths[:, -1])),
    }


def write_benchmark_report(
    path: Path,
    *,
    feature_path: Path,
    parameters: dict[str, object],
    market_normalization_error: float,
    market_elapsed_seconds: float,
    benchmark: dict[str, float | int | bool],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Phase 3 QRW performance benchmark",
        f"Python: {platform.python_version()}",
        f"NumPy: {np.__version__}",
        f"Feature source: {feature_path}",
        f"Calibration rows: {parameters['calibration_rows']}",
        f"rho_1: {parameters['rho_1']:.12g}",
        f"gamma: {parameters['gamma']:.12g}",
        f"alpha_obi: {parameters['alpha_obi']:.12g}",
        f"alpha_direction: {parameters['alpha_direction']:.12g}",
        f"alpha_obi_change: {parameters['alpha_obi_change']:.12g}",
        f"alpha_abs_obi: {parameters['alpha_abs_obi']:.12g}",
        f"gamma_intensity: {parameters['gamma_intensity']:.12g}",
        f"obi_bias: {parameters['obi_bias']:.12g}",
        f"Calibration method: {parameters['calibration_method']}",
        f"Calibration status: {parameters['calibration_status']}",
        (
            "Selected regularization: "
            f"{parameters['selected_regularization']:.12g}"
        ),
        (
            "Validation log loss: "
            f"{parameters['validation_log_loss']:.12g}"
        ),
        f"Validation Brier: {parameters['validation_brier']:.12g}",
        (
            "Market simulation max normalization error: "
            f"{market_normalization_error:.3e}"
        ),
        f"Exact density simulation seconds: {market_elapsed_seconds:.6f}",
        f"Benchmark paths: {benchmark['n_paths']}",
        f"Benchmark steps: {benchmark['n_steps']}",
        f"Elapsed seconds: {benchmark['elapsed_seconds']:.6f}",
        f"Target seconds: {benchmark['target_seconds']:.1f}",
        f"Target passed: {benchmark['target_passed']}",
        (
            "Paths use only local {-1, 0, +1} moves: "
            f"{benchmark['paths_are_local']}"
        ),
        f"Maximum absolute path increment: {benchmark['max_absolute_increment']}",
        (
            "Calibrated event move probability: "
            f"{benchmark['movement_probability']:.6f}"
        ),
        (
            "Simulated event move fraction: "
            f"{benchmark['simulated_move_fraction']:.6f}"
        ),
        f"Final sampled mean: {benchmark['sample_mean_final']:.6f}",
        f"Final sampled variance: {benchmark['sample_variance_final']:.6f}",
        "",
        (
            "Method: AdaptiveDecoherenceQRW with causal microstructure signal, "
            "trade-intensity-modulated basis dephasing, adaptive coin-shift, "
            "and local projective position measurement."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Defaults to the newest feature file with varying OBI.",
    )
    parser.add_argument("--max-calibration-rows", type=int, default=250_000)
    parser.add_argument("--calibrated-output", type=Path, default=Path("results/calibrated_params.json"))
    parser.add_argument("--benchmark-output", type=Path, default=Path("reports/performance_benchmark.txt"))
    parser.add_argument("--market-steps", type=int, default=50)
    parser.add_argument("--benchmark-steps", type=int, default=1_000)
    parser.add_argument("--benchmark-paths", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_calibration_rows < 2:
        raise ValueError("max-calibration-rows must be at least 2")
    if args.market_steps < 1:
        raise ValueError("market-steps must be positive")

    feature_path = resolve_feature_path(args.feature_path)
    data = load_calibration_rows(feature_path, args.max_calibration_rows)
    model = AdaptiveDecoherenceQRW(
        data,
        {
            "n_positions": 2 * args.market_steps + 1,
            "gamma_base": 0.0,
        },
    )
    parameters = model.calibrate_two_stage(args.calibrated_output)
    market_started = time.perf_counter()
    simulation = model.simulate(args.market_steps)
    market_elapsed_seconds = time.perf_counter() - market_started
    totals = simulation.groupby("t", sort=False)["probability"].sum().to_numpy()
    market_normalization_error = float(np.max(np.abs(totals - 1.0)))

    benchmark = benchmark_paths(
        market_data=data,
        parameters=parameters,
        n_steps=args.benchmark_steps,
        n_paths=args.benchmark_paths,
        random_seed=args.random_seed,
    )
    write_benchmark_report(
        args.benchmark_output,
        feature_path=feature_path,
        parameters=parameters,
        market_normalization_error=market_normalization_error,
        market_elapsed_seconds=market_elapsed_seconds,
        benchmark=benchmark,
    )

    print(f"Calibration: {args.calibrated_output}")
    print(f"Benchmark: {args.benchmark_output}")
    print(f"Elapsed: {benchmark['elapsed_seconds']:.6f}s")
    print(f"Target passed: {benchmark['target_passed']}")


if __name__ == "__main__":
    main()
