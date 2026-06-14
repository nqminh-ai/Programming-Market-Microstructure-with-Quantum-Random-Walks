"""Run Phase 4 classical baselines and the common benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_pipeline import resolve_feature_path
from src.evaluation.benchmark_suite import BenchmarkSuite


def write_checkpoint(
    path: Path,
    *,
    feature_path: Path,
    suite: BenchmarkSuite,
) -> None:
    diagnostics = suite.diagnostics
    coherent = diagnostics["coherent_no_decoherence"]
    status = "PASS" if diagnostics["checkpoint_passed"] else "FAIL"
    if suite.results is None:
        raise RuntimeError("benchmark results are unavailable")
    summary = suite.results.pivot(
        index="model",
        columns="metric",
        values="value",
    )
    summary_rows = [
        (
            f"| {model} | "
            f"{row['wasserstein_path_mae']:.6f} | "
            f"{row['hit_rate_h1']:.2%} | "
            f"{row['hit_rate_h5']:.2%} | "
            f"{row['hit_rate_h10']:.2%} | "
            f"{row['mean_direction_log_likelihood']:.6f} |"
        )
        for model, row in summary.iterrows()
    ]
    lines = [
        "# Phase 4 Checkpoint",
        "",
        f"**Status:** {status}",
        "",
        f"Feature source: `{feature_path}`",
        "",
        "## Protocol",
        "",
        f"- Chronological train rows: `{diagnostics['train_rows']}`",
        f"- Later test rows: `{diagnostics['test_rows']}`",
        f"- Simulated paths per model: `{diagnostics['n_paths']}`",
        f"- Steps: `{diagnostics['n_steps']}`",
        f"- Tick size: `{diagnostics['tick_size']:.8g}`",
        f"- Event move probability: `{diagnostics['movement_probability']:.4f}`",
        f"- Forecast protocol: `{diagnostics['forecast_protocol']}`",
        f"- Random seed: `{diagnostics['random_seed']}`",
        "",
        "## Acceptance",
        "",
        "| Check | Status | Observed |",
        "|---|---|---:|",
        (
            f"| At least 4 models x 5 metrics | "
            f"{'PASS' if diagnostics['model_count'] >= 4 and diagnostics['metric_count'] >= 5 else 'FAIL'} | "
            f"{diagnostics['model_count']} models x {diagnostics['metric_count']} metrics |"
        ),
        (
            f"| Simple CRW variance ratio near move probability | "
            f"{'PASS' if diagnostics['simple_crw_passed'] else 'FAIL'} | "
            f"{diagnostics['simple_crw_variance_ratio']:.6f} vs "
            f"{diagnostics['simple_crw_target']:.6f} |"
        ),
        (
            f"| GARCH optimizer converged | "
            f"{'PASS' if diagnostics['garch_converged'] else 'FAIL'} | "
            f"flag={diagnostics['garch_convergence_flag']} |"
        ),
        (
            f"| Coherent QRW variance ratio > 1.3 x CRW | "
            f"{'PASS' if coherent['passed'] else 'FAIL'} | "
            f"{coherent['qrw_variance_ratio']:.6f} vs "
            f"{coherent['crw_theoretical_variance_ratio']:.6f} |"
        ),
        "",
        "## Observed Test Metrics",
        "",
        "| Model | Wasserstein path MAE | Hit@1 | Hit@5 | Hit@10 | Mean direction log likelihood |",
        "|---|---:|---:|---:|---:|---:|",
        *summary_rows,
        "",
        "Lower Wasserstein error and less-negative direction log likelihood are",
        "better. These are single-window descriptive results.",
        "",
        "## Interpretation Notes",
        "",
        "- Paths use the empirical event move probability, so a symmetric",
        "  zero-inflated CRW targets `Var(X_T)/T = P(move)`.",
        "- QRW path forecasts use only the last feature vector observed before",
        "  the holdout; later holdout OBI/intensity values are not inputs.",
        "- `wasserstein_path_mae` is the mean one-dimensional Wasserstein",
        "  distance between each forecast cross-section and the realized price.",
        "- Directional log likelihood is Bernoulli and comparable across models.",
        "  AIC/BIC should only be compared within the same `likelihood_type` in",
        "  `model_comparison_table.csv`.",
        "- QRW superiority is not inferred from this engineering checkpoint.",
        "  Statistical claims require Phase 5 and fresh multi-day holdout data.",
        "",
        "## Artifacts",
        "",
        "- `results/benchmark_results.csv`",
        "- `results/model_comparison_table.csv`",
        "- `results/garch_params.json`",
        "- `reports/phase4_diagnostics.json`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-paths", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("results/benchmark_results.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("results/model_comparison_table.csv"),
    )
    parser.add_argument(
        "--garch-output",
        type=Path,
        default=Path("results/garch_params.json"),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("reports/phase4_diagnostics.json"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("reports/phase4_checkpoint.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path)
    frame = pd.read_parquet(feature_path)
    suite = BenchmarkSuite(
        frame,
        train_fraction=args.train_fraction,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )
    suite.run(
        benchmark_output=args.benchmark_output,
        comparison_output=args.comparison_output,
        garch_output=args.garch_output,
        diagnostics_output=args.diagnostics_output,
    )
    write_checkpoint(
        args.checkpoint_output,
        feature_path=feature_path,
        suite=suite,
    )
    print(f"Benchmark: {args.benchmark_output}")
    print(f"Comparison: {args.comparison_output}")
    print(f"Checkpoint: {args.checkpoint_output}")
    print(f"Status: {'PASS' if suite.diagnostics['checkpoint_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
