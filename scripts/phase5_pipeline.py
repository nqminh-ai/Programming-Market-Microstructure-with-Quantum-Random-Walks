"""Run Phase 5 statistical validation and compile the model scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_pipeline import resolve_feature_path
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.statistical_tests import StatisticalTestSuite


def write_checkpoint(
    output: Path,
    *,
    feature_path: Path,
    suite: BenchmarkSuite,
    statistical_results: dict[str, pd.DataFrame],
    scorecard: pd.DataFrame,
    diagnostics_output: Path,
    results_dir: Path = Path("results"),
) -> dict[str, object]:
    distribution = statistical_results["distribution"]
    scaling = statistical_results["variance_scaling"]
    tail = statistical_results["tail"]
    qrw_distribution = distribution.loc[
        (distribution["model"] == "QRW Adaptive")
        & (distribution["horizon"] == 1)
    ].iloc[0]
    qrw_scaling = scaling.loc[
        scaling["model"] == "QRW Adaptive"
    ].iloc[0]
    qrw_tail = tail.loc[tail["model"] == "QRW Adaptive"].iloc[0]
    empirical_tail = tail.loc[tail["model"] == "Empirical"].iloc[0]
    expected_files = [
        results_dir / "distribution_tests.csv",
        results_dir / "variance_scaling_results.csv",
        results_dir / "autocorrelation_tests.csv",
        results_dir / "tail_analysis.csv",
        results_dir / "diebold_mariano_tests.csv",
        results_dir / "scorecard_bootstrap_ci.csv",
    ]
    complete_categories = all(path.exists() for path in expected_files)
    finite_qrw_pvalue = bool(np.isfinite(qrw_distribution["ks_pvalue"]))
    finite_scaling_ci = bool(
        np.isfinite(
            qrw_scaling[["beta_ci_low", "beta_ci_high"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    )
    finite_tail_indices = bool(
        np.isfinite(
            np.asarray(
                [qrw_tail["tail_index"], empirical_tail["tail_index"]],
                dtype=np.float64,
            )
        ).all()
    )
    scorecard_complete = bool(
        len(scorecard) == len(suite.simulated_paths)
        and "overall_rank" in scorecard
        and scorecard["overall_rank"].notna().all()
    )
    checkpoint_passed = bool(
        complete_categories
        and finite_qrw_pvalue
        and finite_scaling_ci
        and finite_tail_indices
        and scorecard_complete
    )
    diagnostics: dict[str, object] = {
        "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
        "feature_path": str(feature_path),
        "feature_bytes": feature_path.stat().st_size,
        "feature_mtime_ns": feature_path.stat().st_mtime_ns,
        "train_fraction": suite.train_fraction,
        "requested_n_steps": suite.requested_n_steps,
        "random_seed": suite.random_seed,
        "train_rows": len(suite.train),
        "holdout_rows": len(suite.holdout),
        "simulated_steps": suite.n_steps,
        "simulated_paths_per_model": suite.n_paths,
        "movement_probability": suite.movement_probability,
        "model_count": len(suite.simulated_paths),
        "distribution_horizons": sorted(
            distribution["horizon"].unique().astype(int).tolist()
        ),
        "complete_test_categories": complete_categories,
        "qrw_ks_pvalue": float(qrw_distribution["ks_pvalue"]),
        "qrw_ks_pvalue_bh": float(qrw_distribution["ks_pvalue_bh"]),
        "qrw_variance_beta": float(qrw_scaling["beta"]),
        "qrw_variance_beta_ci": [
            float(qrw_scaling["beta_ci_low"]),
            float(qrw_scaling["beta_ci_high"]),
        ],
        "qrw_tail_index": float(qrw_tail["tail_index"]),
        "empirical_tail_index": float(empirical_tail["tail_index"]),
        "scorecard_complete": scorecard_complete,
        "top_ranked_model": str(scorecard.iloc[0]["model"]),
        "checkpoint_passed": checkpoint_passed,
        "inference_scope": "single_window_exploratory",
        "confirmatory_superiority_claim_allowed": False,
    }
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    status = "PASS" if checkpoint_passed else "FAIL"
    lines = [
        "# Phase 5 Checkpoint",
        "",
        f"**Engineering status:** {status}",
        "",
        f"Feature source: `{feature_path}`",
        "",
        "## Protocol",
        "",
        f"- Chronological train rows: `{len(suite.train)}`",
        f"- Later holdout rows: `{len(suite.holdout)}`",
        f"- Simulated paths per model: `{suite.n_paths}`",
        f"- Simulated steps: `{suite.n_steps}`",
        f"- Bootstrap iterations: "
        f"`{int(qrw_scaling['bootstrap_iterations'])}`",
        f"- Random seed: `{suite.random_seed}`",
        "- Distribution and tail tests use matched empirical/simulated sample",
        "  sizes. Variance scaling uses the same comparison horizon for both.",
        "- Scaling intervals use moving-block bootstrap for empirical returns",
        "  and whole-path resampling for simulations.",
        "- Benjamini-Hochberg adjusted p-values are included for each test",
        "  family.",
        "",
        "## Acceptance",
        "",
        "| Check | Status | Observed |",
        "|---|---|---:|",
        (
            f"| Six test categories persisted | "
            f"{'PASS' if complete_categories else 'FAIL'} | "
            f"{sum(path.exists() for path in expected_files)}/6 CSV files |"
        ),
        (
            f"| QRW KS p-value computed | "
            f"{'PASS' if finite_qrw_pvalue else 'FAIL'} | "
            f"raw={qrw_distribution['ks_pvalue']:.6g}, "
            f"BH={qrw_distribution['ks_pvalue_bh']:.6g} |"
        ),
        (
            f"| QRW variance beta has 95% CI | "
            f"{'PASS' if finite_scaling_ci else 'FAIL'} | "
            f"{qrw_scaling['beta']:.4f} "
            f"[{qrw_scaling['beta_ci_low']:.4f}, "
            f"{qrw_scaling['beta_ci_high']:.4f}] |"
        ),
        (
            f"| QRW and empirical tail indices computed | "
            f"{'PASS' if finite_tail_indices else 'FAIL'} | "
            f"{qrw_tail['tail_index']:.4f} vs "
            f"{empirical_tail['tail_index']:.4f} |"
        ),
        (
            f"| Scorecard compiled | "
            f"{'PASS' if scorecard_complete else 'FAIL'} | "
            f"top={scorecard.iloc[0]['model']} |"
        ),
        "",
        "## Scorecard",
        "",
        "| Rank | Model | Mean metric rank |",
        "|---:|---|---:|",
    ]
    for _, row in scorecard.iterrows():
        lines.append(
            f"| {int(row['overall_rank'])} | {row['model']} | "
            f"{row['mean_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This checkpoint validates implementation completeness on one",
            "June 12, 2026 window. It is exploratory and does not establish",
            "QRW superiority. A confirmatory claim still requires the frozen",
            "protocol and fresh multi-day holdout described in the audit.",
            "",
            "## Artifacts",
            "",
            "- `results/distribution_tests.csv`",
            "- `results/variance_scaling_results.csv`",
            "- `results/autocorrelation_tests.csv`",
            "- `results/tail_analysis.csv`",
            "- `results/diebold_mariano_tests.csv`",
            "- `results/scorecard_bootstrap_ci.csv`",
            "- `results/model_aic_bic_comparison.csv`",
            "- `results/final_comparison_table.csv`",
            "- `results/scorecard.csv`",
            "- `figures/variance_scaling.png`",
            "- `figures/acf_comparison.png`",
            "- `reports/phase5_diagnostics.json`",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-paths", type=int, default=5_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("reports/phase5_diagnostics.json"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("reports/phase5_checkpoint.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path)
    frame = pd.read_parquet(feature_path)
    benchmark = BenchmarkSuite(
        frame,
        train_fraction=args.train_fraction,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )
    benchmark.run(
        comparison_output=args.results_dir / "model_aic_bic_comparison.csv"
    )
    tests = StatisticalTestSuite(
        benchmark.holdout["price"].to_numpy(dtype=np.float64),
        benchmark.simulated_paths,
        random_seed=args.random_seed,
        bootstrap_iterations=args.bootstrap_iterations,
        max_lag=args.max_lag,
    )
    statistical_results = tests.run_all(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    tests.run_model_selection_tests(
        benchmark.model_comparison,
        output=args.results_dir / "model_selection_corrected.csv",
    )
    comparison, scorecard = ResultsCompiler().compile(
        statistical_results,
        comparison_output=args.results_dir / "final_comparison_table.csv",
        scorecard_output=args.results_dir / "scorecard.csv",
    )
    diagnostics = write_checkpoint(
        args.checkpoint_output,
        feature_path=feature_path,
        suite=benchmark,
        statistical_results=statistical_results,
        scorecard=scorecard,
        diagnostics_output=args.diagnostics_output,
        results_dir=args.results_dir,
    )
    print(f"Models compiled: {len(comparison)}")
    print(f"Top scorecard model: {scorecard.iloc[0]['model']}")
    print(f"Checkpoint: {args.checkpoint_output}")
    print(
        "Engineering status: "
        f"{'PASS' if diagnostics['checkpoint_passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
