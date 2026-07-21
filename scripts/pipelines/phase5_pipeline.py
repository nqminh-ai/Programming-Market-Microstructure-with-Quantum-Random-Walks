"""Run Phase 5 statistical validation and compile the model scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipelines.phase3_pipeline import resolve_feature_path
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.directional_baselines import evaluate_directional_baselines
from src.evaluation.results_compiler import ResultsCompiler
from src.evaluation.statistical_tests import StatisticalTestSuite
from src.evaluation.provenance import collect_provenance


def write_checkpoint(
    output: Path,
    *,
    feature_path: Path,
    suite: BenchmarkSuite,
    statistical_results: dict[str, pd.DataFrame],
    scorecard: pd.DataFrame,
    directional_summary: pd.DataFrame,
    directional_diagnostics: dict[str, object],
    diagnostics_output: Path,
    provenance: dict[str, str],
    results_dir: Path = Path("results"),
) -> dict[str, object]:
    marginal = statistical_results["marginal_scores"]
    scaling = statistical_results["variance_scaling"]
    qrw_marginal = marginal.loc[
        marginal["model"] == "QRW Adaptive"
    ]
    qrw_scaling = scaling.loc[
        scaling["model"] == "QRW Adaptive"
    ].iloc[0]
    expected_files = [
        results_dir / "marginal_score_tests.csv",
        results_dir / "variance_scaling_results.csv",
        results_dir / "trajectory_autocorrelation_tests.csv",
        results_dir / "trajectory_tail_analysis.csv",
        results_dir / "diebold_mariano_tests.csv",
        results_dir / "scorecard_bootstrap_ci.csv",
        results_dir / "strong_directional_baselines.csv",
        results_dir / "strong_directional_predictions.csv",
        results_dir / "strong_directional_diagnostics.json",
    ]
    complete_categories = all(path.exists() for path in expected_files)
    finite_qrw_scores = bool(
        np.isfinite(
            qrw_marginal[["crps", "direction_log_loss"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    )
    finite_scaling_ci = bool(
        np.isfinite(
            qrw_scaling[["beta_ci_low", "beta_ci_high"]].to_numpy(
                dtype=np.float64
            )
        ).all()
    )
    scorecard_complete = bool(
        len(scorecard) == len(suite.forecast_samples)
        and "overall_rank" in scorecard
        and scorecard["overall_rank"].notna().all()
        and set(scorecard["selection_rule"])
        == {ResultsCompiler.SELECTION_RULE}
    )
    directional_sample_equality = bool(
        directional_diagnostics["same_test_sample_for_all_models"]
        and directional_summary["sample_size"].nunique() == 1
    )
    checkpoint_passed = bool(
        complete_categories
        and finite_qrw_scores
        and finite_scaling_ci
        and scorecard_complete
        and directional_sample_equality
    )
    diagnostics: dict[str, object] = {
        "provenance": provenance,
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
        "marginal_draws_per_model_horizon": suite.n_paths,
        "simulated_paths_per_model": suite.n_paths,
        "movement_probability": suite.movement_probability,
        "model_count": len(suite.forecast_samples),
        "marginal_score_horizons": sorted(
            marginal["horizon"].unique().astype(int).tolist()
        ),
        "complete_test_categories": complete_categories,
        "qrw_mean_marginal_crps": float(qrw_marginal["crps"].mean()),
        "qrw_mean_direction_log_loss": float(
            qrw_marginal["direction_log_loss"].mean()
        ),
        "qrw_variance_beta": float(qrw_scaling["beta"]),
        "qrw_variance_beta_ci": [
            float(qrw_scaling["beta_ci_low"]),
            float(qrw_scaling["beta_ci_high"]),
        ],
        "path_dependent_metrics_include_qrw": False,
        "dm_forecast_alignment": "rolling_origin_one_step",
        "scorecard_selection_rule": ResultsCompiler.SELECTION_RULE,
        "directional_baseline_sample_equality": directional_sample_equality,
        "directional_baseline_models": directional_summary[
            "model"
        ].tolist(),
        "top_directional_log_loss_model": str(
            directional_summary.sort_values("log_loss", kind="stable").iloc[0][
                "model"
            ]
        ),
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
        f"- Marginal draws per model/horizon: `{suite.n_paths}`",
        f"- Simulated steps: `{suite.n_steps}`",
        f"- Bootstrap iterations: "
        f"`{int(qrw_scaling['bootstrap_iterations'])}`",
        f"- Random seed: `{suite.random_seed}`",
        "- QRW is evaluated only through fixed-origin marginal proper scores",
        "  and marginal variance scaling.",
        "- QRW draws are excluded from ACF and tail diagnostics because they",
        "  are not jointly sampled trajectories.",
        "- Diebold-Mariano uses aligned rolling-origin one-step absolute losses.",
        "",
        "## Acceptance",
        "",
        "| Check | Status | Observed |",
        "|---|---|---:|",
        (
            f"| Nine test/strong-baseline artifacts persisted | "
            f"{'PASS' if complete_categories else 'FAIL'} | "
            f"{sum(path.exists() for path in expected_files)}/9 files |"
        ),
        (
            f"| QRW marginal proper scores computed | "
            f"{'PASS' if finite_qrw_scores else 'FAIL'} | "
            f"CRPS={qrw_marginal['crps'].mean():.6g}, "
            f"direction log loss={qrw_marginal['direction_log_loss'].mean():.6g} |"
        ),
        (
            f"| QRW variance beta has 95% CI | "
            f"{'PASS' if finite_scaling_ci else 'FAIL'} | "
            f"{qrw_scaling['beta']:.4f} "
            f"[{qrw_scaling['beta_ci_low']:.4f}, "
            f"{qrw_scaling['beta_ci_high']:.4f}] |"
        ),
        (
            f"| Scorecard compiled | "
            f"{'PASS' if scorecard_complete else 'FAIL'} | "
            f"top={scorecard.iloc[0]['model']} |"
        ),
        (
            f"| Strong baselines share one test sample | "
            f"{'PASS' if directional_sample_equality else 'FAIL'} | "
            f"n={int(directional_summary['sample_size'].iloc[0])} |"
        ),
        "",
        "## Scorecard",
        "",
        "| Rank | Model | Mean marginal CRPS | Direction log loss |",
        "|---:|---|---:|---:|",
    ]
    for _, row in scorecard.iterrows():
        lines.append(
            f"| {int(row['overall_rank'])} | {row['model']} | "
            f"{row['mean_marginal_crps']:.6g} | "
            f"{row['mean_direction_log_loss']:.6g} |"
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
            "- `results/marginal_score_tests.csv`",
            "- `results/variance_scaling_results.csv`",
            "- `results/trajectory_autocorrelation_tests.csv`",
            "- `results/trajectory_tail_analysis.csv`",
            "- `results/diebold_mariano_tests.csv`",
            "- `results/scorecard_bootstrap_ci.csv`",
            "- `results/strong_directional_baselines.csv`",
            "- `results/strong_directional_predictions.csv`",
            "- `results/strong_directional_diagnostics.json`",
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
    provenance = collect_provenance(
        feature_path,
        protocol_version=BenchmarkSuite.PROTOCOL_VERSION,
        repo_root=ROOT,
    )
    frame = pd.read_parquet(feature_path)
    directional_summary, directional_predictions, directional_diagnostics = (
        evaluate_directional_baselines(frame)
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    directional_summary.to_csv(
        args.results_dir / "strong_directional_baselines.csv", index=False
    )
    directional_predictions.to_csv(
        args.results_dir / "strong_directional_predictions.csv", index=False
    )
    directional_diagnostics = {
        **directional_diagnostics,
        "provenance": provenance,
        "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
    }
    (args.results_dir / "strong_directional_diagnostics.json").write_text(
        json.dumps(directional_diagnostics, indent=2), encoding="utf-8"
    )
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
        benchmark.test["price"].to_numpy(dtype=np.float64),
        benchmark.forecast_samples,
        sample_semantics=benchmark.sample_semantics,
        rolling_one_step_losses=benchmark.rolling_one_step_losses,
        rolling_one_step_timestamps=benchmark.rolling_one_step_timestamps,
        random_seed=args.random_seed,
        bootstrap_iterations=args.bootstrap_iterations,
        max_lag=args.max_lag,
    )
    statistical_results = tests.run_all(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
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
        directional_summary=directional_summary,
        directional_diagnostics=directional_diagnostics,
        diagnostics_output=args.diagnostics_output,
        provenance=provenance,
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
