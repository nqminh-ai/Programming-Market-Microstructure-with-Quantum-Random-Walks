"""Generate and validate the complete Phase 6 research deliverables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_pipeline import resolve_feature_path
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.reporting import (
    ReportContext,
    build_final_report_markdown,
    build_presentation_markdown,
    count_pdf_pages,
    render_final_report_pdf,
    render_presentation_pdf,
)
from src.visualization import (
    PHASE6_FIGURES,
    plot_acf_comparison,
    plot_benchmark_scorecard,
    plot_heatmap_coin_operator,
    plot_probability_evolution,
    plot_return_distribution_comparison,
    plot_sample_paths,
    plot_variance_scaling,
    render_dashboard_preview,
)


def load_report_context(
    *,
    feature_path: Path,
    phase5_diagnostics: Path,
    results_dir: Path,
) -> ReportContext:
    """Load the frozen Phase 5 values used by every Phase 6 document."""
    diagnostics = json.loads(phase5_diagnostics.read_text(encoding="utf-8"))
    scorecard = pd.read_csv(results_dir / "scorecard.csv")
    distribution = pd.read_csv(results_dir / "distribution_tests.csv")
    scaling = pd.read_csv(results_dir / "variance_scaling_results.csv")
    autocorrelation = pd.read_csv(results_dir / "autocorrelation_tests.csv")
    tail = pd.read_csv(results_dir / "tail_analysis.csv")

    top = scorecard.sort_values("overall_rank", kind="stable").iloc[0]
    qrw_score = scorecard.loc[scorecard["model"] == "QRW Adaptive"].iloc[0]
    qrw_distribution = distribution.loc[
        (distribution["model"] == "QRW Adaptive")
        & (distribution["horizon"] == 1)
    ].iloc[0]
    qrw_scaling = scaling.loc[
        scaling["model"] == "QRW Adaptive"
    ].iloc[0]
    empirical_scaling = scaling.loc[
        scaling["model"] == "Empirical"
    ].iloc[0]
    qrw_tail = tail.loc[tail["model"] == "QRW Adaptive"].iloc[0]
    empirical_tail = tail.loc[tail["model"] == "Empirical"].iloc[0]
    qrw_acf = autocorrelation.loc[
        autocorrelation["model"] == "QRW Adaptive"
    ].iloc[0]
    return ReportContext(
        feature_path=str(feature_path),
        train_rows=int(diagnostics["train_rows"]),
        holdout_rows=int(diagnostics["holdout_rows"]),
        n_steps=int(diagnostics["simulated_steps"]),
        n_paths=int(diagnostics["simulated_paths_per_model"]),
        random_seed=int(diagnostics["random_seed"]),
        top_model=str(top["model"]),
        top_mean_rank=float(top["mean_rank"]),
        qrw_rank=int(qrw_score["overall_rank"]),
        qrw_mean_rank=float(qrw_score["mean_rank"]),
        qrw_ks_pvalue=float(qrw_distribution["ks_pvalue"]),
        qrw_ks_pvalue_bh=float(qrw_distribution["ks_pvalue_bh"]),
        empirical_beta=float(empirical_scaling["beta"]),
        qrw_beta=float(qrw_scaling["beta"]),
        qrw_beta_ci_low=float(qrw_scaling["beta_ci_low"]),
        qrw_beta_ci_high=float(qrw_scaling["beta_ci_high"]),
        empirical_tail_index=float(empirical_tail["tail_index"]),
        qrw_tail_index=float(qrw_tail["tail_index"]),
        qrw_acf_mse=float(qrw_acf["acf_mse"]),
    )


def validate_phase5_artifacts(
    *,
    feature_path: Path,
    phase5_diagnostics: Path,
    train_fraction: float,
    n_steps: int,
    n_paths: int,
    random_seed: int,
) -> dict[str, Any]:
    """Reject stale Phase 5 results before generating Phase 6 artifacts."""
    diagnostics = json.loads(phase5_diagnostics.read_text(encoding="utf-8"))
    expected = {
        "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
        "train_fraction": float(train_fraction),
        "requested_n_steps": int(n_steps),
        "simulated_paths_per_model": int(n_paths),
        "random_seed": int(random_seed),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {diagnostics.get(key)!r}"
        for key, value in expected.items()
        if diagnostics.get(key) != value
    ]
    stored_feature = Path(str(diagnostics.get("feature_path", "")))
    if not stored_feature.is_absolute():
        stored_feature = ROOT / stored_feature
    if stored_feature.resolve() != feature_path.resolve():
        mismatches.append(
            "feature_path: expected "
            f"{str(feature_path.resolve())!r}, found "
            f"{str(stored_feature.resolve())!r}"
        )
    feature_stat = feature_path.stat()
    for key, value in {
        "feature_bytes": feature_stat.st_size,
        "feature_mtime_ns": feature_stat.st_mtime_ns,
    }.items():
        if diagnostics.get(key) != value:
            mismatches.append(
                f"{key}: expected {value!r}, found {diagnostics.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            "Phase 5 artifacts do not match the requested Phase 6 run:\n- "
            + "\n- ".join(mismatches)
            + "\nRun `python scripts/phase5_pipeline.py` with matching arguments."
        )
    return diagnostics


def _image_diagnostics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.seek(0)
        grayscale = image.convert("L")
        extrema = grayscale.getextrema()
        standard_deviation = float(ImageStat.Stat(grayscale).stddev[0])
        width, height = image.size
        frames = int(getattr(image, "n_frames", 1))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "frames": frames,
        "grayscale_range": int(extrema[1] - extrema[0]),
        "grayscale_stddev": standard_deviation,
        "exists_and_nonblank": bool(
            path.stat().st_size > 0
            and width >= 600
            and height >= 300
            and extrema[1] - extrema[0] >= 10
            and standard_deviation >= 2.0
        ),
        "roadmap_size_over_50kb": bool(path.stat().st_size > 50_000),
    }


def _run_pytest() -> tuple[bool, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--basetemp={ROOT / 'reports' / '.pytest-tmp-phase6'}",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        value.strip()
        for value in (completed.stdout, completed.stderr)
        if value.strip()
    )
    return completed.returncode == 0, output[-4_000:]


def _write_checkpoint(
    path: Path,
    *,
    diagnostics: dict[str, Any],
    context: ReportContext,
) -> None:
    status = "PASS" if diagnostics["checkpoint_passed"] else "FAIL"
    lines = [
        "# Phase 6 Checkpoint",
        "",
        f"**Final engineering status:** {status}",
        "",
        "## Acceptance",
        "",
        "| Check | Status | Observed |",
        "|---|---|---:|",
        (
            f"| Seven required figures exist and are nonblank | "
            f"{'PASS' if diagnostics['figures_complete'] else 'FAIL'} | "
            f"{diagnostics['valid_figure_count']}/7 |"
        ),
        (
            f"| Every required figure exceeds 50 KB | "
            f"{'PASS' if diagnostics['figure_sizes_passed'] else 'FAIL'} | "
            f"{diagnostics['large_figure_count']}/7 |"
        ),
        (
            f"| Final report has at least 15 pages | "
            f"{'PASS' if diagnostics['report_page_count'] >= 15 else 'FAIL'} | "
            f"{diagnostics['report_page_count']} pages |"
        ),
        (
            f"| Presentation has 15-20 slides | "
            f"{'PASS' if 15 <= diagnostics['slide_page_count'] <= 20 else 'FAIL'} | "
            f"{diagnostics['slide_page_count']} slides |"
        ),
        (
            f"| Report contains a quantitative p-value table | "
            f"{'PASS' if diagnostics['quantitative_table_present'] else 'FAIL'} | "
            "KS p-value and adjusted p-value |"
        ),
        (
            f"| Final pytest run passes | "
            f"{'PASS' if diagnostics['pytest_passed'] else 'FAIL'} | "
            f"{diagnostics['pytest_summary']} |"
        ),
        (
            f"| README has installation and run instructions | "
            f"{'PASS' if diagnostics['readme_complete'] else 'FAIL'} | "
            "installation + reproducibility commands |"
        ),
        "",
        "## Scientific Guardrail",
        "",
        f"- Top Phase 5 scorecard model: `{context.top_model}`.",
        f"- QRW Adaptive overall rank: `{context.qrw_rank}`.",
        "- Scope: one June 12, 2026 window; exploratory only.",
        "- The report explicitly rejects a current QRW superiority claim.",
        "- A frozen protocol and fresh multi-day synchronized LOB holdout are",
        "  still required for confirmation.",
        "",
        "## Artifacts",
        "",
        "- `src/visualization/plot_suite.py`",
        "- `figures/prob_evolution.gif`",
        "- `figures/variance_scaling.png`",
        "- `figures/return_distributions.png`",
        "- `figures/acf_comparison.png`",
        "- `figures/sample_paths.png`",
        "- `figures/coin_operator_heatmap.png`",
        "- `figures/scorecard.png`",
        "- `docs/final_report.md`",
        "- `docs/final_report.pdf`",
        "- `docs/presentation_slides.md`",
        "- `docs/presentation_slides.pdf`",
        "- `reports/phase6_diagnostics.json`",
        "",
        "The optional Streamlit dashboard is available at `src/dashboard/app.py`.",
        "Its static preview is `figures/dashboard_screenshot.png`.",
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
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--phase5-diagnostics",
        type=Path,
        default=Path("reports/phase5_diagnostics.json"),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("reports/phase6_diagnostics.json"),
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=Path("reports/phase6_checkpoint.md"),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Generate artifacts without running the final pytest gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path)
    required_results = (
        "scorecard.csv",
        "distribution_tests.csv",
        "variance_scaling_results.csv",
        "autocorrelation_tests.csv",
        "tail_analysis.csv",
    )
    missing = [
        args.results_dir / name
        for name in required_results
        if not (args.results_dir / name).exists()
    ]
    if missing or not args.phase5_diagnostics.exists():
        raise FileNotFoundError(
            "Phase 5 artifacts are required; run "
            "`python scripts/phase5_pipeline.py` first."
        )
    validate_phase5_artifacts(
        feature_path=feature_path,
        phase5_diagnostics=args.phase5_diagnostics,
        train_fraction=args.train_fraction,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )

    frame = pd.read_parquet(feature_path)
    benchmark = BenchmarkSuite(
        frame,
        train_fraction=args.train_fraction,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )
    benchmark.run()
    empirical_prices = benchmark.test["price"].to_numpy(dtype=np.float64)
    paths = benchmark.simulated_paths
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    plot_probability_evolution(
        t_snapshots=(0, 5, 10, 20, 30, 40, 50, 60),
        output_path=args.figures_dir / "prob_evolution.gif",
    )
    plot_variance_scaling(
        empirical_prices,
        paths,
        output_path=args.figures_dir / "variance_scaling.png",
    )
    plot_return_distribution_comparison(
        empirical_prices,
        paths,
        output_path=args.figures_dir / "return_distributions.png",
        random_seed=args.random_seed,
    )
    plot_acf_comparison(
        empirical_prices,
        paths,
        output_path=args.figures_dir / "acf_comparison.png",
    )
    plot_sample_paths(
        empirical_prices,
        paths,
        output_path=args.figures_dir / "sample_paths.png",
    )
    calibration = benchmark.diagnostics["qrw_calibration"]
    plot_heatmap_coin_operator(
        obi=0.75,
        alpha=max(abs(float(calibration["alpha_obi"])), 0.1),
        output_path=args.figures_dir / "coin_operator_heatmap.png",
    )
    plot_benchmark_scorecard(
        args.results_dir / "scorecard.csv",
        output_path=args.figures_dir / "scorecard.png",
    )
    render_dashboard_preview(
        empirical_prices,
        args.results_dir / "scorecard.csv",
        output_path=args.figures_dir / "dashboard_screenshot.png",
    )

    context = load_report_context(
        feature_path=feature_path,
        phase5_diagnostics=args.phase5_diagnostics,
        results_dir=args.results_dir,
    )
    report_markdown = build_final_report_markdown(context)
    slides_markdown = build_presentation_markdown(context)
    report_md_path = args.docs_dir / "final_report.md"
    slides_md_path = args.docs_dir / "presentation_slides.md"
    report_md_path.write_text(report_markdown, encoding="utf-8")
    slides_md_path.write_text(slides_markdown, encoding="utf-8")
    report_pdf = render_final_report_pdf(
        context,
        args.figures_dir,
        args.docs_dir / "final_report.pdf",
    )
    slides_pdf = render_presentation_pdf(
        context,
        args.figures_dir,
        args.docs_dir / "presentation_slides.pdf",
    )

    image_results = {
        name: _image_diagnostics(args.figures_dir / name)
        for name in PHASE6_FIGURES
    }
    pytest_passed, pytest_output = (
        (False, "skipped by --skip-tests")
        if args.skip_tests
        else _run_pytest()
    )
    pytest_summary = next(
        (
            line.strip()
            for line in reversed(pytest_output.splitlines())
            if "passed" in line or "failed" in line or "error" in line
        ),
        pytest_output.splitlines()[-1] if pytest_output else "no output",
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    readme_complete = all(
        marker in readme
        for marker in ("installation", "run phase 6", "python -m pytest")
    )
    quantitative_table_present = all(
        marker in report_markdown
        for marker in (
            "| QRW KS p-value |",
            "| QRW KS p-value, BH adjusted |",
            "does not establish QRW predictive",
        )
    )
    valid_figure_count = sum(
        result["exists_and_nonblank"] for result in image_results.values()
    )
    large_figure_count = sum(
        result["roadmap_size_over_50kb"]
        for result in image_results.values()
    )
    report_pages = count_pdf_pages(report_pdf)
    slide_pages = count_pdf_pages(slides_pdf)
    diagnostics: dict[str, Any] = {
        "feature_path": str(feature_path),
        "random_seed": args.random_seed,
        "benchmark_paths_per_model": args.n_paths,
        "benchmark_steps": benchmark.n_steps,
        "phase5_top_model": context.top_model,
        "phase5_qrw_rank": context.qrw_rank,
        "inference_scope": "single_window_exploratory",
        "confirmatory_superiority_claim_allowed": False,
        "figures": image_results,
        "valid_figure_count": valid_figure_count,
        "large_figure_count": large_figure_count,
        "figures_complete": valid_figure_count == len(PHASE6_FIGURES),
        "figure_sizes_passed": large_figure_count == len(PHASE6_FIGURES),
        "report_page_count": report_pages,
        "slide_page_count": slide_pages,
        "quantitative_table_present": quantitative_table_present,
        "readme_complete": readme_complete,
        "pytest_passed": pytest_passed,
        "pytest_summary": pytest_summary,
        "pytest_output_tail": pytest_output,
    }
    diagnostics["checkpoint_passed"] = bool(
        diagnostics["figures_complete"]
        and diagnostics["figure_sizes_passed"]
        and report_pages >= 15
        and 15 <= slide_pages <= 20
        and quantitative_table_present
        and readme_complete
        and pytest_passed
    )
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    _write_checkpoint(
        args.checkpoint_output,
        diagnostics=diagnostics,
        context=context,
    )

    print(f"Figures: {valid_figure_count}/{len(PHASE6_FIGURES)} valid")
    print(f"Report: {report_pdf} ({report_pages} pages)")
    print(f"Slides: {slides_pdf} ({slide_pages} pages)")
    print(f"Tests: {pytest_summary}")
    print(
        "Final Phase 6 status: "
        f"{'PASS' if diagnostics['checkpoint_passed'] else 'FAIL'}"
    )
    if not diagnostics["checkpoint_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
