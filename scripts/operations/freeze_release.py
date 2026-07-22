"""Validate Phase 3-6 provenance and hash every official release artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipelines.phase3_pipeline import resolve_feature_path
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.provenance import (
    build_sha256_manifest,
    release_dirty_paths,
    validate_diagnostic_set,
    write_sha256_manifest,
)
from src.visualization import PHASE6_FIGURES


DIAGNOSTICS = (
    Path("reports/phase3_holdout_diagnostics.json"),
    Path("reports/audits/phase3_overfitting_audit.json"),
    Path("reports/phase4_diagnostics.json"),
    Path("reports/phase5_diagnostics.json"),
    Path("reports/phase6_diagnostics.json"),
)

OFFICIAL_OUTPUTS = (
    Path("results/calibrated_params.json"),
    Path("results/phase3_holdout_benchmark.csv"),
    Path("results/phase3_holdout_model_comparison.csv"),
    Path("results/phase3_holdout_garch.json"),
    Path("reports/performance_benchmark.txt"),
    Path("reports/phase3_holdout_diagnostics.json"),
    Path("reports/audits/phase3_overfitting_audit.json"),
    Path("reports/audits/phase3_overfitting_audit.md"),
    Path("results/benchmark_results.csv"),
    Path("results/model_comparison_table.csv"),
    Path("results/garch_params.json"),
    Path("reports/phase4_diagnostics.json"),
    Path("reports/phase4_checkpoint.md"),
    Path("results/marginal_score_tests.csv"),
    Path("results/variance_scaling_results.csv"),
    Path("results/trajectory_autocorrelation_tests.csv"),
    Path("results/trajectory_tail_analysis.csv"),
    Path("results/diebold_mariano_tests.csv"),
    Path("results/scorecard_bootstrap_ci.csv"),
    Path("results/model_aic_bic_comparison.csv"),
    Path("results/final_comparison_table.csv"),
    Path("results/scorecard.csv"),
    Path("results/strong_directional_baselines.csv"),
    Path("results/strong_directional_predictions.csv"),
    Path("results/strong_directional_diagnostics.json"),
    Path("reports/phase5_diagnostics.json"),
    Path("reports/phase5_checkpoint.md"),
    *tuple(Path("figures") / name for name in PHASE6_FIGURES),
    Path("docs/final_report.md"),
    Path("docs/final_report.pdf"),
    Path("docs/presentation_slides.md"),
    Path("docs/presentation_slides.pdf"),
    Path("reports/phase6_diagnostics.json"),
    Path("reports/phase6_checkpoint.md"),
)


def freeze_release(
    feature_path: Path,
    *,
    diagnostics: tuple[Path, ...] = DIAGNOSTICS,
    outputs: tuple[Path, ...] = OFFICIAL_OUTPUTS,
    manifest_output: Path = Path("reports/release_manifest.json"),
) -> Path:
    """Hard-fail unless one clean commit produced every declared artifact."""
    dirty = release_dirty_paths(ROOT)
    if dirty:
        raise RuntimeError(
            "release freeze requires a clean source tree; dirty paths: "
            + ", ".join(dirty[:12])
        )
    resolved_diagnostics = tuple(
        path if path.is_absolute() else ROOT / path for path in diagnostics
    )
    provenance = validate_diagnostic_set(
        resolved_diagnostics,
        feature_path,
        protocol_version=BenchmarkSuite.PROTOCOL_VERSION,
        repo_root=ROOT,
    )
    resolved_outputs = tuple(
        path if path.is_absolute() else ROOT / path for path in outputs
    )
    destination = (
        manifest_output
        if manifest_output.is_absolute()
        else ROOT / manifest_output
    )
    manifest = build_sha256_manifest(
        repo_root=ROOT,
        provenance=provenance,
        inputs=(feature_path, ROOT / "requirements.lock", *resolved_diagnostics),
        outputs=resolved_outputs,
    )
    return write_sha256_manifest(destination, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("reports/release_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path).resolve()
    manifest = freeze_release(
        feature_path, manifest_output=args.manifest_output
    )
    print(f"Frozen release manifest: {manifest}")


if __name__ == "__main__":
    main()
