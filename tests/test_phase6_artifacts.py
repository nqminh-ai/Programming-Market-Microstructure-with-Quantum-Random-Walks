"""Tests for Phase 6 artifact consistency guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pipelines.phase6_pipeline import (
    load_report_context,
    validate_document_text,
    validate_phase5_artifacts,
)
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.evaluation.provenance import (
    canonical_repo_path,
    git_commit,
    sha256_file,
)
from src.reporting import ReportContext, build_final_report_markdown
from src.visualization.plot_suite import _variance_by_horizon


def test_phase6_rejects_stale_phase5_artifacts(tmp_path) -> None:
    feature_path = tmp_path / "features.parquet"
    feature_path.write_bytes(b"feature-artifact")
    stat = feature_path.stat()
    diagnostics_path = tmp_path / "phase5.json"
    diagnostics = {
        "provenance": {
            "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
            "code_commit": git_commit(Path(__file__).resolve().parents[1]),
            "feature_path": canonical_repo_path(
                feature_path, Path(__file__).resolve().parents[1]
            ),
            "feature_sha256": sha256_file(feature_path),
        },
        "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
        "feature_path": str(feature_path),
        "feature_bytes": stat.st_size,
        "feature_mtime_ns": stat.st_mtime_ns,
        "train_fraction": 0.6,
        "requested_n_steps": 500,
        "simulated_paths_per_model": 5_000,
        "random_seed": 2026,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics),
        encoding="utf-8",
    )

    validated = validate_phase5_artifacts(
        feature_path=feature_path,
        phase5_diagnostics=diagnostics_path,
        train_fraction=0.6,
        n_steps=500,
        n_paths=5_000,
        random_seed=2026,
    )
    assert validated["protocol_version"] == BenchmarkSuite.PROTOCOL_VERSION

    diagnostics["protocol_version"] = "stale_protocol"
    diagnostics_path.write_text(
        json.dumps(diagnostics),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol_version"):
        validate_phase5_artifacts(
            feature_path=feature_path,
            phase5_diagnostics=diagnostics_path,
            train_fraction=0.6,
            n_steps=500,
            n_paths=5_000,
            random_seed=2026,
        )


def test_phase6_rejects_missing_provenance(tmp_path) -> None:
    feature_path = tmp_path / "features.parquet"
    feature_path.write_bytes(b"feature-artifact")
    diagnostics_path = tmp_path / "phase5.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
                "feature_path": str(feature_path),
                "train_fraction": 0.6,
                "requested_n_steps": 500,
                "simulated_paths_per_model": 5_000,
                "random_seed": 2026,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        validate_phase5_artifacts(
            feature_path=feature_path,
            phase5_diagnostics=diagnostics_path,
            train_fraction=0.6,
            n_steps=500,
            n_paths=5_000,
            random_seed=2026,
        )


def test_variance_plot_skips_single_observation_horizons() -> None:
    variances = _variance_by_horizon(
        np.arange(6, dtype=float),
        np.array([1, 5]),
        paths=False,
    )

    assert np.isfinite(variances[0])
    assert np.isnan(variances[1])


def test_fixed_origin_variance_ignores_cross_horizon_row_pairing() -> None:
    rng = np.random.default_rng(21)
    samples = rng.normal(100.0, 0.2, size=(100, 11))
    samples[:, 0] = 100.0
    shuffled = samples.copy()
    for horizon in range(1, shuffled.shape[1]):
        rng.shuffle(shuffled[:, horizon])

    expected = _variance_by_horizon(
        samples,
        np.array([1, 5, 10]),
        paths=True,
        fixed_origin=True,
    )
    observed = _variance_by_horizon(
        shuffled,
        np.array([1, 5, 10]),
        paths=True,
        fixed_origin=True,
    )
    assert observed == pytest.approx(expected)


def _valid_report_context_kwargs() -> dict:
    return dict(
        feature_path="data/assets/btcusdt/features/example.parquet",
        train_rows=100,
        holdout_rows=50,
        n_steps=40,
        n_paths=200,
        random_seed=2026,
        top_model="CRW Simple",
        top_mean_marginal_crps=0.01,
        qrw_rank=3,
        qrw_mean_marginal_crps=0.02,
        qrw_mean_direction_log_loss=0.7,
        empirical_beta=1.0,
        qrw_beta=1.2,
        qrw_beta_ci_low=0.9,
        qrw_beta_ci_high=1.5,
        walk_forward_folds=3,
        walk_forward_folds_qrw_better=0,
        walk_forward_brier_difference=0.05,
        walk_forward_ci_low=0.02,
        walk_forward_ci_high=0.08,
        walk_forward_verdict="QRW is worse in pooled walk-forward.",
        top_directional_model="Logistic L2 (5F)",
        qrw_directional_rank=4,
        qrw_directional_log_loss=0.72,
        directional_test_events=40,
    )


def test_report_context_rejects_malformed_values() -> None:
    """Regression test for audit finding M9: ReportContext previously had
    no validation, so NaN/negative/reversed-interval values could silently
    reach a "complete" report."""
    base = _valid_report_context_kwargs()
    assert ReportContext(**base) is not None

    bad_variants = {
        "qrw_rank": 0,
        "qrw_directional_rank": 0,
        "train_rows": -1,
        "walk_forward_folds_qrw_better": 4,
        "qrw_mean_marginal_crps": float("nan"),
        "walk_forward_brier_difference": float("inf"),
        "qrw_beta_ci_low": 2.0,
        "walk_forward_ci_low": 0.5,
    }
    for field, bad_value in bad_variants.items():
        kwargs = {**base, field: bad_value}
        with pytest.raises(ValueError):
            ReportContext(**kwargs)


def test_report_does_not_claim_path_statistics_for_qrw() -> None:
    context = ReportContext(
        feature_path="data/assets/btcusdt/features/example.parquet",
        train_rows=100,
        holdout_rows=50,
        n_steps=40,
        n_paths=200,
        random_seed=2026,
        top_model="CRW Simple",
        top_mean_marginal_crps=0.01,
        qrw_rank=3,
        qrw_mean_marginal_crps=0.02,
        qrw_mean_direction_log_loss=0.7,
        empirical_beta=1.0,
        qrw_beta=1.2,
        qrw_beta_ci_low=0.9,
        qrw_beta_ci_high=1.5,
        walk_forward_folds=3,
        walk_forward_folds_qrw_better=0,
        walk_forward_brier_difference=0.05,
        walk_forward_ci_low=0.02,
        walk_forward_ci_high=0.08,
        walk_forward_verdict="QRW is worse in pooled walk-forward.",
        top_directional_model="Logistic L2 (5F)",
        qrw_directional_rank=4,
        qrw_directional_log_loss=0.72,
        directional_test_events=40,
    )
    report = build_final_report_markdown(context)

    assert "QRW KS p-value" not in report
    assert "QRW tail index" not in report
    assert "QRW has return-ACF" not in report
    assert "mẫu độc lập giữa\ncác horizon" in report
    assert "Báo cáo cuối" in report
    assert "Kết luận" in report
    assert "Ä‘" not in report
    assert "á»" not in report
    assert "0/3" in report
    assert "+0.050000" in report


def test_vietnamese_report_and_readme_satisfy_release_markers() -> None:
    root = Path(__file__).resolve().parents[1]
    report = (root / "docs" / "final_report.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    report_ok, readme_ok = validate_document_text(report, readme)

    assert report_ok is True
    assert readme_ok is True


def test_report_context_is_sourced_from_provenanced_result_files(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    feature = tmp_path / "features.parquet"
    feature.write_bytes(b"report-source")
    diagnostics = {
        "provenance": {
            "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
            "code_commit": git_commit(root),
            "feature_path": canonical_repo_path(feature, root),
            "feature_sha256": sha256_file(feature),
        },
        "train_rows": 120,
        "holdout_rows": 80,
        "simulated_steps": 50,
        "simulated_paths_per_model": 300,
        "random_seed": 2026,
    }
    diagnostics_path = tmp_path / "phase5.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    audit_path = tmp_path / "phase3.json"
    audit_path.write_text(
        json.dumps(
            {
                "provenance": diagnostics["provenance"],
                "walk_forward": [{"fold": index} for index in range(3)],
                "walk_forward_edge": {
                    "folds_model_better": 0,
                    "model_minus_comparison": 0.05,
                    "confidence_interval_95": [0.02, 0.08],
                },
                "verdict": "QRW is worse in pooled walk-forward.",
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        {
            "model": ["CRW Simple", "QRW Adaptive"],
            "overall_rank": [1.0, 2.0],
            "mean_marginal_crps": [0.01, 0.02],
        }
    ).to_csv(results / "scorecard.csv", index=False)
    pd.DataFrame(
        {
            "model": ["QRW Adaptive", "QRW Adaptive"],
            "direction_log_loss": [0.6, 0.8],
        }
    ).to_csv(results / "marginal_score_tests.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Empirical", "QRW Adaptive"],
            "beta": [1.0, 1.2],
            "beta_ci_low": [0.9, 1.0],
            "beta_ci_high": [1.1, 1.4],
        }
    ).to_csv(results / "variance_scaling_results.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Logistic L2 (5F)", "QRW Directional Link"],
            "sample_size": [40, 40],
            "log_loss": [0.6, 0.7],
        }
    ).to_csv(results / "strong_directional_baselines.csv", index=False)

    context = load_report_context(
        feature_path=feature,
        phase3_audit=audit_path,
        phase5_diagnostics=diagnostics_path,
        results_dir=results,
    )

    assert context.top_model == "CRW Simple"
    assert context.qrw_mean_marginal_crps == pytest.approx(0.02)
    assert context.qrw_mean_direction_log_loss == pytest.approx(0.7)
    assert context.qrw_beta_ci_high == pytest.approx(1.4)
    assert context.walk_forward_folds_qrw_better == 0
    assert context.walk_forward_ci_low == pytest.approx(0.02)
    assert context.top_directional_model == "Logistic L2 (5F)"
    assert context.qrw_directional_rank == 2
