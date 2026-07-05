"""Tests for Phase 6 artifact consistency guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.phase6_pipeline import validate_phase5_artifacts
from src.evaluation.benchmark_suite import BenchmarkSuite
from src.reporting import ReportContext, build_final_report_markdown
from src.visualization.plot_suite import _variance_by_horizon


def test_phase6_rejects_stale_phase5_artifacts(tmp_path) -> None:
    feature_path = tmp_path / "features.parquet"
    feature_path.write_bytes(b"feature-artifact")
    stat = feature_path.stat()
    diagnostics_path = tmp_path / "phase5.json"
    diagnostics = {
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
    )
    report = build_final_report_markdown(context)

    assert "QRW KS p-value" not in report
    assert "QRW tail index" not in report
    assert "QRW has return-ACF" not in report
    assert "draw độc lập giữa\ncác horizon" in report
