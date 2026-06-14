"""Tests for Phase 6 artifact consistency guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.phase6_pipeline import validate_phase5_artifacts
from src.evaluation.benchmark_suite import BenchmarkSuite
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
