"""Tests for the Phase 8 pretest release bundle and read-only dashboard."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.dashboard.ml_release_dashboard import (
    artifact_table,
    load_release_manifest,
    verify_release_files,
)
from src.evaluation.ml_protocol import load_ml_protocol
from src.evaluation.ml_release import (
    build_ml_release_bundle,
    validate_release_artifacts,
)
from src.evaluation.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ASSET = "BNBUSDT"
HORIZON = 1000


def _payloads() -> dict[str, dict]:
    protocol = load_ml_protocol()
    config_hash = sha256_file(
        ROOT / "config" / "ml_experiment.yaml"
    )
    common = {
        "protocol_version": protocol.protocol_version,
        "config_sha256": config_hash,
    }
    return {
        "phase1_dataset_metadata": {
            **common,
            "kind": "ml_directional_dataset",
            "asset": ASSET,
            "horizons_ticks": [HORIZON],
            "official": False,
        },
        "phase4_sequence_manifest": {
            **common,
            "kind": "causal_sequence_dataset",
            "asset": ASSET,
            "shards": [
                {
                    "horizon_ticks": HORIZON,
                    "fold": "test",
                    "path": "ignored-test-shard.npz",
                }
            ],
            "official": False,
            "test_labels_used_for_normalization": False,
        },
        "phase2_hgb_training": {
            **common,
            "kind": "histogram_gradient_boosting_training",
            "asset": ASSET,
            "horizon_ticks": HORIZON,
            "test_fold_read": False,
            "test_metrics": None,
        },
        "phase5_tcn_training": {
            **common,
            "kind": "temporal_convolutional_network_training",
            "asset": ASSET,
            "horizon_ticks": HORIZON,
            "test_fold_read": False,
            "test_metrics": None,
        },
        "phase6_tcn_robustness": {
            **common,
            "kind": "tcn_pretest_robustness",
            "asset": ASSET,
            "horizon_ticks": HORIZON,
            "test_fold_read": False,
            "test_metrics": None,
        },
        "phase7_hybrid_ablation": {
            **common,
            "kind": "neural_adaptive_qrw_ablation",
            "asset": ASSET,
            "horizon_ticks": HORIZON,
            "test_fold_read": False,
            "test_metrics": None,
        },
    }


def _write_artifacts(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, payload in _payloads().items():
        path = tmp_path / f"{role}.json"
        path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        result[role] = path
    return result


def test_build_release_hashes_artifacts_and_writes_reproduction_script(
    tmp_path: Path,
) -> None:
    build = build_ml_release_bundle(
        _write_artifacts(tmp_path),
        tmp_path / "release",
        repo_root=ROOT,
    )

    assert build.manifest_path.is_file()
    assert build.reproduction_path.is_file()
    assert build.manifest["protocol_version"] == "ml_directional_v7"
    assert build.manifest["holdout_state"] == "closed"
    assert build.manifest["test_fold_read"] is False
    assert len(build.manifest["artifacts"]) == 6
    assert all(
        len(entry["sha256"]) == 64
        for entry in build.manifest["artifacts"]
    )
    script = build.reproduction_path.read_text(encoding="utf-8")
    assert "build_ml_dataset" in script
    assert "evaluate_tcn_robustness" in script
    assert "train_neural_qrw_hybrid" in script
    assert "--open-test" not in script


def test_release_rejects_missing_role_test_metrics_and_scope_drift(
    tmp_path: Path,
) -> None:
    protocol = load_ml_protocol()
    paths = _write_artifacts(tmp_path)
    missing = dict(paths)
    missing.pop("phase7_hybrid_ablation")
    with pytest.raises(ValueError, match="roles mismatch"):
        validate_release_artifacts(missing, protocol)

    contaminated = json.loads(
        paths["phase5_tcn_training"].read_text(encoding="utf-8")
    )
    contaminated["test_metrics"] = {"brier": 0.1}
    paths["phase5_tcn_training"].write_text(
        json.dumps(contaminated), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="test metrics"):
        validate_release_artifacts(paths, protocol)

    paths = _write_artifacts(tmp_path / "scope")
    changed = json.loads(
        paths["phase6_tcn_robustness"].read_text(encoding="utf-8")
    )
    changed["asset"] = "BTCUSDT"
    paths["phase6_tcn_robustness"].write_text(
        json.dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="asset/horizon mismatch"):
        validate_release_artifacts(paths, protocol)


def test_dashboard_loads_inventory_and_verifies_hashes_read_only(
    tmp_path: Path,
) -> None:
    build = build_ml_release_bundle(
        _write_artifacts(tmp_path),
        tmp_path / "release",
        repo_root=ROOT,
    )
    manifest = load_release_manifest(build.manifest_path)
    table = artifact_table(manifest)
    verification = verify_release_files(manifest, repo_root=ROOT)

    assert len(table) == 6
    assert all(row["exists"] for row in verification)
    assert all(row["sha256_matches"] for row in verification)

    unsafe = deepcopy(dict(manifest))
    unsafe["test_fold_read"] = True
    unsafe_path = tmp_path / "unsafe.json"
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe test state"):
        load_release_manifest(unsafe_path)
