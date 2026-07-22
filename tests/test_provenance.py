"""Tests for release provenance and artifact checksum guards."""

from __future__ import annotations

import hashlib
import json

import pytest

import scripts.operations.freeze_release as freeze_module
from src.evaluation.provenance import (
    build_sha256_manifest,
    canonical_repo_path,
    sha256_file,
    validate_diagnostic_set,
    validate_provenance,
)


def test_sha256_file_matches_known_reference(tmp_path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"qrw-release-input\n")

    assert sha256_file(source) == hashlib.sha256(
        b"qrw-release-input\n"
    ).hexdigest()


def test_validate_provenance_rejects_protocol_commit_path_and_hash(
    tmp_path,
) -> None:
    feature = tmp_path / "features.parquet"
    feature.write_bytes(b"feature-data")
    expected = {
        "protocol_version": "protocol-v1",
        "code_commit": "a" * 40,
        "feature_path": canonical_repo_path(feature, tmp_path),
        "feature_sha256": sha256_file(feature),
    }

    assert validate_provenance(
        expected,
        feature,
        protocol_version="protocol-v1",
        repo_root=tmp_path,
        expected_commit="a" * 40,
    ) == expected

    for key in expected:
        stale = dict(expected)
        stale[key] = "stale"
        with pytest.raises(ValueError, match=key):
            validate_provenance(
                stale,
                feature,
                protocol_version="protocol-v1",
                repo_root=tmp_path,
                expected_commit="a" * 40,
            )


def test_manifest_hashes_every_declared_input_and_output(tmp_path) -> None:
    source = tmp_path / "input.dat"
    output = tmp_path / "result.csv"
    source.write_bytes(b"input")
    output.write_text("metric,value\ncrps,1.0\n", encoding="utf-8")

    manifest = build_sha256_manifest(
        repo_root=tmp_path,
        provenance={"protocol_version": "v1", "code_commit": "b" * 40},
        inputs=[source],
        outputs=[output],
    )

    assert manifest["inputs"][0]["sha256"] == sha256_file(source)
    assert manifest["outputs"][0]["sha256"] == sha256_file(output)


def test_diagnostic_set_requires_one_commit_protocol_path_and_hash(
    tmp_path,
) -> None:
    feature = tmp_path / "features.parquet"
    feature.write_bytes(b"daily-features")
    provenance = {
        "protocol_version": "v4",
        "code_commit": "c" * 40,
        "feature_path": canonical_repo_path(feature, tmp_path),
        "feature_sha256": sha256_file(feature),
    }
    diagnostics = []
    for phase in (3, 4, 5, 6):
        path = tmp_path / f"phase{phase}.json"
        path.write_text(
            __import__("json").dumps({"provenance": provenance}),
            encoding="utf-8",
        )
        diagnostics.append(path)

    assert validate_diagnostic_set(
        diagnostics,
        feature,
        protocol_version="v4",
        repo_root=tmp_path,
        expected_commit="c" * 40,
    ) == provenance

    diagnostics[-1].write_text(
        __import__("json").dumps(
            {"provenance": {**provenance, "feature_sha256": "stale"}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feature_sha256"):
        validate_diagnostic_set(
            diagnostics,
            feature,
            protocol_version="v4",
            repo_root=tmp_path,
            expected_commit="c" * 40,
        )


def test_freeze_release_refuses_dirty_source_tree(tmp_path, monkeypatch) -> None:
    feature = tmp_path / "feature.parquet"
    feature.write_bytes(b"feature")
    monkeypatch.setattr(freeze_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "release_dirty_paths",
        lambda _root: ["src/dirty.py"],
    )

    with pytest.raises(RuntimeError, match="clean source tree"):
        freeze_module.freeze_release(
            feature,
            diagnostics=(),
            outputs=(),
            manifest_output=tmp_path / "manifest.json",
        )


def test_freeze_release_hashes_declared_diagnostics_and_outputs(
    tmp_path, monkeypatch
) -> None:
    feature = tmp_path / "feature.parquet"
    lock = tmp_path / "requirements.lock"
    diagnostic = tmp_path / "phase.json"
    output = tmp_path / "result.csv"
    feature.write_bytes(b"feature")
    lock.write_text("numpy==2.4.6\n", encoding="utf-8")
    diagnostic.write_text("{}", encoding="utf-8")
    output.write_text("metric,value\ncrps,0.1\n", encoding="utf-8")
    provenance = {
        "protocol_version": "v4",
        "code_commit": "d" * 40,
        "feature_path": "feature.parquet",
        "feature_sha256": sha256_file(feature),
    }
    monkeypatch.setattr(freeze_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        freeze_module, "release_dirty_paths", lambda _root: []
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_diagnostic_set",
        lambda *_args, **_kwargs: provenance,
    )
    destination = tmp_path / "release_manifest.json"

    freeze_module.freeze_release(
        feature,
        diagnostics=(diagnostic,),
        outputs=(output,),
        manifest_output=destination,
    )

    manifest = json.loads(destination.read_text(encoding="utf-8"))
    input_hashes = {entry["path"]: entry["sha256"] for entry in manifest["inputs"]}
    output_hashes = {
        entry["path"]: entry["sha256"] for entry in manifest["outputs"]
    }
    assert input_hashes == {
        "feature.parquet": sha256_file(feature),
        "phase.json": sha256_file(diagnostic),
        "requirements.lock": sha256_file(lock),
    }
    assert output_hashes == {"result.csv": sha256_file(output)}
    assert manifest["provenance"] == provenance
