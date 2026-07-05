"""Tests for release provenance and artifact checksum guards."""

from __future__ import annotations

import hashlib

import pytest

from src.evaluation.provenance import (
    build_sha256_manifest,
    canonical_repo_path,
    sha256_file,
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
