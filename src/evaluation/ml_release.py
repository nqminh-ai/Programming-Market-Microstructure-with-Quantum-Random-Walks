"""Build an auditable, pre-holdout ML research release bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.ml_protocol import (
    DEFAULT_ML_CONFIG,
    MLDirectionalProtocol,
    load_ml_protocol,
)
from src.evaluation.provenance import (
    canonical_repo_path,
    git_commit,
    release_dirty_paths,
    sha256_file,
)


@dataclass(frozen=True)
class MLReleaseBuild:
    """Paths and payload produced by one Phase 8 bundle build."""

    manifest_path: Path
    reproduction_path: Path
    manifest: Mapping[str, Any]


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"release artifact must be a JSON object: {path}")
    return payload


def _assert_test_closed(value: Any, location: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "test_fold_read" and child is not False:
                raise ValueError(f"test access is not closed at {child_location}")
            if key == "test_metrics" and child is not None:
                raise ValueError(f"test metrics are present at {child_location}")
            _assert_test_closed(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_test_closed(child, f"{location}[{index}]")


def _artifact_identity(
    role: str,
    payload: Mapping[str, Any],
    *,
    asset: str,
    horizon: int,
) -> None:
    if role == "phase1_dataset_metadata":
        if payload.get("asset") != asset or horizon not in {
            int(value) for value in payload.get("horizons_ticks", ())
        }:
            raise ValueError("Phase 1 metadata asset/horizon mismatch")
        return
    if role == "phase4_sequence_manifest":
        registered_horizons = {
            int(entry["horizon_ticks"])
            for entry in payload.get("shards", ())
        }
        if payload.get("asset") != asset or horizon not in registered_horizons:
            raise ValueError("Phase 4 manifest asset/horizon mismatch")
        return
    if role == "phase6_cross_asset":
        if (
            horizon != int(payload.get("horizon_ticks", -1))
            or asset not in payload.get("assets", ())
        ):
            raise ValueError("cross-asset robustness scope mismatch")
        return
    if (
        payload.get("asset") != asset
        or int(payload.get("horizon_ticks", -1)) != horizon
    ):
        raise ValueError(f"{role} asset/horizon mismatch")


def validate_release_artifacts(
    artifact_paths: Mapping[str, str | Path],
    protocol: MLDirectionalProtocol,
    *,
    config_path: str | Path = DEFAULT_ML_CONFIG,
) -> tuple[str, int, dict[str, Mapping[str, Any]]]:
    """Validate roles, schemas, common scope, hashes and closed-test policy."""
    settings = protocol.raw["release"]
    required = dict(settings["required_artifact_roles"])
    optional = dict(settings["optional_artifact_roles"])
    observed = set(artifact_paths)
    missing = sorted(set(required).difference(observed))
    unexpected = sorted(observed.difference(set(required).union(optional)))
    if missing or unexpected:
        raise ValueError(
            f"release artifact roles mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    config_hash = sha256_file(config_path)
    payloads = {
        role: _load_json(Path(path).resolve())
        for role, path in artifact_paths.items()
    }
    dataset = payloads["phase1_dataset_metadata"]
    asset = str(dataset.get("asset"))
    if asset not in protocol.assets:
        raise ValueError("release asset is not registered")
    horizon_candidates = {
        int(payload.get("horizon_ticks"))
        for role, payload in payloads.items()
        if role
        not in (
            "phase1_dataset_metadata",
            "phase4_sequence_manifest",
        )
        and payload.get("horizon_ticks") is not None
    }
    if len(horizon_candidates) != 1:
        raise ValueError("release artifacts must share one horizon")
    horizon = horizon_candidates.pop()
    if horizon not in protocol.evaluation_horizons:
        raise ValueError("release horizon is not registered")
    for role, payload in payloads.items():
        expected_kind = required.get(role, optional.get(role))
        if payload.get("kind") != expected_kind:
            raise ValueError(
                f"{role} kind must be {expected_kind!r}"
            )
        if payload.get("protocol_version") != protocol.protocol_version:
            raise ValueError(f"{role} protocol version mismatch")
        if payload.get("config_sha256") != config_hash:
            raise ValueError(f"{role} config SHA-256 mismatch")
        _assert_test_closed(payload, role)
        _artifact_identity(
            role, payload, asset=asset, horizon=horizon
        )
    return asset, horizon, payloads


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _reproduction_text(
    *,
    asset: str,
    horizon: int,
    artifact_paths: Mapping[str, str | Path],
    official: bool,
) -> str:
    lower = asset.lower()
    metadata = artifact_paths["phase1_dataset_metadata"]
    sequence = artifact_paths["phase4_sequence_manifest"]
    official_flag = " --official" if official else ""
    artifact_arguments = " ".join(
        f"--artifact {role}={_powershell_quote(Path(path).resolve())}"
        for role, path in sorted(artifact_paths.items())
    )
    model_arguments = " ".join(
        (
            f"--model {seed}=results/ml_models/"
            f"temporal_convolutional_{asset}_h{horizon}_s{seed}.pt"
        )
        for seed in (2026, 2027, 2028)
    )
    diagnostic_arguments = " ".join(
        (
            f"--diagnostics {seed}=results/ml_models/"
            f"temporal_convolutional_{asset}_h{horizon}_s{seed}.json"
        )
        for seed in (2026, 2027, 2028)
    )
    release_flag = " --official" if official else ""
    return "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            "",
            "# Phase 1: causal non-overlapping horizon datasets",
            (
                "python -m scripts.research.build_ml_dataset "
                f"--asset {asset}{official_flag}"
            ),
            "",
            "# Phase 4: causal sequence shards",
            (
                "python -m scripts.research.build_sequence_dataset "
                f"--asset {asset} "
                f"--phase1-metadata {_powershell_quote(metadata)}"
                f"{official_flag}"
            ),
            "",
            "# Phase 2: tabular baseline",
            (
                "python -m scripts.research.train_ml_baseline "
                f"--metadata {_powershell_quote(metadata)} "
                f"--horizon {horizon}"
            ),
            "",
            "# Phase 5/6: TCN registered seeds, then pretest robustness",
            "2026, 2027, 2028 | ForEach-Object {",
            (
                "  python -m scripts.research.train_tcn_baseline "
                f"--manifest {_powershell_quote(sequence)} "
                f"--horizon {horizon} --seed $_"
            ),
            "}",
            (
                "python -m scripts.research.evaluate_tcn_robustness "
                f"--manifest {_powershell_quote(sequence)} "
                f"--horizon {horizon} {model_arguments} "
                f"{diagnostic_arguments}"
            ),
            "",
            "# Phase 7: mandatory hybrid ablations",
            (
                "python -m scripts.research.train_neural_qrw_hybrid "
                f"--manifest {_powershell_quote(sequence)} "
                f"--horizon {horizon}"
            ),
            "",
            "# Phase 8: rebuild this release bundle",
            (
                "python -m scripts.research.build_ml_release "
                f"--asset {asset} --horizon {horizon} "
                f"{artifact_arguments}{release_flag}"
            ),
            "",
            f"# Canonical generated-data root: data/assets/{lower}/ml/",
            "# No command in this script opens the test fold.",
            "",
        ]
    )


def build_ml_release_bundle(
    artifact_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    *,
    protocol: MLDirectionalProtocol | None = None,
    config_path: str | Path = DEFAULT_ML_CONFIG,
    repo_root: str | Path | None = None,
    official: bool = False,
) -> MLReleaseBuild:
    """Validate upstream JSON and write a read-only pretest release bundle."""
    selected_protocol = protocol or load_ml_protocol(config_path)
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    asset, horizon, payloads = validate_release_artifacts(
        artifact_paths, selected_protocol, config_path=config_path
    )
    dirty = release_dirty_paths(root)
    if official:
        if dirty:
            raise RuntimeError(
                "official ML release requires a clean source tree; "
                f"dirty paths: {', '.join(dirty[:8])}"
            )
        for role in (
            "phase1_dataset_metadata",
            "phase4_sequence_manifest",
        ):
            if payloads[role].get("official") is not True:
                raise ValueError(
                    f"official ML release requires official {role}"
                )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reproduction_path = output / "reproduce_ml_release.ps1"
    reproduction_path.write_text(
        _reproduction_text(
            asset=asset,
            horizon=horizon,
            artifact_paths=artifact_paths,
            official=official,
        ),
        encoding="utf-8",
    )
    artifact_rows = []
    for role, path_value in sorted(artifact_paths.items()):
        path = Path(path_value).resolve()
        artifact_rows.append(
            {
                "role": role,
                "kind": payloads[role]["kind"],
                "path": canonical_repo_path(path, root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    config_source = Path(config_path).resolve()
    manifest = {
        "kind": "ml_directional_release",
        "status": (
            selected_protocol.raw["release"]["status"]
            if official
            else "development_pretest_bundle"
        ),
        "protocol_version": selected_protocol.protocol_version,
        "asset": asset,
        "horizon_ticks": horizon,
        "official": bool(official),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(root),
        "dirty_source_paths": dirty,
        "config_path": canonical_repo_path(config_source, root),
        "config_sha256": sha256_file(config_source),
        "hash_algorithm": "sha256",
        "artifacts": artifact_rows,
        "reproduction": {
            "shell": "powershell",
            "path": canonical_repo_path(reproduction_path, root),
            "sha256": sha256_file(reproduction_path),
        },
        "dashboard": {
            "policy": "read_only_manifest_no_recompute",
            "command": (
                "streamlit run src/dashboard/ml_release_dashboard.py"
            ),
        },
        "holdout_state": "closed",
        "test_fold_read": False,
        "test_metrics": None,
        "live_trading_authorized": False,
    }
    manifest_path = output / "ml_release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MLReleaseBuild(
        manifest_path=manifest_path,
        reproduction_path=reproduction_path,
        manifest=manifest,
    )
