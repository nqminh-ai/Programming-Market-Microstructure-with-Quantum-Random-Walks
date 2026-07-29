"""Build the Phase 8 read-only pretest release manifest and reproduction script."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.ml_release import (
    build_ml_release_bundle,
    validate_release_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="ROLE=PATH",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--official", action="store_true")
    return parser.parse_args()


def _artifact_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path_text = value.partition("=")
        if not separator or not role or not path_text:
            raise ValueError("--artifact must use ROLE=PATH")
        if role in result:
            raise ValueError(f"duplicate artifact role: {role}")
        result[role] = Path(path_text).resolve()
    return result


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    artifacts = _artifact_paths(args.artifact)
    asset, horizon, _ = validate_release_artifacts(
        artifacts,
        protocol,
        config_path=args.config,
    )
    if asset != args.asset.upper():
        raise ValueError("--asset does not match release artifacts")
    if horizon != args.horizon:
        raise ValueError("--horizon does not match release artifacts")
    output = args.output_directory or (
        ROOT
        / "reports"
        / "research"
        / "ml_release"
        / args.asset.upper()
        / f"h{args.horizon}"
    )
    build = build_ml_release_bundle(
        artifacts,
        output,
        protocol=protocol,
        config_path=args.config,
        repo_root=ROOT,
        official=args.official,
    )
    print(f"Release manifest: {build.manifest_path}")
    print(f"Reproduction script: {build.reproduction_path}")
    print(f"Status: {build.manifest['status']}")
    print("Holdout: closed")


if __name__ == "__main__":
    main()
