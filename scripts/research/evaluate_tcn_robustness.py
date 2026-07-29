"""Evaluate registered TCN seeds on pretest folds and train-defined regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.evaluation.tcn_robustness import evaluate_tcn_robustness
from src.models.temporal_convolutional_direction import (
    load_temporal_convolutional_model,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="repeat once for every registered robustness seed",
    )
    parser.add_argument(
        "--diagnostics",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="repeat for the matching seed training diagnostics",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _model_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text or not path_text:
            raise ValueError("--model must use SEED=PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate model seed: {seed}")
        result[seed] = Path(path_text).resolve()
    return result


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(args.config)
    if manifest.get("protocol_version") != protocol.protocol_version:
        raise ValueError("sequence manifest protocol does not match config")
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("sequence manifest config SHA-256 does not match")
    paths = _model_paths(args.model)
    diagnostic_paths = _model_paths(args.diagnostics)
    if set(diagnostic_paths) != set(paths):
        raise ValueError("model and diagnostics seeds must match")
    manifest_sha256 = sha256_file(manifest_path)
    for seed, diagnostic_path in diagnostic_paths.items():
        diagnostic = json.loads(
            diagnostic_path.read_text(encoding="utf-8")
        )
        if (
            diagnostic.get("random_seed") != seed
            or diagnostic.get("asset") != manifest.get("asset")
            or diagnostic.get("horizon_ticks") != args.horizon
            or diagnostic.get("protocol_version") != protocol.protocol_version
            or diagnostic.get("config_sha256") != config_sha256
            or diagnostic.get("manifest_sha256") != manifest_sha256
            or diagnostic.get("model_sha256") != sha256_file(paths[seed])
            or diagnostic.get("test_fold_read") is not False
            or diagnostic.get("test_metrics") is not None
            or diagnostic.get("status")
            != "trained_calibrated_without_test_access"
        ):
            raise ValueError(
                f"training diagnostics mismatch for seed {seed}"
            )
    models = {
        seed: load_temporal_convolutional_model(path)
        for seed, path in paths.items()
    }
    report = evaluate_tcn_robustness(
        manifest,
        models,
        args.horizon,
        protocol,
        repo_root=ROOT,
        device=args.device,
    )
    report.update(
        {
            "manifest_path": canonical_repo_path(manifest_path, ROOT),
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "models": {
                str(seed): {
                    "path": canonical_repo_path(path, ROOT),
                    "sha256": sha256_file(path),
                }
                for seed, path in sorted(paths.items())
            },
            "training_diagnostics": {
                str(seed): {
                    "path": canonical_repo_path(path, ROOT),
                    "sha256": sha256_file(path),
                }
                for seed, path in sorted(diagnostic_paths.items())
            },
        }
    )
    output = args.output or (
        ROOT
        / "reports"
        / "research"
        / (
            f"tcn_robustness_{report['asset']}"
            f"_h{args.horizon}.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Robustness report: {output}")
    print(f"Seeds: {', '.join(str(seed) for seed in report['seeds'])}")
    print("Test fold: not read")


if __name__ == "__main__":
    main()
