"""Train the Phase 5 causal TCN without reading the test fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.models.temporal_convolutional_direction import (
    save_temporal_convolutional_model,
    train_temporal_convolutional_network,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    return parser.parse_args()


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
    if args.horizon not in protocol.evaluation_horizons:
        raise ValueError("horizon is not registered by the ML protocol")
    selected_seed = (
        protocol.random_seed if args.seed is None else int(args.seed)
    )

    model, diagnostics = train_temporal_convolutional_network(
        manifest,
        args.horizon,
        protocol,
        repo_root=ROOT,
        device=args.device,
        random_seed=selected_seed,
    )
    asset = str(manifest["asset"])
    model_output = args.model_output or (
        ROOT
        / "results"
        / "ml_models"
        / f"temporal_convolutional_{asset}_h{args.horizon}_s{selected_seed}.pt"
    )
    diagnostics_output = args.diagnostics_output or (
        ROOT
        / "results"
        / "ml_models"
        / f"temporal_convolutional_{asset}_h{args.horizon}_s{selected_seed}.json"
    )
    save_temporal_convolutional_model(model, model_output)
    diagnostics.update(
        {
            "asset": asset,
            "manifest_path": canonical_repo_path(manifest_path, ROOT),
            "manifest_sha256": sha256_file(manifest_path),
            "config_sha256": config_sha256,
            "model_path": canonical_repo_path(model_output, ROOT),
            "model_sha256": sha256_file(model_output),
            "test_metrics": None,
            "status": "trained_calibrated_without_test_access",
        }
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Model: {model_output}")
    print(f"Diagnostics: {diagnostics_output}")
    print(
        "Selected epoch: "
        f"{diagnostics['selected_epoch']}; "
        f"calibrator: {diagnostics['selected_calibrator']}"
    )
    print("Test fold: not read")


if __name__ == "__main__":
    main()
