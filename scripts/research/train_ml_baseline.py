"""Train Phase 2 Histogram Gradient Boosting without reading the test fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.temporal_features import TemporalFeatureSpec
from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.models.gradient_boosted_direction import (
    TRAINING_FOLDS,
    save_hist_gradient_boosting_model,
    train_hist_gradient_boosting,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    return parser.parse_args()


def _resolve_dataset(
    metadata: dict, horizon: int, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    entry = metadata["datasets"].get(str(horizon))
    if entry is None:
        raise ValueError(f"metadata does not register horizon {horizon}")
    path = Path(entry["path"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    if args.horizon not in protocol.evaluation_horizons:
        raise ValueError("horizon is not registered by the ML protocol")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != protocol.protocol_version:
        raise ValueError("dataset metadata protocol does not match config")
    config_sha256 = sha256_file(args.config)
    if metadata.get("config_sha256") != config_sha256:
        raise ValueError("dataset metadata config SHA-256 does not match")
    expected_features = TemporalFeatureSpec.from_protocol(
        protocol.raw
    ).feature_names
    if tuple(metadata.get("feature_names", ())) != expected_features:
        raise ValueError("dataset feature schema does not match config")
    dataset = _resolve_dataset(metadata, args.horizon, args.dataset)
    registered_hash = metadata["datasets"][str(args.horizon)].get("sha256")
    if registered_hash and sha256_file(dataset) != registered_hash:
        raise ValueError("dataset SHA-256 does not match metadata")
    feature_names = tuple(metadata["feature_names"])
    columns = ["fold", "target_up", *feature_names]
    frame = pd.read_parquet(
        dataset,
        columns=columns,
        filters=[("fold", "in", list(TRAINING_FOLDS))],
    )
    model, diagnostics = train_hist_gradient_boosting(
        frame, feature_names, protocol
    )

    asset = metadata["asset"]
    model_output = args.model_output or (
        ROOT
        / "results"
        / "ml_models"
        / f"hist_gradient_boosting_{asset}_h{args.horizon}.pkl"
    )
    diagnostics_output = args.diagnostics_output or (
        ROOT
        / "results"
        / "ml_models"
        / f"hist_gradient_boosting_{asset}_h{args.horizon}.json"
    )
    save_hist_gradient_boosting_model(model, model_output)
    model_sha256 = sha256_file(model_output)
    diagnostics.update(
        {
            "asset": asset,
            "horizon_ticks": args.horizon,
            "dataset_path": canonical_repo_path(dataset, ROOT),
            "dataset_sha256": registered_hash,
            "metadata_path": canonical_repo_path(args.metadata, ROOT),
            "config_sha256": config_sha256,
            "model_path": canonical_repo_path(model_output, ROOT),
            "model_sha256": model_sha256,
            "test_metrics": None,
            "status": "trained_without_test_access",
        }
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Model: {model_output}")
    print(f"Diagnostics: {diagnostics_output}")
    print("Test fold: not read")


if __name__ == "__main__":
    main()
