"""Open Phase 3 test data once and write a common-sample ML benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.ml_benchmark import run_ml_common_sample_benchmark
from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.models.gradient_boosted_direction import (
    load_hist_gradient_boosting_model,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--training-diagnostics", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--open-test",
        action="store_true",
        help="Explicitly authorize the protocol's one test-fold evaluation.",
    )
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
    if not args.open_test:
        raise RuntimeError(
            "test fold remains closed; pass --open-test only after model freeze"
        )
    protocol = load_ml_protocol(args.config)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    training = json.loads(
        args.training_diagnostics.read_text(encoding="utf-8")
    )
    config_sha256 = sha256_file(args.config)
    if metadata.get("protocol_version") != protocol.protocol_version:
        raise ValueError("dataset protocol does not match config")
    if metadata.get("config_sha256") != config_sha256:
        raise ValueError("dataset config SHA-256 does not match")
    if training.get("protocol_version") != protocol.protocol_version:
        raise ValueError("model training protocol does not match config")
    if training.get("config_sha256") != config_sha256:
        raise ValueError("model training config SHA-256 does not match")
    if training.get("test_fold_read") is not False:
        raise ValueError("training diagnostics do not prove a closed test fold")
    if training.get("test_metrics") is not None:
        raise ValueError("training diagnostics already contain test metrics")
    if sha256_file(args.model) != training.get("model_sha256"):
        raise ValueError("model SHA-256 does not match training diagnostics")

    dataset = _resolve_dataset(metadata, args.horizon, args.dataset)
    registered_dataset_hash = metadata["datasets"][str(args.horizon)].get(
        "sha256"
    )
    if (
        registered_dataset_hash
        and sha256_file(dataset) != registered_dataset_hash
    ):
        raise ValueError("dataset SHA-256 does not match metadata")
    model = load_hist_gradient_boosting_model(args.model)
    columns = [
        "timestamp",
        "utc_day",
        "anchor_row",
        "fold",
        "target_up",
        *dict.fromkeys(
            [
                *model.feature_names,
                "obi",
                "tick_direction",
                "obi_change",
                "abs_obi",
                "log_trade_intensity",
                *(f"tick_direction_lag_{lag}" for lag in range(1, 6)),
            ]
        ),
    ]
    frame = pd.read_parquet(dataset, columns=columns).sort_values(
        "timestamp", kind="stable"
    )
    result = run_ml_common_sample_benchmark(frame, model, protocol)

    asset = metadata["asset"]
    output = args.output_directory or (
        ROOT / "results" / "ml_benchmarks"
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = f"ml_benchmark_{asset}_h{args.horizon}"
    summary_path = output / f"{stem}_summary.csv"
    predictions_path = output / f"{stem}_predictions.parquet"
    json_path = output / f"{stem}.json"
    result.summary.to_csv(summary_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    payload = {
        "kind": "ml_common_sample_benchmark",
        "status": "exploratory_test_fold_opened",
        "protocol_version": protocol.protocol_version,
        "asset": asset,
        "horizon_ticks": args.horizon,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": canonical_repo_path(dataset, ROOT),
        "dataset_sha256": registered_dataset_hash,
        "model_path": canonical_repo_path(args.model, ROOT),
        "model_sha256": training["model_sha256"],
        "config_sha256": config_sha256,
        "summary": result.summary.to_dict(orient="records"),
        "comparisons_hgb_minus_baseline": result.comparisons,
        "diagnostics": result.diagnostics,
        "artifacts": {
            "summary": {
                "path": canonical_repo_path(summary_path, ROOT),
                "sha256": sha256_file(summary_path),
            },
            "predictions": {
                "path": canonical_repo_path(predictions_path, ROOT),
                "sha256": sha256_file(predictions_path),
            },
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Benchmark: {json_path}")
    print("Test fold: opened and recorded")


if __name__ == "__main__":
    main()
