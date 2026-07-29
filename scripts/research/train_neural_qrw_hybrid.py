"""Train every Phase 7 neural-adaptive QRW ablation without reading test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.models.neural_adaptive_qrw import (
    build_hybrid_ablation_report,
    save_neural_adaptive_qrw_model,
    train_neural_adaptive_qrw,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(args.config)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest.get("protocol_version") != protocol.protocol_version:
        raise ValueError("sequence manifest protocol does not match config")
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("sequence manifest config SHA-256 does not match")
    asset = str(manifest["asset"])
    output = args.output_directory or ROOT / "results" / "ml_models"
    diagnostics: dict[str, dict] = {}
    artifacts: dict[str, dict[str, str]] = {}
    variants = protocol.raw["models"]["phase_7_neural_adaptive_qrw"][
        "variants"
    ]
    for variant in variants:
        model, values = train_neural_adaptive_qrw(
            manifest,
            args.horizon,
            variant,
            protocol,
            repo_root=ROOT,
            device=args.device,
        )
        stem = (
            f"neural_qrw_{asset}_h{args.horizon}_{variant}"
        )
        model_path = output / f"{stem}.pt"
        diagnostic_path = output / f"{stem}.json"
        save_neural_adaptive_qrw_model(model, model_path)
        values.update(
            {
                "asset": asset,
                "manifest_path": canonical_repo_path(manifest_path, ROOT),
                "manifest_sha256": manifest_sha256,
                "config_sha256": config_sha256,
                "model_path": canonical_repo_path(model_path, ROOT),
                "model_sha256": sha256_file(model_path),
            }
        )
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostics[str(variant)] = values
        artifacts[str(variant)] = {
            "model_path": canonical_repo_path(model_path, ROOT),
            "model_sha256": values["model_sha256"],
            "diagnostics_path": canonical_repo_path(
                diagnostic_path, ROOT
            ),
            "diagnostics_sha256": sha256_file(diagnostic_path),
        }
        print(
            f"{variant}: epoch={values['selected_epoch']}, "
            f"calibrator={values['selected_calibrator']}"
        )
    report = build_hybrid_ablation_report(diagnostics, protocol)
    report.update(
        {
            "asset": asset,
            "horizon_ticks": args.horizon,
            "manifest_path": canonical_repo_path(manifest_path, ROOT),
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "artifacts": artifacts,
        }
    )
    report_path = output / (
        f"neural_qrw_{asset}_h{args.horizon}_ablation.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Ablation report: {report_path}")
    print("Test fold: not read")


if __name__ == "__main__":
    main()
