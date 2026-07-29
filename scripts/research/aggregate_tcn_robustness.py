"""Aggregate three independently refitted pretest TCN robustness reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.evaluation.tcn_robustness import aggregate_cross_asset_robustness


ROOT = Path(__file__).resolve().parents[2]


def _resolve_report_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_report_provenance(
    report: Mapping[str, Any],
    *,
    config_sha256: str,
    seeds: list[int],
) -> None:
    asset = report.get("asset")
    models = report.get("models")
    diagnostics = report.get("training_diagnostics")
    if (
        not isinstance(models, Mapping)
        or not isinstance(diagnostics, Mapping)
        or set(models) != {str(seed) for seed in seeds}
        or set(diagnostics) != {str(seed) for seed in seeds}
    ):
        raise ValueError(f"incomplete model provenance for asset {asset}")
    for seed in seeds:
        key = str(seed)
        model_entry = models[key]
        diagnostic_entry = diagnostics[key]
        model_path = _resolve_report_path(str(model_entry["path"]))
        diagnostic_path = _resolve_report_path(str(diagnostic_entry["path"]))
        if (
            sha256_file(model_path) != model_entry.get("sha256")
            or sha256_file(diagnostic_path) != diagnostic_entry.get("sha256")
        ):
            raise ValueError(f"artifact SHA-256 mismatch for {asset}, seed {seed}")
        payload = json.loads(
            diagnostic_path.read_text(encoding="utf-8")
        )
        if (
            payload.get("asset") != asset
            or payload.get("random_seed") != seed
            or payload.get("horizon_ticks") != report.get("horizon_ticks")
            or payload.get("protocol_version") != report.get("protocol_version")
            or payload.get("config_sha256") != config_sha256
            or payload.get("manifest_sha256")
            != report.get("manifest_sha256")
            or payload.get("model_sha256") != model_entry.get("sha256")
            or payload.get("test_fold_read") is not False
            or payload.get("test_metrics") is not None
        ):
            raise ValueError(
                f"training provenance mismatch for {asset}, seed {seed}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="repeat for BTCUSDT, ETHUSDT and BNBUSDT",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    config_sha256 = sha256_file(args.config)
    paths = [path.resolve() for path in args.report]
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    for report in reports:
        if report.get("config_sha256") != config_sha256:
            raise ValueError("robustness report config SHA-256 does not match")
        _validate_report_provenance(
            report,
            config_sha256=config_sha256,
            seeds=list(protocol.raw["robustness"]["seeds"]),
        )
    aggregate = aggregate_cross_asset_robustness(reports, protocol)
    aggregate.update(
        {
            "config_sha256": config_sha256,
            "reports": [
                {
                    "path": canonical_repo_path(path, ROOT),
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        }
    )
    output = args.output or (
        ROOT
        / "reports"
        / "research"
        / f"tcn_cross_asset_h{aggregate['horizon_ticks']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Cross-asset report: {output}")
    print("Policy: independently refit each asset; equal-asset summary")
    print("Test fold: not read")


if __name__ == "__main__":
    main()
