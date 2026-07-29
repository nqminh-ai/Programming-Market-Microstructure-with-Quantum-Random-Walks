"""Build the frozen Phase 1 causal ML datasets without loading 69 days at once."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.ml_dataset import build_ml_dataset_files
from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        choices=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        required=True,
    )
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Development smoke cap; capped outputs cannot support claims.",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Require 69 days, a clean tree and SHA-256 every dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    asset_settings = protocol.raw["assets"][args.asset]
    feature_path = args.feature_path or ROOT / asset_settings["feature_path"]
    output_directory = args.output_directory or (
        ROOT / "data" / "assets" / args.asset.lower() / "ml"
    )
    build = build_ml_dataset_files(
        feature_path,
        output_directory,
        asset=args.asset,
        protocol=protocol,
        config_path=args.config,
        batch_size=args.batch_size,
        max_days=args.max_days,
        official=args.official,
        repo_root=ROOT,
    )
    print(f"Metadata: {build.metadata_path}")
    for horizon, path in build.datasets.items():
        print(f"h={horizon:,}: {path}")


if __name__ == "__main__":
    main()
