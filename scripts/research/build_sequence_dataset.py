"""Build Phase 4 causal sequence shards aligned to Phase 1 samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.sequence_dataset import build_sequence_dataset_shards
from src.evaluation.ml_protocol import DEFAULT_ML_CONFIG, load_ml_protocol


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        choices=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        required=True,
    )
    parser.add_argument("--phase1-metadata", type=Path, required=True)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_ML_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--official", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_ml_protocol(args.config)
    feature_path = args.feature_path or (
        ROOT / protocol.raw["assets"][args.asset]["feature_path"]
    )
    output = args.output_directory or (
        ROOT / "data" / "assets" / args.asset.lower() / "ml" / "sequences"
    )
    build = build_sequence_dataset_shards(
        feature_path,
        args.phase1_metadata,
        output,
        asset=args.asset,
        protocol=protocol,
        config_path=args.config,
        batch_size=args.batch_size,
        max_days=args.max_days,
        official=args.official,
        repo_root=ROOT,
    )
    print(f"Manifest: {build.manifest_path}")
    print(f"Shards: {len(build.shard_paths)}")
    print("Normalization source: train fold only")


if __name__ == "__main__":
    main()
