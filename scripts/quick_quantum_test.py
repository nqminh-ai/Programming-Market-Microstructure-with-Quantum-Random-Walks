"""Quick smoke test: compare the QRW quantum engine against a fair linear
baseline and a neutral baseline on a small subset of one feature file.

Usage
-----
    python scripts/quick_quantum_test.py [--feature-path PATH] [--n-events 50000] [--seed 2026]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qrw_market_sim import MarketQRW

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=ROOT / "data/assets/btcusdt/features/features_BTCUSDT_multiday.parquet",
    )
    parser.add_argument("--n-events", type=int, default=50_000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def quick_test(args: argparse.Namespace) -> None:
    feature_path = args.feature_path.resolve()
    print(f"Loading {feature_path}...")
    frame = pd.read_parquet(feature_path).sort_values("timestamp").reset_index(drop=True)

    frame = frame.iloc[: args.n_events].copy()

    train_size = int(len(frame) * args.train_fraction)
    train_df = frame.iloc[:train_size].copy()
    test_df = frame.iloc[train_size:].copy()

    print(f"Train size: {len(train_df)} events")
    print(f"Test size: {len(test_df)} events")

    print("\nCalibrating Quantum Engine...")
    start_time = time.time()
    config = {
        "n_positions": 101,
        "gamma_base": 0.0,
        "alpha_obi": 0.0,
        "coin_type": "obi_adaptive",
        "quantum_calibration_max_events": 5000,
        "quantum_window": 5,
        "quantum_calibration_seed": args.seed,
    }
    model = MarketQRW(train_df, config)
    # Generated output belongs under results/, not the repository root.
    parameter_path = ROOT / "results" / "quick_quantum_params.json"
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    params = model.calibrate(parameter_path)
    print(f"Calibration took {time.time() - start_time:.1f}s")
    print("Learned Params:")
    print(f"  alpha_obi: {params.get('alpha_obi')}")
    print(f"  alpha_direction: {params.get('alpha_direction')}")
    print(f"  gamma: {params.get('gamma_estimate')}")
    print(f"  alpha_phase (Quantum): {params.get('alpha_phase')}")
    print(f"  quantum_improved: {params.get('quantum_improved')}")

    print("\nEvaluating on Test set...")
    obi = test_df["obi"].to_numpy(dtype=np.float64)[:-1]
    direction = test_df["tick_direction"].to_numpy(dtype=np.float64)[:-1]

    price = test_df["price"].to_numpy(dtype=np.float64)
    delta = np.diff(price)
    target = (delta > 0.0).astype(np.float64)

    # Filter valid events
    valid = np.abs(delta) > 1e-12
    if "obi_valid" in test_df:
        valid &= test_df["obi_valid"].astype(bool).to_numpy()[:-1]

    obi, direction, target = obi[valid], direction[valid], target[valid]

    # 1. Classical Baseline (from Phase 3): fit fair affine baseline on train
    train_obi = train_df["obi"].to_numpy(dtype=np.float64)[:-1]
    train_direction = train_df["tick_direction"].to_numpy(dtype=np.float64)[:-1]
    train_price = train_df["price"].to_numpy(dtype=np.float64)
    train_target = (np.diff(train_price) > 0.0).astype(np.float64)
    train_valid = np.abs(np.diff(train_price)) > 1e-12
    if "obi_valid" in train_df:
        train_valid &= train_df["obi_valid"].astype(bool).to_numpy()[:-1]
    train_obi, train_direction, train_target = (
        train_obi[train_valid],
        train_direction[train_valid],
        train_target[train_valid],
    )

    design = np.column_stack([np.ones(len(train_obi)), train_obi, train_direction])
    linear_coeffs = np.linalg.lstsq(design, train_target, rcond=None)[0]

    linear_prob = np.clip(
        linear_coeffs[0] + linear_coeffs[1] * obi + linear_coeffs[2] * direction,
        0.0,
        1.0,
    )
    linear_brier = np.mean((linear_prob - target) ** 2)

    # 2. Quantum Engine
    start_time = time.time()
    quantum_prob = model.quantum_probabilities(obi, direction)
    quantum_brier = np.mean((quantum_prob - target) ** 2)
    print(f"Evaluation took {time.time() - start_time:.1f}s")

    # 3. Random Baseline
    neutral_brier = np.mean((0.5 - target) ** 2)

    print("\n--- RESULTS (Brier Score, lower is better) ---")
    print(f"Neutral Baseline:  {neutral_brier:.5f}")
    print(f"Fair Linear Model: {linear_brier:.5f}")
    print(f"Genuine QRW Model: {quantum_brier:.5f}")

    if quantum_brier < linear_brier:
        diff = linear_brier - quantum_brier
        print(f"\nSUCCESS! Quantum engine OUTPERFORMS classical linear model by {diff:.5f}")
    else:
        diff = quantum_brier - linear_brier
        print(f"\nFAIL! Classical linear model still beats Quantum engine by {diff:.5f}")


if __name__ == "__main__":
    quick_test(parse_args())
