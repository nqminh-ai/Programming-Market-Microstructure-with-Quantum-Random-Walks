"""Calibrate the classical Pareto directional surrogate for diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qrw_heavy_tail import HeavyTailAdaptiveQRW
from scripts.pipelines.phase3_pipeline import resolve_feature_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path)
    print(f"Loading {feature_path}...")
    frame = pd.read_parquet(feature_path)
    
    print("Calibrating classical Pareto directional surrogate...")
    model = HeavyTailAdaptiveQRW(frame, {"n_positions": 101})
    
    # We will log the results of calibration
    out_path = args.results_dir / "heavy_tail_calibration.json"
    params = model.calibrate_two_stage(output_path=out_path)
    
    print("\n--- Calibration Results ---")
    print(f"Tail Index (alpha): {params['tail_index']:.4f}")
    print(f"Jump Scale: {params['jump_scale']:.6f}")
    print(f"Gamma Base: {params['gamma']:.4f}")
    print(f"Regularization: {params['selected_regularization']}")
    print(f"AIC: {params.get('aic', 'N/A')}")
    print(f"BIC: {params.get('bic', 'N/A')}")
    
    print("Quantum state evolution: False")
    print("Eligible for QRW claims: False")

    print(f"\n[OK] Phase C calibration complete. Saved to {out_path}")

if __name__ == "__main__":
    main()
