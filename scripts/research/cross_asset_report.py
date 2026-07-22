"""Render the final cross-asset ranking figure with Matplotlib only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    scorecard_path = ROOT / "results" / "cross_asset" / "scorecard.csv"
    output_dir = ROOT / "figures" / "cross_asset"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not scorecard_path.exists():
        raise FileNotFoundError(
            f"{scorecard_path} is missing; run cross_asset_benchmark.py first"
        )

    frame = pd.read_csv(scorecard_path).sort_values("avg_overall_rank")
    print("=== Final Cross-Asset Scorecard ===")
    print(frame.to_string(index=False))

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(frame["model"], frame["avg_overall_rank"], color="#00A6C8")
    axis.invert_yaxis()
    axis.set_title("Cross-Asset Model Ranking (Lower is Better)")
    axis.set_xlabel("Average Overall Rank")
    axis.set_ylabel("Model")
    figure.tight_layout()
    output_path = output_dir / "model_ranking.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
