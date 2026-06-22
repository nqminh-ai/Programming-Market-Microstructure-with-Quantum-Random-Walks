"""Phase E: Final Integration & Validation."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    scorecard_path = results_dir / "cross_asset_scorecard.csv"
    if not scorecard_path.exists():
        print(f"Error: {scorecard_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(scorecard_path)
    print("=== Final Cross-Asset Scorecard ===")
    print(df.to_string())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="avg_overall_rank",
        y="model",
        hue="model",
        palette="viridis",
        legend=False
    )
    plt.title("Cross-Asset Model Ranking (Lower is Better)")
    plt.xlabel("Average Overall Rank")
    plt.ylabel("Model")
    plt.tight_layout()
    
    out_png = figures_dir / "cross_asset_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved figure to {out_png}")
    
    print("\n[OK] Phase E Integration Complete.")

if __name__ == "__main__":
    main()
