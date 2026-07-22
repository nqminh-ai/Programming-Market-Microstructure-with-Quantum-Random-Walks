"""Generate the Track A A1 volatility demo artifacts."""

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


def generate_synthetic_tick_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0005, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-12 09:00:00", periods=n, freq="s"),
            "price": 10_000.0 * np.exp(np.cumsum(returns)),
            "volume": rng.uniform(10, 500, n),
            "obi": np.clip(rng.normal(0, 0.3, n), -1, 1),
        }
    )


def load_tick_data(date_str: str, limit: int = 1000) -> pd.DataFrame:
    data_dir = ROOT / "data" / "assets" / "btcusdt" / "processed"
    candidates = sorted(data_dir.glob(f"*{date_str}*.parquet")) or sorted(data_dir.glob("*.parquet"))
    for path in candidates:
        try:
            frame = pd.read_parquet(path).head(limit).copy()
            if "price" not in frame.columns:
                continue
            if "timestamp" not in frame.columns:
                frame["timestamp"] = pd.date_range("2026-06-12", periods=len(frame), freq="s")
            if "obi" not in frame.columns:
                signed = np.sign(frame["price"].diff().fillna(0.0))
                frame["obi"] = signed.rolling(20, min_periods=1).mean()
            print(f"[OK] Loaded {path.name} ({len(frame):,} rows for demo)")
            return frame.reset_index(drop=True)
        except Exception as exc:
            print(f"[WARN] Skipped {path.name}: {exc}")
    print("[INFO] No usable processed data; using deterministic synthetic ticks")
    return generate_synthetic_tick_data()


def run_volatility_demo(
    date_str: str,
    window: int,
    output_dir: Path | None = None,
) -> dict:
    from src.models.qrw_core import QuantumRandomWalk
    from src.models.volatility_forecaster import QRWVolatilityForecaster

    if window not in {5, 15, 30}:
        raise ValueError("window must be one of 5, 15, or 30")
    tick_df = load_tick_data(date_str)
    demo_dir = output_dir or ROOT / "results" / "track_a"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    comparisons: dict[int, pd.DataFrame] = {}
    metrics_by_window: dict[str, dict] = {}
    for candidate in (5, 15, 30):
        qrw = QuantumRandomWalk(n_positions=2 * candidate + 3)
        model = QRWVolatilityForecaster(qrw, window_sizes=[5, 15, 30])
        comparison = model.compare_with_garch(tick_df, window=candidate)
        comparisons[candidate] = comparison
        metrics = model.benchmark_metrics(comparison)
        metrics_by_window[f"window_{candidate}"] = {
            key: (round(value, 8) if isinstance(value, float) else value)
            for key, value in metrics.items()
        }
        print(
            f"[OK] window={candidate:2d}: n={len(comparison):4d}, "
            f"QRW MAE={metrics['mae_qrw']:.6f}, GARCH MAE={metrics['mae_garch']:.6f}"
        )

    selected = comparisons[window].copy()
    selected.insert(0, "timestamp", tick_df.loc[selected.index, "timestamp"].to_numpy())
    selected.to_parquet(demo_dir / "vol_metrics.parquet", index=False)

    latest = selected.iloc[-1]
    regime = QRWVolatilityForecaster(None).vol_regime(float(latest["vol_qrw"]))
    payload = {
        "date": date_str,
        "window": window,
        "n_observations": int(len(selected)),
        "current_qrw_vol": round(float(latest["vol_qrw"]), 8),
        "current_garch_vol": round(float(latest["vol_garch"]), 8),
        "current_realized_vol": round(float(latest["vol_realized"]), 8),
        "vol_regime": regime,
        **metrics_by_window,
    }
    metrics_path = demo_dir / "vol_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#080C12")
    for axis in axes:
        axis.set_facecolor("#0D1420")
        axis.tick_params(colors="#8FA3BE")
        axis.grid(alpha=0.15, color="#4A6080")
        for spine in axis.spines.values():
            spine.set_color("#1C2A3D")

    x = np.arange(len(selected))
    axes[0].plot(x, selected["vol_qrw"] * 100, color="#00D4FF", label="QRW")
    axes[0].plot(x, selected["vol_garch"] * 100, color="#B388FF", label="GARCH", linestyle="--")
    axes[0].plot(x, selected["vol_realized"] * 100, color="#E8F0FA", label="Realized", linewidth=1)
    axes[0].set_title("Causal volatility comparison", color="#E8F0FA")
    axes[0].set_ylabel("Annualized vol (%)", color="#8FA3BE")
    axes[0].legend(facecolor="#0D1420", labelcolor="#E8F0FA")

    steps = np.arange(1, window + 1)
    axes[1].plot(steps, steps**2, color="#00D4FF", label="QRW variance ∝ t²")
    axes[1].plot(steps, steps, color="#B388FF", linestyle="--", label="CRW variance ∝ t")
    axes[1].set_title("Theoretical variance growth", color="#E8F0FA")
    axes[1].set_xlabel("Steps", color="#8FA3BE")
    axes[1].set_ylabel("Relative variance", color="#8FA3BE")
    axes[1].legend(facecolor="#0D1420", labelcolor="#E8F0FA")
    fig.tight_layout(pad=2.0)
    figure_dir = ROOT / "figures" / "track_a"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "volatility_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="#080C12")
    plt.close(fig)

    print(f"[OK] Wrote {demo_dir / 'vol_metrics.parquet'}")
    print(f"[OK] Wrote {metrics_path}")
    print(f"[OK] Wrote {figure_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-06-12")
    parser.add_argument("--window", type=int, choices=[5, 15, 30], default=15)
    args = parser.parse_args()
    run_volatility_demo(args.date, args.window)


if __name__ == "__main__":
    main()
