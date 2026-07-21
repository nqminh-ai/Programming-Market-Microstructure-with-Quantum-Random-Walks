"""Build all pre-computed artifacts for the Track A dashboard.

Usage:
    python scripts/track_a/build_demo.py --date 2026-06-12 --output results/track_a
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.track_a.volatility_demo import load_tick_data, run_volatility_demo
from src.models.anomaly_detector import QRWAnomalyDetector
from src.models.risk_simulator import QRWRiskSimulator, SCENARIOS
from src.strategy.optimizer import QRWStrategyOptimizer
from src.strategy.signal_engine import QRWSignalEngine


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _scalar_metrics(payload: dict) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in payload.items()
        if not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    }


def build_risk_artifacts(tick_df: pd.DataFrame, output: Path) -> dict:
    lattice = np.linspace(-1.0, 1.0, 101)
    probability = np.exp(-0.5 * (lattice / 0.18) ** 2)
    probability /= probability.sum()
    path_frames: list[pd.DataFrame] = []
    metrics: dict[str, dict] = {}
    for offset, scenario in enumerate(SCENARIOS):
        simulator = QRWRiskSimulator(n_paths=1000, seed=2026 + offset)
        result = simulator.scenario_var(probability, scenario)
        paths = result["paths"]
        n_paths, horizon = paths.shape
        path_frames.append(pd.DataFrame({
            "scenario": scenario,
            "path_id": np.repeat(np.arange(n_paths), horizon),
            "step": np.tile(np.arange(1, horizon + 1), n_paths),
            "cumulative_return": paths.reshape(-1),
        }))
        metrics[scenario] = {
            "label": result["scenario"],
            "var_95": result["var_95"],
            "var_99": result["var_99"],
            "es_95": result["es_95"],
            "severity": result["severity"],
        }
    pd.concat(path_frames, ignore_index=True).to_parquet(
        output / "risk_paths.parquet", index=False
    )
    backtest = QRWRiskSimulator(seed=2026).backtest_var(
        tick_df, window=min(100, max(20, len(tick_df) // 5))
    )
    backtest["details"].to_parquet(output / "risk_backtest.parquet", index=False)
    metrics["backtest"] = {
        key: value for key, value in backtest.items() if key != "details"
    }
    _write_json(output / "risk_metrics.json", metrics)
    return {"path_rows": int(sum(len(frame) for frame in path_frames)), **metrics["backtest"]}


def build_signal_artifacts(tick_df: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    engine = QRWSignalEngine()
    signal_log = engine.backtest_signals(tick_df)
    signal_log.to_parquet(output / "signal_log.parquet", index=False)
    metrics = engine.compute_signal_metrics(signal_log)
    _write_json(output / "signal_metrics.json", metrics)
    return signal_log, metrics


def build_optimizer_artifacts(tick_df: pd.DataFrame, output: Path) -> dict:
    split = int(len(tick_df) * 0.70)
    train = tick_df.iloc[:split].copy().reset_index(drop=True)
    test = tick_df.iloc[split:].copy().reset_index(drop=True)
    optimizer = QRWStrategyOptimizer()
    best = optimizer.grid_search(train, objective="sharpe")
    oos = optimizer.evaluate_out_of_sample(test, best_params=best)
    _write_json(output / "optimizer_params.json", best)
    optimizer.search_surface.to_parquet(output / "optimizer_surface.parquet", index=False)
    oos["backtest"].to_parquet(output / "optimizer_oos.parquet", index=False)
    metrics = _scalar_metrics(oos)
    _write_json(output / "optimizer_metrics.json", metrics)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "surface_rows": len(optimizer.search_surface),
        **metrics,
    }


def build_anomaly_artifacts(tick_df: pd.DataFrame, output: Path) -> dict:
    baseline_rows = min(max(250, len(tick_df) // 2), len(tick_df) - 50)
    detector = QRWAnomalyDetector(window=50).fit_baseline(
        tick_df.iloc[:baseline_rows].copy()
    )
    anomaly_log = detector.rolling_anomaly_scan(tick_df, step=5)
    anomaly_log.to_parquet(output / "anomaly_log.parquet", index=False)
    positions = np.arange(detector.n_bins) - detector.n_bins // 2
    pd.DataFrame({
        "position": positions,
        "baseline": detector.baseline,
        "current": detector.current_distribution,
    }).to_parquet(output / "anomaly_distributions.parquet", index=False)
    metrics = {
        "baseline_rows": baseline_rows,
        "scan_rows": int(len(anomaly_log)),
        "alerts": int(anomaly_log["alert"].sum()),
        "latest_sigma": float(anomaly_log["sigma_score"].iloc[-1]),
        "latest_type": str(anomaly_log["anom_type"].iloc[-1]),
    }
    _write_json(output / "anomaly_metrics.json", metrics)
    return metrics


def _module_quality_gate(name: str, metrics: dict) -> str | None:
    """Return a failure reason if ``metrics`` fails a real numeric sanity
    check for ``name``, or ``None`` if the module passes.

    Mirrors the removed_fraction-style threshold already used in
    scripts/pipelines/phase2_pipeline.py: a module is only PASS because its
    own reported numbers clear a real bar, not merely because its output
    file exists on disk (a module that runs to completion on garbage
    numbers used to report PASS unconditionally).
    """

    def _finite(value: object) -> bool:
        return isinstance(value, (int, float)) and np.isfinite(value)

    if name == "A1_volatility":
        if metrics.get("n_observations", 0) <= 0:
            return "no volatility observations"
        for key in ("current_qrw_vol", "current_garch_vol", "current_realized_vol"):
            if not _finite(metrics.get(key)) or metrics[key] < 0:
                return f"{key} missing, non-finite, or negative"
        return None
    if name == "A2_risk":
        if metrics.get("path_rows", 0) <= 0 or metrics.get("n_tests", 0) <= 0:
            return "no simulated risk paths or backtest observations"
        rate = metrics.get("violation_rate")
        if not _finite(rate) or not 0.0 <= rate <= 1.0:
            return "violation_rate missing or out of [0, 1]"
        return None
    if name == "A3_signal":
        for key in ("hit_rate", "profit_factor", "sharpe", "net_pnl"):
            if not _finite(metrics.get(key)):
                return f"{key} missing or non-finite"
        if metrics.get("n_trades", 0) <= 0:
            return "no non-HOLD trades generated"
        return None
    if name == "A4_optimizer":
        for key in ("sharpe", "hit_rate"):
            if not _finite(metrics.get(key)):
                return f"{key} missing or non-finite"
        if metrics.get("sharpe") == 0.0 and metrics.get("hit_rate") == 0.0:
            return "sharpe and hit_rate are both exactly zero"
        if metrics.get("surface_rows", 0) <= 0:
            return "empty optimizer search surface"
        return None
    if name == "A5_anomaly":
        if metrics.get("scan_rows", 0) <= 0:
            return "no anomaly scan rows"
        if not _finite(metrics.get("latest_sigma")):
            return "latest_sigma missing or non-finite"
        return None
    raise ValueError(f"unknown module: {name}")


def run_full_platform(date: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    print("[A1] Volatility Forecaster")
    volatility = run_volatility_demo(date, 15, output)
    tick_df = load_tick_data(date, limit=1000)

    print("[A2] Risk Simulator")
    risk = build_risk_artifacts(tick_df, output)
    print("[A3] Signal Engine")
    _, signal = build_signal_artifacts(tick_df, output)
    print("[A4] Strategy Optimizer")
    optimizer = build_optimizer_artifacts(tick_df, output)
    print("[A5] Anomaly Detector")
    anomaly = build_anomaly_artifacts(tick_df, output)

    required = [
        "vol_metrics.parquet",
        "risk_paths.parquet",
        "risk_backtest.parquet",
        "signal_log.parquet",
        "optimizer_params.json",
        "optimizer_surface.parquet",
        "optimizer_oos.parquet",
        "anomaly_log.parquet",
        "anomaly_distributions.parquet",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"pipeline did not create required artifacts: {missing}")

    module_metrics = {
        "A1_volatility": volatility,
        "A2_risk": risk,
        "A3_signal": signal,
        "A4_optimizer": optimizer,
        "A5_anomaly": anomaly,
    }
    failures = {
        name: reason
        for name, metrics in module_metrics.items()
        if (reason := _module_quality_gate(name, metrics)) is not None
    }

    def _status(name: str) -> str:
        return "FAIL" if name in failures else "PASS"

    manifest = {
        "track": "A",
        "phase": 10,
        "date": date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": int(len(tick_df)),
        "modules": {
            "A1_volatility": {
                "status": _status("A1_volatility"),
                "observations": volatility["n_observations"],
            },
            "A2_risk": {"status": _status("A2_risk"), **risk},
            "A3_signal": {"status": _status("A3_signal"), **signal},
            "A4_optimizer": {"status": _status("A4_optimizer"), **optimizer},
            "A5_anomaly": {"status": _status("A5_anomaly"), **anomaly},
        },
        "artifacts": required,
    }
    _write_json(output / "platform_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"quality gate failed for module(s): {failures}")
    print(f"[PASS] Track A Phase 10 artifacts written to {output}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-06-12")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "track_a")
    args = parser.parse_args()
    run_full_platform(args.date, args.output)


if __name__ == "__main__":
    main()
