"""Full-dataset directional confirmation (closes report limitation #6).

The report notes that after the Phase 2 bias fix the walk-forward could not be
re-run on the full ~32.4M-tick BTC dataset that produced the stale pre-fix
+0.049889 edge, because ``rolling_stability``/``walk_forward_evaluation`` copy
the frame repeatedly and exceed available RAM. This script provides a
memory-efficient path: it loads ONLY the columns the QRW/affine comparison needs
and downcasts them to the smallest safe dtype, then reuses the exact post-fix
``run_config``/``run_affine`` walk-forward from the alpha_phase ablation to
report the current QRW-vs-affine directional edge on the full dataset.

It is deliberately scoped to the directional endpoint (Brier), which is where
the phase mechanism and the windowing edge live; the strong-baseline refutation
(§5c) already shows the picture does not change with a competitive baseline.

EXPLORATORY ONLY. Not a confirmatory run; do not relabel as confirmatory.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.audits.phase3_overfitting_audit import (
    fit_model,
    moving_block_bootstrap_mean,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file
from scripts.research.alpha_phase_ablation import paired_edge, run_affine, run_config

ROOT = Path(__file__).resolve().parents[2]

# Only these columns are needed by MarketQRW.calibrate() and market_events();
# everything else in the 17-column feature frame is dropped before load.
NEEDED_COLUMNS = [
    "timestamp",
    "price",
    "tick_direction",
    "obi",
    "trade_intensity",
    "segment_id",
    "obi_valid",
]
# price stays float64 (tick-level diffs need the precision); the rest downcast.
DOWNCAST = {
    "obi": np.float32,
    "trade_intensity": np.float32,
    "tick_direction": np.float32,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _load_frame_efficient(path: Path, max_rows: int) -> pd.DataFrame:
    """Load only the needed columns, downcast, and sort by timestamp."""
    schema_names = set(pq.ParquetFile(path).schema.names)
    columns = [c for c in NEEDED_COLUMNS if c in schema_names]
    frame = pd.read_parquet(path, columns=columns)
    if max_rows and len(frame) > max_rows:
        frame = frame.iloc[:max_rows]
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for column, dtype in DOWNCAST.items():
        if column in frame.columns:
            frame[column] = frame[column].astype(dtype)
    if "obi_valid" in frame.columns:
        frame["obi_valid"] = frame["obi_valid"].astype(bool)
    if "segment_id" in frame.columns and frame["segment_id"].dtype != np.int32:
        # Keep segment ids compact but lossless where possible.
        try:
            frame["segment_id"] = frame["segment_id"].astype(np.int32)
        except (ValueError, OverflowError, TypeError):
            pass
    gc.collect()
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_multiday.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT_full")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = full dataset. Set a cap if RAM is tight.",
    )
    parser.add_argument(
        "--folds",
        default="2,3,5",
        help="Comma-separated fold counts for the A_full-vs-affine edge.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json-out", default="reports/research/full_dataset_confirmation.json")
    parser.add_argument("--md-out", default="reports/research/full_dataset_confirmation.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    frame = _load_frame_efficient(feature_path, args.max_rows)
    mem_mb = frame.memory_usage(deep=True).sum() / 1e6
    print(
        f"[full] loaded {len(frame):,} rows ({mem_mb:.0f} MB in memory) "
        f"from {feature_path.name}"
    )

    warmup_end = int(len(frame) * 0.4)
    print("[full] calibrating structural model on warmup (40%)...")
    _, structural = fit_model(frame.iloc[:warmup_end].copy())
    tick_size = float(structural["tick_size"])
    print(
        f"[full]   quantum_improved={structural['quantum_improved']} "
        f"alpha_phase={structural['alpha_phase']:.3e} gamma={structural['gamma']:.4f}"
    )
    gc.collect()

    fold_grid = [int(v) for v in args.folds.split(",") if v.strip()]
    seed_seq = np.random.SeedSequence(args.seed)
    children = dict(zip(fold_grid, seed_seq.spawn(len(fold_grid))))
    results: list[dict[str, Any]] = []
    for k in fold_grid:
        print(f"[full] walk-forward A_full vs affine at folds={k}...")
        a_k = run_config(frame, structural=structural, folds=k, tick_size=tick_size)
        c_k = run_affine(frame, folds=k)
        edge = paired_edge(
            a_k, c_k,
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(children[k]),
            block_size=args.block_size,
        )
        lo, hi = edge["confidence_interval_95"]
        row = {
            "folds": k,
            "a_full_brier": a_k["pooled"]["brier"],
            "affine_brier": c_k["pooled"]["brier"],
            "edge_qrw_minus_affine": edge["model_minus_comparison"],
            "confidence_interval_95": [lo, hi],
            "qrw_wins": bool(edge["significant"] and edge["direction"] == "first_better"),
            "events": a_k["pooled"]["events"],
        }
        results.append(row)
        print(
            f"[full]   A_full={row['a_full_brier']:.6f} affine={row['affine_brier']:.6f} "
            f"edge={row['edge_qrw_minus_affine']:+.6f} "
            f"({'QRW wins' if row['qrw_wins'] else 'QRW does not win'})"
        )
        del a_k, c_k
        gc.collect()

    audit = {
        "kind": "full_dataset_directional_confirmation",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "label": args.label,
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "in_memory_mb": round(mem_mb, 1),
        "quantum_improved": bool(structural["quantum_improved"]),
        "alpha_phase": float(structural["alpha_phase"]),
        "gamma": float(structural["gamma"]),
        "tick_size": tick_size,
        "stale_prefix_edge_reference": 0.049889,
        "fold_results": results,
    }
    json_out = (ROOT / args.json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out = (ROOT / args.md_out).resolve()
    md_out.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"[full] wrote {json_out}")
    print(f"[full] wrote {md_out}")


def _render_markdown(audit: dict[str, Any]) -> str:
    rows = [
        f"# Full-dataset directional confirmation — {audit['label']}",
        "",
        f"**Status:** `{audit['status']}` — exploratory, closes report limitation #6 "
        "for the directional endpoint.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` "
        f"({audit['rows']:,} rows, {audit['in_memory_mb']:.0f} MB in memory via "
        "column-subset + float32 downcast)",
        f"- quantum_improved={audit['quantum_improved']} · "
        f"alpha_phase={audit['alpha_phase']:.3e} · gamma={audit['gamma']:.4f}",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
        "",
        "## Post-fix QRW vs affine on the full dataset",
        "",
        "Negative edge = QRW has the lower (better) Brier. Compare against the "
        f"stale pre-fix figure **+{audit['stale_prefix_edge_reference']}** "
        "(reported before the Phase 2 bias fix, never reproducible post-fix).",
        "",
        "| folds | events | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |",
        "|---:|---:|---:|---:|---:|---|:--:|",
    ]
    for r in audit["fold_results"]:
        lo, hi = r["confidence_interval_95"]
        rows.append(
            f"| {r['folds']} | {r['events']:,} | {r['a_full_brier']:.6f} | "
            f"{r['affine_brier']:.6f} | {r['edge_qrw_minus_affine']:+.6f} | "
            f"[{lo:+.6f}, {hi:+.6f}] | {'✔' if r['qrw_wins'] else '✘'} |"
        )
    rows += ["", "## Note", "", _note(audit), ""]
    return "\n".join(rows)


def _note(audit: dict[str, Any]) -> str:
    wins = [r for r in audit["fold_results"] if r["qrw_wins"]]
    if wins and len(wins) == len(audit["fold_results"]):
        stability = "beats the affine baseline stably across every fold count tested"
    elif wins:
        stability = "beats the affine baseline at some fold counts"
    else:
        stability = "does not beat the affine baseline"
    return (
        f"On the full dataset the post-fix windowed-QRW {stability}. This only "
        "restates the §5b affine comparison at full scale; §5c already shows the "
        "windowed-QRW loses to competitive classical baselines (OrderFlow AR(5), "
        "Logistic+Pairwise) regardless, so this does not change the overall "
        "verdict. The old +0.049889 was an artifact of the pre-Phase-2 bias bug."
    )


if __name__ == "__main__":
    main()
