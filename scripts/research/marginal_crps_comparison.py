"""Does the QRW density-matrix marginal beat classical distributional models?

Phases 1-4 covered the DIRECTIONAL endpoint (Brier / accuracy). The
pre-registered PRIMARY endpoint is different: mean fixed-origin marginal CRPS
(``docs/data_collection_todo.md``). This script closes that gap using the
project's own ``BenchmarkSuite`` (protocol v4), which evolves the QRW density
matrix into fixed-origin position marginals and scores them per horizon with
CRPS against the realized holdout path, alongside CRW variants, GARCH(1,1) and
GBM.

A single forecast origin is fragile (Phase 2 taught us not to trust a single
split), so this runs several non-overlapping windows, each with its own
chronological train/holdout, and aggregates the per-model CRPS across windows.

If GARCH/GBM beat the QRW marginal on CRPS as consistently as the strong
directional baselines beat it on Brier (§5c), the QRW offers no advantage on the
registered primary endpoint either.

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

from scripts.research.full_dataset_confirmation import _load_frame_efficient
from scipy.stats import spearmanr

from src.data.common import timestamps_to_nanoseconds

from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.evaluation.benchmark_suite import BenchmarkSuite

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "mean_marginal_crps"
TIE_BREAK = "direction_log_loss"
SECONDARY = "mean_marginal_absolute_error"
METRICS = (PRIMARY, SECONDARY, TIE_BREAK)


QRW_MODEL = "QRW Adaptive"


def window_volatility(window) -> float:
    """Realised volatility of the window: std of tick-to-tick log returns.

    Per-tick and scale-free, so windows of different length and different
    assets are comparable. Segment breaks are excluded -- the jump across a
    data gap is not a price move and would dominate the estimate.
    """
    price = window["price"].to_numpy(dtype=np.float64)
    if len(price) < 3:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(price))
    usable = np.isfinite(returns)
    if "segment_id" in window.columns:
        segment = window["segment_id"].to_numpy()
        usable &= segment[1:] == segment[:-1]
    if usable.sum() < 2:
        return float("nan")
    return float(np.std(returns[usable], ddof=1))


def volatility_relationship(per_window: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the QRW fall further behind as the window gets more volatile?

    The report has carried this as an interpretation -- "QRW wins on quiet
    windows and loses badly on volatile ones, so it does not model volatility
    dynamics" -- read off five windows per asset, with an acknowledged
    counter-example. Five points cannot support it either way, so it is
    measured here instead: Spearman between realised volatility and how far
    the QRW sits behind the best alternative model in that window.

    Rank correlation rather than Pearson: the relationship need not be linear
    and a single turbulent window should not set the answer.
    """
    pairs = [
        (row["realised_volatility"], row["qrw_crps_gap"])
        for row in per_window
        if np.isfinite(row.get("realised_volatility", np.nan))
        and np.isfinite(row.get("qrw_crps_gap", np.nan))
    ]
    if len(pairs) < 5:
        return {
            "windows_used": len(pairs),
            "spearman": None,
            "p_value": None,
            "supports_claim": None,
            "note": "fewer than 5 usable windows; no correlation reported",
        }

    volatility = np.array([p[0] for p in pairs])
    gap = np.array([p[1] for p in pairs])
    result = spearmanr(volatility, gap)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    return {
        "windows_used": len(pairs),
        "spearman": rho,
        "p_value": p_value,
        # The claim is directional: more volatility => QRW further behind.
        "supports_claim": bool(rho > 0 and p_value < 0.05),
        "median_volatility": float(np.median(volatility)),
        "median_gap": float(np.median(gap)),
    }


def day_cluster_boundaries(frame: pd.DataFrame, windows: int) -> np.ndarray:
    """Row boundaries that fall only on UTC-day edges.

    The pre-registration makes the complete UTC day the unit of analysis.
    Splitting on row count instead can put every window inside a single day --
    which is what happened to ETHUSDT -- so a "window" then measures a few
    hours of one session rather than a stretch of market. Days are grouped as
    evenly as the count allows and the boundaries are the first row of each
    group's first day, so no window holds a partial day.

    Raises when there are fewer days than requested windows: silently
    returning fewer would restate the same thinness the day unit exists to fix.
    """
    nanoseconds = timestamps_to_nanoseconds(frame["timestamp"]).to_numpy(dtype="int64")
    day_index = nanoseconds // 86_400_000_000_000
    # First row of each distinct day; the frame is time-ordered by the loader.
    starts = np.flatnonzero(np.r_[True, day_index[1:] != day_index[:-1]])
    if len(starts) < windows:
        raise ValueError(
            f"{len(starts)} UTC day(s) in the frame cannot fill {windows} "
            f"day-cluster windows; widen --max-rows or lower --windows"
        )
    edges = np.linspace(0, len(starts), windows + 1).astype(int)
    return np.r_[starts[edges[:-1]], len(frame)]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _run_window(
    window: pd.DataFrame,
    *,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return {model: {metric: value}} for one chronological window."""
    suite = BenchmarkSuite(
        window,
        train_fraction=0.6,
        n_steps=n_steps,
        n_paths=n_paths,
        random_seed=seed,
    )
    results = suite.run()
    wanted = results[results["metric"].isin(METRICS)]
    out: dict[str, dict[str, float]] = {}
    for _, row in wanted.iterrows():
        out.setdefault(str(row["model"]), {})[str(row["metric"])] = float(
            row["value"]
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=4000000)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument(
        "--window-unit",
        choices=("rows", "utc-day"),
        default="rows",
        help="utc-day keeps every window a whole number of UTC days, "
             "which is the pre-registered unit of analysis.",
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    frame = _load_frame_efficient(feature_path, args.max_rows)
    print(f"[crps] {args.label}: {len(frame):,} rows from {feature_path.name}")

    if args.window_unit == "utc-day":
        boundaries = day_cluster_boundaries(frame, args.windows)
    else:
        boundaries = np.linspace(0, len(frame), args.windows + 1, dtype=int)
    per_window: list[dict[str, Any]] = []
    models: set[str] = set()
    for w in range(args.windows):
        window = frame.iloc[boundaries[w] : boundaries[w + 1]].copy()
        print(f"[crps] window {w + 1}/{args.windows} ({len(window):,} rows)...")
        try:
            scores = _run_window(
                window, n_steps=args.n_steps, n_paths=args.n_paths,
                seed=args.seed + w,
            )
        except (ValueError, RuntimeError) as error:
            print(f"[crps]   window {w} skipped: {error}")
            continue
        models.update(scores)
        best = min(scores, key=lambda m: scores[m].get(PRIMARY, np.inf))
        # How far the QRW sits behind the best *alternative*, so a window the
        # QRW wins scores negative rather than zero.
        rivals = [
            scores[m].get(PRIMARY, np.inf) for m in scores if m != QRW_MODEL
        ]
        qrw_crps = scores.get(QRW_MODEL, {}).get(PRIMARY, np.nan)
        gap = float(qrw_crps - min(rivals)) if rivals else float("nan")
        per_window.append(
            {
                "window": w,
                "scores": scores,
                "best_crps_model": best,
                "realised_volatility": window_volatility(window),
                "qrw_crps_gap": gap,
            }
        )
        print(
            f"[crps]   best CRPS: {best} "
            f"({scores[best][PRIMARY]:.4f}); "
            f"QRW Adaptive={scores.get('QRW Adaptive', {}).get(PRIMARY, float('nan')):.4f}"
        )
        gc.collect()

    if not per_window:
        raise RuntimeError("no window produced a valid benchmark")

    # Aggregate the primary endpoint across windows.
    model_list = sorted(models)
    aggregate: dict[str, dict[str, Any]] = {}
    for model in model_list:
        crps_values = [
            row["scores"][model][PRIMARY]
            for row in per_window
            if model in row["scores"] and PRIMARY in row["scores"][model]
        ]
        if not crps_values:
            continue
        aggregate[model] = {
            "mean_crps": float(np.mean(crps_values)),
            "crps_per_window": [round(v, 6) for v in crps_values],
            "windows_scored": len(crps_values),
            "windows_best": sum(
                1 for row in per_window if row["best_crps_model"] == model
            ),
        }

    ranked = sorted(aggregate, key=lambda m: aggregate[m]["mean_crps"])
    qrw = "QRW Adaptive"
    best_overall = ranked[0]
    qrw_best_count = aggregate.get(qrw, {}).get("windows_best", 0)
    qrw_rank = ranked.index(qrw) + 1 if qrw in ranked else None

    if qrw_best_count == 0:
        verdict = (
            f"On the registered primary endpoint (mean marginal CRPS), the QRW "
            f"density-matrix marginal is NOT the best model in any of "
            f"{len(per_window)} windows; '{best_overall}' wins overall. The QRW "
            f"offers no advantage on the distributional endpoint."
        )
    elif qrw_best_count == len(per_window):
        verdict = (
            "The QRW marginal has the lowest mean marginal CRPS in every window."
        )
    else:
        verdict = (
            f"The QRW marginal wins the CRPS endpoint in {qrw_best_count}/"
            f"{len(per_window)} windows; overall best is '{best_overall}'."
        )
    volatility_test = volatility_relationship(per_window)
    if volatility_test["spearman"] is None:
        verdict += (
            f" Volatility relationship not tested: only "
            f"{volatility_test['windows_used']} usable windows."
        )
    elif volatility_test["supports_claim"]:
        verdict += (
            f" The QRW does fall further behind as realised volatility rises "
            f"(Spearman {volatility_test['spearman']:+.2f}, p="
            f"{volatility_test['p_value']:.3f}, {volatility_test['windows_used']} "
            f"windows) -- consistent with it not modelling volatility dynamics."
        )
    else:
        verdict += (
            f" The volatility story is NOT supported at this sample size: "
            f"Spearman between realised volatility and the QRW's CRPS gap is "
            f"{volatility_test['spearman']:+.2f} (p={volatility_test['p_value']:.3f}, "
            f"{volatility_test['windows_used']} windows)."
        )
    print(f"[crps] VERDICT: {verdict}")

    audit = {
        "kind": "marginal_crps_comparison",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "protocol_version": BenchmarkSuite.PROTOCOL_VERSION,
        "label": args.label,
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "windows": len(per_window),
        "window_unit": args.window_unit,
        "n_steps": args.n_steps,
        "n_paths": args.n_paths,
        "seed": args.seed,
        "primary_endpoint": PRIMARY,
        "aggregate": aggregate,
        "ranked_by_mean_crps": ranked,
        "best_overall": best_overall,
        "qrw_rank": qrw_rank,
        "qrw_windows_best": qrw_best_count,
        "per_window": per_window,
        "volatility_relationship": volatility_test,
        "verdict": verdict,
    }
    label = args.label
    json_out = (ROOT / (args.json_out or f"reports/research/marginal_crps_{label}.json")).resolve()
    md_out = (ROOT / (args.md_out or f"reports/research/marginal_crps_{label}.md")).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"[crps] wrote {json_out}")
    print(f"[crps] wrote {md_out}")


def _render_markdown(audit: dict[str, Any]) -> str:
    agg = audit["aggregate"]
    ranked = audit["ranked_by_mean_crps"]
    rows = [
        f"# Marginal-CRPS comparison — {audit['label']}",
        "",
        f"**Status:** `{audit['status']}` — exploratory. Registered PRIMARY "
        "endpoint: mean fixed-origin marginal CRPS.",
        "",
        f"- Protocol: `{audit['protocol_version']}`",
        f"- Feature file: `{Path(audit['feature_path']).name}` "
        f"({audit['rows']:,} rows)",
        f"- Windows: {audit['windows']} non-overlapping chronological splits · "
        f"n_steps={audit['n_steps']} · n_paths={audit['n_paths']}",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
        "",
        "## Mean marginal CRPS across windows (lower = better)",
        "",
        "| Rank | Model | mean CRPS | windows best | per-window CRPS |",
        "|---:|---|---:|---:|---|",
    ]
    for i, model in enumerate(ranked, 1):
        a = agg[model]
        mark = " **(QRW)**" if model == "QRW Adaptive" else ""
        per = ", ".join(f"{v:.3f}" for v in a["crps_per_window"])
        rows.append(
            f"| {i} | {model}{mark} | {a['mean_crps']:.4f} | "
            f"{a['windows_best']}/{audit['windows']} | {per} |"
        )
    rows += ["", "## Verdict", "", audit["verdict"], ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
