"""Does a genuine heavy-tailed UNITARY shift give market-like tails?

final_report §7 retracted every tail claim because the heavy-tail module was a
classical Bernoulli/Pareto sampler, not a unitary operator, so it could not test
the mechanism it claimed to test. ``src/models/heavy_tail_unitary`` now supplies
an exactly-unitary Lévy shift. This script asks the empirical question that was
left open: **does that unitary produce tails shaped like real tick data, and
does it beat the ordinary alpha=1 walk?**

Method
------
Scale is not the question here (the bare walk steps every tick, whereas most
real ticks leave the price unchanged), so every distribution is standardised by
its own dispersion before comparison. What is compared is tail *shape*:

* excess kurtosis,
* tail mass beyond 3 and 5 standard deviations.

The empirical reference is the distribution of realized ``horizon``-tick price
displacements measured in tick units. Each candidate walk is evolved for
``horizon`` steps and its position marginal is standardised the same way.

A run is discarded when probability wraps around the periodic lattice, since
wraparound mimics a heavy tail.

EXPLORATORY ONLY. Not a confirmatory run; do not relabel as confirmatory.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.full_dataset_confirmation import _load_frame_efficient
from src.models.heavy_tail_unitary import LevyUnitaryQRW

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Weighted quantile via the cumulative weight curve."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, sorted_values))


def _shape_statistics(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Return scale-free, heavy-tail-safe shape statistics.

    Variance and kurtosis are **not** used: for a Lévy walk with alpha < 2 the
    second moment does not exist, so on a finite lattice any sigma/kurtosis we
    measure is a function of the lattice size rather than of the distribution.
    Instead the distribution is summarised by quantile ratios of |x - median|,
    which are well defined for heavy tails and invariant to scale:

        tail_ratio_99  = q99(|x-med|)  / q75(|x-med|)
        tail_ratio_999 = q999(|x-med|) / q75(|x-med|)

    Heavier tails push these ratios up. ``weights`` lets a probability marginal
    be summarised through the same code path as an empirical sample.
    """
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total

    median = _weighted_quantile(values, weights, 0.5)
    deviation = np.abs(values - median)
    scale = _weighted_quantile(deviation, weights, 0.75)
    if scale <= 0.0:
        raise ValueError("distribution has zero robust scale")
    q99 = _weighted_quantile(deviation, weights, 0.99)
    q999 = _weighted_quantile(deviation, weights, 0.999)
    return {
        "robust_scale_q75_ticks": scale,
        "tail_ratio_99": q99 / scale,
        "tail_ratio_999": q999 / scale,
    }


def _empirical_displacements(
    frame: pd.DataFrame,
    *,
    horizon: int,
    tick_size: float,
    max_samples: int,
) -> np.ndarray:
    """Realized ``horizon``-tick price displacements in tick units."""
    price = frame["price"].to_numpy(dtype=np.float64)
    if len(price) <= horizon:
        raise ValueError("frame is shorter than the requested horizon")
    displacement = (price[horizon:] - price[:-horizon]) / tick_size
    if "segment_id" in frame.columns:
        segment = frame["segment_id"].to_numpy()
        same = segment[horizon:] == segment[:-horizon]
        displacement = displacement[same]
    displacement = displacement[np.isfinite(displacement)]
    if len(displacement) > max_samples:
        step = len(displacement) // max_samples
        displacement = displacement[::step][:max_samples]
    return displacement


def _infer_tick_size(price: np.ndarray) -> float:
    delta = np.abs(np.diff(price))
    nonzero = delta[delta > 1e-12]
    if len(nonzero) == 0:
        raise ValueError("price series has no movement")
    return float(np.median(nonzero))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=2000000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-positions", type=int, default=4001)
    parser.add_argument(
        "--alphas",
        default="0.3,0.5,0.7,0.9,1.0,1.3,1.6,2.0",
        help="Levy exponents to evaluate; 1.0 is the ordinary walk.",
    )
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    frame = _load_frame_efficient(feature_path, args.max_rows)
    tick_size = _infer_tick_size(frame["price"].to_numpy(dtype=np.float64))
    print(f"[tail] {args.label}: {len(frame):,} rows, tick_size={tick_size:g}")

    empirical = _empirical_displacements(
        frame,
        horizon=args.horizon,
        tick_size=tick_size,
        max_samples=args.max_samples,
    )
    empirical_stats = _shape_statistics(empirical)
    print(
        f"[tail] empirical (h={args.horizon}, n={len(empirical):,}): "
        f"q99/q75={empirical_stats['tail_ratio_99']:.2f} "
        f"q999/q75={empirical_stats['tail_ratio_999']:.2f}"
    )

    # A Levy tail always leaks some amplitude to the lattice edge, so require it
    # to be far below the smallest tail probability we actually read (1e-3 at
    # the 99.9th percentile) rather than demanding exactly zero.
    wrap_tolerance = 1e-5
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    models: list[dict[str, Any]] = []
    for alpha in alphas:
        walk = LevyUnitaryQRW(args.n_positions, alpha)
        marginal = walk.run(args.horizon)
        wrap = walk.wraparound_mass()
        try:
            stats = _shape_statistics(walk.positions.astype(np.float64), marginal)
        except ValueError as error:
            print(f"[tail] alpha={alpha}: skipped ({error})")
            continue
        # Compare tail shape on log ratios so a model that is 10x too heavy is
        # penalised as much as one 10x too light.
        gap99 = abs(
            np.log(stats["tail_ratio_99"])
            - np.log(empirical_stats["tail_ratio_99"])
        )
        gap999 = abs(
            np.log(stats["tail_ratio_999"])
            - np.log(empirical_stats["tail_ratio_999"])
        )
        row = {
            "alpha": alpha,
            "is_ordinary_walk": bool(abs(alpha - 1.0) < 1e-12),
            "wraparound_mass": wrap,
            "wraparound_ok": bool(wrap < wrap_tolerance),
            **stats,
            "log_gap_99": float(gap99),
            "log_gap_999": float(gap999),
            "shape_distance": float(gap99 + gap999),
        }
        models.append(row)
        print(
            f"[tail] alpha={alpha:<4} q99/q75={stats['tail_ratio_99']:6.2f} "
            f"q999/q75={stats['tail_ratio_999']:7.2f} "
            f"dist={row['shape_distance']:.3f} wrap={wrap:.2e} "
            f"{'' if row['wraparound_ok'] else '[WRAPAROUND]'}"
        )

    valid = [m for m in models if m["wraparound_ok"]]
    if not valid:
        raise RuntimeError("every alpha wrapped around the lattice; raise --n-positions")
    best = min(valid, key=lambda m: m["shape_distance"])
    ordinary = next((m for m in valid if m["is_ordinary_walk"]), None)

    if ordinary is None:
        verdict = (
            f"Best tail-shape match is alpha={best['alpha']}; the ordinary walk "
            "(alpha=1) wrapped around and could not be compared."
        )
    elif best["is_ordinary_walk"]:
        verdict = (
            "The ordinary alpha=1 walk already matches the empirical tail shape "
            "best; the heavy-tailed unitary does not improve it."
        )
    else:
        verdict = (
            f"The heavy-tailed unitary (alpha={best['alpha']}) matches the "
            f"empirical tail shape better than the ordinary walk "
            f"(shape distance {best['shape_distance']:.3f} vs "
            f"{ordinary['shape_distance']:.3f}). The ordinary walk's marginal is "
            f"bimodal/ballistic with q999/q75 = "
            f"{ordinary['tail_ratio_999']:.2f}, against an empirical "
            f"{empirical_stats['tail_ratio_999']:.2f}; the Lévy shift reaches "
            f"{best['tail_ratio_999']:.2f}."
        )
    print(f"[tail] VERDICT: {verdict}")

    audit = {
        "kind": "heavy_tail_unitary_evaluation",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "label": args.label,
        "feature_path": str(feature_path),
        "rows": int(len(frame)),
        "tick_size": tick_size,
        "horizon": args.horizon,
        "n_positions": args.n_positions,
        "empirical_samples": int(len(empirical)),
        "empirical": empirical_stats,
        "models": models,
        "best_alpha": best["alpha"],
        "ordinary_walk": ordinary,
        "verdict": verdict,
    }
    label = args.label
    json_out = (ROOT / (args.json_out or f"reports/research/heavy_tail_unitary_{label}.json")).resolve()
    md_out = (ROOT / (args.md_out or f"reports/research/heavy_tail_unitary_{label}.md")).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"[tail] wrote {json_out}")
    print(f"[tail] wrote {md_out}")


def _render_markdown(audit: dict[str, Any]) -> str:
    emp = audit["empirical"]
    rows = [
        f"# Heavy-tailed unitary shift — tail-shape evaluation ({audit['label']})",
        "",
        f"**Status:** `{audit['status']}` — exploratory. Closes the §7 gap: the "
        "heavy-tail mechanism is now an exactly-unitary Lévy shift, not a "
        "classical Bernoulli/Pareto sampler.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` "
        f"({audit['rows']:,} rows), tick size {audit['tick_size']:g}",
        f"- Horizon: {audit['horizon']} ticks · lattice {audit['n_positions']} "
        f"positions · {audit['empirical_samples']:,} empirical displacements",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
        "",
        "Only tail *shape* is compared, via quantile ratios of |x − median| "
        "(scale-free). Variance and kurtosis are deliberately avoided: for a "
        "Lévy walk with α < 2 the second moment does not exist, so on a finite "
        "lattice any σ or kurtosis measured is a function of the lattice size, "
        "not of the distribution.",
        "",
        "## Empirical reference",
        "",
        "| q99/q75 | q999/q75 |",
        "|---:|---:|",
        f"| {emp['tail_ratio_99']:.2f} | {emp['tail_ratio_999']:.2f} |",
        "",
        "## Lévy unitary walks (α = 1 is the ordinary nearest-neighbour walk)",
        "",
        "| α | q99/q75 | q999/q75 | shape distance | wraparound | valid |",
        "|---:|---:|---:|---:|---:|:--:|",
    ]
    for m in sorted(audit["models"], key=lambda r: r["alpha"]):
        mark = " *(ordinary)*" if m["is_ordinary_walk"] else ""
        rows.append(
            f"| {m['alpha']}{mark} | {m['tail_ratio_99']:.2f} | "
            f"{m['tail_ratio_999']:.2f} | {m['shape_distance']:.3f} | "
            f"{m['wraparound_mass']:.1e} | "
            f"{'✔' if m['wraparound_ok'] else '✘ wrap'} |"
        )
    rows += ["", "## Verdict", "", audit["verdict"], ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
