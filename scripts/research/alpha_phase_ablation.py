"""Alpha-phase ablation: isolate the quantum-interference contribution.

The project's identity claim is that a *quantum* random walk beats classical
baselines. The only place the "quantum interference" mechanism enters the
directional prediction is the phase angle in
``MarketQRW._quantum_windowed_probabilities``::

    phi = (alpha_phase * dir) / window

When ``alpha_phase == 0`` every SU(2) coin collapses to a real SO(2) rotation.
Real rotations about the same axis commute, so the accumulated product no longer
depends on the *order* of recent events -- exactly the interference effect the
model is supposed to exploit disappears. Toggling ``alpha_phase`` is therefore a
theoretically exact isolation of the quantum mechanism.

This script runs an expanding-window walk-forward (mirroring
``phase3_overfitting_audit``) for four configurations that differ *only* in the
phase / dispatch, holding the warmup structural fit otherwise fixed:

    A_full       quantum_improved as selected, alpha_phase free (the reported model)
    B_refit      alpha_phase pinned to 0 and every other structural param refit
    B_posthoc    the A_full structural fit with alpha_phase zeroed after the fact
    C_classical  the closed-form classical directional formula (no windowing)

Because the evaluation event selection depends only on the data (not the model),
the pooled per-event targets are identical across configs, so Brier differences
are computed as genuine *paired* differences and bootstrapped with a moving-block
scheme that preserves tick autocorrelation. The mechanism decomposes cleanly:

    A vs C  = total quantum contribution (windowing + phase)
    B vs C  = windowing / decoherence contribution, phase removed
    A vs B  = pure phase (interference) contribution

An optional phase sweep additionally scores a grid of forced ``alpha_phase``
values to reveal whether *any* phase level improves the holdout Brier on this
data, independent of what the optimizer happened to select.

EXPLORATORY ONLY. This is a mechanism diagnostic, not a confirmatory run: it is
not gated by the frozen pre-registration protocol and its output must never be
relabeled as confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audits.phase3_overfitting_audit import (
    fit_fixed_structure_model,
    fit_linear_market_probability,
    fit_linear_probability,
    fit_model,
    market_events,
    moving_block_bootstrap_mean,
    moving_events,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _pooled_score(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "events": int(len(target)),
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            -np.mean(
                target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)
            )
        ),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
        "mean_probability": float(np.mean(probability)),
    }


def run_config(
    frame: pd.DataFrame,
    *,
    structural: dict[str, Any],
    folds: int,
    tick_size: float,
) -> dict[str, Any]:
    """Walk-forward a single structural config; return pooled per-event arrays.

    The fold boundaries and the ``market_events`` selection depend only on the
    frame, so the returned ``target`` array aligns element-for-element across
    every config evaluated on the same frame -- which is what makes the
    downstream Brier differences properly paired.
    """
    if folds < 2:
        raise ValueError("walk-forward folds must be at least 2")
    first_train_end = int(len(frame) * 0.4)
    boundaries = np.linspace(first_train_end, len(frame), folds + 1, dtype=int)
    warmup = frame.iloc[:first_train_end]

    prior_bias = float(structural["obi_bias"])
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []
    for fold in range(folds):
        train_end = int(boundaries[fold])
        evaluation_end = int(boundaries[fold + 1])
        bias_history = frame.iloc[first_train_end:train_end].copy()
        update_bias = not bias_history.empty
        if not update_bias:
            bias_history = warmup.iloc[-2:].copy()
        evaluation = frame.iloc[train_end:evaluation_end].copy()
        model, bias_update = fit_fixed_structure_model(
            bias_history,
            structural=structural,
            tick_size=tick_size,
            prior_bias=prior_bias,
            update_bias=update_bias,
        )
        prior_bias = float(bias_update["obi_bias"])
        obi, direction, target = market_events(evaluation)
        probability = model.predict_right_probabilities(obi, direction)
        probabilities.append(probability)
        targets.append(target)
        per_fold.append(
            {
                "fold": fold,
                "evaluation_rows": int(len(evaluation)),
                "moving_events": int(len(target)),
                "obi_bias": prior_bias,
                "brier": float(np.mean((probability - target) ** 2)),
            }
        )

    pooled_probability = np.concatenate(probabilities)
    pooled_target = np.concatenate(targets)
    return {
        "quantum_improved": bool(structural.get("quantum_improved", False)),
        "alpha_phase": float(structural.get("alpha_phase", 0.0)),
        "gamma": float(structural["gamma"]),
        "alpha_obi": float(structural.get("alpha_obi", 0.0)),
        "alpha_direction": float(structural.get("alpha_direction", 0.0)),
        "pooled": _pooled_score(pooled_probability, pooled_target),
        "per_fold": per_fold,
        "_probability": pooled_probability,
        "_target": pooled_target,
    }


def run_affine(
    frame: pd.DataFrame,
    *,
    folds: int,
) -> dict[str, Any]:
    """Walk-forward the independently-fit affine (OBI + direction) baseline.

    This is the honest classical reference the reported verdict compares
    against: an ordinary least-squares ``P(up) = c0 + c1*OBI + c2*direction``
    refit on each expanding training window. It shares the exact fold
    boundaries and event selection with :func:`run_config`, so its pooled
    target aligns element-for-element for a proper paired comparison. (Feeding
    the quantum-fit structural parameters into the classical closed form is
    *not* a valid baseline -- those parameters were optimised under the
    windowed formula and invert the sign, which is audit finding C2.)
    """
    first_train_end = int(len(frame) * 0.4)
    boundaries = np.linspace(first_train_end, len(frame), folds + 1, dtype=int)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []
    for fold in range(folds):
        train_end = int(boundaries[fold])
        evaluation_end = int(boundaries[fold + 1])
        train = frame.iloc[:train_end].copy()
        evaluation = frame.iloc[train_end:evaluation_end].copy()
        coefficients = fit_linear_market_probability(train)
        obi, direction, target = market_events(evaluation)
        probability = np.clip(
            coefficients[0] + coefficients[1] * obi + coefficients[2] * direction,
            0.0,
            1.0,
        )
        probabilities.append(probability)
        targets.append(target)
        per_fold.append(
            {
                "fold": fold,
                "moving_events": int(len(target)),
                "brier": float(np.mean((probability - target) ** 2)),
            }
        )
    pooled_probability = np.concatenate(probabilities)
    pooled_target = np.concatenate(targets)
    return {
        "quantum_improved": False,
        "alpha_phase": float("nan"),
        "gamma": float("nan"),
        "alpha_obi": float("nan"),
        "alpha_direction": float("nan"),
        "pooled": _pooled_score(pooled_probability, pooled_target),
        "per_fold": per_fold,
        "_probability": pooled_probability,
        "_target": pooled_target,
    }


def paired_edge(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    *,
    samples: int,
    rng: np.random.Generator,
    block_size: int,
) -> dict[str, Any]:
    """Block-bootstrap CI of the paired Brier difference ``a - b``.

    A negative mean means ``a`` (listed first) has the lower Brier, i.e. is the
    better forecaster.
    """
    target = config_a["_target"]
    if not np.array_equal(target, config_b["_target"]):
        raise ValueError("configs were not evaluated on identical events")
    diff = (config_a["_probability"] - target) ** 2 - (
        config_b["_probability"] - target
    ) ** 2
    edge = moving_block_bootstrap_mean(
        diff, samples=samples, rng=rng, block_size=block_size
    )
    lower, upper = edge["confidence_interval_95"]
    edge["significant"] = bool(upper < 0.0 or lower > 0.0)
    edge["direction"] = (
        "first_better" if edge["model_minus_comparison"] < 0.0 else "second_better"
    )
    return edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
        help="Parquet feature file (defaults to the reported positive-result subset).",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on rows (0 = use all). Keeps memory bounded on large files.",
    )
    parser.add_argument(
        "--phase-sweep",
        default="0.0,0.05,0.1,0.25,0.5,1.0,2.0",
        help="Comma-separated forced alpha_phase grid; empty to skip.",
    )
    parser.add_argument(
        "--fold-sensitivity",
        default="2,3,4,5,6,8",
        help="Comma-separated walk-forward fold counts to test A_full vs affine "
        "robustness; empty to skip.",
    )
    parser.add_argument("--json-out", default="reports/research/alpha_phase_ablation.json")
    parser.add_argument("--md-out", default="reports/research/alpha_phase_ablation.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    frame = (
        pd.read_parquet(feature_path)
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.iloc[: args.max_rows].reset_index(drop=True)
    print(f"[ablation] loaded {len(frame):,} rows from {feature_path.name}")

    warmup_end = int(len(frame) * 0.4)
    warmup = frame.iloc[:warmup_end].copy()

    # A: free alpha_phase (the reported model). B_refit: phase pinned to 0 and
    # every other structural parameter refit under that constraint.
    print("[ablation] calibrating A_full (alpha_phase free)...")
    _, structural_free = fit_model(warmup)
    tick_size = float(structural_free["tick_size"])
    print(
        f"[ablation]   quantum_improved={structural_free['quantum_improved']} "
        f"alpha_phase={structural_free['alpha_phase']:.3e} "
        f"gamma={structural_free['gamma']:.4f}"
    )

    print("[ablation] calibrating B_refit (alpha_phase pinned to 0)...")
    # fit_model() does not expose config overrides, so drive calibrate() directly
    # with the freeze flag for the frozen structural fit.
    from src.models.qrw_market_sim import MarketQRW  # local import: heavy module

    frozen_model = MarketQRW(
        frame.iloc[:warmup_end].copy(),
        {
            "n_positions": 101,
            "gamma_base": 0.0,
            "alpha_obi": 0.0,
            "coin_type": "obi_adaptive",
            "tick_size": tick_size,
            "freeze_alpha_phase": True,
        },
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        structural_frozen = frozen_model.calibrate(Path(directory) / "params.json")
    print(
        f"[ablation]   quantum_improved={structural_frozen['quantum_improved']} "
        f"alpha_phase={structural_frozen['alpha_phase']:.3e} "
        f"gamma={structural_frozen['gamma']:.4f}"
    )

    # B_posthoc shares A's warmup fit with the phase zeroed after the fact,
    # differing from A only at prediction time -- a dispatch sanity check that
    # post-hoc zeroing matches an explicit refit when the fitted phase is tiny.
    structural_posthoc = deepcopy(structural_free)
    structural_posthoc["alpha_phase"] = 0.0

    configs = {
        "A_full": structural_free,
        "B_refit": structural_frozen,
        "B_posthoc": structural_posthoc,
    }
    results: dict[str, Any] = {}
    for name, structural in configs.items():
        print(f"[ablation] walk-forward {name}...")
        results[name] = run_config(
            frame, structural=structural, folds=args.folds, tick_size=tick_size
        )
        print(
            f"[ablation]   pooled Brier={results[name]['pooled']['brier']:.6f} "
            f"log_loss={results[name]['pooled']['log_loss']:.6f} "
            f"acc={results[name]['pooled']['accuracy']:.4f}"
        )

    # The valid classical reference: independently-fit affine baseline.
    print("[ablation] walk-forward C_affine (independent OBI+direction OLS)...")
    results["C_affine"] = run_affine(frame, folds=args.folds)
    print(
        f"[ablation]   pooled Brier={results['C_affine']['pooled']['brier']:.6f} "
        f"log_loss={results['C_affine']['pooled']['log_loss']:.6f} "
        f"acc={results['C_affine']['pooled']['accuracy']:.4f}"
    )

    # Mechanism decomposition (paired, block-bootstrapped).
    seed_seq = np.random.SeedSequence(args.seed)
    child_a, child_b, child_c = seed_seq.spawn(3)
    comparisons = {
        "A_full_vs_C_affine": paired_edge(
            results["A_full"], results["C_affine"],
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(child_a),
            block_size=args.block_size,
        ),
        "B_refit_vs_C_affine": paired_edge(
            results["B_refit"], results["C_affine"],
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(child_b),
            block_size=args.block_size,
        ),
        "A_full_vs_B_refit": paired_edge(
            results["A_full"], results["B_refit"],
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(child_c),
            block_size=args.block_size,
        ),
    }

    # Optional forced-phase sweep on A's structural fit.
    sweep: list[dict[str, Any]] = []
    grid = [float(v) for v in args.phase_sweep.split(",") if v.strip()]
    for phi in grid:
        structural_phi = deepcopy(structural_free)
        structural_phi["alpha_phase"] = phi
        structural_phi["quantum_improved"] = True
        result_phi = run_config(
            frame, structural=structural_phi, folds=args.folds, tick_size=tick_size
        )
        sweep.append(
            {
                "alpha_phase": phi,
                "brier": result_phi["pooled"]["brier"],
                "log_loss": result_phi["pooled"]["log_loss"],
                "accuracy": result_phi["pooled"]["accuracy"],
            }
        )
        print(f"[ablation] phase sweep alpha_phase={phi:.3f} -> Brier={result_phi['pooled']['brier']:.6f}")

    # Fold-sensitivity: does the "QRW beats affine" verdict survive changing the
    # arbitrary walk-forward fold count? The affine baseline is refit cleanly per
    # fold so it is stable; a large swing in the A_full edge exposes a
    # non-robust verdict rather than a real predictive advantage.
    fold_sensitivity: list[dict[str, Any]] = []
    fold_grid = [int(v) for v in args.fold_sensitivity.split(",") if v.strip()]
    for k in fold_grid:
        if k < 2:
            continue
        a_k = run_config(
            frame, structural=structural_free, folds=k, tick_size=tick_size
        )
        c_k = run_affine(frame, folds=k)
        edge_k = paired_edge(
            a_k, c_k,
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(np.random.SeedSequence(args.seed + k)),
            block_size=args.block_size,
        )
        fold_sensitivity.append(
            {
                "folds": k,
                "a_full_brier": a_k["pooled"]["brier"],
                "affine_brier": c_k["pooled"]["brier"],
                "edge_a_minus_affine": edge_k["model_minus_comparison"],
                "confidence_interval_95": edge_k["confidence_interval_95"],
                "significant": edge_k["significant"],
                "qrw_wins": bool(
                    edge_k["significant"] and edge_k["direction"] == "first_better"
                ),
            }
        )
        print(
            f"[ablation] fold-sensitivity folds={k}: A_full={a_k['pooled']['brier']:.6f} "
            f"affine={c_k['pooled']['brier']:.6f} edge={edge_k['model_minus_comparison']:+.6f} "
            f"({'QRW wins' if fold_sensitivity[-1]['qrw_wins'] else 'QRW does not win'})"
        )

    # Strip the heavy per-event arrays before serialising.
    serialisable = {
        name: {k: v for k, v in res.items() if not k.startswith("_")}
        for name, res in results.items()
    }
    audit = {
        "kind": "alpha_phase_ablation",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "folds": int(args.folds),
        "bootstrap_samples": int(args.bootstrap_samples),
        "block_size": int(args.block_size),
        "seed": int(args.seed),
        "tick_size": tick_size,
        "configs": serialisable,
        "comparisons": comparisons,
        "phase_sweep": sweep,
        "fold_sensitivity": fold_sensitivity,
    }

    json_out = (ROOT / args.json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out = (ROOT / args.md_out).resolve()
    md_out.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"[ablation] wrote {json_out}")
    print(f"[ablation] wrote {md_out}")


# Below this pooled-Brier magnitude a paired difference is treated as
# practically zero regardless of large-N statistical significance: with
# hundreds of thousands of events even a ~1e-7 systematic difference clears a
# bootstrap CI, so significance alone would overstate the effect.
PHASE_NEGLIGIBLE_BRIER = 1e-4


def _verdict(comparisons: dict[str, Any]) -> str:
    phase = comparisons["A_full_vs_B_refit"]
    window = comparisons["B_refit_vs_C_affine"]
    total = comparisons["A_full_vs_C_affine"]
    lines = []
    phase_magnitude = abs(phase["model_minus_comparison"])
    if phase_magnitude < PHASE_NEGLIGIBLE_BRIER:
        lines.append(
            "The phase (interference) term is **practically zero** "
            f"({phase['model_minus_comparison']:+.2e} Brier, below the "
            f"{PHASE_NEGLIGIBLE_BRIER:.0e} threshold): despite large-N "
            "significance it is orders of magnitude smaller than the main edge, "
            "so quantum interference does not drive the result."
        )
    elif phase["significant"] and phase["direction"] == "first_better":
        lines.append(
            "The phase (interference) term contributes a statistically "
            "significant and non-negligible Brier improvement (A_full beats "
            "B_refit)."
        )
    else:
        lines.append(
            "The phase (interference) term shows **no** statistically significant "
            "Brier benefit (A_full vs B_refit CI spans 0): on this data the "
            "quantum-interference mechanism does not drive the advantage."
        )
    if window["significant"] and window["direction"] == "first_better":
        lines.append(
            "The phase-free windowed density-matrix model (B_refit) *does* beat "
            "the independently-fit affine baseline (B_refit vs C_affine), so the "
            "edge comes from windowing/decoherence, not from quantum interference."
        )
    else:
        lines.append(
            "The phase-free windowed model shows no significant edge over the "
            "affine baseline (B_refit vs C_affine)."
        )
    total_word = (
        "does" if (total["significant"] and total["direction"] == "first_better")
        else "does not"
    )
    lines.append(
        f"At the reported protocol's fold count, the full quantum model "
        f"{total_word} significantly beat the affine baseline (A_full vs "
        "C_affine) -- but see the fold-sensitivity table for whether that "
        "verdict is robust."
    )
    return " ".join(lines)


def _fold_verdict(fold_sensitivity: list[dict[str, Any]]) -> str:
    if not fold_sensitivity:
        return ""
    wins = [row for row in fold_sensitivity if row["qrw_wins"]]
    losses = [
        row for row in fold_sensitivity
        if row["significant"] and not row["qrw_wins"]
    ]
    if wins and losses:
        win_folds = ", ".join(str(r["folds"]) for r in wins)
        loss_folds = ", ".join(str(r["folds"]) for r in losses)
        return (
            f"**The QRW-vs-affine verdict is not robust to the fold count.** QRW "
            f"wins significantly at folds {{{win_folds}}} but loses significantly "
            f"at folds {{{loss_folds}}}, while the affine baseline stays stable. "
            "A verdict that flips sign under an arbitrary evaluation "
            "hyperparameter is an artifact of that choice, not evidence of a "
            "genuine predictive advantage."
        )
    if wins and not losses:
        return (
            "QRW wins (or ties) the affine baseline across every fold count "
            "tested -- the advantage is robust to this evaluation hyperparameter."
        )
    return (
        "QRW does not significantly beat the affine baseline at any fold count "
        "tested."
    )


def _render_markdown(audit: dict[str, Any]) -> str:
    cfg = audit["configs"]
    cmp = audit["comparisons"]
    rows = [
        "# Alpha-phase ablation — isolating the quantum-interference contribution",
        "",
        f"**Status:** `{audit['status']}` — exploratory mechanism diagnostic, "
        "not a confirmatory run. Do not relabel as confirmatory evidence.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` ({audit['rows']:,} rows)",
        f"- Walk-forward folds: {audit['folds']}; block bootstrap samples: "
        f"{audit['bootstrap_samples']} (block size {audit['block_size']})",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']} · "
        f"seed {audit['seed']}",
        "",
        "## Configurations",
        "",
        "| Config | quantum_improved | alpha_phase | gamma | pooled Brier | log loss | accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("A_full", "B_refit", "B_posthoc", "C_affine"):
        c = cfg[name]
        phase = "n/a" if np.isnan(c["alpha_phase"]) else f"{c['alpha_phase']:.3e}"
        gamma = "n/a" if np.isnan(c["gamma"]) else f"{c['gamma']:.4f}"
        rows.append(
            f"| {name} | {c['quantum_improved']} | {phase} | "
            f"{gamma} | {c['pooled']['brier']:.6f} | "
            f"{c['pooled']['log_loss']:.6f} | {c['pooled']['accuracy']:.4f} |"
        )
    rows += [
        "",
        "## Mechanism decomposition (paired Brier difference, 95% block-bootstrap CI)",
        "",
        "A negative mean = the first-listed config has the lower (better) Brier.",
        "",
        "| Comparison | Meaning | mean (a−b) | 95% CI | significant |",
        "|---|---|---:|---|:--:|",
    ]
    meanings = {
        "A_full_vs_C_affine": "total quantum vs affine baseline (windowing + phase)",
        "B_refit_vs_C_affine": "phase-free quantum vs affine baseline",
        "A_full_vs_B_refit": "pure phase (interference) contribution",
    }
    for key, meaning in meanings.items():
        e = cmp[key]
        lo, hi = e["confidence_interval_95"]
        rows.append(
            f"| {key} | {meaning} | {e['model_minus_comparison']:+.6f} | "
            f"[{lo:+.6f}, {hi:+.6f}] | {'✔' if e['significant'] else '—'} |"
        )
    if audit["phase_sweep"]:
        rows += [
            "",
            "## Forced phase sweep (A's structural fit, alpha_phase overridden)",
            "",
            "| alpha_phase | pooled Brier | log loss | accuracy |",
            "|---:|---:|---:|---:|",
        ]
        for s in audit["phase_sweep"]:
            rows.append(
                f"| {s['alpha_phase']:.3f} | {s['brier']:.6f} | "
                f"{s['log_loss']:.6f} | {s['accuracy']:.4f} |"
            )
    if audit.get("fold_sensitivity"):
        rows += [
            "",
            "## Fold-count robustness (A_full vs affine baseline)",
            "",
            "Negative edge = QRW has the lower (better) Brier. The affine "
            "baseline is refit per fold and stays stable; a sign flip in the "
            "edge exposes a non-robust verdict.",
            "",
            "| folds | A_full Brier | affine Brier | edge (QRW−affine) | 95% CI | QRW wins? |",
            "|---:|---:|---:|---:|---|:--:|",
        ]
        for s in audit["fold_sensitivity"]:
            lo, hi = s["confidence_interval_95"]
            rows.append(
                f"| {s['folds']} | {s['a_full_brier']:.6f} | "
                f"{s['affine_brier']:.6f} | {s['edge_a_minus_affine']:+.6f} | "
                f"[{lo:+.6f}, {hi:+.6f}] | {'✔' if s['qrw_wins'] else '✘'} |"
            )
    rows += ["", "## Verdict", "", _verdict(cmp), ""]
    fold_verdict = _fold_verdict(audit.get("fold_sensitivity", []))
    if fold_verdict:
        rows += [fold_verdict, ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
